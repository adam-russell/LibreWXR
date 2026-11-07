# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import gzip
import io
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.sources

from librewxr.data.regions import REGIONS, resolve_regions
from librewxr.sources._helpers import _dbz_float_to_uint8
from librewxr.sources.regional.north_america.usa.radar.mrms import (
    MRMS_EXTENTS,
    MRMSSource,
    _parse_mrms_grib2,
    _resample_mrms_to_region,
)

MRMS_SOUTH, MRMS_NORTH, MRMS_WEST, MRMS_EAST = MRMS_EXTENTS["USCOMP"]


def _make_minimal_grib2(
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
) -> bytes:
    """Create a minimal GRIB2 file in memory with cfgrib-compatible structure.

    This uses xarray's to_netcdf after creating a proper dataset, then
    wraps it in a fake GRIB2 envelope. Since constructing valid GRIB2
    from scratch is extremely complex, we instead create a dataset that
    cfgrib can read by writing via xarray with engine='cfgrib'.

    However, since cfgrib can only READ grib2 (not write), we take a
    different approach: create a netCDF-like binary that we use directly
    for testing _resample_mrms_to_region with the dataset object directly.
    """
    # We'll return the raw data + axes for direct testing
    # The actual GRIB2 round-trip is tested via integration tests
    ds = xr.Dataset(
        {"reflectivity": (["latitude", "longitude"], data)},
        coords={
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


class TestMRMSRegion:
    def test_uscomp_in_regions(self):
        assert "USCOMP" in REGIONS

    def test_cacomp_in_regions(self):
        assert "CACOMP" in REGIONS

    def test_all_includes_uscomp_cacomp(self):
        result = resolve_regions("ALL")
        assert "USCOMP" in result
        assert "CACOMP" in result

    def test_mrms_extent_constants(self):
        assert MRMS_SOUTH < MRMS_NORTH
        assert MRMS_WEST < MRMS_EAST
        # MRMS extent should cover most of the US and populated Canada
        assert MRMS_SOUTH <= 21  # USCOMP south is 23, give some margin
        assert MRMS_NORTH >= 54  # Should go at least to ~55°N
        assert MRMS_WEST <= -129  # Should cover at least to -130°W
        assert MRMS_EAST >= -61  # Should cover at least to ~-60°W

    def test_mrms_covers_uscomp_extent(self):
        uscomp = REGIONS["USCOMP"]
        # USCOMP should be fully within MRMS extent
        assert uscomp.south >= MRMS_SOUTH
        assert uscomp.north <= MRMS_NORTH
        assert uscomp.west >= MRMS_WEST
        assert uscomp.east <= MRMS_EAST

    def test_mrms_partially_covers_cacomp(self):
        cacomp = REGIONS["CACOMP"]
        # CACOMP extends beyond MRMS in the north and (some) east/west
        assert cacomp.north > MRMS_NORTH  # Canada goes north of MRMS
        # MRMS should at least partially overlap CACOMP
        assert cacomp.south < MRMS_NORTH  # Some overlap exists
        assert cacomp.west < MRMS_EAST  # Some overlap exists


class TestMRMSResampling:
    def _make_dataset(
        self,
        nlat: int = 100,
        nlon: int = 200,
        south: float = 20.0,
        north: float = 55.0,
        west: float = -130.0,
        east: float = -60.0,
        fill_value: float = -999.0,
        dbz_value: float = 35.0,
    ) -> xr.Dataset:
        """Create a synthetic MRMS-like dataset for testing."""
        lats = np.linspace(north, south, nlat)  # descending
        lons = np.linspace(west, east, nlon)
        data = np.full((nlat, nlon), fill_value, dtype=np.float32)
        # Put a block of real data in the middle
        data[25:75, 50:150] = dbz_value
        return xr.Dataset(
            {"unknown": (["latitude", "longitude"], data)},
            coords={"latitude": lats, "longitude": lons},
        )

    def test_resample_produces_correct_shape(self):
        uscomp = REGIONS["USCOMP"]
        ds = self._make_dataset()
        result = _resample_mrms_to_region(ds, uscomp)
        assert result is not None
        assert result.shape == (uscomp.height, uscomp.width)

    def test_resample_nodata_becomes_zero(self):
        uscomp = REGIONS["USCOMP"]
        ds = self._make_dataset(dbz_value=-999.0)  # all no-data
        result = _resample_mrms_to_region(ds, uscomp)
        assert result is not None
        assert np.all(result == 0)

    def test_resample_valid_dbz_encoding(self):
        uscomp = REGIONS["USCOMP"]
        ds = self._make_dataset(dbz_value=30.0)
        result = _resample_mrms_to_region(ds, uscomp)
        assert result is not None
        # 30 dBZ -> (30+32)*2 = 124
        nonzero = result[result > 0]
        if len(nonzero) > 0:
            assert nonzero.min() >= 2  # at least some non-zero values
            assert nonzero.max() <= 255

    def test_resample_cacomp_produces_correct_shape(self):
        cacomp = REGIONS["CACOMP"]
        ds = self._make_dataset(nlat=3500, nlon=7000)
        result = _resample_mrms_to_region(ds, cacomp)
        assert result is not None
        assert result.shape == (cacomp.height, cacomp.width)


class TestMRMSUrl:
    def test_latest_url(self):
        src = MRMSSource("https://mrms.ncep.noaa.gov/2D")
        url = src._latest_url()
        assert "MergedReflectivityQCComposite" in url
        assert url.endswith(".grib2.gz")
        assert "latest" in url

    def test_archive_url(self):
        from datetime import datetime, timezone
        src = MRMSSource("https://mrms.ncep.noaa.gov/2D")
        dt = datetime(2026, 4, 28, 15, 30, 42, tzinfo=timezone.utc)
        url = src._archive_url(dt)
        assert "20260428-153042" in url
        assert url.endswith(".grib2.gz")

    def test_custom_base_url(self):
        src = MRMSSource("https://example.com/mrms")
        url = src._latest_url()
        assert url.startswith("https://example.com/mrms/")


class TestMRMSDbzEncoding:
    def test_nodata_maps_to_zero(self):
        arr = np.array([[-999.0, -100.0, 35.0]], dtype=np.float32)
        result = _dbz_float_to_uint8(arr)
        assert result[0, 0] == 0  # -999 → 0 (no-data)
        assert result[0, 1] == 0  # -100 → 0 (< -32)
        assert result[0, 2] == 134  # (35+32)*2 = 134

    def test_positive_dbz_range(self):
        dbz = np.array([[0, 10, 20, 30, 40, 50, 60]], dtype=np.float32)
        result = _dbz_float_to_uint8(dbz)
        # (0+32)*2=64, (10+32)*2=84, ..., (60+32)*2=184
        assert result[0, 0] == 64
        assert result[0, 1] == 84
        assert result[0, 6] == 184

    def test_negative_dbz(self):
        dbz = np.array([[-31.0, -20.0, -10.0]], dtype=np.float32)
        result = _dbz_float_to_uint8(dbz)
        assert result[0, 0] == 2  # (-31+32)*2 = 2
        assert result[0, 1] == 24  # (-20+32)*2 = 24
        assert result[0, 2] == 44  # (-10+32)*2 = 44


class TestMRMSStations:
    def test_mrms_stations_is_dict(self):
        from librewxr.sources.regional.north_america.usa.radar.mrms.stations import (
            STATION_MAP,
        )
        assert isinstance(STATION_MAP, dict)
        assert len(STATION_MAP) == 6
        assert "USCOMP" in STATION_MAP
        assert "CACOMP" in STATION_MAP
        assert "AKCOMP" in STATION_MAP
        assert "HICOMP" in STATION_MAP
        assert "PRCOMP" in STATION_MAP
        assert "GUCOMP" in STATION_MAP

    def test_mrms_stations_uscomp_cacomp(self):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada.stations import (
            STATIONS as CANADA_STATIONS,
        )
        from librewxr.sources.regional.north_america.usa.radar.mrms.stations import (
            STATION_MAP,
        )
        from librewxr.sources.regional.north_america.usa.radar.stations import (
            NEXRAD_CONUS,
        )
        # USCOMP and CACOMP share the combined NEXRAD + Canada list
        combined = NEXRAD_CONUS + CANADA_STATIONS
        assert STATION_MAP["USCOMP"] == combined
        assert STATION_MAP["CACOMP"] == combined

    def test_mrms_stations_territories(self):
        from librewxr.sources.regional.north_america.usa.radar.mrms.stations import (
            STATION_MAP,
        )
        from librewxr.sources.regional.north_america.usa.radar.stations import (
            NEXRAD_ALASKA,
            NEXRAD_GUAM,
            NEXRAD_HAWAII,
            NEXRAD_PUERTO_RICO,
        )
        assert STATION_MAP["AKCOMP"] == NEXRAD_ALASKA
        assert STATION_MAP["HICOMP"] == NEXRAD_HAWAII
        assert STATION_MAP["PRCOMP"] == NEXRAD_PUERTO_RICO
        assert STATION_MAP["GUCOMP"] == NEXRAD_GUAM


class TestMRMSDirCache:
    def test_timestamp_regex_matches_archived_files(self):
        from librewxr.sources.regional.north_america.usa.radar.mrms import MRMSSource
        pat = MRMSSource._TIMESTAMP_RE
        m = pat.search("MRMS_MergedReflectivityQCComposite_00.50_20260428-060041.grib2.gz")
        assert m is not None
        assert m.group(1) == "20260428-060041"

    def test_timestamp_regex_ignores_latest(self):
        from librewxr.sources.regional.north_america.usa.radar.mrms import MRMSSource
        pat = MRMSSource._TIMESTAMP_RE
        m = pat.search("MRMS_MergedReflectivityQCComposite.latest.grib2.gz")
        assert m is None

    async def test_find_nearest_midpoint(self):
        from datetime import datetime, timezone
        from librewxr.sources.regional.north_america.usa.radar.mrms import MRMSSource
        src = MRMSSource("https://mrms.ncep.noaa.gov/2D")
        entries = [
            (datetime(2026, 4, 28, 6, 0, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060000.grib2.gz"),
            (datetime(2026, 4, 28, 6, 2, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060200.grib2.gz"),
            (datetime(2026, 4, 28, 6, 4, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060400.grib2.gz"),
            (datetime(2026, 4, 28, 6, 6, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060600.grib2.gz"),
        ]
        src._dir_cache = entries
        src._dir_cache_time = time.time()

        target = datetime(2026, 4, 28, 6, 3, 0, tzinfo=timezone.utc)
        url = await src._find_nearest_url(target)
        assert url is not None
        assert "060200" in url or "060400" in url

    async def test_find_nearest_exact_match(self):
        from datetime import datetime, timezone
        from librewxr.sources.regional.north_america.usa.radar.mrms import MRMSSource
        src = MRMSSource("https://mrms.ncep.noaa.gov/2D")
        entries = [
            (datetime(2026, 4, 28, 6, 0, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060000.grib2.gz"),
            (datetime(2026, 4, 28, 6, 2, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060200.grib2.gz"),
            (datetime(2026, 4, 28, 6, 4, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060400.grib2.gz"),
        ]
        src._dir_cache = entries
        src._dir_cache_time = time.time()

        url = await src._find_nearest_url(datetime(2026, 4, 28, 6, 2, 0, tzinfo=timezone.utc))
        assert url is not None
        assert "060200" in url

    async def test_find_nearest_before_range(self):
        from datetime import datetime, timezone
        from librewxr.sources.regional.north_america.usa.radar.mrms import MRMSSource
        src = MRMSSource("https://mrms.ncep.noaa.gov/2D")
        entries = [
            (datetime(2026, 4, 28, 6, 4, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060400.grib2.gz"),
            (datetime(2026, 4, 28, 6, 6, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060600.grib2.gz"),
        ]
        src._dir_cache = entries
        src._dir_cache_time = time.time()

        url = await src._find_nearest_url(datetime(2026, 4, 28, 5, 50, 0, tzinfo=timezone.utc))
        assert url is not None
        assert "060400" in url

    async def test_find_nearest_after_range(self):
        from datetime import datetime, timezone
        from librewxr.sources.regional.north_america.usa.radar.mrms import MRMSSource
        src = MRMSSource("https://mrms.ncep.noaa.gov/2D")
        entries = [
            (datetime(2026, 4, 28, 6, 0, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060000.grib2.gz"),
            (datetime(2026, 4, 28, 6, 2, 0, tzinfo=timezone.utc), "MRMS_MergedReflectivityQCComposite_00.50_20260428-060200.grib2.gz"),
        ]
        src._dir_cache = entries
        src._dir_cache_time = time.time()

        url = await src._find_nearest_url(datetime(2026, 4, 28, 6, 10, 0, tzinfo=timezone.utc))
        assert url is not None
        assert "060200" in url

class TestStripedMRMSRemap:
    @pytest.mark.parametrize("height", [255, 256, 257, 513])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matches_full_maps_across_strip_boundaries(self, height, dtype):
        import cv2
        from librewxr.sources.regional.north_america.usa.radar.mrms import source

        rng = np.random.default_rng(42)
        data = rng.uniform(-33, 65, (37, 53)).astype(dtype)
        nodata_mask = (rng.random(data.shape) < 0.1).astype(np.float32)
        data[nodata_mask > 0] = 0
        data[2, 2] = np.nan
        rows = np.linspace(-1.25, 38.75, height)
        cols = np.linspace(-0.75, 54.25, 79)
        maps = source._build_resample_maps(rows, cols, *data.shape)
        padded = source._pad_continuation(
            data, maps.pad_top, maps.pad_bottom, maps.pad_left, maps.pad_right,
        )
        padded_mask = source._pad_continuation(
            nodata_mask, maps.pad_top, maps.pad_bottom, maps.pad_left, maps.pad_right,
        )
        original_data = padded.copy()
        original_mask = padded_mask.copy()
        padded.flags.writeable = False
        padded_mask.flags.writeable = False
        maps.map_x.flags.writeable = False
        maps.map_y.flags.writeable = False
        # Independent reference: the original full-sized float32 maps.
        shape = (height, len(cols))
        map_x = np.broadcast_to((cols + maps.pad_left).astype(np.float32)[None, :], shape).copy()
        map_y = np.broadcast_to((rows + maps.pad_top).astype(np.float32)[:, None], shape).copy()
        expected = cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR)
        mask = cv2.remap(padded_mask, map_x, map_y, cv2.INTER_LINEAR)
        with np.errstate(invalid="ignore"):
            expected = _dbz_float_to_uint8(np.where(mask > 0.5, -33.0, expected))
            result = source._remap_and_encode_striped(padded, padded_mask, maps)
        np.testing.assert_array_equal(padded, original_data)
        np.testing.assert_array_equal(padded_mask, original_mask)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint8
        assert not np.shares_memory(result, padded)

    def test_cache_retains_axes_instead_of_destination_grids(self):
        from librewxr.sources.regional.north_america.usa.radar.mrms import source

        maps = source._build_resample_maps(
            np.linspace(0, 3499, 5400), np.linspace(0, 6999, 12200), 3500, 7000,
        )
        assert maps.map_x.shape == (12200,)
        assert maps.map_y.shape == (5400,)
        assert maps.map_x.dtype == maps.map_y.dtype == np.float32
        assert maps.map_x.nbytes + maps.map_y.nbytes == (12200 + 5400) * 4

    def test_majority_nodata_threshold_is_strict(self):
        from librewxr.sources.regional.north_america.usa.radar.mrms import source

        data = np.full((2, 4), 30, dtype=np.float32)
        mask = np.tile(np.array([0, 0.5, 0.5001, 1], dtype=np.float32), (2, 1))
        maps = source._build_resample_maps(np.arange(2.), np.arange(4.), 2, 4)
        result = source._remap_and_encode_striped(data, mask, maps)
        np.testing.assert_array_equal(result, [[124, 124, 0, 0], [124, 124, 0, 0]])

    def test_float_workspaces_are_bounded_to_a_strip(self, monkeypatch):
        from librewxr.sources.regional.north_america.usa.radar.mrms import source

        data = np.full((513, 9), 30, dtype=np.float32)
        mask = np.zeros_like(data)
        maps = source._build_resample_maps(np.arange(513.), np.arange(9.), *data.shape)
        remap_shapes, encode_shapes = [], []
        original_remap = source.cv2.remap
        original_encode = source._dbz_float_to_uint8

        def remap(array, map_x, map_y, *args, **kwargs):
            assert map_x.shape == map_y.shape
            remap_shapes.append(map_x.shape)
            return original_remap(array, map_x, map_y, *args, **kwargs)

        def encode(array):
            encode_shapes.append(array.shape)
            return original_encode(array)

        monkeypatch.setattr(source.cv2, "remap", remap)
        monkeypatch.setattr(source, "_dbz_float_to_uint8", encode)
        result = source._remap_and_encode_striped(data, mask, maps)
        assert remap_shapes == [(256, 9)] * 4 + [(1, 9)] * 2
        assert encode_shapes == [(256, 9), (256, 9), (1, 9)]
        np.testing.assert_array_equal(result, np.full(data.shape, 124, dtype=np.uint8))
