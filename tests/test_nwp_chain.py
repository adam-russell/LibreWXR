# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the NWPChain dispatcher.

The Tier 2 precip-bbox fast-path surface (``_probe_domain_bbox`` /
``_bbox_intersects`` / ``has_precip_in_bbox`` / ``_domain_bboxes``) was
removed when the stitched global precip mask (``PrecipMaskStore``)
replaced it — see ``tests/test_precip_mask.py``.  What remains here is
the dispatcher behaviour that is independent of that gate.
"""

import numpy as np
import pytest

from librewxr.data.nwp_source import NWPChain


class FakeRegionalSource:
    """A regional NWP source: finite projection-only domain."""

    def __init__(self, west, south, east, north):
        self._bbox = (west, south, east, north)

    @property
    def name(self) -> str:
        return "fake_regional"

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        w, s, e, n = self._bbox
        return (
            (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
        )

    def feather_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return self.domain_mask(lat, lon).astype(np.float32)

    def has_data(self) -> bool:
        return True

    def has_data_at(self, timestamp: int) -> bool:
        return True

    def sample(self, lat, lon, timestamp=None, bilinear=False) -> np.ndarray:
        return np.zeros(lat.shape, dtype=np.uint8)

    @property
    def supports_snow(self) -> bool:
        return False

    def get_snow_mask(self, lat, lon, timestamp=None) -> np.ndarray:
        return np.zeros(lat.shape, dtype=bool)


class TestNWPChain:
    def test_sources_property_unchanged(self):
        chain = NWPChain([FakeRegionalSource(-125.0, 25.0, -70.0, 50.0)])
        assert len(chain.sources) == 1
        assert chain.sources[0].name == "fake_regional"


class ArraySource:
    """Model that can return its read-only cached data directly."""

    name = "array_source"

    def __init__(self, values, snow, feather=1.0, domain=True, available=True,
                 supports_snow=True):
        self.values = np.asarray(values, dtype=np.uint8)
        self.snow = np.asarray(snow, dtype=bool)
        self.feather = np.broadcast_to(np.float32(feather), self.values.shape)
        self.domain = np.broadcast_to(np.asarray(domain, dtype=bool), self.values.shape)
        self.available = available
        self.supports_snow = supports_snow
        self.values.flags.writeable = False
        self.snow.flags.writeable = False

    def has_data(self):
        return self.available

    def has_data_at(self, timestamp):
        return self.available and timestamp != 99

    def feather_mask(self, lat, lon):
        return self.feather

    def domain_mask(self, lat, lon):
        return self.domain

    @staticmethod
    def _gather(field, lat):
        if lat.shape == field.shape:
            return field
        return field.ravel()[lat.astype(np.intp)]

    def sample(self, lat, lon, timestamp=None, bilinear=False):
        self.last_sample = (timestamp, bilinear)
        return self._gather(self.values, lat)

    def get_snow_mask(self, lat, lon, timestamp=None):
        self.last_snow_timestamp = timestamp
        return self._gather(self.snow, lat)


@pytest.fixture
def model_coordinates():
    lat = np.arange(6, dtype=np.float64).reshape(2, 3)
    lon = np.zeros_like(lat)
    lat.flags.writeable = False
    lon.flags.writeable = False
    return lat, lon


class TestFullCoverageShortcuts:
    @pytest.mark.parametrize("timestamp", [None, 123])
    @pytest.mark.parametrize("bilinear", [False, True])
    def test_full_weight_sample_preserves_values_and_ownership(
        self, model_coordinates, timestamp, bilinear,
    ):
        values = [[0, 1, 83], [84, 128, 255]]
        source = ArraySource(values, [[False, True, False], [True, False, True]])
        fallback = ArraySource(np.full((2, 3), 200), np.ones((2, 3)))
        result = NWPChain([source, fallback]).sample(*model_coordinates, timestamp, bilinear)
        assert source.last_sample == (timestamp, bilinear)
        np.testing.assert_array_equal(result, values)
        assert result.dtype == np.uint8
        assert result.flags.writeable
        assert not np.shares_memory(result, source.values)
        result[:] = 0
        np.testing.assert_array_equal(source.values, values)

    @pytest.mark.parametrize("timestamp", [None, 123])
    def test_full_domain_snow_preserves_rain_and_ownership(
        self, model_coordinates, timestamp,
    ):
        snow = [[False, True, False], [True, False, True]]
        source = ArraySource(np.full((2, 3), 100), snow)
        fallback = ArraySource(np.full((2, 3), 200), np.ones((2, 3)))
        result = NWPChain([source, fallback]).get_snow_mask(*model_coordinates, timestamp)
        assert source.last_snow_timestamp == timestamp
        np.testing.assert_array_equal(result, snow)
        assert result.dtype == np.bool_
        assert result.flags.writeable
        assert not np.shares_memory(result, source.snow)
        result[:] = False
        np.testing.assert_array_equal(source.snow, snow)

    def test_partial_weight_keeps_earlier_contribution(self, model_coordinates):
        first = ArraySource(np.full((2, 3), 40), np.zeros((2, 3)), feather=0.25)
        fallback = ArraySource(np.full((2, 3), 200), np.ones((2, 3)))
        result = NWPChain([first, fallback]).sample(*model_coordinates)
        np.testing.assert_array_equal(result, np.full((2, 3), 160, dtype=np.uint8))

    def test_partial_domain_keeps_authoritative_rain(self, model_coordinates):
        domain = [[True, False, True], [False, True, False]]
        first = ArraySource(np.full((2, 3), 40), np.zeros((2, 3)), domain=domain)
        fallback = ArraySource(np.full((2, 3), 200), np.ones((2, 3)))
        result = NWPChain([first, fallback]).get_snow_mask(*model_coordinates)
        np.testing.assert_array_equal(result, ~np.asarray(domain))

    @pytest.mark.parametrize("timestamp,available", [(None, False), (123, False), (99, True)])
    def test_unavailable_first_source_falls_through(
        self, model_coordinates, timestamp, available,
    ):
        first = ArraySource(np.full((2, 3), 40), np.zeros((2, 3)), available=available)
        fallback = ArraySource(np.full((2, 3), 200), np.ones((2, 3)))
        # The fallback can answer this time even when the first source cannot.
        fallback.has_data_at = lambda timestamp: True
        chain = NWPChain([first, fallback])
        np.testing.assert_array_equal(chain.sample(*model_coordinates, timestamp),
                                      np.full((2, 3), 200, dtype=np.uint8))
        np.testing.assert_array_equal(chain.get_snow_mask(*model_coordinates, timestamp),
                                      np.ones((2, 3), dtype=bool))

    def test_source_without_snow_support_falls_through(self, model_coordinates):
        first = ArraySource(np.full((2, 3), 40), np.zeros((2, 3)), supports_snow=False)
        fallback = ArraySource(np.full((2, 3), 200), np.ones((2, 3)))
        result = NWPChain([first, fallback]).get_snow_mask(*model_coordinates)
        np.testing.assert_array_equal(result, np.ones((2, 3), dtype=bool))

    def test_empty_queries_preserve_shape(self):
        lat = np.empty((0, 3))
        source = ArraySource(lat, lat)
        chain = NWPChain([source])
        assert chain.sample(lat, lat).shape == lat.shape
        assert chain.get_snow_mask(lat, lat).shape == lat.shape
