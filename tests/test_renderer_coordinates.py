# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coordinate shortcuts must preserve the full-map renderer's exact geometry."""
import math
from unittest.mock import Mock

import numpy as np
import pytest

from librewxr.config import settings
from librewxr.data.regions import REGIONS
from librewxr.tiles import coordinates, sampling


def _tile(lat, lon, z):
    return (
        int((lon + 180) / 360 * 2**z),
        int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**z),
    )


def _full_map_radius(region, z, x, y, tile_size):
    """Original blur calculation, independent of the new centre-only path."""
    base = settings.smooth_radius
    if base <= 0:
        return 0.0
    rows, cols = coordinates._compute_region_pixel_indices_fractional(
        region, z, x, y, tile_size,
    )
    mid = tile_size // 2
    dr = abs(float(rows[mid + 1, mid] - rows[mid - 1, mid])) / 2.0
    dc = abs(float(cols[mid, mid + 1] - cols[mid, mid - 1])) / 2.0
    if dr < 1e-6 or dc < 1e-6:
        return base
    return min(base * max(1.0, max(1.0 / dr, 1.0 / dc) * .25), tile_size / 32.0)


@pytest.fixture(autouse=True)
def clear_caches():
    for fn in coordinates.ALL_CACHES:
        fn.cache_clear()
    sampling.sampling_plans.clear()
    yield
    for fn in coordinates.ALL_CACHES:
        fn.cache_clear()
    sampling.sampling_plans.clear()


@pytest.mark.parametrize("region", list(REGIONS.values()), ids=lambda r: r.name)
@pytest.mark.parametrize("size", [16, 256, 512])
@pytest.mark.parametrize("base", [0.0, .2, 2.0])
def test_radius_matches_full_maps(region, size, base, monkeypatch):
    monkeypatch.setattr(settings, "smooth_radius", base)
    locations = [
        (0, 0, 0),
        (7, *_tile((region.north + region.south) / 2, (region.east + region.west) / 2, 7)),
        (12, *_tile(region.south + .001, region.west + .001, 12)),
        (7, 0, 0),  # Clipped/outside the source, including projected grids.
    ]
    for z, x, y in locations:
        assert coordinates.compute_blur_radius(region, z, x, y, size) == _full_map_radius(
            region, z, x, y, size,
        )


@pytest.mark.parametrize("region_name,helper", [("OPERA", "_laea_pixel_coords"), ("ITCOMP", "_tmerc_pixel_coords")])
def test_blur_projects_only_three_by_three_and_caches_scalar(region_name, helper, monkeypatch):
    region = REGIONS[region_name]
    monkeypatch.setattr(settings, "smooth_radius", 1.0)
    project = Mock(wraps=getattr(coordinates, helper))
    monkeypatch.setattr(coordinates, helper, project)
    args = (region, 7, 64, 40, 512)
    first = coordinates.compute_blur_radius(*args)
    assert coordinates.compute_blur_radius(*args) == first
    assert project.call_count == 1
    lon, lat, _ = project.call_args.args
    assert lon.shape == lat.shape == (3,)
    info = coordinates._cached_blur_radius.cache_info()
    assert (info.currsize, info.hits, info.misses) == (1, 1, 1)
    assert coordinates.region_pixel_indices_fractional.cache_info().currsize == 0
    monkeypatch.setattr(settings, "smooth_radius", .2)
    assert coordinates.compute_blur_radius(*args) == _full_map_radius(*args)
    assert coordinates._cached_blur_radius.cache_info().currsize == 2


def test_radius_cache_is_bounded_reported_and_pressure_cleared(monkeypatch):
    from librewxr.main import _clear_coord_caches
    monkeypatch.setattr(settings, "smooth_radius", 1.0)
    cache = coordinates._cached_blur_radius
    limit = cache.cache_info().maxsize
    for x in range(limit + 1):
        coordinates.compute_blur_radius(REGIONS["USCOMP"], 16, x, 20000, 512)
    assert cache.cache_info().currsize == limit
    stats = coordinates.coord_cache_stats()["caches"][cache.__name__]
    assert stats["entries"] == limit
    assert coordinates.coord_cache_bytes() >= limit * coordinates._CACHE_ENTRY_BYTES[cache]
    _clear_coord_caches()
    assert cache.cache_info().currsize == 0
    assert coordinates.coord_cache_bytes() == 0


@pytest.mark.parametrize("region", [r for r in REGIONS.values() if r.proj in ("laea", "tmerc")], ids=lambda r: r.name)
@pytest.mark.parametrize("size,pad", [(16, 0), (256, 3), (512, 48)])
def test_projected_plan_matches_all_original_arrays_with_one_transform(region, size, pad, monkeypatch):
    helper = "_laea_pixel_coords" if region.proj == "laea" else "_tmerc_pixel_coords"
    project = Mock(wraps=getattr(coordinates, helper))
    monkeypatch.setattr(coordinates, helper, project)
    for z, x, y in [
        (0, 0, 0),
        (7, *_tile((region.north + region.south) / 2, (region.east + region.west) / 2, 7)),
        (12, *_tile(region.north - .001, region.east - .001, 12)),
    ]:
        rf, cf = coordinates._compute_region_pixel_indices_fractional_padded(region, z, x, y, size, pad)
        ri, ci = coordinates._compute_region_pixel_indices_padded(region, z, x, y, size, pad)
        r0, c0 = np.floor(rf).astype(np.int32), np.floor(cf).astype(np.int32)
        expected = (
            r0, np.minimum(r0 + 1, region.height - 1),
            c0, np.minimum(c0 + 1, region.width - 1),
            (rf - r0).astype(np.float32), (cf - c0).astype(np.float32),
            ri >= 0, ci >= 0,
        )
        project.reset_mock()
        plan = sampling.build_sampling_plan(region, z, x, y, size, pad)
        assert project.call_count == 1
        assert plan.shape == (size + 2 * pad,) * 2
        assert plan.source_shape == (region.height, region.width)
        assert not plan.separable
        for actual, original in zip(plan.arrays, expected):
            np.testing.assert_array_equal(actual, original, strict=True)
            assert actual.flags.c_contiguous and not actual.flags.writeable
        assert not np.shares_memory(plan.valid_rows, plan.valid_cols)


@pytest.mark.parametrize("size", [256, 512])
def test_render_bytes_match_full_map_radius(size, sample_frame_data, monkeypatch):
    from librewxr.tiles import renderer
    monkeypatch.setattr(settings, "smooth_radius", 1.0)
    regions = {"USCOMP": sample_frame_data}
    kwargs = dict(z=5, x=7, y=12, tile_size=size, smooth=True)
    actual = renderer.compute_tile_geometry(regions, **kwargs)
    monkeypatch.setattr(renderer, "compute_blur_radius", _full_map_radius)
    expected = renderer.compute_tile_geometry(regions, **kwargs)
    assert not actual.is_transparent
    assert actual.blur_radius == expected.blur_radius
    for fmt in ("png", "webp"):
        assert renderer.present_tile(actual, color_scheme=2, fmt=fmt) == renderer.present_tile(
            expected, color_scheme=2, fmt=fmt,
        )
