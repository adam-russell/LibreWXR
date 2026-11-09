# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact rendering, native-buffer safety, and plan-cache ownership regressions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
from threading import Event

import numpy as np
import pytest

from librewxr.data.regions import RegionDef
from librewxr.tiles import coordinates, sampling


def _tile(lat, lon, z):
    return int((lon + 180) / 360 * 2**z), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**z)


def _region(projection):
    return RegionDef(
        "test", -5, 5, 45, 55, .125, "TEST", pixel_size_y=.15625,
        proj=projection, grid_width=80, grid_height=64,
        grid_x_min=-40000, grid_y_max=32000, grid_scale=1000,
        laea_lat0=50, laea_lon0=0, tmerc_lat0=50, tmerc_lon0=0,
    )


def _reference(data, region, z, x, y, size, pad):
    """Original renderer's map, gather, arithmetic, and OOB mask sequence."""
    if pad:
        rf, cf = coordinates._compute_region_pixel_indices_fractional_padded(region, z, x, y, size, pad)
        ri, ci = coordinates._compute_region_pixel_indices_padded(region, z, x, y, size, pad)
    else:
        rf, cf = coordinates._compute_region_pixel_indices_fractional(region, z, x, y, size)
        ri, ci = coordinates._compute_region_pixel_indices(region, z, x, y, size)
    r0, c0 = np.floor(rf).astype(np.int32), np.floor(cf).astype(np.int32)
    r1, c1 = np.minimum(r0 + 1, region.height - 1), np.minimum(c0 + 1, region.width - 1)
    dr, dc = (rf - r0).astype(np.float32), (cf - c0).astype(np.float32)
    a, b = data[r0, c0].astype(np.float32), data[r0, c1].astype(np.float32)
    c, d = data[r1, c0].astype(np.float32), data[r1, c1].astype(np.float32)
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    mixed = a * (1 - dr) * (1 - dc) + b * (1 - dr) * dc + c * dr * (1 - dc) + d * dr * dc
    result = np.clip(np.where(zero, a, mixed) + .5, 0, 255).astype(np.uint8)
    result[(ri == -1) | (ci == -1)] = 0
    return result


@pytest.fixture(params=["numpy", "native"])
def backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(sampling, "_sampling_native", None)
    elif sampling._sampling_native is None:
        pytest.skip("Optional native sampler was not built")
    return request.param


@pytest.mark.parametrize("projection", ["latlon", "laea", "tmerc"])
@pytest.mark.parametrize("size,pad", [(32, 0), (64, 3), (256, 0), (512, 48)])
@pytest.mark.parametrize("location", [(50, 0, 8), (54.8, -4.9, 7), (45.1, 4.9, 7), (50, 0, 0)])
def test_matches_original_sampler(backend, projection, size, pad, location):
    region = _region(projection)
    rng = np.random.default_rng(20261109)
    data = rng.integers(0, 256, (region.height, region.width), dtype=np.uint8)
    data[::4, ::3] = 0
    data.flags.writeable = False
    before = data.copy()
    lat, lon, z = location
    x, y = _tile(lat, lon, z)
    plan = sampling.build_sampling_plan(region, z, x, y, size, pad)
    expected = _reference(data, region, z, x, y, size, pad)
    actual = sampling.sample_with_plan(data, plan)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, before)
    assert actual.flags.owndata and actual.flags.writeable
    assert all(not a.flags.writeable and a.flags.c_contiguous for a in plan.arrays)
    assert all(a.ndim == (1 if projection == "latlon" else 2) for a in plan.arrays)


@pytest.mark.parametrize("layout", ["C", "F", "reversed", "strided", "memmap"])
def test_source_layouts_and_owned_outputs(backend, layout, tmp_path):
    region = _region("latlon")
    base = np.random.default_rng(3).integers(0, 256, (64, 160), dtype=np.uint8)
    if layout == "C":
        data = base[:, :80].copy()
    elif layout == "F":
        data = np.asfortranarray(base[:, :80])
    elif layout == "reversed":
        data = base[::-1, 79::-1]
    elif layout == "strided":
        data = base[:, ::2]
    else:
        path = tmp_path / "frame.dat"
        path.write_bytes(base[:, :80].copy().tobytes())
        data = np.memmap(path, shape=(64, 80), dtype=np.uint8, mode="r")
    data.flags.writeable = False
    x, y = _tile(50, 0, 8)
    plan = sampling.build_sampling_plan(region, 8, x, y, 64, 3)
    first = sampling.sample_with_plan(data, plan)
    second = sampling.sample_with_plan(data, plan)
    np.testing.assert_array_equal(first, _reference(data, region, 8, x, y, 64, 3))
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, data)


def test_native_rounding_boundaries_and_zero_neighbors(backend):
    rng = np.random.default_rng(14)
    data = rng.integers(0, 256, (257, 263), dtype=np.uint8)
    shape = (512, 512)
    r0 = rng.integers(0, 256, shape, dtype=np.int32)
    c0 = rng.integers(0, 262, shape, dtype=np.int32)
    fractions = np.array([0, .25, .5, .75, 1, np.nextafter(np.float32(.5), np.float32(0)), np.nextafter(np.float32(.5), np.float32(1))], dtype=np.float32)
    dr, dc = rng.choice(fractions, shape), rng.choice(fractions, shape)
    valid = np.ones(shape, dtype=bool)
    plan = sampling.SamplingPlan(r0, r0 + 1, c0, c0 + 1, dr, dc, valid, valid, shape, data.shape, False)
    expected = sampling._sample_numpy(data, plan)
    np.testing.assert_array_equal(sampling.sample_with_plan(data, plan), expected)


def test_degenerate_source_axis(backend):
    region = replace(_region("latlon"), grid_width=1, grid_height=1)
    data = np.array([[255]], dtype=np.uint8)
    x, y = _tile(54.98, -4.98, 12)
    plan = sampling.build_sampling_plan(region, 12, x, y, 64, 0)
    np.testing.assert_array_equal(sampling.sample_with_plan(data, plan), _reference(data, region, 12, x, y, 64, 0))


def test_cache_bounds_reuse_and_layout_keys():
    cache = sampling.SamplingPlanCache(max_bytes=10000, max_entries=2)
    region = _region("latlon")
    a = cache.get(region, 7, 60, 40, 256, 0)
    assert cache.get(region, 7, 60, 40, 256, 0) is a
    for x in range(61, 70):
        cache.get(region, 7, x, 40, 256, 0)
    stats = cache.stats()
    assert stats["entries"] <= 2 and stats["bytes"] <= 10000
    assert cache.get(replace(region, pixel_size_y=.2), 7, 69, 40, 256, 0) is not cache.get(region, 7, 69, 40, 256, 0)
    disabled = sampling.SamplingPlanCache(max_bytes=0)
    disabled.get(region, 7, 60, 40, 256, 0)
    assert disabled.stats()["entries"] == 0
    assert a.nbytes == 26 * 256


def test_clear_during_build_does_not_repopulate(monkeypatch):
    cache = sampling.SamplingPlanCache()
    started, release = Event(), Event()
    original = sampling.build_sampling_plan
    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(sampling, "build_sampling_plan", blocked)
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(cache.get, _region("latlon"), 7, 60, 40, 32, 0)
        assert started.wait(5)
        cache.clear()
        release.set()
        assert future.result().shape == (32, 32)
    assert cache.stats()["entries"] == 0


def test_concurrent_sampling_shares_only_geometry(backend):
    cache = sampling.SamplingPlanCache()
    region = _region("latlon")
    data = np.full((64, 80), 130, dtype=np.uint8)
    data.flags.writeable = False
    def call(_):
        p = cache.get(region, 7, 64, 40, 64, 0)
        return sampling.sample_with_plan(data, p)
    with ThreadPoolExecutor(4) as executor:
        outputs = list(executor.map(call, range(16)))
    for output in outputs:
        np.testing.assert_array_equal(output, outputs[0])
    assert len({id(a) for a in outputs}) == 16
    assert cache.stats()["entries"] == 1


def test_pressure_cleanup_and_memory_accounting(monkeypatch):
    from librewxr.main import _clear_coord_caches
    cache = sampling.SamplingPlanCache()
    monkeypatch.setattr(sampling, "sampling_plans", cache)
    plan = cache.get(_region("latlon"), 7, 64, 40, 256, 0)
    assert coordinates.coord_cache_bytes() >= plan.nbytes
    assert coordinates.coord_cache_stats()["caches"]["sampling_plans"]["bytes"] == plan.nbytes
    _clear_coord_caches()
    assert cache.stats()["bytes"] == 0


@pytest.mark.parametrize("bad", ["index", "weight", "dtype", "shape", "readonly_output"])
def test_native_rejects_malformed_buffers(bad):
    native = sampling._sampling_native
    if native is None:
        pytest.skip("Optional native sampler was not built")
    data = np.full((64, 80), 100, dtype=np.uint8)
    plan = sampling.build_sampling_plan(_region("latlon"), 8, *_tile(50, 0, 8), 64, 0)
    arrays = [a.copy() for a in plan.arrays]
    arrays[6][:] = arrays[7][:] = True
    output = np.empty(plan.shape, dtype=np.uint8)
    if bad == "index":
        arrays[0][0] = -1
    elif bad == "weight":
        arrays[4][0] = np.nan
    elif bad == "dtype":
        arrays[0] = arrays[0].astype(np.int64)
    elif bad == "shape":
        arrays[0] = arrays[0][:-1]
    else:
        output.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        native.sample(data, *arrays, output, plan.separable)
