# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Preserve continuation padding, corner arithmetic and MRMS output bytes."""
import numpy as np
import pytest
import xarray as xr

from librewxr.data.regions import RegionDef
from librewxr.sources.regional.north_america.usa.radar.mrms import source

pytestmark = pytest.mark.sources


def _concatenated_padding(arr, top, bottom, left, right):
    """Reference the original sequential row/column padding implementation."""
    if top:
        k = (top - np.arange(top, dtype=np.float32))[:, None]
        arr = np.concatenate([(1 + k) * arr[:1] - k * arr[1:2], arr], axis=0)
    if bottom:
        arr = np.concatenate([arr, np.repeat(arr[-1:], bottom, axis=0)], axis=0)
    if left:
        k = (left - np.arange(left, dtype=np.float32))[None, :]
        arr = np.concatenate([(1 + k) * arr[:, :1] - k * arr[:, 1:2], arr], axis=1)
    if right:
        arr = np.concatenate([arr, np.repeat(arr[:, -1:], right, axis=1)], axis=1)
    return arr


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
@pytest.mark.parametrize("flags", range(16))
def test_padding_combinations_match_original_corner_arithmetic(dtype, flags):
    rng = np.random.default_rng(42)
    pads = [size if flags & (1 << i) else 0 for i, size in enumerate([13, 17, 19, 23])]
    for shape in [(2, 2), (37, 53), (257, 79)]:
        data = rng.uniform(-33, 65, shape).astype(dtype)
        if np.issubdtype(dtype, np.floating):
            data[1, 1] = np.nan
        original = data.copy()
        data.flags.writeable = False
        expected = _concatenated_padding(data, *pads)
        result = source._pad_continuation(data, *pads)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(data, original)
        assert result.dtype == expected.dtype
        if any(pads):
            assert not np.shares_memory(result, data)
        else:
            assert result is data


def test_linear_top_left_and_replicated_bottom_right():
    data = np.array([[10, 20], [30, 40]], dtype=np.float32)
    result = source._pad_continuation(data, 1, 1, 1, 1)
    np.testing.assert_array_equal(result, [
        [-20, -10, 0, 0], [0, 10, 20, 20],
        [20, 30, 40, 40], [20, 30, 40, 40],
    ])


@pytest.mark.parametrize("height", [257, 513])
@pytest.mark.parametrize("crop", [False, True])
def test_resampling_matches_original_padding_with_nodata_and_outside_edges(monkeypatch, height, crop):
    rng = np.random.default_rng(42)
    data = rng.uniform(-33, 65, (height + 8, 100)).astype(np.float32)
    data[rng.random(data.shape) < .1] = -999
    data[2, 2] = np.nan
    original = data.copy()
    data.flags.writeable = False
    ds = xr.Dataset(
        {"r": (("latitude", "longitude"), data)},
        coords={"latitude": np.arange(height + 7., -1, -1), "longitude": np.arange(100.)},
    )
    bounds = (10.25, 30.75, 15.25, height - 7.75) if crop else (-.25, 100.75, -.25, height + 7.75)
    region = RegionDef(f"PAD_{height}_{crop}", *bounds, .5, "US")
    with np.errstate(invalid="ignore"):
        result = source._resample_mrms_to_region(ds, region)
        monkeypatch.setattr(source, "_pad_continuation", _concatenated_padding)
        expected = source._resample_mrms_to_region(ds, region)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(data, original)
