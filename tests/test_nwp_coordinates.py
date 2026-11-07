# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Compact model coordinates preserve rain/snow interpolation and ownership."""
import numpy as np
import pytest

from librewxr.data import nwp_interpolation as source


@pytest.fixture(autouse=True)
def isolated_coordinate_cache(monkeypatch):
    monkeypatch.setattr(source, "_GRID_CACHE", {})


def _full_grid(h, w):
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    return ys, xs


def test_ifs_axes_are_small_and_reused_by_shape():
    axes = source._coordinate_grid(1801, 3600)
    ys, xs = axes
    assert ys.shape == (1801, 1) and xs.shape == (1, 3600)
    assert ys.dtype == xs.dtype == np.float32
    assert ys.nbytes + xs.nbytes == (1801 + 3600) * 4
    assert source._coordinate_grid(1801, 3600) is axes
    np.testing.assert_array_equal(ys[:, 0], np.arange(1801, dtype=np.float32))
    np.testing.assert_array_equal(xs[0], np.arange(3600, dtype=np.float32))


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (127, 239)])
@pytest.mark.parametrize("fraction", [0., .01, .5, .9, 1.])
def test_pair_and_forward_helpers_match_full_coordinates(monkeypatch, shape, fraction):
    rng = np.random.default_rng(42)
    a = rng.integers(0, 256, shape, dtype=np.uint8)
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    flow = rng.uniform(-8, 8, (*shape, 2)).astype(np.float32)
    originals = [value.copy() for value in [a, b, flow]]
    for value in [a, b, flow]:
        value.flags.writeable = False
    ys, xs = source._coordinate_grid(*shape)
    ys.flags.writeable = xs.flags.writeable = False
    actual_pair, pair_flow = source.interpolate_pair_at_fraction(a, b, fraction, flow=flow)
    actual_forward, forward_flow = source.extrapolate_forward(a, b, fraction, flow=flow)
    monkeypatch.setattr(source, "_coordinate_grid", _full_grid)
    expected_pair, _ = source.interpolate_pair_at_fraction(a, b, fraction, flow=flow)
    expected_forward, _ = source.extrapolate_forward(a, b, fraction, flow=flow)
    np.testing.assert_array_equal(actual_pair, expected_pair)
    np.testing.assert_array_equal(actual_forward, expected_forward)
    assert pair_flow is forward_flow is flow
    for value, original in zip([a, b, flow], originals):
        np.testing.assert_array_equal(value, original)
    np.testing.assert_array_equal(ys[:, 0], np.arange(shape[0], dtype=np.float32))
    np.testing.assert_array_equal(xs[0], np.arange(shape[1], dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8])
@pytest.mark.parametrize("fraction", [.01, .25, .5, .9])
def test_snow_warp_preserves_masks_and_dtype(dtype, fraction):
    rng = np.random.default_rng(42)
    a = (rng.random((37, 53)) < .4).astype(dtype)
    b = (rng.random((37, 53)) < .4).astype(dtype)
    a.flags.writeable = b.flags.writeable = False
    flow = rng.uniform(-8, 8, (*a.shape, 2)).astype(np.float32)
    ys, xs = source._coordinate_grid(*a.shape)
    full_ys, full_xs = _full_grid(*a.shape)
    result = source._interpolate_snow(a, b, flow, fraction, xs, ys)
    expected = source._interpolate_snow(a, b, flow, fraction, full_xs, full_ys)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == dtype


@pytest.mark.parametrize("snow_dtype", [None, np.bool_, np.uint8])
def test_complete_hourly_run_matches_dense_coordinates_and_preserves_endpoints(monkeypatch, snow_dtype):
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (37, 53), dtype=np.uint8)
    frames = {0: frame, 3600: np.roll(frame, 3, axis=1)}
    snow = None if snow_dtype is None else {t: (value > 127).astype(snow_dtype) for t, value in frames.items()}
    inputs = list(frames.values()) + ([] if snow is None else list(snow.values()))
    originals = [value.copy() for value in inputs]
    for value in inputs:
        value.flags.writeable = False
    flow = rng.uniform(-3, 3, (*frame.shape, 2)).astype(np.float32)
    flow.flags.writeable = False
    monkeypatch.setattr(source, "_compute_flow", lambda *args: flow)
    ys, xs = source._coordinate_grid(*frame.shape)
    ys.flags.writeable = xs.flags.writeable = False
    actual = source.interpolate_run(frames, snow)
    monkeypatch.setattr(source, "_coordinate_grid", _full_grid)
    expected = source.interpolate_run(frames, snow)
    assert sorted(actual[0]) == list(range(0, 3601, 600))
    assert actual[2] is expected[2] is flow
    for result_frames, reference_frames in zip(actual[:2], expected[:2]):
        if reference_frames is None:
            assert result_frames is None
        else:
            assert set(result_frames) == set(reference_frames)
            for t in reference_frames:
                np.testing.assert_array_equal(result_frames[t], reference_frames[t])
                assert result_frames[t].dtype == reference_frames[t].dtype
    for t in frames:
        assert actual[0][t] is frames[t]
        if snow is not None:
            assert actual[1][t] is snow[t]
    for value, original in zip(inputs, originals):
        np.testing.assert_array_equal(value, original)
