# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Compact coordinate storage and bit-exact regional/global nowcast warps."""
import cv2
import numpy as np
import pytest

from librewxr.data import nowcast as source

pytestmark = pytest.mark.nowcast


def _full_grid_warp(frame, flow, steps, max_px, wrap):
    """Reference the original full-size float32 coordinate calculation."""
    h, w = frame.shape
    if flow.shape[:2] != frame.shape:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    if max_px is not None:
        flow = source._clamp_flow(flow, max_px)
    if wrap:
        pad = min(source._WRAP_FLOW_PAD, w // 8)
        frame = np.pad(frame, ((0, 0), (pad, pad)), mode="wrap")
        flow = np.pad(flow, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
    ys, xs = np.mgrid[:frame.shape[0], :frame.shape[1]].astype(np.float32)
    result = cv2.remap(
        frame, xs - steps * flow[..., 0], ys - steps * flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP if wrap else cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return result[:, pad:pad + w] if wrap else result


def test_us_coordinate_storage_is_linear_in_height_and_width():
    ys, xs = source._coordinate_axes(5400, 12200)
    assert ys.shape == (5400, 1)
    assert xs.shape == (1, 12200)
    assert ys.dtype == xs.dtype == np.float32
    assert ys.nbytes + xs.nbytes == (5400 + 12200) * 4
    np.testing.assert_array_equal(ys[:, 0], np.arange(5400, dtype=np.float32))
    np.testing.assert_array_equal(xs[0], np.arange(12200, dtype=np.float32))


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (513, 79)])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("max_px", [None, 2.])
@pytest.mark.parametrize("steps", [1, 6])
@pytest.mark.parametrize("wrap", [False, True])
def test_warp_matches_full_grids(shape, reduced, max_px, steps, wrap):
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, shape, dtype=np.uint8)
    flow_shape = (max(1, shape[0] // 3), max(1, shape[1] // 3)) if reduced else shape
    flow = rng.uniform(-8, 8, (*flow_shape, 2)).astype(np.float32)
    original_frame, original_flow = frame.copy(), flow.copy()
    frame.flags.writeable = flow.flags.writeable = False
    expected = _full_grid_warp(frame, flow, steps, max_px, wrap)
    result = source._extrapolate_forward(frame, flow, steps, max_px=max_px, wrap=wrap)
    np.testing.assert_array_equal(result, expected)
    if not wrap:
        ys, xs = source._coordinate_axes(*shape)
        ys.flags.writeable = xs.flags.writeable = False
        cached = source._extrapolate_forward(frame, flow, steps, xs=xs, ys=ys, max_px=max_px)
        np.testing.assert_array_equal(cached, expected)
        # Existing callers can still pass complete precomputed grids.
        full_ys, full_xs = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
        dense = source._extrapolate_forward(frame, flow, steps, xs=full_xs, ys=full_ys, max_px=max_px)
        np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(frame, original_frame)
    np.testing.assert_array_equal(flow, original_flow)


@pytest.mark.parametrize("region", ["USCOMP", "RRQPE"])
def test_generation_reuses_small_axes_without_changing_six_forecast_frames(monkeypatch, region):
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (27, 35), dtype=np.uint8)
    flow = rng.uniform(-3, 3, (5, 7, 2)).astype(np.float32)
    frame.flags.writeable = flow.flags.writeable = False
    axes = []
    build = source._coordinate_axes

    def capture(h, w):
        ys, xs = build(h, w)
        assert ys.nbytes + xs.nbytes == (h + w) * 4
        ys.flags.writeable = xs.flags.writeable = False
        axes.append((ys, xs))
        return ys, xs

    monkeypatch.setattr(source, "_coordinate_axes", capture)
    monkeypatch.setattr(source, "_compute_region_flow", lambda name, *args: (name, flow, flow, None))
    monkeypatch.setattr(source.settings, "nowcast_coarsen_enabled", False)
    frames, flows = source.NowcastGenerator._generate_sync(
        {region: frame}, {region: frame}, 1000, 6, 600, "radar",
    )
    assert len(frames) == 6 and axes
    assert flows[region] is flow
    for step, generated in enumerate(frames, 1):
        assert generated.timestamp == 1000 + step * 600
        expected = _full_grid_warp(frame, flow, step, None, region == "RRQPE")
        np.testing.assert_array_equal(generated.regions[region], expected)
    for ys, xs in axes:
        np.testing.assert_array_equal(ys[:, 0], np.arange(len(ys), dtype=np.float32))
        np.testing.assert_array_equal(xs[0], np.arange(xs.shape[1], dtype=np.float32))
