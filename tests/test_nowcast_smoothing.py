# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact smoothing output and private float-buffer reuse."""
import cv2
import numpy as np
import pytest

from librewxr.data import nowcast as source

pytestmark = pytest.mark.nowcast


def _separate_buffer_reference(frame, sigma, wrap):
    """The original float32 blur, rounding and clipping sequence."""
    if sigma < .4:
        return frame
    working = frame.astype(np.float32)
    if wrap:
        pad = min(int(np.ceil(sigma * 3)), max(1, frame.shape[1] // 8))
        working = np.pad(working, ((0, 0), (pad, pad)), mode="wrap")
    working = cv2.GaussianBlur(working, (0, 0), sigma)
    if wrap:
        working = working[:, pad:pad + frame.shape[1]]
    return np.rint(working).clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize("shape", [(1, 1), (1, 31), (31, 1), (7, 9), (127, 239)])
@pytest.mark.parametrize("sigma", [.399, .4, .6756756757, 3., 5.4054054054])
@pytest.mark.parametrize("wrap", [False, True])
def test_matches_original_rounding_and_borders_on_readonly_strided_inputs(shape, sigma, wrap):
    rng = np.random.default_rng(42)
    for pattern in ["zero", "full", "random"]:
        frame = rng.integers(0, 256, shape, dtype=np.uint8)
        if pattern != "random":
            frame[:] = 0 if pattern == "zero" else 255
        frame = frame[::-1, ::-1]
        original = frame.copy()
        frame.flags.writeable = False
        expected = _separate_buffer_reference(frame, sigma, wrap)
        result = source._coarsen_frame(frame, sigma, wrap)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(frame, original)
        assert result.shape == shape and result.dtype == np.uint8
        if sigma < .4:
            assert result is frame
        else:
            assert not np.shares_memory(result, frame)


@pytest.mark.parametrize("wrap", [False, True])
def test_readonly_memmap_is_unchanged(tmp_path, wrap):
    path = tmp_path / "frame.dat"
    original = np.random.default_rng(42).integers(0, 256, (37, 53), dtype=np.uint8)
    path.write_bytes(original.tobytes())
    frame = np.memmap(path, dtype=np.uint8, mode="r", shape=original.shape)
    expected = _separate_buffer_reference(frame, 3., wrap)
    result = source._coarsen_frame(frame, 3., wrap)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(frame, original)
    assert path.read_bytes() == original.tobytes()


@pytest.mark.parametrize("wrap", [False, True])
def test_filter_round_and_clip_reuse_private_storage(monkeypatch, wrap):
    frame = np.random.default_rng(42).integers(0, 256, (37, 53), dtype=np.uint8)
    frame.flags.writeable = False
    expected = _separate_buffer_reference(frame, 3., wrap)
    blur, rint, clip = cv2.GaussianBlur, np.rint, np.clip
    buffers = []

    def checked_blur(array, *args, **kwargs):
        assert not np.shares_memory(array, frame)
        assert array.dtype == np.float32
        assert kwargs.get("dst") is array
        buffers.append(array)
        return blur(array, *args, **kwargs)

    def checked_rint(array, *args, **kwargs):
        assert kwargs.get("out") is array
        assert np.shares_memory(array, buffers[0])
        return rint(array, *args, **kwargs)

    def checked_clip(array, *args, **kwargs):
        assert kwargs.get("out") is array
        assert np.shares_memory(array, buffers[0])
        return clip(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(source.cv2, "GaussianBlur", checked_blur)
        patch.setattr(source.np, "rint", checked_rint)
        patch.setattr(source.np, "clip", checked_clip)
        result = source._coarsen_frame(frame, 3., wrap)
    assert buffers
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("layout", ["transposed", "fortran"])
def test_transposed_and_fortran_inputs_match_original_output(layout, wrap):
    data = np.random.default_rng(42).integers(0, 256, (37, 53), dtype=np.uint8)
    frame = data.T if layout == "transposed" else np.asfortranarray(data)
    assert frame.flags.f_contiguous and not frame.flags.c_contiguous
    original = frame.copy()
    frame.flags.writeable = False
    expected = _separate_buffer_reference(frame, 3., wrap)
    result = source._coarsen_frame(frame, 3., wrap)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(frame, original)
    assert not np.shares_memory(result, frame)
