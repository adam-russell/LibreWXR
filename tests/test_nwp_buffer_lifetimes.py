# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interpolation workspaces are released after use without changing output."""
import weakref

import numpy as np
import pytest

from librewxr.data import nwp_interpolation as source


@pytest.mark.parametrize("kind", ["precip", "snow_bool", "snow_uint8"])
@pytest.mark.parametrize("fraction", [.25, .5, .75])
def test_first_warp_workspaces_are_released_before_second_warp(monkeypatch, kind, fraction):
    shape = (37, 53)
    if kind == "precip":
        a = np.full(shape, 80, dtype=np.uint8)
        b = np.full(shape, 120, dtype=np.uint8)
        expected_value = int((1 - fraction) * 80 + fraction * 120 + .5)
        expected_dtype = np.uint8
        function = source._interpolate_precip
    else:
        expected_dtype = np.bool_ if kind == "snow_bool" else np.uint8
        a = np.zeros(shape, dtype=expected_dtype)
        b = np.ones(shape, dtype=expected_dtype)
        expected_value = fraction > .5
        function = source._interpolate_snow
    originals = a.copy(), b.copy()
    a.flags.writeable = b.flags.writeable = False
    flow = np.zeros((*shape, 2), dtype=np.float32)
    flow.flags.writeable = False
    ys = np.arange(shape[0], dtype=np.float32)[:, None]
    xs = np.arange(shape[1], dtype=np.float32)[None, :]
    ys.flags.writeable = xs.flags.writeable = False
    remap = source.cv2.remap
    references = []

    def trace(array, map_x, map_y, *args, **kwargs):
        if references:
            assert references[0][1]() is None
            assert references[0][2]() is None
            if kind != "precip":
                # The private float conversion must also be gone; the
                # caller's original snow arrays stay alive and unchanged.
                assert references[0][0]() is None
        references.append(tuple(weakref.ref(value) for value in (array, map_x, map_y)))
        return remap(array, map_x, map_y, *args, **kwargs)

    monkeypatch.setattr(source.cv2, "remap", trace)
    result = function(a, b, flow, fraction, xs, ys)
    assert len(references) == 2
    assert all(ref() is None for refs in references for ref in refs[1:])
    if kind != "precip":
        assert all(refs[0]() is None for refs in references)
    np.testing.assert_array_equal(result, np.full(shape, expected_value, dtype=expected_dtype))
    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(a, originals[0])
    np.testing.assert_array_equal(b, originals[1])
    assert not np.shares_memory(result, a)
    assert not np.shares_memory(result, b)
