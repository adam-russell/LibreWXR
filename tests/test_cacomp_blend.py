# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canadian source priority, extent boundaries and input ownership."""
import numpy as np
import pytest

from librewxr.data import fetcher as source
from librewxr.data.regions import RegionDef


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided", "memmap"])
@pytest.mark.parametrize("coverage", ["mixed", "mrms_missing", "msc_missing", "both_missing"])
def test_blend_priority_and_extent_edges_preserve_read_only_inputs(tmp_path, monkeypatch, layout, coverage):
    # Use exact synthetic bounds to exercise inclusive edges. The real
    # source bounds include pixel-center offsets rather than whole degrees.
    monkeypatch.setattr(source, "MRMS_EXTENTS", {"USCOMP": (20, 55, -130, -60)})
    # The extent occupies rows/columns 1..8, with one outside each side.
    region = RegionDef(
        name="BLENDTEST", west=-140, east=-40, south=12.5, north=62.5,
        pixel_size=10, pixel_size_y=5, group="CANADA",
    )
    mrms_values = np.full((10, 10), 134, dtype=np.uint8)
    msc_values = np.full((10, 10), 93, dtype=np.uint8)
    mrms_values[::2, ::2] = 0
    msc_values[4:7, 4:7] = 0
    if coverage in ("mrms_missing", "both_missing"):
        mrms_values.fill(0)
    if coverage in ("msc_missing", "both_missing"):
        msc_values.fill(0)

    def input_array(values, name):
        if layout == "fortran":
            array = np.asfortranarray(values)
        elif layout == "strided":
            backing = np.zeros((20, 20), dtype=np.uint8)
            array = backing[::2, ::2]
            array[:] = values
        elif layout == "memmap":
            path = tmp_path / (name + ".npy")
            np.save(path, values)
            array = np.load(path, mmap_mode="r")
        else:
            array = values.copy()
        array.flags.writeable = False
        return array

    mrms = input_array(mrms_values, "mrms")
    msc = input_array(msc_values, "msc")
    actual = source._blend_cacomp_arrays(mrms, msc, region)

    expected = msc_values.copy()
    for row in range(1, 9):
        for col in range(1, 9):
            if mrms_values[row, col] > 0:
                expected[row, col] = mrms_values[row, col]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mrms, mrms_values)
    np.testing.assert_array_equal(msc, msc_values)
    assert actual.dtype == np.uint8 and actual.shape == (10, 10)
    assert actual.flags.writeable
    assert not np.shares_memory(actual, mrms)
    assert not np.shares_memory(actual, msc)
