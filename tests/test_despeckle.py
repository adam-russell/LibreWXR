# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cleanup output, strip-edge neighborhoods, input ownership and memory bounds."""
import numpy as np
import pytest

from librewxr.data import fetcher as source


def _reference_despeckle(data, min_neighbors):
    """Count each pixel's neighbors independently using the original input."""
    result = data.copy()
    height, width = data.shape
    for row, col in np.ndindex(data.shape):
        if data[row, col] == 0:
            continue
        neighbors = sum(
            data[y, x] > 0
            for y in range(max(0, row - 1), min(height, row + 2))
            for x in range(max(0, col - 1), min(width, col + 2))
            if (y, x) != (row, col)
        )
        if neighbors < min_neighbors:
            result[row, col] = 0
    return result


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (255, 11), (256, 11), (257, 11), (513, 11)])
@pytest.mark.parametrize("min_neighbors", [0, 1, 3, 8, 9])
def test_matches_neighbor_reference_across_strip_boundaries(shape, min_neighbors):
    rng = np.random.default_rng(52)
    data = rng.integers(1, 256, shape, dtype=np.uint8)
    data[rng.random(shape) < 0.7] = 0
    original = data.copy()
    data.flags.writeable = False
    result = source._despeckle(data, min_neighbors)
    np.testing.assert_array_equal(result, _reference_despeckle(data, min_neighbors))
    np.testing.assert_array_equal(data, original)
    assert result.dtype == np.uint8
    assert result.shape == shape
    assert not np.shares_memory(result, data)


@pytest.mark.parametrize("value", [0, 134])
def test_uniform_fields_keep_the_original_border_rules(value):
    data = np.full((513, 5), value, dtype=np.uint8)
    result = source._despeckle(data, 8)
    expected = data.copy()
    expected[[0, -1], :] = 0
    expected[:, [0, -1]] = 0
    np.testing.assert_array_equal(result, expected)


def test_neighbors_across_strip_edges_and_strided_inputs():
    data = np.zeros((520, 19), dtype=np.uint8)
    data[254:258, 3:7] = 134
    data[510:514, 14:18] = 114
    for view in (data, data[:, ::2], data[::-1], data.T):
        original = view.copy()
        view.flags.writeable = False
        result = source._despeckle(view, 3)
        np.testing.assert_array_equal(result, _reference_despeckle(view, 3))
        np.testing.assert_array_equal(view, original)
        assert not np.shares_memory(result, view)
    # These interior pixels need the neighbor row across each strip boundary.
    result = source._despeckle(data, 8)
    assert np.all(result[255:257, 4:6] == 134)
    assert np.all(result[511:513, 15:17] == 114)


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (5, 0)])
def test_empty_fields(shape):
    data = np.empty(shape, dtype=np.uint8)
    result = source._despeckle(data, 3)
    assert result.shape == shape
    assert result.dtype == np.uint8


def test_neighbor_workspaces_are_strip_sized(monkeypatch):
    data = np.full((513, 9), 134, dtype=np.uint8)
    data.flags.writeable = False
    block_shapes = []
    original = source._despeckle_block

    def process_block(block, min_neighbors):
        block_shapes.append(block.shape)
        assert block.shape[0] <= 258  # 256 output rows plus neighbor rows.
        assert np.shares_memory(block, data)
        return original(block, min_neighbors)

    monkeypatch.setattr(source, "_despeckle_block", process_block)
    result = source._despeckle(data, 3)
    assert block_shapes == [(257, 9), (258, 9), (2, 9)]
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("layout", ["fortran", "memmap"])
@pytest.mark.parametrize("min_neighbors", [0, 1, 3, 8, 9, 255])
def test_native_counting_preserves_read_only_inputs_and_strip_halos(tmp_path, layout, min_neighbors):
    rng = np.random.default_rng(1772)
    data = rng.integers(1, 256, (520, 19), dtype=np.uint8)
    data[rng.random(data.shape) < 0.8] = 0
    data[254:258, 3:7] = 134
    data[510:514, 12:16] = 114
    data[0, 0] = data[-1, -1] = 211
    if layout == "fortran":
        array = np.asfortranarray(data)
    else:
        path = tmp_path / "radar.npy"
        np.save(path, data)
        array = np.load(path, mmap_mode="r")
    array.flags.writeable = False
    actual = source._despeckle(array, min_neighbors)
    np.testing.assert_array_equal(actual, _reference_despeckle(data, min_neighbors))
    np.testing.assert_array_equal(array, data)
    assert actual.dtype == np.uint8 and actual.shape == data.shape
    assert actual.flags.writeable
    assert not np.shares_memory(actual, array)
