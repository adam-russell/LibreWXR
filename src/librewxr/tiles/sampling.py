# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reusable radar interpolation geometry, with an optional native executor."""
from collections import OrderedDict
from dataclasses import dataclass
import math
from threading import Lock

import numpy as np

from librewxr.data.regions import RegionDef
from librewxr.tiles import coordinates

try:
    from librewxr.tiles import _sampling_native
except ImportError:
    _sampling_native = None


@dataclass(frozen=True)
class SamplingPlan:
    """Read-only indices/weights; regular grids retain only two compact axes."""

    r0: np.ndarray
    r1: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    dr: np.ndarray
    dc: np.ndarray
    valid_rows: np.ndarray
    valid_cols: np.ndarray
    shape: tuple[int, int]
    source_shape: tuple[int, int]
    separable: bool

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.r0, self.r1, self.c0, self.c1, self.dr, self.dc,
            self.valid_rows, self.valid_cols,
        )

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.arrays)


def build_sampling_plan(
    region: RegionDef, z: int, x: int, y: int, tile_size: int, pad: int,
) -> SamplingPlan:
    """Preserve the renderer's float64 coordinate math and float32 weights."""
    if tile_size <= 0 or pad < 0 or region.height <= 0 or region.width <= 0:
        raise ValueError("Invalid sampling dimensions")
    separable = region.proj not in ("laea", "tmerc")
    if separable:
        axis = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5
        n = 2**z
        lon = (x + axis / tile_size) / n * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * (y + axis / tile_size) / n))))
        raw_r = (region.north - lat) / region._ps_y
        raw_c = (lon - region.west) / region.pixel_size
        rounded_r, rounded_c = np.rint(raw_r), np.rint(raw_c)
        valid_rows = (rounded_r >= 0) & (rounded_r < region.height)
        valid_cols = (rounded_c >= 0) & (rounded_c < region.width)
        row_f = np.clip(raw_r, 0, region.height - 1).astype(np.float32)
        col_f = np.clip(raw_c, 0, region.width - 1).astype(np.float32)
    else:
        # Projected coordinates are not separable. Build their exact existing
        # maps without also pinning a duplicate in the coordinate LRUs.
        if pad:
            row_f, col_f = coordinates._compute_region_pixel_indices_fractional_padded(
                region, z, x, y, tile_size, pad,
            )
            row_i, col_i = coordinates._compute_region_pixel_indices_padded(
                region, z, x, y, tile_size, pad,
            )
        else:
            row_f, col_f = coordinates._compute_region_pixel_indices_fractional(
                region, z, x, y, tile_size,
            )
            row_i, col_i = coordinates._compute_region_pixel_indices(region, z, x, y, tile_size)
        valid_rows, valid_cols = row_i >= 0, col_i >= 0

    r0 = np.floor(row_f).astype(np.int32)
    c0 = np.floor(col_f).astype(np.int32)
    arrays = (
        r0, np.minimum(r0 + 1, region.height - 1),
        c0, np.minimum(c0 + 1, region.width - 1),
        (row_f - r0).astype(np.float32), (col_f - c0).astype(np.float32),
        valid_rows, valid_cols,
    )
    arrays = tuple(np.ascontiguousarray(array) for array in arrays)
    for array in arrays:
        array.flags.writeable = False
    size = tile_size + 2 * pad
    return SamplingPlan(*arrays, (size, size), (region.height, region.width), separable)


class SamplingPlanCache:
    """Per-worker LRU bounded by both array bytes and entry count."""

    def __init__(self, max_bytes: int = 32 * 1024**2, max_entries: int = 32):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("Sampling cache limits must be nonnegative")
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._items: OrderedDict[tuple, SamplingPlan] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._generation = 0
        self._lock = Lock()

    def get(
        self, region: RegionDef, z: int, x: int, y: int, tile_size: int, pad: int,
    ) -> SamplingPlan:
        # The frozen RegionDef includes projection parameters, dimensions and
        # resolution. Weather/timestamp changes do not invalidate this geometry.
        key = (region, z, x, y, tile_size, pad)
        with self._lock:
            if key in self._items:
                self._hits += 1
                self._items.move_to_end(key)
                return self._items[key]
            self._misses += 1
            generation = self._generation
        plan = build_sampling_plan(*key)
        with self._lock:
            if generation != self._generation:
                return plan  # An in-flight build must not undo pressure cleanup.
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            if self._max_entries and plan.nbytes <= self._max_bytes:
                self._items[key] = plan
                self._bytes += plan.nbytes
                while self._bytes > self._max_bytes or len(self._items) > self._max_entries:
                    _, old = self._items.popitem(last=False)
                    self._bytes -= old.nbytes
        return plan

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = self._hits = self._misses = 0
            self._generation += 1

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._items), "hits": self._hits, "misses": self._misses,
                "hit_ratio": round(self._hits / total, 3) if total else None,
                "bytes": self._bytes, "max_bytes": self._max_bytes,
                "max_entries": self._max_entries, "native": _sampling_native is not None,
            }


sampling_plans = SamplingPlanCache()


def _sample_numpy(data: np.ndarray, plan: SamplingPlan) -> np.ndarray:
    r0, r1, c0, c1, dr, dc = plan.r0, plan.r1, plan.c0, plan.c1, plan.dr, plan.dc
    if plan.separable:
        r0, r1, dr = r0[:, None], r1[:, None], dr[:, None]
    v00 = data[r0, c0].astype(np.float32)
    v01 = data[r0, c1].astype(np.float32)
    v10 = data[r1, c0].astype(np.float32)
    v11 = data[r1, c1].astype(np.float32)
    any_zero = (v00 == 0) | (v01 == 0) | (v10 == 0) | (v11 == 0)
    interp = (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )
    result = np.clip(np.where(any_zero, v00, interp) + 0.5, 0, 255).astype(np.uint8)
    valid = (
        plan.valid_rows[:, None] & plan.valid_cols[None, :]
        if plan.separable else plan.valid_rows & plan.valid_cols
    )
    result[~valid] = 0
    return result


def sample_with_plan(data: np.ndarray, plan: SamplingPlan) -> np.ndarray:
    """Return an owned, writable tile; never copy or mutate the source frame."""
    if data.ndim != 2 or data.shape != plan.source_shape:
        raise ValueError("Frame dimensions do not match the sampling plan")
    if _sampling_native is None or data.dtype != np.uint8:
        return _sample_numpy(data, plan)
    result = np.empty(plan.shape, dtype=np.uint8)
    _sampling_native.sample(data, *plan.arrays, result, plan.separable)
    return result


def sample_bilinear(
    data: np.ndarray, region: RegionDef, z: int, x: int, y: int,
    tile_size: int, pad: int = 0,
) -> np.ndarray:
    return sample_with_plan(data, sampling_plans.get(region, z, x, y, tile_size, pad))
