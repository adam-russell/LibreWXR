# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-processing contention, ordering, failure and cancellation checks."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from librewxr.data import fetcher as fetcher_module
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.regions import RegionDef
from librewxr.data.store import FrameStore
from librewxr.tiles.cache import TileCache


@pytest.fixture
def fetcher(monkeypatch, tmp_path):
    # Exercise normal initialization without constructing external providers.
    monkeypatch.setattr(fetcher_module, "collect_radar_contributions", lambda _: [])
    monkeypatch.setattr(fetcher_module.settings, "despeckle_min_neighbors", 3)
    instance = RadarFetcher(FrameStore(max_frames=12, cache_dir=tmp_path), TileCache(max_mb=1))
    region = RegionDef("TESTREG", 0, 4, 0, 4, 1, "US")
    instance._enabled_regions = [region]

    async def fetch_frame(region, minutes_ago):
        return np.full((4, 4), 50 + minutes_ago, dtype=np.uint8)

    instance._sources = {region.name: SimpleNamespace(fetch_frame=fetch_frame)}
    yield instance
    instance._store.cleanup()


def finalize(fetcher, value):
    region = fetcher._enabled_regions[0]
    return fetcher._finalize_result(region, 1000 + value * 600, "live", value, None)


@pytest.mark.parametrize("stage", ["fallback", "blend"])
async def test_two_slots_shared_across_batches_preserve_frame_order(fetcher, monkeypatch, stage):
    entered = asyncio.Queue()
    release = {i: asyncio.Event() for i in range(6)}
    active = peak = 0

    async def processing_stage(*args):
        nonlocal active, peak
        value = args[-1]
        active += 1
        peak = max(peak, active)
        entered.put_nowait(value)
        try:
            await release[value].wait()
            return np.full((4, 4), 50 + value, dtype=np.uint8)
        finally:
            active -= 1

    if stage == "fallback":
        async def missing(*args):
            return None
        monkeypatch.setattr(fetcher._sources["TESTREG"], "fetch_frame", missing)
        monkeypatch.setattr(fetcher, "_try_fallback", processing_stage)
    else:
        region = RegionDef("CACOMP", 0, 4, 0, 4, 1, "CANADA")
        fetcher._enabled_regions = [region]
        fetcher._sources = {region.name: fetcher._sources["TESTREG"]}
        fetcher._ca_source = "mrms_with_msc_blend"
        fetcher._cacomp_msc_source = object()
        monkeypatch.setattr(fetcher, "_blend_cacomp", processing_stage)

    batches = [[(1000 + i * 600, "live", i) for i in values] for values in (range(3), range(3, 6))]
    tasks = [asyncio.create_task(fetcher._fetch_timestamps(batch)) for batch in batches]
    try:
        first = await asyncio.wait_for(entered.get(), 2)
        current = await asyncio.wait_for(entered.get(), 2)
        await asyncio.sleep(0)
        assert entered.empty()
        assert peak == 2
        # Hold the first result while the others finish, proving metadata
        # still associates out-of-order completions with the correct slots.
        for remaining in range(5):
            release[current].set()
            if remaining < 4:
                current = await asyncio.wait_for(entered.get(), 2)
        release[first].set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert peak == 2
        assert active == 0
        assert await fetcher._store.get_timestamps() == [1000 + i * 600 for i in range(6)]
        name = fetcher._enabled_regions[0].name
        for i in range(6):
            frame = await fetcher._store.get_frame(1000 + i * 600)
            np.testing.assert_array_equal(frame.regions[name], np.full((4, 4), 50 + i, dtype=np.uint8))
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_primary_fetch_concurrency_is_independent(fetcher, monkeypatch):
    monkeypatch.setattr(fetcher_module.settings, "radar_fetch_concurrency", 4)
    all_fetches_started = asyncio.Event()
    release = asyncio.Event()
    fetch_active = fetch_peak = post_active = post_peak = 0

    async def source(region, minutes_ago):
        nonlocal fetch_active, fetch_peak
        fetch_active += 1
        fetch_peak = max(fetch_peak, fetch_active)
        if fetch_active == 4:
            all_fetches_started.set()
        try:
            await release.wait()
            return np.full((4, 4), 50, dtype=np.uint8)
        finally:
            fetch_active -= 1

    async def process(region, ts, source_type, source_arg, result):
        nonlocal post_active, post_peak
        post_active += 1
        post_peak = max(post_peak, post_active)
        try:
            await asyncio.sleep(0)
            return result
        finally:
            post_active -= 1

    monkeypatch.setattr(fetcher._sources["TESTREG"], "fetch_frame", source)
    monkeypatch.setattr(fetcher, "_process_result", process)
    task = asyncio.create_task(fetcher._fetch_timestamps([(1000 + i * 600, "live", i) for i in range(4)]))
    try:
        await asyncio.wait_for(all_fetches_started.wait(), 2)
        assert fetch_peak == 4
        release.set()
        await asyncio.wait_for(task, 2)
        assert post_peak == 2
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_processing_exception_releases_slot(fetcher, monkeypatch):
    active = peak = 0

    async def process(region, ts, source_type, source_arg, result):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            if source_arg == 0:
                raise ValueError("processing failed")
            return source_arg
        finally:
            active -= 1

    monkeypatch.setattr(fetcher, "_process_result", process)
    results = await asyncio.wait_for(asyncio.gather(*(finalize(fetcher, i) for i in range(6)), return_exceptions=True), 2)
    assert isinstance(results[0], ValueError)
    assert results[1:] == list(range(1, 6))
    assert active == 0
    peak = 0
    assert await asyncio.wait_for(asyncio.gather(finalize(fetcher, 6), finalize(fetcher, 7)), 2) == [6, 7]
    assert peak == 2


@pytest.mark.parametrize("cancel_waiter", [True, False])
async def test_cancellation_preserves_two_usable_slots(fetcher, monkeypatch, cancel_waiter):
    entered = asyncio.Queue()
    release = asyncio.Event()
    active = peak = 0

    async def process(region, ts, source_type, source_arg, result):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.put_nowait(source_arg)
        try:
            await release.wait()
            return source_arg
        finally:
            active -= 1

    monkeypatch.setattr(fetcher, "_process_result", process)
    tasks = [asyncio.create_task(finalize(fetcher, i)) for i in range(3)]
    try:
        assert await asyncio.wait_for(entered.get(), 2) == 0
        assert await asyncio.wait_for(entered.get(), 2) == 1
        await asyncio.sleep(0)
        assert entered.empty()
        cancelled = tasks[2] if cancel_waiter else tasks[0]
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        if not cancel_waiter:
            assert await asyncio.wait_for(entered.get(), 2) == 2
        assert active == 2
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
        assert active == 0
        assert peak == 2
        # A fresh wave must admit exactly two, including after cancelling
        # a task that never acquired a slot.
        release.clear()
        tasks = [asyncio.create_task(finalize(fetcher, i)) for i in range(3, 6)]
        assert await asyncio.wait_for(entered.get(), 2) == 3
        assert await asyncio.wait_for(entered.get(), 2) == 4
        await asyncio.sleep(0)
        assert entered.empty()
        assert active == 2
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
