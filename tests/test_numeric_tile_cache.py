# SPDX-License-Identifier: AGPL-3.0-or-later
"""Numeric response reuse with real frame versions and shared-store files."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import threading
from unittest.mock import AsyncMock, Mock
import zlib

from fastapi import FastAPI
import httpx
import numpy as np
import pytest

from librewxr.api import data_tiles as api, routes
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.main import _maintain_shared_tiles
from librewxr.tiles.cache import TileCache
from librewxr.tiles.data_tiles import COMPRESSION_LEVEL, decode_data_tile, encode_data_tile
from librewxr.tiles.renderer import TileGeometry
from librewxr.tiles.shared_tile_store import SharedTileStore

URL = '/experimental/radar-data/123/512/3/2/2/1_1.lwxr'


@pytest.fixture
async def context(tmp_path, monkeypatch):
    store = FrameStore(max_frames=2, cache_dir=tmp_path/'frames')
    await store.add_frame(RadarFrame(123, {'USCOMP': np.full((2, 2), 100, np.uint8)}))
    disk = SharedTileStore(tmp_path/'encoded', max_mb=8)
    for name, value in dict(frame_store=store, tile_cache=TileCache(8), shared_tile_store=disk,
                            nowcast_store=None, present_executor=None, io_executor=None,
                            nwp_chain=None, precip_mask=None, enabled_regions=set()).items():
        monkeypatch.setattr(routes, name, value)
    state = dict(store=store, disk=disk, model_offset=0)
    def geometry(**kwargs):
        value = int(kwargs['frame_regions']['USCOMP'][0, 0]) + state['model_offset']
        size = kwargs['tile_size']; pad = 3 if kwargs['smooth'] else 0
        yy, xx = np.indices((size+2*pad, size+2*pad))
        values = ((xx//8 + yy//16 + value) % 255).astype(np.uint8)
        snow = (xx + yy) % 19 == 0 if kwargs['snow'] else None
        return TileGeometry(values, snow, size, pad, 1.1 if kwargs['smooth'] else 0.)
    state['compute'] = Mock(side_effect=geometry)
    monkeypatch.setattr(api, 'compute_tile_geometry', state['compute'])
    app = FastAPI(); app.include_router(api.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://numeric-test') as client:
        state['client'] = client
        yield state


async def merge(context, value):
    await context['store'].add_frame(RadarFrame(123, {'USCOMP':np.full((2,2), value, np.uint8)}))
    # Mirror the render poller after applying a new state snapshot. Leave old
    # disk files in place to prove versioning independently prevents reuse.
    routes.tile_cache.invalidate_timestamp(123)


def test_default_compression_is_lossless_level_six():
    yy, xx = np.indices((518,518))
    geom = TileGeometry(((xx//8 + yy//8)%180).astype(np.uint8), xx%7==0, 512, 3, 1.1)
    data = encode_data_tile(geom)
    length = struct.unpack_from('<I',data,8)[0]
    payload = data[12+length:]
    raw = zlib.decompress(payload)
    assert COMPRESSION_LEVEL==6 and payload==zlib.compress(raw,6)
    assert len(data)<len(encode_data_tile(geom,level=1))
    a, b = decode_data_tile(data), decode_data_tile(encode_data_tile(geom,level=1))
    assert a.metadata==b.metadata
    np.testing.assert_array_equal(a.values,b.values)
    np.testing.assert_array_equal(a.snow,b.snow)


@pytest.mark.parametrize('dedicated_executors',[False,True])
async def test_another_worker_reuses_disk_then_encoded_ram(context, monkeypatch, dedicated_executors):
    main_thread = threading.get_ident()
    io_threads = []
    original_get, original_publish = context['disk'].get, context['disk'].publish
    def get(key):
        io_threads.append(threading.get_ident()); return original_get(key)
    def publish(key,data):
        io_threads.append(threading.get_ident()); return original_publish(key,data)
    monkeypatch.setattr(context['disk'],'get',get)
    monkeypatch.setattr(context['disk'],'publish',publish)
    encode = Mock(wraps=api._encode);monkeypatch.setattr(api,'_encode',encode)
    with ThreadPoolExecutor(2) as io, ThreadPoolExecutor(2) as present:
        if dedicated_executors:
            monkeypatch.setattr(routes,'io_executor',io);monkeypatch.setattr(routes,'present_executor',present)
        first = await context['client'].get(URL)
        assert first.status_code==200 and context['compute'].call_count==encode.call_count==1
        assert first.headers['cache-control']=='public, max-age=300'
        assert len(list(context['disk'].root.rglob('*.tile')))==1
        assert '-numeric-v1-zlib6-' in next(context['disk'].root.rglob('*.tile')).stem
        # Separate worker: fresh memory cache and a distinct shared-store instance.
        monkeypatch.setattr(routes,'tile_cache',TileCache(8))
        other = SharedTileStore(context['disk'].root.parent,max_mb=8)
        monkeypatch.setattr(routes,'shared_tile_store',other)
        read = Mock(wraps=other.get);monkeypatch.setattr(other,'get',read)
        second = await context['client'].get(URL)
        assert second.content==first.content and second.headers['etag']==first.headers['etag']
        assert read.call_count==1 and context['compute'].call_count==encode.call_count==1
        assert routes.tile_cache.size==1  # Encoded bytes only, no geometry.
        for path, headers in [(URL,{}),(URL,{'If-None-Match':first.headers['etag']}),
                              (URL.replace('/123/','/0/'),{})]:
            response = await context['client'].get(path,headers=headers)
            assert response.status_code==(304 if headers else 200)
            assert response.content==(b'' if headers else first.content)
        assert read.call_count==1 and context['compute'].call_count==encode.call_count==1
        assert io_threads and all(t!=main_thread for t in io_threads)


async def test_versions_and_expiry_do_not_reuse_old_files(context):
    first = await context['client'].get(URL)
    await merge(context,160)
    second = await context['client'].get(URL)
    assert second.status_code==200 and second.content!=first.content
    assert second.headers['etag']!=first.headers['etag']
    assert len(list(context['disk'].root.rglob('*.tile')))==2
    assert decode_data_tile(second.content).values[0,0]==160
    for ts in (124,125):
        await context['store'].add_frame(RadarFrame(ts,{'USCOMP':np.full((2,2),80,np.uint8)}))
    routes.tile_cache.invalidate_timestamp(123)
    assert context['store'].frame_version(123) is None
    assert (await context['client'].get(URL)).status_code==404


async def test_model_refresh_uses_existing_shared_store_maintenance(context):
    first = await context['client'].get(URL)
    context['model_offset']=15
    routes.tile_cache.clear()
    await asyncio.to_thread(_maintain_shared_tiles,context['disk'],True,None)
    second = await context['client'].get(URL)
    assert second.status_code==200 and second.content!=first.content
    assert context['compute'].call_count==2
    assert decode_data_tile(second.content).values[0,0]==115


@pytest.mark.parametrize('timestamp',[123,456])
async def test_unversioned_and_nowcast_frames_never_use_disk(context,monkeypatch,timestamp):
    if timestamp==123:
        context['store']._frame_versions.clear()  # Legacy snapshot.
    else:
        nowcast = AsyncMock()
        nowcast.get_frame.return_value=(RadarFrame(456,{'USCOMP':np.full((2,2),90,np.uint8)}),.25)
        monkeypatch.setattr(routes,'nowcast_store',nowcast)
    def forbidden(*args):raise AssertionError('Unversioned/nowcast frame touched shared disk')
    monkeypatch.setattr(context['disk'],'get',forbidden)
    monkeypatch.setattr(context['disk'],'publish',forbidden)
    response = await context['client'].get(URL.replace('/123/',f'/{timestamp}/'))
    assert response.status_code==200
    if timestamp==456:assert context['compute'].call_args.kwargs['nowcast_blend']==.25


async def test_disabled_disk_store_keeps_bounded_memory_reuse(context,monkeypatch):
    monkeypatch.setattr(routes,'shared_tile_store',None)
    first = await context['client'].get(URL)
    assert (await context['client'].get(URL)).content==first.content
    assert context['compute'].call_count==1
    assert routes.tile_cache.total_bytes<=8*1024**2
    assert not list(context['disk'].root.rglob('*.tile'))


@pytest.mark.parametrize('failure',['read','write'])
async def test_disk_errors_fall_back_to_rendering(context,monkeypatch,failure):
    first = await context['client'].get(URL)
    routes.tile_cache.clear()
    method = 'read_bytes' if failure=='read' else 'write_bytes'
    original = getattr(Path,method)
    root = context['disk'].root
    def fail(path,*args):
        if root in path.parents:raise PermissionError('simulated shared-volume failure')
        return original(path,*args)
    if failure=='write':context['disk'].sweep_final_files()
    monkeypatch.setattr(Path,method,fail)
    response = await context['client'].get(URL)
    assert response.status_code==200 and response.content==first.content
    assert context['compute'].call_count==2


async def test_tile_parameters_have_distinct_shared_keys(context):
    paths=[URL,URL.replace('/512/','/256/'),URL.replace('/3/2/2/','/3/3/2/'),
           URL.replace('/3/2/2/','/4/2/2/'),URL.replace('/1_1.','/0_1.'),URL.replace('/1_1.','/1_0.')]
    for path in paths:assert (await context['client'].get(path)).status_code==200
    assert len(list(context['disk'].root.rglob('*.tile')))==len(paths)
    await context['store'].add_frame(RadarFrame(124,{'USCOMP':np.full((2,2),80,np.uint8)}))
    assert (await context['client'].get(URL)).headers['cache-control']=='public, max-age=7200'


async def test_refresh_during_shared_read_retries_new_version(context,monkeypatch):
    await context['client'].get(URL)
    routes.tile_cache.clear()
    original = context['disk'].get
    loop = asyncio.get_running_loop();updated=False
    def get(key):
        nonlocal updated
        data=original(key)
        if not updated:
            updated=True
            asyncio.run_coroutine_threadsafe(merge(context,170),loop).result(timeout=5)
        return data
    monkeypatch.setattr(context['disk'],'get',get)
    response=await context['client'].get(URL)
    assert response.status_code==200 and decode_data_tile(response.content).values[0,0]==170


@pytest.mark.parametrize('evict',[False,True])
async def test_refresh_or_expiry_during_encode_does_not_publish_old_bytes(context,monkeypatch,evict):
    loop=asyncio.get_running_loop();original=api._encode;updated=False
    async def update():
        if evict:
            for ts in (124,125):
                await context['store'].add_frame(RadarFrame(ts,{'USCOMP':np.full((2,2),80,np.uint8)}))
            routes.tile_cache.invalidate_timestamp(123)
        else:await merge(context,180)
    def encode(g):
        nonlocal updated
        result=original(g)
        if not updated:
            updated=True
            asyncio.run_coroutine_threadsafe(update(),loop).result(timeout=5)
        return result
    monkeypatch.setattr(api,'_encode',encode)
    response=await context['client'].get(URL)
    assert response.status_code==(404 if evict else 200)
    if evict:assert not list(context['disk'].root.rglob('*.tile'))
    else:
        assert decode_data_tile(response.content).values[0,0]==180
        assert len(list(context['disk'].root.rglob('*.tile')))==1


async def test_repeated_version_changes_return_retryable_503(context,monkeypatch):
    loop=asyncio.get_running_loop();original=api._encode;value=150
    def encode(g):
        nonlocal value
        result=original(g)
        value+=1
        asyncio.run_coroutine_threadsafe(merge(context,value),loop).result(timeout=5)
        return result
    monkeypatch.setattr(api,'_encode',encode)
    response=await context['client'].get(URL)
    assert response.status_code==503
    assert response.json()['detail']=='Frame changed while rendering; retry'
    assert not list(context['disk'].root.rglob('*.tile'))
