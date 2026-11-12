# SPDX-License-Identifier: AGPL-3.0-or-later
"""Empty numeric responses stay frame-aware, lossless and free of disk I/O."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
import httpx
import numpy as np
import pytest

from librewxr.api import data_tiles as api, routes
from librewxr.api.conditional import compute_etag
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.data_tiles import decode_data_tile, encode_data_tile, MAGIC
from librewxr.tiles.renderer import TileGeometry, compute_tile_geometry
from librewxr.tiles.shared_tile_store import SharedTileStore

URL = '/experimental/radar-data/123/512/3/2/2/1_1.lwxr'


@pytest.fixture
async def context(tmp_path, monkeypatch):
    api._blank_render.cache_clear()
    store = FrameStore(max_frames=2, cache_dir=tmp_path/'frames')
    await store.add_frame(RadarFrame(123, {'USCOMP':np.full((2,2),100,np.uint8)}))
    disk = SharedTileStore(tmp_path/'shared', max_mb=8)
    mask = SimpleNamespace(has_precip_in_bbox=Mock(return_value=False))
    model = SimpleNamespace(has_data=lambda:True)
    for name,value in dict(frame_store=store,tile_cache=TileCache(8),shared_tile_store=disk,
                            nowcast_store=None,nwp_chain=model,precip_mask=mask,
                            enabled_regions=['USCOMP'],present_executor=None,io_executor=None).items():
        monkeypatch.setattr(routes,name,value)
    def wet(**kwargs):
        size=kwargs['tile_size'];pad=3 if kwargs['smooth'] else 0
        value=int(kwargs['frame_regions']['USCOMP'][0,0])
        return TileGeometry(np.full((size+2*pad,size+2*pad),value,np.uint8),None,size,pad,1.1 if pad else 0.)
    compute=Mock(side_effect=wet);encode=Mock(wraps=api._encode)
    monkeypatch.setattr(api,'compute_tile_geometry',compute);monkeypatch.setattr(api,'_encode',encode)
    app=FastAPI();app.include_router(api.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://blank-test') as client:
        yield dict(store=store,disk=disk,mask=mask,model=model,compute=compute,encode=encode,client=client)
    api._blank_render.cache_clear()


@pytest.mark.parametrize('size',[256,512])
@pytest.mark.parametrize('flags',['0_0','0_1','1_0','1_1'])
async def test_certified_blank_reuses_v1_response_without_compute_or_disk(context,monkeypatch,size,flags):
    disk_get=Mock(side_effect=AssertionError('Blank tile read shared disk'))
    disk_put=Mock(side_effect=AssertionError('Blank tile wrote shared disk'))
    monkeypatch.setattr(context['disk'],'get',disk_get);monkeypatch.setattr(context['disk'],'publish',disk_put)
    path=URL.replace('/512/',f'/{size}/').replace('/1_1.',f'/{flags}.')
    first=await context['client'].get(path)
    assert first.status_code==200
    assert first.content==encode_data_tile(TileGeometry.transparent(size))
    assert first.content.startswith(MAGIC) and decode_data_tile(first.content).metadata['version']==1
    assert first.headers['etag']==compute_etag(first.content)
    assert first.headers['x-frame-timestamp']=='123' and first.headers['cache-control']=='public, max-age=300'
    other=await context['client'].get(path.replace('/3/2/2/','/3/3/2/'))
    routes.tile_cache.clear()
    again=await context['client'].get(path)
    latest=await context['client'].get(path.replace('/123/','/0/'))
    conditional=await context['client'].get(path,headers={'If-None-Match':first.headers['etag']})
    assert other.content==again.content==latest.content==first.content
    assert latest.headers['x-frame-timestamp']=='123'
    assert conditional.status_code==304 and not conditional.content
    assert context['compute'].call_count==0 and context['encode'].call_count==1
    disk_get.assert_not_called();disk_put.assert_not_called()
    await context['store'].add_frame(RadarFrame(124,{'USCOMP':np.full((2,2),100,np.uint8)}))
    assert (await context['client'].get(path)).headers['cache-control']=='public, max-age=7200'


async def test_empty_constant_cache_is_bounded_to_two_sizes(context):
    for size in (256,512):
        for flags in ('0_0','0_1','1_0','1_1'):
            path=URL.replace('/512/',f'/{size}/').replace('/1_1.',f'/{flags}.')
            assert (await context['client'].get(path)).status_code==200
    info=api._blank_render.cache_info()
    assert info.maxsize==info.currsize==context['encode'].call_count==2


async def test_no_region_no_model_uses_existing_empty_decision(context,monkeypatch):
    monkeypatch.setattr(routes,'nwp_chain',None)
    response=await context['client'].get(URL.replace('/3/2/2/','/6/2/2/'))
    assert response.status_code==200 and decode_data_tile(response.content).metadata['empty']
    context['compute'].assert_not_called()


async def test_nwp_only_ocean_mask_gate_matches_full_renderer(context):
    context['store']._by_ts[123].regions={}
    expected=compute_tile_geometry({},3,2,2,tile_size=512,smooth=True,snow=True,
        nwp_chain=context['model'],enabled_regions=['USCOMP'],frame_timestamp=123,precip_mask=context['mask'])
    assert expected.is_transparent
    response=await context['client'].get(URL)
    assert response.content==encode_data_tile(expected)
    context['compute'].assert_not_called()


@pytest.mark.parametrize('mode',['unknown-mask','no-mask','no-model','model-unavailable'])
async def test_uncertain_or_inapplicable_mask_preserves_wet_rendering(context,monkeypatch,mode):
    if mode=='unknown-mask':context['mask'].has_precip_in_bbox.return_value=True
    elif mode=='no-mask':monkeypatch.setattr(routes,'precip_mask',None)
    elif mode=='no-model':monkeypatch.setattr(routes,'nwp_chain',None)
    else:monkeypatch.setattr(routes,'nwp_chain',SimpleNamespace(has_data=lambda:False))
    response=await context['client'].get(URL)
    assert response.status_code==200 and not decode_data_tile(response.content).metadata['empty']
    assert context['compute'].call_count==context['encode'].call_count==1
    assert len(list(context['disk'].root.rglob('*.tile')))==1


async def test_post_sample_blank_uses_one_constant_and_never_publishes(context,monkeypatch):
    context['mask'].has_precip_in_bbox.return_value=True
    context['compute'].side_effect=None;context['compute'].return_value=TileGeometry.transparent(512)
    get=Mock(wraps=context['disk'].get);publish=Mock(wraps=context['disk'].publish)
    monkeypatch.setattr(context['disk'],'get',get);monkeypatch.setattr(context['disk'],'publish',publish)
    one=await context['client'].get(URL)
    two=await context['client'].get(URL.replace('/3/2/2/','/3/3/2/'))
    assert one.status_code==two.status_code==200 and one.content==two.content
    assert context['compute'].call_count==get.call_count==2
    assert context['encode'].call_count==1
    publish.assert_not_called()


async def test_wet_frame_is_fetched_once_and_ram_hit_skips_precheck(context,monkeypatch):
    context['mask'].has_precip_in_bbox.return_value=True
    get_frame=AsyncMock(wraps=context['store'].get_frame)
    label=Mock(wraps=api._blank_label)
    monkeypatch.setattr(context['store'],'get_frame',get_frame);monkeypatch.setattr(api,'_blank_label',label)
    first=await context['client'].get(URL)
    second=await context['client'].get(URL)
    assert first.content==second.content and first.status_code==200
    assert get_frame.await_count==label.call_count==context['compute'].call_count==1


async def test_expired_frame_cannot_reuse_memo_or_stale_geometry(context):
    assert (await context['client'].get(URL)).status_code==200
    for ts in (124,125):
        await context['store'].add_frame(RadarFrame(ts,{'USCOMP':np.full((2,2),100,np.uint8)}))
    # Deliberately leave geometry cached: expiry must be checked even before
    # the poller has applied targeted invalidation.
    assert routes.tile_cache.get((123,3,2,2,512,True,True)) is not None
    response=await context['client'].get(URL)
    assert response.status_code==404 and response.json()['detail']=='Frame not found'


async def test_new_precipitation_is_not_hidden_by_empty_memo(context):
    first=await context['client'].get(URL)
    context['mask'].has_precip_in_bbox.return_value=True
    await context['store'].add_frame(RadarFrame(123,{'USCOMP':np.full((2,2),160,np.uint8)}))
    routes.tile_cache.invalidate_timestamp(123)
    second=await context['client'].get(URL)
    assert second.status_code==200 and second.headers['etag']!=first.headers['etag']
    assert not decode_data_tile(second.content).metadata['empty']
    context['mask'].has_precip_in_bbox.return_value=False
    await context['store'].add_frame(RadarFrame(123,{'USCOMP':np.full((2,2),100,np.uint8)}))
    routes.tile_cache.invalidate_timestamp(123)
    third=await context['client'].get(URL)
    assert third.content==first.content and context['encode'].call_count==2


async def test_refresh_during_precheck_does_not_cache_superseded_blank(context,monkeypatch):
    original=api._blank_label;changed=False
    def label(*args):
        nonlocal changed
        if not changed:
            changed=True
            context['store']._frame_versions[123]+=1
            routes.tile_cache.invalidate_timestamp(123)
            context['mask'].has_precip_in_bbox.return_value=True
            return 'superseded-empty-mask'
        return original(*args)
    monkeypatch.setattr(api,'_blank_label',label)
    response=await context['client'].get(URL)
    assert response.status_code==200 and not decode_data_tile(response.content).metadata['empty']
    assert api._blank_render.cache_info().misses==0


async def test_nowcast_blank_and_regenerated_precipitation(context,monkeypatch):
    nowcast=AsyncMock();nowcast.get_frame.return_value=(RadarFrame(456,{}),.25)
    monkeypatch.setattr(routes,'nowcast_store',nowcast)
    path=URL.replace('/123/','/456/')
    first=await context['client'].get(path)
    assert first.status_code==200 and first.headers['x-frame-timestamp']=='456'
    context['compute'].assert_not_called()
    assert not list(context['disk'].root.rglob('*.tile'))
    nowcast.get_frame.return_value=(RadarFrame(456,{'USCOMP':np.full((2,2),150,np.uint8)}),.25)
    context['mask'].has_precip_in_bbox.return_value=True
    routes.tile_cache.invalidate_timestamp(456)
    second=await context['client'].get(path)
    assert second.status_code==200 and not decode_data_tile(second.content).metadata['empty']
    assert context['compute'].call_args.kwargs['nowcast_blend']==.25
    assert not list(context['disk'].root.rglob('*.tile'))
