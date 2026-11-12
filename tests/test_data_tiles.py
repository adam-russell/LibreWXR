# SPDX-License-Identifier: AGPL-3.0-or-later
import base64
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import zlib
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from librewxr.api import routes, data_tiles as api
from librewxr.colors import schemes
from librewxr.config import Settings
from librewxr.tiles.cache import TileCache
from librewxr.tiles.data_tiles import encode_data_tile, decode_data_tile, client_rgba, MAGIC
from librewxr.tiles.renderer import TileGeometry, present_tile


def geometry(size=16, radius=0., mask='mixed'):
    pad=int(np.ceil(radius*3)) if radius>=.5 else 0
    side=size+2*pad
    rng=np.random.default_rng(42)
    values=rng.integers(0,256,(side,side),dtype=np.uint8)
    snow=None if mask=='none' else np.full((side,side),mask=='all',bool)
    if mask=='mixed':snow=rng.random((side,side))>.5
    return TileGeometry(values,snow,size,pad,radius)


@pytest.mark.parametrize('size,radius',[(1,0),(16,.49),(16,.5),(32,.9),(32,1.5),(32,2.75),(256,3.7),(512,1.1)])
@pytest.mark.parametrize('mask',['none','all','mixed'])
def test_roundtrip_and_current_png_parity(size,radius,mask):
    geom=geometry(size,radius,mask)
    tile=decode_data_tile(encode_data_tile(geom))
    np.testing.assert_array_equal(tile.values,geom.values)
    assert not tile.values.flags.writeable
    if geom.snow_mask is not None:np.testing.assert_array_equal(tile.snow,geom.snow_mask)
    for scheme in schemes.SCHEME_NAMES:
        rain,snow,_=schemes.capture_luts(scheme)
        actual=client_rgba(tile,rain,snow)
        expected=np.asarray(Image.open(io.BytesIO(present_tile(geom,scheme,'png'))).convert('RGBA'))
        np.testing.assert_array_equal(actual,expected)


def test_transparent():
    geom=TileGeometry.transparent(512)
    data=encode_data_tile(geom);assert len(data)<512
    tile=decode_data_tile(data)
    assert tile.metadata['empty'] and not client_rgba(tile,*schemes.capture_luts(2)[:2]).any()


def rewrite(data,**changes):
    n=struct.unpack_from('<I',data,8)[0];m=json.loads(data[12:12+n]);m.update(changes)
    h=json.dumps(m).encode();return MAGIC+struct.pack('<I',len(h))+h+data[12+n:]


@pytest.mark.parametrize('changes',[dict(version=True),dict(version=2),dict(size=513),dict(side=2049),dict(pad=-1),dict(size=True),dict(empty=1),dict(snow=1),dict(blur=True),dict(blur=float('nan')),dict(blur=129),dict(kernel={}),dict(decoded_bytes=10**9),dict(intensity='other')])
def test_bad_metadata(changes):
    with pytest.raises(ValueError):decode_data_tile(rewrite(encode_data_tile(geometry()),**changes))


def test_compression_and_header_rejections():
    data=encode_data_tile(geometry());n=struct.unpack_from('<I',data,8)[0];head=data[:12+n]
    for bad in (b'',b'unknown!',data[:-1],data+b'extra',head+zlib.compress(b'x'*10**6),MAGIC+struct.pack('<I',5000)+b'x'*5000,head+zlib.compress(b'')):
        with pytest.raises(ValueError):decode_data_tile(bad)


def test_browser_matches_pillow(tmp_path):
    if not shutil.which('node'):pytest.skip('Node needed for browser algorithm parity')
    cases=[]
    for size,radius in [(1,.5),(16,.9),(32,1.1),(32,2.75),(256,3.7),(512,1.5)]:
        for mask in ('none','all','mixed'):
            geom=geometry(size,radius,mask);data=encode_data_tile(geom)
            for scheme in schemes.SCHEME_NAMES:
                rain,snow,_=schemes.capture_luts(scheme)
                reference=np.asarray(Image.open(io.BytesIO(present_tile(geom,scheme,'png'))).convert('RGBA'))
                cases.append(dict(data=base64.b64encode(data).decode(),palette=dict(rain=rain.tolist(),snow=snow.tolist()),expected=base64.b64encode(reference.tobytes()).decode()))
    fixture=tmp_path/'cases.json';fixture.write_text(json.dumps(cases))
    script=tmp_path/'test.cjs';script.write_text('''const fs=require('node:fs'),assert=require('node:assert/strict');
const codec=require(process.argv[2]);
(async()=>{for(const row of JSON.parse(fs.readFileSync(process.argv[3]))){const b=Buffer.from(row.data,'base64');const tile=await codec.decode(b);assert.deepEqual(Buffer.from(codec.colorize(tile,row.palette)),Buffer.from(row.expected,'base64'));}})().catch(e=>{console.error(e);process.exit(1)});''')
    result=subprocess.run(['node',str(script),str(Path(api.__file__).with_name('data_tiles.js')),str(fixture)],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr[-4000:]


@pytest.fixture
def client(monkeypatch):
    store=AsyncMock();store.get_latest_frame.return_value=type('Frame',(),dict(timestamp=123))()
    store.get_timestamps.return_value=[123]
    store.get_frame.return_value=type('Frame',(),dict(timestamp=123,regions={}))()
    store.frame_version=Mock(side_effect=lambda ts: 1 if ts==123 and store.get_frame.return_value is not None else None)
    monkeypatch.setattr(routes,'frame_store',store);monkeypatch.setattr(routes,'tile_cache',TileCache(8))
    monkeypatch.setattr(routes,'nowcast_store',None)
    monkeypatch.setattr(routes,'shared_tile_store',None)
    monkeypatch.setattr(routes,'present_executor',None)
    monkeypatch.setattr(routes,'io_executor',None)
    # Geometry is supplied by this fixture rather than geographic sampling.
    monkeypatch.setattr(api,'_blank_label',Mock(return_value=None))
    monkeypatch.setattr(api,'compute_tile_geometry',lambda **kwargs:geometry(256,1.5))
    app=FastAPI();app.include_router(routes.router);app.include_router(api.router)
    return TestClient(app)


def test_api_cache_palette_and_invalidation(client,monkeypatch):
    path='/experimental/radar-data/0/256/3/2/2/1_1.lwxr'
    r=client.get(path);assert r.status_code==200
    assert r.headers['x-frame-timestamp']=='123'
    assert client.get(path,headers={'If-None-Match':r.headers['etag']}).status_code==304
    palette=client.get('/experimental/radar-data/palettes/2.json');assert palette.status_code==200
    assert client.get('/experimental/radar-data/palettes/999.json').status_code==404
    assert client.get('/experimental/radar-data/palettes/2.json',headers={'If-None-Match':palette.headers['etag']}).status_code==304
    def forbidden(**kwargs):raise AssertionError('Cached geometry should be reused')
    monkeypatch.setattr(api,'compute_tile_geometry',forbidden)
    # Palette purge removes colored entries, leaving numeric bytes and geometry.
    routes.tile_cache.purge_token('p'+schemes.current_version())
    assert client.get(path).content==r.content
    # Existing colored endpoint shares geometry and is still a regular PNG.
    png=client.get('/v2/radar/123/256/3/2/2/2/1_1.png');assert png.status_code==200 and png.content.startswith(b'\x89PNG')
    routes.tile_cache.invalidate_timestamp(123)
    routes.frame_store.get_frame.return_value=None
    assert client.get(path).status_code==404


@pytest.mark.parametrize('tail',['123/512/3/8/1/1_1.lwxr','123/100/3/2/1/1_1.lwxr','123/512/3/2/1/2_1.lwxr','123/512/3/2/1/1_1.lwxr?arrows=1'])
def test_invalid_api(client,tail):
    assert client.get('/experimental/radar-data/'+tail).status_code==400


def test_disabled_by_default():
    assert not Settings(_env_file=None).experimental_data_tiles_enabled
    # Main mounting is conditional, not a runtime toggle that leaves routes exposed.
    code='from librewxr.main import app; from fastapi.testclient import TestClient; assert TestClient(app).get("/experimental/radar-data/client").status_code == 404'
    subprocess.run([__import__('sys').executable,'-c',code],check=True,timeout=30)


def test_nowcast_and_empty_store(client,monkeypatch):
    routes.frame_store.get_frame.return_value=None
    nowcast=AsyncMock();nowcast.get_frame.return_value=(type('Frame',(),dict(regions={}))(),.25)
    monkeypatch.setattr(routes,'nowcast_store',nowcast)
    observed=[]
    def compute(**kwargs):
        observed.append(kwargs)
        return TileGeometry.transparent(256)
    monkeypatch.setattr(api,'compute_tile_geometry',compute)
    r=client.get('/experimental/radar-data/456/256/3/2/2/1_1.lwxr')
    assert r.status_code==200 and r.headers['x-frame-timestamp']=='456'
    assert observed[0]['nowcast_blend']==.25 and decode_data_tile(r.content).metadata['empty']
    routes.frame_store.get_latest_frame.return_value=None
    assert client.get('/experimental/radar-data/0/256/3/2/2/1_1.lwxr').status_code==404


def test_palette_update_reuses_data(client):
    # Actually replace a valid table, rather than only simulating a cache purge.
    original=Path(schemes.__file__).with_name('color_table.csv').read_bytes()
    path='/experimental/radar-data/123/256/3/2/2/1_1.lwxr'
    data=client.get(path)
    palette=client.get('/experimental/radar-data/palettes/2.json')
    png=client.get('/v2/radar/123/256/3/2/2/2/1_1.png')
    try:
        changed=original.decode().splitlines()
        row=changed[20].split(',');row[3]='#ff0000ff';changed[20]=','.join(row)
        schemes.apply_external_table(('\n'.join(changed)+'\n').encode())
        routes.tile_cache.purge_token('p'+palette.json()['version'])
        assert client.get(path).content==data.content
        assert client.get('/experimental/radar-data/palettes/2.json').headers['etag']!=palette.headers['etag']
        assert client.get('/v2/radar/123/256/3/2/2/2/1_1.png').content!=png.content
    finally:
        schemes.apply_external_table(original)
