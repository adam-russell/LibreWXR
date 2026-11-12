# SPDX-License-Identifier: AGPL-3.0-or-later
"""Separate server timings from the reference client's displaced work."""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
import time

from bootstrap import NATIVE_SHA256
import numpy as np
from PIL import Image
from librewxr.colors import schemes
from librewxr.tiles import coordinates
from librewxr.tiles.sampling import sampling_plans
from librewxr.tiles.renderer import compute_tile_geometry, present_tile, TileGeometry
from librewxr.tiles.data_tiles import encode_data_tile, decode_data_tile, client_rgba
from weather import open_weather

ROOT=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()
def counters():
    return dict(plans=sampling_plans.stats()["misses"], **{f.__name__:f.cache_info().misses for f in coordinates.ALL_CACHES})

def clear():
    for fn in coordinates.ALL_CACHES:fn.cache_clear()
    sampling_plans.clear()
def usage():
    result={}
    for name in ('memory.current','memory.peak','memory.events','memory.stat','cpu.stat'):
        p=Path('/sys/fs/cgroup')/name
        if p.exists():result[name]=p.read_text().strip()
    return result

def measure(fn):
    wall=time.perf_counter();cpu=time.process_time();out=fn()
    return out,dict(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall)

def encode(geom,variant):
    return present_tile(geom,2,'png') if variant=='png' else encode_data_tile(geom,level=int(variant[-1]))

def png_rgba(geom,scheme):
    return np.asarray(Image.open(io.BytesIO(present_tile(geom,scheme,'png'))).convert('RGBA'))

def validation(weather,manifest,output):
    count=0;export=[]
    palettes={str(s):dict(rain=schemes.capture_luts(s)[0].tolist(),snow=schemes.capture_luts(s)[1].tolist(),name=name) for s,name in schemes.SCHEME_NAMES.items()}
    for case in manifest['cases']:
        geoms=geometries(weather,case)
        for index,geom in enumerate(geoms):
            decoded=decode_data_tile(encode_data_tile(geom))
            if not geom.is_transparent:
                np.testing.assert_array_equal(decoded.values,geom.values)
                if geom.snow_mask is not None:np.testing.assert_array_equal(decoded.snow,geom.snow_mask)
            refs={}
            for scheme in schemes.SCHEME_NAMES:
                rain,snow,_=schemes.capture_luts(scheme);expected=png_rgba(geom,scheme)
                np.testing.assert_array_equal(client_rgba(decoded,rain,snow),expected);count+=1
                if index==0:refs[str(scheme)]=dict(png=base64.b64encode(present_tile(geom,scheme,'png')).decode(),rgba_sha256=sha(expected.tobytes()))
            if index==0:export.append(dict(name=case['name'],timestamp=weather['frames'][0][0],data=base64.b64encode(encode_data_tile(geom)).decode(),reference=refs))
    # Explicit transparent, mixed-snow and edge padding supplements, independent of weather.
    rng=np.random.default_rng(123)
    supplements=[TileGeometry.transparent(512)]
    for mask in (None,False,True,'mixed'):
        side=530;values=rng.integers(0,256,(side,side),dtype=np.uint8)
        snow=None if mask is None else (rng.random((side,side))>.5 if mask=='mixed' else np.full((side,side),mask,bool))
        supplements.append(TileGeometry(values,snow,512,9,2.75))
    for geom in supplements:
        for scheme in schemes.SCHEME_NAMES:
            np.testing.assert_array_equal(client_rgba(decode_data_tile(encode_data_tile(geom)),*schemes.capture_luts(scheme)[:2]),png_rgba(geom,scheme));count+=1
    payload=dict(scenes=export,palettes=palettes)
    (output/'comparison.json').write_text(json.dumps(payload,separators=(',',':')))
    template=(ROOT/'viewer.html').read_text();script=(ROOT/'source/librewxr/api/data_tiles.js').read_text()
    (output/'comparison.html').write_text(template.replace('/*CODEC*/',script).replace('/*FIXTURES*/',json.dumps(payload,separators=(',',':')).replace('<', '\\u003c')))
    return dict(transparent_transfer_bytes={v:len(encode(TileGeometry.transparent(512),v)) for v in ('png','data1','data6')},palettes=len(schemes.SCHEME_NAMES),exact_rgba_comparisons=count,real_tiles=len(manifest['cases'])*6,synthetic_geometries=len(supplements),palette_version=schemes.current_version())

def geometries(weather,case,offset=0):
    z=case['z'];n=2**z
    x=int((case['lon']+180)/360*n)+offset
    y=int((1-math.asinh(math.tan(math.radians(case['lat'])))/math.pi)/2*n)
    return [compute_tile_geometry(arrays,z,x,y,tile_size=512,smooth=True,snow=True,nwp_chain=weather['chain'],enabled_regions=weather['enabled'],frame_timestamp=ts,precip_mask=weather['precip_mask']) for ts,arrays in weather['frames']]

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['validate','png','data1','data6'],required=True);p.add_argument('--output',required=True);p.add_argument('--frozen',default='/data/cache');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,want in manifest['runtime_hashes'].items():assert sha((ROOT/name).read_bytes())==want,name
    import librewxr
    assert Path(librewxr.__file__).resolve()==ROOT/'source/librewxr/__init__.py'
    weather=open_weather(args.frozen)
    if not args.smoke:assert sha((Path(args.frozen)/'state.json').read_bytes())==manifest['snapshot_sha256']
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',variant=args.variant,native_sha256=NATIVE_SHA256,source_hashes=manifest['runtime_hashes'],snapshot_sha256=sha((Path(args.frozen)/'state.json').read_bytes()),distinct_frames=weather['distinct_frames'],smoke=args.smoke,cases=[],cgroup_before=usage())
    if args.variant=='validate':result['validation']=validation(weather,manifest,out.parent)
    else:
        for case in manifest['cases']:
            print(case['name'],args.variant,flush=True)
            clear();geoms=geometries(weather,case)
            assert any(not g.is_transparent and g.values.any() for g in geoms),'Workload is empty'
            outputs=[encode(g,args.variant) for g in geoms]
            # Cross-worker geometry signatures are independent of the output codec.
            # All-palette image equality is checked by the separate validation job.
            hashes=[sha(g.values.tobytes()+(g.snow_mask.tobytes() if g.snow_mask is not None else b'')+str((g.tile_size,g.pad,g.blur_radius,g.is_transparent)).encode()) for g in geoms]
            samples=[]
            phases=['presentation','render_warm','render_cold','pan_cold']
            for phase in phases:
                # Prime allocations and coordinates once, then clear outside every cold measurement.
                geometries(weather,case)
                for repetition in range(1 if args.smoke else manifest['rounds']):
                    if phase.endswith('cold'):clear()
                    def run():
                        targets=geoms if phase=='presentation' else geometries(weather,case,offset=1 if phase=='pan_cold' else 0)
                        return [encode(g,args.variant) for g in targets]
                    before=counters()
                    measured,metrics=measure(run)
                    after=counters()
                    metrics["coordinate_plan_misses"]=sum(after[k]-before[k] for k in before)
                    if phase=="render_warm":assert before==after,"Cold work entered warm measurement"
                    samples.append(dict(phase=phase,repetition=repetition,**metrics))
                    if phase in ('presentation','render_warm','render_cold'):assert [sha(x) for x in measured]==[sha(x) for x in outputs]
                    del measured
            result['cases'].append(dict(case=case,geometry_hashes=hashes,bytes_per_tile=[len(x) for x in outputs],samples=samples,plans=sampling_plans.stats()))
        # This peak covers server work only; the separately timed reference
        # client has not run in this process yet.
        result['server_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        for row in result['cases']:
            outputs=[encode(g,args.variant) for g in geometries(weather,row['case'])]
            def client_batch():
                return [client_rgba(decode_data_tile(data),*schemes.capture_luts(2)[:2]) for data in outputs] if args.variant!='png' else [np.asarray(Image.open(io.BytesIO(data)).convert('RGBA')).copy() for data in outputs]
            client_batch() # Untimed allocation warm-up, after the server peak.
            for repetition in range(1 if args.smoke else manifest['rounds']):
                decoded,metrics=measure(client_batch)
                row['samples'].append(dict(phase='reference_client',repetition=repetition,**metrics))
                del decoded
        # Palette switching: transfer/reuse one numeric geometry; colored path re-presents for each scheme.
        geom=geoms[0]
        def palette_switch():
            return [present_tile(geom,s,'png') for s in schemes.SCHEME_NAMES] if args.variant=='png' else [encode(geom,args.variant)]
        switched,metrics=measure(palette_switch)
        result['palette_switch']=dict(**metrics,tiles=len(switched),total_bytes=sum(map(len,switched)),palettes=len(schemes.SCHEME_NAMES),palette_json_bytes=len(json.dumps({s:dict(rain=schemes.capture_luts(s)[0].tolist(),snow=schemes.capture_luts(s)[1].tolist()) for s in schemes.SCHEME_NAMES},separators=(',',':')).encode()))
    result.update(status='completed',cgroup_after=usage(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
