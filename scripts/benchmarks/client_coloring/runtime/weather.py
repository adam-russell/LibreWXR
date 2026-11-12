# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open an immutable saved radar/model snapshot; never fetch live data."""
from pathlib import Path
import numpy as np
from librewxr.colors.schemes import apply_external_table
from librewxr.config import settings
from librewxr.data.coverage import load_masks,build_coverage_masks
from librewxr.data.master_state import load_state
from librewxr.data.nwp_source import NWPChain
from librewxr.data.precip_mask import PrecipMaskStore
from librewxr.sources import collect_radar_coverage_metadata,enabled_regions_with_always_on
from librewxr.sources.world.ifs.grid import ECMWFGrid


def open_weather(root):
    root=Path(root)
    snapshot=load_state(root);assert snapshot is not None
    enabled=enabled_regions_with_always_on(settings)
    stations,ranges,polygons=collect_radar_coverage_metadata(settings)
    if not load_masks(root,enabled,stations,ranges,polygons):
        # Build only in process memory; the mounted snapshot remains read-only.
        build_coverage_masks(stations,range_overrides=ranges,coverage_polygons=polygons)
    state=dict(snapshot['stores']['ecmwf_grid']);state['memmap_dir']=str(root/'ecmwf_ifs')
    grid=ECMWFGrid(cache_dir=Path('/tmp/sampling-model'));grid.__setstate__(state)
    assert grid.has_data()
    chain=NWPChain([grid])
    frame_state=snapshot['stores']['frame_store']
    frames=[]
    for info in sorted(frame_state['frames'],key=lambda f:f['timestamp']):
        arrays={name:np.memmap(root/'radar'/Path(filename).name,dtype=np.dtype(dtype),shape=tuple(shape),mode='r')
                for name,(filename,dtype,shape) in info['regions'].items()}
        assert arrays and all(not a.flags.writeable for a in arrays.values())
        frames.append((info['timestamp'],arrays))
    assert len(frames)>=2
    # This is a six-frame animation workload; disclose reused inputs if the
    # snapshot has fewer than six distinct times instead of synthesizing weather.
    sequence=[frames[i%len(frames)] for i in range(6)]
    precip_mask=PrecipMaskStore()
    mask_state=dict(snapshot['stores'].get('precip_mask',{}))
    mask_state['memmap_dir']=str(root/'mask')
    precip_mask.__setstate__(mask_state)
    apply_external_table((root/'palettes/current.csv').read_bytes())
    return dict(chain=chain,precip_mask=precip_mask,mask_count=len(precip_mask._masks),enabled=enabled,frames=sequence,distinct_frames=len(set(t for t,_ in sequence)),
                available_regions=sorted(set().union(*(set(a) for _,a in sequence))))
