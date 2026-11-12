# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in prototype. The normal colored API remains independent of this router."""
import asyncio
import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response

from librewxr.api import routes
from librewxr.api.conditional import compute_etag, conditional_response
from librewxr.colors import schemes
from librewxr.config import settings
from librewxr.tiles.cache import CachedRender
from librewxr.tiles.data_tiles import COMPRESSION_LEVEL, MEDIA_TYPE, encode_data_tile
from librewxr.tiles.renderer import compute_tile_geometry, TileGeometry, transparent_fast_path_label
from librewxr.tiles.coordinates import tile_bounds

router = APIRouter(prefix="/experimental/radar-data", include_in_schema=False)
ASSETS = Path(__file__).parent


def _blank_label(regions, z, x, y, enabled, nwp, mask, timestamp, blend):
    # The existing NWP-only geometry path applies this same conservative
    # mask gate. Hoist it too, so ocean tiles without radar-region arrays
    # do not need a compute-thread hop just to discover that they are empty.
    if nwp is not None and nwp.has_data() and mask is not None:
        if not mask.has_precip_in_bbox(timestamp, tile_bounds(z, x, y)):
            return "tier2_mask_numeric"
    return transparent_fast_path_label(regions,z,x,y,enabled,nwp,mask,timestamp,blend)


@lru_cache(maxsize=2)
def _blank_render(size: int) -> CachedRender:
    """One immutable encoded empty response per supported tile size."""
    return _encode(TileGeometry.transparent(size))


@router.get("")
async def capabilities():
    return dict(version=1, experimental=True, sizes=[256, 512],
                tiles="/experimental/radar-data/{timestamp}/{size}/{z}/{x}/{y}/{smooth_snow}.lwxr",
                palettes="/experimental/radar-data/palettes/{scheme}.json",
                schemes=schemes.SCHEME_NAMES, palette_version=schemes.current_version(),
                client="/experimental/radar-data/client", overlays=False)


@router.get("/client")
async def client():
    return Response((ASSETS / "data_tiles.html").read_bytes(), media_type="text/html")


@router.get("/client.js")
async def client_script():
    return Response((ASSETS / "data_tiles.js").read_bytes(), media_type="text/javascript")


@router.get("/palettes/{scheme}.json")
async def palette(request: Request, scheme: int):
    if scheme not in schemes.SCHEME_NAMES:
        raise HTTPException(404, "Unknown palette")
    rain, snow, version = schemes.capture_luts(scheme)
    body = json.dumps(dict(scheme=scheme, version=version, rain=rain.tolist(), snow=snow.tolist()),
                      separators=(",", ":")).encode()
    return conditional_response(request=request, body=body, etag=compute_etag(body),
                                content_type="application/json", max_age=0)


def _encode(geometry):
    body = encode_data_tile(geometry)
    return CachedRender(body, compute_etag(body))


async def _shared_io(fn, *args):
    """Keep shared-volume reads, hashing and publication off the event loop."""
    if routes.io_executor is not None:
        return await asyncio.get_running_loop().run_in_executor(routes.io_executor, fn, *args)
    return await asyncio.to_thread(fn, *args)


def _shared_key(timestamp, version, z, x, y, size, smooth, snow):
    # Timestamp first preserves the existing targeted invalidation/pruning.
    # Numeric data has no palette dependency; keep its namespace distinct
    # from colored tiles and from other wire formats/compression levels.
    return (f"{timestamp}-v{version}-numeric-v1-zlib{COMPRESSION_LEVEL}-"
            f"{z}-{x}-{y}-{size}-{int(smooth)}{int(snow)}")


@router.get("/{timestamp}/{size}/{z}/{x}/{y}/{smooth_snow}.lwxr")
async def data_tile(request: Request, timestamp: int, size: int, z: int, x: int, y: int, smooth_snow: str):
    if size not in (256, 512) or not 0 <= z <= settings.max_zoom or not (0 <= x < 2**z and 0 <= y < 2**z):
        raise HTTPException(400, "Invalid tile coordinates or size")
    if smooth_snow not in ("0_0", "0_1", "1_0", "1_1"):
        raise HTTPException(400, "Invalid smooth/snow flags")
    if request.query_params:
        raise HTTPException(400, "Data tiles do not support overlays or query options")
    if routes.frame_store is None or routes.tile_cache is None:
        raise HTTPException(503, "Renderer not ready")
    timestamp = await routes._resolve_radar_timestamp(timestamp)
    smooth, snow = (v == "1" for v in smooth_snow.split("_"))
    key = (timestamp, z, x, y, size, smooth, snow)
    # A refresh can happen while disk I/O or encoding is awaited. Retry once
    # against the new frame version instead of serving/caching the old bytes.
    for _ in range(2):
        frame_loaded = False
        frame = None
        blend = None
        version = routes.frame_store.frame_version(timestamp)
        encoded_key = (*key, "data-v1", "zlib", COMPRESSION_LEVEL, "numeric", version)
        shared_store = routes.shared_tile_store
        shared_key = (_shared_key(timestamp, version, z, x, y, size, smooth, snow)
                      if shared_store is not None and version is not None else None)
        geom = routes.tile_cache.get(key)
        # A past-frame encoded hit remains useful after geometry eviction.
        # Legacy/unversioned and nowcast frames retain the geometry gate and
        # existing poller invalidation; they never enter the shared store.
        cached = routes.tile_cache.get(encoded_key) if version is not None or geom is not None else None
        if not isinstance(cached, CachedRender):
            if geom is None or version is None:
                frame = await routes.frame_store.get_frame(timestamp)
                if frame is None and routes.nowcast_store is not None:
                    frame, blend = await routes.nowcast_store.get_frame(timestamp)
                if frame is None:
                    raise HTTPException(404, "Frame not found")
                frame_loaded = True
                if routes.frame_store.frame_version(timestamp) != version:
                    continue
                label = _blank_label(
                    frame.regions, z, x, y, routes.enabled_regions,
                    routes.nwp_chain, routes.precip_mask, timestamp, blend)
                if routes.frame_store.frame_version(timestamp) != version:
                    continue
                if label is not None:
                    geom = TileGeometry.transparent(size, fast_path=label)
                    routes.tile_cache.put(key, geom)
            if geom is not None and geom.is_transparent:
                cached = _blank_render(size)
        if not isinstance(cached, CachedRender) and shared_key is not None:
            hit = await _shared_io(routes._shared_get_and_hash, shared_store, shared_key)
            if routes.frame_store.frame_version(timestamp) != version:
                continue
            if hit is not None:
                cached = CachedRender(*hit)
        if not isinstance(cached, CachedRender):
            if geom is None:
                if not frame_loaded:
                    frame = await routes.frame_store.get_frame(timestamp)
                    blend = None
                    if frame is None and routes.nowcast_store is not None:
                        frame, blend = await routes.nowcast_store.get_frame(timestamp)
                    if frame is None:
                        raise HTTPException(404, "Frame not found")
                if routes.frame_store.frame_version(timestamp) != version:
                    continue
                geom = await asyncio.to_thread(
                    compute_tile_geometry, frame_regions=frame.regions, z=z, x=x, y=y,
                    tile_size=size, smooth=smooth, snow=snow, nwp_chain=routes.nwp_chain,
                    enabled_regions=routes.enabled_regions, frame_timestamp=timestamp,
                    nowcast_blend=blend, precip_mask=routes.precip_mask,
                )
                if routes.frame_store.frame_version(timestamp) != version:
                    continue
                routes.tile_cache.put(key, geom)
            if geom.is_transparent:
                # Also reuse the constant for emptiness discovered only after
                # sampling. Those tiles still pay their compute/disk lookup.
                cached = _blank_render(size)
            else:
                if routes.present_executor is not None:
                    cached = await asyncio.get_running_loop().run_in_executor(routes.present_executor, _encode, geom)
                else:
                    cached = await asyncio.to_thread(_encode, geom)
                if routes.frame_store.frame_version(timestamp) != version:
                    continue
                if shared_key is not None:
                    await _shared_io(shared_store.publish, shared_key, cached.data)
        if routes.frame_store.frame_version(timestamp) != version:
            continue
        routes.tile_cache.put(encoded_key, cached)
        latest = await routes.frame_store.get_latest_frame()
        if routes.frame_store.frame_version(timestamp) != version:
            continue
        max_age = 7200 if latest is not None and timestamp < latest.timestamp else 300
        return conditional_response(request=request, body=cached.data, etag=cached.etag,
                                    content_type=MEDIA_TYPE, max_age=max_age,
                                    extra_headers={"X-Frame-Timestamp": str(timestamp)})
    raise HTTPException(503, "Frame changed while rendering; retry")
