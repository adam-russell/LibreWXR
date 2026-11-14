# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""PAGASA PANaHON numeric radar mosaic (Philippines).

The public timeline supplies UTC timestamps, geographic bounds and a rain-rate
scale with Z–R coefficients. Its data PNGs contain values in red and coverage in alpha, on a
Web Mercator raster. Decode values directly and reproject to the existing
PHCOMP geographic grid; the web client's display palette is irrelevant.
"""
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import hmac
import io
import logging
import math
import secrets
import time
from urllib.parse import urlencode, urlsplit

import httpx
import numpy as np
from PIL import Image

from librewxr.config import settings
from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get
from librewxr.sources._helpers import _dbz_float_to_uint8
from .regions import PHCOMP

logger = logging.getLogger(__name__)
_IMAGE_SIZE = 1024
_MAX_NEAREST_OFFSET_SEC = 300
_LAYER = "mosaic-rainrate"


class _SessionTokenParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        token = attributes.get("content")
        name = attributes.get("name")
        if (tag == "meta" and name in ("csrf-token", "embed-grant", "api-sig")
                and token and len(token) <= 1024):
            self.values[name] = token


def _mercator_y(latitude):
    return np.log(np.tan(np.pi / 4 + np.deg2rad(latitude) / 2))


@dataclass(frozen=True)
class _ImageSpec:
    bounds: tuple[float, float, float, float]
    maximum: float
    sqrt: bool
    version: int
    size: int = _IMAGE_SIZE
    mode: str = "dbz"
    zr: tuple[float, float] | None = None

    @property
    def dimensions(self) -> tuple[int, int]:
        west, south, east, north = self.bounds
        xspan = math.radians(east - west)
        yspan = float(_mercator_y(north) - _mercator_y(south))
        # Match PANaHON's Web Mercator image/tile geometry, including its
        # JS Math.round (positive halves round up).
        if xspan >= yspan:
            return self.size, max(1, math.floor(self.size * yspan / xspan + 0.5))
        return max(1, math.floor(self.size * xspan / yspan + 0.5)), self.size


def _image_spec(data: dict) -> _ImageSpec:
    bounds = data.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError("missing radar bounds")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bounds):
        raise ValueError("invalid radar bounds")
    bounds = tuple(float(v) for v in bounds)
    west, south, east, north = bounds
    if not (all(math.isfinite(v) for v in bounds)
            and -180 <= west < east <= 180
            and -85.051129 < south < north < 85.051129):
        raise ValueError("invalid radar bounds")
    scale = data.get("scale")
    if not isinstance(scale, dict) or scale.get("mode") not in ("dbz", "rain"):
        raise ValueError("unsupported radar scale")
    mode = scale["mode"]
    zr = None
    if mode == "rain":
        coefficients = scale.get("zr")
        if scale.get("unit") != "mm/hr" or not isinstance(coefficients, dict):
            raise ValueError("rain scale lacks mm/hr units or Z-R coefficients")
        zr = (coefficients.get("a"), coefficients.get("b"))
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0 for v in zr):
            raise ValueError("invalid radar Z-R coefficients")
    maximum = scale.get("max")
    if (isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(maximum) or not 0 < maximum <= 255):
        raise ValueError("invalid radar scale maximum")
    sqrt = scale.get("sqrt", False)
    if not isinstance(sqrt, bool):
        raise ValueError("invalid radar scale sqrt flag")
    version = data.get("tile_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("invalid radar tile version")
    sizes = data.get("image_sizes", [_IMAGE_SIZE])
    if not isinstance(sizes, list) or _IMAGE_SIZE not in sizes:
        raise ValueError("radar no longer advertises the 1024 image size")
    spec = _ImageSpec(bounds, float(maximum), sqrt, version, _IMAGE_SIZE, mode, zr)
    if min(spec.dimensions) < 2:
        raise ValueError("degenerate radar image bounds")
    return spec


def _decode_numeric_png(
    png: bytes, spec: _ImageSpec, region: RegionDef = PHCOMP,
) -> np.ndarray:
    """Decode/reproject without interpolating echoes or a coverage mask.

    Pixel centres are sampled by nearest neighbour. The output's grid
    nodes use RegionDef's existing west/north + index*pixel_size convention.
    PANaHON uses square Mercator pixels and adjusts the image's bottom edge
    for dimension rounding; use the same extent rather than stretching it.
    """
    with Image.open(io.BytesIO(png)) as image:
        if image.format != "PNG":
            raise ValueError("expected a PNG radar image")
        if image.size == (1, 1):
            raise ValueError("upstream returned a 1x1 placeholder")
        if image.size != spec.dimensions:
            raise ValueError(
                f"radar image size {image.size} differs from {spec.dimensions}"
            )
        if image.mode not in ("RGBA", "LA"):
            raise ValueError("numeric radar image lacks an explicit alpha mask")
        rgba = np.asarray(image.convert("RGBA"))

    west, _, east, north = spec.bounds
    width, height = spec.dimensions
    resolution = math.radians(east - west) / width
    lon = region.west + np.arange(region.width) * region.pixel_size
    lat = region.north - np.arange(region.height) * region._ps_y
    columns = np.floor((np.deg2rad(lon) - math.radians(west)) / resolution).astype(np.int32)
    rows = np.floor((float(_mercator_y(north)) - _mercator_y(lat)) / resolution).astype(np.int32)
    valid_columns = (columns >= 0) & (columns < width)
    valid_rows = (rows >= 0) & (rows < height)
    np.clip(columns, 0, width - 1, out=columns)
    np.clip(rows, 0, height - 1, out=rows)

    values = np.arange(256, dtype=np.float32) / np.float32(255)
    if spec.sqrt:
        values *= values
    values *= np.float32(spec.maximum)
    if spec.mode == "rain":
        # Z = a R**b; zero rain is no echo, not zero dBZ. Only the
        # 256-entry LUT needs floating point work, not the full raster.
        a, b = spec.zr
        values[1:] = 10 * math.log10(a) + 10 * b * np.log10(values[1:])
        values[0] = -32
    lut = _dbz_float_to_uint8(values)
    red = rgba[rows[:, None], columns, 0]
    output = lut[red]
    alpha = rgba[rows[:, None], columns, 3]
    output[alpha == 0] = 0
    output[~valid_rows, :] = 0
    output[:, ~valid_columns] = 0
    return output


class PAGASASource:
    """Fetch the current anonymous API, coalescing concurrent store lookups."""

    _REFRESH_TTL_SEC = 120
    _CACHE_MAX = 24
    _IMAGE_CONCURRENCY = 2

    def __init__(self, base_url: str = "https://panahon.gov.ph"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._frame_cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self._last_fetch_unix = 0.0
        self._refresh_lock = asyncio.Lock()
        self._session_ready = False
        self._session_token: str | None = None
        self._embed_grant: str | None = None
        self._signing_key: str | None = None
        # Follow the public viewer linked from PAGASA's radar page. Its
        # anonymous session supplies signing context; no account is used.
        self._viewer_url = self._base_url + "/?trg=iframe&req=radar.rain-rate"
        self._headers = {"User-Agent": "LibreWXR radar client", "Referer": self._viewer_url}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._session_ready = False
            self._session_token = None
            self._embed_grant = None
            self._signing_key = None
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True,
            )
        return self._client

    async def _prepare_session(self, client, *, force=False) -> bool:
        if self._session_ready and not force:
            return True
        self._session_ready = False
        self._session_token = self._embed_grant = self._signing_key = None
        response = await retry_get(
            client, self._viewer_url, log_name="PAGASA-session",
            headers={"User-Agent": self._headers["User-Agent"], "Accept": "text/html"},
        )
        if response is not None and response.status_code == 200:
            parser = _SessionTokenParser()
            parser.feed(response.text)
            values = parser.values
            if all(values.get(name) for name in ("csrf-token", "embed-grant", "api-sig")):
                self._session_token = values["csrf-token"]
                self._embed_grant = values["embed-grant"]
                self._signing_key = values["api-sig"]
                self._session_ready = True
        if not self._session_ready:
            logger.warning("PAGASA public viewer did not supply complete session context")
        return self._session_ready

    async def _api_get(self, client, url, *, accept, log_name):
        # A lost response may already have consumed the nonce. Sign every
        # transport retry afresh, matching PANaHON's public client protocol.
        for attempt in range(settings.download_retries + 1):
            timestamp = str(int(time.time()))
            nonce = secrets.token_hex(16)
            message = f"GET\n{urlsplit(url).path.strip('/')}\n{timestamp}\n{nonce}"
            signature = hmac.new(self._signing_key.encode(), message.encode(), hashlib.sha256).hexdigest()
            try:
                return await client.get(self._with_session(url), headers={
                    **self._headers, "Accept": accept, "X-Ts": timestamp,
                    "X-Nonce": nonce, "X-Sig": signature, "X-Embed-Grant": self._embed_grant,
                }, follow_redirects=False)
            except (httpx.TransportError, httpx.DecodingError):
                if attempt < settings.download_retries:
                    logger.debug("%s: transport/decode error, retrying", log_name)
                    await asyncio.sleep(1)
                else:
                    logger.warning("%s: transport/decode error after %d retries", log_name, attempt)
        return None

    def _with_session(self, url):
        if self._session_token is None:
            return url
        return url + "&" + urlencode({"token": self._session_token})

    @property
    def _timeline_url(self) -> str:
        return self._base_url + "/api/v1/radar/timeline?" + urlencode({"sublayer": _LAYER})

    def _image_url(self, timestamp: int, spec: _ImageSpec) -> str:
        return self._base_url + "/api/v1/radar-data-image?" + urlencode({
            "sublayer": _LAYER, "t": timestamp, "mode": spec.mode,
            "size": spec.size, "v": spec.version,
        })

    async def fetch_frame(self, region: RegionDef, minutes_ago: int) -> np.ndarray | None:
        now_rounded = int(time.time()) // 600 * 600
        return await self._fetch_for_ts(now_rounded - minutes_ago * 60, region)

    async def fetch_archive_frame(self, region: RegionDef, dt: datetime) -> np.ndarray | None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_for_ts(int(dt.timestamp()), region)

    async def _fetch_for_ts(self, ts: int, region: RegionDef) -> np.ndarray | None:
        if region.name != "PHCOMP":
            raise ValueError(f"PAGASASource cannot decode region {region.name!r}; expected PHCOMP")
        nearest = self._nearest_cached(ts)
        if nearest is None and time.time() - self._last_fetch_unix >= self._REFRESH_TTL_SEC:
            await self._refresh()
            nearest = self._nearest_cached(ts)
        return self._frame_cache[nearest] if nearest is not None else None

    def _nearest_cached(self, ts: int) -> int | None:
        if not self._cache_order:
            return None
        nearest = min(self._cache_order, key=lambda t: abs(t - ts))
        return nearest if abs(nearest - ts) <= _MAX_NEAREST_OFFSET_SEC else None

    async def _refresh(self) -> None:
        async with self._refresh_lock:
            if time.time() - self._last_fetch_unix < self._REFRESH_TTL_SEC:
                return
            self._last_fetch_unix = time.time()
            client = await self._get_client()
            # Refresh the opaque public grant once per coalesced refresh;
            # its lifetime is not part of the published timeline metadata.
            if not await self._prepare_session(client, force=True):
                return
            renewed = False
            async def timeline():
                return await self._api_get(
                    client, self._timeline_url, log_name="PAGASA-timeline", accept="application/json",
                )
            response = await timeline()
            if response is not None and response.status_code in (403, 419):
                renewed = True
                if not await self._prepare_session(client, force=True):
                    return
                response = await timeline()
            if response is None or response.status_code != 200:
                logger.warning("PAGASA timeline fetch failed: status=%s",
                               response.status_code if response is not None else "transport error")
                return
            try:
                payload = response.json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, dict) or payload.get("success") is not True:
                    raise ValueError("invalid radar timeline response")
                entries = data.get("timeline")
                if data.get("no_data") or not entries:
                    logger.warning("PAGASA timeline has no radar data")
                    return
                if not isinstance(entries, list):
                    raise ValueError("invalid radar timeline entries")
                spec = _image_spec(data)
            except (ValueError, TypeError) as error:
                logger.warning("PAGASA timeline rejected: %s", error)
                return

            timestamps = set()
            for entry in entries:
                raw = entry.get("observed_at_unix") if isinstance(entry, dict) else None
                if isinstance(raw, bool) or not isinstance(raw, (int, str)):
                    continue
                try:
                    timestamp = int(raw)
                except ValueError:
                    continue
                if 0 < timestamp < 2**63:
                    timestamps.add(timestamp)
            timestamps = sorted(timestamps)[-self._CACHE_MAX:]
            if not timestamps:
                logger.warning("PAGASA timeline has no valid UTC timestamps")
                return
            # The API exposes no per-frame revision. Re-read the window on
            # refresh so corrections at an existing timestamp replace cached
            # data; TTL coalescing still prevents one refresh per store slot.
            semaphore = asyncio.Semaphore(self._IMAGE_CONCURRENCY)

            async def fetch_one(ts):
                async with semaphore:
                    try:
                        response = await self._api_get(
                            client, self._image_url(ts, spec), log_name="PAGASA-image", accept="image/png",
                        )
                        if response is None:
                            return ts, None, "transport error"
                        if response.status_code != 200:
                            return ts, None, f"HTTP {response.status_code}"
                        grid = await asyncio.to_thread(_decode_numeric_png, response.content, spec)
                        return ts, grid, None
                    except Exception as error:
                        # Decoder errors contain useful shape/format details.
                        # Other exceptions may embed a request URL/session
                        # token, so report only their class.
                        reason = str(error) if isinstance(error, (ValueError, OSError)) else type(error).__name__
                        return ts, None, reason or type(error).__name__

            results = await asyncio.gather(*(fetch_one(ts) for ts in timestamps))
            # The image route can express an expired grant as a 1x1 PNG.
            # Renew only once for an entirely placeholder window or explicit
            # session errors, after in-flight requests finish. Partial product
            # gaps must not trigger one session bootstrap per missing frame.
            session_errors = {"HTTP 403", "HTTP 419", "upstream returned a 1x1 placeholder"}
            retry_session = (all(error == "upstream returned a 1x1 placeholder" for _, _, error in results)
                             or any(error in ("HTTP 403", "HTTP 419") for _, _, error in results))
            if retry_session and not renewed and await self._prepare_session(client, force=True):
                retried = await asyncio.gather(*(fetch_one(ts) for ts, _, error in results if error in session_errors))
                replacements = {result[0]: result for result in retried}
                results = [replacements.get(result[0], result) for result in results]
            failures = Counter()
            for timestamp, grid, error in results:
                if grid is None:
                    failures[error] += 1
                else:
                    self._frame_cache[timestamp] = grid
            self._cache_order = sorted(self._frame_cache)
            while len(self._cache_order) > self._CACHE_MAX:
                self._frame_cache.pop(self._cache_order.pop(0))
            if failures:
                logger.warning("PAGASA: %d/%d radar images unavailable (%s)",
                               sum(failures.values()), len(timestamps),
                               "; ".join(f"{reason}: {count}" for reason, count in sorted(failures.items())))

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
