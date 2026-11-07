# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Environment and Climate Change Canada radar composite source.

Fetches the RADAR_1KM_RRAI composite from MSC GeoMet WMS as a
pre-colored PNG (MSC does not publish raw radar data in any open
format).  Uses the "Radar-Rain_Dis-14colors" discrete style and
reverse-engineers the color palette back to precipitation rate,
then converts to dBZ via Marshall-Palmer.

The WMS time dimension gives a rolling ~3-hour history at 6-minute
cadence — sufficient for live + archive playback.

License: ECCC open data — see the
`Canadian Open Government Licence <https://open.canada.ca/en/open-government-licence-canada>`_.
Attribution recorded in the top-level README and ``docs/coverage.md``.
"""
import io
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
from PIL import Image

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get
from librewxr.sources._helpers import _dbz_float_to_uint8

logger = logging.getLogger(__name__)


# RADAR_1KM_RRAI discrete palette (Radar-Rain_Dis-14colors style).
#
# MSC's WMS serves Canadian radar as pre-colored PNG only (no WCS/TIFF
# access to raw data).  The "Dis" style uses 14 discrete buckets mapping
# RGB → precipitation rate in mm/h.  Each entry is (R, G, B, rate) where
# rate is the geometric mean of the bucket's [lower, upper) edges — the
# typical value within the bucket, given that precipitation rates are
# log-distributed.  Using lower-edge labels directly systematically
# under-reports by ~30% within each bucket and (crucially) pushes the
# lowest bucket below the 10 dBZ noise floor, making broad light-rain
# regions invisible compared to Rain Viewer and other MSC consumers.
#
# The top bucket (≥200 mm/h, unbounded) is represented as 250 mm/h —
# a reasonable "typical extreme" that preserves headroom below the
# clamp ceiling without requiring an arbitrary upper edge.
#
# Bucket edges from the legend: 0.1, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0,
# 24.0, 32.0, 50.0, 64.0, 100.0, 125.0, 200.0, (∞).
#
# Colors were extracted from the legend graphic; composite pixels are
# within ±1 of these values due to server-side rendering rounding, so we
# use nearest-anchor lookup with a small Euclidean distance threshold.
_MSC_CANADA_PALETTE: tuple[tuple[int, int, int, float], ...] = (
    (152, 203, 254, 0.3162),   # √(0.1 × 1.0)
    (0, 152, 254, 1.4142),     # √(1.0 × 2.0)
    (0, 254, 102, 2.8284),     # √(2.0 × 4.0)
    (0, 203, 0, 5.6569),       # √(4.0 × 8.0)
    (0, 152, 0, 9.7980),       # √(8.0 × 12.0)
    (0, 102, 0, 13.8564),      # √(12.0 × 16.0)
    (254, 254, 0, 19.5959),    # √(16.0 × 24.0)
    (254, 203, 0, 27.7128),    # √(24.0 × 32.0)
    (254, 152, 0, 40.0),       # √(32.0 × 50.0)
    (254, 102, 0, 56.5685),    # √(50.0 × 64.0)
    (254, 0, 0, 80.0),         # √(64.0 × 100.0)
    (254, 2, 152, 111.8034),   # √(100.0 × 125.0)
    (152, 51, 203, 158.1139),  # √(125.0 × 200.0)
    (102, 0, 152, 250.0),      # top bucket (unbounded): typical extreme
)

# Max Euclidean RGB distance for nearest-anchor matching.  Legend vs
# composite colors differ by ≤±1 per channel (≈1.7 total); any pixel
# farther than this from every anchor is probably an artifact or
# unexpected color and is treated as no-data.
_MSC_CANADA_MAX_RGB_DIST = 8.0

# Bound the (pixels, palette, RGB) distance buffers independently of image size.
_MSC_CANADA_DECODE_CHUNK_PIXELS = 65536


def _mmhr_to_dbz(rate: np.ndarray) -> np.ndarray:
    """Convert precipitation rate (mm/h) to reflectivity (dBZ).

    Uses the Marshall-Palmer Z-R relationship: Z = 200 * R^1.6.
    NaN inputs propagate as NaN (no-data).  Rates ≤ 0 map to NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        z = 200.0 * np.power(rate, 1.6)
        dbz = 10.0 * np.log10(z)
    dbz[~np.isfinite(dbz)] = np.nan
    return dbz


def _decode_msc_canada_png(data: bytes) -> np.ndarray | None:
    """Decode an MSC GeoMet WMS PNG into a uint8 dBZ array.

    Steps:
    1. Open as RGBA (transparent pixels → no-data).
    2. For each opaque pixel, find the nearest palette anchor in RGB
       space.  Pixels beyond the distance threshold become no-data.
    3. Convert anchor mm/h values to dBZ via Marshall-Palmer.
    4. Encode to uint8 using the shared scheme: (dBZ + 32) * 2, clamped.
    """
    try:
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
    except Exception:
        logger.exception("Failed to decode MSC Canada PNG")
        return None

    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, 4)
    out = np.empty(len(pixels), dtype=np.uint8)
    anchors_rgb = np.array(
        [(r, g, b) for r, g, b, _ in _MSC_CANADA_PALETTE], dtype=np.int32
    )
    anchors_rate = np.array(
        [rate for *_, rate in _MSC_CANADA_PALETTE], dtype=np.float32
    )

    # Keep the original per-pixel distance and conversion calculations, but
    # materialize their large palette-axis buffers only for a bounded block.
    for start in range(0, len(pixels), _MSC_CANADA_DECODE_CHUNK_PIXELS):
        block = pixels[start:start + _MSC_CANADA_DECODE_CHUNK_PIXELS]
        # Squared RGB distances overflow int16; retain int32 differences
        # and NumPy's original sum dtype and first-minimum tie-breaking.
        rgb = block[:, :3].astype(np.int32)
        diffs = rgb[:, None, :] - anchors_rgb[None, :, :]
        dist2 = np.sum(diffs * diffs, axis=2)
        nearest_idx = np.argmin(dist2, axis=1)
        nearest_dist2 = dist2[np.arange(len(block)), nearest_idx]
        rate = anchors_rate[nearest_idx]
        valid = (block[:, 3] > 0) & (
            nearest_dist2 <= _MSC_CANADA_MAX_RGB_DIST ** 2
        )
        rate = np.where(valid, rate, np.nan)
        dbz = _mmhr_to_dbz(rate)
        dbz = np.where(np.isnan(dbz), -33.0, dbz)
        out[start:start + len(block)] = _dbz_float_to_uint8(dbz)

    return out.reshape(h, w)


class MSCCanadaSource:
    """Environment and Climate Change Canada radar composite source.

    Fetches the RADAR_1KM_RRAI composite from MSC GeoMet WMS as a
    pre-colored PNG (MSC does not publish raw radar data in any open
    format).  Uses the "Radar-Rain_Dis-14colors" discrete style and
    reverse-engineers the color palette back to precipitation rate,
    then converts to dBZ via Marshall-Palmer.

    The WMS time dimension gives a rolling ~3-hour history at 6-minute
    cadence — sufficient for live + archive playback.
    """

    _WMS_PATH = "/geomet"
    _STYLE = "Radar-Rain_Dis-14colors"
    _LAYER = "RADAR_1KM_RRAI"

    def __init__(self, base_url: str = "https://geo.weather.gc.ca"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    def _build_url(self, region: RegionDef, time_iso: str | None) -> str:
        params = [
            "SERVICE=WMS",
            "VERSION=1.3.0",
            "REQUEST=GetMap",
            f"LAYERS={self._LAYER}",
            f"STYLES={self._STYLE}",
            "CRS=EPSG:4326",
            # WMS 1.3.0 EPSG:4326 axis order is (lat, lon)
            f"BBOX={region.south},{region.west},{region.north},{region.east}",
            f"WIDTH={region.width}",
            f"HEIGHT={region.height}",
            "FORMAT=image/png",
            "TRANSPARENT=TRUE",
        ]
        if time_iso:
            params.append(f"TIME={time_iso}")
        return f"{self._base_url}{self._WMS_PATH}?" + "&".join(params)

    # MSC publishes radar composites on a ~6-minute cadence.  When a
    # requested timestamp is not yet available (common for the most recent
    # slots or when our aligned timestamps drift slightly ahead of real
    # time), we fall back to progressively older 6-minute slots.
    _CADENCE_SEC = 360  # 6 minutes
    _MAX_TIME_RETRIES = 2

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        """Fetch a live frame.  minutes_ago=0 → server default (latest)."""
        if minutes_ago <= 0:
            return await self._fetch(region, None)
        target_ts = int(time.time()) - minutes_ago * 60
        target = datetime.fromtimestamp(target_ts, tz=timezone.utc)
        return await self._fetch_with_fallback(region, target)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        """Fetch a specific historical frame via WMS TIME parameter."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_with_fallback(region, dt)

    async def _fetch(
        self, region: RegionDef, time_iso: str | None
    ) -> np.ndarray | None:
        url = self._build_url(region, time_iso)
        try:
            client = await self._get_client()
            resp = await retry_get(client, url, log_name="MSC Canada")
            if resp is None:
                return None
            if resp.status_code != 200:
                logger.warning(
                    "MSC Canada WMS fetch failed: HTTP %d (time=%s)",
                    resp.status_code, time_iso,
                )
                return None
            # MSC returns a ServiceExceptionReport (XML) with 200 when the
            # requested TIME is not yet available — this is normal for the
            # most recent slots, not an error worth warning about.
            if resp.headers.get("content-type", "").startswith("text/xml"):
                logger.debug(
                    "MSC Canada WMS returned XML exception (time=%s)",
                    time_iso,
                )
                return None
            return _decode_msc_canada_png(resp.content)
        except Exception:
            logger.exception("Error fetching MSC Canada WMS")
            return None

    async def _fetch_with_fallback(
        self, region: RegionDef, target_dt: datetime
    ) -> np.ndarray | None:
        """Fetch with progressive fallback to older MSC timesteps.

        MSC cadence is ~6 minutes and the most recent 1-2 slots are often
        not yet published.  If the requested TIME returns an XML exception,
        step backwards in 6-minute increments up to ``_MAX_TIME_RETRIES``.
        """
        for attempt in range(self._MAX_TIME_RETRIES + 1):
            ts_iso = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            result = await self._fetch(region, ts_iso)
            if result is not None:
                if attempt > 0:
                    logger.debug(
                        "MSC Canada fallback succeeded: %s (attempt %d)",
                        ts_iso, attempt,
                    )
                return result
            target_dt -= timedelta(seconds=self._CADENCE_SEC)

        logger.debug(
            "MSC Canada WMS no data after %d fallback attempts",
            self._MAX_TIME_RETRIES,
        )
        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
