# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""PAGASA's current numeric API, projection, missing data and cache contract."""
import asyncio
import io
import math
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import httpx
import numpy as np
from PIL import Image
import pytest

from librewxr.data.regions import REGIONS, RegionDef, resolve_regions
from librewxr.config import Settings
from librewxr.sources.regional.southeast_asia.philippines.radar import pagasa
from librewxr.sources.regional.southeast_asia.philippines.radar.pagasa import source as mod
from librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source import (
    PAGASASource, _ImageSpec, _decode_numeric_png, _image_spec,
)

pytestmark = pytest.mark.sources
BOUNDS = [115.41549141305251, 3.801613036809332, 129.51730887177652, 22.45850950564088]
START = 1794670800


def metadata(count=9, start=START):
    return {
        "success": True,
        "data": {
            "timeline": [{"observed_at_unix": start + i * 600, "image_url": ""} for i in range(count)],
            "bounds": list(BOUNDS), "scale": {"mode": "dbz", "max": 80, "sqrt": False},
            "image_sizes": [896, 1024, 1536, 2048], "tile_version": 5,
            "no_data": False,
        },
    }


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def image(spec, value=128, alpha=255):
    width, height = spec.dimensions
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., 0] = value
    pixels[..., 3] = alpha
    return png(pixels)


def client_source(handler):
    source = PAGASASource()
    def public_session(request):
        if request.url.path == "/":
            return httpx.Response(200, text='<meta name="csrf-token" content="anonymous-test-token">',
                                  headers={"Set-Cookie": "pagasa_test=anonymous; Path=/"})
        return handler(request)
    source._client = httpx.AsyncClient(transport=httpx.MockTransport(public_session))
    return source


@pytest.fixture
def small_region():
    return RegionDef(name="TEST", group="TEST", west=1, east=4, south=30, north=60,
                     pixel_size=3/8, pixel_size_y=30/8, grid_width=8, grid_height=8)


class TestRegionAndSettings:
    def test_grid_is_preserved(self):
        region = REGIONS["PHCOMP"]
        assert region.width == region.height == 2048
        assert (region.west, region.east) == pytest.approx((115.4154914129329, 129.51727937415484))
        assert (region.south, region.north) == pytest.approx((3.8016540706290445, 22.458510294136033))
        assert region._ps_y > region.pixel_size
        assert "PHCOMP" in resolve_regions("ALL")
        assert "PHCOMP" in resolve_regions("SOUTHEAST_ASIA")

    def test_stations_and_provider(self, monkeypatch):
        monkeypatch.delenv("LIBREWXR_PAGASA_BASE_URL", raising=False)
        monkeypatch.delenv("LIBREWXR_PAGASA_ENABLED", raising=False)
        settings = Settings(_env_file=None)
        assert settings.pagasa_base_url == "https://panahon.gov.ph"
        contribution = pagasa.radar_provider(settings)
        assert contribution.instance._base_url == settings.pagasa_base_url
        assert len(pagasa.PHCOMP_STATIONS) == 9
        settings.pagasa_enabled = False
        assert pagasa.radar_provider(settings) is None

    def test_custom_origin_is_respected(self):
        source = PAGASASource("https://mirror.example/radar/")
        spec = _image_spec(metadata()["data"])
        assert source._timeline_url == "https://mirror.example/radar/api/v1/radar/timeline?sublayer=mosaic-reflectivity"
        url = urlsplit(source._image_url(START, spec))
        assert url.path == "/radar/api/v1/radar-data-image"
        assert parse_qs(url.query) == {
            "sublayer": ["mosaic-reflectivity"], "t": [str(START)],
            "mode": ["dbz"], "size": ["2048"], "v": ["5"],
        }

    def test_davao_station_is_preserved(self):
        assert pagasa.PHCOMP_STATIONS[6] == pytest.approx((7.13, 125.65), abs=.05)


class TestMetadata:
    def test_uses_mercator_aspect_ratio(self):
        spec = _image_spec(metadata()["data"])
        width, height = spec.dimensions
        xspan = math.radians(BOUNDS[2] - BOUNDS[0])
        yspan = math.asinh(math.tan(math.radians(BOUNDS[3]))) - math.asinh(math.tan(math.radians(BOUNDS[1])))
        assert height == 2048
        assert width == math.floor(2048 * xspan / yspan + .5)
        assert width != round(2048 * (BOUNDS[2] - BOUNDS[0]) / (BOUNDS[3] - BOUNDS[1]))

    @pytest.mark.parametrize("key,value", [
        ("bounds", None), ("bounds", [1, 2, 3]), ("bounds", [3, 2, 1, 4]),
        ("bounds", [1, 2, float("nan"), 4]), ("bounds", [1, -90, 3, 90]),
        ("bounds", [True, 2, 3, 4]), ("bounds", [1, 2, 3, 2]),
        ("scale", None), ("scale", {"mode": "rain", "max": 80}),
        ("scale", {"mode": "dbz", "max": 0}), ("scale", {"mode": "dbz", "max": float("inf")}),
        ("scale", {"mode": "dbz", "max": True}),
        ("scale", {"mode": "dbz", "max": 80, "sqrt": "false"}),
        ("tile_version", 0), ("tile_version", True), ("image_sizes", []),
    ])
    def test_rejects_unusable_contract(self, key, value):
        data = metadata()["data"]
        data[key] = value
        with pytest.raises(ValueError):
            _image_spec(data)


class TestNumericDecode:
    def test_value_alpha_and_scale(self):
        spec = _ImageSpec((0, 0, 10, 10), 80, False, 5, 16)
        width, height = spec.dimensions
        region = RegionDef(name="TEST", group="TEST", west=0, east=10, south=0, north=10,
                           pixel_size=10/width, pixel_size_y=10/height, grid_width=width, grid_height=height)
        data = np.zeros((height, width, 4), dtype=np.uint8)
        data[..., 0] = 128
        data[..., 3] = 255
        data[0, :4, 0] = [0, 128, 255, 255]
        data[0, 3, 3] = 0
        decoded = _decode_numeric_png(png(data), spec, region)
        assert decoded.dtype == np.uint8 and decoded.shape == (height, width)
        assert decoded[0, :4].tolist() == [64, 144, 224, 0]

    def test_sqrt_scale_is_applied_before_librewxr_encoding(self, small_region):
        spec = _ImageSpec((1, 30, 4, 60), 80, True, 5, 128)
        decoded = _decode_numeric_png(image(spec, value=128), spec, small_region)
        expected = int((80 * (128/255)**2 + 32) * 2)
        assert np.all(decoded == expected)

    def test_mercator_rows_are_reprojected_not_stretched(self, small_region):
        spec = _ImageSpec((1, 30, 4, 60), 80, False, 5, 256)
        width, height = spec.dimensions
        values = np.zeros((height, width, 4), dtype=np.uint8)
        values[..., 0] = np.arange(height)[:, None]
        values[..., 3] = 255
        decoded = _decode_numeric_png(png(values), spec, small_region)
        resolution = math.radians(3) / width
        # Target row 4 is latitude 45. Mercator is not linear in latitude.
        row = math.floor((math.asinh(math.tan(math.radians(60))) - math.asinh(math.tan(math.radians(45)))) / resolution)
        assert row != round(height/2)
        assert decoded[4, 3] == int((row/255*80 + 32)*2)

    def test_outside_source_extent_is_nodata(self):
        region = RegionDef(name="TEST", group="TEST", west=0, east=10, south=0, north=10,
                           pixel_size=1, grid_width=10, grid_height=10)
        spec = _ImageSpec((2, 2, 8, 8), 80, False, 5, 64)
        decoded = _decode_numeric_png(image(spec, 255), spec, region)
        assert decoded[5, 5] == 224
        assert not decoded[:, :2].any() and not decoded[:2, :].any()
        assert not decoded[:, 8:].any() and not decoded[9:, :].any()

    def test_full_sized_clear_image_is_valid(self, small_region):
        spec = _ImageSpec((1, 30, 4, 60), 80, False, 5, 128)
        assert not _decode_numeric_png(image(spec, alpha=0), spec, small_region).any()

    @pytest.mark.parametrize("kind", ["placeholder", "wrong-size", "corrupt", "rgb"])
    def test_bad_image_is_rejected(self, kind, small_region):
        spec = _ImageSpec((1, 30, 4, 60), 80, False, 5, 128)
        if kind == "placeholder":
            body = png(np.zeros((1, 1, 4), dtype=np.uint8))
        elif kind == "corrupt":
            body = b"not a png"
        elif kind == "wrong-size":
            body = png(np.zeros((4, 4, 4), dtype=np.uint8))
        else:
            width, height = spec.dimensions
            body = png(np.zeros((height, width, 3), dtype=np.uint8))
        with pytest.raises((ValueError, OSError)):
            _decode_numeric_png(body, spec, small_region)


class TestRefresh:
    async def test_anonymous_session_refreshes_expired_token(self):
        sessions = timelines = 0
        def handler(request):
            nonlocal sessions, timelines
            if request.url.path == "/":
                sessions += 1
                return httpx.Response(200, text=f"<meta content='token{sessions}' name='csrf-token'>",
                                      headers={"Set-Cookie": f"session={sessions}; Path=/"})
            timelines += 1
            assert request.url.params["token"] == f"token{sessions}"
            assert request.headers["cookie"] == f"session={sessions}"
            assert request.url.params["sublayer"] == "mosaic-reflectivity"
            return httpx.Response(419) if timelines == 1 else httpx.Response(200, json=metadata(0))
        source = PAGASASource()
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await source._refresh()
            assert sessions == timelines == 2
            source._last_fetch_unix = 0
            await source._refresh()
            assert sessions == 2 and timelines == 3
        finally:
            await source.close()

    async def test_tokenless_compatible_mirror_is_supported(self):
        def handler(request):
            if request.url.path == "/":
                return httpx.Response(404)
            assert "token" not in request.url.params
            return httpx.Response(200, json=metadata(0))
        source = PAGASASource()
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await source._refresh()
            assert not source._cache_order
        finally:
            await source.close()

    async def test_all_nine_frames_and_timestamp_requests(self):
        data = metadata()
        spec = _image_spec(data["data"])
        body = image(spec)
        seen = []
        def handler(request):
            assert request.headers["referer"] == "https://panahon.gov.ph/"
            assert request.headers["user-agent"].startswith("LibreWXR")
            assert request.url.params["token"] == "anonymous-test-token"
            assert request.headers["cookie"] == "pagasa_test=anonymous"
            assert request.url.params["sublayer"] == "mosaic-reflectivity"
            if request.url.path.endswith("/timeline"):
                return httpx.Response(200, json=data)
            seen.append(int(request.url.params["t"]))
            assert request.url.path == "/api/v1/radar-data-image"
            assert request.url.params["mode"] == "dbz" and request.url.params["v"] == "5"
            return httpx.Response(200, content=body)
        source = client_source(handler)
        try:
            await source._refresh()
            assert source._cache_order == sorted(seen) == [START + i*600 for i in range(9)]
            assert all(grid.shape == (2048, 2048) for grid in source._frame_cache.values())
            assert all(grid[1000, 1000] == 144 for grid in source._frame_cache.values())
            body = image(spec, value=255)
            source._last_fetch_unix = 0
            await source._refresh()
            assert len(seen) == 18
            assert all(grid[1000, 1000] == 224 for grid in source._frame_cache.values())
        finally:
            await source.close()

    async def test_partial_failure_does_not_poison_other_frames(self, caplog):
        data = metadata(3)
        spec = _image_spec(data["data"])
        valid = image(spec)
        placeholder = png(np.zeros((1, 1, 4), dtype=np.uint8))
        def handler(request):
            if request.url.path.endswith("/timeline"):
                return httpx.Response(200, json=data)
            timestamp = int(request.url.params["t"])
            if timestamp == START:
                return httpx.Response(200, content=placeholder)
            return httpx.Response(404) if timestamp == START+600 else httpx.Response(200, content=valid)
        source = client_source(handler)
        try:
            await source._refresh()
            assert source._cache_order == [START+1200]
            assert "1x1 placeholder" in caplog.text and "HTTP 404" in caplog.text
        finally:
            await source.close()

    async def test_placeholder_retry_recovers_without_fabricating_clear_frames(self):
        data = metadata(1)
        valid = image(_image_spec(data["data"]), value=255)
        placeholder = png(np.zeros((1, 1, 4), dtype=np.uint8))
        available = False
        def handler(request):
            if request.url.path.endswith("/timeline"):
                return httpx.Response(200, json=data)
            return httpx.Response(200, content=valid if available else placeholder)
        source = client_source(handler)
        try:
            await source._refresh()
            assert not source._frame_cache and source._nearest_cached(START) is None
            available = True
            source._last_fetch_unix = 0
            await source._refresh()
            assert source._cache_order == [START]
            assert source._frame_cache[START][1000, 1000] == 224
            available = False
            source._last_fetch_unix = 0
            await source._refresh()
            assert source._frame_cache[START][1000, 1000] == 224
        finally:
            await source.close()

    @pytest.mark.parametrize("payload", [[], {}, {"success": False}, {"success": True, "data": []},
        {"success": True, "data": {"timeline": []}}, {"success": True, "data": {"timeline": [1], "no_data": True}}])
    async def test_invalid_or_empty_timeline_does_not_fetch_images(self, payload):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, json=payload)
        source = client_source(handler)
        try:
            await source._refresh()
            assert not source._cache_order and calls == ["/api/v1/radar/timeline"]
        finally:
            await source.close()

    @pytest.mark.parametrize("failure", ["status", "transport", "non-json", "metadata"])
    async def test_timeline_failure_preserves_cache_and_has_diagnostics(self, failure, caplog, monkeypatch):
        monkeypatch.setattr(mod.retry_get.__globals__["settings"], "download_retries", 0)
        def handler(request):
            if failure == "transport":
                raise httpx.ConnectError("DNS failure", request=request)
            if failure == "status":
                return httpx.Response(503)
            if failure == "non-json":
                return httpx.Response(200, content=b"generic image instead of JSON")
            payload = metadata(); payload["data"]["scale"]["mode"] = "rain"
            return httpx.Response(200, json=payload)
        source = client_source(handler)
        source._frame_cache[START] = np.ones((2, 2), dtype=np.uint8)
        source._cache_order = [START]
        try:
            await source._refresh()
            assert source._cache_order == [START] and np.all(source._frame_cache[START] == 1)
            assert "PAGASA" in caplog.text
        finally:
            await source.close()

    async def test_refresh_coalesces_and_limits_parallel_images(self):
        data = metadata(5); spec = _image_spec(data["data"]); body = image(spec)
        active = maximum = timelines = 0
        async def handler(request):
            nonlocal active, maximum, timelines
            if request.url.path.endswith("/timeline"):
                timelines += 1
                await asyncio.sleep(.01)
                return httpx.Response(200, json=data)
            active += 1; maximum = max(maximum, active)
            await asyncio.sleep(.01)
            active -= 1
            return httpx.Response(200, content=body)
        source = client_source(handler)
        try:
            await asyncio.gather(*(source._refresh() for _ in range(6)))
            assert timelines == 1 and maximum <= 2 and len(source._cache_order) == 5
        finally:
            await source.close()

    async def test_malformed_timestamps_deduplicate_and_bound_downloads(self, monkeypatch):
        data = metadata(30)
        data["data"]["timeline"] += [{"observed_at_unix": START}, {}, None, {"observed_at_unix": True}, {"observed_at_unix": "bad"}]
        seen = []
        def handler(request):
            if request.url.path.endswith("/timeline"):
                return httpx.Response(200, json=data)
            seen.append(int(request.url.params["t"]))
            return httpx.Response(200, content=b"mock image")
        monkeypatch.setattr(mod, "_decode_numeric_png", lambda *args:np.ones((2, 2), dtype=np.uint8))
        source = client_source(handler)
        try:
            await source._refresh()
            assert sorted(seen) == [START+i*600 for i in range(6, 30)]
            assert len(source._cache_order) == 24
        finally:
            await source.close()


class TestLookup:
    def test_nearest_frame_does_not_extend_staleness(self):
        source = PAGASASource()
        assert source._nearest_cached(START) is None
        source._cache_order = [START, START+600]
        assert source._nearest_cached(START) == START
        assert source._nearest_cached(START+300) == START
        assert source._nearest_cached(START+901) is None

    async def test_real_timestamps_and_timezone_are_preserved(self, monkeypatch):
        source = PAGASASource()
        grid = np.ones((2, 2), dtype=np.uint8)
        source._frame_cache[START] = grid; source._cache_order = [START]
        monkeypatch.setattr(mod.time, "time", lambda:START+600)
        try:
            assert await source.fetch_frame(REGIONS["PHCOMP"], 10) is grid
            for date in (datetime.fromtimestamp(START, timezone.utc), datetime.fromtimestamp(START, timezone.utc).replace(tzinfo=None)):
                assert await source.fetch_archive_frame(REGIONS["PHCOMP"], date) is grid
            source._last_fetch_unix = START+600
            assert await source.fetch_frame(REGIONS["PHCOMP"], 0) is None
            assert await source.fetch_archive_frame(REGIONS["PHCOMP"], datetime(2024, 1, 1, tzinfo=timezone.utc)) is None
            with pytest.raises(ValueError, match="PHCOMP"):
                await source.fetch_frame(REGIONS["MYPENINSULAR"], 0)
        finally:
            await source.close()
