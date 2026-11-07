# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.sources

from librewxr.data.regions import REGIONS, resolve_regions
from librewxr.sources.regional.north_america.canada.radar.msc_canada import (
    _MSC_CANADA_PALETTE,
    _decode_msc_canada_png,
    _mmhr_to_dbz,
)
from librewxr.tiles.coordinates import tile_overlaps_region


class TestCanadaRegion:
    def test_cacomp_in_regions(self):
        assert "CACOMP" in REGIONS

    def test_cacomp_is_latlon(self):
        r = REGIONS["CACOMP"]
        assert r.proj == "latlon"
        assert r.group == "CANADA"

    def test_cacomp_bounds(self):
        r = REGIONS["CACOMP"]
        assert r.west == -141.0
        assert r.east == -52.0
        assert r.south == 41.0
        assert r.north == 84.0

    def test_cacomp_dimensions_reasonable(self):
        r = REGIONS["CACOMP"]
        # Must stay under typical WMS server size caps (~4096 on each axis)
        assert r.width < 4096
        assert r.height < 4096
        # Must be big enough to be useful
        assert r.width > 1000
        assert r.height > 500

    def test_canada_group_resolution(self):
        assert resolve_regions("CANADA") == ["CACOMP"]

    def test_all_includes_canada(self):
        assert "CACOMP" in resolve_regions("ALL")

    def test_mixed_regions(self):
        result = resolve_regions("CONUS,CANADA")
        assert "USCOMP" in result
        assert "CACOMP" in result


class TestMarshallPalmer:
    def test_known_values(self):
        # Z = 200 * R^1.6, dBZ = 10*log10(Z)
        # R=1 mm/h → Z=200 → dBZ=23.01
        assert _mmhr_to_dbz(np.array([1.0]))[0] == pytest.approx(23.01, abs=0.1)
        # R=100 mm/h → Z=200 * 100^1.6 = 200 * 10^3.2 ≈ 316228 → dBZ≈55.0
        assert _mmhr_to_dbz(np.array([100.0]))[0] == pytest.approx(55.0, abs=0.5)

    def test_monotonic(self):
        rates = np.array([0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0])
        dbz = _mmhr_to_dbz(rates)
        assert np.all(np.diff(dbz) > 0), "dBZ must be monotonic in rain rate"

    def test_nan_propagates(self):
        rates = np.array([1.0, np.nan, 5.0])
        dbz = _mmhr_to_dbz(rates)
        assert not np.isnan(dbz[0])
        assert np.isnan(dbz[1])
        assert not np.isnan(dbz[2])

    def test_zero_becomes_nan(self):
        # log(0) is undefined; zero rates must become no-data, not -inf
        dbz = _mmhr_to_dbz(np.array([0.0]))
        assert np.isnan(dbz[0])


class TestPaletteDecoder:
    def _make_png(self, pixels: list[tuple[int, int, int, int]]) -> bytes:
        """Build a tiny Nx1 RGBA PNG from a list of pixels."""
        arr = np.array([pixels], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_transparent_becomes_zero(self):
        png = self._make_png([(0, 0, 0, 0), (0, 0, 0, 0)])
        out = _decode_msc_canada_png(png)
        assert out is not None
        assert out.shape == (1, 2)
        assert np.all(out == 0)

    def test_every_anchor_decodes(self):
        """Every palette anchor must round-trip to a non-zero uint8 dBZ."""
        pixels = [(r, g, b, 255) for r, g, b, _ in _MSC_CANADA_PALETTE]
        png = self._make_png(pixels)
        out = _decode_msc_canada_png(png)
        assert out is not None
        assert out.shape == (1, len(_MSC_CANADA_PALETTE))
        # All anchors encode rates ≥ 0.1 mm/h → dBZ ≥ 7 → uint8 ≥ 78
        assert np.all(out[0] > 0)
        # Higher rates must encode to higher uint8 values (monotonic palette)
        assert np.all(np.diff(out[0].astype(int)) > 0)

    def test_off_by_one_rgb_tolerance(self):
        """Composite pixels are within ±1 of legend colors — must still decode."""
        # Take anchor #1 (1.0 mm/h) and shift every channel by +1
        r, g, b, _ = _MSC_CANADA_PALETTE[1]
        png = self._make_png([(r + 1, g + 1, b + 1, 255)])
        out = _decode_msc_canada_png(png)
        # Should decode to the same value as the exact anchor
        exact_png = self._make_png([(r, g, b, 255)])
        exact = _decode_msc_canada_png(exact_png)
        assert out[0, 0] == exact[0, 0]

    def test_unknown_color_becomes_nodata(self):
        # Pure magenta (255, 0, 255) isn't in the palette and is far from
        # every anchor — should be treated as no-data.
        png = self._make_png([(255, 0, 255, 255)])
        out = _decode_msc_canada_png(png)
        assert out is not None
        assert out[0, 0] == 0

    def test_lowest_bucket_encoding(self):
        # Lowest bucket [0.1, 1.0) mm/h, geomean = 0.316 mm/h → ~15 dBZ
        # → uint8 = (15+32)*2 = 94.  Critically this is ABOVE the default
        # 10 dBZ noise floor (uint8 84), so light rain stays visible.
        r, g, b, _ = _MSC_CANADA_PALETTE[0]
        png = self._make_png([(r, g, b, 255)])
        out = _decode_msc_canada_png(png)
        assert out[0, 0] == pytest.approx(94, abs=2)

    def test_lowest_bucket_clears_noise_floor(self):
        # Explicit guard: the lowest bucket must round-trip to a dBZ
        # value clearly above the default noise_floor_dbz (10.0).
        r, g, b, _ = _MSC_CANADA_PALETTE[0]
        png = self._make_png([(r, g, b, 255)])
        out = _decode_msc_canada_png(png)
        dbz = (int(out[0, 0]) / 2) - 32
        assert dbz > 12.0, f"lowest bucket dBZ={dbz} too close to noise floor"

    def test_highest_bucket_encoding(self):
        # Highest bucket (≥200 mm/h, represented as 250) → ~61.3 dBZ
        # → uint8 = (61.3+32)*2 ≈ 187
        r, g, b, _ = _MSC_CANADA_PALETTE[-1]
        png = self._make_png([(r, g, b, 255)])
        out = _decode_msc_canada_png(png)
        assert out[0, 0] == pytest.approx(187, abs=2)

    def test_bad_png_returns_none(self):
        out = _decode_msc_canada_png(b"not a png")
        assert out is None


class TestCanadaTileOverlap:
    def test_tile_over_canada_overlaps(self):
        region = REGIONS["CACOMP"]
        # z=4, x=4, y=5 is over northern North America
        assert tile_overlaps_region(region, z=4, x=4, y=5)

    def test_tile_over_europe_does_not_overlap(self):
        region = REGIONS["CACOMP"]
        # z=4, x=8, y=5 is over central Europe
        assert not tile_overlaps_region(region, z=4, x=8, y=5)


def _scalar_palette_reference(rgba):
    """Independent integer-distance oracle, without pixel/palette broadcasting."""
    from librewxr.sources._helpers import _dbz_float_to_uint8
    from librewxr.sources.regional.north_america.canada.radar.msc_canada import source

    palette = source._MSC_CANADA_PALETTE
    rates = np.array([row[3] for row in palette], dtype=np.float32)
    codes = _dbz_float_to_uint8(_mmhr_to_dbz(rates))
    result = np.zeros(rgba.shape[:2], dtype=np.uint8)
    for y, x in np.ndindex(result.shape):
        pixel = rgba[y, x]
        if pixel[3] == 0:
            continue
        distances = [sum((int(pixel[c]) - anchor[c]) ** 2 for c in range(3)) for anchor in palette]
        nearest = min(range(len(palette)), key=distances.__getitem__)
        if distances[nearest] <= source._MSC_CANADA_MAX_RGB_DIST ** 2:
            result[y, x] = codes[nearest]
    return result


def _save_test_png(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class TestBoundedPaletteDecoder:
    @pytest.mark.parametrize("shape", [(1, 1), (1, 6), (1, 7), (1, 8), (2, 7), (3, 5), (5, 3), (4, 4)])
    def test_exact_output_across_pixel_block_boundaries(self, monkeypatch, shape):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada import source

        monkeypatch.setattr(source, "_MSC_CANADA_DECODE_CHUNK_PIXELS", 7)
        rng = np.random.default_rng(42)
        count = shape[0] * shape[1]
        anchors = np.array([row[:3] for row in _MSC_CANADA_PALETTE], dtype=np.int32)
        colors = anchors[np.arange(count) % len(anchors)] + rng.integers(-9, 10, (count, 3))
        pixels = np.empty((count, 4), dtype=np.uint8)
        pixels[:, :3] = np.clip(colors, 0, 255)
        pixels[:, 3] = np.resize([0, 1, 127, 255], count)
        rgba = pixels.reshape(*shape, 4)
        result = _decode_msc_canada_png(_save_test_png(Image.fromarray(rgba)))
        np.testing.assert_array_equal(result, _scalar_palette_reference(rgba))
        assert result.shape == shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("mode", ["RGBA", "RGB", "P", "L"])
    def test_png_modes_match_scalar_reference(self, mode):
        pixels = [(r, g, b, [0, 1, 127, 255][i % 4])
                  for i, (r, g, b, _) in enumerate(_MSC_CANADA_PALETTE * 2)]
        image = Image.fromarray(np.array(pixels, dtype=np.uint8).reshape(4, 7, 4)).convert(mode)
        png = _save_test_png(image)
        rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
        np.testing.assert_array_equal(_decode_msc_canada_png(png), _scalar_palette_reference(rgba))

    def test_distance_threshold_and_nonzero_alpha(self):
        rgba = np.array([[(8, 152, 254, 255), (8, 153, 254, 255),
                          (8, 152, 254, 0), (8, 152, 254, 1), (255, 255, 255, 255)]], dtype=np.uint8)
        result = _decode_msc_canada_png(_save_test_png(Image.fromarray(rgba)))
        assert result[0, 0] > 0  # Squared distance exactly 64 is accepted.
        assert result[0, 1] == 0  # Squared distance 65 is rejected.
        assert result[0, 2] == 0
        assert result[0, 3] == result[0, 0]
        assert result[0, 4] == 0

    def test_equal_distances_choose_first_palette_entry(self, monkeypatch):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada import source

        monkeypatch.setattr(source, "_MSC_CANADA_PALETTE", ((100, 100, 100, 1.4142), (102, 100, 100, 2.8284)))
        rgba = np.array([[(101, 100, 100, 255), (100, 100, 100, 255), (102, 100, 100, 255)]], dtype=np.uint8)
        result = _decode_msc_canada_png(_save_test_png(Image.fromarray(rgba)))
        assert result[0, 0] == result[0, 1]
        assert result[0, 0] != result[0, 2]
        np.testing.assert_array_equal(result, _scalar_palette_reference(rgba))

    def test_distance_workspaces_are_bounded(self, monkeypatch):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada import source

        monkeypatch.setattr(source, "_MSC_CANADA_DECODE_CHUNK_PIXELS", 7)
        pixels = [(r, g, b, 255) for r, g, b, _ in _MSC_CANADA_PALETTE * 2][:23]
        rgba = np.array([pixels], dtype=np.uint8)
        block_sizes = []
        original_argmin = np.argmin

        def argmin(distances, *args, **kwargs):
            block_sizes.append(distances.shape[0])
            assert distances.shape[0] <= 7
            return original_argmin(distances, *args, **kwargs)

        monkeypatch.setattr(source.np, "argmin", argmin)
        result = _decode_msc_canada_png(_save_test_png(Image.fromarray(rgba)))
        assert block_sizes == [7, 7, 7, 2]
        np.testing.assert_array_equal(result, _scalar_palette_reference(rgba))


    @pytest.mark.parametrize("alpha", [0, 1, 127, 255])
    def test_uniform_alpha_skips_only_fully_transparent_pixels(self, monkeypatch, alpha):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada import source

        monkeypatch.setattr(source, "_MSC_CANADA_DECODE_CHUNK_PIXELS", 7)
        rgba = np.empty((3, 5, 4), dtype=np.uint8)
        rgba[:, :, :3] = _MSC_CANADA_PALETTE[1][:3]
        rgba[:, :, 3] = alpha
        expected = _scalar_palette_reference(rgba)
        sizes = []
        original_argmin = np.argmin

        def argmin(distances, *args, **kwargs):
            sizes.append(distances.shape[0])
            return original_argmin(distances, *args, **kwargs)

        monkeypatch.setattr(source.np, "argmin", argmin)
        result = _decode_msc_canada_png(_save_test_png(Image.fromarray(rgba)))
        assert sizes == ([] if alpha == 0 else [7, 7, 1])
        np.testing.assert_array_equal(result, expected)
        assert bool(np.any(result)) == (alpha > 0)

    def test_mixed_blocks_do_palette_work_only_for_visible_pixels(self, monkeypatch):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada import source

        monkeypatch.setattr(source, "_MSC_CANADA_DECODE_CHUNK_PIXELS", 7)
        rgba = np.empty((3, 7, 4), dtype=np.uint8)
        pixels = rgba.reshape(-1, 4)
        pixels[:, :3] = np.array([row[:3] for row in _MSC_CANADA_PALETTE * 2][:21])
        pixels[:, 3] = 0
        # An empty block, a mixed block, then a fully visible block; nonzero
        # alpha at both sides of the boundary must be decoded and scattered.
        pixels[7:14, 3] = [1, 0, 127, 0, 0, 0, 255]
        pixels[14:, 3] = 255
        expected = _scalar_palette_reference(rgba)
        sizes = []
        original_argmin = np.argmin

        def argmin(distances, *args, **kwargs):
            sizes.append(distances.shape[0])
            return original_argmin(distances, *args, **kwargs)

        monkeypatch.setattr(source.np, "argmin", argmin)
        result = _decode_msc_canada_png(_save_test_png(Image.fromarray(rgba)))
        assert sizes == [3, 7]
        np.testing.assert_array_equal(result, expected)
