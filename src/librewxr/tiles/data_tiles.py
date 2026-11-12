# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experimental, palette-independent tiles. Numeric bytes never enter an image codec."""
from dataclasses import dataclass
import json
import math
import struct
import zlib

import numpy as np
from PIL import Image, ImageFilter

from librewxr.tiles.renderer import TileGeometry

MAGIC = b"LWXRT01\n"
MEDIA_TYPE = "application/vnd.librewxr.data-tile"
MAX_HEADER = 4096
MAX_SIDE = 2048
COMPRESSION_LEVEL = 6


@dataclass(frozen=True)
class DataTile:
    metadata: dict
    values: np.ndarray
    snow: np.ndarray | None


def blur_recipe(sigma: float) -> dict | None:
    """Fixed-point extended-box recipe for the current Pillow RGBA blur.

    Pillow uses three horizontal, then three vertical passes, rounding to
    uint8 after each pass. Keeping integer weights on the wire avoids client
    floating-point/implementation differences. See docs/client-coloring.md.
    """
    if sigma < 0.5:
        return None
    f = np.float32
    variance = f(f(f(sigma) * f(sigma)) / f(3))
    length = f(math.sqrt(12.0 * float(variance) + 1.0))
    radius = f(math.floor((float(length) - 1.0) / 2.0))
    numerator = f(f(f(2) * radius + f(1)) * f(f(radius * f(radius + f(1))) - f(f(3) * variance)))
    denominator = f(f(6) * f(variance - f(f(radius + f(1)) * f(radius + f(1)))))
    box = f(radius + f(numerator / denominator))
    whole = int(box)
    weight = int(f(f(1 << 24) / f(f(box * f(2)) + f(1))))
    far = ((1 << 24) - (whole * 2 + 1) * weight) // 2
    return dict(radius=whole, center=weight, edge=far, passes=3, scale=1 << 24)


def encode_data_tile(geometry: TileGeometry, *, level: int = COMPRESSION_LEVEL) -> bytes:
    """Preserve the sampled values, snow mask, padding and blur, without colorizing."""
    size, pad = geometry.tile_size, geometry.pad
    side = size + 2 * pad
    if not (1 <= size <= 512 and 0 <= pad and side <= MAX_SIDE):
        raise ValueError("Invalid data-tile dimensions")
    if not math.isfinite(geometry.blur_radius) or not 0 <= geometry.blur_radius <= 128:
        raise ValueError("Invalid data-tile blur radius")
    if geometry.blur_radius < 0.5 and pad:
        raise ValueError("Padding without blur")
    empty = bool(geometry.is_transparent)
    snow = geometry.snow_mask is not None and not empty
    if empty:
        raw = b""
    else:
        if geometry.values.shape != (side, side) or geometry.values.dtype != np.uint8:
            raise ValueError("Invalid intensity plane")
        raw = geometry.values.tobytes(order="C")
        if snow:
            if geometry.snow_mask.shape != (side, side) or geometry.snow_mask.dtype != np.bool_:
                raise ValueError("Invalid snow plane")
            raw += np.packbits(geometry.snow_mask.ravel(), bitorder="little").tobytes()
    metadata = dict(version=1, size=size, pad=pad, side=side, empty=empty,
                    snow=snow, blur=geometry.blur_radius, kernel=blur_recipe(geometry.blur_radius),
                    decoded_bytes=len(raw), intensity="uint8: dbz=value/2-32; zero=no-echo",
                    snow_encoding="little-bit-first")
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    if len(header) > MAX_HEADER:
        raise ValueError("Header too large")
    return MAGIC + struct.pack("<I", len(header)) + header + zlib.compress(raw, level)


def decode_data_tile(data: bytes) -> DataTile:
    """Bound all allocations and reject truncation, extra streams and malformed planes."""
    if len(data) < 12 or data[:8] != MAGIC:
        raise ValueError("Unknown data-tile format")
    length = struct.unpack_from("<I", data, 8)[0]
    if not 1 <= length <= MAX_HEADER or len(data) < 12 + length:
        raise ValueError("Invalid header size")
    try:
        m = json.loads(data[12:12 + length])
        if not isinstance(m, dict) or type(m.get("version")) is not int or m["version"] != 1:
            raise ValueError("Unknown version")
        size, pad, side = m["size"], m["pad"], m["side"]
        if any(type(v) is not int for v in (size, pad, side, m["decoded_bytes"])):
            raise ValueError("Invalid dimensions")
        if not (1 <= size <= 512 and pad >= 0 and side == size + 2 * pad and side <= MAX_SIDE):
            raise ValueError("Invalid dimensions")
        if type(m["empty"]) is not bool or type(m["snow"]) is not bool:
            raise ValueError("Invalid flags")
        if type(m["blur"]) not in (int, float) or not math.isfinite(m["blur"]) or not 0 <= m["blur"] <= 128:
            raise ValueError("Invalid blur")
        if m["blur"] < 0.5 and pad:
            raise ValueError("Padding without blur")
        if m["kernel"] != blur_recipe(m["blur"]):
            raise ValueError("Unsupported blur recipe")
        if m["intensity"] != "uint8: dbz=value/2-32; zero=no-echo" or m["snow_encoding"] != "little-bit-first":
            raise ValueError("Unsupported plane encoding")
        if m["empty"] and m["snow"]:
            raise ValueError("Empty tile has a snow plane")
        count = side * side
        expected = 0 if m["empty"] else count + ((count + 7) // 8 if m["snow"] else 0)
        if m["decoded_bytes"] != expected:
            raise ValueError("Invalid decoded size")
        decoder = zlib.decompressobj()
        raw = decoder.decompress(data[12 + length:], expected + 1)
        if len(raw) != expected or decoder.unconsumed_tail or decoder.unused_data or not decoder.eof:
            raise ValueError("Invalid compressed payload")
    except (KeyError, TypeError, OverflowError, zlib.error, json.JSONDecodeError) as exc:
        raise ValueError("Malformed data tile") from exc
    if m["empty"]:
        values = np.zeros((side, side), dtype=np.uint8)
        snow = None
    else:
        values = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(side, side)
        snow = np.unpackbits(np.frombuffer(raw[count:], dtype=np.uint8), bitorder="little", count=count).reshape(side, side).astype(bool) if m["snow"] else None
    values.flags.writeable = False
    if snow is not None:
        snow.flags.writeable = False
    return DataTile(m, values, snow)


def client_rgba(tile: DataTile, rain_lut: np.ndarray, snow_lut: np.ndarray) -> np.ndarray:
    """Independent reference client: decode -> color -> RGBA blur -> crop.

    No call to present_tile: this is the oracle for the browser prototype and
    the Linux correctness benchmark, not a claim about native GPU performance.
    """
    m = tile.metadata
    if m["empty"]:
        return np.zeros((m["size"], m["size"], 4), dtype=np.uint8)
    for lut in (rain_lut, snow_lut):
        if lut.shape != (256, 4) or lut.dtype != np.uint8:
            raise ValueError("Invalid palette")
    rgba = rain_lut[tile.values]
    if tile.snow is not None:
        rgba[tile.snow] = snow_lut[tile.values[tile.snow]]
    image = Image.fromarray(rgba, "RGBA")
    if m["blur"] >= 0.5:
        image = image.filter(ImageFilter.GaussianBlur(m["blur"]))
        if m["pad"]:
            p, n = m["pad"], m["size"]
            image = image.crop((p, p, p + n, p + n))
    return np.asarray(image).copy()
