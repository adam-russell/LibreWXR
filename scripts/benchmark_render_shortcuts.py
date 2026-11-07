#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare local rendering against a Git revision, without fetching live data.

Run with the project's Python environment and --baseline-ref <commit>, or
--baseline-dir containing the original renderer.py and png_palette.py.
Synthetic precipitation uses real US/RRQPE grid dimensions and NEXRAD coverage.
Optional --png-dir adds saved real-world RGBA tiles to the encoder comparison.
These single-process timings exclude HTTP, queueing, disk caches, and NWP.
This is a manual benchmark, not a pytest test or a timing-based CI gate.

Usage (from the repository root, after installing .[dev] in .venv):

    .venv/bin/python scripts/benchmark_render_shortcuts.py \
        --baseline-ref 9bf2c5dc675f29fdf2966d4a59bb291c68356aa6 \
        --repeats 15 --output /tmp/librewxr-render-benchmark.json

That revision predates these optimizations; select the appropriate pre-change
revision when comparing subsequent work. --baseline-dir can instead point to
saved original renderer.py and png_palette.py files from the same codebase.
Only those two modules are replaced; all other code and dependencies come from
the current environment. Baseline modules must be compatible with that code.

Both variants disable the shared disk coordinate store and use 32 entries per
coordinate cache, overriding those two environment settings. Cold-coordinate
measurements clear the in-process coordinate caches before each variant,
outside the timed interval. Warm-coordinate measurements retain those caches;
two unmeasured rounds warm both variants. Full-tile timings clear coordinate
caches before each variant. These direct calls bypass the HTTP tile caches.
Other settings may come from the local environment/.env; use matching settings
when comparing separate runs. The JSON records the runtime and cache settings.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

os.environ["LIBREWXR_COORD_STORE_ENABLED"] = "false"
os.environ["LIBREWXR_COORD_CACHE_SIZE"] = "32"

import numpy as np
from PIL import Image, __version__ as pillow_version

from librewxr.config import settings
from librewxr.data.coverage import build_coverage_masks, sample_coverage
from librewxr.data.regions import REGIONS
from librewxr.sources.regional.north_america.usa.radar.stations import NEXRAD_CONUS
from librewxr.tiles import coordinates, png_palette, renderer

ROOT = Path(__file__).resolve().parents[1]


def load_revision(commit, filename, name, directory=None):
    if directory is not None:
        source = (directory / filename).read_text()
    else:
        source = subprocess.check_output(
            ["git", "show", f"{commit}:src/librewxr/tiles/{filename}"], cwd=ROOT, text=True,
        )
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"{commit}:{filename}", "exec"), module.__dict__)
    return module


def clear_coordinates():
    for fn in coordinates.ALL_CACHES:
        fn.cache_clear()


def measure(variants, repeats, reset=lambda: None):
    times = {name: [] for name in variants}
    # Alternate ordering to reduce systematic warmup/order bias.
    items = list(variants.items())
    for i in range(repeats + 2):
        for name, fn in (items if i % 2 == 0 else items[::-1]):
            reset()
            start = time.perf_counter_ns()
            fn()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if i >= 2:
                times[name].append(elapsed)
    return {name: round(statistics.median(values), 3) for name, values in times.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-ref")
    baseline.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--png-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    commit = "saved-source-files"
    if args.baseline_ref:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", args.baseline_ref + "^{commit}"],
            cwd=ROOT, text=True,
        ).strip()
    old_png = load_revision(commit, "png_palette.py", "baseline_png", args.baseline_dir)
    old_renderer = load_revision(commit, "renderer.py", "baseline_renderer", args.baseline_dir)
    old_renderer.encode_png = old_png.encode_png
    sources_only = load_revision(commit, "renderer.py", "sources_only_renderer", args.baseline_dir)
    sources_only._composite_regions = renderer._composite_regions
    sources_only.encode_png = old_png.encode_png
    png_only = load_revision(commit, "renderer.py", "png_only_renderer", args.baseline_dir)
    png_only.encode_png = png_palette.encode_png

    report = {
        "baseline": commit,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__, "pillow": pillow_version,
        "repeats": args.repeats,
        "coordinate_cache": {
            "shared_store_enabled": settings.coord_store_enabled,
            "entries_per_cache": settings.coord_cache_size,
        },
        "units": "median milliseconds",
        "limitations": "Synthetic precipitation; local CPU; no HTTP, executor queue, disk cache, or NWP.",
        "encoding": [], "geometry": [], "render": [],
    }
    images = {}
    for count in (1, 2, 40, 256, 257):
        index = (np.arange(512 * 512) % count).reshape(512, 512)
        arr = np.zeros((512, 512, 4), dtype=np.uint8)
        arr[..., 0] = index % 256
        arr[..., 1] = index // 256
        arr[..., 3] = index % 256
        images[f"synthetic_{count}_colors"] = Image.fromarray(arr)
    if args.png_dir:
        for path in sorted(args.png_dir.glob("*.png")):
            with Image.open(path) as img:
                images[path.stem] = img.convert("RGBA")
    for name, img in images.items():
        old = old_png.encode_png(img)
        new = png_palette.encode_png(img)
        assert old == new, f"Encoded bytes changed: {name}"
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(new)).convert("RGBA")), np.asarray(img))
        report["encoding"].append({"case": name, "bytes": len(new), **measure({
            "baseline": lambda: old_png.encode_png(img),
            "optimized": lambda: png_palette.encode_png(img),
        }, args.repeats)})

    build_coverage_masks({"USCOMP": NEXRAD_CONUS})
    enabled = ["USCOMP", "RRQPE"]
    frames = {}
    for name in enabled:
        region = REGIONS[name]
        # Broad bands with dry gaps exercise smoothing, no-echo precedence,
        # and nontrivial PNG colors without allocating giant float grids.
        row = np.arange(region.height, dtype=np.uint32)[:, None]
        col = np.arange(region.width, dtype=np.uint32)[None, :]
        frames[name] = (((row // 7 + col // 11) % 7) * 25).astype(np.uint8)
    for name, lat, lon, z in [("inland", 38.5, -97, 8), ("coastal", 35, -75, 7), ("ocean", 35, -65, 7)]:
        x = int((lon + 180) / 360 * 2**z)
        y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**z)
        kwargs = dict(z=z, x=x, y=y, tile_size=512, smooth=True, snow=False, enabled_regions=enabled)
        old_geom = old_renderer.compute_tile_geometry(frames, **kwargs)
        new_geom = renderer.compute_tile_geometry(frames, **kwargs)
        assert (old_geom.pad, old_geom.blur_radius, old_geom.is_transparent) == (new_geom.pad, new_geom.blur_radius, new_geom.is_transparent)
        np.testing.assert_array_equal(old_geom.values, new_geom.values)
        pad = new_geom.pad
        latlon = coordinates.tile_pixel_latlons_padded(z, x, y, 512, pad) if pad else coordinates.tile_pixel_latlons(z, x, y, 512)
        fraction = float(sample_coverage("USCOMP", *latlon).mean())
        for cold in (False, True):
            # Clear before each variant, outside timing, for cold coordinates.
            reset = clear_coordinates if cold else lambda: None
            report["geometry"].append({"case": name, "cold_coordinates": cold, "fine_coverage": round(fraction, 4), **measure({
                "baseline": lambda: old_renderer.compute_tile_geometry(frames, **kwargs),
                "optimized": lambda: renderer.compute_tile_geometry(frames, **kwargs),
            }, args.repeats, reset)})
        variants = {
            "baseline": old_renderer, "sources_only": sources_only,
            "png_only": png_only, "both": renderer,
        }
        def render(module):
            return module.render_tile(frames, color_scheme=6, **kwargs)
        expected = render(old_renderer)
        for variant in variants.values():
            assert render(variant) == expected, f"Rendered bytes changed: {name}"
        report["render"].append({"case": name, **measure({
            label: lambda module=module: render(module) for label, module in variants.items()
        }, args.repeats, clear_coordinates)})
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
