# Radar sampling plans and native sampler

Smooth radar rendering reuses interpolation geometry across weather frames.
Regular latitude/longitude sources use compact row and column vectors;
LAEA and transverse-Mercator sources retain exact two-dimensional plans.
The key includes the full immutable source definition, tile coordinates,
tile size and smoothing padding. Plans contain no weather values.
Projected plans derive weights and validity masks from a single projection,
with the same rounding and clipping as the separate coordinate maps.

Adaptive smoothing computes its radius from a 3×3 centre sample instead of
allocating a full tile coordinate map. A per-worker scalar LRU is bounded by
`LIBREWXR_COORD_CACHE_SIZE` and keyed by the source, tile, size, and configured
base radius. It is reported as `_cached_blur_radius` in coordinate-cache health
stats and cleared with those caches under memory pressure. Cached payload
accounting includes the scalar values; as with the array caches, it excludes
Python cache-key and container overhead.

Each rendering worker retains at most 32 plans and 32 MiB of plan arrays.
Oversized plans are used for the request without being retained. Plans are
read-only; every sample returns a separate writable output. Memory-pressure
cleanup clears the plans together with the coordinate caches. `/health`
reports plan bytes, limits, entries, hits/misses, and native availability
under `coord_caches.caches.sampling_plans`. Plan bytes remain private heap
even if the shared coordinate store is enabled.

The optional `_sampling_native` CPython extension performs interpolation
without full-tile NumPy intermediates. It uses Python's buffer protocol,
does not require NumPy headers, supports strided/read-only frames, and
releases the GIL while sampling. The NumPy implementation remains available
when the extension is absent. Both preserve float32 evaluation order,
zero-neighbor fallback, out-of-region masking, rounding, and source ownership.
Nearest-neighbor rendering, coverage authority, model sampling, palettes,
image blur and image encoders are unchanged.

## Building

`pip install .` and `pip install -e .` attempt to build the extension on
Linux and macOS using a C compiler and Python development headers. On Debian
hosts these are normally provided by `gcc` and the matching `python3-dev`;
macOS uses the Command Line Tools. If compilation is unavailable, installation
continues with NumPy sampling. No compiler is invoked at runtime.

Build-time switches:

- `LIBREWXR_REQUIRE_NATIVE=1`: fail installation if compilation fails.
- `LIBREWXR_DISABLE_NATIVE=1`: build a Python-only installation.

Docker uses a build stage with the compiler and requires native compilation
to succeed. The final runtime image installs the resulting wheel and
contains no compiler. Native wheels carry interpreter/platform tags; source
distributions contain the C source and build hook. Editable builds place an
ignored native artifact beside the Python sources.

The prototype Linux comparison measured all four combinations (existing
sampler, plans with NumPy, native sampling, and both) against identical saved
weather. Both reduced aggregate full-render CPU by about 39%, warm geometry
CPU by 56%, and repeated-pan CPU by 61%; peak process RSS fell from about
531 to 410 MiB. These are controlled rendering results with completed-tile
caches bypassed, not HTTP throughput estimates. Individual projected plans
can be larger than the old maps; the byte limit bounds retained memory.
