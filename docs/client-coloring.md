# Experimental client-side radar coloring

The current colored PNG/WebP API remains the default. This prototype is an
additional transport, disabled unless
`LIBREWXR_EXPERIMENTAL_DATA_TILES_ENABLED=true`. No TidesPro renderer or setting
is changed. Once enabled on a development instance, open
`/experimental/radar-data/client` for an isolated comparison.

The browser client uses CPU JavaScript, not a GPU shader. It tests fidelity and
palette reuse; it does not establish native app speed, battery cost, or Metal
performance. Its neutral light/dark backgrounds are not app map screenshots.

## API and wire format (experimental v1)

- `GET /experimental/radar-data`: capabilities and current scheme names/version.
- `GET /experimental/radar-data/palettes/{scheme}.json`: 256-entry rain and snow
  RGBA tables plus their atomically captured palette version; ETag revalidation.
- `GET /experimental/radar-data/{timestamp}/{size}/{z}/{x}/{y}/{smooth_snow}.lwxr`:
  XYZ only, size 256 or 512, flags `0_0`, `0_1`, `1_0`, `1_1`. Timestamp 0 resolves
  to the latest past frame; `X-Frame-Timestamp` identifies the resolved frame.
  Nowcast timestamps use the existing nowcast store when available.

Centered windows, arrows, storm cells, lightning and query options are not
implemented in this experimental transport (unsupported options return 400).
The existing colored routes continue to support their existing features.

The body is `application/vnd.librewxr.data-tile`:

1. Eight bytes `LWXRT01\n`.
2. Four-byte little-endian JSON-header length (at most 4096).
3. UTF-8 JSON header: version, final size, padding, plane side, transparent and
   snow flags, blur sigma, integer blur kernel, uncompressed byte count and
   encoding identifiers.
4. One zlib stream (level 6 by default): row-major uint8 intensity plane, followed by a little-bit-first
   packed snow plane when present. Transparent sentinels contain no planes.

Intensity uses the renderer's existing indices (`dBZ = value / 2 - 32`, zero
means no echo). No resampling, quantization or gamma transformation is added.
Decoders bound allocations before decompression and reject malformed lengths.
Raw intensity and snow values never enter a browser image decoder or canvas.
They should use unnormalized integer or linear numeric textures in any future
GPU implementation, with no sRGB conversion, premultiplication or interpolation
of packed channels.

Client order is **LUT coloring → straight-RGBA blur → crop padding**. Blurring
intensity or applying a typical premultiplied-alpha canvas blur changes the
result. The header carries the fixed-point extended-box kernel corresponding to
Pillow's three horizontal and three vertical Gaussian approximation passes.
The independently structured JavaScript implementation uses prefix sums and
edge replication, rounding each pass to uint8. See
[Pillow's implementation](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/BoxBlur.c).
Tests compare its raw RGBA bytes with the existing server PNG decoder, before
canvas alpha conversion. A different Pillow implementation must pass these
checks before using this protocol unchanged.

Level 6 compression is lossless and retains the same v1 framing and decoder;
existing clients do not need a format migration. Explicit levels remain
available to encoding callers, including the level-1 benchmark comparison.

Numeric tiles share the existing bounded geometry cache and encoded byte budget,
including timestamp invalidation. Their encoded keys contain no palette version:
a palette change reuses the same data bytes. When the existing shared encoded-tile
store is enabled, past-frame numeric responses also use that store and its
existing disk budget. Keys include frame content version, numeric format version,
compression level, dimensions, coordinates and smooth/snow flags. A second worker
can serve the encoded response without sampling or encoding. Encoded memory hits
do not require the geometry to remain resident for versioned past frames. No
additional memory cache or larger cache budget is introduced.

Disk reads, hashing and publication run off the event loop; first-request
publication is awaited. The existing best-effort shared-store behavior falls
back to rendering on read errors and still serves a response on write errors.
Nowcast and legacy frames without content versions use only the existing local
cache and poller invalidation. Model refreshes clear numeric geometry/encoded
entries and sweep shared files through the same existing maintenance as colored
tiles. Frame-version changes during a request trigger one retry; continued
changes return a retryable 503 rather than serving the superseded bytes.

Conditional requests use strong
ETags and the same past/latest cache ages as normal tiles. Palette responses
must be revalidated separately; the comparison page captures palettes for its
session, so reload it after changing the server color table. The experimental
router does not contribute request counters or timing histories to the existing
public radar diagnostics.

## Linux benchmark

Reproducible source: `scripts/benchmarks/client_coloring/`. From the project venv:

```sh
.venv/bin/python scripts/benchmarks/client_coloring/prepare.py /tmp/client-coloring-trial
```

The prepared bundle contains a source snapshot and hashes. It uses the existing
frozen November 6 weather and verified native sampling artifact on radar1;
it never fetches weather or builds an image. Copy the bundle into a new services
subdirectory and run its `run-benchmark.py` with sudo there. It needs Docker
permission. It does not commit, push, deploy, restart or edit running services.

The arms are current PNG scheme 2 (Universal Blue), numeric zlib level 1, and
numeric zlib level 6. All use current geometry/native sampling, smooth+snow,
512px tiles, and the same four wet locations: Halifax, Genoa, Osaka and Taipei.
The five real saved frames are cycled into six animation steps, explicitly
reported as five distinct frames. A separate validation job compares all
palettes and synthetic transparent, edge-padding, no-snow, all-snow and mixed-snow
supplements. The offline `validation/comparison.html` lets a browser independently
check each exported RGBA hash and change palettes without another data tile.

Nine serial fresh-worker jobs use three balanced arm orders. Each runs five
repetitions of presentation/encoding only, full rendering with warm coordinate
and sampling caches, full rendering with cleared caches, and an adjacent-tile
render with cleared caches. Geometry/encoded completed-tile caches are bypassed,
matching cache-miss rendering; encoded cache hits and HTTP overhead are not
measured. Warm measurements assert zero coordinate/plan cache misses. Separate
reference-client timings include numeric inflate/color/blur/crop versus PNG
decode, using Python/Pillow; they are **not native GPU or browser speed estimates**.

Reports include process CPU, wall time, bytes, server-only peak RSS (before
reference-client work), whole-worker peak RSS, cgroup memory/CPU diagnostics,
and a one-geometry multi-palette transfer demonstration. Palette JSON overhead
is reported separately. Server peak RSS includes six retained geometry/output
objects per batch; it is a controlled comparison, not a live worker RAM forecast.
Balanced medians are descriptive measurements, not confidence intervals. The
CPU work displaced to the client is never counted as server savings twice.

Each job has two CPUs, 1.5 GiB RAM, no swap or network, and read-only source/weather
mounts. The controller retains a 1 GiB host memory reserve, checks service health
and identities/settings, saves partial results, and enforces 60-second RAM waits
and a 20-minute overall deadline. It removes only containers bearing its own
label. No memory/concurrency limits of running services are changed. The pinned
dependency image is reused even if the live service image has since changed;
the full working source snapshot is mounted ahead of installed code, and both
native source and binary hashes must match.
