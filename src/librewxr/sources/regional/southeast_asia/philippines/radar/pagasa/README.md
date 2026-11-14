# PAGASA PANaHON radar composite (Philippines)

National radar mosaic from **PAGASA / DOST**, served anonymously by the
PANaHON web app at `https://panahon.gov.ph`. The former
`cdn.panahon.gov.ph` host and palette-image API are retired.

## Coverage and grid

`PHCOMP` covers the Philippines (Luzon, Visayas, Mindanao and western
Palawan), plus edges of Sabah and northern Sulawesi. The existing
2048×2048 geographic grid and station inventory are unchanged.

The upstream numeric images use **Web Mercator**, with their longest side
1024 pixels (matching the public viewer) and an aspect ratio derived from the timeline bounds. The
adapter reprojects them by nearest neighbour onto PHCOMP's geographic
grid, retaining the alpha coverage mask and avoiding blended reflectivity
or fabricated echoes outside the image extent.

## Timeline and endpoints

The public [rain-rate viewer](https://panahon.gov.ph/?trg=iframe&req=radar.rain-rate),
linked from [PAGASA's radar page](https://bagong.pagasa.dost.gov.ph/radar),
starts an anonymous session. Its HTML supplies `csrf-token`, `embed-grant`
and `api-sig` metadata, retained in memory alongside its cookies. No account,
operator credentials or configured API key are required.

- `GET /api/v1/radar/timeline?sublayer=mosaic-rainrate` returns native UTC
  `observed_at_unix` timestamps, bounds, scale and tile version. The current
  public buffer carries six frames at **10-minute cadence**; ingestion follows
  returned timestamps rather than assuming a fixed frame count.
- `GET /api/v1/radar-data-image?sublayer=mosaic-rainrate&t=TIMESTAMP&mode=rain&size=1024&v=VERSION`
  returns the corresponding numeric PNG (currently 750×1024, luminance + alpha).
  Legacy `image_url` fields may be empty and are not used.

Each API request sends the CSRF token, cookies and `X-Embed-Grant`, with
`X-Ts`, a fresh `X-Nonce` and `X-Sig`. The signature is HMAC-SHA256 using the
public session's `api-sig` as a UTF-8 key over four newline-separated fields:
`GET`, the request path stripped of leading/trailing slashes (no query),
the Unix timestamp and the nonce. Even transport retries get fresh signatures.
Signed API requests do not follow redirects.

The client identifies itself as LibreWXR and uses the public viewer's Referer.
Each coalesced refresh obtains fresh session context. One additional renewal
is allowed for HTTP 403/419 or an entirely placeholder image window, with no
per-image renewal storm. Session values are never included in adapter logs.
`LIBREWXR_PAGASA_BASE_URL` still supports a custom origin or mirror implementing
this public session flow; missing context fails without unsigned API requests.

## Numeric encoding and missing data

The PNG red/luminance channel stores a normalized rain rate; alpha marks
coverage. Apply `scale.sqrt` (square the normalized value when true), then
multiply by `scale.max` to obtain mm/hr. Convert to equivalent reflectivity
using the supplied `scale.zr` coefficients: `Z = a R^b`, or
`dBZ = 10 log10(a) + 10 b log10(R)`. Zero rain and zero alpha become no echo.
The current advertised coefficients are `a=409.78`, `b=1.21`, with a
square-root scale capped at 80 mm/hr; these values are read from metadata,
not hardcoded in the decoder. This is rain-rate-derived equivalent dBZ,
not the separate reflectivity product.

LibreWXR's standard uint8 encoding is then applied. Display palettes have
no effect on decoded values. Unexpected units, scales, coefficients,
bounds, dimensions and images without an alpha mask are rejected.

A transparent **1×1 placeholder** is unavailable imagery, not a valid dry
radar frame. These responses are summarized in a warning and never cached
as real data. A correctly sized, fully transparent image remains valid.
Failed refreshes preserve previously decoded native timestamps; they are
never relabelled as current observations. Lookups must be within five
minutes of a real cached timestamp.

Concurrent lookups share one refresh per 120 seconds, with at most two
image downloads/decodes in flight and 24 decoded frames retained. Existing
timestamps are refreshed too, so upstream corrections can replace them.

## License & attribution

Public domain per Philippine IP code **RA 8293 §176** (government-works
exception).  Caveat: §176.1 requires PAGASA approval for for-profit
exploitation of the work — fine for a self-hosted AGPL instance;
worth flagging if you ever operate a paid hosting service.  Attribution
to **PAGASA / DOST**.

## Stations

Nine stations (Baguio, Baler, Echague, Daet, Guiuan, Iloilo, Davao,
Kabacan, Panabo).  Coordinates and range settings live in `stations.py`
and feed `data/coverage.py` via the `RadarSourceContribution`.