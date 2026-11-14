# PAGASA PANaHON radar composite (Philippines)

National radar mosaic from **PAGASA / DOST**, served anonymously by the
PANaHON web app at `https://panahon.gov.ph`. The former
`cdn.panahon.gov.ph` host and palette-image API are retired.

## Coverage and grid

`PHCOMP` covers the Philippines (Luzon, Visayas, Mindanao and western
Palawan), plus edges of Sabah and northern Sulawesi. The existing
2048×2048 geographic grid and station inventory are unchanged.

The upstream numeric images use **Web Mercator**, with their longest side
2048 pixels and an aspect ratio derived from the timeline bounds. The
adapter reprojects them by nearest neighbour onto PHCOMP's geographic
grid, retaining the alpha coverage mask and avoiding blended reflectivity
or fabricated echoes outside the image extent.

## Timeline and endpoints

- `GET /api/v1/radar/timeline?sublayer=mosaic-reflectivity` returns native
  UTC `observed_at_unix` timestamps, bounds, scale and tile version. The
  current buffer carries nine frames at **10-minute cadence**; ingestion
  follows the returned timestamps rather than assuming six frame indices.
- `GET /api/v1/radar-data-image?sublayer=mosaic-reflectivity&t=TIMESTAMP&mode=dbz&size=2048&v=VERSION`
  returns the corresponding numeric PNG. The timeline's legacy `image_url`
  fields may be empty and are not used.

The client identifies itself as LibreWXR, supplies the origin's normal
Referer and Accept headers, and starts an anonymous public web session.
When the homepage provides a CSRF token, the client forwards it with the
session cookies, refreshing once on an expired-session response. No
account or operator credentials are required. `LIBREWXR_PAGASA_BASE_URL` continues to support a
custom origin or compatible mirror.

## Numeric encoding and missing data

The PNG red channel stores a normalized value; alpha marks coverage. The
adapter applies the timeline's dBZ `scale.max` and `scale.sqrt` metadata,
then LibreWXR's standard uint8 encoding. Display palettes have no effect
on the decoded values. Unexpected scales, bounds, dimensions and images
without an alpha mask are rejected.

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