# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Stable geographic grid for the PAGASA PANaHON Philippines mosaic.

Current upstream numeric PNGs use Web Mercator and variable dimensions.
The source reprojects them to this existing 2048×2048 geographic grid;
its unequal longitude/latitude pixel sizes and coverage stay unchanged.
"""
from __future__ import annotations

from librewxr.data.regions import RegionDef


PHCOMP = RegionDef(
    name="PHCOMP",
    west=115.4154914129329, east=129.51727937415484,
    south=3.8016540706290445, north=22.458510294136033,
    pixel_size=(129.51727937415484 - 115.4154914129329) / 2048,    # 0.006886°/px (lon)
    pixel_size_y=(22.458510294136033 - 3.8016540706290445) / 2048, # 0.009110°/px (lat)
    group="SOUTHEAST_ASIA",
    grid_width=2048, grid_height=2048,
)

REGIONS: list[RegionDef] = [PHCOMP]