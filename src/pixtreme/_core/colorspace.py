"""Shared named RGB colorspace definitions."""

from __future__ import annotations

from collections.abc import Mapping

_ColorSpaceDefinition = tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]]

# CIE 1931 xy chromaticities from the named standards. S-Gamut3 uses the
# published S-Gamut boundary; S-Gamut3.Cine uses Sony's technical-summary set.
_COLORSPACE_DEFINITIONS: Mapping[str, _ColorSpaceDefinition] = {
    "sRGB": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290)),
    "Rec.709": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290)),
    "Rec.2020": (((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), (0.3127, 0.3290)),
    "ACES2065-1": (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), (0.32168, 0.33767)),
    "ACEScg": (((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), (0.32168, 0.33767)),
    "S-Gamut3": (((0.730, 0.280), (0.140, 0.855), (0.100, -0.050)), (0.3127, 0.3290)),
    "S-Gamut3.Cine": (((0.766, 0.275), (0.225, 0.800), (0.089, -0.087)), (0.3127, 0.3290)),
}
