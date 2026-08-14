"""Color and lookup-table transforms."""

from pixtreme._color.histogram import clahe, equalize_histogram
from pixtreme._color.hsv import hsv_to_rgb, rgb_to_hsv
from pixtreme._color.lut import apply_lut
from pixtreme._color.semantics import (
    gamma_to_linear,
    linear_to_gamma,
    rgb_to_grayscale,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    ycbcr_to_ycbcr,
)
from pixtreme._color.transform import rgb_to_rgb

__all__ = (
    "apply_lut",
    "gamma_to_linear",
    "hsv_to_rgb",
    "linear_to_gamma",
    "rgb_to_grayscale",
    "rgb_to_hsv",
    "rgb_to_rgb",
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
    "ycbcr_to_ycbcr",
    "equalize_histogram",
    "clahe",
)
