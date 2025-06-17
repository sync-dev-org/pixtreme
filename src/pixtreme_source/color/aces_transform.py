import os

import cupy as cp
import numpy as np

from ..utils.dtypes import to_float32
from .apply_lut import apply_lut
from .read_lut import read_lut

current_dir = os.path.dirname(os.path.abspath(__file__))

ap0_to_ap1_matrix = cp.array(
    [
        [1.4514393161, -0.2365107469, -0.2149285693],
        [-0.0765537734, 1.1762296998, -0.0996759264],
        [0.0083161484, -0.0060324498, 0.9977163014],
    ]
)
ap1_to_ap0_matrix = cp.array(
    [
        [0.6954522414, 0.1406786965, 0.1638690622],
        [0.0447945634, 0.8596711185, 0.0955343182],
        [-0.0055258826, 0.0040252103, 1.0015006723],
    ]
)


# ACES2065-1 to CIE XYZ transformation matrix
aces_to_xyz_matrix = cp.array(
    [
        [0.9525523959, 0.0000000000, 0.0000936786],
        [0.3439664498, 0.7281660966, -0.0721325464],
        [0.0000000000, 0.0000000000, 1.0088251844],
    ]
)

# CIE XYZ to Rec.709(a.k.a sRGB) transformation matrix
xyz_to_709_matrix = cp.array(
    [
        [3.2096, -1.55743, -0.495805],
        [-0.970989, 1.88517, 0.0394894],
        [0.0597193, -0.210104, 1.14312],
    ]
)


rrt_lut = read_lut(os.path.join(current_dir, "./lut/Log2_48_nits_Shaper.RRT.Rec.709.cube"))
inv_rrt_lut = read_lut(os.path.join(current_dir, "./lut/InvRRT.Rec.709.Log2_48_nits_Shaper.cube"))


def rec709_to_aces2065_1(image: cp.ndarray | np.ndarray, tonemap: bool = True) -> cp.ndarray | np.ndarray:
    image = to_float32(image)
    if tonemap:
        a = 1.186 * 10**-3
        b = 12.144
        c = -1.500 * 10**-8
        image = cp.log((image - c) / a) / b
    image = apply_lut(image, rrt_lut)
    image = cp.clip(image, 0, 1)
    return image


def aces2065_1_to_rec709(image: cp.ndarray | np.ndarray, tonemap: bool = True) -> cp.ndarray | np.ndarray:
    image = to_float32(image)
    image = apply_lut(image, inv_rrt_lut)
    if tonemap:
        a = 1.186 * 10**-3
        b = 12.144
        c = -1.500 * 10**-8
        image = cp.exp(image * b) * a + c
    image = cp.clip(image, 0, 1)
    return image


def aces2065_1_to_acescct(image: cp.ndarray | np.ndarray) -> cp.ndarray | np.ndarray:
    image = to_float32(image)
    ap1_image = cp.dot(image, ap0_to_ap1_matrix.T)
    acescct_image = cp.where(
        ap1_image <= 0.0078125,
        (cp.log2(ap1_image) + 9.72) / 17.52,
        (10.5402377416545 * ap1_image + 0.0729055341958355) / 17.52,
    )

    return acescct_image


def acescct_to_aces2065_1(image: cp.ndarray | np.ndarray) -> cp.ndarray | np.ndarray:
    image = to_float32(image)

    ap1_image = cp.where(
        image <= 0.155251141552511,
        (image * 17.52 - 0.0729055341958355) / 10.5402377416545,
        cp.where(
            image < cp.log2(65504),
            2 ** ((image * 17.52) - 9.72),
            65504,
        ),
    )
    ap0_image = cp.dot(ap1_image, ap1_to_ap0_matrix.T)
    return ap0_image


def aces2065_1_to_acescg(image: cp.ndarray | np.ndarray) -> cp.ndarray | np.ndarray:
    image = to_float32(image)
    ap1_image = cp.dot(image, ap0_to_ap1_matrix.T)
    return ap1_image


def acescg_to_aces2065_1(image: cp.ndarray | np.ndarray) -> cp.ndarray | np.ndarray:
    image = to_float32(image)
    ap0_image = cp.dot(image, ap1_to_ap0_matrix.T)
    return ap0_image
