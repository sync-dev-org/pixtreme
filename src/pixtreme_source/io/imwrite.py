import os
from typing import Union

import cupy as cp
import cv2
import Imath
import numpy as np
import OpenEXR
import torch

from ..color.bgr import bgr_to_rgb, rgb_to_bgr
from ..utils.dlpack import to_numpy
from ..utils.dtypes import to_float16, to_uint8, to_uint16


def imwrite(output_path: str, image: Union[cp.ndarray, np.ndarray], param: int = -1, is_rgb: bool = False) -> None:
    filename, ext = os.path.splitext(output_path)
    if isinstance(image, cp.ndarray):
        image = to_numpy(image)
    elif isinstance(image, torch.Tensor):
        image = to_numpy(image)

    if ext in [".exr"]:
        if not is_rgb:
            image = bgr_to_rgb(image)

        image_rgb = to_float16(image)
        header = OpenEXR.Header(image_rgb.shape[1], image_rgb.shape[0])
        header["compression"] = Imath.Compression(Imath.Compression.DWAA_COMPRESSION)
        if param == -1:
            param = 1
        header["dwaCompressionLevel"] = float(param)
        header["channels"] = {
            "R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
            "B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
        }
        output = OpenEXR.OutputFile(output_path, header)
        output.writePixels(
            {
                "R": image_rgb[:, :, 0].astype(np.float16).tobytes(),
                "G": image_rgb[:, :, 1].astype(np.float16).tobytes(),
                "B": image_rgb[:, :, 2].astype(np.float16).tobytes(),
            }
        )

    else:
        if is_rgb:
            image = rgb_to_bgr(image)

        image_bgr = image

        if ext in [".jpg", ".jpeg"]:
            if param == -1:
                param = 100
            image_bgr = to_uint8(image_bgr)
            options = [cv2.IMWRITE_JPEG_QUALITY, param]
        elif ext in [".png"]:
            if param == -1:
                param = 3
            if image_bgr.dtype != np.uint8:
                image_bgr = to_uint16(image_bgr)
            options = [cv2.IMWRITE_PNG_COMPRESSION, param]
        elif ext in [".tif", ".tiff"]:
            if param == -1:
                param = 5
            if image_bgr.dtype != np.uint8:
                image_bgr = to_uint16(image_bgr)
            options = [cv2.IMWRITE_TIFF_COMPRESSION, param]
        else:
            options = []

        cv2.imwrite(output_path, image_bgr, options)
