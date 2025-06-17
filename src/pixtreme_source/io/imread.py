import os

import cupy as cp
import cv2
import Imath
import numpy as np
import OpenEXR
from nvidia import nvimgcodec

from ..color.bgr import bgr_to_rgb, rgb_to_bgr


def imread(input_path: str, is_rgb=False, is_nvimgcodec=False) -> cp.ndarray:
    """
    Read an image from a file into a CuPy array.

    Args:
        input_path (str): Path to the image file.
        is_rgb (bool): If True, the image will be read in RGB format. Default is False (BGR).
        is_nvimgcodec (bool): If True, use NVIDIA's nvimgcodec for reading the image. Default is False.
    Returns:
        cp.ndarray: The image as a CuPy array.
    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image not found at {input_path}")

    filename, ext = os.path.splitext(input_path)
    ext = ext.lower()

    if ext in [".exr"]:
        image_exr = OpenEXR.InputFile(input_path)

        # Determine pixel type
        r_header = image_exr.header()["channels"]["R"]
        if "HALF" in str(r_header):
            pixeltype = Imath.PixelType(Imath.PixelType.HALF)
            dtype = "float16"
        else:
            pixeltype = Imath.PixelType(Imath.PixelType.FLOAT)
            dtype = "float32"
        r_str, g_str, b_str = image_exr.channels("RGB", pixeltype)

        # Convert to numpy array
        red: cp.ndarray = cp.frombuffer(r_str, dtype=dtype)
        green: cp.ndarray = cp.frombuffer(g_str, dtype=dtype)
        blue: cp.ndarray = cp.frombuffer(b_str, dtype=dtype)
        dw = image_exr.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        if is_rgb:
            image: cp.ndarray = cp.dstack([red, green, blue])
        else:
            image: cp.ndarray = cp.dstack([blue, green, red])

        image = image.reshape(size[1], size[0], 3)

    else:
        if is_nvimgcodec:
            decoder = nvimgcodec.Decoder()
            nv_image = decoder.read(nvimgcodec.DecodeSource(input_path))  # RGB
            if nv_image is None:
                raise RuntimeError(f"Failed to read image from {input_path}")
            image: cp.ndarray = cp.asarray(nv_image)

            if len(image.shape) == 2:
                # If the image is grayscale, convert it to a 3-channel BGR image
                image = cp.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                # If the image has an alpha channel, remove it
                image = image[:, :, :3]

            if not is_rgb:
                image = rgb_to_bgr(image)

        else:
            cv2_image: np.ndarray = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if cv2_image is None:
                raise RuntimeError(f"Failed to read image from {input_path}")
            image: cp.ndarray = cp.asarray(cv2_image)

            if len(image.shape) == 2:
                # If the image is grayscale, convert it to a 3-channel BGR image
                image = cp.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                # If the image has an alpha channel, remove it
                image = image[:, :, :3]

            if is_rgb:
                image = bgr_to_rgb(image)

    return image
