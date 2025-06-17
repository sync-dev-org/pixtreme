import cupy as cp
import cv2
import numpy as np
import torch
from nvidia import nvimgcodec

from ..color.bgr import rgb_to_bgr
from ..transform.resize import resize
from ..utils.dlpack import to_cupy, to_numpy
from ..utils.dtypes import to_float32, to_uint8


def imshow(
    title: str, image: np.ndarray | cp.ndarray | nvimgcodec.Image, scale: float = 1.0, is_rgb: bool = False
) -> None:
    """
    Image show function for numpy and cupy arrays. in RGB format.

    Parameters
    ----------
    title : str
        Window title
    image : np.ndarray | cp.ndarray | nvimgcodec.Image
        Image to show
    scale : float, optional
        Scale factor, by default 1.0
    is_rgb : bool, optional
        If True, the image will be shown in RGB format, by default False

    Raises
    ------
    KeyboardInterrupt
        If the user presses the ESC key, the window will close and the KeyboardInterrupt will be raised.
    """
    try:
        sufix = ""
        if isinstance(image, cp.ndarray):
            sufix = " (Cupy"
        elif isinstance(image, np.ndarray):
            sufix = " (Numpy"
            image = to_cupy(image)
        elif isinstance(image, torch.Tensor):
            sufix = " (Torch"
            image = to_cupy(image)
            image = rgb_to_bgr(image)
        elif isinstance(image, nvimgcodec.Image):
            sufix = " (nvimgcodec"
            image = to_cupy(image)
            if is_rgb:
                image = rgb_to_bgr(image)
        else:
            raise ValueError(f"Invalid image type: {type(image)}")

        if scale != 1.0:
            image = to_float32(image)
            image = resize(image, None, fx=scale, fy=scale, interpolation=2)

        if is_rgb:
            sufix += " / RGB"
            image = rgb_to_bgr(image)
        else:
            sufix += " / BGR"

        image = to_numpy(image)

        if image.dtype == np.uint8:
            sufix += " / UINT8"
        elif image.dtype == np.uint16:
            sufix += " / UINT16"
        elif image.dtype == np.float16:
            sufix += " / FLOAT16"
        elif image.dtype == np.float32:
            sufix += " / FLOAT32"
        elif image.dtype == np.float64:
            sufix += " / FLOAT64"

        sufix += ")"
        image = to_uint8(image)

        title = f"{title} - {sufix}"
        cv2.imshow(title, image)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        raise
    except Exception as e:
        cv2.destroyAllWindows()
        raise e


def waitkey(delay: int) -> int:
    """
    Wait for a pressed key.

    Parameters
    ----------
    delay : int, optional
        Delay in milliseconds, by default 0

    Returns
    -------
    int
        Key code
    """
    return cv2.waitKey(delay)


def destroy_all_windows() -> None:
    """
    Destroy all windows.
    """
    return cv2.destroyAllWindows()
