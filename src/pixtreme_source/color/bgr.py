import cupy as cp
import numpy as np


def rgb_to_bgr(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    """
    Convert RGB to BGR

    Parameters
    ----------
    image : np.ndarray | cp.ndarray
        Input image. Shape 3D array (height, width, 3) in RGB format.

    Returns
    -------
    image : np.ndarray | cp.ndarray
        Output image. Shape 3D array (height, width, 3) in BGR format.
    """
    image = image[:, :, [2, 1, 0]]
    if isinstance(image, np.ndarray):
        return np.ascontiguousarray(image)
    else:
        return cp.ascontiguousarray(image)


def bgr_to_rgb(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
    """
    Convert BGR to RGB

    Parameters
    ----------
    image : np.ndarray | cp.ndarray
        Input image. Shape 3D array (height, width, 3) in BGR format.

    Returns
    -------
    image : np.ndarray | cp.ndarray
        Output image. Shape 3D array (height, width, 3) in RGB format.
    """
    image = image[:, :, [2, 1, 0]]
    if isinstance(image, np.ndarray):
        return np.ascontiguousarray(image)
    else:
        return cp.ascontiguousarray(image)
