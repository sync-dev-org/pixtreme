import cupy as cp

from ..utils.dtypes import to_float32
from .bgr import bgr_to_rgb


def bgr_to_grayscale(image: cp.ndarray) -> cp.ndarray:
    """
    Convert BGR to Grayscale

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in BGR format.

    Returns
    -------
    image_gray : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in RGB format.
    """
    return rgb_to_grayscale(bgr_to_rgb(image))


def rgb_to_grayscale(image: cp.ndarray) -> cp.ndarray:
    """
    Convert RGB to Grayscale

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in RGB format.

    Returns
    -------
    frame_gray : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in RGB format.
    """
    image_rgb: cp.ndarray = to_float32(image)

    height, width, _ = image_rgb.shape
    image_gray: cp.ndarray = cp.empty_like(image_rgb)
    grid_size: tuple = ((width + 31) // 32, (height + 31) // 32)
    block_size: tuple = (32, 32)
    rgb_to_grayscale_kernel(grid_size, block_size, (image_rgb, image_gray, width, height))
    return image_gray


rgb_to_grayscale_kernel_code = """
extern "C" __global__
void rgb_to_grayscale_kernel(const float* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float r = rgb[idx];
    float g = rgb[idx + 1];
    float b = rgb[idx + 2];

    float gray_val = 0.2126f * r + 0.7152f * g + 0.0722f * b;

    gray[idx] = gray_val;
    gray[idx + 1] = gray_val;
    gray[idx + 2] = gray_val;
}
"""
rgb_to_grayscale_kernel = cp.RawKernel(code=rgb_to_grayscale_kernel_code, name="rgb_to_grayscale_kernel")
