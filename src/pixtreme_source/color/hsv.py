import cupy as cp

from ..utils.dtypes import to_float32
from .bgr import bgr_to_rgb, rgb_to_bgr


def hsv_to_bgr(image: cp.ndarray) -> cp.ndarray:
    """
    Convert HSV to BGR

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in HSV format.

    Returns
    -------
    image_bgr : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in BGR format.
    """
    return rgb_to_bgr(hsv_to_rgb(image))


def hsv_to_rgb(image: cp.ndarray) -> cp.ndarray:
    """
    Convert HSV to RGB

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in HSV format.

    Returns
    -------
    image_rgb : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in RGB format.
    """
    image_hsv: cp.ndarray = to_float32(image)
    assert 0.0 <= image_hsv[..., 0].max() <= 1.0
    height, width, _ = image_hsv.shape
    image_rgb: cp.ndarray = cp.empty_like(image_hsv)
    block_size: tuple = (32, 32)
    grid_size: tuple = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])
    hsv_to_rgb_kernel(grid_size, block_size, (image_hsv, image_rgb, height, width))
    return image_rgb


hsv_to_rgb_kernel_code = """
extern "C" __global__
void hsv_to_rgb_kernel_optimized(const float* hsv, float* rgb, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float h = hsv[idx] * 360.0f;
    float s = hsv[idx + 1];
    float v = hsv[idx + 2];

    float c = v * s;
    //float h_prime = fmodf(h / 60.0, 6);
    h = fmodf(h, 360.0f);
    if (h < 0) h += 360.0f;
    float h_prime = h / 60.0f;

    float x_tmp = c * (1 - fabsf(fmodf(h_prime, 2) - 1));
    float m = v - c;

    float r, g, b;

    if (0 <= h_prime && h_prime < 1) {
        r = c; g = x_tmp; b = 0;
    } else if (1 <= h_prime && h_prime < 2) {
        r = x_tmp; g = c; b = 0;
    } else if (2 <= h_prime && h_prime < 3) {
        r = 0; g = c; b = x_tmp;
    } else if (3 <= h_prime && h_prime < 4) {
        r = 0; g = x_tmp; b = c;
    } else if (4 <= h_prime && h_prime < 5) {
        r = x_tmp; g = 0; b = c;
    } else if (5 <= h_prime && h_prime < 6) {
        r = c; g = 0; b = x_tmp;
    } else {
        r = 0; g = 0; b = 0;
    }

    r += m;
    g += m;
    b += m;

    rgb[idx] = r;
    rgb[idx + 1] = g;
    rgb[idx + 2] = b;
}
"""

hsv_to_rgb_kernel = cp.RawKernel(code=hsv_to_rgb_kernel_code, name="hsv_to_rgb_kernel_optimized")


def bgr_to_hsv(image: cp.ndarray) -> cp.ndarray:
    """
    Convert BGR to HSV

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in BGR format.

    Returns
    -------
    image_hsv : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in
    """
    return rgb_to_hsv(bgr_to_rgb(image))


def rgb_to_hsv(image: cp.ndarray) -> cp.ndarray:
    """
    Convert RGB to HSV

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in RGB format.

    Returns
    -------
    image_hsv : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in HSV format.
    """
    image_rgb: cp.ndarray = to_float32(image)
    height, width, _ = image_rgb.shape
    image_hsv: cp.ndarray = cp.empty_like(image_rgb)
    block_size: tuple = (32, 32)
    grid_size: tuple = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])
    rgb_to_hsv_kernel(grid_size, block_size, (image_rgb, image_hsv, height, width))
    return image_hsv


rgb_to_hsv_kernel_code = """
extern "C" __global__
void rgb_to_hsv_kernel(const float* rgb, float* hsv, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float r = rgb[idx];
    float g = rgb[idx + 1];
    float b = rgb[idx + 2];

    float maxc = fmaxf(r, fmaxf(g, b));
    float minc = fminf(r, fminf(g, b));
    float delta = maxc - minc;

    // Value
    float v = maxc;

    // Saturation
    float s = maxc == 0 ? 0 : delta / maxc;

    // Hue
    float h = 0;
    if (delta > 0) {
        if (r == maxc) {
            h = (g - b) / delta;
        } else if (g == maxc) {
            h = 2.0f + (b - r) / delta;
        } else {
            h = 4.0f + (r - g) / delta;
        }
        h *= 60.0f;
        if (h < 0) h += 360.0f;
    }

    h /= 360.0f;
    hsv[idx] = h;
    hsv[idx + 1] = s;
    hsv[idx + 2] = v;
}
"""
rgb_to_hsv_kernel = cp.RawKernel(code=rgb_to_hsv_kernel_code, name="rgb_to_hsv_kernel")
