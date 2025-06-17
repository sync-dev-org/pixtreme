from typing import Union

import cupy as cp
import numpy as np

from ..utils.dlpack import to_cupy, to_numpy
from ..utils.dtypes import to_float32

# Horizontal blur kernel for RGB images
horizontal_blur_kernel_code = r"""
extern "C" __global__
void horizontal_blur_kernel(const float* input, float* output, const float* kernel,
                        int height, int width, int kernel_size) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < height && x < width) {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float kernel_sum = 0.0f;
        int radius = kernel_size / 2;

        for (int k = -radius; k <= radius; k++) {
            int px = min(max(x + k, 0), width - 1);
            int idx = (y * width + px) * 3;
            float kernel_val = kernel[k + radius];
            kernel_sum += kernel_val;
            sum_r += input[idx] * kernel_val;
            sum_g += input[idx + 1] * kernel_val;
            sum_b += input[idx + 2] * kernel_val;
        }

        int out_idx = (y * width + x) * 3;
        if (kernel_sum > 0.0f) {
            sum_r /= kernel_sum;
            sum_g /= kernel_sum;
            sum_b /= kernel_sum;
        }
        output[out_idx] = sum_r;
        output[out_idx + 1] = sum_g;
        output[out_idx + 2] = sum_b;
    }
}
"""
horizontal_blur_kernel = cp.RawKernel(horizontal_blur_kernel_code, "horizontal_blur_kernel")

# Vertical blur kernel for RGB images
vertical_blur_kernel_code = r"""
extern "C" __global__
void vertical_blur_kernel(const float* input, float* output, const float* kernel,
                      int height, int width, int kernel_size) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < height && x < width) {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float kernel_sum = 0.0f;
        int radius = kernel_size / 2;

        for (int k = -radius; k <= radius; k++) {
            int py = min(max(y + k, 0), height - 1);
            int idx = (py * width + x) * 3;
            float kernel_val = kernel[k + radius];
            kernel_sum += kernel_val;
            sum_r += input[idx] * kernel_val;
            sum_g += input[idx + 1] * kernel_val;
            sum_b += input[idx + 2] * kernel_val;
        }

        int out_idx = (y * width + x) * 3;
        if (kernel_sum > 0.0f) {
            sum_r /= kernel_sum;
            sum_g /= kernel_sum;
            sum_b /= kernel_sum;
        }
        output[out_idx] = fmaxf(0.0f, fminf(1.0f, sum_r));
        output[out_idx + 1] = fmaxf(0.0f, fminf(1.0f, sum_g));
        output[out_idx + 2] = fmaxf(0.0f, fminf(1.0f, sum_b));
    }
}
"""

vertical_blur_kernel = cp.RawKernel(vertical_blur_kernel_code, "vertical_blur_kernel")


def create_gaussian_kernel(kernel_size: int, sigma: float) -> cp.ndarray:
    """
    Generate 1D Gaussian kernel

    Parameters:
    kernel_size (int): Kernel size (odd number)
    sigma (float): Standard deviation of Gaussian distribution

    Returns:
    cupy.ndarray: Normalized 1D Gaussian kernel
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    x = cp.arange(kernel_size) - (kernel_size - 1) / 2
    kernel = cp.exp(-(x**2) / (2 * sigma**2))

    return kernel.astype(cp.float32)


def gaussian_blur(image: cp.ndarray, kernel_size: int, sigma: float, kernel=None) -> cp.ndarray:
    """
    Apply Gaussian blur to RGB image using CuPy's RawKernel

    Parameters:
    -----------
    image: cp.ndarray
        Input image (HxWx3, float32 [0-1])
    kernel_size: int
        Kernel size (odd number)
    sigma: float
        Standard deviation of Gaussian distribution

    Returns:
    --------
    cp.ndarray
        Output image (HxWx3, float32 [0-1])
    """
    image = to_float32(image)

    if image.ndim == 2:
        image = image[:, :, cp.newaxis]

    height, width = image.shape[:2]

    if kernel is None:
        kernel = create_gaussian_kernel(kernel_size, sigma)

    # Allocate memory
    temp = cp.empty_like(image)
    output = cp.empty_like(image)

    # Set thread block and grid sizes
    block_size = (16, 16)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

    # Convert to contiguous memory layout
    image_cont = cp.ascontiguousarray(image)

    # Horizontal blur
    horizontal_blur_kernel(
        grid_size, block_size, (image_cont.reshape(-1), temp.reshape(-1), kernel, height, width, kernel_size)
    )

    # Debug information for intermediate results
    # print(f"After horizontal blur - max: {temp.max()}, min: {temp.min()}")

    # Vertical blur
    vertical_blur_kernel(
        grid_size, block_size, (temp.reshape(-1), output.reshape(-1), kernel, height, width, kernel_size)
    )

    return output


class GaussianBlur:
    def __init__(self):
        self.kernel_size: int | None = None
        self.sigma: float | None = None
        self.kernel: cp.ndarray | None = None

    def get(self, image: cp.ndarray, kernel_size: int, sigma: float) -> cp.ndarray:
        if self.kernel_size != kernel_size or self.sigma != sigma:
            self.kernel_size = kernel_size
            self.sigma = sigma
            self.kernel = create_gaussian_kernel(kernel_size, sigma)

        return gaussian_blur(image, kernel_size, sigma, self.kernel)
