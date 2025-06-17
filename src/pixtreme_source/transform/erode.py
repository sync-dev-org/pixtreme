from typing import Union

import cupy as cp
import numpy as np

from ..utils.dlpack import to_cupy, to_numpy
from ..utils.dtypes import to_float32

erode_kernel_code = r"""
extern "C" __global__ void erode_kernel(
    const float* input,
    float* output,
    const int* kernel,
    const int kernel_size,
    const int width,
    const int height,
    const int kernel_center,
    const float border_value
) {
    // Calculate the pixel coordinates for the current thread
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Out of bounds check
    if (x >= width || y >= height) return;

    // Process each channel (RGB)
    for (int c = 0; c < 3; c++) {
        float min_val = 1.0f;  // Initialize to maximum for float32

        // Find minimum value in kernel area
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                // Coordinates in input image with kernel offset
                const int img_x = x + (kx - kernel_center);
                const int img_y = y + (ky - kernel_center);

                // Check if current kernel position is 1
                if (kernel[ky * kernel_size + kx] == 1) {
                    float pixel_value;

                    // Check if within image bounds
                    if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                        // Get value from input image
                        pixel_value = input[(img_y * width + img_x) * 3 + c];
                    } else {
                        // Use border_value for out of bounds
                        pixel_value = border_value;
                    }

                    min_val = min(min_val, pixel_value);
                }
            }
        }

        // Output result
        output[(y * width + x) * 3 + c] = min_val;
    }
}
"""

erode_kernel = cp.RawKernel(erode_kernel_code, "erode_kernel")


def create_erode_kernel(kernel_size: int) -> cp.ndarray:
    """
    Create kernel for erosion processing

    Parameters:
    -----------
    kernel_size : int
        Kernel size

    Returns:
    --------
    cp.ndarray
        Kernel
    """
    kernel = cp.ones((kernel_size, kernel_size), dtype=cp.int32)
    # kernel[kernel_size // 2, :] = 0
    # kernel[:, kernel_size // 2] = 0

    return kernel


def erode(image: cp.ndarray, kernel_size: int, kernel=None, border_value=0.0):
    """
    Perform GPU-based erosion processing on RGB images

    Parameters:
    -----------
    image : cp.ndarray (float32)
        Input RGB image (HxWx3), value range [0, 1]
    kernel : np.ndarray or cp.ndarray
        Structuring element (kernel). Binary 2D array
    border_value : float
        Pixel value outside boundaries

    Returns:
    --------
    cp.ndarray
        RGB image after erosion processing
    """

    image = to_float32(image)

    if kernel is None:
        kernel = create_erode_kernel(kernel_size)

    height, width = image.shape[:2]
    kernel_size = kernel.shape[0]
    kernel_center = kernel_size // 2

    # Prepare output array
    output_image = cp.empty_like(image)

    # Calculate block size and grid size
    block_size = (16, 16)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

    # Execute kernel
    erode_kernel(
        grid_size,
        block_size,
        (
            image.ravel(),
            output_image.ravel(),
            kernel,
            kernel_size,
            width,
            height,
            kernel_center,
            np.float32(border_value),
        ),
    )

    return output_image
