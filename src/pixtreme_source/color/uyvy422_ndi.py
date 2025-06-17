import cupy as cp

from ..transform.resize import resize


def ndi_uyvy422_to_ycbcr444(uyvy_data: cp.ndarray, use_bilinear: bool = True) -> cp.ndarray:
    """
    Convert NDI UYVY422 to YCbCr444 using CUDA kernel.

    Parameters
    ----------
    uyvy_data : cp.ndarray
        The input UYVY422 data. The 3d array of the shape (height, width, 2).
    use_bilinear : bool
        Whether to use bilinear interpolation for UV components.

    Returns
    -------
    yuv444p : cp.ndarray
        The output YCbCr444 data. The 3d array of the shape (height, width, 3).
    """
    height, width, channels = uyvy_data.shape
    if channels != 2:
        raise ValueError("Input must have 2 channels (UV, Y)")

    total_pixels = height * width

    # Flatten input for kernel processing
    uyvy_flat = uyvy_data.flatten()

    # Output buffer
    yuv444_flat = cp.empty(total_pixels * 3, dtype=cp.uint8)

    # Kernel launch parameters
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    # Choose kernel based on interpolation method
    if use_bilinear:
        ndi_uyvy422_to_ycbcr444_bilinear_kernel(
            (blocks_per_grid,), (threads_per_block,), (uyvy_flat, yuv444_flat, height, width)
        )
    else:
        ndi_uyvy422_to_ycbcr444_kernel((blocks_per_grid,), (threads_per_block,), (uyvy_flat, yuv444_flat, height, width))

    # Reshape to 3D array
    yuv444p = yuv444_flat.reshape(height, width, 3)

    return yuv444p


ndi_uyvy422_to_ycbcr444_kernel_code = """
extern "C" __global__
void ndi_uyvy422_to_ycbcr444_kernel(
    const unsigned char* uyvy_data,
    unsigned char* yuv444_data,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = height * width;

    if (idx >= total_pixels) return;

    int row = idx / width;
    int col = idx % width;

    // NDI UYVY422 format: (height, width, 2)
    // channel 0: UV components [U0, V0, U1, V1, ...]
    // channel 1: Y components

    // Y component is directly available
    int y_idx = row * width * 2 + col * 2 + 1;  // channel 1
    unsigned char y_val = uyvy_data[y_idx];

    // UV components need interpolation
    int uv_idx = row * width * 2 + col * 2;  // channel 0

    // For UV interpolation, find the nearest UV pair
    int uv_col = col / 2;  // UV sample position
    int uv_base_idx = row * width * 2 + uv_col * 4;  // Base index for UV pair

    unsigned char u_val, v_val;

    if (col % 2 == 0) {
        // Even column: use current UV values
        u_val = uyvy_data[uv_base_idx];      // U0
        v_val = uyvy_data[uv_base_idx + 2];  // V0
    } else {
        // Odd column: interpolate between current and next UV
        if (uv_col * 2 + 1 < width / 2) {
            // Linear interpolation between UV pairs
            unsigned char u0 = uyvy_data[uv_base_idx];
            unsigned char v0 = uyvy_data[uv_base_idx + 2];
            unsigned char u1 = uyvy_data[uv_base_idx + 4];
            unsigned char v1 = uyvy_data[uv_base_idx + 6];

            u_val = (u0 + u1) / 2;
            v_val = (v0 + v1) / 2;
        } else {
            // At the edge, use the current UV values
            u_val = uyvy_data[uv_base_idx];
            v_val = uyvy_data[uv_base_idx + 2];
        }
    }

    // Output YUV444
    int output_idx = idx * 3;
    yuv444_data[output_idx] = y_val;      // Y
    yuv444_data[output_idx + 1] = u_val;  // U
    yuv444_data[output_idx + 2] = v_val;  // V
}
"""
ndi_uyvy422_to_ycbcr444_kernel = cp.RawKernel(code=ndi_uyvy422_to_ycbcr444_kernel_code, name="ndi_uyvy422_to_ycbcr444_kernel")

ndi_uyvy422_to_ycbcr444_bilinear_kernel_code = """
extern "C" __global__
void ndi_uyvy422_to_ycbcr444_bilinear_kernel(
    const unsigned char* uyvy_data,
    unsigned char* yuv444_data,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = height * width;

    if (idx >= total_pixels) return;

    int row = idx / width;
    int col = idx % width;

    // Y component (channel 1)
    int y_idx = row * width * 2 + col * 2 + 1;
    unsigned char y_val = uyvy_data[y_idx];

    // UV interpolation with bilinear sampling
    float uv_x = (float)col / 2.0f;
    int uv_x0 = (int)uv_x;
    int uv_x1 = min(uv_x0 + 1, width / 2 - 1);
    float weight = uv_x - (float)uv_x0;

    // UV indices
    int uv_idx0 = row * width * 2 + uv_x0 * 4;  // channel 0
    int uv_idx1 = row * width * 2 + uv_x1 * 4;  // channel 0

    // Get UV values
    unsigned char u0 = uyvy_data[uv_idx0];
    unsigned char v0 = uyvy_data[uv_idx0 + 2];
    unsigned char u1 = uyvy_data[uv_idx1];
    unsigned char v1 = uyvy_data[uv_idx1 + 2];

    // Bilinear interpolation
    unsigned char u_val = (unsigned char)((1.0f - weight) * u0 + weight * u1);
    unsigned char v_val = (unsigned char)((1.0f - weight) * v0 + weight * v1);

    // Output YUV444
    int output_idx = idx * 3;
    yuv444_data[output_idx] = y_val;
    yuv444_data[output_idx + 1] = u_val;
    yuv444_data[output_idx + 2] = v_val;
}
"""

ndi_uyvy422_to_ycbcr444_bilinear_kernel = cp.RawKernel(
    code=ndi_uyvy422_to_ycbcr444_bilinear_kernel_code, name="ndi_uyvy422_to_ycbcr444_bilinear_kernel"
)


def ndi_uyvy422_to_ycbcr444_cp(uyvy_data: cp.ndarray) -> cp.ndarray:
    """
    Convert NDI UYVY422 to YCbCr444.

    Parameters
    ----------
    uyvy_data : cp.ndarray
        The input UYVY422 data. The 3d array of the shape (height, width, 2).

    Returns
    -------
    yuv444p : cp.ndarray
        The output YCbCr444 data. The 3d array of the shape (height, width, 3).
    """
    # channel 1: Y
    # channel 0: U and V, [U0, Y0, U1, Y1, ...]
    y_component = uyvy_data[:, :, 1]
    uv_component = uyvy_data[:, :, 0]

    # Divide U and V components
    uv_component_flat = uv_component.flatten()
    u_component_flat = uv_component_flat[0::2]
    v_component_flat = uv_component_flat[1::2]

    u_component = u_component_flat.reshape((y_component.shape[0], y_component.shape[1] // 2))
    v_component = v_component_flat.reshape((y_component.shape[0], y_component.shape[1] // 2))

    u_component = resize(u_component, (y_component.shape[0], y_component.shape[1]), interpolation=2)
    v_component = resize(v_component, (y_component.shape[0], y_component.shape[1]), interpolation=2)

    # Resize and place U and V components
    # Repeat U and V to each pixel
    yuv444p = cp.dstack((y_component, u_component, v_component))

    return yuv444p
