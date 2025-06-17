import cupy as cp

from ..utils.dtypes import to_float32


def apply_lut(frame_rgb: cp.ndarray, lut: cp.ndarray, interpolation: int = 0) -> cp.ndarray:
    """
    Apply a 3D LUT to an frame_rgb with trilinear interpolation.

    Parameters:
    ----------
    frame_rgb : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in RGB format.
    lut : cp.ndarray
        3D LUT. Shape 3D array (N, N, N, 3) in RGB format.
    interpolation : int
        Interpolation method. 0 for trilinear, 1 for tetrahedral.

    Returns
    -------
    result : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in RGB format.
    """
    frame_rgb = to_float32(frame_rgb)
    height, width, channels = frame_rgb.shape
    N = lut.shape[0]
    result = cp.zeros_like(frame_rgb)
    block_size = (32, 32)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

    if interpolation == 0:
        # Flatten the LUT for trilinear interpolation
        lut_flat = lut.reshape(-1)
        lut_trilinear_kernel(grid_size, block_size, (frame_rgb, lut_flat, result, height, width, N))
    elif interpolation == 1:
        lut_tetrahedral_kernel(grid_size, block_size, (frame_rgb, result, lut, height, width, N, N * N))

    return result


lut_trilinear_kernel_code = """
extern "C" __global__
void lut_trilinear_kernel(const float* frame_rgb, const float* lut, float* result, int height, int width, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    int frame_rgb_index = (idy * width + idx) * 3;
    float r = frame_rgb[frame_rgb_index] * (N - 1);
    float g = frame_rgb[frame_rgb_index + 1] * (N - 1);
    float b = frame_rgb[frame_rgb_index + 2] * (N - 1);

    int r_low = max(0, min(int(r), N - 2));
    int g_low = max(0, min(int(g), N - 2));
    int b_low = max(0, min(int(b), N - 2));
    int r_high = r_low + 1;
    int g_high = g_low + 1;
    int b_high = b_low + 1;

    float r_ratio = r - r_low;
    float g_ratio = g - g_low;
    float b_ratio = b - b_low;

    for (int channel = 0; channel < 3; channel++) {
        float c000 = lut[((r_low * N + g_low) * N + b_low) * 3 + channel];
        float c001 = lut[((r_low * N + g_low) * N + b_high) * 3 + channel];
        float c010 = lut[((r_low * N + g_high) * N + b_low) * 3 + channel];
        float c011 = lut[((r_low * N + g_high) * N + b_high) * 3 + channel];
        float c100 = lut[((r_high * N + g_low) * N + b_low) * 3 + channel];
        float c101 = lut[((r_high * N + g_low) * N + b_high) * 3 + channel];
        float c110 = lut[((r_high * N + g_high) * N + b_low) * 3 + channel];
        float c111 = lut[((r_high * N + g_high) * N + b_high) * 3 + channel];

        float c00 = c000 * (1 - r_ratio) + c100 * r_ratio;
        float c01 = c001 * (1 - r_ratio) + c101 * r_ratio;
        float c10 = c010 * (1 - r_ratio) + c110 * r_ratio;
        float c11 = c011 * (1 - r_ratio) + c111 * r_ratio;

        float c0 = c00 * (1 - g_ratio) + c10 * g_ratio;
        float c1 = c01 * (1 - g_ratio) + c11 * g_ratio;

        float c = c0 * (1 - b_ratio) + c1 * b_ratio;

        result[frame_rgb_index + channel] = c;
    }
}

"""

lut_trilinear_kernel = cp.RawKernel(lut_trilinear_kernel_code, "lut_trilinear_kernel")


lut_tetrahedral_kernel_code = """
__device__ float3 get_lut_value(const float *lut, int x, int y, int z, int lutSize, int lutSizeSquared) {
    int index = (x * lutSizeSquared + y * lutSize + z) * 3;
    return {lut[index], lut[index + 1], lut[index + 2]};
}

extern "C" __global__
void lut_tetrahedral_kernel(const float *frame_rgb, float *output, const float *lut, int height, int width, int lutSize, int lutSizeSquared) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    float r = frame_rgb[idx] * (lutSize - 1);
    float g = frame_rgb[idx + 1] * (lutSize - 1);
    float b = frame_rgb[idx + 2] * (lutSize - 1);

    int x0 = static_cast<int>(r);
    int x1 = min(x0 + 1, lutSize - 1);
    int y0 = static_cast<int>(g);
    int y1 = min(y0 + 1, lutSize - 1);
    int z0 = static_cast<int>(b);
    int z1 = min(z0 + 1, lutSize - 1);

    float dx = r - x0;
    float dy = g - y0;
    float dz = b - z0;

    float3 c000 = get_lut_value(lut, x0, y0, z0, lutSize, lutSizeSquared);
    float3 c111 = get_lut_value(lut, x1, y1, z1, lutSize, lutSizeSquared);
    float3 cA, cB;
    float s0, s1, s2, s3;

    if (dx > dy) {
        if (dy > dz) { // dx > dy > dz
            cA = get_lut_value(lut, x1, y0, z0, lutSize, lutSizeSquared);
            cB = get_lut_value(lut, x1, y1, z0, lutSize, lutSizeSquared);
            s0 = 1.0 - dx;
            s1 = dx - dy;
            s2 = dy - dz;
            s3 = dz;
        } else if (dx > dz) { // dx > dz > dy
            cA = get_lut_value(lut, x1, y0, z0, lutSize, lutSizeSquared);
            cB = get_lut_value(lut, x1, y0, z1, lutSize, lutSizeSquared);
            s0 = 1.0 - dx;
            s1 = dx - dz;
            s2 = dz - dy;
            s3 = dy;
        } else { // dz > dx > dy
            cA = get_lut_value(lut, x0, y0, z1, lutSize, lutSizeSquared);
            cB = get_lut_value(lut, x1, y0, z1, lutSize, lutSizeSquared);
            s0 = 1.0 - dz;
            s1 = dz - dx;
            s2 = dx - dy;
            s3 = dy;
        }
    } else {
        if (dz > dy) { // dz > dy > dx
            cA = get_lut_value(lut, x0, y0, z1, lutSize, lutSizeSquared);
            cB = get_lut_value(lut, x0, y1, z1, lutSize, lutSizeSquared);
            s0 = 1.0 - dz;
            s1 = dz - dy;
            s2 = dy - dx;
            s3 = dx;
        } else if (dz > dx) { // dy > dz > dx
            cA = get_lut_value(lut, x0, y1, z0, lutSize, lutSizeSquared);
            cB = get_lut_value(lut, x0, y1, z1, lutSize, lutSizeSquared);
            s0 = 1.0 - dy;
            s1 = dy - dz;
            s2 = dz - dx;
            s3 = dx;
        } else { // dy > dx > dz
            cA = get_lut_value(lut, x0, y1, z0, lutSize, lutSizeSquared);
            cB = get_lut_value(lut, x1, y1, z0, lutSize, lutSizeSquared);
            s0 = 1.0 - dy;
            s1 = dy - dx;
            s2 = dx - dz;
            s3 = dz;
        }
    }

    output[idx] = s0 * c000.x + s1 * cA.x + s2 * cB.x + s3 * c111.x;
    output[idx + 1] = s0 * c000.y + s1 * cA.y + s2 * cB.y + s3 * c111.y;
    output[idx + 2] = s0 * c000.z + s1 * cA.z + s2 * cB.z + s3 * c111.z;
}
"""
lut_tetrahedral_kernel = cp.RawKernel(lut_tetrahedral_kernel_code, "lut_tetrahedral_kernel")


def apply_lut_cp(frame_rgb: cp.ndarray, lut: cp.ndarray, interpolation: int = 0) -> cp.ndarray:
    """
    Apply a 3D LUT to an frame_rgb with trilinear interpolation.

    Parameters:
    frame_rgb : cp.ndarray
        Input frame_rgb. The shape is (height, width, 3). dtype is float32.
    lut : cp.ndarray
        Input LUT. The shape is (N, N, N, 3). dtype is float32.
    interpolation : int (optional)
        The interpolation method to use. by default 0, options are: 0 for trilinear, 1 for tetrahedral.

    Returns:
    result : cp.ndarray
        Output frame_rgb. The shape is (height, width, 3). dtype is float32.
    """
    try:
        height, width, _ = frame_rgb.shape
        result = cp.zeros_like(frame_rgb)

        if interpolation == 0:
            # Get the number of LUT entries minus 1 (for zero-based indexing)
            N = lut.shape[0] - 1

            # Scale the frame_rgb to the LUT size
            scaled_frame_rgb = frame_rgb * N

            # Calculate the indices for the corners of the cube for interpolation
            index_low = cp.floor(scaled_frame_rgb).astype(cp.int32)
            index_high = cp.clip(index_low + 1, 0, N)

            # Calculate the fractional part for interpolation
            fractional = scaled_frame_rgb - index_low

            # Interpolate
            for i in range(3):  # Iterate over each channel
                # Retrieve values from the LUT
                val000 = lut[index_low[..., 0], index_low[..., 1], index_low[..., 2], i]
                val001 = lut[index_low[..., 0], index_low[..., 1], index_high[..., 2], i]
                val010 = lut[index_low[..., 0], index_high[..., 1], index_low[..., 2], i]
                val011 = lut[index_low[..., 0], index_high[..., 1], index_high[..., 2], i]
                val100 = lut[index_high[..., 0], index_low[..., 1], index_low[..., 2], i]
                val101 = lut[index_high[..., 0], index_low[..., 1], index_high[..., 2], i]
                val110 = lut[index_high[..., 0], index_high[..., 1], index_low[..., 2], i]
                val111 = lut[index_high[..., 0], index_high[..., 1], index_high[..., 2], i]

                # Perform trilinear interpolation
                val00 = val000 * (1 - fractional[..., 0]) + val100 * fractional[..., 0]
                val01 = val001 * (1 - fractional[..., 0]) + val101 * fractional[..., 0]
                val10 = val010 * (1 - fractional[..., 0]) + val110 * fractional[..., 0]
                val11 = val011 * (1 - fractional[..., 0]) + val111 * fractional[..., 0]

                val0 = val00 * (1 - fractional[..., 1]) + val10 * fractional[..., 1]
                val1 = val01 * (1 - fractional[..., 1]) + val11 * fractional[..., 1]

                final_val = val0 * (1 - fractional[..., 2]) + val1 * fractional[..., 2]

                result[..., i] = final_val

                result = result.astype(cp.float32)

        elif interpolation == 1:
            # Get the number of LUT entries minus 1 (for zero-based indexing)
            dim = lut.shape[0]
            dim_minus_one = dim - 1

            # Scale the frame_rgb to the LUT size
            scaled_frame_rgb = cp.clip(frame_rgb * dim_minus_one, 0, dim_minus_one - 1e-5)

            # Calculate the indices for the corners of the cube for interpolation
            index_floor = cp.floor(scaled_frame_rgb).astype(cp.int32)
            index_ceil = cp.ceil(scaled_frame_rgb).astype(cp.int32)

            # Calculate the fractional part for interpolation
            weights = scaled_frame_rgb - index_floor

            # Interpolate
            for i in range(height):
                for j in range(width):
                    fx, fy, fz = weights[i, j]
                    if fx > fy:
                        if fy > fz:
                            w0, w1, w2, w3 = (1 - fx, fx - fy, fy - fz, fz)
                        elif fx > fz:
                            w0, w1, w2, w3 = (1 - fx, fx - fz, fz - fy, fy)
                        else:
                            w0, w1, w2, w3 = (1 - fz, fz - fx, fx - fy, fy)
                    else:
                        if fz > fy:
                            w0, w1, w2, w3 = (1 - fz, fz - fy, fy - fx, fx)
                        elif fz > fx:
                            w0, w1, w2, w3 = (1 - fy, fy - fz, fz - fx, fx)
                        else:
                            w0, w1, w2, w3 = (1 - fy, fy - fx, fx - fz, fz)

                    # Calculate corresponding LUT indices
                    indices = index_floor[i, j], index_ceil[i, j]
                    c000 = lut[indices[0][0], indices[0][1], indices[0][2]]
                    c100 = lut[indices[1][0], indices[0][1], indices[0][2]]
                    c110 = lut[indices[1][0], indices[1][1], indices[0][2]]
                    c111 = lut[indices[1][0], indices[1][1], indices[1][2]]

                    # Calculate interpolated color
                    result[i, j] = w0 * c000 + w1 * c100 + w2 * c110 + w3 * c111
        return result

    except Exception as e:
        print(f"Error apply_lut: {e}")
        raise e
