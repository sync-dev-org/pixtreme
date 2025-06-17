import cupy as cp

nearest_kernel_code = r"""
extern "C" __global__ void nearest_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width_in,
    const int height_in,
    const int width_out,
    const int height_out,
    const int channels
) {
    // Calculate global thread index
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    // Check output image bounds
    if (x_out >= width_out || y_out >= height_out) {
        return;
    }

    // Calculate scale ratio
    const float scale_x = static_cast<float>(width_in) / width_out;
    const float scale_y = static_cast<float>(height_in) / height_out;

    // Calculate corresponding coordinates in input image
    const int x_in = min(static_cast<int>(x_out * scale_x), width_in - 1);
    const int y_in = min(static_cast<int>(y_out * scale_y), height_in - 1);

    // Process for each channel
    for (int c = 0; c < channels; c++) {
        // Calculate input image index
        const int idx_in = (y_in * width_in + x_in) * channels + c;
        // Calculate output image index
        const int idx_out = (y_out * width_out + x_out) * channels + c;

        // Copy pixel value
        output[idx_out] = input[idx_in];
    }
}
"""

nearest_kernel = cp.RawKernel(nearest_kernel_code, "nearest_kernel")


nearest_affine_kernel_code = r"""
extern "C" __global__ void nearest_affine(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ matrix,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    // Calculate global thread index
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    // Check output image bounds
    if (x_out >= output_width || y_out >= output_height) {
        return;
    }

    // Calculate input coordinates using affine transformation
    const float src_x = matrix[0] * x_out + matrix[1] * y_out + matrix[2];
    const float src_y = matrix[3] * x_out + matrix[4] * y_out + matrix[5];

    // Calculate nearest integer coordinates (rounding)
    const int x_in = __float2int_rn(src_x);
    const int y_in = __float2int_rn(src_y);

    // Check bounds
    if (x_in < 0 || x_in >= input_width || y_in < 0 || y_in >= input_height) {
        const int idx_out = (y_out * output_width + x_out) * 3;
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            output[idx_out + c] = 0.0f;
        }
        return;
    }

    // Calculate base indices for input and output
    const int idx_in_base = (y_in * input_width + x_in) * 3;
    const int idx_out_base = (y_out * output_width + x_out) * 3;

    // Optimize: Process all 3 channels at once
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        output[idx_out_base + c] = input[idx_in_base + c];
    }
}
"""

nearest_affine_kernel = cp.RawKernel(nearest_affine_kernel_code, "nearest_affine")
