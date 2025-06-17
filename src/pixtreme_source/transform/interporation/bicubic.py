import cupy as cp

bicubic_kernel_code = r"""
extern "C" __global__
void bicubic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int channels
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height) return;

    const float scale_x = (float)input_width / output_width;
    const float scale_y = (float)input_height / output_height;

    const float src_x = x * scale_x;
    const float src_y = y * scale_y;

    const int x0 = int(src_x) - 1;
    const int y0 = int(src_y) - 1;

    auto cubic_kernel = [](float x) -> float {
        x = abs(x);
        if (x <= 1.0f)
            return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
        else if (x < 2.0f)
            return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
        return 0.0f;
    };

    // Position of the output pixel in the output array
    const int out_pos = (y * output_width + x) * channels;

    // Calculate interpolation for each channel
    #pragma unroll
    for (int c = 0; c < channels; c++) {
        float result = 0.0f;
        float weight_sum = 0.0f;

        // Calculate weighted sum for 16 neighboring pixels
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                const int src_y_idx = max(0, min(y0 + i, input_height - 1));
                const int src_x_idx = max(0, min(x0 + j, input_width - 1));

                const float dx = abs(src_x - (x0 + j));
                const float dy = abs(src_y - (y0 + i));

                const float weight = cubic_kernel(dx) * cubic_kernel(dy);

                // Calculate direct index from input image
                const int src_idx = (src_y_idx * input_width + src_x_idx) * channels + c;
                result += input[src_idx] * weight;
                weight_sum += weight;
            }
        }

        // Write result
        output[out_pos + c] = result / weight_sum;
    }
}
"""

bicubic_kernel = cp.RawKernel(bicubic_kernel_code, "bicubic_kernel")


bicubic_affine_kernel_code = r"""
extern "C" __global__
void bicubic_affine(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ matrix,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_width || y >= output_height) return;

    // Define bicubic interpolation kernel as a lambda
    auto cubic_kernel = [](float x) -> float {
        x = abs(x);
        if (x <= 1.0f)
            return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
        else if (x < 2.0f)
            return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
        return 0.0f;
    };

    // Calculate input coordinates using affine transformation
    const float src_x = matrix[0] * x + matrix[1] * y + matrix[2];
    const float src_y = matrix[3] * x + matrix[4] * y + matrix[5];

    // Calculate nearest integer coordinates (rounding)
    const int x0 = int(src_x) - 1;
    const int y0 = int(src_y) - 1;

    // Check bounds
    if (src_x < 2.0f || src_x >= input_width - 2.0f ||
        src_y < 2.0f || src_y >= input_height - 2.0f) {
        const int out_pos = (y * output_width + x) * 3;
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            output[out_pos + c] = 0.0f;
        }
        return;
    }

    // Position of the output pixel in the output array
    const int out_pos = (y * output_width + x) * 3;

    // Calculate interpolation for each channel
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        float result = 0.0f;
        float weight_sum = 0.0f;

        // Calculate weighted sum for 16 neighboring pixels
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int src_y_idx = max(0, min(y0 + i, input_height - 1));
            const float dy = abs(src_y - (y0 + i));
            const float wy = cubic_kernel(dy);
            if (wy == 0.0f) continue;

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int src_x_idx = max(0, min(x0 + j, input_width - 1));
                const float dx = abs(src_x - (x0 + j));
                const float wx = cubic_kernel(dx);
                if (wx == 0.0f) continue;

                const float weight = wx * wy;
                const int src_idx = (src_y_idx * input_width + src_x_idx) * 3 + c;
                result += input[src_idx] * weight;
                weight_sum += weight;
            }
        }

        // Write result
        output[out_pos + c] = weight_sum > 0.0f ? result / weight_sum : 0.0f;
    }
}
"""

bicubic_affine_kernel = cp.RawKernel(bicubic_affine_kernel_code, "bicubic_affine")
