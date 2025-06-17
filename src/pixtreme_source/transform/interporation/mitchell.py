import cupy as cp

mitchell_kernel_code = r"""
extern "C" __device__ float mitchell_weight(float x, float B, float C) {
    const float ax = fabsf(x);
    float result;
    if (ax < 1.0f) {
        result = ((12.0f - 9.0f * B - 6.0f * C) * ax * ax * ax
            + (-18.0f + 12.0f * B + 6.0f * C) * ax * ax
            + (6.0f - 2.0f * B)) / 6.0f;
    } else if (ax < 2.0f) {
        result = ((-B - 6.0f * C) * ax * ax * ax
            + (6.0f * B + 30.0f * C) * ax * ax
            + (-12.0f * B - 48.0f * C) * ax
            + (8.0f * B + 24.0f * C)) / 6.0f;
    } else {
        result = 0.0f;
    }
    return result;
}

extern "C" __global__
void mitchell_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int channels,
    const float B,
    const float C
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_width || y >= output_height) return;

    const float src_x = (x + 0.5f) * input_width / output_width - 0.5f;
    const float src_y = (y + 0.5f) * input_height / output_height - 0.5f;

    const int x1 = floorf(src_x - 1);
    const int y1 = floorf(src_y - 1);

    // Optimization: Check bounds before processing
    float channel_results[3];  // Support up to 3 channels

    #pragma unroll
    for (int c = 0; c < min(channels, 4); c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        // Optimization 2: Dynamically limit y loop range
        const int start_y = max(0, y1);
        const int end_y = min(input_height, y1 + 4);

        for (int iy = start_y; iy < end_y; iy++) {
            const float dy_dist = fabsf(src_y - iy);
            if (dy_dist >= 2.0f) continue;
            const float wy = mitchell_weight(dy_dist, B, C);

            // Optimization 3: Dynamically limit x loop range
            const int start_x = max(0, x1);
            const int end_x = min(input_width, x1 + 4);

            for (int ix = start_x; ix < end_x; ix++) {
                const float dx_dist = fabsf(src_x - ix);
                if (dx_dist >= 2.0f) continue;
                const float wx = mitchell_weight(dx_dist, B, C);
                const float weight = wx * wy;

                const float pixel = input[(iy * input_width + ix) * channels + c];
                sum += pixel * weight;
                weight_sum += weight;
            }
        }

        channel_results[c] = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
    }

    // Optimization 4: Write results in a single operation
    const int output_idx = (y * output_width + x) * channels;
    #pragma unroll
    for (int c = 0; c < min(channels, 4); c++) {
        output[output_idx + c] = channel_results[c];
    }
}
"""

mitchell_kernel = cp.RawKernel(mitchell_kernel_code, "mitchell_kernel")


# Mitchell-Netravali interpolation kernel
mitchell_affine_kernel_code = r"""
extern "C" __device__ float mitchell_weight(float x, float B, float C) {
    const float ax = fabsf(x);
    float result;
    if (ax < 1.0f) {
        result = ((12.0f - 9.0f * B - 6.0f * C) * ax * ax * ax
            + (-18.0f + 12.0f * B + 6.0f * C) * ax * ax
            + (6.0f - 2.0f * B)) / 6.0f;
    } else if (ax < 2.0f) {
        result = ((-B - 6.0f * C) * ax * ax * ax
            + (6.0f * B + 30.0f * C) * ax * ax
            + (-12.0f * B - 48.0f * C) * ax
            + (8.0f * B + 24.0f * C)) / 6.0f;
    } else {
        result = 0.0f;
    }
    return result;
}

extern "C" __global__
void mitchell_affine_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ matrix,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float B,
    const float C
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_width || y >= output_height) return;

    // Calculate input coordinates using affine transformation
    const float src_x = matrix[0] * x + matrix[1] * y + matrix[2];
    const float src_y = matrix[3] * x + matrix[4] * y + matrix[5];

    // Calculate nearest integer coordinates (rounding)
    const int x1 = floorf(src_x - 1);
    const int y1 = floorf(src_y - 1);

    // Check bounds
    if (src_x < 1 || src_x >= input_width - 2 || src_y < 1 || src_y >= input_height - 2) {
        const int output_idx = (y * output_width + x) * 3;
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            output[output_idx + c] = 0.0f;
        }
        return;
    }

    // Optimization 1: Store results in temporary variables for each channel
    float channel_results[3];

    #pragma unroll
    for (int c = 0; c < 3; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        // Optimization 2: Dynamically limit y loop range
        const int start_y = max(0, y1);
        const int end_y = min(input_height, y1 + 4);

        for (int iy = start_y; iy < end_y; iy++) {
            const float dy_dist = fabsf(src_y - iy);
            if (dy_dist >= 2.0f) continue;
            const float wy = mitchell_weight(dy_dist, B, C);

            // Optimization 3: Dynamically limit x loop range
            const int start_x = max(0, x1);
            const int end_x = min(input_width, x1 + 4);

            for (int ix = start_x; ix < end_x; ix++) {
                const float dx_dist = fabsf(src_x - ix);
                if (dx_dist >= 2.0f) continue;
                const float wx = mitchell_weight(dx_dist, B, C);
                const float weight = wx * wy;
                const float pixel = input[(iy * input_width + ix) * 3 + c];
                sum += pixel * weight;
                weight_sum += weight;
            }
        }
        channel_results[c] = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
    }

    // Optimization 4: Write results in a single operation
    const int output_idx = (y * output_width + x) * 3;
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        output[output_idx + c] = channel_results[c];
    }
}
"""

mitchell_affine_kernel = cp.RawKernel(mitchell_affine_kernel_code, "mitchell_affine_kernel")
