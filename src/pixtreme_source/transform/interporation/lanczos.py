import cupy as cp

lanczos_kernel_code = r"""
extern "C" __global__ void lanczos_kernel(
    const float* input,
    float* output,
    const int width_in,
    const int height_in,
    const int width_out,
    const int height_out,
    const int a,
    const int channels
) {
    const float PI = 3.14159265358979323846f;

    // Calculate thread indices
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_out >= width_out || y_out >= height_out) return;

    // Calculate scaling factors
    const float scale_x = (float)width_out / width_in;
    const float scale_y = (float)height_out / height_in;

    // Calculate source position
    const float x_in = x_out / scale_x;
    const float y_in = y_out / scale_y;

    // Calculate kernel centers
    const int kernel_center_x = (int)x_in;
    const int kernel_center_y = (int)y_in;

    // Calculate kernel boundaries
    const int start_x = max(kernel_center_x - a, 0);
    const int end_x = min(kernel_center_x + a + 1, width_in);
    const int start_y = max(kernel_center_y - a, 0);
    const int end_y = min(kernel_center_y + a + 1, height_in);

    // Process each channel
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        // Apply Lanczos kernel
        for (int y_k = start_y; y_k < end_y; ++y_k) {
            const float dy = y_in - y_k;
            float wy;

            // Calculate y-direction kernel weight
            if (dy == 0.0f) {
                wy = 1.0f;
            }
            else if (abs(dy) < a) {
                const float y_pi = PI * dy;
                wy = a * sin(y_pi) * sin(y_pi / a) / (y_pi * y_pi);
            }
            else {
                wy = 0.0f;
            }

            for (int x_k = start_x; x_k < end_x; ++x_k) {
                const float dx = x_in - x_k;
                float wx;

                // Calculate x-direction kernel weight
                if (dx == 0.0f) {
                    wx = 1.0f;
                }
                else if (abs(dx) < a) {
                    const float x_pi = PI * dx;
                    wx = a * sin(x_pi) * sin(x_pi / a) / (x_pi * x_pi);
                }
                else {
                    wx = 0.0f;
                }

                const float weight = wx * wy;
                const float pixel_value = input[(y_k * width_in + x_k) * channels + c];
                sum += weight * pixel_value;
                weight_sum += weight;
            }
        }

        // Normalize and store result
        if (weight_sum > 0.0f) {
            output[(y_out * width_out + x_out) * channels + c] = sum / weight_sum;
        }
        else {
            output[(y_out * width_out + x_out) * channels + c] = 0.0f;
        }
    }
}
"""


lanczos_kernel = cp.RawKernel(lanczos_kernel_code, "lanczos_kernel")

lanczos_affine_kernel_code = r"""
extern "C" __device__ float sinc(float x) {
    if (x == 0.0f) return 1.0f;
    const float pi_x = 3.14159265358979323846f * x;
    return sinf(pi_x) / pi_x;
}

extern "C" __device__ float lanczos(float x, int a) {
    if (x < -a || x > a) return 0.0f;
    if (x == 0.0f) return 1.0f;
    return sinc(x) * sinc(x / a);
}

extern "C" __global__
void lanczos_affine(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ matrix,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int a
) {
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_out >= output_width || y_out >= output_height) return;


    // Calculate affine transformation coordinates
    const float src_x = matrix[0] * x_out + matrix[1] * y_out + matrix[2];
    const float src_y = matrix[3] * x_out + matrix[4] * y_out + matrix[5];

    // Check bounds
    const int margin = a + 1;
    if (src_x < margin || src_x >= input_width - margin ||
        src_y < margin || src_y >= input_height - margin) {
        const int output_idx = (y_out * output_width + x_out) * 3;
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            output[output_idx + c] = 0.0f;
        }
        return;
    }

    // Calculate nearest integer coordinates (rounding)
    const int ix = floorf(src_x);
    const int iy = floorf(src_y);

    // Store results in temporary variables for each channel
    float channel_results[3] = {0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int c = 0; c < 3; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        // Limit y loop range
        const int start_y = max(0, iy - a + 1);
        const int end_y = min(input_height - 1, iy + a);

        #pragma unroll
        for (int sy = start_y; sy < end_y + 1; sy++) {
            const float dy = src_y - sy;
            if (fabsf(dy) >= a) continue;
            const float wy = lanczos(dy, a);

            // Limit x loop range
            const int start_x = max(0, ix - a + 1);
            const int end_x = min(input_width - 1, ix + a);

            #pragma unroll
            for (int sx = start_x; sx < end_x + 1; sx++) {
                const float dx = src_x - sx;
                if (fabsf(dx) >= a) continue;
                const float wx = lanczos(dx, a);

                const float weight = wx * wy;
                if (weight == 0.0f) continue;

                const int input_idx = (sy * input_width + sx) * 3 + c;
                sum += input[input_idx] * weight;
                weight_sum += weight;
            }
        }

        channel_results[c] = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
    }

    // Write results
    const int output_idx = (y_out * output_width + x_out) * 3;
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        output[output_idx + c] = channel_results[c];
    }
}
"""

lanczos_affine_kernel = cp.RawKernel(lanczos_affine_kernel_code, "lanczos_affine")
