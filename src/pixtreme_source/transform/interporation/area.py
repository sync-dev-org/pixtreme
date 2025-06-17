import cupy as cp

area_kernel_code = r"""
extern "C" __global__
void area_kernel(const float* __restrict__ input,
                 float* __restrict__ output,
                 int input_width, int input_height,
                 int new_width, int new_height,
                 int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_width || y >= new_height) return;

    // Calculate ratio for scaling
    float x_ratio = (float)input_width / new_width;
    float y_ratio = (float)input_height / new_height;

    // Calculate source coordinate range in input image
    float input_x_start = x * x_ratio;
    float input_y_start = y * y_ratio;
    float input_x_end = (x + 1) * x_ratio;
    float input_y_end = (y + 1) * y_ratio;

    // Get integer parts (cover the entire necessary range)
    int x_start = (int)floorf(input_x_start);
    int y_start = (int)floorf(input_y_start);
    int x_end = (int)ceilf(input_x_end);
    int y_end = (int)ceilf(input_y_end);

    // Check bounds
    x_start = max(0, x_start);
    y_start = max(0, y_start);
    x_end = min(input_width, x_end);
    y_end = min(input_height, y_end);

    for (int c = 0; c < channels; ++c) {
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;

        // Calculate weighted sum for each channel
        for (int sy = y_start; sy < y_end; sy++) {
            for (int sx = x_start; sx < x_end; sx++) {
                // Calculate pixel overlap area
                float wx = fminf(input_x_end, sx + 1) - fmaxf(input_x_start, sx);
                float wy = fminf(input_y_end, sy + 1) - fmaxf(input_y_start, sy);
                float weight = wx * wy;

                int idx = (sy * input_width + sx) * channels + c;
                weighted_sum += input[idx] * weight;
                weight_sum += weight;
            }
        }

        int out_idx = (y * new_width + x) * channels + c;
        output[out_idx] = (weight_sum > 0) ? weighted_sum / weight_sum : 0.0f;
    }
}
"""

area_kernel = cp.RawKernel(area_kernel_code, "area_kernel")


# Area interpolation with affine transformation
area_affine_kernel_code = r"""
extern "C" __global__
void area_affine_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ matrix,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    unsigned int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int output_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (output_x >= output_width || output_y >= output_height) return;

    // Calculate affine transformation coordinates
    float center_x = matrix[0] * (output_x + 0.5f) + matrix[1] * (output_y + 0.5f) + matrix[2];
    float center_y = matrix[3] * (output_x + 0.5f) + matrix[4] * (output_y + 0.5f) + matrix[5];

    // Calculate actual scaling factors from transformation matrix
    float scale_x = sqrtf(matrix[0] * matrix[0] + matrix[1] * matrix[1]);
    float scale_y = sqrtf(matrix[3] * matrix[3] + matrix[4] * matrix[4]);

    // Calculate size of sampling area
    float half_width = 0.5f * scale_x;
    float half_height = 0.5f * scale_y;

    // Calculate boundaries of sampling area
    float left = center_x - half_width;
    float right = center_x + half_width;
    float top = center_y - half_height;
    float bottom = center_y + half_height;

    // Convert to integer coordinate range
    int ix0 = max(0, (int)floor(left));
    int ix1 = min(input_width, (int)ceil(right));
    int iy0 = max(0, (int)floor(top));
    int iy1 = min(input_height, (int)ceil(bottom));

    // If sampling area is outside the image
    if (ix0 >= input_width || ix1 <= 0 || iy0 >= input_height || iy1 <= 0) {
        for (int c = 0; c < 3; c++) {
            output[(output_y * output_width + output_x) * 3 + c] = 0.0f;
        }
        return;
    }

    // Calculate average value for each channel
    for (int c = 0; c < 3; c++) {
        float sum = 0.0f;
        float total_weight = 0.0f;

        // Sample each pixel in the area
        for (int y = iy0; y < iy1; y++) {
            // Calculate vertical weight for the pixel
            float wy = 1.0f;
            if (y == iy0) {
                wy = 1.0f - (top - y);
            }
            if (y == iy1 - 1) {
                wy = 1.0f - (y + 1 - bottom);
            }

            for (int x = ix0; x < ix1; x++) {
                // Calculate horizontal weight for the pixel
                float wx = 1.0f;
                if (x == ix0) {
                    wx = 1.0f - (left - x);
                }
                if (x == ix1 - 1) {
                    wx = 1.0f - (x + 1 - right);
                }

                // Calculate overall weight
                float weight = wx * wy;

                // Calculate weighted sum
                sum += input[(y * input_width + x) * 3 + c] * weight;
                total_weight += weight;
            }
        }

        // Write normalized value
        output[(output_y * output_width + output_x) * 3 + c] = total_weight > 0.0f ? sum / total_weight : 0.0f;
    }
}
"""

area_affine_kernel = cp.RawKernel(area_affine_kernel_code, "area_affine_kernel")
