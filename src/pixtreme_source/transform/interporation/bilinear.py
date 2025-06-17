import cupy as cp

bilinear_kernel_code = r"""
extern "C" __global__ void bilinear_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width_in,
    const int height_in,
    const int width_out,
    const int height_out,
    const int channels
) {
    // Get the output pixel coordinates
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the output image bounds
    if (x_out >= width_out || y_out >= height_out) {
        return;
    }

    // Calculate scaling factors
    const float scale_x = (float)(width_in - 1) / (float)(width_out - 1);
    const float scale_y = (float)(height_in - 1) / (float)(height_out - 1);

    // Calculate the corresponding position in the input image
    const float x_in = x_out * scale_x;
    const float y_in = y_out * scale_y;

    // Get the four neighboring pixels
    const int x0 = __float2int_rd(x_in);  // floor
    const int x1 = min(x0 + 1, width_in - 1);
    const int y0 = __float2int_rd(y_in);  // floor
    const int y1 = min(y0 + 1, height_in - 1);

    // Calculate interpolation weights
    const float wx = x_in - x0;
    const float wy = y_in - y0;

    // Interpolate for each channel
    for (int c = 0; c < channels; c++) {
        // Get the values of the four neighboring pixels for this channel
        const float f00 = input[(y0 * width_in + x0) * channels + c];
        const float f10 = input[(y0 * width_in + x1) * channels + c];
        const float f01 = input[(y1 * width_in + x0) * channels + c];
        const float f11 = input[(y1 * width_in + x1) * channels + c];

        // Perform bilinear interpolation
        // First interpolate in x direction
        const float fx0 = f00 * (1.0f - wx) + f10 * wx;
        const float fx1 = f01 * (1.0f - wx) + f11 * wx;

        // Then interpolate in y direction
        const float result = fx0 * (1.0f - wy) + fx1 * wy;

        // Write the result to the output image
        output[(y_out * width_out + x_out) * channels + c] = result;
    }
}
"""


bilinear_kernel = cp.RawKernel(bilinear_kernel_code, "bilinear_kernel")


bilinear_affine_kernel_code = r"""
extern "C" __global__
void bilinear_affine_kernel(const float *src, float *dst, const float *matrix, int src_height, int src_width, int dst_height, int dst_width) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_width || dst_y >= dst_height) return;

    // Affine transformation
    float tx = matrix[0] * dst_x + matrix[1] * dst_y + matrix[2];
    float ty = matrix[3] * dst_x + matrix[4] * dst_y + matrix[5];

    int x0 = floor(tx);
    int y0 = floor(ty);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Out of bounds
    if (tx < 0 || tx >= src_width || ty < 0 || ty >= src_height) {
        for (int c = 0; c < 3; c++) {
            dst[(dst_y * dst_width + dst_x) * 3 + c] = 0.0;
        }
        return;
    }

    // Bilinear interpolation
    float wa = (x1 - tx) * (y1 - ty);
    float wb = (x1 - tx) * (ty - y0);
    float wc = (tx - x0) * (y1 - ty);
    float wd = (tx - x0) * (ty - y0);

    x0 = max(0, min(x0, src_width - 1));
    y0 = max(0, min(y0, src_height - 1));
    x1 = max(0, min(x1, src_width - 1));
    y1 = max(0, min(y1, src_height - 1));

    // Output
    for (int c = 0; c < 3; c++) {
        float val00 = src[(y0 * src_width + x0) * 3 + c];
        float val01 = src[(y0 * src_width + x1) * 3 + c];
        float val10 = src[(y1 * src_width + x0) * 3 + c];
        float val11 = src[(y1 * src_width + x1) * 3 + c];

        dst[(dst_y * dst_width + dst_x) * 3 + c] = wa * val00 + wb * val10 + wc * val01 + wd * val11;
    }
}
"""

bilinear_affine_kernel = cp.RawKernel(bilinear_affine_kernel_code, "bilinear_affine_kernel")
