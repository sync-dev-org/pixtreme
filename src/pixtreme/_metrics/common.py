"""GPU kernel source and launch geometry for quality metrics."""

from __future__ import annotations

_SSIM_TILE_WIDTH = 16
_SSIM_TILE_HEIGHT = 64
_SSIM_BLOCK_ROWS = 32

_SSIM_KERNEL_SOURCE = (
    f"#define TILE_WIDTH {_SSIM_TILE_WIDTH}\n"
    f"#define TILE_HEIGHT {_SSIM_TILE_HEIGHT}\n"
    f"#define BLOCK_ROWS {_SSIM_BLOCK_ROWS}\n"
    "#define TILE_AREA ((TILE_HEIGHT + 10) * TILE_WIDTH)\n"
    + r"""
struct Pair {
    float high;
    float low;
};

// Keep the rounding error of each sum and product in the second float.
__device__ __forceinline__ Pair pair_add(const Pair a, const Pair b) {
    const float sum = a.high + b.high;
    const float part = sum - a.high;
    float error = (a.high - (sum - part)) + (b.high - part);
    error += a.low + b.low;
    const float high = sum + error;
    return {high, error - (high - sum)};
}

__device__ __forceinline__ Pair pair_multiply(const Pair a, const Pair b) {
    const float product = a.high * b.high;
    float error = fmaf(a.high, b.high, -product);
    error = fmaf(a.high, b.low, error);
    error = fmaf(a.low, b.high, error);
    error = fmaf(a.low, b.low, error);
    const float high = product + error;
    return {high, error - (high - product)};
}

__device__ __forceinline__ Pair pair_subtract(const Pair a, const Pair b) {
    return pair_add(a, {-b.high, -b.low});
}

__device__ __forceinline__ Pair load_pair(const float* values, const int offset, const int stride = 1) {
    return {values[offset], values[offset + stride]};
}

__device__ __forceinline__ void store_horizontal_pair(float* values, const int offset, const Pair value) {
    values[offset] = value.high;
    values[offset + TILE_AREA] = value.low;
}

extern "C" __global__ void pixtreme_ssim_map(
    const float* __restrict__ reference,
    const float* __restrict__ candidate,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const float c1,
    const float c2
) {
    // Each component has its own plane so neighboring threads access neighboring shared words.
    // Ten extra rows supply the vertical window at the bottom of this output tile.
    __shared__ float horizontal[10 * TILE_AREA];
    const long long output_width = width - 10;
    const long long output_height = height - 10;
    const long long x = (long long)blockIdx.x * TILE_WIDTH + threadIdx.x;
    const long long origin_y = (long long)blockIdx.y * TILE_HEIGHT;
    float channel_sum[TILE_HEIGHT / BLOCK_ROWS] = {};
    for (long long channel = 0; channel < channel_count; ++channel) {
        for (int row = threadIdx.y; row < TILE_HEIGHT + 10; row += BLOCK_ROWS) {
            const long long y = origin_y + row;
            if (x < output_width && y < height) {
                Pair mean_reference = {0.0f, 0.0f};
                Pair mean_candidate = {0.0f, 0.0f};
                Pair square_reference = {0.0f, 0.0f};
                Pair square_candidate = {0.0f, 0.0f};
                Pair product = {0.0f, 0.0f};
                #pragma unroll
                for (int tap = 0; tap < 11; ++tap) {
                    const long long source_index = ((y * width + x + tap) * channel_count) + channel;
                    const Pair weight = load_pair(weights, tap * 2);
                    const Pair reference_value = {reference[source_index], 0.0f};
                    const Pair candidate_value = {candidate[source_index], 0.0f};
                    mean_reference = pair_add(mean_reference, pair_multiply(weight, reference_value));
                    mean_candidate = pair_add(mean_candidate, pair_multiply(weight, candidate_value));
                    square_reference = pair_add(
                        square_reference, pair_multiply(weight, pair_multiply(reference_value, reference_value))
                    );
                    square_candidate = pair_add(
                        square_candidate, pair_multiply(weight, pair_multiply(candidate_value, candidate_value))
                    );
                    product = pair_add(product, pair_multiply(weight, pair_multiply(reference_value, candidate_value)));
                }

                const int destination = row * TILE_WIDTH + threadIdx.x;
                store_horizontal_pair(horizontal, destination, mean_reference);
                store_horizontal_pair(horizontal, destination + 2 * TILE_AREA, mean_candidate);
                store_horizontal_pair(horizontal, destination + 4 * TILE_AREA, square_reference);
                store_horizontal_pair(horizontal, destination + 6 * TILE_AREA, square_candidate);
                store_horizontal_pair(horizontal, destination + 8 * TILE_AREA, product);
            }
        }
        // All horizontal rows, including the halo, are ready before any vertical reads.
        __syncthreads();
        #pragma unroll
        for (int row = threadIdx.y; row < TILE_HEIGHT; row += BLOCK_ROWS) {
            if (x < output_width && origin_y + row < output_height) {
                Pair mean_reference = {0.0f, 0.0f};
                Pair mean_candidate = {0.0f, 0.0f};
                Pair square_reference = {0.0f, 0.0f};
                Pair square_candidate = {0.0f, 0.0f};
                Pair product = {0.0f, 0.0f};
                #pragma unroll
                for (int tap = 0; tap < 11; ++tap) {
                    const int source = (row + tap) * TILE_WIDTH + threadIdx.x;
                    const Pair weight = load_pair(weights, tap * 2);
                    mean_reference = pair_add(
                        mean_reference, pair_multiply(weight, load_pair(horizontal, source, TILE_AREA))
                    );
                    mean_candidate = pair_add(
                        mean_candidate, pair_multiply(weight, load_pair(horizontal, source + 2 * TILE_AREA, TILE_AREA))
                    );
                    square_reference = pair_add(
                        square_reference, pair_multiply(weight, load_pair(horizontal, source + 4 * TILE_AREA, TILE_AREA))
                    );
                    square_candidate = pair_add(
                        square_candidate, pair_multiply(weight, load_pair(horizontal, source + 6 * TILE_AREA, TILE_AREA))
                    );
                    product = pair_add(
                        product, pair_multiply(weight, load_pair(horizontal, source + 8 * TILE_AREA, TILE_AREA))
                    );
                }
                const Pair mean_product = pair_multiply(mean_reference, mean_candidate);
                const Pair reference_squared = pair_multiply(mean_reference, mean_reference);
                const Pair candidate_squared = pair_multiply(mean_candidate, mean_candidate);
                const Pair variance_reference = pair_subtract(square_reference, reference_squared);
                const Pair variance_candidate = pair_subtract(square_candidate, candidate_squared);
                const Pair covariance = pair_subtract(product, mean_product);
                const float luminance_numerator = (mean_product.high + mean_product.low) * 2.0f + c1;
                const float structure_numerator = (covariance.high + covariance.low) * 2.0f + c2;
                const float luminance_denominator =
                    (reference_squared.high + reference_squared.low) + (candidate_squared.high + candidate_squared.low) + c1;
                const float structure_denominator =
                    (variance_reference.high + variance_reference.low) + (variance_candidate.high + variance_candidate.low) + c2;
                channel_sum[row / BLOCK_ROWS] +=
                    (luminance_numerator * structure_numerator) / (luminance_denominator * structure_denominator);
            }
        }
        // Finish all reads before reusing the shared planes for the next channel.
        __syncthreads();
    }
    #pragma unroll
    for (int row = threadIdx.y; row < TILE_HEIGHT; row += BLOCK_ROWS) {
        if (x < output_width && origin_y + row < output_height) {
            output[(origin_y + row) * output_width + x] = channel_sum[row / BLOCK_ROWS] / (float)channel_count;
        }
    }
}
"""
)
