"""GPU kernel source and launch geometry for quality metrics."""

from __future__ import annotations

_THREADS_PER_BLOCK = 256

_SSIM_KERNEL_SOURCE = r"""
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
    const long long output_width = width - 10;
    const long long output_height = height - 10;
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= output_width * output_height) {
        return;
    }
    const long long output_x = index % output_width;
    const long long output_y = index / output_width;
    float channel_sum = 0.0f;
    for (long long channel = 0; channel < channel_count; ++channel) {
        float mu_reference = 0.0f;
        float mu_candidate = 0.0f;
        for (long long window_y = 0; window_y < 11; ++window_y) {
            for (long long window_x = 0; window_x < 11; ++window_x) {
                const long long source_index =
                    (((output_y + window_y) * width + output_x + window_x) * channel_count) + channel;
                const float weight = weights[window_y * 11 + window_x];
                mu_reference += weight * reference[source_index];
                mu_candidate += weight * candidate[source_index];
            }
        }
        float variance_reference = 0.0f;
        float variance_candidate = 0.0f;
        float covariance = 0.0f;
        for (long long window_y = 0; window_y < 11; ++window_y) {
            for (long long window_x = 0; window_x < 11; ++window_x) {
                const long long source_index =
                    (((output_y + window_y) * width + output_x + window_x) * channel_count) + channel;
                const float weight = weights[window_y * 11 + window_x];
                const float centered_reference = reference[source_index] - mu_reference;
                const float centered_candidate = candidate[source_index] - mu_candidate;
                variance_reference += weight * centered_reference * centered_reference;
                variance_candidate += weight * centered_candidate * centered_candidate;
                covariance += weight * centered_reference * centered_candidate;
            }
        }
        const float mean_product = mu_reference * mu_candidate;
        const float luminance_numerator = mean_product + mean_product + c1;
        const float structure_numerator = covariance + covariance + c2;
        const float luminance_denominator =
            (mu_reference * mu_reference) + (mu_candidate * mu_candidate) + c1;
        const float structure_denominator = variance_reference + variance_candidate + c2;
        channel_sum +=
            (luminance_numerator * structure_numerator) / (luminance_denominator * structure_denominator);
    }
    output[index] = channel_sum / (float)channel_count;
}
"""


def _block_count(element_count: int) -> int:
    return (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
