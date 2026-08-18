"""Shared GPU kernel source and launch geometry for analysis operations."""

from __future__ import annotations

from pixtreme._core.border import _BORDER_PREAMBLE

_THREADS_PER_BLOCK = 256


_ANALYZE_KERNEL_SOURCE = (
    _BORDER_PREAMBLE
    + r"""
extern "C" __global__ void pixtreme_harris_tensor(
    const float* __restrict__ source,
    float* __restrict__ tensor,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const int border,
    const float border_value
) {
    const long long extended_width = width + 2 * radius;
    const long long extended_height = height + 2 * radius;
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= extended_width * extended_height) {
        return;
    }
    const long long x = index % extended_width - radius;
    const long long y = index / extended_width - radius;
    float tensor_a = 0.0f;
    float tensor_b = 0.0f;
    float tensor_d = 0.0f;
    for (long long channel = 0; channel < channel_count; ++channel) {
        const float top_left = pixtreme_border_sample(
            source, x - 1, y - 1, width, height, channel_count, channel, border, border_value
        );
        const float top = pixtreme_border_sample(
            source, x, y - 1, width, height, channel_count, channel, border, border_value
        );
        const float top_right = pixtreme_border_sample(
            source, x + 1, y - 1, width, height, channel_count, channel, border, border_value
        );
        const float left = pixtreme_border_sample(
            source, x - 1, y, width, height, channel_count, channel, border, border_value
        );
        const float right = pixtreme_border_sample(
            source, x + 1, y, width, height, channel_count, channel, border, border_value
        );
        const float bottom_left = pixtreme_border_sample(
            source, x - 1, y + 1, width, height, channel_count, channel, border, border_value
        );
        const float bottom = pixtreme_border_sample(
            source, x, y + 1, width, height, channel_count, channel, border, border_value
        );
        const float bottom_right = pixtreme_border_sample(
            source, x + 1, y + 1, width, height, channel_count, channel, border, border_value
        );
        const float derivative_x =
            -top_left + top_right - 2.0f * left + 2.0f * right - bottom_left + bottom_right;
        const float derivative_y =
            -top_left - 2.0f * top - top_right + bottom_left + 2.0f * bottom + bottom_right;
        tensor_a += derivative_x * derivative_x;
        tensor_b += derivative_x * derivative_y;
        tensor_d += derivative_y * derivative_y;
    }
    tensor[index * 3] = tensor_a;
    tensor[index * 3 + 1] = tensor_b;
    tensor[index * 3 + 2] = tensor_d;
}

extern "C" __global__ void pixtreme_harris_response(
    const float* __restrict__ tensor,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long radius,
    const float k
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= width * height) {
        return;
    }
    const long long x = index % width;
    const long long y = index / width;
    const long long extended_width = width + 2 * radius;
    float aggregate_a = 0.0f;
    float aggregate_b = 0.0f;
    float aggregate_d = 0.0f;
    for (long long offset_y = 0; offset_y <= 2 * radius; ++offset_y) {
        for (long long offset_x = 0; offset_x <= 2 * radius; ++offset_x) {
            const long long tensor_index = ((y + offset_y) * extended_width + x + offset_x) * 3;
            aggregate_a += tensor[tensor_index];
            aggregate_b += tensor[tensor_index + 1];
            aggregate_d += tensor[tensor_index + 2];
        }
    }
    const float trace = aggregate_a + aggregate_d;
    output[index] = aggregate_a * aggregate_d - aggregate_b * aggregate_b - k * trace * trace;
}

extern "C" __global__ void pixtreme_match_template(
    const float* __restrict__ source,
    const float* __restrict__ template_data,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long template_width,
    const long long template_height,
    const int method
) {
    const long long output_width = width - template_width + 1;
    const long long output_height = height - template_height + 1;
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= output_width * output_height) {
        return;
    }
    const long long output_x = index % output_width;
    const long long output_y = index / output_width;
    const long long spatial_count = template_width * template_height;
    float numerator = 0.0f;
    float source_energy = 0.0f;
    float template_energy = 0.0f;

    if (method == 4 || method == 5) {
        for (long long channel = 0; channel < channel_count; ++channel) {
            const long long source_reference_index =
                ((output_y * width + output_x) * channel_count) + channel;
            const long long template_reference_index = channel;
            const float source_reference = source[source_reference_index];
            const float template_reference = template_data[template_reference_index];
            // Anchor before accumulation so every stored constant centers to exact zero.
            float source_offset_sum = 0.0f;
            float template_offset_sum = 0.0f;
            for (long long template_y = 0; template_y < template_height; ++template_y) {
                for (long long template_x = 0; template_x < template_width; ++template_x) {
                    const long long source_index =
                        (((output_y + template_y) * width + output_x + template_x) * channel_count) + channel;
                    const long long template_index =
                        ((template_y * template_width + template_x) * channel_count) + channel;
                    source_offset_sum += source[source_index] - source_reference;
                    template_offset_sum += template_data[template_index] - template_reference;
                }
            }
            const float source_offset_mean = source_offset_sum / (float)spatial_count;
            const float template_offset_mean = template_offset_sum / (float)spatial_count;
            for (long long template_y = 0; template_y < template_height; ++template_y) {
                for (long long template_x = 0; template_x < template_width; ++template_x) {
                    const long long source_index =
                        (((output_y + template_y) * width + output_x + template_x) * channel_count) + channel;
                    const long long template_index =
                        ((template_y * template_width + template_x) * channel_count) + channel;
                    const float centered_source = (source[source_index] - source_reference) - source_offset_mean;
                    const float centered_template =
                        (template_data[template_index] - template_reference) - template_offset_mean;
                    numerator += centered_source * centered_template;
                    source_energy += centered_source * centered_source;
                    template_energy += centered_template * centered_template;
                }
            }
        }
    } else {
        for (long long template_y = 0; template_y < template_height; ++template_y) {
            for (long long template_x = 0; template_x < template_width; ++template_x) {
                const long long source_base =
                    ((output_y + template_y) * width + output_x + template_x) * channel_count;
                const long long template_base = (template_y * template_width + template_x) * channel_count;
                for (long long channel = 0; channel < channel_count; ++channel) {
                    const float source_value = source[source_base + channel];
                    const float template_value = template_data[template_base + channel];
                    if (method == 0 || method == 1) {
                        const float difference = source_value - template_value;
                        numerator += difference * difference;
                    } else {
                        numerator += source_value * template_value;
                    }
                    if (method == 1 || method == 3) {
                        source_energy += source_value * source_value;
                        template_energy += template_value * template_value;
                    }
                }
            }
        }
    }

    if (method == 0 || method == 2 || method == 4) {
        output[index] = numerator;
        return;
    }
    const float denominator_squared = source_energy * template_energy;
    if (denominator_squared > 0.0f) {
        output[index] = numerator / sqrtf(denominator_squared);
    } else if (method == 1 && numerator > 0.0f) {
        output[index] = __int_as_float(0x7f800000);
    } else {
        output[index] = 0.0f;
    }
}

extern "C" __global__ void pixtreme_match_template_fft_response(
    const float* __restrict__ correlation,
    const float* __restrict__ window_sums,
    const float* __restrict__ squared_window_sums,
    const float* __restrict__ template_sums,
    const float* __restrict__ template_energy,
    const float* __restrict__ centered_template_energy,
    const bool* __restrict__ zero_variance,
    float* __restrict__ output,
    const long long element_count,
    const long long channel_count,
    const float spatial_count,
    const int method
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const long long channel_base = index * channel_count;
    float source_energy = 0.0f;
    float centered_sum = 0.0f;
    float centered_source_sum = 0.0f;
    for (long long channel = 0; channel < channel_count; ++channel) {
        if (method == 3 || method == 5) {
            source_energy = __fadd_rn(source_energy, squared_window_sums[channel_base + channel]);
        }
        if (method == 4 || method == 5) {
            const float window_sum = window_sums[channel_base + channel];
            const float weighted_sum = __fdiv_rn(
                __fmul_rn(window_sum, template_sums[channel]),
                spatial_count
            );
            centered_sum = __fadd_rn(centered_sum, weighted_sum);
            if (method == 5) {
                const float squared_sum = __fdiv_rn(__fmul_rn(window_sum, window_sum), spatial_count);
                centered_source_sum = __fadd_rn(centered_source_sum, squared_sum);
            }
        }
    }

    float numerator = correlation[index];
    if (method == 4 || method == 5) {
        numerator = __fsub_rn(numerator, centered_sum);
        if (zero_variance[index]) {
            output[index] = 0.0f;
            return;
        }
    }
    if (method == 4) {
        output[index] = numerator;
        return;
    }

    float denominator_squared;
    if (method == 3) {
        denominator_squared = __fmul_rn(source_energy, template_energy[0]);
    } else {
        const float centered_source_energy = __fsub_rn(source_energy, centered_source_sum);
        denominator_squared = __fmul_rn(centered_source_energy, centered_template_energy[0]);
    }
    output[index] = denominator_squared > 0.0f ? numerator / sqrtf(denominator_squared) : 0.0f;
}

"""
)


def _block_count(element_count: int) -> int:
    return (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
