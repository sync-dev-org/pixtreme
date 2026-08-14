"""Shared CUDA source for three-dimensional LUT lookup."""

from __future__ import annotations

_LUT_TETRAHEDRAL_CUDA_SOURCE = r"""
__device__ __forceinline__ float3 pixtreme_lut_load(
    const float* __restrict__ lut,
    const long long offset,
    const long long stride_c,
    const int packed
) {
    if (packed) {
        const float4 value = __ldg(reinterpret_cast<const float4*>(lut + offset));
        return make_float3(value.x, value.y, value.z);
    }
    return make_float3(
        lut[offset],
        lut[offset + stride_c],
        lut[offset + 2 * stride_c]
    );
}

__device__ __forceinline__ float3 pixtreme_lut_tetrahedral(
    const float* __restrict__ lut,
    const long long stride_r,
    const long long stride_g,
    const long long stride_b,
    const long long stride_c,
    const int red,
    const int green,
    const int blue,
    const float red_fraction,
    const float green_fraction,
    const float blue_fraction,
    const int packed
) {
    float first_fraction = red_fraction;
    float second_fraction = green_fraction;
    float third_fraction = blue_fraction;
    int first_axis = 0;
    int second_axis = 1;
    int third_axis = 2;

    // Stable descending sort preserves R, G, B tie order.
    if (first_fraction < second_fraction) {
        const float fraction = first_fraction;
        first_fraction = second_fraction;
        second_fraction = fraction;
        const int axis = first_axis;
        first_axis = second_axis;
        second_axis = axis;
    }
    if (second_fraction < third_fraction) {
        const float fraction = second_fraction;
        second_fraction = third_fraction;
        third_fraction = fraction;
        const int axis = second_axis;
        second_axis = third_axis;
        third_axis = axis;
    }
    if (first_fraction < second_fraction) {
        const float fraction = first_fraction;
        first_fraction = second_fraction;
        second_fraction = fraction;
        const int axis = first_axis;
        first_axis = second_axis;
        second_axis = axis;
    }

    const int first_red = red + (first_axis == 0);
    const int first_green = green + (first_axis == 1);
    const int first_blue = blue + (first_axis == 2);
    const int second_red = first_red + (second_axis == 0);
    const int second_green = first_green + (second_axis == 1);
    const int second_blue = first_blue + (second_axis == 2);
    const long long offset000 =
        (long long)red * stride_r + (long long)green * stride_g + (long long)blue * stride_b;
    const long long offset1 =
        (long long)first_red * stride_r + (long long)first_green * stride_g + (long long)first_blue * stride_b;
    const long long offset2 =
        (long long)second_red * stride_r + (long long)second_green * stride_g + (long long)second_blue * stride_b;
    const float3 c000 = pixtreme_lut_load(lut, offset000, stride_c, packed);
    const float3 c1 = pixtreme_lut_load(lut, offset1, stride_c, packed);
    const float3 c2 = pixtreme_lut_load(lut, offset2, stride_c, packed);
    const float3 c111 = pixtreme_lut_load(lut, offset000 + stride_r + stride_g + stride_b, stride_c, packed);
    return make_float3(
        c000.x
            + first_fraction * (c1.x - c000.x)
            + second_fraction * (c2.x - c1.x)
            + third_fraction * (c111.x - c2.x),
        c000.y
            + first_fraction * (c1.y - c000.y)
            + second_fraction * (c2.y - c1.y)
            + third_fraction * (c111.y - c2.y),
        c000.z
            + first_fraction * (c1.z - c000.z)
            + second_fraction * (c2.z - c1.z)
            + third_fraction * (c111.z - c2.z)
    );
}

__device__ __forceinline__ void pixtreme_lut_coordinate(
    const float value,
    const float domain_min,
    const float domain_max,
    const int size,
    int* lower,
    float* fraction
) {
    float position = (value - domain_min) / (domain_max - domain_min) * (size - 1);
    position = fminf(fmaxf(position, 0.0f), (float)(size - 1));
    *lower = min((int)floorf(position), size - 2);
    *fraction = position - *lower;
}
"""
