"""GPU-native transformed source-over image composition."""

from __future__ import annotations

import math
from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._channel.shuffle import _route_frame
from pixtreme._color.semantics import rgb_to_ycbcr, ycbcr_to_rgb
from pixtreme._color.transform import rgb_to_rgb
from pixtreme._core.blend import _BLEND_CODES, _BLEND_DEVICE_SOURCE, _BLEND_TOKENS
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _validate_float32_frame
from pixtreme._core.interpolation import _POINT_INTERPOLATION_DEVICE_SOURCE, _POINT_INTERPOLATION_TOKENS
from pixtreme._core.validation import (
    _bounded_real,
    _closed_str_token,
    _finite_pair,
    _finite_real,
    _positive_scalar_or_pair,
    _strict_bool,
)
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _ALPHA_TOKENS, Alpha, Blend, Interpolation

_COMPOSITE_INTERPOLATION_TOKENS = _POINT_INTERPOLATION_TOKENS
_RGB_CHANNELS = ("R", "G", "B")
_YCBCR_CHANNELS = ("Y", "Cb", "Cr")
_COMPOSITE_BLOCK = (16, 16)

_COMPOSITE_KERNEL_SOURCE = (
    _POINT_INTERPOLATION_DEVICE_SOURCE
    + _BLEND_DEVICE_SOURCE
    + r"""
__device__ int pixtreme_axis_plan(
    const float coordinate,
    const long long extent,
    const int interpolation,
    long long indices[8],
    float weights[8]
) {
    const float source_coordinate = coordinate - 0.5f;
    if (interpolation == 0) {
        const long long index = (long long)floorf(source_coordinate + 0.5f);
        indices[0] = index >= 0 && index < extent ? index : -1;
        weights[0] = 1.0f;
        return 1;
    }

    const long long base = (long long)floorf(source_coordinate);
    const int lobes = interpolation >= 5 ? interpolation - 3 : 0;
    const int sample_count = interpolation == 1 ? 2 : (lobes > 0 ? 2 * lobes : 4);
    const long long start = interpolation == 1 ? base : base - (lobes > 0 ? lobes - 1 : 1);
    float weight_sum = 0.0f;
    #pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
        if (offset >= sample_count) {
            indices[offset] = -1;
            weights[offset] = 0.0f;
            continue;
        }
        const long long index = start + offset;
        const float weight = pixtreme_point_weight(interpolation, source_coordinate - (float)index);
        indices[offset] = index >= 0 && index < extent ? index : -1;
        weights[offset] = weight;
        weight_sum += weight;
    }
    const float inverse_weight_sum = weight_sum != 0.0f ? 1.0f / weight_sum : 0.0f;
    #pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
        if (offset < sample_count) {
            weights[offset] *= inverse_weight_sum;
        }
    }
    return sample_count;
}

__device__ float pixtreme_sample(
    const float* __restrict__ foreground,
    const long long foreground_width,
    const long long foreground_channel_count,
    const int channel,
    const int alpha_channel,
    const int associate,
    const long long x_indices[8],
    const float x_weights[8],
    const int x_count,
    const long long y_indices[8],
    const float y_weights[8],
    const int y_count
) {
    float result = 0.0f;
    #pragma unroll
    for (int y_offset = 0; y_offset < 8; ++y_offset) {
        if (y_offset >= y_count || y_indices[y_offset] < 0) {
            continue;
        }
        #pragma unroll
        for (int x_offset = 0; x_offset < 8; ++x_offset) {
            if (x_offset >= x_count || x_indices[x_offset] < 0) {
                continue;
            }
            const long long foreground_offset =
                (y_indices[y_offset] * foreground_width + x_indices[x_offset]) * foreground_channel_count;
            float sample = channel < 0 ? 1.0f : foreground[foreground_offset + channel];
            if (associate && channel >= 0 && alpha_channel >= 0) {
                sample *= foreground[foreground_offset + alpha_channel];
            }
            result += sample * x_weights[x_offset] * y_weights[y_offset];
        }
    }
    return result;
}

extern "C" __global__ void pixtreme_composite_images(
    const float* __restrict__ background,
    const float* __restrict__ foreground,
    const float* __restrict__ mask,
    const int* __restrict__ foreground_indices,
    float* __restrict__ output,
    const long long background_width,
    const long long background_height,
    const long long background_channel_count,
    const long long foreground_width,
    const long long foreground_height,
    const long long foreground_channel_count,
    const int background_alpha_index,
    const int foreground_alpha_index,
    const int has_mask,
    const int alpha_mode,
    const int interpolation,
    const int blend,
    const float opacity,
    const float position_x,
    const float position_y,
    const float anchor_x,
    const float anchor_y,
    const float inverse_scale_x,
    const float inverse_scale_y,
    const float cosine,
    const float sine
) {
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= background_width || y >= background_height) {
        return;
    }

    const float dx = (float)x + 0.5f - position_x;
    const float dy = (float)y + 0.5f - position_y;
    const float source_x = anchor_x + (cosine * dx - sine * dy) * inverse_scale_x;
    const float source_y = anchor_y + (sine * dx + cosine * dy) * inverse_scale_y;
    long long x_indices[8];
    long long y_indices[8];
    float x_weights[8];
    float y_weights[8];
    const int x_count = pixtreme_axis_plan(
        source_x,
        foreground_width,
        interpolation,
        x_indices,
        x_weights
    );
    const int y_count = pixtreme_axis_plan(
        source_y,
        foreground_height,
        interpolation,
        y_indices,
        y_weights
    );
    const float source_alpha = pixtreme_sample(
        foreground,
        foreground_width,
        foreground_channel_count,
        foreground_alpha_index,
        foreground_alpha_index,
        0,
        x_indices,
        x_weights,
        x_count,
        y_indices,
        y_weights,
        y_count
    );
    const long long background_offset = (y * background_width + x) * background_channel_count;
    const float background_alpha =
        background_alpha_index >= 0 ? background[background_offset + background_alpha_index] : 1.0f;
    const float mask_value = has_mask ? mask[y * background_width + x] : 1.0f;
    const float effective_alpha = source_alpha * mask_value * opacity;
    const float output_alpha =
        effective_alpha + background_alpha * (1.0f - effective_alpha);

    for (long long channel = 0; channel < background_channel_count; ++channel) {
        if (channel == background_alpha_index) {
            continue;
        }
        const float source_premultiplied = pixtreme_sample(
            foreground,
            foreground_width,
            foreground_channel_count,
            foreground_indices[channel],
            foreground_alpha_index,
            alpha_mode == 1,
            x_indices,
            x_weights,
            x_count,
            y_indices,
            y_weights,
            y_count
        );
        const float source_color = source_alpha != 0.0f ? source_premultiplied / source_alpha : 0.0f;
        const float stored_background = background[background_offset + channel];
        const float background_color =
            alpha_mode == 0 && background_alpha_index >= 0
                ? (background_alpha != 0.0f ? stored_background / background_alpha : 0.0f)
                : stored_background;
        const float blend_value = pixtreme_blend(background_color, source_color, blend);
        const float composite_source =
            (1.0f - background_alpha) * source_color + background_alpha * blend_value;
        const float output_premultiplied =
            effective_alpha * composite_source
            + background_alpha * (1.0f - effective_alpha) * background_color;
        output[background_offset + channel] =
            alpha_mode == 1 && background_alpha_index >= 0
                ? (output_alpha != 0.0f ? output_premultiplied / output_alpha : 0.0f)
                : output_premultiplied;
    }
    if (background_alpha_index >= 0) {
        output[background_offset + background_alpha_index] = output_alpha;
    }
}
"""
)


@lru_cache(maxsize=1)
def _composite_kernel() -> cp.RawKernel:
    return cp.RawKernel(_COMPOSITE_KERNEL_SOURCE, "pixtreme_composite_images")


def _require_float32(frame: Frame, *, name: str) -> None:
    dtype = np.dtype(frame.dtype)
    if dtype != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why=f"merge requires float32 {name} data",
                what=f"received {name} dtype {dtype.name!r}",
                how=_float32_conversion_guidance(dtype),
            )
        )


def _color_channels(frame: Frame) -> tuple[str, ...]:
    return tuple(label for label in frame.channels if label != "A")


def _channel_indices(source: tuple[str, ...], target: tuple[str, ...]) -> tuple[int, ...] | None:
    remaining = list(enumerate(source))
    result: list[int] = []
    for label in target:
        match = next((index for index, (_, candidate) in enumerate(remaining) if candidate == label), None)
        if match is None:
            return None
        source_index, _ = remaining.pop(match)
        result.append(source_index)
    return tuple(result)


def _compatibility_error(
    *,
    field: str,
    background_value: object,
    foreground_value: object,
    adapt: bool,
) -> ValueError:
    mode = "adapt=True cannot deterministically rescue" if adapt else "adapt=False requires matching"
    return ValueError(
        _actionable_error(
            why=f"merge {mode} {field}",
            what=f"background {field}={background_value!r}, foreground {field}={foreground_value!r}",
            how="make the inputs compatible before composition or choose a supported adapt=True conversion",
        )
    )


def _associate(frame: Frame, *, inverse: bool) -> Frame:
    if "A" not in frame.channels:
        return frame
    alpha_index = frame.channels.index("A")
    alpha = frame.data[..., alpha_index]
    output = frame.data.copy()
    for channel_index, label in enumerate(frame.channels):
        if label == "A":
            continue
        if inverse:
            output[..., channel_index] = cp.where(
                alpha != np.float32(0.0),
                frame.data[..., channel_index] / alpha,
                np.float32(0.0),
            )
        else:
            output[..., channel_index] *= alpha
    return Frame(
        data=output,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
        matrix=frame.matrix,
    )


def _adapt_foreground(background: Frame, foreground: Frame, *, alpha: str, adapt: bool) -> Frame:
    background_colors = _color_channels(background)
    foreground_colors = _color_channels(foreground)
    background_set = frozenset(background_colors)
    foreground_set = frozenset(foreground_colors)
    target_channels = (*background_colors, *(("A",) if "A" in foreground.channels else ()))

    if not adapt:
        if background_set != foreground_set:
            raise _compatibility_error(
                field="channels",
                background_value=background_set,
                foreground_value=foreground_set,
                adapt=False,
            )
        if _channel_indices(foreground.channels, target_channels) is None:
            raise _compatibility_error(
                field="channels",
                background_value=background.channels,
                foreground_value=foreground.channels,
                adapt=False,
            )
        if foreground.colorspace != background.colorspace:
            raise _compatibility_error(
                field="colorspace",
                background_value=background.colorspace,
                foreground_value=foreground.colorspace,
                adapt=False,
            )
        if foreground.gamma != background.gamma:
            raise _compatibility_error(
                field="gamma",
                background_value=background.gamma,
                foreground_value=foreground.gamma,
                adapt=False,
            )
        try:
            return _route_frame(foreground, target_channels)
        except ValueError as error:
            raise _compatibility_error(
                field="channels",
                background_value=background.channels,
                foreground_value=foreground.channels,
                adapt=False,
            ) from error

    same_colorimetry = foreground.colorspace == background.colorspace and foreground.gamma == background.gamma
    if background_set == foreground_set and same_colorimetry:
        try:
            return _route_frame(foreground, target_channels)
        except ValueError as error:
            raise _compatibility_error(
                field="channels",
                background_value=background.channels,
                foreground_value=foreground.channels,
                adapt=True,
            ) from error

    supported_sets = {frozenset(_RGB_CHANNELS), frozenset(_YCBCR_CHANNELS)}
    if (
        foreground_set not in supported_sets
        or background_set not in supported_sets
        or len(foreground_colors) != 3
        or len(background_colors) != 3
    ):
        raise _compatibility_error(
            field="channels",
            background_value=background_set,
            foreground_value=foreground_set,
            adapt=True,
        )

    current = _associate(foreground, inverse=True) if alpha == "premultiplied" else foreground
    alpha_suffix = ("A",) if "A" in current.channels else ()
    try:
        if foreground_set == frozenset(_YCBCR_CHANNELS):
            current = ycbcr_to_rgb(current)
        current = rgb_to_rgb(
            current,
            output_colorspace=background.colorspace,
            output_gamma=background.gamma,
        )
        if background_set == frozenset(_YCBCR_CHANNELS):
            if background.matrix is None and background.colorspace not in {"sRGB", "Rec.709", "Rec.2020"}:
                raise ValueError(
                    _actionable_error(
                        why="target YCbCr Frame has no deterministic matrix provenance",
                        what=f"target colorspace={background.colorspace!r}, matrix={background.matrix!r}",
                        how=(
                            "set target Frame matrix metadata or use a YCbCr-defining colorspace: "
                            "'sRGB', 'Rec.709', or 'Rec.2020'"
                        ),
                    )
                )
            current = rgb_to_ycbcr(current, matrix=background.matrix)
        current = _route_frame(current, (*background_colors, *alpha_suffix))
    except ValueError as error:
        raise ValueError(
            _actionable_error(
                why="merge adapt=True found no complete deterministic channel/color conversion",
                what=(
                    f"foreground channels={foreground.channels!r}, colorspace={foreground.colorspace!r}, "
                    f"gamma={foreground.gamma!r}; background channels={background.channels!r}, "
                    f"colorspace={background.colorspace!r}, gamma={background.gamma!r}"
                ),
                how="use RGB or colorspace-derived YCbCr channels with a transformable colorspace and gamma",
            )
        ) from error
    return _associate(current, inverse=False) if alpha == "premultiplied" else current


def _validate_mask(mask: object | None, *, background: Frame) -> Frame | None:
    if mask is None:
        return None
    if not isinstance(mask, Frame):
        raise ValueError(
            _actionable_error(
                why="merge mask must be a metadata-bearing Frame",
                what=f"received {type(mask).__module__}.{type(mask).__qualname__}",
                how="pass mask=None or a same-size one-channel float32 Frame",
            )
        )
    if mask.height != background.height or mask.width != background.width or len(mask.channels) != 1:
        raise ValueError(
            _actionable_error(
                why="merge mask must have background geometry and exactly one channel",
                what=f"background shape={background.shape!r}, mask shape={mask.shape!r}",
                how="provide a one-channel mask with the background height and width",
            )
        )
    _require_float32(mask, name="mask")
    return mask


def merge(
    background: Frame,
    foreground: Frame,
    *,
    blend: Blend = "normal",
    opacity: float = 1.0,
    mask: Frame | None = None,
    alpha: Alpha = "premultiplied",
    position: tuple[float, float] | None = None,
    scale: float | tuple[float, float] = 1.0,
    rotation: float = 0.0,
    interpolation: Interpolation = "bilinear",
    adapt: bool = False,
) -> Frame:
    """Transform and composite a foreground Frame over a background Frame.

    The background fixes output geometry, channels, colorspace, and gamma.
    ``position`` places the foreground center in background continuous
    coordinates; ``scale`` applies per axis before ``rotation`` in degrees.
    ``interpolation`` samples transparent zero beyond foreground edges.

    ``alpha`` declares both inputs and the output as ``straight`` or
    ``premultiplied`` when an A channel exists. Foreground A, ``mask``, and
    ``opacity`` determine source coverage. Blend and source-over arithmetic are
    float32 and never clamp scene values. ``adapt=True`` uses deterministic
    RGB/YCbCr channel conversion plus colorspace/gamma transformation; dtype is
    never adapted. Inputs must be float32 Frames. The result always has new
    storage and never shares its allocation with either input.
    """
    checked_background = _validate_float32_frame(background, operation="composite.merge")
    checked_foreground = _validate_float32_frame(foreground, operation="composite.merge")
    checked_adapt = _strict_bool(
        adapt,
        name="adapt",
        why="merge adapt is an explicit boolean choice",
        how="pass adapt=False or adapt=True",
    )
    checked_blend = _closed_str_token(blend, axis="blend", accepted=_BLEND_TOKENS)
    checked_opacity = _bounded_real(
        opacity,
        name="opacity",
        minimum=0.0,
        maximum=1.0,
        why="opacity must be in the closed interval from 0 through 1",
        how="pass a finite real opacity from 0.0 through 1.0",
    )
    checked_alpha = _closed_str_token(alpha, axis="alpha", accepted=_ALPHA_TOKENS)
    checked_interpolation = _closed_str_token(
        interpolation,
        axis="interpolation",
        accepted=_COMPOSITE_INTERPOLATION_TOKENS,
    )
    checked_position = (
        (checked_background.width / 2.0, checked_background.height / 2.0)
        if position is None
        else _finite_pair(position, name="position")
    )
    scale_x, scale_y = _positive_scalar_or_pair(
        scale,
        name="scale",
        why="scale must be one positive real number or one (sx, sy) pair",
        how="pass a positive finite scalar or a two-element positive finite sequence",
    )
    checked_rotation = _finite_real(rotation, name="rotation")
    checked_mask = _validate_mask(mask, background=checked_background)
    prepared_foreground = _adapt_foreground(
        checked_background,
        checked_foreground,
        alpha=checked_alpha,
        adapt=checked_adapt,
    )

    output = checked_background.data.copy(order="C")
    if checked_opacity == 0.0 or checked_background.width == 0 or checked_background.height == 0:
        return Frame(
            data=output,
            colorspace=checked_background.colorspace,
            gamma=checked_background.gamma,
            channels=checked_background.channels,
            matrix=checked_background.matrix,
        )

    background_alpha_index = checked_background.channels.index("A") if "A" in checked_background.channels else -1
    foreground_alpha_index = prepared_foreground.channels.index("A") if "A" in prepared_foreground.channels else -1
    foreground_mapping = np.full(len(checked_background.channels), -1, dtype=np.int32)
    background_colors = _color_channels(checked_background)
    foreground_colors = _color_channels(prepared_foreground)
    color_indices = _channel_indices(foreground_colors, background_colors)
    if color_indices is None:
        raise _compatibility_error(
            field="channels",
            background_value=checked_background.channels,
            foreground_value=prepared_foreground.channels,
            adapt=adapt,
        )
    prepared_color_positions = tuple(index for index, label in enumerate(prepared_foreground.channels) if label != "A")
    for background_position, prepared_color_index in zip(
        (index for index, label in enumerate(checked_background.channels) if label != "A"),
        color_indices,
        strict=True,
    ):
        foreground_mapping[background_position] = prepared_color_positions[prepared_color_index]
    device_mapping = cp.asarray(foreground_mapping)
    radians = math.radians(checked_rotation)
    block_x, block_y = _COMPOSITE_BLOCK
    grid = (
        (checked_background.width + block_x - 1) // block_x,
        (checked_background.height + block_y - 1) // block_y,
    )
    mask_data = checked_background.data if checked_mask is None else checked_mask.data
    _composite_kernel()(
        grid,
        _COMPOSITE_BLOCK,
        (
            checked_background.data,
            prepared_foreground.data,
            mask_data,
            device_mapping,
            output,
            np.int64(checked_background.width),
            np.int64(checked_background.height),
            np.int64(len(checked_background.channels)),
            np.int64(prepared_foreground.width),
            np.int64(prepared_foreground.height),
            np.int64(len(prepared_foreground.channels)),
            np.int32(background_alpha_index),
            np.int32(foreground_alpha_index),
            np.int32(checked_mask is not None),
            np.int32(_ALPHA_TOKENS.index(checked_alpha)),
            np.int32(_COMPOSITE_INTERPOLATION_TOKENS.index(checked_interpolation)),
            np.int32(_BLEND_CODES[checked_blend]),
            np.float32(checked_opacity),
            np.float32(checked_position[0]),
            np.float32(checked_position[1]),
            np.float32(prepared_foreground.width / 2.0),
            np.float32(prepared_foreground.height / 2.0),
            np.float32(1.0 / scale_x),
            np.float32(1.0 / scale_y),
            np.float32(math.cos(radians)),
            np.float32(math.sin(radians)),
        ),
    )
    return Frame(
        data=output,
        colorspace=checked_background.colorspace,
        gamma=checked_background.gamma,
        channels=checked_background.channels,
        matrix=checked_background.matrix,
    )
