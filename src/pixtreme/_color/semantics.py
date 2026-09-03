"""Fused declarative RGB, YCbCr, grayscale, and transfer operations."""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

import cupy as cp
import numpy as np

from pixtreme._color.transform import (
    _COLOR_TRANSFORM_KERNEL,
    _GAMMA_CODES,
    _RGB_TO_XYZ,
    _compose_matrix,
)
from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    _COLORSPACE_TOKENS,
    _GAMMA_TOKENS,
    _MATRIX_TOKENS,
    Frame,
)
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_domain import _BIT_DEPTHS, _float32_conversion_guidance, _legal_parameters
from pixtreme._core.vocabulary import _RANGE_TOKENS, Colorspace, Gamma, Matrix, Range

_RGB_CHANNELS = ("R", "G", "B")
_YCBCR_CHANNELS = ("Y", "Cb", "Cr")
_INPUT_RGB = 0
_INPUT_YCBCR = 1
_OUTPUT_RGB = 0
_OUTPUT_YCBCR = 1
_OUTPUT_GRAYSCALE = 2

_STANDARD_MATRIX_COEFFICIENTS = {
    "BT.601": (np.float32(0.299), np.float32(0.114)),
    "BT.709": (np.float32(0.2126), np.float32(0.0722)),
    "BT.2020": (np.float32(0.2627), np.float32(0.0593)),
}

_COLOR_SEMANTICS_KERNEL = (
    _COLOR_TRANSFORM_KERNEL.partition('extern "C" __global__')[0]
    + r"""
extern "C" __global__
void color_semantics_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const int input_channel_count,
    const int output_channel_count,
    const int first_index,
    const int second_index,
    const int third_index,
    const int input_mode,
    const int output_mode,
    const int apply_technical_transform,
    const int input_gamma,
    const int output_gamma,
    const float input_kr,
    const float input_kb,
    const float output_kr,
    const float output_kb,
    const int input_legal,
    const float input_lower,
    const float input_luma_extent,
    const float input_chroma_extent,
    const int output_legal,
    const float output_lower,
    const float output_luma_extent,
    const float output_chroma_extent,
    const float m00,
    const float m01,
    const float m02,
    const float m10,
    const float m11,
    const float m12,
    const float m20,
    const float m21,
    const float m22
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }

    const long long input_base = pixel * input_channel_count;
    float first = input[input_base + first_index];
    float second = input[input_base + second_index];
    float third = input[input_base + third_index];
    float encoded_red;
    float encoded_green;
    float encoded_blue;

    if (input_mode == 1) {
        if (input_legal != 0) {
            first = (first - input_lower) / input_luma_extent;
            second = (second - input_lower) / input_chroma_extent;
            third = (third - input_lower) / input_chroma_extent;
        }
        const float input_kg = 1.0f - input_kr - input_kb;
        encoded_red = first + 2.0f * (1.0f - input_kr) * (third - 0.5f);
        encoded_blue = first + 2.0f * (1.0f - input_kb) * (second - 0.5f);
        encoded_green = (first - input_kr * encoded_red - input_kb * encoded_blue) / input_kg;
    } else {
        encoded_red = first;
        encoded_green = second;
        encoded_blue = third;
    }

    float output_red = encoded_red;
    float output_green = encoded_green;
    float output_blue = encoded_blue;
    if (apply_technical_transform != 0) {
        const float linear_red = decode_transfer(encoded_red, input_gamma);
        const float linear_green = decode_transfer(encoded_green, input_gamma);
        const float linear_blue = decode_transfer(encoded_blue, input_gamma);
        output_red = encode_transfer(m00 * linear_red + m01 * linear_green + m02 * linear_blue, output_gamma);
        output_green = encode_transfer(m10 * linear_red + m11 * linear_green + m12 * linear_blue, output_gamma);
        output_blue = encode_transfer(m20 * linear_red + m21 * linear_green + m22 * linear_blue, output_gamma);
    }

    float output_first = output_red;
    float output_second = output_green;
    float output_third = output_blue;
    if (output_mode != 0) {
        const float output_kg = 1.0f - output_kr - output_kb;
        output_first = output_kr * output_red + output_kg * output_green + output_kb * output_blue;
        output_second = (output_blue - output_first) / (2.0f * (1.0f - output_kb)) + 0.5f;
        output_third = (output_red - output_first) / (2.0f * (1.0f - output_kr)) + 0.5f;
        if (output_legal != 0) {
            output_first = output_first * output_luma_extent + output_lower;
            output_second = output_second * output_chroma_extent + output_lower;
            output_third = output_third * output_chroma_extent + output_lower;
        }
    }

    if (output_mode == 2) {
        output[pixel] = output_first;
        return;
    }

    const long long output_base = pixel * output_channel_count;
    for (int channel = 0; channel < input_channel_count; ++channel) {
        output[output_base + channel] = input[input_base + channel];
    }
    output[output_base + first_index] = output_first;
    output[output_base + second_index] = output_second;
    output[output_base + third_index] = output_third;
}
"""
)


@lru_cache(maxsize=1)
def _color_semantics_kernel() -> cp.RawKernel:
    return cp.RawKernel(_COLOR_SEMANTICS_KERNEL, "color_semantics_kernel")


def _error(*, why: str, what: str, how: str) -> ValueError:
    return ValueError(_actionable_error(why=why, what=what, how=how))


def _validate_matrix(value: str | None, *, parameter: str) -> Matrix | None:
    return (
        None
        if value is None
        else _normalized_closed_token(
            value,
            axis=parameter,
            accepted=_MATRIX_TOKENS,
            why=f"{parameter} accepts only documented matrix bases",
            how=f"use one of the canonical tokens {_MATRIX_TOKENS!r} or None",
        )
    )


_Token = TypeVar("_Token", bound=str)


def _validate_axis(value: str | None, *, parameter: str, accepted: tuple[_Token, ...]) -> _Token | None:
    return None if value is None else _normalized_closed_token(value, axis=parameter, accepted=accepted)


def _validate_range_and_bit_depth(range_value: str, bit_depth: int, *, operation: str) -> tuple[Range, int]:
    range_value = _normalized_closed_token(range_value, axis="range", accepted=_RANGE_TOKENS)
    if type(bit_depth) is not int or bit_depth not in _BIT_DEPTHS:
        raise _error(
            why=f"{operation} bit_depth selects an H.273 code grid",
            what=f"received bit_depth={bit_depth!r}",
            how=f"use an integer from {_BIT_DEPTHS!r}",
        )
    return range_value, bit_depth


def _validate_color_frame(
    frame: Frame,
    *,
    operation: str,
    required: tuple[str, str, str],
    forbidden: tuple[str, ...],
) -> Frame:
    if not isinstance(frame, Frame):
        raise _error(
            why=f"{operation} is a Frame-to-Frame color operation",
            what=f"received {type(frame).__module__}.{type(frame).__qualname__}",
            how="construct a float32 px.core.Frame first",
        )
    if np.dtype(frame.data.dtype) != np.dtype(np.float32):
        raise _error(
            why=f"{operation} evaluates color formulae in float32",
            what=f"received dtype={frame.data.dtype!s}",
            how=_float32_conversion_guidance(np.dtype(frame.data.dtype)),
        )
    missing = tuple(label for label in required if frame.channels.count(label) != 1)
    conflicting = tuple(label for label in forbidden if label in frame.channels)
    if missing or conflicting:
        raise _error(
            why=f"{operation} requires exactly one {required!r} triplet and no destination color labels",
            what=f"received channels={frame.channels!r}; missing-or-duplicated={missing!r}; conflicting={conflicting!r}",
            how=f"use a Frame with one each of {required!r} and move channel structure separately",
        )
    return frame


def _validate_rgb_frame(frame: Frame, *, operation: str) -> Frame:
    return _validate_color_frame(
        frame,
        operation=operation,
        required=_RGB_CHANNELS,
        forbidden=_YCBCR_CHANNELS,
    )


def _validate_rgb_transfer_frame(frame: Frame, *, operation: str) -> Frame:
    return _validate_color_frame(frame, operation=operation, required=_RGB_CHANNELS, forbidden=())


def _validate_ycbcr_frame(frame: Frame, *, operation: str) -> Frame:
    return _validate_color_frame(
        frame,
        operation=operation,
        required=_YCBCR_CHANNELS,
        forbidden=_RGB_CHANNELS,
    )


def _replace_triplet(
    channels: tuple[str, ...],
    source: tuple[str, str, str],
    destination: tuple[str, str, str],
) -> tuple[str, ...]:
    replacements = dict(zip(source, destination, strict=True))
    return tuple(replacements.get(label, label) for label in channels)


def _matrix_coefficients(matrix: Matrix, *, colorspace: Colorspace) -> tuple[np.float32, np.float32]:
    if matrix != "native":
        return _STANDARD_MATRIX_COEFFICIENTS[matrix]
    own_row = _RGB_TO_XYZ[colorspace][1]
    return np.float32(own_row[0]), np.float32(own_row[2])


def _resolve_encode_matrix(matrix: str | None, *, colorspace: Colorspace, gamma: Gamma) -> Matrix:
    validated = _validate_matrix(matrix, parameter="matrix")
    if validated is not None:
        return validated
    if colorspace in {"sRGB", "Rec.709"}:
        return "BT.709"
    if colorspace == "Rec.2020":
        return "BT.2020"
    return "native" if gamma == "linear" else "BT.709"


def _resolve_decode_matrix(matrix: str | None, *, frame: Frame, parameter: str = "matrix") -> Matrix:
    validated = _validate_matrix(matrix, parameter=parameter)
    if validated is not None:
        return validated
    if frame.matrix is not None:
        return frame.matrix
    if frame.colorspace in {"sRGB", "Rec.709"}:
        return "BT.709"
    if frame.colorspace == "Rec.2020":
        return "BT.2020"
    raise _error(
        why=f"{parameter} cannot be inferred from colorspace {frame.colorspace!r}",
        what="the Frame has no matrix provenance",
        how=f"pass {parameter}= explicitly; for camera material BT.709 is the most common practical starting point",
    )


def _technical_transform_required(
    input_colorspace: str,
    input_gamma: str,
    output_colorspace: str,
    output_gamma: str,
) -> bool:
    return not (
        _COLORSPACE_DEFINITIONS[input_colorspace] == _COLORSPACE_DEFINITIONS[output_colorspace]
        and input_gamma == output_gamma
    )


def _run_transform(
    frame: Frame,
    *,
    input_mode: int,
    output_mode: int,
    input_colorspace: Colorspace,
    input_gamma: Gamma,
    output_colorspace: Colorspace,
    output_gamma: Gamma,
    input_matrix: Matrix | None = None,
    output_matrix: Matrix | None = None,
    input_range: Range = "full",
    input_bit_depth: int = 8,
    output_range: Range = "full",
    output_bit_depth: int = 8,
) -> cp.ndarray:
    output_shape = (frame.height, frame.width, 1 if output_mode == _OUTPUT_GRAYSCALE else len(frame.channels))
    output = cp.empty(output_shape, dtype=cp.float32)
    pixel_count = frame.width * frame.height
    if pixel_count == 0:
        return output

    source_labels = _RGB_CHANNELS if input_mode == _INPUT_RGB else _YCBCR_CHANNELS
    first_index, second_index, third_index = (frame.channels.index(label) for label in source_labels)
    input_kr, input_kb = (
        _matrix_coefficients(input_matrix, colorspace=input_colorspace)
        if input_matrix is not None
        else (np.float32(0.0), np.float32(0.0))
    )
    output_kr, output_kb = (
        _matrix_coefficients(output_matrix, colorspace=output_colorspace)
        if output_matrix is not None
        else (np.float32(0.0), np.float32(0.0))
    )
    input_lower, input_luma_extent, input_chroma_extent = _legal_parameters(input_bit_depth)
    output_lower, output_luma_extent, output_chroma_extent = _legal_parameters(output_bit_depth)
    matrix = _compose_matrix(input_colorspace, output_colorspace)
    flat_matrix = matrix.reshape(9)
    block_size = 256
    block_count = (pixel_count + block_size - 1) // block_size
    _color_semantics_kernel()(
        (block_count,),
        (block_size,),
        (
            frame.data,
            output,
            np.int64(pixel_count),
            np.int32(len(frame.channels)),
            np.int32(output_shape[2]),
            np.int32(first_index),
            np.int32(second_index),
            np.int32(third_index),
            np.int32(input_mode),
            np.int32(output_mode),
            np.int32(
                _technical_transform_required(
                    input_colorspace,
                    input_gamma,
                    output_colorspace,
                    output_gamma,
                )
            ),
            np.int32(_GAMMA_CODES[input_gamma]),
            np.int32(_GAMMA_CODES[output_gamma]),
            input_kr,
            input_kb,
            output_kr,
            output_kb,
            np.int32(input_range == "legal"),
            input_lower,
            input_luma_extent,
            input_chroma_extent,
            np.int32(output_range == "legal"),
            output_lower,
            output_luma_extent,
            output_chroma_extent,
            *(np.float32(value) for value in flat_matrix),
        ),
    )
    return output


def rgb_to_ycbcr(
    frame: Frame,
    *,
    colorspace: Colorspace | None = None,
    gamma: Gamma | None = None,
    matrix: Matrix | None = None,
    range: Range = "full",
    bit_depth: int = 8,
) -> Frame:
    """Encode RGB as full- or legal-range YCbCr in one fused GPU pass.

    ``frame`` must be a float32 Frame containing exactly one R, G, and B and no
    YCbCr destination labels. ``colorspace`` and ``gamma`` declare the output
    representation; ``None`` inherits the corresponding Frame metadata. They use
    the same case- and separator-insensitive closed vocabularies as Frame, including ``sRGB`` /
    ``Rec.709`` / ``Rec.2020`` / ACES and S-Gamut colorspaces and ``linear`` /
    ``sRGB`` / ``Rec.709`` / ``BT.1886`` / ``PQ`` / ``HLG`` / ``S-Log`` / ``S-Log2`` /
    ``S-Log3`` / ``ARRI-LogC3`` / ``ARRI-LogC4`` / ``Blackmagic-Film-Gen-5`` / ``DaVinci-Intermediate`` /
    ``RED-Log3G10`` / ``REDlogFilm`` / ``Cineon`` / ``Gamma-2.2`` / ``Gamma-2.4`` / ``Gamma-2.6`` transfers.

    ``matrix`` accepts ``"BT.601"``, ``"BT.709"``, ``"BT.2020"``, or ``"native"``.
    When omitted, the target representation resolves it to BT.709, BT.2020, native,
    or the documented non-linear fallback. ``range`` is ``"full"`` or ``"legal"``;
    ``bit_depth`` is 8, 10, 12, 14, or 16 and affects only legal-range scaling.

    The result replaces R/G/B labels in place with Y/Cb/Cr, stamps the declared
    representation and resolved matrix, and preserves auxiliary channels bit for
    bit. Technical conversion, matrix encoding, and range mapping are fused; no
    scene values are clipped. Invalid frame types, dtypes, labels, tokens, ranges,
    or bit depths raise :class:`ValueError`.
    """
    frame = _validate_rgb_frame(frame, operation="rgb_to_ycbcr")
    colorspace = _validate_axis(colorspace, parameter="colorspace", accepted=_COLORSPACE_TOKENS)
    gamma = _validate_axis(gamma, parameter="gamma", accepted=_GAMMA_TOKENS)
    range, bit_depth = _validate_range_and_bit_depth(range, bit_depth, operation="rgb_to_ycbcr")
    output_colorspace = frame.colorspace if colorspace is None else colorspace
    output_gamma = frame.gamma if gamma is None else gamma
    resolved_matrix = _resolve_encode_matrix(matrix, colorspace=output_colorspace, gamma=output_gamma)
    data = _run_transform(
        frame,
        input_mode=_INPUT_RGB,
        output_mode=_OUTPUT_YCBCR,
        input_colorspace=frame.colorspace,
        input_gamma=frame.gamma,
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        output_matrix=resolved_matrix,
        output_range=range,
        output_bit_depth=bit_depth,
    )
    return Frame(
        data=data,
        colorspace=output_colorspace,
        gamma=output_gamma,
        channels=_replace_triplet(frame.channels, _RGB_CHANNELS, _YCBCR_CHANNELS),
        matrix=resolved_matrix,
    )


def ycbcr_to_rgb(
    frame: Frame,
    *,
    colorspace: Colorspace | None = None,
    gamma: Gamma | None = None,
    matrix: Matrix | None = None,
    range: Range = "full",
    bit_depth: int = 8,
) -> Frame:
    """Decode full- or legal-range YCbCr to a declared RGB representation.

    ``frame`` must be a float32 Frame containing exactly one Y, Cb, and Cr and no
    RGB destination labels. ``range`` accepts ``"full"`` or ``"legal"`` and
    ``bit_depth`` accepts 8, 10, 12, 14, or 16; legal code values are expanded
    before matrix decoding. ``matrix`` accepts ``"BT.601"``, ``"BT.709"``,
    ``"BT.2020"``, or ``"native"`` and resolves in order from the explicit value,
    ``frame.matrix``, or the sRGB/Rec.709/Rec.2020 convention. Other colorspaces
    require an explicit matrix when provenance is absent.

    ``colorspace`` and ``gamma`` declare the output RGB representation, with
    ``None`` inheriting Frame metadata. They accept the Frame colorspace vocabulary
    and the ``linear``, ``sRGB``, ``Rec.709``, ``BT.1886``, ``PQ``, ``HLG``,
    ``S-Log``, ``S-Log2``, ``S-Log3``, ``ARRI-LogC3``, ``ARRI-LogC4``, ``Blackmagic-Film-Gen-5``,
    ``DaVinci-Intermediate``, ``RED-Log3G10``, ``REDlogFilm``, ``Cineon``, ``Gamma-2.2``, ``Gamma-2.4``, and
    ``Gamma-2.6`` transfer tokens.

    The result replaces Y/Cb/Cr labels in place with R/G/B, preserves auxiliary
    channels bit for bit, stamps the declared colorspace and gamma, and clears
    matrix metadata. Range expansion, matrix decoding, and technical conversion
    are fused on the GPU without clipping. Invalid frame types, dtypes, labels,
    tokens, ranges, bit depths, or unresolvable matrix provenance raise
    :class:`ValueError`.
    """
    frame = _validate_ycbcr_frame(frame, operation="ycbcr_to_rgb")
    colorspace = _validate_axis(colorspace, parameter="colorspace", accepted=_COLORSPACE_TOKENS)
    gamma = _validate_axis(gamma, parameter="gamma", accepted=_GAMMA_TOKENS)
    range, bit_depth = _validate_range_and_bit_depth(range, bit_depth, operation="ycbcr_to_rgb")
    input_matrix = _resolve_decode_matrix(matrix, frame=frame)
    output_colorspace = frame.colorspace if colorspace is None else colorspace
    output_gamma = frame.gamma if gamma is None else gamma
    data = _run_transform(
        frame,
        input_mode=_INPUT_YCBCR,
        output_mode=_OUTPUT_RGB,
        input_colorspace=frame.colorspace,
        input_gamma=frame.gamma,
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        input_matrix=input_matrix,
        input_range=range,
        input_bit_depth=bit_depth,
    )
    return Frame(
        data=data,
        colorspace=output_colorspace,
        gamma=output_gamma,
        channels=_replace_triplet(frame.channels, _YCBCR_CHANNELS, _RGB_CHANNELS),
        matrix=None,
    )


def rgb_to_grayscale(
    frame: Frame,
    *,
    colorspace: Colorspace | None = None,
    gamma: Gamma | None = None,
    matrix: Matrix | None = None,
) -> Frame:
    """Project RGB to the Y channel of a declared full-range representation.

    ``frame`` must be a float32 Frame containing exactly one R, G, and B and no
    YCbCr destination labels. ``colorspace`` and ``gamma`` declare the projection
    representation; ``None`` inherits Frame metadata. They use the closed Frame
    colorspace vocabulary and the ``linear``, ``sRGB``, ``Rec.709``, ``BT.1886``,
    ``PQ``, ``HLG``, ``S-Log``, ``S-Log2``, ``S-Log3``, ``ARRI-LogC3``, ``ARRI-LogC4``, ``Blackmagic-Film-Gen-5``,
    ``DaVinci-Intermediate``, ``RED-Log3G10``, ``REDlogFilm``, ``Cineon``, ``Gamma-2.2``, ``Gamma-2.4``, and
    ``Gamma-2.6`` gamma tokens. ``matrix`` accepts
    ``"BT.601"``, ``"BT.709"``, ``"BT.2020"``, or
    ``"native"`` and otherwise resolves from the declared representation.

    The GPU result is a new C-contiguous float32 Frame with only the ``("Y",)``
    channel; auxiliary channels are intentionally omitted. It stamps the declared
    colorspace and gamma plus the resolved matrix. Linear output represents
    luminance and non-linear output represents luma. The projection matches the Y
    channel of :func:`rgb_to_ycbcr` with full range and does not clip scene values.
    Invalid frame types, dtypes, labels, or tokens raise :class:`ValueError`.
    """
    frame = _validate_rgb_frame(frame, operation="rgb_to_grayscale")
    colorspace = _validate_axis(colorspace, parameter="colorspace", accepted=_COLORSPACE_TOKENS)
    gamma = _validate_axis(gamma, parameter="gamma", accepted=_GAMMA_TOKENS)
    output_colorspace = frame.colorspace if colorspace is None else colorspace
    output_gamma = frame.gamma if gamma is None else gamma
    resolved_matrix = _resolve_encode_matrix(matrix, colorspace=output_colorspace, gamma=output_gamma)
    data = _run_transform(
        frame,
        input_mode=_INPUT_RGB,
        output_mode=_OUTPUT_GRAYSCALE,
        input_colorspace=frame.colorspace,
        input_gamma=frame.gamma,
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        output_matrix=resolved_matrix,
    )
    return Frame(
        data=data,
        colorspace=output_colorspace,
        gamma=output_gamma,
        channels=("Y",),
        matrix=resolved_matrix,
    )


def gamma_to_linear(frame: Frame, *, gamma: Gamma | None = None) -> Frame:
    """Decode a claimed RGB transfer to scene-linear values on the GPU.

    ``frame`` must be a float32 Frame containing exactly one R, G, and B.
    ``gamma`` is an input metadata claim; ``None`` uses ``frame.gamma``. Explicit
    canonical values are ``linear``, ``sRGB``, ``Rec.709``, ``BT.1886``,
    ``PQ``, ``HLG``, ``S-Log``, ``S-Log2``, ``S-Log3``, ``ARRI-LogC3``, ``ARRI-LogC4``, ``Blackmagic-Film-Gen-5``,
    ``DaVinci-Intermediate``, ``RED-Log3G10``, ``REDlogFilm``, ``Cineon``, ``Gamma-2.2``, ``Gamma-2.4``, or
    ``Gamma-2.6`` tokens.
    The claim controls interpretation without mutating the input Frame metadata.

    Only R/G/B values are decoded. Auxiliary channels and channel order are
    preserved bit for bit, colorspace is inherited, output gamma is ``"linear"``,
    and matrix metadata is cleared. Negative and above-one scene values follow
    each transfer's documented extension and are not clipped. A new Frame and GPU
    allocation are returned even for a linear claim. Invalid frame types, dtypes,
    RGB labels, or gamma tokens raise :class:`ValueError`.

    S-Log / S-Log2 / S-Log3 apply their lower linear branches directly to signed inputs. For S-Log and S-Log2,
    public scene-linear reflectance uses x = r / 0.9 and Sony encoded IRE uses the public embedding
    e = (64 + 876 * y) / 1023. S-Log3 / ARRI-LogC4 do not use sign/magnitude mirroring; ARRI-LogC4 retains its negative
    scene cut. Established S-Log3 and ARRI-LogC4 results for nonnegative inputs remain float32 bit-identical.
    ARRI-LogC3 is the ARRI EI 800 relative scene-exposure curve, maps 18% gray to 400 / 1023, and extends its lower
    linear branch to negative values without clipping or sign/magnitude mirroring.
    Blackmagic Film Gen 5 uses its published natural-log branches. DaVinci Intermediate uses its published base-2
    branches and a derived decode threshold. Both apply their lower linear branches directly to negative values.
    RED-Log3G10 uses RED's published base-10 curve with a directly extended lower branch below scene-linear -0.01.
    REDlogFilm uses the Cineon sign-preserving mirror and exact float32 bits while preserving its own metadata.
    """
    frame = _validate_rgb_transfer_frame(frame, operation="gamma_to_linear")
    gamma = _validate_axis(gamma, parameter="gamma", accepted=_GAMMA_TOKENS)
    input_gamma = frame.gamma if gamma is None else gamma
    data = _run_transform(
        frame,
        input_mode=_INPUT_RGB,
        output_mode=_OUTPUT_RGB,
        input_colorspace=frame.colorspace,
        input_gamma=input_gamma,
        output_colorspace=frame.colorspace,
        output_gamma="linear",
    )
    return Frame(data=data, colorspace=frame.colorspace, gamma="linear", channels=frame.channels, matrix=None)


def linear_to_gamma(frame: Frame, *, gamma: Gamma) -> Frame:
    """Encode scene-linear RGB with an explicit transfer on the GPU.

    ``frame`` must be a float32 Frame with ``frame.gamma == "linear"`` and exactly
    one R, G, and B. ``gamma`` is a required normalized output token:
    ``linear``, ``sRGB``, ``Rec.709``, ``BT.1886``, ``PQ``, ``HLG``, ``S-Log``, ``S-Log2``, ``S-Log3``,
    ``ARRI-LogC3``, ``ARRI-LogC4``, ``Blackmagic-Film-Gen-5``, ``DaVinci-Intermediate``, ``RED-Log3G10``,
    ``REDlogFilm``, ``Cineon``, ``Gamma-2.2``, ``Gamma-2.4``, or ``Gamma-2.6``. Passing ``None`` or an unknown token
    is rejected rather than inferred.

    Only R/G/B values are encoded. Auxiliary channels and channel order are
    preserved bit for bit, colorspace is inherited, output gamma is the requested
    token, and matrix metadata is cleared. Negative and above-one scene values
    follow each transfer's documented extension and are not clipped; a new Frame
    and GPU allocation are always returned. Invalid frame types, dtypes, RGB
    labels, gamma tokens, or non-linear input metadata raise :class:`ValueError`.

    S-Log / S-Log2 / S-Log3 apply their lower linear branches directly to signed inputs. For S-Log and S-Log2,
    public scene-linear reflectance uses x = r / 0.9 and Sony encoded IRE uses the public embedding
    e = (64 + 876 * y) / 1023. S-Log3 / ARRI-LogC4 do not use sign/magnitude mirroring; ARRI-LogC4 retains its negative
    scene cut. Established S-Log3 and ARRI-LogC4 results for nonnegative inputs remain float32 bit-identical.
    ARRI-LogC3 is the ARRI EI 800 relative scene-exposure curve, maps 18% gray to 400 / 1023, and extends its lower
    linear branch to negative values without clipping or sign/magnitude mirroring.
    Blackmagic Film Gen 5 uses its published natural-log branches. DaVinci Intermediate uses its published base-2
    branches and a derived decode threshold. Both apply their lower linear branches directly to negative values.
    RED-Log3G10 uses RED's published base-10 curve with a directly extended lower branch below scene-linear -0.01.
    REDlogFilm uses the Cineon sign-preserving mirror and exact float32 bits while preserving its own metadata.
    """
    frame = _validate_rgb_transfer_frame(frame, operation="linear_to_gamma")
    validated_gamma = _validate_axis(gamma, parameter="gamma", accepted=_GAMMA_TOKENS)
    if validated_gamma is None:
        raise _error(
            why="linear_to_gamma requires an explicit output transfer",
            what="received gamma=None",
            how=f"use one of {_GAMMA_TOKENS!r}",
        )
    if frame.gamma != "linear":
        raise _error(
            why="linear_to_gamma requires a linear input representation",
            what=f"received frame.gamma={frame.gamma!r}",
            how="correct the metadata explicitly or call gamma_to_linear first",
        )
    data = _run_transform(
        frame,
        input_mode=_INPUT_RGB,
        output_mode=_OUTPUT_RGB,
        input_colorspace=frame.colorspace,
        input_gamma="linear",
        output_colorspace=frame.colorspace,
        output_gamma=validated_gamma,
    )
    return Frame(
        data=data,
        colorspace=frame.colorspace,
        gamma=validated_gamma,
        channels=frame.channels,
        matrix=None,
    )


def ycbcr_to_ycbcr(
    frame: Frame,
    *,
    colorspace: Colorspace | None = None,
    gamma: Gamma | None = None,
    input_matrix: Matrix | None = None,
    output_matrix: Matrix | None = None,
    input_range: Range = "full",
    input_bit_depth: int = 8,
    output_range: Range = "full",
    output_bit_depth: int = 8,
) -> Frame:
    """Re-express YCbCr in one fused pass without exposing an RGB Frame.

    For example, ``input_matrix="BT.709", output_matrix="native"`` converts a
    Rec.709 container convention to the working colorspace's own-row basis.
    Reversing those arguments performs the inverse rematrix.

    Parameters
    ----------
    frame:
        A float32 Frame containing exactly one Y, Cb, and Cr and no RGB
        destination labels. Its colorspace and gamma declare the input
        representation.
    colorspace, gamma:
        Output representation tokens. ``None`` independently inherits the
        corresponding input Frame metadata.
    input_matrix:
        Input YCbCr basis: ``"BT.601"``, ``"BT.709"``, ``"BT.2020"``, or
        ``"native"``. Omission resolves in order from ``frame.matrix``, the
        sRGB/Rec.709/Rec.2020 convention, or an error when provenance is absent.
    input_range, input_bit_depth:
        Independent input code grid. Range is ``"full"`` or ``"legal"`` and
        bit depth is 8, 10, 12, 14, or 16.
    output_matrix:
        Output YCbCr basis from the same four-token set. When omitted, the same
        colorspace retains the resolved input basis; a changed colorspace uses
        the encode resolver (BT.709, BT.2020, native for other linear output, or
        BT.709 for other encoded output).
    output_range, output_bit_depth:
        Independent output code grid with the same range and bit-depth domains.

    Returns
    -------
    Frame
        A new float32 Frame with Y/Cb/Cr labels and auxiliary channel order
        preserved, output metadata stamped, and the resolved output matrix.
        Input range expansion, matrix decoding, technical conversion, output
        matrix encoding, and range mapping run in one fused GPU pass without
        clipping scene values.

    Raises
    ------
    ValueError
        If the Frame type, dtype, labels, representation tokens, matrix
        provenance, ranges, or bit depths are invalid.
    """
    frame = _validate_ycbcr_frame(frame, operation="ycbcr_to_ycbcr")
    colorspace = _validate_axis(colorspace, parameter="colorspace", accepted=_COLORSPACE_TOKENS)
    gamma = _validate_axis(gamma, parameter="gamma", accepted=_GAMMA_TOKENS)
    input_range, input_bit_depth = _validate_range_and_bit_depth(
        input_range, input_bit_depth, operation="ycbcr_to_ycbcr input"
    )
    output_range, output_bit_depth = _validate_range_and_bit_depth(
        output_range, output_bit_depth, operation="ycbcr_to_ycbcr output"
    )
    resolved_input_matrix = _resolve_decode_matrix(input_matrix, frame=frame, parameter="input_matrix")
    output_colorspace = frame.colorspace if colorspace is None else colorspace
    output_gamma = frame.gamma if gamma is None else gamma
    validated_output_matrix = _validate_matrix(output_matrix, parameter="output_matrix")
    if validated_output_matrix is not None:
        resolved_output_matrix = validated_output_matrix
    elif output_colorspace == frame.colorspace:
        resolved_output_matrix = resolved_input_matrix
    else:
        resolved_output_matrix = _resolve_encode_matrix(
            None,
            colorspace=output_colorspace,
            gamma=output_gamma,
        )
    data = _run_transform(
        frame,
        input_mode=_INPUT_YCBCR,
        output_mode=_OUTPUT_YCBCR,
        input_colorspace=frame.colorspace,
        input_gamma=frame.gamma,
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        input_matrix=resolved_input_matrix,
        output_matrix=resolved_output_matrix,
        input_range=input_range,
        input_bit_depth=input_bit_depth,
        output_range=output_range,
        output_bit_depth=output_bit_depth,
    )
    return Frame(
        data=data,
        colorspace=output_colorspace,
        gamma=output_gamma,
        channels=frame.channels,
        matrix=resolved_output_matrix,
    )
