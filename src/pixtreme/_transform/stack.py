"""GPU-native concatenation of metadata-bearing Frames."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import cupy as cp
import numpy as np

from pixtreme._channel.shuffle import _route_frame
from pixtreme._color.semantics import rgb_to_ycbcr, ycbcr_to_rgb
from pixtreme._color.transform import rgb_to_rgb
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _STACK_DIRECTION_TOKENS, StackDirection
from pixtreme._transform.resize import resize

_RGB_CHANNELS = ("R", "G", "B")
_YCBCR_CHANNELS = ("Y", "Cb", "Cr")


def _normalize_images(images: Sequence[Frame]) -> tuple[Frame, ...]:
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
        raise ValueError(
            _actionable_error(
                why="stack requires a sequence of Frame values",
                what=f"received {type(images).__module__}.{type(images).__qualname__}",
                how="pass a nonempty list or tuple of Frame values",
            )
        )
    normalized = tuple(images)
    if not normalized:
        raise ValueError(
            _actionable_error(
                why="stack cannot concatenate an empty image sequence",
                what="received zero Frame values",
                how="pass one or more Frame values",
            )
        )
    for index, image in enumerate(normalized):
        if not isinstance(image, Frame):
            raise ValueError(
                _actionable_error(
                    why="every stack member must be a Frame",
                    what=(f"images[{index}] is {type(image).__module__}.{type(image).__qualname__}"),
                    how="construct every input with px.io.from_array or another Frame-returning operation",
                )
            )
    return normalized


def _validate_direction(direction: str) -> str:
    return _normalized_closed_token(direction, axis="direction", accepted=_STACK_DIRECTION_TOKENS)


def _mismatch_error(
    *,
    field: str,
    first_value: object,
    other_value: object,
    other_index: int,
    adapt: bool,
) -> ValueError:
    mode = "adapt=True cannot rescue" if adapt else "adapt=False requires matching"
    return ValueError(
        _actionable_error(
            why=f"stack {mode} {field} values",
            what=(f"images[0] {field}={first_value!r}, images[{other_index}] {field}={other_value!r}"),
            how="make the values equal before stacking or change the input collection",
        )
    )


def _validate_shared_structure(images: tuple[Frame, ...], *, adapt: bool) -> None:
    first = images[0]
    for index, image in enumerate(images[1:], start=1):
        if not adapt and image.channels != first.channels:
            raise _mismatch_error(
                field="channels",
                first_value=first.channels,
                other_value=image.channels,
                other_index=index,
                adapt=adapt,
            )
        if np.dtype(image.dtype) != np.dtype(first.dtype):
            raise _mismatch_error(
                field="dtype",
                first_value=str(first.dtype),
                other_value=str(image.dtype),
                other_index=index,
                adapt=adapt,
            )


def _validate_default_compatibility(images: tuple[Frame, ...], *, direction: str) -> None:
    first = images[0]
    orthogonal_field = "width" if direction == "vertical" else "height"
    first_orthogonal = first.width if direction == "vertical" else first.height
    for index, image in enumerate(images[1:], start=1):
        image_orthogonal = image.width if direction == "vertical" else image.height
        if image_orthogonal != first_orthogonal:
            raise _mismatch_error(
                field=orthogonal_field,
                first_value=first_orthogonal,
                other_value=image_orthogonal,
                other_index=index,
                adapt=False,
            )
        if image.colorspace != first.colorspace:
            raise _mismatch_error(
                field="colorspace",
                first_value=first.colorspace,
                other_value=image.colorspace,
                other_index=index,
                adapt=False,
            )
        if image.gamma != first.gamma:
            raise _mismatch_error(
                field="gamma",
                first_value=first.gamma,
                other_value=image.gamma,
                other_index=index,
                adapt=False,
            )


def _half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _resize_to_first(image: Frame, first: Frame, *, direction: str) -> Frame:
    if direction == "vertical":
        if image.width == first.width:
            return image
        height = _half_up(image.height * first.width / image.width)
        if height < 1:
            raise ValueError(
                _actionable_error(
                    why="vertical adapt produced a non-positive aspect-preserving height",
                    what=f"source shape={image.shape!r}, master width={first.width}, rounded height={height}",
                    how="provide geometry whose half-up resized height is at least 1",
                )
            )
        return resize(image, width=first.width, height=height)

    if image.height == first.height:
        return image
    width = _half_up(image.width * first.height / image.height)
    if width < 1:
        raise ValueError(
            _actionable_error(
                why="horizontal adapt produced a non-positive aspect-preserving width",
                what=f"source shape={image.shape!r}, master height={first.height}, rounded width={width}",
                how="provide geometry whose half-up resized width is at least 1",
            )
        )
    return resize(image, width=width, height=first.height)


def _color_channels(frame: Frame) -> tuple[str, ...]:
    return tuple(label for label in frame.channels if label != "A")


def _adapt_conversion_error(first: Frame, image: Frame, *, index: int, detail: str | None = None) -> ValueError:
    description = (
        f"images[0] channels={first.channels!r}, colorspace={first.colorspace!r}, gamma={first.gamma!r}; "
        f"images[{index}] channels={image.channels!r}, colorspace={image.colorspace!r}, gamma={image.gamma!r}"
    )
    if detail is not None:
        description = f"{description}; conversion failed: {detail}"
    return ValueError(
        _actionable_error(
            why="stack adapt=True found no complete deterministic channels/color conversion",
            what=description,
            how=(
                "use equal channel sets or RGB/YCbCr color triplets with matching channel counts, alpha presence, "
                "and colorspace-derived matrices"
            ),
        )
    )


def _adapt_color_to_first(image: Frame, first: Frame, *, index: int) -> Frame:
    image_colors = _color_channels(image)
    first_colors = _color_channels(first)
    same_color_labels = Counter(image_colors) == Counter(first_colors)
    same_colorimetry = image.colorspace == first.colorspace and image.gamma == first.gamma
    if (
        len(image.channels) != len(first.channels)
        or ("A" in image.channels) != ("A" in first.channels)
        or len(image_colors) != len(first_colors)
    ):
        raise _adapt_conversion_error(first, image, index=index)

    if same_color_labels and same_colorimetry:
        if image.channels == first.channels:
            return image
        try:
            return _route_frame(image, first.channels)
        except ValueError as error:
            raise _adapt_conversion_error(first, image, index=index, detail=str(error)) from error

    supported_sets = {frozenset(_RGB_CHANNELS), frozenset(_YCBCR_CHANNELS)}
    image_set = frozenset(image_colors)
    first_set = frozenset(first_colors)
    if (
        image_set not in supported_sets
        or first_set not in supported_sets
        or len(image_colors) != 3
        or len(first_colors) != 3
    ):
        raise _adapt_conversion_error(first, image, index=index)

    current = image
    try:
        if image_set == frozenset(_YCBCR_CHANNELS):
            current = ycbcr_to_rgb(current)
        if not same_colorimetry:
            current = rgb_to_rgb(
                current,
                output_colorspace=first.colorspace,
                output_gamma=first.gamma,
            )
        if first_set == frozenset(_YCBCR_CHANNELS):
            if first.matrix is None and first.colorspace not in {"sRGB", "Rec.709", "Rec.2020"}:
                raise ValueError(
                    _actionable_error(
                        why="target YCbCr Frame has no deterministic matrix provenance",
                        what=f"target colorspace={first.colorspace!r}, matrix={first.matrix!r}",
                        how=(
                            "set target Frame matrix metadata or use a YCbCr-defining colorspace: "
                            "'sRGB', 'Rec.709', or 'Rec.2020'"
                        ),
                    )
                )
            current = rgb_to_ycbcr(current, matrix=first.matrix)
        if current.channels != first.channels:
            current = _route_frame(current, first.channels)
    except ValueError as error:
        raise _adapt_conversion_error(first, image, index=index, detail=str(error)) from error
    return current


def _adapt_images(images: tuple[Frame, ...], *, direction: str) -> tuple[Frame, ...]:
    first = images[0]
    if np.dtype(first.dtype) != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why="stack adapt=True uses float32 color and resize operations",
                what=f"received shared Frame dtype {first.dtype!s}",
                how=_float32_conversion_guidance(np.dtype(first.dtype)),
            )
        )

    adapted = [first]
    for index, image in enumerate(images[1:], start=1):
        current = _adapt_color_to_first(image, first, index=index)
        adapted.append(_resize_to_first(current, first, direction=direction))
    return tuple(adapted)


def stack(
    images: Sequence[Frame],
    *,
    direction: StackDirection = "vertical",
    adapt: bool = False,
) -> Frame:
    """Concatenate Frames vertically or horizontally into a new Frame.

    ``vertical`` places inputs from top to bottom and requires a common width;
    ``horizontal`` places them from left to right and requires a common height.
    With ``adapt=False``, channels, colorspace, gamma, dtype, and that
    orthogonal dimension must already match.

    With ``adapt=True``, the first Frame is the master. Later Frames are first
    transformed to its channels, colorspace, and gamma when a deterministic
    equal-set reorder or RGB/YCbCr conversion exists, then resized to its
    orthogonal dimension with aspect preservation and half-up rounding. Dtype
    still must match, and multi-input adaptation requires float32. The result
    always owns new storage, including a single-input call.
    """
    normalized = _normalize_images(images)
    resolved_direction = _validate_direction(direction)
    if not isinstance(adapt, bool):
        raise ValueError(
            _actionable_error(
                why="stack adapt must be a bool",
                what=f"received adapt={adapt!r}",
                how="pass adapt=False or adapt=True",
            )
        )

    if len(normalized) == 1:
        first = normalized[0]
        return Frame(
            data=first.data.copy(),
            colorspace=first.colorspace,
            gamma=first.gamma,
            channels=first.channels,
            matrix=first.matrix,
        )

    _validate_shared_structure(normalized, adapt=adapt)
    if adapt:
        prepared = _adapt_images(normalized, direction=resolved_direction)
    else:
        _validate_default_compatibility(normalized, direction=resolved_direction)
        prepared = normalized

    axis = 0 if resolved_direction == "vertical" else 1
    first = prepared[0]
    return Frame(
        data=cp.concatenate(tuple(image.data for image in prepared), axis=axis),
        colorspace=first.colorspace,
        gamma=first.gamma,
        channels=first.channels,
        matrix=first.matrix,
    )
