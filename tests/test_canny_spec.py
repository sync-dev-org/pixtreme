"""Specification and independent numerical-oracle tests for Canny edge detection."""

from __future__ import annotations

import inspect
from collections import deque
from typing import Any

import numpy as np
import pytest

import pixtreme as px

BORDERS = ("mirror", "replicate", "wrap", "constant")
_SOBEL_X = np.asarray(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)), dtype=np.float32)
_SOBEL_Y = _SOBEL_X.T


def _frame(
    values: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | tuple[str, ...] = ("signal",),
    matrix: str | None = None,
) -> px.core.Frame:
    import cupy as cp

    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=dtype)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix=matrix,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _border_index(index: int, extent: int, border: str) -> int:
    if extent <= 1:
        return 0
    if border == "replicate":
        return min(max(index, 0), extent - 1)
    if border == "wrap":
        return index % extent
    period = 2 * extent - 2
    reflected = index % period
    return reflected if reflected < extent else period - reflected


def _sample(
    source: np.ndarray,
    *,
    x: int,
    y: int,
    channel: int,
    border: str,
    border_value: float,
) -> float:
    height, width, _ = source.shape
    if border == "constant" and not (0 <= x < width and 0 <= y < height):
        return border_value
    return float(source[_border_index(y, height, border), _border_index(x, width, border), channel])


def _sobel_reference(
    source: np.ndarray,
    *,
    border: str,
    border_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the specified 3x3 Sobel pair in scalar host NumPy."""
    dx = np.empty_like(source, dtype=np.float32)
    dy = np.empty_like(source, dtype=np.float32)
    height, width, channel_count = source.shape
    for y in range(height):
        for x in range(width):
            for channel in range(channel_count):
                total_x = 0.0
                total_y = 0.0
                for kernel_y in range(3):
                    for kernel_x in range(3):
                        value = _sample(
                            source,
                            x=x + kernel_x - 1,
                            y=y + kernel_y - 1,
                            channel=channel,
                            border=border,
                            border_value=border_value,
                        )
                        total_x += float(_SOBEL_X[kernel_y, kernel_x]) * value
                        total_y += float(_SOBEL_Y[kernel_y, kernel_x]) * value
                dx[y, x, channel] = np.float32(total_x)
                dy[y, x, channel] = np.float32(total_y)
    return dx, dy


def _nms_reference(
    magnitude: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    border: str,
    border_value: float,
) -> np.ndarray:
    """Apply the four half-open sectors and asymmetric plateau rule on the host."""
    output = np.zeros_like(magnitude, dtype=np.float32)
    height, width, channel_count = magnitude.shape
    for y in range(height):
        for x in range(width):
            for channel in range(channel_count):
                theta = float(np.degrees(np.arctan2(float(dy[y, x, channel]), float(dx[y, x, channel])))) % 180.0
                if theta < 22.5 or theta >= 157.5:
                    offset_x, offset_y = 1, 0
                elif theta < 67.5:
                    offset_x, offset_y = 1, 1
                elif theta < 112.5:
                    offset_x, offset_y = 0, 1
                else:
                    offset_x, offset_y = -1, 1
                current = float(magnitude[y, x, channel])
                negative = _sample(
                    magnitude,
                    x=x - offset_x,
                    y=y - offset_y,
                    channel=channel,
                    border=border,
                    border_value=border_value,
                )
                positive = _sample(
                    magnitude,
                    x=x + offset_x,
                    y=y + offset_y,
                    channel=channel,
                    border=border,
                    border_value=border_value,
                )
                if current > negative and current >= positive:
                    output[y, x, channel] = np.float32(current)
    return output


def _hysteresis_reference(nms: np.ndarray, *, threshold_low: float, threshold_high: float) -> np.ndarray:
    """Find strong-reachable weak components with a host deque, not GPU propagation."""
    strong = nms >= np.float32(threshold_high)
    weak = (nms >= np.float32(threshold_low)) & (nms < np.float32(threshold_high))
    output = strong.copy()
    height, width, channel_count = nms.shape
    for channel in range(channel_count):
        queue = deque((int(y), int(x)) for y, x in np.argwhere(strong[..., channel]))
        while queue:
            y, x = queue.popleft()
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    neighbor_y = y + offset_y
                    neighbor_x = x + offset_x
                    if (
                        (offset_x != 0 or offset_y != 0)
                        and 0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                        and weak[neighbor_y, neighbor_x, channel]
                        and not output[neighbor_y, neighbor_x, channel]
                    ):
                        output[neighbor_y, neighbor_x, channel] = True
                        queue.append((neighbor_y, neighbor_x))
    return output.astype(np.float32)


def _canny_reference(
    source: np.ndarray,
    *,
    threshold_low: float,
    threshold_high: float,
    border: str,
    border_value: float,
) -> np.ndarray:
    """Independent host pipeline derived only from v1-canny acceptance 5-12."""
    dx, dy = _sobel_reference(source, border=border, border_value=border_value)
    magnitude = np.hypot(dx, dy).astype(np.float32)
    nms = _nms_reference(magnitude, dx, dy, border=border, border_value=border_value)
    return _hysteresis_reference(nms, threshold_low=threshold_low, threshold_high=threshold_high)


def test_canny_public_signature_and_single_canonical_path_are_exact() -> None:
    """v1-canny acceptance 1 and 14: required thresholds exist only on px.filter.canny."""
    import cupy as cp

    signature = inspect.signature(px.filter.canny)
    assert tuple(signature.parameters) == ("frame", "threshold_low", "threshold_high", "border", "border_value")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("threshold_low", "threshold_high", "border", "border_value"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["threshold_low"].default is inspect.Parameter.empty
    assert signature.parameters["threshold_high"].default is inspect.Parameter.empty
    assert signature.parameters["border"].default == "mirror"
    assert signature.parameters["border_value"].default is None
    assert px.filter.__all__.count("canny") == 1
    assert not hasattr(px, "canny")
    assert not hasattr(px.core.Frame, "canny")

    with pytest.raises(ValueError) as error:
        px.filter.canny(cp.zeros((2, 2, 1), dtype=cp.float32), threshold_low=0.5, threshold_high=1.0)
    _assert_actionable(error)


@pytest.mark.parametrize("border", BORDERS)
def test_canny_matches_independent_pipeline_oracle_for_every_border(border: str) -> None:
    """v1-canny acceptance 2, 4-6, and 9-11: every border and channel matches the host pipeline."""
    rng = np.random.default_rng(20260803)
    values = rng.uniform(-0.75, 1.75, size=(7, 8, 3)).astype(np.float32)
    border_value = -0.6
    expected = _canny_reference(
        values,
        threshold_low=1.25,
        threshold_high=3.75,
        border=border,
        border_value=border_value,
    )
    kwargs = {"border_value": border_value} if border == "constant" else {}

    actual = px.io.to_array(
        px.filter.canny(
            _frame(values, channels=("Z", "custom", "A")),
            threshold_low=1.25,
            threshold_high=3.75,
            border=border,
            **kwargs,
        ),
    ).get()

    np.testing.assert_array_equal(actual, expected)
    assert set(np.unique(actual)).issubset({0.0, 1.0})


@pytest.mark.parametrize(
    "angle_degrees",
    (0.0, 22.0, 22.5, 23.0, 67.0, 67.5, 68.0, 112.0, 112.5, 113.0, 157.0, 157.5, 158.0),
)
def test_canny_four_sector_boundaries_match_the_half_open_host_oracle(angle_degrees: float) -> None:
    """v1-canny acceptance 7: sector centers and every half-open boundary follow atan2 modulo 180 degrees."""
    coordinates_y, coordinates_x = np.indices((19, 19), dtype=np.float32)
    coordinates_x -= np.float32(9.0)
    coordinates_y -= np.float32(9.0)
    angle = np.deg2rad(angle_degrees)
    projection = coordinates_x * np.float32(np.cos(angle)) + coordinates_y * np.float32(np.sin(angle))
    values = np.exp(np.float32(-0.5) * (projection / np.float32(2.0)) ** 2)[..., None].astype(np.float32)
    expected = _canny_reference(
        values,
        threshold_low=0.25,
        threshold_high=0.5,
        border="replicate",
        border_value=0.0,
    )

    actual = px.io.to_array(
        px.filter.canny(_frame(values), threshold_low=0.25, threshold_high=0.5, border="replicate"),
    ).get()

    np.testing.assert_array_equal(actual, expected)


def test_canny_asymmetric_nms_tie_break_keeps_only_the_negative_side_of_a_plateau() -> None:
    """v1-canny acceptance 8: current > negative and current >= positive retains one plateau side."""
    values = np.zeros((5, 7, 1), dtype=np.float32)
    values[:, 3:, 0] = np.float32(1.0)

    actual = px.io.to_array(
        px.filter.canny(_frame(values), threshold_low=4.0, threshold_high=4.0, border="replicate"),
    ).get()

    expected = np.zeros_like(values)
    expected[:, 2, 0] = np.float32(1.0)
    np.testing.assert_array_equal(actual, expected)


def test_canny_hysteresis_fully_converges_along_a_long_weak_path_and_stays_per_channel() -> None:
    """v1-canny acceptance 4 and 10-12: long 8-connected paths converge while isolated channel edges do not."""
    values = np.zeros((70, 8, 2), dtype=np.float32)
    values[:, 4:, :] = np.float32(0.3)
    values[0, 4:, 0] = np.float32(1.0)
    source = _frame(values, channels=("connected", "isolated"))
    expected = _canny_reference(
        values,
        threshold_low=0.5,
        threshold_high=3.0,
        border="replicate",
        border_value=0.0,
    )

    first = px.io.to_array(
        px.filter.canny(source, threshold_low=0.5, threshold_high=3.0, border="replicate"),
    ).get()
    second = px.io.to_array(
        px.filter.canny(source, threshold_low=0.5, threshold_high=3.0, border="replicate"),
    ).get()

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, first)
    assert np.any(first[-1, :, 0] == np.float32(1.0))
    assert not np.any(first[..., 1])


@pytest.mark.parametrize("border", BORDERS)
def test_canny_one_pixel_extent_matches_the_oracle_without_virtual_hysteresis_neighbors(border: str) -> None:
    """v1-canny acceptance 9 and 11: one-pixel axes use border for filters but not hysteresis connectivity."""
    values = np.asarray([[[-0.5], [0.25], [1.5], [0.1], [0.8]]], dtype=np.float32)
    border_value = 1.7
    expected = _canny_reference(
        values,
        threshold_low=0.4,
        threshold_high=1.2,
        border=border,
        border_value=border_value,
    )
    kwargs = {"border_value": border_value} if border == "constant" else {}

    actual = px.io.to_array(
        px.filter.canny(_frame(values), threshold_low=0.4, threshold_high=1.2, border=border, **kwargs),
    ).get()

    np.testing.assert_array_equal(actual, expected)


def test_canny_equal_threshold_is_a_valid_single_threshold_with_no_weak_set() -> None:
    """v1-canny acceptance 3 and 10: equal low/high thresholds are accepted without swapping."""
    values = np.asarray(
        [[[0.0], [0.0], [1.0], [1.0]], [[0.0], [0.5], [1.0], [1.5]], [[-0.5], [0.0], [1.0], [2.0]]],
        dtype=np.float32,
    )
    expected = _canny_reference(
        values,
        threshold_low=2.0,
        threshold_high=2.0,
        border="mirror",
        border_value=0.0,
    )

    actual = px.io.to_array(
        px.filter.canny(_frame(values), threshold_low=2.0, threshold_high=2.0),
    ).get()

    np.testing.assert_array_equal(actual, expected)


def test_canny_accepts_zero_thresholds_with_the_specified_inclusive_classification() -> None:
    """v1-canny acceptance 3 and 10: zero is valid and x >= threshold_high remains inclusive."""
    values = np.zeros((2, 3, 1), dtype=np.float32)
    expected = _canny_reference(
        values,
        threshold_low=0.0,
        threshold_high=0.0,
        border="mirror",
        border_value=0.0,
    )

    actual = px.io.to_array(
        px.filter.canny(_frame(values), threshold_low=0.0, threshold_high=0.0),
    ).get()

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, np.ones_like(values))


@pytest.mark.parametrize("name", ("threshold_low", "threshold_high"))
@pytest.mark.parametrize("value", (True, "1", 1 + 0j, float("nan"), float("inf"), float("-inf"), -0.01))
def test_canny_rejects_invalid_threshold_values_actionably(name: str, value: object) -> None:
    """v1-canny acceptance 3 and 13: thresholds reject bool, non-real, non-finite, and negative values."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32))
    kwargs: dict[str, object] = {"threshold_low": 0.5, "threshold_high": 1.0}
    kwargs[name] = value

    with pytest.raises(ValueError) as error:
        px.filter.canny(source, **kwargs)

    _assert_actionable(error)
    assert name in str(error.value)


def test_canny_rejects_reversed_thresholds_instead_of_swapping() -> None:
    """v1-canny acceptance 3 and 13: threshold_low above threshold_high fails fast."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32))

    with pytest.raises(ValueError) as error:
        px.filter.canny(source, threshold_low=2.0, threshold_high=1.0)

    _assert_actionable(error)
    assert "threshold_low" in str(error.value)
    assert "threshold_high" in str(error.value)


@pytest.mark.parametrize(
    ("border", "border_value"),
    (
        ("unknown", None),
        ("reflect", None),
        (None, None),
        ("constant", None),
        ("constant", True),
        ("constant", float("nan")),
        ("mirror", 0.0),
        ("replicate", -1.0),
        ("wrap", 2.0),
    ),
)
def test_canny_rejects_invalid_border_combinations_actionably(border: object, border_value: object) -> None:
    """v1-canny acceptance 9 and 13: border and border_value use the shared fail-fast contract."""
    source = _frame(np.zeros((2, 2, 1), dtype=np.float32))

    with pytest.raises(ValueError) as error:
        px.filter.canny(
            source,
            threshold_low=0.5,
            threshold_high=1.0,
            border=border,
            border_value=border_value,
        )

    _assert_actionable(error)


@pytest.mark.parametrize(
    ("dtype", "guidance"),
    ((np.float16, "cast_dtype"), (np.uint8, "recode_dtype"), (np.uint16, "dequantize")),
)
def test_canny_rejects_non_float32_frames_with_conversion_guidance(
    dtype: np.dtype[Any] | type[np.generic], guidance: str
) -> None:
    """v1-canny acceptance 4 and 13: non-fp32 Frames name a value-aware conversion path."""
    source = _frame(np.zeros((2, 2, 1), dtype=dtype), dtype=dtype)

    with pytest.raises(ValueError) as error:
        px.filter.canny(source, threshold_low=0.5, threshold_high=1.0)

    _assert_actionable(error)
    assert guidance in str(error.value)


def test_canny_returns_private_contiguous_binary_storage_and_preserves_metadata_and_input() -> None:
    """v1-canny acceptance 2 and 4; v1-red-tokens acceptance 68: renamed ARRI metadata is fixed."""
    values = np.linspace(-0.5, 1.5, 6 * 7 * 4, dtype=np.float32).reshape(6, 7, 4)
    source = _frame(
        values,
        colorspace="ACEScg",
        gamma="ARRI-LogC4",
        channels=("R", "G", "custom", "A"),
        matrix="native",
    )
    source_before = px.io.to_array(source, copy=True).get()

    result = px.filter.canny(source, threshold_low=0.25, threshold_high=0.75, border="wrap")
    actual = px.io.to_array(
        result,
    ).get()

    assert result.shape == source.shape
    assert result.dtype == np.dtype(np.float32)
    assert result.data.flags.c_contiguous
    assert result.data.data.ptr != source.data.data.ptr
    assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
        source.colorspace,
        source.gamma,
        source.channels,
        source.matrix,
    )
    assert set(np.unique(actual)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        source_before,
    )


def test_canny_docstring_is_a_self_contained_operational_contract() -> None:
    """v1-canny acceptance 14: the public docstring names every user-visible pipeline contract."""
    docstring = inspect.getdoc(px.filter.canny) or ""
    for required in (
        "threshold_low",
        "threshold_high",
        "non-normalized 3x3 Sobel",
        "current > magnitude(-v)",
        "current >= magnitude(+v)",
        "8-connected",
        "complete convergence",
        "mirror",
        "replicate",
        "wrap",
        "constant",
        "per channel",
        "float32",
        "metadata",
        "px.values.cast_dtype",
        "px.values.recode_dtype",
        "px.values.dequantize",
    ):
        assert required in docstring


def test_canny_vocabulary_documents_shared_border_tokens_and_both_internal_stages(
    vocabulary_markdown: str,
) -> None:
    """v1-canny acceptance 15: vocabulary fixes four border tokens, mirror default, and Sobel/NMS scope."""
    section = vocabulary_markdown.split("## border\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]

    for required in ("px.filter.canny", "mirror", "replicate", "wrap", "constant", "Sobel", "NMS"):
        assert required in section
