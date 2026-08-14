"""Specification, contract, property, and documentation tests for RGB / HSV conversion."""

from __future__ import annotations

import inspect
import itertools

import cupy as cp
import numpy as np
import pytest

import pixtreme as px


def _frame(
    values: object,
    *,
    channels: tuple[str, ...] = ("R", "G", "B"),
    dtype: object = np.float32,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    matrix: str | None = None,
) -> px.core.Frame:
    data = cp.asarray(values, dtype=dtype).reshape(-1, 1, len(channels))
    return px.core.Frame(data=data, colorspace=colorspace, gamma=gamma, channels=channels, matrix=matrix)


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _numpy_rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Independent host implementation of v1-hsv acceptance 5-6."""
    values = np.asarray(rgb, dtype=np.float64)
    red, green, blue = np.moveaxis(values, -1, 0)
    maximum = np.maximum(red, np.maximum(green, blue))
    minimum = np.minimum(red, np.minimum(green, blue))
    delta = maximum - minimum

    saturation = np.zeros_like(maximum)
    np.divide(delta, maximum, out=saturation, where=maximum != 0.0)
    hue = np.zeros_like(maximum)
    chromatic = delta > 0.0
    red_sector = chromatic & (red == maximum)
    green_sector = chromatic & ~red_sector & (green == maximum)
    blue_sector = chromatic & ~red_sector & ~green_sector
    hue[red_sector] = ((green[red_sector] - blue[red_sector]) / delta[red_sector]) / 6.0
    hue[green_sector] = (2.0 + (blue[green_sector] - red[green_sector]) / delta[green_sector]) / 6.0
    hue[blue_sector] = (4.0 + (red[blue_sector] - green[blue_sector]) / delta[blue_sector]) / 6.0
    hue = np.mod(hue, 1.0)
    return np.stack((hue, saturation, maximum), axis=-1)


def _numpy_hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Independent host implementation of v1-hsv acceptance 8-9."""
    values = np.asarray(hsv, dtype=np.float64)
    hue, saturation, value = np.moveaxis(values, -1, 0)
    h6 = 6.0 * np.mod(hue, 1.0)
    sector = np.floor(h6).astype(np.int64)
    chroma = value * saturation
    x_value = chroma * (1.0 - np.abs(np.mod(h6, 2.0) - 1.0))
    minimum = value - chroma
    zero = np.zeros_like(chroma)
    red = np.select(
        (sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5),
        (chroma, x_value, zero, zero, x_value, chroma),
    )
    green = np.select(
        (sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5),
        (x_value, chroma, chroma, x_value, zero, zero),
    )
    blue = np.select(
        (sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5),
        (zero, zero, x_value, chroma, chroma, x_value),
    )
    return np.stack((red + minimum, green + minimum, blue + minimum), axis=-1)


def test_hsv_public_signatures_and_namespace_are_exact() -> None:
    """v1-hsv acceptance 1 and 12: two frame-only color paths exist without aliases or methods."""
    assert tuple(inspect.signature(px.color.rgb_to_hsv).parameters) == ("frame",)
    assert tuple(inspect.signature(px.color.hsv_to_rgb).parameters) == ("frame",)
    assert inspect.signature(px.color.rgb_to_hsv).parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert inspect.signature(px.color.hsv_to_rgb).parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("rgb_to_hsv", "hsv_to_rgb"):
        assert px.color.__all__.count(name) == 1
        assert not hasattr(px, name)
        assert not hasattr(px.core.Frame, name)
    for name in ("rgb_to_bgr", "bgr_to_hsv", "hsv_to_bgr"):
        assert not hasattr(px.color, name)


def test_rgb_to_hsv_matches_hand_calculated_primary_secondary_and_achromatic_values() -> None:
    """v1-hsv acceptance 5-7 and 10: known colors, gray, black, and scene values have fixed HSV values."""
    rgb = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 4.0, 4.0),
            (0.0, 0.0, 5.0),
            (6.0, 0.0, 6.0),
        ),
        dtype=np.float32,
    )
    expected = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5),
            (0.0, 1.0, 2.0),
            (1.0 / 6.0, 1.0, 2.0),
            (2.0 / 6.0, 1.0, 3.0),
            (3.0 / 6.0, 1.0, 4.0),
            (4.0 / 6.0, 1.0, 5.0),
            (5.0 / 6.0, 1.0, 6.0),
        ),
        dtype=np.float32,
    )

    result = px.color.rgb_to_hsv(_frame(rgb))

    np.testing.assert_allclose(result.data.get()[:, 0], expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("channels", tuple(itertools.permutations(("R", "G", "B"))))
def test_rgb_to_hsv_reads_each_rgb_permutation_by_label(channels: tuple[str, ...]) -> None:
    """v1-hsv acceptance 3-4: every RGB input order is label-driven and output is canonical HSV."""
    rgb_by_label = {"R": 0.25, "G": 1.5, "B": 0.75}
    source = _frame(tuple(rgb_by_label[label] for label in channels), channels=channels)

    result = px.color.rgb_to_hsv(source)

    expected = _numpy_rgb_to_hsv(np.asarray((0.25, 1.5, 0.75), dtype=np.float32))
    assert result.channels == ("H", "S", "V")
    np.testing.assert_allclose(result.data.get()[0, 0], expected, rtol=0.0, atol=2e-7)


@pytest.mark.parametrize("channels", tuple(itertools.permutations(("H", "S", "V"))))
def test_hsv_to_rgb_reads_each_hsv_permutation_by_label(channels: tuple[str, ...]) -> None:
    """v1-hsv acceptance 3-4 and 8: every HSV input order is label-driven and output is canonical RGB."""
    hsv_by_label = {"H": 4.5 / 6.0, "S": 0.75, "V": 1.8}
    source = _frame(tuple(hsv_by_label[label] for label in channels), channels=channels)

    result = px.color.hsv_to_rgb(source)

    expected = _numpy_hsv_to_rgb(np.asarray((4.5 / 6.0, 0.75, 1.8), dtype=np.float32))
    assert result.channels == ("R", "G", "B")
    np.testing.assert_allclose(result.data.get()[0, 0], expected, rtol=0.0, atol=3e-7)


def test_rgb_to_hsv_matches_independent_numpy_for_nonnegative_scene_values() -> None:
    """v1-hsv acceptance 5-7: independent host equations cover random finite RGB values above one."""
    generator = np.random.default_rng(20260804)
    rgb = generator.uniform(0.0, 8.0, size=(9, 11, 3)).astype(np.float32)
    source = px.core.Frame(data=cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB")

    result = px.color.rgb_to_hsv(source)

    expected = _numpy_rgb_to_hsv(rgb)
    np.testing.assert_allclose(result.data.get(), expected, rtol=3e-7, atol=3e-7)
    assert np.all((result.data.get()[..., 0] >= 0.0) & (result.data.get()[..., 0] < 1.0))
    assert np.all((result.data.get()[..., 1] >= 0.0) & (result.data.get()[..., 1] <= 1.0))
    np.testing.assert_array_equal(result.data.get()[..., 2], np.max(rgb, axis=-1))


def test_rgb_to_hsv_preserves_positive_scale_in_hue_saturation_and_value() -> None:
    """v1-hsv acceptance 7: positive RGB scaling preserves H/S and scales V without clipping."""
    rgb = np.asarray(((0.2, 0.7, 1.3), (2.0, 0.4, 0.9), (0.1, 4.0, 1.2)), dtype=np.float32)
    source = px.color.rgb_to_hsv(_frame(rgb))
    scaled = px.color.rgb_to_hsv(_frame(rgb * np.float32(3.25)))

    np.testing.assert_allclose(scaled.data.get()[..., :2], source.data.get()[..., :2], rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(
        scaled.data.get()[..., 2], source.data.get()[..., 2] * np.float32(3.25), rtol=2e-7, atol=2e-7
    )


def test_hsv_to_rgb_wraps_hue_and_matches_all_six_host_sectors() -> None:
    """v1-hsv acceptance 8-9: negative, seam, and over-one hues wrap across all six exact sectors."""
    hues = np.asarray((-1.0, -1.0 / 6.0, 0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0, 1.0, 2.25))
    hsv = np.stack((hues, np.ones_like(hues), np.full_like(hues, 2.0)), axis=-1).astype(np.float32)
    expected = np.asarray(
        (
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 2.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 2.0, 2.0),
            (0.0, 0.0, 2.0),
            (2.0, 0.0, 2.0),
            (2.0, 0.0, 0.0),
            (1.0, 2.0, 0.0),
        ),
        dtype=np.float32,
    )

    result = px.color.hsv_to_rgb(_frame(hsv, channels=("H", "S", "V")))

    np.testing.assert_allclose(result.data.get()[:, 0], expected, rtol=0.0, atol=3e-7)


def test_hsv_to_rgb_does_not_clip_saturation_or_value() -> None:
    """v1-hsv acceptance 9 and 14: S/V outside nominal bounds are formula inputs rather than errors."""
    hsv = np.asarray(
        ((0.25, 2.0, 3.0), (0.75, -0.5, 1.5), (0.4, 0.0, -2.0), (123.0, 0.0, 0.0), (-123.0, 0.0, 2.0)),
        dtype=np.float32,
    )

    result = px.color.hsv_to_rgb(_frame(hsv, channels=("H", "S", "V")))

    np.testing.assert_allclose(result.data.get()[:, 0], _numpy_hsv_to_rgb(hsv), rtol=3e-7, atol=3e-7)
    np.testing.assert_array_equal(
        result.data.get()[2:, 0],
        np.asarray(((-2.0, -2.0, -2.0), (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), dtype=np.float32),
    )


def test_nonnegative_rgb_round_trip_restores_values_above_one() -> None:
    """v1-hsv acceptance 10: fp32 round-trip restores nonnegative scene RGB within operation-depth tolerance."""
    generator = np.random.default_rng(20360804)
    rgb = generator.uniform(0.0, 12.0, size=(13, 17, 3)).astype(np.float32)
    source = px.core.Frame(data=cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB")

    restored = px.color.hsv_to_rgb(px.color.rgb_to_hsv(source))

    # Two divisions, modulo, and the inverse multiply/add chain account for this fp32 bound.
    np.testing.assert_allclose(restored.data.get(), rgb, rtol=2e-6, atol=2e-6)


def test_negative_and_nonfinite_rgb_values_are_not_prevalidated_or_clipped() -> None:
    """v1-hsv acceptance 11 and 14: negative and nonfinite pixels reach the documented equations."""
    finite = np.asarray(((-2.0, -1.0, -3.0), (-1.0, 0.0, -0.5), (0.5, -0.5, 0.25)), dtype=np.float32)
    result = px.color.rgb_to_hsv(_frame(finite))
    np.testing.assert_allclose(result.data.get()[:, 0], _numpy_rgb_to_hsv(finite), rtol=3e-7, atol=3e-7)

    nonfinite = _frame(((np.nan, 0.0, 1.0), (np.inf, 1.0, 0.0)))
    output = px.color.rgb_to_hsv(nonfinite)
    assert output.shape == nonfinite.shape


def test_hsv_operations_preserve_claims_reset_matrix_and_leave_input_storage_unchanged() -> None:
    """v1-hsv acceptance 4: outputs are private canonical Frames and inputs remain byte-for-byte unchanged."""
    source = _frame(
        ((0.25, 1.5, 0.75), (2.0, 0.5, 0.1)),
        channels=("B", "R", "G"),
        colorspace="S-Gamut3.Cine",
        gamma="s-log3",
        matrix="bt601",
    )
    snapshot = source.data.copy()
    source_pointer = source.data.data.ptr

    hsv = px.color.rgb_to_hsv(source)
    rgb = px.color.hsv_to_rgb(hsv)

    assert hsv.colorspace == rgb.colorspace == source.colorspace
    assert hsv.gamma == rgb.gamma == source.gamma
    assert hsv.matrix is rgb.matrix is None
    assert hsv.channels == ("H", "S", "V")
    assert rgb.channels == ("R", "G", "B")
    assert hsv.data.flags.c_contiguous and rgb.data.flags.c_contiguous
    assert hsv.data.dtype == rgb.data.dtype == cp.float32
    assert hsv.data.data.ptr != source_pointer
    assert rgb.data.data.ptr != hsv.data.data.ptr
    cp.testing.assert_array_equal(source.data, snapshot)
    assert source.channels == ("B", "R", "G")
    assert source.matrix == "bt601"


@pytest.mark.parametrize("operation", ("rgb_to_hsv", "hsv_to_rgb"))
def test_hsv_operations_reject_non_frame_with_actionable_error(operation: str) -> None:
    """v1-hsv acceptance 2 and 14: non-Frame inputs fail before pixel processing with recovery guidance."""
    with pytest.raises(ValueError) as error:
        getattr(px.color, operation)(object())
    _assert_actionable(error)
    assert "Frame" in str(error.value)


@pytest.mark.parametrize("operation", ("rgb_to_hsv", "hsv_to_rgb"))
@pytest.mark.parametrize("dtype", (np.float16, np.uint8, np.uint16))
def test_hsv_operations_reject_each_non_float32_dtype_with_cast_guidance(operation: str, dtype: object) -> None:
    """v1-hsv acceptance 2 and 14: every supported non-fp32 Frame dtype names the three conversion paths."""
    channels = ("R", "G", "B") if operation == "rgb_to_hsv" else ("H", "S", "V")
    source = _frame((0, 0, 0), channels=channels, dtype=dtype)

    with pytest.raises(ValueError) as error:
        getattr(px.color, operation)(source)

    _assert_actionable(error)
    for required in ("px.values.cast_dtype", "px.values.recode_dtype", "px.values.dequantize"):
        assert required in str(error.value)


@pytest.mark.parametrize(
    ("operation", "channels"),
    (
        ("rgb_to_hsv", ("R", "G")),
        ("rgb_to_hsv", ("R", "G", "R")),
        ("rgb_to_hsv", ("R", "G", "A")),
        ("rgb_to_hsv", ("R", "G", "B", "A")),
        ("hsv_to_rgb", ("H", "S")),
        ("hsv_to_rgb", ("H", "S", "H")),
        ("hsv_to_rgb", ("H", "S", "A")),
        ("hsv_to_rgb", ("H", "S", "V", "A")),
    ),
)
def test_hsv_operations_reject_non_exact_triplets_with_expected_and_received_channels(
    operation: str, channels: tuple[str, ...]
) -> None:
    """v1-hsv acceptance 3 and 14: missing, duplicate, foreign, and extra labels fail without implicit routing."""
    source = _frame(np.zeros(len(channels), dtype=np.float32), channels=channels)

    with pytest.raises(ValueError) as error:
        getattr(px.color, operation)(source)

    _assert_actionable(error)
    expected = ("R", "G", "B") if operation == "rgb_to_hsv" else ("H", "S", "V")
    assert repr(expected) in str(error.value)
    assert repr(channels) in str(error.value)
    assert "px.channel.shuffle" in str(error.value)


def test_hsv_channel_vocabulary_and_documentation_are_self_contained(vocabulary_markdown: str) -> None:
    """v1-hsv acceptance 13: channel tokens and the RGB / HSV section fix the full numeric contract."""
    assert px.core.channels("HSV") == ("H", "S", "V")
    section = vocabulary_markdown.split("## RGB / HSV conversion\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for required in (
        "px.color.rgb_to_hsv",
        "px.color.hsv_to_rgb",
        '("H", "S", "V")',
        '("R", "G", "B")',
        "maximum",
        "minimum",
        "delta",
        "modulo 1",
        "[0, 1)",
        "[0, 1]",
        "上限なし",
        "負値",
        "clip",
    ):
        assert required in section


@pytest.mark.parametrize("operation", ("rgb_to_hsv", "hsv_to_rgb"))
def test_hsv_docstrings_state_the_complete_public_contract(operation: str) -> None:
    """v1-hsv acceptance 12: each public docstring is self-contained for invisible numeric and Frame rules."""
    docstring = inspect.getdoc(getattr(px.color, operation)) or ""
    for required in (
        "float32",
        "Frame",
        "label",
        "H",
        "S",
        "V",
        "modulo",
        "scene",
        "negative",
        "clip",
        "domain",
        "matrix",
        "None",
        "unchanged",
        "px.values.cast_dtype",
        "px.values.recode_dtype",
        "px.values.dequantize",
    ):
        assert required in docstring
