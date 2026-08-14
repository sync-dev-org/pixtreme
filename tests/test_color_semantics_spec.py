"""Specification tests for the declarative color API."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

MATRIX_TOKENS = ("bt601", "bt709", "bt2020", "native")

_COLORSPACE_XY = {
    "sRGB": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290)),
    "Rec.709": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290)),
    "Rec.2020": (((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), (0.3127, 0.3290)),
    "ACES2065-1": (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), (0.32168, 0.33767)),
    "ACEScg": (((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), (0.32168, 0.33767)),
    "S-Gamut3": (((0.730, 0.280), (0.140, 0.855), (0.100, -0.050)), (0.3127, 0.3290)),
    "S-Gamut3.Cine": (((0.766, 0.275), (0.225, 0.800), (0.089, -0.087)), (0.3127, 0.3290)),
}


def _frame(
    values: object,
    *,
    colorspace: str = "Rec.709",
    gamma: str = "rec709",
    channels: tuple[str, ...] = ("R", "G", "B"),
    matrix: str | None = None,
) -> px.core.Frame:
    data = cp.asarray(values, dtype=cp.float32).reshape(1, -1, len(channels))
    return px.core.Frame(data=data, colorspace=colorspace, gamma=gamma, channels=channels, matrix=matrix)


def _h273(rgb: np.ndarray, kr: float, kb: float) -> np.ndarray:
    red, green, blue = rgb.astype(np.float64)
    y = kr * red + (1.0 - kr - kb) * green + kb * blue
    return np.asarray((y, (blue - y) / (2.0 * (1.0 - kb)) + 0.5, (red - y) / (2.0 * (1.0 - kr)) + 0.5))


def _independent_own_row(colorspace: str) -> np.ndarray:
    primaries, (white_x, white_y) = _COLORSPACE_XY[colorspace]
    primary_matrix = np.asarray(
        (
            tuple(x / y for x, y in primaries),
            (1.0, 1.0, 1.0),
            tuple((1.0 - x - y) / y for x, y in primaries),
        ),
        dtype=np.float64,
    )
    white_xyz = np.asarray((white_x / white_y, 1.0, (1.0 - white_x - white_y) / white_y), dtype=np.float64)
    scales = np.linalg.solve(primary_matrix, white_xyz)
    return (primary_matrix @ np.diag(scales))[1]


def test_frame_matrix_is_independent_mutable_metadata() -> None:
    """v1-color-semantics acceptance 1-3: matrix is validated but independent mutable metadata."""
    frame = _frame((0.1, 0.2, 0.3), matrix="native")

    assert frame.matrix == "native"
    assert frame.model_dump()["matrix"] == "native"
    assert "matrix='native'" in repr(frame)

    frame.colorspace = "ACEScg"
    frame.gamma = "linear"
    frame.channels = ("Y", "Cb", "Cr")
    frame.matrix = "bt601"
    assert frame.matrix == "bt601"

    with pytest.raises(ValueError, match="matrix"):
        frame.matrix = "BT709"


def test_from_array_stamps_matrix_without_changing_ownership_or_values() -> None:
    """v1-color-semantics acceptance 4: from_array stamps matrix without changing array behavior."""
    data = cp.arange(12, dtype=cp.float32).reshape(2, 2, 3)
    frame = px.io.from_array(
        data,
        colorspace="Rec.709",
        gamma="rec709",
        channels="RGB",
        matrix="bt709",
        copy=False,
    )

    assert frame.matrix == "bt709"
    assert frame.data.data.ptr == data.data.ptr
    cp.testing.assert_array_equal(frame.data, data)


def test_from_format_entry_points_accept_matrix_none_by_default() -> None:
    """v1-color-semantics acceptance 5: all YCbCr unpackers expose matrix=None."""
    names = (
        "from_uyvy422",
        "from_v210",
        "from_nv12",
        "from_p010",
        "from_yuv420p",
        "from_yuv422p",
        "from_yuv444p",
        "from_yuva444p",
    )

    for name in names:
        parameter = inspect.signature(getattr(px.io, name)).parameters["matrix"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_color_public_surface_uses_semantic_operation_names_only() -> None:
    """v1-color-semantics acceptance 9, 15, 20, 24-25, 33, 40 and v1-hsv acceptance 12: names are exact."""
    assert px.color.__all__ == (
        "apply_lut",
        "gamma_to_linear",
        "hsv_to_rgb",
        "linear_to_gamma",
        "rgb_to_grayscale",
        "rgb_to_hsv",
        "rgb_to_rgb",
        "rgb_to_ycbcr",
        "ycbcr_to_rgb",
        "ycbcr_to_ycbcr",
        "equalize_histogram",
        "clahe",
    )
    assert not hasattr(px.color, "view_transform")
    assert not hasattr(px.color, "channel_transform")


@pytest.mark.parametrize(
    ("matrix", "kr", "kb"),
    (("bt601", 0.299, 0.114), ("bt709", 0.2126, 0.0722), ("bt2020", 0.2627, 0.0593)),
)
def test_rgb_to_ycbcr_matches_h273_and_preserves_auxiliary_channels(matrix: str, kr: float, kb: float) -> None:
    """v1-color-semantics acceptance 6 and 9-14: RGB encoding uses H.273 and label-driven passthrough."""
    source = _frame((0.25, 9.0, -0.5, 0.75), channels=("B", "A", "R", "G"))

    result = px.color.rgb_to_ycbcr(source, matrix=matrix)

    expected = _h273(np.asarray((-0.5, 0.75, 0.25)), kr, kb)
    assert result.channels == ("Cr", "A", "Y", "Cb")
    assert result.matrix == matrix
    np.testing.assert_allclose(result.data.get()[0, 0, (2, 3, 0)], expected, rtol=2e-6, atol=2e-6)
    assert result.data.get()[0, 0, 1] == np.float32(9.0)


@pytest.mark.parametrize(
    ("colorspace", "gamma", "expected"),
    (
        ("sRGB", "srgb", "bt709"),
        ("Rec.709", "linear", "bt709"),
        ("Rec.2020", "linear", "bt2020"),
        ("ACEScg", "linear", "native"),
        ("ACEScg", "2.6", "bt709"),
    ),
)
def test_rgb_encode_matrix_resolver_is_representation_dependent(colorspace: str, gamma: str, expected: str) -> None:
    """v1-color-semantics acceptance 7-8 and 11-12: the encode resolver stamps its selected basis."""
    source = _frame((0.2, 0.4, 0.6), colorspace=colorspace, gamma=gamma)
    assert px.color.rgb_to_ycbcr(source).matrix == expected
    assert px.color.rgb_to_grayscale(source).matrix == expected


@pytest.mark.parametrize("colorspace", tuple(_COLORSPACE_XY))
def test_native_matrix_uses_independently_derived_primary_own_row(colorspace: str) -> None:
    """v1-color-semantics acceptance 3 and 7-8: native follows the current colorspace's independently derived Y row."""
    rgb = np.asarray((0.2, 1.1, -0.4), dtype=np.float64)
    source = _frame(rgb, colorspace=colorspace, gamma="linear")

    result = px.color.rgb_to_grayscale(source, matrix="native")

    expected = float(_independent_own_row(colorspace) @ rgb)
    assert result.data.get()[0, 0, 0] == pytest.approx(expected, abs=2e-6)


def test_decode_matrix_resolver_honors_override_then_metadata_and_refuses_unsafe_guess() -> None:
    """v1-color-semantics acceptance 16 and 42: decode resolution is explicit, provenance-aware, and conservative."""
    source = _frame((0.15, 0.7, 1.3), colorspace="ACEScg", gamma="linear")
    encoded = px.color.rgb_to_ycbcr(source, matrix="native")
    misleading = encoded.model_copy(update={"matrix": "bt601"})

    explicit = px.color.ycbcr_to_rgb(misleading, matrix="native")
    metadata = px.color.ycbcr_to_rgb(encoded)
    cp.testing.assert_allclose(explicit.data, source.data, rtol=3e-5, atol=3e-5)
    cp.testing.assert_allclose(metadata.data, source.data, rtol=3e-5, atol=3e-5)

    unknown = encoded.model_copy(update={"matrix": None})
    with pytest.raises(ValueError) as captured:
        px.color.ycbcr_to_rgb(unknown)
    message = str(captured.value)
    assert "why=" in message and "what=" in message and "how=" in message
    assert "matrix=" in message and "bt709" in message


def test_rgb_ycbcr_round_trip_preserves_scene_values_and_metadata() -> None:
    """v1-color-semantics acceptance 15-19: the paired conversion round-trips unrestricted values."""
    source = _frame((-0.25, 0.5, 1.75, 7.0), channels=("R", "G", "B", "Z"))
    encoded = px.color.rgb_to_ycbcr(source, matrix="bt709", range="legal", bit_depth=10)
    restored = px.color.ycbcr_to_rgb(encoded, range="legal", bit_depth=10)

    assert restored.channels == source.channels
    assert restored.colorspace == source.colorspace
    assert restored.gamma == source.gamma
    assert restored.matrix is None
    cp.testing.assert_allclose(restored.data, source.data, rtol=3e-5, atol=3e-5)


def test_rgb_legal_encode_matches_full_encode_then_public_range_conversion() -> None:
    """v1-color-semantics acceptance 10 and 13: legal RGB encode matches the public one-way composition."""
    source = _frame((-0.2, 0.45, 1.4), colorspace="S-Gamut3", gamma="s-log3")

    result = px.color.rgb_to_ycbcr(
        source,
        colorspace="ACEScg",
        gamma="linear",
        matrix="native",
        range="legal",
        bit_depth=10,
    )
    full = px.color.rgb_to_ycbcr(
        source,
        colorspace="ACEScg",
        gamma="linear",
        matrix="native",
    )
    expected = px.values.full_to_legal(full, bit_depth=10)

    assert result.matrix == expected.matrix == "native"
    cp.testing.assert_allclose(result.data, expected.data, rtol=0.0, atol=2e-7)


def test_ycbcr_legal_decode_matches_public_range_conversion_then_full_decode() -> None:
    """v1-color-semantics acceptance 17-18: legal YCbCr decode matches the public one-way composition."""
    source = _frame(
        (-0.1, 0.35, 1.2),
        colorspace="Rec.709",
        gamma="rec709",
        channels=("Y", "Cb", "Cr"),
        matrix="bt709",
    )

    result = px.color.ycbcr_to_rgb(
        source,
        colorspace="ACEScg",
        gamma="linear",
        range="legal",
        bit_depth=12,
    )
    full = px.values.legal_to_full(source, bit_depth=12)
    expected = px.color.ycbcr_to_rgb(full, colorspace="ACEScg", gamma="linear")

    assert result.matrix is expected.matrix is None
    cp.testing.assert_allclose(result.data, expected.data, rtol=0.0, atol=2e-6)


def test_ycbcr_to_ycbcr_matches_three_public_ops_across_different_legal_code_grids() -> None:
    """v1-color-semantics acceptance 43 and 46-47: both legal-range code grids compose independently."""
    source = _frame(
        (0.15, 0.7, 1.1),
        colorspace="Rec.709",
        gamma="rec709",
        channels=("Y", "Cb", "Cr"),
        matrix="bt709",
    )

    result = px.color.ycbcr_to_ycbcr(
        source,
        input_matrix="bt709",
        output_matrix="bt709",
        input_range="legal",
        input_bit_depth=10,
        output_range="legal",
        output_bit_depth=12,
    )
    rgb = px.color.ycbcr_to_rgb(source, matrix="bt709", range="legal", bit_depth=10)
    transformed = px.color.rgb_to_rgb(
        rgb,
        output_colorspace=source.colorspace,
        output_gamma=source.gamma,
    )
    expected = px.color.rgb_to_ycbcr(transformed, matrix="bt709", range="legal", bit_depth=12)

    cp.testing.assert_allclose(result.data, expected.data, rtol=0.0, atol=2e-6)


def test_ycbcr_to_ycbcr_matches_three_public_ops_for_explicit_bt709_to_native_rematrix() -> None:
    """v1-color-semantics acceptance 47-48: explicit input and output matrices compose for a pure rematrix."""
    source = _frame(
        (0.15, 0.7, 1.1),
        colorspace="S-Gamut3",
        gamma="s-log3",
        channels=("Y", "Cb", "Cr"),
        matrix="bt709",
    )

    result = px.color.ycbcr_to_ycbcr(source, input_matrix="bt709", output_matrix="native")
    rgb = px.color.ycbcr_to_rgb(source, matrix="bt709")
    transformed = px.color.rgb_to_rgb(
        rgb,
        output_colorspace=source.colorspace,
        output_gamma=source.gamma,
    )
    expected = px.color.rgb_to_ycbcr(transformed, matrix="native")

    assert result.matrix == expected.matrix == "native"
    cp.testing.assert_allclose(result.data, expected.data, rtol=0.0, atol=2e-6)


@pytest.mark.parametrize(
    ("parameter", "invalid"),
    (
        ("input_range", "FULL"),
        ("input_range", True),
        ("output_range", "FULL"),
        ("output_range", True),
        ("input_bit_depth", True),
        ("input_bit_depth", "10"),
        ("input_bit_depth", 9),
        ("output_bit_depth", True),
        ("output_bit_depth", "10"),
        ("output_bit_depth", 9),
    ),
)
def test_ycbcr_to_ycbcr_validates_each_range_and_bit_depth_axis(parameter: str, invalid: object) -> None:
    """v1-color-semantics acceptance 46: all four range/code-grid arguments reject values outside their domains."""
    source = _frame(
        (0.2, 0.5, 0.8),
        channels=("Y", "Cb", "Cr"),
        matrix="bt709",
    )

    with pytest.raises(ValueError) as captured:
        px.color.ycbcr_to_ycbcr(source, **{parameter: invalid})

    message = str(captured.value)
    assert "why=" in message and "what=" in message and "how=" in message


@pytest.mark.parametrize("direction", ("encode", "decode"))
def test_declarative_ycbcr_conversion_matches_a_separated_rgb_to_rgb_call(direction: str) -> None:
    """v1-color-semantics acceptance 10 and 18-19: fused declarations match a separated technical conversion."""
    if direction == "encode":
        source = _frame((-0.2, 0.45, 1.4), colorspace="S-Gamut3", gamma="s-log3")
        result = px.color.rgb_to_ycbcr(
            source,
            colorspace="ACEScg",
            gamma="linear",
            matrix="native",
        )
        transformed = px.color.rgb_to_rgb(source, output_colorspace="ACEScg", output_gamma="linear")
        expected = px.color.rgb_to_ycbcr(transformed, matrix="native")
    else:
        source = _frame(
            (-0.1, 0.35, 1.2),
            colorspace="S-Gamut3",
            gamma="s-log3",
            channels=("Y", "Cb", "Cr"),
            matrix="bt709",
        )
        result = px.color.ycbcr_to_rgb(
            source,
            colorspace="ACEScg",
            gamma="linear",
            matrix="bt709",
        )
        decoded = px.color.ycbcr_to_rgb(source, matrix="bt709")
        expected = px.color.rgb_to_rgb(decoded, output_colorspace="ACEScg", output_gamma="linear")

    cp.testing.assert_allclose(result.data, expected.data, rtol=0.0, atol=2e-6)


def test_rgb_to_grayscale_is_the_full_range_y_channel_bit_for_bit() -> None:
    """v1-color-semantics acceptance 20-23: grayscale is the matching Y projection with no auxiliary labels."""
    source = _frame((0.2, 1.2, -0.1, 42.0), colorspace="ACEScg", gamma="linear", channels=("R", "G", "B", "Z"))
    encoded = px.color.rgb_to_ycbcr(source, matrix="native")
    gray = px.color.rgb_to_grayscale(source, matrix="native")

    assert gray.channels == ("Y",)
    assert gray.shape == (1, 1, 1)
    assert gray.matrix == "native"
    cp.testing.assert_array_equal(gray.data[..., 0], encoded.data[..., encoded.channels.index("Y")])


def test_gamma_directional_pair_supports_pure_power_2_6_without_clipping() -> None:
    """v1-color-semantics acceptance 24-28: gamma pairs expose pure-power 2.6 and preserve scene values."""
    encoded_values = np.asarray((-1.4, -0.25, 2.0), dtype=np.float64)
    source = _frame(encoded_values, gamma="2.6", matrix="native")

    linear = px.color.gamma_to_linear(source)
    expected = np.sign(encoded_values) * np.abs(encoded_values) ** 2.6
    np.testing.assert_allclose(linear.data.get()[0, 0], expected, rtol=2e-6, atol=2e-6)
    assert linear.gamma == "linear"
    assert linear.matrix is None

    restored = px.color.linear_to_gamma(linear, gamma="2.6")
    np.testing.assert_allclose(restored.data.get()[0, 0], encoded_values, rtol=2e-6, atol=2e-6)
    assert restored.gamma == "2.6"
    assert restored.matrix is None


def test_rgb_to_rgb_signature_integrates_tonemap_and_always_clears_matrix() -> None:
    """v1-color-semantics acceptance 29-33: rgb_to_rgb owns tonemap and clears matrix provenance."""
    signature = inspect.signature(px.color.rgb_to_rgb)
    assert tuple(signature.parameters) == (
        "frame",
        "input_colorspace",
        "input_gamma",
        "output_colorspace",
        "output_gamma",
        "tonemap",
    )
    assert signature.parameters["tonemap"].default is None

    source = _frame((-0.1, 0.5, 1.2), matrix="bt709")
    technical = px.color.rgb_to_rgb(source)
    assert technical.matrix is None
    cp.testing.assert_array_equal(technical.data, source.data)


def test_ycbcr_to_ycbcr_matches_public_composition_and_preserves_auxiliary_values() -> None:
    """v1-color-semantics acceptance 40-48: fused YCbCr conversion matches the declared public composition."""
    source = _frame(
        (0.35, 0.1, 0.9, 6.0),
        colorspace="Rec.709",
        gamma="rec709",
        channels=("Y", "Cb", "Cr", "A"),
        matrix="bt709",
    )
    result = px.color.ycbcr_to_ycbcr(
        source,
        colorspace="ACEScg",
        gamma="linear",
        output_matrix="native",
        input_range="legal",
        input_bit_depth=10,
        output_range="full",
    )
    rgb = px.color.ycbcr_to_rgb(source, matrix="bt709", range="legal", bit_depth=10)
    transformed = px.color.rgb_to_rgb(rgb, output_colorspace="ACEScg", output_gamma="linear")
    expected = px.color.rgb_to_ycbcr(transformed, matrix="native")

    assert result.channels == source.channels
    assert result.colorspace == "ACEScg"
    assert result.gamma == "linear"
    assert result.matrix == "native"
    cp.testing.assert_allclose(result.data[..., :3], expected.data[..., :3], rtol=3e-5, atol=3e-5)
    cp.testing.assert_array_equal(result.data[..., 3], source.data[..., 3])


def test_ycbcr_to_ycbcr_preserves_resolved_input_matrix_when_colorspace_is_unchanged() -> None:
    """v1-color-semantics acceptance 45 and 48: same-colorspace transfer changes retain the input matrix basis."""
    source = _frame(
        (0.35, 0.1, 0.9),
        colorspace="Rec.709",
        gamma="rec709",
        channels=("Y", "Cb", "Cr"),
        matrix="bt601",
    )

    result = px.color.ycbcr_to_ycbcr(source, gamma="2.6")

    assert result.matrix == "bt601"
    assert result.colorspace == source.colorspace
    assert result.gamma == "2.6"


def test_ycbcr_to_ycbcr_docstring_is_symmetric_with_the_directional_pair() -> None:
    """v1-color-semantics acceptance 40-48: the fused API documents both independent sides and resolvers."""
    docstring = " ".join((inspect.getdoc(px.color.ycbcr_to_ycbcr) or "").split())
    for required in (
        "Parameters",
        "input_matrix",
        "frame.matrix",
        "input_range",
        "input_bit_depth",
        "output_matrix",
        "output_range",
        "output_bit_depth",
        "same colorspace",
        "Returns",
        "Raises",
        "ValueError",
        'input_matrix="bt709"',
        'output_matrix="native"',
        "inverse rematrix",
    ):
        assert required in docstring


@pytest.mark.parametrize("operation", ("rgb_to_ycbcr", "ycbcr_to_rgb"))
@pytest.mark.parametrize(("parameter", "value"), (("range", "FULL"), ("bit_depth", True), ("bit_depth", 9)))
def test_directional_range_and_bit_depth_validation_is_closed(operation: str, parameter: str, value: object) -> None:
    """v1-color-semantics acceptance 14, 17, and 46: range and code-grid axes fail actionably."""
    channels = ("Y", "Cb", "Cr") if operation == "ycbcr_to_rgb" else ("R", "G", "B")
    frame = _frame((0.1, 0.2, 0.3), channels=channels, matrix="bt709")

    with pytest.raises(ValueError) as captured:
        getattr(px.color, operation)(frame, **{parameter: value})

    message = str(captured.value)
    assert "why=" in message and "what=" in message and "how=" in message


def test_every_non_source_frame_constructor_declares_matrix_provenance() -> None:
    """v1-color-semantics acceptance 39: Frame construction points cannot silently drop matrix provenance."""
    source_modules = {"_io.py"}
    source_root = Path(px.__file__).parent
    missing: list[str] = []
    for module in sorted(source_root.glob("*.py")):
        if module.name in source_modules:
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Frame":
                continue
            if not any(keyword.arg == "matrix" for keyword in node.keywords):
                missing.append(f"{module.name}:{node.lineno}")

    assert missing == []


@pytest.mark.parametrize("operation", ("rgb_to_ycbcr", "ycbcr_to_rgb", "rgb_to_grayscale"))
def test_color_validation_errors_are_actionable(operation: str) -> None:
    """v1-color-semantics acceptance 9, 15, and 20: invalid input errors carry why/what/how."""
    channels = ("R", "G", "B") if operation == "ycbcr_to_rgb" else ("Y", "Cb", "Cr")
    frame = _frame((0.1, 0.2, 0.3), channels=channels)

    with pytest.raises(ValueError) as captured:
        getattr(px.color, operation)(frame)

    message = str(captured.value)
    assert "why=" in message and "what=" in message and "how=" in message
