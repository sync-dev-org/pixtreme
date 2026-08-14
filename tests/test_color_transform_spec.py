"""Specification tests for rgb_to_rgb."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _frame(
    values: Any,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | list[str] = "RGB",
    dtype: Any = np.float32,
) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    return px.io.from_array(cp.asarray(array), colorspace=colorspace, gamma=gamma, channels=channels)


def _gray_frame(values: tuple[float, ...], *, gamma: str = "linear") -> px.core.Frame:
    gray = np.repeat(np.asarray(values, dtype=np.float32)[None, :, None], 3, axis=2)
    return _frame(gray, gamma=gamma)


_TRANSFER_FIXTURES = (
    ("linear", (0.0, 0.18, 1.0), (0.0, 0.18, 1.0), 2e-7),
    # IEC 61966-2-1: the 0.04045 encoded breakpoint maps to 0.003130805 linear.
    ("srgb", (0.0, 0.04045, 0.5, 1.0), (0.0, 0.003130805, 0.21404114, 1.0), 2e-6),
    # BT.709 using the published 10-bit constants alpha=1.099 and beta=0.018.
    ("rec709", (0.0, 0.08124794, 0.5, 1.0), (0.0, 0.018, 0.2595894, 1.0), 2e-6),
    ("bt1886", (0.0, 0.25, 0.5, 1.0), (0.0, 0.03589682, 0.18946457, 1.0), 2e-6),
    # SMPTE ST 2084 normalized by its 10,000 cd/m2 peak.
    (
        "pq",
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (0.0, 0.0005154176, 0.009224571, 0.098337786, 1.0),
        8e-5,
    ),
    # ARIB STD-B67 / BT.2100 inverse OETF, including the encoded 0.5 branch boundary.
    ("hlg", (0.0, 0.5, 0.75, 1.0), (0.0, 1.0 / 12.0, 0.26496256, 1.0), 3e-6),
    # Sony's published normalized 10-bit code points: black, cut, 18% grey, and 90% white.
    (
        "s-log3",
        (95.0 / 1023.0, 171.2102946929 / 1023.0, 420.0 / 1023.0, 598.0 / 1023.0),
        (0.0, 0.01125, 0.18, 0.90083967),
        4e-6,
    ),
    # ARRI LogC4 equations 2/3: c encodes scene-linear zero.
    (
        "logc4",
        (95.0 / 1023.0, 0.2783958365, 0.4275193648),
        (0.0, 0.18, 1.0),
        5e-6,
    ),
    # Kodak Cineon: black CV=95, 18% scene-linear grey, and white CV=685.
    (
        "cineon",
        (95.0 / 1023.0, 0.4573196130854184, 685.0 / 1023.0),
        (0.0, 0.18, 1.0),
        5e-6,
    ),
    ("2.2", (0.0, 0.25, 0.5, 1.0), (0.0, 0.047366143, 0.21763764, 1.0), 2e-6),
    ("2.4", (0.0, 0.25, 0.5, 1.0), (0.0, 0.035896824, 0.18946457, 1.0), 2e-6),
    ("2.6", (0.0, 0.25, 0.5, 1.0), (0.0, 0.027204706, 0.16493849, 1.0), 2e-6),
)


@pytest.mark.parametrize(("gamma", "encoded", "linear", "atol"), _TRANSFER_FIXTURES)
def test_every_gamma_decodes_and_encodes_published_representative_and_boundary_points(
    gamma: str,
    encoded: tuple[float, ...],
    linear: tuple[float, ...],
    atol: float,
) -> None:
    """v1-color-semantics acceptance 26-28: all transfers match independent public-formula fixtures."""
    decoded = px.color.rgb_to_rgb(_gray_frame(encoded, gamma=gamma), output_gamma="linear")
    encoded_again = px.color.rgb_to_rgb(_gray_frame(linear), output_gamma=gamma)

    expected_linear = np.repeat(np.asarray(linear, dtype=np.float32)[None, :, None], 3, axis=2)
    expected_encoded = np.repeat(np.asarray(encoded, dtype=np.float32)[None, :, None], 3, axis=2)
    np.testing.assert_allclose(
        px.io.to_array(
            decoded,
        ).get(),
        expected_linear,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        px.io.to_array(
            encoded_again,
        ).get(),
        expected_encoded,
        rtol=0.0,
        atol=atol,
    )


def test_cineon_matches_independent_numpy_float64_kodak_equations() -> None:
    """v1-dpx acceptance 5 and 13: Cineon encode/decode matches an independent host float64 equation oracle."""
    black_offset = np.float64(10.0) ** ((np.float64(95.0) - np.float64(685.0)) / np.float64(300.0))
    encoded = np.asarray((95.0, 250.0, 445.0, 685.0, 750.0), dtype=np.float64) / np.float64(1023.0)
    expected_linear = (
        np.power(np.float64(10.0), (np.float64(1023.0) * encoded - np.float64(685.0)) / np.float64(300.0))
        - black_offset
    ) / (np.float64(1.0) - black_offset)
    linear = np.asarray((0.0, 0.01, 0.18, 1.0, 1.5), dtype=np.float64)
    expected_encoded = (
        np.float64(300.0) * np.log10(linear * (np.float64(1.0) - black_offset) + black_offset) + np.float64(685.0)
    ) / np.float64(1023.0)

    decoded = px.color.rgb_to_rgb(_gray_frame(tuple(encoded), gamma="cineon"), output_gamma="linear")
    encoded_frame = px.color.rgb_to_rgb(_gray_frame(tuple(linear)), output_gamma="cineon")

    np.testing.assert_allclose(
        px.io.to_array(
            decoded,
        ).get()[0, :, 0],
        expected_linear,
        rtol=0.0,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        px.io.to_array(
            encoded_frame,
        ).get()[0, :, 0],
        expected_encoded,
        rtol=0.0,
        atol=5e-6,
    )


@pytest.mark.parametrize("gamma", tuple(fixture[0] for fixture in _TRANSFER_FIXTURES))
def test_every_gamma_decode_encode_round_trip_stays_within_float32_transfer_error(gamma: str) -> None:
    """v1-color-semantics acceptance 26-28: decode then encode round-trips within float32 error."""
    fixture = next(candidate for candidate in _TRANSFER_FIXTURES if candidate[0] == gamma)
    # BT.709's published rounded 10-bit constants have a deliberate 0.000248
    # discontinuity at the branch boundary, so that boundary is tested against
    # each published branch above rather than treated as a round-trip point.
    encoded = tuple(value for index, value in enumerate(fixture[1]) if not (gamma == "rec709" and index == 1))
    source = _gray_frame(encoded, gamma=gamma)

    decoded = px.color.rgb_to_rgb(source, output_gamma="linear")
    restored = px.color.rgb_to_rgb(decoded, output_gamma=gamma)

    # 8e-5 covers ST 2084's steep float32 exponentiation near black. The other
    # curves are materially tighter, but one documented bound keeps the invariant explicit.
    np.testing.assert_allclose(
        px.io.to_array(
            restored,
        ).get(),
        px.io.to_array(
            source,
        ).get(),
        rtol=0.0,
        atol=8e-5,
    )


@pytest.mark.parametrize("gamma", ("pq", "s-log3", "logc4", "cineon", "2.2", "2.4", "2.6"))
def test_log_and_pure_power_extensions_are_sign_preserving_mirrors(gamma: str) -> None:
    """v1-color-semantics acceptance 26-28: log and pure-power extensions are sign-preserving."""
    source = _gray_frame((-0.18, 0.18))

    encoded = px.io.to_array(
        px.color.rgb_to_rgb(source, output_gamma=gamma),
    ).get()[0, :, 0]
    assert encoded[0] == pytest.approx(-encoded[1], abs=6e-6)

    restored = px.color.rgb_to_rgb(
        _gray_frame((float(encoded[0]), float(encoded[1])), gamma=gamma), output_gamma="linear"
    )
    np.testing.assert_allclose(
        px.io.to_array(
            restored,
        ).get()[0, :, 0],
        np.array([-0.18, 0.18]),
        rtol=0.0,
        atol=8e-5,
    )


@pytest.mark.parametrize(
    ("gamma", "linear", "encoded"),
    (
        ("srgb", -0.01, -0.1292),
        ("rec709", -0.01, -0.045),
        ("bt1886", -0.25, -0.561231),
        ("hlg", -(1.0 / 12.0), -0.5),
    ),
)
def test_piecewise_transfer_extensions_follow_their_natural_negative_branch(
    gamma: str, linear: float, encoded: float
) -> None:
    """v1-color-transform acceptance 7: piecewise curves extend naturally without clipping negative values."""
    result = px.color.rgb_to_rgb(_gray_frame((linear,)), output_gamma=gamma)
    assert px.io.to_array(
        result,
    ).get()[0, 0, 0] == pytest.approx(encoded, abs=3e-6)

    restored = px.color.rgb_to_rgb(result, output_gamma="linear")
    assert px.io.to_array(
        restored,
    ).get()[0, 0, 0] == pytest.approx(linear, abs=3e-6)


def test_transfer_functions_preserve_scene_overshoot_without_clipping() -> None:
    """v1-color-transform acceptance 7: negative and above-one scene values survive a transfer round trip."""
    source = _gray_frame((-0.25, 1.5))

    encoded = px.color.rgb_to_rgb(source, output_gamma="2.2")
    assert (
        px.io.to_array(
            encoded,
        ).get()[0, 0, 0]
        < 0.0
    )
    assert (
        px.io.to_array(
            encoded,
        ).get()[0, 1, 0]
        > 1.0
    )
    restored = px.color.rgb_to_rgb(encoded, output_gamma="linear")

    np.testing.assert_allclose(
        px.io.to_array(
            restored,
        ).get(),
        px.io.to_array(
            source,
        ).get(),
        rtol=0.0,
        atol=3e-6,
    )


@pytest.mark.parametrize("gamma", tuple(fixture[0] for fixture in _TRANSFER_FIXTURES))
def test_every_transfer_preserves_negative_and_above_one_scene_values(gamma: str) -> None:
    """v1-color-transform acceptance 7: every transfer token preserves signed scene-range excursions."""
    source = _gray_frame((-0.18, 1.5))

    encoded = px.color.rgb_to_rgb(source, output_gamma=gamma)
    restored = px.color.rgb_to_rgb(encoded, output_gamma="linear")

    assert np.isfinite(
        px.io.to_array(
            encoded,
        ).get()
    ).all()
    np.testing.assert_allclose(
        px.io.to_array(
            restored,
        ).get(),
        px.io.to_array(
            source,
        ).get(),
        rtol=0.0,
        atol=2e-4,
    )


_RED_TO_REC709_FIXTURES = (
    ("sRGB", (1.0, 0.0, 0.0)),
    ("Rec.709", (1.0, 0.0, 0.0)),
    ("Rec.2020", (1.6604910, -0.12455047, -0.01815076)),
    ("ACES2065-1", (2.5216862, -0.2764799, -0.01537807)),
    ("ACEScg", (1.7050510, -0.13025641, -0.02400336)),
    ("S-Gamut3", (1.8779151, -0.17680699, -0.02620113)),
    ("S-Gamut3.Cine", (1.6269474, -0.17851552, -0.04443612)),
)


@pytest.mark.parametrize(("colorspace", "expected_red"), _RED_TO_REC709_FIXTURES)
def test_every_colorspace_uses_its_published_primaries_and_bradford_white_adaptation(
    colorspace: str, expected_red: tuple[float, float, float]
) -> None:
    """v1-color-transform acceptance 8: all primaries and white points match independent matrix fixtures."""
    red = _frame([1.0, 0.0, 0.0], colorspace=colorspace)
    white = _frame([1.0, 1.0, 1.0], colorspace=colorspace)

    converted_red = px.color.rgb_to_rgb(red, output_colorspace="Rec.709")
    converted_white = px.color.rgb_to_rgb(white, output_colorspace="Rec.709")

    # Values were hand-derived in float64 from each published xy primary/white
    # table and the Bradford cone-response matrix, then fixed independently here.
    np.testing.assert_allclose(
        px.io.to_array(
            converted_red,
        ).get()[0, 0],
        expected_red,
        rtol=0.0,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        px.io.to_array(
            converted_white,
        ).get()[0, 0],
        (1.0, 1.0, 1.0),
        rtol=0.0,
        atol=5e-6,
    )


def test_srgb_and_rec709_colorspace_conversion_is_value_exact_identity() -> None:
    """v1-color-transform acceptance 9: sRGB and Rec.709 are exactly equivalent primaries and white."""
    values = np.array([[[-0.25, 0.125, 1.5], [0.2, 0.4, 0.8]]], dtype=np.float32)
    source = _frame(values, colorspace="sRGB")

    result = px.color.rgb_to_rgb(source, output_colorspace="Rec.709")

    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        values,
    )
    assert result.colorspace == "Rec.709"


@pytest.mark.parametrize("colorspace", tuple(fixture[0] for fixture in _RED_TO_REC709_FIXTURES))
def test_every_colorspace_round_trip_stays_within_accumulated_float32_matrix_error(colorspace: str) -> None:
    """v1-color-transform acceptance 10: A-to-B-to-A round trips remain within two float32 matrix passes."""
    values = np.array([[[-0.25, 0.125, 1.5], [0.2, 0.4, 0.8]]], dtype=np.float32)
    source = _frame(values, colorspace=colorspace)
    intermediate = "ACEScg" if colorspace != "ACEScg" else "Rec.709"

    converted = px.color.rgb_to_rgb(source, output_colorspace=intermediate)
    restored = px.color.rgb_to_rgb(converted, output_colorspace=colorspace)

    # 1e-5 is about 84 ulp at magnitude 1 and covers two condition-dependent
    # float32 3x3 products (including Bradford) without hiding visible error.
    np.testing.assert_allclose(
        px.io.to_array(
            restored,
        ).get(),
        values,
        rtol=0.0,
        atol=1e-5,
    )


def test_input_claims_override_metadata_without_mutating_the_input_frame() -> None:
    """v1-color-transform acceptance 2 and 4: per-call input claims win while input state stays immutable."""
    source = _gray_frame((0.5,), gamma="linear")
    original = (
        px.io.to_array(
            source,
        )
        .get()
        .copy()
    )

    result = px.color.rgb_to_rgb(source, input_gamma="srgb", output_gamma="linear")

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        np.full((1, 1, 3), 0.21404114),
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            source,
        ).get(),
        original,
    )
    assert (source.colorspace, source.gamma, source.channels) == ("sRGB", "linear", ("R", "G", "B"))
    assert (result.colorspace, result.gamma, result.channels) == ("sRGB", "linear", ("R", "G", "B"))


def test_input_colorspace_claim_overrides_metadata_without_relabeling_the_input() -> None:
    """v1-color-transform acceptance 2: an input colorspace claim drives the matrix but not Frame mutation."""
    source = _frame([1.0, 0.0, 0.0], colorspace="ACEScg")

    result = px.color.rgb_to_rgb(source, input_colorspace="Rec.2020", output_colorspace="Rec.709")

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get()[0, 0],
        (1.6604910, -0.12455047, -0.01815076),
        rtol=0.0,
        atol=5e-6,
    )
    assert source.colorspace == "ACEScg"
    assert result.colorspace == "Rec.709"


def test_partial_output_changes_only_that_axis_and_updates_its_metadata() -> None:
    """v1-color-transform acceptance 3 and 4: output omission preserves the other axis and all channels."""
    source = _gray_frame((0.5,), gamma="srgb")

    result = px.color.rgb_to_rgb(source, output_gamma="linear")

    assert (result.colorspace, result.gamma, result.channels) == ("sRGB", "linear", source.channels)
    assert (source.colorspace, source.gamma, source.channels) == ("sRGB", "srgb", ("R", "G", "B"))


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {
            "input_colorspace": "sRGB",
            "input_gamma": "linear",
            "output_colorspace": "sRGB",
            "output_gamma": "linear",
        },
    ),
)
def test_omitted_or_equal_outputs_return_a_new_frame_no_op(kwargs: dict[str, str]) -> None:
    """v1-color-transform acceptance 3: omitted or equal output axes allocate a value-preserving Frame."""
    source = _frame([0.25, 0.5, 0.75])

    result = px.color.rgb_to_rgb(source, **kwargs)

    assert result is not source
    assert result.data.data.ptr != source.data.data.ptr
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        px.io.to_array(
            source,
        ).get(),
    )
    assert (result.colorspace, result.gamma, result.channels) == (source.colorspace, source.gamma, source.channels)


def test_combined_colorspace_and_gamma_transform_matches_decode_then_matrix_fixture() -> None:
    """v1-color-transform acceptance 11: a combined call applies decode, matrix, then encode in that order."""
    source = _frame([0.5, 0.0, 0.0], colorspace="Rec.709", gamma="srgb")

    result = px.color.rgb_to_rgb(source, output_colorspace="ACEScg", output_gamma="linear")

    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get()[0, 0],
        (0.13122807, 0.01502434, 0.00441259),
        rtol=0.0,
        atol=3e-6,
    )


def test_color_transform_kernel_is_one_fused_decode_matrix_encode_pass() -> None:
    """v1-color-transform acceptance 11 and v1-tonemap-bt2408 acceptance 8: one fused transform pass."""
    import pixtreme._color.transform as implementation

    transform_source = inspect.getsource(implementation._transform_data)
    kernel_source = implementation._COLOR_TRANSFORM_KERNEL

    assert transform_source.count("_color_transform_kernel()(") == 1
    assert "Frame(" not in transform_source
    assert kernel_source.index("const float linear_red = decode_transfer") < kernel_source.index(
        "const float transformed_red ="
    )
    assert kernel_source.index("const float transformed_red =") < kernel_source.index("const float scaled_red =")
    assert kernel_source.index("const float scaled_red =") < kernel_source.index("encode_transfer(scaled_red")


def test_color_transform_docstring_warns_that_split_partial_calls_add_passes() -> None:
    """v1-color-transform acceptance 11: API docs identify the pass cost of splitting a combined transform."""
    docstring = inspect.getdoc(px.color.rgb_to_rgb)
    assert docstring is not None
    for required in ("single fused pass", "separate partial calls", "additional passes"):
        assert required in docstring


def test_color_transform_docstring_scopes_rendering_to_tonemap_selection() -> None:
    """REQ-API-003: the opening contract distinguishes technical conversion, rendering, and direct mapping."""
    docstring = inspect.getdoc(px.color.rgb_to_rgb)
    assert docstring is not None
    opening = " ".join(docstring.split("\n\n", maxsplit=1)[0].split())
    for required in ("tonemap=None", "without rendering", "rendering", "direct mapping"):
        assert required in opening


def test_rgb_labels_drive_conversion_and_non_rgb_labels_pass_through_exactly() -> None:
    """v1-color-transform acceptance 12: RGB is transformed by label while Z and A remain bit-exact."""
    source = _frame([9.0, 0.3, 0.8, 0.1, 0.2], gamma="2.2", channels=["Z", "B", "A", "R", "G"])

    result = px.color.rgb_to_rgb(source, output_gamma="linear")
    output = px.io.to_array(
        result,
    ).get()[0, 0]

    assert result.channels == ("Z", "B", "A", "R", "G")
    assert output[0] == np.float32(9.0)
    assert output[2] == np.float32(0.8)
    np.testing.assert_allclose(output[[1, 3, 4]], np.power([0.3, 0.1, 0.2], 2.2), rtol=0.0, atol=2e-6)


@pytest.mark.parametrize("dtype", (np.float16, np.uint8, np.uint16))
def test_color_transform_rejects_non_float32_actionably(dtype: type[np.generic]) -> None:
    """REQ-API-012 / v1-color-transform acceptance 12: non-fp32 input names a concrete recovery route."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(_frame([0, 0, 0], dtype=dtype), output_gamma="linear")
    _assert_actionable(error)
    assert str(np.dtype(dtype)) in str(error.value)


@pytest.mark.parametrize("channels", ("YCbCr", "RG", ["R", "G", "A"]))
def test_color_transform_rejects_frames_without_all_rgb_labels(channels: str | list[str]) -> None:
    """REQ-API-012 / v1-color-transform acceptance 12: missing RGB labels fail actionably."""
    source = _frame(np.zeros(len(px.core.channels(channels)), dtype=np.float32), channels=channels)

    with pytest.raises(ValueError, match="R, G, and B") as error:
        px.color.rgb_to_rgb(source, output_gamma="linear")
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("input_colorspace", "rec709"),
        ("output_colorspace", "ACES"),
        ("input_gamma", "sRGB"),
        ("output_gamma", "gamma2.2"),
    ),
)
def test_color_transform_rejects_unknown_axis_tokens_case_sensitively(parameter: str, value: str) -> None:
    """v1-color-transform acceptance 2 and 4: per-call axis tokens use the canonical case-sensitive vocabulary."""
    with pytest.raises(ValueError, match=parameter):
        px.color.rgb_to_rgb(_frame([0.1, 0.2, 0.3]), **{parameter: value})
