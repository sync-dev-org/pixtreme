"""Specification tests for reference-white device simulation."""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from pathlib import Path
from typing import get_args, get_type_hints

import numpy as np
import pytest

import pixtreme as px

_D65 = (0.3127, 0.3290)
_D93 = (0.2831, 0.2971)
_D50 = (0.3457, 0.3585)
_ACES = (0.32168, 0.33767)
_REFERENCE_WHITES = {"d65": _D65, "d93": _D93, "d50": _D50, "aces": _ACES}
_CAT_TOKENS = ("bradford", "cat02", "cat16", "von-kries")
_PRIMARIES = {
    "sRGB": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "ACES2065-1": ((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)),
}


def _frame(
    values: object,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: tuple[str, ...] = ("R", "G", "B"),
    dtype: np.dtype[object] | type[np.generic] = np.float32,
    matrix: str | None = "bt709",
) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array[None, None, :]
    return px.core.Frame(
        data=cp.asarray(array),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix=matrix,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert message.startswith("why=")
    assert "; what=" in message
    assert "; how=" in message
    assert message.index("why=") < message.index("; what=") < message.index("; how=")


def _xy_to_xyz(white: tuple[float, float]) -> np.ndarray:
    x, y = white
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _device_matrix(colorspace: str, white: tuple[float, float]) -> np.ndarray:
    primaries = _PRIMARIES[colorspace]
    unscaled = np.asarray(
        (
            tuple(x / y for x, y in primaries),
            (1.0, 1.0, 1.0),
            tuple((1.0 - x - y) / y for x, y in primaries),
        ),
        dtype=np.float64,
    )
    scale = np.linalg.solve(unscaled, _xy_to_xyz(white))
    return unscaled @ np.diag(scale)


def _simulation_matrix(
    colorspace: str,
    input_white: tuple[float, float],
    output_white: tuple[float, float],
) -> np.ndarray:
    return np.linalg.inv(_device_matrix(colorspace, output_white)) @ _device_matrix(colorspace, input_white)


def _decode_srgb(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    linear = values <= np.float64(0.04045)
    result[linear] = values[linear] / np.float64(12.92)
    result[~linear] = np.power(
        (values[~linear] + np.float64(0.055)) / np.float64(1.055),
        np.float64(2.4),
    )
    return result


def _encode_srgb(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    linear = values <= np.float64(0.0031308)
    result[linear] = np.float64(12.92) * values[linear]
    result[~linear] = np.float64(1.055) * np.power(
        values[~linear],
        np.float64(1.0) / np.float64(2.4),
    ) - np.float64(0.055)
    return result


def _expected(
    values: np.ndarray,
    *,
    colorspace: str,
    gamma: str,
    input_white: tuple[float, float],
    output_white: tuple[float, float],
) -> np.ndarray:
    decoded = _decode_srgb(values.astype(np.float64)) if gamma == "srgb" else values.astype(np.float64)
    transformed = np.einsum(
        "ij,...j->...i",
        _simulation_matrix(colorspace, input_white, output_white),
        decoded,
    )
    return _encode_srgb(transformed) if gamma == "srgb" else transformed


def test_public_surface_signature_alias_counts_and_docs_are_synchronized() -> None:
    """v1-white-point-simulation acceptance 1 and 13: API, alias, counts, and public docs agree."""
    assert get_args(px.core.ReferenceWhite) == ("d65", "d93", "d50", "aces")
    assert px.color.__all__[-1] == "white_point_simulation"
    assert len(px.color.__all__) == 15

    signature = inspect.signature(px.color.white_point_simulation)
    assert tuple(signature.parameters) == ("frame", "input_white", "output_white")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["input_white"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["output_white"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["input_white"].default is None
    assert signature.parameters["output_white"].default is inspect.Parameter.empty
    hints = get_type_hints(px.color.white_point_simulation)
    assert hints["input_white"] == px.core.ReferenceWhite | Sequence[float] | None
    assert hints["output_white"] == px.core.ReferenceWhite | Sequence[float]
    assert hints["return"] is px.core.Frame

    adaptation_signature = inspect.signature(px.color.chromatic_adaptation)
    assert adaptation_signature.parameters["input_white"].default is inspect.Parameter.empty
    assert adaptation_signature.parameters["output_white"].default is inspect.Parameter.empty
    assert adaptation_signature.parameters["cat"].default == "cat02"
    adaptation_hints = get_type_hints(px.color.chromatic_adaptation)
    assert adaptation_hints["input_white"] == px.core.ReferenceWhite | Sequence[float]
    assert adaptation_hints["output_white"] == px.core.ReferenceWhite | Sequence[float]

    root = Path(__file__).resolve().parents[1]
    requirements = (root / "docs" / "requirements.md").read_text(encoding="utf-8")
    tokens = (root / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    color_row = next(line for line in requirements.splitlines() if line.startswith("| `color` |"))
    assert "| 15 |" in color_row
    assert "公開 operation は計 94 関数" in requirements
    reference_section = tokens.split("## reference white\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    reference_table = reference_section.split("| Token |", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert tuple(
        line.split("|")[1].strip().strip("`") for line in reference_table.splitlines() if line.startswith("| `")
    ) == ("d65", "d93", "d50", "aces")
    assert "(0.3127, 0.3290)" in reference_section
    assert "(0.2831, 0.2971)" in reference_section
    assert "(0.3457, 0.3585)" in reference_section
    assert "(0.32168, 0.33767)" in reference_section
    assert "ICC absolute colorimetric intent" in tokens
    assert "Temperature / Tint" in tokens


@pytest.mark.parametrize(("token", "xy"), tuple(_REFERENCE_WHITES.items()))
def test_reference_white_tokens_are_case_sensitive_and_match_only_their_fixed_xy(
    token: str,
    xy: tuple[float, float],
) -> None:
    """v1-white-point-simulation acceptance 2: each closed token resolves to one specification-fixed xy."""
    import cupy as cp

    source = _frame((0.17, 0.41, 1.23))
    token_output = px.color.white_point_simulation(source, input_white="d65", output_white=token)
    xy_output = px.color.white_point_simulation(source, input_white=_D65, output_white=xy)
    assert cp.array_equal(token_output.data, xy_output.data)


@pytest.mark.parametrize(
    "distinct_xy",
    (
        (0.283145007105, 0.297112885246),
        (0.281, 0.311),
    ),
)
def test_noncanonical_d93_coordinates_remain_distinct_direct_xy_inputs(
    distinct_xy: tuple[float, float],
) -> None:
    """v1-white-point-simulation acceptance 2: unrounded daylight and 27 MPCD coordinates do not become d93."""
    import cupy as cp

    source = _frame((0.17, 0.41, 1.23))
    token_output = px.color.white_point_simulation(source, input_white="d65", output_white="d93")
    direct_output = px.color.white_point_simulation(source, input_white="d65", output_white=distinct_xy)
    assert not cp.array_equal(token_output.data, direct_output.data)


@pytest.mark.parametrize("cat", _CAT_TOKENS)
@pytest.mark.parametrize(("token", "xy"), tuple(_REFERENCE_WHITES.items()))
def test_chromatic_adaptation_tokens_match_direct_xy_for_every_cat(
    cat: str,
    token: str,
    xy: tuple[float, float],
) -> None:
    """v1-white-point-simulation acceptance 4: both white inputs share token resolution for all CATs."""
    import cupy as cp

    source = _frame((0.19, 0.53, 1.17), gamma="srgb")
    token_output = px.color.chromatic_adaptation(
        source,
        input_white=token,
        output_white="d93" if token == "d65" else "d65",
        cat=cat,
    )
    xy_output = px.color.chromatic_adaptation(
        source,
        input_white=list(xy),
        output_white=list(_D93 if token == "d65" else _D65),
        cat=cat,
    )
    assert cp.array_equal(token_output.data, xy_output.data)


@pytest.mark.parametrize(
    "invalid",
    (
        "D65",
        "D93",
        "D50",
        "ACES",
        "Aces",
        "d93-8mpcd",
        "d93-27mpcd",
        "cie-d93",
        None,
        object(),
        (0.3,),
        (0.3, 0.3, 0.4),
        (True, 0.3),
        ("0.3", 0.3),
        (math.nan, 0.3),
        (math.inf, 0.3),
        (0.0, 0.3),
        (0.3, 0.0),
        (0.6, 0.4),
    ),
)
def test_invalid_reference_whites_fail_actionably_before_pixel_processing(
    monkeypatch: pytest.MonkeyPatch,
    invalid: object,
) -> None:
    """v1-white-point-simulation acceptance 2-3: aliases, malformed xy, and invalid domains fail before GPU work."""
    import pixtreme._color.white_point as implementation

    def forbidden_transform(*args: object, **kwargs: object) -> object:
        raise AssertionError("pixel processing must not start for invalid arguments")

    monkeypatch.setattr(implementation, "_transform_data", forbidden_transform)
    with pytest.raises(ValueError) as error:
        px.color.white_point_simulation(
            _frame((0.2, 0.3, 0.4)),
            input_white="d65",
            output_white=invalid,  # type: ignore[arg-type]
        )
    _assert_actionable(error)


@pytest.mark.parametrize("invalid", ((0.640, 0.330), (5e-324, 5e-324)))
def test_nonfinite_or_singular_device_matrices_fail_before_pixel_processing(
    monkeypatch: pytest.MonkeyPatch,
    invalid: tuple[float, float],
) -> None:
    """v1-white-point-simulation acceptance 3: unconstructible device matrices fail actionably before GPU work."""
    import pixtreme._color.white_point as implementation

    def forbidden_transform(*args: object, **kwargs: object) -> object:
        raise AssertionError("pixel processing must not start for invalid device matrices")

    monkeypatch.setattr(implementation, "_transform_data", forbidden_transform)
    with pytest.raises(ValueError) as error:
        px.color.white_point_simulation(
            _frame((0.2, 0.3, 0.4)),
            input_white="d65",
            output_white=invalid,
        )
    _assert_actionable(error)


@pytest.mark.parametrize("invalid", ((0.640, 0.330), (5e-324, 5e-324)))
def test_equal_singular_whites_fail_before_the_identity_copy(
    monkeypatch: pytest.MonkeyPatch,
    invalid: tuple[float, float],
) -> None:
    """v1-white-point-simulation acceptance 3: equal unconstructible whites fail instead of returning an identity copy."""
    import pixtreme._color.white_point as implementation

    def forbidden_transform(*args: object, **kwargs: object) -> object:
        raise AssertionError("pixel processing must not start for invalid device matrices")

    monkeypatch.setattr(implementation, "_transform_data", forbidden_transform)
    with pytest.raises(ValueError) as error:
        px.color.white_point_simulation(
            _frame((0.2, 0.3, 0.4)),
            input_white=invalid,
            output_white=invalid,
        )
    _assert_actionable(error)


def test_implicit_input_white_uses_the_frame_colorspace_nominal_white() -> None:
    """v1-white-point-simulation acceptance 2-3: ACES token, omitted, and None match the Frame nominal white."""
    import cupy as cp

    source = _frame((0.18, 0.47, 1.31), colorspace="ACES2065-1")
    omitted = px.color.white_point_simulation(source, output_white="d93")
    explicit_none = px.color.white_point_simulation(source, input_white=None, output_white="d93")
    explicit_token = px.color.white_point_simulation(source, input_white="aces", output_white="d93")
    explicit_xy = px.color.white_point_simulation(
        source,
        input_white=_ACES,
        output_white=_D93,
    )
    assert cp.array_equal(omitted.data, explicit_none.data)
    assert cp.array_equal(omitted.data, explicit_token.data)
    assert cp.array_equal(omitted.data, explicit_xy.data)


@pytest.mark.parametrize(
    "invalid_frame",
    (
        object(),
        pytest.param("float16", id="float16"),
        pytest.param("missing-rgb", id="missing-rgb"),
    ),
)
def test_white_point_simulation_requires_one_float32_rgb_triplet(invalid_frame: object) -> None:
    """v1-white-point-simulation acceptance 5: the operation requires a float32 Frame with one RGB triplet."""
    if invalid_frame == "float16":
        invalid_frame = _frame((0.2, 0.3, 0.4), dtype=np.float16)
    elif invalid_frame == "missing-rgb":
        invalid_frame = _frame((0.2, 0.3), channels=("R", "G"))
    with pytest.raises(ValueError) as error:
        px.color.white_point_simulation(invalid_frame, output_white="d93")  # type: ignore[arg-type]
    _assert_actionable(error)


@pytest.mark.parametrize(("input_white", "output_white"), ((_D65, _D93), (_D93, _D65)))
@pytest.mark.parametrize("colorspace", ("sRGB", "ACES2065-1"))
def test_linear_output_matches_independent_device_matrix_and_preserves_signed_primary_scales(
    input_white: tuple[float, float],
    output_white: tuple[float, float],
    colorspace: str,
) -> None:
    """v1-white-point-simulation acceptance 6-7 and 11: device composition preserves scene values without CAT or clip."""
    values = np.asarray(
        [[[-0.20, 0.35, 1.40], [1.75, 0.08, 0.60], [0.04, 1.20, 0.22]]],
        dtype=np.float32,
    )
    matrix = _simulation_matrix(colorspace, input_white, output_white)
    np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)), rtol=0.0, atol=1e-14)
    if colorspace == "ACES2065-1":
        assert np.any(
            np.linalg.solve(
                np.asarray(
                    (
                        tuple(x / y for x, y in _PRIMARIES[colorspace]),
                        (1.0, 1.0, 1.0),
                        tuple((1.0 - x - y) / y for x, y in _PRIMARIES[colorspace]),
                    ),
                    dtype=np.float64,
                ),
                _xy_to_xyz(input_white),
            )
            < 0.0
        )
    source = _frame(values, colorspace=colorspace)
    actual = px.color.white_point_simulation(
        source,
        input_white=input_white,
        output_white=output_white,
    )
    expected = _expected(
        values,
        colorspace=colorspace,
        gamma="linear",
        input_white=input_white,
        output_white=output_white,
    )
    np.testing.assert_allclose(actual.data.get(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("input_white", "output_white"), (("d65", _D65), (_D93, "d93")))
def test_equal_resolved_whites_return_an_all_channel_bit_preserving_private_copy(
    input_white: object,
    output_white: object,
) -> None:
    """v1-white-point-simulation acceptance 8: token/xy identity preserves all channel bits in private storage."""
    import cupy as cp

    source = _frame(
        np.asarray([[[-0.0, math.nan, 1.25, -2.0], [math.inf, -math.inf, 0.5, 7.0]]], dtype=np.float32),
        channels=("B", "A", "R", "G"),
    )
    output = px.color.white_point_simulation(
        source,
        input_white=input_white,  # type: ignore[arg-type]
        output_white=output_white,  # type: ignore[arg-type]
    )
    assert not cp.shares_memory(output.data, source.data)
    assert output.data.view(cp.uint32).tobytes() == source.data.view(cp.uint32).tobytes()
    assert output.matrix is None


@pytest.mark.parametrize("gamma", ("linear", "srgb"))
def test_asymmetric_scene_values_match_host_oracle_and_reverse_round_trip(gamma: str) -> None:
    """v1-white-point-simulation acceptance 9 and 11: both transfer paths match an oracle and reverse compensation."""
    values = np.asarray(
        [[[-0.20, 0.35, 1.40], [1.75, 0.08, 0.60], [0.04, 1.20, 0.22]]],
        dtype=np.float32,
    )
    encoded = _encode_srgb(values.astype(np.float64)).astype(np.float32) if gamma == "srgb" else values
    source = _frame(encoded, gamma=gamma)
    forward = px.color.white_point_simulation(source, input_white="d65", output_white="d93")
    reverse = px.color.white_point_simulation(forward, input_white="d93", output_white="d65")
    expected = _expected(
        encoded,
        colorspace="sRGB",
        gamma=gamma,
        input_white=_D65,
        output_white=_D93,
    )
    tolerance = 1e-5 if gamma == "linear" else 8e-5
    np.testing.assert_allclose(forward.data.get(), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(reverse.data.get(), source.data.get(), rtol=tolerance, atol=tolerance)


def test_different_whites_use_one_label_driven_private_metadata_preserving_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-white-point-simulation acceptance 10: decode, device matrix, and encode use one label-driven GPU pass."""
    import cupy as cp

    import pixtreme._color.white_point as implementation

    values = np.asarray(
        [[[-0.25, 0.60, 17.0, 0.15, 1.40, -3.0], [1.25, -0.10, 23.0, 0.70, 0.05, 8.0]]],
        dtype=np.float32,
    )
    source = _frame(values, gamma="srgb", channels=("A", "B", "custom", "R", "G", "Z"))
    source_snapshot = source.data.copy()
    metadata_snapshot = (source.colorspace, source.gamma, source.channels, source.matrix)
    original_transform = implementation._transform_data
    calls = 0

    def counted_transform(*args: object, **kwargs: object) -> cp.ndarray:
        nonlocal calls
        calls += 1
        return original_transform(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(implementation, "_transform_data", counted_transform)
    output = px.color.white_point_simulation(source, input_white="d65", output_white="d93")

    assert calls == 1
    assert output is not source
    assert not cp.shares_memory(output.data, source.data)
    assert output.data.flags.c_contiguous
    assert output.data.dtype == cp.float32
    assert (output.colorspace, output.gamma, output.channels, output.matrix) == (
        source.colorspace,
        source.gamma,
        source.channels,
        None,
    )
    assert cp.array_equal(output.data[..., (0, 2, 5)], source.data[..., (0, 2, 5)])
    assert cp.array_equal(source.data, source_snapshot)
    assert (source.colorspace, source.gamma, source.channels, source.matrix) == metadata_snapshot


def test_calls_are_bit_deterministic_and_do_not_stamp_implicit_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-white-point-simulation acceptance 12: results ignore environment and call order without metadata state."""
    import cupy as cp

    source = _frame((0.21, 0.47, 1.19), gamma="srgb")
    first = px.color.white_point_simulation(source, input_white="d65", output_white="d93")
    monkeypatch.setenv("PIXTREME_WHITE_POINT", "ignored")
    px.color.white_point_simulation(source, input_white="d93", output_white="d65")
    second = px.color.white_point_simulation(source, input_white="d65", output_white="d93")
    assert cp.array_equal(first.data, second.data)
    assert tuple(px.core.Frame.model_fields) == ("data", "colorspace", "gamma", "channels", "matrix")
    assert (first.colorspace, first.gamma, first.channels, first.matrix) == (
        source.colorspace,
        source.gamma,
        source.channels,
        None,
    )
