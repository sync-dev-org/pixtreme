"""Specification tests for per-channel Lift / Gamma / Gain grading."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _frame(
    values: Any,
    *,
    channels: str | tuple[str, ...] = "RGB",
    dtype: Any = np.float32,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    matrix: str | None = None,
) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    return px.core.Frame(
        data=cp.asarray(array),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix=matrix,
    )


def _host(frame: px.core.Frame) -> np.ndarray:
    return frame.data.get()


def _signed_power(value: np.ndarray, exponent: np.ndarray | float) -> np.ndarray:
    return np.copysign(np.power(np.abs(value), exponent), value)


def _assert_z_contract(
    actual: np.ndarray,
    source: np.ndarray,
    *,
    lift: np.ndarray | float,
    gamma: np.ndarray | float,
    gain: np.ndarray | float,
) -> None:
    """Apply the v1-grade fp64 inverse-power oracle without production helpers."""
    x64 = np.asarray(source, dtype=np.float32).astype(np.float64)
    lift64 = np.asarray(lift, dtype=np.float32).astype(np.float64)
    gamma64 = np.asarray(gamma, dtype=np.float32).astype(np.float64)
    gain64 = np.asarray(gain, dtype=np.float32).astype(np.float64)
    z_exact = gain64 * x64 + lift64 * (np.float64(1.0) - x64)
    actual64 = np.asarray(actual, dtype=np.float32).astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        z_reconstructed = _signed_power(actual64, gamma64)
    rtol_z = np.float64(2e-6) * gamma64
    atol_z = np.float64(2.0**-22) * (np.abs(gain64 * x64) + np.abs(lift64 * (np.float64(1.0) - x64)))
    error = np.abs(z_reconstructed - z_exact)
    allowed = rtol_z * np.abs(z_exact) + atol_z
    assert np.all(error <= allowed), f"maximum z-space excess={np.max(error - allowed)!r}"
    np.testing.assert_array_equal(np.signbit(actual64), np.signbit(z_exact))
    assert np.isfinite(actual64).all()


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert message.startswith("why=")
    assert "; what=" in message
    assert "; how=" in message
    assert message.index("why=") < message.index("; what=") < message.index("; how=")


def test_neutral_grade_bit_preserves_every_channel_in_a_private_copy() -> None:
    """v1-grade acceptance 2: neutral resolution preserves all fp32 bits in new private storage."""
    import cupy as cp

    bits = np.asarray(
        (
            0x00000000,
            0x80000000,
            0x7FC01234,
            0xFFC05678,
            0x7F800000,
            0xFF800000,
            0x3F800000,
            0xBF800000,
        ),
        dtype=np.uint32,
    )
    source = _frame(bits.view(np.float32), channels=("R", "G", "B", "A", "Z", "custom", "R", "aux"))
    output = px.color.grade(source, lift={}, gamma={}, gain={})

    assert output is not source
    assert not cp.shares_memory(output.data, source.data)
    assert output.data.flags.c_contiguous
    assert output.data.dtype == cp.float32
    np.testing.assert_array_equal(_host(output).view(np.uint32), bits.reshape(1, 1, -1))


def test_nuke_default_reduction_matches_the_independent_fp64_oracle() -> None:
    """v1-grade acceptance 3: the Nuke-default reduction matches within the fixed z-space contract."""
    values = np.asarray((0.0, 0.03125, 0.19, 0.53, 0.77, 1.0), dtype=np.float32).reshape(1, 2, 3)
    lift = np.float32(0.08)
    gamma = np.float32(0.73)
    gain = np.float32(0.91)
    source = _frame(values)

    actual = _host(px.color.grade(source, lift=float(lift), gamma=float(gamma), gain=float(gain)))
    x64 = values.astype(np.float64)
    nuke_affine = (np.float64(gain) - np.float64(lift)) * x64 + np.float64(lift)
    assert np.all((nuke_affine >= 0.0) & (nuke_affine <= 1.0))
    expected = np.power(nuke_affine, np.float64(1.0) / np.float64(gamma))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    _assert_z_contract(actual, values, lift=lift, gamma=gamma, gain=gain)


def test_scene_extended_values_use_signed_pure_power_without_clipping() -> None:
    """v1-grade acceptance 4: negative, overshoot, extrema, cancellation, and signed zero use pure power."""
    values = np.asarray(
        (-4.0, -1.25, -0.0, 0.0, 0.125, 0.99999994, 1.0000001, 2.5, 4.0),
        dtype=np.float32,
    ).reshape(1, 3, 3)
    source = _frame(values)
    actual = _host(px.color.grade(source, lift=-0.0, gamma=2.0, gain=1.0))

    _assert_z_contract(actual, values, lift=-0.0, gamma=2.0, gain=1.0)
    assert np.min(actual) < 0.0
    assert np.max(actual) > 1.0
    assert np.signbit(actual.reshape(-1)[2])
    assert not np.signbit(actual.reshape(-1)[3])

    cancellation = np.asarray(
        (
            np.nextafter(np.float32(0.5), np.float32(0.0)),
            np.float32(0.5),
            np.nextafter(np.float32(0.5), np.float32(1.0)),
        ),
        dtype=np.float32,
    ).reshape(1, 1, 3)
    cancellation_actual = _host(px.color.grade(_frame(cancellation), lift=0.5, gamma=4.0, gain=-0.5))
    _assert_z_contract(cancellation_actual, cancellation, lift=0.5, gamma=4.0, gain=-0.5)

    for boundary_lift, boundary_gamma, boundary_gain in ((-2.0, 0.25, -4.0), (2.0, 4.0, 4.0)):
        boundary_actual = _host(
            px.color.grade(
                _frame(values),
                lift=boundary_lift,
                gamma=boundary_gamma,
                gain=boundary_gain,
            )
        )
        _assert_z_contract(
            boundary_actual,
            values,
            lift=boundary_lift,
            gamma=boundary_gamma,
            gain=boundary_gain,
        )


def test_cdl_and_classic_lgg_parameterizations_match_their_independent_cores() -> None:
    """v1-grade acceptance 5: CDL full/valid subsets and classic LGG match only their stated affine-power cores."""
    values = np.asarray((-3.0, -0.5, 0.0, 0.2, 1.0, 2.75), dtype=np.float32).reshape(1, 2, 3)
    lift = np.float32(-0.2)
    gamma = np.float32(1.3)
    gain = np.float32(1.5)
    actual = _host(px.color.grade(_frame(values), lift=float(lift), gamma=float(gamma), gain=float(gain)))
    slope = np.float64(gain) - np.float64(lift)
    offset = np.float64(lift)
    power = np.float64(1.0) / np.float64(gamma)
    cdl_core = _signed_power(slope * values.astype(np.float64) + offset, power)
    np.testing.assert_allclose(actual, cdl_core, rtol=3e-6, atol=3e-7)
    _assert_z_contract(actual, values, lift=lift, gamma=gamma, gain=gain)

    valid_values = np.asarray((0.0, 0.1, 0.3, 0.6, 0.9, 1.0), dtype=np.float32).reshape(1, 2, 3)
    valid_lift = np.float32(0.1)
    valid_gain = np.float32(0.9)
    valid_gamma = np.float32(0.7)
    valid_z = np.float64(valid_gain - valid_lift) * valid_values.astype(np.float64) + np.float64(valid_lift)
    assert float(valid_gain - valid_lift) >= 0.0
    assert np.all((valid_z >= 0.0) & (valid_z <= 1.0))
    valid_actual = _host(
        px.color.grade(
            _frame(valid_values),
            lift=float(valid_lift),
            gamma=float(valid_gamma),
            gain=float(valid_gain),
        )
    )
    _assert_z_contract(valid_actual, valid_values, lift=valid_lift, gamma=valid_gamma, gain=valid_gain)

    classic_values = np.asarray((-2.0, -0.25, 0.0, 0.4, 1.0, 3.0), dtype=np.float32).reshape(1, 2, 3)
    classic_gain = np.float32(1.2)
    lift_control = np.float32(0.15)
    grade_lift = np.float32(classic_gain * lift_control)
    classic_gamma = np.float32(1.8)
    classic_actual = _host(
        px.color.grade(
            _frame(classic_values),
            lift=float(grade_lift),
            gamma=float(classic_gamma),
            gain=float(classic_gain),
        )
    )
    classic_z = np.float64(classic_gain) * (
        classic_values.astype(np.float64)
        + np.float64(lift_control) * (np.float64(1.0) - classic_values.astype(np.float64))
    )
    classic_expected = _signed_power(classic_z, np.float64(1.0) / np.float64(classic_gamma))
    np.testing.assert_allclose(classic_actual, classic_expected, rtol=3e-6, atol=3e-7)
    _assert_z_contract(
        classic_actual,
        classic_values,
        lift=grade_lift,
        gamma=classic_gamma,
        gain=classic_gain,
    )


def test_mapping_resolves_duplicate_labels_and_bit_preserves_unspecified_channels() -> None:
    """v1-grade acceptance 6 and 9: mappings affect every exact duplicate label and preserve all unspecified bits."""
    import cupy as cp

    values = np.asarray(
        ((-0.5, -0.0, 0.25, 1.5, 17.0), (2.0, 0.75, -1.0, -0.0, -23.0)),
        dtype=np.float32,
    ).reshape(1, 2, 5)
    source = _frame(
        values,
        channels=("R", "A", "R", "Z", "custom"),
        colorspace="Rec.709",
        gamma="Gamma-2.4",
        matrix="BT.709",
    )
    source_bits = _host(source).view(np.uint32).copy()
    metadata = (source.colorspace, source.gamma, source.channels, source.matrix)

    output = px.color.grade(source, lift={"R": 0.2}, gamma={"R": 2.0}, gain={"R": 1.4})
    actual = _host(output)

    _assert_z_contract(
        actual[..., (0, 2)],
        values[..., (0, 2)],
        lift=0.2,
        gamma=2.0,
        gain=1.4,
    )
    np.testing.assert_array_equal(actual[..., (1, 3, 4)].view(np.uint32), source_bits[..., (1, 3, 4)])
    np.testing.assert_array_equal(_host(source).view(np.uint32), source_bits)
    assert not cp.shares_memory(output.data, source.data)
    assert output.data.flags.c_contiguous and output.data.dtype == cp.float32
    assert (output.colorspace, output.gamma, output.channels, output.matrix) == metadata
    assert (source.colorspace, source.gamma, source.channels, source.matrix) == metadata


def test_scalar_broadcast_and_mixed_mappings_ignore_channel_semantics() -> None:
    """v1-grade acceptance 7: scalar broadcast and independent mappings include every named and custom channel."""
    labels = ("R", "G", "B", "A", "Z", "Cb", "Cr", "H", "application.depth")
    values = np.asarray((-1.0, -0.5, -0.0, 0.0, 0.25, 0.8, 1.0, 1.5, 3.0), dtype=np.float32).reshape(1, 1, -1)
    actual = _host(
        px.color.grade(
            _frame(values, channels=labels),
            lift=0.1,
            gamma={"G": 2.0, "H": 0.5},
            gain=0.8,
        )
    )
    gamma_by_label = {"G": np.float32(2.0), "H": np.float32(0.5)}
    gammas = np.asarray([gamma_by_label.get(label, np.float32(1.0)) for label in labels], dtype=np.float32)

    _assert_z_contract(actual, values, lift=0.1, gamma=gammas, gain=0.8)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"lift": True},
        {"lift": "0.1"},
        {"lift": math.nan},
        {"lift": math.inf},
        {"lift": 1e300},
        {"gain": -math.inf},
        {"gain": 1e300},
        {"gamma": True},
        {"gamma": 0.0},
        {"gamma": -1.0},
        {"gamma": math.nan},
        {"gamma": math.inf},
        {"gamma": 1e-300},
        {"gamma": 10**400},
        {"lift": {1: 0.0}},
        {"lift": {"": 0.0}},
        {"lift": {"missing": 0.0}},
        {"gain": {"R": True}},
        {"gain": {"R": math.nan}},
        {"gain": {"R": 1e300}},
        {"gamma": {"R": 0.0}},
        {"gamma": {"R": 1e-300}},
    ),
)
def test_invalid_parameters_fail_actionably_before_pixel_processing(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
) -> None:
    """v1-grade acceptance 8: invalid scalar, mapping, key, fp32 overflow, and gamma underflow fail first."""
    import pixtreme._color.grade as implementation

    source = _frame((0.2, 0.3, 0.4))

    def forbidden_kernel() -> object:
        raise AssertionError("pixel processing must not start for invalid parameters")

    monkeypatch.setattr(implementation, "_grade_kernel", forbidden_kernel)
    with pytest.raises(ValueError) as error:
        px.color.grade(source, **kwargs)  # type: ignore[arg-type]
    _assert_actionable(error)


@pytest.mark.parametrize("invalid_frame", (object(), pytest.param("float16", id="float16")))
def test_invalid_frame_contract_fails_actionably(invalid_frame: object) -> None:
    """v1-grade acceptance 8: grade accepts only metadata-bearing float32 Frames with a cast recovery path."""
    is_float16 = invalid_frame == "float16"
    if is_float16:
        invalid_frame = _frame((0.2, 0.3, 0.4), dtype=np.float16)
    with pytest.raises(ValueError) as error:
        px.color.grade(invalid_frame)  # type: ignore[arg-type]
    _assert_actionable(error)
    if is_float16:
        assert "float32" in str(error.value)
        assert "px.values.cast_dtype" in str(error.value)


def test_backend_failure_is_translated_with_its_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-grade acceptance 8: a backend launch failure retains its cause behind an actionable public error."""
    import pixtreme._color.grade as implementation

    def failed_kernel() -> object:
        raise RuntimeError("synthetic CUDA failure")

    monkeypatch.setattr(implementation, "_grade_kernel", failed_kernel)
    with pytest.raises(ValueError) as error:
        px.color.grade(_frame((0.2, 0.3, 0.4)), lift=0.1)
    _assert_actionable(error)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_frame_gamma_metadata_does_not_change_the_curve_and_is_preserved() -> None:
    """v1-grade acceptance 9-10: frame.gamma is neither interpreted nor changed by the grade curve."""
    values = np.asarray((-0.5, 0.18, 1.5), dtype=np.float32)
    linear = _frame(values, gamma="linear")
    encoded_claim = _frame(values, gamma="sRGB")
    kwargs = {"lift": 0.05, "gamma": 1.7, "gain": 1.2}

    linear_output = px.color.grade(linear, **kwargs)
    encoded_output = px.color.grade(encoded_claim, **kwargs)

    np.testing.assert_array_equal(_host(linear_output).view(np.uint32), _host(encoded_output).view(np.uint32))
    assert linear_output.gamma == "linear"
    assert encoded_output.gamma == "sRGB"
