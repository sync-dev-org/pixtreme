"""Specification and independent-oracle tests for full-reference quality metrics."""

from __future__ import annotations

import inspect
from fractions import Fraction
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import pytest

import pixtreme as px


def _frame(
    values: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    channels: str | tuple[str, ...] | list[str] = "RGB",
    matrix: str | None = None,
) -> px.core.Frame:
    return px.io.from_array(
        cp.asarray(np.asarray(values, dtype=dtype)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix=matrix,
    )


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> str:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message
    return message


def _psnr_reference(reference: np.ndarray, candidate: np.ndarray, *, data_range: float) -> np.float64:
    """Evaluate the acceptance formula in host NumPy float64."""
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = np.sum(difference * difference, dtype=np.float64) / np.float64(reference.size)
    if mse == 0.0:
        return np.float64(np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.float64(10.0) * np.log10(np.float64(data_range) ** 2 / mse)


def _gaussian_weights() -> np.ndarray:
    """Construct the specified normalized 11x11 Gaussian in host float64."""
    offsets = np.arange(-5, 6, dtype=np.float64)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    weights = np.exp(-(xx * xx + yy * yy) / (np.float64(2.0) * np.float64(1.5) ** 2))
    return weights / np.sum(weights, dtype=np.float64)


def _ssim_reference(reference: np.ndarray, candidate: np.ndarray, *, data_range: float) -> np.ndarray:
    """Evaluate valid per-channel population moments in independent host float64 loops."""
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    weights = _gaussian_weights()
    output = np.empty((reference.shape[0] - 10, reference.shape[1] - 10), dtype=np.float64)
    c1 = (np.float64(0.01) * np.float64(data_range)) ** 2
    c2 = (np.float64(0.03) * np.float64(data_range)) ** 2
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            channel_values = np.empty(reference.shape[2], dtype=np.float64)
            for channel in range(reference.shape[2]):
                reference_window = reference64[y : y + 11, x : x + 11, channel]
                candidate_window = candidate64[y : y + 11, x : x + 11, channel]
                mu_reference = np.sum(weights * reference_window, dtype=np.float64)
                mu_candidate = np.sum(weights * candidate_window, dtype=np.float64)
                centered_reference = reference_window - mu_reference
                centered_candidate = candidate_window - mu_candidate
                variance_reference = np.sum(weights * centered_reference * centered_reference, dtype=np.float64)
                variance_candidate = np.sum(weights * centered_candidate * centered_candidate, dtype=np.float64)
                covariance = np.sum(weights * centered_reference * centered_candidate, dtype=np.float64)
                channel_values[channel] = ((2.0 * mu_reference * mu_candidate + c1) * (2.0 * covariance + c2)) / (
                    (mu_reference * mu_reference + mu_candidate * mu_candidate + c1)
                    * (variance_reference + variance_candidate + c2)
                )
            output[y, x] = np.mean(channel_values, dtype=np.float64)
    return output


def test_quality_metrics_public_signatures_paths_and_array_contract_are_exact() -> None:
    """v1-quality-metrics acceptance 1, 2, and 7: three exact paths return private raw device arrays."""
    for name in ("psnr", "ssim", "ssim_map"):
        function = getattr(px.metrics, name)
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == ("reference", "candidate", "data_range")
        assert signature.parameters["reference"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert signature.parameters["candidate"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert signature.parameters["data_range"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["data_range"].default == 1.0

    assert px.metrics.__all__ == ("psnr", "ssim", "ssim_map")
    public_modules = (
        px,
        px.color,
        px.filter,
        px.filter,
        px.morphology,
        px.transform,
        px.draw,
        px.generate,
        px.channel,
        px.values,
        px.composite,
    )
    for name in ("psnr", "ssim", "ssim_map"):
        assert px.metrics.__all__.count(name) == 1
        assert not hasattr(px, name)
        assert not hasattr(px.core.Frame, name)
        assert all(not hasattr(module, name) for module in public_modules)

    values = np.linspace(-0.25, 1.5, 12 * 13 * 3, dtype=np.float32).reshape(12, 13, 3)
    reference = _frame(values)
    candidate = _frame(values + np.float32(0.01))
    first_results = (
        px.metrics.psnr(reference, candidate),
        px.metrics.ssim(reference, candidate),
        px.metrics.ssim_map(reference, candidate),
    )
    second_results = (
        px.metrics.psnr(reference, candidate),
        px.metrics.ssim(reference, candidate),
        px.metrics.ssim_map(reference, candidate),
    )
    for index, (result, shape) in enumerate(zip(first_results, ((), (), (2, 3)), strict=True)):
        assert isinstance(result, cp.ndarray)
        assert not isinstance(result, px.core.Frame)
        assert result.shape == shape
        assert result.dtype == cp.float32
        assert result.flags.c_contiguous
        assert result.data.ptr not in (reference.data.data.ptr, candidate.data.data.ptr, second_results[index].data.ptr)
        for metadata_name in ("colorspace", "gamma", "channels", "matrix"):
            assert not hasattr(result, metadata_name)


@pytest.mark.parametrize("name", ("psnr", "ssim", "ssim_map"))
def test_quality_metrics_reject_non_frame_and_non_float32_inputs_actionably(name: str) -> None:
    """v1-quality-metrics acceptance 3: both inputs must be float32 Frames with conversion guidance."""
    function = getattr(px.metrics, name)
    good = _frame(np.zeros((11, 11, 1), dtype=np.float32), channels=["Y"])
    for value in (cp.zeros((11, 11, 1), dtype=cp.float32), np.zeros((11, 11, 1), dtype=np.float32), object()):
        with pytest.raises(ValueError) as reference_error:
            function(value, good)
        _assert_actionable(reference_error)
        with pytest.raises(ValueError) as candidate_error:
            function(good, value)
        _assert_actionable(candidate_error)

    expected_guidance = {
        np.dtype(np.float16): ("cast_dtype",),
        np.dtype(np.uint8): ("recode_dtype", "dequantize"),
        np.dtype(np.uint16): ("recode_dtype", "dequantize"),
    }
    for dtype, required in expected_guidance.items():
        bad = _frame(np.ones((11, 11, 1)), dtype=dtype, channels=["Y"])
        for arguments in ((bad, good), (good, bad)):
            with pytest.raises(ValueError) as dtype_error:
                function(*arguments)
            message = _assert_actionable(dtype_error)
            assert dtype.name in message
            for guidance in required:
                assert f"px.values.{guidance}" in message


@pytest.mark.parametrize("name", ("psnr", "ssim", "ssim_map"))
@pytest.mark.parametrize(
    ("field", "candidate"),
    (
        ("height", lambda: _frame(np.zeros((12, 12, 3), dtype=np.float32))),
        ("width", lambda: _frame(np.zeros((11, 13, 3), dtype=np.float32))),
        ("channel count", lambda: _frame(np.zeros((11, 12, 2), dtype=np.float32), channels="RG")),
        ("channels", lambda: _frame(np.zeros((11, 12, 3), dtype=np.float32), channels="BGR")),
        ("colorspace", lambda: _frame(np.zeros((11, 12, 3), dtype=np.float32), colorspace="sRGB")),
        ("gamma", lambda: _frame(np.zeros((11, 12, 3), dtype=np.float32), gamma="srgb")),
        ("matrix", lambda: _frame(np.zeros((11, 12, 3), dtype=np.float32), matrix="native")),
    ),
)
def test_quality_metrics_reject_every_pair_mismatch_actionably(name: str, field: str, candidate: Any) -> None:
    """v1-quality-metrics acceptance 4: geometry and metadata must match literally and fail fast."""
    reference = _frame(np.zeros((11, 12, 3), dtype=np.float32))
    with pytest.raises(ValueError) as error:
        getattr(px.metrics, name)(reference, candidate())
    message = _assert_actionable(error)
    assert field in message
    assert "reference" in message
    assert "candidate" in message


def test_ssim_metrics_enforce_the_11x11_minimum_geometry() -> None:
    """v1-quality-metrics acceptance 5: SSIM needs an 11x11 valid window beyond the Frame invariant."""
    for shape in ((10, 11, 1), (11, 10, 1)):
        too_small = _frame(np.zeros(shape, dtype=np.float32), channels=["Y"])
        for function in (px.metrics.ssim, px.metrics.ssim_map):
            with pytest.raises(ValueError) as error:
                function(too_small, too_small)
            message = _assert_actionable(error)
            assert repr(shape) in message
            assert "11" in message


@pytest.mark.parametrize("name", ("psnr", "ssim", "ssim_map"))
def test_quality_metrics_validate_data_range_without_inference(name: str) -> None:
    """v1-quality-metrics acceptance 6 and 8: data_range is explicit, finite, positive, and never inferred."""
    function = getattr(px.metrics, name)
    frame = _frame(np.full((11, 11, 1), np.float32(2.5)), channels=["Y"])
    for value in (True, False, "1.0", object(), complex(1.0), 0.0, -1.0, np.inf, -np.inf, np.nan, 10**1000):
        with pytest.raises(ValueError) as error:
            function(frame, frame, data_range=value)
        message = _assert_actionable(error)
        assert "positive finite real" in message

    for value in (1, 2.5, np.float32(1.25), np.float64(4.0), Fraction(3, 2)):
        result = function(frame, frame, data_range=value)
        assert isinstance(result, cp.ndarray)


def test_psnr_matches_one_global_float64_oracle_and_ieee_special_values() -> None:
    """v1-quality-metrics acceptance 8-12 and 25: PSNR uses one all-sample MSE and propagates IEEE values."""
    cases = (
        (
            np.asarray([[[-0.5]]], dtype=np.float32),
            np.asarray([[[1.5]]], dtype=np.float32),
            2.0,
        ),
        (
            np.asarray([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
            np.asarray([[[0.1, 2.0], [0.2, 4.0]]], dtype=np.float32),
            1.0,
        ),
        (
            np.asarray([[[-2.0, 0.5, 3.0], [1.25, -0.75, 2.5]]], dtype=np.float32),
            np.asarray([[[-1.5, 0.25, 2.0], [1.0, -0.5, 4.0]]], dtype=np.float32),
            3.5,
        ),
    )
    for reference_values, candidate_values, data_range in cases:
        reference = _frame(reference_values, channels=[f"c{index}" for index in range(reference_values.shape[2])])
        candidate = _frame(candidate_values, channels=reference.channels)
        actual = px.metrics.psnr(reference, candidate, data_range=data_range)
        expected = _psnr_reference(reference_values, candidate_values, data_range=data_range)
        assert actual.shape == ()
        assert actual.dtype == cp.float32
        np.testing.assert_allclose(actual.get(), np.float32(expected), rtol=2e-6, atol=2e-6)

    identical_values = np.asarray([[[-3.0, 2.0, 9.0]]], dtype=np.float32)
    identical = _frame(identical_values, channels=("A", "custom", "Z"), matrix="native")
    assert bool(cp.isposinf(px.metrics.psnr(identical, identical)))

    nan_reference = _frame(np.asarray([[[np.nan]]], dtype=np.float32), channels=["Y"])
    finite = _frame(np.asarray([[[0.0]]], dtype=np.float32), channels=["Y"])
    assert bool(cp.isnan(px.metrics.psnr(nan_reference, finite)))
    infinite_reference = _frame(np.asarray([[[np.inf]]], dtype=np.float32), channels=["Y"])
    assert bool(cp.isneginf(px.metrics.psnr(infinite_reference, finite)))


def test_ssim_map_matches_direct_population_oracle_and_scalar_is_its_fp32_mean() -> None:
    """v1-quality-metrics acceptance 13-18 and 25: valid Gaussian SSIM matches an independent float64 oracle."""
    generator = np.random.default_rng(20260806)
    reference_values = generator.uniform(-0.75, 1.75, size=(13, 14, 3)).astype(np.float32)
    candidate_values = reference_values.copy()
    candidate_values[2:8, 3:10, 0] += np.float32(0.35)
    candidate_values[6:12, 5:13, 1] *= np.float32(0.72)
    candidate_values[..., 2] = np.roll(candidate_values[..., 2], shift=1, axis=1)
    reference = _frame(reference_values, channels=("A", "custom", "Z"))
    candidate = _frame(candidate_values, channels=reference.channels)

    for data_range in (1.0, 3.25):
        expected_map = _ssim_reference(reference_values, candidate_values, data_range=data_range)
        actual_map = px.metrics.ssim_map(reference, candidate, data_range=data_range)
        actual_scalar = px.metrics.ssim(reference, candidate, data_range=data_range)
        assert actual_map.shape == (3, 4)
        np.testing.assert_allclose(actual_map.get(), expected_map.astype(np.float32), rtol=4e-5, atol=4e-5)
        assert bool(actual_scalar == cp.mean(actual_map, dtype=cp.float32))

    minimal_values = reference_values[:11, :11, :]
    minimal = _frame(minimal_values, channels=reference.channels)
    assert px.metrics.ssim_map(minimal, minimal).shape == (1, 1)


@pytest.mark.parametrize("constant", (-2.5, 0.0, 1.0, 4.25))
def test_ssim_identical_pairs_are_exact_one_without_clamp(constant: float) -> None:
    """v1-quality-metrics acceptance 8, 9, and 19: finite identical scene values produce exact one."""
    values = np.full((12, 12, 2), np.float32(constant), dtype=np.float32)
    frame = _frame(values, channels=("A", "custom"))
    result_map = px.metrics.ssim_map(frame, frame, data_range=0.5)
    result_scalar = px.metrics.ssim(frame, frame, data_range=0.5)
    cp.testing.assert_array_equal(result_map, cp.ones((2, 2), dtype=cp.float32))
    cp.testing.assert_array_equal(result_scalar, cp.asarray(np.float32(1.0)))


def test_ssim_nonconstant_identical_pair_is_exact_one() -> None:
    """v1-quality-metrics acceptance 19: finite nonconstant identical windows are exact fp32 one."""
    values = np.linspace(-0.45, 2.5, 13 * 14 * 3, dtype=np.float32).reshape(13, 14, 3)
    frame = _frame(values)
    cp.testing.assert_array_equal(px.metrics.ssim_map(frame, frame), cp.ones((3, 4), dtype=cp.float32))
    cp.testing.assert_array_equal(px.metrics.ssim(frame, frame), cp.asarray(np.float32(1.0)))


def test_ssim_evaluates_distinct_constants_and_nonfinite_values_without_clamp() -> None:
    """v1-quality-metrics acceptance 9 and 19: constants and IEEE values follow the formula unchanged."""
    reference_values = np.full((11, 11, 2), np.float32(-2.0), dtype=np.float32)
    candidate_values = np.full((11, 11, 2), np.float32(3.5), dtype=np.float32)
    reference = _frame(reference_values, channels=("A", "custom"))
    candidate = _frame(candidate_values, channels=reference.channels)
    expected = _ssim_reference(reference_values, candidate_values, data_range=0.75)
    actual_map = px.metrics.ssim_map(reference, candidate, data_range=0.75)
    np.testing.assert_allclose(actual_map.get(), expected.astype(np.float32), rtol=3e-6, atol=3e-6)
    assert bool(px.metrics.ssim(reference, candidate, data_range=0.75) == actual_map[0, 0])

    nan_values = reference_values.copy()
    nan_values[5, 5, 0] = np.nan
    nan_frame = _frame(nan_values, channels=reference.channels)
    assert bool(cp.isnan(px.metrics.ssim_map(nan_frame, candidate)[0, 0]))
    assert bool(cp.isnan(px.metrics.ssim(nan_frame, candidate)))


def test_quality_metrics_preserve_both_inputs_and_do_not_share_storage() -> None:
    """v1-quality-metrics acceptance 7-9: metrics do not mutate, normalize, clip, scan, or alias either input."""
    values = np.linspace(-2.0, 3.0, 12 * 12 * 2, dtype=np.float32).reshape(12, 12, 2)
    reference = _frame(values, channels=("A", "custom"), matrix="native")
    candidate = _frame(values[::-1].copy(), channels=reference.channels, matrix=reference.matrix)
    reference_before = reference.data.copy()
    candidate_before = candidate.data.copy()
    reference_metadata = (reference.colorspace, reference.gamma, reference.channels, reference.matrix)
    candidate_metadata = (candidate.colorspace, candidate.gamma, candidate.channels, candidate.matrix)

    results = (
        px.metrics.psnr(reference, candidate, data_range=0.25),
        px.metrics.ssim(reference, candidate, data_range=0.25),
        px.metrics.ssim_map(reference, candidate, data_range=0.25),
    )
    cp.testing.assert_array_equal(reference.data, reference_before)
    cp.testing.assert_array_equal(candidate.data, candidate_before)
    assert (reference.colorspace, reference.gamma, reference.channels, reference.matrix) == reference_metadata
    assert (candidate.colorspace, candidate.gamma, candidate.channels, candidate.matrix) == candidate_metadata
    for result in results:
        assert result.data.ptr not in (reference.data.data.ptr, candidate.data.data.ptr)


def test_quality_metric_docstrings_are_self_contained_operational_contracts() -> None:
    """v1-quality-metrics acceptance 20 and 21: public docs explain formulas, boundaries, and explicit composition."""
    docstrings = {
        name: " ".join((inspect.getdoc(getattr(px.metrics, name)) or "").split())
        for name in ("psnr", "ssim", "ssim_map")
    }
    for name, docstring in docstrings.items():
        for required in (
            f"px.metrics.{name}(reference, candidate, *, data_range=1.0)",
            "float32 Frame",
            "height, width, channels, colorspace, gamma, and matrix",
            "data_range",
            "1.0",
            "all channels",
            "cupy.ndarray",
            "float32",
            "new storage",
            "does not mutate",
            "does not clamp",
            "cast_dtype",
            "recode_dtype",
            "dequantize",
            "rgb_to_grayscale",
            "channel.shuffle",
        ):
            assert required in docstring
    for name in ("psnr", "ssim"):
        for required in ("0D", "GPU-resident", "float(result)", "result.item()", "explicit synchronization"):
            assert required in docstrings[name]
    for required in ("MSE", "10 * log10", "+inf"):
        assert required in docstrings["psnr"]
    for name in ("ssim", "ssim_map"):
        for required in ("11x11", "sigma=1.5", "population", "C1", "C2", "channel mean"):
            assert required in docstrings[name]
    for required in ("2D", "(H - 10, W - 10)", "length-one channel dimension", "px.io.from_array", "explicit metadata"):
        assert required in docstrings["ssim_map"]


def test_quality_metric_requirements_preserve_module_and_vocabulary_boundaries() -> None:
    """v1-quality-metrics acceptance 22 / v1-public-namespace acceptance 1 and 8: boundaries stay exact."""
    repository = Path(__file__).resolve().parents[1]
    requirements_path = repository / "docs" / "requirements.md"
    vocabulary_path = repository / "docs_site" / "tokens.md"
    if not requirements_path.is_file() or not vocabulary_path.is_file():
        pytest.skip("repo-only documentation contract: canonical docs are absent from this distribution")
    requirements = requirements_path.read_text(encoding="utf-8")
    vocabulary = vocabulary_path.read_text(encoding="utf-8")
    architecture = requirements.split("**REQ-ARCH-008:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    modules = requirements.split("**REQ-API-009:", maxsplit=1)[1].split("**REQ-API-010:", maxsplit=1)[0]
    boundaries = requirements.split("**REQ-API-010:", maxsplit=1)[1].split("**REQ-API-011:", maxsplit=1)[0]
    assert "13 module" in architecture
    assert "`px.io.from_array`" in architecture
    assert "`px.io.to_array`" in architecture
    assert "13 module" in modules
    assert "`metrics`" in modules
    for path in ("px.metrics.psnr", "px.metrics.ssim", "px.metrics.ssim_map"):
        assert path in boundaries
    for required in ("0 次元", "2 次元", "scalar", "map", "Frame → device 配列"):
        assert required in boundaries
    assert not any(token in vocabulary.lower() for token in ("psnr", "ssim", "data_range"))


def test_quality_metric_tests_carry_acceptance_backreferences() -> None:
    """v1-quality-metrics acceptance 25: every test in this module names its feature acceptance source."""
    for name, value in globals().items():
        if name.startswith("test_") and inspect.isfunction(value):
            assert "v1-quality-metrics acceptance" in (inspect.getdoc(value) or "")
