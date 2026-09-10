"""Specification tests for Frame validation-overhead reduction."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import pixtreme as px


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert message.index("why=") < message.index("; what=") < message.index("; how=")


def _frame_module_tree() -> ast.Module:
    import pixtreme._core.frame as frame_module

    return ast.parse(inspect.getsource(frame_module))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    return matches[0]


def _host_snapshot(frame: px.core.Frame) -> tuple[np.ndarray, str, str, tuple[str, ...], str | None]:
    import cupy as cp

    return (cp.asnumpy(frame.data), frame.colorspace, frame.gamma, frame.channels, frame.matrix)


def test_validation_fast_path_stays_private_and_preserves_the_public_surface() -> None:
    """v1-frame-validation-overhead acceptance 1: the fast constructor is private and public surfaces stay fixed."""
    import pixtreme.core as public_core

    assert tuple(px.core.Frame.model_fields) == ("data", "colorspace", "gamma", "channels", "matrix")
    assert tuple(inspect.signature(px.io.from_array).parameters) == (
        "data",
        "colorspace",
        "gamma",
        "channels",
        "matrix",
        "layout",
        "dtype",
        "bit_depth",
        "scale",
        "mean",
        "std",
        "copy",
    )
    assert "_construct_frame" not in public_core.__all__
    assert not hasattr(public_core, "_construct_frame")

    constructor = _function(_frame_module_tree(), "_construct_frame")
    model_construct_calls = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "model_construct"
    ]
    assert len(model_construct_calls) == 1


def test_direct_frame_construction_keeps_full_validation_and_normalization() -> None:
    """v1-frame-validation-overhead acceptance 2: direct Frame construction retains the complete valid/invalid corpus."""
    import cupy as cp

    source = cp.arange(4 * 4 * 3, dtype=cp.float32).reshape(4, 4, 3)[:, ::2, :]
    assert not source.flags.c_contiguous
    result = px.core.Frame(
        data=source,
        colorspace="a c_e-s.c.g",
        gamma="S_R-G.B",
        channels="RGB",
        matrix="b t_7-0.9",
    )

    assert result.data.flags.c_contiguous
    assert result.data.data.ptr != source.data.ptr
    assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
        "ACEScg",
        "sRGB",
        ("R", "G", "B"),
        "BT.709",
    )

    invalid = (
        {"data": cp.zeros((2, 3), dtype=cp.float32), "channels": "RGB"},
        {"data": cp.zeros((0, 2, 3), dtype=cp.float32), "channels": "RGB"},
        {"data": cp.zeros((1, 2, 3), dtype=cp.float64), "channels": "RGB"},
        {"data": cp.zeros((1, 2, 3), dtype=cp.float32), "channels": "RGBA"},
    )
    for case in invalid:
        with pytest.raises(ValidationError) as error:
            px.core.Frame(data=case["data"], colorspace="sRGB", gamma="linear", channels=case["channels"])
        _assert_actionable(error)

    with pytest.raises(ValidationError):
        px.core.Frame(
            data=cp.zeros((1, 2, 3), dtype=cp.float32),
            colorspace="sRGB",
            gamma="linear",
            channels="RGB",
            rejected_field=True,
        )


def test_frame_assignment_keeps_transactional_full_validation() -> None:
    """v1-frame-validation-overhead acceptance 3: all five assignable fields still validate transactionally."""
    import cupy as cp

    frame = px.core.Frame(
        data=cp.zeros((1, 2, 3), dtype=cp.float32),
        colorspace="sRGB",
        gamma="linear",
        channels="RGB",
    )
    replacement = cp.ones((2, 1, 3), dtype=cp.float16)
    frame.data = replacement
    frame.colorspace = "a c_e-s.c.g"
    frame.gamma = "2.2"
    frame.channels = ["red", "green", "blue"]
    frame.matrix = "b t_7-0.9"
    assert frame.data is replacement
    assert (frame.colorspace, frame.gamma, frame.channels, frame.matrix) == (
        "ACEScg",
        "Gamma-2.2",
        ("red", "green", "blue"),
        "BT.709",
    )

    original = (frame.data, frame.colorspace, frame.gamma, frame.channels, frame.matrix)
    invalid_assignments: tuple[tuple[str, Any], ...] = (
        ("data", cp.zeros((1, 1, 4), dtype=cp.float32)),
        ("colorspace", "unknown-colorspace"),
        ("gamma", "unknown-gamma"),
        ("channels", ("R", "G")),
        ("matrix", "unknown-matrix"),
    )
    for field, value in invalid_assignments:
        with pytest.raises(ValidationError) as error:
            setattr(frame, field, value)
        _assert_actionable(error)
        assert (frame.data, frame.colorspace, frame.gamma, frame.channels, frame.matrix) == original


def test_from_array_fast_path_preserves_valid_storage_and_stays_within_scope() -> None:
    """v1-frame-validation-overhead acceptance 4: from_array preserves valid storage while other entries keep full validation."""
    import cupy as cp

    from pixtreme._io.wire import array as array_module

    values = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    source = cp.asarray(values)
    zero_copy = px.io.from_array(
        source,
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
        copy=False,
    )
    private = px.io.from_array(
        source,
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
        copy=True,
    )
    non_contiguous = source[:, ::2, :]
    contiguous = px.io.from_array(
        non_contiguous,
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )

    assert zero_copy.data.data.ptr == source.data.ptr
    assert private.data.data.ptr != source.data.ptr
    assert contiguous.data.flags.c_contiguous
    assert contiguous.data.data.ptr != non_contiguous.data.ptr
    np.testing.assert_array_equal(cp.asnumpy(zero_copy.data), values)
    np.testing.assert_array_equal(cp.asnumpy(private.data), values)
    np.testing.assert_array_equal(cp.asnumpy(contiguous.data), values[:, ::2, :])

    from_array_tree = ast.parse(inspect.getsource(array_module.from_array))
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_construct_frame"
        for node in ast.walk(from_array_tree)
    )

    root = Path(inspect.getsourcefile(px) or "").resolve().parent
    non_target_sources = tuple((root / "_io").rglob("*.py"))
    assert all(
        "_construct_frame" not in path.read_text(encoding="utf-8")
        for path in non_target_sources
        if path != Path(array_module.__file__)
    )


def test_from_array_invalid_public_inputs_keep_their_entry_rejection_timing_and_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-frame-validation-overhead acceptance 5: invalid layout, dtype, and channel counts keep entry ValueError timing."""
    import cupy as cp

    from pixtreme._io.wire import array as array_module

    launches: list[tuple[object, ...]] = []
    original_factory = array_module._from_array_kernel

    def traced_factory(*args: object, **kwargs: object) -> Any:
        kernel = original_factory(*args, **kwargs)

        def traced_launch(*launch_args: object, **launch_kwargs: object) -> Any:
            launches.append(launch_args)
            return kernel(*launch_args, **launch_kwargs)

        return traced_launch

    monkeypatch.setattr(array_module, "_from_array_kernel", traced_factory)
    source = cp.zeros((2, 2, 3), dtype=cp.float16)
    invalid_calls: tuple[tuple[dict[str, object], str], ...] = (
        ({"layout": "unknown-layout"}, "unknown-layout"),
        ({"dtype": "unknown-dtype"}, "unknown-dtype"),
        ({"channels": "RGBA"}, "4 labels"),
    )
    for overrides, raw_input in invalid_calls:
        arguments: dict[str, object] = {
            "colorspace": "ACEScg",
            "gamma": "linear",
            "channels": "RGB",
            "dtype": "float32",
        }
        arguments.update(overrides)
        with pytest.raises(ValueError) as error:
            px.io.from_array(source, **arguments)  # type: ignore[arg-type]
        assert type(error.value) is ValueError
        _assert_actionable(error)
        assert raw_input in str(error.value)
    assert launches == []


def test_from_array_token_spellings_resolve_to_identical_canonical_results() -> None:
    """v1-frame-validation-overhead acceptance 6: canonical, variant, and permanent-alias inputs remain equivalent."""
    import cupy as cp

    values = np.asarray([[[0.0, 0.5, 1.0], [1.0, -0.25, 1.25]]], dtype=np.float32)
    spellings = (
        ("ACEScg", "Gamma-2.2", "BT.709"),
        ("a c_e-s.c.g", "g a_m-m.a 2_2", "b t_7-0.9"),
        ("ACEScg", "2.2", "BT.709"),
    )
    results: list[px.core.Frame] = []
    for colorspace, gamma, matrix in spellings:
        source = cp.asarray(values)
        result = px.io.from_array(
            source,
            colorspace=colorspace,
            gamma=gamma,
            channels="RGB",
            matrix=matrix,
            copy=False,
        )
        assert result.data.data.ptr == source.data.ptr
        results.append(result)

    expected_metadata = ("ACEScg", "Gamma-2.2", ("R", "G", "B"), "BT.709")
    for result in results:
        assert (result.colorspace, result.gamma, result.channels, result.matrix) == expected_metadata
        np.testing.assert_array_equal(cp.asnumpy(result.data), values)


def test_common_operations_keep_bits_metadata_ownership_and_reject_corrupt_current_shapes() -> None:
    """v1-frame-validation-overhead acceptance 7: common outputs stay exact and current invalid shapes fall back to rejection."""
    import cupy as cp

    float_values = np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    float_frame = px.io.from_array(
        cp.asarray(float_values),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )
    input_before = cp.asnumpy(float_frame.data).copy()
    quantized = px.values.quantize(float_frame, bit_depth=8)
    dequantized = px.values.dequantize(quantized, bit_depth=8)
    converted = px.color.rgb_to_rgb(
        px.io.from_array(cp.zeros((2, 2, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB"),
        output_colorspace="sRGB",
        output_gamma="srgb",
    )
    blurred = px.filter.gaussian_blur(
        px.io.from_array(cp.zeros((2, 2, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB"),
        sigma=1.0,
    )

    np.testing.assert_array_equal(cp.asnumpy(quantized.data), np.asarray([[[0, 128, 255]]], dtype=np.uint8))
    np.testing.assert_array_equal(
        cp.asnumpy(dequantized.data),
        np.asarray([[[0, 128, 255]]], dtype=np.uint8).astype(np.float32) * np.float32(1.0 / 255.0),
    )
    np.testing.assert_array_equal(cp.asnumpy(converted.data), np.zeros((2, 2, 3), dtype=np.float32))
    np.testing.assert_array_equal(cp.asnumpy(blurred.data), np.zeros((2, 2, 3), dtype=np.float32))
    assert (quantized.colorspace, quantized.gamma, quantized.channels, quantized.matrix) == (
        float_frame.colorspace,
        float_frame.gamma,
        float_frame.channels,
        float_frame.matrix,
    )
    assert quantized.data.data.ptr != float_frame.data.data.ptr
    assert dequantized.data.data.ptr != quantized.data.data.ptr
    assert converted.data.flags.c_contiguous and blurred.data.flags.c_contiguous
    np.testing.assert_array_equal(cp.asnumpy(float_frame.data), input_before)

    for corrupt_shape in ((2, 6), (2, 3, 2)):
        corrupt = px.io.from_array(
            cp.zeros((2, 2, 3), dtype=cp.float32),
            colorspace="ACEScg",
            gamma="linear",
            channels="RGB",
        )
        corrupt.data.shape = corrupt_shape
        with pytest.raises(ValueError) as error:
            px.values.quantize(corrupt, bit_depth=8)
        _assert_actionable(error)


def test_existing_cache_state_does_not_change_results_and_fast_validation_adds_no_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-frame-validation-overhead acceptance 8: cache-state behavior is deterministic and fast validation is stateless.

    This is an intentional source-AST structural contract for the absence of new registry, ambient-state,
    identity, or validation-result memoization in the optimized construction path.
    """
    import cupy as cp

    import pixtreme._core.frame as frame_module
    import pixtreme._core.validation as validation_module
    from pixtreme._io.wire import array as array_module

    source = cp.asarray(np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32))

    def make_result() -> tuple[np.ndarray, str, str, tuple[str, ...], str | None]:
        return _host_snapshot(
            px.io.from_array(
                source,
                colorspace="a c_e-s.c.g",
                gamma="2.2",
                channels="RGB",
                matrix="b t_7-0.9",
            )
        )

    frame_module._compact_channels.cache_clear()
    validation_module._canonical_token_map.cache_clear()
    validation_module._permanent_alias_map.cache_clear()
    cold = make_result()
    hit = make_result()

    for count in range(1, 130):
        px.core.channels("R" * count)
    misses_before = frame_module._compact_channels.cache_info().misses
    monkeypatch.setenv("PIXtreme_VALIDATION_OVERHEAD_PROBE", "ignored")
    after_eviction_and_environment_change = make_result()
    assert frame_module._compact_channels.cache_info().misses == misses_before + 1

    for observed in (hit, after_eviction_and_environment_change):
        np.testing.assert_array_equal(observed[0], cold[0])
        assert observed[1:] == cold[1:]

    frame_tree = ast.parse(inspect.getsource(frame_module))
    array_tree = ast.parse(inspect.getsource(array_module))
    constructor = _function(frame_tree, "_construct_frame")
    assert not constructor.decorator_list
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "id"
        for node in ast.walk(constructor)
    )
    assert not any(isinstance(node, ast.Attribute) and node.attr == "ptr" for node in ast.walk(constructor))
    forbidden_names = {"environ", "getenv", "registry", "cache", "memo"}
    assert not ({node.id for node in ast.walk(constructor) if isinstance(node, ast.Name)} & forbidden_names)

    decorated = {
        node.name
        for tree in (frame_tree, array_tree)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            (isinstance(decorator, ast.Name) and decorator.id == "lru_cache")
            or (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "lru_cache"
            )
            for decorator in node.decorator_list
        )
    }
    assert decorated == {"_compact_channels", "_select_normalized_channel_indices"}
