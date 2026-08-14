"""Contracts for the shared semantic validation primitives."""

from __future__ import annotations

import ast
import importlib
import inspect
from types import ModuleType

import numpy as np
import pytest


def _validation() -> ModuleType:
    return importlib.import_module("pixtreme._core.validation")


def _error_slots(error: BaseException) -> dict[str, str]:
    parts = str(error).split("; ")
    assert len(parts) == 3
    slots = dict(part.split("=", maxsplit=1) for part in parts)
    assert tuple(slots) == ("why", "what", "how")
    assert all(slots.values())
    return slots


def _assert_error_case(
    error: BaseException,
    *,
    parameter: str,
    rejected_value: object,
) -> dict[str, str]:
    slots = _error_slots(error)
    assert slots["what"] == f"received {parameter}={rejected_value!r}"
    return slots


def _requirement_for(message: str, *, parameter: str) -> str:
    prefix = f"{parameter} must be "
    assert message.startswith(prefix)
    return message.removeprefix(prefix)


def _accepted_values(message: str) -> tuple[str, ...]:
    prefix = "pass one of "
    assert message.startswith(prefix)
    accepted = ast.literal_eval(message.removeprefix(prefix))
    assert isinstance(accepted, tuple)
    assert all(isinstance(value, str) for value in accepted)
    return accepted


def test_real_validation_contract_preserves_numeric_boundaries_and_error_semantics() -> None:
    """REQ-TEST-001: preserve finite, positive, and bounded-real behavior and actionable error identity."""
    validation = _validation()
    bounded_why = "opacity-boundary-why-sentinel"
    bounded_how = "opacity-boundary-how-sentinel"

    assert validation._finite_real(np.int64(3), name="size") == 3.0
    assert validation._positive_real(np.float32(2.5), name="size") == 2.5
    assert (
        validation._bounded_real(
            0.25,
            name="opacity",
            minimum=0.0,
            maximum=1.0,
            why=bounded_why,
            how=bounded_how,
        )
        == 0.25
    )

    with pytest.raises(ValueError) as type_error:
        validation._finite_real(True, name="size")
    _assert_error_case(type_error.value, parameter="size", rejected_value=True)

    with pytest.raises(ValueError) as finite_error:
        validation._finite_real(float("inf"), name="size")
    slots = _assert_error_case(finite_error.value, parameter="size", rejected_value=float("inf"))
    assert _requirement_for(slots["why"], parameter="size") == "finite"

    with pytest.raises(ValueError) as positive_error:
        validation._positive_real(0.0, name="size")
    slots = _assert_error_case(positive_error.value, parameter="size", rejected_value=0.0)
    assert _requirement_for(slots["why"], parameter="size") == "greater than 0"

    with pytest.raises(ValueError) as bounded_error:
        validation._bounded_real(
            1.25,
            name="opacity",
            minimum=0.0,
            maximum=1.0,
            why=bounded_why,
            how=bounded_how,
        )
    slots = _assert_error_case(bounded_error.value, parameter="opacity", rejected_value=1.25)
    assert (slots["why"], slots["how"]) == (bounded_why, bounded_how)


def test_array_pair_and_scalar_or_pair_contract_preserves_numpy_scalar_acceptance() -> None:
    """REQ-TEST-001: preserve host coercion, NumPy scalars, pair shape, and recovery semantics."""
    validation = _validation()
    array_why = "array-conversion-why-sentinel"
    array_how = "array-conversion-how-sentinel"
    scale_why = "scale-shape-why-sentinel"
    scale_how = "scale-shape-how-sentinel"

    np.testing.assert_array_equal(
        validation._host_array((1, 2), why=array_why, how=array_how),
        np.asarray((1, 2)),
    )
    assert validation._finite_pair((np.float32(1.5), np.int64(2)), name="point") == (1.5, 2.0)
    assert validation._positive_scalar_or_pair(
        np.float32(2.0),
        name="scale",
        why=scale_why,
        how=scale_how,
    ) == (2.0, 2.0)
    assert validation._positive_scalar_or_pair(
        np.asarray((np.float32(2.0), np.int64(3))),
        name="scale",
        why=scale_why,
        how=scale_how,
    ) == (2.0, 3.0)

    with pytest.raises(ValueError) as array_error:
        validation._host_array([[1], [2, 3]], why=array_why, how=array_how)
    slots = _assert_error_case(array_error.value, parameter="value", rejected_value=[[1], [2, 3]])
    assert slots["why"] == array_why
    assert slots["how"] == array_how

    with pytest.raises(ValueError) as pair_error:
        validation._finite_pair((1.0, 2.0, 3.0), name="point")
    _assert_error_case(pair_error.value, parameter="point", rejected_value=(1.0, 2.0, 3.0))

    with pytest.raises(ValueError) as pair_value_error:
        validation._finite_pair((1.0, float("nan")), name="point")
    slots = _assert_error_case(pair_value_error.value, parameter="point[1]", rejected_value=float("nan"))
    assert _requirement_for(slots["why"], parameter="point[1]") == "finite"

    with pytest.raises(ValueError) as scalar_or_pair_error:
        validation._positive_scalar_or_pair(
            True,
            name="scale",
            why=scale_why,
            how=scale_how,
        )
    slots = _assert_error_case(scalar_or_pair_error.value, parameter="scale", rejected_value=True)
    assert (slots["why"], slots["how"]) == (scale_why, scale_how)


def test_bool_and_closed_token_contract_preserves_validation_variants() -> None:
    """REQ-TEST-001: preserve strict bool and all three closed-token boundary variants."""
    validation = _validation()
    bool_why = "adapt-type-why-sentinel"
    bool_how = "adapt-type-how-sentinel"
    colorspace_why = "output-colorspace-why-sentinel"
    colorspace_how = "output-colorspace-how-sentinel"

    assert (
        validation._strict_bool(
            True,
            name="adapt",
            why=bool_why,
            how=bool_how,
        )
        is True
    )
    with pytest.raises(ValueError) as bool_error:
        validation._strict_bool(
            np.bool_(True),
            name="adapt",
            why=bool_why,
            how=bool_how,
        )
    slots = _assert_error_case(bool_error.value, parameter="adapt", rejected_value=np.bool_(True))
    assert (slots["why"], slots["how"]) == (bool_why, bool_how)

    numpy_token = np.str_("linear")
    preserved = validation._closed_token(numpy_token, axis="gamma", accepted=("linear",))
    assert preserved is numpy_token
    normalized = validation._normalized_closed_token(numpy_token, axis="gamma", accepted=("linear",))
    assert normalized == "linear" and type(normalized) is str

    class TokenLookalike:
        def __eq__(self, other: object) -> bool:
            return other == "normal"

        def __str__(self) -> str:
            return "normal"

        def __repr__(self) -> str:
            return "LOOKALIKE"

    lookalike = TokenLookalike()
    assert validation._normalized_closed_token(lookalike, axis="blend", accepted=("normal",)) == "normal"
    with pytest.raises(ValueError) as explicit_str_error:
        validation._closed_str_token(lookalike, axis="blend", accepted=("normal",))
    slots = _assert_error_case(explicit_str_error.value, parameter="blend", rejected_value=lookalike)
    assert _accepted_values(slots["how"]) == ("normal",)

    with pytest.raises(ValueError) as custom_token_error:
        validation._closed_token(
            "ACES",
            axis="output_colorspace",
            accepted=("sRGB",),
            why=colorspace_why,
            how=colorspace_how,
        )
    slots = _assert_error_case(custom_token_error.value, parameter="output_colorspace", rejected_value="ACES")
    assert (slots["why"], slots["how"]) == (colorspace_why, colorspace_how)


def test_validation_primitives_bind_consumer_adapters_to_core_source() -> None:
    """REQ-TEST-003 structure contract: consumer adapters return the core call result; as a scoped legacy
    regression, removed domain-local primitive definitions remain absent."""
    validation = _validation()
    primitive_names = {
        "_bounded_real",
        "_closed_str_token",
        "_closed_token",
        "_finite_pair",
        "_finite_real",
        "_host_array",
        "_normalized_closed_token",
        "_positive_real",
        "_positive_scalar_or_pair",
        "_strict_bool",
    }

    validation_definitions = {
        node.name
        for node in ast.parse(inspect.getsource(validation)).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert primitive_names <= validation_definitions

    local_primitive_names = {
        "pixtreme._draw.shapes": {
            "_boolean",
            "_finite_real",
            "_nonnegative_real",
            "_opacity",
            "_point",
            "_positive_real",
            "_token",
            "_validate_frame",
        },
        "pixtreme._draw.text": {"_strict_bool"},
        "pixtreme._generate.patterns": {
            "_cell",
            "_finite_real",
            "_point",
            "_positive_real",
            "_token",
        },
        "pixtreme._generate.noise": {"_nonnegative_real", "_strict_bool"},
        "pixtreme._composite.merge": {"_closed_token", "_require_image_float32", "_scale"},
        "pixtreme._color.transform": {"_validate_axis_token"},
        "pixtreme._io.wire.sampling": {"_token"},
    }
    for module_name, absent in local_primitive_names.items():
        module = importlib.import_module(module_name)
        local_definitions = {
            node.name
            for node in ast.parse(inspect.getsource(module)).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not local_definitions & absent, f"{module_name} retained {sorted(local_definitions & absent)}"

    for module_name in ("pixtreme._draw.shapes", "pixtreme._draw.text", "pixtreme._generate.patterns"):
        module = importlib.import_module(module_name)
        adapter = ast.parse(inspect.getsource(module._host_array)).body[0]
        assert isinstance(adapter, (ast.FunctionDef, ast.AsyncFunctionDef))
        assert len(adapter.body) == 1 and isinstance(adapter.body[0], ast.Return)
        call = adapter.body[0].value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Attribute)
        assert isinstance(call.func.value, ast.Name)
        assert (call.func.value.id, call.func.attr) == ("_validation", "_host_array")
        assert len(call.args) == 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == "value"
        assert {keyword.arg for keyword in call.keywords} == {"why", "how"}

    frame = importlib.import_module("pixtreme._core.frame")
    frame_definitions = {
        node.name
        for node in ast.parse(inspect.getsource(frame)).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"_validate_frame", "_validate_float32_frame"} <= frame_definitions
