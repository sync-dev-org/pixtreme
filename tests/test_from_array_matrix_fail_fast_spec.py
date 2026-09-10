"""Specification tests for from_array matrix token entry validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert message.index("why=") < message.index("; what=") < message.index("; how=")


def _trace_from_array_kernel(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
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
    return launches


@pytest.mark.parametrize(
    ("matrix", "repacking"),
    (
        ("unknown-matrix", {"dtype": "float32"}),
        ("unknown-matrix", {"scale": 2.0}),
        ("unknown-matrix", {"bit_depth": None, "copy": True}),
        (7, {"dtype": "float32"}),
        ("", {"dtype": "float32"}),
    ),
)
def test_from_array_rejects_invalid_matrix_before_any_repacking(
    monkeypatch: pytest.MonkeyPatch, matrix: object, repacking: dict[str, object]
) -> None:
    """v1-from-array-matrix-fail-fast acceptance 1 and 2: invalid matrix fails at the entry with a plain ValueError."""
    import cupy as cp

    launches = _trace_from_array_kernel(monkeypatch)
    values = np.asarray([[[0.0, 0.5, 1.0], [1.0, -0.25, 1.25]]], dtype=np.float16)
    source = cp.asarray(values)
    source_pointer = source.data.ptr
    with pytest.raises(ValueError) as error:
        px.io.from_array(
            source,
            colorspace="ACEScg",
            gamma="linear",
            channels="RGB",
            matrix=matrix,  # type: ignore[arg-type]
            **repacking,  # type: ignore[arg-type]
        )

    assert type(error.value) is ValueError
    assert launches == []
    _assert_actionable(error)
    message = str(error.value)
    assert repr(matrix) in message.split("; what=")[1].split("; how=")[0]
    assert "BT.709" in message.split("; how=")[1]
    assert source.data.ptr == source_pointer
    assert source.shape == (1, 2, 3)
    assert source.dtype == cp.float16
    np.testing.assert_array_equal(cp.asnumpy(source), values)


def test_from_array_invalid_matrix_is_rejected_at_the_same_stage_as_other_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-from-array-matrix-fail-fast acceptance 4: matrix joins layout, colorspace, gamma, and channels at the entry."""
    import cupy as cp

    launches = _trace_from_array_kernel(monkeypatch)
    source = cp.zeros((2, 2, 3), dtype=cp.float16)
    invalid_calls: tuple[dict[str, object], ...] = (
        {"layout": "unknown-layout"},
        {"colorspace": "unknown-colorspace"},
        {"gamma": "unknown-gamma"},
        {"channels": "RGBA"},
        {"matrix": "unknown-matrix"},
    )
    for overrides in invalid_calls:
        arguments: dict[str, object] = {
            "colorspace": "ACEScg",
            "gamma": "linear",
            "channels": "RGB",
            "matrix": "BT.709",
            "dtype": "float32",
        }
        arguments.update(overrides)
        with pytest.raises(ValueError) as error:
            px.io.from_array(source, **arguments)  # type: ignore[arg-type]
        assert type(error.value) is ValueError
        _assert_actionable(error)
    assert launches == []


def test_from_array_valid_matrix_spellings_and_none_keep_canonical_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-from-array-matrix-fail-fast acceptance 3: valid matrix spellings and None resolve as before the entry check."""
    import cupy as cp

    values = np.asarray([[[0.0, 0.5, 1.0], [1.0, -0.25, 1.25]]], dtype=np.float32)
    expectations: tuple[tuple[object, str | None], ...] = (
        ("BT.709", "BT.709"),
        ("b t_7-0.9", "BT.709"),
        ("bt709", "BT.709"),
        (None, None),
    )
    for matrix, canonical in expectations:
        source = cp.asarray(values)
        zero_copy = px.io.from_array(
            source,
            colorspace="ACEScg",
            gamma="linear",
            channels="RGB",
            matrix=matrix,  # type: ignore[arg-type]
        )
        repacked = px.io.from_array(
            source,
            colorspace="ACEScg",
            gamma="linear",
            channels="RGB",
            matrix=matrix,  # type: ignore[arg-type]
            dtype="float16",
        )
        assert zero_copy.matrix == canonical
        assert repacked.matrix == canonical
        assert zero_copy.data is source
        assert repacked.data is not source
        assert (zero_copy.colorspace, zero_copy.gamma, zero_copy.channels) == ("ACEScg", "linear", ("R", "G", "B"))
        assert repacked.data.dtype == cp.float16
        np.testing.assert_array_equal(cp.asnumpy(zero_copy.data), values)
        np.testing.assert_array_equal(cp.asnumpy(repacked.data), values.astype(np.float16))
