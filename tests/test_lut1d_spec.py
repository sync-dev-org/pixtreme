"""Specification tests for GPU-resident one-dimensional LUTs."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_args, get_type_hints

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]


def _assert_actionable(error: pytest.ExceptionInfo[BaseException]) -> None:
    message = str(error.value)
    assert message.index("why=") < message.index("what=") < message.index("how=")


def _frame(values: np.ndarray, *, channels: tuple[str, ...] = ("R", "G", "B")) -> px.core.Frame:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    return px.io.from_array(
        cp.asarray(array),
        colorspace="ACEScg",
        gamma="linear",
        channels=channels,
        matrix="native",
    )


def _identity_lut1d(*, size: int = 4) -> px.core.Lut1D:
    axis = cp.linspace(0.0, 1.0, size, dtype=cp.float32)
    return px.core.Lut1D(cp.repeat(axis[:, None], 3, axis=1))


def _host_apply_lut1d(
    values: np.ndarray,
    table: np.ndarray,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
) -> np.ndarray:
    """Independent scalar host oracle derived from v1-lut-extensions acceptance 6 and 28."""
    output = np.empty_like(values)
    size = table.shape[0]
    for row in range(values.shape[0]):
        for channel in range(3):
            position = (
                (float(values[row, channel]) - domain_min[channel])
                / (domain_max[channel] - domain_min[channel])
                * (size - 1)
            )
            position = min(max(position, 0.0), float(size - 1))
            lower = min(int(np.floor(position)), size - 2)
            fraction = position - lower
            output[row, channel] = table[lower, channel] + fraction * (
                table[lower + 1, channel] - table[lower, channel]
            )
    return output


def test_lut1d_is_public_frozen_slotted_and_reference_preserving() -> None:
    """v1-lut-extensions acceptance 1 and 3: Lut1D is a frozen slotted public GPU value type."""
    data = cp.asarray(
        ((-1.0, 2.0, 0.0), (0.5, -3.0, 4.0), (2.0, 1.0, -2.0)),
        dtype=cp.float32,
    )

    lut = px.core.Lut1D(data)

    assert lut.data is data
    assert lut.data.shape == (3, 3)
    assert lut.data.dtype == cp.float32
    assert lut.domain_min == (0.0, 0.0, 0.0)
    assert lut.domain_max == (1.0, 1.0, 1.0)
    assert not hasattr(lut, "__dict__")
    with pytest.raises(FrozenInstanceError):
        lut.domain_min = (-1.0, -1.0, -1.0)  # type: ignore[misc]


@pytest.mark.parametrize(
    "case",
    (
        "not-cupy",
        "rank",
        "channels",
        "size",
        "dtype",
        "domain-length",
        "domain-string",
        "domain-bool",
        "domain-nonfinite",
        "domain-order",
    ),
)
def test_lut1d_rejects_every_construction_invariant_actionably(case: str) -> None:
    """v1-lut-extensions acceptance 2: invalid Lut1D data and domains fail with three-part ValueError."""
    data: object = cp.zeros((2, 3), dtype=cp.float32)
    kwargs: dict[str, object] = {}
    if case == "not-cupy":
        data = np.zeros((2, 3), dtype=np.float32)
    elif case == "rank":
        data = cp.zeros((2, 1, 3), dtype=cp.float32)
    elif case == "channels":
        data = cp.zeros((2, 4), dtype=cp.float32)
    elif case == "size":
        data = cp.zeros((1, 3), dtype=cp.float32)
    elif case == "dtype":
        data = cp.zeros((2, 3), dtype=cp.float16)
    elif case == "domain-length":
        kwargs["domain_min"] = (0.0, 0.0)
    elif case == "domain-string":
        kwargs["domain_min"] = "0 0 0"
    elif case == "domain-bool":
        kwargs["domain_min"] = (False, 0.0, 0.0)
    elif case == "domain-nonfinite":
        kwargs["domain_max"] = (1.0, np.inf, 1.0)
    else:
        kwargs["domain_min"] = (0.0, 2.0, 0.0)
        kwargs["domain_max"] = (1.0, 2.0, 1.0)

    with pytest.raises(ValueError) as error:
        px.core.Lut1D(data, **kwargs)
    _assert_actionable(error)


def test_lut1d_normalizes_domain_float_conversion_failures_actionably() -> None:
    """v1-lut-extensions acceptance 2: domain conversion failures stay three-part ValueError values."""
    data = cp.zeros((2, 3), dtype=cp.float32)

    with pytest.raises(ValueError) as error:
        px.core.Lut1D(data, domain_max=(10**10000, 1, 1))

    _assert_actionable(error)


def test_lut1d_and_apply_lut_public_signatures_are_exact() -> None:
    """v1-lut-extensions acceptance 4: Lut1D and apply_lut expose the specified public grammar and union."""
    constructor = inspect.signature(px.core.Lut1D)
    assert tuple(constructor.parameters) == ("data", "domain_min", "domain_max")
    assert constructor.parameters["domain_min"].default == (0.0, 0.0, 0.0)
    assert constructor.parameters["domain_max"].default == (1.0, 1.0, 1.0)

    apply_signature = inspect.signature(px.color.apply_lut)
    assert tuple(apply_signature.parameters) == ("frame", "lut", "interpolation")
    assert apply_signature.parameters["interpolation"].default is None
    hints = get_type_hints(px.color.apply_lut)
    assert set(get_args(hints["lut"])) == {px.core.Lut, px.core.Lut1D}
    assert set(get_args(hints["interpolation"])) == {px.core.Interpolation, type(None)}
    assert hints["return"] is px.core.Frame


@pytest.mark.parametrize("interpolation", (None, "linear"))
def test_apply_lut1d_matches_an_independent_asymmetric_host_oracle(interpolation: str | None) -> None:
    """v1-lut-extensions acceptance 5-6 and 28: 1D linear lookup matches an independent host oracle."""
    table = np.asarray(
        (
            (-1.0, 4.0, -2.0),
            (0.5, 3.0, 0.25),
            (2.0, 0.0, 0.75),
            (-3.0, 1.5, 3.0),
        ),
        dtype=np.float32,
    )
    domain_min = (-1.0, 10.0, -4.0)
    domain_max = (2.0, 14.0, 4.0)
    values = np.asarray(
        ((-2.0, 10.0, -4.0), (-0.25, 11.0, -1.0), (1.5, 13.5, 2.0), (3.0, 15.0, 5.0)),
        dtype=np.float32,
    )
    source = _frame(values.reshape(1, 4, 3))
    lut = px.core.Lut1D(cp.asarray(table), domain_min=domain_min, domain_max=domain_max)

    result = px.color.apply_lut(source, lut=lut, interpolation=interpolation)

    expected = _host_apply_lut1d(values, table, domain_min, domain_max).reshape(1, 4, 3)
    np.testing.assert_allclose(cp.asnumpy(result.data), expected, rtol=0.0, atol=3e-7)
    assert float(cp.min(result.data).get()) < 0.0
    assert float(cp.max(result.data).get()) > 1.0


def test_apply_lut1d_preserves_wide_finite_domain_affine_mapping() -> None:
    """v1-lut-extensions acceptance 6: finite domains wider than float32 retain their affine mapping."""
    table = np.asarray(((0.0, 2.0, -1.0), (1.0, -2.0, 3.0)), dtype=np.float32)
    values = np.zeros((1, 3), dtype=np.float32)
    domain_min = (-1e300, -1e300, -1e300)
    domain_max = (1e300, 1e300, 1e300)
    lut = px.core.Lut1D(cp.asarray(table), domain_min=domain_min, domain_max=domain_max)

    with np.errstate(over="raise"):
        result = px.color.apply_lut(_frame(values.reshape(1, 1, 3)), lut=lut)

    expected = _host_apply_lut1d(values, table, domain_min, domain_max).reshape(1, 1, 3)
    np.testing.assert_allclose(cp.asnumpy(result.data), expected, rtol=0.0, atol=1e-7)


def test_apply_lut_resolves_type_specific_defaults_and_rejects_mismatched_tokens() -> None:
    """v1-lut-extensions acceptance 5: None resolves per LUT type and cross-type interpolation tokens fail."""
    scalar_vertices = np.asarray(
        (
            ((0.0, 4.0), (2.0, 32.0)),
            ((1.0, 16.0), (8.0, 64.0)),
        ),
        dtype=np.float32,
    ) / np.float32(64.0)
    lut3d = px.core.Lut(cp.asarray(np.repeat(scalar_vertices[..., None], 3, axis=-1)))
    source3d = _frame(np.asarray((0.75, 0.5, 0.25), dtype=np.float32))
    cp.testing.assert_array_equal(
        px.color.apply_lut(source3d, lut=lut3d).data,
        px.color.apply_lut(source3d, lut=lut3d, interpolation="tetrahedral").data,
    )

    lut1d = _identity_lut1d()
    source1d = _frame(np.asarray((0.125, 0.5, 0.875), dtype=np.float32))
    cp.testing.assert_array_equal(
        px.color.apply_lut(source1d, lut=lut1d).data,
        px.color.apply_lut(source1d, lut=lut1d, interpolation="linear").data,
    )

    for lut, token, expected_tokens in (
        (lut3d, "linear", ("trilinear", "tetrahedral")),
        (lut1d, "trilinear", ("linear",)),
        (lut1d, "tetrahedral", ("linear",)),
        (lut1d, "Linear", ("linear",)),
    ):
        with pytest.raises(ValueError) as error:
            px.color.apply_lut(source1d, lut=lut, interpolation=token)
        _assert_actionable(error)
        assert all(expected in str(error.value) for expected in expected_tokens)


def test_apply_lut1d_preserves_metadata_non_rgb_bits_storage_and_inputs() -> None:
    """v1-lut-extensions acceptance 7 and 28: 1D lookup preserves metadata, non-RGB bits, and both inputs."""
    values = np.asarray(
        ((0.0, 0.25, 0.5, 0.0, 0.75), (0.0, 0.9, 0.1, 0.0, 0.4)),
        dtype=np.float32,
    ).reshape(1, 2, 5)
    bits = values.view(np.uint32)
    bits[..., 0] = np.uint32(0x80000000)
    bits[..., 3] = np.uint32(0x7FC01234)
    source = _frame(values, channels=("A", "B", "R", "Z", "G"))
    source_before = source.data.copy()

    table = np.asarray(
        ((-2.0, 0.0, 2.0), (0.0, 1.0, 1.0), (4.0, -1.0, 3.0), (1.0, 2.0, -4.0)),
        dtype=np.float32,
    )
    backing = cp.zeros((4, 6), dtype=cp.float32)
    backing[:, ::2] = cp.asarray(table)
    strided = backing[:, ::2]
    assert not strided.flags.c_contiguous
    lut = px.core.Lut1D(strided)
    lut_before = lut.data.copy()

    result = px.color.apply_lut(source, lut=lut)

    assert result.data is not source.data
    assert result.data.flags.c_contiguous
    assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
        source.colorspace,
        source.gamma,
        source.channels,
        source.matrix,
    )
    output_bits = cp.asnumpy(result.data).view(np.uint32)
    source_bits = cp.asnumpy(source.data).view(np.uint32)
    np.testing.assert_array_equal(output_bits[..., (0, 3)], source_bits[..., (0, 3)])
    cp.testing.assert_array_equal(source.data, source_before)
    cp.testing.assert_array_equal(lut.data, lut_before)


@pytest.mark.parametrize("case", ("dtype", "channels"))
def test_apply_lut1d_enforces_the_shared_frame_contract(case: str) -> None:
    """v1-lut-extensions acceptance 7: Lut1D shares float32 and complete RGB-label validation with Lut."""
    if case == "dtype":
        source = _frame(np.zeros(3, dtype=np.float16))
    else:
        source = _frame(np.zeros(3, dtype=np.float32), channels=("R", "G", "A"))

    with pytest.raises(ValueError) as error:
        px.color.apply_lut(source, lut=_identity_lut1d())
    _assert_actionable(error)


def test_lut1d_token_documentation_is_type_specific_and_complete() -> None:
    """v1-lut-extensions acceptance 27: token reference documents linear and both type-specific LUT subsets."""
    path = ROOT / "docs_site" / "tokens.md"
    if not path.is_file():
        pytest.skip("repo-only documentation contract: docs_site/tokens.md is absent from this distribution")
    section = path.read_text(encoding="utf-8").split("## interpolation\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for required in (
        "`linear`",
        "`Lut1D`",
        "default `linear`",
        "per-channel declared `domain`",
        "clamped",
        "not clipped",
    ):
        assert required in section
