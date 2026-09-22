"""Behavior and structural contracts for the opt-in shared 3DL shaper."""

from __future__ import annotations

import dataclasses
import inspect
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import get_args, get_type_hints

import cupy as cp
import numpy as np
import pytest
from lut_shaper_oracle import (
    CASES,
    boundary_inputs,
    fixture_bytes,
    float32_simplex_variants,
    host_apply,
    host_cube,
    load_fixture,
    quantized_tables,
    three_dl_text,
)
from test_lut_io_extensions_spec import (
    _cube_1d_text,
    _cube_3d_text,
    _spi1d_text,
    _spi3d_text,
    _tetrahedral_host,
)

import pixtreme as px

FIXTURES = Path(__file__).parent / "data" / "lut_shaper"


def _frame(points: np.ndarray, *, channels: tuple[str, ...] = ("R", "G", "B")) -> px.core.Frame:
    return px.io.from_array(
        cp.asarray(points.reshape(1, -1, len(channels))),
        colorspace="ACEScg",
        gamma="linear",
        channels=channels,
        matrix="BT.709",
    )


def _cube(edge: int) -> np.ndarray:
    r, g, b = np.meshgrid(*([np.linspace(0, 1, edge)] * 3), indexing="ij")
    return np.stack((1.7 * r * g - 0.3 * b, 0.2 * r + g * b, 1.2 * b + 0.3 * r * g), axis=-1).astype(np.float32)


def _bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def _actionable(error: BaseException) -> str:
    message = str(error)
    assert re.fullmatch(r"why=.+; what=.+; how=.+", message, re.DOTALL)
    return message


def test_lut_shaper_constructor_and_read_signatures_are_keyword_only() -> None:
    """v1-lut-shaper acceptance 1 and 16: only the fixed optional public parameters are added."""
    constructor = inspect.signature(px.core.Lut)
    assert tuple(constructor.parameters) == ("data", "domain_min", "domain_max", "shaper")
    assert constructor.parameters["shaper"].kind is inspect.Parameter.KEYWORD_ONLY
    assert constructor.parameters["shaper"].default is None
    assert constructor.parameters["domain_min"].default == (0.0, 0.0, 0.0)
    assert constructor.parameters["domain_max"].default == (1.0, 1.0, 1.0)
    assert set(get_args(get_type_hints(px.core.Lut)["shaper"])) == {cp.ndarray, type(None)}
    for function, first in ((px.io.read_lut, "path"), (px.io.decode_lut, "data")):
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == (first, "preserve_shaper")
        option = signature.parameters["preserve_shaper"]
        assert option.kind is inspect.Parameter.KEYWORD_ONLY and option.default is False
        hints = get_type_hints(function)
        assert hints["preserve_shaper"] is bool
        assert set(get_args(hints["return"])) == {px.core.Lut, px.core.Lut1D}


@pytest.mark.parametrize("strided", (False, True))
def test_shaper_retains_shared_finite_nonmonotonic_output_by_reference(strided: bool) -> None:
    """v1-lut-shaper acceptance 2, 6 and 19: decreasing/equal/out-of-range samples remain caller-owned."""
    cube = cp.asarray(_cube(4))
    host = np.asarray((-0.5, 1.5, 1.5, -0.25), dtype=np.float32)
    backing = cp.asarray(np.repeat(host, 2) if strided else host)
    shaper = backing[::2] if strided else backing
    lut = px.core.Lut(cube, shaper=shaper)
    assert lut.data is cube and lut.shaper is shaper
    _bits(cp.asnumpy(shaper), host)
    assert px.core.Lut(cube).shaper is None
    assert {field.name for field in dataclasses.fields(lut)} == {"data", "domain_min", "domain_max", "shaper"}
    assert not hasattr(lut, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        lut.shaper = None
    shaper[1] = np.float32(0.25)
    assert float(lut.shaper[1]) == 0.25


@pytest.mark.parametrize(
    "case", ("host", "rank", "rgb-table", "float16", "float64", "empty", "one", "edge", "nan", "inf", "-inf")
)
def test_shaper_rejects_invalid_storage_and_nonfinite_samples(case: str) -> None:
    """v1-lut-shaper acceptance 2: each invalid shaper invariant is an actionable ValueError."""
    cube = cp.asarray(_cube(3))
    shaper = cp.asarray((0, 0.2, 1), dtype=cp.float32)
    if case == "host":
        shaper = np.asarray((0, 0.2, 1), dtype=np.float32)
    elif case == "rank":
        shaper = shaper.reshape(1, 3)
    elif case == "rgb-table":
        shaper = cp.zeros((3, 3), dtype=cp.float32)
    elif case in ("float16", "float64"):
        shaper = shaper.astype(case)
    elif case in ("empty", "one", "edge"):
        shaper = cp.zeros({"empty": 0, "one": 1, "edge": 2}[case], dtype=cp.float32)
    else:
        shaper[1] = float(case)
    with pytest.raises(ValueError) as error:
        px.core.Lut(cube, shaper=shaper)
    assert "shaper" in _actionable(error.value).lower()


@pytest.mark.skipif(
    "cp.cuda.runtime.getDeviceCount() < 2",
    reason="requires two visible CUDA devices; remove this guard when all test lanes provide two devices",
)
def test_shaper_rejects_a_different_cuda_device() -> None:
    """v1-lut-shaper acceptance 2: a foreign-device table fails without an implicit device copy.

    The string skipif is evaluated at test setup, never during collection.
    """
    with cp.cuda.Device(0):
        cube = cp.asarray(_cube(2))
    with cp.cuda.Device(1):
        shaper = cp.asarray((0, 1), dtype=cp.float32)
    with cp.cuda.Device(0), pytest.raises(ValueError) as error:
        px.core.Lut(cube, shaper=shaper)
    assert "device" in _actionable(error.value).lower()


@pytest.mark.parametrize("lustre", (False, True))
@pytest.mark.parametrize("mode", ("identity", "half-code", "nonidentity"))
def test_3dl_default_bake_and_opt_in_use_independent_code_oracles(tmp_path: Path, lustre: bool, mode: str) -> None:
    """v1-lut-shaper acceptance 3-4 and 18: format codes fix bit-exact default bake and preserved state.

    Headerless identity/nonidentity use edge 4 and half-code uses edge 11
    (255/10 = 25.5), keeping the spacing row above the existing sniff's three-token
    cutoff. Lustre uses rounded edge 17. Bake uses the independent float64 host tetrahedral formula
    on unrounded normalized source codes, followed by exactly one float32 cast.
    """
    edge = 17 if lustre else {"identity": 4, "half-code": 11, "nonidentity": 4}[mode]
    scale = 1023 if lustre else 255
    axis = np.linspace(0, scale, edge)
    if mode == "nonidentity":
        spacing = np.floor((axis / scale) ** 0.45 * scale + 0.5).astype(np.int64)
    else:
        # For Lustre identity is represented within half a source code.
        spacing = np.floor(axis + (0 if mode == "half-code" else 0.5)).astype(np.int64)
        if lustre and mode == "half-code":
            spacing = np.floor(axis + 0.5).astype(np.int64)
    _, codes = quantized_tables(edge, "log")
    if not lustre:
        codes = np.floor(codes / 1023 * 255 + 0.5).astype(np.int64)
    text = three_dl_text(spacing, codes, lustre=lustre)
    path = tmp_path / "input.3dl"
    path.write_text(text, encoding="utf-8")
    raw = codes.astype(np.float64) / scale
    normalized = spacing.astype(np.float64) / scale
    baked = raw.copy()
    if mode == "nonidentity":
        for index in np.ndindex((edge, edge, edge)):
            baked[index] = _tetrahedral_host(raw, normalized[list(index)])
    expected_baked = baked.astype(np.float32)
    for reader, source in ((px.io.read_lut, path), (px.io.decode_lut, text.encode())):
        implicit = reader(source)
        _bits(cp.asnumpy(implicit.data), expected_baked)
        explicit = reader(source, preserve_shaper=False)
        strict = reader(source, preserve_shaper=True)
        for lut in (implicit, explicit, strict):
            assert type(lut) is px.core.Lut
            assert lut.domain_min == (0.0, 0.0, 0.0) and lut.domain_max == (1.0, 1.0, 1.0)
            assert lut.data.shape == (edge, edge, edge, 3)
        for lut in (implicit, explicit):
            assert lut.shaper is None
            _bits(cp.asnumpy(lut.data), expected_baked)
        _bits(cp.asnumpy(strict.data), raw.astype(np.float32))
        if mode == "nonidentity":
            _bits(cp.asnumpy(strict.shaper), normalized.astype(np.float32))
        else:
            assert strict.shaper is None


@pytest.mark.parametrize("spacing", ((255,), (0, 255), (0, 127, 255)), ids=("one-token", "two-tokens", "three-tokens"))
@pytest.mark.parametrize("option", ("default", False, True), ids=("default", "bake", "preserve"))
def test_decode_lut_rejects_short_headerless_spacing_rows(spacing: tuple[int, ...], option: str | bool) -> None:
    """v1-lut-shaper acceptance 3: opt-in preserves the existing format-identification boundary.

    The fixed specification's read/state and non-scope clauses freeze format
    sniffing and existing grammar. Numeric payloads with at most three tokens in
    their first row remain unidentified, even with an otherwise complete cube.
    """
    _, codes = quantized_tables(len(spacing), "log")
    payload = three_dl_text(np.asarray(spacing), codes).encode("utf-8")
    with pytest.raises(ValueError) as error:
        if option == "default":
            px.io.decode_lut(payload)
        else:
            px.io.decode_lut(payload, preserve_shaper=option)
    message = _actionable(error.value).lower()
    assert "format" in message and "could not be identified" in message


@pytest.mark.parametrize(
    "suffix,text",
    ((".cube", _cube_1d_text()), (".cube", _cube_3d_text()), (".spi1d", _spi1d_text()), (".spi3d", _spi3d_text())),
)
def test_preserve_shaper_leaves_other_formats_unshaped(tmp_path: Path, suffix: str, text: str) -> None:
    """v1-lut-shaper acceptance 4 and 18: opt-in never invents a shaper for another format."""
    path = tmp_path / f"input{suffix}"
    path.write_text(text, encoding="utf-8")
    for reader, source in ((px.io.read_lut, path), (px.io.decode_lut, text.encode())):
        old = reader(source)
        strict = reader(source, preserve_shaper=True)
        assert type(strict) is type(old)
        assert strict.domain_min == old.domain_min and strict.domain_max == old.domain_max
        _bits(cp.asnumpy(strict.data), cp.asnumpy(old.data))
        if isinstance(strict, px.core.Lut):
            assert strict.shaper is None
        else:
            assert not hasattr(strict, "shaper")


@pytest.mark.parametrize(
    "payload,detail",
    (
        (b"LUT_1D_SIZE nope\n0 255\n" + b"0 0 0\n" * 8, "LUT_1D_SIZE"),
        (b"LUT_1D_SIZE 2\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n", "Cube"),
    ),
    ids=("selected-parser-error", "combined-cube"),
)
def test_opt_in_keeps_parser_failure_and_rejects_combined_cube(payload: bytes, detail: str) -> None:
    """v1-lut-shaper acceptance 4 and 18: opting in cannot enable fallback parsing or a 1D+3D Cube variant."""
    with pytest.raises(ValueError) as error:
        px.io.decode_lut(payload, preserve_shaper=True)
    assert detail.lower() in _actionable(error.value).lower()


@pytest.mark.parametrize("bad", (None, 0, 1, "True", "false", [], np.bool_(True)))
@pytest.mark.parametrize("entry", ("file", "bytes"))
def test_non_boolean_opt_in_fails_before_io_parse_or_transfer(
    monkeypatch: pytest.MonkeyPatch, bad: object, entry: str
) -> None:
    """v1-lut-shaper acceptance 5: raw invalid option wins over corrupt bytes and unreadable paths."""
    touched = []

    def forbidden(*args, **kwargs):
        touched.append(True)
        raise AssertionError("invalid Boolean must be rejected before this boundary")

    class UnreadablePath:
        def __fspath__(self):
            return forbidden()

    class UndecodableBytes(bytes):
        def decode(self, *args, **kwargs):
            return forbidden()

    monkeypatch.setattr(cp, "asarray", forbidden)
    source = UnreadablePath() if entry == "file" else UndecodableBytes(b"\xff")
    reader = px.io.read_lut if entry == "file" else px.io.decode_lut
    with pytest.raises(ValueError) as error:
        reader(source, preserve_shaper=bad)
    message = _actionable(error.value)
    assert repr(bad) in message and "True" in message and "False" in message
    assert touched == []


@pytest.mark.parametrize("edge", (2, 3, 17))
@pytest.mark.parametrize("interpolation", (None, "tetrahedral", "trilinear"))
def test_shaped_apply_matches_independent_two_stage_domain_and_clamp_oracle(
    edge: int, interpolation: str | None
) -> None:
    """v1-lut-shaper acceptance 6-8 and 13: domain -> linear shaper -> cube clamp -> unclipped output.

    2e-6 absolute allows about 17 float32 epsilons for two stages, as AC-11-10;
    this host contract also covers out-of-domain points excluded from OCIO parity.
    """
    cube = _cube(edge)
    shaper = np.resize(np.asarray((-0.25, 1.5, 0.1, 0.1), dtype=np.float32), edge)
    lower, upper = (-2.0, 1.0, -0.5), (2.0, 3.0, 4.5)
    normalized = boundary_inputs(shaper.astype(np.float64))
    normalized = np.concatenate(
        (normalized, [[-1, 0.25, 2], [0, 0, 0], [1, 1, 1]], np.stack([np.linspace(0, 1, edge)] * 3, axis=-1))
    )
    points = (np.asarray(lower) + normalized * (np.asarray(upper) - lower)).astype(np.float32)
    lut = px.core.Lut(cp.asarray(cube), lower, upper, shaper=cp.asarray(shaper))
    result = px.color.apply_lut(_frame(points), lut=lut, interpolation=interpolation)
    expected = host_apply(cube, shaper, points, lower, upper, interpolation or "tetrahedral")
    np.testing.assert_allclose(cp.asnumpy(result.data).reshape(-1, 3), expected, rtol=0, atol=2e-6)
    assert np.any(expected < 0) and np.any(expected > 1)


@pytest.mark.parametrize("direction", (1, -1), ids=("increasing", "decreasing"))
@pytest.mark.parametrize("interpolation", (None, "tetrahedral", "trilinear"))
def test_shaped_apply_interpolates_extreme_finite_shaper_before_cube_clamp(
    direction: int, interpolation: str | None
) -> None:
    """v1-lut-shaper acceptance 6, 7 and 13: finite endpoints cannot overflow into the wrong cube edge.

    Opposite float32 extrema have a finite linear midpoint of zero, although
    their difference exceeds float32 range. Both knots and interior points must
    follow the float64 host model. The identity cube exposes the clamped shaper
    result; 2e-6 allows AC-11-10's two-stage float32 interpolation budget.
    """
    axis = np.asarray((0, 1), dtype=np.float32)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    shaper = np.asarray((-direction, direction), dtype=np.float32) * np.finfo(np.float32).max
    points = np.asarray(((0.5, 0.5, 0.5), (0, 0.25, 0.75), (1, 0.75, 0.25)), dtype=np.float32)
    expected = host_apply(cube, shaper, points, interpolation=interpolation or "tetrahedral")
    lut = px.core.Lut(cp.asarray(cube), shaper=cp.asarray(shaper))
    result = px.color.apply_lut(_frame(points), lut=lut, interpolation=interpolation)
    np.testing.assert_allclose(cp.asnumpy(result.data).reshape(-1, 3), expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize("interpolation", (None, "tetrahedral", "trilinear"))
def test_shaped_apply_preserves_labels_metadata_storage_and_all_input_bits(interpolation: str | None) -> None:
    """v1-lut-shaper acceptance 2, 6 and 14: strided state and non-RGB NaN/-zero bits survive.

    RGB comparison allows AC-11-10's 2e-6 interpolation budget; storage is exact.
    """
    cube = _cube(3)
    backing = cp.zeros((3, 3, 3, 6), dtype=cp.float32)
    backing[..., ::2] = cp.asarray(cube)
    shaper_values = np.asarray((0, 0.8, 1), dtype=np.float32)
    shaper_backing = cp.asarray(np.repeat(shaper_values, 2))
    lut = px.core.Lut(backing[..., ::2], shaper=shaper_backing[::2])
    host = np.asarray([[0.2, 0, 0.6, 0, 0.4], [0.9, 0, 0.1, 0, 0.7]], dtype=np.float32)
    host.view(np.uint32)[:, 1] = [0x80000000, 0x7FC12345]
    host.view(np.uint32)[:, 3] = [0x7FC54321, 0x80000000]
    source = _frame(host, channels=("B", "A", "R", "Z", "G"))
    result = px.color.apply_lut(source, lut=lut, interpolation=interpolation)
    assert result.data is not source.data and result.data.data.ptr != source.data.data.ptr
    assert result.data.flags.c_contiguous
    for name in ("colorspace", "gamma", "channels", "matrix"):
        assert getattr(result, name) == getattr(source, name)
    actual = cp.asnumpy(result.data).reshape(-1, 5)
    _bits(actual[:, [1, 3]], host[:, [1, 3]])
    expected = host_apply(cube, shaper_values, host[:, [2, 4, 0]], interpolation=interpolation or "tetrahedral")
    np.testing.assert_allclose(actual[:, [2, 4, 0]], expected, rtol=0, atol=2e-6)
    _bits(cp.asnumpy(source.data).reshape(-1, 5), host)
    _bits(cp.asnumpy(lut.data), cube)
    _bits(cp.asnumpy(lut.shaper), shaper_values)
    # Mutability belongs to the caller: the next call must see updated state, not a cached bake.
    lut.shaper[...] = np.float32(0)
    after = px.color.apply_lut(source, lut=lut, interpolation=interpolation)
    np.testing.assert_array_equal(cp.asnumpy(after.data)[..., [2, 4, 0]], np.broadcast_to(cube[0, 0, 0], (1, 2, 3)))


@pytest.mark.parametrize("case", ("float16", "uint8", "uint16", "uint32", "missing-rgb", "linear", "cubic", "strict"))
def test_shaped_apply_keeps_float32_rgb_and_interpolation_validation(case: str) -> None:
    """v1-lut-shaper acceptance 8, 14 and 18: shaper state cannot broaden dtype, labels or token subsets."""
    lut = px.core.Lut(cp.asarray(_cube(2)), shaper=cp.asarray((0, 0.75), dtype=cp.float32))
    dtype = case if case in ("float16", "uint8", "uint16", "uint32") else "float32"
    labels = ("R", "G", "A") if case == "missing-rgb" else ("R", "G", "B")
    frame = _frame(np.zeros((1, 3), dtype=dtype), channels=labels)
    token = case if case in ("linear", "cubic", "strict") else None
    with pytest.raises(ValueError) as error:
        px.color.apply_lut(frame, lut=lut, interpolation=token)
    message = _actionable(error.value)
    if dtype != "float32":
        guidance = {
            "float16": "cast_dtype",
            "uint8": "dequantize",
            "uint16": "dequantize",
            "uint32": "recode_dtype",
        }[dtype]
        assert "float32" in message and guidance in message
    elif case == "missing-rgb":
        assert "R" in message and "G" in message and "B" in message
    else:
        assert "trilinear" in message and "tetrahedral" in message


@pytest.mark.parametrize("interpolation", (None, "tetrahedral", "trilinear"))
def test_shaped_apply_captures_one_kernel_without_host_transfer_or_intermediate_frame(
    interpolation: str | None,
) -> None:
    """v1-lut-shaper acceptance 7: structural contract observes real CUDA graph nodes and returned Frames.

    Capture forbids synchronous host round-trips; all nodes must be one kernel.
    Profiling observes actual Frame returns rather than replacing internal helpers.
    No kernel, helper, module, or launch-argument name is prescribed.
    """
    lut = px.core.Lut(cp.asarray(_cube(3)), shaper=cp.asarray((0, 0.8, 1), dtype=cp.float32))
    source = _frame(np.asarray([[0.12, 0.43, 0.78]], dtype=np.float32))
    px.color.apply_lut(source, lut=lut, interpolation=interpolation)
    cp.cuda.runtime.deviceSynchronize()
    returned_frames = {}

    def profile(frame, event, arg):
        if event == "return" and isinstance(arg, px.core.Frame):
            returned_frames[id(arg)] = arg
        if event == "return" and frame.f_code.co_name == "__init__":
            instance = frame.f_locals.get("self")
            if isinstance(instance, px.core.Frame):
                returned_frames[id(instance)] = instance

    stream = cp.cuda.Stream(non_blocking=True)
    previous = sys.getprofile()
    with stream:
        stream.begin_capture()
        try:
            sys.setprofile(profile)
            result = px.color.apply_lut(source, lut=lut, interpolation=interpolation)
        finally:
            sys.setprofile(previous)
            graph = stream.end_capture()
    dot = graph.debug_dot_str()
    declarations = re.findall(r'"graph_\d+_node_\d+"\[([^;]+);', dot)
    assert len(declarations) == 1 and 'shape="octagon"' in declarations[0], dot
    assert set(returned_frames) == {id(result)}
    graph.launch(stream=stream)
    stream.synchronize()
    expected = host_apply(
        _cube(3),
        np.asarray((0, 0.8, 1), dtype=np.float32),
        cp.asnumpy(source.data).reshape(-1, 3),
        interpolation=interpolation or "tetrahedral",
    )
    np.testing.assert_allclose(cp.asnumpy(result.data).reshape(-1, 3), expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize("edge,curve", CASES)
@pytest.mark.parametrize("interpolation", (None, "tetrahedral"))
def test_preserved_shaper_matches_pinned_ocio_cpu_fixture(edge: int, curve: str, interpolation: str | None) -> None:
    """v1-lut-shaper acceptance 10: both strict selectors match all seed and boundary points.

    rtol=0, atol=2e-6 is AC-11-10's tenfold margin over the measured host error.
    """
    metadata, text, inputs, expected = load_fixture(FIXTURES, edge, curve)
    lut = px.io.decode_lut(text, preserve_shaper=True)
    result = cp.asnumpy(px.color.apply_lut(_frame(inputs), lut=lut, interpolation=interpolation).data).reshape(-1, 3)
    np.testing.assert_allclose(result, expected, rtol=metadata["rtol"], atol=metadata["atol"])
    assert float(np.max(np.abs(result.astype(np.float64) - expected))) <= 2e-6


def test_default_bake_retains_a_counterexample_to_strict_ocio_parity() -> None:
    """v1-lut-shaper acceptance 3, 10 and 18: same-edge bake must not silently stand in for two stages.

    A point in the first input cell of the fixed 17/log corpus has >0.01 error:
    this loose witness bound distinguishes approximation from the 2e-6 budget.
    """
    _, text, points, expected = load_fixture(FIXTURES, 17, "log")
    baked = px.io.decode_lut(text)
    result = cp.asnumpy(px.color.apply_lut(_frame(points), lut=baked).data).reshape(-1, 3)
    assert baked.data.shape == (17, 17, 17, 3)
    assert np.max(np.abs(result - expected)) > 0.01


@pytest.mark.parametrize("edge,curve", CASES)
def test_ocio_fixture_regenerates_all_bytes_and_matches_3dl_cpu_processor(
    tmp_path: Path, edge: int, curve: str
) -> None:
    """v1-lut-shaper acceptance 10-11: independent OCIO regeneration fixes bytes, provenance and boundary corpus.

    FileTransform and explicit normalized transforms must agree within AC-11-10's
    2e-6 budget. Host barycentric interpolation is a second independent witness.
    """
    import PyOpenColorIO as ocio

    generated = fixture_bytes(edge, curve)
    assert list(generated) == ["metadata.json", "source.3dl", "input.f32", "output.f32"]
    for name, content in generated.items():
        assert content == (FIXTURES / f"{edge}-{curve}" / name).read_bytes(), name
    metadata, text, inputs, expected = load_fixture(FIXTURES, edge, curve)
    assert list(metadata) == sorted(metadata)
    assert metadata["ocio_version"] == "2.5.2" and metadata["dtype"] == "<f4"
    assert metadata["processor_cache_id"] and metadata["cpu_cache_id"] and metadata["tolerance_reason"]
    spacing, cube = quantized_tables(edge, curve)
    _bits(inputs, boundary_inputs(spacing.astype(np.float64) / 1023))
    assert np.all((0 <= inputs) & (inputs <= 1))
    path = tmp_path / "oracle.3dl"
    path.write_bytes(text)
    processor = ocio.Config.CreateRaw().getProcessor(
        ocio.FileTransform(str(path), interpolation=ocio.INTERP_TETRAHEDRAL)
    )
    file_output = inputs.copy()
    processor.getDefaultCPUProcessor().applyRGB(file_output)
    np.testing.assert_allclose(file_output, expected, rtol=0, atol=2e-6)
    host = host_apply((cube / 1023).astype(np.float32), (spacing / 1023).astype(np.float32), inputs)
    np.testing.assert_allclose(host, expected, rtol=0, atol=2e-6)


def test_shaped_write_bakes_nodes_on_gpu_once_and_roundtrips_baked_bits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """v1-lut-shaper acceptance 9 and 19: GPU node bake loses shaper but retains domain and baked bits.

    A dyadic table gives exact float64->float32 node results; near-half-ULP
    values separately distinguish float32 intermediate arithmetic below.
    """
    cube = _cube(3)
    shaper = np.asarray((0, 0.75, 1), dtype=np.float32)
    lower, upper = (-2.0, 1.0, -0.5), (2.0, 3.0, 4.5)
    normalized_nodes = np.stack(np.meshgrid(*([np.linspace(0, 1, 3)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    expected = host_apply(cube, shaper, normalized_nodes).astype(np.float32).reshape(3, 3, 3, 3)
    lut = px.core.Lut(cp.asarray(cube), lower, upper, shaper=cp.asarray(shaper))
    original_asnumpy = cp.asnumpy
    transfers = []

    def transfer(array, *args, **kwargs):
        host = original_asnumpy(array, *args, **kwargs)
        transfers.append((array, host.copy()))
        return host

    path = tmp_path / "baked.cube"
    with monkeypatch.context() as patch:
        patch.setattr(cp, "asnumpy", transfer)
        assert px.io.write_lut(path, lut) is None
    assert len(transfers) == 1
    assert transfers[0][0] is not lut.data and transfers[0][0] is not lut.shaper
    _bits(transfers[0][1], expected)
    text = path.read_bytes()
    assert text.endswith(b"\n") and b"\r" not in text and b"SHAPER" not in text.upper()
    lines = text.decode("utf-8").splitlines()
    assert lines[:3] == ["LUT_3D_SIZE 3", "DOMAIN_MIN -2.0 1.0 -0.5", "DOMAIN_MAX 2.0 3.0 4.5"]
    # Independently parse Red-fastest rows and compare exact float32 payloads.
    rows = np.asarray([line.split() for line in lines[3:]], dtype=np.float32)
    _bits(rows, expected.transpose(2, 1, 0, 3).reshape(-1, 3))
    restored = px.io.read_lut(path)
    assert restored.shaper is None and restored.domain_min == lower and restored.domain_max == upper
    _bits(cp.asnumpy(restored.data), expected)
    _bits(cp.asnumpy(lut.data), cube)
    _bits(cp.asnumpy(lut.shaper), shaper)
    assert not np.array_equal(expected, cube)
    second = tmp_path / "repeat.cube"
    px.io.write_lut(second, lut)
    assert second.read_bytes() == text
    points = np.random.default_rng(11).uniform(0, 1, (128, 3))
    assert np.max(np.abs(host_apply(cube, shaper, points) - host_cube(expected, points))) > 0.01
    assert sorted(p.name for p in tmp_path.iterdir()) == ["baked.cube", "repeat.cube"]


@pytest.mark.parametrize("edge", (2, 3), ids=("diagonal", "four-vertex-simplex"))
def test_shaped_write_uses_float64_intermediate_before_float32_table(tmp_path: Path, edge: int) -> None:
    """v1-lut-shaper acceptance 9: independent f64 node bits reject f32 bake arithmetic.

    Diagonal: endpoints 6409297/-57657176 at float32(.1) give f64->f32
    2649.79052734375 (0x45259ca6); f32 delta and weighted sum both give
    2649.5. Delta FMA gives 2649.70458984375; the two weighted FMA orders
    give 2649.647216796875 and 2649.31396484375.

    Four vertices: 12084003, 13622878, 28477278, -63258292 at ranked
    fractions (.875, .625, .25) give exactly -219373.875 (0xc8563b78).
    Sequential f32 weighted sums range -219375..-219374, paired sums give
    -219375/-219374, and delta sums range -219373.375..-219372. Fused
    weighted sums range -219375..-219373 but never equal the f64 result;
    fused deltas also differ. Every term permutation is checked below.

    These are precision counterexamples, independent of production structure.
    All expected table bits come from the host float64 two-stage model, with
    no tolerance; the countermodels only verify the fixtures' detection power.
    """
    if edge == 2:
        vertices = np.asarray((6409297, 6409297, 6409297, -57657176), dtype=np.float32)
        shaper = np.asarray((0.1, 0.1), dtype=np.float32)
        fractions = np.repeat(shaper[0], 3)
        probe = (0, 0, 0, 0)
    else:
        vertices = np.asarray((12084003, 13622878, 28477278, -63258292), dtype=np.float32)
        shaper = np.asarray((0.125, 0.3125, 0.4375), dtype=np.float32)
        fractions = shaper[::-1] * np.float32(2)
        probe = (2, 1, 0, 0)
    cube = np.full((edge, edge, edge, 3), vertices[0], dtype=np.float32)
    cube[1, 0, 0], cube[1, 1, 0], cube[1, 1, 1] = vertices[1:]
    nodes = np.stack(np.meshgrid(*([np.linspace(0, 1, edge)] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    expected = host_apply(cube, shaper, nodes).astype(np.float32).reshape(cube.shape)
    for form, value in float32_simplex_variants(vertices, fractions).items():
        assert value.view(np.uint32) != expected[probe].view(np.uint32), (form, value, expected[probe])

    lut = px.core.Lut(cp.asarray(cube), shaper=cp.asarray(shaper))
    path = tmp_path / "cancellation.cube"
    px.io.write_lut(path, lut)
    rows = np.asarray([line.split() for line in path.read_text().splitlines()[3:]], dtype=np.float32)
    _bits(rows, expected.transpose(2, 1, 0, 3).reshape(-1, 3))
    _bits(cp.asnumpy(px.io.read_lut(path).data), expected)


@pytest.mark.parametrize("case", ("extension", "nonfinite-cube", "missing-parent"))
def test_shaped_write_preserves_pre_mutation_errors(tmp_path: Path, case: str) -> None:
    """v1-lut-shaper acceptance 9 and 19: shaped values keep the existing Cube exit and error boundaries."""
    cube = _cube(2)
    if case == "nonfinite-cube":
        cube[0, 0, 0, 0] = np.nan
    lut = px.core.Lut(cp.asarray(cube), shaper=cp.asarray((0.1, 0.9), dtype=cp.float32))
    path = tmp_path / ("output.look" if case == "extension" else "output.cube")
    if case == "missing-parent":
        path = tmp_path / "absent" / "output.cube"
    else:
        path.write_bytes(b"keep")
    with pytest.raises(RuntimeError if case == "missing-parent" else ValueError) as error:
        px.io.write_lut(path, lut)
    _actionable(error.value)
    if case == "missing-parent":
        assert isinstance(error.value.__cause__, FileNotFoundError)
        assert not path.parent.exists()
    else:
        assert path.read_bytes() == b"keep"


@pytest.mark.parametrize("entry", ("file", "bytes"))
def test_shaped_read_transfers_each_table_once_and_has_no_cache_or_filesystem_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entry: str,
) -> None:
    """v1-lut-shaper acceptance 4 and 19: two whole-array H2D transfers, independent calls and no hidden writes.

    Structural boundary observation spies on real CuPy transfer calls and NumPy
    bulk conversion; backing-allocation sharing is deliberately unconstrained.
    """
    spacing, codes = quantized_tables(17, "log")
    text = three_dl_text(spacing, codes).encode()
    path = tmp_path / "source.3dl"
    path.write_bytes(text)
    original_transfer, original_parse = cp.asarray, np.fromstring
    transfers, bulk_sizes = [], []

    def transfer(value, *args, **kwargs):
        result = original_transfer(value, *args, **kwargs)
        if isinstance(value, np.ndarray):
            transfers.append(value.copy())
        return result

    def parse(*args, **kwargs):
        result = original_parse(*args, **kwargs)
        bulk_sizes.append(result.size)
        return result

    reader, source = (px.io.read_lut, path) if entry == "file" else (px.io.decode_lut, text)
    with monkeypatch.context() as patch:
        patch.setattr(cp, "asarray", transfer)
        patch.setattr(np, "fromstring", parse)
        first = reader(source, preserve_shaper=True)
    assert len(transfers) == 2
    assert sorted(array.shape[0] for array in transfers) == [17, 17]
    assert any(array.shape == (17,) for array in transfers)
    assert any(array.size >= codes.size for array in transfers)
    assert codes.size in bulk_sizes
    first.data[...] = np.float32(-4)
    first.shaper[...] = np.float32(-3)
    second = reader(source, preserve_shaper=True)
    _bits(cp.asnumpy(second.data), (codes / 1023).astype(np.float32))
    _bits(cp.asnumpy(second.shaper), (spacing / 1023).astype(np.float32))
    assert not hasattr(reader, "cache_info")
    assert list(tmp_path.iterdir()) == [path] and path.read_bytes() == text


def test_ocio_is_declared_only_as_a_development_dependency() -> None:
    """v1-lut-shaper acceptance 11 and 18: metadata must not install an OCIO runtime dependency.

    This structural packaging contract is already Green: the current project
    declares opencolorio in dependency-groups.dev only. Adding either OCIO
    distribution name to project.dependencies must fail even without an import.
    """
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    def ocio_requirements(requirements):
        # Extract the distribution name before extras, version, URL or markers.
        return [
            requirement
            for requirement in requirements
            if re.split(r"[\s\[<>=!~@;]", requirement.strip(), maxsplit=1)[0].casefold()
            in {"opencolorio", "pyopencolorio"}
        ]

    assert not ocio_requirements(pyproject["project"]["dependencies"])
    assert ocio_requirements(pyproject["dependency-groups"]["dev"])


def test_ocio_is_absent_from_runtime_imports_and_package_assets(tmp_path: Path) -> None:
    """v1-lut-shaper acceptance 11, 16, 18 and 19: fresh runtime applies/writes with OCIO imports forbidden.

    Subprocess isolation checks the real import graph; timeout bounds the probe.
    """
    code = """
import importlib.abc, sys
class RejectOCIO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if "opencolorio" in fullname.casefold():
            raise AssertionError("runtime OCIO dependency: " + fullname)
sys.meta_path.insert(0, RejectOCIO())
import cupy as cp
import pixtreme as px
from pathlib import Path
lut = px.io.decode_lut(Path(sys.argv[1]).read_bytes(), preserve_shaper=True)
frame = px.io.from_array(cp.zeros((1, 1, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB")
px.color.apply_lut(frame, lut=lut)
px.io.write_lut(Path(sys.argv[2]), lut)
assert not any("opencolorio" in name.casefold() for name in sys.modules)
root = Path(px.__file__).parent
assert not any("ocio" in path.name.casefold() or path.suffix in {".3dl", ".clf", ".ctf", ".ocio"}
               for path in root.rglob("*") if path.is_file())
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(FIXTURES / "17-log" / "source.3dl"), str(tmp_path / "output.cube")],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_public_docstrings_describe_shaper_and_lossy_cube_exit() -> None:
    """v1-lut-shaper acceptance 16: public help describes the new state, strict path and irreversible exit."""
    for obj in (px.core.Lut, px.io.read_lut, px.io.decode_lut, px.io.write_lut, px.color.apply_lut):
        doc = inspect.getdoc(obj).lower()
        assert "shaper" in doc, obj
    for obj in (px.io.read_lut, px.io.decode_lut):
        assert "preserve_shaper" in inspect.getdoc(obj)
    for obj in (px.core.Lut, px.io.write_lut):
        doc = inspect.getdoc(obj).lower()
        assert "cube" in doc and any(word in doc for word in ("loss", "irrevers", "lost", "not recover")), obj
    apply = inspect.getdoc(px.color.apply_lut).lower()
    assert all(word in apply for word in ("linear", "tetrahedral", "shaper", "clamp"))
    assert "no shaper stage is implied" not in apply


@pytest.mark.parametrize(
    "text,size",
    ((_cube_1d_text(), 9), (_cube_3d_text(), 24), (_spi1d_text(), 9), (_spi3d_text(), 24)),
    ids=("cube1d", "cube3d", "spi1d", "spi3d"),
)
def test_unshaped_parsers_keep_bulk_numeric_conversion(monkeypatch: pytest.MonkeyPatch, text: str, size: int) -> None:
    """v1-lut-shaper acceptance 19: structural contract observes whole-table numeric parsing at NumPy's boundary."""
    original = np.fromstring
    sizes = []

    def observed(*args, **kwargs):
        output = original(*args, **kwargs)
        sizes.append(output.size)
        return output

    monkeypatch.setattr(np, "fromstring", observed)
    px.io.decode_lut(text.encode())
    assert size in sizes, f"expected one bulk numeric table of {size} entries, got {sizes}"


def test_performance_pair_has_fixed_registry_conditions_and_executable_inputs() -> None:
    """v1-lut-shaper acceptance 12: case setup is testable without running a timed benchmark."""
    import test_performance_spec as registry

    assert (registry._WIDTH, registry._HEIGHT, registry._CHANNELS, registry._LUT_SIZE) == (1920, 1080, 3, 65)
    assert registry._WARMUP_MINIMUM_SECONDS >= 0.5
    assert {case.case_id for case in registry._LUT_SHAPER_CASES} <= {
        case.case_id for case in registry._PERFORMANCE_CASES
    }
    for case in registry._LUT_SHAPER_CASES:
        assert case.operation is px.color.apply_lut and dict(case.kwargs) == {"interpolation": None}
        assert case.minimum_frames >= 1000 and case.minimum_seconds >= 3
    shaped, baked = registry._lut_shaper_pair()
    assert shaped.data.shape == baked.data.shape == (65, 65, 65, 3)
    assert shaped.shaper.shape == (65,) and baked.shaper is None
    shaper = cp.asnumpy(shaped.shaper)
    assert not np.array_equal(shaper, np.linspace(0, 1, 65, dtype=np.float32))
    assert shaped.data.dtype == baked.data.dtype == shaped.shaper.dtype == cp.float32
    assert shaped.data.device.id == baked.data.device.id == shaped.shaper.device.id
    frame = _frame(np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32))
    for lut in (shaped, baked):
        result = px.color.apply_lut(frame, lut=lut, interpolation=None)
        expected = np.interp([0.25, 0.5, 0.75], np.linspace(0, 1, 65), shaper)
        # Identity cube and exact input knots leave at most float32 arithmetic error.
        np.testing.assert_allclose(cp.asnumpy(result.data)[0, 0], expected, rtol=0, atol=2e-6)
