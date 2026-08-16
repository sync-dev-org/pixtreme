"""Specification tests for user-provided three-dimensional LUTs."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

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
    return px.io.from_array(cp.asarray(array), colorspace="ACEScg", gamma="linear", channels=channels)


def _identity_lut(
    *,
    size: int = 2,
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> px.core.Lut:
    axes = tuple(np.linspace(lower, upper, size, dtype=np.float32) for lower, upper in zip(domain_min, domain_max))
    data = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return px.core.Lut(data=cp.asarray(data), domain_min=domain_min, domain_max=domain_max)


def test_lut_constructs_directly_with_defaults_and_reference_data() -> None:
    """v1-lut acceptance 1-2: Lut validates and retains a programmatic GPU grid by reference."""
    data = cp.zeros((2, 2, 2, 3), dtype=cp.float32)

    lut = px.core.Lut(data=data)

    assert lut.data is data
    assert lut.data.shape == (2, 2, 2, 3)
    assert lut.data.dtype == cp.float32
    assert lut.domain_min == (0.0, 0.0, 0.0)
    assert lut.domain_max == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "case",
    (
        "not-cupy",
        "rank",
        "channels",
        "non-cube",
        "size",
        "dtype",
        "domain-length",
        "domain-order",
    ),
)
def test_lut_rejects_invalid_construction_actionably(case: str) -> None:
    """v1-lut acceptance 2: every Lut construction invariant fails with a three-part ValueError."""
    data: object = cp.zeros((2, 2, 2, 3), dtype=cp.float32)
    kwargs: dict[str, object] = {}
    if case == "not-cupy":
        data = np.zeros((2, 2, 2, 3), dtype=np.float32)
    elif case == "rank":
        data = cp.zeros((2, 2, 3), dtype=cp.float32)
    elif case == "channels":
        data = cp.zeros((2, 2, 2, 4), dtype=cp.float32)
    elif case == "non-cube":
        data = cp.zeros((2, 3, 2, 3), dtype=cp.float32)
    elif case == "size":
        data = cp.zeros((1, 1, 1, 3), dtype=cp.float32)
    elif case == "dtype":
        data = cp.zeros((2, 2, 2, 3), dtype=cp.float16)
    elif case == "domain-length":
        kwargs["domain_min"] = (0.0, 0.0)
    else:
        kwargs["domain_min"] = (0.0, 2.0, 0.0)
        kwargs["domain_max"] = (1.0, 2.0, 1.0)

    with pytest.raises(ValueError) as error:
        px.core.Lut(data=data, **kwargs)
    _assert_actionable(error)


def test_read_lut_parses_cube_domain_metadata_comments_and_red_fastest_order(tmp_path: Path) -> None:
    """v1-lut acceptance 3 and 7: .cube metadata and red-fastest rows map to the RGB-indexed grid."""
    path = tmp_path / "asymmetric.cube"
    path.write_text(
        """# asymmetric oracle
TITLE "ignored metadata"
LUT_3D_SIZE 2
DOMAIN_MIN -1 -2 -3
DOMAIN_MAX 1 2 3

0 0.25 0.5
1 1.25 1.5 # red advances fastest
10 10.25 10.5
11 11.25 11.5
100 100.25 100.5
101 101.25 101.5
110 110.25 110.5
111 111.25 111.5
""",
        encoding="utf-8",
    )

    lut = px.io.read_lut(path)

    indices = np.indices((2, 2, 2), dtype=np.float32)
    scalar = indices[0] + 10.0 * indices[1] + 100.0 * indices[2]
    expected = np.stack((scalar, scalar + 0.25, scalar + 0.5), axis=-1)
    np.testing.assert_array_equal(cp.asnumpy(lut.data), expected)
    assert lut.domain_min == (-1.0, -2.0, -3.0)
    assert lut.domain_max == (1.0, 2.0, 3.0)


def test_read_lut_parses_recognized_metadata_between_data_rows(tmp_path: Path) -> None:
    """v1-lut acceptance 3 and 7: recognized metadata is position-independent and excluded from data."""
    path = tmp_path / "interleaved-metadata.cube"
    path.write_text(
        """LUT_3D_SIZE 2
0 0 0
TITLE "metadata after data begins"
DOMAIN_MIN -1 -2 -3
1 0 0
0 1 0
1 1 0
DOMAIN_MAX 1 2 3
0 0 1
1 0 1
0 1 1
1 1 1
""",
        encoding="utf-8",
    )

    lut = px.io.read_lut(path)

    indices = np.indices((2, 2, 2), dtype=np.float32)
    expected = np.stack((indices[0], indices[1], indices[2]), axis=-1)
    np.testing.assert_array_equal(cp.asnumpy(lut.data), expected)
    assert lut.domain_min == (-1.0, -2.0, -3.0)
    assert lut.domain_max == (1.0, 2.0, 3.0)


def test_parse_cube_accepts_untranslated_crlf_directives_and_data_rows() -> None:
    """v1-lut acceptance 3: the parser accepts CRLF before the file boundary translates newlines."""
    import pixtreme._io.formats.lut as implementation

    body = (
        'TITLE "CRLF fixture"\r\n'
        "LUT_3D_SIZE 2\r\n"
        "DOMAIN_MIN -1 -2 -3\r\n"
        "DOMAIN_MAX 1 2 3\r\n"
        "# comment\r\n"
        "0 0 0\r\n"
        "1 0 0\r\n"
        "0 1 0\r\n"
        "1 1 0\r\n"
        "0 0 1\r\n"
        "1 0 1\r\n"
        "0 1 1\r\n"
        "1 1 1\r\n"
    )

    parsed = implementation._parse_cube(body, source="crlf fixture")

    indices = np.indices((2, 2, 2), dtype=np.float32)
    expected = np.stack((indices[0], indices[1], indices[2]), axis=-1)
    np.testing.assert_array_equal(parsed.data, expected)
    assert parsed.domain_min == (-1.0, -2.0, -3.0)
    assert parsed.domain_max == (1.0, 2.0, 3.0)


def test_read_lut_accepts_crlf_files_at_the_public_boundary(tmp_path: Path) -> None:
    """v1-lut acceptance 3: CRLF .cube files parse through the public file boundary."""
    path = tmp_path / "crlf.cube"
    body = (
        'TITLE "CRLF fixture"\r\n'
        "LUT_3D_SIZE 2\r\n"
        "DOMAIN_MIN -1 -2 -3\r\n"
        "DOMAIN_MAX 1 2 3\r\n"
        "# comment\r\n"
        "0 0 0\r\n"
        "1 0 0\r\n"
        "0 1 0\r\n"
        "1 1 0\r\n"
        "0 0 1\r\n"
        "1 0 1\r\n"
        "0 1 1\r\n"
        "1 1 1\r\n"
    )
    path.write_bytes(body.encode("utf-8"))

    lut = px.io.read_lut(path)

    assert lut.domain_min == (-1.0, -2.0, -3.0)
    assert lut.domain_max == (1.0, 2.0, 3.0)


def test_read_lut_accepts_str_paths_and_default_domain(tmp_path: Path) -> None:
    """v1-lut acceptance 2-3: str paths and omitted .cube domains use the documented unit cube."""
    path = tmp_path / "identity.cube"
    path.write_text(
        "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
        encoding="utf-8",
    )

    lut = px.io.read_lut(str(path))

    assert lut.domain_min == (0.0, 0.0, 0.0)
    assert lut.domain_max == (1.0, 1.0, 1.0)


def test_read_lut_translates_invalid_path_types_actionably() -> None:
    """REQ-API-012: the LUT path boundary translates pathlib type failures and retains their cause."""
    with pytest.raises(ValueError) as error:
        px.io.read_lut(None)  # type: ignore[arg-type]

    _assert_actionable(error)
    assert "path" in str(error.value)
    assert "str or os.PathLike" in str(error.value)
    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "body",
    (
        "0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
        "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n",
        "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n2 2 2\n",
        "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\nnope 1 1\n",
        "LUT_3D_SIZE 2\n0 0 0 1\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
        "LUT_3D_SIZE 2\n0 0\n1 0 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
    ),
)
def test_read_lut_rejects_broken_cube_actionably(tmp_path: Path, body: str) -> None:
    """v1-lut acceptance 5: missing size, row-count, numeric, and row-shape corruption fail actionably."""
    path = tmp_path / "broken.cube"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError) as error:
        px.io.read_lut(path)
    _assert_actionable(error)


def test_read_lut_rejects_missing_files_actionably(tmp_path: Path) -> None:
    """v1-lut acceptance 5: an absent LUT file is a three-part FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as error:
        px.io.read_lut(tmp_path / "absent.cube")
    _assert_actionable(error)


def test_read_lut_parser_is_bulk_vectorized_uncached_and_pure(tmp_path: Path) -> None:
    """v1-lut acceptance 6: parsing has no per-line loop or cache and performs no filesystem writes."""
    import pixtreme._io.formats.lut as implementation

    tree = ast.parse(inspect.getsource(implementation._parse_cube))
    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
    calls = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == "fromstring" for node in calls)
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "split" for node in calls)
    assert not hasattr(implementation._parse_cube, "cache_info")

    path = tmp_path / "identity.cube"
    body = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n"
    path.write_text(body, encoding="utf-8")
    before = tuple(sorted(item.name for item in tmp_path.iterdir()))

    first = px.io.read_lut(path)
    second = px.io.read_lut(path)

    assert first.data is not second.data
    np.testing.assert_array_equal(cp.asnumpy(first.data), cp.asnumpy(second.data))
    assert path.read_text(encoding="utf-8") == body
    assert tuple(sorted(item.name for item in tmp_path.iterdir())) == before


def test_lut_transform_signature_is_keyword_only_after_frame() -> None:
    """v1-lut acceptance 8 and 10; v1-lut-extensions acceptance 4-5: the transform grammar stays exact."""
    signature = inspect.signature(px.color.apply_lut)

    assert tuple(signature.parameters) == ("frame", "lut", "interpolation")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["lut"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["lut"].default is inspect.Parameter.empty
    assert signature.parameters["interpolation"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["interpolation"].default is None


@pytest.mark.parametrize(
    ("interpolation", "expected"),
    (("trilinear", 11.34375 / 64.0), ("tetrahedral", 0.28515625)),
)
def test_lut_transform_matches_hand_calculated_two_cube_oracles(interpolation: str, expected: float) -> None:
    """v1-lut acceptance 10-11: both interpolation tokens match independent 2x2x2 hand calculations."""
    scalar_vertices = np.asarray(
        (
            ((0.0, 4.0), (2.0, 32.0)),
            ((1.0, 16.0), (8.0, 64.0)),
        ),
        dtype=np.float32,
    ) / np.float32(64.0)
    lut = px.core.Lut(data=cp.asarray(np.repeat(scalar_vertices[..., None], 3, axis=-1)))
    frame = _frame(np.asarray((0.75, 0.5, 0.25), dtype=np.float32))

    result = px.color.apply_lut(frame, lut=lut, interpolation=interpolation)

    np.testing.assert_allclose(cp.asnumpy(result.data)[0, 0], (expected, expected, expected), rtol=0.0, atol=2e-7)


def test_tetrahedral_lut_preserves_arbitrary_channel_stride_fallback() -> None:
    """v1-lut acceptance 10-11: non-packed LUT storage uses its declared strides without changing the oracle."""
    scalar_vertices = np.asarray(
        (
            ((0.0, 4.0), (2.0, 32.0)),
            ((1.0, 16.0), (8.0, 64.0)),
        ),
        dtype=np.float32,
    ) / np.float32(64.0)
    values = np.repeat(scalar_vertices[..., None], 3, axis=-1)
    backing = cp.zeros((2, 2, 2, 6), dtype=cp.float32)
    backing[..., ::2] = cp.asarray(values)
    strided = backing[..., ::2]
    assert not strided.flags.c_contiguous
    lut = px.core.Lut(data=strided)
    frame = _frame(np.asarray((0.75, 0.5, 0.25), dtype=np.float32))

    result = px.color.apply_lut(frame, lut=lut, interpolation="tetrahedral")

    np.testing.assert_allclose(cp.asnumpy(result.data)[0, 0], (0.28515625,) * 3, rtol=0.0, atol=2e-7)


@pytest.mark.parametrize("interpolation", ("trilinear", "tetrahedral"))
def test_lut_transform_selects_the_correct_cell_in_a_three_cube(interpolation: str) -> None:
    """v1-lut acceptance 11: both interpolation kernels match a hand-computed 3x3x3 cell fixture."""
    indices = np.indices((3, 3, 3), dtype=np.float32)
    scalar = (indices[0] ** 2 + 2.0 * indices[1] ** 2 + 4.0 * indices[2] ** 2) / 28.0
    lut = px.core.Lut(data=cp.asarray(np.repeat(scalar[..., None], 3, axis=-1)))
    frame = _frame(np.asarray((0.625, 0.375, 0.875), dtype=np.float32))

    result = px.color.apply_lut(frame, lut=lut, interpolation=interpolation)

    expected = 16.25 / 28.0
    np.testing.assert_allclose(cp.asnumpy(result.data)[0, 0], (expected, expected, expected), rtol=0.0, atol=2e-7)


@pytest.mark.parametrize("interpolation", ("trilinear", "tetrahedral"))
def test_lut_transform_preserves_wide_finite_domain_affine_mapping(interpolation: str) -> None:
    """v1-lut acceptance 13; GitHub issue #10: wide finite domains retain their affine midpoint."""
    axis = np.asarray((0.0, 1.0), dtype=np.float32)
    table = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    lut = px.core.Lut(
        cp.asarray(table),
        domain_min=(-1e300, -1e300, -1e300),
        domain_max=(1e300, 1e300, 1e300),
    )

    with np.errstate(over="raise"):
        result = px.color.apply_lut(_frame(np.zeros(3, dtype=np.float32)), lut=lut, interpolation=interpolation)

    np.testing.assert_allclose(cp.asnumpy(result.data)[0, 0], (0.5, 0.5, 0.5), rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("interpolation", ("trilinear", "tetrahedral"))
def test_identity_lut_uses_declared_domain_and_clamps_lookup_without_output_clip(interpolation: str) -> None:
    """v1-lut acceptance 12-13: identity, domain affine, input clamp, and unclipped output share one oracle."""
    lut = _identity_lut(domain_min=(-1.0, 0.0, 2.0), domain_max=(1.0, 2.0, 4.0))
    frame = _frame(np.asarray(((-2.0, 1.0, 5.0), (0.0, 1.5, 3.0)), dtype=np.float32).reshape(1, 2, 3))

    result = px.color.apply_lut(frame, lut=lut, interpolation=interpolation)

    expected = np.asarray(((-1.0, 1.0, 4.0), (0.0, 1.5, 3.0)), dtype=np.float32).reshape(1, 2, 3)
    np.testing.assert_allclose(cp.asnumpy(result.data), expected, rtol=0.0, atol=2e-7)


def test_lut_transform_preserves_metadata_and_non_rgb_channels() -> None:
    """v1-lut acceptance 8-9: RGB labels drive lookup while metadata and every other label pass through."""
    data = cp.empty((2, 2, 2, 3), dtype=cp.float32)
    data[..., 0] = np.float32(2.0)
    data[..., 1] = np.float32(-3.0)
    data[..., 2] = np.float32(4.0)
    lut = px.core.Lut(data=data)
    source = _frame(
        np.asarray((0.25, 0.5, 0.75, 9.0, 0.125), dtype=np.float32),
        channels=("B", "A", "R", "Z", "G"),
    )

    result = px.color.apply_lut(source, lut=lut)

    assert result.colorspace == source.colorspace
    assert result.gamma == source.gamma
    assert result.channels == source.channels
    np.testing.assert_array_equal(cp.asnumpy(result.data)[0, 0], (4.0, 0.5, 2.0, 9.0, -3.0))
    docstring = inspect.getdoc(px.color.apply_lut) or ""
    for required in ("metadata", "meaning", "clip", "shaper"):
        assert required in docstring


def test_lut_transform_does_not_mutate_frame_or_lut() -> None:
    """v1-lut acceptance 14: frame storage and referenced LUT storage remain byte-for-byte unchanged."""
    lut = _identity_lut(size=3)
    source = _frame(np.asarray((0.2, 0.4, 0.6), dtype=np.float32))
    source_before = source.data.copy()
    lut_before = lut.data.copy()

    result = px.color.apply_lut(source, lut=lut)

    assert result.data is not source.data
    cp.testing.assert_array_equal(source.data, source_before)
    cp.testing.assert_array_equal(lut.data, lut_before)


@pytest.mark.parametrize("channels", (("Y", "Cb", "Cr"), ("R", "G"), ("R", "G", "A")))
def test_lut_transform_rejects_frames_without_all_rgb_labels(channels: tuple[str, ...]) -> None:
    """v1-lut acceptance 8: every missing RGB label fails before lookup."""
    source = _frame(np.zeros(len(channels), dtype=np.float32), channels=channels)

    with pytest.raises(ValueError) as error:
        px.color.apply_lut(source, lut=_identity_lut())
    _assert_actionable(error)
    assert "R, G, and B" in str(error.value)


@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        (np.float16, ("cast_dtype",)),
        (np.uint8, ("recode_dtype", "dequantize")),
        (np.uint16, ("recode_dtype", "dequantize")),
    ),
)
def test_lut_transform_rejects_non_float32_with_dtype_specific_guidance(
    dtype: type[np.generic], routes: tuple[str, ...]
) -> None:
    """v1-lut acceptance 8: non-fp32 errors retain the shared conversion guidance."""
    source = _frame(np.zeros(3, dtype=dtype))

    with pytest.raises(ValueError) as error:
        px.color.apply_lut(source, lut=_identity_lut())
    _assert_actionable(error)
    message = str(error.value)
    assert "float32" in message
    assert tuple(message.index(route) for route in routes) == tuple(sorted(message.index(route) for route in routes))


@pytest.mark.parametrize("interpolation", ("linear", "Tetrahedral"))
def test_lut_transform_rejects_unknown_interpolation_tokens(interpolation: object) -> None:
    """v1-lut acceptance 10; v1-lut-extensions acceptance 5: 3D keeps its case-sensitive two-token subset."""
    source = _frame(np.zeros(3, dtype=np.float32))

    with pytest.raises(ValueError) as error:
        px.color.apply_lut(source, lut=_identity_lut(), interpolation=interpolation)  # type: ignore[arg-type]
    _assert_actionable(error)
    for token in ("trilinear", "tetrahedral"):
        assert token in str(error.value)


def test_lut_transform_rejects_non_frame_and_non_lut_inputs() -> None:
    """v1-lut acceptance 8: both public currencies fail actionably when their types are wrong."""
    with pytest.raises(ValueError) as frame_error:
        px.color.apply_lut(object(), lut=_identity_lut())  # type: ignore[arg-type]
    _assert_actionable(frame_error)

    with pytest.raises(ValueError) as lut_error:
        px.color.apply_lut(_frame(np.zeros(3, dtype=np.float32)), lut=object())  # type: ignore[arg-type]
    _assert_actionable(lut_error)


def test_lut_documentation_contracts_are_present() -> None:
    """v1-lut acceptance 15-16; v1-lut-extensions acceptance 26: LUT boundary canon stays current."""
    vocabulary_path = ROOT / "docs_site" / "tokens.md"
    requirements_path = ROOT / "docs" / "requirements.md"
    if not vocabulary_path.is_file() or not requirements_path.is_file():
        pytest.skip("repo-only documentation contract: canonical docs are absent from this distribution")

    vocabulary = vocabulary_path.read_text(encoding="utf-8")
    interpolation = vocabulary.split("## interpolation\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for required in ("apply_lut", "trilinear", "tetrahedral", "default", "domain", "clamp", "clip"):
        assert required in interpolation

    requirements = requirements_path.read_text(encoding="utf-8")
    boundary_table = requirements.split("**REQ-API-010", maxsplit=1)[1].split("\n\n統一則", maxsplit=1)[0]
    lut_boundary_rows = tuple(line for line in boundary_table.splitlines() if "LUT" in line)
    assert any(
        "LUT file" in line and "`px.io.read_lut`" in line and "`px.io.write_lut`" in line for line in lut_boundary_rows
    )
    assert any("LUT bytes" in line and "`px.io.decode_lut`" in line for line in lut_boundary_rows)
