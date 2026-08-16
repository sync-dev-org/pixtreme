"""Specification tests for LUT file, byte, and serialization boundaries."""

from __future__ import annotations

import inspect
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]


def _assert_actionable(error: BaseException) -> None:
    message = str(error)
    assert message.index("why=") < message.index("what=") < message.index("how=")


def _write_lut_fixture(tmp_path: Path, suffix: str, text: str) -> Path:
    path = tmp_path / f"fixture{suffix}"
    path.write_text(text, encoding="utf-8", newline="")
    return path


def _cube_3d_text() -> str:
    return (
        "LUT_3D_SIZE 2\nDOMAIN_MIN -1 -2 -3\nDOMAIN_MAX 1 2 3\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n"
    )


def _cube_1d_text() -> str:
    return "LUT_1D_SIZE 3\nDOMAIN_MIN -1 -2 -3\nDOMAIN_MAX 1 2 3\n-0.25 1.25 0.75\n2 -1 0.5\n0.125 3 -2\n"


def _headerless_3dl_text(
    spacing: tuple[int, ...] = (0, 255),
    cube: np.ndarray | None = None,
) -> str:
    edge = len(spacing)
    if cube is None:
        indices = np.indices((edge, edge, edge), dtype=np.int64)
        cube = np.stack(indices, axis=-1) * (255 // max(edge - 1, 1))
    rows = [" ".join(str(int(value)) for value in spacing)]
    rows.extend(
        " ".join(str(int(value)) for value in cube[red, green, blue])
        for red in range(edge)
        for green in range(edge)
        for blue in range(edge)
    )
    return "\n".join(rows) + "\n"


def _spi1d_text(*, components: int = 3) -> str:
    rows = {
        1: ("-0.5", "0.25", "1.5"),
        2: ("-0.5 1", "0.25 -1", "1.5 2"),
        3: ("-0.5 1 2", "0.25 -1 3", "1.5 2 -2"),
    }[components]
    return f"Version 1\nFrom -2 4\nLength 3\nComponents {components}\n{{\n" + "\n".join(rows) + "\n}\n"


def _spi3d_text(*, rows: list[str] | None = None, size_line: str = "2 2 2") -> str:
    if rows is None:
        rows = [
            f"{red} {green} {blue} {red + 10 * green + 100 * blue} {red + 0.25} {green - blue}"
            for red in range(2)
            for green in range(2)
            for blue in range(2)
        ]
    return "SPILUT 1.0\n3 3\n" + size_line + "\n" + "\n".join(rows) + "\n"


def _infer_3dl_scale(maximum: int) -> float:
    if maximum <= 511:
        return 255.0
    if maximum <= 2047:
        return 1023.0
    if maximum <= 8191:
        return 4095.0
    return 65535.0


def _tetrahedral_host(cube: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    size = cube.shape[0]
    scaled = np.clip(coordinate, 0.0, 1.0) * (size - 1)
    lower = np.minimum(np.floor(scaled).astype(np.int64), size - 2)
    fraction = scaled - lower
    red, green, blue = (int(value) for value in lower)
    fr, fg, fb = (float(value) for value in fraction)
    v000 = cube[red, green, blue]
    v100 = cube[red + 1, green, blue]
    v010 = cube[red, green + 1, blue]
    v001 = cube[red, green, blue + 1]
    v110 = cube[red + 1, green + 1, blue]
    v101 = cube[red + 1, green, blue + 1]
    v011 = cube[red, green + 1, blue + 1]
    v111 = cube[red + 1, green + 1, blue + 1]
    if fr >= fg >= fb:
        return v000 + fr * (v100 - v000) + fg * (v110 - v100) + fb * (v111 - v110)
    if fr >= fb >= fg:
        return v000 + fr * (v100 - v000) + fb * (v101 - v100) + fg * (v111 - v101)
    if fb >= fr >= fg:
        return v000 + fb * (v001 - v000) + fr * (v101 - v001) + fg * (v111 - v101)
    if fg >= fr >= fb:
        return v000 + fg * (v010 - v000) + fr * (v110 - v010) + fb * (v111 - v110)
    if fg >= fb >= fr:
        return v000 + fg * (v010 - v000) + fb * (v011 - v010) + fr * (v111 - v011)
    return v000 + fb * (v001 - v000) + fg * (v011 - v001) + fr * (v111 - v011)


def test_cube_1d_preserves_independent_curves_domains_and_unbounded_outputs(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 8: Cube 1D maps independent RGB curves and per-channel domains."""
    path = _write_lut_fixture(tmp_path, ".cube", _cube_1d_text())

    lut = px.io.read_lut(path)

    assert isinstance(lut, px.core.Lut1D)
    np.testing.assert_array_equal(
        cp.asnumpy(lut.data),
        np.asarray(((-0.25, 1.25, 0.75), (2.0, -1.0, 0.5), (0.125, 3.0, -2.0)), dtype=np.float32),
    )
    assert lut.domain_min == (-1.0, -2.0, -3.0)
    assert lut.domain_max == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "text",
    (
        "LUT_1D_SIZE 2\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n",
        "LUT_1D_SIZE 2\nLUT_1D_SIZE 2\n0 0 0\n1 1 1\n",
        "LUT_1D_SIZE 1\n0 0 0\n",
        "LUT_1D_SIZE 65537\n0 0 0\n",
        "LUT_1D_SIZE 2\n0 0\n1 1 1\n",
        "LUT_1D_SIZE 2\n0 0 0\n1 nope 1\n",
        "LUT_1D_SIZE 2\n0 0 0\n",
        "LUT_1D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MIN 0 0 0\n0 0 0\n1 1 1\n",
        "LUT_1D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 0 1\n0 0 0\n1 1 1\n",
    ),
)
def test_cube_parser_rejects_ambiguous_or_malformed_1d_inputs(tmp_path: Path, text: str) -> None:
    """v1-lut-extensions acceptance 9 and 24: Cube declarations, rows, numbers, and domains fail actionably."""
    path = _write_lut_fixture(tmp_path, ".cube", text)

    with pytest.raises(ValueError) as error:
        px.io.read_lut(path)

    _assert_actionable(error.value)


def test_lustre_3dl_header_controls_edge_scale_and_blue_fastest_mapping(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 10 and 14: Lustre headers validate and Blue-fastest rows map by RGB index."""
    edge = 17
    spacing = np.rint(np.linspace(0.0, 255.0, edge)).astype(np.int64)
    indices = np.indices((edge, edge, edge), dtype=np.int64)
    cube = np.stack(
        (
            np.rint(indices[0] * 255.0 / (edge - 1)),
            np.rint(indices[1] * 255.0 / (edge - 1)),
            np.rint(indices[2] * 255.0 / (edge - 1)),
        ),
        axis=-1,
    )
    body = "3DMESH\nMesh 4 8\n" + _headerless_3dl_text(tuple(int(value) for value in spacing), cube)
    path = _write_lut_fixture(tmp_path, ".3dl", body)

    lut = px.io.read_lut(path)

    assert isinstance(lut, px.core.Lut)
    assert lut.data.shape == (17, 17, 17, 3)
    np.testing.assert_array_equal(cp.asnumpy(lut.data)[8, 3, 14], cube[8, 3, 14].astype(np.float32) / 255.0)


@pytest.mark.parametrize(
    ("maximum", "full_scale"),
    ((511, 255), (512, 1023), (2047, 1023), (2048, 4095), (8191, 4095), (8192, 65535), (65535, 65535)),
)
def test_headerless_3dl_uses_ocio_integer_scale_boundaries(tmp_path: Path, maximum: int, full_scale: int) -> None:
    """v1-lut-extensions acceptance 11: headerless 3DL scale inference uses the pinned OCIO boundaries."""
    cube = np.zeros((2, 2, 2, 3), dtype=np.int64)
    cube[1, 1, 1, 0] = maximum
    path = _write_lut_fixture(tmp_path, ".3dl", _headerless_3dl_text(cube=cube))

    lut = px.io.read_lut(path)

    expected = np.float32(maximum / full_scale)
    assert cp.asnumpy(lut.data)[1, 1, 1, 0] == expected


def test_headerless_3dl_near_identity_spacing_preserves_normalized_cube_bits(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 12: a half-source-code near-identity shaper is discarded without resampling."""
    edge = 3
    indices = np.indices((edge, edge, edge), dtype=np.int64)
    cube = np.stack((11 * indices[0] + indices[2], 7 * indices[1], 13 * indices[2]), axis=-1)
    cube[2, 2, 2] = (255, 128, 64)
    path = _write_lut_fixture(tmp_path, ".3dl", _headerless_3dl_text((0, 127, 255), cube))

    lut = px.io.read_lut(path)

    expected = (cube.astype(np.float64) / 255.0).astype(np.float32)
    np.testing.assert_array_equal(cp.asnumpy(lut.data), expected)


def test_headerless_3dl_bakes_nonidentity_shaper_with_independent_tetrahedral_oracle(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 13 and 28: independent float64 tetrahedral oracle fixes 3DL shaper baking."""
    edge = 3
    indices = np.indices((edge, edge, edge), dtype=np.int64)
    cube_codes = np.stack(
        (
            20 * indices[0] ** 2 + 5 * indices[1] + 3 * indices[2],
            10 * indices[0] + 15 * indices[1] ** 2 + 2 * indices[2],
            4 * indices[0] + 6 * indices[1] + 12 * indices[2] ** 2,
        ),
        axis=-1,
    )
    spacing_codes = np.asarray((0, 64, 511), dtype=np.float64)
    path = _write_lut_fixture(
        tmp_path,
        ".3dl",
        _headerless_3dl_text(tuple(int(value) for value in spacing_codes), cube_codes),
    )

    lut = px.io.read_lut(path)

    cube = cube_codes.astype(np.float64) / _infer_3dl_scale(int(cube_codes.max()))
    spacing = spacing_codes / _infer_3dl_scale(int(spacing_codes.max()))
    source_axis = np.linspace(0.0, 1.0, edge)
    expected = np.empty((edge, edge, edge, 3), dtype=np.float64)
    for red in range(edge):
        for green in range(edge):
            for blue in range(edge):
                uniform = np.asarray((red, green, blue), dtype=np.float64) / (edge - 1)
                shaped = np.asarray([np.interp(value, source_axis, spacing) for value in uniform])
                expected[red, green, blue] = _tetrahedral_host(cube, shaped)
    np.testing.assert_allclose(cp.asnumpy(lut.data), expected.astype(np.float32), rtol=0.0, atol=1e-7)


def test_headerless_3dl_blue_fastest_rows_transpose_into_rgb_indexed_data(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 14: Blue-fastest 3DL rows land at Lut.data[R, G, B]."""
    cube = np.empty((2, 2, 2, 3), dtype=np.int64)
    for red in range(2):
        for green in range(2):
            for blue in range(2):
                scalar = 100 * red + 10 * green + blue
                cube[red, green, blue] = (scalar, scalar + 1, scalar + 2)
    path = _write_lut_fixture(tmp_path, ".3dl", _headerless_3dl_text(cube=cube))

    lut = px.io.read_lut(path)

    scale = _infer_3dl_scale(int(cube.max()))
    np.testing.assert_array_equal(cp.asnumpy(lut.data), (cube.astype(np.float64) / scale).astype(np.float32))


@pytest.mark.parametrize(
    "text",
    (
        "0 255\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n-1 0 0\n",
        "0 255\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n1.5 0 0\n",
        "0 255\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n65536 0 0\n",
        "0 255\n0 255\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n",
        "0 255\n0 0 0\n0 0 0\n",
        "3DMESH\nMesh 4 8\n0 255\n0 0 0\n",
        "3dmesh\nMesh 4 8\n0 255\n0 0 0\n",
    ),
)
def test_3dl_rejects_invalid_codes_spacing_rows_and_headers(tmp_path: Path, text: str) -> None:
    """v1-lut-extensions acceptance 10 and 14: malformed headers, codes, spacing, and counts fail actionably."""
    path = _write_lut_fixture(tmp_path, ".3dl", text)

    with pytest.raises(ValueError) as error:
        px.io.read_lut(path)

    _assert_actionable(error.value)


@pytest.mark.parametrize(
    ("components", "expected"),
    (
        (1, ((-0.5, -0.5, -0.5), (0.25, 0.25, 0.25), (1.5, 1.5, 1.5))),
        (2, ((-0.5, 1.0, 0.0), (0.25, -1.0, 0.0), (1.5, 2.0, 0.0))),
        (3, ((-0.5, 1.0, 2.0), (0.25, -1.0, 3.0), (1.5, 2.0, -2.0))),
    ),
)
def test_spi1d_maps_each_component_variant_and_shared_domain(
    tmp_path: Path, components: int, expected: tuple[tuple[float, float, float], ...]
) -> None:
    """v1-lut-extensions acceptance 15-16: SPI1D grammar maps 1/2/3 components and From into Lut1D."""
    path = _write_lut_fixture(tmp_path, ".spi1d", _spi1d_text(components=components))

    lut = px.io.read_lut(path)

    assert isinstance(lut, px.core.Lut1D)
    np.testing.assert_array_equal(cp.asnumpy(lut.data), np.asarray(expected, dtype=np.float32))
    assert lut.domain_min == (-2.0, -2.0, -2.0)
    assert lut.domain_max == (4.0, 4.0, 4.0)


@pytest.mark.parametrize(
    "text",
    (
        "version 1\nLength 2\nComponents 1\n{\n0\n1\n}\n",
        "Version 2\nLength 2\nComponents 1\n{\n0\n1\n}\n",
        "Version 1\nLength 1\nComponents 1\n{\n0\n}\n",
        "Version 1\nLength 2\nComponents 4\n{\n0 0 0 0\n1 1 1 1\n}\n",
        "Version 1\nLength 2\nComponents 2\n{\n0\n1 1\n}\n",
        "Version 1\nLength 3\nComponents 1\n{\n0\n1\n}\n",
        "Version 1\nFrom 1 1\nLength 2\nComponents 1\n{\n0\n1\n}\n",
        "Version 1\nLength 2\nComponents 1\n0\n1\n",
        "Version 1\nLength 2\nComponents 3\n{\n0 nope 0\n1 1 1\n}\n",
    ),
)
def test_spi1d_rejects_invalid_grammar_and_table_shape(tmp_path: Path, text: str) -> None:
    """v1-lut-extensions acceptance 15-16 and 24: malformed SPI1D payloads fail actionably."""
    path = _write_lut_fixture(tmp_path, ".spi1d", text)

    with pytest.raises(ValueError) as error:
        px.io.read_lut(path)

    _assert_actionable(error.value)


def test_spi3d_uses_explicit_indices_independent_of_row_order(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 17: SPI3D rows populate explicit indices independent of physical order."""
    rows = _spi3d_text().splitlines()[3:]
    path = _write_lut_fixture(tmp_path, ".spi3d", _spi3d_text(rows=list(reversed(rows))))

    lut = px.io.read_lut(path)

    expected = np.empty((2, 2, 2, 3), dtype=np.float32)
    for red in range(2):
        for green in range(2):
            for blue in range(2):
                expected[red, green, blue] = (red + 10 * green + 100 * blue, red + 0.25, green - blue)
    np.testing.assert_array_equal(cp.asnumpy(lut.data), expected)
    assert lut.domain_min == (0.0, 0.0, 0.0)
    assert lut.domain_max == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("case", ("header", "sizes", "duplicate", "missing", "range", "width", "numeric"))
def test_spi3d_rejects_invalid_headers_and_index_sets(tmp_path: Path, case: str) -> None:
    """v1-lut-extensions acceptance 17 and 24: SPI3D validates headers, sizes, and a complete unique index set."""
    rows = _spi3d_text().splitlines()[3:]
    if case == "header":
        text = _spi3d_text().replace("3 3", "3 4", 1)
    elif case == "sizes":
        text = _spi3d_text(size_line="2 3 2")
    elif case == "duplicate":
        text = _spi3d_text(rows=[rows[0], *rows[:-1]])
    elif case == "missing":
        text = _spi3d_text(rows=rows[:-1])
    elif case == "range":
        text = _spi3d_text(rows=["2 0 0 0 0 0", *rows[1:]])
    elif case == "width":
        text = _spi3d_text(rows=["0 0 0 0 0", *rows[1:]])
    else:
        text = _spi3d_text(rows=["0 0 0 nope 0 0", *rows[1:]])
    path = _write_lut_fixture(tmp_path, ".spi3d", text)

    with pytest.raises(ValueError) as error:
        px.io.read_lut(path)

    _assert_actionable(error.value)


@pytest.mark.parametrize(
    ("suffix", "text", "expected_type"),
    (
        (".CUBE", _cube_1d_text(), px.core.Lut1D),
        (".3DL", _headerless_3dl_text(), px.core.Lut),
        (".SPI1D", _spi1d_text(), px.core.Lut1D),
        (".SPI3D", _spi3d_text(), px.core.Lut),
    ),
)
def test_read_lut_routes_case_insensitive_closed_extensions(
    tmp_path: Path, suffix: str, text: str, expected_type: type[object]
) -> None:
    """v1-lut-extensions acceptance 18: read_lut routes only the four case-insensitive extensions."""
    path = _write_lut_fixture(tmp_path, suffix, text)

    assert isinstance(px.io.read_lut(path), expected_type)


def test_read_lut_rejects_unsupported_extension_before_filesystem_access(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 18 and 24: unsupported extensions fail before existence checks."""
    unsupported = tmp_path / "does-not-exist.look"

    with pytest.raises(ValueError) as error:
        px.io.read_lut(unsupported)

    _assert_actionable(error.value)
    assert not isinstance(error.value, FileNotFoundError)


def test_read_lut_distinguishes_missing_utf8_and_malformed_files(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 18 and 24: file failures preserve their specified public exception boundaries."""
    with pytest.raises(FileNotFoundError) as missing:
        px.io.read_lut(tmp_path / "missing.cube")
    _assert_actionable(missing.value)

    invalid_utf8 = tmp_path / "invalid.cube"
    invalid_utf8.write_bytes(b"LUT_1D_SIZE 2\n\xff")
    with pytest.raises(ValueError) as decoded:
        px.io.read_lut(invalid_utf8)
    _assert_actionable(decoded.value)
    assert isinstance(decoded.value.__cause__, UnicodeDecodeError)

    malformed = _write_lut_fixture(tmp_path, ".cube", "LUT_1D_SIZE 2\n0 0 0\n")
    with pytest.raises(ValueError) as parsed:
        px.io.read_lut(malformed)
    _assert_actionable(parsed.value)


@pytest.mark.parametrize(
    ("suffix", "text"),
    (
        (".spi3d", _spi3d_text()),
        (".spi1d", _spi1d_text()),
        (".cube", _cube_1d_text()),
        (
            ".3dl",
            "3DMESH\nMesh 4 8\n"
            + _headerless_3dl_text(
                tuple(np.rint(np.linspace(0, 255, 17)).astype(int)), np.zeros((17, 17, 17, 3), dtype=np.int64)
            ),
        ),
        (".3dl", _headerless_3dl_text((0, 85, 170, 255))),
    ),
)
def test_decode_lut_sniffs_each_format_and_matches_read_lut(tmp_path: Path, suffix: str, text: str) -> None:
    """v1-lut-extensions acceptance 19-20: byte sniffing selects one format and matches extension-directed reading."""
    path = _write_lut_fixture(tmp_path, suffix, text)

    decoded = px.io.decode_lut(text.encode("utf-8"))
    read = px.io.read_lut(path)

    assert type(decoded) is type(read)
    assert decoded.domain_min == read.domain_min
    assert decoded.domain_max == read.domain_max
    np.testing.assert_array_equal(cp.asnumpy(decoded.data), cp.asnumpy(read.data))


@pytest.mark.parametrize(
    "payload",
    (
        b"SPILUT 1.0\nLUT_1D_SIZE 2\n0 0 0\n1 1 1\n",
        b"not a supported LUT",
        b"\xff\xfe",
        "LUT_1D_SIZE 2\n0 0 0\n1 1 1\n",
    ),
)
def test_decode_lut_rejects_ambiguous_unknown_non_utf8_and_non_bytes(payload: object) -> None:
    """v1-lut-extensions acceptance 19 and 24: decode_lut rejects invalid byte-boundary inputs actionably."""
    with pytest.raises(ValueError) as error:
        px.io.decode_lut(payload)  # type: ignore[arg-type]

    _assert_actionable(error.value)


def test_decode_lut_does_not_fall_through_after_marker_parse_failure() -> None:
    """v1-lut-extensions acceptance 20: a selected Cube parser failure cannot fall through to headerless 3DL."""
    payload = b"LUT_1D_SIZE nope\n0 255\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n"

    with pytest.raises(ValueError) as error:
        px.io.decode_lut(payload)

    _assert_actionable(error.value)
    assert "LUT_1D_SIZE" in str(error.value)


def test_decode_lut_sniffs_headerless_3dl_after_ignoring_vendor_metadata() -> None:
    """v1-lut-extensions acceptance 19: headerless 3DL sniffing begins at the first active numeric row."""
    text = "Flame export metadata\nLUT name ignored\n" + _headerless_3dl_text((0, 85, 170, 255))

    decoded = px.io.decode_lut(text.encode("utf-8"))

    assert isinstance(decoded, px.core.Lut)
    assert decoded.data.shape == (4, 4, 4, 3)


def test_write_lut_emits_deterministic_cube_text_for_1d_and_3d(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 21-22: Cube output is deterministic, self-contained, ordered UTF-8 text."""
    one_d = px.core.Lut1D(
        cp.asarray(((-0.0, 0.1, 1.0), (2.0, -1.5, 0.25)), dtype=cp.float32),
        domain_min=(-1.0, -2.0, -3.0),
        domain_max=(1.0, 2.0, 3.0),
    )
    one_path = tmp_path / "one.CUBE"

    assert px.io.write_lut(one_path, one_d) is None

    assert one_path.read_bytes() == (
        b"LUT_1D_SIZE 2\nDOMAIN_MIN -1.0 -2.0 -3.0\nDOMAIN_MAX 1.0 2.0 3.0\n-0.0 0.1 1.0\n2.0 -1.5 0.25\n"
    )

    values = np.empty((2, 2, 2, 3), dtype=np.float32)
    for red in range(2):
        for green in range(2):
            for blue in range(2):
                values[red, green, blue] = (red + 10 * green + 100 * blue, red, blue)
    three_path = tmp_path / "three.cube"
    px.io.write_lut(three_path, px.core.Lut(cp.asarray(values)))
    lines = three_path.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["LUT_3D_SIZE 2", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    assert lines[3:] == [
        "0.0 0.0 0.0",
        "1.0 1.0 0.0",
        "10.0 0.0 0.0",
        "11.0 1.0 0.0",
        "100.0 0.0 1.0",
        "101.0 1.0 1.0",
        "110.0 0.0 1.0",
        "111.0 1.0 1.0",
    ]
    assert three_path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("kind", ("1d", "3d"))
def test_write_read_roundtrip_preserves_every_float32_bit_and_domain(tmp_path: Path, kind: str) -> None:
    """v1-lut-extensions acceptance 23: finite programmatic LUTs round-trip every float32 data bit."""
    values = np.asarray(
        (
            -0.0,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            np.float32(1.0e20),
            np.float32(-1.25),
            np.float32(3.5),
        ),
        dtype=np.float32,
    )
    if kind == "1d":
        data = values.reshape(2, 3)
        original: px.core.Lut | px.core.Lut1D = px.core.Lut1D(
            cp.asarray(data), domain_min=(-3.0, -2.0, -1.0), domain_max=(1.0, 2.0, 3.0)
        )
    else:
        data = np.resize(values, (2, 2, 2, 3)).astype(np.float32)
        original = px.core.Lut(cp.asarray(data), domain_min=(-3.0, -2.0, -1.0), domain_max=(1.0, 2.0, 3.0))
    path = tmp_path / f"{kind}.cube"

    px.io.write_lut(path, original)
    restored = px.io.read_lut(path)

    assert type(restored) is type(original)
    assert restored.domain_min == original.domain_min
    assert restored.domain_max == original.domain_max
    np.testing.assert_array_equal(
        cp.asnumpy(restored.data).view(np.uint32),
        cp.asnumpy(original.data).view(np.uint32),
    )


def test_write_lut_rejects_before_mutation_and_preserves_backend_causes(tmp_path: Path) -> None:
    """v1-lut-extensions acceptance 21 and 24: validation is pre-mutation and write failures retain causes."""
    finite = px.core.Lut1D(cp.zeros((2, 3), dtype=cp.float32))
    existing = tmp_path / "existing.look"
    existing.write_bytes(b"unchanged")
    with pytest.raises(ValueError) as extension_error:
        px.io.write_lut(existing, finite)
    _assert_actionable(extension_error.value)
    assert existing.read_bytes() == b"unchanged"

    nonfinite_path = tmp_path / "nonfinite.cube"
    nonfinite = px.core.Lut1D(cp.asarray(((0.0, np.nan, 1.0), (1.0, 2.0, 3.0)), dtype=cp.float32))
    with pytest.raises(ValueError) as finite_error:
        px.io.write_lut(nonfinite_path, nonfinite)
    _assert_actionable(finite_error.value)
    assert not nonfinite_path.exists()

    missing_parent = tmp_path / "missing" / "output.cube"
    with pytest.raises(RuntimeError) as parent_error:
        px.io.write_lut(missing_parent, finite)
    _assert_actionable(parent_error.value)
    assert isinstance(parent_error.value.__cause__, FileNotFoundError)
    assert not missing_parent.parent.exists()

    directory_path = tmp_path / "directory.cube"
    directory_path.mkdir()
    with pytest.raises(RuntimeError) as write_error:
        px.io.write_lut(directory_path, finite)
    _assert_actionable(write_error.value)
    assert isinstance(write_error.value.__cause__, OSError)


@pytest.mark.parametrize(
    "text",
    (
        _cube_1d_text(),
        _cube_3d_text(),
        _headerless_3dl_text((0, 85, 170, 255)),
        _spi1d_text(),
        _spi3d_text(),
    ),
)
def test_each_decode_parser_performs_one_bulk_host_to_device_transfer(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """v1-lut-extensions acceptance 25: every decoded LUT crosses the host-to-device boundary exactly once."""
    import pixtreme._io.formats.lut as implementation

    original = implementation.cp.asarray
    transfers: list[np.ndarray] = []

    def counted(value: np.ndarray, *args: object, **kwargs: object) -> cp.ndarray:
        transfers.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(implementation.cp, "asarray", counted)

    px.io.decode_lut(text.encode("utf-8"))

    assert len(transfers) == 1
    assert isinstance(transfers[0], np.ndarray)


@pytest.mark.parametrize("kind", ("1d", "3d"))
def test_write_lut_performs_one_device_to_host_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    """v1-lut-extensions acceptance 25: write_lut transfers each LUT table to host exactly once without caching."""
    import pixtreme._io.formats.lut as implementation

    lut: px.core.Lut | px.core.Lut1D
    if kind == "1d":
        lut = px.core.Lut1D(cp.zeros((2, 3), dtype=cp.float32))
    else:
        lut = px.core.Lut(cp.zeros((2, 2, 2, 3), dtype=cp.float32))
    original = implementation.cp.asnumpy
    transfers: list[cp.ndarray] = []

    def counted(value: cp.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        transfers.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(implementation.cp, "asnumpy", counted)

    px.io.write_lut(tmp_path / f"{kind}.cube", lut)

    assert transfers == [lut.data]
    assert not hasattr(px.io.read_lut, "cache_info")
    assert not hasattr(px.io.decode_lut, "cache_info")


def test_lut_io_public_signatures_and_documentation_contract_match_the_feature() -> None:
    """v1-lut-extensions acceptance 4 and 26; v1-white-balance acceptance 1;
    v1-white-point-simulation acceptance 1:
    public signatures, counts, types, and boundary canon stay aligned.
    """
    assert tuple(inspect.signature(px.io.read_lut).parameters) == ("path",)
    assert tuple(inspect.signature(px.io.decode_lut).parameters) == ("data",)
    assert tuple(inspect.signature(px.io.write_lut).parameters) == ("path", "lut")
    assert len([name for name in px.io.__all__ if inspect.isfunction(getattr(px.io, name))]) == 26

    requirements_path = ROOT / "docs" / "requirements.md"
    if not requirements_path.is_file():
        pytest.skip("repo-only documentation contract: canonical requirements are absent from this distribution")
    requirements = requirements_path.read_text(encoding="utf-8")
    public_section = requirements.split("**REQ-API-009", maxsplit=1)[1].split("**REQ-API-010", maxsplit=1)[0]
    assert "| `io`" in public_section and "| 26 |" in public_section
    assert "公開 operation は計 94 関数" in public_section
    assert "公開型" in public_section and "`core.Lut1D`" in public_section and "5 点" in public_section
    boundary = requirements.split("**REQ-API-010", maxsplit=1)[1].split("**REQ-API-011", maxsplit=1)[0]
    assert "LUT file" in boundary and "`px.io.read_lut`" in boundary and "`px.io.write_lut`" in boundary
    assert "LUT bytes" in boundary and "`px.io.decode_lut`" in boundary
