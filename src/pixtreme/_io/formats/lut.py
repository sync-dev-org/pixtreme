"""User-provided one- and three-dimensional LUT boundaries."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.lut import _DEFAULT_DOMAIN_MAX, _DEFAULT_DOMAIN_MIN, Lut, Lut1D
from pixtreme._io.common import _coerce_path

_INTEGER_TOKEN = re.compile(r"[+-]?\d+\Z")
_CUBE_MARKER = re.compile(r"(?mi)^[ \t]*LUT_(?:1D|3D)_SIZE\b")
_SPI3D_MARKER = re.compile(r"(?m)^[ \t]*SPILUT\b")
_THREE_DL_MARKER = re.compile(r"(?m)^[ \t]*(?:3DMESH|Mesh)\b")
_CUBE_DIRECTIVE_LINE = re.compile(
    r"(?mi)^[ \t]*(?:TITLE|LUT_1D_SIZE|LUT_3D_SIZE|DOMAIN_MIN|DOMAIN_MAX)\b[^\r\n]*(?:\r?\n|$)"
)
_SUPPORTED_EXTENSIONS = frozenset({".cube", ".3dl", ".spi1d", ".spi3d"})


@dataclass(frozen=True, slots=True)
class _ParsedLut:
    data: np.ndarray
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    dimension: Literal[1, 3]


@dataclass(frozen=True, slots=True)
class _CubeParts:
    data_text: str
    row_count: int
    size: int
    dimension: Literal[1, 3]
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]


def _error(*, why: str, what: str, how: str) -> ValueError:
    return ValueError(_actionable_error(why=why, what=what, how=how))


def _without_comments(text: str) -> str:
    return text if "#" not in text else re.sub(r"(?m)#.*$", "", text)


def _active_lines(text: str) -> list[str]:
    return [line.strip() for line in _without_comments(text).splitlines() if line.strip()]


def _finite_float_tokens(
    tokens: tuple[str, ...],
    *,
    count: int,
    label: str,
    source: str,
) -> tuple[float, ...]:
    if len(tokens) != count:
        raise _error(
            why=f"{label} must declare exactly {count} numeric value{'s' if count != 1 else ''}",
            what=f"{source} provides tokens {tokens!r}",
            how=f"write {label} followed by {count} finite decimal value{'s' if count != 1 else ''}",
        )
    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as error:
        raise _error(
            why=f"{label} contains a non-numeric value",
            what=f"{source} provides tokens {tokens!r}",
            how=f"write {label} with finite decimal values",
        ) from error
    if not all(np.isfinite(values)):
        raise _error(
            why=f"{label} values must be finite",
            what=f"{source} provides {values!r}",
            how=f"replace NaN or infinite {label} values with finite decimal values",
        )
    return values


def _validate_domain(
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    *,
    source: str,
) -> None:
    if not all(lower < upper for lower, upper in zip(domain_min, domain_max)):
        raise _error(
            why="a LUT input domain must increase in every RGB channel",
            what=f"{source} declares domain_min={domain_min!r}, domain_max={domain_max!r}",
            how="choose finite DOMAIN_MIN values strictly below the corresponding DOMAIN_MAX values",
        )


def _single_declaration(
    declarations: dict[str, list[tuple[str, ...]]],
    name: str,
    *,
    required: bool,
    source: str,
) -> tuple[str, ...] | None:
    matches = declarations.get(name, [])
    if len(matches) > 1:
        raise _error(
            why=f"a LUT payload must declare {name} at most once",
            what=f"{source} contains {len(matches)} declarations",
            how=f"retain one valid {name} declaration",
        )
    if not matches:
        if required:
            raise _error(
                why=f"a LUT payload must declare {name}",
                what=f"{source} has no {name} declaration",
                how=f"add one valid {name} declaration",
            )
        return None
    return matches[0]


def _cube_declarations(text: str) -> dict[str, list[tuple[str, ...]]]:
    declarations: dict[str, list[tuple[str, ...]]] = {}
    for name in ("LUT_1D_SIZE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX"):
        matches = re.findall(rf"(?mi)^[ \t]*{name}\b([^\r\n]*)\r?$", text)
        if matches:
            declarations[name] = [tuple(match.split()) for match in matches]
    return declarations


def _data_row_widths(data_text: str) -> np.ndarray:
    encoded = data_text.encode("utf-8")
    if not encoded:
        return np.empty(0, dtype=np.int64)
    raw = np.frombuffer(encoded, dtype=np.uint8)
    whitespace = raw <= ord(" ")
    token_starts = ~whitespace
    token_starts[1:] &= whitespace[:-1]
    line_starts = np.concatenate((np.asarray((0,), dtype=np.intp), np.flatnonzero(raw == ord("\n")) + 1))
    line_starts = line_starts[line_starts < raw.size]
    return np.add.reduceat(token_starts, line_starts)


def _cube_parts(text: str, *, source: str) -> _CubeParts:
    without_comments = _without_comments(text)
    declarations = _cube_declarations(without_comments)
    data_text = _CUBE_DIRECTIVE_LINE.sub("", without_comments)

    one_d = _single_declaration(declarations, "LUT_1D_SIZE", required=False, source=source)
    three_d = _single_declaration(declarations, "LUT_3D_SIZE", required=False, source=source)
    if one_d is None and three_d is None:
        raise _error(
            why="a .cube payload must declare exactly one LUT size",
            what=f"{source} has neither LUT_1D_SIZE nor LUT_3D_SIZE",
            how="add one LUT_1D_SIZE or LUT_3D_SIZE declaration before or among the data rows",
        )
    if one_d is not None and three_d is not None:
        raise _error(
            why="a .cube payload cannot combine one- and three-dimensional LUTs",
            what=f"{source} declares both LUT_1D_SIZE and LUT_3D_SIZE",
            how="retain exactly one size declaration and its matching data table",
        )
    dimension: Literal[1, 3] = 1 if one_d is not None else 3
    size_tokens = one_d if one_d is not None else three_d
    assert size_tokens is not None
    label = "LUT_1D_SIZE" if dimension == 1 else "LUT_3D_SIZE"
    if len(size_tokens) != 1 or _INTEGER_TOKEN.fullmatch(size_tokens[0]) is None:
        raise _error(
            why=f"{label} must declare one integer",
            what=f"{source} provides tokens {size_tokens!r}",
            how=f"write {label} <N> with N in the supported range",
        )
    size = int(size_tokens[0])
    valid_size = 2 <= size <= 65536 if dimension == 1 else size >= 2
    if not valid_size:
        supported = "2 through 65536" if dimension == 1 else "at least 2"
        raise _error(
            why=f"{label} is outside its supported range",
            what=f"{source} declares {label} {size}",
            how=f"choose a size {supported}",
        )

    domain_min_tokens = _single_declaration(declarations, "DOMAIN_MIN", required=False, source=source)
    domain_max_tokens = _single_declaration(declarations, "DOMAIN_MAX", required=False, source=source)
    domain_min = (
        _DEFAULT_DOMAIN_MIN
        if domain_min_tokens is None
        else tuple(_finite_float_tokens(domain_min_tokens, count=3, label="DOMAIN_MIN", source=source))
    )
    domain_max = (
        _DEFAULT_DOMAIN_MAX
        if domain_max_tokens is None
        else tuple(_finite_float_tokens(domain_max_tokens, count=3, label="DOMAIN_MAX", source=source))
    )
    domain_min = (float(domain_min[0]), float(domain_min[1]), float(domain_min[2]))
    domain_max = (float(domain_max[0]), float(domain_max[1]), float(domain_max[2]))
    _validate_domain(domain_min, domain_max, source=source)

    row_widths = _data_row_widths(data_text)
    invalid_width = bool(np.any((row_widths != 0) & (row_widths != 3)))
    if invalid_width:
        raise _error(
            why="every .cube data row must contain exactly three numeric values",
            what=f"{source} contains a data row whose width is not three",
            how="write one red green blue output triplet per data row",
        )
    row_count = int(np.count_nonzero(row_widths))
    expected_rows = size if dimension == 1 else size**3
    if row_count != expected_rows:
        raise _error(
            why="the .cube data row count must exactly match its declared size",
            what=f"{source} requires {expected_rows} rows and contains {row_count}",
            how="add or remove complete RGB rows to match the size declaration",
        )
    return _CubeParts(
        data_text=data_text,
        row_count=row_count,
        size=size,
        dimension=dimension,
        domain_min=domain_min,
        domain_max=domain_max,
    )


def _parse_cube(text: str, *, source: str) -> _ParsedLut:
    parts = _cube_parts(text, source=source)
    expected_values = parts.row_count * 3
    try:
        with np.errstate(invalid="ignore"):
            values = np.fromstring(parts.data_text, dtype=np.float32, sep=" ")
    except ValueError as error:
        raise _error(
            why=".cube data rows must contain numeric values",
            what=f"{source} contains a non-numeric data token",
            how="replace each invalid token with a decimal RGB value",
        ) from error
    if values.size != expected_values:
        raise _error(
            why=".cube data rows must contain numeric values",
            what=f"{source} contains a non-numeric data token",
            how="replace each invalid token with a decimal RGB value",
        )
    if parts.dimension == 1:
        data = np.ascontiguousarray(values.reshape(parts.size, 3))
    else:
        red_fastest = values.reshape(parts.size, parts.size, parts.size, 3)
        data = np.ascontiguousarray(red_fastest.transpose(2, 1, 0, 3))
    return _ParsedLut(data, parts.domain_min, parts.domain_max, parts.dimension)


def _integer_rows(lines: list[str], *, source: str) -> tuple[np.ndarray, np.ndarray]:
    numeric_lines: list[tuple[str, ...]] = []
    for line in lines:
        tokens = tuple(line.split())
        first = tokens[0]
        if _INTEGER_TOKEN.fullmatch(first) is None:
            continue
        if any(_INTEGER_TOKEN.fullmatch(token) is None for token in tokens):
            raise _error(
                why="every numeric .3dl row must contain integers only",
                what=f"{source} contains row tokens {tokens!r}",
                how="replace fractional or non-numeric tokens with integer sample codes",
            )
        numeric_lines.append(tokens)
    if not numeric_lines:
        raise _error(
            why="a .3dl payload requires a sample-spacing row and cube rows",
            what=f"{source} contains no active numeric rows",
            how="write one integer spacing row followed by RGB integer output rows",
        )
    spacing_tokens = numeric_lines[0]
    cube_rows = numeric_lines[1:]
    invalid_cube_row = next((row for row in cube_rows if len(row) != 3), None)
    if invalid_cube_row is not None:
        raise _error(
            why="a .3dl payload permits exactly one sample-spacing row",
            what=f"{source} contains another row with {len(invalid_cube_row)} integer values",
            how="retain one spacing row followed only by three-integer RGB output rows",
        )
    spacing = np.fromstring(" ".join(spacing_tokens), dtype=np.int64, sep=" ")
    cube = np.fromstring("\n".join(" ".join(row) for row in cube_rows), dtype=np.int64, sep=" ")
    if np.any(spacing < 0) or np.any(cube < 0):
        raise _error(
            why=".3dl integer sample codes cannot be negative",
            what=f"{source} contains a negative spacing or output code",
            how="use integer codes from zero through 65535",
        )
    if np.any(spacing > 65535) or np.any(cube > 65535):
        raise _error(
            why=".3dl integer sample codes cannot exceed 65535",
            what=f"{source} contains a spacing or output code above 65535",
            how="use integer codes from zero through 65535",
        )
    return spacing, cube.reshape(-1, 3)


def _three_dl_scale(maximum: int) -> float:
    if maximum <= 511:
        return 255.0
    if maximum <= 2047:
        return 1023.0
    if maximum <= 8191:
        return 4095.0
    return 65535.0


def _tetrahedral_grid(cube: np.ndarray, shaped_axis: np.ndarray) -> np.ndarray:
    edge = cube.shape[0]
    scaled = np.clip(shaped_axis, 0.0, 1.0) * (edge - 1)
    lower = np.minimum(np.floor(scaled).astype(np.intp), edge - 2)
    fraction = scaled - lower
    red = lower[:, None, None]
    green = lower[None, :, None]
    blue = lower[None, None, :]
    fr = np.broadcast_to(fraction[:, None, None], (edge, edge, edge))
    fg = np.broadcast_to(fraction[None, :, None], (edge, edge, edge))
    fb = np.broadcast_to(fraction[None, None, :], (edge, edge, edge))
    v000 = cube[red, green, blue]
    v100 = cube[red + 1, green, blue]
    v010 = cube[red, green + 1, blue]
    v001 = cube[red, green, blue + 1]
    v110 = cube[red + 1, green + 1, blue]
    v101 = cube[red + 1, green, blue + 1]
    v011 = cube[red, green + 1, blue + 1]
    v111 = cube[red + 1, green + 1, blue + 1]
    result = np.empty((edge, edge, edge, 3), dtype=np.float64)
    cases = (
        (
            (fr >= fg) & (fg >= fb),
            v000 + fr[..., None] * (v100 - v000) + fg[..., None] * (v110 - v100) + fb[..., None] * (v111 - v110),
        ),
        (
            (fr >= fb) & (fb > fg),
            v000 + fr[..., None] * (v100 - v000) + fb[..., None] * (v101 - v100) + fg[..., None] * (v111 - v101),
        ),
        (
            (fb > fr) & (fr >= fg),
            v000 + fb[..., None] * (v001 - v000) + fr[..., None] * (v101 - v001) + fg[..., None] * (v111 - v101),
        ),
        (
            (fg > fr) & (fr >= fb),
            v000 + fg[..., None] * (v010 - v000) + fr[..., None] * (v110 - v010) + fb[..., None] * (v111 - v110),
        ),
        (
            (fg >= fb) & (fb > fr),
            v000 + fg[..., None] * (v010 - v000) + fb[..., None] * (v011 - v010) + fr[..., None] * (v111 - v011),
        ),
        (
            (fb > fg) & (fg > fr),
            v000 + fb[..., None] * (v001 - v000) + fg[..., None] * (v011 - v001) + fr[..., None] * (v111 - v011),
        ),
    )
    assigned = np.zeros((edge, edge, edge), dtype=np.bool_)
    for mask, values in cases:
        result[mask] = values[mask]
        assigned |= mask
    assert bool(np.all(assigned))
    return result


def _parse_3dl(text: str, *, source: str) -> _ParsedLut:
    lines = _active_lines(text)
    mesh_marker_count = 0
    mesh_declarations: list[tuple[str, ...]] = []
    for line in lines:
        tokens = tuple(line.split())
        first = tokens[0]
        if first.lower() == "3dmesh":
            if first != "3DMESH" or len(tokens) != 1:
                raise _error(
                    why="the Lustre 3DMESH marker is case-sensitive and has no arguments",
                    what=f"{source} contains {tokens!r}",
                    how="write the exact line 3DMESH",
                )
            mesh_marker_count += 1
        elif first.lower() == "mesh":
            if first != "Mesh":
                raise _error(
                    why="the Lustre Mesh declaration is case-sensitive",
                    what=f"{source} contains {first!r}",
                    how="write Mesh <input-bits> <output-bits>",
                )
            mesh_declarations.append(tokens[1:])
    lustre = mesh_marker_count > 0 or bool(mesh_declarations)
    if lustre and (mesh_marker_count != 1 or len(mesh_declarations) != 1):
        raise _error(
            why="a Lustre .3dl payload requires one 3DMESH marker and one Mesh declaration",
            what=f"{source} contains {mesh_marker_count} markers and {len(mesh_declarations)} declarations",
            how="retain one exact 3DMESH line and one Mesh <input-bits> <output-bits> line",
        )

    spacing_codes, cube_codes = _integer_rows(lines, source=source)
    edge = int(spacing_codes.size)
    if edge < 2:
        raise _error(
            why="a .3dl spacing row requires at least two samples",
            what=f"{source} contains {edge} spacing values",
            how="provide a spacing row whose value count is the cube edge",
        )
    expected_rows = edge**3
    if cube_codes.shape[0] != expected_rows:
        raise _error(
            why="the .3dl cube row count must equal the spacing edge cubed",
            what=f"{source} requires {expected_rows} rows and contains {cube_codes.shape[0]}",
            how="add or remove complete RGB integer rows to match the spacing count",
        )

    if lustre:
        declaration = mesh_declarations[0]
        if len(declaration) != 2 or any(_INTEGER_TOKEN.fullmatch(token) is None for token in declaration):
            raise _error(
                why="a Lustre Mesh declaration requires two integer bit depths",
                what=f"{source} provides tokens {declaration!r}",
                how="write Mesh <input-bits> <output-bits>",
            )
        input_bits, output_bits = (int(token) for token in declaration)
        if input_bits not in {4, 5, 6} or output_bits not in {8, 10, 12, 16}:
            raise _error(
                why="Lustre .3dl supports edges 17, 33, or 65 and output depths 8, 10, 12, or 16",
                what=f"{source} declares Mesh {input_bits} {output_bits}",
                how="choose input bits 4, 5, or 6 and a supported output bit depth",
            )
        declared_edge = 2**input_bits + 1
        if edge != declared_edge:
            raise _error(
                why="the Lustre spacing row must match the Mesh input depth",
                what=f"{source} declares edge {declared_edge} but contains {edge} spacing values",
                how=f"provide exactly {declared_edge} spacing values and {declared_edge**3} cube rows",
            )
        output_scale = float(2**output_bits - 1)
        if cube_codes.size and int(cube_codes.max()) > output_scale:
            raise _error(
                why="a Lustre output code exceeds its declared output bit depth",
                what=f"{source} declares {output_bits} bits and contains code {int(cube_codes.max())}",
                how=f"use output codes from zero through {int(output_scale)}",
            )
    else:
        output_scale = _three_dl_scale(int(cube_codes.max(initial=0)))

    spacing_scale = _three_dl_scale(int(spacing_codes.max(initial=0)))
    spacing = spacing_codes.astype(np.float64) / spacing_scale
    expected_spacing_codes = np.linspace(0.0, spacing_scale, edge, dtype=np.float64)
    identity_spacing = bool(np.all(np.abs(spacing_codes.astype(np.float64) - expected_spacing_codes) <= 0.5))
    cube = cube_codes.astype(np.float64).reshape(edge, edge, edge, 3) / output_scale
    if not identity_spacing:
        cube = _tetrahedral_grid(cube, spacing)
    return _ParsedLut(
        np.ascontiguousarray(cube.astype(np.float32)),
        _DEFAULT_DOMAIN_MIN,
        _DEFAULT_DOMAIN_MAX,
        3,
    )


def _parse_spi1d(text: str, *, source: str) -> _ParsedLut:
    declarations: dict[str, list[tuple[str, ...]]] = {}
    data_lines: list[str] = []
    state: Literal["header", "table", "closed"] = "header"
    for line in _active_lines(text):
        if line == "{":
            if state != "header":
                raise _error(
                    why="an SPI1D payload requires one brace-delimited table",
                    what=f"{source} contains an unexpected opening brace",
                    how="place one opening brace after the declarations",
                )
            state = "table"
            continue
        if line == "}":
            if state != "table":
                raise _error(
                    why="an SPI1D payload requires one brace-delimited table",
                    what=f"{source} contains an unexpected closing brace",
                    how="close the table once after all sample rows",
                )
            state = "closed"
            continue
        tokens = tuple(line.split())
        if state == "header":
            name = tokens[0]
            if name not in {"Version", "From", "Length", "Components"}:
                raise _error(
                    why="SPI1D declarations are case-sensitive",
                    what=f"{source} contains unrecognized header {tokens!r}",
                    how="use Version, From, Length, and Components with their exact spelling",
                )
            declarations.setdefault(name, []).append(tokens[1:])
        elif state == "table":
            data_lines.append(line)
        else:
            raise _error(
                why="SPI1D data cannot follow the closing brace",
                what=f"{source} contains trailing content {line!r}",
                how="remove content after the table's closing brace",
            )
    if state != "closed":
        raise _error(
            why="an SPI1D payload requires a complete brace-delimited table",
            what=f"{source} ended while parser state was {state}",
            how="place the sample rows between one opening and one closing brace",
        )

    version = _single_declaration(declarations, "Version", required=True, source=source)
    length_tokens = _single_declaration(declarations, "Length", required=True, source=source)
    component_tokens = _single_declaration(declarations, "Components", required=True, source=source)
    from_tokens = _single_declaration(declarations, "From", required=False, source=source)
    if version != ("1",):
        raise _error(
            why="SPI1D supports exactly Version 1",
            what=f"{source} declares Version tokens {version!r}",
            how="write the exact declaration Version 1",
        )
    assert length_tokens is not None and component_tokens is not None
    if len(length_tokens) != 1 or _INTEGER_TOKEN.fullmatch(length_tokens[0]) is None:
        raise _error(
            why="SPI1D Length must be one integer",
            what=f"{source} provides {length_tokens!r}",
            how="write Length <N> with N at least 2",
        )
    length = int(length_tokens[0])
    if length < 2:
        raise _error(
            why="SPI1D Length must be at least 2",
            what=f"{source} declares Length {length}",
            how="provide at least two uniformly spaced sample rows",
        )
    if len(component_tokens) != 1 or component_tokens[0] not in {"1", "2", "3"}:
        raise _error(
            why="SPI1D Components must be 1, 2, or 3",
            what=f"{source} provides {component_tokens!r}",
            how="write Components 1, Components 2, or Components 3",
        )
    components = int(component_tokens[0])
    invalid_width = next((tuple(line.split()) for line in data_lines if len(line.split()) != components), None)
    if invalid_width is not None:
        raise _error(
            why="each SPI1D table row must match Components",
            what=f"{source} contains row tokens {invalid_width!r}",
            how=f"write exactly {components} float values per sample row",
        )
    if len(data_lines) != length:
        raise _error(
            why="the SPI1D row count must match Length",
            what=f"{source} declares {length} rows and contains {len(data_lines)}",
            how="add or remove complete sample rows to match Length",
        )
    try:
        values = np.fromstring("\n".join(data_lines), dtype=np.float32, sep=" ")
    except ValueError as error:
        raise _error(
            why="SPI1D table rows must contain numeric values",
            what=f"{source} contains a non-numeric sample token",
            how="replace each invalid token with a decimal output value",
        ) from error
    if values.size != length * components:
        raise _error(
            why="SPI1D table rows must contain numeric values",
            what=f"{source} contains a non-numeric sample token",
            how="replace each invalid token with a decimal output value",
        )
    table = values.reshape(length, components)
    if components == 1:
        data = np.repeat(table, 3, axis=1)
    elif components == 2:
        data = np.concatenate((table, np.zeros((length, 1), dtype=np.float32)), axis=1)
    else:
        data = table
    if from_tokens is None:
        lower, upper = 0.0, 1.0
    else:
        lower, upper = _finite_float_tokens(from_tokens, count=2, label="From", source=source)
    if lower >= upper:
        raise _error(
            why="SPI1D From must declare an increasing finite domain",
            what=f"{source} declares From {lower} {upper}",
            how="choose a finite minimum strictly below the maximum",
        )
    domain_min = (lower, lower, lower)
    domain_max = (upper, upper, upper)
    return _ParsedLut(np.ascontiguousarray(data), domain_min, domain_max, 1)


def _parse_spi3d(text: str, *, source: str) -> _ParsedLut:
    lines = _active_lines(text)
    if len(lines) < 3:
        raise _error(
            why="SPI3D requires three header lines",
            what=f"{source} contains {len(lines)} active lines",
            how="write SPILUT 1.0, 3 3, and the three lattice sizes before data",
        )
    if tuple(lines[0].split()) != ("SPILUT", "1.0"):
        raise _error(
            why="SPI3D requires the exact case-sensitive marker SPILUT 1.0",
            what=f"{source} begins with {lines[0]!r}",
            how="write SPILUT 1.0 as the first active line",
        )
    if tuple(lines[1].split()) != ("3", "3"):
        raise _error(
            why="SPI3D requires the exact channel declaration 3 3",
            what=f"{source} declares {lines[1]!r}",
            how="write 3 3 as the second active line",
        )
    size_tokens = tuple(lines[2].split())
    if len(size_tokens) != 3 or any(_INTEGER_TOKEN.fullmatch(token) is None for token in size_tokens):
        raise _error(
            why="SPI3D lattice sizes must be three integers",
            what=f"{source} provides {size_tokens!r}",
            how="write equal red green blue sizes with each size at least 2",
        )
    sizes = tuple(int(token) for token in size_tokens)
    if sizes[0] < 2 or sizes[0] != sizes[1] or sizes[0] != sizes[2]:
        raise _error(
            why="SPI3D requires one equal lattice size of at least 2 on all axes",
            what=f"{source} declares sizes {sizes!r}",
            how="write N N N with N at least 2",
        )
    size = sizes[0]
    rows = [tuple(line.split()) for line in lines[3:]]
    invalid_width = next((row for row in rows if len(row) != 6), None)
    if invalid_width is not None:
        raise _error(
            why="each SPI3D data row must contain three indices and three outputs",
            what=f"{source} contains row tokens {invalid_width!r}",
            how="write red-index green-index blue-index red green blue",
        )
    if len(rows) != size**3:
        raise _error(
            why="the SPI3D row count must equal the lattice size cubed",
            what=f"{source} requires {size**3} rows and contains {len(rows)}",
            how="provide one row for every RGB lattice coordinate",
        )
    if any(_INTEGER_TOKEN.fullmatch(token) is None for row in rows for token in row[:3]):
        raise _error(
            why="SPI3D indices must be integers",
            what=f"{source} contains a non-integer coordinate token",
            how="write integer red, green, and blue indices before each output triplet",
        )
    index_text = "\n".join(" ".join(row[:3]) for row in rows)
    output_text = "\n".join(" ".join(row[3:]) for row in rows)
    indices = np.fromstring(index_text, dtype=np.int64, sep=" ").reshape(-1, 3)
    try:
        outputs = np.fromstring(output_text, dtype=np.float32, sep=" ")
    except ValueError as error:
        raise _error(
            why="SPI3D outputs must be numeric",
            what=f"{source} contains a non-numeric output token",
            how="replace each invalid token with a decimal RGB output value",
        ) from error
    if outputs.size != size**3 * 3:
        raise _error(
            why="SPI3D outputs must be numeric",
            what=f"{source} contains a non-numeric output token",
            how="replace each invalid token with a decimal RGB output value",
        )
    if np.any(indices < 0) or np.any(indices >= size):
        raise _error(
            why="SPI3D indices must remain inside the declared lattice",
            what=f"{source} contains an index outside zero through {size - 1}",
            how="replace the coordinate with an in-range RGB index",
        )
    linear = (indices[:, 0] * size + indices[:, 1]) * size + indices[:, 2]
    if np.unique(linear).size != size**3:
        raise _error(
            why="SPI3D requires each lattice index exactly once",
            what=f"{source} contains a duplicate index and therefore a missing index",
            how="provide one unique row for every RGB coordinate",
        )
    data = np.empty((size, size, size, 3), dtype=np.float32)
    data[indices[:, 0], indices[:, 1], indices[:, 2]] = outputs.reshape(-1, 3)
    return _ParsedLut(data, _DEFAULT_DOMAIN_MIN, _DEFAULT_DOMAIN_MAX, 3)


def _looks_like_headerless_3dl(text: str) -> bool:
    lines = _active_lines(text)
    rows: list[tuple[str, ...]] = []
    for line in lines:
        tokens = tuple(line.split())
        if _INTEGER_TOKEN.fullmatch(tokens[0]) is None:
            continue
        if any(_INTEGER_TOKEN.fullmatch(token) is None for token in tokens):
            return False
        rows.append(tokens)
    if len(rows) < 2:
        return False
    if len(rows[0]) <= 3:
        return False
    if any(len(row) != 3 for row in rows[1:]):
        return False
    edge = len(rows[0])
    return len(rows) - 1 == edge**3


def _sniff_parser(text: str) -> Literal[".cube", ".3dl", ".spi1d", ".spi3d"]:
    spi3d = _SPI3D_MARKER.search(_without_comments(text)) is not None
    spi1d_lines = _active_lines(text)
    spi1d_names = {line.split()[0] for line in spi1d_lines if line not in {"{", "}"}}
    spi1d = {"Version", "Length", "Components"}.issubset(spi1d_names) and "{" in spi1d_lines and "}" in spi1d_lines
    cube = _CUBE_MARKER.search(_without_comments(text)) is not None
    three_dl = _THREE_DL_MARKER.search(_without_comments(text)) is not None
    markers = [
        name for name, present in (("SPI3D", spi3d), ("SPI1D", spi1d), ("Cube", cube), ("3DL", three_dl)) if present
    ]
    if len(markers) > 1:
        raise _error(
            why="a LUT byte payload cannot contain markers for multiple formats",
            what=f"found markers {tuple(markers)!r}",
            how="provide one complete Cube, 3DL, SPI1D, or SPI3D payload",
        )
    if markers:
        return {"SPI3D": ".spi3d", "SPI1D": ".spi1d", "Cube": ".cube", "3DL": ".3dl"}[markers[0]]  # type: ignore[return-value]
    if _looks_like_headerless_3dl(text):
        return ".3dl"
    raise _error(
        why="the LUT byte payload format could not be identified",
        what="no unique supported format marker or headerless 3DL structure was found",
        how="provide UTF-8 Cube, 3DL, SPI1D, or SPI3D bytes with their required grammar",
    )


def _parse_text(text: str, *, format_name: str, source: str) -> _ParsedLut:
    if format_name == ".cube":
        return _parse_cube(text, source=source)
    if format_name == ".3dl":
        return _parse_3dl(text, source=source)
    if format_name == ".spi1d":
        return _parse_spi1d(text, source=source)
    if format_name == ".spi3d":
        return _parse_spi3d(text, source=source)
    raise AssertionError(f"unreachable LUT format {format_name!r}")


def _to_device_lut(parsed: _ParsedLut) -> Lut | Lut1D:
    if parsed.dimension == 1:
        device_data = cp.asarray(parsed.data)
        return Lut1D(device_data, domain_min=parsed.domain_min, domain_max=parsed.domain_max)
    packed = np.empty((*parsed.data.shape[:3], 4), dtype=np.float32)
    packed[..., :3] = parsed.data
    packed[..., 3] = np.float32(0.0)
    device_packed = cp.asarray(packed)
    return Lut(device_packed[..., :3], domain_min=parsed.domain_min, domain_max=parsed.domain_max)


def read_lut(path: str | os.PathLike[str]) -> Lut | Lut1D:
    """Read a supported LUT text file into GPU memory without caching.

    The extension selects Cube, 3DL, SPI1D, or SPI3D parsing. Cube red-fastest,
    3DL blue-fastest, and SPI explicit-index forms normalize to RGB-indexed
    public values. A nonidentity 3DL shaper is baked onto an equally sized grid;
    that single-LUT approximation depends on the source edge density and local
    transform curvature. Parsing performs one bulk host-to-device transfer.
    """
    file_path = _coerce_path(path, kind="LUT")
    extension = file_path.suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise _error(
            why="read_lut supports a closed set of LUT file extensions",
            what=f"received extension {file_path.suffix!r} for {file_path}",
            how="use a .cube, .3dl, .spi1d, or .spi3d path",
        )
    if not file_path.is_file():
        raise FileNotFoundError(
            _actionable_error(
                why="the LUT file does not exist",
                what=str(file_path),
                how=f"provide an existing {extension} file path",
            )
        )
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _error(
            why="the LUT file could not be read as UTF-8 text",
            what=f"{file_path} failed with {type(error).__name__}",
            how="provide a readable UTF-8 LUT text file",
        ) from error
    return _to_device_lut(_parse_text(text, format_name=extension, source=str(file_path)))


def decode_lut(data: bytes) -> Lut | Lut1D:
    """Decode supported UTF-8 LUT bytes into GPU memory without caching.

    Format markers are resolved in SPI3D, SPI1D, Cube, then 3DL specificity
    order. Headerless 3DL is selected only by its complete numeric structure.
    A selected parser never falls through. Nonidentity 3DL shapers are baked as
    the equally sized single-grid approximation documented by :func:`read_lut`.
    """
    if not isinstance(data, bytes):
        raise _error(
            why="decode_lut requires a bytes payload",
            what=f"received {type(data).__module__}.{type(data).__qualname__}",
            how="encode a supported LUT text as UTF-8 bytes",
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _error(
            why="decode_lut requires UTF-8 text bytes",
            what=f"decoding failed at byte offset {error.start}",
            how="provide a supported LUT payload encoded as UTF-8",
        ) from error
    format_name = _sniff_parser(text)
    return _to_device_lut(_parse_text(text, format_name=format_name, source="LUT bytes"))


def _format_float32(value: np.float32) -> str:
    return str(np.float32(value))


def _format_domain(domain: tuple[float, float, float]) -> str:
    return " ".join(repr(value) for value in domain)


def write_lut(path: str | os.PathLike[str], lut: Lut | Lut1D) -> None:
    """Write a finite one- or three-dimensional LUT as deterministic Cube text.

    The output always includes the RGB input domain and uses float32 shortest
    round-trip decimals. The complete table crosses device-to-host once. Parent
    directories are not created and no cache or ambient registry is consulted.
    """
    file_path = _coerce_path(path, kind="LUT")
    if file_path.suffix.lower() != ".cube":
        raise _error(
            why="write_lut supports Cube output only",
            what=f"received extension {file_path.suffix!r} for {file_path}",
            how="choose a path ending in .cube",
        )
    if not isinstance(lut, (Lut, Lut1D)):
        raise _error(
            why="write_lut requires a Lut or Lut1D value",
            what=f"received {type(lut).__module__}.{type(lut).__qualname__}",
            how="pass a validated px.core.Lut or px.core.Lut1D",
        )
    host_data = cp.asnumpy(lut.data)
    if not bool(np.all(np.isfinite(host_data))):
        raise _error(
            why="Cube output requires finite table values",
            what="the LUT table contains NaN or infinity",
            how="replace every non-finite output with a finite float32 value",
        )
    if isinstance(lut, Lut1D):
        header = f"LUT_1D_SIZE {host_data.shape[0]}"
        rows = host_data.reshape(-1, 3)
    else:
        header = f"LUT_3D_SIZE {host_data.shape[0]}"
        rows = host_data.transpose(2, 1, 0, 3).reshape(-1, 3)
    lines = [
        header,
        f"DOMAIN_MIN {_format_domain(lut.domain_min)}",
        f"DOMAIN_MAX {_format_domain(lut.domain_max)}",
    ]
    lines.extend(" ".join(_format_float32(value) for value in row) for row in rows)
    text = "\n".join(lines) + "\n"
    try:
        file_path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as error:
        raise RuntimeError(
            _actionable_error(
                why="the Cube LUT file could not be written",
                what=f"{file_path} failed with {type(error).__name__}",
                how="choose a writable .cube path whose parent directory already exists",
            )
        ) from error
