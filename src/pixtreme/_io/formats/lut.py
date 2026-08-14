"""User-provided three-dimensional LUT file boundary."""

from __future__ import annotations

import re
from pathlib import Path

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.lut import _DEFAULT_DOMAIN_MAX, _DEFAULT_DOMAIN_MIN, Lut

_DIRECTIVE_LINE = re.compile(
    r"(?mi)^[ \t]*(?:TITLE|LUT_3D_SIZE|LUT_1D_SIZE|DOMAIN_MIN|DOMAIN_MAX)\b[^\r\n]*(?:\r?\n|$)"
)
_DATA_START_LINE = re.compile(r"(?mi)^[ \t]*(?=[+-]?(?:\d|\.\d|inf(?:inity)?\b|nan\b))")


def _directive_tokens(text: str, directive: str) -> tuple[str, ...] | None:
    matches = re.findall(rf"(?mi)^[ \t]*{directive}[ \t]+([^\r\n]+?)[ \t]*\r?$", text)
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            _actionable_error(
                why=f"a .cube file must declare {directive} at most once",
                what=f"found {len(matches)} declarations",
                how=f"retain one valid {directive} line",
            )
        )
    return tuple(matches[0].split())


def _parse_domain(text: str, directive: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    tokens = _directive_tokens(text, directive)
    if tokens is None:
        return default
    if len(tokens) != 3:
        raise ValueError(
            _actionable_error(
                why=f"{directive} must declare exactly three RGB numbers",
                what=f"received tokens {tokens!r}",
                how=f"write {directive} <red> <green> <blue>",
            )
        )
    try:
        values = np.asarray(tokens, dtype=np.float64)
    except ValueError as error:
        raise ValueError(
            _actionable_error(
                why=f"{directive} contains a non-numeric value",
                what=f"received tokens {tokens!r}",
                how=f"write {directive} with three finite decimal numbers",
            )
        ) from error
    return (float(values[0]), float(values[1]), float(values[2]))


def _reject_one_dimensional(text: str, *, source: str) -> None:
    if re.search(r"(?mi)^[ \t]*LUT_1D_SIZE\b", text):
        raise ValueError(
            _actionable_error(
                why="read_lut supports three-dimensional LUTs only",
                what=f"{source} declares LUT_1D_SIZE",
                how="provide a .cube file with LUT_3D_SIZE; 1D LUT support is reserved for a future feature",
            )
        )


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


def _parse_cube(text: str, *, source: str) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    without_comments = text if "#" not in text else re.sub(r"(?m)#.*$", "", text)
    data_match = _DATA_START_LINE.search(without_comments)
    data_start = len(without_comments) if data_match is None else data_match.start()
    header_text = without_comments[:data_start]
    data_text = without_comments[data_start:]
    directive_text = header_text
    _reject_one_dimensional(directive_text, source=source)
    size_tokens = _directive_tokens(directive_text, "LUT_3D_SIZE")
    if size_tokens is None:
        cleaned_data_text = _DIRECTIVE_LINE.sub("", data_text)
        if cleaned_data_text != data_text:
            data_text = cleaned_data_text
            directive_text = without_comments
            _reject_one_dimensional(directive_text, source=source)
            size_tokens = _directive_tokens(directive_text, "LUT_3D_SIZE")
    if size_tokens is None:
        raise ValueError(
            _actionable_error(
                why="a three-dimensional .cube file must declare LUT_3D_SIZE",
                what=f"{source} has no LUT_3D_SIZE line",
                how="add one LUT_3D_SIZE <N> declaration before the data rows",
            )
        )
    if len(size_tokens) != 1:
        raise ValueError(
            _actionable_error(
                why="LUT_3D_SIZE must declare one integer",
                what=f"received tokens {size_tokens!r}",
                how="write LUT_3D_SIZE <N> with N >= 2",
            )
        )
    try:
        size = int(size_tokens[0])
    except ValueError as error:
        raise ValueError(
            _actionable_error(
                why="LUT_3D_SIZE must be an integer",
                what=f"received {size_tokens[0]!r}",
                how="write LUT_3D_SIZE <N> with N >= 2",
            )
        ) from error
    if size < 2:
        raise ValueError(
            _actionable_error(
                why="LUT_3D_SIZE must be at least 2",
                what=f"received LUT_3D_SIZE {size}",
                how="provide a cubic grid with at least two samples per axis",
            )
        )

    if _DIRECTIVE_LINE.sub("", header_text).strip():
        raise ValueError(
            _actionable_error(
                why="every .cube data row must contain exactly three numeric values",
                what=f"{source} contains a malformed data row",
                how="write one red green blue output triplet per data row",
            )
        )

    row_widths = _data_row_widths(data_text)
    received_rows = int(np.count_nonzero(row_widths))
    expected_values = size * size * size * 3
    invalid_row_width = bool(np.any((row_widths != 0) & (row_widths != 3)))
    if invalid_row_width or received_rows != expected_values // 3:
        cleaned_data_text = _DIRECTIVE_LINE.sub("", data_text)
        if cleaned_data_text != data_text:
            data_text = cleaned_data_text
            directive_text = without_comments
            _reject_one_dimensional(directive_text, source=source)
            _directive_tokens(directive_text, "LUT_3D_SIZE")
            row_widths = _data_row_widths(data_text)
            received_rows = int(np.count_nonzero(row_widths))
            invalid_row_width = bool(np.any((row_widths != 0) & (row_widths != 3)))

    domain_min = _parse_domain(directive_text, "DOMAIN_MIN", _DEFAULT_DOMAIN_MIN)
    domain_max = _parse_domain(directive_text, "DOMAIN_MAX", _DEFAULT_DOMAIN_MAX)
    if invalid_row_width:
        raise ValueError(
            _actionable_error(
                why="every .cube data row must contain exactly three numeric values",
                what=f"{source} contains a malformed data row",
                how="write one red green blue output triplet per data row",
            )
        )
    if received_rows != expected_values // 3:
        raise ValueError(
            _actionable_error(
                why="the .cube data row count must exactly match LUT_3D_SIZE cubed",
                what=f"expected {expected_values // 3} rows, received {received_rows}",
                how="add or remove complete RGB data rows to match the declared cube size",
            )
        )
    try:
        values = np.fromstring(data_text, dtype=np.float32, sep=" ")
    except ValueError as error:
        raise ValueError(
            _actionable_error(
                why=".cube data rows must contain numeric values",
                what=f"{source} contains a non-numeric data token",
                how="replace the invalid token with a decimal RGB value",
            )
        ) from error
    if values.size != expected_values:
        raise ValueError(
            _actionable_error(
                why=".cube data rows must contain numeric values",
                what=f"{source} contains a non-numeric data token",
                how="replace the invalid token with a decimal RGB value",
            )
        )

    red_fastest = values.reshape(size, size, size, 3)
    rgb_indexed = np.ascontiguousarray(red_fastest.transpose(2, 1, 0, 3))
    return rgb_indexed, domain_min, domain_max


def read_lut(path: str | Path) -> Lut:
    """Read one .cube 3D LUT into GPU memory without caching.

    The file's red-fastest data order is normalized to ``Lut.data[R, G, B]``.
    Parsing performs one bulk token conversion followed by one host-to-device
    transfer. ``TITLE`` metadata and ``#`` comments do not affect the result.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            _actionable_error(
                why="the LUT file does not exist",
                what=str(file_path),
                how="provide an existing .cube file path",
            )
        )
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(
            _actionable_error(
                why="the LUT file could not be read as UTF-8 text",
                what=str(file_path),
                how="provide a readable text .cube file",
            )
        ) from error
    data, domain_min, domain_max = _parse_cube(text, source=str(file_path))
    packed = np.empty((*data.shape[:3], 4), dtype=np.float32)
    packed[..., :3] = data
    packed[..., 3] = np.float32(0.0)
    device_packed = cp.asarray(packed)
    return Lut(data=device_packed[..., :3], domain_min=domain_min, domain_max=domain_max)
