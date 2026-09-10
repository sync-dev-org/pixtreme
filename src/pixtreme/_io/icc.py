"""Pure-CPU parsing and vocabulary mapping for embedded RGB ICC profiles."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS
from pixtreme._io.models import _ImageColorInfo

_MAX_PROFILE_SIZE = 16_777_216
_MAX_PNG_COMPRESSED_SIZE = 17_825_792
_MAX_JPEG_SEGMENT_DATA = 65_519
_MAX_JPEG_CARRIER_SIZE = _MAX_JPEG_SEGMENT_DATA * 255
_D50 = np.asarray((0.9642, 1.0, 0.8249), dtype=np.float64)
_BRADFORD = np.asarray(
    (
        (0.8951, 0.2664, -0.1614),
        (-0.7502, 1.7135, 0.0367),
        (0.0389, -0.0685, 1.0296),
    ),
    dtype=np.float64,
)
_REQUIRED_TAGS = frozenset((b"rXYZ", b"gXYZ", b"bXYZ", b"wtpt", b"rTRC", b"gTRC", b"bTRC"))
_HYBRID_TAGS = frozenset(
    (
        b"A2B0",
        b"A2B1",
        b"A2B2",
        b"B2A0",
        b"B2A1",
        b"B2A2",
        b"D2B0",
        b"D2B1",
        b"D2B2",
        b"D2B3",
        b"B2D0",
        b"B2D1",
        b"B2D2",
        b"B2D3",
    )
)
_CURVE_GRID = np.unique(
    np.concatenate(
        (
            np.linspace(0.0, 1.0, 4096, dtype=np.float64),
            np.asarray((0.04045, 0.081, 1.0 / 32.0), dtype=np.float64),
        )
    )
)
_PURE_EXPONENTS = {
    "linear": 1.0,
    "Gamma-1.8": 1.8,
    "Gamma-2.2": 2.2,
    "Gamma-2.4": 2.4,
    "Gamma-2.5": 2.5,
    "Gamma-2.6": 2.6,
    "Adobe-RGB": 563.0 / 256.0,
}


@dataclass(frozen=True, slots=True)
class _IccCarrier:
    present: bool
    profile: bytes | None


@dataclass(frozen=True, slots=True)
class _Curve:
    kind: str
    function: int | None = None
    values: tuple[float, ...] | NDArray[np.float64] = ()


def _invalid_icc_color(raw: dict[str, object] | None = None) -> _ImageColorInfo:
    return _ImageColorInfo(raw={} if raw is None else raw, colorspace=None, gamma=None, mappable=False)


def _icc_carrier_color(
    carrier: _IccCarrier,
    *,
    compatible: bool,
    raw: dict[str, object] | None = None,
) -> _ImageColorInfo:
    retained = {} if raw is None else raw
    if not carrier.present:
        return _ImageColorInfo(raw=retained, colorspace=None, gamma=None, mappable=None)
    if carrier.profile is None:
        return _invalid_icc_color(retained)
    return _icc_color_info(carrier.profile, compatible=compatible, raw=retained)


def _icc_color_info(
    profile: bytes,
    *,
    compatible: bool,
    raw: dict[str, object] | None = None,
) -> _ImageColorInfo:
    """Map one fully reconstructed profile while retaining its exact bytes on every profile-level failure."""
    retained = {} if raw is None else dict(raw)
    retained["ICC"] = profile
    try:
        colorspace, gamma = _map_profile(profile)
    except Exception:
        return _invalid_icc_color(retained)
    if not compatible:
        return _invalid_icc_color(retained)
    return _ImageColorInfo(
        raw=retained,
        colorspace=colorspace,
        gamma=gamma,
        mappable=colorspace is not None and gamma is not None,
    )


def _png_icc_carrier(payloads: list[bytes]) -> _IccCarrier:
    if not payloads:
        return _IccCarrier(present=False, profile=None)
    if len(payloads) != 1:
        return _IccCarrier(present=True, profile=None)
    payload = payloads[0]
    separator = payload.find(b"\x00")
    if separator < 1 or separator > 79 or separator + 2 > len(payload):
        return _IccCarrier(present=True, profile=None)
    name = payload[:separator]
    if name[0] == 0x20 or name[-1] == 0x20 or b"  " in name:
        return _IccCarrier(present=True, profile=None)
    if any(not (32 <= value <= 126 or 161 <= value <= 255) for value in name):
        return _IccCarrier(present=True, profile=None)
    if payload[separator + 1] != 0:
        return _IccCarrier(present=True, profile=None)
    compressed = payload[separator + 2 :]
    if len(compressed) > _MAX_PNG_COMPRESSED_SIZE:
        return _IccCarrier(present=True, profile=None)
    profile = _bounded_zlib_decompress(compressed)
    return _IccCarrier(present=True, profile=profile)


def _bounded_zlib_decompress(compressed: bytes) -> bytes | None:
    try:
        decompressor = zlib.decompressobj()
        profile = decompressor.decompress(compressed, _MAX_PROFILE_SIZE + 1)
        if len(profile) > _MAX_PROFILE_SIZE:
            return None
        while not decompressor.eof and len(profile) <= _MAX_PROFILE_SIZE:
            tail = decompressor.unconsumed_tail
            more = decompressor.decompress(tail, _MAX_PROFILE_SIZE + 1 - len(profile))
            profile += more
            if len(profile) > _MAX_PROFILE_SIZE:
                return None
            if not tail and not more:
                break
        if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
            return None
        if decompressor.flush(1):
            return None
        return profile
    except zlib.error:
        return None


def _map_profile(profile: bytes) -> tuple[str | None, str | None]:
    if len(profile) < 132 or len(profile) > _MAX_PROFILE_SIZE:
        raise ValueError("profile size is outside the supported bounds")
    declared_size = struct.unpack_from(">I", profile, 0)[0]
    version = profile[8]
    if (
        declared_size != len(profile)
        or version not in (2, 4)
        or profile[12:16] not in (b"mntr", b"scnr", b"spac")
        or profile[16:20] != b"RGB "
        or profile[20:24] != b"XYZ "
        or profile[36:40] != b"acsp"
    ):
        raise ValueError("unsupported ICC header")
    tag_count = struct.unpack_from(">I", profile, 128)[0]
    table_end = 132 + 12 * tag_count
    if table_end > len(profile):
        raise ValueError("truncated ICC tag table")

    tags: dict[bytes, bytes] = {}
    ranges: list[tuple[int, int]] = []
    for index in range(tag_count):
        record = 132 + 12 * index
        signature, offset, size = struct.unpack_from(">4sII", profile, record)
        end = offset + size
        if signature in tags or offset % 4 or offset < table_end or end < offset or end > len(profile):
            raise ValueError("invalid ICC tag record")
        tags[signature] = profile[offset:end]
        ranges.append((offset, end))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:], strict=False):
        if current[0] < previous[1] and current != previous:
            raise ValueError("partially overlapping ICC tag data")
    if not _REQUIRED_TAGS.issubset(tags) or _HYBRID_TAGS.intersection(tags):
        raise ValueError("profile is not a complete matrix/TRC profile")

    colorspace = _map_colorspace(tags, version=version)
    gamma = _map_gamma(tags, colorspace=colorspace)
    return colorspace, gamma


def _fixed_values(payload: bytes, *, offset: int, count: int) -> NDArray[np.float64]:
    if len(payload) != offset + 4 * count:
        raise ValueError("fixed-point tag payload has the wrong count")
    values = np.asarray(struct.unpack_from(f">{count}i", payload, offset), dtype=np.float64) / 65536.0
    if not np.all(np.isfinite(values)):
        raise ValueError("fixed-point tag payload is not finite")
    return values


def _xyz_tag(payload: bytes) -> NDArray[np.float64]:
    if payload[:8] != b"XYZ \x00\x00\x00\x00":
        raise ValueError("XYZ tag has the wrong type")
    return _fixed_values(payload, offset=8, count=3)


def _chad_tag(payload: bytes) -> NDArray[np.float64]:
    if payload[:8] != b"sf32\x00\x00\x00\x00":
        raise ValueError("chad tag has the wrong type")
    return _fixed_values(payload, offset=8, count=9).reshape(3, 3)


def _map_colorspace(tags: dict[bytes, bytes], *, version: int) -> str | None:
    try:
        pcs_matrix = np.column_stack(tuple(_xyz_tag(tags[name]) for name in (b"rXYZ", b"gXYZ", b"bXYZ")))
        white_tag = _xyz_tag(tags[b"wtpt"])
        if b"chad" in tags:
            inverse = np.linalg.inv(_chad_tag(tags[b"chad"]))
            source_matrix = inverse @ pcs_matrix
            source_white = inverse @ _D50
        elif version == 2:
            source_white = white_tag
            adaptation = _bradford_adaptation(source_white, _D50)
            source_matrix = np.linalg.inv(adaptation) @ pcs_matrix
        else:
            if not np.allclose(white_tag, _D50, rtol=0.0, atol=5e-4):
                return None
            source_matrix = pcs_matrix
            source_white = _D50
        if not np.all(np.isfinite(source_matrix)) or not np.all(np.isfinite(source_white)):
            return None
        if not np.allclose(source_matrix.sum(axis=1), source_white, rtol=0.0, atol=5e-4):
            return None
        if not np.isclose(source_white[1], 1.0, rtol=0.0, atol=5e-4):
            return None
        points = tuple(_xyz_to_xy(source_matrix[:, index]) for index in range(3)) + (_xyz_to_xy(source_white),)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None

    candidates = []
    observed = np.asarray(points, dtype=np.float64)
    for token, (primaries, white) in _COLORSPACE_DEFINITIONS.items():
        expected = np.asarray((*primaries, white), dtype=np.float64)
        if np.allclose(observed, expected, rtol=0.0, atol=1e-4):
            candidates.append(token)
    candidate_set = frozenset(candidates)
    if candidate_set == frozenset(("sRGB", "Rec.709")):
        return "sRGB"
    if candidate_set == frozenset(("S-Gamut", "S-Gamut3")):
        return "S-Gamut"
    return candidates[0] if len(candidates) == 1 else None


def _bradford_adaptation(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    source_cones = _BRADFORD @ source
    target_cones = _BRADFORD @ target
    if np.any(source_cones == 0.0) or not np.all(np.isfinite(source_cones)):
        raise ValueError("source white cannot be adapted")
    return np.asarray(np.linalg.inv(_BRADFORD) @ np.diag(target_cones / source_cones) @ _BRADFORD, dtype=np.float64)


def _xyz_to_xy(value: NDArray[np.float64]) -> tuple[float, float]:
    denominator = float(np.sum(value))
    if denominator == 0.0 or not np.isfinite(denominator):
        raise ValueError("XYZ value cannot be normalized")
    xy = (float(value[0] / denominator), float(value[1] / denominator))
    if not all(np.isfinite(component) for component in xy):
        raise ValueError("xy value is not finite")
    return xy


def _curve(payload: bytes) -> _Curve:
    if len(payload) < 12 or payload[4:8] != b"\x00\x00\x00\x00":
        raise ValueError("TRC tag is truncated or has nonzero reserved bytes")
    if payload[:4] == b"curv":
        count = struct.unpack_from(">I", payload, 8)[0]
        if len(payload) != 12 + 2 * count:
            raise ValueError("curv count does not match its payload")
        if count == 0:
            return _Curve("identity")
        if count == 1:
            return _Curve("exponent", values=(struct.unpack_from(">H", payload, 12)[0] / 256.0,))
        encoded = np.frombuffer(payload, dtype=">u2", count=count, offset=12)
        return _Curve("sampled", values=np.asarray(encoded, dtype=np.float64) / 65535.0)
    if payload[:4] == b"para":
        function, reserved = struct.unpack_from(">HH", payload, 8)
        parameter_counts = (1, 3, 4, 5, 7)
        if function >= len(parameter_counts) or reserved:
            raise ValueError("unsupported para function")
        count = parameter_counts[function]
        values = _fixed_values(payload, offset=12, count=count)
        return _Curve("parametric", function=function, values=tuple(float(value) for value in values))
    raise ValueError("unsupported TRC type")


def _map_gamma(tags: dict[bytes, bytes], *, colorspace: str | None) -> str | None:
    try:
        candidate_sets = [set(_curve_candidates(_curve(tags[name]))) for name in (b"rTRC", b"gTRC", b"bTRC")]
    except (ValueError, struct.error, OverflowError):
        return None
    common = set.intersection(*candidate_sets)
    if common == {"Gamma-2.2", "Adobe-RGB"}:
        return "Adobe-RGB" if colorspace == "Adobe-RGB" else "Gamma-2.2"
    return next(iter(common)) if len(common) == 1 else None


def _curve_candidates(curve: _Curve) -> tuple[str, ...]:
    if curve.kind == "identity":
        return ("linear",)
    if curve.kind == "exponent":
        exponent = curve.values[0]
        return tuple(token for token, target in _PURE_EXPONENTS.items() if abs(exponent - target) <= 1.0 / 512.0)

    points = _CURVE_GRID
    if curve.kind == "sampled":
        samples = np.asarray(curve.values, dtype=np.float64)
        knots = np.linspace(0.0, 1.0, samples.size, dtype=np.float64)
        points = np.unique(np.concatenate((points, knots)))
        realized = np.interp(points, knots, samples)
    elif curve.kind == "parametric":
        realized = _evaluate_parametric(curve, points)
    else:
        return ()
    if not np.all(np.isfinite(realized)):
        return ()
    candidates = []
    for token in (
        "linear",
        "sRGB",
        "Rec.709",
        "Gamma-1.8",
        "Gamma-2.2",
        "Gamma-2.4",
        "Gamma-2.5",
        "Gamma-2.6",
        "Adobe-RGB",
        "ProPhoto-RGB",
    ):
        target = _target_curve(token, points)
        if np.max(np.abs(realized - target)) <= 5e-5:
            candidates.append(token)
    return tuple(candidates)


def _evaluate_parametric(curve: _Curve, x: NDArray[np.float64]) -> NDArray[np.float64]:
    assert curve.function is not None
    with np.errstate(all="ignore"):
        if curve.function == 0:
            (g,) = curve.values
            return np.asarray(x**g, dtype=np.float64)
        if curve.function == 1:
            g, a, b = curve.values
            if a == 0.0:
                return np.full_like(x, np.nan)
            return np.where(x >= -b / a, (a * x + b) ** g, 0.0)
        if curve.function == 2:
            g, a, b, c = curve.values
            if a == 0.0:
                return np.full_like(x, np.nan)
            return np.where(x >= -b / a, (a * x + b) ** g + c, c)
        if curve.function == 3:
            g, a, b, c, d = curve.values
            return np.where(x >= d, (a * x + b) ** g, c * x)
        g, a, b, c, d, e, f = curve.values
        return np.where(x >= d, (a * x + b) ** g + e, c * x + f)


def _target_curve(token: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
    if token in _PURE_EXPONENTS:
        return np.asarray(x ** _PURE_EXPONENTS[token], dtype=np.float64)
    if token == "sRGB":
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    if token == "Rec.709":
        return np.where(x < 0.081, x / 4.5, ((x + 0.099) / 1.099) ** (1.0 / 0.45))
    if token == "ProPhoto-RGB":
        return np.where(x < 1.0 / 32.0, x / 16.0, x**1.8)
    raise ValueError(f"unknown ICC target curve {token}")
