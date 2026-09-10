"""Independent hand-built ICC profiles and raster carriers for specification tests."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable, Mapping

import numpy as np

D50 = np.asarray((0.9642, 1.0, 0.8249), dtype=np.float64)
BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)
COLORSPACES = {
    "sRGB": (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290)),
    "Rec.2020": (((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), (0.3127, 0.3290)),
    "P3-D65": (((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), (0.3127, 0.3290)),
    "S-Gamut": (((0.730, 0.280), (0.140, 0.855), (0.100, -0.050)), (0.3127, 0.3290)),
    "Adobe-RGB": (((0.6400, 0.3300), (0.2100, 0.7100), (0.1500, 0.0600)), (0.3127, 0.3290)),
    "ProPhoto-RGB": (((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)), (0.3457, 0.3585)),
}


def xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def rgb_to_xyz(
    primaries: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    white: tuple[float, float],
) -> np.ndarray:
    unscaled = np.asarray(
        (tuple(x / y for x, y in primaries), (1.0, 1.0, 1.0), tuple((1.0 - x - y) / y for x, y in primaries)),
        dtype=np.float64,
    )
    return unscaled @ np.diag(np.linalg.solve(unscaled, xy_to_xyz(white)))


def bradford_xyz(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_cones = BRADFORD @ source
    target_cones = BRADFORD @ target
    return np.linalg.inv(BRADFORD) @ np.diag(target_cones / source_cones) @ BRADFORD


def _fixed(value: float) -> bytes:
    return struct.pack(">i", round(value * 65536.0))


def _fixed_values(values: Iterable[float]) -> bytes:
    return b"".join(_fixed(float(value)) for value in values)


def xyz_tag(values: Iterable[float]) -> bytes:
    return b"XYZ \x00\x00\x00\x00" + _fixed_values(values)


def chad_tag(matrix: np.ndarray) -> bytes:
    return b"sf32\x00\x00\x00\x00" + _fixed_values(np.asarray(matrix, dtype=np.float64).reshape(-1))


def curv_tag(*, exponent: float | None = None, samples: Iterable[float] | None = None) -> bytes:
    if exponent is not None and samples is not None:
        raise ValueError("choose exponent or samples")
    if exponent is not None:
        return b"curv\x00\x00\x00\x00" + struct.pack(">IH", 1, round(exponent * 256.0))
    values = () if samples is None else tuple(samples)
    encoded = tuple(max(0, min(65535, round(value * 65535.0))) for value in values)
    return b"curv\x00\x00\x00\x00" + struct.pack(">I", len(encoded)) + struct.pack(f">{len(encoded)}H", *encoded)


def para_tag(function: int, parameters: Iterable[float]) -> bytes:
    values = tuple(parameters)
    return b"para\x00\x00\x00\x00" + struct.pack(">HH", function, 0) + _fixed_values(values)


def trc_payload(kind: str) -> bytes:
    values = {
        "linear": curv_tag(),
        "Gamma-1.8": curv_tag(exponent=1.8),
        "Gamma-2.2": para_tag(0, (2.2,)),
        "Gamma-2.4": para_tag(0, (2.4,)),
        "Adobe-RGB": curv_tag(exponent=563.0 / 256.0),
        "ProPhoto-RGB": para_tag(3, (1.8, 1.0, 0.0, 1.0 / 16.0, 1.0 / 32.0)),
        "sRGB": para_tag(3, (2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045)),
        "Rec.709": para_tag(3, (1.0 / 0.45, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081)),
    }
    return values[kind]


def icc_profile(
    *,
    colorspace: str = "sRGB",
    gamma: str = "sRGB",
    version: int = 4,
    profile_class: bytes = b"mntr",
    data_space: bytes = b"RGB ",
    pcs: bytes = b"XYZ ",
    signature: bytes = b"acsp",
    with_chad: bool = True,
    trcs: tuple[bytes, bytes, bytes] | None = None,
    extra_tags: Mapping[bytes, bytes] | None = None,
    omit_tags: Iterable[bytes] = (),
    colorant_scale: float = 1.0,
) -> bytes:
    """Build a minimal v2/v4 matrix/TRC profile without using a profile writer."""
    primaries, white_xy = COLORSPACES[colorspace]
    source_matrix = rgb_to_xyz(primaries, white_xy) * colorant_scale
    source_white = xy_to_xyz(white_xy)
    tags: dict[bytes, bytes] = {}
    if with_chad:
        adaptation = bradford_xyz(source_white, D50)
        adaptation = np.round(adaptation * 65536.0) / 65536.0
        pcs_matrix = adaptation @ source_matrix
        tags[b"chad"] = chad_tag(adaptation)
        white_tag = D50
    elif version == 2:
        pcs_matrix = bradford_xyz(source_white, D50) @ source_matrix
        white_tag = source_white
    else:
        pcs_matrix = source_matrix
        white_tag = source_white
    for name, column in zip((b"rXYZ", b"gXYZ", b"bXYZ"), pcs_matrix.T, strict=True):
        tags[name] = xyz_tag(column)
    tags[b"wtpt"] = xyz_tag(white_tag)
    curve_payloads = trcs or (trc_payload(gamma),) * 3
    for name, payload in zip((b"rTRC", b"gTRC", b"bTRC"), curve_payloads, strict=True):
        tags[name] = payload
    tags.update(extra_tags or {})
    for name in omit_tags:
        tags.pop(name, None)

    tag_items = tuple(tags.items())
    table_end = 128 + 4 + 12 * len(tag_items)
    data_offset = (table_end + 3) & ~3
    payload_blob = bytearray(data_offset - table_end)
    records: list[tuple[bytes, int, int]] = []
    shared: dict[bytes, tuple[int, int]] = {}
    for name, payload in tag_items:
        location = shared.get(payload)
        if location is None:
            offset = table_end + len(payload_blob)
            size = len(payload)
            payload_blob.extend(payload)
            payload_blob.extend(b"\x00" * (-len(payload_blob) % 4))
            location = (offset, size)
            shared[payload] = location
        records.append((name, *location))

    profile = bytearray(128)
    profile[8] = version
    profile[12:16] = profile_class
    profile[16:20] = data_space
    profile[20:24] = pcs
    profile[36:40] = signature
    profile[68:80] = _fixed_values(D50)
    profile.extend(struct.pack(">I", len(records)))
    for name, offset, size in records:
        profile.extend(struct.pack(">4sII", name, offset, size))
    profile.extend(payload_blob)
    struct.pack_into(">I", profile, 0, len(profile))
    return bytes(profile)


def mutate_tag_record(profile: bytes, signature: bytes, *, offset_delta: int = 0, size_delta: int = 0) -> bytes:
    result = bytearray(profile)
    count = struct.unpack_from(">I", result, 128)[0]
    for index in range(count):
        record = 132 + 12 * index
        if result[record : record + 4] == signature:
            offset, size = struct.unpack_from(">II", result, record + 4)
            struct.pack_into(">II", result, record + 4, offset + offset_delta, size + size_delta)
            return bytes(result)
    raise KeyError(signature)


def duplicate_tag_record(profile: bytes, signature: bytes) -> bytes:
    result = bytearray(profile)
    count = struct.unpack_from(">I", result, 128)[0]
    record = next(
        result[132 + 12 * index : 144 + 12 * index]
        for index in range(count)
        if result[132 + 12 * index : 136 + 12 * index] == signature
    )
    result[132 + 12 * count : 132 + 12 * count] = record
    struct.pack_into(">I", result, 128, count + 1)
    struct.pack_into(">I", result, 0, len(result))
    return bytes(result)


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def stored_zlib_stream(*, output_size: int, stream_size: int) -> bytes:
    """Build an exact-size zlib stream from stored blocks without using production decompression."""
    block_count, remainder = divmod(stream_size - 6 - output_size, 5)
    if remainder or block_count <= 0 or output_size > 65535 * block_count:
        raise ValueError("requested sizes cannot be represented by stored deflate blocks")
    output = b"\x00" * output_size
    result = bytearray(b"\x78\x01")
    cursor = 0
    for index in range(block_count):
        remaining_blocks = block_count - index
        length = min(65535, output_size - cursor) if remaining_blocks > 1 else output_size - cursor
        final = index == block_count - 1
        result.append(1 if final else 0)
        result.extend(struct.pack("<HH", length, 0xFFFF - length))
        result.extend(output[cursor : cursor + length])
        cursor += length
    result.extend(struct.pack(">I", zlib.adler32(output) & 0xFFFFFFFF))
    assert cursor == output_size and len(result) == stream_size
    return bytes(result)


def png_header(
    *,
    profile: bytes | None = None,
    name: bytes = b"pixtreme",
    method: int = 0,
    compressed: bytes | None = None,
    duplicate_iccp: bool = False,
    color_type: int = 2,
    cicp: bytes | None = None,
    srgb: bool = False,
    gama: int | None = None,
) -> bytes:
    chunks = [png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, color_type, 0, 0, 0))]
    if profile is not None or compressed is not None:
        stream = zlib.compress(profile or b"") if compressed is None else compressed
        carrier = png_chunk(b"iCCP", name + b"\x00" + bytes((method,)) + stream)
        chunks.append(carrier)
        if duplicate_iccp:
            chunks.append(carrier)
    if srgb:
        chunks.append(png_chunk(b"sRGB", b"\x00"))
    if gama is not None:
        chunks.append(png_chunk(b"gAMA", struct.pack(">I", gama)))
    if cicp is not None:
        chunks.append(png_chunk(b"cICP", cicp))
    chunks.append(png_chunk(b"IEND", b""))
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def jpeg_header(segments: Iterable[tuple[int, int, bytes]] = (), *, components: int = 3) -> bytes:
    markers = []
    for sequence, count, payload in segments:
        app2 = b"ICC_PROFILE\x00" + bytes((sequence, count)) + payload
        markers.append(b"\xff\xe2" + struct.pack(">H", len(app2) + 2) + app2)
    sof = bytes((8,)) + struct.pack(">HHB", 1, 2, components) + b"\x00" * (3 * components)
    return b"\xff\xd8" + b"".join(markers) + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof + b"\xff\xd9"


def tiff_header(
    *,
    profiles: Iterable[bytes] = (),
    profile_type: int = 7,
    photometric: int = 2,
    samples: int = 3,
) -> bytes:
    profile_values = tuple(profiles)
    base_entries = [(256, 4, 1, struct.pack("<I", 2)), (257, 4, 1, struct.pack("<I", 1))]
    base_entries.extend(
        (
            (258, 3, 1, struct.pack("<H", 8) + b"\x00\x00"),
            (262, 3, 1, struct.pack("<H", photometric) + b"\x00\x00"),
            (277, 3, 1, struct.pack("<H", samples) + b"\x00\x00"),
        )
    )
    entry_count = len(base_entries) + len(profile_values)
    data_offset = 8 + 2 + 12 * entry_count + 4
    entries = list(base_entries)
    data = bytearray()
    for profile in profile_values:
        entries.append((34675, profile_type, len(profile), struct.pack("<I", data_offset + len(data))))
        data.extend(profile)
    entries.sort(key=lambda item: item[0])
    table = b"".join(struct.pack("<HHI", tag, kind, count) + value for tag, kind, count, value in entries)
    return b"II" + struct.pack("<HIH", 42, 8, entry_count) + table + b"\x00\x00\x00\x00" + bytes(data)


def webp_chunk(kind: bytes, payload: bytes, *, declared_size: int | None = None) -> bytes:
    size = len(payload) if declared_size is None else declared_size
    return kind + struct.pack("<I", size) + payload + (b"\x00" if len(payload) & 1 else b"")


def webp_header(
    *,
    profiles: Iterable[bytes] = (),
    icc_flag: bool = True,
    iccp_before_vp8x: bool = False,
    iccp_after_image: bool = False,
) -> bytes:
    profile_chunks = [webp_chunk(b"ICCP", profile) for profile in profiles]
    flags = 0x20 if icc_flag else 0
    vp8x = webp_chunk(b"VP8X", bytes((flags, 0, 0, 0, 1, 0, 0, 0, 0, 0)))
    vp8 = webp_chunk(b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a\x02\x00\x01\x00")
    if iccp_before_vp8x:
        chunks = [*profile_chunks, vp8x, vp8]
    elif iccp_after_image:
        chunks = [vp8x, vp8, *profile_chunks]
    else:
        chunks = [vp8x, *profile_chunks, vp8]
    body = b"WEBP" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body
