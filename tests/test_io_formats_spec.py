"""Specification tests for the JPEG 2000, WebP, BMP, and PNM I/O expansion."""

from __future__ import annotations

import inspect
import io
import os
import struct
import subprocess
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from PIL import Image

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
_ACTIONABLE = r"why=.*what=.*how="


def _frame(dtype: type[np.generic], channels: str, *, height: int = 3, width: int = 5) -> px.core.Frame:
    channel_count = len(channels)
    if np.issubdtype(dtype, np.floating):
        values = np.linspace(0.0, 1.0, height * width * channel_count, dtype=dtype).reshape(
            height, width, channel_count
        )
    else:
        maximum = np.iinfo(dtype).max
        values = np.arange(height * width * channel_count, dtype=np.uint64).reshape(height, width, channel_count)
        values = ((values * 997 + 31) % (maximum + 1)).astype(dtype)
    return px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels=channels)


def _assert_frame_equal(actual: px.core.Frame, expected: px.core.Frame) -> None:
    assert (actual.colorspace, actual.gamma, actual.channels, actual.dtype) == (
        expected.colorspace,
        expected.gamma,
        expected.channels,
        expected.dtype,
    )
    cp.testing.assert_array_equal(actual.data, expected.data)


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def _minimal_jp2(width: int, height: int, components: int, bit_depth: int) -> bytes:
    signature = _box(b"jP  ", b"\r\n\x87\n")
    ihdr = struct.pack(">IIHBBBB", height, width, components, bit_depth - 1, 7, 0, 0)
    return signature + _box(b"jp2h", _box(b"ihdr", ihdr))


def _minimal_j2k(width: int, height: int, components: int, bit_depth: int) -> bytes:
    component_fields = bytes((bit_depth - 1, 1, 1)) * components
    siz_payload = (
        struct.pack(
            ">HIIIIIIIIH",
            0,
            width,
            height,
            0,
            0,
            width,
            height,
            0,
            0,
            components,
        )
        + component_fields
    )
    return b"\xff\x4f\xff\x51" + struct.pack(">H", len(siz_payload) + 2) + siz_payload


def _riff_webp(chunk_type: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) % 2 else b""
    body = b"WEBP" + chunk_type + struct.pack("<I", len(payload)) + payload + padding
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _minimal_vp8(width: int, height: int) -> bytes:
    payload = b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height)
    return _riff_webp(b"VP8 ", payload)


def _minimal_vp8l(width: int, height: int) -> bytes:
    packed = (width - 1) | ((height - 1) << 14)
    return _riff_webp(b"VP8L", b"\x2f" + packed.to_bytes(4, "little"))


def _minimal_vp8x(width: int, height: int) -> bytes:
    payload = b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return _riff_webp(b"VP8X", payload)


def _minimal_bmp(width: int, height: int, bits_per_pixel: int) -> bytes:
    row_size = ((width * bits_per_pixel + 31) // 32) * 4
    image_size = row_size * height
    pixel_offset = 54
    file_header = struct.pack("<2sIHHI", b"BM", pixel_offset + image_size, 0, 0, pixel_offset)
    dib_header = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,
        1,
        bits_per_pixel,
        0,
        image_size,
        0,
        0,
        0,
        0,
    )
    return file_header + dib_header + bytes(image_size)


def _pnm_payload(magic: str, values: np.ndarray) -> bytes:
    height, width = values.shape[:2]
    maxval = np.iinfo(values.dtype).max
    header = f"{magic}\n# deterministic fixture\n{width} {height}\n{maxval}\n".encode()
    if magic in ("P2", "P3"):
        return header + " ".join(str(int(value)) for value in values.reshape(-1)).encode() + b"\n"
    samples = values.astype(">u2", copy=False).tobytes() if values.dtype == np.uint16 else values.tobytes()
    return header + samples


def test_new_format_public_signatures_are_exact() -> None:
    """v1-io-formats acceptance 1; v1-exr-runtime-independence acceptance 1 and 3:
    file output exposes only the fixed selectors, including the trailing EXR dtype selector.
    """
    read = inspect.signature(px.io.read_image)
    decode = inspect.signature(px.io.decode_image)
    write = inspect.signature(px.io.write_image)
    encode = inspect.signature(px.io.encode_image)

    assert tuple(read.parameters) == ("path", "channels", "unchanged", "colorspace", "gamma")
    assert tuple(decode.parameters) == ("data", "channels", "unchanged", "colorspace", "gamma")
    assert tuple(write.parameters) == (
        "path",
        "frame",
        "quality",
        "compression",
        "compression_level",
        "lossless",
        "dwa_level",
        "bit_depth",
        "dtype",
    )
    assert tuple(encode.parameters) == (
        "frame",
        "format",
        "quality",
        "compression",
        "compression_level",
        "lossless",
    )
    assert write.parameters["lossless"].kind is inspect.Parameter.KEYWORD_ONLY
    assert encode.parameters["lossless"].kind is inspect.Parameter.KEYWORD_ONLY
    assert write.parameters["lossless"].default is None
    assert write.parameters["bit_depth"].default is None
    assert encode.parameters["lossless"].default is None


@pytest.mark.parametrize("token", ("jpeg2000", "webp", "bmp", "pnm"))
def test_encode_image_accepts_the_extended_closed_token_set(token: str) -> None:
    """v1-io-formats acceptance 3 / v1-write-dtype-convert acceptance 5: format tokens stay closed."""
    from pixtreme._io.common import _ENCODE_FORMAT_TOKENS

    assert _ENCODE_FORMAT_TOKENS == ("jpeg", "png", "tiff", "jpeg2000", "webp", "bmp", "pnm")
    frame = _frame(np.uint8, "RGB")
    assert isinstance(
        px.io.encode_image(frame, format=token, lossless=True if token in ("jpeg2000", "webp") else None), bytes
    )
    assert isinstance(
        px.io.encode_image(frame, format=token.upper(), lossless=True if token in ("jpeg2000", "webp") else None),
        bytes,
    )


@pytest.mark.parametrize(
    ("suffix", "format_name"),
    (
        (".jp2", "JPEG2000"),
        (".J2K", "JPEG2000"),
        (".j2c", "JPEG2000"),
        (".WEBP", "WEBP"),
        (".bmp", "BMP"),
        (".PNM", "PNM"),
        (".ppm", "PNM"),
        (".PGM", "PNM"),
    ),
)
def test_file_extensions_are_case_insensitive_and_report_the_new_formats(
    tmp_path: Path, suffix: str, format_name: str
) -> None:
    """v1-io-formats acceptance 2: all new file extensions select the intended format case-insensitively."""
    channels = "Y" if suffix.lower() == ".pgm" else "RGB"
    frame = _frame(np.uint8, channels)
    path = tmp_path / f"round-trip{suffix}"

    assert px.io.write_image(path, frame, lossless=True if format_name in ("JPEG2000", "WEBP") else None) is None
    assert px.io.read_header(path).format == format_name


@pytest.mark.parametrize(
    ("suffix", "payload", "format_name", "width", "height", "channels", "dtype"),
    (
        (".jp2", _minimal_jp2(7, 5, 3, 8), "JPEG2000", 7, 5, ("R", "G", "B"), "uint8"),
        (".j2k", _minimal_j2k(9, 4, 1, 16), "JPEG2000", 9, 4, ("Y",), "uint16"),
        (".webp", _minimal_vp8(8, 6), "WEBP", 8, 6, ("R", "G", "B"), "uint8"),
        (".webp", _minimal_vp8l(11, 3), "WEBP", 11, 3, ("R", "G", "B"), "uint8"),
        (".webp", _minimal_vp8x(13, 2), "WEBP", 13, 2, ("R", "G", "B"), "uint8"),
        (".bmp", _minimal_bmp(10, 3, 24), "BMP", 10, 3, ("R", "G", "B"), "uint8"),
        (".pnm", b"P5\n# c\n12 2\n65535\n" + bytes(12 * 2 * 2), "PNM", 12, 2, ("Y",), "uint16"),
    ),
)
def test_read_header_parses_independent_minimal_new_format_headers(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
    format_name: str,
    width: int,
    height: int,
    channels: tuple[str, ...],
    dtype: str,
) -> None:
    """v1-io-formats acceptance 16-17: independent headers expose dimensions, labels, and storage dtype."""
    path = tmp_path / f"minimal{suffix}"
    path.write_bytes(payload)

    header = px.io.read_header(path)

    assert (header.format, header.width, header.height) == (format_name, width, height)
    assert header.parts[0].channels == dict.fromkeys(channels, dtype)
    assert header.color.raw == {}


def test_read_header_rejects_bmp_pixel_offset_beyond_the_declared_payload(tmp_path: Path) -> None:
    """v1-io-formats acceptance 17: BMP payload offsets stay within declared and available bytes."""
    payload = bytearray(_minimal_bmp(1, 1, 24)[:54])
    struct.pack_into("<I", payload, 2, len(payload))
    struct.pack_into("<I", payload, 10, 1000)
    path = tmp_path / "invalid-offset.bmp"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError, match=_ACTIONABLE) as error:
        px.io.read_header(path)

    assert "file_size=54" in str(error.value)
    assert "pixel_offset=1000" in str(error.value)


def test_read_header_stops_before_ascii_pnm_raster_while_decode_rejects_corruption(tmp_path: Path) -> None:
    """v1-io-formats acceptance 16-17: header probing skips ASCII samples that decode must validate."""
    payload = b"P2\n1 1\n255\nnot-a-decimal-sample\n"
    path = tmp_path / "corrupt-raster.pgm"
    path.write_bytes(payload)

    header = px.io.read_header(path)

    assert (header.format, header.width, header.height) == ("PNM", 1, 1)
    assert header.parts[0].channels == {"Y": "uint8"}
    with pytest.raises(RuntimeError, match=_ACTIONABLE):
        px.io.decode_image(payload)


@pytest.mark.parametrize(
    ("payload", "observed"),
    (
        (b"\x00\x00\x00\x0cjP  \r\n\x87\n", "JPEG2000"),
        (b"\xff\x4f\xff\x51\x00\x02", "JPEG2000"),
        (b"RIFF\x04\x00\x00\x00WEBP", "WEBP"),
        (b"BM" + bytes(12), "BMP"),
        (b"P5\n0 2\n255\n", "PNM"),
    ),
)
def test_decode_image_rejects_truncated_or_invalid_new_format_headers(payload: bytes, observed: str) -> None:
    """v1-io-formats acceptance 4 and 17: recognized corrupt payloads retain actionable parse context."""
    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.decode_image(payload)
    assert observed.lower() in str(error.value).lower()


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16))
@pytest.mark.parametrize("channels", ("Y", "RGB", "RGBA"))
@pytest.mark.parametrize("suffix", (".jp2", ".j2k"))
def test_jpeg2000_lossless_file_round_trip_preserves_every_supported_layout(
    tmp_path: Path, dtype: type[np.generic], channels: str, suffix: str
) -> None:
    """v1-io-formats acceptance 7-8 and 13: JP2/J2K preserve all uint depth and channel combinations."""
    frame = _frame(dtype, channels)
    path = tmp_path / f"lossless{suffix}"

    px.io.write_image(path, frame, lossless=True)
    decoded = px.io.read_image(path, unchanged=True, colorspace="ACEScg", gamma="linear")

    _assert_frame_equal(decoded, frame)
    if suffix == ".jp2":
        assert path.read_bytes().startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n")
    else:
        assert path.read_bytes().startswith(b"\xff\x4f\xff\x51")


@pytest.mark.parametrize(
    ("format_token", "suffix", "channels"),
    (("webp", ".webp", "RGB"), ("bmp", ".bmp", "Y"), ("bmp", ".bmp", "RGB")),
)
def test_webp_and_bmp_supported_layouts_round_trip_at_file_and_bytes_boundaries(
    tmp_path: Path, format_token: str, suffix: str, channels: str
) -> None:
    """v1-io-formats acceptance 5, 9, and 15: WebP/BMP file and bytes boundaries agree."""
    frame = _frame(np.uint8, channels)
    kwargs = {"lossless": True} if format_token == "webp" else {}
    path = tmp_path / f"round-trip{suffix}"

    payload = px.io.encode_image(frame, format=format_token, **kwargs)
    px.io.write_image(path, frame, **kwargs)
    from_bytes = px.io.decode_image(payload, unchanged=True, colorspace="ACEScg", gamma="linear")
    from_file = px.io.read_image(path, unchanged=True, colorspace="ACEScg", gamma="linear")

    _assert_frame_equal(from_bytes, frame)
    _assert_frame_equal(from_file, frame)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16))
@pytest.mark.parametrize(("magic", "channels"), (("P2", "Y"), ("P3", "RGB"), ("P5", "Y"), ("P6", "RGB")))
def test_pnm_ascii_and_binary_decode_preserve_samples_and_normalize(
    dtype: type[np.generic], magic: str, channels: str
) -> None:
    """v1-io-formats acceptance 4, 7, and 10: P2/P3/P5/P6 parse comments, depth, and samples exactly."""
    source = _frame(dtype, channels, height=2, width=4)
    values = px.io.to_array(
        source,
    ).get()
    payload = _pnm_payload(magic, values)

    unchanged = px.io.decode_image(payload, unchanged=True, colorspace="ACEScg", gamma="linear")
    normalized = px.io.decode_image(payload)

    _assert_frame_equal(unchanged, source)
    np.testing.assert_array_equal(
        px.io.to_array(
            normalized,
        ).get(),
        values.astype(np.float32) / np.float32(np.iinfo(dtype).max),
    )


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16))
@pytest.mark.parametrize(("channels", "magic", "suffix"), (("Y", b"P5", ".pgm"), ("RGB", b"P6", ".ppm")))
def test_pnm_encode_is_bit_exact_big_endian_and_extension_checked(
    tmp_path: Path, dtype: type[np.generic], channels: str, magic: bytes, suffix: str
) -> None:
    """v1-io-formats acceptance 10: P5/P6 preserve bits and uint16 samples use network byte order."""
    frame = _frame(dtype, channels)
    path = tmp_path / f"round-trip{suffix}"

    payload = px.io.encode_image(frame, format="pnm")
    px.io.write_image(path, frame)

    assert payload.startswith(magic)
    _assert_frame_equal(px.io.decode_image(payload, unchanged=True, colorspace="ACEScg", gamma="linear"), frame)
    _assert_frame_equal(px.io.read_image(path, unchanged=True, colorspace="ACEScg", gamma="linear"), frame)
    if dtype == np.uint16:
        header_end = payload.find(b"\n", payload.find(b"65535")) + 1
        assert (
            payload[header_end:]
            == px.io.to_array(
                frame,
            )
            .get()
            .astype(">u2", copy=False)
            .tobytes()
        )

    wrong_suffix = ".ppm" if suffix == ".pgm" else ".pgm"
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / f"wrong{wrong_suffix}", frame)


@pytest.mark.parametrize(
    ("format_token", "dtype", "channels"),
    (
        ("jpeg2000", np.uint8, "YA"),
        ("webp", np.uint8, "RGBA"),
        ("bmp", np.uint8, "RGBA"),
        ("pnm", np.uint8, "YA"),
    ),
)
def test_new_formats_reject_unsupported_channel_layouts_before_codec(
    format_token: str, dtype: type[np.generic], channels: str
) -> None:
    """v1-write-dtype-convert acceptance 5; v1-io-formats acceptance 11.

    Dtype conversion does not expand the closed channel-layout table.
    """
    frame = _frame(dtype, channels)
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format=format_token)


@pytest.mark.parametrize(
    ("format_token", "kwargs"),
    (
        ("jpeg2000", {"quality": 90}),
        ("bmp", {"quality": 90}),
        ("webp", {"quality": 0}),
        ("webp", {"quality": True}),
        ("webp", {"lossless": 1}),
        ("jpeg2000", {"lossless": "yes"}),
        ("webp", {"quality": 90, "lossless": True}),
        ("pnm", {"lossless": True}),
        ("bmp", {"compression_level": 4}),
    ),
)
def test_new_encode_parameters_are_typed_format_specific_and_non_conflicting(
    format_token: str, kwargs: dict[str, object]
) -> None:
    """v1-write-dtype-convert acceptance 5; v1-io-formats acceptance 12-14.

    Quality/lossless and legacy options retain fail-fast validation.
    """
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(_frame(np.uint8, "RGB"), format=format_token, **kwargs)  # type: ignore[arg-type]


def test_webp_quality_controls_payload_and_independent_mse() -> None:
    """v1-io-formats acceptance 12: quality 1/50/100 changes WebP payloads and improves independent MSE."""
    generator = np.random.default_rng(20260806)
    values = generator.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="sRGB", channels="RGB")

    payloads = [px.io.encode_image(frame, format="webp", quality=quality) for quality in (1, 50, 100)]
    decoded = [np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.float32) for payload in payloads]
    mse = [float(np.mean((result - values.astype(np.float32)) ** 2)) for result in decoded]

    assert len(set(payloads)) == 3
    assert mse[2] <= mse[0]


@pytest.mark.parametrize("format_token", ("jpeg2000", "webp"))
def test_lossless_true_is_exact_and_false_or_none_is_lossy(format_token: str) -> None:
    """v1-io-formats acceptance 13: explicit lossless is exact while default/False selects lossy coding."""
    generator = np.random.default_rng(20260807)
    values = generator.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="sRGB", channels="RGB")

    exact = px.io.decode_image(px.io.encode_image(frame, format=format_token, lossless=True), unchanged=True)
    default = px.io.decode_image(px.io.encode_image(frame, format=format_token), unchanged=True)
    explicit_lossy = px.io.decode_image(px.io.encode_image(frame, format=format_token, lossless=False), unchanged=True)

    cp.testing.assert_array_equal(exact.data, frame.data)
    assert bool(cp.any(default.data != frame.data).get())
    assert bool(cp.any(explicit_lossy.data != frame.data).get())


def test_new_format_metadata_defaults_overrides_and_channel_selection() -> None:
    """v1-io-formats acceptance 6 and 18: defaults, claims, and label-driven selection match existing raster I/O."""
    frame = _frame(np.uint8, "RGB")
    payload = px.io.encode_image(frame, format="bmp")

    default = px.io.decode_image(payload)
    selected = px.io.decode_image(payload, channels="BR", unchanged=True, colorspace="Rec.2020", gamma="PQ")

    assert (default.colorspace, default.gamma, default.channels) == ("sRGB", "sRGB", ("R", "G", "B"))
    assert (selected.colorspace, selected.gamma, selected.channels) == ("Rec.2020", "PQ", ("B", "R"))
    cp.testing.assert_array_equal(selected.data, frame.data[..., [2, 0]])


def test_new_header_parsers_remain_codec_lazy_and_gpu_free(tmp_path: Path) -> None:
    """v1-io-formats acceptance 16 and 19: pure header inspection avoids codecs and CUDA allocation."""
    paths = []
    for name, payload in (
        ("header.jp2", _minimal_jp2(3, 2, 3, 8)),
        ("header.j2k", _minimal_j2k(3, 2, 3, 8)),
        ("header.webp", _minimal_vp8x(3, 2)),
        ("header.bmp", _minimal_bmp(3, 2, 24)),
        ("header.pnm", b"P6\n3 2\n255\n" + bytes(18)),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        paths.append(path)
    script = """
import sys
import pixtreme as px
for value in sys.argv[1:]:
    assert px.io.read_header(value).width == 3
assert "nvidia.nvimgcodec" not in sys.modules
assert "OpenEXR" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *(str(path) for path in paths)],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_canonical_docs_and_vocabulary_match_the_extended_format_contract() -> None:
    """v1-io-formats acceptance 20: requirements, feature sheets, and vocabulary share the implementation tokens."""
    requirements_path = ROOT / "docs" / "requirements.md"
    io_feature_path = ROOT / "docs" / "features" / "v1-io.md"
    bytes_feature_path = ROOT / "docs" / "features" / "v1-bytes-boundary.md"
    if not (requirements_path.is_file() and io_feature_path.is_file() and bytes_feature_path.is_file()):
        pytest.skip("repo-only documentation contract: docs canon is absent from this distribution")
    requirements = requirements_path.read_text(encoding="utf-8")
    vocabulary = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    io_feature = io_feature_path.read_text(encoding="utf-8")
    bytes_feature = bytes_feature_path.read_text(encoding="utf-8")

    for format_name in ("JPEG 2000", "WebP", "BMP", "PNM"):
        assert format_name in requirements
        assert format_name in vocabulary
    for token in ("jpeg2000", "webp", "bmp", "pnm"):
        assert f"`{token}`" in vocabulary
    for feature in (io_feature, bytes_feature):
        assert "v1-io-formats" in feature
        assert "将来" not in feature.split("v1-io-formats", maxsplit=1)[-1][:120]
