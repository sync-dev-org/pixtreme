"""Specification tests for embedded ICC carriers at public raster boundaries."""

from __future__ import annotations

import gc
import inspect
import os
import struct
import subprocess
import sys
import tracemalloc
import warnings
from io import BytesIO
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from generate_io_fixtures import encode_oriented_raster
from icc_test_utils import (
    icc_profile,
    jpeg_header,
    png_header,
    stored_zlib_stream,
    tiff_header,
    trc_payload,
    webp_chunk,
    webp_header,
)
from PIL import Image

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
_SUFFIXES = {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tiff", "WEBP": ".webp"}


def _header(tmp_path: Path, payload: bytes, suffix: str) -> px.io.ImageHeader:
    path = tmp_path / f"image{suffix}"
    path.write_bytes(payload)
    return px.io.read_header(path)


def _header_with_peak_allocation(path: Path) -> tuple[px.io.ImageHeader, int]:
    gc.collect()
    tracemalloc.start()
    try:
        header = px.io.read_header(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return header, peak


def _encoded_image(format_name: str, profile: bytes | None, *, mode: str = "RGB") -> bytes:
    shape = (4, 6) if mode in ("L", "P") else (4, 6, len(mode))
    values = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    image = Image.fromarray(values, mode=mode)
    if mode == "P":
        image.putpalette(np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 3)).reshape(-1))
    output = BytesIO()
    options: dict[str, object] = {}
    if profile is not None:
        options["icc_profile"] = profile
    if format_name == "WEBP":
        options["lossless"] = True
    image.save(output, format=format_name, **options)
    return output.getvalue()


def _warnings(call: object, *args: object, **kwargs: object) -> tuple[px.core.Frame, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call(*args, **kwargs)  # type: ignore[operator]
    return result, caught


def test_public_signatures_and_image_header_shape_do_not_change_for_icc() -> None:
    """v1-io-icc acceptance 4: ICC reuses the existing public boundaries and nested color fields."""
    assert tuple(inspect.signature(px.io.read_image).parameters) == (
        "path",
        "channels",
        "unchanged",
        "colorspace",
        "gamma",
        "apply_exif_orientation",
    )
    assert tuple(inspect.signature(px.io.decode_image).parameters) == (
        "data",
        "channels",
        "unchanged",
        "colorspace",
        "gamma",
        "apply_exif_orientation",
    )
    assert tuple(inspect.signature(px.io.read_header).parameters) == ("path",)
    assert set(px.io.ImageHeader.model_fields) == {"format", "width", "height", "parts", "color", "orientation"}


@pytest.mark.parametrize("name", (b"A", b"profile-name", b"\xa1\xff", b"A" * 79))
def test_png_iccp_accepts_valid_names_and_maps_exact_decompressed_profile(tmp_path: Path, name: bytes) -> None:
    """v1-io-icc acceptance 5 and 18: a valid iCCP exposes exact decompressed bytes independent of its name."""
    profile = icc_profile(colorspace="Adobe-RGB", gamma="Adobe-RGB")
    color = _header(tmp_path, png_header(profile=profile, name=name), ".png").color

    assert color.raw["ICC"] == profile
    assert (color.colorspace, color.gamma, color.mappable) == ("Adobe-RGB", "Adobe-RGB", True)


@pytest.mark.parametrize("name", (b"", b" A", b"A ", b"A  B", b"\x1f", b"\x7f", b"\xa0", b"A" * 80))
def test_png_iccp_rejects_invalid_names_without_rejecting_the_image(tmp_path: Path, name: bytes) -> None:
    """v1-io-icc acceptance 5 and 16: invalid iCCP names select unmappable optional metadata."""
    profile = icc_profile()
    color = _header(tmp_path, png_header(profile=profile, name=name), ".png").color

    assert color.raw == {}
    assert (color.colorspace, color.gamma, color.mappable) == (None, None, False)


@pytest.mark.parametrize(
    "payload",
    (
        png_header(profile=icc_profile(), method=1),
        png_header(compressed=b"not-zlib"),
        png_header(profile=icc_profile(), duplicate_iccp=True),
    ),
    ids=("method", "stream", "duplicate"),
)
def test_png_iccp_carrier_failures_have_no_raw_profile(tmp_path: Path, payload: bytes) -> None:
    """v1-io-icc acceptance 5, 16, and 18: malformed or duplicate PNG carriers are recoverable and not raw."""
    color = _header(tmp_path, payload, ".png").color

    assert color.raw == {}
    assert (color.colorspace, color.gamma, color.mappable) == (None, None, False)


def test_png_color_source_priority_never_falls_through_to_shadowed_iccp(tmp_path: Path) -> None:
    """v1-io-icc acceptance 8 and 25: cICP outranks iCCP, sRGB, and gAMA even when lower data is bad."""
    profile = icc_profile(version=3)
    cicp = _header(
        tmp_path,
        png_header(profile=profile, compressed=b"not-zlib", cicp=bytes((9, 16, 0, 1)), srgb=True, gama=45455),
        ".png",
    ).color
    invalid_icc = _header(tmp_path, png_header(profile=profile, srgb=True, gama=45455), ".png").color

    assert cicp.raw == {"cICP": (9, 16, 0, 1), "sRGB": 0, "gAMA": 45455}
    assert (cicp.colorspace, cicp.gamma, cicp.mappable) == ("Rec.2020", "PQ", True)
    assert "ICC" not in cicp.raw
    assert invalid_icc.raw["ICC"] == profile
    assert (invalid_icc.colorspace, invalid_icc.gamma, invalid_icc.mappable) == (None, None, False)


def test_png_cicp_priority_skips_oversized_preceding_iccp_payload(tmp_path: Path) -> None:
    """v1-io-icc acceptance 8 and 25: selected cICP leaves a preceding oversized iCCP unmaterialized."""
    compressed_limit = 17_825_792
    payload = png_header(compressed=b"\x00" * (compressed_limit + 1), cicp=bytes((9, 16, 0, 1)))
    path = tmp_path / "shadowed-oversized.png"
    path.write_bytes(payload)
    del payload

    header, peak = _header_with_peak_allocation(path)

    assert header.color.raw == {"cICP": (9, 16, 0, 1)}
    assert (header.color.colorspace, header.color.gamma, header.color.mappable) == ("Rec.2020", "PQ", True)
    assert peak < 2 * 1024 * 1024


def test_png_oversized_compressed_stream_is_rejected_before_payload_read(tmp_path: Path) -> None:
    """v1-io-icc acceptance 25: a declared 17 MiB + 1 compressed stream is skipped before materialization."""
    compressed_limit = 17_825_792
    payload = png_header(compressed=b"\x00" * (compressed_limit + 1))
    path = tmp_path / "oversized-compressed.png"
    path.write_bytes(payload)
    del payload

    header, peak = _header_with_peak_allocation(path)

    assert header.color.raw == {}
    assert header.color.mappable is False
    assert peak < 2 * 1024 * 1024


def test_jpeg_app2_reassembles_shuffled_segments_and_rejects_invalid_sequences(tmp_path: Path) -> None:
    """v1-io-icc acceptance 6 and 18: APP2 reconstruction is ordered, complete, unique, and count-consistent."""
    profile = icc_profile(colorspace="P3-D65", gamma="sRGB")
    split = len(profile) // 2
    valid = jpeg_header(((2, 2, profile[split:]), (1, 2, profile[:split])))
    color = _header(tmp_path, valid, ".jpg").color
    assert color.raw["ICC"] == profile
    assert (color.colorspace, color.gamma, color.mappable) == ("P3-D65", "sRGB", True)

    invalid_sets = (
        ((1, 2, profile[:split]),),
        ((1, 2, profile[:split]), (1, 2, profile[split:])),
        ((0, 1, profile),),
        ((1, 1, profile[:split]), (2, 2, profile[split:])),
    )
    for segments in invalid_sets:
        invalid = _header(tmp_path, jpeg_header(segments), ".jpg").color
        assert invalid.raw == {}
        assert invalid.mappable is False

    absent = _header(tmp_path, jpeg_header(), ".jpg").color
    assert absent.raw == {}
    assert absent.mappable is None


def test_tiff_intercolorprofile_requires_one_undefined_first_ifd_tag(tmp_path: Path) -> None:
    """v1-io-icc acceptance 7 and 18: TIFF reads one exact type-UNDEFINED first-IFD profile."""
    profile = icc_profile(colorspace="Rec.2020", gamma="Gamma-2.4")
    valid = _header(tmp_path, tiff_header(profiles=(profile,)), ".tiff").color
    wrong_type = _header(tmp_path, tiff_header(profiles=(profile,), profile_type=1), ".tiff").color
    duplicate = _header(tmp_path, tiff_header(profiles=(profile, profile)), ".tiff").color
    absent = _header(tmp_path, tiff_header(), ".tiff").color

    assert valid.raw["ICC"] == profile
    assert (valid.colorspace, valid.gamma, valid.mappable) == ("Rec.2020", "Gamma-2.4", True)
    for invalid in (wrong_type, duplicate):
        assert invalid.raw == {}
        assert invalid.mappable is False
    assert absent.mappable is None


def test_tiff_broken_optional_profile_offset_does_not_become_container_corruption(tmp_path: Path) -> None:
    """v1-io-icc acceptance 7 and 16: an unreadable ICC value is carrier failure when the first IFD stays valid."""
    payload = bytearray(tiff_header(profiles=(icc_profile(),)))
    entry_count = struct.unpack_from("<H", payload, 8)[0]
    for index in range(entry_count):
        entry = 10 + 12 * index
        if struct.unpack_from("<H", payload, entry)[0] == 34675:
            struct.pack_into("<I", payload, entry + 8, len(payload) + 100)
            break
    else:
        raise AssertionError("fixture lacks InterColorProfile")

    color = _header(tmp_path, bytes(payload), ".tiff").color

    assert color.raw == {}
    assert color.mappable is False


def test_tiff_duplicate_profiles_are_rejected_before_payload_read(tmp_path: Path) -> None:
    """v1-io-icc acceptance 7 and 25: duplicate first-IFD ICC tags are rejected before materialization."""
    profile = b"\x00" * (4 * 1024 * 1024)
    payload = tiff_header(profiles=(profile, profile))
    path = tmp_path / "duplicate-profiles.tiff"
    path.write_bytes(payload)
    del payload

    header, peak = _header_with_peak_allocation(path)

    assert header.color.raw == {}
    assert header.color.mappable is False
    assert peak < 2 * 1024 * 1024


def test_webp_iccp_requires_vp8x_flag_unique_chunk_and_normative_order(tmp_path: Path) -> None:
    """v1-io-icc acceptance 7 and 18: WebP validates the ICC flag, chunk uniqueness, and pre-image order."""
    profile = icc_profile(colorspace="Adobe-RGB", gamma="Adobe-RGB")
    valid = _header(tmp_path, webp_header(profiles=(profile,)), ".webp").color
    assert valid.raw["ICC"] == profile
    assert (valid.colorspace, valid.gamma, valid.mappable) == ("Adobe-RGB", "Adobe-RGB", True)

    invalid_payloads = (
        webp_header(profiles=(), icc_flag=True),
        webp_header(profiles=(profile,), icc_flag=False),
        webp_header(profiles=(profile,), iccp_before_vp8x=True),
        webp_header(profiles=(profile,), iccp_after_image=True),
        webp_header(profiles=(profile, profile)),
    )
    for payload in invalid_payloads:
        invalid = _header(tmp_path, payload, ".webp").color
        assert invalid.raw == {}
        assert invalid.mappable is False
    assert _header(tmp_path, webp_header(profiles=(), icc_flag=False), ".webp").color.mappable is None


def test_webp_icc_is_independent_from_exif_orientation_and_pixel_samples(tmp_path: Path) -> None:
    """v1-io-icc acceptance 4 and 7: ICC labeling leaves WebP orientation, dimensions, and samples unchanged."""
    plain = encode_oriented_raster("WEBP", 6)
    profile = icc_profile(colorspace="P3-D65", gamma="sRGB")
    assert plain[12:16] == b"VP8X"
    vp8x_size = struct.unpack_from("<I", plain, 16)[0]
    vp8x_end = 20 + vp8x_size + (vp8x_size & 1)
    embedded = bytearray(plain[:vp8x_end] + webp_chunk(b"ICCP", profile) + plain[vp8x_end:])
    embedded[20] |= 0x20
    struct.pack_into("<I", embedded, 4, len(embedded) - 8)
    payload = bytes(embedded)
    path = tmp_path / "oriented-icc.webp"
    path.write_bytes(payload)

    plain_frame = px.io.decode_image(plain, unchanged=True)
    embedded_frame = px.io.decode_image(payload, unchanged=True)
    file_frame = px.io.read_image(path, unchanged=True)
    header = px.io.read_header(path)

    cp.testing.assert_array_equal(embedded_frame.data, plain_frame.data)
    cp.testing.assert_array_equal(file_frame.data, embedded_frame.data)
    assert (embedded_frame.shape, embedded_frame.dtype, embedded_frame.channels, embedded_frame.matrix) == (
        plain_frame.shape,
        plain_frame.dtype,
        plain_frame.channels,
        plain_frame.matrix,
    )
    assert (embedded_frame.colorspace, embedded_frame.gamma) == ("P3-D65", "sRGB")
    assert (header.orientation, header.width, header.height) == (6, embedded_frame.width, embedded_frame.height)


@pytest.mark.parametrize("format_name", ("PNG", "TIFF", "WEBP"))
def test_profile_size_limit_accepts_exact_16_mib_and_rejects_one_more_byte(tmp_path: Path, format_name: str) -> None:
    """v1-io-icc acceptance 25: PNG, TIFF, and WebP enforce the exact common reconstructed-size boundary."""
    limit = 16_777_216
    exact_profile = b"\x00" * limit
    oversized_profile = b"\x00" * (limit + 1)
    builders = {
        "PNG": lambda value: png_header(profile=value),
        "TIFF": lambda value: tiff_header(profiles=(value,)),
        "WEBP": lambda value: webp_header(profiles=(value,)),
    }
    suffix = _SUFFIXES[format_name]

    exact = _header(tmp_path, builders[format_name](exact_profile), suffix).color
    oversized = _header(tmp_path, builders[format_name](oversized_profile), suffix).color

    assert len(exact.raw["ICC"]) == limit
    assert exact.mappable is False
    assert oversized.raw == {}
    assert oversized.mappable is False


def test_png_compressed_input_limit_accepts_exact_17_mib_and_rejects_one_more_byte(tmp_path: Path) -> None:
    """v1-io-icc acceptance 25: PNG bounds compressed input separately before bounded decompression."""
    compressed_limit = 17_825_792
    stream = stored_zlib_stream(output_size=16_777_211, stream_size=compressed_limit)

    exact = _header(tmp_path, png_header(compressed=stream), ".png").color
    oversized = _header(tmp_path, png_header(compressed=stream + b"\x00"), ".png").color

    assert len(exact.raw["ICC"]) == 16_777_211
    assert exact.mappable is False
    assert oversized.raw == {}
    assert oversized.mappable is False


def test_jpeg_accepts_the_maximum_255_segment_reconstruction(tmp_path: Path) -> None:
    """v1-io-icc acceptance 6 and 25: JPEG reconstructs the exact carrier maximum within bounded counts."""
    segment_size = 65_519
    count = 255
    profile = bytearray(segment_size * count)
    struct.pack_into(">I", profile, 0, len(profile))
    segments = (
        (sequence, count, bytes(profile[(sequence - 1) * segment_size : sequence * segment_size]))
        for sequence in range(1, 256)
    )

    color = _header(tmp_path, jpeg_header(segments), ".jpg").color

    assert len(color.raw["ICC"]) == segment_size * count
    assert color.mappable is False


def test_jpeg_rejects_oversized_duplicate_aggregate_with_bounded_retention(tmp_path: Path) -> None:
    """v1-io-icc acceptance 6 and 25: invalid APP2 aggregates stop retaining data after duplication is known."""
    segment = b"\x00" * 65_519
    payload = jpeg_header(((1, 1, segment) for _ in range(300)))
    path = tmp_path / "duplicate-aggregate.jpg"
    path.write_bytes(payload)
    del payload

    header, peak = _header_with_peak_allocation(path)

    assert header.color.raw == {}
    assert header.color.mappable is False
    assert peak < 2 * 1024 * 1024


def test_webp_duplicate_iccp_chunks_discard_and_skip_payloads(tmp_path: Path) -> None:
    """v1-io-icc acceptance 7 and 25: duplicate ICCP chunks do not retain a RIFF-sized payload collection."""
    profile = b"\x00" * (4 * 1024 * 1024)
    payload = webp_header(profiles=(profile,) * 5)
    path = tmp_path / "duplicate-aggregate.webp"
    path.write_bytes(payload)
    del payload

    header, peak = _header_with_peak_allocation(path)

    assert header.color.raw == {}
    assert header.color.mappable is False
    assert peak < 8 * 1024 * 1024


@pytest.mark.parametrize("format_name", tuple(_SUFFIXES))
def test_file_and_bytes_boundaries_resolve_valid_partial_invalid_absent_and_overrides_identically(
    tmp_path: Path, format_name: str
) -> None:
    """v1-io-icc acceptance 4, 15-17, and 19: both decoded boundaries share component-wise metadata resolution."""
    valid_profile = icc_profile(colorspace="Adobe-RGB", gamma="Adobe-RGB")
    partial_profile = icc_profile(
        colorspace="P3-D65",
        trcs=(trc_payload("sRGB"), trc_payload("Rec.709"), trc_payload("sRGB")),
    )
    invalid_profile = icc_profile(version=3)
    cases = (
        (valid_profile, ("Adobe-RGB", "Adobe-RGB"), 0),
        (partial_profile, ("P3-D65", "sRGB"), 1),
        (invalid_profile, ("sRGB", "sRGB"), 1),
        (None, ("sRGB", "sRGB"), 0),
    )
    for index, (profile, expected, warning_count) in enumerate(cases):
        payload = _encoded_image(format_name, profile)
        path = tmp_path / f"{format_name.lower()}-{index}{_SUFFIXES[format_name]}"
        path.write_bytes(payload)
        from_bytes, byte_warnings = _warnings(px.io.decode_image, payload, unchanged=True)
        from_file, file_warnings = _warnings(px.io.read_image, path, unchanged=True)

        assert (from_bytes.colorspace, from_bytes.gamma) == expected
        assert (from_file.colorspace, from_file.gamma) == expected
        assert len(byte_warnings) == len(file_warnings) == warning_count
        assert (from_file.shape, from_file.dtype, from_file.channels, from_file.matrix) == (
            from_bytes.shape,
            from_bytes.dtype,
            from_bytes.channels,
            from_bytes.matrix,
        )
        cp.testing.assert_array_equal(from_file.data, from_bytes.data)

        overridden, override_warnings = _warnings(
            px.io.decode_image,
            payload,
            unchanged=True,
            colorspace="ProPhoto RGB",
            gamma="Gamma 1.8",
        )
        assert (overridden.colorspace, overridden.gamma) == ("ProPhoto-RGB", "Gamma-1.8")
        assert len(override_warnings) == warning_count


@pytest.mark.parametrize(("mode", "expected"), (("RGB", True), ("RGBA", True), ("P", True), ("L", False)))
def test_icc_application_uses_preselection_standard_container_channels(
    tmp_path: Path, mode: str, expected: bool
) -> None:
    """v1-io-icc acceptance 26: RGB(A)/palette apply ICC before selection while grayscale remains incompatible."""
    profile = icc_profile(colorspace="P3-D65", gamma="sRGB")
    payload = _encoded_image("PNG", profile, mode=mode)
    path = tmp_path / f"channels-{mode}.png"
    path.write_bytes(payload)
    header = px.io.read_header(path)

    assert header.color.raw["ICC"] == profile
    assert header.color.mappable is expected
    if expected:
        selected = px.io.decode_image(payload, channels="BR", unchanged=True)
        assert (selected.colorspace, selected.gamma, selected.channels) == ("P3-D65", "sRGB", ("B", "R"))
    else:
        result, caught = _warnings(px.io.decode_image, payload, unchanged=True)
        assert (result.colorspace, result.gamma, len(caught)) == ("sRGB", "sRGB", 1)


def test_read_header_maps_valid_and_invalid_icc_without_codec_or_cuda_initialization(tmp_path: Path) -> None:
    """v1-io-icc acceptance 20: header-only carrier/profile mapping stays CPU-only for success and fallback."""
    valid = tmp_path / "valid.png"
    invalid = tmp_path / "invalid.png"
    valid.write_bytes(png_header(profile=icc_profile()))
    invalid.write_bytes(png_header(profile=icc_profile(version=3)))
    script = """
import sys
import pixtreme as px
valid = px.io.read_header(sys.argv[1])
invalid = px.io.read_header(sys.argv[2])
assert valid.color.mappable is True
assert invalid.color.mappable is False
assert "nvidia.nvimgcodec" not in sys.modules
assert "OpenEXR" not in sys.modules
assert "PIL" not in sys.modules
assert "colour" not in sys.modules
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", script, str(valid), str(invalid)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_production_icc_import_graph_has_no_dev_profile_oracle_dependencies() -> None:
    """v1-io-icc acceptance 21: production ICC handling adds no Pillow, lcms2, or colour-science import."""
    source = (ROOT / "src" / "pixtreme" / "_io" / "icc.py").read_text(encoding="utf-8")

    assert "PIL" not in source
    assert "ImageCms" not in source
    assert "colour" not in source
