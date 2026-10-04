"""Specification tests for raster image I/O."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

import pixtreme as px


def _save_png(path: Path, values: np.ndarray, *, chunks: dict[bytes, bytes] | None = None) -> None:
    info = PngImagePlugin.PngInfo()
    for chunk, payload in (chunks or {}).items():
        info.add(chunk, payload)
    Image.fromarray(values).save(path, pnginfo=info)


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-008")
@pytest.mark.parametrize(
    ("name", "values", "mode", "labels"),
    (
        ("rgb.PNG", np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8), "RGB", ("R", "G", "B")),
        ("rgba.png", np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.uint8), "RGBA", ("R", "G", "B", "A")),
        ("gray.png", np.array([[1, 2]], dtype=np.uint8), "L", ("Y",)),
    ),
)
def test_read_image_returns_normalized_contiguous_rgb_alpha_and_gray(
    tmp_path: Path,
    name: str,
    values: np.ndarray,
    mode: str,
    labels: tuple[str, ...],
) -> None:
    """Raster image reading returns contiguous normalized float32 pixels with the documented shape and channel order."""
    path = tmp_path / name
    Image.fromarray(values, mode=mode).save(path, format="PNG")

    result = px.io.read_image(path)

    assert result.dtype == np.dtype(np.float32)
    assert result.channels == labels
    assert result.data.flags.c_contiguous
    assert (result.colorspace, result.gamma) == ("sRGB", "sRGB")
    expected = values.reshape(*values.shape[:2], -1).astype(np.float32) / np.float32(255.0)
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        expected,
    )


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-008")
def test_read_image_unchanged_preserves_native_16_bit_and_default_normalizes(tmp_path: Path) -> None:
    """Unchanged raster reading preserves uint16 codes, while ordinary reading normalizes by the container maximum."""
    values = np.array([[0, 1, 32768, 65535]], dtype=np.uint16)
    path = tmp_path / "gray16.png"
    Image.fromarray(values).save(path)

    unchanged = px.io.read_image(path, unchanged=True)
    normalized = px.io.read_image(path)

    assert unchanged.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(
        px.io.to_array(
            unchanged,
        ).get()[..., 0],
        values,
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            normalized,
        ).get()[..., 0],
        values.astype(np.float32) / np.float32(65535.0),
    )


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-017")
def test_read_image_channel_selection_is_label_driven_and_ordered(tmp_path: Path) -> None:
    """Raster reading selects requested channel labels in order and rejects labels absent from the image."""
    values = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.uint8)
    path = tmp_path / "rgba.png"
    Image.fromarray(values, mode="RGBA").save(path)

    result = px.io.read_image(path, channels="BAR", unchanged=True)

    assert result.channels == ("B", "A", "R")
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get(),
        values[..., [2, 3, 0]],
    )
    with pytest.raises(ValueError, match=r"why=.*what=.*how=.*read_header"):
        px.io.read_image(path, channels="Z")


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_png_file_metadata_and_per_call_claims_follow_the_fixed_priority(tmp_path: Path) -> None:
    """PNG reading resolves per-call color claims before mapped file metadata and then format defaults."""
    values = np.array([[[1, 2, 3]]], dtype=np.uint8)
    path = tmp_path / "cicp.png"
    _save_png(path, values, chunks={b"cICP": bytes((9, 16, 0, 1))})

    file_claim = px.io.read_image(path)
    call_claim = px.io.read_image(path, colorspace="ACEScg", gamma="linear")
    header = px.io.read_header(path)

    assert (file_claim.colorspace, file_claim.gamma) == ("Rec.2020", "PQ")
    assert (call_claim.colorspace, call_claim.gamma) == ("ACEScg", "linear")
    assert header.color.raw == {"cICP": (9, 16, 0, 1)}
    assert (header.color.colorspace, header.color.gamma, header.color.mappable) == ("Rec.2020", "PQ", True)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_png_srgb_gama_and_unmappable_metadata_have_deterministic_mapping(tmp_path: Path) -> None:
    """PNG color chunks map supported values to metadata and warn before falling back from unsupported values."""
    values = np.array([[[1, 2, 3]]], dtype=np.uint8)
    srgb_path = tmp_path / "srgb.png"
    gama_path = tmp_path / "gama.png"
    unknown_path = tmp_path / "unknown.png"
    _save_png(srgb_path, values, chunks={b"sRGB": b"\x00"})
    _save_png(gama_path, values, chunks={b"gAMA": struct.pack(">I", 45455)})
    _save_png(unknown_path, values, chunks={b"cICP": bytes((99, 99, 0, 1))})

    assert (px.io.read_image(srgb_path).colorspace, px.io.read_image(srgb_path).gamma) == ("sRGB", "sRGB")
    assert px.io.read_image(gama_path).gamma == "Gamma-2.2"
    with pytest.warns(UserWarning, match="file color metadata"):
        result = px.io.read_image(unknown_path)
    assert (result.colorspace, result.gamma) == ("sRGB", "sRGB")
    assert px.io.read_header(unknown_path).color.mappable is False


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.parametrize("suffix", (".png", ".jpg", ".tiff"))
def test_write_image_uint8_returns_none_and_external_reader_opens_output(tmp_path: Path, suffix: str) -> None:
    """Writing a uint8 raster image returns None and produces a file readable by an independent image library."""
    import cupy as cp

    values = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="sRGB", channels="RGB")
    path = tmp_path / f"output{suffix}"

    result = px.io.write_image(path, frame, quality=95 if suffix == ".jpg" else None)

    assert result is None
    with Image.open(path) as image:
        assert image.size == (2, 1)
        assert image.mode == "RGB"


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-008")
@pytest.mark.parametrize("suffix", (".png", ".tiff"))
def test_write_image_preserves_uint16_gray_depth_round_trip(tmp_path: Path, suffix: str) -> None:
    """PNG and TIFF writing preserve uint16 grayscale codes through a file round trip."""
    import cupy as cp

    values = np.array([[[0], [1], [32768], [65535]]], dtype=np.uint16)
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="sRGB", channels="Y")
    path = tmp_path / f"output{suffix}"

    px.io.write_image(path, frame)

    np.testing.assert_array_equal(np.asarray(Image.open(path)), values[..., 0])
    round_trip = px.io.read_image(path, unchanged=True)
    assert round_trip.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(
        px.io.to_array(
            round_trip,
        ).get(),
        values,
    )


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-017")
def test_write_image_validates_named_quality_and_writes_no_raster_color_chunks(tmp_path: Path) -> None:
    """Raster writing accepts a named quality option and omits unsupported color metadata chunks."""
    import cupy as cp

    frame = px.io.from_array(cp.zeros((1, 1, 3), dtype=cp.uint8), colorspace="sRGB", gamma="sRGB", channels="RGB")
    png_path = tmp_path / "output.png"
    with pytest.raises(ValueError, match="quality"):
        px.io.write_image(tmp_path / "output.jpg", frame, quality=0)
    with pytest.raises(ValueError, match="quality"):
        px.io.write_image(png_path, frame, quality=90)

    px.io.write_image(png_path, frame)

    payload = png_path.read_bytes()
    assert b"sRGB" not in payload and b"cICP" not in payload and b"gAMA" not in payload


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-017")
def test_read_image_errors_distinguish_missing_extension_and_decode_failure(tmp_path: Path) -> None:
    """Image reading distinguishes missing files, unsupported extensions, and decode failures with actionable errors."""
    missing = tmp_path / "missing.png"
    unsupported = tmp_path / "image.gif"
    corrupt = tmp_path / "corrupt.png"
    unsupported.write_bytes(b"GIF89a")
    corrupt.write_bytes(b"not a png")

    with pytest.raises(FileNotFoundError, match=r"why=.*what=.*how="):
        px.io.read_image(missing)
    with pytest.raises(ValueError, match=r"why=.*what=.*how="):
        px.io.read_image(unsupported)
    with pytest.raises(RuntimeError, match=r"why=.*what=.*how="):
        px.io.read_image(corrupt)


@pytest.mark.req("REQ-PIX-007")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize(
    ("name", "args"),
    (
        ("read_image", (None,)),
        ("write_image", (None, None)),
        ("read_header", (None,)),
    ),
)
def test_image_path_boundaries_translate_invalid_path_types_actionably(name: str, args: tuple[object, ...]) -> None:
    """Image path APIs explain invalid path types while retaining the original pathlib error as their cause."""
    with pytest.raises(ValueError, match=r"^why=.*; what=.*; how=.*") as error:
        getattr(px.io, name)(*args)

    assert "path" in str(error.value)
    assert "str or os.PathLike" in str(error.value)
    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.req("REQ-PIX-007")
def test_read_header_reports_jpeg_tiff_and_float_tiff_without_decoding_pixels(tmp_path: Path) -> None:
    """Header inspection reports JPEG and TIFF dimensions, channels, and storage dtype without decoding pixels."""
    jpeg = tmp_path / "sample.jpg"
    tiff = tmp_path / "sample.tiff"
    float_tiff = tmp_path / "float.tiff"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8), mode="RGB").save(jpeg)
    Image.fromarray(np.zeros((2, 3), dtype=np.uint16)).save(tiff)
    Image.fromarray(np.array([[0.25, 1.5]], dtype=np.float32), mode="F").save(float_tiff)

    assert (px.io.read_header(jpeg).format, px.io.read_header(jpeg).parts[0].channels) == (
        "JPEG",
        {"R": "uint8", "G": "uint8", "B": "uint8"},
    )
    assert px.io.read_header(tiff).parts[0].channels == {"Y": "uint16"}
    assert px.io.read_header(float_tiff).parts[0].channels == {"Y": "float32"}
    np.testing.assert_array_equal(
        px.io.to_array(
            px.io.read_image(float_tiff, unchanged=True),
        ).get()[..., 0],
        np.array([[0.25, 1.5]], dtype=np.float32),
    )
