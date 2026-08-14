"""Specification tests for the Radiance HDR file boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
_ACTIONABLE = r"why=.*what=.*how="


def _header(
    *,
    width: int,
    height: int,
    format_value: str = "32-bit_rle_rgbe",
    resolution: str | None = None,
    variables: tuple[str, ...] = (),
) -> bytes:
    lines = ("#?RADIANCE", *variables, f"FORMAT={format_value}", "", resolution or f"-Y {height} +X {width}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _decode_rgbe_oracle(rgbe: np.ndarray) -> np.ndarray:
    """Evaluate the published Radiance colr_color equation independently on the host."""
    exponent = rgbe[..., 3].astype(np.int32)
    scale = np.ldexp(np.ones(exponent.shape, dtype=np.float32), exponent - 136)
    decoded = (rgbe[..., :3].astype(np.float32) + np.float32(0.5)) * scale[..., None]
    return np.where((exponent == 0)[..., None], np.float32(0.0), decoded).astype(np.float32)


def _encode_rgbe_oracle(rgb: np.ndarray) -> np.ndarray:
    """Evaluate the published Radiance setcolr frexp equation independently on the host."""
    source = rgb.astype(np.float32, copy=False)
    maximum = np.max(source, axis=2)
    mantissa, exponent = np.frexp(maximum)
    non_black = maximum > np.float32(1e-32)
    scale = np.zeros(maximum.shape, dtype=np.float32)
    scale[non_black] = mantissa[non_black].astype(np.float32) * np.float32(256.0) / maximum[non_black]
    output = np.zeros((*source.shape[:2], 4), dtype=np.uint8)
    positive = source > np.float32(0.0)
    scaled = source * scale[..., None]
    output[..., :3] = np.where(positive, scaled, np.float32(0.0)).astype(np.uint8)
    output[..., 3] = np.where(non_black, exponent + 128, 0).astype(np.uint8)
    return output


def _new_style_fixture(rgbe: np.ndarray) -> bytes:
    height, width, _ = rgbe.shape
    payload = bytearray()
    for row in rgbe:
        payload.extend((2, 2, width >> 8, width & 0xFF))
        for component in range(4):
            values = row[:, component]
            for start in range(0, width, 128):
                literal = values[start : start + 128]
                payload.append(len(literal))
                payload.extend(literal.tobytes())
    return _header(width=width, height=height) + bytes(payload)


def _old_style_fixture(rows: tuple[bytes, ...], *, width: int) -> bytes:
    return _header(width=width, height=len(rows)) + b"".join(rows)


def _decode_new_style_file(payload: bytes) -> tuple[np.ndarray, tuple[tuple[str, int], ...], str]:
    """Decode writer output from literal Radiance packet rules without using pixtreme internals."""
    header_end = payload.index(b"\n\n")
    header_text = payload[:header_end].decode("ascii")
    resolution_end = payload.index(b"\n", header_end + 2)
    resolution = payload[header_end + 2 : resolution_end].decode("ascii")
    _, height_text, _, width_text = resolution.split()
    height = int(height_text)
    width = int(width_text)
    offset = resolution_end + 1
    output = np.empty((height, width, 4), dtype=np.uint8)
    packets: list[tuple[str, int]] = []
    for row_index in range(height):
        assert payload[offset : offset + 4] == bytes((2, 2, width >> 8, width & 0xFF))
        offset += 4
        for component in range(4):
            position = 0
            while position < width:
                code = payload[offset]
                offset += 1
                if code > 128:
                    count = code - 128
                    value = payload[offset]
                    offset += 1
                    output[row_index, position : position + count, component] = value
                    packets.append(("run", count))
                else:
                    count = code
                    output[row_index, position : position + count, component] = np.frombuffer(
                        payload[offset : offset + count], dtype=np.uint8
                    )
                    offset += count
                    packets.append(("literal", count))
                position += count
    assert offset == len(payload)
    return output, tuple(packets), header_text


def _host_recode_float32(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.dtype(np.float32):
        return values.copy()
    if values.dtype == np.dtype(np.float16):
        return values.astype(np.float32)
    return values.astype(np.float32) / np.float32(np.iinfo(values.dtype).max)


def test_hdr_read_decodes_flat_rgbe_with_published_ldexp_oracle(tmp_path: Path) -> None:
    """v1-hdr acceptance 1 and 3: flat RGBE lands as top-down fp32 Rec.709/linear data."""
    rgbe = np.array(
        [
            [[128, 64, 32, 129], [0, 0, 0, 0], [255, 1, 127, 140]],
            [[1, 2, 3, 120], [64, 128, 192, 130], [5, 0, 10, 134]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "flat.HDR"
    path.write_bytes(_header(width=3, height=2) + rgbe.tobytes())

    actual = px.io.read_image(path)

    assert (actual.dtype, actual.channels, actual.colorspace, actual.gamma) == (
        np.dtype(np.float32),
        ("R", "G", "B"),
        "Rec.709",
        "linear",
    )
    assert actual.data.flags.c_contiguous
    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        _decode_rgbe_oracle(rgbe),
    )


def test_hdr_read_decodes_old_style_runs_and_multibyte_counts(tmp_path: Path) -> None:
    """v1-hdr acceptance 1 and 11: old-style repeat markers include consecutive higher-order count bytes."""
    pixel = bytes((17, 34, 51, 130))
    row = pixel + bytes((1, 1, 1, 1)) + bytes((1, 1, 1, 1))
    path = tmp_path / "old.hdr"
    path.write_bytes(_old_style_fixture((row,), width=258))
    expected_rgbe = np.repeat(np.frombuffer(pixel, dtype=np.uint8)[None, None, :], 258, axis=1)

    actual = px.io.read_image(path)

    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        _decode_rgbe_oracle(expected_rgbe),
    )


def test_hdr_read_decodes_new_style_component_runs_and_literals(tmp_path: Path) -> None:
    """v1-hdr acceptance 1 and 11: hand-built adaptive component packets decode independently."""
    width = 8
    expected = np.array(
        [
            [
                [10, 1, 50, 129],
                [10, 2, 51, 129],
                [10, 3, 52, 129],
                [10, 4, 53, 129],
                [20, 5, 54, 130],
                [20, 6, 55, 130],
                [20, 7, 56, 130],
                [20, 8, 57, 130],
            ]
        ],
        dtype=np.uint8,
    )
    scanline = bytes((2, 2, 0, width))
    scanline += bytes((132, 10, 132, 20))
    scanline += bytes((8, 1, 2, 3, 4, 5, 6, 7, 8))
    scanline += bytes((8, 50, 51, 52, 53, 54, 55, 56, 57))
    scanline += bytes((132, 129, 132, 130))
    path = tmp_path / "new.hdr"
    path.write_bytes(_header(width=width, height=1) + scanline)

    actual = px.io.read_image(path)

    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        _decode_rgbe_oracle(expected),
    )


def test_hdr_read_accepts_new_style_literal_packets_of_length_128(tmp_path: Path) -> None:
    """v1-hdr acceptance 1 and 11: the full uint8 literal-count boundary remains valid."""
    rgbe = np.empty((1, 128, 4), dtype=np.uint8)
    rgbe[..., :3] = np.arange(128 * 3, dtype=np.uint8).reshape(1, 128, 3)
    rgbe[..., 3] = 136
    path = tmp_path / "literal-128.hdr"
    path.write_bytes(_new_style_fixture(rgbe))

    actual = px.io.read_image(path)

    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        _decode_rgbe_oracle(rgbe),
    )


def test_hdr_read_unchanged_selects_channels_and_overrides_metadata(tmp_path: Path) -> None:
    """v1-hdr acceptance 3: unchanged is fp32-equivalent and read metadata/channel overrides remain label-driven."""
    rgbe = np.array([[[128, 64, 32, 129] for _ in range(8)]], dtype=np.uint8)
    path = tmp_path / "selection.hdr"
    path.write_bytes(_new_style_fixture(rgbe))

    default = px.io.read_image(path, channels=("B", "R", "R"))
    unchanged = px.io.read_image(
        path,
        channels=("B", "R", "R"),
        unchanged=True,
        colorspace="ACEScg",
        gamma="2.2",
    )

    assert (default.dtype, unchanged.dtype) == (np.dtype(np.float32), np.dtype(np.float32))
    assert unchanged.channels == ("B", "R", "R")
    assert (unchanged.colorspace, unchanged.gamma) == ("ACEScg", "2.2")
    np.testing.assert_array_equal(
        px.io.to_array(
            default,
        ).get(),
        px.io.to_array(
            unchanged,
        ).get(),
    )


def test_hdr_header_variables_are_raw_only_and_do_not_change_pixels_or_metadata(tmp_path: Path) -> None:
    """v1-hdr acceptance 5 and 7: EXPOSURE, PRIMARIES, and COLORCORR are inspectable but unapplied."""
    rgbe = np.array([[[128, 64, 32, 129] for _ in range(8)]], dtype=np.uint8)
    variables = (
        "EXPOSURE=8.0",
        "PRIMARIES=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
        "COLORCORR=2.0 3.0 4.0",
    )
    plain = tmp_path / "plain.hdr"
    decorated = tmp_path / "decorated.hdr"
    plain.write_bytes(_new_style_fixture(rgbe))
    decorated.write_bytes(
        _header(width=8, height=1, variables=variables) + _new_style_fixture(rgbe).split(b"\n", 4)[-1]
    )

    plain_frame = px.io.read_image(plain)
    decorated_frame = px.io.read_image(decorated)
    header = px.io.read_header(decorated)

    cp.testing.assert_array_equal(plain_frame.data, decorated_frame.data)
    assert (decorated_frame.colorspace, decorated_frame.gamma) == ("Rec.709", "linear")
    assert header.color.raw == {
        "EXPOSURE": ("8.0",),
        "PRIMARIES": ("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",),
        "COLORCORR": ("2.0 3.0 4.0",),
    }


@pytest.mark.parametrize(
    ("format_value", "resolution", "observed"),
    (
        ("32-bit_rle_xyze", None, "XYZE"),
        ("32-bit_rle_rgbe", "+Y 1 +X 8", "orientation"),
        ("32-bit_rle_rgbe", "-X 8 +Y 1", "orientation"),
    ),
    ids=("xyze", "bottom-up", "axis-order"),
)
def test_hdr_read_rejects_xyze_and_nonstandard_orientations_before_transfer(
    tmp_path: Path, format_value: str, resolution: str | None, observed: str
) -> None:
    """v1-hdr acceptance 1 and 2: out-of-scope header configurations fail fast as actionable ValueError."""
    path = tmp_path / "unsupported.hdr"
    path.write_bytes(
        _header(width=8, height=1, format_value=format_value, resolution=resolution)
        + np.zeros((1, 8, 4), dtype=np.uint8).tobytes()
    )

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.read_image(path)

    assert observed in str(error.value)


@pytest.mark.parametrize(
    ("scanline", "observed"),
    (
        (bytes((2, 2, 0, 7)), "length"),
        (bytes((2, 2, 0, 8, 0)), "zero"),
        (bytes((2, 2, 0, 8, 137, 7)), "exceeds"),
        (bytes((2, 2, 0, 8, 8, 1, 2, 3)), "truncated"),
    ),
    ids=("width-mismatch", "zero-count", "overrun", "truncated-literal"),
)
def test_hdr_read_rejects_malformed_new_style_scanlines(tmp_path: Path, scanline: bytes, observed: str) -> None:
    """v1-hdr acceptance 1 and 11: malformed adaptive RLE fails with actionable RuntimeError."""
    path = tmp_path / "malformed.hdr"
    path.write_bytes(_header(width=8, height=1) + scanline)

    with pytest.raises(RuntimeError, match=_ACTIONABLE) as error:
        px.io.read_image(path)

    assert observed in str(error.value)


def test_hdr_read_transfers_only_flat_uint8_rgbe_before_gpu_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-hdr acceptance 4: the host-to-device image transfer is one flat uint8 RGBE buffer, never host floats."""
    rgbe = np.array([[[128, 64, 32, 129] for _ in range(8)]], dtype=np.uint8)
    path = tmp_path / "transfer.hdr"
    path.write_bytes(_new_style_fixture(rgbe))
    original_asarray = cp.asarray
    host_inputs: list[np.ndarray] = []

    def capture_asarray(value: object, *args: object, **kwargs: object) -> cp.ndarray:
        if isinstance(value, np.ndarray):
            host_inputs.append(value)
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr("pixtreme._io.formats.hdr.cp.asarray", capture_asarray)

    px.io.read_image(path)

    assert len(host_inputs) == 1
    assert (host_inputs[0].dtype, host_inputs[0].shape) == (np.dtype(np.uint8), (1 * 8 * 4,))


@pytest.mark.parametrize(
    "dtype", (np.uint8, np.uint16, np.uint32, np.float16, np.float32), ids=lambda value: np.dtype(value).name
)
def test_hdr_write_accepts_every_dtype_and_matches_independent_frexp_oracle(
    tmp_path: Path, dtype: type[np.generic]
) -> None:
    """v1-exr-runtime-independence acceptance 9: all five dtype recodes match independent host equations."""
    if np.issubdtype(dtype, np.integer):
        maximum = np.iinfo(dtype).max
        row = np.array([[0, maximum // 4, maximum], [maximum, maximum // 2, maximum // 8]], dtype=dtype)
    else:
        row = np.array([[0.0, 0.25, 1.0], [4.0, 2.0, -1.0]], dtype=dtype)
    values = np.resize(row, (2, 8, 3)).astype(dtype)
    bgr = values[..., [2, 1, 0]]
    frame = px.io.from_array(cp.asarray(bgr), colorspace="Rec.709", gamma="linear", channels="BGR")
    before = frame.data.copy()
    path = tmp_path / f"{np.dtype(dtype).name}.hdr"

    assert px.io.write_image(path, frame) is None

    encoded, packets, header_text = _decode_new_style_file(path.read_bytes())
    expected_float = _host_recode_float32(values)
    np.testing.assert_array_equal(encoded, _encode_rgbe_oracle(expected_float))
    assert all(kind in ("run", "literal") and 1 <= count <= 128 for kind, count in packets)
    assert header_text.splitlines() == ["#?RADIANCE", "FORMAT=32-bit_rle_rgbe"]
    assert frame.dtype == np.dtype(dtype)
    cp.testing.assert_array_equal(frame.data, before)


def test_hdr_write_float32_native_bypasses_recode_dtype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-hdr acceptance 6: the native float32 container reaches the RGBE kernel without numeric recoding."""
    frame = px.io.from_array(cp.ones((1, 8, 3), dtype=cp.float32), colorspace="Rec.709", gamma="linear", channels="RGB")

    def fail_recode(*args: object, **kwargs: object) -> px.core.Frame:
        raise AssertionError("native float32 must not be recoded")

    monkeypatch.setattr("pixtreme._values.cast.recode_dtype", fail_recode)

    assert px.io.write_image(tmp_path / "native.hdr", frame) is None


@pytest.mark.parametrize(
    ("shape", "channels", "observed"),
    (
        ((1, 8, 4), "RGBA", "RGB"),
        ((1, 8, 3), ("R", "R", "B"), "unique"),
        ((1, 7, 3), "RGB", "width"),
        ((1, 32768, 3), "RGB", "width"),
    ),
    ids=("rgba", "duplicate", "too-narrow", "too-wide"),
)
def test_hdr_write_rejects_non_rgb_layouts_and_widths_outside_new_style(
    tmp_path: Path, shape: tuple[int, int, int], channels: str | tuple[str, ...], observed: str
) -> None:
    """v1-hdr acceptance 6: writer layout and new-style width are actionable closed sets."""
    frame = px.io.from_array(cp.ones(shape, dtype=cp.float32), colorspace="Rec.709", gamma="linear", channels=channels)
    path = tmp_path / "invalid.hdr"

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.write_image(path, frame)

    assert observed in str(error.value)
    assert not path.exists()


def test_hdr_write_read_matches_encoded_rgbe_decode_oracle(tmp_path: Path) -> None:
    """v1-hdr acceptance 3, 6, and 11: round-trip expectation derives from independent encode/decode equations."""
    values = np.array(
        [[[0.0, 0.5, 1.0], [2.0, 1.0, 0.25], [8.0, 0.0, 4.0], [1e-4, 2e-4, 3e-4]]],
        dtype=np.float32,
    )
    values = np.resize(values, (1, 8, 3))
    frame = px.io.from_array(cp.asarray(values), colorspace="Rec.709", gamma="linear", channels="RGB")
    path = tmp_path / "roundtrip.hdr"

    px.io.write_image(path, frame)
    actual = px.io.read_image(path)

    expected = _decode_rgbe_oracle(_encode_rgbe_oracle(values))
    np.testing.assert_array_equal(
        px.io.to_array(
            actual,
        ).get(),
        expected,
    )


def test_hdr_read_header_is_gpu_free_and_preserves_the_public_model(tmp_path: Path) -> None:
    """v1-hdr acceptance 7: pure CPU header probing reports HDR fp32 RGB without public field changes."""
    path = tmp_path / "header.hdr"
    path.write_bytes(_header(width=8, height=5, variables=("EXPOSURE=2.0",)))
    script = """
import sys
import pixtreme as px
h = px.io.read_header(sys.argv[1])
assert (h.format, h.width, h.height) == ("HDR", 8, 5)
assert h.parts[0].channels == {"R": "float32", "G": "float32", "B": "float32"}
assert (h.color.colorspace, h.color.gamma, h.color.mappable) == ("Rec.709", "linear", True)
assert h.color.raw == {"EXPOSURE": ("2.0",)}
assert set(px.io.ImageHeader.model_fields) == {"format", "width", "height", "parts", "color"}
assert "nvidia.nvimgcodec" not in sys.modules
assert "OpenEXR" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_hdr_remains_outside_bytes_boundaries_and_encode_options(tmp_path: Path) -> None:
    """v1-hdr acceptance 8: HDR stays file-only and accepts no raster/EXR encode option."""
    payload = _header(width=8, height=1) + np.zeros((1, 8, 4), dtype=np.uint8).tobytes()
    frame = px.io.from_array(cp.ones((1, 8, 3), dtype=cp.float32), colorspace="Rec.709", gamma="linear", channels="RGB")

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.decode_image(payload)
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(frame, format="hdr")
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "options.hdr", frame, quality=90)
