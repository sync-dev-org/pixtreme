"""Specification tests for mixed-dtype OpenEXR channel writing."""

from __future__ import annotations

import inspect
import math
import os
import struct
from collections.abc import Sequence
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from repository_contracts import require_repo_file

import pixtreme as px
from pixtreme._core.vocabulary import ExrCompression

_ACTIONABLE = r"why=.*what=.*how="
_COMPRESSIONS = ("none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab")
_LOSSLESS = frozenset(("none", "rle", "zip", "zips", "piz"))
_DWA_COLOR_UNITS = (("R", "G", "B"), ("Y",), ("BY",), ("RY",))
# Both DWA encoders transmit HALF DCT coefficients. At this fixture's declared
# [-0.2, 1.6] range, two binary16 unit-roundoff steps cover coefficient-choice
# and inverse-transfer rounding without admitting a finite zero replacement.
_DWA_DECODE_ATOL = np.float32(2.0 * np.finfo(np.float16).eps)
_EXR_MAGIC = 20000630
_LONG_NAMES_FLAG = 0x00000400
_ROOT = Path(__file__).resolve().parents[1]


def _frame(
    values: np.ndarray,
    channels: Sequence[str],
    *,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    matrix: str | None = None,
) -> px.core.Frame:
    return px.io.from_array(
        cp.asarray(np.ascontiguousarray(values)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix=matrix,
    )


def _mixed_frames(*, height: int = 17, width: int = 19) -> tuple[tuple[px.core.Frame, ...], dict[str, np.ndarray]]:
    y, x = np.mgrid[:height, :width]
    half = np.stack(
        (
            (x - 3.0) / 16.0,
            (y + 1.0) / 13.0,
            (x + y + 2.0) / 23.0,
        ),
        axis=2,
    ).astype(np.float16)
    floating_y = ((x * 0.03125 - y * 0.015625) + 0.25).astype(np.float32)
    floating_a = ((x + 2 * y + 1) / np.float32(height + 2 * width)).astype(np.float32)
    floating = np.stack((floating_y, floating_a), axis=2)
    object_ids = np.resize(
        np.asarray((0, 1, 2**24 - 1, 2**24, 2**24 + 1, 2**32 - 1), dtype=np.uint32),
        (height, width, 1),
    )
    ids = np.concatenate((object_ids, np.bitwise_xor(object_ids, np.uint32(0xA5A5A5A5))), axis=2)
    frames = (
        _frame(half, ("R", "G", "B")),
        _frame(floating, ("layer.Y", "matte.A"), gamma="sRGB", matrix="native"),
        _frame(ids, ("object_id", "id.A"), gamma="Gamma-2.4", matrix="BT.709"),
    )
    expected = {
        "R": half[..., 0],
        "G": half[..., 1],
        "B": half[..., 2],
        "layer.Y": floating_y,
        "matte.A": floating_a,
        "object_id": object_ids[..., 0],
        "id.A": ids[..., 1],
    }
    return frames, expected


def _invalid_frame(values: np.ndarray, channels: tuple[str, ...]) -> px.core.Frame:
    return px.core.Frame.model_construct(
        data=cp.asarray(np.ascontiguousarray(values)),
        colorspace="ACEScg",
        gamma="linear",
        channels=channels,
        matrix=None,
    )


def _cstring(payload: bytes, offset: int) -> tuple[str, int]:
    end = payload.index(b"\x00", offset)
    return payload[offset:end].decode("utf-8"), end + 1


def _wire_header(path: Path) -> tuple[int, dict[str, tuple[str, bytes]]]:
    payload = path.read_bytes()
    magic, version = struct.unpack_from("<II", payload)
    assert magic == _EXR_MAGIC
    attributes: dict[str, tuple[str, bytes]] = {}
    offset = 8
    while payload[offset] != 0:
        name, offset = _cstring(payload, offset)
        attribute_type, offset = _cstring(payload, offset)
        size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        attributes[name] = (attribute_type, payload[offset : offset + size])
        offset += size
    return version, attributes


def _wire_channels(path: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    _, attributes = _wire_header(path)
    attribute_type, payload = attributes["channels"]
    assert attribute_type == "chlist"
    result: list[tuple[str, int, int, int, int]] = []
    offset = 0
    while payload[offset] != 0:
        name, offset = _cstring(payload, offset)
        pixel_type, p_linear, x_sampling, y_sampling = struct.unpack_from("<iB3xii", payload, offset)
        offset += 16
        result.append((name, pixel_type, p_linear, x_sampling, y_sampling))
    assert offset + 1 == len(payload)
    return tuple(result)


def _assert_unchanged(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    if expected.dtype == np.float16:
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    elif expected.dtype == np.float32:
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    else:
        np.testing.assert_array_equal(actual, expected)


def _assert_dwa_decode_matches_oracle(actual: np.ndarray, oracle: np.ndarray) -> None:
    assert actual.dtype == oracle.dtype
    np.testing.assert_allclose(actual, oracle, rtol=0.0, atol=_DWA_DECODE_ATOL)


def test_write_exr_channels_is_the_only_new_public_operation_with_exact_signature() -> None:
    """v1-exr-mixed-dtype-write acceptance 1: the file-only mixed EXR API has one exact public path."""
    signature = inspect.signature(px.io.write_exr_channels, eval_str=True)

    assert tuple(signature.parameters) == ("path", "frames", "compression", "dwa_level")
    assert signature.parameters["path"].annotation == str | os.PathLike[str]
    assert signature.parameters["frames"].annotation == Sequence[px.core.Frame]
    assert signature.parameters["compression"].annotation == ExrCompression | None
    assert signature.parameters["compression"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["compression"].default is None
    assert signature.parameters["dwa_level"].annotation == float | None
    assert signature.parameters["dwa_level"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["dwa_level"].default is None
    assert signature.return_annotation is None
    assert px.io.__all__.count("write_exr_channels") == 1
    assert len(tuple(name for name in px.io.__all__ if name != "ImageHeader")) == 27
    assert not hasattr(px, "write_exr_channels")
    assert not hasattr(px.io, "encode_exr_channels")


@pytest.mark.parametrize("count", (1, 3))
def test_write_exr_channels_accepts_one_or_multiple_frames(tmp_path: Path, count: int) -> None:
    """v1-exr-mixed-dtype-write acceptance 2-3: every nonempty Frame sequence uses literal storage dtype."""
    frames, expected = _mixed_frames()
    selected = frames[:count]
    path = tmp_path / f"frames-{count}.exr"

    result = px.io.write_exr_channels(path, selected, compression="none")

    assert result is None
    from openexr_dev_oracle import read_frame

    actual = read_frame(path)
    assert tuple(actual) == tuple(sorted(label for frame in selected for label in frame.channels))
    for label, values in actual.items():
        _assert_unchanged(values, expected[label])


def test_mixed_header_is_single_part_native_and_independently_parseable(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 5 and 7: wire metadata is canonical per channel."""
    frames, _ = _mixed_frames()
    path = tmp_path / "mixed-header.exr"

    px.io.write_exr_channels(path, frames, compression="none")

    version, attributes = _wire_header(path)
    assert version & ~_LONG_NAMES_FLAG == 2
    assert _wire_channels(path) == (
        ("B", 1, 0, 1, 1),
        ("G", 1, 0, 1, 1),
        ("R", 1, 0, 1, 1),
        ("id.A", 0, 0, 1, 1),
        ("layer.Y", 2, 0, 1, 1),
        ("matte.A", 2, 0, 1, 1),
        ("object_id", 0, 0, 1, 1),
    )
    assert struct.unpack("<iiii", attributes["dataWindow"][1]) == (0, 0, 18, 16)
    assert struct.unpack("<iiii", attributes["displayWindow"][1]) == (0, 0, 18, 16)
    assert attributes["lineOrder"] == ("lineOrder", b"\x00")
    assert px.io.read_header(path).parts[0].channels == {
        "B": "float16",
        "G": "float16",
        "R": "float16",
        "id.A": "uint32",
        "layer.Y": "float32",
        "matte.A": "float32",
        "object_id": "uint32",
    }


def test_grouping_sequence_metadata_and_channel_order_do_not_change_file_bytes(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 6 and 8: labels and samples alone own canonical file identity."""
    frames, _ = _mixed_frames(height=5, width=7)
    rgb, floating, ids = frames
    split_rgb = tuple(
        _frame(
            cp.asnumpy(rgb.data[..., index : index + 1]),
            (label,),
            gamma=("sRGB", "Gamma-2.2", "linear")[index],
            matrix=("BT.601", "BT.709", "native")[index],
        )
        for index, label in enumerate(rgb.channels)
    )
    first = tmp_path / "grouped.exr"
    second = tmp_path / "split-and-permuted.exr"

    px.io.write_exr_channels(first, (rgb, floating, ids), compression="zip")
    px.io.write_exr_channels(second, (ids, split_rgb[2], floating, split_rgb[0], split_rgb[1]), compression="ZIP")

    assert first.read_bytes() == second.read_bytes()


def test_colorspace_metadata_and_compression_defaults_match_the_existing_exr_writer(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 8-9: common colorspace and omitted options use EXR defaults."""
    from openexr_dev_oracle import OpenEXR

    half = _frame(
        np.ones((8, 9, 1), dtype=np.float16),
        ("R",),
        colorspace="ACES2065-1",
        gamma="linear",
    )
    ids = _frame(
        np.ones((8, 9, 1), dtype=np.uint32),
        ("object_id",),
        colorspace="ACES2065-1",
        gamma="Gamma-2.4",
        matrix="BT.709",
    )
    zip_path = tmp_path / "default-zip.exr"
    dwa_path = tmp_path / "default-dwaa.exr"

    px.io.write_exr_channels(zip_path, (ids, half))
    px.io.write_exr_channels(dwa_path, (half, ids), compression="DWAA")

    zip_header = OpenEXR.File(str(zip_path), header_only=True).header()
    dwa_header = OpenEXR.File(str(dwa_path), header_only=True).header()
    assert zip_header["compression"] == OpenEXR.ZIP_COMPRESSION
    assert dwa_header["compression"] == OpenEXR.DWAA_COMPRESSION
    assert float(dwa_header["dwaCompressionLevel"]) == 45.0
    assert zip_header["acesImageContainerFlag"] == dwa_header["acesImageContainerFlag"] == 1
    expected_chromaticities = (
        0.7347,
        0.2653,
        0.0,
        1.0,
        0.0001,
        -0.077,
        0.32168,
        0.33767,
    )
    observed = zip_header["chromaticities"]
    assert tuple(observed) == pytest.approx(expected_chromaticities)


@pytest.mark.parametrize("compression", _COMPRESSIONS)
def test_every_codec_writes_three_pixel_types_and_preserves_uint_ids(tmp_path: Path, compression: str) -> None:
    """v1-exr-mixed-dtype-write acceptance 9-12: all codecs consume mixed descriptors and preserve UINT bits."""
    from openexr_dev_oracle import read_frame, write_frames

    frames, expected = _mixed_frames(height=18, width=19)
    actual_path = tmp_path / f"actual-{compression}.exr"
    oracle_path = tmp_path / f"oracle-{compression}.exr"
    dwa_level = 45.0 if compression in ("dwaa", "dwab") else None

    px.io.write_exr_channels(actual_path, frames, compression=compression, dwa_level=dwa_level)
    write_frames(oracle_path, frames, compression=compression, dwa_level=dwa_level)

    actual = read_frame(actual_path)
    oracle = read_frame(oracle_path)
    assert {values.dtype.name for values in actual.values()} == {"float16", "float32", "uint32"}
    _assert_unchanged(actual["object_id"], expected["object_id"])
    if compression in _LOSSLESS:
        for label, values in actual.items():
            _assert_unchanged(values, expected[label])
    elif compression not in ("dwaa", "dwab"):
        for label in actual:
            _assert_unchanged(actual[label], oracle[label])
    else:
        for label in ("object_id", "id.A", "matte.A"):
            _assert_unchanged(actual[label], expected[label])
        for label in ("R", "G", "B", "layer.Y"):
            _assert_dwa_decode_matches_oracle(actual[label], oracle[label])
        with pytest.raises(AssertionError):
            _assert_dwa_decode_matches_oracle(np.zeros_like(oracle["layer.Y"]), oracle["layer.Y"])


@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize("dtype", (np.float16, np.float32), ids=("half", "float"))
@pytest.mark.parametrize("channels", _DWA_COLOR_UNITS, ids=("rgb", "y", "by", "ry"))
def test_dwa_mixed_color_routes_match_the_existing_coefficient_oracle(
    tmp_path: Path,
    compression: str,
    dtype: type[np.floating],
    channels: tuple[str, ...],
) -> None:
    """v1-exr-mixed-dtype-write acceptance 11 and 17: every lossy suffix consumes the Phase 2 oracle."""
    from openexr_dev_oracle import read_frame
    from test_io_exr_gpu_phase2_write_spec import (
        _candidate_file_coefficients,
        _minimum_population_half,
        _oracle_forward_coefficients,
        _oracle_quantization_tables,
    )

    height, width = 32, 8
    y, x = np.mgrid[:height, :width]
    color_planes = (
        np.float32(-0.2) + x / np.float32(17.0) + y / np.float32(71.0),
        np.float32(0.7) - x / np.float32(23.0) + y / np.float32(89.0),
        np.float32(1.6) - x / np.float32(29.0) - y / np.float32(97.0),
    )
    color_values = np.stack(color_planes[: len(channels)], axis=2).astype(dtype)
    lossless_dtype = np.float32 if dtype is np.float16 else np.float16
    lossless_values = np.stack(
        (
            np.float32(0.25) + (x + y) / np.float32(128.0),
            np.float32(-0.5) + x / np.float32(31.0),
        ),
        axis=2,
    ).astype(lossless_dtype)
    ids = np.stack(
        (
            np.resize(np.asarray((0, 2**24 + 1, 2**32 - 1), dtype=np.uint32), (height, width)),
            np.resize(np.asarray((7, 2**24, 0xA5A5A5A5), dtype=np.uint32), (height, width)),
        ),
        axis=2,
    )
    frames = (
        _frame(color_values, channels),
        _frame(lossless_values, ("matte.A", "other.Z")),
        _frame(ids, ("id.A", "object_id")),
    )
    path = tmp_path / f"mixed-{compression}-{np.dtype(dtype).name}-{channels[0]}.exr"

    px.io.write_exr_channels(path, frames, compression=compression, dwa_level=45.0)

    actual_coefficients = _candidate_file_coefficients(path)
    source_coefficients = _oracle_forward_coefficients(color_values)
    luminance, chroma = _oracle_quantization_tables(45.0)
    expected_coefficients = source_coefficients.astype(np.float16).view(np.uint16)
    tolerances = (luminance, chroma, chroma) if len(channels) == 3 else (luminance,)
    for block in range(source_coefficients.shape[0]):
        for component, component_tolerances in enumerate(tolerances):
            for position, tolerance in enumerate(component_tolerances):
                expected_coefficients[block, component, position] = _minimum_population_half(
                    float(source_coefficients[block, component, position]), float(tolerance)
                )

    np.testing.assert_array_equal(actual_coefficients, expected_coefficients)
    decoded = read_frame(path)
    for index, label in enumerate(("matte.A", "other.Z")):
        _assert_unchanged(decoded[label], lossless_values[..., index])
    for index, label in enumerate(("id.A", "object_id")):
        _assert_unchanged(decoded[label], ids[..., index])


def test_lossless_codecs_preserve_special_float_bit_patterns(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 10: lossless mixed files retain signed zero, subnormal, NaN, and infinity."""
    half_bits = np.asarray((0x0000, 0x8000, 0x0001, 0x7C00, 0xFC00, 0x7E01), dtype=np.uint16)
    float_bits = np.asarray((0x00000000, 0x80000000, 0x00000001, 0x7F800000, 0xFF800000, 0x7FC00001), dtype=np.uint32)
    half = np.resize(half_bits.view(np.float16), (2, 3, 1))
    floating = np.resize(float_bits.view(np.float32), (2, 3, 1))
    ids = np.resize(np.asarray((2**24 - 1, 2**24, 2**24 + 1, 2**32 - 1), dtype=np.uint32), (2, 3, 1))
    frames = (_frame(half, ("H",)), _frame(floating, ("F",)), _frame(ids, ("object_id",)))

    from openexr_dev_oracle import read_frame

    for compression in _LOSSLESS:
        path = tmp_path / f"special-{compression}.exr"
        px.io.write_exr_channels(path, frames, compression=compression)
        actual = read_frame(path)
        _assert_unchanged(actual["H"], half[..., 0])
        _assert_unchanged(actual["F"], floating[..., 0])
        _assert_unchanged(actual["object_id"], ids[..., 0])


def test_mixed_readback_keeps_existing_asymmetric_selection_contract(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 12-13: exact IDs require a homogeneous unchanged selection."""
    frames, expected = _mixed_frames(height=3, width=6)
    path = tmp_path / "readback.exr"
    px.io.write_exr_channels(path, frames, compression="zip")

    default = px.io.read_image(path)
    mixed = px.io.read_image(path, channels=("R", "G", "B", "object_id"))
    exact_id = px.io.read_image(path, channels=("object_id",), unchanged=True)

    assert default.channels == ("R", "G", "B")
    assert default.dtype == np.dtype("float32")
    assert mixed.dtype == np.dtype("float32")
    np.testing.assert_array_equal(cp.asnumpy(exact_id.data[..., 0]), expected["object_id"])
    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.read_image(path, channels=("R", "object_id"), unchanged=True)
    assert "float16" in str(error.value) and "uint32" in str(error.value)


@pytest.mark.parametrize("frames", (None, "RGB", b"RGB", bytearray(b"RGB"), (), (object(),), iter(())))
def test_frames_validation_happens_before_file_creation(tmp_path: Path, frames: object) -> None:
    """v1-exr-mixed-dtype-write acceptance 2 and 14: invalid Frame containers fail before file creation."""
    path = tmp_path / "invalid-frames.exr"

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_exr_channels(path, frames)  # type: ignore[arg-type]

    assert not path.exists()


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16))
def test_non_native_exr_dtypes_explain_literal_and_normalized_preparation(
    tmp_path: Path, dtype: type[np.generic]
) -> None:
    """v1-exr-mixed-dtype-write acceptance 3 and 14: rejected dtypes distinguish cast from recode preparation."""
    frame = _frame(np.zeros((2, 3, 1), dtype=dtype), ("object_id",))
    path = tmp_path / f"invalid-{np.dtype(dtype).name}.exr"

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.write_exr_channels(path, (frame,))

    message = str(error.value)
    assert "frame_index=0" in message and np.dtype(dtype).name in message
    assert "cast_dtype" in message and "uint32" in message
    assert "recode_dtype" in message and "float16" in message and "float32" in message
    assert not path.exists()


def test_shape_colorspace_and_duplicate_validation_precedes_truncation(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 4-5, 8, and 14: cross-Frame conflicts preserve existing files."""
    base = _frame(np.zeros((2, 3, 1), dtype=np.float16), ("same",))
    cases = (
        (_frame(np.zeros((3, 3, 1), dtype=np.float16), ("other",)), ("frame_index=1", "shape=")),
        (
            _frame(np.zeros((2, 3, 1), dtype=np.float16), ("other",), colorspace="ACES2065-1"),
            ("frame_index=1", "colorspace="),
        ),
        (_frame(np.zeros((2, 3, 1), dtype=np.float32), ("same",)), ("same", "frame_index=0", "frame_index=1")),
    )
    for index, (other, fragments) in enumerate(cases):
        path = tmp_path / f"preserved-{index}.exr"
        path.write_bytes(b"keep-me")
        with pytest.raises(ValueError, match=_ACTIONABLE) as error:
            px.io.write_exr_channels(path, (base, other))
        assert all(fragment in str(error.value) for fragment in fragments)
        assert path.read_bytes() == b"keep-me"


def test_device_mismatch_is_rejected_before_writing_when_two_visible_devices_exist(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 4 and 14: cross-device input is never copied implicitly."""
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two visible CUDA devices; remove when the single-device CI lane changes")
    with cp.cuda.Device(0):
        first = _frame(np.zeros((2, 3, 1), dtype=np.float16), ("A",))
    with cp.cuda.Device(1):
        second = _frame(np.zeros((2, 3, 1), dtype=np.float16), ("B",))
    path = tmp_path / "cross-device.exr"

    with pytest.raises(ValueError, match=_ACTIONABLE) as error:
        px.io.write_exr_channels(path, (first, second))

    assert "frame_index=1" in str(error.value) and "device" in str(error.value)
    assert not path.exists()


@pytest.mark.parametrize(
    ("label", "accepted"),
    (
        ("a" * 31, True),
        ("a" * 32, True),
        ("\ud800", False),
        ("nul\x00label", False),
        ("é" * 127 + "a", True),
        ("é" * 128, False),
    ),
    ids=("31-bytes", "32-bytes", "not-utf8", "nul", "255-bytes", "256-bytes"),
)
def test_channel_label_byte_boundaries_and_long_name_flag(tmp_path: Path, label: str, accepted: bool) -> None:
    """v1-exr-mixed-dtype-write acceptance 5 and 14: UTF-8 byte limits and the long-name flag are exact."""
    frame = _invalid_frame(np.zeros((1, 1, 1), dtype=np.float16), (label,))
    path = tmp_path / "label.exr"

    if not accepted:
        with pytest.raises(ValueError, match=_ACTIONABLE):
            px.io.write_exr_channels(path, (frame,), compression="none")
        assert not path.exists()
        return

    px.io.write_exr_channels(path, (frame,), compression="none")
    version, _ = _wire_header(path)
    assert bool(version & _LONG_NAMES_FLAG) is (len(label.encode("utf-8")) > 31)


@pytest.mark.parametrize("compression", ("gzip", 1, True))
def test_compression_validation_precedes_truncation(tmp_path: Path, compression: object) -> None:
    """v1-exr-mixed-dtype-write acceptance 9 and 14: compression remains a shared closed vocabulary."""
    frames, _ = _mixed_frames(height=2, width=3)
    path = tmp_path / "invalid-compression.exr"
    path.write_bytes(b"keep-me")

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_exr_channels(path, frames, compression=compression)  # type: ignore[arg-type]

    assert path.read_bytes() == b"keep-me"


@pytest.mark.parametrize(
    ("compression", "dwa_level"),
    (
        ("zip", 45.0),
        ("dwaa", 0.0),
        ("dwaa", math.inf),
        ("dwab", 45),
        ("dwab", True),
        ("dwab", "45"),
    ),
)
def test_dwa_validation_precedes_truncation(tmp_path: Path, compression: str, dwa_level: object) -> None:
    """v1-exr-mixed-dtype-write acceptance 9 and 14: DWA level type and converted range match write_image."""
    frames, _ = _mixed_frames(height=2, width=3)
    path = tmp_path / "invalid-dwa.exr"
    path.write_bytes(b"keep-me")

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_exr_channels(path, frames, compression=compression, dwa_level=dwa_level)  # type: ignore[arg-type]

    assert path.read_bytes() == b"keep-me"


def test_path_and_post_validation_io_failures_follow_public_error_contract(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 14: path validation is fail-fast and later I/O preserves its cause."""
    frames, _ = _mixed_frames(height=2, width=3)
    wrong_extension = tmp_path / "mixed.png"
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_exr_channels(wrong_extension, frames)
    assert not wrong_extension.exists()

    missing_parent = tmp_path / "missing" / "mixed.exr"
    with pytest.raises(RuntimeError, match=_ACTIONABLE) as error:
        px.io.write_exr_channels(missing_parent, frames, compression="none")
    assert isinstance(error.value.__cause__, OSError)
    assert not missing_parent.exists()


def test_cuda_runtime_failure_is_actionable_and_preserves_its_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-mixed-dtype-write acceptance 14: CUDA failures are classified before crossing the public boundary."""
    import pixtreme._io.formats.exr.mixed as exr_mixed

    frames, _ = _mixed_frames(height=2, width=3)
    path = tmp_path / "cuda-failure.exr"
    backend_error = cp.cuda.runtime.CUDARuntimeError(2)

    def fail(*args: object, **kwargs: object) -> tuple[cp.ndarray, int]:
        raise backend_error

    monkeypatch.setattr(exr_mixed, "_pack_channel_rows", fail)

    with pytest.raises(RuntimeError, match=_ACTIONABLE) as error:
        px.io.write_exr_channels(path, frames, compression="none")

    assert type(error.value) is RuntimeError
    assert error.value.__cause__ is backend_error
    assert not path.exists()


def test_inputs_are_not_mutated_by_mixed_write(tmp_path: Path) -> None:
    """v1-exr-mixed-dtype-write acceptance 8: writing preserves every input array and metadata field."""
    frames, _ = _mixed_frames(height=4, width=5)
    snapshots = tuple(
        (
            cp.asnumpy(frame.data).copy(),
            frame.dtype,
            frame.colorspace,
            frame.gamma,
            frame.channels,
            frame.matrix,
            frame.data.device.id,
            frame.data.data.ptr,
        )
        for frame in frames
    )

    px.io.write_exr_channels(tmp_path / "immutable.exr", frames, compression="zip")

    for frame, snapshot in zip(frames, snapshots, strict=True):
        data, dtype, colorspace, gamma, channels, matrix, device, pointer = snapshot
        np.testing.assert_array_equal(cp.asnumpy(frame.data), data)
        assert (frame.dtype, frame.colorspace, frame.gamma, frame.channels, frame.matrix) == (
            dtype,
            colorspace,
            gamma,
            channels,
            matrix,
        )
        assert (frame.data.device.id, frame.data.data.ptr) == (device, pointer)


def test_public_docstring_describes_mixed_exr_boundary_contract() -> None:
    """v1-exr-mixed-dtype-write acceptance 15: the public docstring carries the invisible boundary contract."""
    docstring = inspect.getdoc(px.io.write_exr_channels) or ""
    for fragment in (
        "Sequence[Frame]",
        "same shape",
        "same CUDA device",
        "same colorspace",
        "float16",
        "float32",
        "uint32",
        "cast_dtype",
        "recode_dtype",
        "compression",
        "gamma",
        "matrix",
        "read_image",
        "unchanged=True",
        "file-only",
        "ValueError",
        "RuntimeError",
    ):
        assert fragment in docstring


def test_canonical_docs_describe_the_mixed_exr_public_boundary() -> None:
    """v1-exr-mixed-dtype-write acceptance 15; v1-fonts-module acceptance 1 and 14;
    v1-grade acceptance 1: API canon is current.
    """
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    tokens = (_ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    io_feature = require_repo_file("docs/features/v1-io.md").read_text(encoding="utf-8")

    assert "| `io` | file / bytes / device array / wire format の from・to 境界、ImageHeader | 27 |" in requirements
    assert "公開 operation は計 98 関数" in requirements
    assert "`px.io.write_image` / `px.io.write_exr_channels`" in requirements
    for fragment in (
        "px.io.write_exr_channels(",
        "frames: Sequence[Frame]",
        "compression: ExrCompression | None = None",
        "dwa_level: float | None = None",
        "`float16` | HALF",
        "`float32` | FLOAT",
        "`uint32` | UINT",
        "`none` / `rle` / `zip` / `zips` / `piz`",
        "px.values.cast_dtype",
        "px.values.recode_dtype",
        "unchanged=True",
    ):
        assert fragment in tokens
    assert "`Sequence[Frame]` を受ける `px.io.write_exr_channels`" in io_feature
    assert "`write_image` / `encode_image` の dtype 変換契約は変えない" in io_feature
