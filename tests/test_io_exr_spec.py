"""Specification tests for OpenEXR image I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from generate_io_fixtures import write_exr, write_multipart_exr

import pixtreme as px

ACES_CHROMATICITIES = (
    0.7347,
    0.2653,
    0.0,
    1.0,
    0.0001,
    -0.077,
    0.32168,
    0.33767,
)


def _assert_actionable(error: pytest.ExceptionInfo[BaseException]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _rgb(dtype: type[np.generic] = np.float16) -> dict[str, np.ndarray]:
    return {
        "R": np.array([[0.0, 0.25], [0.5, 1.0]], dtype=dtype),
        "G": np.array([[1.0, 0.5], [0.25, 0.0]], dtype=dtype),
        "B": np.array([[0.125, 0.25], [0.5, 0.75]], dtype=dtype),
    }


def test_read_exr_defaults_to_rgb_float32_and_preserves_half_unchanged(tmp_path: Path) -> None:
    """v1-io acceptance 3, 4, 6, and 9: EXR defaults select RGB and promote HALF unless unchanged."""
    path = tmp_path / "rgb.EXR"
    channels = _rgb(np.float16)
    write_exr(path, channels)

    normalized = px.io.read_image(path)
    unchanged = px.io.read_image(path, unchanged=True)

    expected = np.stack([channels[label] for label in "RGB"], axis=2)
    assert (normalized.dtype, normalized.colorspace, normalized.gamma, normalized.channels) == (
        np.dtype(np.float32),
        "ACES2065-1",
        "linear",
        ("R", "G", "B"),
    )
    assert unchanged.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        px.io.to_array(
            normalized,
        ).get(),
        expected.astype(np.float32),
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            unchanged,
        ).get(),
        expected,
    )


def test_read_exr_keeps_alpha_and_selects_dotted_channels_by_sequence(tmp_path: Path) -> None:
    """v1-io acceptance 3 and 8: EXR keeps A by default and supports ordered dotted-name selection."""
    path = tmp_path / "multilayer.exr"
    channels = {
        **_rgb(np.float32),
        "A": np.ones((2, 2), dtype=np.float32),
        "diffuse.R": np.full((2, 2), 4.0, dtype=np.float32),
        "diffuse.G": np.full((2, 2), 5.0, dtype=np.float32),
    }
    write_exr(path, channels)

    default = px.io.read_image(path)
    selected = px.io.read_image(path, channels=["diffuse.G", "diffuse.R"], unchanged=True)

    assert default.channels == ("R", "G", "B", "A")
    assert selected.channels == ("diffuse.G", "diffuse.R")
    np.testing.assert_array_equal(
        px.io.to_array(
            selected,
        ).get()[0, 0],
        np.array([5.0, 4.0], dtype=np.float32),
    )


def test_read_exr_without_rgb_requires_explicit_selection_and_names_read_header(tmp_path: Path) -> None:
    """v1-io acceptance 8 and 9: non-RGB EXR requires selection and the error points to header inspection."""
    path = tmp_path / "depth.exr"
    write_exr(path, {"Z": np.ones((2, 3), dtype=np.float32)})

    with pytest.raises(ValueError, match=r"why=.*what=.*how=.*read_header"):
        px.io.read_image(path)
    result = px.io.read_image(path, channels="Z")
    assert result.channels == ("Z",)


def test_true_multipart_exr_resolves_unique_and_qualified_names_and_rejects_ambiguity(tmp_path: Path) -> None:
    """v1-io acceptance 10: multi-part EXR uses unique naked names and part-qualified collision resolution."""
    path = tmp_path / "multipart.exr"
    write_multipart_exr(
        path,
        (
            (
                "beauty",
                {"R": np.full((2, 2), 1.0, np.float32), "G": np.full((2, 2), 2.0, np.float32)},
                {},
            ),
            (
                "utility",
                {"R": np.full((2, 2), 3.0, np.float32), "Z": np.full((2, 2), 4.0, np.float32)},
                {},
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"ambiguous.*beauty\.R"):
        px.io.read_image(path, channels="R")
    selected = px.io.read_image(path, channels=["utility.Z", "beauty.G", "beauty.R"])

    assert selected.channels == ("utility.Z", "beauty.G", "beauty.R")
    np.testing.assert_array_equal(
        px.io.to_array(
            selected,
        ).get()[0, 0],
        np.array([4.0, 2.0, 1.0], dtype=np.float32),
    )


def test_true_multipart_exr_rejects_selected_channels_with_different_dimensions(tmp_path: Path) -> None:
    """v1-io acceptance 10: selected channels from differently sized parts fail before stacking."""
    path = tmp_path / "dimensions.exr"
    write_multipart_exr(
        path,
        (
            ("small", {"R": np.zeros((2, 2), np.float32)}, {}),
            ("large", {"G": np.zeros((3, 2), np.float32)}, {}),
        ),
    )

    with pytest.raises(ValueError, match=r"why=.*dimensions.*what=.*how="):
        px.io.read_image(path, channels=["small.R", "large.G"])


def test_exr_mixed_channel_types_promote_by_default_and_reject_unchanged(tmp_path: Path) -> None:
    """v1-io acceptance 11: mixed HALF/FLOAT is absorbed into float32 only on the default path."""
    path = tmp_path / "mixed.exr"
    channels = {
        "R": np.ones((2, 2), np.float16),
        "G": np.ones((2, 2), np.float32),
        "B": np.ones((2, 2), np.float16),
    }
    write_exr(path, channels)

    assert px.io.read_image(path).dtype == np.dtype(np.float32)
    with pytest.raises(ValueError, match=r"why=.*mixed.*what=.*how=.*unchanged=False"):
        px.io.read_image(path, unchanged=True)


def test_exr_uint32_channels_use_literal_float32_default_and_native_unchanged_reads(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 10 and 11: UINT reads are literal fp32 or native uint32."""
    path = tmp_path / "uint.exr"
    values = np.asarray([[0, 1], [16777217, 4294967295]], dtype=np.uint32)
    write_exr(path, {label: values for label in "RGB"})

    default = px.io.read_image(path)
    unchanged = px.io.read_image(path, unchanged=True)

    assert default.dtype == np.dtype(np.float32)
    assert unchanged.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(
        px.io.to_array(
            default,
        ).get(),
        np.repeat(values[..., None].astype(np.float32), 3, axis=2),
    )
    np.testing.assert_array_equal(
        px.io.to_array(
            unchanged,
        ).get(),
        np.repeat(values[..., None], 3, axis=2),
    )


def test_exr_file_color_metadata_maps_and_per_call_claims_override_it(tmp_path: Path) -> None:
    """v1-io acceptance 5, 6, and 7: EXR ACES/chromaticities metadata maps below per-call claims."""
    path = tmp_path / "aces.exr"
    write_exr(
        path,
        _rgb(np.float16),
        header={"chromaticities": ACES_CHROMATICITIES, "acesImageContainerFlag": 1},
    )

    file_claim = px.io.read_image(path)
    call_claim = px.io.read_image(path, colorspace="ACEScg", gamma="Gamma-2.2")
    header = px.io.read_header(path)

    assert (file_claim.colorspace, file_claim.gamma) == ("ACES2065-1", "linear")
    assert (call_claim.colorspace, call_claim.gamma) == ("ACEScg", "Gamma-2.2")
    assert header.color.colorspace == "ACES2065-1"
    assert header.color.mappable is True
    assert "chromaticities" in header.color.raw and "acesImageContainerFlag" in header.color.raw


def test_read_header_exr_reports_every_part_channel_type_and_data_window_dimensions(tmp_path: Path) -> None:
    """v1-io acceptance 10, 17, and provisional 2/5: EXR header exposes all parts without pixel arrays."""
    path = tmp_path / "header.exr"
    write_multipart_exr(
        path,
        (
            ("half", {"R": np.zeros((2, 3), np.float16)}, {}),
            ("float", {"Z": np.zeros((2, 3), np.float32)}, {}),
        ),
    )

    header = px.io.read_header(path)

    assert (header.format, header.width, header.height) == ("EXR", 3, 2)
    assert tuple((part.name, part.channels) for part in header.parts) == (
        ("half", {"R": "float16"}),
        ("float", {"Z": "float32"}),
    )


@pytest.mark.parametrize(("dtype", "expected_dtype"), (("float16", "float16"), ("float32", "float16")))
def test_write_exr_uses_default_half_storage_and_writes_mappable_chromaticities(
    tmp_path: Path,
    dtype: str,
    expected_dtype: str,
) -> None:
    """v1-exr-runtime-independence acceptance 13: default EXR output uses HALF for float Frames."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    values = np.array([[[0.0, 0.25, 1.0], [2.0, -0.5, 0.75]]], dtype=dtype)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")
    path = tmp_path / f"output-{dtype}.exr"

    result = px.io.write_image(path, frame)

    assert result is None
    round_trip = px.io.read_image(path, unchanged=True, colorspace="ACEScg")
    assert round_trip.dtype == np.dtype(expected_dtype)
    np.testing.assert_array_equal(
        px.io.to_array(
            round_trip,
        ).get(),
        values.astype(expected_dtype),
    )
    exr_header = OpenEXR.File(str(path), header_only=True).header()
    assert "chromaticities" in exr_header


def test_write_exr_preserves_custom_channel_names(tmp_path: Path) -> None:
    """v1-io acceptance 8 and 14: EXR output uses Frame channel labels as file channel names."""
    import cupy as cp

    values = np.array([[[1.0, 2.0]]], dtype=np.float32)
    frame = px.io.from_array(
        cp.asarray(values),
        colorspace="ACES2065-1",
        gamma="linear",
        channels=["diffuse.R", "depth.Z"],
    )
    path = tmp_path / "custom.exr"

    px.io.write_image(path, frame)

    result = px.io.read_image(path, channels=["depth.Z", "diffuse.R"])
    assert result.channels == ("depth.Z", "diffuse.R")
    np.testing.assert_array_equal(
        px.io.to_array(
            result,
        ).get()[0, 0],
        np.array([2.0, 1.0], np.float32),
    )


def test_write_exr_rejects_duplicate_channel_labels_with_actionable_context(tmp_path: Path) -> None:
    """REQ-API-012: duplicate EXR output labels name the observed channels and the unique-label recovery."""
    import cupy as cp

    frame = px.io.from_array(
        cp.zeros((1, 1, 2), dtype=cp.float32),
        colorspace="ACES2065-1",
        gamma="linear",
        channels=["R", "R"],
    )

    with pytest.raises(ValueError) as error:
        px.io.write_image(tmp_path / "duplicate.exr", frame)

    _assert_actionable(error)
    assert "channels=('R', 'R')" in str(error.value)
