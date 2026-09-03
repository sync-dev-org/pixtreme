"""Specification tests for selectable OpenEXR write compression."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import pixtreme as px

_ACTIONABLE = r"why=.*what=.*how="


def _smooth_frame(dtype: type[np.generic] = np.float32) -> tuple[px.core.Frame, np.ndarray]:
    import cupy as cp

    height = width = 32
    y, x = np.mgrid[:height, :width]
    values = np.stack(
        (
            0.125 + x / 8.0,
            0.25 + y / 12.0,
            0.5 + (x + y) / 16.0,
        ),
        axis=2,
    ).astype(dtype)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")
    return frame, values


def _exr_header(path: Path) -> dict[str, object]:
    from openexr_dev_oracle import OpenEXR

    image = OpenEXR.File(str(path), header_only=True)
    return dict(image.header())


def test_write_image_signature_appends_optional_dwa_level() -> None:
    """v1-exr-runtime-independence acceptance 1; v1-exr-write-compression acceptance 1:
    write dtype and dwa_level follow the existing optional file keywords.
    """
    signature = inspect.signature(px.io.write_image)

    assert tuple(signature.parameters) == (
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
    parameter = signature.parameters["dwa_level"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_exr_compression_tokens_are_a_case_sensitive_closed_set() -> None:
    """v1-exr-write-compression acceptance 2-3: the ten public EXR compression tokens are exact."""
    from pixtreme._io.common import _EXR_COMPRESSION_TOKENS

    assert _EXR_COMPRESSION_TOKENS == (
        "none",
        "rle",
        "zip",
        "zips",
        "piz",
        "pxr24",
        "b44",
        "b44a",
        "dwaa",
        "dwab",
    )


@pytest.mark.parametrize("compression", ("gzip", "lzw", "htj2k256", 1, True))
def test_exr_compression_rejects_unknown_tokens_and_non_strings_before_writing(
    tmp_path: Path, compression: object
) -> None:
    """v1-exr-write-compression acceptance 3: invalid compression fails fast with actionable context."""
    frame, _ = _smooth_frame()
    path = tmp_path / "invalid.exr"

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(path, frame, compression=compression)  # type: ignore[arg-type]

    assert not path.exists()


def test_exr_default_compression_remains_zip(tmp_path: Path) -> None:
    """v1-exr-write-compression acceptance 2: omitting compression preserves the ZIP write default."""
    from openexr_dev_oracle import OpenEXR

    frame, _ = _smooth_frame()
    path = tmp_path / "default.exr"

    px.io.write_image(path, frame)

    assert _exr_header(path)["compression"] == OpenEXR.ZIP_COMPRESSION


@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize(("dwa_level", "expected"), ((None, 45.0), (23.5, 23.5)))
def test_dwa_level_defaults_and_explicit_values_are_written_to_the_header(
    tmp_path: Path,
    compression: str,
    dwa_level: float | None,
    expected: float,
) -> None:
    """v1-exr-write-compression acceptance 4: DWA writes the fixed default or explicit positive level."""
    frame, _ = _smooth_frame()
    path = tmp_path / f"{compression}-{expected}.exr"

    px.io.write_image(path, frame, compression=compression, dwa_level=dwa_level)

    assert _exr_header(path)["dwaCompressionLevel"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("dwa_level", "expected"),
    (
        (1e-45, float(np.nextafter(np.float32(0.0), np.float32(1.0)))),
        (
            math.nextafter(float(np.finfo(np.float32).max), math.inf),
            float(np.finfo(np.float32).max),
        ),
    ),
    ids=("rounds-up-to-smallest-subnormal", "rounds-down-to-float32-max"),
)
def test_dwa_level_accepts_values_that_round_to_a_positive_finite_header_float(
    tmp_path: Path,
    dwa_level: float,
    expected: float,
) -> None:
    """v1-exr-write-compression acceptance 4: validity follows the converted OpenEXR header float."""
    frame, _ = _smooth_frame()
    path = tmp_path / "rounded-level.exr"

    px.io.write_image(path, frame, compression="dwaa", dwa_level=dwa_level)

    assert _exr_header(path)["dwaCompressionLevel"] == expected


@pytest.mark.parametrize(
    "dwa_level",
    (0.0, -1.0, math.inf, -math.inf, math.nan, 1e308, 5e-324, 45, True, "45"),
    ids=(
        "zero",
        "negative",
        "inf",
        "negative-inf",
        "nan",
        "float32-overflow",
        "float32-underflow",
        "int",
        "bool",
        "str",
    ),
)
def test_dwa_level_requires_an_exact_positive_finite_float(tmp_path: Path, dwa_level: object) -> None:
    """v1-exr-write-compression acceptance 4: DWA levels must stay finite after OpenEXR float conversion."""
    frame, _ = _smooth_frame()

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "invalid-level.exr", frame, compression="dwaa", dwa_level=dwa_level)  # type: ignore[arg-type]


@pytest.mark.parametrize("compression", (None, "none", "zip", "piz", "pxr24", "b44a"))
def test_dwa_level_is_rejected_for_every_non_dwa_exr_compression(tmp_path: Path, compression: str | None) -> None:
    """v1-exr-write-compression acceptance 4: an explicit DWA level cannot accompany non-DWA output."""
    frame, _ = _smooth_frame()

    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "not-dwa.exr", frame, compression=compression, dwa_level=45.0)


@pytest.mark.parametrize(
    ("compression", "constant_name", "dtype", "oracle"),
    (
        ("none", "NO_COMPRESSION", np.float32, "exact"),
        ("rle", "RLE_COMPRESSION", np.float32, "exact"),
        ("zip", "ZIP_COMPRESSION", np.float32, "exact"),
        ("zips", "ZIPS_COMPRESSION", np.float32, "exact"),
        ("piz", "PIZ_COMPRESSION", np.float32, "exact"),
        ("pxr24", "PXR24_COMPRESSION", np.float32, "pxr24"),
        ("pxr24", "PXR24_COMPRESSION", np.float16, "exact"),
        ("b44", "B44_COMPRESSION", np.float16, "lossy-finite"),
        ("b44", "B44_COMPRESSION", np.float32, "exact"),
        ("b44a", "B44A_COMPRESSION", np.float16, "lossy-finite"),
        ("b44a", "B44A_COMPRESSION", np.float32, "exact"),
        ("dwaa", "DWAA_COMPRESSION", np.float32, "lossy-finite"),
        ("dwab", "DWAB_COMPRESSION", np.float32, "lossy-finite"),
    ),
)
def test_every_exr_compression_selects_its_header_and_round_trips_with_the_correct_oracle(
    tmp_path: Path,
    compression: str,
    constant_name: str,
    dtype: type[np.generic],
    oracle: str,
) -> None:
    """v1-exr-runtime-independence acceptance 44; v1-exr-write-compression acceptance 5-6:
    every token controls the header and preserves each supported documented lossy class.

    The OpenEXR header reader is independent of pixtreme's write implementation.
    PXR24 FLOAT uses the official Technical Introduction's approximately 3e-5
    relative-error characterization with a 4e-5 test margin. Codecs without an
    official numeric bound are checked only for successful finite lossy decode.
    """
    from openexr_dev_oracle import OpenEXR

    frame, expected = _smooth_frame(dtype)
    path = tmp_path / f"{compression}.exr"

    px.io.write_image(path, frame, compression=compression, dtype=np.dtype(dtype).name)
    actual = px.io.to_array(
        px.io.read_image(path, unchanged=True, colorspace="ACEScg", gamma="linear"),
    ).get()

    assert _exr_header(path)["compression"] == getattr(OpenEXR, constant_name)
    if oracle == "exact":
        np.testing.assert_array_equal(actual, expected)
    elif oracle == "pxr24":
        relative_error = np.abs(actual - expected) / np.abs(expected)
        assert float(relative_error.max()) <= 4e-5
        assert np.any(actual != expected)
    else:
        assert np.all(np.isfinite(actual))
        assert np.any(actual != expected)


def test_exr_and_raster_encode_options_remain_format_specific(tmp_path: Path) -> None:
    """v1-exr-write-compression acceptance 7-9: EXR, TIFF, PNG, and bytes options do not mix."""
    import cupy as cp

    exr_frame, _ = _smooth_frame()
    raster_frame = px.io.from_array(
        cp.arange(4 * 4 * 3, dtype=cp.uint8).reshape(4, 4, 3),
        colorspace="sRGB",
        gamma="sRGB",
        channels="RGB",
    )

    for compression in ("none", "lzw"):
        px.io.write_image(tmp_path / f"valid-{compression}.tiff", raster_frame, compression=compression)
    for compression in ("zip", "dwaa"):
        with pytest.raises(ValueError, match=_ACTIONABLE):
            px.io.write_image(tmp_path / f"invalid-{compression}.tiff", raster_frame, compression=compression)
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "invalid-lzw.exr", exr_frame, compression="lzw")
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "invalid-level.exr", exr_frame, compression_level=4)
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.write_image(tmp_path / "invalid-dwa.png", raster_frame, dwa_level=45.0)

    assert "dwa_level" not in inspect.signature(px.io.encode_image).parameters
    with pytest.raises(TypeError):
        px.io.encode_image(raster_frame, format="png", dwa_level=45.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match=_ACTIONABLE):
        px.io.encode_image(exr_frame, format="exr")
