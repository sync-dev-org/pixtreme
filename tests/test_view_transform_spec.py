"""Specification and mathematical-oracle tests for the rendering transform."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px
import pixtreme._color.view_transform as view_transform_module

ROOT = Path(__file__).resolve().parents[1]

_COMBINATIONS = (
    ("aces-1.3-lut", "Rec.709", "bt1886"),
    ("aces-1.3-lut", "sRGB", "srgb"),
    ("aces-2.0-lut", "Rec.709", "bt1886"),
    ("aces-2.0-lut", "sRGB", "srgb"),
)
_LUT_FILENAMES = (
    "view_transform_aces-1.3_rec709_bt1886.npz",
    "view_transform_aces-1.3_srgb_srgb.npz",
    "view_transform_aces-2.0_rec709_bt1886.npz",
    "view_transform_aces-2.0_srgb_srgb.npz",
)


def _assert_actionable(error: pytest.ExceptionInfo[BaseException]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _frame(
    values: np.ndarray | list[float],
    *,
    colorspace: str = "ACES2065-1",
    gamma: str = "linear",
    channels: str | list[str] = "RGB",
    dtype: np.dtype[np.generic] | type[np.generic] = np.float32,
) -> px.core.Frame:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    elif array.ndim == 2:
        array = array.reshape(1, *array.shape)
    return px.io.from_array(cp.asarray(array), colorspace=colorspace, gamma=gamma, channels=channels)


def _tetrahedral_reference(lut: np.ndarray, point: tuple[float, float, float]) -> np.ndarray:
    position = np.clip(np.asarray(point, dtype=np.float64), 0.0, 1.0) * (lut.shape[0] - 1)
    lower = np.minimum(np.floor(position).astype(np.int64), lut.shape[0] - 2)
    fractions = position - lower
    axes = sorted(range(3), key=lambda axis: -fractions[axis])
    first = lower.copy()
    first[axes[0]] += 1
    second = first.copy()
    second[axes[1]] += 1
    upper = lower + 1
    c000 = lut[tuple(lower)]
    c1 = lut[tuple(first)]
    c2 = lut[tuple(second)]
    c111 = lut[tuple(upper)]
    return c000 + fractions[axes[0]] * (c1 - c000) + fractions[axes[1]] * (c2 - c1) + fractions[axes[2]] * (c111 - c2)


@pytest.mark.parametrize("case", ("frame", "dtype", "channels", "tonemap", "combination"))
def test_lut_tonemap_public_boundary_failures_are_actionable(case: str) -> None:
    """v1-color-semantics acceptance 29-30 and 33: LUT rendering failures stay on rgb_to_rgb."""
    frame: object = _frame([0.1, 0.2, 0.3])
    kwargs = {
        "input_colorspace": None,
        "input_gamma": None,
        "output_colorspace": "Rec.709",
        "output_gamma": "bt1886",
        "tonemap": "aces-1.3-lut",
    }
    if case == "frame":
        frame = object()
    elif case == "dtype":
        frame = _frame([0.1, 0.2, 0.3], dtype=np.float16)
    elif case == "channels":
        frame = _frame([0.1, 0.2], channels="RG")
    elif case == "tonemap":
        kwargs["tonemap"] = "aces-latest"
    else:
        kwargs["output_colorspace"] = "Rec.2020"
        kwargs["output_gamma"] = "pq"

    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(frame, **kwargs)  # type: ignore[arg-type]
    _assert_actionable(error)
    assert case in str(error.value) or case == "frame" and "Frame" in str(error.value)


def test_retired_view_transform_name_and_dead_validation_constants_are_absent() -> None:
    """v1-color-semantics acceptance 33: the retired view_transform route leaves no alias or shim."""
    assert not hasattr(px.color, "view_transform")
    assert not hasattr(view_transform_module, "view_transform")
    for name in ("_RGB_CHANNELS", "_VERSION_TOKENS", "_INTERNAL_LUT_COMBINATIONS"):
        assert not hasattr(view_transform_module, name)


@pytest.mark.parametrize("corruption", ("metadata", "shape"))
def test_view_transform_lut_archive_corruption_fails_actionably(
    corruption: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-API-012: corrupt LUT identity and shape failures explain the archive repair contract."""
    combination = ("aces-1.3", "Rec.709", "bt1886")
    filename = view_transform_module._LUT_FILES[combination]
    stored = ("aces-2.0", "Rec.709", "bt1886") if corruption == "metadata" else combination
    np.savez(
        tmp_path / filename,
        version=stored[0],
        output_colorspace=stored[1],
        output_gamma=stored[2],
        shaper=np.zeros((1,), dtype=np.float32),
        shaper_domain=np.zeros((1,), dtype=np.float32),
        lut=np.zeros((1, 1, 1, 3), dtype=np.float32),
    )
    monkeypatch.setattr(view_transform_module, "files", lambda _package: tmp_path)
    view_transform_module._load_lut_identity.cache_clear()
    try:
        with pytest.raises(RuntimeError) as error:
            view_transform_module._load_lut_identity(*combination)
    finally:
        view_transform_module._load_lut_identity.cache_clear()
    _assert_actionable(error)
    assert corruption in str(error.value) or (corruption == "shape" and "shape" in str(error.value))


@pytest.mark.parametrize(("version", "output_colorspace", "output_gamma"), _COMBINATIONS)
def test_every_documented_version_and_sdr_output_combination_is_supplied(
    version: str, output_colorspace: str, output_gamma: str
) -> None:
    """v1-tonemap-aces13-analytic acceptance 2 and 10: the four LUT/output combinations are supplied."""
    result = px.color.rgb_to_rgb(
        _frame([0.18, 0.18, 0.18]),
        output_colorspace=output_colorspace,
        output_gamma=output_gamma,
        tonemap=version,
    )

    assert (result.colorspace, result.gamma) == (output_colorspace, output_gamma)
    assert np.isfinite(
        px.io.to_array(
            result,
        ).get()
    ).all()


def test_unknown_tonemap_and_unsupported_output_combinations_fail_fast() -> None:
    """v1-tonemap-aces13-analytic acceptance 3: unknown tokens and unsupported SDR exits fail with guidance."""
    source = _frame([0.18, 0.18, 0.18])

    with pytest.raises(ValueError, match="aces-latest"):
        px.color.rgb_to_rgb(
            source,
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-latest",
        )

    for colorspace, gamma in (("Rec.709", "srgb"), ("sRGB", "bt1886"), ("Rec.2020", "pq")):
        with pytest.raises(ValueError, match="combination"):
            px.color.rgb_to_rgb(
                source,
                output_colorspace=colorspace,
                output_gamma=gamma,
                tonemap="aces-2.0-lut",
            )


def test_input_claims_override_metadata_without_mutating_the_source() -> None:
    """v1-tonemap-aces13-analytic acceptance 4 and 10: per-call claims win without mutating the source."""
    mislabeled = _frame([0.18, 0.18, 0.18], colorspace="sRGB", gamma="srgb")
    original = (
        px.io.to_array(
            mislabeled,
        )
        .get()
        .copy()
    )
    reference = _frame([0.18, 0.18, 0.18])

    result = px.color.rgb_to_rgb(
        mislabeled,
        input_colorspace="ACES2065-1",
        input_gamma="linear",
        output_colorspace="sRGB",
        output_gamma="srgb",
        tonemap="aces-2.0-lut",
    )
    expected = px.color.rgb_to_rgb(
        reference,
        output_colorspace="sRGB",
        output_gamma="srgb",
        tonemap="aces-2.0-lut",
    )

    np.testing.assert_array_equal(
        px.io.to_array(
            mislabeled,
        ).get(),
        original,
    )
    assert (mislabeled.colorspace, mislabeled.gamma, mislabeled.channels) == ("sRGB", "srgb", ("R", "G", "B"))
    assert (result.colorspace, result.gamma, result.channels) == ("sRGB", "srgb", ("R", "G", "B"))
    np.testing.assert_allclose(
        px.io.to_array(
            result,
        ).get(),
        px.io.to_array(
            expected,
        ).get(),
        rtol=0.0,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "input_colorspace",
    ("sRGB", "Rec.709", "Rec.2020", "ACES2065-1", "ACEScg", "S-Gamut3", "S-Gamut3.Cine"),
)
def test_every_colorspace_token_is_accepted_as_an_input_claim(input_colorspace: str) -> None:
    """v1-tonemap-aces13-analytic acceptance 2 and 4: every colorspace token may feed the analytic path."""
    result = px.color.rgb_to_rgb(
        _frame([0.18, 0.18, 0.18]),
        input_colorspace=input_colorspace,
        output_colorspace="sRGB",
        output_gamma="srgb",
        tonemap="aces-1.3",
    )
    assert np.isfinite(
        px.io.to_array(
            result,
        ).get()
    ).all()


@pytest.mark.parametrize(
    "input_gamma",
    ("linear", "srgb", "rec709", "bt1886", "pq", "hlg", "s-log3", "logc4", "cineon", "2.2", "2.4", "2.6"),
)
def test_every_gamma_token_is_accepted_as_an_input_claim(input_gamma: str) -> None:
    """v1-tonemap-aces13-analytic acceptance 2 and 4: every gamma token may feed the analytic path."""
    result = px.color.rgb_to_rgb(
        _frame([0.18, 0.18, 0.18]),
        input_gamma=input_gamma,
        output_colorspace="Rec.709",
        output_gamma="bt1886",
        tonemap="aces-1.3",
    )
    assert np.isfinite(
        px.io.to_array(
            result,
        ).get()
    ).all()


@pytest.mark.parametrize(("parameter", "value"), (("input_colorspace", "ACES"), ("input_gamma", "gamma2.4")))
def test_unknown_input_axis_tokens_fail_fast(parameter: str, value: str) -> None:
    """v1-color-semantics acceptance 29-30: input claims use the case-sensitive public vocabulary."""
    with pytest.raises(ValueError, match=parameter):
        px.color.rgb_to_rgb(
            _frame([0.18, 0.18, 0.18]),
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-1.3",
            **{parameter: value},
        )


@pytest.mark.parametrize("filename", _LUT_FILENAMES)
def test_prebaked_lut_path_matches_the_external_ocio_oracle(filename: str) -> None:
    """v1-tonemap-aces13-analytic acceptance 10-12: renamed LUT tokens retain the baked OCIO oracle."""
    with np.load(ROOT / "src" / "pixtreme" / "data" / filename, allow_pickle=False) as archive:
        source = _frame(archive["oracle_input"])
        result = px.color.rgb_to_rgb(
            source,
            output_colorspace=str(archive["output_colorspace"].item()),
            output_gamma=str(archive["output_gamma"].item()),
            tonemap=f"{archive['version'].item()}-lut",
        )
        tolerance = float(archive["oracle_tolerance"].item())

        assert tolerance == pytest.approx(5e-3, rel=0.0, abs=1e-9)
        np.testing.assert_allclose(
            np.clip(
                px.io.to_array(
                    result,
                )
                .get()
                .reshape(-1, 3),
                0.0,
                1.0,
            ),
            archive["oracle_output"],
            rtol=0.0,
            atol=tolerance,
        )


@pytest.mark.parametrize(
    ("point", "expected"),
    (
        ((0.75, 0.50, 0.25), 0.28515625),
        ((0.75, 0.25, 0.50), 0.31640625),
        ((0.50, 0.25, 0.75), 0.32812500),
        ((0.50, 0.75, 0.25), 0.28906250),
        ((0.25, 0.75, 0.50), 0.38281250),
        ((0.25, 0.50, 0.75), 0.39062500),
    ),
)
def test_tetrahedral_kernel_matches_each_hand_calculated_ordering_for_a_two_cube(
    point: tuple[float, float, float], expected: float
) -> None:
    """v1-color-semantics acceptance 31 and 36: all tetrahedra match a hand-computed fixture."""
    import pixtreme._color.view_transform as implementation

    scalar_vertices = np.asarray(
        (
            ((0.0, 4.0), (2.0, 32.0)),
            ((1.0, 16.0), (8.0, 64.0)),
        ),
        dtype=np.float32,
    ) / np.float32(64.0)
    lut = np.repeat(scalar_vertices[..., None], 3, axis=-1)

    result = implementation._apply_lut_data(
        cp.asarray(np.asarray(point, dtype=np.float32).reshape(1, 1, 3)),
        ("R", "G", "B"),
        shaper=cp.asarray((0.0, 1.0), dtype=cp.float32),
        shaper_domain=(-0.0, 1.0),
        lut=cp.asarray(lut),
        output_gamma="linear",
    )

    np.testing.assert_allclose(cp.asnumpy(result)[0, 0], (expected, expected, expected), rtol=0.0, atol=2e-7)


def test_tetrahedral_kernel_selects_the_correct_cell_in_a_three_cube() -> None:
    """v1-color-semantics acceptance 31 and 36: a hand-computed fixture fixes cell indexing and weights."""
    import pixtreme._color.view_transform as implementation

    indices = np.indices((3, 3, 3), dtype=np.float32)
    scalar = (indices[0] ** 2 + 2.0 * indices[1] ** 2 + 4.0 * indices[2] ** 2) / 28.0
    lut = np.repeat(scalar[..., None], 3, axis=-1)
    point = np.asarray((0.625, 0.375, 0.875), dtype=np.float32).reshape(1, 1, 3)

    result = implementation._apply_lut_data(
        cp.asarray(point),
        ("R", "G", "B"),
        shaper=cp.asarray((0.0, 1.0), dtype=cp.float32),
        shaper_domain=(0.0, 1.0),
        lut=cp.asarray(lut),
        output_gamma="linear",
    )

    expected = 16.25 / 28.0
    np.testing.assert_allclose(cp.asnumpy(result)[0, 0], (expected, expected, expected), rtol=0.0, atol=2e-7)


@pytest.mark.parametrize(
    "point",
    (
        (-0.25, 0.0, 0.0),
        (1.25, 1.0, 1.0),
        (0.75, 0.75, 0.25),
        (0.75, 0.25, 0.75),
        (0.25, 0.75, 0.75),
        (0.5, 0.5, 0.5),
    ),
)
def test_packed_tetrahedral_kernel_matches_independent_boundary_and_stable_tie_oracle(
    point: tuple[float, float, float],
) -> None:
    """v1-color-semantics acceptance 31-32 and 36: packed LUT boundaries and stable RGB ties use a float64 oracle."""
    lut = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) / np.float32(32.0)
    packed_lut = np.zeros((2, 2, 2, 4), dtype=np.float32)
    packed_lut[..., :3] = lut

    result = view_transform_module._apply_lut_data(
        cp.asarray(np.asarray(point, dtype=np.float32).reshape(1, 1, 3)),
        ("R", "G", "B"),
        shaper=cp.asarray((0.0, 1.0), dtype=cp.float32),
        shaper_domain=(0.0, 1.0),
        lut=cp.asarray(packed_lut),
        output_gamma="linear",
    )

    expected = _tetrahedral_reference(lut.astype(np.float64), point)
    np.testing.assert_allclose(cp.asnumpy(result)[0, 0], expected, rtol=0.0, atol=2e-7)


@pytest.mark.parametrize(
    "point",
    (
        (0.75, 0.75, 0.25),
        (0.75, 0.25, 0.25),
        (0.75, 0.25, 0.75),
    ),
)
def test_packed_tetrahedral_kernel_preserves_stable_rgb_tie_order_against_non_affine_oracle(
    point: tuple[float, float, float],
) -> None:
    """v1-color-semantics acceptance 31-32 and 36: exact ties preserve stable RGB order against float64."""
    lut = (np.sin(np.arange(24, dtype=np.float64) * 0.9 + 0.1) * 4.0).astype(np.float32).reshape(2, 2, 2, 3)
    packed_lut = np.zeros((2, 2, 2, 4), dtype=np.float32)
    packed_lut[..., :3] = lut

    result = view_transform_module._apply_lut_data(
        cp.asarray(np.asarray(point, dtype=np.float32).reshape(1, 1, 3)),
        ("R", "G", "B"),
        shaper=cp.asarray((0.0, 1.0), dtype=cp.float32),
        shaper_domain=(0.0, 1.0),
        lut=cp.asarray(packed_lut),
        output_gamma="linear",
    )

    expected = _tetrahedral_reference(lut.astype(np.float64), point)
    # This fixture's stable fp32 path stays within 1.79e-7; 2e-7 excludes every non-strict comparison mutant.
    np.testing.assert_allclose(cp.asnumpy(result)[0, 0], expected, rtol=0.0, atol=2e-7)


def test_generated_lut_kernels_share_one_tetrahedral_source_and_keep_the_packed_view_path() -> None:
    """v1-color-semantics acceptance 31-32: generated CUDA source identity and packed float4 load are structural."""
    import pixtreme._color.lut as lut_implementation
    from pixtreme._color._lut_cuda import _LUT_TETRAHEDRAL_CUDA_SOURCE

    sources = (lut_implementation._LUT_TRANSFORM_KERNEL_SOURCE, view_transform_module._VIEW_TRANSFORM_KERNEL)
    for source in sources:
        assert source.count(_LUT_TETRAHEDRAL_CUDA_SOURCE) == 1
        assert source.count("__device__ __forceinline__ float3 pixtreme_lut_tetrahedral(") == 1
    assert "reinterpret_cast<const float4*>(lut + offset)" in _LUT_TETRAHEDRAL_CUDA_SOURCE
    view_source = view_transform_module._VIEW_TRANSFORM_KERNEL
    call_start = view_source.index("const float3 linear = pixtreme_lut_tetrahedral(")
    packed_call = view_source[call_start : view_source.index(");", call_start) + 2]
    assert "const long long stride_blue = 4;" in view_source
    assert "reinterpret_cast<const float*>(lut)" in packed_call
    call_arguments = packed_call.split("pixtreme_lut_tetrahedral(", maxsplit=1)[1].rsplit(")", maxsplit=1)[0]
    argument_tokens = tuple(argument.strip() for argument in call_arguments.split(","))
    assert argument_tokens[-1] == "1"


def test_rendering_preserves_out_of_range_rgb_and_passes_non_rgb_labels_unchanged() -> None:
    """v1-color-semantics acceptance 29 and 32: rendering preserves scene values and auxiliary channels."""
    source = _frame(
        [9.0, 10.0, 0.7, 0.0, 0.0],
        channels=["Z", "R", "A", "G", "B"],
    )

    result = px.color.rgb_to_rgb(
        source,
        output_colorspace="sRGB",
        output_gamma="srgb",
        tonemap="aces-1.3",
    )
    output = px.io.to_array(
        result,
    ).get()[0, 0]

    assert result.channels == source.channels
    assert output[0] == np.float32(9.0)
    assert output[2] == np.float32(0.7)
    assert output[[1, 3, 4]].max() > 1.0
    assert output[[1, 3, 4]].min() < 0.0


@pytest.mark.parametrize("channels", ("YCbCr", "RG", ["R", "G", "A"]))
def test_tonemap_rejects_frames_without_all_rgb_labels(channels: str | list[str]) -> None:
    """v1-color-semantics acceptance 29-30: a missing RGB label fails before rendering."""
    source = _frame(np.zeros(len(px.core.channels(channels)), dtype=np.float32), channels=channels)

    with pytest.raises(ValueError, match="R, G, and B"):
        px.color.rgb_to_rgb(
            source,
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-1.3",
        )


@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        (np.float16, ("cast_dtype",)),
        (np.uint8, ("recode_dtype", "dequantize")),
        (np.uint16, ("recode_dtype", "dequantize")),
    ),
)
def test_tonemap_rejects_non_float32_with_the_dtype_specific_recipe(
    dtype: type[np.generic],
    routes: tuple[str, ...],
) -> None:
    """v1-color-semantics acceptance 29-30: tonemap errors retain dtype-specific conversion guidance."""
    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(
            _frame([0, 0, 0], dtype=dtype),
            output_colorspace="sRGB",
            output_gamma="srgb",
            tonemap="aces-1.3",
        )
    message = str(error.value)
    assert "float32" in message
    positions = tuple(message.index(route) for route in routes)
    assert positions == tuple(sorted(positions))


def test_bake_tool_recreates_all_four_archives_byte_for_byte(tmp_path: Path) -> None:
    """v1-tonemap-aces13-analytic acceptance 12: LUT baking reproduces all packaged archives byte-for-byte."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    command = [sys.executable, str(ROOT / "tools" / "bake_view_transform_luts.py"), "--output-dir"]

    subprocess.run([*command, str(first)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([*command, str(second)], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)

    first_files = sorted(path.name for path in first.glob("*.npz"))
    second_files = sorted(path.name for path in second.glob("*.npz"))
    assert first_files == second_files == sorted(_LUT_FILENAMES)
    for filename in first_files:
        packaged = ROOT / "src" / "pixtreme" / "data" / filename
        assert (first / filename).read_bytes() == (second / filename).read_bytes() == packaged.read_bytes()


def test_pixtreme_import_graph_has_no_runtime_ocio_dependency() -> None:
    """v1-color-semantics acceptance 31: importing the runtime package never imports PyOpenColorIO."""
    command = [
        sys.executable,
        "-c",
        "import sys; import pixtreme; print(any(name.startswith('PyOpenColorIO') for name in sys.modules))",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
    assert completed.stdout.strip() == "False"
