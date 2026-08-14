"""Specification tests for the generic device-array boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _host(array: Any) -> np.ndarray:
    import cupy as cp

    return cp.asnumpy(array)


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message


def _sample_hwc(*, dtype: Any = np.float32) -> np.ndarray:
    return np.arange(2 * 3 * 3, dtype=dtype).reshape(2, 3, 3)


def test_array_layout_and_dtype_errors_are_actionable() -> None:
    """REQ-API-012: layout rank/batch and dtype-token rejections identify why, the input, and a valid declaration."""
    import cupy as cp

    source = cp.asarray(_sample_hwc())
    calls = (
        lambda: px.io.from_array(source, colorspace="sRGB", gamma="linear", channels="RGB", layout="hwc"),
        lambda: px.io.from_array(source, colorspace="sRGB", gamma="linear", channels="RGB", dtype=32),
        lambda: px.io.from_array(source, colorspace="sRGB", gamma="linear", channels="RGB", dtype="float64"),
        lambda: px.io.from_array(
            cp.zeros((2, 3), dtype=cp.float32), colorspace="sRGB", gamma="linear", channels="RGB", layout="HWC"
        ),
        lambda: px.io.from_array(
            cp.zeros((2, 3, 3), dtype=cp.float32),
            colorspace="sRGB",
            gamma="linear",
            channels="RGB",
            layout="NHWC",
        ),
        lambda: px.io.from_array(
            cp.zeros((2, 2, 3, 3), dtype=cp.float32),
            colorspace="sRGB",
            gamma="linear",
            channels="RGB",
            layout="NHWC",
        ),
        lambda: px.io.from_array(
            cp.zeros((2, 3), dtype=cp.float32), colorspace="sRGB", gamma="linear", channels="RGB", layout="CHW"
        ),
        lambda: px.io.from_array(
            cp.zeros((3, 2, 3), dtype=cp.float32),
            colorspace="sRGB",
            gamma="linear",
            channels="RGB",
            layout="NCHW",
        ),
        lambda: px.io.from_array(
            cp.zeros((2, 3, 2, 3), dtype=cp.float32),
            colorspace="sRGB",
            gamma="linear",
            channels="RGB",
            layout="NCHW",
        ),
    )

    for call in calls:
        with pytest.raises(ValueError) as error:
            call()
        _assert_actionable(error)


def test_array_affine_value_errors_are_actionable() -> None:
    """REQ-API-012: each affine scalar/sequence rejection reports the received value and channel-count recovery."""
    import cupy as cp

    source = cp.asarray(_sample_hwc())
    invalid_values = (
        "invalid",
        np.void(b"invalid"),
        ("invalid", "invalid", "invalid"),
        ["invalid", "invalid", "invalid"],
        [1.0, 2.0],
    )
    for value in invalid_values:
        with pytest.raises(ValueError) as error:
            px.io.from_array(
                source,
                colorspace="sRGB",
                gamma="linear",
                channels="RGB",
                scale=value,
            )
        _assert_actionable(error)


def test_array_channel_and_storage_errors_are_actionable() -> None:
    """REQ-API-012: unavailable labels, channel-count mismatches, and unsupported storage dtypes provide recovery."""
    import cupy as cp

    source = cp.asarray(_sample_hwc())
    frame = px.io.from_array(source, colorspace="sRGB", gamma="linear", channels="RGB")
    calls = (
        lambda: px.io.to_array(frame, channels="A"),
        lambda: px.io.from_array(source, colorspace="sRGB", gamma="linear", channels="RGBA"),
        lambda: px.io.from_array(
            cp.asarray(_sample_hwc(dtype=np.float64)), colorspace="sRGB", gamma="linear", channels="RGB"
        ),
    )

    for call in calls:
        with pytest.raises(ValueError) as error:
            call()
        _assert_actionable(error)


def test_to_array_rejects_non_frame_input_actionably() -> None:
    """REQ-API-012: the generic Frame exit rejects non-Frame values at the public boundary with recovery."""
    with pytest.raises(ValueError) as error:
        px.io.to_array(None)  # type: ignore[arg-type]

    _assert_actionable(error)
    assert "builtins.NoneType" in str(error.value)
    assert "px.io.to_array" in str(error.value)


def test_to_array_docstring_names_the_function_boundary_and_public_quantize_path() -> None:
    """REQ-API-012: to_array documentation identifies the function input and an existing recovery API."""
    docstring = px.io.to_array.__doc__ or ""

    assert "Export a Frame" in docstring
    assert ":func:`pixtreme.values.quantize`" in docstring
    assert "Export this Frame" not in docstring


def test_array_bit_depth_option_errors_are_actionable() -> None:
    """REQ-API-012: import/export bit-depth conflicts identify the conflicting values and a valid recovery call."""
    import cupy as cp

    float_frame = px.io.from_array(
        cp.zeros((1, 1, 1), dtype=cp.float32), colorspace="sRGB", gamma="linear", channels="Y"
    )
    float16_frame = px.io.from_array(
        cp.zeros((1, 1, 1), dtype=cp.float16), colorspace="sRGB", gamma="linear", channels="Y"
    )
    uint16_source = cp.zeros((1, 1, 1), dtype=cp.uint16)
    calls = (
        lambda: px.io.to_array(float_frame, bit_depth=10, scale=1.0),
        lambda: px.io.to_array(float16_frame, bit_depth=10),
        lambda: px.io.to_array(float_frame, bit_depth=10, dtype="uint8"),
        lambda: px.io.from_array(
            uint16_source,
            colorspace="sRGB",
            gamma="linear",
            channels="Y",
            bit_depth=10,
            scale=1.0,
        ),
        lambda: px.io.from_array(
            cp.zeros((1, 1, 1), dtype=cp.uint8),
            colorspace="sRGB",
            gamma="linear",
            channels="Y",
            bit_depth=10,
        ),
        lambda: px.io.from_array(
            uint16_source,
            colorspace="sRGB",
            gamma="linear",
            channels="Y",
            bit_depth=10,
            dtype="float16",
        ),
    )

    for call in calls:
        with pytest.raises(ValueError) as error:
            call()
        _assert_actionable(error)


def test_from_array_wraps_hwc_zero_copy_and_treats_channels_as_an_interpretation_claim() -> None:
    """v1-boundary-api acceptance 1, 3, and 5: the sole constructor wraps HWC data without reordering it."""
    import cupy as cp

    source = cp.asarray(_sample_hwc())
    result = px.io.from_array(source, colorspace="sRGB", gamma="srgb", channels="BGR")

    assert isinstance(result, px.core.Frame)
    assert result.data.data.ptr == source.data.ptr
    assert result.channels == ("B", "G", "R")
    np.testing.assert_array_equal(_host(result.data), _sample_hwc())


def test_uint32_array_boundary_preserves_storage_and_zero_copy() -> None:
    """v1-exr-runtime-independence acceptance 7: uint32 crosses from_array/to_array without copying or recoding."""
    import cupy as cp

    source = cp.asarray(_sample_hwc(dtype=np.uint32))
    frame = px.io.from_array(source, colorspace="ACEScg", gamma="linear", channels="RGB", copy=False)
    output = px.io.to_array(frame, copy=False)

    assert frame.dtype == np.dtype(np.uint32)
    assert frame.data.data.ptr == source.data.ptr
    assert output.data.ptr == source.data.ptr
    np.testing.assert_array_equal(_host(output), _sample_hwc(dtype=np.uint32))


def test_from_array_rejects_host_arrays_with_an_actionable_device_recovery_path() -> None:
    """v1-boundary-api acceptance 2: host arrays fail with why/what/how guidance instead of implicit transfer."""
    source = _sample_hwc()

    with pytest.raises(ValueError) as error:
        px.io.from_array(source, colorspace="sRGB", gamma="srgb", channels="RGB")

    _assert_actionable(error)
    assert "CUDA" in str(error.value)
    assert "cp.asarray" in str(error.value)


def test_from_array_copy_tristate_controls_contiguity_and_private_ownership() -> None:
    """v1-boundary-api acceptance 4 and 5: copy None/False/True mean opportunistic, strict, and private."""
    import cupy as cp

    contiguous = cp.asarray(_sample_hwc())
    private = px.io.from_array(contiguous, colorspace="sRGB", gamma="srgb", channels="RGB", copy=True)
    assert private.data.data.ptr != contiguous.data.ptr
    np.testing.assert_array_equal(_host(private.data), _sample_hwc())

    non_contiguous = contiguous[:, ::2, :]
    copied = px.io.from_array(non_contiguous, colorspace="sRGB", gamma="srgb", channels="RGB")
    assert copied.data.flags.c_contiguous
    assert copied.data.data.ptr != non_contiguous.data.ptr
    np.testing.assert_array_equal(_host(copied.data), _host(non_contiguous))

    with pytest.raises(ValueError) as non_contiguous_error:
        px.io.from_array(
            non_contiguous,
            colorspace="sRGB",
            gamma="srgb",
            channels="RGB",
            copy=False,
        )
    _assert_actionable(non_contiguous_error)

    with pytest.raises(ValueError) as affine_error:
        px.io.from_array(
            contiguous,
            colorspace="sRGB",
            gamma="srgb",
            channels="RGB",
            scale=255.0,
            copy=False,
        )
    _assert_actionable(affine_error)

    with pytest.raises(ValueError) as dtype_error:
        px.io.from_array(
            contiguous,
            colorspace="sRGB",
            gamma="srgb",
            channels="RGB",
            dtype="float32",
            copy=False,
        )
    _assert_actionable(dtype_error)


@pytest.mark.parametrize("layout", ("HWC", "NHWC", "CHW", "NCHW"))
def test_from_array_normalizes_each_layout_against_an_independent_numpy_oracle(layout: str) -> None:
    """v1-boundary-api acceptance 6: each declared layout is normalized to contiguous HWC storage."""
    import cupy as cp

    expected = _sample_hwc()
    layouts = {
        "HWC": expected,
        "NHWC": expected[np.newaxis, ...],
        "CHW": np.transpose(expected, (2, 0, 1)).copy(),
        "NCHW": np.transpose(expected, (2, 0, 1))[np.newaxis, ...].copy(),
    }
    source = cp.asarray(layouts[layout])

    result = px.io.from_array(
        source,
        colorspace="sRGB",
        gamma="srgb",
        channels="RGB",
        layout=layout,
    )

    assert result.shape == expected.shape
    assert result.data.flags.c_contiguous
    np.testing.assert_array_equal(_host(result.data), expected)
    if layout in {"HWC", "NHWC"}:
        assert result.data.data.ptr == source.data.ptr
    else:
        assert result.data.data.ptr != source.data.ptr


def test_from_array_layout_rejects_batches_and_strict_transpose_copy() -> None:
    """v1-boundary-api acceptance 4 and 6: N must be one and transpose layouts cannot promise zero-copy."""
    import cupy as cp

    with pytest.raises(ValueError, match="N.*1"):
        px.io.from_array(
            cp.zeros((2, 2, 3, 3), dtype=cp.float32),
            colorspace="sRGB",
            gamma="srgb",
            channels="RGB",
            layout="NHWC",
        )

    chw = cp.asarray(np.transpose(_sample_hwc(), (2, 0, 1)).copy())
    with pytest.raises(ValueError) as error:
        px.io.from_array(chw, colorspace="sRGB", gamma="srgb", channels="RGB", layout="CHW", copy=False)
    _assert_actionable(error)


def test_from_array_affine_and_dtype_follow_the_inverse_fp32_numpy_oracle() -> None:
    """v1-boundary-api acceptance 7 and 8: import applies inverse per-channel affine in one destination pass."""
    import cupy as cp

    source = np.asarray(
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]],
        dtype=np.float16,
    )
    scale = np.asarray((2.0, 4.0, 8.0), dtype=np.float32)
    mean = np.asarray((0.5, 1.0, 1.5), dtype=np.float32)
    std = np.asarray((1.5, 2.0, 2.5), dtype=np.float32)
    expected = (source.astype(np.float32) * std + mean) / scale

    result = px.io.from_array(
        cp.asarray(np.transpose(source, (2, 0, 1)).copy()),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
        layout="CHW",
        dtype="float32",
        scale=scale,
        mean=mean,
        std=std,
    )

    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(_host(result.data), expected, rtol=0.0, atol=2e-7)


@pytest.mark.parametrize("bit_depth", (8, 10, 12, 14, 16))
def test_from_array_bit_depth_normalizes_the_declared_integer_grid(bit_depth: int) -> None:
    """v1-quantize-values acceptance 11: from_array divides the matching container by 2^B-1."""
    import cupy as cp

    maximum = (1 << bit_depth) - 1
    container = np.uint8 if bit_depth == 8 else np.uint16
    codes = np.asarray([0, maximum // 2, maximum], dtype=container).reshape(3, 1, 1)

    result = px.io.from_array(
        cp.asarray(codes),
        colorspace="Rec.709",
        gamma="rec709",
        channels=["low", "middle", "high"],
        layout="CHW",
        bit_depth=bit_depth,
    )

    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        _host(result.data).reshape(-1),
        codes.astype(np.float32).reshape(-1) * np.float32(1.0 / maximum),
    )


def test_from_array_bit_depth_matches_the_full_range_planar_format_entry() -> None:
    """v1-quantize-values acceptance 17: independent full-range planar and generic entries converge."""
    import cupy as cp

    width = 2
    height = 2
    planar = np.asarray(
        [
            0,
            256,
            512,
            1023,
            64,
            320,
            640,
            960,
            1023,
            768,
            384,
            0,
        ],
        dtype=np.uint16,
    )
    device = cp.asarray(planar)

    named = px.io.from_yuv444p(device, width=width, height=height, bit_depth=10, range="full")
    generic = px.io.from_array(
        device.reshape(3, height, width),
        colorspace="Rec.709",
        gamma="rec709",
        channels="YCbCr",
        layout="CHW",
        bit_depth=10,
    )

    np.testing.assert_array_equal(_host(generic.data), _host(named.data))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"bit_depth": 10, "scale": 1023.0},
        {"bit_depth": 10, "mean": 0.5},
        {"bit_depth": 10, "std": 2.0},
        {"bit_depth": 10, "dtype": "float16"},
        {"bit_depth": 10, "copy": False},
    ),
)
def test_from_array_bit_depth_rejects_affine_non_fp32_and_strict_zero_copy(
    kwargs: dict[str, object],
) -> None:
    """v1-quantize-values acceptance 11: the normalization path has one fp32 write and no affine composition."""
    import cupy as cp

    with pytest.raises(ValueError):
        px.io.from_array(
            cp.zeros((1, 1, 1), dtype=cp.uint16),
            colorspace="Rec.709",
            gamma="rec709",
            channels="Y",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("bit_depth", "dtype"),
    ((8, "uint16"), (10, "uint8"), (12, "float32"), (14, "float16")),
)
def test_from_array_bit_depth_requires_the_matching_uint_container(bit_depth: int, dtype: str) -> None:
    """v1-quantize-values acceptance 11 and REQ-API-012: declared code depth and input container must agree,
    and every offered recovery is valid for the dtype that reached the branch."""
    import cupy as cp

    expected = "uint8" if bit_depth == 8 else "uint16"
    with pytest.raises(ValueError, match=expected) as error:
        px.io.from_array(
            cp.zeros((1, 1, 1), dtype=dtype),
            colorspace="Rec.709",
            gamma="rec709",
            channels="Y",
            bit_depth=bit_depth,
        )
    message = str(error.value)
    assert "why=" in message
    assert "; what=" in message
    assert "; how=" in message
    if dtype.startswith("float"):
        assert "omit bit_depth" in message


def test_array_boundary_rejects_unknown_tokens_and_malformed_affine_sequences() -> None:
    """v1-boundary-api acceptance 6, 7, 12, and 14: named axes and per-channel constants fail fast."""
    import cupy as cp

    source = cp.asarray(_sample_hwc())
    for kwargs in ({"layout": "hwc"}, {"dtype": "float64"}, {"scale": (1.0, 2.0)}):
        with pytest.raises(ValueError):
            px.io.from_array(source, colorspace="sRGB", gamma="srgb", channels="RGB", **kwargs)

    frame = px.io.from_array(source, colorspace="sRGB", gamma="srgb", channels="RGB")
    for kwargs in ({"layout": "hwc"}, {"dtype": "float64"}, {"mean": (0.0, 1.0)}):
        with pytest.raises(ValueError):
            px.io.to_array(frame, **kwargs)


def test_to_array_default_and_nhwc_are_zero_copy_views_while_copy_true_is_private() -> None:
    """v1-boundary-api acceptance 9, 10, and 12: identity exits are views unless private ownership is requested."""
    import cupy as cp

    source = cp.asarray(_sample_hwc())
    frame = px.io.from_array(source, colorspace="sRGB", gamma="srgb", channels="RGB")

    view = px.io.to_array(
        frame,
    )
    nhwc = px.io.to_array(frame, layout="NHWC", copy=False)
    private = px.io.to_array(frame, copy=True)

    assert isinstance(view, cp.ndarray)
    assert callable(view.__dlpack__)
    assert view.data.ptr == frame.data.data.ptr
    assert nhwc.shape == (1, 2, 3, 3)
    assert nhwc.data.ptr == frame.data.data.ptr
    assert private.data.ptr != frame.data.data.ptr
    view[0, 0, 0] = np.float32(99.0)
    assert float(frame.data[0, 0, 0]) == pytest.approx(99.0)


def test_to_array_selects_channels_greedily_and_emits_each_layout() -> None:
    """v1-boundary-api acceptance 11 and 12: label selection and layout match an independent NumPy oracle."""
    import cupy as cp

    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    frame = px.io.from_array(
        cp.asarray(values),
        colorspace="Rec.709",
        gamma="rec709",
        channels=["Y", "Y", "Cr", "A"],
    )

    hwc = px.io.to_array(frame, channels=["Y", "Cr"])
    chw = px.io.to_array(frame, channels=["A", "Y"], layout="CHW")
    nchw = px.io.to_array(frame, channels=["A", "Y"], layout="NCHW")

    np.testing.assert_array_equal(_host(hwc), values[..., [0, 2]])
    expected_chw = np.transpose(values[..., [3, 0]], (2, 0, 1))
    np.testing.assert_array_equal(_host(chw), expected_chw)
    np.testing.assert_array_equal(_host(nchw), expected_chw[np.newaxis, ...])
    with pytest.raises(ValueError, match="Cb"):
        px.io.to_array(frame, channels=["Cb"])


def test_to_array_affine_round_trips_through_from_array_with_the_same_constants() -> None:
    """v1-boundary-api acceptance 7 and 13: export/import affine formulas are inverse with shared constants."""
    import cupy as cp

    values = np.linspace(-0.25, 1.25, 18, dtype=np.float32).reshape(2, 3, 3)
    scale = (255.0, 127.5, 63.75)
    mean = (127.5, 31.0, 15.0)
    std = (128.0, 64.0, 32.0)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")

    exported = px.io.to_array(frame, scale=scale, mean=mean, std=std)
    expected = (
        values.astype(np.float32) * np.asarray(scale, dtype=np.float32) - np.asarray(mean, dtype=np.float32)
    ) / np.asarray(std, dtype=np.float32)
    restored = px.io.from_array(
        exported,
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
        scale=scale,
        mean=mean,
        std=std,
    )

    np.testing.assert_allclose(_host(exported), expected, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(_host(restored.data), values, rtol=0.0, atol=3e-7)


def test_to_array_dtype_is_a_faithful_cast_without_rounding_or_clipping() -> None:
    """v1-boundary-api acceptance 14: dtype conversion follows CuPy cast semantics after fp32 affine."""
    import cupy as cp

    values = np.asarray([[[1.9, 260.0, 0.9]]], dtype=np.float32)
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="linear", channels="RGB")

    result = px.io.to_array(frame, dtype="uint8")

    np.testing.assert_array_equal(_host(result), values.astype(np.uint8))


@pytest.mark.parametrize("bit_depth", (8, 10, 12, 14, 16))
def test_to_array_bit_depth_quantizes_to_the_declared_integer_grid(bit_depth: int) -> None:
    """v1-quantize-values acceptance 12: to_array clips, rounds ties upward, and derives the container."""
    import cupy as cp

    maximum = (1 << bit_depth) - 1
    container = np.uint8 if bit_depth == 8 else np.uint16
    values = np.asarray([-1.0, 0.5 / maximum, 0.5, 1.0, 2.0], dtype=np.float32).reshape(1, 1, 5)
    frame = px.io.from_array(
        cp.asarray(values),
        colorspace="sRGB",
        gamma="linear",
        channels=[f"channel-{index}" for index in range(5)],
    )

    result = px.io.to_array(frame, bit_depth=bit_depth)

    assert result.dtype == np.dtype(container)
    np.testing.assert_array_equal(
        _host(result).reshape(-1),
        np.asarray([0, 1, (maximum + 1) // 2, maximum, maximum], dtype=container),
    )


def test_to_array_bit_depth_composes_with_channels_layout_out_and_copy() -> None:
    """v1-quantize-values acceptance 13: quantization composes with every independent export axis."""
    import cupy as cp

    values = np.asarray(
        [
            [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]],
            [[0.25, 0.75, 0.5], [0.75, 0.25, 0.5]],
        ],
        dtype=np.float32,
    )
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="linear", channels="RGB")
    out = cp.empty((2, 2, 2), dtype=cp.uint16)

    result = px.io.to_array(frame, channels="BR", layout="CHW", bit_depth=10, out=out)
    private = px.io.to_array(frame, channels="BR", layout="CHW", bit_depth=10, copy=True)
    expected = np.floor(np.clip(np.transpose(values[..., [2, 0]], (2, 0, 1)), 0.0, 1.0) * 1023.0 + 0.5).astype(
        np.uint16
    )

    assert result is out
    np.testing.assert_array_equal(_host(result), expected)
    np.testing.assert_array_equal(_host(private), expected)
    assert private.data.ptr != out.data.ptr


@pytest.mark.parametrize(
    "kwargs",
    (
        {"bit_depth": 10, "scale": 1023.0},
        {"bit_depth": 10, "mean": 0.5},
        {"bit_depth": 10, "std": 2.0},
        {"bit_depth": 10, "dtype": "uint8"},
        {"bit_depth": 10, "copy": False},
    ),
)
def test_to_array_bit_depth_rejects_affine_wrong_container_and_strict_zero_copy(
    kwargs: dict[str, object],
) -> None:
    """v1-quantize-values acceptance 12 and 13: the grid path is an explicit write with one derived dtype."""
    import cupy as cp

    frame = px.io.from_array(
        cp.zeros((1, 1, 1), dtype=cp.float32),
        colorspace="sRGB",
        gamma="linear",
        channels="Y",
    )

    with pytest.raises(ValueError):
        px.io.to_array(frame, **kwargs)


@pytest.mark.parametrize("dtype", ("float16", "uint8", "uint16"))
def test_to_array_bit_depth_requires_float32_frame_values(dtype: str) -> None:
    """v1-quantize-values acceptance 12: generic quantized export starts from fp32 normalized values."""
    import cupy as cp

    frame = px.io.from_array(
        cp.zeros((1, 1, 1), dtype=dtype),
        colorspace="sRGB",
        gamma="linear",
        channels="Y",
    )

    with pytest.raises(ValueError, match="float32"):
        px.io.to_array(frame, bit_depth=8)


def test_to_array_copy_false_rejects_copy_requiring_repacking() -> None:
    """v1-boundary-api acceptance 15: strict zero-copy fails for selection, transpose, dtype, and affine."""
    import cupy as cp

    frame = px.io.from_array(cp.asarray(_sample_hwc()), colorspace="sRGB", gamma="srgb", channels="RGB")
    calls = (
        {"channels": "BGR"},
        {"layout": "CHW"},
        {"dtype": "float16"},
        {"scale": 255.0},
    )

    for kwargs in calls:
        with pytest.raises(ValueError) as error:
            px.io.to_array(frame, copy=False, **kwargs)
        _assert_actionable(error)


def test_to_array_writes_directly_to_cupy_out_and_returns_the_same_array() -> None:
    """v1-boundary-api acceptance 16: out receives the fused result in place and is returned by identity."""
    import cupy as cp

    values = _sample_hwc()
    frame = px.io.from_array(cp.asarray(values), colorspace="sRGB", gamma="srgb", channels="RGB")
    out = cp.empty((3, 2, 3), dtype=cp.float16)
    pointer = out.data.ptr

    result = px.io.to_array(frame, channels="BGR", layout="CHW", dtype="float16", scale=2.0, out=out)

    expected = np.transpose(values[..., [2, 1, 0]] * np.float32(2.0), (2, 0, 1)).astype(np.float16)
    assert result is out
    assert out.data.ptr == pointer
    np.testing.assert_array_equal(_host(out), expected)


def test_to_array_out_rejects_non_cupy_and_incompatible_destinations_actionably() -> None:
    """v1-boundary-api acceptance 16: out requires writable-shape evidence from a C-contiguous CuPy array."""
    import cupy as cp
    import torch

    frame = px.io.from_array(cp.asarray(_sample_hwc()), colorspace="sRGB", gamma="srgb", channels="RGB")
    invalid = (
        np.empty((2, 3, 3), dtype=np.float32),
        torch.empty((2, 3, 3), device="cuda"),
        cp.empty((2, 3, 2), dtype=cp.float32),
        cp.empty((2, 3, 3), dtype=cp.float16),
        cp.empty((2, 3, 6), dtype=cp.float32)[..., ::2],
    )

    for out in invalid:
        with pytest.raises(ValueError) as error:
            px.io.to_array(frame, out=out)
        _assert_actionable(error)
    with pytest.raises(ValueError) as copy_error:
        px.io.to_array(frame, out=cp.empty_like(frame.data), copy=True)
    _assert_actionable(copy_error)
