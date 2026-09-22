"""P216 boundary behavior, with independent H.273/layout and fp64 sampling oracles."""

from __future__ import annotations

import inspect
from typing import Any, get_args

import cupy as cp
import numpy as np
import pytest
from p216_oracle import (
    FROM_FILTERS,
    TO_FILTERS,
    assert_corpus_codes,
    asymmetric_codes,
    decode_values,
    fixed_corpus,
    from_reference,
    pack_planes,
    to_reference,
    unpack_codes,
)
from test_to_format_spec import _frame

import pixtreme as px


@pytest.mark.parametrize("direction", ("from", "to"))
def test_p216_has_the_exact_static_signature(direction: str) -> None:
    """v1-p216-wire-format acceptance 1 and 19: no bit-depth, siting, options, pitch, or extra positional surface."""
    function = getattr(px.io, f"{direction}_p216")
    assert not hasattr(px, f"{direction}_p216")
    assert not hasattr(px.core.Frame, f"{direction}_p216")
    expected = (
        ("buf", "width", "height", "colorspace", "gamma", "matrix", "range", "interpolation")
        if direction == "from"
        else ("frame", "range", "interpolation")
    )
    parameters = inspect.signature(function).parameters
    assert tuple(parameters) == expected
    assert parameters[expected[0]].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[expected[0]].default is inspect.Parameter.empty
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in expected[1:])
    assert parameters["range"].default == "legal"
    assert parameters["interpolation"].default == ("bilinear" if direction == "from" else "area")
    if direction == "from":
        assert all(parameters[name].default is inspect.Parameter.empty for name in ("width", "height"))
        assert all(parameters[name].default is None for name in ("colorspace", "gamma", "matrix"))


@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_from_p216_asymmetric_words_keep_layout_low_bits_and_row_phase(range_token: str) -> None:
    """v1-p216-wire-format acceptance 2, 4, 6 and 10: explicit indices catch planar, UV-swap, mask and 420 errors.

    AC-43-10 fixes the range/layout anchor tolerance at 2e-7 (fp32 affine only).
    """
    codes = asymmetric_codes()
    # nearest half-up: luma x=1 chooses chroma sample 1, not sample 0.
    y, cb, cr = unpack_codes(codes, 3, 6)
    indices = [0, 1, 1, 2, 2, 2]
    expected = np.stack(
        (
            decode_values(y, range_token, 0),
            decode_values(cb[:, indices], range_token, 1),
            decode_values(cr[:, indices], range_token, 2),
        ),
        axis=-1,
    )
    result = px.io.from_p216(cp.asarray(codes), width=6, height=3, range=range_token, interpolation="nearest")
    np.testing.assert_allclose(result.data.get(), expected, rtol=0, atol=2e-7)


@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_to_p216_direct_layout_owns_all_sixteen_bits(range_token: str) -> None:
    """v1-p216-wire-format acceptance 2, 4 and 10: independent code-derived 444 input packs directly and exactly."""
    expected = asymmetric_codes()
    y, cb, cr = unpack_codes(expected, 3, 6)
    values = np.stack(
        (
            decode_values(y, range_token, 0),
            np.repeat(decode_values(cb, range_token, 1), 2, axis=1),
            np.repeat(decode_values(cr, range_token, 2), 2, axis=1),
        ),
        axis=-1,
    ).astype(np.float32)
    result = px.io.to_p216(_frame(values), range=range_token, interpolation="nearest")
    assert isinstance(result, cp.ndarray)
    assert result.shape == (36,)
    assert result.dtype == cp.uint16 and result.flags.c_contiguous
    np.testing.assert_array_equal(result.get(), expected)


@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_from_p216_h273_endpoints_center_and_headroom(range_token: str) -> None:
    """v1-p216-wire-format acceptance 4 and 10: n=16 hand anchors preserve legal headroom without clipping.

    Expected fractions use 219*256/224*256 or 65535 from H.273, with AC's 2e-7 fp32 tolerance.
    """
    levels = np.asarray([0, 4096, 32768, 60160, 61440, 65535], dtype=np.uint16)
    # One chroma pair per row avoids interpolation in this range-only anchor.
    codes = pack_planes(np.repeat(levels[:, None], 2, axis=1), levels[:, None], levels[::-1, None])
    result = px.io.from_p216(cp.asarray(codes), width=2, height=6, range=range_token).data.get()
    if range_token == "legal":
        y = [-16 / 219, 0, 112 / 219, 1, 224 / 219, 61439 / 56064]
        c = [-1 / 14, 0, 0.5, 219 / 224, 1, 61439 / 57344]
    else:
        y = c = [0, 4096 / 65535, 32768 / 65535, 60160 / 65535, 61440 / 65535, 1]
    expected = np.repeat(np.asarray(list(zip(y, c, c[::-1], strict=True)))[:, None, :], 2, axis=1)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-7)


@pytest.mark.parametrize("range_token", ("legal", "full"))
@pytest.mark.parametrize("interpolation", FROM_FILTERS)
def test_from_p216_every_filter_matches_fp64_horizontal_reference(range_token: str, interpolation: str) -> None:
    """v1-p216-wire-format acceptance 6 and 10: all 8 filters preserve row identity, co-siting and replicate edges.

    AC-43-10 fixes atol=3e-6 for fp32 filter evaluation versus the independent fp64 weights.
    """
    codes = asymmetric_codes()
    expected = from_reference(codes, 3, 6, range_token, interpolation)
    actual = px.io.from_p216(cp.asarray(codes), width=6, height=3, range=range_token, interpolation=interpolation)
    np.testing.assert_allclose(actual.data.get(), expected, rtol=0, atol=3e-6)


@pytest.mark.parametrize("range_token", ("legal", "full"))
@pytest.mark.parametrize("interpolation", TO_FILTERS)
def test_to_p216_fixed_corpus_matches_bounded_near_tie_oracle(range_token: str, interpolation: str) -> None:
    """v1-p216-wire-format acceptance 5, 6 and 10: fixed seed-43 corpus uses per-case ±1/2%/0.125-code bounds.

    The tolerance is AC-43-10's fp32 sampling/affine budget; it does not apply to exact anchors or round trips.
    """
    values = fixed_corpus()
    expected, q64 = to_reference(values, range_token, interpolation)
    actual = px.io.to_p216(_frame(values), range=range_token, interpolation=interpolation)
    assert actual.flags.c_contiguous
    assert_corpus_codes(actual.get(), expected, q64)


@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_to_p216_hand_rounded_ties_and_container_only_clip(range_token: str) -> None:
    """v1-p216-wire-format acceptance 4, 5 and 10: half-away, signed near-ties and headroom have exact hand codes.

    Binary fractions make the mapped positive ties exact. Negative mapped ties must clip to zero;
    their rounding direction is unobservable after unsigned clipping. Negative input with positive
    legal mapped code still distinguishes truncation and ties-to-even from half-away.
    """
    if range_token == "legal":
        ys = [-1, -129 / 512, -1 / 32, -1 / 512, 0, 1 / 512, 1 / 512 - 1 / 65536, 1 / 512 + 1 / 65536, 1, 33 / 32, 2]
        cs = [
            -1,
            -2049 / 16384,
            -1 / 32,
            -1 / 16384,
            0,
            1 / 16384,
            1 / 16384 - 1 / 65536,
            1 / 16384 + 1 / 65536,
            1,
            33 / 32,
            2,
        ]
        ycodes = [0, 0, 2344, 3987, 4096, 4206, 4205, 4206, 60160, 61912, 65535]
        ccodes = [0, 0, 2304, 4093, 4096, 4100, 4099, 4100, 61440, 63232, 65535]
    else:
        ys = cs = [-1, -0.5 - 1 / 65536, -0.5, -0.5 + 1 / 65536, 0, 0.5 - 1 / 65536, 0.5, 0.5 + 1 / 65536, 1, 2]
        ycodes = ccodes = [0, 0, 0, 0, 0, 32767, 32768, 32768, 65535, 65535]
    values = np.repeat(np.asarray(list(zip(ys, cs, cs[::-1], strict=True)), dtype=np.float32)[:, None, :], 2, axis=1)
    expected = pack_planes(
        np.repeat(np.asarray(ycodes)[:, None], 2, axis=1),
        np.asarray(ccodes)[:, None],
        np.asarray(ccodes[::-1])[:, None],
    ).astype(np.uint16)
    actual = px.io.to_p216(_frame(values), range=range_token, interpolation="nearest")
    np.testing.assert_array_equal(actual.get(), expected)


@pytest.mark.parametrize("bit_depth", (10, 12))
def test_to_p216_low_depth_origin_is_quantized_to_sixteen_effective_bits(bit_depth: int) -> None:
    """v1-p216-wire-format acceptance 4 and 19: lower-depth signal values do not select an MSB-padding mode."""
    maximum = (1 << bit_depth) - 1
    codes = np.asarray([1, 17, maximum // 2, maximum - 1], dtype=np.int64)
    values = np.repeat(np.repeat((codes / maximum).astype(np.float32)[:, None, None], 2, axis=1), 3, axis=2)
    expected_codes = ((codes * 65535 * 2 + maximum) // (2 * maximum)).astype(np.uint16)
    expected = np.tile(np.repeat(expected_codes, 2), 2)
    actual = px.io.to_p216(_frame(values), range="full", interpolation="nearest")
    np.testing.assert_array_equal(actual.get(), expected)


@pytest.mark.parametrize("range_token", ("legal", "full"))
@pytest.mark.parametrize("from_filter", FROM_FILTERS)
@pytest.mark.parametrize("to_filter", TO_FILTERS)
def test_p216_constant_chroma_round_trips_all_32_filter_pairs(
    range_token: str, from_filter: str, to_filter: str
) -> None:
    """v1-p216-wire-format acceptance 11: both ranges preserve all words in all 32 constant-chroma filter pairs."""
    codes = asymmetric_codes()
    codes[18::2] = 17011
    codes[19::2] = 53003
    frame = px.io.from_p216(cp.asarray(codes), width=6, height=3, range=range_token, interpolation=from_filter)
    result = px.io.to_p216(frame, range=range_token, interpolation=to_filter)
    np.testing.assert_array_equal(result.get(), codes)


@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_p216_nonconstant_nearest_code_origin_round_trip(range_token: str) -> None:
    """v1-p216-wire-format acceptance 12: nearest preserves nonconstant code-origin words, including low bits."""
    codes = asymmetric_codes()
    frame = px.io.from_p216(cp.asarray(codes), width=6, height=3, range=range_token, interpolation="nearest")
    np.testing.assert_array_equal(px.io.to_p216(frame, range=range_token, interpolation="nearest").get(), codes)


def test_p216_default_metadata_and_filters_have_the_specified_behavior() -> None:
    """v1-p216-wire-format acceptance 6 and 7: defaults act as legal/bilinear and legal/area, with fp32 placeholders."""
    codes = asymmetric_codes()
    default = px.io.from_p216(cp.asarray(codes), width=6, height=3)
    explicit_none = px.io.from_p216(cp.asarray(codes), width=6, height=3, colorspace=None, gamma=None, matrix=None)
    for result in (default, explicit_none):
        assert isinstance(result, px.core.Frame)
        assert result.shape == (3, 6, 3) and result.dtype == cp.float32 and result.data.flags.c_contiguous
        assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
            "Rec.709",
            "Rec.709",
            ("Y", "Cb", "Cr"),
            None,
        )
        np.testing.assert_allclose(
            result.data.get(), from_reference(codes, 3, 6, "legal", "bilinear"), rtol=0, atol=3e-6
        )
    values = fixed_corpus()
    expected, q64 = to_reference(values, "legal", "area")
    assert_corpus_codes(px.io.to_p216(_frame(values)).get(), expected, q64)


@pytest.mark.parametrize(
    ("axis", "token", "canonical"),
    [
        (axis, token, token)
        for axis, alias in (("colorspace", px.core.Colorspace), ("gamma", px.core.Gamma), ("matrix", px.core.Matrix))
        for token in get_args(alias)
    ]
    + [("colorspace", "rec_2020", "Rec.2020"), ("gamma", "p q", "PQ"), ("matrix", "bt_2020", "BT.2020")],
)
def test_from_p216_metadata_tokens_stamp_without_changing_pixels(axis: str, token: str, canonical: str) -> None:
    """v1-p216-wire-format acceptance 7: every canonical metadata token and normalized spelling is a value-neutral claim."""
    source = cp.asarray(asymmetric_codes())
    result = px.io.from_p216(source, width=6, height=3, **{axis: token})
    assert getattr(result, axis) == canonical
    expected_tags = {"colorspace": "Rec.709", "gamma": "Rec.709", "matrix": None} | {axis: canonical}
    assert {name: getattr(result, name) for name in expected_tags} == expected_tags
    np.testing.assert_allclose(
        result.data.get(), from_reference(asymmetric_codes(), 3, 6, "legal", "bilinear"), rtol=0, atol=3e-6
    )


@pytest.mark.parametrize("direction", ("from", "to"))
def test_p216_calls_leave_inputs_unchanged_and_return_private_storage(direction: str) -> None:
    """v1-p216-wire-format acceptance 2 and 9: repeated calls allocate independent output without mutating inputs."""
    function = getattr(px.io, f"{direction}_p216")
    source = cp.asarray(asymmetric_codes()) if direction == "from" else _frame(fixed_corpus())
    data = source if direction == "from" else source.data
    before = data.get()
    metadata = None if direction == "from" else (source.colorspace, source.gamma, source.channels, source.matrix)
    kwargs = {"width": 6, "height": 3} if direction == "from" else {}
    first, second = function(source, **kwargs), function(source, **kwargs)
    first_data = first.data if direction == "from" else first
    second_data = second.data if direction == "from" else second
    assert len({data.data.ptr, first_data.data.ptr, second_data.data.ptr}) == 3
    np.testing.assert_array_equal(data.get(), before)
    np.testing.assert_array_equal(first_data.get(), second_data.get())
    first_data.fill(0)
    np.testing.assert_array_equal(data.get(), before)
    assert np.any(second_data.get() != 0)
    if direction == "to":
        assert (source.colorspace, source.gamma, source.channels, source.matrix) == metadata


def _three_part_error(error: pytest.ExceptionInfo[ValueError], recovery: str) -> None:
    message = str(error.value)
    why, what, how = message.split("; ", 2)
    assert why.startswith("why=") and why[4:].strip()
    assert what.startswith("what=") and what[5:].strip()
    assert how.startswith("how=") and how[4:].strip()
    assert recovery.casefold() in how.casefold(), message


def _reject_without_pixel_work(function: Any, source: Any, kwargs: dict[str, Any], recovery: str) -> None:
    # Real CUDA capture records any erroneous pixel work before ValueError; validation itself needs no kernels.
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        try:
            with pytest.raises(ValueError) as error:
                function(source, **kwargs)
        finally:
            graph = stream.end_capture()
    _three_part_error(error, recovery)
    assert not graph.debug_dot_str(cp.cuda.runtime.cudaGraphDebugDotFlagsVerbose).count('label="{KERNEL')


@pytest.mark.parametrize(
    ("case", "recovery"),
    [
        ("numpy", "cupy"),
        ("list", "cupy"),
        ("object", "cupy"),
        ("uint8", "uint16"),
        ("uint32", "uint16"),
        ("float32", "uint16"),
        ("short", "36"),
        ("long", "36"),
        ("2d", "shape"),
        ("3d", "shape"),
        ("strided", "contiguous"),
        ("reversed", "contiguous"),
    ],
)
def test_from_p216_rejects_invalid_buffer_before_pixel_work(case: str, recovery: str) -> None:
    """v1-p216-wire-format acceptance 3 and 17: device/type/dtype/count/shape/contiguity fail with actionable errors."""
    function = px.io.from_p216
    codes = asymmetric_codes()
    source: Any = cp.asarray(codes)
    if case in ("numpy", "list", "object"):
        source = {"numpy": codes, "list": codes.tolist(), "object": object()}[case]
    elif case in ("uint8", "uint32", "float32"):
        source = source.astype(case)
    elif case == "short":
        source = source[:-1]
    elif case == "long":
        source = cp.concatenate((source, source[:1]))
    elif case == "2d":
        source = source.reshape(6, 6)
    elif case == "3d":
        source = source.reshape(3, 6, 2)
    elif case == "strided":
        source = cp.tile(source, 2)[::2]
    elif case == "reversed":
        source = source[::-1]
    _reject_without_pixel_work(function, source, {"width": 6, "height": 3}, recovery)


@pytest.mark.parametrize(
    ("axis", "value"),
    [("width", v) for v in (0, -2, 1, 3, 2.5, True, "6", None)]
    + [("height", v) for v in (0, -1, 1.5, True, "3", None)],
)
def test_from_p216_rejects_invalid_dimensions_before_pixel_work(axis: str, value: Any) -> None:
    """v1-p216-wire-format acceptance 3 and 17: dimensions are positive integers and width is even, never coerced."""
    function = px.io.from_p216
    _reject_without_pixel_work(
        function, cp.asarray(asymmetric_codes()), {"width": 6, "height": 3} | {axis: value}, axis
    )


@pytest.mark.parametrize(
    "case", ("object", "array", "RGB", "swapped", "alpha", "Y", "float16", "uint8", "uint16", "uint32", "odd-width")
)
def test_to_p216_rejects_invalid_frames_before_pixel_work(case: str) -> None:
    """v1-p216-wire-format acceptance 3, 8 and 17: require fp32 YCbCr Frame with even width and concrete recovery paths."""
    function = px.io.to_p216
    source: Any = _frame(np.zeros((3, 6, 3), dtype=np.float32))
    recovery = "Frame"
    if case == "object":
        source = object()
    elif case == "array":
        source = source.data
    elif case in ("RGB", "swapped", "alpha", "Y"):
        channels = {"RGB": ("R", "G", "B"), "swapped": ("Y", "Cr", "Cb"), "alpha": ("Y", "Cb", "Cr", "A"), "Y": ("Y",)}[
            case
        ]
        source = _frame(np.zeros((3, 6, len(channels)), dtype=np.float32), channels=channels)
        recovery = "px.color.rgb_to_ycbcr"
    elif case == "odd-width":
        source = _frame(np.zeros((3, 5, 3), dtype=np.float32))
        recovery = "width"
    else:
        source = _frame(np.zeros((3, 6, 3), dtype=case))
        recovery = "float32"
    _reject_without_pixel_work(function, source, {}, recovery)


@pytest.mark.parametrize(
    ("direction", "axis", "value", "recovery"),
    [
        (direction, axis, value, recovery)
        for direction in ("from", "to")
        for axis, recovery in (("range", "legal"), ("interpolation", "nearest"))
        for value in ("unknown", "", " ._- ", 17, None)
    ]
    + [("from", "interpolation", "area", "nearest")]
    + [
        ("to", "interpolation", token, "nearest")
        for token in ("b-spline", "mitchell", "lanczos2", "lanczos3", "lanczos4")
    ]
    + [
        ("from", axis, value, recovery)
        for axis, recovery in (("colorspace", "Rec.709"), ("gamma", "Rec.709"), ("matrix", "BT.709"))
        for value in ("unknown", "", 17)
    ],
)
def test_p216_rejects_invalid_tokens_before_pixel_work(direction: str, axis: str, value: Any, recovery: str) -> None:
    """v1-p216-wire-format acceptance 7, 8 and 17: closed token/subset failures precede GPU work and show canonical choices."""
    function = getattr(px.io, f"{direction}_p216")
    source = cp.asarray(asymmetric_codes()) if direction == "from" else _frame(fixed_corpus())
    kwargs = {"width": 6, "height": 3} if direction == "from" else {}
    _reject_without_pixel_work(function, source, kwargs | {axis: value}, recovery)
