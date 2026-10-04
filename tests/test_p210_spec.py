"""P210 boundary behavior, with independent H.273/layout and fp64 sampling oracles."""

from __future__ import annotations

import inspect
from typing import Any, get_args

import cupy as cp
import numpy as np
import pytest
from p210_oracle import (
    FROM_FILTERS,
    TO_FILTERS,
    assert_near_tie_codes,
    asymmetric_words,
    decode_values,
    fixed_corpus,
    from_reference,
    pack_planes,
    quality_values,
    quality_words,
    to_reference,
    unpack_codes,
)
from test_to_format_spec import _frame

import pixtreme as px


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize("direction", ("from", "to"))
def test_p210_has_the_exact_static_signature(direction: str) -> None:
    """P210 conversion exposes its documented parameters without extra bit-depth, siting, pitch, or positional
    options.
    """
    function = getattr(px.io, f"{direction}_p210")
    assert not hasattr(px, f"{direction}_p210")
    assert not hasattr(px.core.Frame, f"{direction}_p210")
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


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_from_p210_asymmetric_words_keep_layout_low_bits_and_row_phase(range_token: str) -> None:
    """P210 reading decodes asymmetric packed words in the correct plane order and row phase while ignoring low
    padding bits.
    """
    codes = asymmetric_words()
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
    result = px.io.from_p210(cp.asarray(codes), width=6, height=3, range=range_token, interpolation="nearest")
    np.testing.assert_allclose(result.data.get(), expected, rtol=0, atol=2e-7)


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_to_p210_direct_layout_sets_padding_to_zero(range_token: str) -> None:
    """P210 writing packs ten-bit codes in high bits and clears every low padding bit."""
    expected = asymmetric_words() & np.uint16(0xFFC0)
    y, cb, cr = unpack_codes(expected, 3, 6)
    values = np.stack(
        (
            decode_values(y, range_token, 0),
            np.repeat(decode_values(cb, range_token, 1), 2, axis=1),
            np.repeat(decode_values(cr, range_token, 2), 2, axis=1),
        ),
        axis=-1,
    ).astype(np.float32)
    result = px.io.to_p210(_frame(values), range=range_token, interpolation="nearest")
    assert isinstance(result, cp.ndarray)
    assert result.shape == (36,)
    assert result.dtype == cp.uint16 and result.flags.c_contiguous
    np.testing.assert_array_equal(result.get(), expected)


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-103")
@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_from_p210_h273_endpoints_center_and_headroom(range_token: str) -> None:
    """P210 reading maps H.273 endpoints and chroma center while preserving legal-range headroom without clipping."""
    levels = np.asarray([0, 64, 512, 940, 960, 1023], dtype=np.uint16)
    # One chroma pair per row avoids interpolation in this range-only anchor.
    codes = pack_planes(np.repeat(levels[:, None], 2, axis=1), levels[:, None], levels[::-1, None]) << 6
    result = px.io.from_p210(cp.asarray(codes), width=2, height=6, range=range_token).data.get()
    if range_token == "legal":
        y = [-64 / 876, 0, 448 / 876, 1, 896 / 876, 959 / 876]
        c = [-64 / 896, 0, 0.5, 876 / 896, 1, 959 / 896]
    else:
        y = c = [0, 64 / 1023, 512 / 1023, 940 / 1023, 960 / 1023, 1]
    expected = np.repeat(np.asarray(list(zip(y, c, c[::-1], strict=True)))[:, None, :], 2, axis=1)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-7)


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("range_token", ("legal", "full"))
@pytest.mark.parametrize("interpolation", FROM_FILTERS)
def test_from_p210_every_filter_matches_fp64_horizontal_reference(range_token: str, interpolation: str) -> None:
    """Every P210 input filter matches an independent horizontal sampling reference with co-sited chroma and
    replicated edges.
    """
    codes = asymmetric_words()
    expected = from_reference(codes, 3, 6, range_token, interpolation)
    actual = px.io.from_p210(cp.asarray(codes), width=6, height=3, range=range_token, interpolation=interpolation)
    np.testing.assert_allclose(actual.data.get(), expected, rtol=0, atol=3e-6)


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("range_token", ("legal", "full"))
@pytest.mark.parametrize("interpolation", TO_FILTERS)
def test_to_p210_fixed_corpus_matches_bounded_near_tie_oracle(range_token: str, interpolation: str) -> None:
    """P210 writing matches an independent sampling and quantization oracle except within the stated near-tie code
    tolerance.
    """
    values = fixed_corpus()
    expected, q64 = to_reference(values, range_token, interpolation)
    actual = px.io.to_p210(_frame(values), range=range_token, interpolation=interpolation)
    assert actual.flags.c_contiguous
    assert_near_tie_codes(actual.get(), expected, q64, corpus=True)


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-103")
@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_to_p210_hand_rounded_ties_and_code_only_clip(range_token: str) -> None:
    """P210 writing rounds half away from zero and clips only at the unsigned container bounds while retaining legal
    headroom.
    """
    if range_token == "legal":
        ys = [-1, -1 / 8, -1 / 256, 0, 1 / 8 - 1 / 65536, 1 / 8, 1 / 8 + 1 / 65536, 1, 33 / 32, 2]
        cs = [-1, -1 / 8, -1 / 256, 0, 1 / 256 - 1 / 65536, 1 / 256, 1 / 256 + 1 / 65536, 1, 33 / 32, 2]
        ycodes = [0, 0, 61, 64, 173, 174, 174, 940, 967, 1023]
        ccodes = [0, 0, 61, 64, 67, 68, 68, 960, 988, 1023]
    else:
        ys = cs = [-1, -0.5, 0, 0.5 - 1 / 65536, 0.5, 0.5 + 1 / 65536, 1, 2]
        ycodes = ccodes = [0, 0, 0, 511, 512, 512, 1023, 1023]
    values = np.repeat(np.asarray(list(zip(ys, cs, cs[::-1], strict=True)), dtype=np.float32)[:, None, :], 2, axis=1)
    expected = (
        pack_planes(
            np.repeat(np.asarray(ycodes)[:, None], 2, axis=1),
            np.asarray(ccodes)[:, None],
            np.asarray(ccodes[::-1])[:, None],
        ).astype(np.uint16)
        << 6
    )
    actual = px.io.to_p210(_frame(values), range=range_token, interpolation="nearest")
    np.testing.assert_array_equal(actual.get(), expected)


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("range_token", ("legal", "full"))
@pytest.mark.parametrize("from_filter", FROM_FILTERS)
@pytest.mark.parametrize("to_filter", TO_FILTERS)
def test_p210_constant_chroma_round_trips_all_32_filter_pairs(
    range_token: str, from_filter: str, to_filter: str
) -> None:
    """P210 data with constant chroma preserves active codes through every supported input and output filter pair."""
    codes = asymmetric_words()
    codes[18::2] = (267 << 6) | 17
    codes[19::2] = (731 << 6) | 35
    frame = px.io.from_p210(cp.asarray(codes), width=6, height=3, range=range_token, interpolation=from_filter)
    result = px.io.to_p210(frame, range=range_token, interpolation=to_filter)
    np.testing.assert_array_equal(result.get(), codes & np.uint16(0xFFC0))


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("range_token", ("legal", "full"))
def test_p210_nonconstant_nearest_code_origin_round_trip(range_token: str) -> None:
    """P210 data with varying chroma round-trips active codes through nearest sampling and clears padding bits."""
    codes = asymmetric_words()
    frame = px.io.from_p210(cp.asarray(codes), width=6, height=3, range=range_token, interpolation="nearest")
    np.testing.assert_array_equal(
        px.io.to_p210(frame, range=range_token, interpolation="nearest").get(), codes & np.uint16(0xFFC0)
    )
    clean = cp.asarray(codes & np.uint16(0xFFC0))
    np.testing.assert_array_equal(
        px.io.from_p210(clean, width=6, height=3, range=range_token, interpolation="nearest").data.get(),
        frame.data.get(),
    )


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-009")
def test_p210_default_metadata_and_filters_have_the_specified_behavior() -> None:
    """P210 conversion uses legal range, bilinear input sampling, area output sampling, and documented float32
    metadata defaults.
    """
    codes = asymmetric_words()
    default = px.io.from_p210(cp.asarray(codes), width=6, height=3)
    explicit_none = px.io.from_p210(cp.asarray(codes), width=6, height=3, colorspace=None, gamma=None, matrix=None)
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
    assert_near_tie_codes(px.io.to_p210(_frame(values)).get(), expected, q64)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize(
    ("axis", "token", "canonical"),
    [
        (axis, token, token)
        for axis, alias in (("colorspace", px.core.Colorspace), ("gamma", px.core.Gamma), ("matrix", px.core.Matrix))
        for token in get_args(alias)
    ]
    + [("colorspace", "rec_2020", "Rec.2020"), ("gamma", "p q", "PQ"), ("matrix", "bt_2020", "BT.2020")],
)
def test_from_p210_metadata_tokens_stamp_without_changing_pixels(axis: str, token: str, canonical: str) -> None:
    """P210 input accepts supported color metadata names without changing decoded pixel values."""
    source = cp.asarray(asymmetric_words())
    result = px.io.from_p210(source, width=6, height=3, **{axis: token})
    assert getattr(result, axis) == canonical
    expected_tags = {"colorspace": "Rec.709", "gamma": "Rec.709", "matrix": None} | {axis: canonical}
    assert {name: getattr(result, name) for name in expected_tags} == expected_tags
    np.testing.assert_allclose(
        result.data.get(), from_reference(asymmetric_words(), 3, 6, "legal", "bilinear"), rtol=0, atol=3e-6
    )


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("direction", ("from", "to"))
def test_p210_calls_leave_inputs_unchanged_and_return_private_storage(direction: str) -> None:
    """Repeated P210 conversions leave inputs unchanged and return independent output storage."""
    function = getattr(px.io, f"{direction}_p210")
    source = cp.asarray(asymmetric_words()) if direction == "from" else _frame(fixed_corpus())
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


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-017")
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
def test_from_p210_rejects_invalid_buffer_before_pixel_work(case: str, recovery: str) -> None:
    """P210 input rejects buffers with an invalid device, type, dtype, element count, shape, or contiguity before
    pixel work.
    """
    function = px.io.from_p210
    codes = asymmetric_words()
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


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize(
    ("axis", "value"),
    [("width", v) for v in (0, -2, 1, 3, 2.5, True, "6", None)]
    + [("height", v) for v in (0, -1, 1.5, True, "3", None)],
)
def test_from_p210_rejects_invalid_dimensions_before_pixel_work(axis: str, value: Any) -> None:
    """P210 input rejects dimensions that are not positive integers or have an odd width before pixel work."""
    function = px.io.from_p210
    _reject_without_pixel_work(
        function, cp.asarray(asymmetric_words()), {"width": 6, "height": 3} | {axis: value}, axis
    )


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize(
    "case", ("object", "array", "RGB", "swapped", "alpha", "Y", "float16", "uint8", "uint16", "uint32", "odd-width")
)
def test_to_p210_rejects_invalid_frames_before_pixel_work(case: str) -> None:
    """P210 output requires an even-width float32 YCbCr Frame and explains how to correct invalid inputs."""
    function = px.io.to_p210
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


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-017")
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
def test_p210_rejects_invalid_tokens_before_pixel_work(direction: str, axis: str, value: Any, recovery: str) -> None:
    """P210 conversion rejects unsupported range, filter, and metadata names before GPU work and lists valid choices."""
    function = getattr(px.io, f"{direction}_p210")
    source = cp.asarray(asymmetric_words()) if direction == "from" else _frame(fixed_corpus())
    kwargs = {"width": 6, "height": 3} if direction == "from" else {}
    _reject_without_pixel_work(function, source, kwargs | {axis: value}, recovery)


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("pattern", ("ramp", "neutral", "edge", "alternation", "zone"))
@pytest.mark.parametrize("range_token", ("legal", "full"))
@pytest.mark.parametrize("interpolation", FROM_FILTERS)
def test_from_p210_quality_patterns_match_fp64_phase_and_filter_reference(
    pattern: str, range_token: str, interpolation: str
) -> None:
    """P210 reading reproduces independent phase and filter results for patterns that reveal channel swaps and
    ringing.
    """
    words = quality_words(pattern, range_token)
    expected = from_reference(words, 17, 64, range_token, interpolation)
    actual = px.io.from_p210(cp.asarray(words), width=64, height=17, range=range_token, interpolation=interpolation)
    np.testing.assert_allclose(actual.data.get(), expected, rtol=0, atol=3e-6)
    if pattern == "ramp":
        codes, _, _ = unpack_codes(words, 17, 64)
        assert np.all(np.diff(codes, axis=1) > 0)
        np.testing.assert_allclose(actual.data.get()[..., 0], decode_values(codes, range_token, 0), rtol=0, atol=2e-7)
    if pattern == "neutral":
        _, cb, cr = unpack_codes(words, 17, 64)
        assert np.all(cb == 512) and np.all(cr == 512)
        np.testing.assert_allclose(actual.data.get()[..., 1:], expected[..., 1:], rtol=0, atol=3e-6)
    if pattern in {"edge", "alternation", "zone"} and interpolation in {"nearest", "bilinear", "b-spline"}:
        assert np.all(actual.data.get()[..., 1:] >= 0.25 - 2e-7)
        assert np.all(actual.data.get()[..., 1:] <= 0.75 + 2e-7)


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.req("REQ-PIX-009")
@pytest.mark.parametrize("pattern", ("ramp", "neutral", "edge", "alternation", "zone"))
@pytest.mark.parametrize("range_token", ("legal", "full"))
@pytest.mark.parametrize("interpolation", TO_FILTERS)
def test_to_p210_quality_patterns_match_fp64_phase_and_quantization_reference(
    pattern: str, range_token: str, interpolation: str
) -> None:
    """P210 writing reproduces independent phase and quantization results for diagnostic patterns within its near-tie
    code tolerance.
    """
    values = quality_values(pattern).astype(np.float32)
    expected, q64 = to_reference(values, range_token, interpolation)
    actual = px.io.to_p210(_frame(values), range=range_token, interpolation=interpolation).get()
    assert_near_tie_codes(actual, expected, q64)
    if pattern == "ramp":
        ycodes = (actual[: 64 * 17] >> 6).reshape(17, 64)
        expected_y = (expected[: 64 * 17] >> 6).reshape(17, 64)
        assert np.all(np.diff(ycodes, axis=1) > 0)
        np.testing.assert_array_equal(np.diff(ycodes, axis=1), np.diff(expected_y, axis=1))
    if pattern == "neutral":
        assert np.all((actual[64 * 17 :] >> 6) == 512)
