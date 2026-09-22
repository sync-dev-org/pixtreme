"""Specification and numerical-property tests for antialiased Lanczos resize."""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable
from typing import get_args

import numpy as np
import pytest

import pixtreme as px

AA_TO_POINT = {
    "lanczos2-aa": "lanczos2",
    "lanczos3-aa": "lanczos3",
    "lanczos4-aa": "lanczos4",
}
AA_TOKENS = tuple(AA_TO_POINT)
EXISTING_RESIZE_TOKENS = (
    "nearest",
    "bilinear",
    "bicubic",
    "b-spline",
    "mitchell",
    "lanczos2",
    "lanczos3",
    "lanczos4",
    "area",
)
_BASELINE_DIGESTS = {
    "nearest": "0996f8889820defc768f31f7a8d6d8eabb1f0afd6517a964461a46f71631794c",
    "bilinear": "344cd1be6b1c55bedf082fa7083cae1f9c14d0411c24e5e87632074d490f5df5",
    "bicubic": "181bc66b11c8ac00c0cb593246ddecadfe749f21c84f1128a2aca3233f697acb",
    "b-spline": "a65d8fb59164ddb4eb0fa6359e530eec8e4755e1adea1ae976f760145ba5fb33",
    "mitchell": "1882d67cc0cc4edd488751263c73f9d49b20eb6099ce78dc396fb453fe2b5728",
    "lanczos2": "c3e8337c489568e1d8404df911e48d71570c15c9692d810196c06008aa964aa2",
    "lanczos3": "232092346b29573db1872e458261b3b370bb0bf44b4eddf7230ffbce7b3126df",
    "lanczos4": "440af51f69b4c9f0806f1294eeec623bc9fd51d37719e3807ad7465ff2fa56d1",
    "area": "731038c3e700a1684c0931a1c09647942faf7d716e8ec68ab7e6d15ce16e373c",
}


def _frame(
    values: np.ndarray,
    *,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    channels: tuple[str, ...] | None = None,
    matrix: str | None = None,
) -> px.core.Frame:
    import cupy as cp

    channel_labels = channels or tuple(f"channel-{index}" for index in range(values.shape[2]))
    return px.io.from_array(
        cp.asarray(np.ascontiguousarray(values)),
        colorspace=colorspace,
        gamma=gamma,
        channels=channel_labels,
        matrix=matrix,
    )


def _host(frame: px.core.Frame) -> np.ndarray:
    return frame.data.get()


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> str:
    message = str(error.value)
    assert message.startswith("why=")
    assert "; what=" in message
    assert "; how=" in message
    return message


def _strict_support_indices(coordinate: float, scale: float, lobes: int) -> tuple[int, ...]:
    lower = coordinate - lobes * scale
    upper = coordinate + lobes * scale
    return tuple(range(math.floor(lower) + 1, math.ceil(upper)))


def _lanczos_weight(distance: float, *, scale: float, lobes: int) -> float:
    x = abs(distance) / scale
    if x == 0.0:
        return 1.0
    if x >= lobes:
        return 0.0
    return float(np.sinc(x) * np.sinc(x / lobes))


def _axis_plan(input_extent: int, output_extent: int, lobes: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    scale = max(input_extent / output_extent, 1.0)
    plans: list[tuple[np.ndarray, np.ndarray]] = []
    for output_coordinate in range(output_extent):
        coordinate = (output_coordinate + 0.5) * input_extent / output_extent - 0.5
        unclamped = _strict_support_indices(coordinate, scale, lobes)
        raw_weights = np.asarray(
            [_lanczos_weight(coordinate - index, scale=scale, lobes=lobes) for index in unclamped],
            dtype=np.float32,
        )
        weights = np.asarray(raw_weights / raw_weights.sum(dtype=np.float32), dtype=np.float32)
        indices = np.asarray([min(max(index, 0), input_extent - 1) for index in unclamped], dtype=np.int64)
        plans.append((indices, weights))
    return tuple(plans)


def _resize_aa_reference(source: np.ndarray, *, width: int, height: int, lobes: int) -> np.ndarray:
    """Independent fp32 separable oracle for exact-support, replicate-mapped Lanczos."""
    input_height, input_width, channels = source.shape
    horizontal = _axis_plan(input_width, width, lobes)
    vertical = _axis_plan(input_height, height, lobes)
    intermediate = np.empty((input_height, width, channels), dtype=np.float32)
    output = np.empty((height, width, channels), dtype=np.float32)

    for source_y in range(input_height):
        for output_x, (indices, weights) in enumerate(horizontal):
            for channel in range(channels):
                value = np.float32(0.0)
                for source_x, weight in zip(indices, weights, strict=True):
                    value = np.float32(value + np.float32(source[source_y, source_x, channel] * weight))
                intermediate[source_y, output_x, channel] = value

    for output_y, (indices, weights) in enumerate(vertical):
        for output_x in range(width):
            for channel in range(channels):
                value = np.float32(0.0)
                for source_y, weight in zip(indices, weights, strict=True):
                    value = np.float32(value + np.float32(intermediate[source_y, output_x, channel] * weight))
                output[output_y, output_x, channel] = value
    return output


def _full_support_mask(input_extent: int, output_extent: int, lobes: int) -> np.ndarray:
    scale = input_extent / output_extent
    mask = np.empty(output_extent, dtype=np.bool_)
    for output_coordinate in range(output_extent):
        coordinate = (output_coordinate + 0.5) * input_extent / output_extent - 0.5
        indices = _strict_support_indices(coordinate, scale, lobes)
        mask[output_coordinate] = all(0 <= index < input_extent for index in indices)
    return mask


def test_interpolation_vocabulary_and_resize_variants_are_canonical() -> None:
    """v1-resize-antialias acceptance 1: the three canonical tokens and runtime variants select resize AA."""
    expected = (
        *EXISTING_RESIZE_TOKENS[:-1],
        *AA_TOKENS,
        "area",
        "trilinear",
        "tetrahedral",
        "linear",
    )
    assert get_args(px.core.Interpolation) == expected

    values = np.arange(9 * 7, dtype=np.float32).reshape(7, 9, 1)
    source = _frame(values)
    variants = {
        "lanczos2-aa": ("LANCZOS2_AA", "lanczos 2 aa"),
        "lanczos3-aa": ("Lanczos.3-AA", "lanczos3aa"),
        "lanczos4-aa": ("LANCZOS-4_AA", "lanczos 4.aa"),
    }
    for canonical, accepted_variants in variants.items():
        expected_data = _host(px.transform.resize(source, width=4, height=3, interpolation=canonical))
        for variant in accepted_variants:
            np.testing.assert_array_equal(
                _host(px.transform.resize(source, width=4, height=3, interpolation=variant)),
                expected_data,
            )


@pytest.mark.parametrize(("token", "lobes"), tuple(zip(AA_TOKENS, (2, 3, 4), strict=True)))
@pytest.mark.parametrize(("width", "height"), ((4, 3), (4, 11), (13, 3)))
def test_antialiased_lanczos_matches_exact_support_numpy_oracle(
    token: str,
    lobes: int,
    width: int,
    height: int,
) -> None:
    """v1-resize-antialias acceptance 2, 3, and 5: pure and mixed reductions match an independent oracle."""
    rng = np.random.default_rng(3811)
    values = rng.uniform(-4.0, 8.0, size=(7, 9, 3)).astype(np.float32)
    expected = _resize_aa_reference(values, width=width, height=height, lobes=lobes)

    actual = _host(px.transform.resize(_frame(values), width=width, height=height, interpolation=token))

    # 3e-5 covers CUDA sinf versus NumPy sinc and sequential fp32 accumulation
    # while remaining below 0.0003% of this fixture's 12-unit value span.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-5)


@pytest.mark.parametrize("token", AA_TOKENS)
def test_antialiased_lanczos_normalizes_before_replicate_mapping(token: str) -> None:
    """v1-resize-antialias acceptance 3: exact support includes edge taps before replicate mapping."""
    lobes = int(token.removeprefix("lanczos").removesuffix("-aa"))
    edge_impulse = np.zeros((5, 7, 1), dtype=np.float32)
    edge_impulse[:, 0, 0] = 3.0
    expected = _resize_aa_reference(edge_impulse, width=2, height=3, lobes=lobes)
    actual = _host(px.transform.resize(_frame(edge_impulse), width=2, height=3, interpolation=token))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-6)

    constant = np.full((5, 7, 2), (-2.5, 6.0), dtype=np.float32)
    constant_result = _host(px.transform.resize(_frame(constant), width=2, height=3, interpolation=token))
    expected_constant = np.broadcast_to(np.asarray((-2.5, 6.0), dtype=np.float32), (3, 2, 2))
    np.testing.assert_allclose(constant_result, expected_constant, rtol=0.0, atol=2e-6)


@pytest.mark.parametrize(("aa_token", "point_token"), tuple(AA_TO_POINT.items()))
@pytest.mark.parametrize(("width", "height"), ((7, 5), (13, 11), (7, 11), (13, 5)))
def test_nonshrinking_antialiased_lanczos_is_bit_identical_to_point_sampled(
    aa_token: str,
    point_token: str,
    width: int,
    height: int,
) -> None:
    """v1-resize-antialias acceptance 4: same-size and enlargement reuse point-sampled Lanczos bits."""
    values = np.random.default_rng(384).uniform(-4.0, 8.0, size=(5, 7, 2)).astype(np.float32)
    source = _frame(values)
    antialiased = _host(px.transform.resize(source, width=width, height=height, interpolation=aa_token))
    point_sampled = _host(px.transform.resize(source, width=width, height=height, interpolation=point_token))
    np.testing.assert_array_equal(antialiased, point_sampled)


def test_existing_resize_output_bits_remain_frozen_characterization() -> None:
    """characterization: freeze issue #38's pre-change existing-token output on CUDA device 0.

    v1-resize-antialias acceptance 6: the hashes cover reduction, enlargement,
    same-size, and both mixed-axis directions. Correctness remains owned by the
    existing independent resize oracles; remove these hashes only when the
    bit-invariance requirement is deliberately superseded.
    """
    values = np.random.default_rng(3807).uniform(-4.0, 8.0, size=(7, 9, 3)).astype(np.float32)
    source = _frame(values, channels=("left", "middle", "right"))
    cases = ((4, 3), (13, 11), (9, 7), (5, 11), (13, 4))
    for token in EXISTING_RESIZE_TOKENS:
        digest = hashlib.sha256()
        for width, height in cases:
            result = _host(px.transform.resize(source, width=width, height=height, interpolation=token))
            digest.update(np.asarray((width, height), dtype=np.int64).tobytes())
            digest.update(result.tobytes(order="C"))
        assert digest.hexdigest() == _BASELINE_DIGESTS[token]


@pytest.mark.parametrize(
    ("width", "height", "explicit"),
    ((4, 3, "area"), (4, 11, "area"), (13, 3, "area"), (9, 7, "lanczos4"), (13, 11, "lanczos4")),
)
def test_resize_auto_default_remains_bit_identical(width: int, height: int, explicit: str) -> None:
    """v1-resize-antialias acceptance 6: auto remains area for any reduction and lanczos4 otherwise."""
    values = np.random.default_rng(386).uniform(-1.0, 2.0, size=(7, 9, 2)).astype(np.float32)
    source = _frame(values)
    automatic = _host(px.transform.resize(source, width=width, height=height))
    selected = _host(px.transform.resize(source, width=width, height=height, interpolation=explicit))
    np.testing.assert_array_equal(automatic, selected)


def test_lanczos3_aa_matches_pillow_on_the_fixed_full_support_corpus() -> None:
    """v1-resize-antialias acceptance 7: Pillow 12.3.0 agrees on the fixed full-support interior corpus."""
    import PIL
    from PIL import Image

    assert PIL.__version__ == "12.3.0"
    dimensions = ((97, 83, 31, 29, 575), (53, 47, 19, 17, 143))
    seeds = (38, 3807, 7, 1234, 65537)
    worst = 0.0
    for input_width, input_height, output_width, output_height, expected_mask_size in dimensions:
        horizontal = _full_support_mask(input_width, output_width, 3)
        vertical = _full_support_mask(input_height, output_height, 3)
        mask = vertical[:, None] & horizontal[None, :]
        assert int(mask.sum()) == expected_mask_size
        assert mask.any()
        for seed in seeds:
            values = np.random.default_rng(seed).uniform(-4.0, 8.0, size=(input_height, input_width)).astype(np.float32)
            actual = _host(
                px.transform.resize(
                    _frame(values[..., None]),
                    width=output_width,
                    height=output_height,
                    interpolation="lanczos3-aa",
                )
            )[..., 0]
            pillow = np.asarray(
                Image.fromarray(values, mode="F").resize(
                    (output_width, output_height),
                    Image.Resampling.LANCZOS,
                    box=None,
                    reducing_gap=None,
                ),
                dtype=np.float32,
            )
            worst = max(worst, float(np.max(np.abs(actual[mask] - pillow[mask]))))

            replicate = _resize_aa_reference(
                values[..., None],
                width=output_width,
                height=output_height,
                lobes=3,
            )[..., 0]
            np.testing.assert_allclose(actual[~mask], replicate[~mask], rtol=0.0, atol=3e-5)
    assert worst <= 2e-5, f"production-path Pillow interior max absolute error was {worst:.9g}"


@pytest.mark.parametrize("token", AA_TOKENS)
def test_antialiased_lanczos_preserves_frame_and_unclamped_channel_contract(token: str) -> None:
    """v1-resize-antialias acceptance 8: AA is float32, per-channel, unclamped, private, and metadata-stable."""
    values = np.asarray(
        [
            [[-4.0, 8.0], [3.0, -2.0], [7.0, 1.0], [-1.0, 6.0], [5.0, -3.0]],
            [[8.0, -4.0], [-2.0, 3.0], [1.0, 7.0], [6.0, -1.0], [-3.0, 5.0]],
            [[-4.0, 8.0], [3.0, -2.0], [7.0, 1.0], [-1.0, 6.0], [5.0, -3.0]],
        ],
        dtype=np.float32,
    )
    values[..., 0] -= 10.0
    values[..., 1] += 10.0
    source = _frame(
        values,
        colorspace="Rec.2020",
        gamma="PQ",
        channels=("depth", "mask"),
        matrix="BT.2020",
    )
    lobes = int(token.removeprefix("lanczos").removesuffix("-aa"))
    expected = _resize_aa_reference(values, width=3, height=2, lobes=lobes)

    result = px.transform.resize(source, width=3, height=2, interpolation=token)

    np.testing.assert_allclose(_host(result), expected, rtol=0.0, atol=3e-5)
    assert result.dtype == np.dtype(np.float32)
    assert result.data.data.ptr != source.data.data.ptr
    assert result is not source
    assert (result.width, result.height) == (3, 2)
    assert (result.colorspace, result.gamma, result.channels, result.matrix) == (
        "Rec.2020",
        "PQ",
        ("depth", "mask"),
        "BT.2020",
    )
    assert float(_host(result).min()) < 0.0
    assert float(_host(result).max()) > 1.0


def _non_resize_calls(token: str) -> tuple[tuple[str, Callable[[], object]], ...]:
    import cupy as cp

    from_cases = (
        ("from_uyvy422", [128, 16, 128, 235], np.uint8, {"width": 2, "height": 1}),
        ("from_v210", np.zeros(32, dtype=np.uint32), np.uint32, {"width": 2, "height": 1}),
        ("from_nv12", [16, 235, 16, 235, 128, 128], np.uint8, {"width": 2, "height": 2}),
        (
            "from_p010",
            np.asarray([64, 940, 64, 940, 512, 512], dtype=np.uint16) << 6,
            np.uint16,
            {"width": 2, "height": 2},
        ),
        ("from_yuv420p", [16, 235, 16, 235, 128, 128], np.uint8, {"width": 2, "height": 2}),
        ("from_yuv422p", [16, 235, 128, 128], np.uint8, {"width": 2, "height": 1}),
    )
    calls: list[tuple[str, Callable[[], object]]] = []
    for name, values, dtype, kwargs in from_cases:
        buffer = cp.asarray(np.asarray(values, dtype=dtype))
        function = getattr(px.io, name)
        calls.append(
            (
                name,
                lambda function=function, buffer=buffer, kwargs=kwargs: function(buffer, **kwargs, interpolation=token),
            )
        )

    ycbcr_values = np.zeros((4, 6, 3), dtype=np.float32)
    ycbcr_values[..., 1:] = 0.5
    ycbcr = _frame(
        ycbcr_values,
        colorspace="Rec.709",
        gamma="Rec.709",
        channels=("Y", "Cb", "Cr"),
        matrix="BT.709",
    )
    for name in ("to_uyvy422", "to_v210", "to_nv12", "to_p010", "to_yuv420p", "to_yuv422p"):
        function = getattr(px.io, name)
        calls.append((name, lambda function=function: function(ycbcr, interpolation=token)))

    rgb = _frame(np.zeros((3, 4, 3), dtype=np.float32), channels=("R", "G", "B"))
    calls.append(
        (
            "warp_affine",
            lambda: px.transform.warp_affine(rgb, np.eye(2, 3, dtype=np.float32), interpolation=token),
        )
    )
    calls.append(("merge", lambda: px.composite.merge(rgb, rgb, interpolation=token)))
    axis = np.linspace(0.0, 1.0, 2, dtype=np.float32)
    lut_data = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    lut = px.core.Lut(data=cp.asarray(lut_data))
    calls.append(("apply_lut", lambda: px.color.apply_lut(rgb, lut=lut, interpolation=token)))
    return tuple(calls)


@pytest.mark.parametrize(
    "token",
    ("lanczos2-aa", "LANCZOS2_AA", "lanczos3-aa", "Lanczos.3 AA", "lanczos4-aa", "LANCZOS4AA"),
)
def test_non_resize_interpolation_subsets_reject_antialiased_lanczos_before_pixel_work(token: str) -> None:
    """v1-resize-antialias acceptance 9: all non-resize Interpolation subsets stay closed against AA variants."""
    for name, call in _non_resize_calls(token):
        with pytest.raises(ValueError) as error:
            call()
        message = _assert_actionable(error)
        assert repr(token) in message
        how = message.partition("; how=")[2]
        assert all(candidate not in how for candidate in AA_TOKENS), name


def test_resize_fail_fast_boundaries_include_the_expanded_canonical_subset() -> None:
    """v1-resize-antialias acceptance 9: resize preserves input errors and advertises its expanded subset."""
    import cupy as cp

    source = _frame(np.zeros((3, 4, 1), dtype=np.float32))
    with pytest.raises(ValueError) as unknown:
        px.transform.resize(source, width=2, height=2, interpolation="unknown")
    unknown_message = _assert_actionable(unknown)
    assert all(token in unknown_message.partition("; how=")[2] for token in AA_TOKENS)

    with pytest.raises(ValueError) as non_string:
        px.transform.resize(source, width=2, height=2, interpolation=3)  # type: ignore[arg-type]
    _assert_actionable(non_string)
    with pytest.raises(ValueError) as raw_array:
        px.transform.resize(cp.zeros((3, 4, 1), dtype=cp.float32), width=2, height=2)  # type: ignore[arg-type]
    _assert_actionable(raw_array)

    non_float = px.core.Frame(
        data=cp.zeros((3, 4, 1), dtype=cp.float16),
        colorspace="ACEScg",
        gamma="linear",
        channels=("signal",),
    )
    with pytest.raises(ValueError) as dtype_error:
        px.transform.resize(non_float, width=2, height=2, interpolation="lanczos3-aa")
    _assert_actionable(dtype_error)
    with pytest.raises(ValueError) as size_error:
        px.transform.resize(source, width=2, interpolation="lanczos3-aa")
    _assert_actionable(size_error)


def test_resize_antialias_docstring_is_self_contained_and_llm_readable() -> None:
    """v1-resize-antialias acceptance 10: resize docstring states selection, widening, edge, and oracle limits."""
    docstring = inspect.getdoc(px.transform.resize)
    assert docstring is not None
    for required in (
        *AA_TOKENS,
        "explicit",
        "shrinking axis",
        "s = max(input / output, 1)",
        "exact support",
        "replicate",
        "point-sampled",
        "area",
        "lanczos4",
        "does not clamp",
        "Pillow 12.3.0",
        "full-support interior",
    ):
        assert required in docstring
