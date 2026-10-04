"""Specification, contract, and property tests for ``channel.shuffle``."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pixtreme as px


def _frame(
    values: Any,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | tuple[str, ...] = "RGB",
    matrix: str | None = None,
    dtype: Any = np.float32,
) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    return px.io.from_array(
        cp.asarray(array),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix=matrix,
    )


def _host(frame: px.core.Frame) -> np.ndarray:
    return px.io.to_array(
        frame,
    ).get()


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> str:
    message = str(error.value)
    assert message.index("why=") < message.index("what=") < message.index("how=")
    return message


@pytest.mark.req("REQ-PIX-001")
def test_shuffle_signature_public_surface_and_output_order() -> None:
    """Channel shuffle exposes one public keyword-based call and returns output channels in the requested order."""
    signature = inspect.signature(px.channel.shuffle)
    assert tuple(signature.parameters) == ("adapt", "outputs")
    assert signature.parameters["adapt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["adapt"].default is False
    assert signature.parameters["outputs"].kind is inspect.Parameter.VAR_KEYWORD
    assert px.channel.__all__ == ("shuffle",)
    assert not hasattr(px.channel, "assemble_channels")
    assert not hasattr(px.channel, "channel_transform")

    source = _frame([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], colorspace="ACEScg")
    result = px.channel.shuffle(green=(source, "G"), negative=-1.25, blue=(source, "B"))

    assert result.channels == ("green", "negative", "blue")
    np.testing.assert_array_equal(
        _host(result),
        np.stack(
            (
                _host(source)[..., 1],
                np.full((1, 2), -1.25, dtype=np.float32),
                _host(source)[..., 2],
            ),
            axis=-1,
        ),
    )
    with pytest.raises(TypeError):
        px.channel.shuffle({"R": (source, "R")})  # type: ignore[misc]


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize(
    "outputs_factory",
    (
        lambda: {"": None},
        lambda: {"R": []},
        lambda: {"R": (object(), "R")},
        lambda: {"R": (None,)},
        lambda: {"R": (None, "R", "G")},
        lambda: {"R": (_frame([0.0], channels=("Y",)), "")},
        lambda: {"R": (_frame([0.0], channels=("Y",)), 1)},
        lambda: {"R": True},
        lambda: {"R": object()},
    ),
)
def test_shuffle_rejects_malformed_output_sources_with_actionable_errors(
    outputs_factory: Callable[[], dict[str, object]],
) -> None:
    """Channel shuffle rejects invalid output labels and source declarations with a consistent grammar and corrective
    error.
    """
    outputs = outputs_factory()
    with pytest.raises(ValueError) as error:
        px.channel.shuffle(**outputs)
    _assert_actionable(error)


@pytest.mark.req("REQ-PIX-001")
def test_shuffle_requires_outputs_and_a_frame_source() -> None:
    """Channel shuffle rejects calls without output channels or a Frame source because they cannot define image metadata."""
    with pytest.raises(ValueError) as empty_error:
        px.channel.shuffle()
    with pytest.raises(ValueError) as constants_error:
        px.channel.shuffle(Y=0.0, Cb=0.5, Cr=0.5)

    assert "zero output" in _assert_actionable(empty_error)
    assert "constants only" in _assert_actionable(constants_error)


@pytest.mark.req("REQ-PIX-001")
def test_source_lookup_uses_first_matching_label_and_bit_exact_reuse() -> None:
    """Channel shuffle uses the first matching source label and copies its pixel bits on repeated routes."""
    first_bits = np.asarray([[0x80000000, 0x7FC00001], [0xBF800000, 0x3FC00000]], dtype=np.uint32)
    second_bits = np.asarray([[0x00000000, 0x7FC01234], [0x40000000, 0xC0200000]], dtype=np.uint32)
    source = _frame(
        np.stack((first_bits.view(np.float32), second_bits.view(np.float32)), axis=-1),
        channels=("signal", "signal"),
    )

    result = px.channel.shuffle(copy_a=(source, "signal"), copy_b=(source, "signal"))

    np.testing.assert_array_equal(_host(result).view(np.uint32), np.stack((first_bits, first_bits), axis=-1))


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-017")
def test_missing_source_label_names_available_labels_and_repair() -> None:
    """Channel shuffle reports available labels and a correction when a requested source label is absent."""
    source = _frame([0.1, 0.2, 0.3])

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(depth=(source, "Z"))

    message = _assert_actionable(error)
    assert "Z" in message and str(source.channels) in message and "choose" in message


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-004")
@pytest.mark.req("REQ-PIX-103")
def test_fill_and_literal_labels_preserve_scene_values_without_semantic_checks() -> None:
    """Channel shuffle preserves out-of-range scene values when filling channels or relabeling them literally."""
    source = _frame([[[0.25], [0.75]]], colorspace="Rec.2020", channels=("Y",), matrix="BT.2020")

    result = px.channel.shuffle(
        **{
            "left.diffuse.R": (source, "Y"),
            "depth.Z": -3.25,
            "application label": 2,
            "high": 4.5,
        }
    )

    assert result.channels == ("left.diffuse.R", "depth.Z", "application label", "high")
    assert result.matrix is None
    np.testing.assert_array_equal(
        _host(result),
        np.asarray([[[0.25, -3.25, 2.0, 4.5], [0.75, -3.25, 2.0, 4.5]]], dtype=np.float32),
    )


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-003")
@pytest.mark.parametrize("adapt", (1, 0.0, None, "false"))
def test_adapt_is_a_reserved_strict_bool_option(adapt: object) -> None:
    """Channel shuffle reserves adapt as an option and accepts only a built-in boolean for it."""
    source = _frame([0.0, 0.0, 0.0])
    outputs = {"adapt": (source, "R")} if adapt is None else {"adapt": adapt}

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(**outputs)

    message = _assert_actionable(error)
    assert "bool" in message and "reserved" in message and "different label" in message


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-002")
def test_first_frame_after_leading_fills_defines_geometry_and_metadata() -> None:
    """Channel shuffle takes geometry and color metadata from the first Frame even when constant fills precede it."""
    master = _frame(
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        colorspace="ACEScg",
        gamma="Gamma-2.6",
        matrix="native",
    )

    result = px.channel.shuffle(fill=0.5, blue=(master, "B"))

    assert (result.width, result.height, result.colorspace, result.gamma) == (2, 2, "ACEScg", "Gamma-2.6")


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        (np.float16, ("cast_dtype",)),
        (np.uint8, ("recode_dtype", "dequantize")),
        (np.uint16, ("recode_dtype", "dequantize")),
    ),
)
def test_every_source_requires_float32_with_shared_conversion_guidance(
    dtype: Any,
    routes: tuple[str, ...],
) -> None:
    """Channel shuffle rejects every non-float32 Frame source and explains the public conversion route."""
    source = _frame([1], channels=("Y",), dtype=dtype)

    for adapt in (False, True):
        with pytest.raises(ValueError) as error:
            px.channel.shuffle(adapt=adapt, Y=(source, "Y"))
        message = _assert_actionable(error)
        assert "float32" in message
        positions = tuple(message.index(route) for route in routes)
        assert positions == tuple(sorted(positions))


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize(
    ("field", "source_kwargs", "required"),
    (
        ("width", {"values": np.zeros((2, 3, 3), dtype=np.float32)}, ("2", "3", "resize")),
        ("height", {"values": np.zeros((3, 2, 3), dtype=np.float32)}, ("2", "3", "resize")),
        ("colorspace", {"colorspace": "sRGB"}, ("ACEScg", "sRGB")),
        ("gamma", {"gamma": "sRGB"}, ("linear", "sRGB")),
    ),
)
def test_source_mismatch_errors_name_field_values_and_repair(
    field: str,
    source_kwargs: dict[str, object],
    required: tuple[str, ...],
) -> None:
    """Channel shuffle reports mismatched geometry or metadata values and how to align the sources."""
    master = _frame(np.zeros((2, 2, 3), dtype=np.float32), colorspace="ACEScg", gamma="linear")
    resolved_source_kwargs = {"colorspace": "ACEScg", "gamma": "linear", **source_kwargs}
    values = resolved_source_kwargs.pop("values", np.zeros((2, 2, 3), dtype=np.float32))
    source = _frame(values, **resolved_source_kwargs)

    for adapt in (False, True) if field in {"width", "height"} else (False,):
        with pytest.raises(ValueError) as error:
            px.channel.shuffle(adapt=adapt, master=(master, "R"), source=(source, "G"))
        message = _assert_actionable(error)
        assert field in message and all(value in message for value in required)


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-003")
def test_adapt_matches_public_rgb_to_rgb_composition_bit_exactly() -> None:
    """Adaptive channel shuffle produces the same pixels as explicit public color conversion followed by ordinary
    shuffling.
    """
    master = _frame(
        np.asarray([[[0.02, 0.08, 0.20], [0.10, 0.30, 0.70]]], dtype=np.float32),
        colorspace="ACEScg",
        gamma="linear",
    )
    source = _frame(
        np.asarray([[[0.02, 0.30, 0.90], [0.80, 0.10, 0.04]]], dtype=np.float32),
        colorspace="sRGB",
        gamma="sRGB",
    )

    result = px.channel.shuffle(
        adapt=True,
        constant=1.5,
        master_red=(master, "R"),
        adapted_green=(source, "G"),
        adapted_blue=(source, "B"),
    )
    transformed = px.color.rgb_to_rgb(
        source,
        output_colorspace=master.colorspace,
        output_gamma=master.gamma,
    )
    expected = px.channel.shuffle(
        constant=1.5,
        master_red=(master, "R"),
        adapted_green=(transformed, "G"),
        adapted_blue=(transformed, "B"),
    )

    np.testing.assert_array_equal(_host(result), _host(expected))


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-003")
@pytest.mark.req("REQ-PIX-017")
def test_adapt_preserves_public_color_conversion_fail_fast_as_actionable_error() -> None:
    """Adaptive channel shuffle reports unsupported color conversion with the same actionable cause as the public
    converter.
    """
    master = _frame([0.2], channels=("Y",), gamma="linear")
    source = _frame([0.4], channels=("Y",), gamma="sRGB")

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(adapt=True, master=(master, "Y"), source=(source, "Y"))

    message = _assert_actionable(error)
    assert all(value in message for value in ("rgb_to_rgb", "R", "G", "B"))


@pytest.mark.req("REQ-PIX-001")
def test_shuffle_allocates_contiguous_storage_without_mutating_inputs() -> None:
    """Channel shuffle returns new contiguous Frame storage and leaves every input unchanged."""
    source = _frame(np.arange(12, dtype=np.float32).reshape(2, 2, 3), matrix="BT.709")
    original_data = _host(source).copy()
    original_metadata = source.model_dump(exclude={"data"})

    result = px.channel.shuffle(B=(source, "B"), R=(source, "R"))

    assert result is not source and result.data.data.ptr != source.data.data.ptr
    assert result.dtype == np.dtype(np.float32) and result.data.flags.c_contiguous
    np.testing.assert_array_equal(_host(source), original_data)
    assert source.model_dump(exclude={"data"}) == original_metadata


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-002")
@pytest.mark.parametrize(
    ("outputs", "expected_matrix"),
    (
        ({"R": "BT.709"}, None),
        ({"Z": "BT.709"}, None),
        ({"R": "BT.709", "Y": "BT.601"}, None),
        ({"Y": "BT.709"}, "BT.709"),
        ({"Y": "native"}, "native"),
        ({"Y": None}, None),
        ({"Y": "BT.709", "Cb": "fill"}, "BT.709"),
        ({"Y": "BT.709", "Cb": "BT.709"}, "BT.709"),
        ({"Y": "native", "Cb": "native", "Cr": "fill"}, "native"),
        ({"Y": "BT.709", "Cb": None}, None),
        ({"Y": "fill", "Cb": "fill", "Z": "BT.709"}, None),
    ),
)
def test_matrix_provenance_decision_table(outputs: dict[str, str | None], expected_matrix: str | None) -> None:
    """Channel shuffle sets the output matrix from the sources according to the declared provenance cases."""
    routed: dict[str, tuple[px.core.Frame, str] | float] = {}
    for output_label, matrix in outputs.items():
        if matrix == "fill":
            routed[output_label] = 0.5
        else:
            source = _frame([0.25], channels=("source",), matrix=matrix)
            routed[output_label] = (source, "source")

    result = px.channel.shuffle(**routed)

    assert result.matrix == expected_matrix


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-002")
@pytest.mark.parametrize("adapt", (False, True))
def test_conflicting_matrix_claims_fail_without_implicit_rematrix(adapt: bool) -> None:
    """Channel shuffle rejects conflicting matrix claims without silently changing pixel values or matrices."""
    first = _frame([0.1, 0.2, 0.3], matrix="BT.601")
    second = _frame([0.4, 0.5, 0.6], matrix="BT.709")

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(adapt=adapt, Y=(first, "R"), Cb=(second, "G"))

    message = _assert_actionable(error)
    assert all(value in message for value in ("BT.601", "BT.709", "Y", "Cb", "rematrix"))


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-003")
def test_adapt_matrix_claim_comes_from_call_site_source_not_temporary_frame() -> None:
    """Adaptive channel shuffle derives the matrix claim from the original source Frame after color conversion."""
    master = _frame([0.1, 0.2, 0.3], colorspace="ACEScg", matrix="BT.709")
    source = _frame([0.4, 0.5, 0.6], colorspace="sRGB", gamma="sRGB", matrix="native")

    result = px.channel.shuffle(adapt=True, Z=(master, "R"), Y=(source, "R"))

    assert result.matrix == "native"


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-003")
@pytest.mark.req("REQ-PIX-017")
def test_shuffle_docstring_maps_each_adapt_mode_to_its_conversion_recipe() -> None:
    """The channel shuffle documentation explains the conversion recipe for adapt=False and adapt=True."""
    docstring = " ".join((inspect.getdoc(px.channel.shuffle) or "").split())
    modes = tuple(re.findall(r"With ``adapt=(False|True)``", docstring))
    false_recipe = re.search(r"With ``adapt=False``(?P<recipe>.*?)With ``adapt=True``", docstring)
    true_recipe = re.search(r"With ``adapt=True``(?P<recipe>.*?)Routing then", docstring)

    assert modes == ("False", "True")
    assert false_recipe is not None
    assert "all source colorspace and gamma metadata must match the first Frame" in false_recipe.group("recipe")
    assert true_recipe is not None
    assert re.search(
        r"each mismatched source identity is converted once through :func:`px\.color\.rgb_to_rgb`",
        true_recipe.group("recipe"),
    )
