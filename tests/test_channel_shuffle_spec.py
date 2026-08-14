"""Specification, contract, and property tests for ``channel.shuffle``."""

from __future__ import annotations

import ast
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


def test_shuffle_signature_public_surface_and_output_order() -> None:
    """v1-channel-shuffle acceptance 1 and 19: shuffle is the sole kwargs-only channel operation."""
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
    """v1-channel-shuffle acceptance 2 and 6: output labels and sources use one strict uniform grammar.

    Sources are built lazily inside the test so that GPU-less collection never initializes CUDA (I-60).
    """
    outputs = outputs_factory()
    with pytest.raises(ValueError) as error:
        px.channel.shuffle(**outputs)
    _assert_actionable(error)


def test_shuffle_requires_outputs_and_a_frame_source() -> None:
    """v1-channel-shuffle acceptance 3: empty and constants-only calls cannot establish Frame metadata."""
    with pytest.raises(ValueError) as empty_error:
        px.channel.shuffle()
    with pytest.raises(ValueError) as constants_error:
        px.channel.shuffle(Y=0.0, Cb=0.5, Cr=0.5)

    assert "zero output" in _assert_actionable(empty_error)
    assert "constants only" in _assert_actionable(constants_error)


def test_source_lookup_uses_first_matching_label_and_bit_exact_reuse() -> None:
    """v1-channel-shuffle acceptance 4 and 22: lookup selects the first label and copies its bits repeatedly."""
    first_bits = np.asarray([[0x80000000, 0x7FC00001], [0xBF800000, 0x3FC00000]], dtype=np.uint32)
    second_bits = np.asarray([[0x00000000, 0x7FC01234], [0x40000000, 0xC0200000]], dtype=np.uint32)
    source = _frame(
        np.stack((first_bits.view(np.float32), second_bits.view(np.float32)), axis=-1),
        channels=("signal", "signal"),
    )

    result = px.channel.shuffle(copy_a=(source, "signal"), copy_b=(source, "signal"))

    np.testing.assert_array_equal(_host(result).view(np.uint32), np.stack((first_bits, first_bits), axis=-1))


def test_missing_source_label_names_available_labels_and_repair() -> None:
    """v1-channel-shuffle acceptance 4: missing labels report the source label set and correction path."""
    source = _frame([0.1, 0.2, 0.3])

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(depth=(source, "Z"))

    message = _assert_actionable(error)
    assert "Z" in message and str(source.channels) in message and "choose" in message


def test_fill_and_literal_labels_preserve_scene_values_without_semantic_checks() -> None:
    """v1-channel-shuffle acceptance 5-7 and 20-22: fills and literal relabels are value-only routing."""
    source = _frame([[[0.25], [0.75]]], colorspace="Rec.2020", channels=("Y",), matrix="bt2020")

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


@pytest.mark.parametrize("adapt", (1, 0.0, None, "false"))
def test_adapt_is_a_reserved_strict_bool_option(adapt: object) -> None:
    """v1-channel-shuffle acceptance 8: adapt cannot be used as an output label and accepts only bool."""
    source = _frame([0.0, 0.0, 0.0])
    outputs = {"adapt": (source, "R")} if adapt is None else {"adapt": adapt}

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(**outputs)

    message = _assert_actionable(error)
    assert "bool" in message and "reserved" in message and "different label" in message


def test_first_frame_after_leading_fills_defines_geometry_and_metadata() -> None:
    """v1-channel-shuffle acceptance 9: the first Frame source, not a leading fill, is the master."""
    master = _frame(
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        colorspace="ACEScg",
        gamma="2.6",
        matrix="native",
    )

    result = px.channel.shuffle(fill=0.5, blue=(master, "B"))

    assert (result.width, result.height, result.colorspace, result.gamma) == (2, 2, "ACEScg", "2.6")


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
    """v1-channel-shuffle acceptance 10: every source is fp32 for both adapt modes."""
    source = _frame([1], channels=("Y",), dtype=dtype)

    for adapt in (False, True):
        with pytest.raises(ValueError) as error:
            px.channel.shuffle(adapt=adapt, Y=(source, "Y"))
        message = _assert_actionable(error)
        assert "float32" in message
        positions = tuple(message.index(route) for route in routes)
        assert positions == tuple(sorted(positions))


@pytest.mark.parametrize(
    ("field", "source_kwargs", "required"),
    (
        ("width", {"values": np.zeros((2, 3, 3), dtype=np.float32)}, ("2", "3", "resize")),
        ("height", {"values": np.zeros((3, 2, 3), dtype=np.float32)}, ("2", "3", "resize")),
        ("colorspace", {"colorspace": "sRGB"}, ("ACEScg", "sRGB")),
        ("gamma", {"gamma": "srgb"}, ("linear", "srgb")),
    ),
)
def test_source_mismatch_errors_name_field_values_and_repair(
    field: str,
    source_kwargs: dict[str, object],
    required: tuple[str, ...],
) -> None:
    """v1-channel-shuffle acceptance 10-11: geometry and default metadata mismatches are actionable."""
    master = _frame(np.zeros((2, 2, 3), dtype=np.float32), colorspace="ACEScg", gamma="linear")
    resolved_source_kwargs = {"colorspace": "ACEScg", "gamma": "linear", **source_kwargs}
    values = resolved_source_kwargs.pop("values", np.zeros((2, 2, 3), dtype=np.float32))
    source = _frame(values, **resolved_source_kwargs)

    for adapt in (False, True) if field in {"width", "height"} else (False,):
        with pytest.raises(ValueError) as error:
            px.channel.shuffle(adapt=adapt, master=(master, "R"), source=(source, "G"))
        message = _assert_actionable(error)
        assert field in message and all(value in message for value in required)


def test_adapt_matches_public_rgb_to_rgb_composition_bit_exactly() -> None:
    """v1-channel-shuffle acceptance 11-12 and 22: adapt equals explicit rgb_to_rgb then default shuffle."""
    master = _frame(
        np.asarray([[[0.02, 0.08, 0.20], [0.10, 0.30, 0.70]]], dtype=np.float32),
        colorspace="ACEScg",
        gamma="linear",
    )
    source = _frame(
        np.asarray([[[0.02, 0.30, 0.90], [0.80, 0.10, 0.04]]], dtype=np.float32),
        colorspace="sRGB",
        gamma="srgb",
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


def test_adapt_transforms_each_source_identity_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-channel-shuffle acceptance 11: repeated routes adapt one source Frame only once."""
    import pixtreme._channel.shuffle as implementation

    master = _frame([0.1, 0.2, 0.3], colorspace="ACEScg")
    source = _frame([0.4, 0.5, 0.6], colorspace="sRGB", gamma="srgb")
    calls: list[px.core.Frame] = []
    original = implementation.rgb_to_rgb

    def counted(frame: px.core.Frame, **kwargs: object) -> px.core.Frame:
        calls.append(frame)
        return original(frame, **kwargs)

    monkeypatch.setattr(implementation, "rgb_to_rgb", counted)
    px.channel.shuffle(adapt=True, master=(master, "R"), green=(source, "G"), blue=(source, "B"))

    assert calls == [source]


def test_adapt_preserves_public_color_conversion_fail_fast_as_actionable_error() -> None:
    """v1-channel-shuffle acceptance 12: unsupported rgb_to_rgb inputs remain explicit three-part errors."""
    master = _frame([0.2], channels=("Y",), gamma="linear")
    source = _frame([0.4], channels=("Y",), gamma="srgb")

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(adapt=True, master=(master, "Y"), source=(source, "Y"))

    message = _assert_actionable(error)
    assert all(value in message for value in ("rgb_to_rgb", "R", "G", "B"))


def test_shuffle_allocates_contiguous_storage_without_mutating_inputs() -> None:
    """v1-channel-shuffle acceptance 13: output is a new contiguous Frame and inputs remain unchanged."""
    source = _frame(np.arange(12, dtype=np.float32).reshape(2, 2, 3), matrix="bt709")
    original_data = _host(source).copy()
    original_metadata = source.model_dump(exclude={"data"})

    result = px.channel.shuffle(B=(source, "B"), R=(source, "R"))

    assert result is not source and result.data.data.ptr != source.data.data.ptr
    assert result.dtype == np.dtype(np.float32) and result.data.flags.c_contiguous
    np.testing.assert_array_equal(_host(source), original_data)
    assert source.model_dump(exclude={"data"}) == original_metadata


@pytest.mark.parametrize(
    ("outputs", "expected_matrix"),
    (
        ({"R": "bt709"}, None),
        ({"Z": "bt709"}, None),
        ({"R": "bt709", "Y": "bt601"}, None),
        ({"Y": "bt709"}, "bt709"),
        ({"Y": "native"}, "native"),
        ({"Y": None}, None),
        ({"Y": "bt709", "Cb": "fill"}, "bt709"),
        ({"Y": "bt709", "Cb": "bt709"}, "bt709"),
        ({"Y": "native", "Cb": "native", "Cr": "fill"}, "native"),
        ({"Y": "bt709", "Cb": None}, None),
        ({"Y": "fill", "Cb": "fill", "Z": "bt709"}, None),
    ),
)
def test_matrix_provenance_decision_table(outputs: dict[str, str | None], expected_matrix: str | None) -> None:
    """v1-channel-shuffle acceptance 14-18 and 22: matrix follows the independent provenance decision table."""
    routed: dict[str, tuple[px.core.Frame, str] | float] = {}
    for output_label, matrix in outputs.items():
        if matrix == "fill":
            routed[output_label] = 0.5
        else:
            source = _frame([0.25], channels=("source",), matrix=matrix)
            routed[output_label] = (source, "source")

    result = px.channel.shuffle(**routed)

    assert result.matrix == expected_matrix


@pytest.mark.parametrize("adapt", (False, True))
def test_conflicting_matrix_claims_fail_without_implicit_rematrix(adapt: bool) -> None:
    """v1-channel-shuffle acceptance 15-16 and 20: distinct claims fail and adapt never rematrices."""
    first = _frame([0.1, 0.2, 0.3], matrix="bt601")
    second = _frame([0.4, 0.5, 0.6], matrix="bt709")

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(adapt=adapt, Y=(first, "R"), Cb=(second, "G"))

    message = _assert_actionable(error)
    assert all(value in message for value in ("bt601", "bt709", "Y", "Cb", "rematrix"))


def test_adapt_matrix_claim_comes_from_call_site_source_not_temporary_frame() -> None:
    """v1-channel-shuffle acceptance 15: adapt provenance uses the original source Frame matrix."""
    master = _frame([0.1, 0.2, 0.3], colorspace="ACEScg", matrix="bt709")
    source = _frame([0.4, 0.5, 0.6], colorspace="sRGB", gamma="srgb", matrix="native")

    result = px.channel.shuffle(adapt=True, Z=(master, "R"), Y=(source, "R"))

    assert result.matrix == "native"


def test_shuffle_source_binds_adaptation_to_the_public_color_operation_without_local_kernels() -> None:
    """v1-channel-shuffle acceptance 20 and 24: AST binds adaptation to rgb_to_rgb and no local GPU kernel."""
    import pixtreme._channel.shuffle as shuffle_module

    tree = ast.parse(inspect.getsource(shuffle_module))
    prepare_sources = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_sources"
    )
    called_names = tuple(
        node.func.id
        for node in ast.walk(prepare_sources)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    kernel_constructors = tuple(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"RawKernel", "ElementwiseKernel"}
    )

    assert called_names.count("rgb_to_rgb") == 1
    assert not kernel_constructors


def test_shuffle_docstring_maps_each_adapt_mode_to_its_conversion_recipe() -> None:
    """v1-channel-shuffle acceptance 11: adapt paragraphs bind False and True to opposite routing recipes."""
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
