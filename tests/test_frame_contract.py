"""Contract tests for Frame structure and the pixtreme public surface."""

from __future__ import annotations

import ast
import inspect

import pytest

import pixtreme as px


def test_frame_raises_use_the_shared_actionable_error_contract() -> None:
    """REQ-API-012 structural contract: Frame raises use the shared helper rather than local or plain messages."""
    import pixtreme._core.frame as frame_module

    tree = ast.parse(inspect.getsource(frame_module))
    local_helpers = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_actionable_error"
    ]
    shared_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "pixtreme._core.errors"
        for alias in node.names
    ]
    target_raises = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id in {"ValueError", "TypeError", "RuntimeError"}
    ]

    assert not local_helpers
    assert "_actionable_error" in shared_imports
    assert target_raises
    for node in target_raises:
        assert node.exc is not None
        assert isinstance(node.exc, ast.Call)
        assert len(node.exc.args) == 1
        message = node.exc.args[0]
        assert isinstance(message, ast.Call), f"raise at line {node.lineno} does not call _actionable_error"
        assert isinstance(message.func, ast.Name)
        assert message.func.id == "_actionable_error"


def test_frame_has_the_four_color_metadata_fields_and_data() -> None:
    """v1-color-semantics acceptance 1: Frame exposes matrix beside its existing color axes."""
    assert set(px.core.Frame.model_fields) == {"data", "colorspace", "gamma", "channels", "matrix"}
    rejected = {
        "range",
        "bit_depth",
        "chroma_sampling",
        "alpha",
        "scene_referred",
        "display_referred",
        "pixel_aspect_ratio",
        "interlace",
        "orientation",
        "source",
        "history",
    }
    assert rejected.isdisjoint(px.core.Frame.model_fields)


def test_dlpack_protocol_delegates_consumer_arguments_to_data() -> None:
    """v1-frame-core acceptance 12: Frame delegates DLPack protocol calls, including the consumer stream."""

    class DLPackProbe:
        def __init__(self) -> None:
            self.received: tuple[object, ...] | None = None
            self.capsule = object()

        def __dlpack__(
            self,
            *,
            stream: int | None = None,
            max_version: tuple[int, int] | None = None,
            dl_device: tuple[int, int] | None = None,
            copy: bool | None = None,
        ) -> object:
            self.received = (stream, max_version, dl_device, copy)
            return self.capsule

        def __dlpack_device__(self) -> tuple[int, int]:
            return (2, 7)

    probe = DLPackProbe()
    source = px.core.Frame.model_construct(data=probe, colorspace="sRGB", gamma="srgb", channels=("R", "G", "B"))

    assert source.__dlpack__(stream=23, max_version=(1, 0), dl_device=(2, 7), copy=False) is probe.capsule
    assert probe.received == (23, (1, 0), (2, 7), False)
    assert source.__dlpack_device__() == (2, 7)


def test_tensor_helpers_are_absent_in_favor_of_the_dlpack_protocol() -> None:
    """v1-frame-core acceptance 14: no tensor/DLPack helper duplicates the Python DLPack protocol."""
    assert not hasattr(px.core.Frame, "to_tensor")
    for name in ("to_tensor", "to_dlpack", "from_dlpack"):
        assert not hasattr(px, name)


def test_public_api_is_the_feature_minimum() -> None:
    """v1-public-namespace acceptance 1 and 4: root exports stay module-only."""
    assert px.__all__ == (
        "core",
        "io",
        "color",
        "filter",
        "transform",
        "draw",
        "generate",
        "morphology",
        "metrics",
        "feature",
        "values",
        "channel",
        "composite",
        "__version__",
    )
    for removed in (
        "Frame",
        "Lut",
        "ImageHeader",
        "channels",
        "from_array",
        "read_image",
        "frame",
        "recode_range",
        "unpack_uyvy422",
        "unpack_yuv420p",
        "unpack_yuv422p10le",
        "unpack_uyvy422_raw",
        "from_yuv422p10le",
    ):
        assert not hasattr(px, removed)


def test_directional_color_signatures_match_the_declarative_contract() -> None:
    """v1-color-semantics acceptance 9, 15, 20, 24-25, 40 and v1-hsv acceptance 1 fix signatures."""
    expected = {
        px.color.rgb_to_hsv: ("frame",),
        px.color.hsv_to_rgb: ("frame",),
        px.color.rgb_to_ycbcr: ("frame", "colorspace", "gamma", "matrix", "range", "bit_depth"),
        px.color.ycbcr_to_rgb: ("frame", "colorspace", "gamma", "matrix", "range", "bit_depth"),
        px.color.rgb_to_grayscale: ("frame", "colorspace", "gamma", "matrix"),
        px.color.gamma_to_linear: ("frame", "gamma"),
        px.color.linear_to_gamma: ("frame", "gamma"),
        px.color.ycbcr_to_ycbcr: (
            "frame",
            "colorspace",
            "gamma",
            "input_matrix",
            "output_matrix",
            "input_range",
            "input_bit_depth",
            "output_range",
            "output_bit_depth",
        ),
    }
    for operation, names in expected.items():
        signature = inspect.signature(operation)
        assert tuple(signature.parameters) == names
        assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name in names[1:]:
            assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(px.color.linear_to_gamma).parameters["gamma"].default is inspect.Parameter.empty


def test_color_transform_signature_integrates_optional_tonemap() -> None:
    """v1-color-semantics acceptance 29-30: rgb_to_rgb integrates the optional rendering axis."""
    signature = inspect.signature(px.color.rgb_to_rgb)

    assert tuple(signature.parameters) == (
        "frame",
        "input_colorspace",
        "input_gamma",
        "output_colorspace",
        "output_gamma",
        "tonemap",
    )
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("input_colorspace", "input_gamma", "output_colorspace", "output_gamma", "tonemap"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_stack_images_signature_uses_one_positional_collection_and_keyword_controls() -> None:
    """v1-stack acceptance 1-2: stack has the exact collection, direction, and adapt grammar."""
    signature = inspect.signature(px.transform.stack)

    assert tuple(signature.parameters) == ("images", "direction", "adapt")
    assert signature.parameters["images"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["direction"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["direction"].default == "vertical"
    assert signature.parameters["adapt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["adapt"].default is False


def test_shuffle_signature_uses_keyword_only_adapt_and_output_collector() -> None:
    """v1-channel-shuffle acceptance 1: shuffle has the exact kwargs routing grammar."""
    signature = inspect.signature(px.channel.shuffle)

    assert tuple(signature.parameters) == ("adapt", "outputs")
    assert signature.parameters["adapt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["adapt"].default is False
    assert signature.parameters["outputs"].kind is inspect.Parameter.VAR_KEYWORD


def test_from_format_signatures_match_each_static_format_contract() -> None:
    """v1-color-semantics acceptance 5: format constructors expose matrix=None without changing other axes."""
    expected = {
        px.io.from_uyvy422: (
            ("buf", "width", "height", "colorspace", "gamma", "matrix", "range", "interpolation"),
            {"colorspace": None, "gamma": None, "matrix": None, "range": "legal", "interpolation": "bilinear"},
        ),
        px.io.from_v210: (
            ("buf", "width", "height", "colorspace", "gamma", "matrix", "range", "interpolation"),
            {"colorspace": None, "gamma": None, "matrix": None, "range": "legal", "interpolation": "bilinear"},
        ),
        px.io.from_nv12: (
            ("buf", "width", "height", "colorspace", "gamma", "matrix", "range", "siting", "interpolation"),
            {
                "colorspace": None,
                "gamma": None,
                "matrix": None,
                "range": "legal",
                "siting": "left",
                "interpolation": "bilinear",
            },
        ),
        px.io.from_p010: (
            ("buf", "width", "height", "colorspace", "gamma", "matrix", "range", "siting", "interpolation"),
            {
                "colorspace": None,
                "gamma": None,
                "matrix": None,
                "range": "legal",
                "siting": "left",
                "interpolation": "bilinear",
            },
        ),
        px.io.from_yuv420p: (
            (
                "buf",
                "width",
                "height",
                "bit_depth",
                "colorspace",
                "gamma",
                "matrix",
                "range",
                "siting",
                "interpolation",
            ),
            {
                "bit_depth": 8,
                "colorspace": None,
                "gamma": None,
                "matrix": None,
                "range": "legal",
                "siting": "left",
                "interpolation": "bilinear",
            },
        ),
        px.io.from_yuv422p: (
            ("buf", "width", "height", "bit_depth", "colorspace", "gamma", "matrix", "range", "interpolation"),
            {
                "bit_depth": 8,
                "colorspace": None,
                "gamma": None,
                "matrix": None,
                "range": "legal",
                "interpolation": "bilinear",
            },
        ),
        px.io.from_yuv444p: (
            ("buf", "width", "height", "bit_depth", "colorspace", "gamma", "matrix", "range"),
            {"bit_depth": 10, "colorspace": None, "gamma": None, "matrix": None, "range": "legal"},
        ),
        px.io.from_yuva444p: (
            ("buf", "width", "height", "bit_depth", "colorspace", "gamma", "matrix", "range"),
            {"bit_depth": 12, "colorspace": None, "gamma": None, "matrix": None, "range": "legal"},
        ),
    }

    for function, (names, defaults) in expected.items():
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == names
        assert signature.parameters["buf"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name in names[1:]:
            parameter = signature.parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            expected_default = defaults.get(name, inspect.Parameter.empty)
            assert parameter.default == expected_default


@pytest.mark.parametrize(
    "operation_name",
    ("quantize", "dequantize", "legal_to_full", "full_to_legal"),
)
def test_value_operation_signatures_require_the_bit_depth_claim(operation_name: str) -> None:
    """v1-subpackage-reorg acceptance 3-4: value signatures stay fixed except the range pair defaults to 8."""
    operation = getattr(px.values, operation_name)
    signature = inspect.signature(operation)

    assert tuple(signature.parameters) == ("frame", "bit_depth")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameter = signature.parameters["bit_depth"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    expected_default = 8 if operation_name in {"legal_to_full", "full_to_legal"} else inspect.Parameter.empty
    assert parameter.default == expected_default


def test_cast_dtype_signature_requires_the_dtype_claim() -> None:
    """v1-io acceptance 19: cast_dtype exposes one required keyword-only dtype token."""
    signature = inspect.signature(px.values.cast_dtype)

    assert tuple(signature.parameters) == ("frame", "dtype")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameter = signature.parameters["dtype"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_image_io_signatures_are_keyword_only_after_the_primary_inputs() -> None:
    """v1-exr-runtime-independence acceptance 1: write_image adds one keyword-only dtype selector."""
    read = inspect.signature(px.io.read_image)
    assert tuple(read.parameters) == ("path", "channels", "unchanged", "colorspace", "gamma")
    assert read.parameters["path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name, default in (("channels", None), ("unchanged", False), ("colorspace", None), ("gamma", None)):
        assert read.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert read.parameters[name].default is default

    write = inspect.signature(px.io.write_image)
    assert tuple(write.parameters) == (
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
    assert write.parameters["path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert write.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("quality", "compression", "compression_level", "lossless", "dwa_level", "bit_depth", "dtype"):
        assert write.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert write.parameters[name].default is None

    header = inspect.signature(px.io.read_header)
    assert tuple(header.parameters) == ("path",)


@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        ("float16", ("cast_dtype",)),
        ("uint8", ("recode_dtype", "dequantize")),
        ("uint16", ("recode_dtype", "dequantize")),
        ("uint32", ("recode_dtype",)),
    ),
)
def test_channel_shuffle_contract_rejects_non_float32_frame_data(
    dtype: str,
    routes: tuple[str, ...],
) -> None:
    """v1-channel-shuffle acceptance 10: shuffle errors prioritize recoding and retain bit-grid guidance."""
    import cupy as cp

    source = px.io.from_array(
        cp.zeros((1, 1, 3), dtype=dtype),
        colorspace="sRGB",
        gamma="linear",
        channels="RGB",
    )

    with pytest.raises(ValueError) as error:
        px.channel.shuffle(R=(source, "R"))
    positions = tuple(str(error.value).index(route) for route in routes)
    assert positions == tuple(sorted(positions))


@pytest.mark.parametrize(
    ("dtype", "routes"),
    (
        ("float16", ("cast_dtype",)),
        ("uint8", ("recode_dtype", "dequantize")),
        ("uint16", ("recode_dtype", "dequantize")),
        ("uint32", ("recode_dtype",)),
    ),
)
def test_color_transform_contract_rejects_non_float32_frame_data(
    dtype: str,
    routes: tuple[str, ...],
) -> None:
    """v1-recode-dtype acceptance 9: color errors prioritize recoding and retain bit-grid guidance."""
    import cupy as cp

    source = px.io.from_array(
        cp.zeros((1, 1, 3), dtype=dtype),
        colorspace="sRGB",
        gamma="linear",
        channels="RGB",
    )

    with pytest.raises(ValueError) as error:
        px.color.rgb_to_rgb(source)
    positions = tuple(str(error.value).index(route) for route in routes)
    assert positions == tuple(sorted(positions))


def test_frame_constructor_signature_requires_explicit_keyword_metadata() -> None:
    """v1-color-semantics acceptance 4: from_array adds optional matrix metadata."""
    signature = inspect.signature(px.io.from_array)
    assert tuple(signature.parameters) == (
        "data",
        "colorspace",
        "gamma",
        "channels",
        "matrix",
        "layout",
        "dtype",
        "bit_depth",
        "scale",
        "mean",
        "std",
        "copy",
    )
    assert signature.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("colorspace", "gamma", "channels"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    for name in ("matrix", "layout", "dtype", "bit_depth", "scale", "mean", "std", "copy"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_frame_boundary_functions_expose_only_the_array_exit_contract() -> None:
    """v1-public-namespace acceptance 10: io owns the generic array exit and Frame has no exits."""
    signature = inspect.signature(px.io.to_array)
    assert tuple(signature.parameters) == (
        "frame",
        "channels",
        "layout",
        "dtype",
        "bit_depth",
        "scale",
        "mean",
        "std",
        "out",
        "copy",
    )
    for name in tuple(signature.parameters)[1:]:
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
    assert not hasattr(px.core.Frame, "to_numpy")

    assert not hasattr(px.core.Frame, "to_array")


def test_frame_rejects_extra_model_fields() -> None:
    """v1-frame-core acceptance 9: direct model construction cannot smuggle rejected metadata into Frame."""
    import cupy as cp

    with pytest.raises(ValueError):
        px.core.Frame(
            data=cp.zeros((1, 1, 3), dtype=cp.float32),
            colorspace="sRGB",
            gamma="srgb",
            channels=("R", "G", "B"),
            range="full",
        )
