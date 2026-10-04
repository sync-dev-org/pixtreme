"""Contract tests for Frame structure and the pixtreme public surface."""

from __future__ import annotations

import inspect

import pytest

import pixtreme as px


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-003")
def test_frame_has_the_four_color_metadata_fields_and_data() -> None:
    """Frame exposes pixels together with colorspace, gamma, channels, and matrix metadata."""
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


@pytest.mark.req("REQ-PIX-015")
@pytest.mark.req("REQ-PIX-105")
def test_dlpack_protocol_delegates_consumer_arguments_to_data() -> None:
    """Frame delegates DLPack requests, including the consumer stream, to its GPU pixel storage."""

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
    source = px.core.Frame.model_construct(data=probe, colorspace="sRGB", gamma="sRGB", channels=("R", "G", "B"))

    assert source.__dlpack__(stream=23, max_version=(1, 0), dl_device=(2, 7), copy=False) is probe.capsule
    assert probe.received == (23, (1, 0), (2, 7), False)
    assert source.__dlpack_device__() == (2, 7)


@pytest.mark.req("REQ-PIX-015")
@pytest.mark.req("REQ-PIX-105")
def test_tensor_helpers_are_absent_in_favor_of_the_dlpack_protocol() -> None:
    """Frame exposes GPU array interchange through the DLPack protocol without duplicate tensor helper methods."""
    assert not hasattr(px.core.Frame, "to_tensor")
    for name in ("to_tensor", "to_dlpack", "from_dlpack"):
        assert not hasattr(px, name)


@pytest.mark.req("REQ-PIX-017")
def test_public_api_is_the_feature_minimum() -> None:
    """The root package exposes public modules and version without duplicate operation aliases."""
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
        "fonts",
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


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-003")
def test_directional_color_signatures_match_the_declarative_contract() -> None:
    """Public color conversion functions expose the declared directional signatures and color arguments."""
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


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-003")
@pytest.mark.req("REQ-PIX-005")
def test_color_transform_signature_integrates_optional_tonemap() -> None:
    """RGB color conversion exposes the optional rendering transform in its public call signature."""
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


@pytest.mark.req("REQ-PIX-010")
def test_stack_images_signature_uses_one_positional_collection_and_keyword_controls() -> None:
    """Image stacking accepts one positional collection and keyword controls for direction and adaptation."""
    signature = inspect.signature(px.transform.stack)

    assert tuple(signature.parameters) == ("images", "direction", "adapt")
    assert signature.parameters["images"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["direction"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["direction"].default == "vertical"
    assert signature.parameters["adapt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["adapt"].default is False


@pytest.mark.req("REQ-PIX-001")
def test_shuffle_signature_uses_keyword_only_adapt_and_output_collector() -> None:
    """Channel shuffle accepts keyword-only adaptation and output channel declarations."""
    signature = inspect.signature(px.channel.shuffle)

    assert tuple(signature.parameters) == ("adapt", "outputs")
    assert signature.parameters["adapt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["adapt"].default is False
    assert signature.parameters["outputs"].kind is inspect.Parameter.VAR_KEYWORD


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-002")
def test_from_format_signatures_match_each_static_format_contract() -> None:
    """Named-format constructors accept optional matrix metadata without changing the other color arguments."""
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


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.parametrize(
    "operation_name",
    ("quantize", "dequantize", "legal_to_full", "full_to_legal"),
)
def test_value_operation_signatures_require_the_bit_depth_claim(operation_name: str) -> None:
    """Value conversion signatures require a declared bit depth except where range conversion defaults to eight bits."""
    operation = getattr(px.values, operation_name)
    signature = inspect.signature(operation)

    assert tuple(signature.parameters) == ("frame", "bit_depth")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameter = signature.parameters["bit_depth"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    expected_default = 8 if operation_name in {"legal_to_full", "full_to_legal"} else inspect.Parameter.empty
    assert parameter.default == expected_default


@pytest.mark.req("REQ-PIX-008")
def test_cast_dtype_signature_requires_the_dtype_claim() -> None:
    """Numeric dtype casting requires a keyword-only destination dtype."""
    signature = inspect.signature(px.values.cast_dtype)

    assert tuple(signature.parameters) == ("frame", "dtype")
    assert signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameter = signature.parameters["dtype"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.req("REQ-PIX-007")
def test_image_io_signatures_are_keyword_only_after_the_primary_inputs() -> None:
    """Image reading accepts orientation and image writing accepts dtype as trailing keyword-only choices."""
    read = inspect.signature(px.io.read_image)
    assert tuple(read.parameters) == (
        "path",
        "channels",
        "unchanged",
        "colorspace",
        "gamma",
        "apply_exif_orientation",
    )
    assert read.parameters["path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name, default in (
        ("channels", None),
        ("unchanged", False),
        ("colorspace", None),
        ("gamma", None),
        ("apply_exif_orientation", True),
    ):
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


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-017")
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
    """Channel shuffle rejects non-float32 Frame pixels and explains recoding and bit-depth conversion routes."""
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


@pytest.mark.req("REQ-PIX-003")
@pytest.mark.req("REQ-PIX-017")
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
    """Color conversion rejects non-float32 Frame pixels and explains recoding and bit-depth conversion routes."""
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


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-102")
def test_frame_constructor_signature_requires_explicit_keyword_metadata() -> None:
    """Array import accepts color metadata, including matrix, only through explicit keyword arguments."""
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


@pytest.mark.req("REQ-PIX-017")
def test_frame_boundary_functions_expose_only_the_array_exit_contract() -> None:
    """Generic GPU array export belongs to the I/O module and is absent from Frame methods."""
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


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-017")
def test_frame_rejects_extra_model_fields() -> None:
    """Frame construction rejects unrecognized metadata fields instead of retaining them silently."""
    import cupy as cp

    with pytest.raises(ValueError):
        px.core.Frame(
            data=cp.zeros((1, 1, 3), dtype=cp.float32),
            colorspace="sRGB",
            gamma="sRGB",
            channels=("R", "G", "B"),
            range="full",
        )
