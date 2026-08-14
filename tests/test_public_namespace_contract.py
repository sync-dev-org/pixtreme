"""Exact public namespace contracts for the v1 namespace redesign."""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, get_args, get_origin

import numpy as np
import pytest

import pixtreme as px

ROOT_MODULES = (
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
)

IO_FUNCTIONS = (
    "read_image",
    "write_image",
    "read_header",
    "read_lut",
    "decode_image",
    "encode_image",
    "from_array",
    "to_array",
    "from_uyvy422",
    "to_uyvy422",
    "from_v210",
    "to_v210",
    "from_nv12",
    "to_nv12",
    "from_p010",
    "to_p010",
    "from_yuv420p",
    "to_yuv420p",
    "from_yuv422p",
    "to_yuv422p",
    "from_yuv444p",
    "to_yuv444p",
    "from_yuva444p",
    "to_yuva444p",
)

FUNCTION_MODULES = {
    "io": IO_FUNCTIONS,
    "color": (
        "apply_lut",
        "gamma_to_linear",
        "hsv_to_rgb",
        "linear_to_gamma",
        "rgb_to_grayscale",
        "rgb_to_hsv",
        "rgb_to_rgb",
        "rgb_to_ycbcr",
        "ycbcr_to_rgb",
        "ycbcr_to_ycbcr",
        "equalize_histogram",
        "clahe",
    ),
    "filter": (
        "gaussian_blur",
        "box_blur",
        "median_blur",
        "bilateral_blur",
        "directional_blur",
        "zoom_blur",
        "spin_blur",
        "vector_blur",
        "lens_blur",
        "sobel",
        "laplacian",
        "difference_of_gaussians",
        "canny",
        "sharpen",
        "unsharp_mask",
        "convolve_box",
    ),
    "transform": ("resize", "warp_affine", "stack"),
    "draw": ("line", "polyline", "rectangle", "circle", "ellipse", "polygon", "text"),
    "generate": ("ramp", "grid", "checkerboard", "color_bars", "fractal_noise", "turbulent_noise", "grain"),
    "morphology": (
        "erosion",
        "dilation",
        "opening",
        "closing",
        "morphological_gradient",
        "white_tophat",
        "black_tophat",
    ),
    "metrics": ("psnr", "ssim", "ssim_map"),
    "feature": ("corner_harris", "match_template"),
    "values": ("quantize", "dequantize", "full_to_legal", "legal_to_full", "cast_dtype", "recode_dtype"),
    "channel": ("shuffle",),
    "composite": ("merge",),
}

_ERROR_PATH_CASES = (
    ("filter", "gaussian_blur", {"sigma": 1.0}),
    ("filter", "box_blur", {"size": 1}),
    ("filter", "median_blur", {"size": 1}),
    ("filter", "bilateral_blur", {"sigma_space": 1.0, "sigma_value": 1.0}),
    ("filter", "directional_blur", {"angle": 0.0, "length": 1.0}),
    ("filter", "zoom_blur", {"amount": 1.0}),
    ("filter", "spin_blur", {"angle": 1.0}),
    ("filter", "vector_blur", {"vector": object()}),
    ("filter", "lens_blur", {"radius": 1.0}),
    ("filter", "sobel", {}),
    ("filter", "laplacian", {}),
    ("filter", "difference_of_gaussians", {"sigma1": 1.0, "sigma2": 2.0}),
    ("filter", "sharpen", {"amount": 1.0}),
    ("filter", "unsharp_mask", {"sigma": 1.0, "amount": 1.0}),
    ("filter", "convolve_box", {"size": 1, "normalize": True}),
    ("color", "equalize_histogram", {}),
    ("color", "clahe", {}),
    ("morphology", "erosion", {"radius": 1}),
    ("morphology", "dilation", {"radius": 1}),
    ("morphology", "opening", {"radius": 1}),
    ("morphology", "closing", {"radius": 1}),
    ("morphology", "morphological_gradient", {"radius": 1}),
    ("morphology", "white_tophat", {"radius": 1}),
    ("morphology", "black_tophat", {"radius": 1}),
    ("values", "quantize", {"bit_depth": 8}),
    ("values", "dequantize", {"bit_depth": 8}),
    ("values", "cast_dtype", {"dtype": "float32"}),
    ("values", "recode_dtype", {"dtype": "float32"}),
    ("values", "legal_to_full", {"bit_depth": 8}),
    ("values", "full_to_legal", {"bit_depth": 8}),
    ("draw", "line", {"start": (0.0, 0.0), "end": (1.0, 1.0), "color": (1.0,), "thickness": 1.0}),
    ("draw", "polyline", {"points": ((0.0, 0.0), (1.0, 1.0)), "color": (1.0,), "thickness": 1.0}),
    (
        "draw",
        "rectangle",
        {"top_left": (0.0, 0.0), "bottom_right": (1.0, 1.0), "color": (1.0,), "fill": True},
    ),
    ("draw", "circle", {"center": (0.0, 0.0), "radius": 1.0, "color": (1.0,), "fill": True}),
    ("draw", "ellipse", {"center": (0.0, 0.0), "radii": (1.0, 1.0), "color": (1.0,), "fill": True}),
    ("draw", "polygon", {"points": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), "color": (1.0,)}),
    ("draw", "text", {"text": "x", "position": (0.0, 0.0), "size": 1.0, "color": (1.0,)}),
    ("feature", "corner_harris", {}),
)

ALIAS_TOKENS = {
    "Colorspace": ("sRGB", "Rec.709", "Rec.2020", "ACES2065-1", "ACEScg", "S-Gamut3", "S-Gamut3.Cine"),
    "Gamma": ("linear", "srgb", "rec709", "bt1886", "pq", "hlg", "s-log3", "logc4", "cineon", "2.2", "2.4", "2.6"),
    "Matrix": ("bt601", "bt709", "bt2020", "native"),
    "Dtype": ("float32", "float16", "uint8", "uint16", "uint32"),
    "Layout": ("HWC", "NHWC", "CHW", "NCHW"),
    "Tonemap": ("aces-1.3", "aces-1.3-lut", "aces-2.0", "aces-2.0-lut", "bt2408"),
    "Range": ("legal", "full"),
    "Interpolation": (
        "nearest",
        "bilinear",
        "bicubic",
        "b-spline",
        "mitchell",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "area",
        "trilinear",
        "tetrahedral",
    ),
    "Border": ("mirror", "replicate", "wrap", "constant"),
    "ChromaSiting": ("left", "center", "topleft"),
    "StackDirection": ("vertical", "horizontal"),
    "SobelDirection": ("x", "y", "magnitude"),
    "TemplateMatchingMethod": ("sqdiff", "sqdiff_normed", "ccorr", "ccorr_normed", "ccoeff", "ccoeff_normed"),
    "Blend": (
        "normal",
        "lighten",
        "add",
        "screen",
        "darken",
        "multiply",
        "difference",
        "overlay",
        "hardlight",
        "softlight",
    ),
    "Alpha": ("premultiplied", "straight"),
    "Antialiasing": ("distance", "supersample", "off"),
    "TextLanguage": ("ja", "zh-hans", "zh-hant", "ko"),
    "TextAnchor": (
        "top-left",
        "top-center",
        "top-right",
        "center-left",
        "center-center",
        "center-right",
        "baseline-left",
        "baseline-center",
        "baseline-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ),
    "TextAlign": ("left", "center", "right", "justify"),
    "TextFont": ("sans", "mono"),
    "GeneratorKind": ("linear", "radial"),
    "ColorBarsStandard": (
        "arib-std-b28",
        "smpte-rp219",
        "bt2111-hlg",
        "bt2111-pq",
        "bt2111-pq-full",
        "full-100",
        "full-75",
    ),
    "ColorBarsOutput": ("normalized", "code"),
    "MorphologyShape": ("disk", "square"),
    "ImageFormat": ("jpeg", "png", "tiff", "jpeg2000", "webp", "bmp", "pnm"),
    "TiffCompression": ("none", "lzw"),
    "ExrCompression": ("none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"),
    "VectorBlurShutter": ("centered", "forward", "backward"),
}

ALIAS_SECTIONS = {
    "Layout": "layout",
    "Gamma": "gamma",
    "Colorspace": "colorspace",
    "Tonemap": "tonemap",
    "Matrix": "matrix",
    "Range": "range",
    "Interpolation": "interpolation",
    "StackDirection": "stack direction",
    "SobelDirection": "sobel direction",
    "TemplateMatchingMethod": "template matching method",
    "ChromaSiting": "chroma siting",
    "TextLanguage": "language",
    "TextAnchor": "anchor",
    "TextAlign": "text align",
    "TextFont": "text font",
    "Blend": "blend",
    "Alpha": "alpha",
    "Antialiasing": "aa",
    "GeneratorKind": "generator kind",
    "ColorBarsStandard": "color bars standard",
    "ColorBarsOutput": "color bars output",
    "MorphologyShape": "morphology shape",
    "Border": "border",
    "VectorBlurShutter": "vector blur shutter",
    "Dtype": "dtype",
    "ImageFormat": "image format",
    "TiffCompression": "TIFF compression",
    "ExrCompression": "EXR compression",
}


def _public_names(module: ModuleType) -> set[str]:
    return {name for name in vars(module) if not name.startswith("_") or name == "__version__"}


def _table_tokens(markdown: str, heading: str) -> tuple[str, ...]:
    section = markdown.split(f"## {heading}\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return tuple(
        cells[1].strip().removeprefix("`").removesuffix("`")
        for line in section.splitlines()
        if line.startswith("| `")
        for cells in (line.split("|"),)
    )


def test_root_surface_is_exact_and_version_matches_distribution() -> None:
    """v1-public-namespace acceptance 1-2: root is 13 modules plus the installed version string."""
    assert px.__all__ == (*ROOT_MODULES, "__version__")
    assert _public_names(px) == set(px.__all__)
    assert all(isinstance(getattr(px, name), ModuleType) for name in ROOT_MODULES)
    assert px.__version__ == importlib.metadata.version("pixtreme")
    assert isinstance(px.__version__, str) and px.__version__


def test_public_modules_expose_the_exact_function_type_helper_and_alias_contract() -> None:
    """v1-public-namespace acceptance 3 and 7-8: every leaf has one exact module owner."""
    assert sum(len(leaves) for leaves in FUNCTION_MODULES.values()) == 89
    for module_name, leaves in FUNCTION_MODULES.items():
        module = getattr(px, module_name)
        expected_all = (*leaves, *(("ImageHeader",) if module_name == "io" else ()))
        expected_public = set(expected_all)
        assert module.__all__ == expected_all
        assert _public_names(module) == expected_public
        assert all(inspect.isfunction(getattr(module, leaf)) for leaf in leaves)

    expected_core_all = ("Frame", "Lut", "channels", *ALIAS_TOKENS)
    expected_core = set(expected_core_all)
    assert px.core.__all__ == expected_core_all
    assert _public_names(px.core) == expected_core
    assert inspect.isclass(px.core.Frame)
    assert inspect.isclass(px.core.Lut)
    assert inspect.isclass(px.io.ImageHeader)
    assert inspect.isfunction(px.core.channels)


def test_legacy_root_and_module_imports_fail_in_fresh_processes() -> None:
    """v1-public-namespace acceptance 4-5: removed paths have no aliases, shims, or import fallback."""
    legacy_root = (
        "Frame",
        "Lut",
        "ImageHeader",
        "channels",
        "read_image",
        "write_image",
        "read_header",
        "read_lut",
        "decode_image",
        "encode_image",
        "from_array",
        "from_uyvy422",
        "from_v210",
        "from_nv12",
        "from_p010",
        "from_yuv420p",
        "from_yuv422p",
        "from_yuv444p",
        "from_yuva444p",
    )
    assert all(not hasattr(px, name) for name in legacy_root)
    assert not hasattr(px, "blur")
    assert not hasattr(px, "analyze")

    snippets = [
        *(f"from pixtreme import {name}" for name in legacy_root),
        "import pixtreme.blur",
        "import pixtreme.analyze",
    ]
    for snippet in snippets:
        result = subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, check=False, text=True, timeout=10
        )
        assert result.returncode != 0, snippet


def test_frame_is_data_metadata_properties_and_dlpack_only() -> None:
    """v1-public-namespace acceptance 6: Frame keeps its structural surface and loses all nine exit methods."""
    assert tuple(px.core.Frame.model_fields) == ("data", "colorspace", "gamma", "channels", "matrix")
    assert all(isinstance(getattr(px.core.Frame, name), property) for name in ("width", "height", "shape", "dtype"))
    assert all(callable(getattr(px.core.Frame, name)) for name in ("__dlpack__", "__dlpack_device__"))
    removed_methods = (
        "to_array",
        "to_uyvy422",
        "to_v210",
        "to_nv12",
        "to_p010",
        "to_yuv420p",
        "to_yuv422p",
        "to_yuv444p",
        "to_yuva444p",
    )
    assert all(not hasattr(px.core.Frame, name) for name in removed_methods)

    for name in removed_methods:
        signature = inspect.signature(getattr(px.io, name))
        parameters = tuple(signature.parameters.values())
        assert parameters[0].name == "frame"
        assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:])


def test_literal_aliases_and_vocabulary_tables_are_identical() -> None:
    """v1-public-namespace acceptance 9 and 12: aliases, runtime tokens, and parsed docs tables stay identical."""
    from pixtreme._core import vocabulary as runtime_vocabulary

    vocabulary_path = Path(__file__).resolve().parents[1] / "docs" / "vocabulary.md"
    if not vocabulary_path.is_file():
        import pytest

        pytest.skip("repo-only documentation contract: docs/vocabulary.md is absent from this distribution")
    markdown = vocabulary_path.read_text(encoding="utf-8")

    for alias_name, expected_tokens in ALIAS_TOKENS.items():
        alias = getattr(px.core, alias_name)
        assert get_origin(alias) is Literal
        assert get_args(alias) == expected_tokens
        constant_name = f"_{''.join(('_' + character if character.isupper() else character) for character in alias_name).lstrip('_').upper()}_TOKENS"
        assert getattr(runtime_vocabulary, constant_name) == get_args(alias)
        assert _table_tokens(markdown, ALIAS_SECTIONS[alias_name]) == expected_tokens

    assert not hasattr(px.core, "Channels")


@pytest.mark.parametrize(
    ("module_name", "function_name", "kwargs"),
    _ERROR_PATH_CASES,
    ids=tuple(f"{module_name}.{function_name}" for module_name, function_name, _ in _ERROR_PATH_CASES),
)
def test_moved_operation_errors_name_only_the_canonical_public_path(
    module_name: str, function_name: str, kwargs: dict[str, object]
) -> None:
    """v1-public-namespace acceptance 8 and 12."""
    function = getattr(getattr(px, module_name), function_name)

    with pytest.raises(ValueError) as error:
        function(object(), **kwargs)

    message = str(error.value)
    why, separator, remainder = message.partition("; what=")
    assert separator
    _, separator, how = remainder.partition("; how=")
    assert separator
    canonical_path = f"{module_name}.{function_name}"
    assert why.startswith(f"why={canonical_path} ")
    assert how.endswith(f"px.{canonical_path}")
    public_root, public_module, public_name = how.rsplit(" ", maxsplit=1)[-1].split(".")
    assert public_root == "px"
    assert getattr(getattr(px, public_module), public_name) is function
    assert f"px.{function_name}" not in how
    assert "px.filter.equalize_histogram" not in message
    assert "px.filter.clahe" not in message
    assert "px.analyze.corner_harris" not in message


def test_exr_write_dtype_subset_is_alias_derived_and_shared_by_validation() -> None:
    """v1-public-namespace acceptance 9 and 12: EXR dtype registry dataflow derives from the canonical Dtype alias."""
    from pixtreme._core import vocabulary as runtime_vocabulary
    from pixtreme._io import dtype as runtime_dtype

    assert runtime_dtype._EXR_WRITE_DTYPES == ("float16", "float32", "uint32")
    assert set(runtime_dtype._EXR_WRITE_DTYPES) <= set(runtime_vocabulary._DTYPE_TOKENS)
    assert runtime_dtype._WRITE_NATIVE_DTYPES["EXR"] == frozenset(runtime_dtype._EXR_WRITE_DTYPES)

    tree = ast.parse(inspect.getsource(runtime_dtype))
    exr_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_EXR_WRITE_DTYPES" for target in node.targets)
    )
    exr_dataflow_names = {node.id for node in ast.walk(exr_assignment.value) if isinstance(node, ast.Name)}
    assert {"_DTYPE_TOKENS", "_is_exr_write_dtype"} <= exr_dataflow_names

    native_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_WRITE_NATIVE_DTYPES" for target in node.targets)
    )
    assert isinstance(native_assignment.value, ast.Dict)
    exr_value = next(
        value
        for key, value in zip(native_assignment.value.keys, native_assignment.value.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "EXR"
    )
    assert isinstance(exr_value, ast.Call) and isinstance(exr_value.func, ast.Name) and exr_value.func.id == "frozenset"
    assert len(exr_value.args) == 1
    assert isinstance(exr_value.args[0], ast.Name) and exr_value.args[0].id == "_EXR_WRITE_DTYPES"

    for token in runtime_dtype._EXR_WRITE_DTYPES:
        frame = SimpleNamespace(dtype=np.dtype(token))
        assert runtime_dtype._prepare_exr_write_frame(frame, dtype=token) is frame

    with pytest.raises(ValueError) as error:
        runtime_dtype._prepare_exr_write_frame(object(), dtype="uint16")
    assert repr(runtime_dtype._EXR_WRITE_DTYPES) in str(error.value)
