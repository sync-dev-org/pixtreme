"""Canonical closed-token type aliases for the public API."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast, get_args

Colorspace: TypeAlias = Literal["sRGB", "Rec.709", "Rec.2020", "ACES2065-1", "ACEScg", "S-Gamut3", "S-Gamut3.Cine"]
Gamma: TypeAlias = Literal[
    "linear", "srgb", "rec709", "bt1886", "pq", "hlg", "s-log3", "logc4", "cineon", "2.2", "2.4", "2.6"
]
Matrix: TypeAlias = Literal["bt601", "bt709", "bt2020", "native"]
Dtype: TypeAlias = Literal["float32", "float16", "uint8", "uint16", "uint32"]
Layout: TypeAlias = Literal["HWC", "NHWC", "CHW", "NCHW"]
Tonemap: TypeAlias = Literal["aces-1.3", "aces-1.3-lut", "aces-2.0", "aces-2.0-lut", "bt2408"]
Range: TypeAlias = Literal["legal", "full"]
Interpolation: TypeAlias = Literal[
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
]
Border: TypeAlias = Literal["mirror", "replicate", "wrap", "constant"]
ChromaSiting: TypeAlias = Literal["left", "center", "topleft"]
StackDirection: TypeAlias = Literal["vertical", "horizontal"]
SobelDirection: TypeAlias = Literal["x", "y", "magnitude"]
TemplateMatchingMethod: TypeAlias = Literal[
    "sqdiff", "sqdiff_normed", "ccorr", "ccorr_normed", "ccoeff", "ccoeff_normed"
]
Blend: TypeAlias = Literal[
    "normal", "lighten", "add", "screen", "darken", "multiply", "difference", "overlay", "hardlight", "softlight"
]
Alpha: TypeAlias = Literal["premultiplied", "straight"]
Antialiasing: TypeAlias = Literal["distance", "supersample", "off"]
TextLanguage: TypeAlias = Literal["ja", "zh-hans", "zh-hant", "ko"]
TextAnchor: TypeAlias = Literal[
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
]
TextAlign: TypeAlias = Literal["left", "center", "right", "justify"]
TextFont: TypeAlias = Literal["sans", "mono"]
GeneratorKind: TypeAlias = Literal["linear", "radial"]
ColorBarsStandard: TypeAlias = Literal[
    "arib-std-b28", "smpte-rp219", "bt2111-hlg", "bt2111-pq", "bt2111-pq-full", "full-100", "full-75"
]
ColorBarsOutput: TypeAlias = Literal["normalized", "code"]
MorphologyShape: TypeAlias = Literal["disk", "square"]
ImageFormat: TypeAlias = Literal["jpeg", "png", "tiff", "jpeg2000", "webp", "bmp", "pnm"]
TiffCompression: TypeAlias = Literal["none", "lzw"]
ExrCompression: TypeAlias = Literal["none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"]
VectorBlurShutter: TypeAlias = Literal["centered", "forward", "backward"]


def _tokens(alias: object) -> tuple[str, ...]:
    """Return the runtime token set for one canonical Literal alias."""
    return cast(tuple[str, ...], get_args(alias))


_COLORSPACE_TOKENS = _tokens(Colorspace)
_GAMMA_TOKENS = _tokens(Gamma)
_MATRIX_TOKENS = _tokens(Matrix)
_DTYPE_TOKENS = _tokens(Dtype)
_LAYOUT_TOKENS = _tokens(Layout)
_TONEMAP_TOKENS = _tokens(Tonemap)
_TONEMAP_ACES_TOKENS = _TONEMAP_TOKENS[0:4:2]
_TONEMAP_LUT_TOKENS = _TONEMAP_TOKENS[1:4:2]
_TONEMAP_DIRECT_TOKENS = _TONEMAP_TOKENS[4:]
_RANGE_TOKENS = _tokens(Range)
_INTERPOLATION_TOKENS = _tokens(Interpolation)
_BORDER_TOKENS = _tokens(Border)
_CHROMA_SITING_TOKENS = _tokens(ChromaSiting)
_STACK_DIRECTION_TOKENS = _tokens(StackDirection)
_SOBEL_DIRECTION_TOKENS = _tokens(SobelDirection)
_TEMPLATE_MATCHING_METHOD_TOKENS = _tokens(TemplateMatchingMethod)
_BLEND_TOKENS = _tokens(Blend)
_ALPHA_TOKENS = _tokens(Alpha)
_ANTIALIASING_TOKENS = _tokens(Antialiasing)
_TEXT_LANGUAGE_TOKENS = _tokens(TextLanguage)
_TEXT_ANCHOR_TOKENS = _tokens(TextAnchor)
_TEXT_ALIGN_TOKENS = _tokens(TextAlign)
_TEXT_FONT_TOKENS = _tokens(TextFont)
_GENERATOR_KIND_TOKENS = _tokens(GeneratorKind)
_COLOR_BARS_STANDARD_TOKENS = _tokens(ColorBarsStandard)
_COLOR_BARS_OUTPUT_TOKENS = _tokens(ColorBarsOutput)
_MORPHOLOGY_SHAPE_TOKENS = _tokens(MorphologyShape)
_IMAGE_FORMAT_TOKENS = _tokens(ImageFormat)
_TIFF_COMPRESSION_TOKENS = _tokens(TiffCompression)
_EXR_COMPRESSION_TOKENS = _tokens(ExrCompression)
_VECTOR_BLUR_SHUTTER_TOKENS = _tokens(VectorBlurShutter)
