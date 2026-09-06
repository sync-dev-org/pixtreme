"""Canonical closed-token type aliases for the public API."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypeVar, cast, get_args

ChromaticAdaptation: TypeAlias = Literal["Bradford", "CAT02", "CAT16", "von-Kries"]
ReferenceWhite: TypeAlias = Literal["D65", "D93", "D50", "ACES"]
Colorspace: TypeAlias = Literal[
    "sRGB",
    "Rec.709",
    "Rec.2020",
    "P3-DCI",
    "P3-D60",
    "P3-D65",
    "SMPTE-C",
    "ACES2065-1",
    "ACEScg",
    "S-Gamut",
    "S-Gamut3",
    "S-Gamut3.Cine",
    "ARRI-Wide-Gamut-3",
    "ARRI-Wide-Gamut-4",
    "Blackmagic-Wide-Gamut-Gen-5",
    "DaVinci-Wide-Gamut",
    "REDWideGamutRGB",
    "DRAGONcolor",
    "DRAGONcolor2",
    "REDcolor2",
    "REDcolor3",
    "REDcolor4",
    "Canon-Cinema-Gamut",
    "V-Gamut",
    "D-Gamut",
    "F-Gamut-C",
    "Apple-Wide-Gamut",
]
Gamma: TypeAlias = Literal[
    "linear",
    "sRGB",
    "Rec.709",
    "BT.1886",
    "PQ",
    "HLG",
    "ACEScc",
    "ACEScct",
    "S-Log",
    "S-Log2",
    "S-Log3",
    "ARRI-LogC3",
    "ARRI-LogC4",
    "Blackmagic-Film-Gen-5",
    "DaVinci-Intermediate",
    "RED-Log3G10",
    "REDlogFilm",
    "Canon-Log",
    "Canon-Log-2",
    "Canon-Log-3",
    "V-Log",
    "D-Log",
    "F-Log",
    "F-Log2",
    "N-Log",
    "L-Log",
    "Apple-Log",
    "Samsung-Log",
    "Cineon",
    "Gamma-2.2",
    "Gamma-2.4",
    "Gamma-2.5",
    "Gamma-2.6",
]
Matrix: TypeAlias = Literal["BT.601", "BT.709", "BT.2020", "native"]
Dtype: TypeAlias = Literal["float32", "float16", "uint8", "uint16", "uint32"]
Layout: TypeAlias = Literal["HWC", "NHWC", "CHW", "NCHW"]
Tonemap: TypeAlias = Literal["ACES-1.3", "ACES-2.0", "BT.2408"]
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
    "linear",
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
    "ARIB-STD-B28", "SMPTE-RP219", "BT.2111-HLG", "BT.2111-PQ", "BT.2111-PQ-full", "full-100", "full-75"
]
ColorBarsOutput: TypeAlias = Literal["normalized", "code"]
MorphologyShape: TypeAlias = Literal["disk", "square"]
ImageFormat: TypeAlias = Literal["jpeg", "png", "tiff", "jpeg2000", "webp", "bmp", "pnm"]
TiffCompression: TypeAlias = Literal["none", "lzw"]
ExrCompression: TypeAlias = Literal["none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"]
VectorBlurShutter: TypeAlias = Literal["centered", "forward", "backward"]


_PERMANENT_TOKEN_ALIASES = (
    ("logc4", "ARRI-LogC4"),
    ("2.2", "Gamma-2.2"),
    ("2.4", "Gamma-2.4"),
    ("2.6", "Gamma-2.6"),
)


_Token = TypeVar("_Token", bound=str)


def _tokens(alias: object) -> tuple[_Token, ...]:
    """Return the runtime token set for one canonical Literal alias."""
    return cast(tuple[_Token, ...], get_args(alias))


_CHROMATIC_ADAPTATION_TOKENS: tuple[ChromaticAdaptation, ...] = _tokens(ChromaticAdaptation)
_REFERENCE_WHITE_TOKENS: tuple[ReferenceWhite, ...] = _tokens(ReferenceWhite)
_COLORSPACE_TOKENS: tuple[Colorspace, ...] = _tokens(Colorspace)
_GAMMA_TOKENS: tuple[Gamma, ...] = _tokens(Gamma)
_MATRIX_TOKENS: tuple[Matrix, ...] = _tokens(Matrix)
_DTYPE_TOKENS: tuple[Dtype, ...] = _tokens(Dtype)
_LAYOUT_TOKENS: tuple[Layout, ...] = _tokens(Layout)
_TONEMAP_TOKENS: tuple[Tonemap, ...] = _tokens(Tonemap)
_TONEMAP_ACES_TOKENS = _TONEMAP_TOKENS[:2]
_TONEMAP_DIRECT_TOKENS = _TONEMAP_TOKENS[2:]
_RANGE_TOKENS: tuple[Range, ...] = _tokens(Range)
_INTERPOLATION_TOKENS: tuple[Interpolation, ...] = _tokens(Interpolation)
_BORDER_TOKENS: tuple[Border, ...] = _tokens(Border)
_CHROMA_SITING_TOKENS: tuple[ChromaSiting, ...] = _tokens(ChromaSiting)
_STACK_DIRECTION_TOKENS: tuple[StackDirection, ...] = _tokens(StackDirection)
_SOBEL_DIRECTION_TOKENS: tuple[SobelDirection, ...] = _tokens(SobelDirection)
_TEMPLATE_MATCHING_METHOD_TOKENS: tuple[TemplateMatchingMethod, ...] = _tokens(TemplateMatchingMethod)
_BLEND_TOKENS: tuple[Blend, ...] = _tokens(Blend)
_ALPHA_TOKENS: tuple[Alpha, ...] = _tokens(Alpha)
_ANTIALIASING_TOKENS: tuple[Antialiasing, ...] = _tokens(Antialiasing)
_TEXT_LANGUAGE_TOKENS: tuple[TextLanguage, ...] = _tokens(TextLanguage)
_TEXT_ANCHOR_TOKENS: tuple[TextAnchor, ...] = _tokens(TextAnchor)
_TEXT_ALIGN_TOKENS: tuple[TextAlign, ...] = _tokens(TextAlign)
_TEXT_FONT_TOKENS: tuple[TextFont, ...] = _tokens(TextFont)
_GENERATOR_KIND_TOKENS: tuple[GeneratorKind, ...] = _tokens(GeneratorKind)
_COLOR_BARS_STANDARD_TOKENS: tuple[ColorBarsStandard, ...] = _tokens(ColorBarsStandard)
_COLOR_BARS_OUTPUT_TOKENS: tuple[ColorBarsOutput, ...] = _tokens(ColorBarsOutput)
_MORPHOLOGY_SHAPE_TOKENS: tuple[MorphologyShape, ...] = _tokens(MorphologyShape)
_IMAGE_FORMAT_TOKENS: tuple[ImageFormat, ...] = _tokens(ImageFormat)
_TIFF_COMPRESSION_TOKENS: tuple[TiffCompression, ...] = _tokens(TiffCompression)
_EXR_COMPRESSION_TOKENS: tuple[ExrCompression, ...] = _tokens(ExrCompression)
_VECTOR_BLUR_SHUTTER_TOKENS: tuple[VectorBlurShutter, ...] = _tokens(VectorBlurShutter)
