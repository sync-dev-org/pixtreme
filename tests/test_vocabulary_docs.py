"""Documentation contract tests for the public token reference."""

from __future__ import annotations

import re
from pathlib import Path


def _section(markdown: str, heading: str) -> str:
    return markdown.split(f"## {heading}\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]


def _table_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _table_records(markdown: str, heading: str, headers: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    """Parse one named Markdown table into header-keyed records, independent of column order."""
    lines = _section(markdown, heading).splitlines()
    header_index, header_cells = next(
        (index, _table_cells(line))
        for index, line in enumerate(lines)
        if line.startswith("|") and set(_table_cells(line)) == set(headers)
    )
    assert len(header_cells) == len(headers)
    separator_cells = _table_cells(lines[header_index + 1])
    assert len(separator_cells) == len(header_cells)
    assert all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator_cells)
    records: list[dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = _table_cells(line)
        assert len(cells) == len(header_cells)
        row = dict(zip(header_cells, cells, strict=True))
        records.append({header: row[header] for header in headers})
    return tuple(records)


def _table_rows(markdown: str, heading: str, headers: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Parse a named Markdown table into exact semantic rows in the requested field order."""
    return tuple(tuple(record[header] for header in headers) for record in _table_records(markdown, heading, headers))


def _table_tokens(markdown: str, heading: str) -> tuple[str, ...]:
    return tuple(
        cells[1].strip().removeprefix("`").removesuffix("`")
        for line in _section(markdown, heading).splitlines()
        if line.startswith("| `")
        for cells in (line.split("|"),)
    )


def test_token_reference_is_the_english_canon_characterization() -> None:
    """characterization: I-80 replaces the internal Japanese vocabulary canon with the hosted English reference."""
    repository = Path(__file__).resolve().parents[1]
    token_reference = repository / "docs_site" / "tokens.md"
    retired_vocabulary = repository / "docs" / "vocabulary.md"
    if not token_reference.is_file() and not retired_vocabulary.is_file():
        import pytest

        pytest.skip("repo-only documentation contract: token reference is absent from this distribution")

    assert token_reference.is_file()
    assert not retired_vocabulary.exists()
    markdown = token_reference.read_text(encoding="utf-8")
    assert markdown.startswith("# Token Reference\n")
    assert re.search(r"[ぁ-んァ-ヶ一-龠]", markdown) is None


def test_documented_tokens_equal_the_validator_token_sets(vocabulary_markdown: str) -> None:
    """v1-format-boundary acceptance 39; v1-frame-core acceptance 16; v1-recode-dtype acceptance 8;
    v1-white-balance acceptance 1; v1-sony-tokens acceptance 1 and 12.

    Boundary/resizing/blurring docs, including v1-blur-vector acceptance 14 and
    v1-lut-extensions acceptance 27, match code token sets.
    """
    from pixtreme._color.lut import _LUT_INTERPOLATION_TOKENS
    from pixtreme._core.frame import _CHANNEL_LABELS, _COLORSPACE_TOKENS, _GAMMA_TOKENS, _LAYOUT_TOKENS, _MATRIX_TOKENS
    from pixtreme._core.value_domain import _RANGE_TOKENS
    from pixtreme._core.vocabulary import _CHROMATIC_ADAPTATION_TOKENS
    from pixtreme._filter.common import _BORDER_TOKENS
    from pixtreme._io.wire.sampling import _INTERPOLATION_TOKENS, _SITING_TOKENS, _TO_INTERPOLATION_TOKENS
    from pixtreme._transform.resize import _INTERPOLATION_TOKENS as _RESIZE_INTERPOLATION_TOKENS
    from pixtreme._transform.warp_affine import _BORDER_TOKENS as _WARP_BORDER_TOKENS
    from pixtreme._transform.warp_affine import _INTERPOLATION_TOKENS as _WARP_INTERPOLATION_TOKENS
    from pixtreme._values.cast import _DTYPE_TOKENS

    expected_interpolations = (
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
    )
    assert (
        _table_tokens(vocabulary_markdown, "channels")
        == _CHANNEL_LABELS
        == (
            "R",
            "G",
            "B",
            "H",
            "S",
            "V",
            "A",
            "Y",
            "Cb",
            "Cr",
            "Z",
        )
    )
    assert _table_tokens(vocabulary_markdown, "gamma") == _GAMMA_TOKENS
    assert _table_tokens(vocabulary_markdown, "colorspace") == _COLORSPACE_TOKENS
    assert _table_tokens(vocabulary_markdown, "chromatic adaptation") == _CHROMATIC_ADAPTATION_TOKENS
    assert _table_tokens(vocabulary_markdown, "matrix") == _MATRIX_TOKENS
    assert _table_tokens(vocabulary_markdown, "range") == _RANGE_TOKENS
    assert _table_tokens(vocabulary_markdown, "dtype") == _DTYPE_TOKENS
    assert _table_tokens(vocabulary_markdown, "interpolation") == expected_interpolations
    assert _RESIZE_INTERPOLATION_TOKENS == _WARP_INTERPOLATION_TOKENS == expected_interpolations[:9]
    assert _INTERPOLATION_TOKENS == expected_interpolations[:8]
    assert _TO_INTERPOLATION_TOKENS == ("nearest", "bilinear", "bicubic", "area")
    assert _LUT_INTERPOLATION_TOKENS == expected_interpolations[9:]
    assert _table_tokens(vocabulary_markdown, "chroma siting") == _SITING_TOKENS == ("left", "center", "topleft")
    assert _table_tokens(vocabulary_markdown, "layout") == _LAYOUT_TOKENS == ("HWC", "NHWC", "CHW", "NCHW")
    assert _table_tokens(vocabulary_markdown, "border") == _BORDER_TOKENS == _WARP_BORDER_TOKENS


def test_vocabulary_defines_every_frame_channel_gamma_and_colorspace_record(vocabulary_markdown: str) -> None:
    """v1-frame-core acceptance 16; v1-log-negative-extension acceptance 8; v1-sony-tokens acceptance 12;
    v1-arri-tokens acceptance 16 and 29; v1-blackmagic-tokens acceptance 50; v1-red-tokens acceptance 54 and 72.
    """
    assert _table_rows(
        vocabulary_markdown,
        "channels",
        ("Token", "Definition", "Standard or convention", "Notes"),
    ) == (
        (
            "`R`",
            "Red component of an RGB representation",
            "RGB standard named by the colorspace token",
            "Chromaticity and transfer are determined by the Frame colorspace and gamma",
        ),
        (
            "`G`",
            "Green component of an RGB representation",
            "RGB standard named by the colorspace token",
            "Chromaticity and transfer are determined by the Frame colorspace and gamma",
        ),
        (
            "`B`",
            "Blue component of an RGB representation",
            "RGB standard named by the colorspace token",
            "Chromaticity and transfer are determined by the Frame colorspace and gamma",
        ),
        (
            "`H`",
            "Hue turn",
            "HSV cylindrical coordinates",
            "Period 1; canonical output is `[0, 1)`, and the inverse conversion accepts all real values modulo 1",
        ),
        (
            "`S`",
            "HSV saturation",
            "HSV cylindrical coordinates",
            "In `[0, 1]` for nonnegative RGB input; the range is not enforced for arbitrary input",
        ),
        (
            "`V`",
            "HSV value",
            "HSV cylindrical coordinates",
            "The RGB maximum; an unbounded scene scale that may exceed 1",
        ),
        (
            "`A`",
            "Alpha or opacity component",
            "OpenEXR and general image-API convention",
            "Premultiplication state is not stored in Frame metadata",
        ),
        (
            "`Y`",
            "Luma or nonlinear luminance component",
            "ITU-T H.273 and grayscale convention",
            "Read as luma in YCbCr and as achromatic intensity in a one-channel Frame",
        ),
        (
            "`Cb`",
            "Blue-difference chroma component",
            "ITU-T H.273",
            "Longest matching treats it as one label in compact notation",
        ),
        (
            "`Cr`",
            "Red-difference chroma component",
            "ITU-T H.273",
            "Longest matching treats it as one label in compact notation",
        ),
        (
            "`Z`",
            "Depth component",
            "OpenEXR channel-naming convention",
            "Outside golden-path color processing",
        ),
    )
    assert _table_rows(
        vocabulary_markdown,
        "gamma",
        ("Token", "Definition", "Standard or convention", "Out-of-domain extension and notes"),
    ) == (
        (
            "`linear`",
            "Scene-linear light",
            "ACES working convention",
            "Identity extended naturally to all real values; fixed as **scene-referred** and never used to mean display-linear",
        ),
        (
            "`sRGB`",
            "Piecewise sRGB transfer",
            "IEC 61966-2-1",
            "Linear and power branches extend naturally below 0 and above 1; the standard transfer for the `sRGB` colorspace",
        ),
        (
            "`Rec.709`",
            "Rec.709 camera OETF",
            "ITU-R BT.709",
            "Linear and power branches extend naturally below 0 and above 1; independent of the `Rec.709` primaries token",
        ),
        (
            "`BT.1886`",
            "Reference-display EOTF",
            "ITU-R BT.1886",
            "Annex 1 ideal-black (`L_B = 0`) specialization; pure 2.4 power with sign-preserving reflection, numerically equivalent to `Gamma-2.4` but semantically distinct; matches industry production and conversion practice",
        ),
        (
            "`PQ`",
            "Perceptual quantizer",
            "SMPTE ST 2084 / ITU-R BT.2100",
            "Apply the standard formula to the nonnegative magnitude and reflect the negative side with preserved sign, `f(-x) = -f(x)`; absolute-luminance encoding",
        ),
        (
            "`HLG`",
            "Hybrid log-gamma",
            "ITU-R BT.2100",
            "Extend the piecewise low power and high logarithmic branches naturally with sign; scene-referred broadcast HDR transfer",
        ),
        (
            "`S-Log`",
            "S-Log camera log transfer",
            "Sony S-Log whitepaper; Sony S-Log2 technical paper (decoder branch)",
            "Public scene-linear reflectance `r` uses `x = r / 0.9`; Sony encoded IRE `y` uses `e = (64 + 876 * y) / 1023`; the lower linear branch below zero is the algebraic inverse of Sony's published S-Log1 decoder linear branch, not a separately published Sony forward equation, and extends without clipping or sign/magnitude mirroring; 0% / 18% / 90% reflection rounds to 10-bit code `90 / 394 / 636`",
        ),
        (
            "`S-Log2`",
            "S-Log2 camera log transfer",
            "Sony S-Log2 technical paper",
            "Uses the same public reflectance and legal-range embedding as S-Log with Sony's distinct positive log scale and negative linear slope; the lower linear branch extends below zero without clipping or sign/magnitude mirroring; 0% / 18% / 90% reflection rounds to 10-bit code `90 / 347 / 582`",
        ),
        (
            "`S-Log3`",
            "S-Log3 camera log transfer",
            "Sony S-Log3 specification",
            "S-Log3 applies the Sony piecewise formula directly to signed inputs; the lower linear branch extends below zero, maps linear 0 to `95 / 1023`, and does not use sign/magnitude mirroring; validate independently of S-Gamut colorspaces",
        ),
        (
            "`ARRI-LogC3`",
            "ARRI-LogC3 EI 800 camera log transfer",
            "ARRI Log C Curve Usage in VFX; OpenColorIO built-in transform",
            "EI 800 relative scene exposure with 18% gray at `400 / 1023`; the high branch uses ARRI's logarithmic equation and the tangent-derived lower linear branch extends to negative values without clipping or sign/magnitude mirroring; values above 1 remain unclipped; specify colorspace independently",
        ),
        (
            "`ARRI-LogC4`",
            "ARRI-LogC4 camera log transfer",
            "ARRI LogC4 specification",
            "ARRI-LogC4 applies the ARRI piecewise formula directly to signed inputs: the log branch covers `x >= t`, the lower linear branch covers `x < t`, and negative encoded values decode linearly without sign/magnitude mirroring; specify `ARRI-Wide-Gamut-4` independently",
        ),
        (
            "`Blackmagic-Film-Gen-5`",
            "Blackmagic Film Generation 5 camera log transfer",
            "Blackmagic Design Generation 5 Color Science",
            "Uses a natural logarithm above linear input `0.005`, with the published lower linear branch applied directly to negative values; decode uses the threshold derived from that branch; no clipping or sign/magnitude mirroring; specify colorspace independently",
        ),
        (
            "`DaVinci-Intermediate`",
            "DaVinci Intermediate working log transfer",
            "Blackmagic Design DaVinci Wide Gamut / Intermediate",
            "Uses a base-2 logarithm above linear input `0.00262409`, with the published lower linear branch applied directly to negative values; decode uses the derived decode threshold rather than the printed rounded cut; no clipping or sign/magnitude mirroring; specify colorspace independently",
        ),
        (
            "`RED-Log3G10`",
            "RED Log3G10 camera log transfer",
            "RED Log3G10 whitepaper revision C",
            "Uses the published `0.224282 / 155.975327 / 0.01 / 15.1927` piecewise constants; the lower linear branch applies directly below scene-linear `-0.01`, the logarithmic branch includes the boundary, and neither negative values nor scene overshoot are clipped or mirrored; specify colorspace independently",
        ),
        (
            "`REDlogFilm`",
            "RED Cineon-compatible printing-density transfer",
            "RED logarithmic exposure paper; Kodak Cineon specification",
            "Numerically identical to `Cineon`, including its sign-preserving mirror and zero offset, while preserving independent gamma metadata; specify colorspace independently",
        ),
        (
            "`Cineon`",
            "Cineon printing-density log transfer",
            "Kodak Cineon specification",
            "Formula with black CV=95, white CV=685, 0.002 density/code, and film gamma=0.6; apply to nonnegative magnitude and reflect the negative side with preserved sign",
        ),
        (
            "`Gamma-2.2`",
            "Power transfer with exponent 2.2",
            "Conventional value",
            "**Pure power**, reflected with preserved sign; not a piecewise function",
        ),
        (
            "`Gamma-2.4`",
            "Power transfer with exponent 2.4",
            "Conventional value",
            "**Pure power**, reflected with preserved sign; numerically equivalent to the ideal-black `BT.1886` implementation but semantically distinct",
        ),
        (
            "`Gamma-2.6`",
            "Power transfer with exponent 2.6",
            "Conventional value",
            "Decode with `sign(x) * abs(x) ** 2.6` and encode with `sign(x) * abs(x) ** (1 / 2.6)`; no offset, piecewise branch, or clipping",
        ),
    )
    assert _table_rows(
        vocabulary_markdown,
        "colorspace",
        ("Token", "Definition", "Standard or convention", "Notes"),
    ) == (
        (
            "`sRGB`",
            "sRGB primaries and D65 white",
            "IEC 61966-2-1",
            "Primaries and white point are identical to Rec.709",
        ),
        (
            "`Rec.709`",
            "BT.709 primaries and D65 white",
            "ITU-R BT.709",
            "Primaries and white point are identical to sRGB",
        ),
        (
            "`Rec.2020`",
            "BT.2020 wide-gamut primaries and D65 white",
            "ITU-R BT.2020",
            "Specify the HDR transfer separately with gamma",
        ),
        (
            "`ACES2065-1`",
            "ACES AP0 primaries and ACES white",
            "SMPTE ST 2065-1",
            "ACES interchange colorspace",
        ),
        (
            "`ACEScg`",
            "ACES AP1 primaries and ACES white",
            "Academy ACES specification",
            "Scene-linear working colorspace",
        ),
        (
            "`S-Gamut`",
            "Sony S-Gamut primaries",
            "Sony S-Log whitepaper",
            "Numerically identical to `S-Gamut3`; token identity remains distinct; specify a transfer such as `S-Log` separately",
        ),
        (
            "`S-Gamut3`",
            "Sony S-Gamut3 primaries",
            "Sony technical specification",
            "Camera gamut; specify a transfer such as `S-Log3` separately",
        ),
        (
            "`S-Gamut3.Cine`",
            "Sony S-Gamut3.Cine primaries",
            "Sony technical specification",
            "Cinema-oriented camera gamut",
        ),
        (
            "`ARRI-Wide-Gamut-3`",
            "ARRI Wide Gamut 3 primaries and D65 white",
            "ARRI Wide Gamut 3 specification",
            "Scene-referred camera gamut; selected independently from gamma, including `ARRI-LogC3`",
        ),
        (
            "`ARRI-Wide-Gamut-4`",
            "ARRI Wide Gamut 4 primaries and D65 white",
            "ARRI Wide Gamut 4 specification",
            "Scene-referred camera gamut; selected independently from gamma, including `ARRI-LogC4`",
        ),
        (
            "`Blackmagic-Wide-Gamut-Gen-5`",
            "Blackmagic Wide Gamut Generation 5 primaries and D65 white",
            "Blackmagic Design Generation 5 Color Science",
            "Scene-referred camera gamut; selected independently from gamma, including `Blackmagic-Film-Gen-5`; does not assert numerical identity with Gen 4",
        ),
        (
            "`DaVinci-Wide-Gamut`",
            "DaVinci Wide Gamut revision 1.1 primaries and D65 white",
            "Blackmagic Design DaVinci Wide Gamut / Intermediate",
            "Scene-referred working gamut; selected independently from gamma, including `DaVinci-Intermediate`",
        ),
        (
            "`REDWideGamutRGB`",
            "REDWideGamutRGB primaries and D65 white",
            "REDWideGamutRGB / Log3G10 whitepaper revision C",
            "Scene-referred IPP2 gamut; selected independently from gamma, including `RED-Log3G10`",
        ),
        (
            "`DRAGONcolor`",
            "Legacy RED DRAGONcolor-derived primaries and D65 white",
            "ACES 1.0.3 OpenColorIO config",
            "Scene-referred gamut reconstructed from the published RGB-to-ACES2065-1 matrix; selected independently from gamma",
        ),
        (
            "`DRAGONcolor2`",
            "Legacy RED DRAGONcolor2-derived primaries and D65 white",
            "ACES 1.0.3 OpenColorIO config",
            "Scene-referred gamut reconstructed from the published RGB-to-ACES2065-1 matrix; selected independently from gamma",
        ),
        (
            "`REDcolor2`",
            "Legacy REDcolor2-derived primaries and D65 white",
            "ACES 1.0.3 OpenColorIO config",
            "Scene-referred gamut reconstructed from the published RGB-to-ACES2065-1 matrix; selected independently from gamma",
        ),
        (
            "`REDcolor3`",
            "Legacy REDcolor3-derived primaries and D65 white",
            "ACES 1.0.3 OpenColorIO config",
            "Scene-referred gamut reconstructed from the published RGB-to-ACES2065-1 matrix; selected independently from gamma",
        ),
        (
            "`REDcolor4`",
            "Legacy REDcolor4-derived primaries and D65 white",
            "ACES 1.0.3 OpenColorIO config",
            "Scene-referred gamut reconstructed from the published RGB-to-ACES2065-1 matrix; selected independently from gamma",
        ),
    )


def test_vocabulary_defines_chroma_siting_records_and_applicability(vocabulary_markdown: str) -> None:
    """v1-format-boundary acceptance 39: H.273 siting records and format applicability are exact."""
    assert _table_records(
        vocabulary_markdown, "chroma siting", ("Token", "Offset `(x, y)`", "H.273", "Definition")
    ) == (
        {
            "Token": "`left`",
            "Offset `(x, y)`": "`(0, 0.5)`",
            "H.273": "H.273 type 0",
            "Definition": "Horizontally co-sited and vertically interstitial; typical BT.601/BT.709 SDR delivery convention",
        },
        {
            "Token": "`center`",
            "Offset `(x, y)`": "`(0.5, 0.5)`",
            "H.273": "H.273 type 1",
            "Definition": "Geometric center of the 2×2 luma block",
        },
        {
            "Token": "`topleft`",
            "Offset `(x, y)`": "`(0, 0)`",
            "H.273": "H.273 type 2",
            "Definition": "Co-sited on both axes; standard BT.2020/BT.2100 position",
        },
    )
    section = _section(vocabulary_markdown, "chroma siting")
    for required in (
        "top-left luma-sample center is `(0, 0)`",
        "default `left`",
        "State `topleft` explicitly",
        "4:2:2",
        "neither siting nor interpolation arguments",
    ):
        assert required in section


def test_vocabulary_image_format_and_compression_tokens_match_the_bytes_boundary(
    vocabulary_markdown: str,
) -> None:
    """v1-bytes-boundary acceptance 12; v1-exr-write-compression acceptance 11."""
    from pixtreme._io.common import _ENCODE_FORMAT_TOKENS, _EXR_COMPRESSION_TOKENS, _TIFF_COMPRESSION_TOKENS

    assert (
        _table_tokens(vocabulary_markdown, "image format")
        == _ENCODE_FORMAT_TOKENS
        == (
            "jpeg",
            "png",
            "tiff",
            "jpeg2000",
            "webp",
            "bmp",
            "pnm",
        )
    )
    assert _table_tokens(vocabulary_markdown, "TIFF compression") == _TIFF_COMPRESSION_TOKENS == ("none", "lzw")
    assert (
        _table_tokens(vocabulary_markdown, "EXR compression")
        == _EXR_COMPRESSION_TOKENS
        == (
            "none",
            "rle",
            "zip",
            "zips",
            "piz",
            "pxr24",
            "b44",
            "b44a",
            "dwaa",
            "dwab",
        )
    )


def test_vocabulary_write_dtype_matrix_matches_the_boundary_contract(vocabulary_markdown: str) -> None:
    """v1-write-dtype-convert acceptance 8 / v1-hdr acceptance 9: docs match write containers."""
    from pixtreme._io.dtype import _WRITE_DEFAULT_DTYPES, _WRITE_NATIVE_DTYPES

    assert _WRITE_NATIVE_DTYPES == {
        "PNG": frozenset(("uint8", "uint16")),
        "JPEG": frozenset(("uint8",)),
        "TIFF": frozenset(("uint8", "uint16")),
        "EXR": frozenset(("float16", "float32", "uint32")),
        "JPEG2000": frozenset(("uint8", "uint16")),
        "WEBP": frozenset(("uint8",)),
        "BMP": frozenset(("uint8",)),
        "PNM": frozenset(("uint8", "uint16")),
        "HDR": frozenset(("float32",)),
        "DPX": frozenset(("float32",)),
    }
    assert _WRITE_DEFAULT_DTYPES == {
        "PNG": "uint8",
        "JPEG": "uint8",
        "TIFF": "uint8",
        "EXR": "float16",
        "JPEG2000": "uint8",
        "WEBP": "uint8",
        "BMP": "uint8",
        "PNM": "uint8",
        "HDR": "float32",
        "DPX": "float32",
    }
    assert _table_rows(
        vocabulary_markdown,
        "image write dtype",
        ("Format", "Native dtype", "Default container for nonnative input"),
    ) == (
        ("PNG / TIFF / JPEG 2000 / PNM", "`uint8` / `uint16`", "`uint8`"),
        ("JPEG / WebP / BMP", "`uint8`", "`uint8`"),
        ("EXR", "`float16` / `float32` / `uint32`", "Normally `float16`; `uint32` for a uint32 Frame"),
        ("TGA", "`uint8`", "`uint8`"),
        ("HDR", "`float32`", "`float32`"),
        ("DPX", "`float32`", "`float32`"),
    )


def test_vocabulary_stack_directions_equal_the_validator_and_document_join_rules(
    vocabulary_markdown: str,
) -> None:
    """v1-stack acceptance 7: direction tokens and join rules equal the implementation contract."""
    from pixtreme._transform.stack import _STACK_DIRECTION_TOKENS

    section = _section(vocabulary_markdown, "stack direction")
    assert "default `vertical`" in section
    assert (
        _table_tokens(vocabulary_markdown, "stack direction")
        == _STACK_DIRECTION_TOKENS
        == (
            "vertical",
            "horizontal",
        )
    )
    assert _table_records(vocabulary_markdown, "stack direction", ("Token", "Concatenation rule")) == (
        {
            "Token": "`vertical`",
            "Concatenation rule": "Arrange top to bottom; output height is the sum of input heights and width is common",
        },
        {
            "Token": "`horizontal`",
            "Concatenation rule": "Arrange left to right; output width is the sum of input widths and height is common",
        },
    )


def test_vocabulary_documents_channel_shuffle_routing_and_provenance(vocabulary_markdown: str) -> None:
    """v1-channel-shuffle acceptance 21: routing, reserved words, and matrix provenance are fixed."""
    assert _table_rows(vocabulary_markdown, "channel routing", ("Output structure", "`Frame.matrix`")) == (
        ("R / G / B mixed with Y / Cb / Cr", "`None`"),
        ("RGB-only, or no Y / Cb / Cr channels", "`None`"),
        (
            "All Y / Cb / Cr Frame-source claims are the same non-`None` token",
            "That token, preserving `native` literally",
        ),
        ("At least one claim is `None`, or all Y / Cb / Cr outputs are fills", "`None`"),
        ("Non-`None` claims contain multiple tokens", "Three-part error; no implicit rematrixing"),
    )
    section = _section(vocabulary_markdown, "channel routing")
    for required in ("**{...}", "reserved option", "first Frame source", "`adapt=True`", "never clips values"):
        assert required in section


def test_vocabulary_documents_from_format_conventions(vocabulary_markdown: str) -> None:
    """v1-format-boundary acceptance 39; v1-from-format-metadata acceptance 7: defaults and layouts are exact."""
    assert _table_rows(
        vocabulary_markdown,
        "from_<format> conventions",
        ("Item", "Specification default", "Notes"),
    ) == (
        ("colorspace", "`Rec.709`", "Placeholder; an explicit per-call `colorspace=` token takes precedence"),
        ("gamma", "`Rec.709`", "Placeholder; an explicit per-call `gamma=` token takes precedence"),
        (
            "matrix",
            "`None`",
            "Unknown provenance; an explicit per-call `matrix=` token is normalized and stamped as its canonical spelling",
        ),
        ("channels", '`("Y", "Cb", "Cr")`', "Fixed channel order after format resolution"),
        (
            "range",
            "`legal`",
            'Default assumption for video-family YCbCr input; override per call with `range="full"`',
        ),
        (
            "interpolation",
            "`bilinear`",
            "Default for the six subsampled formats; accepts the first eight interpolation tokens",
        ),
        (
            "siting",
            "`left`",
            "Present only on the three 4:2:0 formats; accepts the three chroma-siting tokens",
        ),
    )
    assert _table_rows(
        vocabulary_markdown,
        "from_<format> conventions",
        ("Format", "bit_depth", "Container dtype", "Plane order"),
    ) == (
        (
            "`yuv420p`",
            "8 (default) / 10",
            "8 = uint8, 10 = uint16",
            "Y, then Cb, then Cr; each chroma plane is H/2 × W/2",
        ),
        (
            "`yuv422p`",
            "8 (default) / 10 / 12",
            "8 = uint8, 10 / 12 = uint16",
            "Y, then Cb, then Cr; each chroma plane is H × W/2",
        ),
        ("`yuv444p`", "10 (default) / 12", "uint16", "Y, then Cb, then Cr; each plane is H × W"),
        (
            "`yuva444p`",
            "12 (default)",
            "uint16",
            "Y, then Cb, then Cr, then A; each plane is H × W, and A is full-scale regardless of range",
        ),
    )
    assert _table_rows(
        vocabulary_markdown,
        "from_<format> conventions",
        ("Format", "Container dtype", "C-contiguous 1D layout"),
    ) == (
        (
            "`uyvy422`",
            "uint8",
            "U0 Y0 V0 Y1; input also accepts NDI shape `(H, W, 2)`, which can reshape to 1D as a zero-copy view",
        ),
        (
            "`v210`",
            "uint32",
            "Six pixels in four words, with three 10-bit samples from the low bits of each word; rows align to 128 bytes, or 48 pixels, with zero padding",
        ),
        ("`NV12`", "uint8", "Y plane followed by an interleaved Cb Cr plane"),
        ("`P010`", "uint16", "Same arrangement as NV12; 10-bit codes are MSB-aligned and the lower six bits are zero"),
    )


def test_vocabulary_documents_to_format_conventions(vocabulary_markdown: str) -> None:
    """v1-format-boundary acceptance 39: output defaults and layouts are exact."""
    assert _table_rows(
        vocabulary_markdown,
        "to_<format> conventions",
        ("Item", "Specification default", "Notes"),
    ) == (
        (
            "range",
            "`legal`",
            "Also accepts `full`; legal placement preserves headroom codes without clipping to the legal interval",
        ),
        (
            "interpolation",
            "`area`",
            "Default for the six subsampled formats; accepts nearest, bilinear, bicubic, and area",
        ),
        (
            "siting",
            "`left`",
            "Present only on the three 4:2:0 formats; accepts the three chroma-siting tokens",
        ),
        ("rounding", "Half away from zero", "Nearest rounding from fp32 to code"),
        (
            "clipping",
            "Container range only",
            "Do not clip to the legal interval; clip only to physical `[0, 2^n - 1]`",
        ),
    )
    assert _table_rows(
        vocabulary_markdown,
        "to_<format> conventions",
        ("Format", "bit_depth", "Container dtype", "C-contiguous 1D layout"),
    ) == (
        (
            "`yuv420p`",
            "8 (default) / 10",
            "8 = uint8, 10 = uint16",
            "Y, then Cb, then Cr; each chroma plane is H/2 × W/2",
        ),
        (
            "`yuv422p`",
            "8 (default) / 10 / 12",
            "8 = uint8, 10 / 12 = uint16",
            "Y, then Cb, then Cr; each chroma plane is H × W/2",
        ),
        ("`yuv444p`", "10 (default) / 12", "uint16", "Y, then Cb, then Cr; each plane is H × W"),
        (
            "`yuva444p`",
            "12 (default)",
            "uint16",
            "Y, then Cb, then Cr, then A; A is full-scale regardless of range",
        ),
        ("`uyvy422`", "Fixed 8", "uint8", "U0 Y0 V0 Y1; reshape to `(H, W, 2)` is a zero-copy view"),
        (
            "`v210`",
            "Fixed 10",
            "uint32",
            "Six pixels in four words; the function zero-fills 128-byte row padding",
        ),
        ("`NV12`", "Fixed 8", "uint8", "Y plane followed by an interleaved Cb Cr plane"),
        (
            "`P010`",
            "Fixed 10",
            "uint16",
            "Same arrangement as NV12; MSB-aligned with the lower six bits zero",
        ),
    )


def test_vocabulary_documents_range_dtype_and_quantization_semantics(vocabulary_markdown: str) -> None:
    """v1-quantize-values acceptance 15: range tokens and bit-depth lanes are exact records."""
    assert _table_rows(
        vocabulary_markdown,
        "range",
        ("Token", "Definition", "Standard or convention", "Notes"),
    ) == (
        (
            "`legal`",
            "H.273 limited-range code positions, `video_full_range_flag = 0`",
            "ITU-T H.273",
            "Y and limited-range RGB use the luma interval; Cb and Cr use the chroma interval",
        ),
        (
            "`full`",
            "Full-range values spanning the entire unsigned container, `video_full_range_flag = 1`",
            "ITU-T H.273",
            "Normal state for float working; not stored as a Frame state token",
        ),
    )
    assert _table_rows(
        vocabulary_markdown,
        "pixel value quantization",
        ("Path", "API", "Meaning of `bit_depth`", "Value and container handling"),
    ) == (
        (
            "Range pair",
            "`px.values.legal_to_full` / `px.values.full_to_legal`",
            "Effective code bits for H.273 legal code positions",
            "Linear float32 mapping without clipping",
        ),
        (
            "Quantization pair",
            "`px.values.quantize` / `px.values.dequantize`",
            "Effective code bits for the unsigned full-scale grid",
            "float32 Frame to or from uint Frame",
        ),
        (
            "Named format",
            "`px.io.from_<format>` / `px.io.to_<format>`",
            "Effective code bits carried by the format",
            "Packing, subsampling, and container resolved by the format contract",
        ),
        (
            "General array boundary",
            "`px.io.from_array` / `px.io.to_array`",
            "Effective code bits for an unsigned full-scale grid in a raw array",
            "Composes orthogonally with layout, channel selection, and `out=`",
        ),
    )


def test_legal_to_full_docstring_contains_the_reverse_composition_recipe() -> None:
    """v1-color-semantics acceptance 34: API docs use the directional color pair in the repair recipe."""
    import inspect

    docstring = inspect.getdoc(__import__("pixtreme").values.legal_to_full)
    assert docstring is not None
    first_transform = docstring.index("px.color.rgb_to_ycbcr")
    range_conversion = docstring.index("px.values.legal_to_full", first_transform)
    second_transform = docstring.index("px.color.ycbcr_to_rgb", range_conversion)
    assert first_transform < range_conversion < second_transform
    for required in ('matrix="BT.709"', "bit_depth=8"):
        assert required in docstring


def test_vocabulary_bit_depths_equal_the_shared_validator_set(vocabulary_markdown: str) -> None:
    """v1-quantize-values acceptance 15: the documented five-value axis equals the implementation."""
    from pixtreme._values.quantize import _BIT_DEPTHS

    section = _section(vocabulary_markdown, "pixel value quantization")
    bit_depth_line = next(line for line in section.splitlines() if "`bit_depth` accepts" in line)
    documented = tuple(int(value) for value in re.findall(r"`(\d+)`", bit_depth_line))
    assert documented == _BIT_DEPTHS


def test_vocabulary_documents_matrix_semantics_and_colorspace_derivation(vocabulary_markdown: str) -> None:
    """v1-color-semantics acceptance 35; v1-arri-tokens acceptance 23-24 and 29;
    v1-blackmagic-tokens acceptance 44-45 and 50; v1-red-tokens acceptance 64 and 72.

    Matrix tokens and own-row values are complete records.
    """
    assert _table_rows(
        vocabulary_markdown,
        "matrix",
        ("Token", "Formal name", "Definition", "Standard or convention", "Notes"),
    ) == (
        (
            "`BT.601`",
            "BT.601",
            "Kr = 0.299, Kb = 0.114",
            "ITU-T H.273 / ITU-R BT.601",
            "SD-family non-constant-luminance coefficients",
        ),
        (
            "`BT.709`",
            "BT.709",
            "Kr = 0.2126, Kb = 0.0722",
            "ITU-T H.273 / ITU-R BT.709",
            "Specification-fixed result for sRGB and Rec.709",
        ),
        (
            "`BT.2020`",
            "BT.2020",
            "Kr = 0.2627, Kb = 0.0593",
            "ITU-T H.273 / ITU-R BT.2020",
            "Specification-fixed result for Rec.2020",
        ),
        (
            "`native`",
            "Colorspace own-row",
            "Y row of the normalized RGB-to-XYZ matrix constructed from the Frame's current published colorspace primaries and white point",
            "Published standard for each colorspace",
            "Relative token, not native to a file or device; gamma does not change the coefficients",
        ),
    )
    assert _table_rows(
        vocabulary_markdown,
        "matrix own-row",
        ("Colorspace", "own-row `(Kr, Kg, Kb)`", "Relationship to a known H.273 basis"),
    ) == (
        ("`sRGB`", "`(0.2126390059, 0.7151686788, 0.0721923154)`", "Numerically identical to `BT.709`"),
        ("`Rec.709`", "`(0.2126390059, 0.7151686788, 0.0721923154)`", "Numerically identical to `BT.709`"),
        ("`Rec.2020`", "`(0.2627002120, 0.6779980715, 0.0593017165)`", "Numerically identical to `BT.2020`"),
        ("`ACES2065-1`", "`(0.3439664498, 0.7281660966, -0.0721325464)`", "AP0 own-row"),
        ("`ACEScg`", "`(0.2722287168, 0.6740817658, 0.0536895174)`", "AP1 own-row"),
        ("`S-Gamut`", "`(0.2709796708, 0.7866064112, -0.0575860820)`", "Numerically identical to `S-Gamut3`"),
        ("`S-Gamut3`", "`(0.2709796708, 0.7866064112, -0.0575860820)`", "Sony S-Gamut3 own-row"),
        (
            "`S-Gamut3.Cine`",
            "`(0.2150758201, 0.8850685017, -0.1001443219)`",
            "Sony S-Gamut3.Cine own-row",
        ),
        (
            "`ARRI-Wide-Gamut-3`",
            "`(0.2919537790, 0.8238410415, -0.1157948205)`",
            "ARRI Wide Gamut 3 own-row",
        ),
        (
            "`ARRI-Wide-Gamut-4`",
            "`(0.2545241764, 0.7814777327, -0.0360019091)`",
            "ARRI Wide Gamut 4 own-row",
        ),
        (
            "`Blackmagic-Wide-Gamut-Gen-5`",
            "`(0.2679929401, 0.8327484091, -0.1007413492)`",
            "Blackmagic Wide Gamut Generation 5 own-row",
        ),
        (
            "`DaVinci-Wide-Gamut`",
            "`(0.2741185109, 0.8736318959, -0.1477504068)`",
            "DaVinci Wide Gamut own-row",
        ),
        (
            "`REDWideGamutRGB`",
            "`(0.2866940995, 0.8429791340, -0.1296732335)`",
            "REDWideGamutRGB own-row",
        ),
        ("`DRAGONcolor`", "`(0.2169921791, 0.8380223380, -0.0550145171)`", "DRAGONcolor own-row"),
        ("`DRAGONcolor2`", "`(0.1909714594, 0.7375309361, 0.0714976045)`", "DRAGONcolor2 own-row"),
        ("`REDcolor2`", "`(0.1657102643, 0.8636624823, -0.0293727466)`", "REDcolor2 own-row"),
        ("`REDcolor3`", "`(0.2255112277, 0.7798000805, -0.0053113082)`", "REDcolor3 own-row"),
        ("`REDcolor4`", "`(0.2088065893, 0.7220385248, 0.0691548859)`", "REDcolor4 own-row"),
    )


def test_vocabulary_documents_view_versions_combinations_and_scope_boundary(vocabulary_markdown: str) -> None:
    """v1-view-transform-lut-removal acceptance 4: three tonemap tokens and six exits are explicit."""
    from pixtreme._color.transform import _SUPPORTED_COMBINATIONS

    assert _table_tokens(vocabulary_markdown, "tonemap") == (
        "ACES-1.3",
        "ACES-2.0",
        "BT.2408",
    )
    rows = _table_rows(
        vocabulary_markdown,
        "tonemap combinations",
        ("Tonemap", "Output colorspace", "Output gamma", "Destination"),
    )
    assert rows == (
        ("`ACES-1.3`", "`Rec.709`", "`BT.1886`", "Rec.1886 Rec.709 display"),
        ("`ACES-1.3`", "`sRGB`", "`sRGB`", "sRGB display"),
        ("`ACES-2.0`", "`Rec.709`", "`BT.1886`", "Rec.1886 Rec.709 display"),
        ("`ACES-2.0`", "`sRGB`", "`sRGB`", "sRGB display"),
        ("`BT.2408`", "`Rec.2020`", "`HLG`", "BT.2100 HLG; SDR reference white = 75% signal"),
        ("`BT.2408`", "`Rec.2020`", "`PQ`", "BT.2100 PQ; SDR reference white = 203 cd/m²"),
    )
    documented = tuple(
        (tonemap.strip("`"), colorspace.strip("`"), gamma.strip("`"))
        for tonemap, colorspace, gamma, _destination in rows
    )
    assert set(documented) == set(_SUPPORTED_COMBINATIONS)


def test_vocabulary_documents_image_read_conventions_and_metadata_priority(vocabulary_markdown: str) -> None:
    """v1-io acceptance 5, 6, and 7: image-read defaults and metadata priority are exact."""
    assert _table_rows(
        vocabulary_markdown,
        "image read conventions",
        ("Format", "Default colorspace", "Default gamma", "`channels=None`"),
    ) == (
        ("PNG / JPEG / TIFF", "`sRGB`", "`sRGB`", 'RGB or RGBA; grayscale is one-channel `("Y",)`'),
        ("JPEG 2000", "`sRGB`", "`sRGB`", "Y, RGB, or RGBA"),
        ("WebP", "`sRGB`", "`sRGB`", "RGB"),
        ("BMP / PNM", "`sRGB`", "`sRGB`", "Y or RGB"),
        ("TGA", "`sRGB`", "`sRGB`", "RGB or RGBA"),
        ("HDR", "`Rec.709`", "`linear`", "RGB"),
        (
            "DPX",
            "`Rec.709`",
            "Header transfer; unknown maps to `Cineon` at 10 bit, `Rec.709` at 8 bit, and `linear` at 12 or 16 bit",
            "RGB or RGBA",
        ),
        ("EXR", "`ACES2065-1`", "`linear`", "R, G, B, and A when present"),
    )
    priority = re.search(
        r"Metadata priority is \*\*([^*]+)\*\*", _section(vocabulary_markdown, "image read conventions")
    )
    assert priority is not None
    assert tuple(part.strip() for part in priority.group(1).split(">")) == (
        "explicit per-call value",
        "explicit file value",
        "specification default",
    )


def test_vocabulary_documents_cast_quantization_and_encode_kwargs(vocabulary_markdown: str) -> None:
    """v1-recode-dtype acceptance 8 and v1-quantize-values acceptance 15."""
    assert _table_rows(
        vocabulary_markdown,
        "dtype operation comparison",
        ("API", "Preserved property", "uint↔float behavior", "Primary use"),
    ) == (
        (
            "`px.values.cast_dtype`",
            "Numeric value",
            "Faithful delegation to CuPy `astype`; no scaling, clipping, or explicit rounding",
            "Change the container of depth, label, or other raw values read unchanged",
        ),
        (
            "`px.values.recode_dtype`",
            "Meaning",
            "Normalize uint by container maximum; clip float to `[0, 1]`, scale to full range, and round half away from zero for float to uint; literal cast between floats",
            "Convert between ordinary uint images and normalized float Frames",
        ),
        (
            "`px.values.quantize`",
            "Pixel-value scale",
            "Clip and scale float32 to the uint full-scale grid at the declared bit depth, then round half away from zero",
            "Produce a code-value Frame from normalized values",
        ),
        (
            "`px.values.dequantize`",
            "Pixel-value scale",
            "Normalize uint codes at the declared bit depth by maximum code without clipping",
            "Return a code-value Frame to float32 working values",
        ),
    )
    assert _table_rows(
        vocabulary_markdown,
        "image encode kwargs",
        ("Kwarg", "API and target format", "Value domain", "Meaning"),
    ) == (
        (
            "`quality`",
            "Both APIs; JPEG and WebP",
            "Integer `1` through `100`",
            "Lossy quality; specifying it for JPEG 2000, PNG, TIFF, BMP, PNM, or EXR raises `ValueError`",
        ),
        (
            "`compression`",
            "Both APIs; TIFF",
            "Token `none` or `lzw`",
            "TIFF uncompressed or lossless LZW compression",
        ),
        (
            "`compression`",
            "`px.io.write_image`; EXR",
            "EXR compression token",
            "Default `zip`; distinct from TIFF tokens",
        ),
        (
            "`compression_level`",
            "Both APIs; PNG",
            "Integer `0` through `9`",
            "PNG zlib compression level; specifying it for another format raises `ValueError`",
        ),
        (
            "`lossless`",
            "Both APIs; JPEG 2000 and WebP",
            "Exact `bool` or `None`",
            "`True` is lossless, `False` lossy, and `None` the codec default; WebP `quality` conflicts with `True`",
        ),
        (
            "`dwa_level`",
            "`px.io.write_image`; EXR DWAA and DWAB",
            "Positive finite exact `float` or `None`, including as a header float",
            "`None` means `45.0`; specifying it for non-DWA compression raises `ValueError`",
        ),
        (
            "`bit_depth`",
            "`px.io.write_image`; DPX",
            "Integer `8`, `10`, `12`, `16`, or `None`",
            "`None` means 10 bit; specifying it for non-DPX output raises `ValueError`",
        ),
        (
            "`dtype`",
            "`px.io.write_image`; EXR",
            "`float16`, `float32`, `uint32`, or `None`",
            "An explicit value overrides the Frame-dependent default; specifying it for non-EXR output raises `ValueError`",
        ),
    )


def test_vocabulary_documents_array_layout_affine_copy_and_out_contracts(vocabulary_markdown: str) -> None:
    """v1-frame-core acceptance 16 and v1-boundary-api acceptance 21."""
    assert _table_rows(
        vocabulary_markdown,
        "layout",
        ("Token", "Rank / shape", "`px.io.from_array`", "`px.io.to_array`"),
    ) == (
        ("`HWC`", "`(H, W, C)`", "Interpreted directly as Frame HWC", "HWC view or repacked result"),
        (
            "`NHWC`",
            "`(1, H, W, C)`",
            "HWC view after removing the leading size-1 axis; N > 1 raises `ValueError`",
            "Zero-copy view with a leading size-1 axis",
        ),
        (
            "`CHW`",
            "`(C, H, W)`",
            "Transposed into HWC",
            "Repacked as a channel-first C-contiguous array",
        ),
        (
            "`NCHW`",
            "`(1, C, H, W)`",
            "Validates N == 1 and transposes into HWC",
            "Repacked as a channel-first array with a leading size-1 axis",
        ),
    )
    assert _table_rows(vocabulary_markdown, "device array affine / copy", ("Value", "Meaning")) == (
        (
            "`copy=None`",
            "Use a zero-copy view when possible; otherwise make exactly one copy when required by layout transposition, channel selection, dtype, affine processing, or another requested operation",
        ),
        ("`copy=False`", "Strict zero-copy guarantee; a request that requires writing raises a three-part error"),
        ("`copy=True`", "Always return a private copy that shares no storage with the caller"),
    )
    assert _table_rows(vocabulary_markdown, "Frame boundary contract", ("Contract", "Requirement")) == (
        (
            "GPU layout",
            "`Frame.data` is HWC and C-contiguous; `px.io.from_array` selects a view or GPU copy under the three-state copy contract",
        ),
        ("pointer lifetime", "Retain the allocation-owning Frame while using a raw pointer to `Frame.data`"),
        ("stream", "Pass a DLPack consumer stream through unchanged to `Frame.data.__dlpack__`"),
        (
            "device export",
            "`px.io.to_array(frame)` returns `cupy.ndarray`; both the Frame and returned array are DLPack producers",
        ),
        (
            "direct destination",
            "`out=` accepts only a C-contiguous `cupy.ndarray` with matching shape and dtype, and returns that same object",
        ),
        (
            "host transfer",
            "The canonical `to_array(...).get()` path is `px.io.to_array(frame, ...).get()`; alternatively call `cp.asnumpy(px.io.to_array(frame, ...))` explicitly",
        ),
    )


def test_vocabulary_anchor_definitions_match_the_block_layout_contract(vocabulary_markdown: str) -> None:
    """v1-draw-text-unification acceptance 13: all anchor meanings use the integrated block box."""
    assert _table_rows(vocabulary_markdown, "anchor", ("Token", "Definition")) == (
        ("`top-left`", "First-line ascender and left edge of the block box"),
        ("`top-center`", "First-line ascender and horizontal midpoint of the block box"),
        ("`top-right`", "First-line ascender and right edge of the block box"),
        ("`center-left`", "Midpoint between top and bottom, and left edge of the block box"),
        ("`center-center`", "Midpoint between top and bottom, and horizontal midpoint of the block box"),
        ("`center-right`", "Midpoint between top and bottom, and right edge of the block box"),
        ("`baseline-left`", "First-line baseline and left edge of the block box"),
        ("`baseline-center`", "First-line baseline and horizontal midpoint of the block box"),
        ("`baseline-right`", "First-line baseline and right edge of the block box"),
        ("`bottom-left`", "Final-line descender and left edge of the block box"),
        ("`bottom-center`", "Final-line descender and horizontal midpoint of the block box"),
        ("`bottom-right`", "Final-line descender and right edge of the block box"),
    )


def test_vocabulary_composite_blend_alpha_and_interpolation_match_implementation(
    vocabulary_markdown: str,
) -> None:
    """v1-composite acceptance 18: docs and every validator share exact composite token subsets."""
    from pixtreme._composite.merge import _ALPHA_TOKENS, _COMPOSITE_INTERPOLATION_TOKENS
    from pixtreme._core.blend import _BLEND_TOKENS, _DRAW_BLEND_TOKENS
    from pixtreme._draw.shapes import _BLEND_TOKENS as _SHAPE_BLEND_TOKENS
    from pixtreme._draw.text import _BLEND_TOKENS as _TEXT_BLEND_TOKENS

    assert (
        _table_tokens(vocabulary_markdown, "blend")
        == _BLEND_TOKENS
        == (
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
        )
    )
    assert (
        _DRAW_BLEND_TOKENS
        == _SHAPE_BLEND_TOKENS
        == _TEXT_BLEND_TOKENS
        == (
            "normal",
            "add",
            "multiply",
            "screen",
        )
    )
    assert _table_tokens(vocabulary_markdown, "alpha") == _ALPHA_TOKENS == ("premultiplied", "straight")
    assert _COMPOSITE_INTERPOLATION_TOKENS == (
        "nearest",
        "bilinear",
        "bicubic",
        "b-spline",
        "mitchell",
        "lanczos2",
        "lanczos3",
        "lanczos4",
    )
    assert _table_rows(vocabulary_markdown, "blend", ("Token", "`B(Cb, Cs)`")) == (
        ("`normal`", "`Cs`"),
        ("`lighten`", "`max(Cb, Cs)`"),
        ("`add`", "`Cb + Cs`"),
        ("`screen`", "`1 - (1 - Cb) × (1 - Cs)`"),
        ("`darken`", "`min(Cb, Cs)`"),
        ("`multiply`", "`Cb × Cs`"),
        ("`difference`", "`abs(Cb - Cs)`"),
        ("`overlay`", "`2 × Cb × Cs` when `Cb <= 0.5`; otherwise `1 - 2 × (1 - Cb) × (1 - Cs)`"),
        ("`hardlight`", "`2 × Cb × Cs` when `Cs <= 0.5`; otherwise `1 - 2 × (1 - Cb) × (1 - Cs)`"),
        (
            "`softlight`",
            "`Cb - (1 - 2 × Cs) × Cb × (1 - Cb)` when `Cs <= 0.5`; otherwise `Cb + (2 × Cs - 1) × (D(Cb) - Cb)`",
        ),
    )
    assert _table_rows(vocabulary_markdown, "alpha", ("Token", "Definition")) == (
        (
            "`premultiplied`",
            "Color channels have already been multiplied by the same pixel's `A`; unassociated color at alpha 0 is defined as 0",
        ),
        (
            "`straight`",
            "Color channels have not been multiplied by `A`; foreground color is associated with `A` before interpolation",
        ),
    )
