"""Documentation contract tests for the public token vocabulary."""

from __future__ import annotations

import re


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
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        assert len(cells) == len(header_cells)
        row = dict(zip(header_cells, cells, strict=True))
        records.append({header: row[header] for header in headers})
    return tuple(records)


def _table_tokens(markdown: str, heading: str) -> tuple[str, ...]:
    section = _section(markdown, heading)
    return tuple(
        cells[1].strip().removeprefix("`").removesuffix("`")
        for line in section.splitlines()
        if line.startswith("| `")
        for cells in (line.split("|"),)
    )


def test_documented_tokens_equal_the_validator_token_sets(vocabulary_markdown: str) -> None:
    """v1-format-boundary acceptance 39; v1-frame-core acceptance 16; v1-recode-dtype acceptance 8.

    Boundary/resizing/blurring docs, including v1-blur-vector acceptance 14, match code token sets.
    """
    from pixtreme._color.lut import _LUT_INTERPOLATION_TOKENS
    from pixtreme._core.frame import _CHANNEL_LABELS, _COLORSPACE_TOKENS, _GAMMA_TOKENS, _LAYOUT_TOKENS, _MATRIX_TOKENS
    from pixtreme._core.value_domain import _RANGE_TOKENS
    from pixtreme._filter.common import _BORDER_TOKENS
    from pixtreme._io.wire.sampling import _INTERPOLATION_TOKENS, _SITING_TOKENS, _TO_INTERPOLATION_TOKENS
    from pixtreme._transform.resize import _INTERPOLATION_TOKENS as _RESIZE_INTERPOLATION_TOKENS
    from pixtreme._transform.warp_affine import _BORDER_TOKENS as _WARP_BORDER_TOKENS
    from pixtreme._transform.warp_affine import _INTERPOLATION_TOKENS as _WARP_INTERPOLATION_TOKENS
    from pixtreme._values.cast import _DTYPE_TOKENS

    markdown = vocabulary_markdown
    expected_channels = ("R", "G", "B", "H", "S", "V", "A", "Y", "Cb", "Cr", "Z")
    expected_gamma = (
        "linear",
        "srgb",
        "rec709",
        "bt1886",
        "pq",
        "hlg",
        "s-log3",
        "logc4",
        "cineon",
        "2.2",
        "2.4",
        "2.6",
    )
    expected_colorspaces = ("sRGB", "Rec.709", "Rec.2020", "ACES2065-1", "ACEScg", "S-Gamut3", "S-Gamut3.Cine")
    expected_matrices = ("bt601", "bt709", "bt2020", "native")
    expected_ranges = ("legal", "full")
    expected_dtypes = ("float32", "float16", "uint8", "uint16", "uint32")
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
    )
    expected_layouts = ("HWC", "NHWC", "CHW", "NCHW")
    expected_borders = ("mirror", "replicate", "wrap", "constant")

    assert _table_tokens(markdown, "channels") == expected_channels == _CHANNEL_LABELS
    assert _table_tokens(markdown, "gamma") == expected_gamma == _GAMMA_TOKENS
    assert _table_tokens(markdown, "colorspace") == expected_colorspaces == _COLORSPACE_TOKENS
    assert _table_tokens(markdown, "matrix") == expected_matrices == _MATRIX_TOKENS
    assert _table_tokens(markdown, "range") == expected_ranges == _RANGE_TOKENS
    assert _table_tokens(markdown, "dtype") == expected_dtypes == _DTYPE_TOKENS
    assert _table_tokens(markdown, "interpolation") == expected_interpolations
    assert _RESIZE_INTERPOLATION_TOKENS == expected_interpolations[:9]
    assert _WARP_INTERPOLATION_TOKENS == expected_interpolations[:9]
    assert _INTERPOLATION_TOKENS == expected_interpolations[:8]
    assert _TO_INTERPOLATION_TOKENS == ("nearest", "bilinear", "bicubic", "area")
    assert _LUT_INTERPOLATION_TOKENS == expected_interpolations[9:]
    assert _table_tokens(markdown, "chroma siting") == ("left", "center", "topleft") == _SITING_TOKENS
    assert _table_tokens(markdown, "layout") == expected_layouts == _LAYOUT_TOKENS
    assert _table_tokens(markdown, "border") == expected_borders == _BORDER_TOKENS
    assert _WARP_BORDER_TOKENS == expected_borders


def test_vocabulary_defines_every_frame_channel_gamma_and_colorspace_record(vocabulary_markdown: str) -> None:
    """v1-frame-core acceptance 16: every foundational token record is exact, not just its spelling."""
    assert _table_records(
        vocabulary_markdown,
        "channels",
        ("Token", "定義", "準拠規格・慣習", "注記"),
    ) == (
        {
            "Token": "`R`",
            "定義": "RGB 表現の red 成分",
            "準拠規格・慣習": "colorspace token が指す RGB 規格",
            "注記": "色度と transfer は Frame の colorspace / gamma が決める",
        },
        {
            "Token": "`G`",
            "定義": "RGB 表現の green 成分",
            "準拠規格・慣習": "colorspace token が指す RGB 規格",
            "注記": "色度と transfer は Frame の colorspace / gamma が決める",
        },
        {
            "Token": "`B`",
            "定義": "RGB 表現の blue 成分",
            "準拠規格・慣習": "colorspace token が指す RGB 規格",
            "注記": "色度と transfer は Frame の colorspace / gamma が決める",
        },
        {
            "Token": "`H`",
            "定義": "hue turn",
            "準拠規格・慣習": "HSV cylindrical coordinates",
            "注記": "周期は 1。正規出力は `[0, 1)`、逆変換は全実数を modulo 1 で受ける",
        },
        {
            "Token": "`S`",
            "定義": "HSV saturation",
            "準拠規格・慣習": "HSV cylindrical coordinates",
            "注記": "非負 RGB 由来では `[0, 1]`。任意入力では範囲を強制しない",
        },
        {
            "Token": "`V`",
            "定義": "HSV value",
            "準拠規格・慣習": "HSV cylindrical coordinates",
            "注記": "RGB の maximum。1 を超えうる上限なしの scene scale",
        },
        {
            "Token": "`A`",
            "定義": "alpha / opacity 成分",
            "準拠規格・慣習": "OpenEXR・一般画像 API の慣習",
            "注記": "premultiplied 状態は Frame metadata に置かない",
        },
        {
            "Token": "`Y`",
            "定義": "luma または非線形輝度成分",
            "準拠規格・慣習": "ITU-T H.273 / grayscale 慣習",
            "注記": "YCbCr 内では luma、1ch Frame では無彩色強度として読む",
        },
        {
            "Token": "`Cb`",
            "定義": "blue-difference chroma 成分",
            "準拠規格・慣習": "ITU-T H.273",
            "注記": "compact 表記では単一ラベルとして最長一致する",
        },
        {
            "Token": "`Cr`",
            "定義": "red-difference chroma 成分",
            "準拠規格・慣習": "ITU-T H.273",
            "注記": "compact 表記では単一ラベルとして最長一致する",
        },
        {
            "Token": "`Z`",
            "定義": "depth 成分",
            "準拠規格・慣習": "OpenEXR channel naming convention",
            "注記": "golden path の color 処理対象外",
        },
    )
    assert _table_records(
        vocabulary_markdown,
        "gamma",
        ("Token", "定義", "準拠規格・慣習", "定義域外延長・注記"),
    ) == (
        {
            "Token": "`linear`",
            "定義": "scene-linear light",
            "準拠規格・慣習": "ACES working convention",
            "定義域外延長・注記": "恒等式を全実数へ自然延長。**scene-referred 固定**であり display-linear の意味には用いない",
        },
        {
            "Token": "`srgb`",
            "定義": "sRGB の区分線形 transfer",
            "準拠規格・慣習": "IEC 61966-2-1",
            "定義域外延長・注記": "線形部と冪部の区分式を負値・1 超へ自然延長。`sRGB` colorspace の標準 transfer",
        },
        {
            "Token": "`rec709`",
            "定義": "Rec.709 camera OETF",
            "準拠規格・慣習": "ITU-R BT.709",
            "定義域外延長・注記": "線形部と冪部の区分式を負値・1 超へ自然延長。`Rec.709` primaries と独立した transfer token",
        },
        {
            "Token": "`bt1886`",
            "定義": "reference display EOTF",
            "準拠規格・慣習": "ITU-R BT.1886",
            "定義域外延長・注記": "nominal exponent 2.4 の符号付き冪 branch として自然延長する display transfer",
        },
        {
            "Token": "`pq`",
            "定義": "perceptual quantizer",
            "準拠規格・慣習": "SMPTE ST 2084 / ITU-R BT.2100",
            "定義域外延長・注記": "非負 magnitude に規格式を適用し、負側は sign-preserving 鏡映 (`f(-x) = -f(x)`)。absolute luminance encoding",
        },
        {
            "Token": "`hlg`",
            "定義": "hybrid log-gamma",
            "準拠規格・慣習": "ITU-R BT.2100",
            "定義域外延長・注記": "低域冪部と高域 log 部の区分式を符号付きで自然延長。scene-referred broadcast HDR transfer",
        },
        {
            "Token": "`s-log3`",
            "定義": "S-Log3 camera log transfer",
            "準拠規格・慣習": "Sony S-Log3 specification",
            "定義域外延長・注記": "非負 magnitude に規格式を適用し、負側は sign-preserving 鏡映 (`f(-x) = -f(x)`)。S-Gamut 系 colorspace と独立に検証する",
        },
        {
            "Token": "`logc4`",
            "定義": "LogC4 camera log transfer",
            "準拠規格・慣習": "ARRI LogC4 specification",
            "定義域外延長・注記": "非負 magnitude に規格式を適用し、負側は sign-preserving 鏡映 (`f(-x) = -f(x)`)。ARRI Wide Gamut 4 は現時点の colorspace 語彙外",
        },
        {
            "Token": "`cineon`",
            "定義": "Cineon printing-density log transfer",
            "準拠規格・慣習": "Kodak Cineon specification",
            "定義域外延長・注記": "black CV=95、white CV=685、0.002 density/code、film gamma=0.6 の式。非負 magnitude に適用し、負側は sign-preserving 鏡映",
        },
        {
            "Token": "`2.2`",
            "定義": "exponent 2.2 の power transfer",
            "準拠規格・慣習": "慣習値",
            "定義域外延長・注記": "**純冪**を sign-preserving 鏡映 (`f(-x) = -f(x)`)。区分関数ではない",
        },
        {
            "Token": "`2.4`",
            "定義": "exponent 2.4 の power transfer",
            "準拠規格・慣習": "慣習値",
            "定義域外延長・注記": "**純冪**を sign-preserving 鏡映 (`f(-x) = -f(x)`)。BT.1886 ではない",
        },
        {
            "Token": "`2.6`",
            "定義": "exponent 2.6 の power transfer",
            "準拠規格・慣習": "慣習値",
            "定義域外延長・注記": "decode は `sign(x) * abs(x) ** 2.6`、encode は `sign(x) * abs(x) ** (1 / 2.6)` の純冪。offset・区分・clip を持たない",
        },
    )
    assert _table_records(
        vocabulary_markdown,
        "colorspace",
        ("Token", "定義", "準拠規格・慣習", "注記"),
    ) == (
        {
            "Token": "`sRGB`",
            "定義": "sRGB primaries・D65 white",
            "準拠規格・慣習": "IEC 61966-2-1",
            "注記": "Rec.709 と primaries / white point が同一",
        },
        {
            "Token": "`Rec.709`",
            "定義": "BT.709 primaries・D65 white",
            "準拠規格・慣習": "ITU-R BT.709",
            "注記": "sRGB と primaries / white point が同一",
        },
        {
            "Token": "`Rec.2020`",
            "定義": "BT.2020 wide-gamut primaries・D65 white",
            "準拠規格・慣習": "ITU-R BT.2020",
            "注記": "HDR transfer は gamma で別指定",
        },
        {
            "Token": "`ACES2065-1`",
            "定義": "ACES AP0 primaries・ACES white",
            "準拠規格・慣習": "SMPTE ST 2065-1",
            "注記": "ACES interchange colorspace",
        },
        {
            "Token": "`ACEScg`",
            "定義": "ACES AP1 primaries・ACES white",
            "準拠規格・慣習": "Academy ACES specification",
            "注記": "scene-linear working colorspace",
        },
        {
            "Token": "`S-Gamut3`",
            "定義": "Sony S-Gamut3 primaries",
            "準拠規格・慣習": "Sony technical specification",
            "注記": "camera gamut。transfer は `s-log3` 等を別指定",
        },
        {
            "Token": "`S-Gamut3.Cine`",
            "定義": "Sony S-Gamut3.Cine primaries",
            "準拠規格・慣習": "Sony technical specification",
            "注記": "cinema-oriented camera gamut",
        },
    )


def test_vocabulary_defines_chroma_siting_records_and_applicability(vocabulary_markdown: str) -> None:
    """v1-format-boundary acceptance 39: H.273 siting records and format applicability are exact."""
    assert _table_records(
        vocabulary_markdown,
        "chroma siting",
        ("Token", "Offset `(x, y)`", "H.273", "定義"),
    ) == (
        {
            "Token": "`left`",
            "Offset `(x, y)`": "`(0, 0.5)`",
            "H.273": "H.273 type 0",
            "定義": "水平 co-sited・垂直 interstitial。BT.601 / BT.709 SDR 配信の典型慣行",
        },
        {
            "Token": "`center`",
            "Offset `(x, y)`": "`(0.5, 0.5)`",
            "H.273": "H.273 type 1",
            "定義": "2×2 luma block の幾何中心",
        },
        {
            "Token": "`topleft`",
            "Offset `(x, y)`": "`(0, 0)`",
            "H.273": "H.273 type 2",
            "定義": "両軸 co-sited。BT.2020 / BT.2100 の規格位置",
        },
    )
    section = re.sub(r"\s+", " ", _section(vocabulary_markdown, "chroma siting")).strip()
    assert (
        "chroma siting token は progressive 4:2:0 入力の chroma sample 中心を、左上 luma sample 中心 = `(0, 0)`、 "
        "luma 間隔 = 1 の frame 座標で定義する。`px.io.from_nv12` / `px.io.from_p010` / "
        "`px.io.from_yuv420p` と `px.io.to_nv12` / `px.io.to_p010` / `px.io.to_yuv420p` にだけ露出し、既定は "
        "`left`。colorimetry から 自動導出せず、BT.2020 / BT.2100 素材で規格位置を使う場合は `topleft` を明示する。"
    ) in section
    assert (
        "4:2:2 (`px.io.from_uyvy422` / `px.io.from_v210` / `px.io.from_yuv422p`) は水平 co-sited・垂直 full に固定する。"
        "選択余地が ないため siting 引数を持たない。4:4:4 (`px.io.from_yuv444p` / `px.io.from_yuva444p`) は "
        "subsampling がないため siting と interpolation の両方を持たない。"
    ) in section


def test_vocabulary_image_format_and_compression_tokens_match_the_bytes_boundary(
    vocabulary_markdown: str,
) -> None:
    """v1-bytes-boundary acceptance 12; v1-exr-write-compression acceptance 11:
    documented encode tokens equal the implementation sets.
    """
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
    section = vocabulary_markdown.split("## image write dtype\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    for row in (
        "| PNG / TIFF / JPEG 2000 / PNM | `uint8` / `uint16` | `uint8` |",
        "| JPEG / WebP / BMP | `uint8` | `uint8` |",
        "| EXR | `float16` / `float32` / `uint32` | 通常 `float16`、uint32 Frame は `uint32` |",
        "| TGA | `uint8` | `uint8` |",
        "| HDR | `float32` | `float32` |",
        "| DPX | `float32` | `float32` |",
    ):
        assert row in section


def test_vocabulary_stack_directions_equal_the_validator_and_document_join_rules(
    vocabulary_markdown: str,
) -> None:
    """v1-stack acceptance 7: documented direction tokens and join rules equal the implementation contract."""
    from pixtreme._transform.stack import _STACK_DIRECTION_TOKENS

    section = _section(vocabulary_markdown, "stack direction")
    default = re.search(r"既定は `([^`]+)`", section)

    assert default is not None and default.group(1) == "vertical"
    assert (
        _table_tokens(vocabulary_markdown, "stack direction") == ("vertical", "horizontal") == _STACK_DIRECTION_TOKENS
    )
    assert _table_records(vocabulary_markdown, "stack direction", ("Token", "結合規則")) == (
        {
            "Token": "`vertical`",
            "結合規則": "上から下へ並べる。出力 height は各入力 height の合計、width は共通値",
        },
        {
            "Token": "`horizontal`",
            "結合規則": "左から右へ並べる。出力 width は各入力 width の合計、height は共通値",
        },
    )


def test_vocabulary_documents_channel_shuffle_routing_and_provenance(vocabulary_markdown: str) -> None:
    """v1-channel-shuffle acceptance 21: vocabulary fixes routing, reserved words, and matrix provenance."""
    assert _table_records(vocabulary_markdown, "channel routing", ("Output 構造", "`Frame.matrix`")) == (
        {"Output 構造": "R / G / B と Y / Cb / Cr が混在", "`Frame.matrix`": "`None`"},
        {"Output 構造": "RGB-only、または Y / Cb / Cr を含まない", "`Frame.matrix`": "`None`"},
        {
            "Output 構造": "Y / Cb / Cr の Frame source claim がすべて同じ non-`None` token",
            "`Frame.matrix`": "その token (`native` も逐語で保持)",
        },
        {
            "Output 構造": "claim に `None` が一つ以上ある、または Y / Cb / Cr がすべて fill",
            "`Frame.matrix`": "`None`",
        },
        {
            "Output 構造": "non-`None` claim が複数 token に分かれる",
            "`Frame.matrix`": "3 要素 error。暗黙 rematrix は行わない",
        },
    )


def test_vocabulary_documents_from_format_conventions(vocabulary_markdown: str) -> None:
    """v1-format-boundary acceptance 39; v1-from-format-metadata acceptance 7.

    v1-color-semantics acceptance 5 and 35: from-format defaults and layouts are exact records.
    """
    assert _table_records(vocabulary_markdown, "from_<format> 慣習値", ("項目", "仕様既定", "注記")) == (
        {
            "項目": "colorspace",
            "仕様既定": "`Rec.709`",
            "注記": "placeholder。`colorspace=` の per-call 明示がある場合は指定 token を優先する",
        },
        {
            "項目": "gamma",
            "仕様既定": "`rec709`",
            "注記": "placeholder。`gamma=` の per-call 明示がある場合は指定 token を優先する",
        },
        {
            "項目": "matrix",
            "仕様既定": "`None`",
            "注記": "provenance 不明。`matrix=` の per-call 明示がある場合は指定 token をそのまま stamp する",
        },
        {"項目": "channels", "仕様既定": '`("Y", "Cb", "Cr")`', "注記": "format 解決後の固定 channel 順"},
        {
            "項目": "range",
            "仕様既定": "`legal`",
            "注記": '映像系 YCbCr 入力の既定仮定。`range="full"` で per-call 上書きする',
        },
        {
            "項目": "interpolation",
            "仕様既定": "`bilinear`",
            "注記": "subsampling を持つ 6 format の既定。受理集合は interpolation 節の 8 token",
        },
        {
            "項目": "siting",
            "仕様既定": "`left`",
            "注記": "4:2:0 の 3 format だけが持つ。受理集合は chroma siting 節の 3 token",
        },
    )
    assert _table_records(
        vocabulary_markdown,
        "from_<format> 慣習値",
        ("Format", "bit_depth", "Container dtype", "Plane order"),
    ) == (
        {
            "Format": "`yuv420p`",
            "bit_depth": "8 (既定) / 10",
            "Container dtype": "8 = uint8、10 = uint16",
            "Plane order": "Y → Cb → Cr、各 chroma plane は H/2 × W/2",
        },
        {
            "Format": "`yuv422p`",
            "bit_depth": "8 (既定) / 10 / 12",
            "Container dtype": "8 = uint8、10 / 12 = uint16",
            "Plane order": "Y → Cb → Cr、各 chroma plane は H × W/2",
        },
        {
            "Format": "`yuv444p`",
            "bit_depth": "10 (既定) / 12",
            "Container dtype": "uint16",
            "Plane order": "Y → Cb → Cr、各 plane は H × W",
        },
        {
            "Format": "`yuva444p`",
            "bit_depth": "12 (既定)",
            "Container dtype": "uint16",
            "Plane order": "Y → Cb → Cr → A、各 plane は H × W。A は range に依らず full scale",
        },
    )
    assert _table_records(
        vocabulary_markdown,
        "from_<format> 慣習値",
        ("Format", "Container dtype", "C-contiguous 1D layout"),
    ) == (
        {
            "Format": "`uyvy422`",
            "Container dtype": "uint8",
            "C-contiguous 1D layout": "U0 Y0 V0 Y1。入力だけ `(H, W, 2)` NDI 形も受理し、1D との reshape は zero-copy view で行える",
        },
        {
            "Format": "`v210`",
            "Container dtype": "uint32",
            "C-contiguous 1D layout": "6 pixel = 4 word、各 word は下位から 10-bit × 3。row は 128-byte 境界 (48 pixel 単位)、padding は 0",
        },
        {
            "Format": "`NV12`",
            "Container dtype": "uint8",
            "C-contiguous 1D layout": "Y plane の後に Cb Cr interleaved plane",
        },
        {
            "Format": "`P010`",
            "Container dtype": "uint16",
            "C-contiguous 1D layout": "NV12 と同配置。10-bit code は MSB 詰めで lower 6 bit は 0",
        },
    )


def test_vocabulary_documents_to_format_conventions(vocabulary_markdown: str) -> None:
    """v1-format-boundary acceptance 39; v1-color-semantics acceptance 34-35.

    To-format defaults and layouts are exact records.
    """
    assert _table_records(vocabulary_markdown, "to_<format> 慣習値", ("項目", "仕様既定", "注記")) == (
        {
            "項目": "range",
            "仕様既定": "`legal`",
            "注記": "`full` も受理する。legal 位置では clip せず headroom code を保持する",
        },
        {
            "項目": "interpolation",
            "仕様既定": "`area`",
            "注記": "subsampling を持つ 6 format の既定。受理集合は nearest / bilinear / bicubic / area",
        },
        {
            "項目": "siting",
            "仕様既定": "`left`",
            "注記": "4:2:0 の 3 format だけが持つ。受理集合は chroma siting 節の 3 token",
        },
        {"項目": "rounding", "仕様既定": "half away from zero", "注記": "fp32 から code への最近傍丸め"},
        {
            "項目": "clipping",
            "仕様既定": "container 全域のみ",
            "注記": "legal interval では clip せず、物理的な `[0, 2^n − 1]` だけで clip する",
        },
    )
    assert _table_records(
        vocabulary_markdown,
        "to_<format> 慣習値",
        ("Format", "bit_depth", "Container dtype", "C-contiguous 1D layout"),
    ) == (
        {
            "Format": "`yuv420p`",
            "bit_depth": "8 (既定) / 10",
            "Container dtype": "8 = uint8、10 = uint16",
            "C-contiguous 1D layout": "Y → Cb → Cr、各 chroma plane は H/2 × W/2",
        },
        {
            "Format": "`yuv422p`",
            "bit_depth": "8 (既定) / 10 / 12",
            "Container dtype": "8 = uint8、10 / 12 = uint16",
            "C-contiguous 1D layout": "Y → Cb → Cr、各 chroma plane は H × W/2",
        },
        {
            "Format": "`yuv444p`",
            "bit_depth": "10 (既定) / 12",
            "Container dtype": "uint16",
            "C-contiguous 1D layout": "Y → Cb → Cr、各 plane は H × W",
        },
        {
            "Format": "`yuva444p`",
            "bit_depth": "12 (既定)",
            "Container dtype": "uint16",
            "C-contiguous 1D layout": "Y → Cb → Cr → A。A は range に依らず full scale",
        },
        {
            "Format": "`uyvy422`",
            "bit_depth": "8 固定",
            "Container dtype": "uint8",
            "C-contiguous 1D layout": "U0 Y0 V0 Y1。`(H, W, 2)` への reshape は zero-copy view",
        },
        {
            "Format": "`v210`",
            "bit_depth": "10 固定",
            "Container dtype": "uint32",
            "C-contiguous 1D layout": "6 pixel = 4 word。128-byte row padding は関数が 0 埋めする",
        },
        {
            "Format": "`NV12`",
            "bit_depth": "8 固定",
            "Container dtype": "uint8",
            "C-contiguous 1D layout": "Y plane の後に Cb Cr interleave plane",
        },
        {
            "Format": "`P010`",
            "bit_depth": "10 固定",
            "Container dtype": "uint16",
            "C-contiguous 1D layout": "NV12 と同配置。MSB 詰めで lower 6 bit は 0",
        },
    )


def test_vocabulary_documents_range_dtype_and_quantization_semantics(vocabulary_markdown: str) -> None:
    """v1-quantize-values acceptance 15: range tokens and bit-depth lanes are exact records."""
    assert _table_records(vocabulary_markdown, "range", ("Token", "定義", "準拠規格・慣習", "注記")) == (
        {
            "Token": "`legal`",
            "定義": "H.273 の limited-range code 位置 (`video_full_range_flag = 0`)",
            "準拠規格・慣習": "ITU-T H.273",
            "注記": "Y と limited-range RGB は luma interval、Cb / Cr は chroma interval を使う",
        },
        {
            "Token": "`full`",
            "定義": "unsigned container 全域に対応する full-range 値 (`video_full_range_flag = 1`)",
            "準拠規格・慣習": "ITU-T H.273",
            "注記": "float working の通常状態。Frame に状態 token として保存しない",
        },
    )
    assert _table_records(
        vocabulary_markdown,
        "pixel value quantization",
        ("経路", "API", "`bit_depth` が指すもの", "値・container の扱い"),
    ) == (
        {
            "経路": "range 対",
            "API": "`px.values.legal_to_full` / `px.values.full_to_legal`",
            "`bit_depth` が指すもの": "H.273 legal code 位置の有効 code bit 数",
            "値・container の扱い": "float32 の線形写像、clip なし",
        },
        {
            "経路": "量子化対",
            "API": "`px.values.quantize` / `px.values.dequantize`",
            "`bit_depth` が指すもの": "unsigned full-scale grid の有効 code bit 数",
            "値・container の扱い": "float32 Frame ↔ uint Frame",
        },
        {
            "経路": "named format",
            "API": "`px.io.from_<format>` / `px.io.to_<format>`",
            "`bit_depth` が指すもの": "format が運ぶ code の有効 code bit 数",
            "値・container の扱い": "packing・subsampling・container を format 契約で解決",
        },
        {
            "経路": "汎用出入口",
            "API": "`px.io.from_array` / `px.io.to_array`",
            "`bit_depth` が指すもの": "素配列上の unsigned full-scale grid の有効 code bit 数",
            "値・container の扱い": "layout・channel 選択・`out=` と直交して合成",
        },
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
    for required in ('matrix="bt709"', "bit_depth=8"):
        assert required in docstring


def test_vocabulary_bit_depths_equal_the_shared_validator_set(vocabulary_markdown: str) -> None:
    """v1-quantize-values acceptance 15: the documented five-value axis equals the implementation."""
    from pixtreme._values.quantize import _BIT_DEPTHS

    markdown = vocabulary_markdown
    section = markdown.split("## pixel value quantization\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    bit_depth_line = next(line for line in section.splitlines() if line.startswith("`bit_depth` は"))
    documented = tuple(int(value) for value in re.findall(r"`(\d+)`", bit_depth_line))

    assert documented == _BIT_DEPTHS


def test_vocabulary_documents_matrix_semantics_and_colorspace_derivation(vocabulary_markdown: str) -> None:
    """v1-color-semantics acceptance 35: matrix tokens and own-row values are complete records."""
    assert _table_records(
        vocabulary_markdown,
        "matrix",
        ("Token", "正式表記", "定義", "準拠規格・慣習", "注記"),
    ) == (
        {
            "Token": "`bt601`",
            "正式表記": "BT.601",
            "定義": "Kr = 0.299、Kb = 0.114",
            "準拠規格・慣習": "ITU-T H.273 / ITU-R BT.601",
            "注記": "SD 系 non-constant luminance 係数",
        },
        {
            "Token": "`bt709`",
            "正式表記": "BT.709",
            "定義": "Kr = 0.2126、Kb = 0.0722",
            "準拠規格・慣習": "ITU-T H.273 / ITU-R BT.709",
            "注記": "sRGB / Rec.709 の仕様固定導出先",
        },
        {
            "Token": "`bt2020`",
            "正式表記": "BT.2020",
            "定義": "Kr = 0.2627、Kb = 0.0593",
            "準拠規格・慣習": "ITU-T H.273 / ITU-R BT.2020",
            "注記": "Rec.2020 の仕様固定導出先",
        },
        {
            "Token": "`native`",
            "正式表記": "colorspace own-row",
            "定義": "Frame の現在の colorspace の公表 primaries / white point から作る正規化 RGB→XYZ matrix の Y 行",
            "準拠規格・慣習": "colorspace の各公表規格",
            "注記": "file / hardware の native ではない相対 token。gamma は係数を変えない",
        },
    )
    assert _table_records(
        vocabulary_markdown,
        "matrix own-row",
        ("Colorspace", "own-row `(Kr, Kg, Kb)`", "既知の H.273 基底との関係"),
    ) == (
        {
            "Colorspace": "`sRGB`",
            "own-row `(Kr, Kg, Kb)`": "`(0.2126390059, 0.7151686788, 0.0721923154)`",
            "既知の H.273 基底との関係": "`bt709` と数値一致",
        },
        {
            "Colorspace": "`Rec.709`",
            "own-row `(Kr, Kg, Kb)`": "`(0.2126390059, 0.7151686788, 0.0721923154)`",
            "既知の H.273 基底との関係": "`bt709` と数値一致",
        },
        {
            "Colorspace": "`Rec.2020`",
            "own-row `(Kr, Kg, Kb)`": "`(0.2627002120, 0.6779980715, 0.0593017165)`",
            "既知の H.273 基底との関係": "`bt2020` と数値一致",
        },
        {
            "Colorspace": "`ACES2065-1`",
            "own-row `(Kr, Kg, Kb)`": "`(0.3439664498, 0.7281660966, -0.0721325464)`",
            "既知の H.273 基底との関係": "AP0 own-row",
        },
        {
            "Colorspace": "`ACEScg`",
            "own-row `(Kr, Kg, Kb)`": "`(0.2722287168, 0.6740817658, 0.0536895174)`",
            "既知の H.273 基底との関係": "AP1 own-row",
        },
        {
            "Colorspace": "`S-Gamut3`",
            "own-row `(Kr, Kg, Kb)`": "`(0.2709796708, 0.7866064112, -0.0575860820)`",
            "既知の H.273 基底との関係": "Sony S-Gamut3 own-row",
        },
        {
            "Colorspace": "`S-Gamut3.Cine`",
            "own-row `(Kr, Kg, Kb)`": "`(0.2150758201, 0.8850685017, -0.1001443219)`",
            "既知の H.273 基底との関係": "Sony S-Gamut3.Cine own-row",
        },
    )


def test_vocabulary_documents_view_versions_combinations_and_scope_boundary(vocabulary_markdown: str) -> None:
    """v1-tonemap-aces20-analytic acceptance 18-19: five tonemap tokens and ten exits are explicit."""
    from pixtreme._color.transform import _BT2408_COMBINATIONS
    from pixtreme._color.view_transform import _PUBLIC_TO_INTERNAL_LUT, _SUPPORTED_COMBINATIONS

    assert _PUBLIC_TO_INTERNAL_LUT == {"aces-1.3-lut": "aces-1.3", "aces-2.0-lut": "aces-2.0"}
    assert _table_tokens(vocabulary_markdown, "tonemap") == (
        "aces-1.3",
        "aces-1.3-lut",
        "aces-2.0",
        "aces-2.0-lut",
        "bt2408",
    )
    records = _table_records(
        vocabulary_markdown,
        "tonemap 供給組合せ",
        ("Tonemap", "Output colorspace", "Output gamma", "出口"),
    )
    documented_combinations = tuple(
        (record["Tonemap"].strip("`"), record["Output colorspace"].strip("`"), record["Output gamma"].strip("`"))
        for record in records
    )
    assert documented_combinations == (
        ("aces-1.3", "Rec.709", "bt1886"),
        ("aces-1.3", "sRGB", "srgb"),
        ("aces-1.3-lut", "Rec.709", "bt1886"),
        ("aces-1.3-lut", "sRGB", "srgb"),
        ("aces-2.0", "Rec.709", "bt1886"),
        ("aces-2.0", "sRGB", "srgb"),
        ("aces-2.0-lut", "Rec.709", "bt1886"),
        ("aces-2.0-lut", "sRGB", "srgb"),
        ("bt2408", "Rec.2020", "hlg"),
        ("bt2408", "Rec.2020", "pq"),
    )
    assert set(documented_combinations) == set((*_SUPPORTED_COMBINATIONS, *_BT2408_COMBINATIONS))


def test_vocabulary_documents_image_read_conventions_and_metadata_priority(vocabulary_markdown: str) -> None:
    """v1-io acceptance 5, 6, and 7: image read defaults and metadata priority are exact records."""
    assert _table_records(
        vocabulary_markdown,
        "image read 慣習値",
        ("Format", "仕様既定 colorspace", "仕様既定 gamma", "`channels=None`"),
    ) == (
        {
            "Format": "PNG / JPEG / TIFF",
            "仕様既定 colorspace": "`sRGB`",
            "仕様既定 gamma": "`srgb`",
            "`channels=None`": 'RGB / RGBA、grayscale は 1ch `("Y",)`',
        },
        {
            "Format": "JPEG 2000",
            "仕様既定 colorspace": "`sRGB`",
            "仕様既定 gamma": "`srgb`",
            "`channels=None`": "Y / RGB / RGBA",
        },
        {
            "Format": "WebP",
            "仕様既定 colorspace": "`sRGB`",
            "仕様既定 gamma": "`srgb`",
            "`channels=None`": "RGB",
        },
        {
            "Format": "BMP / PNM",
            "仕様既定 colorspace": "`sRGB`",
            "仕様既定 gamma": "`srgb`",
            "`channels=None`": "Y / RGB",
        },
        {
            "Format": "TGA",
            "仕様既定 colorspace": "`sRGB`",
            "仕様既定 gamma": "`srgb`",
            "`channels=None`": "RGB / RGBA",
        },
        {
            "Format": "HDR",
            "仕様既定 colorspace": "`Rec.709`",
            "仕様既定 gamma": "`linear`",
            "`channels=None`": "RGB",
        },
        {
            "Format": "DPX",
            "仕様既定 colorspace": "`Rec.709`",
            "仕様既定 gamma": "header transfer。unknown は 10-bit=`cineon`、8-bit=`rec709`、12/16-bit=`linear`",
            "`channels=None`": "RGB / RGBA",
        },
        {
            "Format": "EXR",
            "仕様既定 colorspace": "`ACES2065-1`",
            "仕様既定 gamma": "`linear`",
            "`channels=None`": "R / G / B と、存在する場合は A",
        },
    )
    priority = re.search(r"metadata の優先順位は \*\*([^*]+)\*\*", _section(vocabulary_markdown, "image read 慣習値"))
    assert priority is not None
    assert tuple(part.strip() for part in priority.group(1).split(">")) == ("per-call 明示", "file 明示", "仕様既定")


def test_vocabulary_documents_cast_quantization_and_encode_kwargs(vocabulary_markdown: str) -> None:
    """v1-recode-dtype acceptance 8; v1-quantize-values acceptance 15.

    Dtype lanes and encoder kwargs are exact records.
    """
    assert _table_records(
        vocabulary_markdown,
        "dtype operation 対比",
        ("API", "保存する対象", "uint↔float の挙動", "主用途"),
    ) == (
        {
            "API": "`px.values.cast_dtype`",
            "保存する対象": "数字保存",
            "uint↔float の挙動": "CuPy `astype` への忠実委譲。scale / clip / 明示丸めなし",
            "主用途": "unchanged 読みした depth・label 生値の容れ物替え",
        },
        {
            "API": "`px.values.recode_dtype`",
            "保存する対象": "意味保存",
            "uint↔float の挙動": "uint は器の maximum code で正規化。float → uint は `[0, 1]` clip・full-scale・half away from zero 丸め。float 間は literal cast",
            "主用途": "平時の uint 画像と正規化 float Frame の相互変換",
        },
        {
            "API": "`px.values.quantize`",
            "保存する対象": "画素値目盛り",
            "uint↔float の挙動": "float32 を宣言 bit depth の uint full-scale grid へ clip・scale・half away from zero 丸め",
            "主用途": "正規化値から code 値 Frame を作る",
        },
        {
            "API": "`px.values.dequantize`",
            "保存する対象": "画素値目盛り",
            "uint↔float の挙動": "宣言 bit depth の uint code を maximum code で正規化し、clip しない",
            "主用途": "code 値 Frame を float32 working 値へ戻す",
        },
    )
    assert _table_records(
        vocabulary_markdown, "image encode kwargs", ("Kwarg", "API / 対象 format", "値域", "意味")
    ) == (
        {
            "Kwarg": "`quality`",
            "API / 対象 format": "両 API / JPEG・WebP",
            "値域": "integer `1〜100`",
            "意味": "lossy 画質指定。JPEG 2000 / PNG / TIFF / BMP / PNM / EXR への指定は `ValueError`",
        },
        {
            "Kwarg": "`compression`",
            "API / 対象 format": "両 API / TIFF",
            "値域": "token `none` / `lzw`",
            "意味": "TIFF の無圧縮 / LZW 可逆圧縮",
        },
        {
            "Kwarg": "`compression`",
            "API / 対象 format": "`px.io.write_image` / EXR",
            "値域": "EXR compression token",
            "意味": "省略時は `zip`。TIFF token と混同しない",
        },
        {
            "Kwarg": "`compression_level`",
            "API / 対象 format": "両 API / PNG",
            "値域": "integer `0〜9`",
            "意味": "PNG zlib 圧縮 level。PNG 以外への指定は `ValueError`",
        },
        {
            "Kwarg": "`lossless`",
            "API / 対象 format": "両 API / JPEG 2000・WebP",
            "値域": "exact `bool` / `None`",
            "意味": "`True` は可逆、`False` は lossy、`None` は codec 既定。WebP の `quality` と `True` は競合",
        },
        {
            "Kwarg": "`dwa_level`",
            "API / 対象 format": "`px.io.write_image` / EXR DWAA・DWAB",
            "値域": "header float でも positive finite な exact `float` / `None`",
            "意味": "`None` は `45.0`。DWA 以外への明示は `ValueError`",
        },
        {
            "Kwarg": "`bit_depth`",
            "API / 対象 format": "`px.io.write_image` / DPX",
            "値域": "integer `8` / `10` / `12` / `16` / `None`",
            "意味": "`None` は 10-bit。非 DPX への明示は `ValueError`",
        },
        {
            "Kwarg": "`dtype`",
            "API / 対象 format": "`px.io.write_image` / EXR",
            "値域": "`float16` / `float32` / `uint32` / `None`",
            "意味": "明示値は Frame 依存の既定より優先する。非 EXR への明示は `ValueError`",
        },
    )


def test_vocabulary_documents_array_layout_affine_copy_and_out_contracts(vocabulary_markdown: str) -> None:
    """v1-frame-core acceptance 16; v1-boundary-api acceptance 21.

    Layout, copy, and export contracts are exact records.
    """
    assert _table_records(
        vocabulary_markdown,
        "layout",
        ("Token", "Rank / shape", "`px.io.from_array`", "`px.io.to_array`"),
    ) == (
        {
            "Token": "`HWC`",
            "Rank / shape": "`(H, W, C)`",
            "`px.io.from_array`": "そのまま Frame の HWC と解釈",
            "`px.io.to_array`": "HWC view または詰め替え結果",
        },
        {
            "Token": "`NHWC`",
            "Rank / shape": "`(1, H, W, C)`",
            "`px.io.from_array`": "先頭 size-1 軸を除いた HWC view。N > 1 は `ValueError`",
            "`px.io.to_array`": "先頭 size-1 軸を加えた zero-copy view",
        },
        {
            "Token": "`CHW`",
            "Rank / shape": "`(C, H, W)`",
            "`px.io.from_array`": "HWC へ transpose して取り込む",
            "`px.io.to_array`": "channel-first の C-contiguous 配列へ詰め替える",
        },
        {
            "Token": "`NCHW`",
            "Rank / shape": "`(1, C, H, W)`",
            "`px.io.from_array`": "N == 1 を検証し HWC へ transpose して取り込む",
            "`px.io.to_array`": "先頭 size-1 軸付き channel-first 配列へ詰め替える",
        },
    )
    assert _table_records(vocabulary_markdown, "device array affine / copy", ("値", "意味")) == (
        {
            "値": "`copy=None`",
            "意味": "可能なら zero-copy view、layout transpose・channel 選択・dtype / affine 等で必要なら 1 回だけ copy",
        },
        {
            "値": "`copy=False`",
            "意味": "厳格 zero-copy 保証。要求を満たすために書き込みが必要なら 3 要素 error",
        },
        {"値": "`copy=True`", "意味": "常に呼び出し元と storage を共有しない私有複製"},
    )
    assert _table_records(vocabulary_markdown, "Frame 境界契約", ("契約", "必須事項")) == (
        {
            "契約": "GPU layout",
            "必須事項": "`Frame.data` は HWC・C-contiguous。`px.io.from_array` は copy 三値に従って view または GPU 上の copy を選ぶ",
        },
        {
            "契約": "pointer lifetime",
            "必須事項": "`Frame.data` の生ポインタを利用している間は、allocation owner である Frame を保持する",
        },
        {
            "契約": "stream",
            "必須事項": "DLPack consumer stream は変更せず `Frame.data.__dlpack__` へ素通しする",
        },
        {
            "契約": "device export",
            "必須事項": "`px.io.to_array(frame)` の返り値は `cupy.ndarray` であり、Frame と返り値はいずれも DLPack producer である",
        },
        {
            "契約": "direct destination",
            "必須事項": "`out=` は shape / dtype 一致・C-contiguous の `cupy.ndarray` だけを受け、同じ object を返す",
        },
        {
            "契約": "host transfer",
            "必須事項": "`to_array(...).get()` idiom の正規 path は `px.io.to_array(frame, ...).get()`。または `cp.asnumpy(px.io.to_array(frame, ...))` を明示して使う",
        },
    )


def test_vocabulary_anchor_definitions_match_the_block_layout_contract(vocabulary_markdown: str) -> None:
    """v1-draw-text-unification acceptance 13: all anchor meanings use the integrated block box."""
    section = vocabulary_markdown.split("## anchor\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    documented = tuple(
        (
            cells[1].strip().removeprefix("`").removesuffix("`"),
            cells[2].strip(),
        )
        for line in section.splitlines()
        if line.startswith("| `")
        for cells in (line.split("|"),)
    )

    assert documented == (
        ("top-left", "首行 ascender 線と block box 左端"),
        ("top-center", "首行 ascender 線と block box 水平中点"),
        ("top-right", "首行 ascender 線と block box 右端"),
        ("center-left", "top と bottom の中点と block box 左端"),
        ("center-center", "top と bottom の中点と block box 水平中点"),
        ("center-right", "top と bottom の中点と block box 右端"),
        ("baseline-left", "首行 baseline と block box 左端"),
        ("baseline-center", "首行 baseline と block box 水平中点"),
        ("baseline-right", "首行 baseline と block box 右端"),
        ("bottom-left", "末行 descender 線と block box 左端"),
        ("bottom-center", "末行 descender 線と block box 水平中点"),
        ("bottom-right", "末行 descender 線と block box 右端"),
    )


def test_vocabulary_composite_blend_alpha_and_interpolation_match_implementation(
    vocabulary_markdown: str,
) -> None:
    """v1-composite acceptance 18: docs and every validator share the exact composite token subsets."""
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
    assert _table_records(vocabulary_markdown, "blend", ("Token", "`B(Cb, Cs)`")) == (
        {"Token": "`normal`", "`B(Cb, Cs)`": "`Cs`"},
        {"Token": "`lighten`", "`B(Cb, Cs)`": "`max(Cb, Cs)`"},
        {"Token": "`add`", "`B(Cb, Cs)`": "`Cb + Cs`"},
        {"Token": "`screen`", "`B(Cb, Cs)`": "`1 − (1 − Cb) × (1 − Cs)`"},
        {"Token": "`darken`", "`B(Cb, Cs)`": "`min(Cb, Cs)`"},
        {"Token": "`multiply`", "`B(Cb, Cs)`": "`Cb × Cs`"},
        {"Token": "`difference`", "`B(Cb, Cs)`": "`abs(Cb − Cs)`"},
        {
            "Token": "`overlay`",
            "`B(Cb, Cs)`": "`Cb <= 0.5` なら `2 × Cb × Cs`、それ以外は `1 − 2 × (1 − Cb) × (1 − Cs)`",
        },
        {
            "Token": "`hardlight`",
            "`B(Cb, Cs)`": "`Cs <= 0.5` なら `2 × Cb × Cs`、それ以外は `1 − 2 × (1 − Cb) × (1 − Cs)`",
        },
        {
            "Token": "`softlight`",
            "`B(Cb, Cs)`": "`Cs <= 0.5` なら `Cb − (1 − 2 × Cs) × Cb × (1 − Cb)`、それ以外は `Cb + (2 × Cs − 1) × (D(Cb) − Cb)`",
        },
    )
    assert _table_records(vocabulary_markdown, "alpha", ("Token", "定義")) == (
        {
            "Token": "`premultiplied`",
            "定義": "color channel は同じ画素の `A` を乗算済み。alpha 0 の unassociated color は 0 と定義する",
        },
        {
            "Token": "`straight`",
            "定義": "color channel は `A` を未乗算。foreground は color を `A` で associate してから補間する",
        },
    )
