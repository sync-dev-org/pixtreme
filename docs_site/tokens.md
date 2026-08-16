# Token Reference

This page defines the exact spelling and meaning of every named token accepted by the public API, together with the
subset accepted by each API. The single definition point for each closed vocabulary is the corresponding
`typing.Literal` alias exported by `pixtreme.core`; documentation tests mechanically compare the token columns on this
page with those aliases. Runtime validators derive their accepted sets with `typing.get_args(alias)`. Tokens are
case-sensitive exactly as shown. Alternate casing and unlisted tokens raise `ValueError`. Channel sequences are the
only exception: they may contain application-defined labels not listed here.

## Literal aliases and synchronization contract

Closed-vocabulary aliases use PascalCase and map to the sections below. When multiple APIs accept subsets of one
vocabulary, the alias remains a single definition of the complete token set. Each API-specific subset is fixed in the
corresponding section.

| Alias | Vocabulary section |
|---|---|
| `Layout` | layout |
| `Gamma` | gamma |
| `Colorspace` | colorspace |
| `Tonemap` | tonemap |
| `Matrix` | matrix |
| `Range` | range |
| `Interpolation` | interpolation |
| `StackDirection` | stack direction |
| `SobelDirection` | sobel direction |
| `TemplateMatchingMethod` | template matching method |
| `ChromaSiting` | chroma siting |
| `TextLanguage` | language |
| `TextAnchor` | anchor |
| `TextAlign` | text align |
| `TextFont` | text font |
| `Blend` | blend |
| `Alpha` | alpha |
| `Antialiasing` | aa |
| `GeneratorKind` | generator kind |
| `ColorBarsStandard` | color bars standard |
| `ColorBarsOutput` | color bars output |
| `MorphologyShape` | morphology shape |
| `Border` | border |
| `VectorBlurShutter` | vector blur shutter |
| `Dtype` | dtype |
| `ImageFormat` | image format |
| `TiffCompression` | TIFF compression |
| `ExrCompression` | EXR compression |

`channels` is an open set because it permits unknown labels, so it has no Literal alias. Closed numeric and Boolean
arguments such as `bit_depth`, the three-state `copy` option, dimensions, and quality values are not named tokens and
are outside this alias table.

## channels

A compact channel string is split greedily by longest match against the known labels below. An unlisted label can be
specified only through `Sequence[str]`. `px.core.channels` is the single public normalizer; this open vocabulary
therefore retains `str` plus runtime validation.

| Token | Definition | Standard or convention | Notes |
|---|---|---|---|
| `R` | Red component of an RGB representation | RGB standard named by the colorspace token | Chromaticity and transfer are determined by the Frame colorspace and gamma |
| `G` | Green component of an RGB representation | RGB standard named by the colorspace token | Chromaticity and transfer are determined by the Frame colorspace and gamma |
| `B` | Blue component of an RGB representation | RGB standard named by the colorspace token | Chromaticity and transfer are determined by the Frame colorspace and gamma |
| `H` | Hue turn | HSV cylindrical coordinates | Period 1; canonical output is `[0, 1)`, and the inverse conversion accepts all real values modulo 1 |
| `S` | HSV saturation | HSV cylindrical coordinates | In `[0, 1]` for nonnegative RGB input; the range is not enforced for arbitrary input |
| `V` | HSV value | HSV cylindrical coordinates | The RGB maximum; an unbounded scene scale that may exceed 1 |
| `A` | Alpha or opacity component | OpenEXR and general image-API convention | Premultiplication state is not stored in Frame metadata |
| `Y` | Luma or nonlinear luminance component | ITU-T H.273 and grayscale convention | Read as luma in YCbCr and as achromatic intensity in a one-channel Frame |
| `Cb` | Blue-difference chroma component | ITU-T H.273 | Longest matching treats it as one label in compact notation |
| `Cr` | Red-difference chroma component | ITU-T H.273 | Longest matching treats it as one label in compact notation |
| `Z` | Depth component | OpenEXR channel-naming convention | Outside golden-path color processing |

## RGB / HSV conversion

`px.color.rgb_to_hsv` accepts R, G, and B values in any label order and produces a Frame in canonical
`("H", "S", "V")` order. Per pixel, let `maximum = max(R, G, B)`, `minimum = min(R, G, B)`,
`delta = maximum - minimum`, and `V = maximum`. If `maximum == 0`, then `S = 0`; otherwise,
`S = delta / maximum`. If `delta == 0`, then `H = 0`. A colored pixel uses the row selected by its maximum component,
then maps the result modulo 1 into `[0, 1)`.

| Maximum component | `H` before wrapping |
|---|---:|
| R | `((G - B) / delta) / 6` |
| G | `(2 + (B - R) / delta) / 6` |
| B | `(4 + (R - G) / delta) / 6` |

`px.color.hsv_to_rgb` accepts H, S, and V values in any label order and produces canonical
`("R", "G", "B")` order. Let `h = H modulo 1`, `h6 = 6 * h`, `i = floor(h6)`, `C = V * S`,
`X = C * (1 - abs((h6 modulo 2) - 1))`, and `m = V - C`. For sectors `i = 0..5`, `(R', G', B')` is respectively
`(C, X, 0)`, `(X, C, 0)`, `(0, C, X)`, `(0, X, C)`, `(X, 0, C)`, or `(C, 0, X)`. The output is
`(R' + m, G' + m, B' + m)`. When `S = 0`, the result is `(V, V, V)` regardless of H.

Neither operation infers the input domain, clips, or normalizes. For finite nonnegative RGB, H is in `[0, 1)`, S is
in `[0, 1]`, V is an unbounded nonnegative scene scale, and RGB to HSV to RGB is reversible within fp32 tolerance.
Negative values and NaN or infinity are evaluated by the same formulas rather than rejected, but fall outside the
nominal HSV ranges and round-trip guarantee.

## channel routing

`px.channel.shuffle` with `adapt=False` and `**outputs` is the only public operation for channel reordering,
extraction, routing across Frames, and constant filling. Each `**outputs` key is the literal output label. Each value is
either `(source_frame, source_label)` or a real number other than `bool`. Keyword insertion order becomes output
channel order. The operation does not split compact labels or validate semantic compatibility between source and
output labels.

```python
px.channel.shuffle(B=(frame, "B"), G=(frame, "G"), R=(frame, "R"))
px.channel.shuffle(R=(foreground, "R"), G=(foreground, "G"), B=(foreground, "B"), A=(matte, "Y"))
px.channel.shuffle(Y=(frame, "Y"), Cb=0.5, Cr=0.5)
px.channel.shuffle(**{"left.diffuse.R": (frame, "R"), "depth.Z": (depth, "Z")})
```

Pass non-identifier labels, including dotted OpenEXR-style labels, through `**{...}`. `adapt` is a reserved option
name and cannot be an output label. The first Frame source establishes width, height, colorspace, and gamma; preceding
fills do not participate in this selection. All Frame sources must be float32 with identical geometry. With
`adapt=False`, colorspace and gamma must also match. With `adapt=True`, each distinct mismatching source is converted
once to the reference representation through `px.color.rgb_to_rgb`. Routing itself only bit-copies source slices and
fills float32 constants; it never clips values.

Matrix provenance is determined from the output labels and the source Frames passed to the call:

| Output structure | `Frame.matrix` |
|---|---|
| R / G / B mixed with Y / Cb / Cr | `None` |
| RGB-only, or no Y / Cb / Cr channels | `None` |
| All Y / Cb / Cr Frame-source claims are the same non-`None` token | That token, preserving `native` literally |
| At least one claim is `None`, or all Y / Cb / Cr outputs are fills | `None` |
| Non-`None` claims contain multiple tokens | Three-part error; no implicit rematrixing |

Relabeling a source channel to a different Y / Cb / Cr output label still makes that source Frame contribute its
claim. Even with `adapt=True`, provenance comes from the Frames passed by the caller rather than temporary converted
Frames. Use `px.color.rgb_to_ycbcr` or `px.color.ycbcr_to_rgb` for RGB/YCbCr conversion and
`px.color.ycbcr_to_ycbcr` for explicit rematrixing; channel routing never combines those operations.

## layout

Layout tokens declare dimension order at the device-array boundary. They determine how `px.io.from_array` interprets
input shape and which shape `px.io.to_array` produces. The default is `HWC`. Tokens are case-sensitive.

| Token | Rank / shape | `px.io.from_array` | `px.io.to_array` |
|---|---|---|---|
| `HWC` | `(H, W, C)` | Interpreted directly as Frame HWC | HWC view or repacked result |
| `NHWC` | `(1, H, W, C)` | HWC view after removing the leading size-1 axis; N > 1 raises `ValueError` | Zero-copy view with a leading size-1 axis |
| `CHW` | `(C, H, W)` | Transposed into HWC | Repacked as a channel-first C-contiguous array |
| `NCHW` | `(1, C, H, W)` | Validates N == 1 and transposes into HWC | Repacked as a channel-first array with a leading size-1 axis |

## device array affine / copy

The `px.io.to_array` affine is `y = (x × scale - mean) / std`; the inverse affine in `px.io.from_array` is
`x = (y × std + mean) / scale`. A `px.io.to_array` followed by `px.io.from_array` with the same constants returns the
original Frame values. Each constant accepts either a scalar or a sequence whose length equals the channel count.
Defaults are `scale=1`, `mean=0`, and `std=1`. Arithmetic is fp32, followed by a faithful CuPy cast to the requested
dtype. For meaning-preserving rounding and clipping, use `px.io.to_array(frame, bit_depth=...)` or the Frame-domain
`px.values.quantize`.

`px.io.from_array` and `px.io.to_array` share this three-state `copy` contract:

| Value | Meaning |
|---|---|
| `copy=None` | Use a zero-copy view when possible; otherwise make exactly one copy when required by layout transposition, channel selection, dtype, affine processing, or another requested operation |
| `copy=False` | Strict zero-copy guarantee; a request that requires writing raises a three-part error |
| `copy=True` | Always return a private copy that shares no storage with the caller |

`px.io.to_array(out=...)` is outside this three-state contract and cannot be combined with `copy=`. `out` must be a
C-contiguous `cupy.ndarray` with the expected shape and dtype. The fused pass writes directly into it and returns the
same object. Other DLPack producers cannot be destinations. When the caller can guarantee a writable zero-copy view,
it may explicitly pass a CuPy view created with `cp.from_dlpack(tensor)`.

## gamma

Gamma tokens describe the transfer characteristic applied to pixel values.

| Token | Definition | Standard or convention | Out-of-domain extension and notes |
|---|---|---|---|
| `linear` | Scene-linear light | ACES working convention | Identity extended naturally to all real values; fixed as **scene-referred** and never used to mean display-linear |
| `srgb` | Piecewise sRGB transfer | IEC 61966-2-1 | Linear and power branches extend naturally below 0 and above 1; the standard transfer for the `sRGB` colorspace |
| `rec709` | Rec.709 camera OETF | ITU-R BT.709 | Linear and power branches extend naturally below 0 and above 1; independent of the `Rec.709` primaries token |
| `bt1886` | Reference-display EOTF | ITU-R BT.1886 | Display transfer naturally extended as a signed-power branch with nominal exponent 2.4 |
| `pq` | Perceptual quantizer | SMPTE ST 2084 / ITU-R BT.2100 | Apply the standard formula to the nonnegative magnitude and reflect the negative side with preserved sign, `f(-x) = -f(x)`; absolute-luminance encoding |
| `hlg` | Hybrid log-gamma | ITU-R BT.2100 | Extend the piecewise low power and high logarithmic branches naturally with sign; scene-referred broadcast HDR transfer |
| `s-log3` | S-Log3 camera log transfer | Sony S-Log3 specification | Apply the standard formula to nonnegative magnitude and reflect the negative side with preserved sign; validate independently of S-Gamut colorspaces |
| `logc4` | LogC4 camera log transfer | ARRI LogC4 specification | Apply the standard formula to nonnegative magnitude and reflect the negative side with preserved sign; ARRI Wide Gamut 4 is not currently a colorspace token |
| `cineon` | Cineon printing-density log transfer | Kodak Cineon specification | Formula with black CV=95, white CV=685, 0.002 density/code, and film gamma=0.6; apply to nonnegative magnitude and reflect the negative side with preserved sign |
| `2.2` | Power transfer with exponent 2.2 | Conventional value | **Pure power**, reflected with preserved sign; not a piecewise function |
| `2.4` | Power transfer with exponent 2.4 | Conventional value | **Pure power**, reflected with preserved sign; not BT.1886 |
| `2.6` | Power transfer with exponent 2.6 | Conventional value | Decode with `sign(x) * abs(x) ** 2.6` and encode with `sign(x) * abs(x) ** (1 / 2.6)`; no offset, piecewise branch, or clipping |

## reference white

`ReferenceWhite` is the case-sensitive display-white axis accepted by
`px.color.white_point_simulation` and `px.color.chromatic_adaptation`. A white
without a token is supplied directly as a two-element CIE 1931 xy sequence.

| Token | CIE 1931 xy | Standard or convention | Notes |
|---|---|---|---|
| `d65` | `(0.3127, 0.3290)` | CIE D65; ITU / SMPTE signal and display white | Identical to the D65 coordinate in the named colorspace definitions |
| `d93` | `(0.2831, 0.2971)` | SMPTE ST 2080-1 regional 9300 K display white | The four-decimal D93 / 9300 K + 8 MPCD broadcast-monitor convention |
| `d50` | `(0.3457, 0.3585)` | CIE D50; ISO 12646 / ICC PCS print and proofing white | Four-decimal soft-proof monitor calibration white |
| `aces` | `(0.32168, 0.33767)` | Academy-published ACES white point | Bit-identical to the ACES2065-1 and ACEScg nominal white coordinate |

`d93` does not denote a separate illuminant standardized by ISO/CIE 11664-2.
It does not normalize the unrounded CIE daylight-formula coordinate, 9300 K +
27 MPCD, a manufacturing target, or a tolerance. Alternate spellings such as
`d93-8mpcd`, `d93-27mpcd`, and `cie-d93` are not tokens.

The three white-related operations have separate responsibilities:

| Operation | Responsibility | White description | Matrix model |
|---|---|---|---|
| chromatic_adaptation | Perceptual adaptation between a pair of white points | `ReferenceWhite` or CIE 1931 xy | Selected CAT cone-response adaptation |
| white_balance | Source-illuminant correction | Temperature / Tint (Kelvin and signed raw Duv) | Black-body-locus white pair and selected CAT |
| white_point_simulation | Physical re-encoding between displays with the same RGB primaries | `ReferenceWhite` or CIE 1931 xy | Input and output normalized device matrices |

`white_point_simulation` preserves absolute XYZ rather than adapting media
white. Its meaning is equivalent to ICC absolute colorimetric intent: it does
not perform perceptual chromatic adaptation, gamut mapping, tone mapping, or
implicit clipping.

## chromatic adaptation

`ChromaticAdaptation` is the case-sensitive CAT axis used by
`px.color.chromatic_adaptation` and `px.color.white_balance`. Both functions
default to `cat02`; `None`, alternate spelling, and case variants are invalid.
`chromatic_adaptation` accepts either `ReferenceWhite` tokens or direct CIE
1931 xy sequences for both white points. `white_point_simulation` does not
accept or apply a CAT.

| Token | Definition | Standard or convention | Notes |
|---|---|---|---|
| `bradford` | Bradford cone-response transform | Bradford chromatic adaptation | Explicit opt-in |
| `cat02` | CIECAM02 chromatic adaptation transform | CAT02 | Default for both public functions |
| `cat16` | CAM16 chromatic adaptation transform | CAT16 | Explicit opt-in |
| `von-kries` | von Kries cone-response transform | von Kries | Explicit opt-in |

`white_balance` describes its source illuminant with Kelvin Temperature and a
signed raw Duv Tint in CIE 1960 UCS. Positive Tint is the green side and its
correction moves the output toward magenta; negative Tint is the magenta side.
For UI explanation only, the Adobe-style display scale is approximately
`adobe_tint = -3000 * Duv`. The public calculation accepts raw Duv without a
vendor slider range or clamp.

## colorspace

Colorspace tokens identify a set of RGB primaries and a white point. Gamma tokens independently describe transfer
characteristics. sRGB and Rec.709 have identical primaries but remain separate tokens because their standard transfers
and usage contexts differ.

| Token | Definition | Standard or convention | Notes |
|---|---|---|---|
| `sRGB` | sRGB primaries and D65 white | IEC 61966-2-1 | Primaries and white point are identical to Rec.709 |
| `Rec.709` | BT.709 primaries and D65 white | ITU-R BT.709 | Primaries and white point are identical to sRGB |
| `Rec.2020` | BT.2020 wide-gamut primaries and D65 white | ITU-R BT.2020 | Specify the HDR transfer separately with gamma |
| `ACES2065-1` | ACES AP0 primaries and ACES white | SMPTE ST 2065-1 | ACES interchange colorspace |
| `ACEScg` | ACES AP1 primaries and ACES white | Academy ACES specification | Scene-linear working colorspace |
| `S-Gamut3` | Sony S-Gamut3 primaries | Sony technical specification | Camera gamut; specify a transfer such as `s-log3` separately |
| `S-Gamut3.Cine` | Sony S-Gamut3.Cine primaries | Sony technical specification | Cinema-oriented camera gamut |

`px.color.rgb_to_rgb` constructs normalized primary matrices from the published RGB primaries and white point in each
row. Conversions between different white points, such as D65 and ACES white, use the **Bradford** CAT. A colorspace
conversion between sRGB and Rec.709 is the identity because their primaries and white point match. This is a technical
conversion and does not include a tone scale or display rendering.

## golden path

The recommended working state is **ACEScg, linear, full-range, float32**, preserving negative values and overshoots
above 1.0. This golden path is a recommendation, not a requirement. It does not apply to non-color channels such as Z,
alpha, or masks, or to ML boundaries. Explicit output operations own display-gamut clipping and tone scaling.

## tonemap

Tonemap tokens select explicit rendering or direct mapping performed by `tonemap=` in `px.color.rgb_to_rgb`. `None`
performs only the technical conversion, without rendering or white placement.

| Token | Definition | ACES generation in the OCIO built-in config | Notes |
|---|---|---|---|
| `aces-1.3` | ACES 1.3 RRT + ODT family SDR rendering | ACES 1.3 | Evaluate the ACES 1.0 SDR Video view directly in one analytic CUDA pass |
| `aces-1.3-lut` | ACES 1.3 RRT + ODT family SDR rendering | ACES 1.3 | Evaluate the same view with the existing 4,096-point shaper and 65³ LUT |
| `aces-2.0` | ACES 2.0 Output Transform SDR rendering | ACES 2.0 | Evaluate the ACES 2.0 SDR 100-nit view directly with one analytic CUDA pass and a fixed algorithm table |
| `aces-2.0-lut` | ACES 2.0 Output Transform SDR rendering | ACES 2.0 | Evaluate the ACES 2.0 SDR 100-nit view with the existing 4,096-point shaper and 65³ LUT |
| `bt2408` | Direct mapping that places SDR reference white at HDR Reference White, 203 cd/m² | — | Not inverse tone mapping with a gradation curve; multiplies every Rec.2020-linear RGB component by the same positive gain |

An ACES token without a suffix selects the analytic implementation; `-lut` selects the LUT implementation. The
`aces-2.0` algorithm table contains hue-dependent reach M and gamut-boundary parameters fixed for 100-nit Rec.709/D65;
it is not a 65³ grid approximation of output RGB. The immutable source constants contain 363 reach-M records,
non-uniform hue, and cusp and upper-hull data: 1,815 float32 values, or 7,260 bytes. CAM, tone, chroma, gamut, and display
encoding are evaluated analytically per pixel. ACES generation identities stored in LUT archives remain `aces-1.3` and
`aces-2.0`; public `-lut` tokens map to them explicitly.

`bt2408` applies gain after input-transfer decoding and the Rec.2020 primary matrix, but before output-transfer
encoding. For HLG, derive `G_HLG = (exp((0.75 - c) / a) + b) / 12` in closed form from the BT.2100 OETF constants
`a = 0.17883277`, `b = 1 - 4a`, and `c = 0.5 - a × ln(4a)`. For PQ, `G_PQ = 203 / 10000` under the 10,000 cd/m²
normalization. Thus Rec.2020-linear `(1, 1, 1)` maps to HLG signal `0.75` or approximately 58% PQ signal. The 58% PQ
figure is an explanatory approximation derived from ST 2084, not an arithmetic constant or numerical oracle. Neither
intermediate gain application nor output is clipped, preserving negative values and values above reference white.

## tonemap combinations

Version 1.0 supplies exactly these ten tonemap/output combinations. Both output colorspace and output gamma must be
explicit when a tonemap is selected. Any unlisted combination raises `ValueError` immediately.

| Tonemap | Output colorspace | Output gamma | Destination |
|---|---|---|---|
| `aces-1.3` | `Rec.709` | `bt1886` | Rec.1886 Rec.709 display |
| `aces-1.3` | `sRGB` | `srgb` | sRGB display |
| `aces-1.3-lut` | `Rec.709` | `bt1886` | Rec.1886 Rec.709 display |
| `aces-1.3-lut` | `sRGB` | `srgb` | sRGB display |
| `aces-2.0` | `Rec.709` | `bt1886` | Rec.1886 Rec.709 display |
| `aces-2.0` | `sRGB` | `srgb` | sRGB display |
| `aces-2.0-lut` | `Rec.709` | `bt1886` | Rec.1886 Rec.709 display |
| `aces-2.0-lut` | `sRGB` | `srgb` | sRGB display |
| `bt2408` | `Rec.2020` | `hlg` | BT.2100 HLG; SDR reference white = 75% signal |
| `bt2408` | `Rec.2020` | `pq` | BT.2100 PQ; SDR reference white = 203 cd/m² |

OCIO DisplayViewTransform is a general boundary that dynamically selects display, view, and look from the caller's
configuration. In contrast, `aces-1.3` fixes the `ACES 1.0 - SDR Video` view from Studio Config
`studio-config-v2.2.0_aces-v1.3_ocio-v2.4` as analytic formulas in a fused GPU kernel, without OCIO, shapers, 3D LUTs,
or package data. `aces-2.0` likewise fixes the `ACES 2.0 - SDR 100 nits (Rec.709)` view from Studio Config
`studio-config-v4.0.0_aces-v2.0_ocio-v2.5`, evaluating AP1 input limiting, Hellwig JMh, tone/chroma/gamut compression,
Rec.709/D65 limiting RGB, the reference's intrinsic range, and display encoding in that order. The reference's
intrinsic `[0, 1]` range is part of the Output Transform, not an added post-processor clip. `aces-1.3-lut` and
`aces-2.0-lut` fix each generation's image with pre-baked shaper and 65³ LUT data plus the **pixtreme version**; they are
not runtime OCIO-compatible APIs. `bt2408` also uses neither LUT data nor ACES shaper or bake tooling, evaluating the
analytic formula above directly in the fused GPU kernel. A change to a fixed ACES rendering is a pixtreme release and
changelog event. No path adds a post-clip to rendered output RGB.

## matrix

Matrix tokens identify the luma coefficients used by H.273 non-constant-luminance equations. They always operate on
full-range float with chroma centered at 0.5. A **token denotes coefficients only; it never bundles a range**.

| Token | Formal name | Definition | Standard or convention | Notes |
|---|---|---|---|---|
| `bt601` | BT.601 | Kr = 0.299, Kb = 0.114 | ITU-T H.273 / ITU-R BT.601 | SD-family non-constant-luminance coefficients |
| `bt709` | BT.709 | Kr = 0.2126, Kb = 0.0722 | ITU-T H.273 / ITU-R BT.709 | Specification-fixed result for sRGB and Rec.709 |
| `bt2020` | BT.2020 | Kr = 0.2627, Kb = 0.0593 | ITU-T H.273 / ITU-R BT.2020 | Specification-fixed result for Rec.2020 |
| `native` | Colorspace own-row | Y row of the normalized RGB-to-XYZ matrix constructed from the Frame's current published colorspace primaries and white point | Published standard for each colorspace | Relative token, not native to a file or device; gamma does not change the coefficients |

## matrix own-row

`native` uses the following own-row `(Kr, Kg, Kb)` values. They result from constructing normalized primary matrices
in float64 from published xy primaries and white points. Correcting a colorspace attribute therefore changes the row
to which `native` resolves.

| Colorspace | own-row `(Kr, Kg, Kb)` | Relationship to a known H.273 basis |
|---|---|---|
| `sRGB` | `(0.2126390059, 0.7151686788, 0.0721923154)` | Numerically identical to `bt709` |
| `Rec.709` | `(0.2126390059, 0.7151686788, 0.0721923154)` | Numerically identical to `bt709` |
| `Rec.2020` | `(0.2627002120, 0.6779980715, 0.0593017165)` | Numerically identical to `bt2020` |
| `ACES2065-1` | `(0.3439664498, 0.7281660966, -0.0721325464)` | AP0 own-row |
| `ACEScg` | `(0.2722287168, 0.6740817658, 0.0536895174)` | AP1 own-row |
| `S-Gamut3` | `(0.2709796708, 0.7866064112, -0.0575860820)` | Sony S-Gamut3 own-row |
| `S-Gamut3.Cine` | `(0.2150758201, 0.8850685017, -0.1001443219)` | Sony S-Gamut3.Cine own-row |

## matrix resolver

When matrix is omitted for RGB encoding (`px.color.rgb_to_ycbcr` or `px.color.rgb_to_grayscale`), resolve from the
output representation in this order: sRGB or Rec.709 to `bt709`; Rec.2020 to `bt2020`; any remaining colorspace with
`linear` gamma to `native`; and any remaining colorspace with nonlinear gamma to `bt709`. An explicit token is never
canonicalized even when its coefficients are equal to another token. Read Y as luminance under `linear` gamma and as
luma under nonlinear gamma.

When matrix is omitted for YCbCr decoding, on the input side of `px.color.ycbcr_to_rgb` or
`px.color.ycbcr_to_ycbcr`, resolve in this order: explicit per-call value, `frame.matrix`, `bt709` for sRGB or Rec.709,
`bt2020` for Rec.2020, then error. ACES and S-Gamut families are never guessed. For camera material without provenance,
`matrix="bt709"` is the most common practical candidate, but the caller must state it.

`px.color.ycbcr_to_ycbcr` keeps input and output matrix, range, and bit depth independent. When output matrix is
omitted, preserve the resolved input token if output colorspace equals input colorspace; only a colorspace change uses
the RGB-encode resolver. Both ends accept `full` or `legal` range and bit depth 8, 10, 12, 14, or 16.

## range

Range tokens describe code-range interpretation at named-format boundaries. Frame metadata does not store range
state. The explicit recovery operations `px.values.legal_to_full` and `px.values.full_to_legal` are directionally
named and anchored to the H.273 `video_full_range_flag`; they map legal code positions for a selected bit depth to
normalized full-range float values and back. The mapping is linear and does not clip, preserving negative values and
values above 1.0.

`bit_depth` accepts the closed set `8`, `10`, `12`, `14`, and `16`.

| Token | Definition | Standard or convention | Notes |
|---|---|---|---|
| `legal` | H.273 limited-range code positions, `video_full_range_flag = 0` | ITU-T H.273 | Y and limited-range RGB use the luma interval; Cb and Cr use the chroma interval |
| `full` | Full-range values spanning the entire unsigned container, `video_full_range_flag = 1` | ITU-T H.273 | Normal state for float working; not stored as a Frame state token |

OCIO RangeTransform is a general transform that remaps or clamps arbitrary input and output bounds, so it differs
semantically from these range tokens. The pixtreme operations are limited to unclipped H.273 legal/full recovery.

## pixel value quantization

Across every path, `bit_depth` accepts `8`, `10`, `12`, `14`, or `16` and means the same **number of effective code
bits**. Maximum code is `2^bit_depth - 1`. The container is `uint8` at 8 bits and `uint16` at 10 through 16 bits. Bit
depth is a per-call conversion claim, never persistent Frame metadata.

`px.values.quantize` clips a float32 Frame to `[0, 1]`, multiplies by maximum code, and rounds half away from zero.
`px.values.dequantize` multiplies by the reciprocal of the declared maximum code and returns a float32 Frame. It does
not validate or clip values in the container, so input above maximum code remains above 1.0. This is mapping to a
uniform unsigned full-scale grid, not palette quantization or ML affine quantization.

`px.io.from_array(bit_depth=...)` and `px.io.to_array(bit_depth=...)` apply the same numeric rule at the raw-array
boundary. `bit_depth` cannot be combined with `scale`, `mean`, or `std`. A `from_array` bit-depth conversion returns
float32; a `to_array` bit-depth conversion fixes its output dtype to the container derived from bit depth.

| Path | API | Meaning of `bit_depth` | Value and container handling |
|---|---|---|---|
| Range pair | `px.values.legal_to_full` / `px.values.full_to_legal` | Effective code bits for H.273 legal code positions | Linear float32 mapping without clipping |
| Quantization pair | `px.values.quantize` / `px.values.dequantize` | Effective code bits for the unsigned full-scale grid | float32 Frame to or from uint Frame |
| Named format | `px.io.from_<format>` / `px.io.to_<format>` | Effective code bits carried by the format | Packing, subsampling, and container resolved by the format contract |
| General array boundary | `px.io.from_array` / `px.io.to_array` | Effective code bits for an unsigned full-scale grid in a raw array | Composes orthogonally with layout, channel selection, and `out=` |

## interpolation

Interpolation tokens form the shared vocabulary for interpolation and resampling kernels. Every API fixes its accepted
subset and coordinate rules.

- `px.io.from_uyvy422`, `px.io.from_v210`, `px.io.from_nv12`, `px.io.from_p010`, `px.io.from_yuv420p`, and
  `px.io.from_yuv422p` accept the first eight tokens in the table, excluding `area`; default `bilinear`.
- Chroma upsampling on a `from_` path places chroma samples at the frame coordinates defined in chroma siting and
  evaluates filter weights from each luma-sample coordinate. Edges replicate the final chroma row or column.
- `px.io.to_uyvy422`, `px.io.to_v210`, `px.io.to_nv12`, `px.io.to_p010`, `px.io.to_yuv420p`, and
  `px.io.to_yuv422p` accept `nearest`, `bilinear`, `bicubic`, and `area`; default `area`.
- Chroma downsampling on a `to_` path centers each output sample at the siting offset. Bilinear and bicubic use a
  scale-two reduction kernel; area averages coverage over the owned interval. Edges replicate.
- 4:2:2 filters horizontally and reads the original row directly vertically. Equidistant nearest ties round half up.
- `px.transform.resize` accepts the first nine table tokens. When omitted, it selects `area` if either dimension
  shrinks, and `lanczos4` otherwise.
- `px.transform.warp_affine` accepts the first nine table tokens. When omitted, it inspects the column norms of the
  effective forward 2×2 matrix: either norm below 1 selects `area`; both norms at least 1 select `lanczos4`.
- `px.composite.merge` accepts the first eight table tokens, excluding `area`; default `bilinear`. Each background
  pixel center maps inversely into foreground coordinates.
- Outside foreground support, merge samples transparent zero. It does not replicate edges or renormalize weights after
  discarding taps outside support.
- `px.color.apply_lut` chooses a subset by LUT type. A `Lut` accepts `trilinear` and `tetrahedral`, with default
  `tetrahedral`. A `Lut1D` accepts only `linear`, with default `linear`. `None` selects that type-specific default and
  never converts a token from the other subset.
- Input RGB maps through the LUT's per-channel declared `domain` to grid or curve coordinates. Lookup coordinates
  outside the domain are clamped to its edge. LUT output is not clipped; negative values and values above 1 are
  returned unchanged.

Point-sampled `px.transform.resize` kernels share pixel-center alignment
`src = (dst + 0.5) × (input / output) - 0.5` and edge replication. Channels are independent, and neither scene values
nor kernel overshoot or undershoot is clipped. Each factor-derived output dimension is
`floor(dim × factor + 0.5)`.

`px.transform.warp_affine` places the top-left input and output pixel centers at `(0, 0)` and inverse-maps every output
center through the caller-declared forward matrix. Nearest uses `floor(coordinate + 0.5)`. Bilinear, cubic, and Lanczos
use fixed-support separable kernels; Lanczos normalizes by the sum of all x- and y-direction tap weights. `area` is the
full area average of intersections between the inverse-mapped output-pixel parallelogram and unit input-pixel cells.
Neither point nor area renormalizes only in-canvas taps; both apply the border section's infinite-plane extension.

| Token | Definition | Standard or convention | Notes |
|---|---|---|---|
| `nearest` | Nearest sample | Nearest-neighbor interpolation | Equidistant ties round half up, on both format-input paths and resize |
| `bilinear` | 2×2 linear interpolation | Bilinear interpolation | Format-input paths align siting offsets; resize aligns pixel centers |
| `bicubic` | 4×4 support with Keys cubic, `a = -0.5` | Mathematically identical to Catmull-Rom | Interpolating kernel |
| `b-spline` | 4×4 support in the Mitchell family, `B = 1, C = 0` | Cubic B-spline | Approximating kernel; smooths even at unchanged size |
| `mitchell` | 4×4 support in the Mitchell family, `B = 1/3, C = 1/3` | Mitchell-Netravali | Approximating kernel; smooths even at unchanged size |
| `lanczos2` | `sinc(x) sinc(x / 2)`, two lobes | Windowed sinc | Normalize by the weight sum inside support |
| `lanczos3` | `sinc(x) sinc(x / 3)`, three lobes | Windowed sinc | Normalize by the weight sum inside support |
| `lanczos4` | `sinc(x) sinc(x / 4)`, four lobes | Windowed sinc | Normalize by the weight sum inside support |
| `area` | Box average over the source region covered by each output pixel | Area or box resampling | The same definition also applies to enlargement |
| `trilinear` | Linear interpolation along three axes over the eight vertices of a 3D grid cell | Trilinear interpolation | Subset exclusive to `px.color.apply_lut` |
| `tetrahedral` | Linear interpolation after splitting a 3D grid cell into six tetrahedra | Tetrahedral interpolation | Subset exclusive to `px.color.apply_lut`; default |
| `linear` | Linear interpolation between adjacent samples of each independent RGB curve | One-dimensional LUT interpolation | Subset exclusive to `px.color.apply_lut` with `Lut1D`; default for that type |

The cubic and Lanczos support widths in `px.transform.resize` do not expand on the source side during reduction; these
kernels provide no scale-aware antialiasing. Use `area` for antialiased reduction.

## stack direction

`px.transform.stack` direction tokens are case-sensitive; default `vertical`. Input order is preserved, and output
always owns new storage.

| Token | Concatenation rule |
|---|---|
| `vertical` | Arrange top to bottom; output height is the sum of input heights and width is common |
| `horizontal` | Arrange left to right; output width is the sum of input widths and height is common |

## sobel direction

`px.filter.sobel` direction tokens are case-sensitive; default `magnitude`. Each channel is processed independently,
without dispatch based on channel labels.

| Token | Definition |
|---|---|
| `x` | First derivative in the horizontal direction; responds to vertical edges |
| `y` | First derivative in the vertical direction; responds to horizontal edges |
| `magnitude` | Per-channel gradient magnitude `sqrt(x² + y²)`; default |

## template matching method

`px.feature.match_template` method tokens are case-sensitive; default `ccoeff_normed`. Sums over window `I` and
template `T` include spatial dimensions and every channel. In ccoeff methods, `mean_I[c]` and `mean_T[c]` are spatial
arithmetic means per channel; channels are not combined into one mean.

| Token | Response | Direction of better match |
|---|---|---|
| `sqdiff` | `sum((I - T)²)` | Lower is better; an exact match is 0 |
| `sqdiff_normed` | `sum((I - T)²) / sqrt(sum(I²) × sum(T²))` | Lower is better |
| `ccorr` | `sum(I × T)` | Higher is better |
| `ccorr_normed` | `sum(I × T) / sqrt(sum(I²) × sum(T²))` | Higher is better |
| `ccoeff` | `sum((I - mean_I[c]) × (T - mean_T[c]))` | Higher is better |
| `ccoeff_normed` | `ccoeff / sqrt(sum((I - mean_I[c])²) × sum((T - mean_T[c])²))` | Higher is better; default |

When a normalized method's denominator is zero, `sqdiff_normed` returns 0 if its numerator is also zero and `+inf` if
the numerator is positive. `ccorr_normed` and `ccoeff_normed` return 0. No epsilon or result clipping is introduced.

## chroma siting

Chroma-siting tokens define the centers of chroma samples in progressive 4:2:0 input, using frame coordinates where
the top-left luma-sample center is `(0, 0)` and luma spacing is 1. They appear only on `px.io.from_nv12`,
`px.io.from_p010`, `px.io.from_yuv420p`, `px.io.to_nv12`, `px.io.to_p010`, and `px.io.to_yuv420p`; default `left`.
Siting is not inferred from colorimetry. State `topleft` explicitly for BT.2020 or BT.2100 material that uses the
standard position.

| Token | Offset `(x, y)` | H.273 | Definition |
|---|---:|---|---|
| `left` | `(0, 0.5)` | H.273 type 0 | Horizontally co-sited and vertically interstitial; typical BT.601/BT.709 SDR delivery convention |
| `center` | `(0.5, 0.5)` | H.273 type 1 | Geometric center of the 2×2 luma block |
| `topleft` | `(0, 0)` | H.273 type 2 | Co-sited on both axes; standard BT.2020/BT.2100 position |

4:2:2 (`px.io.from_uyvy422`, `px.io.from_v210`, and `px.io.from_yuv422p`) is fixed as horizontally co-sited and
vertically full resolution, so it has no siting argument. 4:4:4 (`px.io.from_yuv444p` and `px.io.from_yuva444p`) has no
subsampling and therefore has neither siting nor interpolation arguments.

## draw continuous coordinates

Drawing APIs use coordinate pairs in `(x, y)` order: x follows columns and y follows rows. `(0, 0)` is the top-left
corner of the top-left pixel, and the center of pixel at column `i`, row `j` is `(i + 0.5, j + 0.5)`. Coordinates and
dimensions accept finite real numbers and preserve subpixel positions. Negative and off-image coordinates are valid;
only the intersection with the image is drawn.

## language

Text-language tokens select the OpenType shaping language and therefore the CJK glyph forms provided by the selected
bundled font's `locl` feature. The default for `px.draw.text` is `ja`. Tokens are case-sensitive.

| Token | Definition |
|---|---|
| `ja` | Japanese glyph forms |
| `zh-hans` | Simplified Chinese glyph forms |
| `zh-hant` | Traditional Chinese glyph forms |
| `ko` | Korean glyph forms |

## anchor

Text-anchor tokens define which point on the text box `position` identifies, in `<vertical>-<horizontal>` order.
Default `baseline-left`. `px.draw.text` defines horizontal positions from block width and vertical positions from the
first-line ascender, first-line baseline, and final-line descender. This is independent of actual ink, outline,
bearing, or glyph overhang for both single-line and multiline text.

| Token | Definition |
|---|---|
| `top-left` | First-line ascender and left edge of the block box |
| `top-center` | First-line ascender and horizontal midpoint of the block box |
| `top-right` | First-line ascender and right edge of the block box |
| `center-left` | Midpoint between top and bottom, and left edge of the block box |
| `center-center` | Midpoint between top and bottom, and horizontal midpoint of the block box |
| `center-right` | Midpoint between top and bottom, and right edge of the block box |
| `baseline-left` | First-line baseline and left edge of the block box |
| `baseline-center` | First-line baseline and horizontal midpoint of the block box |
| `baseline-right` | First-line baseline and right edge of the block box |
| `bottom-left` | Final-line descender and left edge of the block box |
| `bottom-center` | Final-line descender and horizontal midpoint of the block box |
| `bottom-right` | Final-line descender and right edge of the block box |

## text placement

`px.draw.text` splits text exactly like `str.split("\n")`, preserving empty lines from consecutive or trailing
newlines. Carriage returns are rejected rather than normalized. Each nonempty line is independently shaped left to
right with OpenType GSUB and GPOS in the selected font; shaping never spans lines. Advances and offsets returned by
the shaper accumulate at subpixel precision.

## text align

Text-align tokens choose each line's pen origin relative to the left edge of the `px.draw.text` block. Default `left`.
Tokens are case-sensitive.

| Token | Line placement |
|---|---|
| `left` | Start at the block's left edge |
| `center` | Start at `(block width - line advance) / 2` |
| `right` | Start at `block width - line advance` |
| `justify` | Start at the block's left edge and distribute only positive remaining width evenly between shaped glyphs |

## text font

`px.draw.text` accepts either a text-font token or an immutable `px.draw.Font`. Default `sans`. Tokens select bundled
package data and retain their string cache identities. `px.draw.Font.from_file(path, face_index=0)` accepts a regular
font file without an extension whitelist, reads its bytes completely during construction, and validates the selected
face with both FreeType and HarfBuzz. The resulting asset remains usable after the source file is changed or removed.
Its shaping, FreeType face, glyph, layout, and atlas cache identity is the content bytes plus face index, not the path
or Python object identity.

| Token | Bundled font | Accepted `wght` range |
|---|---|---:|
| `sans` | Noto Sans CJK JP variable | 100.0 through 900.0 |
| `mono` | Noto Sans Mono CJK JP variable | Measured 400.0 through 700.0 |

For a user `Font` with a measured `wght` axis, `weight` accepts finite values in that axis's closed range. A static
font has no `wght` axis and therefore accepts only the effective default `weight=400.0`. `variations` is `None` or a
partial mapping from case-sensitive OpenType axis tags to finite values in each measured closed range. Unspecified
axes use their measured defaults. The `wght` key is rejected in `variations`; pass that value through `weight`
instead. Unknown axes, non-real or non-finite values, and out-of-range values fail before shaping without saturation.
The complete resolved axis coordinates are applied identically to HarfBuzz shaping and FreeType metrics and raster.

Missing code points use glyph 0 (`.notdef`) from the selected face. There is no fallback search across bundled fonts,
other user fonts, system-font registries, or the network. `Font.from_file` never performs system-font discovery or
network access.

## text block layout

`line_spacing` in `px.draw.text` is a positive finite multiplier on the selected font's default OpenType horizontal
line advance, `(ascender - descender + lineGap) × size / unitsPerEm`; default `1.0`. Empty lines receive the same
baseline interval. `tracking` is a finite em ratio; default `0.0`. After shaping, `tracking × size` pixels are added
between adjacent shaped glyphs. Negative tracking and overlap are permitted. `kerning=False` disables only the
OpenType `kern` feature, preserving `liga` and `locl`.

With `width=None`, block width is the maximum of zero and every signed line advance. An explicit width is a finite
nonnegative pixel value and fixes the basis for alignment, justification, and horizontal anchors. Justification first
applies tracking, then adds positive remaining width to glyph gaps; it never contracts negative remaining width. A
line wider than the block is not clipped, shrunk, wrapped, or rejected. It overflows the box from its aligned origin,
and only its intersection with the image is drawn.

Horizontal block anchors use block width. Vertical top, baseline, bottom, and center refer to the first-line ascender,
first-line baseline, final-line descender, and midpoint of top and bottom. Line spacing, tracking, justification gaps,
baselines, and pen origins retain subpixel precision equivalent to 26.6 fixed point. `font="sans"` accepts weight
100.0 through 900.0; `font="mono"` accepts the measured range 400.0 through 700.0. Values outside the ranges are not
saturated.

## blend

Blend tokens choose a separable blend function `B(Cb, Cs)` for each corresponding color channel. `Cb` is unassociated
background color and `Cs` is unassociated source color. `px.composite.merge` accepts all ten tokens below. Drawing
shapes and text accept the `normal`, `add`, `multiply`, and `screen` subset. Both families default to `normal`. The
former token `over` is rejected. Every expression is evaluated in fp32 without clamping intermediate values, results,
negative values, or values above 1.

| Token | `B(Cb, Cs)` |
|---|---|
| `normal` | `Cs` |
| `lighten` | `max(Cb, Cs)` |
| `add` | `Cb + Cs` |
| `screen` | `1 - (1 - Cb) × (1 - Cs)` |
| `darken` | `min(Cb, Cs)` |
| `multiply` | `Cb × Cs` |
| `difference` | `abs(Cb - Cs)` |
| `overlay` | `2 × Cb × Cs` when `Cb <= 0.5`; otherwise `1 - 2 × (1 - Cb) × (1 - Cs)` |
| `hardlight` | `2 × Cb × Cs` when `Cs <= 0.5`; otherwise `1 - 2 × (1 - Cb) × (1 - Cs)` |
| `softlight` | `Cb - (1 - 2 × Cs) × Cb × (1 - Cb)` when `Cs <= 0.5`; otherwise `Cb + (2 × Cs - 1) × (D(Cb) - Cb)` |

For softlight, `D(Cb) = ((16 × Cb - 12) × Cb + 4) × Cb` when `Cb <= 0.25`, and `D(Cb) = sqrt(Cb)` otherwise.
Drawing operations use coverage times opacity as `a` and evaluate
`out = dst × (1 - a) + B(dst, color) × a`. `px.composite.merge` constructs `a` from source alpha, mask, and opacity,
then passes the blend result into source-over composition.

## alpha

Alpha tokens jointly declare stored color representation for the background and foreground of `px.composite.merge`
and for output when the background has `A`. Default `premultiplied`. Frame metadata does not store alpha state. A
foreground without `A` receives implicit alpha coverage of 1 inside its support and 0 outside. Alpha, mask, and
composition results are not clamped.

| Token | Definition |
|---|---|
| `premultiplied` | Color channels have already been multiplied by the same pixel's `A`; unassociated color at alpha 0 is defined as 0 |
| `straight` | Color channels have not been multiplied by `A`; foreground color is associated with `A` before interpolation |

## aa

Antialiasing tokens for drawing shapes choose how geometric coverage is computed. Default `distance`. `softness` is a
nonnegative edge-feather width that expands the transition around `distance` and `supersample` boundaries. Because
`off` produces binary coverage, it cannot be combined with positive softness.

Grid and checkerboard generation share the same three tokens and coverage definitions, also defaulting to `distance`.
Generators expose no `softness` argument: `distance` transitions over about one pixel, `supersample` uses fixed 4×4
samples, and `off` makes a binary pixel-center decision.

| Token | Coverage |
|---|---|
| `distance` | Continuous transition about one pixel wide from signed boundary distance, expanded by `softness` |
| `supersample` | Average fixed 4×4 samples in each pixel; with `softness=0`, produces 17 values from 0/16 through 16/16 |
| `off` | 1 when the pixel center is inside, otherwise 0 |

`px.draw.text(supersample=True)` also averages fixed 4×4 samples in fp32, matching the shape `supersample` token.
However, the public text API is `supersample: bool = False`; it accepts neither the aa token vocabulary nor an `aa`
argument.

## generator kind

Ramp kind tokens determine the interpolation coordinate between two colors. Default `linear`. Interpolation occurs
directly in the declared gamma value space, with no linearization or value-range clamp.

| Token | Definition |
|---|---|
| `linear` | Use vector projection from start toward end as the interpolation factor, saturating regions beyond either endpoint to the endpoint color |
| `radial` | Use start as center, distance from start to end as radius, and distance from center as the interpolation factor |

## color bars standard

Standard tokens jointly select pattern geometry, 10-bit code values, colorspace, gamma, and normalization for the
normalized output. Bar channels are always `RGB`, and region edges have no antialiasing.

| Token | Standard and pattern | Colorspace | Gamma | Code range |
|---|---|---|---|---|
| `arib-std-b28` | ARIB STD-B28 multiformat color bar | `Rec.709` | `rec709` | narrow |
| `smpte-rp219` | SMPTE RP 219-1 basic pattern; identical output to ARIB STD-B28 | `Rec.709` | `rec709` | narrow |
| `bt2111-hlg` | ITU-R BT.2111-2 HLG pattern | `Rec.2020` | `hlg` | narrow |
| `bt2111-pq` | ITU-R BT.2111-2 PQ pattern | `Rec.2020` | `pq` | narrow |
| `bt2111-pq-full` | ITU-R BT.2111-2 PQ full-range pattern | `Rec.2020` | `pq` | full |
| `full-100` | ITU-R BT.471 100/0/100/0 full-field bars with BT.1729 BT.709 code values | `Rec.709` | `rec709` | narrow |
| `full-75` | ITU-R BT.471 100/0/75/0 full-field bars, retaining 100% white | `Rec.709` | `rec709` | narrow |

Normalized output for a narrow standard applies `(code - 64) / 876` to every RGB code and preserves negative
sub-black regions such as PLUGE. `bt2111-pq-full` uses `code / 1023`. Region boundaries scale proportionally from the
standard's reference-resolution ratios to output dimensions and round to nearest integer pixel boundaries. A region
that rounds to zero pixels in a very small image may disappear.

## color bars output

Output tokens select storage scale and dtype for the same standard pattern. Default `normalized`. Frame metadata
follows the standard-token table for either output; the Frame does not store code scale or bit depth as metadata.

| Token | Data |
|---|---|
| `normalized` | Normalize standard codes to full-range `float32` with the formulas above |
| `code` | Store the standard's 10-bit code values directly in `uint16` |

## morphology shape

Morphology-shape tokens define structuring-element support as integer pixel offsets. `radius` is an integer at least 1,
and support always contains the center pixel. Raw kernel arrays and iterations are not accepted. Tokens are
case-sensitive.

| Token | Support set | Width and height |
|---|---|---|
| `disk` | Integer offsets `(dx, dy)` from center satisfying `dx² + dy² <= radius²`; default | `2 × radius + 1` |
| `square` | Integer offsets with Chebyshev distance `max(abs(dx), abs(dy)) <= radius` | `2 × radius + 1` |

`px.morphology.erosion` takes the per-channel minimum over support; `px.morphology.dilation` takes the maximum.
`px.morphology.opening` erodes then dilates, and `px.morphology.closing` dilates then erodes with the same element.
`px.morphology.morphological_gradient` is dilation minus erosion. `px.morphology.white_tophat` subtracts opening from
the input to extract small bright details. `px.morphology.black_tophat` subtracts the input from closing to extract
small dark details.

## border

Border tokens define virtual pixels when a neighborhood operation reads beyond an image. They do not change output
dimensions and are case-sensitive. If an image dimension is one pixel, every border reference on that axis maps to
the single pixel.

Blur APIs default to `mirror` and include `px.filter.gaussian_blur`, `px.filter.unsharp_mask`, `px.filter.box_blur`,
`px.filter.median_blur`, `px.filter.bilateral_blur`, `px.filter.convolve_box`, `px.filter.directional_blur`,
`px.filter.zoom_blur`, `px.filter.spin_blur`, `px.filter.vector_blur`, and `px.filter.lens_blur`.
Derivative, sharpening, and edge-detection filters (`px.filter.sobel`, `px.filter.laplacian`,
`px.filter.difference_of_gaussians`, `px.filter.sharpen`, and `px.filter.canny`) also default to `mirror` and accept the
same four tokens. Canny applies the same border and `border_value` to both source reads in Sobel and magnitude-neighbor
reads in NMS. Hysteresis traverses eight-connectivity only within the real image, never virtual border pixels or
opposite-edge connections. `px.feature.corner_harris` accepts the same four tokens with default `mirror`, applying one
input-extension rule consistently to the Sobel gradient stage and structure-tensor aggregation window; it does not
reapply a pixel `border_value` to an already derived tensor.

`px.transform.warp_affine` accepts the same four tokens with default `constant`. As a warp-specific exception,
`border_value=None` resolves to effective 0.0 shared by all channels. Point kernels apply the infinite-plane extension
to every support tap; `area` applies it to every input-pixel cell intersecting the parallelogram. Canvas dimensions do
not change.

Morphology operations default to `replicate`: `px.morphology.erosion`, `px.morphology.dilation`,
`px.morphology.opening`, `px.morphology.closing`, `px.morphology.morphological_gradient`,
`px.morphology.white_tophat`, and `px.morphology.black_tophat`. Unlike blur, replicate is the neutral boundary for a
uniform surface under min and max.

| Token | Mathematical definition | `np.pad` | `scipy.ndimage` | `cv2` |
|---|---|---|---|---|
| `mirror` | Reflection without repeating the edge pixel; reflected indices have period `2n - 2` | `reflect` | `mirror` | `BORDER_REFLECT_101` |
| `replicate` | Edge clamp that repeats the edge pixel | `edge` | `nearest` | `BORDER_REPLICATE` |
| `wrap` | Periodic indices modulo image size, retaining the same modulo definition even when the kernel is wider than the image | `wrap` | `grid-wrap` | `BORDER_WRAP` |
| `constant` | Every virtual pixel outside the image equals `border_value` | `constant` | `constant` | `BORDER_CONSTANT` |

Except for `px.transform.warp_affine`, `constant` requires a finite real `border_value`; negative values and values
above 1 are valid, while `bool` is not treated as real. Every API rejects `border_value` with another border token.
Both violations fail immediately. The rule corresponds to
`np.pad(mode="constant", constant_values=border_value)`, with one value shared by all channels.

The name `reflect` describes different behavior in `np.pad` and `scipy.ndimage`.
`np.pad(mode="reflect")` does not repeat edge pixels and matches pixtreme `mirror`; `scipy.ndimage` `reflect` is
half-sample symmetric and repeats edge pixels. The SciPy token corresponding to pixtreme `mirror` is `mirror`.

## vector blur shutter

Vector-blur shutter tokens select the sampling interval along a motion vector. Default `centered`. Tokens are
case-sensitive.

| Token | Sampling interval |
|---|---|
| `centered` | From `-0.5` through `+0.5` of the motion vector, centered on the current position |
| `forward` | From `0` through `+1` of the motion vector, starting at the current position |
| `backward` | From `-1` through `0` of the motion vector, ending at the current position |

## from_<format> conventions

The eight `px.io.from_<format>` functions resolve uint-code packing, subsampling, and range at input, returning a
C-contiguous float32 YCbCr444 Frame (`px.io.from_yuva444p` alone returns YCbCrA4444). Only `buf` is positional; width
and height are required keyword-only arguments. The following are specification defaults for a boundary whose input
buffer contains no metadata, not guaranteed truths about the material.

| Item | Specification default | Notes |
|---|---|---|
| colorspace | `Rec.709` | Placeholder; an explicit per-call `colorspace=` token takes precedence |
| gamma | `rec709` | Placeholder; an explicit per-call `gamma=` token takes precedence |
| matrix | `None` | Unknown provenance; an explicit per-call `matrix=` token is stamped literally |
| channels | `("Y", "Cb", "Cr")` | Fixed channel order after format resolution |
| range | `legal` | Default assumption for video-family YCbCr input; override per call with `range="full"` |
| interpolation | `bilinear` | Default for the six subsampled formats; accepts the first eight interpolation tokens |
| siting | `left` | Present only on the three 4:2:0 formats; accepts the three chroma-siting tokens |

`colorspace=`, `gamma=`, and `matrix=` are per-call metadata claims. Colorspace and gamma priority is
**explicit per-call value > placeholder**; omitted matrix is `None`. If only colorspace or gamma is explicit, the other
remains its placeholder. Passing `None` for either has the same effect as omission. Explicit correction by assigning an
attribute on the returned Frame remains available. Both mechanisms correct metadata rather than transform pixel
values.

Planar formats accept these `bit_depth` values and container dtypes. Ten- and twelve-bit planar data is packed into the
low bits of uint16; nonsignal high bits are ignored on read.

| Format | bit_depth | Container dtype | Plane order |
|---|---|---|---|
| `yuv420p` | 8 (default) / 10 | 8 = uint8, 10 = uint16 | Y, then Cb, then Cr; each chroma plane is H/2 × W/2 |
| `yuv422p` | 8 (default) / 10 / 12 | 8 = uint8, 10 / 12 = uint16 | Y, then Cb, then Cr; each chroma plane is H × W/2 |
| `yuv444p` | 10 (default) / 12 | uint16 | Y, then Cb, then Cr; each plane is H × W |
| `yuva444p` | 12 (default) | uint16 | Y, then Cb, then Cr, then A; each plane is H × W, and A is full-scale regardless of range |

Packed and semiplanar formats use these conventions:

| Format | Container dtype | C-contiguous 1D layout |
|---|---|---|
| `uyvy422` | uint8 | U0 Y0 V0 Y1; input also accepts NDI shape `(H, W, 2)`, which can reshape to 1D as a zero-copy view |
| `v210` | uint32 | Six pixels in four words, with three 10-bit samples from the low bits of each word; rows align to 128 bytes, or 48 pixels, with zero padding |
| `NV12` | uint8 | Y plane followed by an interleaved Cb Cr plane |
| `P010` | uint16 | Same arrangement as NV12; 10-bit codes are MSB-aligned and the lower six bits are zero |

`range="legal"` maps Y, Cb, and Cr to full-range float using the general H.273 formula for the bit depth and does not
clip code headroom. `range="full"` uses `code / (2^n - 1)`. YUVA alpha always uses `code / (2^n - 1)` independently
of the range token.

## to_<format> conventions

The eight `px.io.to_<format>` functions derive output dimensions from the width and height of the Frame passed as the
first positional argument, resolving packing, subsampling, and range in one CUDA pass. Input must be a float32 Frame
with channels `("Y", "Cb", "Cr")`; only `px.io.to_yuva444p` accepts `("Y", "Cb", "Cr", "A")`. Convert RGB Frames
explicitly with `px.color.rgb_to_ycbcr` before passing them.

Every function returns a newly allocated C-contiguous 1D `cupy.ndarray`. There are no width or height arguments. 4:2:0
requires even width and height; ordinary 4:2:2 requires even width. v210 alone accepts any positive width and aligns
row storage to 128 bytes.

| Item | Specification default | Notes |
|---|---|---|
| range | `legal` | Also accepts `full`; legal placement preserves headroom codes without clipping to the legal interval |
| interpolation | `area` | Default for the six subsampled formats; accepts nearest, bilinear, bicubic, and area |
| siting | `left` | Present only on the three 4:2:0 formats; accepts the three chroma-siting tokens |
| rounding | Half away from zero | Nearest rounding from fp32 to code |
| clipping | Container range only | Do not clip to the legal interval; clip only to physical `[0, 2^n - 1]` |

Planar `bit_depth`, container dtype, and plane order are symmetric with the input path.

| Format | bit_depth | Container dtype | C-contiguous 1D layout |
|---|---|---|---|
| `yuv420p` | 8 (default) / 10 | 8 = uint8, 10 = uint16 | Y, then Cb, then Cr; each chroma plane is H/2 × W/2 |
| `yuv422p` | 8 (default) / 10 / 12 | 8 = uint8, 10 / 12 = uint16 | Y, then Cb, then Cr; each chroma plane is H × W/2 |
| `yuv444p` | 10 (default) / 12 | uint16 | Y, then Cb, then Cr; each plane is H × W |
| `yuva444p` | 12 (default) | uint16 | Y, then Cb, then Cr, then A; A is full-scale regardless of range |
| `uyvy422` | Fixed 8 | uint8 | U0 Y0 V0 Y1; reshape to `(H, W, 2)` is a zero-copy view |
| `v210` | Fixed 10 | uint32 | Six pixels in four words; the function zero-fills 128-byte row padding |
| `NV12` | Fixed 8 | uint8 | Y plane followed by an interleaved Cb Cr plane |
| `P010` | Fixed 10 | uint16 | Same arrangement as NV12; MSB-aligned with the lower six bits zero |

Range mapping composes inversely with the input path. Legal range uses an extent of `219 × 2^(n-8)` for Y and
`224 × 2^(n-8)` for Cb and Cr, with lower code `16 × 2^(n-8)`. Full range scales every component by `2^n - 1`.
YUVA alpha is always full-scale independently of the range token.

## dtype

Dtype tokens describe only the storage representation of Frame data. Never infer a pixel-value scale from dtype;
state it with `bit_depth` on `px.values.quantize`, `px.values.dequantize`, or a general array boundary.

| Token | Definition | Standard or convention | Notes |
|---|---|---|---|
| `float32` | IEEE 754 binary32 working storage | NumPy dtype name | Baseline dtype for processing and transforms |
| `float16` | IEEE 754 binary16 transport and storage | NumPy dtype name | Permitted storage; cast explicitly to float32 before processing |
| `uint8` | 8-bit unsigned integer code storage | NumPy dtype name | Full-range maximum 255 |
| `uint16` | 16-bit unsigned integer code storage | NumPy dtype name | Full-range maximum 65,535 |
| `uint32` | 32-bit unsigned integer code or ID storage | NumPy dtype name | Full-range maximum 4,294,967,295; permitted transport and storage |

`uint32` can preserve and transport integer identities such as object IDs bit-exactly. Numeric conversion to float32,
whether normalized or not, cannot represent every integer above `2^24` exactly and may round to the nearest binary32
value; this is documented loss. Processing and transform operations accept only float32 and never promote uint32
implicitly.

## dtype operation comparison

`px.values.cast_dtype` is a literal cast selected by a dtype token. `px.values.recode_dtype` preserves meaning relative
to container full scale. The quantization pair changes pixel-value scale according to bit depth. `recode_dtype` accepts
the same five dtype tokens (`float32`, `float16`, `uint8`, `uint16`, `uint32`) and implements all 25 source/destination
pairs, including identical dtype. All operations preserve metadata and always return a new GPU allocation.

| API | Preserved property | uint↔float behavior | Primary use |
|---|---|---|---|
| `px.values.cast_dtype` | Numeric value | Faithful delegation to CuPy `astype`; no scaling, clipping, or explicit rounding | Change the container of depth, label, or other raw values read unchanged |
| `px.values.recode_dtype` | Meaning | Normalize uint by container maximum; clip float to `[0, 1]`, scale to full range, and round half away from zero for float to uint; literal cast between floats | Convert between ordinary uint images and normalized float Frames |
| `px.values.quantize` | Pixel-value scale | Clip and scale float32 to the uint full-scale grid at the declared bit depth, then round half away from zero | Produce a code-value Frame from normalized values |
| `px.values.dequantize` | Pixel-value scale | Normalize uint codes at the declared bit depth by maximum code without clipping | Return a code-value Frame to float32 working values |

## image read conventions

`px.io.read_image` identifies extensions case-insensitively and supports JPEG, PNG, TIFF, JPEG 2000, WebP, BMP, PNM,
TGA, HDR, DPX, and EXR. `px.io.decode_image` detects JPEG, PNG, TIFF, JPEG 2000, WebP, BMP, and PNM from encoded-byte
signatures; it does not support TGA, HDR, DPX, or EXR. Both APIs use the immutable specification defaults below.

A standard read normalizes ordinary uint by container maximum and decodes EXR HALF and RGBE into float32 Frames. EXR
UINT alone converts literal unnormalized integers numerically to float32 and permits documented loss above `2^24`.
`unchanged=True` preserves the input's self-declared storage dtype (`uint8`, `uint16`, `uint32`, `float16`, or
`float32`), including exact EXR UINT sample bits. HDR whose native representation is fp32-equivalent returns the same
values and dtype as a default read. EXR and HDR are file-only boundaries.

| Format | Default colorspace | Default gamma | `channels=None` |
|---|---|---|---|
| PNG / JPEG / TIFF | `sRGB` | `srgb` | RGB or RGBA; grayscale is one-channel `("Y",)` |
| JPEG 2000 | `sRGB` | `srgb` | Y, RGB, or RGBA |
| WebP | `sRGB` | `srgb` | RGB |
| BMP / PNM | `sRGB` | `srgb` | Y or RGB |
| TGA | `sRGB` | `srgb` | RGB or RGBA |
| HDR | `Rec.709` | `linear` | RGB |
| DPX | `Rec.709` | Header transfer; unknown maps to `cineon` at 10 bit, `rec709` at 8 bit, and `linear` at 12 or 16 bit | RGB or RGBA |
| EXR | `ACES2065-1` | `linear` | R, G, B, and A when present |

Metadata priority is **explicit per-call value > explicit file value > specification default**. Explicit file values
also include metadata inside encoded bytes accepted by `px.io.decode_image`. Per-call `colorspace=` and `gamma=` are
metadata claims and do not transform pixel values. File metadata is limited to PNG cICP, sRGB, and gAMA chunks, plus
EXR chromaticities and the ACES container flag. HDR EXPOSURE, PRIMARIES, and COLORCORR can be inspected as raw header
values but are applied to neither pixels nor metadata. DPX maps printing-density, logarithmic, and ADX transfer
characteristics to `cineon`; linear to `linear`; and video-family characteristics to `rec709`. ICC profiles are not
read. If a file value cannot map into the public vocabulary, pixtreme falls back to the specification default and emits
a Python warning.

`px.io.read_header` performs no pixel decoding or GPU allocation. It returns an `ImageHeader` with format, dimensions,
stored channel dtype per part, and raw values, mapped tokens, and mapping availability for the file color information
above. Each `ImageHeader.parts[]` entry has a per-part `deep: bool`, allowing flat/deep classification before pixel
decode.

## image write dtype

`px.io.write_image` and `px.io.encode_image` accept all five Frame storage dtypes (`uint8`, `uint16`, `uint32`,
`float16`, `float32`) for every supported format. Native dtypes pass through unchanged; nonnative dtypes convert on the
GPU to the default container below. Input data, dtype, and metadata are unchanged. Conversion matches
`px.values.recode_dtype`: uint scales by container full range, uint-to-float normalizes by container maximum, and
float-to-uint clips to `[0, 1]`, scales to full range, and rounds half away from zero.

| Format | Native dtype | Default container for nonnative input |
|---|---|---|
| PNG / TIFF / JPEG 2000 / PNM | `uint8` / `uint16` | `uint8` |
| JPEG / WebP / BMP | `uint8` | `uint8` |
| EXR | `float16` / `float32` / `uint32` | Normally `float16`; `uint32` for a uint32 Frame |
| TGA | `uint8` | `uint8` |
| HDR | `float32` | `float32` |
| DPX | `float32` | `float32` |

Dtype conversion does not expand each format's closed channel-layout or encode-parameter set. For EXR, `dtype=None`
selects HALF (`float16`) normally and native UINT (`uint32`) only for a uint32 Frame. Explicit `dtype=` overrides the
Frame-dependent default. Conversion among `float16`, `float32`, and `uint32` is numerically equivalent to
`recode_dtype`. EXR, TGA, HDR, and DPX are file-only boundaries. TGA, HDR, and DPX convert every nonnative dtype,
including uint32, to uint8, float32, and float32 respectively with `recode_dtype` full-scale semantics. HDR requires
exactly one R, G, and B channel.

## image encode kwargs

`format` is a required keyword-only token for `px.io.encode_image`; `px.io.write_image` derives format from its file
extension. Encode parameters are named keywords only, never a magic integer list. EXR is file-only, so EXR compression
and `dwa_level` exist only on `px.io.write_image`. Omission (`None`) uses the specification or codec default below.

| Kwarg | API and target format | Value domain | Meaning |
|---|---|---|---|
| `quality` | Both APIs; JPEG and WebP | Integer `1` through `100` | Lossy quality; specifying it for JPEG 2000, PNG, TIFF, BMP, PNM, or EXR raises `ValueError` |
| `compression` | Both APIs; TIFF | Token `none` or `lzw` | TIFF uncompressed or lossless LZW compression |
| `compression` | `px.io.write_image`; EXR | EXR compression token | Default `zip`; distinct from TIFF tokens |
| `compression_level` | Both APIs; PNG | Integer `0` through `9` | PNG zlib compression level; specifying it for another format raises `ValueError` |
| `lossless` | Both APIs; JPEG 2000 and WebP | Exact `bool` or `None` | `True` is lossless, `False` lossy, and `None` the codec default; WebP `quality` conflicts with `True` |
| `dwa_level` | `px.io.write_image`; EXR DWAA and DWAB | Positive finite exact `float` or `None`, including as a header float | `None` means `45.0`; specifying it for non-DWA compression raises `ValueError` |
| `bit_depth` | `px.io.write_image`; DPX | Integer `8`, `10`, `12`, `16`, or `None` | `None` means 10 bit; specifying it for non-DPX output raises `ValueError` |
| `dtype` | `px.io.write_image`; EXR | `float16`, `float32`, `uint32`, or `None` | An explicit value overrides the Frame-dependent default; specifying it for non-EXR output raises `ValueError` |

## image format

Case-sensitive closed tokens accepted by `px.io.encode_image(format=...)`. They are not file extensions.

| Token | Meaning |
|---|---|
| `jpeg` | JPEG encoded bytes |
| `png` | PNG encoded bytes |
| `tiff` | TIFF encoded bytes |
| `jpeg2000` | JPEG 2000 encoded bytes in a JP2 container |
| `webp` | WebP encoded bytes |
| `bmp` | BMP encoded bytes |
| `pnm` | P5 or P6 encoded bytes according to channel layout |

TGA is file-only and is not part of this encoded-bytes token set. `px.io.read_image` accepts TGA image type 2
(uncompressed true color) or 10 (RLE true color), 24-bit BGR or 32-bit BGRA, and bottom-left or top-left origin. It
rejects right-to-left origin, colormapped or grayscale types, 16 bit, reserved descriptor bits, nonzero attribute bits
in 24-bit input, and attribute bits other than 8 in 32-bit input. A default read normalizes to fp32 RGB or RGBA;
`unchanged=True` preserves uint8.

`px.io.write_image` writes every RGB or RGBA Frame storage dtype to `.tga` as type-10 RLE, 24- or 32-bit TGA. uint8
codes are unchanged. uint16 and uint32 use `recode_dtype` full-scale rescaling. Float input clips to `[0, 1]`,
multiplies by 255, and quantizes half away from zero. Origin is top-left. RLE packets contain 1 through 128 pixels and
never cross scanlines.

Radiance HDR is also file-only. `px.io.read_image` accepts `.hdr` with `FORMAT=32-bit_rle_rgbe` and standard orientation
`-Y H +X W`, decoding flat, old-style RLE, or new-style adaptive RLE into fp32 RGB. XYZE and other orientations are
unsupported. E=0 yields 0; otherwise Radiance `colr_color` evaluates `(mantissa + 0.5) * 2 ** (E - 136)`. EXPOSURE,
PRIMARIES, and COLORCORR are not applied to values.

`px.io.write_image` accepts every storage dtype on a Frame with exactly one R, G, and B channel for `.hdr`. It converts
nonnative input, including uint32, to the native/default float32 container with `recode_dtype` full-scale semantics,
generates RGBE with frexp semantics, and writes only new-style RLE in standard orientation. Writer width is 8 through
32,767. No optional header variables are generated.

DPX is also file-only. `px.io.read_image` accepts `.dpx` with `SDPX` or `XPDS`, orientation 0, one unsigned RGB or RGBA
element, and uncompressed 8-, 10-, 12-, or 16-bit samples. Eight- and sixteen-bit samples require packing 0. Ten-bit
input is limited to Method A filled, placing three samples from the high end of a 32-bit word. Twelve-bit input is
limited to Method A filled, using the high 12 bits of a 16-bit word. The CPU handles only the header and packed byte
range. One GPU pass resolves endian order, unpacks, normalizes, and selects channels. `unchanged=True` returns 8-bit
input as uint8 and 10-, 12-, or 16-bit codes as uint16.

`px.io.write_image` accepts every Frame storage dtype with unique RGB or RGBA channels for `.dpx`. It converts
nonnative input, including uint32, to the native/default float32 container with `recode_dtype` full-scale semantics,
then clips to `[0, 1]`, scales by the depth maximum, rounds half away from zero, and packs big-endian `SDPX` raw bytes
in one GPU pass. `bit_depth` defaults to 10 and accepts only 8, 10, 12, or 16. The writer fixes orientation 0, one
unsigned element, no compression, and Method A filled for 10- and 12-bit output. Frame gamma records `cineon` as
printing density, `linear` as linear, `s-log3` and `logc4` as logarithmic, and video or power families as the BT.709
transfer characteristic.

## TIFF compression

Case-sensitive tokens accepted by `compression=` for TIFF output from `px.io.encode_image` or `px.io.write_image`.

| Token | Meaning |
|---|---|
| `none` | Uncompressed |
| `lzw` | Lossless LZW compression |

## EXR compression

Case-sensitive tokens accepted by `compression=` for EXR file output from `px.io.write_image`. `None` selects `zip`.
PXR24 is lossless for HALF and rounds FLOAT to 24-bit precision. B44 and B44A lossily compress 4×4 blocks of HALF and
do not compress FLOAT. DWAA and DWAB are lossy DCT compression; `dwa_level=None` resolves to `45.0`.

| Token | Meaning |
|---|---|
| `none` | Uncompressed |
| `rle` | Lossless RLE compression |
| `zip` | Lossless ZIP compression in 16-scanline blocks; specification default |
| `zips` | Lossless ZIP compression one scanline at a time |
| `piz` | Lossless wavelet plus Huffman compression |
| `pxr24` | Lossless for HALF and lossy 24-bit compression for FLOAT |
| `b44` | Fixed-ratio lossy compression of 4×4 HALF blocks |
| `b44a` | B44 with abbreviated encoding for uniform blocks |
| `dwaa` | Lossy DCT compression in 32-scanline blocks |
| `dwab` | Lossy DCT compression in 256-scanline blocks |

## Frame boundary contract

| Contract | Requirement |
|---|---|
| GPU layout | `Frame.data` is HWC and C-contiguous; `px.io.from_array` selects a view or GPU copy under the three-state copy contract |
| pointer lifetime | Retain the allocation-owning Frame while using a raw pointer to `Frame.data` |
| stream | Pass a DLPack consumer stream through unchanged to `Frame.data.__dlpack__` |
| device export | `px.io.to_array(frame)` returns `cupy.ndarray`; both the Frame and returned array are DLPack producers |
| direct destination | `out=` accepts only a C-contiguous `cupy.ndarray` with matching shape and dtype, and returns that same object |
| host transfer | The canonical `to_array(...).get()` path is `px.io.to_array(frame, ...).get()`; alternatively call `cp.asnumpy(px.io.to_array(frame, ...))` explicitly |
