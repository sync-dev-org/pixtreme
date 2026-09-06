# pixtreme

[![PyPI](https://img.shields.io/pypi/v/pixtreme.svg)](https://pypi.org/project/pixtreme/)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.12-3776AB.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sync-dev-org/pixtreme/blob/main/LICENSE)

GPU-first image processing for Python, built on CUDA and CuPy.

**[API reference and documentation](https://sync-dev-org.github.io/pixtreme/)** — the complete public API (13 modules, 94 operations) with per-function contracts, plus the full performance report.

## Why pixtreme

pixtreme keeps image data on the NVIDIA GPU and makes a metadata-bearing `Frame` the common value passed between
operations. A `Frame` owns an HWC `cupy.ndarray` together with colorspace, transfer, channel, and YCbCr-matrix claims,
so color meaning travels with pixels instead of living in ambient configuration.

- Purpose-built `RawKernel` implementations and CuPy operations avoid unnecessary host round trips.
- Color conversion, format conversion, channel routing, and affine numeric transforms fuse work into a single pass
  where their contracts allow it.
- Floating-point working values are not clipped to `[0, 1]`: negative values, highlights above 1.0, and filter
  overshoot remain valid scene data until an explicit quantization or clipping boundary.
- The package root exposes `core`, 12 focused operation modules, and `__version__`. Named tokens are case-sensitive,
  validated immediately, and have no environment-dependent defaults.
- `Frame` is the working currency; device arrays and encoded/file formats cross explicit `from_*`, `to_*`,
  `read`/`write`, and `decode`/`encode` boundaries.

## Performance

The following measurements are selected from the current 204-case registry, taken from the single full run at
commit `ca83e10`. GPU cases use at least 1,000 FHD frames and 3 seconds after warmup, while file boundaries use at
least 20 iterations and the same 3-second floor. The test system used an NVIDIA RTX A6000, CUDA 12.9, CuPy 14.1.1, and
Python 3.12 under WSL2.

| Operation | Representative parameters | Median (ms) | FPS | Effective GB/s |
|---|---|---:|---:|---:|
| `resize` | 1920x1080 -> 960x540, `nearest` | 0.073 | 13755.0 | 427.8 |
| `resize` | 1920x1080 -> 3840x2160, `lanczos4` | 0.935 | 1069.2 | 133.0 |
| `from_array` | CHW uint16, 10-bit -> float32 HWC | 0.123 | 8162.1 | 304.6 |
| `px.io.to_yuva444p` | 12-bit legal, alpha full | 0.111 | 8991.4 | 447.5 |
| `rgb_to_rgb` | ACEScg linear -> sRGB sRGB | 0.134 | 7486.7 | 372.6 |
| `rgb_to_hsv` | label-driven scene values | 0.113 | 8814.3 | 438.7 |
| `rgb_to_rgb` | BT.2408 direct mapping -> Rec.2020 pq | 0.143 | 6990.0 | 347.9 |
| `apply_lut` | 65^3 LUT, tetrahedral | 0.353 | 2831.3 | 140.9 |
| `text` | single-line CJK, size 64, one outline, 4x supersampling | 0.391 | 2555.6 | 127.2 |
| `color_bars` | FHD ARIB STD-B28 normalized | 0.085 | 11745.6 | 292.3 |
| `read_image` | FHD HALF RGB EXR ZIP, unchanged, temporary-file I/O included | 37.909 | 26.4 | 0.7 |
| `write_image` | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, temporary-file I/O included | 40.614 | 24.6 | 0.9 |

These figures describe this system and workload, not a hardware-independent guarantee. File and encoded-byte
boundaries have different I/O-inclusive conditions. See [the full performance report](https://sync-dev-org.github.io/pixtreme/performance/) for every
case, distribution statistics, and the complete methodology.

EXR reads support scanline and tiled, single- and multipart files across all ten compression tokens; writes produce
single-part scanline files for the same ten tokens. Every path uses a pixtreme-owned implementation, with no OpenEXR
runtime dependency or fallback. Routing is fixed in source rather than benchmarked at runtime: NONE uses the native
read lane; ZIP, ZIPS, and PXR24 use custom CPU reads; the remaining reads and every write use GPU lanes. A float32
Frame written without `dtype` stores HALF by default; pass `dtype="float32"` for explicit FLOAT storage. On the system
above, the default ZIP/HALF path measured 37.909 ms to read unchanged and 40.614 ms to write, including temporary-file
I/O.

## Requirements

- Python 3.12 or newer
- CUDA 12.x
- An NVIDIA GPU

WSL2 is supported with the Windows NVIDIA driver and a working CUDA device. Depending on the WSL installation,
`nvidia-smi` may be available at `/usr/lib/wsl/lib/nvidia-smi` rather than on the default `PATH`.

## Installation

With pip:

```console
python -m pip install pixtreme
```

With uv:

```console
uv add pixtreme
```

Upgrading from 0.x? The 1.x series is a ground-up rewrite — see the [changelog](https://github.com/sync-dev-org/pixtreme/blob/main/CHANGELOG.md) for the migration summary.

## Quickstart

Read an image into GPU memory, work in scene-linear ACEScg, and quantize only at the file boundary:

```python
import pixtreme as px

frame = px.io.read_image("input.png")
working = px.color.rgb_to_rgb(frame, output_colorspace="ACEScg", output_gamma="linear")
working = px.filter.sharpen(working, amount=0.5)
output = px.color.rgb_to_rgb(working, output_colorspace="sRGB", output_gamma="sRGB")
px.io.write_image("output.png", px.values.quantize(output, bit_depth=8))
```

`px.io.read_image` returns a `px.core.Frame` whose pixels already reside on the GPU. Processing remains float32 and
unclipped; `px.values.quantize` is the explicit normalized-float-to-integer boundary required by PNG.

## API tour

The package root exposes 13 modules and `__version__`. Types, helpers, and all 94 operations live under one canonical
two-level path; the root does not re-export them, and `Frame` has no operation methods:

| Namespace | Public members | Responsibility |
|---|---|---|
| `px.core` | `Frame`, `Lut`, `Lut1D`, `channels`, and the named-token `Literal` aliases | Core types, channel normalization, and closed vocabulary |
| `px.io` | `read_image`, `write_image`, `read_header`, `read_lut`, `decode_lut`, `write_lut`, `decode_image`, `encode_image`, `from_array`, `to_array`, and eight named-format `from_*` / `to_*` pairs | File, byte, device-array, LUT, and wire-format boundaries |
| `px.color` | `apply_lut`, `gamma_to_linear`, `hsv_to_rgb`, `linear_to_gamma`, `rgb_to_grayscale`, `rgb_to_hsv`, `rgb_to_rgb`, `rgb_to_ycbcr`, `ycbcr_to_rgb`, `ycbcr_to_ycbcr`, `equalize_histogram`, `clahe`, `chromatic_adaptation`, `white_balance`, `white_point_simulation` | Colorimetry, transfer functions, YCbCr/HSV, LUTs, histogram operations, white-point adaptation, and explicit tonemapping |
| `px.filter` | `gaussian_blur`, `box_blur`, `median_blur`, `bilateral_blur`, `directional_blur`, `zoom_blur`, `spin_blur`, `vector_blur`, `lens_blur`, `sobel`, `laplacian`, `difference_of_gaussians`, `canny`, `sharpen`, `unsharp_mask`, `convolve_box` | Blur, derivatives, edges, sharpening, and convolution |
| `px.transform` | `resize`, `warp_affine`, `stack` | Geometry and multi-image layout |
| `px.draw` | `line`, `polyline`, `rectangle`, `circle`, `ellipse`, `polygon`, `text`, and the `Font` type | Shape and text drawing |
| `px.generate` | `ramp`, `grid`, `checkerboard`, `color_bars`, `fractal_noise`, `turbulent_noise`, `grain` | Procedural frames, test patterns, and noise |
| `px.morphology` | `erosion`, `dilation`, `opening`, `closing`, `morphological_gradient`, `white_tophat`, `black_tophat` | Morphological image operations |
| `px.metrics` | `psnr`, `ssim`, `ssim_map` | Image-quality scalars and response maps |
| `px.feature` | `corner_harris`, `match_template` | Image-feature response maps |
| `px.values` | `quantize`, `dequantize`, `full_to_legal`, `legal_to_full`, `cast_dtype`, `recode_dtype` | Range, quantization, and storage representation |
| `px.channel` | `shuffle` | Channel routing and assembly without implicit color meaning changes |
| `px.composite` | `merge` | Transform-aware multi-image compositing |

Render the Quickstart's scene-linear `working` frame through the analytic ACES 2.0 Output Transform:

```python
display = px.color.rgb_to_rgb(
    working,
    output_colorspace="sRGB",
    output_gamma="sRGB",
    tonemap="ACES-2.0",
)
```

The bytes boundary mirrors the file boundary without inventing a host-array API:

```python
png_bytes = px.io.encode_image(
    px.values.quantize(display, bit_depth=8),
    format="png",
    compression_level=4,
)
round_trip = px.io.decode_image(png_bytes)
```

Text shaping supports bundled CJK fonts, user OpenType/TrueType fonts through `px.draw.Font`, and an opt-in 4x
supersampled raster path:

```python
captioned = px.draw.text(
    display,
    text="極彩",
    position=(48, 96),
    size=64,
    color=(1.0, 0.8, 0.2),
    supersample=True,
)
```

## Full performance

[the full performance report](https://sync-dev-org.github.io/pixtreme/performance/) contains the recorded measured cases, including mean, median, FPS, p5, p95,
effective bandwidth, parameters, and the 88 cases whose median exceeds 1 ms. It also separates GPU-device throughput
from temporary-file and encoded-byte I/O measurements.

## Color management

Color processing is explicit, metadata-aware, and designed to preserve scene values until a declared output boundary.

- The closed vocabulary carries 27 colorspace tokens and 33 transfer (gamma) tokens: the sRGB, Rec.709, Rec.2020,
  P3, SMPTE-C, and ACES families plus the Sony, ARRI, Blackmagic Design, DaVinci, RED, Canon, Panasonic, DJI,
  Fujifilm, Nikon, Leica, Apple, and Samsung camera encodings. Each token is derived from its published
  specification and documented with its provenance and domain-extension rules in the
  [token reference](https://sync-dev-org.github.io/pixtreme/tokens/).
- The analytic ACES 2.0 SDR 100-nit Output Transform evaluates the complete AP1 limit, Hellwig JMh, tone, chroma,
  gamut-compression, limiting-RGB, reference-range, and display-encoding chain in one fused CUDA pass. Its 363-record
  hue table is an algorithm parameter, not an RGB-grid output approximation; runtime evaluation uses no LUT
  interpolation and matches direct OpenColorIO 2.5.2 reference evaluation with `rtol=0`, `atol=2e-4`.
- ACES 1.3 is also available as a formula-based one-pass transform. Both ACES generations are exposed only through
  their analytic tonemap tokens.
- BT.2408 direct mapping places SDR reference white at 203 cd/m2 for Rec.2020 HLG or PQ output.
- RGB/YCbCr conversion, legal/full-range code positions, and chroma siting follow H.273-aligned contracts, with the
  matrix basis carried in `Frame` metadata.
- Broadcast test-pattern generation covers ARIB STD-B28, SMPTE RP 219-1, and ITU-R BT.2111-2 HLG/PQ variants,
  including exact 10-bit code output.

## Status & license

The 1.x series is the current release line. It is a ground-up implementation and does not connect to the 0.x codebase. The
final 0.x release, 0.9.0, remains available from the [`v0.9.0` Git tag](https://github.com/sync-dev-org/pixtreme/tree/v0.9.0)
and the [PyPI release history](https://pypi.org/project/pixtreme/#history). See the [changelog](https://github.com/sync-dev-org/pixtreme/blob/main/CHANGELOG.md) for the
1.0 migration-impact summary.

pixtreme is released under the [MIT License](https://github.com/sync-dev-org/pixtreme/blob/main/LICENSE).
