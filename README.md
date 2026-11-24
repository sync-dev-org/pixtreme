# pixtreme

[![PyPI](https://img.shields.io/pypi/v/pixtreme.svg)](https://pypi.org/project/pixtreme/)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.12-3776AB.svg)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sync-dev-org/pixtreme/blob/main/LICENSE)

GPU-first image processing for C, C++ and Python, built on CUDA.

**[API reference and documentation](https://sync-dev-org.github.io/pixtreme/docs/)** — the complete public API (15 modules, 169 public functions) with per-function contracts, plus the full performance report.

## Why pixtreme

pixtreme keeps image data on the NVIDIA GPU and makes a metadata-bearing `Frame` the common value passed between
operations. A `Frame` holds HWC GPU pixels together with colorspace, transfer, channel, and YCbCr-matrix claims,
so color meaning travels with pixels instead of living in ambient configuration.

Frame and `px.NDArray` provide `view()` to share pixels with an independent description,
and `clone(stream=None)` to duplicate pixels. Python `copy.copy` and `copy.deepcopy`
also duplicate pixels; use `view()` when sharing is intended.

GPU arrays enter through DLPack; CuPy and PyTorch are optional external libraries.
Install the library you use separately. Use `cupy.from_dlpack(frame)` or
`torch.from_dlpack(frame)` to share writable pixels.
DLPack orders the consumer stream after recorded writers and keeps the pixel owner alive.
Read shape and dtype through `frame.shape` and `frame.dtype`. To replace the pixel storage,
construct a new `px.core.Frame` and copy its colorspace, gamma, channels, matrix, range, and bit_depth.

- Native C++ and CUDA implementations process pixels without a GPU array library dependency.
- Color conversion, format conversion, channel routing, and affine numeric transforms fuse work into a single pass
  where their contracts allow it.
- Floating-point working values are not clipped to `[0, 1]`: negative values, highlights above 1.0, and filter
  overshoot remain valid scene data until an explicit quantization or clipping boundary.
- The package root exposes 15 focused modules, including `core`, and `__version__`. Named tokens match
  case-insensitively and separator-insensitively (`Rec.709`, `rec 709`, and `REC_709` name the same token; see the
  [token reference](https://sync-dev-org.github.io/pixtreme/tokens/)), are validated immediately, and have no
  environment-dependent defaults.
- `Frame` is the working currency; device arrays and encoded/file formats cross explicit `from_*`, `to_*`,
  `read`/`write`, and `decode`/`encode` boundaries.

## Performance

pixtreme does its image work on the GPU. The table compares pixtreme's median time with a widely used CPU library
doing the same work on the same machine, measured in two ways:

- **A**: from a GPU `Frame` to a GPU `Frame`. The CPU library's time includes copying the image from the GPU and
  back; file reads end in a GPU `Frame`, and file writes start from one.
- **B**: the CPU library starts and ends with host arrays and makes no GPU copies, while pixtreme still starts and
  ends with GPU `Frame`s.

Each speed-up is the CPU library's median time divided by pixtreme's. This summary lists the conditions where
pixtreme is at least 1.5 times as fast in both ways.

| Operation | Compared with | Input and settings | Speed-up (A) | Speed-up (B) |
|---|---|---|---:|---:|
| Gaussian blur | OpenCV, 24 threads | FHD float32 RGB, σ = 1 | 36.02× | 4.45× |
| Lens blur | OpenCV `filter2D`, 24 threads | FHD float32 RGB, disk radius 4 | 30.01× | 6.63× |
| Resize | OpenCV, 24 threads | FHD float32 RGB to 960×540, area | 46.44× | 4.35× |
| JPEG read | Pillow with libjpeg-turbo, 1 thread | FHD uint8 RGB, quality 75, 4:2:0 | 24.00× | 21.12× |
| JPEG write | Pillow with libjpeg-turbo, 1 thread | FHD uint8 RGB, quality 75, 4:2:0 | 8.31× | 7.65× |
| PNG read | Pillow with zlib, 1 thread | FHD uint8 RGB, level 6 | 2.99× | 2.64× |
| PNG write | Pillow with zlib, 1 thread | FHD uint8 RGB, level 6 | 63.60× | 55.33× |
| EXR read | OpenEXR, 1 thread | FHD HALF RGB, DWAB level 45 | 7.43× | 6.65× |
| EXR write | OpenEXR, 1 thread | FHD HALF RGB, ZIP | 17.21× | 16.96× |
| JPEG 2000 read | OpenJPEG through imagecodecs, 1 thread | FHD 12-bit RGB, lossless | 27.21× | 27.10× |
| JPEG 2000 write | OpenJPEG through imagecodecs, 1 thread | FHD 12-bit RGB, lossless | 19.03× | 18.96× |
| HTJ2K read | OpenJPH through imagecodecs, 1 thread | FHD 12-bit RGB, lossless | 24.67× | 22.52× |
| HTJ2K write | OpenJPH through imagecodecs, 1 thread | FHD 12-bit RGB, lossless | 3.40× | 3.40× |
| ACES 2.0 output transform | OpenColorIO, 1 thread | FHD float32 RGB, ACEScg to Rec.709 100 nit | 7397.27× | 7116.69× |
| 3D LUT | OpenColorIO, 1 thread | FHD float32 RGB, 33³ tetrahedral | 450.69× | 394.82× |
| VMAF, per frame | libvmaf with AVX, 1 thread | FHD float32 RGB, 48 frames, `vmaf_v0.6.1` | 134.55× | 120.63× |

FHD is 1920×1080. Measured on an NVIDIA GeForce RTX 5090 and an Intel Xeon w5-3425 with NVIDIA driver 617.42 and
CUDA runtime 13.2, under WSL2 on Windows, from 2026-11-17 to 2026-11-22, with pixtreme development builds after
2.26.0. Results depend on the workload and the comparison method. The
[performance report](https://sync-dev-org.github.io/pixtreme/performance/#library-comparisons) keeps every measured
condition, including those where the CPU library is faster, with library versions, thread counts and output checks.
Its [full results](https://sync-dev-org.github.io/pixtreme/performance/#full-results) give pixtreme's own timings,
including operations without a CPU comparison.

## Requirements

- Linux x86_64 (including WSL2) or Windows x86_64
- Python 3.12 or newer
- CUDA 13.x
- NVIDIA driver R580 or newer
- An NVIDIA GPU with compute capability 7.5 or higher

The 2.x wheels target Linux x86_64 (`manylinux_2_28`) and native Windows x86_64 (`win_amd64`).
Each platform has one `cp312-abi3` wheel for standard CPython 3.12 or newer; free-threaded builds are not supported.
WSL2 uses the Linux wheel. On native Windows, `import pixtreme` registers the package's DLL directory before
loading its native modules.
Binary wheels need no C++ compiler or CUDA toolkit. Source builds require a C++17 compiler (MSVC on Windows)
and C/C++ and CUDA 13 build dependencies supplied through Conan; Windows builds also need the Windows SDK.
DWA compression uses native CPU workers. DWA reads decode entropy streams
on the CPU and reconstruct dense colour blocks on the supplied CUDA stream. When `stream` is omitted, they use
the C++ interface's default stream. Subsequent image processing continues on the GPU.

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

Upgrading from 0.x? The 1.x series is a ground-up rewrite — see the [changelog](https://sync-dev-org.github.io/pixtreme/changelog/) for the migration summary.

## Release packages

Download Linux and Windows wheels, prebuilt SDK archives, and `SHA256SUMS` from
[GitHub Releases](https://github.com/sync-dev-org/pixtreme/releases). Both platform wheels are also available on PyPI.
The [C and C++ SDK guide](https://sync-dev-org.github.io/pixtreme/cpp/) explains how to use the headers,
shared library, and CMake package supplied in each platform's SDK archive and wheel.
The Windows SDK includes `pixtreme-2.dll` and the `pixtreme.lib` import library.

## Quickstart

Read an image into GPU memory, work in scene-linear ACEScg, and quantize only at the file boundary:

```python
import pixtreme as px

frame = px.io.read_image("input.png")
working = px.color.rgb_to_rgb(frame, output_colorspace="ACEScg", output_gamma="linear")
working = px.filter.sharpen(working, amount=0.5)
output = px.color.rgb_to_rgb(working, output_colorspace="sRGB", output_gamma="sRGB")
px.io.write_png("output.png", px.values.quantize(output, bit_depth=8))
```

`px.io.read_image` returns a `px.core.Frame` whose pixels already reside on the GPU. Processing remains float32 and
unclipped; `px.values.quantize` is the explicit normalized-float-to-integer boundary required by PNG.

## API tour

The package root exposes the feature modules, `Lut1D`, `Lut3D`, `NDArray`, `PixelBuffer`, `PixelBufferLayout` and `__version__`.
Operations live under their feature modules.
`Frame` has no operation methods.

Packed pixels use [`px.PixelBuffer`](https://sync-dev-org.github.io/pixtreme/reference/io/#pixelbuffer), which holds GPU byte storage together with
format, dimensions, plane offsets and pitches, and color declarations. The `px.io` conversion functions exchange
pixels between a PixelBuffer and a Frame.

| Namespace | Public members | Responsibility |
|---|---|---|
| `px.core` | `Frame`, `Lut3D`, `Lut1D`, `Stream`, `Event`, `Graph`, `capture_begin`, `capture_end`, `channels`, and the named-token `Literal` aliases | Core types, CUDA graph capture, channel normalization, and closed vocabulary |
| `px.io` | `read_image`, `read_<format>`, `write_<format>`, `write_exr_channels`, `encode_exr_channels`, `read_header`, `read_lut`, `decode_lut`, `write_lut`, `decode_image`, `decode_<format>`, `encode_<format>`, `from_tensor`, `to_tensor`, `from_pixel_buffer`, `to_pixel_buffer`, `repack_pixel_buffer`, `read_image_as_pixel_buffer`, `decode_image_as_pixel_buffer`, `write_jpeg_from_pixel_buffer`, `encode_jpeg_from_pixel_buffer`, `write_jpeg2000_from_pixel_buffer`, `encode_jpeg2000_from_pixel_buffer` | File, byte, device-array, LUT, and PixelBuffer conversion boundaries |
| `px.color` | `apply_lut`, `gamma_to_linear`, `hsv_to_rgb`, `linear_to_gamma`, `rgb_to_grayscale`, `rgb_to_hsv`, `rgb_to_rgb`, `rgb_to_ycbcr`, `ycbcr_to_rgb`, `ycbcr_to_ycbcr`, `equalize_histogram`, `clahe`, `chromatic_adaptation`, `white_balance`, `white_point_simulation`, `grade` | Colorimetry, transfer functions, YCbCr/HSV, LUTs, histogram operations, white-point adaptation, per-channel Lift / Gamma / Gain, and explicit tonemapping |
| `px.filter` | `gaussian_blur`, `box_blur`, `median_blur`, `bilateral_blur`, `directional_blur`, `zoom_blur`, `spin_blur`, `vector_blur`, `lens_blur`, `sobel`, `laplacian`, `difference_of_gaussians`, `canny`, `sharpen`, `unsharp_mask`, `convolve_box` | Blur, derivatives, edges, sharpening, and convolution |
| `px.transform` | `resize`, `warp_affine`, `stack` | Geometry and multi-image layout |
| `px.draw` | `Path`, `Drawing`, `add_*`, `transform`, `round_corners`, `dash`, `group`, `render`, `text`, `Font` | Immutable shape construction, GPU rendering, and text |
| `px.generate` | `ramp`, `grid`, `checkerboard`, `color_bars`, `fractal_noise`, `turbulent_noise`, `grain` | Procedural frames, test patterns, and noise |
| `px.morphology` | `erosion`, `dilation`, `opening`, `closing`, `morphological_gradient`, `white_tophat`, `black_tophat` | Morphological image operations |
| `px.metrics` | `psnr`, `ssim`, `ssim_map`, `vmaf` | Image and video quality scores and response maps |
| `px.feature` | `corner_harris`, `match_template` | Image-feature response maps |
| `px.values` | `quantize`, `dequantize`, `cast_dtype`, `recode_dtype` | Pixel-value scale conversion, quantization, and storage representation |
| `px.channel` | `shuffle` | Channel routing and assembly without implicit color meaning changes |
| `px.composite` | `merge` | Transform-aware multi-image compositing |
| `px.fonts` | `font_path`, `available` | Bundled and installed font-file discovery |

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
png_bytes = px.io.encode_png(
    px.values.quantize(display, bit_depth=8),
    compression_level=4,
)
round_trip = px.io.decode_image(png_bytes)
```

For an RGB Frame, build shapes as immutable values, attach paint, and render them into a new Frame:

```python
path = px.draw.add_rectangle(px.draw.Path(), (10, 10), (240, 120))
path = px.draw.round_corners(path, radius=12)
drawing = px.draw.add_fill(path, color=(0.1, 0.4, 0.8))
drawing = px.draw.add_stroke(drawing, color=(1.0, 1.0, 1.0), width=2)
outlined = px.draw.render(frame, drawing, aa="distance")
```

Geometry construction uses host values. `render` evaluates the graph on its CUDA stream and preserves the input
Frame. See the [drawing tokens](https://sync-dev-org.github.io/pixtreme/docs/tokens/#draw-path-commands) for fill
rules, stroke styles, modes, edits, and paint order.

Text uses the same native Path and Drawing graph. Choose a catalog family name or load a user font with
`px.draw.Font.from_file(path)`. Catalog fonts are downloaded to the user's cache when first needed; no font files
are bundled in the package. To download them explicitly, call `px.fonts.download()`.

```python
text_path = px.draw.add_text(px.draw.Path(), "極彩", size=64, font="noto-sans-cjk-jp")
text_path = px.draw.transform(text_path, translate=(48, 96))
captioned = px.draw.render(display, px.draw.add_fill(text_path, color=(1.0, 0.8, 0.2)), aa="supersample")
```

## Image formats

`px.io` reads and writes PNG, JPEG, TIFF, JPEG 2000, HTJ2K, WebP, BMP, PNM / PAM, EXR, TGA, HDR and DPX images as
files or bytes.

EXR reads support scanline and tiled, single- and multipart files across all ten compression tokens; writes produce
single-part scanline files for the same ten tokens. Every path uses a pixtreme-owned implementation, with no OpenEXR
runtime dependency or fallback. Routing is fixed in source rather than benchmarked at runtime: NONE uses the native
read lane; ZIP, ZIPS, and PXR24 use custom CPU reads; the remaining reads and every write use GPU lanes. A float32
Frame written without `dtype` stores HALF by default; pass `dtype="float32"` for explicit FLOAT storage.

WebP file reads and decoding from bytes use pixtreme's C++ and CUDA decoder (VP8, VP8L, and alpha), writing pixels
directly into a GPU Frame. Still-image I/O supports RGBA with alpha. Malformed WebP input identified under RFC 6386
or RFC 9649 is rejected with a reason and a remedy. Encoding uses statically linked libwebp on the CPU and accepts
`method` and `quality`.

## Color management

Color processing is explicit, metadata-aware, and designed to preserve scene values until a declared output boundary.

- The closed vocabulary carries 31 colorspace tokens and 37 transfer (gamma) tokens: the sRGB, Rec.709, Rec.2020,
  P3, Rec.601-525, Rec.601-625, Adobe RGB, ProPhoto RGB, and ACES families plus the Sony, ARRI, Blackmagic Design, DaVinci, RED, Canon, Panasonic, DJI,
  Fujifilm, Nikon, Leica, Apple, and Samsung camera encodings, plus `CIE-XYZ` and `DCDM`. Each token is derived from its published
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

The 2.x series provides native C++ support behind the Python API on Linux and Windows.
The 1.x series accepts bug fixes only. The 1.x implementation
is a ground-up rewrite and does not connect to the 0.x codebase. The
final 0.x release remains available as [pixtreme 0.9.0 on PyPI](https://pypi.org/project/pixtreme/0.9.0/). See the [changelog](https://sync-dev-org.github.io/pixtreme/changelog/) for the
1.0 migration-impact summary.

pixtreme is released under the [MIT License](https://github.com/sync-dev-org/pixtreme/blob/main/LICENSE).
JPEG decoding includes libjpeg-turbo components under the IJG License. This software is based in part on
the work of the Independent JPEG Group. The wheel includes the complete third-party license texts in its
distribution metadata.
