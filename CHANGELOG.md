# Changelog

Notable changes to pixtreme are documented in this file.

## 1.1.0 - 2026-08-13

pixtreme 1.1.0 reshapes the public namespace into 13 responsibility modules with a single canonical path per
operation, adds `uint32` as the fifth Frame storage dtype with end-to-end EXR UINT support, and completes the
source-fixed EXR lanes so OpenEXR is no longer a runtime dependency. It is a breaking release within the 1.x
line; the Migration section below maps every removed path to its canonical replacement. A hosted API reference
covering the full public surface accompanies this release.

### Added

- `px.io.write_image()` has a keyword-only `dtype` option for EXR output. It accepts `float16`, `float32`, or `uint32`;
  explicitly passing it for another format raises `ValueError`.
- `uint32` is the fifth Frame storage dtype. `px.core.Frame`, `px.io.from_array()`, `px.io.to_array(frame, ...)`,
  `px.values.cast_dtype()`, and `px.values.recode_dtype()` preserve or convert it under the same five-dtype contracts.
- EXR UINT channels are supported end to end. Default reads convert integer values literally to `float32`, with
  documented precision loss possible above `2^24`; `unchanged=True` preserves the original `uint32` sample bits.
- The EXR reader handles tiled ONE_LEVEL, MIPMAP, and RIPMAP images at full-resolution level 0, flat multi-part
  images, and subsampled channels. Flat channels remain readable in mixed deep/flat files; explicitly selecting a
  deep channel reports the unsupported part and channel.

### Changed

- **Breaking:** Frame construction and `data` assignment now reject zero height, width, or channel count with
  `ValueError`. Values operations no longer accept zero-sized Frames because such Frames cannot be constructed.
- **Breaking:** `px.transform.resize()` now rejects non-`float32` Frame storage with `ValueError` and public
  value-conversion guidance instead of silently converting the input to `float32`.
- The seven `px.draw` operations and `px.filter.gaussian_blur()`, `box_blur()`, `median_blur()`,
  `bilateral_blur()`, `directional_blur()`, `zoom_blur()`, `spin_blur()`, `vector_blur()`, `lens_blur()`, and
  `convolve_box()` now reject non-`float32` Frame storage with `ValueError` and public value-conversion guidance,
  before any identity shortcut. `vector_blur()` applies the same check to its vector Frame. These inputs previously
  reached fp32 CUDA paths with undefined behavior.
- **Breaking:** omitting `dtype` when writing a `float32` Frame to EXR now writes HALF instead of FLOAT. Pass
  `dtype="float32"` to preserve FLOAT output. A `uint32` Frame is the only dtype that defaults to native UINT.
- `px.io.write_image()` and `px.io.encode_image()` accept all five Frame storage dtypes for every supported output
  format.
  Non-EXR native container matches are preserved; other non-EXR inputs are full-scale recoded to the format's
  default storage. EXR uses the Frame-dependent defaults or explicit `dtype` selection described above.
- **Breaking:** the package root now exposes only 13 responsibility modules and `__version__`. Types, helpers, and all
  89 operations use a single canonical `px.<module>.<leaf>` path, without compatibility aliases or deprecation shims.
  Publishing this breaking change within the 1.x line is an intentional departure from semantic-versioning
  major-version rules.
- I/O boundaries are grouped under `px.io`; blur operations are grouped under `px.filter`; quality metrics and image
  features are split into `px.metrics` and `px.feature`; histogram equalization and CLAHE are grouped under
  `px.color`.
- Frame-to-array and Frame-to-wire-format exits are `px.io.to_array(frame, ...)` and
  `px.io.to_<format>(frame, ...)` functions. `px.core.Frame` no longer has operation methods.
- EXR read and write use source-fixed pixtreme GPU or CPU lanes for all ten compression modes: NONE, RLE, ZIPS, ZIP,
  PIZ, PXR24, B44, B44A, DWAA, and DWAB. OpenEXR is no longer a runtime dependency or fallback; it remains a
  development dependency only as a test and fixture oracle.
- Reading a subsampled channel by itself now returns its stored dimensions and samples. It no longer places samples
  at the start of a full-resolution buffer with an uninitialized remainder.
- Known trade-off: the current GPU writers produce larger files for ZIP (+11.5% / +28.8%), ZIPS (+4.0% / +10.3%),
  PXR24 (+25.2% / +38.6%), and DWA (+2.8% / +4.0%), for fp16 / fp32 respectively. Other codecs are effectively
  unchanged; reducing these differences is future work.

### Fixed

- The GPU PIZ writer now serializes Huffman code-length tables with canonical zero-run tokens instead of writing
  every zero length literally. Files written by earlier versions remain valid and readable; new files are
  byte-identical to the canonical form and smaller for sparse code-length tables.

### Removed

- **Breaking:** root-level `Frame`, `Lut`, `ImageHeader`, `channels`, and the 15 root-level I/O functions were removed.
- **Breaking:** `px.blur` and `px.analyze` were removed; their operations are available only from their new canonical
  modules.
- **Breaking:** `Frame.to_array()` and the eight `Frame.to_<format>()` wire-format methods were removed.

### Migration

- Pass `dtype="float32"` to `px.io.write_image()` when EXR FLOAT precision must be preserved; omission now selects
  HALF for every Frame storage dtype except `uint32`.
- Keep object IDs and other exact integers in a `uint32` Frame. Omit `dtype` or pass `dtype="uint32"` to write native
  EXR UINT, and use `unchanged=True` to read the original bits. Default reads return literal `float32` values and can
  lose integer precision above `2^24`.
- OpenEXR no longer needs to be installed in runtime environments. Keep it only in development environments that run
  the EXR oracle tests or generate fixtures.

| Previous path | New canonical path |
|---|---|
| `px.Frame` / `px.Lut` / `px.channels` | `px.core.Frame` / `px.core.Lut` / `px.core.channels` |
| `px.ImageHeader` | `px.io.ImageHeader` |
| `px.read_image` / `px.write_image` / `px.read_header` / `px.read_lut` | `px.io.read_image` / `px.io.write_image` / `px.io.read_header` / `px.io.read_lut` |
| `px.decode_image` / `px.encode_image` / `px.from_array` | `px.io.decode_image` / `px.io.encode_image` / `px.io.from_array` |
| `px.from_uyvy422` / `px.from_v210` | `px.io.from_uyvy422` / `px.io.from_v210` |
| `px.from_nv12` / `px.from_p010` | `px.io.from_nv12` / `px.io.from_p010` |
| `px.from_yuv420p` / `px.from_yuv422p` | `px.io.from_yuv420p` / `px.io.from_yuv422p` |
| `px.from_yuv444p` / `px.from_yuva444p` | `px.io.from_yuv444p` / `px.io.from_yuva444p` |
| `frame.to_array(...)` / `frame.to_<format>(...)` | `px.io.to_array(frame, ...)` / `px.io.to_<format>(frame, ...)` |
| `px.blur.<name>` | `px.filter.<name>` |
| `px.filter.equalize_histogram` / `px.filter.clahe` | `px.color.equalize_histogram` / `px.color.clahe` |
| `px.analyze.psnr` / `px.analyze.ssim` / `px.analyze.ssim_map` | `px.metrics.psnr` / `px.metrics.ssim` / `px.metrics.ssim_map` |
| `px.analyze.corner_harris` / `px.analyze.match_template` | `px.feature.corner_harris` / `px.feature.match_template` |

## 1.0.1 - 2026-08-06

pixtreme 1.0.1 is the first supported release of the 1.x line, a ground-up rewrite. It does not provide a
compatibility layer for the 0.x API, so applications upgrading from 0.9.0 should treat 1.x as a new public API.

### Added

- A single `pixtreme` distribution centered on a metadata-bearing GPU `Frame`, with operations grouped into 12
  focused modules: `io`, `color`, `blur`, `filter`, `analyze`, `morphology`, `transform`, `draw`, `generate`,
  `channel`, `values`, and `composite`.
- Analytic ACES 1.3 and ACES 2.0 output transforms, explicit pre-baked LUT variants, and BT.2408 direct mapping for
  supported Rec.2020 HLG and PQ outputs.
- Explicit GPU format boundaries for UYVY422, v210, NV12, P010, YUV420p, YUV422p, YUV444p, and YUVA444p, alongside
  file, encoded-byte, and device-array boundaries.
- GPU drawing primitives including CJK text, morphology and image-analysis modules, and histogram equalization and
  CLAHE operations.
- A reproducible [performance report](docs/performance.md) covering the public GPU operation and I/O registry.

### Changed

- The 0.x component distributions `pixtreme-core`, `pixtreme-aces`, `pixtreme-filter`, `pixtreme-draw`, and
  `pixtreme-upscale` are discontinued. Version 1.0 ships only as the single `pixtreme` distribution.
- `Frame` replaces raw arrays as the primary value passed between image operations. It carries colorspace, transfer,
  channel, and YCbCr-matrix claims with HWC GPU data.
- Public operations now use canonical module paths. The package root exports only `Frame`, `Lut`, `ImageHeader`, and
  `channels`; it no longer aggregates operation aliases from installed component packages.
- `Frame` methods are limited to exits from the Frame domain: `to_array()` and eight `to_<format>()` methods. Image
  processing remains in module functions.
- The default channel order returned by image reading is RGB. Version 0.x returned BGR by default following the
  OpenCV convention; callers with BGR-specific indexing or conversion steps must update them.
- Named option tokens are case-sensitive and validated immediately. They replace the 0.x pattern of root-level
  interpolation, codec, template-matching, and border constants where the corresponding 1.0 operation exists.
- Floating-point working values are not implicitly clipped to `[0, 1]`. Negative scene values, highlights above 1.0,
  and filter overshoot remain valid until an explicit quantization or clipping boundary.
- CUDA array interoperability uses CuPy arrays and the DLPack protocol. `io.from_array()` accepts CUDA producers, and
  `Frame` and `to_array()` provide the device-array exit; CPU NumPy arrays are not accepted by this boundary.

### Removed

- The 0.x compatibility surface, including legacy root-level operation exports and import paths under
  `pixtreme_core`, `pixtreme_aces`, `pixtreme_filter`, `pixtreme_draw`, and `pixtreme_upscale`.
- Built-in ONNX Runtime, PyTorch, and TensorRT upscalers and model-conversion utilities. Super-resolution and inference
  runtime integration now belong in application-side packages built on pixtreme's DLPack and array boundaries.
- GUI lifecycle helpers such as `imshow`, `waitkey`, and `destroy_all_windows`.
- Dedicated `to_numpy`, `to_tensor`, and `to_cupy` conversion helpers. Host transfer uses CuPy's APIs; CUDA tensor
  exchange uses DLPack.

### Migration

- Replace component-package dependencies and imports with the single `pixtreme` distribution and canonical module
  paths such as `px.io.read_image`, `px.color.rgb_to_rgb`, and `px.transform.resize`.
- Pass `Frame` values between operations, and cross array or encoded-format boundaries explicitly with `px.io.from_*`
  and `Frame.to_*`.
- Audit every image-reading call site for the BGR-to-RGB default change.
- Replace integer option constants and permissive aliases with the documented named tokens, and quantize or clip only
  at the intended output boundary.

## 1.0.0 - 2026-08-06 (yanked)

Yanked shortly after publication. Use 1.0.1 instead; the changes above describe the 1.x line.
