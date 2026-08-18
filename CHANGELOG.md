# Changelog

Notable changes to pixtreme are documented in this file.

## 1.2.1 - 2026-08-17

pixtreme 1.2.1 shortens seven morphology operations by 25 to 48 times, cuts `match_template`
FFT-path runtime nearly in half, and reduces long-path blur, procedural noise, LUT text
serialization, and white-management host composition times by 22 to 71 percent. Public
signatures, observable behavior, and output bits are unchanged; measured outputs are
byte-identical to 1.2.0 across independent verification. Absolute figures below were
remeasured on an NVIDIA RTX A6000 (WSL2, CUDA runtime 12.9, CuPy 14.1.1) with GPU 0
dedicated to a single full-suite `uv run pytest -m performance` run at commit `5f71032`.

### Changed

- `px.morphology.erosion()`, `dilation()`, `opening()`, `closing()`, `morphological_gradient()`,
  `white_tophat()`, and `black_tophat()` now load an output tile with radius halo into shared
  memory once per block, reduce from that shared substrate, and select disk membership from an
  integer row-limit cache keyed on `(device, radius)`. `morphological_gradient()` fuses the
  min/max into a single pass and the tophat pair fuses the trailing subtraction. FHD RGB
  radius-5 disk cases shorten by 25 to 48 times (`morphological_gradient` from 17.16 ms to
  0.36 ms; `erosion` and `dilation` from 8.65 ms to 0.34 ms). Channel counts above 4 or tiles
  exceeding 48 KiB fall back to the original global kernel unchanged.
- `px.feature.match_template()` fuses the FFT-path response reduction, centering, denominator
  construction, zero-variance handling, square root, and division into one RawKernel for
  channel counts 1 through 3. FHD RGB with a 64x64 template at `ccoeff_normed` shortens from
  24.72 ms to 13.10 ms (47.0%). Channel counts 4 and above retain the original composition
  path.
- `px.generate.grain()` at `size=1` uses a lattice-aligned kernel that evaluates only the two
  contributing z-layer lattice values per channel. `px.generate.fractal_noise()` and
  `px.generate.turbulent_noise()` at `scale >= 16` use a per-block shared gradient kernel that
  reuses at most 32 lattice gradients across each 16x16 output block. FHD RGB grain at
  `intensity=0.1` shortens from 12.35 ms to 3.60 ms (70.9%); FHD `scale=64`, `octaves=4`
  fractal and turbulent cases each shorten from about 4.4 ms to 2.16 ms (roughly 50%).
- `px.filter.directional_blur()`, `zoom_blur()`, `spin_blur()`, and `vector_blur()` use a
  noinline global fast-gather for path sample counts of 65 and above; shorter paths retain
  the shared-memory loop with the original gather via macro separation. Representative
  long-path FHD RGB cases shorten by 22 to 32% (`vector_blur` uniform |v|=128 from 10.78 ms
  to 7.30 ms; `directional_blur` length=128 from 11.00 ms to 7.52 ms; `zoom_blur` amount=0.2
  from 12.29 ms to 9.14 ms; `spin_blur` angle=10 from 10.38 ms to 8.07 ms).
- `px.color.chromatic_adaptation()`, `px.color.white_balance()`, and
  `px.color.white_point_simulation()` memoize their pure host matrix composition in a bounded
  LRU (128 identities per operation) keyed on the resolved binary64 x/y values, Frame
  colorspace, and CAT token. FHD RGB cases shorten by 27 to 49%
  (`white_point_simulation` from 0.266 ms to 0.137 ms, `white_balance` from 0.281 ms to
  0.157 ms, `chromatic_adaptation` from 0.238 ms to 0.148 ms), reaching the effective
  bandwidth of the same-shape `px.color.rgb_to_rgb` operation.
- `px.io.write_lut()`, `px.io.read_lut()`, and `px.io.decode_lut()` avoid full data-row
  materialization during Cube sniffing, replace repeated directive scans with a single
  `finditer` pass, and serialize Cube rows through a bulk zipped `%s` operation that
  preserves the `str(numpy.float32)` shortest round-trip spelling. FHD RGB 65^3 Cube write
  shortens from 918.76 ms to 323.92 ms (64.7%), decode from 452.96 ms to 154.03 ms (66.0%),
  and read from 294.87 ms to 138.60 ms (53.0%).

## 1.2.0 - 2026-08-16

pixtreme 1.2.0 adds user-supplied fonts for text drawing, expands LUT support with 1D LUTs, new file formats, and
bytes/write entry points, and introduces a white-management trio: chromatic adaptation, temperature/tint white
balance, and physical white point simulation with named reference whites. The release is purely additive; public
signatures and existing behavior are unchanged.

### Added

- `px.draw.Font` and `Font.from_file(path, face_index=0)` load user OpenType/TrueType fonts (static faces,
  variable fonts, and collections). `px.draw.text()` accepts a `Font` in place of the bundled font tokens, maps
  `weight` to the font's real `wght` axis and `variations` to its other axes, and renders missing glyphs with the
  selected face's `.notdef` glyph without fallback lookup.
- `px.core.Lut1D` is a public 1D LUT type with per-channel domains. `px.color.apply_lut()` accepts `Lut | Lut1D`
  and a new `linear` interpolation token; interpolation defaults stay `tetrahedral` for 3D and `linear` for 1D.
- `px.io.read_lut()` additionally reads `.cube` 1D, `.3dl` (Lustre and headerless dialects), `.spi1d`, and
  `.spi3d` files. `px.io.decode_lut()` parses LUT bytes with signature sniffing, and `px.io.write_lut()` writes
  deterministic `.cube` text for both LUT types with full `float32` round-trip preservation.
- `px.color.chromatic_adaptation()` adapts a `float32` RGB Frame between CIE 1931 xy white points using
  `px.core.ChromaticAdaptation` CAT tokens (`bradford`, `cat02`, `cat16`, `von-kries`; default `cat02`).
- `px.color.white_balance()` corrects a source illuminant described by temperature (Kelvin on the Planckian
  locus) and tint (signed Duv offset, positive toward green).
- `px.color.white_point_simulation()` physically re-encodes absolute colorimetry between display device whites
  (the ICC absolute-colorimetric-intent analogue) without chromatic adaptation, reducing to per-channel gain for
  same-primaries device pairs.
- `px.core.ReferenceWhite` names the reference-white tokens `d65`, `d93`, `d50`, and `aces`, accepted by
  `white_point_simulation()` and `chromatic_adaptation()` alongside direct CIE 1931 xy pairs.

## 1.1.1 - 2026-08-14

pixtreme 1.1.1 hardens the public error contract. Out-of-contract inputs that previously leaked raw interpreter
exceptions now raise the documented actionable `ValueError` diagnostics. Public signatures and successful-path
behavior are unchanged.

### Fixed

- `px.io.to_array()` and the eight wire-format exporters now reject non-Frame inputs with actionable `ValueError`
  diagnostics instead of leaking attribute errors.
- `px.io.read_image()`, `write_image()`, `read_header()`, and `read_lut()` now translate invalid path types into
  actionable `ValueError` diagnostics while retaining the underlying cause.
- `px.transform.resize()` now rejects non-finite or unrepresentable factor-derived dimensions before rounding or
  allocation and guides callers to a smaller factor or explicit dimensions.
- `px.composite.merge()` now names its actual public path when guiding callers after a non-Frame input.
- Shared finite-real validation now translates numeric conversion failures into the parameter's actionable
  `ValueError` instead of leaking `OverflowError`.
- The `px.io.to_array()` docstring now describes its function input and links to `px.values.quantize()` correctly.

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
