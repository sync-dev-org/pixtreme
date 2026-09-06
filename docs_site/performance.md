# Performance

This is the complete 204-case FHD measurement report from a single full-suite `uv run pytest -m performance` run
at pixtreme commit `ca83e10`. The run completed with 258 passed and 4,390 deselected in 3,326.84 seconds (0:55:26).
These measurements characterize this hardware and workload; they are not performance guarantees for other systems.

## Measurement conditions

- **GPU:** NVIDIA RTX A6000, driver 596.72
596.72
596.72. GPU 0 was dedicated to the run.
- **Environment:** WSL2 on Linux 6.18, CUDA runtime 12.9 (`nvidia-cuda-runtime-cu12` 12.9.79), CuPy 14.1.1,
  and Python 3.12.
- **Default input:** 1920x1080, fp32, three-channel RGB. A row's `representative parameters` are authoritative when
  the case uses a different shape, dtype, channel layout, format, or auxiliary input.
- **Sampling:** GPU-device cases use at least 1,000 consecutive frame executions and 3 seconds. Slow file/byte boundary
  cases use at least 20 executions and the same 3-second floor. Warmup and JIT compilation were excluded. The table
  reports mean, median, p5, and p95 timing; FPS is reported from the same run.
- **I/O boundaries:** `read_image`, `write_image`, `read_header`, `read_lut`, `decode_image`, and
  `encode_image` include temporary-file I/O or host-byte exchange in wall-clock time. The OS cache was not cleared.
  These measurements are not GPU-device throughput and do not represent latency for durable persistence to physical
  media.
- **Threshold marker:** `yes` in `> 1 ms` means the case median exceeded 1 ms. There are 88 such cases.

## EXR source-fixed routing

EXR read and write use pixtreme-owned implementations for all ten compression tokens. OpenEXR is a dev-only oracle,
not a runtime route. The following routing table is fixed in source: runtime capabilities, environment, and measured
performance do not alter it. The public medians use the current route after at least 0.5 seconds of excluded warmup and
at least 20 iterations and 3 seconds of measurement. All registry EXR read fixtures use HALF storage; FLOAT read
characteristics remain recorded in the all-combination adoption-gate measurements that selected the source-fixed
routes. Write cases use the current HALF default unless the fp16 input is already native HALF.

| Compression | Read lane | Write lane | Public read median (ms) | Public write median (ms) |
|---|---|---|---:|---:|
| NONE | native | GPU | 30.919 | 8.239 |
| RLE | GPU | GPU | 168.247 | 16.757 |
| ZIPS | custom CPU | GPU | 65.697 | 97.735 |
| ZIP | custom CPU | GPU | 33.141 | 32.783 |
| PIZ | GPU | GPU | 35.387 | 83.270 |
| PXR24 | custom CPU | GPU | 320.833 | 40.804 |
| B44 | GPU | GPU | 70.414 | 5.789 |
| B44A | GPU | GPU | 99.651 | 6.543 |
| DWAA | GPU | GPU | 26.957 | 90.543 |
| DWAB | GPU | GPU | 26.575 | 33.259 |

The default float32-frame write case omits `dtype`, stores ZIP-compressed HALF, and measured 40.614 ms. Reading that
HALF fixture unchanged through the fixed custom CPU ZIP lane measured 37.909 ms. These general default-path cases use
a different deterministic corpus from the compression rows above.

## Full results

| target | representative parameters | mean ms | median ms | fps | p5 ms | p95 ms | effective GB/s | > 1 ms |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| copy | FHD fp32 RGB read+write | 0.101 | 0.096 | 10413.4 | 0.094 | 0.117 | 518.2 |  |
| resize | 1920x1080 -> 960x540, interpolation=nearest | 0.076 | 0.073 | 13755.0 | 0.069 | 0.091 | 427.8 |  |
| resize | 1920x1080 -> 960x540, interpolation=bilinear | 0.094 | 0.091 | 11041.8 | 0.089 | 0.113 | 343.4 |  |
| resize | 1920x1080 -> 960x540, interpolation=bicubic | 0.140 | 0.135 | 7400.6 | 0.132 | 0.162 | 230.2 |  |
| resize | 1920x1080 -> 960x540, interpolation=b-spline | 0.155 | 0.139 | 7181.3 | 0.132 | 0.202 | 223.4 |  |
| resize | 1920x1080 -> 960x540, interpolation=mitchell | 0.147 | 0.136 | 7358.2 | 0.132 | 0.197 | 228.9 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos2 | 0.140 | 0.134 | 7467.1 | 0.131 | 0.165 | 232.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos3 | 0.143 | 0.138 | 7230.5 | 0.135 | 0.165 | 224.9 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos4 | 0.223 | 0.217 | 4599.2 | 0.213 | 0.246 | 143.1 |  |
| resize | 1920x1080 -> 960x540, interpolation=area | 0.158 | 0.154 | 6508.3 | 0.151 | 0.178 | 202.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=nearest | 0.293 | 0.289 | 3465.7 | 0.285 | 0.308 | 431.2 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bilinear | 0.258 | 0.256 | 3910.9 | 0.248 | 0.272 | 486.6 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bicubic | 0.460 | 0.457 | 2187.3 | 0.452 | 0.473 | 272.1 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=b-spline | 0.464 | 0.460 | 2174.9 | 0.452 | 0.496 | 270.6 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=mitchell | 0.460 | 0.457 | 2188.1 | 0.453 | 0.473 | 272.2 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos2 | 0.463 | 0.460 | 2175.9 | 0.454 | 0.476 | 270.7 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos3 | 0.563 | 0.554 | 1804.9 | 0.540 | 0.604 | 224.6 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos4 | 0.945 | 0.935 | 1069.2 | 0.919 | 0.984 | 133.0 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=area | 0.661 | 0.659 | 1517.6 | 0.651 | 0.676 | 188.8 |  |
| warp_affine | FHD fp32 RGB, centered 1.01x scale + 5deg rotation, auto lanczos4, constant 0 | 3.084 | 3.077 | 325.0 | 3.062 | 3.122 | 16.2 | yes |
| stack | 2x FHD fp32 RGB, direction=vertical, adapt=False | 0.206 | 0.200 | 4988.4 | 0.197 | 0.231 | 496.5 |  |
| shuffle | single FHD fp32 Frame BGR reorder, adapt=False | 0.131 | 0.126 | 7918.8 | 0.124 | 0.150 | 394.1 |  |
| shuffle | FHD fp32 RGBA from 2 Frames + constant, adapt=False | 0.188 | 0.182 | 5498.5 | 0.177 | 0.215 | 319.2 |  |
| shuffle | 2 FHD fp32 RGB Frames, sRGB/sRGB source adapted to ACEScg/linear | 0.267 | 0.258 | 3876.7 | 0.253 | 0.319 | 385.9 |  |
| merge | FHD background + transformed 960x540 foreground, bilinear, normal | 0.452 | 0.442 | 2261.7 | 0.432 | 0.500 | 126.6 |  |
| gaussian_blur | sigma=1 | 0.552 | 0.549 | 1821.5 | 0.544 | 0.566 | 90.7 |  |
| gaussian_blur | sigma=2 | 0.645 | 0.642 | 1556.7 | 0.636 | 0.660 | 77.5 |  |
| gaussian_blur | sigma=4 | 0.884 | 0.883 | 1132.6 | 0.874 | 0.897 | 56.4 |  |
| unsharp_mask | sigma=2, amount=1 | 0.750 | 0.748 | 1337.7 | 0.741 | 0.762 | 66.6 |  |
| box_blur | size=3 | 0.466 | 0.464 | 2155.3 | 0.458 | 0.478 | 107.3 |  |
| box_blur | size=9 | 0.525 | 0.521 | 1920.5 | 0.515 | 0.542 | 95.6 |  |
| median_blur | size=3 | 0.334 | 0.331 | 3017.0 | 0.329 | 0.345 | 150.1 |  |
| median_blur | size=5 | 0.721 | 0.717 | 1394.2 | 0.714 | 0.736 | 69.4 |  |
| median_blur | size=7 | 0.976 | 0.974 | 1026.4 | 0.965 | 0.991 | 51.1 |  |
| bilateral_blur | sigma_space=1, sigma_value=0.1 | 0.391 | 0.388 | 2574.4 | 0.384 | 0.402 | 128.1 |  |
| bilateral_blur | sigma_space=2, sigma_value=0.1 | 0.867 | 0.867 | 1153.7 | 0.848 | 0.893 | 57.4 |  |
| convolve_box | size=(1,31), normalize=True | 0.552 | 0.550 | 1817.4 | 0.533 | 0.581 | 90.4 |  |
| erosion | radius=5, shape=disk | 0.333 | 0.332 | 3009.6 | 0.316 | 0.363 | 149.8 |  |
| dilation | radius=5, shape=disk | 0.330 | 0.329 | 3037.8 | 0.317 | 0.349 | 151.2 |  |
| opening | radius=5, shape=disk | 0.595 | 0.591 | 1691.3 | 0.588 | 0.608 | 84.2 |  |
| closing | radius=5, shape=disk | 0.590 | 0.587 | 1704.6 | 0.581 | 0.603 | 84.8 |  |
| morphological_gradient | radius=5, shape=disk | 0.347 | 0.344 | 2909.5 | 0.340 | 0.362 | 144.8 |  |
| white_tophat | radius=5, shape=disk | 0.594 | 0.592 | 1689.9 | 0.584 | 0.610 | 84.1 |  |
| black_tophat | radius=5, shape=disk | 0.591 | 0.587 | 1703.5 | 0.579 | 0.611 | 84.8 |  |
| sobel | direction=x | 0.471 | 0.469 | 2132.0 | 0.460 | 0.490 | 106.1 |  |
| sobel | direction=y | 0.491 | 0.487 | 2055.5 | 0.467 | 0.537 | 102.3 |  |
| sobel | direction=magnitude | 0.489 | 0.487 | 2055.0 | 0.476 | 0.510 | 102.3 |  |
| laplacian | kernel=3x3 | 0.292 | 0.288 | 3471.4 | 0.284 | 0.308 | 172.8 |  |
| canny | threshold_low=0.5, threshold_high=1.0, border=mirror | 2.286 | 2.257 | 443.1 | 2.189 | 2.414 | 22.1 | yes |
| sharpen | amount=1, border=mirror | 0.479 | 0.479 | 2089.3 | 0.457 | 0.496 | 104.0 |  |
| difference_of_gaussians | sigma1=1, sigma2=2 | 1.256 | 1.252 | 798.7 | 1.239 | 1.275 | 39.7 | yes |
| corner_harris | FHD fp32 RGB, block_size=3, k=0.04, border=mirror | 0.561 | 0.559 | 1787.9 | 0.542 | 0.584 | 59.3 |  |
| match_template | FHD fp32 RGB + 64x64 fp32 RGB, method=ccoeff_normed | 12.820 | 12.828 | 78.0 | 12.635 | 13.064 | 2.5 | yes |
| psnr | FHD fp32 RGB reference/candidate, data_range=1.0 default | 0.341 | 0.338 | 2961.1 | 0.274 | 0.439 | 147.4 |  |
| ssim | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.055 | 2.052 | 487.2 | 2.035 | 2.079 | 24.2 | yes |
| ssim_map | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.041 | 2.039 | 490.5 | 2.020 | 2.081 | 28.4 | yes |
| equalize_histogram | domain=(0,1), bins=1024 | 0.897 | 0.879 | 1137.7 | 0.863 | 0.965 | 56.6 |  |
| clahe | clip_limit=2, tiles_y=8, tiles_x=8, domain=(0,1), bins=1024 | 2.947 | 2.937 | 340.5 | 2.908 | 3.014 | 16.9 | yes |
| directional_blur | angle=30, length=8 | 0.512 | 0.508 | 1966.8 | 0.502 | 0.544 | 97.9 |  |
| directional_blur | angle=30, length=32 | 1.568 | 1.567 | 638.2 | 1.559 | 1.583 | 31.8 | yes |
| directional_blur | angle=30, length=128 | 7.432 | 7.425 | 134.7 | 7.335 | 7.558 | 6.7 | yes |
| zoom_blur | amount=0.05 | 1.882 | 1.879 | 532.1 | 1.859 | 1.907 | 26.5 | yes |
| zoom_blur | amount=0.2 | 9.051 | 9.043 | 110.6 | 8.923 | 9.222 | 5.5 | yes |
| spin_blur | angle=2 | 1.229 | 1.228 | 814.6 | 1.218 | 1.245 | 40.5 | yes |
| spin_blur | angle=10 | 7.966 | 7.944 | 125.9 | 7.838 | 8.152 | 6.3 | yes |
| vector_blur | uniform \|v\|=8, shutter=centered | 0.802 | 0.805 | 1242.5 | 0.786 | 0.815 | 82.4 |  |
| vector_blur | uniform \|v\|=32, shutter=centered | 1.787 | 1.785 | 560.2 | 1.769 | 1.805 | 37.2 | yes |
| vector_blur | uniform \|v\|=128, shutter=centered | 7.315 | 7.301 | 137.0 | 7.236 | 7.446 | 9.1 | yes |
| vector_blur | rotation field, corner \|v\|=32, shutter=centered | 1.133 | 1.132 | 883.4 | 1.120 | 1.150 | 58.6 | yes |
| lens_blur | circle radius=4 | 0.709 | 0.706 | 1417.4 | 0.701 | 0.724 | 70.5 |  |
| lens_blur | circle radius=8 | 1.547 | 1.544 | 647.8 | 1.534 | 1.563 | 32.2 | yes |
| lens_blur | circle radius=16 | 1.553 | 1.550 | 645.3 | 1.544 | 1.566 | 32.1 | yes |
| lens_blur | circle radius=32 | 1.346 | 1.342 | 745.3 | 1.338 | 1.360 | 37.1 | yes |
| lens_blur | blades=6, radius=16 | 1.552 | 1.549 | 645.7 | 1.543 | 1.564 | 32.1 | yes |
| lens_blur | blades=6, radius=32 | 1.348 | 1.343 | 744.7 | 1.338 | 1.362 | 37.1 | yes |
| line | diagonal thickness=4, aa=distance | 0.167 | 0.162 | 6166.0 | 0.160 | 0.190 | 306.9 |  |
| polyline | 5 points, closed, thickness=6, aa=distance | 0.198 | 0.191 | 5245.1 | 0.187 | 0.238 | 261.0 |  |
| rectangle | 1280x720 fill, corner_radius=48, aa=distance | 0.200 | 0.192 | 5217.5 | 0.186 | 0.224 | 259.7 |  |
| circle | fill radius=320, aa=supersample | 0.191 | 0.193 | 5178.5 | 0.168 | 0.214 | 257.7 |  |
| ellipse | radii=(520,260), rotation=25, thickness=8 | 0.166 | 0.162 | 6189.0 | 0.158 | 0.190 | 308.0 |  |
| polygon | 8-point concave fill, aa=distance | 0.225 | 0.220 | 4545.1 | 0.217 | 0.248 | 226.2 |  |
| text | single-line CJK, size=64, one outline, supersample=False | 0.385 | 0.358 | 2794.5 | 0.335 | 0.485 | 139.1 |  |
| text | single-line CJK, size=64, one outline, supersample=True | 0.409 | 0.391 | 2555.6 | 0.363 | 0.495 | 127.2 |  |
| ramp | FHD linear RGB | 0.135 | 0.130 | 7689.4 | 0.127 | 0.159 | 191.3 |  |
| grid | FHD cell=(64,64), line_width=2, aa=distance | 0.134 | 0.130 | 7721.7 | 0.127 | 0.155 | 192.1 |  |
| checkerboard | FHD cell=(64,64), aa=distance | 0.133 | 0.129 | 7751.8 | 0.127 | 0.153 | 192.9 |  |
| color_bars | FHD ARIB STD-B28 normalized | 0.088 | 0.085 | 11745.6 | 0.084 | 0.103 | 292.3 |  |
| fractal_noise | FHD scale=64, octaves=4 | 2.150 | 2.149 | 465.4 | 2.139 | 2.169 | 3.9 | yes |
| turbulent_noise | FHD scale=64, octaves=4 | 2.149 | 2.148 | 465.6 | 2.139 | 2.164 | 3.9 | yes |
| grain | FHD intensity=0.1, size=1, RGB | 3.576 | 3.572 | 280.0 | 3.560 | 3.610 | 7.0 | yes |
| from_array | CHW + affine scale=255 -> float32 HWC | 0.146 | 0.141 | 7073.3 | 0.139 | 0.169 | 352.0 |  |
| from_array | CHW uint16, bit_depth=10 -> float32 HWC | 0.127 | 0.123 | 8162.1 | 0.120 | 0.148 | 304.6 |  |
| to_array | BGR + NCHW + float16 + affine | 0.121 | 0.117 | 8548.5 | 0.114 | 0.143 | 319.1 |  |
| to_array | bit_depth=10 -> uint16 HWC | 0.111 | 0.107 | 9334.9 | 0.104 | 0.132 | 348.4 |  |
| rgb_to_rgb | ACEScg linear -> sRGB sRGB | 0.138 | 0.134 | 7486.7 | 0.131 | 0.159 | 372.6 |  |
| rgb_to_rgb | ACES 1.3 analytic -> sRGB sRGB | 0.170 | 0.165 | 6057.9 | 0.154 | 0.199 | 301.5 |  |
| rgb_to_rgb | ACES 2.0 analytic -> sRGB sRGB | 0.165 | 0.157 | 6377.6 | 0.149 | 0.193 | 317.4 |  |
| rgb_to_rgb | BT.2408 direct mapping -> Rec.2020 pq | 0.150 | 0.143 | 6990.0 | 0.140 | 0.179 | 347.9 |  |
| chromatic_adaptation | FHD fp32 RGB, D50 input -> D60 output, CAT02 | 0.171 | 0.170 | 5884.3 | 0.143 | 0.200 | 292.8 |  |
| white_balance | FHD fp32 RGB, Temperature=5000 K, Tint=0 Duv, CAT02 | 0.184 | 0.184 | 5447.7 | 0.155 | 0.214 | 271.1 |  |
| white_point_simulation | FHD fp32 RGB, D65 input display -> D93 output display | 0.160 | 0.160 | 6266.5 | 0.133 | 0.188 | 311.9 |  |
| rgb_to_ycbcr | RGB -> YCbCr, matrix=native | 0.151 | 0.146 | 6865.4 | 0.143 | 0.172 | 341.7 |  |
| rgb_to_hsv | RGB -> HSV, label-driven scene values | 0.118 | 0.113 | 8814.3 | 0.111 | 0.135 | 438.7 |  |
| hsv_to_rgb | HSV six sectors, S=[0,1], V=[0,2] -> RGB | 0.123 | 0.116 | 8596.9 | 0.110 | 0.146 | 427.8 |  |
| ycbcr_to_rgb | YCbCr -> RGB, matrix=bt709 | 0.157 | 0.150 | 6685.5 | 0.142 | 0.184 | 332.7 |  |
| rgb_to_grayscale | RGB -> Y, matrix=native | 0.127 | 0.123 | 8147.3 | 0.120 | 0.147 | 270.3 |  |
| gamma_to_linear | gamma=Gamma-2.6 claim -> linear | 0.153 | 0.145 | 6900.4 | 0.138 | 0.178 | 343.4 |  |
| linear_to_gamma | linear -> gamma=Gamma-2.6 | 0.153 | 0.148 | 6768.2 | 0.139 | 0.179 | 336.8 |  |
| ycbcr_to_ycbcr | YCbCr bt709 -> native rematrix | 0.153 | 0.148 | 6742.7 | 0.146 | 0.179 | 335.6 |  |
| full_to_legal | full -> legal, bit_depth=10 | 0.135 | 0.137 | 7308.6 | 0.119 | 0.159 | 363.7 |  |
| legal_to_full | legal -> full, bit_depth=10 | 0.133 | 0.125 | 8012.8 | 0.120 | 0.157 | 398.8 |  |
| quantize | float32 -> uint8, bit_depth=8 | 0.101 | 0.103 | 9756.0 | 0.086 | 0.121 | 303.5 |  |
| dequantize | uint8 -> float32, bit_depth=8 | 0.107 | 0.109 | 9205.3 | 0.087 | 0.138 | 286.3 |  |
| cast_dtype | float32 -> float16 | 0.106 | 0.100 | 9986.2 | 0.092 | 0.126 | 372.7 |  |
| recode_dtype | uint8 -> float32 | 0.096 | 0.092 | 10813.2 | 0.090 | 0.116 | 336.3 |  |
| recode_dtype | float32 -> uint8 | 0.107 | 0.110 | 9114.6 | 0.086 | 0.131 | 283.5 |  |
| from_uyvy422 | legal range | 0.114 | 0.115 | 8689.3 | 0.091 | 0.139 | 252.3 |  |
| from_v210 | legal range | 0.109 | 0.114 | 8800.4 | 0.092 | 0.140 | 267.6 |  |
| from_nv12 | legal range, siting=left, interpolation=bilinear | 0.112 | 0.113 | 8856.8 | 0.090 | 0.161 | 247.9 |  |
| from_p010 | legal range, siting=left, interpolation=bilinear | 0.114 | 0.098 | 10209.8 | 0.095 | 0.172 | 317.6 |  |
| from_yuv420p | legal range, interpolation=bilinear | 0.120 | 0.112 | 8891.2 | 0.090 | 0.177 | 248.9 |  |
| from_yuv422p | legal range | 0.120 | 0.120 | 8349.5 | 0.096 | 0.170 | 277.0 |  |
| from_yuv444p | 10-bit legal range | 0.124 | 0.105 | 9535.4 | 0.097 | 0.175 | 355.9 |  |
| from_yuva444p | 12-bit legal range | 0.129 | 0.121 | 8240.8 | 0.117 | 0.160 | 410.1 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=trilinear | 0.361 | 0.356 | 2808.8 | 0.353 | 0.380 | 139.8 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=tetrahedral | 0.357 | 0.353 | 2831.3 | 0.350 | 0.373 | 140.9 |  |
| apply_lut | FHD fp32 RGB, 65-sample 1D LUT, interpolation=linear | 0.357 | 0.350 | 2857.9 | 0.346 | 0.393 | 142.2 |  |
| to_uyvy422 | FHD area, legal | 0.090 | 0.086 | 11581.4 | 0.084 | 0.108 | 336.2 |  |
| to_v210 | FHD area, legal, 128-byte rows | 0.152 | 0.147 | 6801.6 | 0.144 | 0.173 | 206.9 |  |
| to_nv12 | FHD area, legal, siting=left | 0.139 | 0.134 | 7470.0 | 0.131 | 0.158 | 209.1 |  |
| to_p010 | FHD area, legal, siting=left | 0.143 | 0.137 | 7314.1 | 0.133 | 0.165 | 227.5 |  |
| to_yuv420p | 8-bit area, legal, siting=left | 0.139 | 0.134 | 7441.9 | 0.132 | 0.159 | 208.3 |  |
| to_yuv422p | 10-bit area, legal | 0.125 | 0.121 | 8273.1 | 0.119 | 0.146 | 274.5 |  |
| to_yuv444p | 10-bit legal | 0.098 | 0.094 | 10617.6 | 0.092 | 0.116 | 396.3 |  |
| to_yuva444p | 12-bit legal, alpha full | 0.115 | 0.111 | 8991.4 | 0.108 | 0.134 | 447.5 |  |
| read_lut | 65^3 RGB .cube file, parse, float4 packing, and host-to-device transfer included | 136.170 | 134.987 | 7.4 | 133.540 | 144.879 | 0.1 | yes |
| read_lut | 65-sample RGB Cube 1D file, parse and host-to-device transfer included | 0.243 | 0.231 | 4335.5 | 0.226 | 0.286 | 0.0 |  |
| read_lut | 17^3 RGB headerless 3DL file, parse, packing, and host-to-device transfer included | 8.267 | 8.247 | 121.3 | 7.996 | 8.561 | 0.0 | yes |
| read_lut | 65-sample RGB SPI1D file, parse and host-to-device transfer included | 0.204 | 0.196 | 5090.5 | 0.193 | 0.238 | 0.0 |  |
| read_lut | 17^3 RGB SPI3D file, explicit-index parse, packing, and host-to-device transfer included | 8.781 | 8.769 | 114.0 | 8.556 | 9.017 | 0.0 | yes |
| write_lut | 65-sample RGB Lut1D, device-to-host transfer and Cube file write included | 0.295 | 0.283 | 3536.6 | 0.268 | 0.334 | 0.0 |  |
| write_lut | 65^3 RGB Lut, device-to-host transfer and Cube file write included | 328.474 | 328.190 | 3.0 | 325.667 | 330.313 | 0.0 | yes |
| read_image | FHD uint8 RGB PNG file, unchanged, temporary-file I/O included | 25.160 | 25.139 | 39.8 | 24.518 | 25.904 | 0.5 | yes |
| read_image | FHD uint8 RGB JPEG file, unchanged, temporary-file I/O included | 37.995 | 37.990 | 26.3 | 36.962 | 38.887 | 0.3 | yes |
| read_image | FHD uint8 RGB TIFF file, unchanged, temporary-file I/O included | 32.050 | 32.042 | 31.2 | 31.452 | 32.668 | 0.4 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 37.904 | 37.909 | 26.4 | 37.026 | 38.899 | 0.7 | yes |
| read_image | FHD HALF RGB EXR NONE file, unchanged, source-fixed native lane, temporary-file I/O included | 32.109 | 30.919 | 32.3 | 30.706 | 31.757 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 33.167 | 33.141 | 30.2 | 32.390 | 34.183 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIPS file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 65.664 | 65.697 | 15.2 | 63.852 | 67.166 | 0.4 | yes |
| read_image | FHD HALF RGB EXR DWAA file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 27.047 | 26.957 | 37.1 | 23.626 | 30.209 | 0.9 | yes |
| read_image | FHD HALF RGB EXR DWAB file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 26.702 | 26.575 | 37.6 | 25.263 | 28.442 | 0.9 | yes |
| read_image | FHD HALF RGB EXR RLE file, unchanged, source-fixed GPU lane, temporary-file I/O included | 182.663 | 168.247 | 5.9 | 160.272 | 226.780 | 0.1 | yes |
| read_image | FHD HALF RGB EXR PXR24 file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 331.418 | 320.833 | 3.1 | 299.751 | 370.715 | 0.1 | yes |
| read_image | FHD HALF RGB EXR B44 file, unchanged, source-fixed GPU lane, temporary-file I/O included | 82.542 | 70.414 | 14.2 | 69.638 | 127.428 | 0.4 | yes |
| read_image | FHD HALF RGB EXR B44A file, unchanged, source-fixed GPU lane, temporary-file I/O included | 111.786 | 99.651 | 10.0 | 97.784 | 156.419 | 0.2 | yes |
| read_image | FHD HALF RGB EXR PIZ file, unchanged, source-fixed GPU lane, temporary-file I/O included | 35.448 | 35.387 | 28.3 | 34.740 | 36.348 | 0.7 | yes |
| read_image | FHD uint8 RGB JPEG 2000 file, unchanged, temporary-file I/O included | 25.141 | 24.909 | 40.1 | 24.600 | 26.056 | 0.5 | yes |
| read_image | FHD uint8 RGB WebP file, unchanged, temporary-file I/O included | 41.623 | 41.443 | 24.1 | 40.536 | 43.212 | 0.3 | yes |
| read_image | FHD uint8 RGB BMP file, unchanged, temporary-file I/O included | 17.595 | 17.564 | 56.9 | 17.147 | 18.240 | 0.7 | yes |
| read_image | FHD uint8 RGB PNM file, unchanged, temporary-file I/O included | 18.632 | 18.590 | 53.8 | 18.134 | 19.269 | 0.7 | yes |
| read_image | FHD uint8 RGB TGA file, unchanged, temporary-file I/O and CPU RLE included | 12.187 | 12.173 | 82.1 | 11.907 | 12.509 | 1.0 | yes |
| read_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 132.230 | 132.381 | 7.6 | 129.772 | 135.580 | 0.3 | yes |
| read_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU unpack included | 5.212 | 5.143 | 194.4 | 4.889 | 5.590 | 6.5 | yes |
| write_image | FHD uint8 RGB PNG file, compression_level=4, temporary-file I/O included | 198.381 | 197.928 | 5.1 | 196.352 | 201.130 | 0.1 | yes |
| write_image | FHD uint8 RGB JPEG file, quality=95, temporary-file I/O included | 41.506 | 41.172 | 24.3 | 39.354 | 44.292 | 0.3 | yes |
| write_image | FHD uint8 RGB TIFF file, temporary-file I/O included | 56.524 | 55.439 | 18.0 | 53.078 | 61.591 | 0.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 40.550 | 40.614 | 24.6 | 38.416 | 42.397 | 0.9 | yes |
| write_image | FHD fp32 RGB to EXR NONE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 8.254 | 8.239 | 121.4 | 6.447 | 9.490 | 4.5 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 33.403 | 32.783 | 30.5 | 30.916 | 35.912 | 1.1 | yes |
| write_image | FHD fp32 RGB to EXR ZIPS/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 101.060 | 97.735 | 10.2 | 87.587 | 128.069 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAA/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 88.250 | 90.543 | 11.0 | 79.567 | 96.031 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAB/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 33.156 | 33.259 | 30.1 | 29.715 | 38.280 | 1.1 | yes |
| write_image | FHD fp32 RGB to EXR RLE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 17.057 | 16.757 | 59.7 | 15.287 | 19.433 | 2.2 | yes |
| write_image | FHD fp32 RGB to EXR PXR24/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 40.935 | 40.804 | 24.5 | 40.152 | 42.202 | 0.9 | yes |
| write_image | FHD fp16 RGB to EXR B44/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 6.052 | 5.789 | 172.7 | 5.156 | 7.217 | 4.3 | yes |
| write_image | FHD fp16 RGB to EXR B44A/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 6.682 | 6.543 | 152.8 | 5.337 | 8.162 | 3.8 | yes |
| write_image | FHD fp16 RGB to EXR PIZ/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 83.124 | 83.270 | 12.0 | 80.189 | 86.277 | 0.3 | yes |
| write_image | FHD uint8 RGB JPEG 2000 file, lossless, temporary-file I/O included | 64.183 | 64.084 | 15.6 | 62.593 | 66.070 | 0.2 | yes |
| write_image | FHD uint8 RGB WebP file, lossless, temporary-file I/O included | 311.061 | 312.245 | 3.2 | 300.972 | 320.474 | 0.0 | yes |
| write_image | FHD uint8 RGB BMP file, temporary-file I/O included | 24.838 | 24.783 | 40.4 | 23.343 | 26.264 | 0.5 | yes |
| write_image | FHD uint8 RGB PNM file, temporary-file I/O included | 27.188 | 27.134 | 36.9 | 25.241 | 29.037 | 0.5 | yes |
| write_image | FHD uint8 RGB TGA file, temporary-file I/O and CPU RLE included | 53.209 | 53.162 | 18.8 | 52.803 | 53.914 | 0.2 | yes |
| write_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 314.092 | 317.911 | 3.1 | 306.467 | 320.768 | 0.1 | yes |
| write_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU packing included | 10.555 | 10.442 | 95.8 | 10.182 | 10.969 | 3.2 | yes |
| read_header | FHD uint8 RGB PNG header, temporary-file I/O included | 2.095 | 2.089 | 478.6 | 2.050 | 2.160 | 0.0 | yes |
| decode_lut | 65-sample RGB Cube 1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.190 | 0.181 | 5510.8 | 0.178 | 0.237 | 0.0 |  |
| decode_lut | 65^3 RGB Cube 3D UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 145.519 | 144.398 | 6.9 | 143.079 | 151.134 | 0.1 | yes |
| decode_lut | 17^3 RGB headerless 3DL UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 15.078 | 15.003 | 66.7 | 14.646 | 15.749 | 0.0 | yes |
| decode_lut | 65-sample RGB SPI1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.187 | 0.177 | 5664.4 | 0.170 | 0.229 | 0.0 |  |
| decode_lut | 17^3 RGB SPI3D UTF-8 bytes, sniff, explicit-index parse, and host-to-device transfer included | 9.959 | 9.878 | 101.2 | 9.641 | 10.532 | 0.0 | yes |
| decode_image | FHD uint8 RGB PNG, unchanged, host bytes exchange included | 23.573 | 23.528 | 42.5 | 23.122 | 24.336 | 0.5 | yes |
| decode_image | FHD uint8 RGB JPEG, unchanged, host bytes exchange included | 39.258 | 39.351 | 25.4 | 38.079 | 40.492 | 0.3 | yes |
| decode_image | FHD uint8 RGB TIFF, unchanged, host bytes exchange included | 32.041 | 32.081 | 31.2 | 31.606 | 32.448 | 0.4 | yes |
| decode_image | FHD uint8 RGB JPEG 2000, unchanged, host bytes exchange included | 23.583 | 23.473 | 42.6 | 23.183 | 24.292 | 0.5 | yes |
| decode_image | FHD uint8 RGB WebP, unchanged, host bytes exchange included | 39.249 | 39.346 | 25.4 | 38.311 | 40.422 | 0.3 | yes |
| decode_image | FHD uint8 RGB BMP, unchanged, host bytes exchange included | 16.426 | 16.341 | 61.2 | 15.905 | 17.199 | 0.8 | yes |
| decode_image | FHD uint8 RGB PNM, unchanged, host bytes exchange included | 17.902 | 17.786 | 56.2 | 17.381 | 18.437 | 0.7 | yes |
| encode_image | FHD uint8 RGB PNG, compression_level=4, host bytes exchange included | 193.943 | 193.278 | 5.2 | 191.168 | 198.246 | 0.1 | yes |
| encode_image | FHD uint8 RGB JPEG, quality=95, host bytes exchange included | 39.123 | 39.148 | 25.5 | 37.510 | 40.605 | 0.3 | yes |
| encode_image | FHD uint8 RGB TIFF, host bytes exchange included | 53.474 | 52.670 | 19.0 | 49.841 | 60.008 | 0.2 | yes |
| encode_image | FHD uint8 RGB JPEG 2000, lossless, host bytes exchange included | 64.710 | 64.819 | 15.4 | 61.640 | 67.389 | 0.2 | yes |
| encode_image | FHD uint8 RGB WebP, lossless, host bytes exchange included | 306.134 | 308.056 | 3.2 | 298.082 | 314.397 | 0.0 | yes |
| encode_image | FHD uint8 RGB BMP, host bytes exchange included | 21.982 | 21.875 | 45.7 | 20.346 | 23.682 | 0.6 | yes |
| encode_image | FHD uint8 RGB PNM, host bytes exchange included | 24.077 | 24.151 | 41.4 | 22.259 | 26.053 | 0.5 | yes |
