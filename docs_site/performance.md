# Performance

This is the complete 206-case FHD measurement report from a single full-suite `uv run pytest -m performance` run
at pixtreme commit `d217e6f`. The run completed with 261 passed, 1 failed, and 5,043 deselected in 3,214.02 seconds
(0:53:34). The one failure is a separate 31-sample absolute-limit gate for the ACES 1.3 analytic path, not a
registry case: it measured 0.217 ms against its 0.20 ms limit inside the suite and passed on three isolated re-runs
on the same GPU, while the registry row for that path below comes from the same full run.
These measurements characterize this hardware and workload; they are not performance guarantees for other systems.

## Measurement conditions

- **GPU:** NVIDIA RTX A6000, driver 596.72. GPU 0 was dedicated to the run.
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
- **Threshold marker:** `yes` in `> 1 ms` means the case median exceeded 1 ms. There are 89 such cases.

## EXR source-fixed routing

EXR read and write use pixtreme-owned implementations for all ten compression tokens. OpenEXR is a dev-only oracle,
not a runtime route. The following routing table is fixed in source: runtime capabilities, environment, and measured
performance do not alter it. The public medians use the current route after at least 0.5 seconds of excluded warmup and
at least 20 iterations and 3 seconds of measurement. All registry EXR read fixtures use HALF storage; FLOAT read
characteristics remain recorded in the all-combination adoption-gate measurements that selected the source-fixed
routes. Write cases use the current HALF default unless the fp16 input is already native HALF.

| Compression | Read lane | Write lane | Public read median (ms) | Public write median (ms) |
|---|---|---|---:|---:|
| NONE | native | GPU | 30.473 | 8.076 |
| RLE | GPU | GPU | 162.042 | 17.131 |
| ZIPS | custom CPU | GPU | 63.678 | 106.632 |
| ZIP | custom CPU | GPU | 33.078 | 35.355 |
| PIZ | GPU | GPU | 33.939 | 85.830 |
| PXR24 | custom CPU | GPU | 341.551 | 46.104 |
| B44 | GPU | GPU | 71.853 | 5.782 |
| B44A | GPU | GPU | 96.095 | 6.910 |
| DWAA | GPU | GPU | 23.922 | 85.176 |
| DWAB | GPU | GPU | 25.136 | 33.067 |

The default float32-frame write case omits `dtype`, stores ZIP-compressed HALF, and measured 37.539 ms. Reading that
HALF fixture unchanged through the fixed custom CPU ZIP lane measured 40.360 ms. These general default-path cases use
a different deterministic corpus from the compression rows above.

## Full results

| target | representative parameters | mean ms | median ms | fps | p5 ms | p95 ms | effective GB/s | > 1 ms |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| copy | FHD fp32 RGB read+write | 0.103 | 0.101 | 9916.2 | 0.094 | 0.122 | 493.5 |  |
| resize | 1920x1080 -> 960x540, interpolation=nearest | 0.075 | 0.069 | 14413.6 | 0.067 | 0.097 | 448.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=bilinear | 0.095 | 0.090 | 11171.1 | 0.085 | 0.116 | 347.5 |  |
| resize | 1920x1080 -> 960x540, interpolation=bicubic | 0.139 | 0.135 | 7425.2 | 0.130 | 0.159 | 231.0 |  |
| resize | 1920x1080 -> 960x540, interpolation=b-spline | 0.141 | 0.136 | 7374.3 | 0.132 | 0.164 | 229.4 |  |
| resize | 1920x1080 -> 960x540, interpolation=mitchell | 0.140 | 0.135 | 7432.4 | 0.132 | 0.162 | 231.2 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos2 | 0.141 | 0.137 | 7314.4 | 0.133 | 0.163 | 227.5 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos3 | 0.143 | 0.138 | 7222.2 | 0.135 | 0.164 | 224.6 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos4 | 0.227 | 0.224 | 4470.3 | 0.214 | 0.253 | 139.0 |  |
| resize | 1920x1080 -> 960x540, interpolation=area | 0.169 | 0.161 | 6219.3 | 0.152 | 0.208 | 193.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=nearest | 0.295 | 0.295 | 3388.8 | 0.284 | 0.306 | 421.6 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bilinear | 0.260 | 0.259 | 3859.7 | 0.249 | 0.274 | 480.2 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bicubic | 0.464 | 0.462 | 2165.1 | 0.452 | 0.482 | 269.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=b-spline | 0.469 | 0.467 | 2143.0 | 0.453 | 0.491 | 266.6 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=mitchell | 0.464 | 0.463 | 2162.0 | 0.453 | 0.480 | 269.0 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos2 | 0.465 | 0.463 | 2159.0 | 0.454 | 0.479 | 268.6 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos3 | 0.559 | 0.558 | 1793.3 | 0.547 | 0.577 | 223.1 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos4 | 0.944 | 0.942 | 1062.1 | 0.929 | 0.962 | 132.1 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=area | 0.663 | 0.660 | 1514.8 | 0.654 | 0.676 | 188.5 |  |
| warp_affine | FHD fp32 RGB, centered 1.01x scale + 5deg rotation, auto lanczos4, constant 0 | 3.110 | 3.101 | 322.4 | 3.081 | 3.178 | 16.0 | yes |
| stack | 2x FHD fp32 RGB, direction=vertical, adapt=False | 0.209 | 0.202 | 4948.5 | 0.198 | 0.237 | 492.5 |  |
| shuffle | single FHD fp32 Frame BGR reorder, adapt=False | 0.133 | 0.128 | 7814.5 | 0.124 | 0.159 | 388.9 |  |
| shuffle | FHD fp32 RGBA from 2 Frames + constant, adapt=False | 0.197 | 0.185 | 5396.3 | 0.179 | 0.254 | 313.3 |  |
| shuffle | 2 FHD fp32 RGB Frames, sRGB/sRGB source adapted to ACEScg/linear | 0.284 | 0.279 | 3582.4 | 0.255 | 0.337 | 356.6 |  |
| merge | FHD background + transformed 960x540 foreground, bilinear, normal | 0.506 | 0.509 | 1964.4 | 0.438 | 0.584 | 110.0 |  |
| gaussian_blur | sigma=1 | 0.561 | 0.560 | 1784.4 | 0.546 | 0.585 | 88.8 |  |
| gaussian_blur | sigma=2 | 0.648 | 0.646 | 1548.2 | 0.637 | 0.666 | 77.0 |  |
| gaussian_blur | sigma=4 | 0.888 | 0.888 | 1126.7 | 0.878 | 0.904 | 56.1 |  |
| unsharp_mask | sigma=2, amount=1 | 0.754 | 0.751 | 1331.4 | 0.744 | 0.770 | 66.3 |  |
| box_blur | size=3 | 0.471 | 0.463 | 2158.7 | 0.456 | 0.510 | 107.4 |  |
| box_blur | size=9 | 0.526 | 0.521 | 1921.1 | 0.515 | 0.546 | 95.6 |  |
| median_blur | size=3 | 0.338 | 0.333 | 3002.9 | 0.326 | 0.356 | 149.4 |  |
| median_blur | size=5 | 0.725 | 0.722 | 1384.3 | 0.713 | 0.743 | 68.9 |  |
| median_blur | size=7 | 0.984 | 0.982 | 1018.7 | 0.971 | 1.004 | 50.7 |  |
| bilateral_blur | sigma_space=1, sigma_value=0.1 | 0.393 | 0.390 | 2562.7 | 0.385 | 0.407 | 127.5 |  |
| bilateral_blur | sigma_space=2, sigma_value=0.1 | 0.869 | 0.866 | 1155.4 | 0.856 | 0.884 | 57.5 |  |
| convolve_box | size=(1,31), normalize=True | 0.549 | 0.547 | 1827.6 | 0.536 | 0.566 | 91.0 |  |
| erosion | radius=5, shape=disk | 0.324 | 0.321 | 3111.3 | 0.318 | 0.338 | 154.8 |  |
| dilation | radius=5, shape=disk | 0.323 | 0.320 | 3123.2 | 0.317 | 0.337 | 155.4 |  |
| opening | radius=5, shape=disk | 0.595 | 0.593 | 1686.3 | 0.587 | 0.609 | 83.9 |  |
| closing | radius=5, shape=disk | 0.593 | 0.591 | 1693.4 | 0.584 | 0.607 | 84.3 |  |
| morphological_gradient | radius=5, shape=disk | 0.348 | 0.345 | 2901.7 | 0.340 | 0.363 | 144.4 |  |
| white_tophat | radius=5, shape=disk | 0.602 | 0.600 | 1666.8 | 0.592 | 0.618 | 83.0 |  |
| black_tophat | radius=5, shape=disk | 0.603 | 0.600 | 1667.2 | 0.592 | 0.621 | 83.0 |  |
| sobel | direction=x | 0.482 | 0.477 | 2094.8 | 0.471 | 0.500 | 104.3 |  |
| sobel | direction=y | 0.482 | 0.479 | 2086.5 | 0.469 | 0.501 | 103.8 |  |
| sobel | direction=magnitude | 0.488 | 0.486 | 2057.8 | 0.476 | 0.504 | 102.4 |  |
| laplacian | kernel=3x3 | 0.283 | 0.279 | 3590.3 | 0.273 | 0.304 | 178.7 |  |
| canny | threshold_low=0.5, threshold_high=1.0, border=mirror | 2.250 | 2.257 | 443.0 | 2.117 | 2.381 | 22.0 | yes |
| sharpen | amount=1, border=mirror | 0.466 | 0.463 | 2159.7 | 0.456 | 0.483 | 107.5 |  |
| difference_of_gaussians | sigma1=1, sigma2=2 | 1.259 | 1.257 | 795.3 | 1.246 | 1.277 | 39.6 | yes |
| corner_harris | FHD fp32 RGB, block_size=3, k=0.04, border=mirror | 0.565 | 0.564 | 1773.9 | 0.554 | 0.580 | 58.9 |  |
| match_template | FHD fp32 RGB + 64x64 fp32 RGB, method=ccoeff_normed | 13.122 | 13.094 | 76.4 | 12.974 | 13.306 | 2.5 | yes |
| psnr | FHD fp32 RGB reference/candidate, data_range=1.0 default | 0.345 | 0.304 | 3294.3 | 0.287 | 0.559 | 163.9 |  |
| ssim | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.059 | 2.056 | 486.4 | 2.045 | 2.083 | 24.2 | yes |
| ssim_map | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.051 | 2.049 | 488.0 | 2.027 | 2.083 | 28.3 | yes |
| equalize_histogram | domain=(0,1), bins=1024 | 0.899 | 0.887 | 1127.3 | 0.863 | 0.967 | 56.1 |  |
| clahe | clip_limit=2, tiles_y=8, tiles_x=8, domain=(0,1), bins=1024 | 2.980 | 2.977 | 336.0 | 2.928 | 3.050 | 16.7 | yes |
| directional_blur | angle=30, length=8 | 0.537 | 0.546 | 1831.0 | 0.498 | 0.574 | 91.1 |  |
| directional_blur | angle=30, length=32 | 1.602 | 1.596 | 626.5 | 1.575 | 1.646 | 31.2 | yes |
| directional_blur | angle=30, length=128 | 7.583 | 7.571 | 132.1 | 7.444 | 7.769 | 6.6 | yes |
| zoom_blur | amount=0.05 | 1.900 | 1.896 | 527.6 | 1.877 | 1.933 | 26.3 | yes |
| zoom_blur | amount=0.2 | 9.085 | 9.074 | 110.2 | 8.937 | 9.287 | 5.5 | yes |
| spin_blur | angle=2 | 1.251 | 1.252 | 799.0 | 1.215 | 1.288 | 39.8 | yes |
| spin_blur | angle=10 | 7.915 | 7.891 | 126.7 | 7.724 | 8.155 | 6.3 | yes |
| vector_blur | uniform \|v\|=8, shutter=centered | 0.813 | 0.807 | 1239.0 | 0.786 | 0.848 | 82.2 |  |
| vector_blur | uniform \|v\|=32, shutter=centered | 1.800 | 1.795 | 557.0 | 1.775 | 1.835 | 37.0 | yes |
| vector_blur | uniform \|v\|=128, shutter=centered | 7.375 | 7.363 | 135.8 | 7.272 | 7.514 | 9.0 | yes |
| vector_blur | rotation field, corner \|v\|=32, shutter=centered | 1.147 | 1.145 | 873.6 | 1.124 | 1.177 | 58.0 | yes |
| lens_blur | circle radius=4 | 0.719 | 0.720 | 1388.7 | 0.702 | 0.741 | 69.1 |  |
| lens_blur | circle radius=8 | 1.582 | 1.589 | 629.1 | 1.546 | 1.621 | 31.3 | yes |
| lens_blur | circle radius=16 | 1.587 | 1.577 | 634.1 | 1.557 | 1.621 | 31.6 | yes |
| lens_blur | circle radius=32 | 1.366 | 1.363 | 733.8 | 1.352 | 1.390 | 36.5 | yes |
| lens_blur | blades=6, radius=16 | 1.571 | 1.568 | 637.9 | 1.558 | 1.585 | 31.7 | yes |
| lens_blur | blades=6, radius=32 | 1.372 | 1.365 | 732.3 | 1.357 | 1.411 | 36.4 | yes |
| line | diagonal thickness=4, aa=distance | 0.188 | 0.184 | 5421.4 | 0.160 | 0.233 | 269.8 |  |
| polyline | 5 points, closed, thickness=6, aa=distance | 0.215 | 0.213 | 4690.0 | 0.186 | 0.245 | 233.4 |  |
| rectangle | 1280x720 fill, corner_radius=48, aa=distance | 0.193 | 0.188 | 5328.1 | 0.185 | 0.216 | 265.2 |  |
| circle | fill radius=320, aa=supersample | 0.175 | 0.170 | 5876.0 | 0.168 | 0.199 | 292.4 |  |
| ellipse | radii=(520,260), rotation=25, thickness=8 | 0.170 | 0.161 | 6202.6 | 0.156 | 0.195 | 308.7 |  |
| polygon | 8-point concave fill, aa=distance | 0.230 | 0.224 | 4472.6 | 0.214 | 0.255 | 222.6 |  |
| text | single-line CJK, size=64, one outline, supersample=False | 0.377 | 0.356 | 2811.1 | 0.333 | 0.496 | 139.9 |  |
| text | single-line CJK, size=64, one outline, supersample=True | 0.397 | 0.388 | 2575.5 | 0.360 | 0.459 | 128.2 |  |
| ramp | FHD linear RGB | 0.131 | 0.126 | 7943.8 | 0.123 | 0.151 | 197.7 |  |
| grid | FHD cell=(64,64), line_width=2, aa=distance | 0.130 | 0.125 | 7980.6 | 0.122 | 0.151 | 198.6 |  |
| checkerboard | FHD cell=(64,64), aa=distance | 0.129 | 0.125 | 8024.2 | 0.122 | 0.152 | 199.7 |  |
| color_bars | FHD ARIB STD-B28 normalized | 0.087 | 0.083 | 12017.8 | 0.081 | 0.102 | 299.0 |  |
| fractal_noise | FHD scale=64, octaves=4 | 2.143 | 2.136 | 468.2 | 2.119 | 2.184 | 3.9 | yes |
| turbulent_noise | FHD scale=64, octaves=4 | 2.137 | 2.136 | 468.1 | 2.121 | 2.159 | 3.9 | yes |
| grain | FHD intensity=0.1, size=1, RGB | 3.609 | 3.609 | 277.1 | 3.551 | 3.660 | 6.9 | yes |
| from_array | CHW + affine scale=255 -> float32 HWC | 0.147 | 0.138 | 7247.9 | 0.132 | 0.177 | 360.7 |  |
| from_array | CHW uint16, bit_depth=10 -> float32 HWC | 0.131 | 0.125 | 8025.1 | 0.114 | 0.158 | 299.5 |  |
| to_array | BGR + NCHW + float16 + affine | 0.130 | 0.127 | 7845.3 | 0.112 | 0.159 | 292.8 |  |
| to_array | bit_depth=10 -> uint16 HWC | 0.110 | 0.106 | 9453.7 | 0.102 | 0.132 | 352.9 |  |
| rgb_to_rgb | ACEScg linear -> sRGB sRGB | 0.143 | 0.135 | 7403.6 | 0.130 | 0.168 | 368.4 |  |
| rgb_to_rgb | ACES 1.3 analytic -> sRGB sRGB | 0.166 | 0.159 | 6297.9 | 0.155 | 0.193 | 313.4 |  |
| rgb_to_rgb | ACES 2.0 analytic -> sRGB sRGB | 0.181 | 0.172 | 5830.1 | 0.146 | 0.239 | 290.1 |  |
| rgb_to_rgb | BT.2408 direct mapping -> Rec.2020 pq | 0.154 | 0.144 | 6954.0 | 0.134 | 0.207 | 346.1 |  |
| chromatic_adaptation | FHD fp32 RGB, D50 input -> D60 output, CAT02 | 0.164 | 0.165 | 6065.9 | 0.142 | 0.201 | 301.9 |  |
| grade | FHD fp32 RGB, per-channel Lift / Gamma / Gain | 0.236 | 0.223 | 4493.0 | 0.212 | 0.311 | 223.6 |  |
| white_balance | FHD fp32 RGB, Temperature=5000 K, Tint=0 Duv, CAT02 | 0.178 | 0.171 | 5862.8 | 0.156 | 0.235 | 291.8 |  |
| white_point_simulation | FHD fp32 RGB, D65 input display -> D93 output display | 0.159 | 0.162 | 6169.5 | 0.136 | 0.195 | 307.0 |  |
| rgb_to_ycbcr | RGB -> YCbCr, matrix=native | 0.161 | 0.164 | 6088.9 | 0.139 | 0.189 | 303.0 |  |
| rgb_to_hsv | RGB -> HSV, label-driven scene values | 0.134 | 0.127 | 7894.8 | 0.107 | 0.191 | 392.9 |  |
| hsv_to_rgb | HSV six sectors, S=[0,1], V=[0,2] -> RGB | 0.126 | 0.114 | 8753.9 | 0.108 | 0.179 | 435.6 |  |
| ycbcr_to_rgb | YCbCr -> RGB, matrix=bt709 | 0.160 | 0.155 | 6431.2 | 0.136 | 0.214 | 320.1 |  |
| rgb_to_grayscale | RGB -> Y, matrix=native | 0.129 | 0.120 | 8358.3 | 0.115 | 0.182 | 277.3 |  |
| gamma_to_linear | gamma=Gamma-2.6 claim -> linear | 0.156 | 0.149 | 6723.6 | 0.134 | 0.212 | 334.6 |  |
| linear_to_gamma | linear -> gamma=Gamma-2.6 | 0.150 | 0.141 | 7098.1 | 0.136 | 0.184 | 353.2 |  |
| ycbcr_to_ycbcr | YCbCr bt709 -> native rematrix | 0.150 | 0.145 | 6913.8 | 0.141 | 0.176 | 344.1 |  |
| full_to_legal | full -> legal, bit_depth=10 | 0.143 | 0.145 | 6919.9 | 0.120 | 0.182 | 344.4 |  |
| legal_to_full | legal -> full, bit_depth=10 | 0.149 | 0.142 | 7066.5 | 0.117 | 0.210 | 351.7 |  |
| quantize | float32 -> uint8, bit_depth=8 | 0.104 | 0.104 | 9603.8 | 0.081 | 0.157 | 298.7 |  |
| dequantize | uint8 -> float32, bit_depth=8 | 0.099 | 0.090 | 11083.9 | 0.082 | 0.152 | 344.8 |  |
| cast_dtype | float32 -> float16 | 0.104 | 0.097 | 10359.2 | 0.092 | 0.131 | 386.7 |  |
| recode_dtype | uint8 -> float32 | 0.109 | 0.109 | 9198.0 | 0.086 | 0.136 | 286.1 |  |
| recode_dtype | float32 -> uint8 | 0.106 | 0.106 | 9418.9 | 0.084 | 0.133 | 293.0 |  |
| from_uyvy422 | legal range | 0.110 | 0.112 | 8912.5 | 0.090 | 0.140 | 258.7 |  |
| from_v210 | legal range | 0.098 | 0.094 | 10623.5 | 0.092 | 0.117 | 323.1 |  |
| from_nv12 | legal range, siting=left, interpolation=bilinear | 0.097 | 0.093 | 10732.4 | 0.090 | 0.114 | 300.4 |  |
| from_p010 | legal range, siting=left, interpolation=bilinear | 0.100 | 0.096 | 10430.0 | 0.092 | 0.122 | 324.4 |  |
| from_yuv420p | legal range, interpolation=bilinear | 0.095 | 0.091 | 11043.9 | 0.088 | 0.116 | 309.2 |  |
| from_yuv422p | legal range | 0.101 | 0.097 | 10353.6 | 0.095 | 0.119 | 343.5 |  |
| from_yuv444p | 10-bit legal range | 0.111 | 0.099 | 10104.7 | 0.095 | 0.167 | 377.2 |  |
| from_yuva444p | 12-bit legal range | 0.125 | 0.119 | 8423.0 | 0.113 | 0.148 | 419.2 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=trilinear | 0.357 | 0.353 | 2836.5 | 0.346 | 0.376 | 141.2 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=tetrahedral | 0.361 | 0.357 | 2800.4 | 0.351 | 0.379 | 139.4 |  |
| apply_lut | FHD fp32 RGB, 65-sample 1D LUT, interpolation=linear | 0.358 | 0.353 | 2829.0 | 0.350 | 0.376 | 140.8 |  |
| to_uyvy422 | FHD area, legal | 0.102 | 0.093 | 10750.7 | 0.085 | 0.160 | 312.1 |  |
| to_v210 | FHD area, legal, 128-byte rows | 0.166 | 0.165 | 6066.2 | 0.143 | 0.213 | 184.5 |  |
| to_nv12 | FHD area, legal, siting=left | 0.145 | 0.139 | 7171.4 | 0.127 | 0.192 | 200.8 |  |
| to_p010 | FHD area, legal, siting=left | 0.146 | 0.139 | 7177.2 | 0.128 | 0.200 | 223.2 |  |
| to_yuv420p | 8-bit area, legal, siting=left | 0.145 | 0.144 | 6933.8 | 0.126 | 0.182 | 194.1 |  |
| to_yuv422p | 10-bit area, legal | 0.123 | 0.118 | 8447.3 | 0.116 | 0.147 | 280.3 |  |
| to_yuv444p | 10-bit legal | 0.097 | 0.093 | 10779.5 | 0.090 | 0.120 | 402.3 |  |
| to_yuva444p | 12-bit legal, alpha full | 0.116 | 0.112 | 8936.6 | 0.108 | 0.139 | 444.7 |  |
| read_lut | 65^3 RGB .cube file, parse, float4 packing, and host-to-device transfer included | 139.428 | 138.396 | 7.2 | 137.225 | 143.670 | 0.1 | yes |
| read_lut | 65-sample RGB Cube 1D file, parse and host-to-device transfer included | 0.263 | 0.250 | 4006.8 | 0.234 | 0.320 | 0.0 |  |
| read_lut | 17^3 RGB headerless 3DL file, parse, packing, and host-to-device transfer included | 8.934 | 8.919 | 112.1 | 8.489 | 9.373 | 0.0 | yes |
| read_lut | 65-sample RGB SPI1D file, parse and host-to-device transfer included | 0.226 | 0.217 | 4598.3 | 0.196 | 0.280 | 0.0 |  |
| read_lut | 17^3 RGB SPI3D file, explicit-index parse, packing, and host-to-device transfer included | 9.108 | 9.038 | 110.6 | 8.624 | 9.710 | 0.0 | yes |
| write_lut | 65-sample RGB Lut1D, device-to-host transfer and Cube file write included | 0.267 | 0.246 | 4070.6 | 0.230 | 0.365 | 0.0 |  |
| write_lut | 65^3 RGB Lut, device-to-host transfer and Cube file write included | 321.364 | 320.647 | 3.1 | 317.594 | 329.073 | 0.0 | yes |
| read_image | FHD uint8 RGB PNG file, unchanged, temporary-file I/O included | 25.395 | 25.356 | 39.4 | 24.832 | 25.942 | 0.5 | yes |
| read_image | FHD uint8 RGB JPEG file, unchanged, temporary-file I/O included | 38.844 | 38.824 | 25.8 | 37.419 | 40.110 | 0.3 | yes |
| read_image | FHD uint8 RGB TIFF file, unchanged, temporary-file I/O included | 33.842 | 34.109 | 29.3 | 32.445 | 35.157 | 0.4 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 40.157 | 40.360 | 24.8 | 38.418 | 41.719 | 0.6 | yes |
| read_image | FHD HALF RGB EXR NONE file, unchanged, source-fixed native lane, temporary-file I/O included | 32.202 | 30.473 | 32.8 | 29.530 | 41.743 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 33.119 | 33.078 | 30.2 | 32.031 | 34.289 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIPS file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 65.184 | 63.678 | 15.7 | 60.908 | 66.483 | 0.4 | yes |
| read_image | FHD HALF RGB EXR DWAA file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 24.442 | 23.922 | 41.8 | 23.019 | 27.388 | 1.0 | yes |
| read_image | FHD HALF RGB EXR DWAB file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 25.364 | 25.136 | 39.8 | 24.415 | 26.952 | 1.0 | yes |
| read_image | FHD HALF RGB EXR RLE file, unchanged, source-fixed GPU lane, temporary-file I/O included | 179.532 | 162.042 | 6.2 | 156.061 | 225.771 | 0.2 | yes |
| read_image | FHD HALF RGB EXR PXR24 file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 339.096 | 341.551 | 2.9 | 301.812 | 379.757 | 0.1 | yes |
| read_image | FHD HALF RGB EXR B44 file, unchanged, source-fixed GPU lane, temporary-file I/O included | 84.353 | 71.853 | 13.9 | 67.825 | 134.814 | 0.3 | yes |
| read_image | FHD HALF RGB EXR B44A file, unchanged, source-fixed GPU lane, temporary-file I/O included | 108.920 | 96.095 | 10.4 | 93.792 | 158.452 | 0.3 | yes |
| read_image | FHD HALF RGB EXR PIZ file, unchanged, source-fixed GPU lane, temporary-file I/O included | 34.137 | 33.939 | 29.5 | 33.628 | 35.168 | 0.7 | yes |
| read_image | FHD uint8 RGB JPEG 2000 file, unchanged, temporary-file I/O included | 25.149 | 25.152 | 39.8 | 24.572 | 25.720 | 0.5 | yes |
| read_image | FHD uint8 RGB WebP file, unchanged, temporary-file I/O included | 40.513 | 40.172 | 24.9 | 39.604 | 41.595 | 0.3 | yes |
| read_image | FHD uint8 RGB BMP file, unchanged, temporary-file I/O included | 17.392 | 17.414 | 57.4 | 16.612 | 18.404 | 0.7 | yes |
| read_image | FHD uint8 RGB PNM file, unchanged, temporary-file I/O included | 19.348 | 19.273 | 51.9 | 18.632 | 20.456 | 0.6 | yes |
| read_image | FHD uint8 RGB TGA file, unchanged, temporary-file I/O and CPU RLE included | 12.404 | 12.394 | 80.7 | 12.136 | 12.745 | 1.0 | yes |
| read_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 133.639 | 134.319 | 7.4 | 130.385 | 135.178 | 0.2 | yes |
| read_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU unpack included | 5.389 | 5.338 | 187.3 | 5.011 | 5.895 | 6.2 | yes |
| write_image | FHD uint8 RGB PNG file, compression_level=4, temporary-file I/O included | 197.498 | 197.281 | 5.1 | 195.912 | 199.600 | 0.1 | yes |
| write_image | FHD uint8 RGB JPEG file, quality=95, temporary-file I/O included | 39.681 | 39.561 | 25.3 | 37.796 | 42.548 | 0.3 | yes |
| write_image | FHD uint8 RGB TIFF file, temporary-file I/O included | 55.913 | 55.793 | 17.9 | 52.688 | 59.847 | 0.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 37.994 | 37.539 | 26.6 | 36.846 | 39.585 | 1.0 | yes |
| write_exr_channels | FHD HALF RGB + UINT ID to EXR ZIP, native mixed dtypes, temporary-file I/O included | 37.458 | 37.179 | 26.9 | 36.207 | 38.680 | 1.1 | yes |
| write_image | FHD fp32 RGB to EXR NONE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 8.142 | 8.076 | 123.8 | 6.445 | 9.552 | 4.6 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 35.283 | 35.355 | 28.3 | 33.287 | 37.114 | 1.1 | yes |
| write_image | FHD fp32 RGB to EXR ZIPS/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 111.557 | 106.632 | 9.4 | 92.697 | 152.352 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAA/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 87.659 | 85.176 | 11.7 | 82.580 | 93.409 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAB/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 33.223 | 33.067 | 30.2 | 32.335 | 34.623 | 1.1 | yes |
| write_image | FHD fp32 RGB to EXR RLE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 17.148 | 17.131 | 58.4 | 15.242 | 19.917 | 2.2 | yes |
| write_image | FHD fp32 RGB to EXR PXR24/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 46.259 | 46.104 | 21.7 | 44.823 | 47.550 | 0.8 | yes |
| write_image | FHD fp16 RGB to EXR B44/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 6.095 | 5.782 | 172.9 | 5.389 | 7.317 | 4.3 | yes |
| write_image | FHD fp16 RGB to EXR B44A/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 6.644 | 6.910 | 144.7 | 5.093 | 7.823 | 3.6 | yes |
| write_image | FHD fp16 RGB to EXR PIZ/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 85.882 | 85.830 | 11.7 | 85.248 | 86.592 | 0.3 | yes |
| write_image | FHD uint8 RGB JPEG 2000 file, lossless, temporary-file I/O included | 66.026 | 66.108 | 15.1 | 63.280 | 68.793 | 0.2 | yes |
| write_image | FHD uint8 RGB WebP file, lossless, temporary-file I/O included | 329.877 | 331.910 | 3.0 | 321.974 | 337.405 | 0.0 | yes |
| write_image | FHD uint8 RGB BMP file, temporary-file I/O included | 25.972 | 25.976 | 38.5 | 24.145 | 28.313 | 0.5 | yes |
| write_image | FHD uint8 RGB PNM file, temporary-file I/O included | 27.541 | 27.455 | 36.4 | 25.554 | 29.794 | 0.5 | yes |
| write_image | FHD uint8 RGB TGA file, temporary-file I/O and CPU RLE included | 53.303 | 53.201 | 18.8 | 51.831 | 54.897 | 0.2 | yes |
| write_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 292.913 | 291.210 | 3.4 | 287.077 | 303.595 | 0.1 | yes |
| write_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU packing included | 10.855 | 10.654 | 93.9 | 10.469 | 11.021 | 3.1 | yes |
| read_header | FHD uint8 RGB PNG header, temporary-file I/O included | 2.187 | 2.165 | 461.9 | 2.120 | 2.314 | 0.0 | yes |
| decode_lut | 65-sample RGB Cube 1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.204 | 0.197 | 5068.0 | 0.186 | 0.238 | 0.0 |  |
| decode_lut | 65^3 RGB Cube 3D UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 150.350 | 148.100 | 6.8 | 147.210 | 160.354 | 0.1 | yes |
| decode_lut | 17^3 RGB headerless 3DL UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 15.256 | 15.222 | 65.7 | 14.697 | 16.026 | 0.0 | yes |
| decode_lut | 65-sample RGB SPI1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.183 | 0.175 | 5710.6 | 0.171 | 0.221 | 0.0 |  |
| decode_lut | 17^3 RGB SPI3D UTF-8 bytes, sniff, explicit-index parse, and host-to-device transfer included | 9.538 | 9.453 | 105.8 | 9.177 | 10.090 | 0.0 | yes |
| decode_image | FHD uint8 RGB PNG, unchanged, host bytes exchange included | 22.953 | 22.908 | 43.7 | 22.347 | 23.809 | 0.5 | yes |
| decode_image | FHD uint8 RGB JPEG, unchanged, host bytes exchange included | 37.925 | 37.797 | 26.5 | 36.841 | 39.048 | 0.3 | yes |
| decode_image | FHD uint8 RGB TIFF, unchanged, host bytes exchange included | 32.598 | 32.534 | 30.7 | 31.916 | 33.657 | 0.4 | yes |
| decode_image | FHD uint8 RGB JPEG 2000, unchanged, host bytes exchange included | 24.522 | 24.541 | 40.7 | 23.898 | 25.107 | 0.5 | yes |
| decode_image | FHD uint8 RGB WebP, unchanged, host bytes exchange included | 41.922 | 41.964 | 23.8 | 39.931 | 44.264 | 0.3 | yes |
| decode_image | FHD uint8 RGB BMP, unchanged, host bytes exchange included | 16.784 | 16.556 | 60.4 | 16.111 | 17.907 | 0.8 | yes |
| decode_image | FHD uint8 RGB PNM, unchanged, host bytes exchange included | 18.026 | 18.010 | 55.5 | 17.204 | 18.798 | 0.7 | yes |
| encode_image | FHD uint8 RGB PNG, compression_level=4, host bytes exchange included | 191.492 | 191.328 | 5.2 | 189.911 | 193.450 | 0.1 | yes |
| encode_image | FHD uint8 RGB JPEG, quality=95, host bytes exchange included | 37.736 | 37.639 | 26.6 | 35.265 | 41.001 | 0.3 | yes |
| encode_image | FHD uint8 RGB TIFF, host bytes exchange included | 52.616 | 52.900 | 18.9 | 49.230 | 56.221 | 0.2 | yes |
| encode_image | FHD uint8 RGB JPEG 2000, lossless, host bytes exchange included | 61.811 | 61.833 | 16.2 | 59.351 | 64.160 | 0.2 | yes |
| encode_image | FHD uint8 RGB WebP, lossless, host bytes exchange included | 317.957 | 317.783 | 3.1 | 313.777 | 323.538 | 0.0 | yes |
| encode_image | FHD uint8 RGB BMP, host bytes exchange included | 23.308 | 23.115 | 43.3 | 21.023 | 26.147 | 0.5 | yes |
| encode_image | FHD uint8 RGB PNM, host bytes exchange included | 25.395 | 25.375 | 39.4 | 23.116 | 27.597 | 0.5 | yes |
