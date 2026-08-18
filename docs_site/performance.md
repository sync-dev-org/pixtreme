# Performance

This is the complete 205-case FHD measurement report from a single full-suite `uv run pytest -m performance` run
at pixtreme commit `5f71032`. The run completed with 259 passed and 4,023 deselected in 3,534.01 seconds (58:54).
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
- **Threshold marker:** `yes` in `> 1 ms` means the case median exceeded 1 ms. There are 90 such cases.

## EXR source-fixed routing

EXR read and write use pixtreme-owned implementations for all ten compression tokens. OpenEXR is a dev-only oracle,
not a runtime route. The following routing table is fixed in source: runtime capabilities, environment, and measured
performance do not alter it. The public medians use the current route after at least 0.5 seconds of excluded warmup and
at least 20 iterations and 3 seconds of measurement. All registry EXR read fixtures use HALF storage; FLOAT read
characteristics remain recorded in the all-combination adoption-gate measurements that selected the source-fixed
routes. Write cases use the current HALF default unless the fp16 input is already native HALF.

| Compression | Read lane | Write lane | Public read median (ms) | Public write median (ms) |
|---|---|---|---:|---:|
| NONE | native | GPU | 30.418 | 8.605 |
| RLE | GPU | GPU | 162.049 | 17.822 |
| ZIPS | custom CPU | GPU | 64.289 | 89.132 |
| ZIP | custom CPU | GPU | 33.472 | 32.234 |
| PIZ | GPU | GPU | 35.055 | 80.759 |
| PXR24 | custom CPU | GPU | 330.448 | 42.980 |
| B44 | GPU | GPU | 71.782 | 6.336 |
| B44A | GPU | GPU | 102.995 | 7.812 |
| DWAA | GPU | GPU | 24.606 | 87.287 |
| DWAB | GPU | GPU | 26.185 | 33.830 |

The default float32-frame write case omits `dtype`, stores ZIP-compressed HALF, and measured 38.921 ms. Reading that
HALF fixture unchanged through the fixed custom CPU ZIP lane measured 39.981 ms. These general default-path cases use
a different deterministic corpus from the compression rows above.

## Full results

| target | representative parameters | mean ms | median ms | fps | p5 ms | p95 ms | effective GB/s | > 1 ms |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| copy | FHD fp32 RGB read+write | 0.116 | 0.112 | 8932.0 | 0.094 | 0.147 | 444.5 |  |
| resize | 1920x1080 -> 960x540, interpolation=nearest | 0.092 | 0.090 | 11112.2 | 0.068 | 0.127 | 345.6 |  |
| resize | 1920x1080 -> 960x540, interpolation=bilinear | 0.103 | 0.101 | 9909.3 | 0.088 | 0.133 | 308.2 |  |
| resize | 1920x1080 -> 960x540, interpolation=bicubic | 0.178 | 0.172 | 5824.1 | 0.134 | 0.242 | 181.2 |  |
| resize | 1920x1080 -> 960x540, interpolation=b-spline | 0.154 | 0.148 | 6748.2 | 0.135 | 0.195 | 209.9 |  |
| resize | 1920x1080 -> 960x540, interpolation=mitchell | 0.152 | 0.149 | 6710.7 | 0.134 | 0.184 | 208.7 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos2 | 0.158 | 0.152 | 6566.7 | 0.139 | 0.189 | 204.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos3 | 0.160 | 0.155 | 6447.3 | 0.138 | 0.195 | 200.5 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos4 | 0.243 | 0.236 | 4235.7 | 0.222 | 0.276 | 131.7 |  |
| resize | 1920x1080 -> 960x540, interpolation=area | 0.171 | 0.168 | 5943.8 | 0.153 | 0.202 | 184.9 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=nearest | 0.308 | 0.306 | 3271.9 | 0.289 | 0.335 | 407.1 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bilinear | 0.266 | 0.265 | 3778.0 | 0.251 | 0.291 | 470.0 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bicubic | 0.480 | 0.476 | 2099.2 | 0.455 | 0.512 | 261.2 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=b-spline | 0.481 | 0.478 | 2090.0 | 0.456 | 0.513 | 260.0 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=mitchell | 0.477 | 0.474 | 2108.0 | 0.462 | 0.501 | 262.3 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos2 | 0.482 | 0.474 | 2108.3 | 0.458 | 0.524 | 262.3 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos3 | 0.572 | 0.569 | 1757.7 | 0.546 | 0.606 | 218.7 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos4 | 0.960 | 0.954 | 1047.8 | 0.933 | 1.000 | 130.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=area | 0.678 | 0.675 | 1482.1 | 0.665 | 0.700 | 184.4 |  |
| warp_affine | FHD fp32 RGB, centered 1.01x scale + 5deg rotation, auto lanczos4, constant 0 | 3.128 | 3.125 | 320.0 | 3.096 | 3.182 | 15.9 | yes |
| stack | 2x FHD fp32 RGB, direction=vertical, adapt=False | 0.220 | 0.217 | 4614.1 | 0.201 | 0.251 | 459.3 |  |
| shuffle | single FHD fp32 Frame BGR reorder, adapt=False | 0.330 | 0.328 | 3047.9 | 0.311 | 0.354 | 151.7 |  |
| shuffle | FHD fp32 RGBA from 2 Frames + constant, adapt=False | 0.442 | 0.437 | 2287.8 | 0.416 | 0.479 | 132.8 |  |
| shuffle | 2 FHD fp32 RGB Frames, sRGB/srgb source adapted to ACEScg/linear | 0.440 | 0.431 | 2320.2 | 0.401 | 0.506 | 230.9 |  |
| merge | FHD background + transformed 960x540 foreground, bilinear, normal | 0.598 | 0.578 | 1729.6 | 0.498 | 0.769 | 96.8 |  |
| gaussian_blur | sigma=1 | 0.579 | 0.575 | 1740.5 | 0.552 | 0.614 | 86.6 |  |
| gaussian_blur | sigma=2 | 0.672 | 0.667 | 1499.1 | 0.646 | 0.706 | 74.6 |  |
| gaussian_blur | sigma=4 | 0.901 | 0.899 | 1112.0 | 0.888 | 0.917 | 55.3 |  |
| unsharp_mask | sigma=2, amount=1 | 0.767 | 0.766 | 1305.7 | 0.754 | 0.784 | 65.0 |  |
| box_blur | size=3 | 0.481 | 0.478 | 2090.3 | 0.465 | 0.507 | 104.0 |  |
| box_blur | size=9 | 0.538 | 0.537 | 1863.1 | 0.523 | 0.555 | 92.7 |  |
| median_blur | size=3 | 0.353 | 0.350 | 2857.1 | 0.334 | 0.378 | 142.2 |  |
| median_blur | size=5 | 0.739 | 0.739 | 1352.6 | 0.722 | 0.759 | 67.3 |  |
| median_blur | size=7 | 1.005 | 1.005 | 994.8 | 0.983 | 1.031 | 49.5 | yes |
| bilateral_blur | sigma_space=1, sigma_value=0.1 | 0.401 | 0.400 | 2498.4 | 0.387 | 0.417 | 124.3 |  |
| bilateral_blur | sigma_space=2, sigma_value=0.1 | 0.877 | 0.874 | 1144.5 | 0.860 | 0.903 | 57.0 |  |
| convolve_box | size=(1,31), normalize=True | 0.567 | 0.563 | 1775.8 | 0.550 | 0.592 | 88.4 |  |
| erosion | radius=5, shape=disk | 0.345 | 0.340 | 2937.4 | 0.327 | 0.369 | 146.2 |  |
| dilation | radius=5, shape=disk | 0.339 | 0.337 | 2971.7 | 0.322 | 0.368 | 147.9 |  |
| opening | radius=5, shape=disk | 0.616 | 0.613 | 1631.9 | 0.591 | 0.646 | 81.2 |  |
| closing | radius=5, shape=disk | 0.603 | 0.602 | 1660.9 | 0.589 | 0.619 | 82.7 |  |
| morphological_gradient | radius=5, shape=disk | 0.360 | 0.359 | 2789.0 | 0.344 | 0.381 | 138.8 |  |
| white_tophat | radius=5, shape=disk | 0.614 | 0.612 | 1635.0 | 0.596 | 0.640 | 81.4 |  |
| black_tophat | radius=5, shape=disk | 0.616 | 0.612 | 1635.2 | 0.594 | 0.648 | 81.4 |  |
| sobel | direction=x | 0.516 | 0.512 | 1953.5 | 0.486 | 0.558 | 97.2 |  |
| sobel | direction=y | 0.494 | 0.493 | 2028.4 | 0.479 | 0.514 | 100.9 |  |
| sobel | direction=magnitude | 0.507 | 0.504 | 1984.1 | 0.489 | 0.528 | 98.7 |  |
| laplacian | kernel=3x3 | 0.300 | 0.297 | 3366.2 | 0.283 | 0.322 | 167.5 |  |
| canny | threshold_low=0.5, threshold_high=1.0, border=mirror | 2.388 | 2.365 | 422.8 | 2.228 | 2.605 | 21.0 | yes |
| sharpen | amount=1, border=mirror | 0.482 | 0.480 | 2085.0 | 0.465 | 0.505 | 103.8 |  |
| difference_of_gaussians | sigma1=1, sigma2=2 | 1.277 | 1.271 | 786.9 | 1.252 | 1.310 | 39.2 | yes |
| corner_harris | FHD fp32 RGB, block_size=3, k=0.04, border=mirror | 0.587 | 0.584 | 1712.1 | 0.565 | 0.616 | 56.8 |  |
| match_template | FHD fp32 RGB + 64x64 fp32 RGB, method=ccoeff_normed | 13.133 | 13.104 | 76.3 | 13.035 | 13.328 | 2.5 | yes |
| psnr | FHD fp32 RGB reference/candidate, data_range=1.0 default | 0.341 | 0.320 | 3122.7 | 0.297 | 0.451 | 155.4 |  |
| ssim | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.072 | 2.068 | 483.7 | 2.057 | 2.098 | 24.1 | yes |
| ssim_map | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.055 | 2.052 | 487.3 | 2.044 | 2.077 | 28.2 | yes |
| equalize_histogram | domain=(0,1), bins=1024 | 0.889 | 0.879 | 1138.2 | 0.869 | 0.939 | 56.6 |  |
| clahe | clip_limit=2, tiles_y=8, tiles_x=8, domain=(0,1), bins=1024 | 2.949 | 2.943 | 339.8 | 2.934 | 2.984 | 16.9 | yes |
| directional_blur | angle=30, length=8 | 0.521 | 0.521 | 1920.4 | 0.508 | 0.537 | 95.6 |  |
| directional_blur | angle=30, length=32 | 1.589 | 1.585 | 630.8 | 1.574 | 1.617 | 31.4 | yes |
| directional_blur | angle=30, length=128 | 7.532 | 7.523 | 132.9 | 7.418 | 7.677 | 6.6 | yes |
| zoom_blur | amount=0.05 | 1.895 | 1.892 | 528.4 | 1.883 | 1.913 | 26.3 | yes |
| zoom_blur | amount=0.2 | 9.149 | 9.137 | 109.4 | 9.006 | 9.339 | 5.4 | yes |
| spin_blur | angle=2 | 1.245 | 1.239 | 807.1 | 1.228 | 1.261 | 40.2 | yes |
| spin_blur | angle=10 | 8.095 | 8.073 | 123.9 | 7.949 | 8.310 | 6.2 | yes |
| vector_blur | uniform \|v\|=8, shutter=centered | 0.804 | 0.801 | 1247.7 | 0.792 | 0.825 | 82.8 |  |
| vector_blur | uniform \|v\|=32, shutter=centered | 1.804 | 1.801 | 555.1 | 1.789 | 1.825 | 36.8 | yes |
| vector_blur | uniform \|v\|=128, shutter=centered | 7.382 | 7.373 | 135.6 | 7.299 | 7.524 | 9.0 | yes |
| vector_blur | rotation field, corner \|v\|=32, shutter=centered | 1.145 | 1.143 | 874.8 | 1.132 | 1.161 | 58.0 | yes |
| lens_blur | circle radius=4 | 0.713 | 0.710 | 1407.6 | 0.702 | 0.731 | 70.1 |  |
| lens_blur | circle radius=8 | 1.565 | 1.559 | 641.5 | 1.550 | 1.597 | 31.9 | yes |
| lens_blur | circle radius=16 | 1.570 | 1.568 | 637.8 | 1.560 | 1.586 | 31.7 | yes |
| lens_blur | circle radius=32 | 1.367 | 1.364 | 733.4 | 1.354 | 1.390 | 36.5 | yes |
| lens_blur | blades=6, radius=16 | 1.579 | 1.574 | 635.4 | 1.562 | 1.615 | 31.6 | yes |
| lens_blur | blades=6, radius=32 | 1.372 | 1.365 | 732.8 | 1.354 | 1.418 | 36.5 | yes |
| line | diagonal thickness=4, aa=distance | 0.179 | 0.176 | 5689.0 | 0.161 | 0.212 | 283.1 |  |
| polyline | 5 points, closed, thickness=6, aa=distance | 0.200 | 0.192 | 5215.1 | 0.188 | 0.230 | 259.5 |  |
| rectangle | 1280x720 fill, corner_radius=48, aa=distance | 0.196 | 0.190 | 5268.8 | 0.186 | 0.222 | 262.2 |  |
| circle | fill radius=320, aa=supersample | 0.181 | 0.174 | 5760.9 | 0.170 | 0.212 | 286.7 |  |
| ellipse | radii=(520,260), rotation=25, thickness=8 | 0.188 | 0.184 | 5446.8 | 0.160 | 0.235 | 271.1 |  |
| polygon | 8-point concave fill, aa=distance | 0.268 | 0.277 | 3615.9 | 0.221 | 0.316 | 180.0 |  |
| text | single-line CJK, size=64, one outline, supersample=False | 0.468 | 0.434 | 2303.3 | 0.371 | 0.698 | 114.6 |  |
| text | single-line CJK, size=64, one outline, supersample=True | 0.498 | 0.462 | 2164.3 | 0.367 | 0.701 | 107.7 |  |
| ramp | FHD linear RGB | 0.193 | 0.179 | 5584.7 | 0.146 | 0.252 | 139.0 |  |
| grid | FHD cell=(64,64), line_width=2, aa=distance | 0.158 | 0.151 | 6612.1 | 0.131 | 0.198 | 164.5 |  |
| checkerboard | FHD cell=(64,64), aa=distance | 0.150 | 0.133 | 7517.4 | 0.129 | 0.214 | 187.1 |  |
| color_bars | FHD ARIB STD-B28 normalized | 0.087 | 0.084 | 11901.7 | 0.081 | 0.103 | 296.2 |  |
| fractal_noise | FHD scale=64, octaves=4 | 2.166 | 2.163 | 462.4 | 2.155 | 2.186 | 3.8 | yes |
| turbulent_noise | FHD scale=64, octaves=4 | 2.166 | 2.163 | 462.3 | 2.155 | 2.185 | 3.8 | yes |
| grain | FHD intensity=0.1, size=1, RGB | 3.601 | 3.597 | 278.0 | 3.591 | 3.622 | 6.9 | yes |
| from_array | CHW + affine scale=255 -> float32 HWC | 0.144 | 0.139 | 7212.8 | 0.136 | 0.166 | 359.0 |  |
| from_array | CHW uint16, bit_depth=10 -> float32 HWC | 0.124 | 0.120 | 8345.9 | 0.117 | 0.145 | 311.5 |  |
| to_array | BGR + NCHW + float16 + affine | 0.119 | 0.114 | 8751.6 | 0.111 | 0.142 | 326.7 |  |
| to_array | bit_depth=10 -> uint16 HWC | 0.112 | 0.107 | 9389.3 | 0.104 | 0.139 | 350.5 |  |
| rgb_to_rgb | ACEScg linear -> sRGB srgb | 0.139 | 0.133 | 7495.8 | 0.128 | 0.166 | 373.0 |  |
| rgb_to_rgb | ACES 1.3 analytic -> sRGB srgb | 0.202 | 0.194 | 5142.9 | 0.184 | 0.236 | 255.9 |  |
| rgb_to_rgb | ACES 2.0 analytic -> sRGB srgb | 1.349 | 1.347 | 742.3 | 1.340 | 1.363 | 36.9 | yes |
| rgb_to_rgb | ACES 2.0 LUT -> sRGB srgb | 0.146 | 0.140 | 7146.2 | 0.136 | 0.171 | 355.6 |  |
| rgb_to_rgb | BT.2408 direct mapping -> Rec.2020 pq | 0.146 | 0.139 | 7187.5 | 0.135 | 0.173 | 357.7 |  |
| chromatic_adaptation | FHD fp32 RGB, D50 input -> D60 output, CAT02 | 0.156 | 0.148 | 6763.9 | 0.140 | 0.186 | 336.6 |  |
| white_balance | FHD fp32 RGB, Temperature=5000 K, Tint=0 Duv, CAT02 | 0.165 | 0.157 | 6355.3 | 0.154 | 0.196 | 316.3 |  |
| white_point_simulation | FHD fp32 RGB, D65 input display -> D93 output display | 0.146 | 0.137 | 7307.7 | 0.131 | 0.175 | 363.7 |  |
| rgb_to_ycbcr | RGB -> YCbCr, matrix=native | 0.153 | 0.145 | 6890.6 | 0.140 | 0.179 | 342.9 |  |
| rgb_to_hsv | RGB -> HSV, label-driven scene values | 0.120 | 0.115 | 8683.3 | 0.112 | 0.139 | 432.1 |  |
| hsv_to_rgb | HSV six sectors, S=[0,1], V=[0,2] -> RGB | 0.127 | 0.122 | 8218.6 | 0.113 | 0.151 | 409.0 |  |
| ycbcr_to_rgb | YCbCr -> RGB, matrix=bt709 | 0.150 | 0.144 | 6957.9 | 0.139 | 0.175 | 346.3 |  |
| rgb_to_grayscale | RGB -> Y, matrix=native | 0.125 | 0.120 | 8353.2 | 0.116 | 0.149 | 277.1 |  |
| gamma_to_linear | gamma=2.6 claim -> linear | 0.148 | 0.142 | 7057.4 | 0.138 | 0.171 | 351.2 |  |
| linear_to_gamma | linear -> gamma=2.6 | 0.153 | 0.145 | 6919.7 | 0.139 | 0.181 | 344.4 |  |
| ycbcr_to_ycbcr | YCbCr bt709 -> native rematrix | 0.150 | 0.144 | 6943.0 | 0.141 | 0.173 | 345.5 |  |
| full_to_legal | full -> legal, bit_depth=10 | 0.131 | 0.125 | 8017.5 | 0.122 | 0.159 | 399.0 |  |
| legal_to_full | legal -> full, bit_depth=10 | 0.130 | 0.125 | 7999.0 | 0.122 | 0.154 | 398.1 |  |
| quantize | float32 -> uint8, bit_depth=8 | 0.103 | 0.098 | 10208.0 | 0.087 | 0.131 | 317.5 |  |
| dequantize | uint8 -> float32, bit_depth=8 | 0.096 | 0.090 | 11098.5 | 0.087 | 0.124 | 345.2 |  |
| cast_dtype | float32 -> float16 | 0.100 | 0.096 | 10466.5 | 0.093 | 0.119 | 390.7 |  |
| recode_dtype | uint8 -> float32 | 0.188 | 0.178 | 5624.4 | 0.170 | 0.237 | 174.9 |  |
| recode_dtype | float32 -> uint8 | 0.101 | 0.095 | 10560.3 | 0.091 | 0.124 | 328.5 |  |
| from_uyvy422 | legal range | 0.099 | 0.094 | 10627.1 | 0.091 | 0.119 | 308.5 |  |
| from_v210 | legal range | 0.099 | 0.095 | 10481.8 | 0.093 | 0.119 | 318.8 |  |
| from_nv12 | legal range, siting=left, interpolation=bilinear | 0.106 | 0.102 | 9761.3 | 0.090 | 0.136 | 273.3 |  |
| from_p010 | legal range, siting=left, interpolation=bilinear | 0.103 | 0.097 | 10302.9 | 0.094 | 0.126 | 320.5 |  |
| from_yuv420p | legal range, interpolation=bilinear | 0.102 | 0.095 | 10544.3 | 0.090 | 0.130 | 295.2 |  |
| from_yuv422p | legal range | 0.109 | 0.102 | 9850.1 | 0.097 | 0.135 | 326.8 |  |
| from_yuv444p | 10-bit legal range | 0.111 | 0.103 | 9724.3 | 0.099 | 0.137 | 363.0 |  |
| from_yuva444p | 12-bit legal range | 0.134 | 0.129 | 7752.5 | 0.118 | 0.164 | 385.8 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=trilinear | 0.364 | 0.358 | 2789.9 | 0.353 | 0.386 | 138.8 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=tetrahedral | 0.362 | 0.358 | 2795.2 | 0.353 | 0.382 | 139.1 |  |
| apply_lut | FHD fp32 RGB, 65-sample 1D LUT, interpolation=linear | 0.367 | 0.357 | 2804.1 | 0.349 | 0.414 | 139.5 |  |
| to_uyvy422 | FHD area, legal | 0.113 | 0.104 | 9577.6 | 0.084 | 0.171 | 278.0 |  |
| to_v210 | FHD area, legal, 128-byte rows | 0.169 | 0.156 | 6402.1 | 0.145 | 0.232 | 194.7 |  |
| to_nv12 | FHD area, legal, siting=left | 0.168 | 0.155 | 6446.9 | 0.130 | 0.223 | 180.5 |  |
| to_p010 | FHD area, legal, siting=left | 0.142 | 0.137 | 7319.0 | 0.133 | 0.165 | 227.6 |  |
| to_yuv420p | 8-bit area, legal, siting=left | 0.139 | 0.134 | 7439.3 | 0.131 | 0.160 | 208.3 |  |
| to_yuv422p | 10-bit area, legal | 0.126 | 0.121 | 8278.3 | 0.118 | 0.150 | 274.7 |  |
| to_yuv444p | 10-bit legal | 0.099 | 0.094 | 10592.9 | 0.091 | 0.120 | 395.4 |  |
| to_yuva444p | 12-bit legal, alpha full | 0.128 | 0.122 | 8211.5 | 0.109 | 0.181 | 408.7 |  |
| read_lut | 65^3 RGB .cube file, parse, float4 packing, and host-to-device transfer included | 139.451 | 138.599 | 7.2 | 137.770 | 140.834 | 0.1 | yes |
| read_lut | 65-sample RGB Cube 1D file, parse and host-to-device transfer included | 0.249 | 0.238 | 4203.6 | 0.233 | 0.295 | 0.0 |  |
| read_lut | 17^3 RGB headerless 3DL file, parse, packing, and host-to-device transfer included | 8.623 | 8.499 | 117.7 | 8.272 | 9.290 | 0.0 | yes |
| read_lut | 65-sample RGB SPI1D file, parse and host-to-device transfer included | 0.220 | 0.205 | 4870.5 | 0.201 | 0.276 | 0.0 |  |
| read_lut | 17^3 RGB SPI3D file, explicit-index parse, packing, and host-to-device transfer included | 8.935 | 8.858 | 112.9 | 8.595 | 9.531 | 0.0 | yes |
| write_lut | 65-sample RGB Lut1D, device-to-host transfer and Cube file write included | 0.320 | 0.292 | 3428.7 | 0.271 | 0.539 | 0.0 |  |
| write_lut | 65^3 RGB Lut, device-to-host transfer and Cube file write included | 325.118 | 323.924 | 3.1 | 321.219 | 332.531 | 0.0 | yes |
| read_image | FHD uint8 RGB PNG file, unchanged, temporary-file I/O included | 25.679 | 25.537 | 39.2 | 25.141 | 26.433 | 0.5 | yes |
| read_image | FHD uint8 RGB JPEG file, unchanged, temporary-file I/O included | 41.004 | 41.125 | 24.3 | 39.467 | 42.106 | 0.3 | yes |
| read_image | FHD uint8 RGB TIFF file, unchanged, temporary-file I/O included | 33.054 | 33.057 | 30.3 | 32.561 | 33.636 | 0.4 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 40.090 | 39.981 | 25.0 | 39.144 | 41.098 | 0.6 | yes |
| read_image | FHD HALF RGB EXR NONE file, unchanged, source-fixed native lane, temporary-file I/O included | 30.675 | 30.418 | 32.9 | 29.628 | 32.997 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 33.470 | 33.472 | 29.9 | 32.594 | 34.272 | 0.7 | yes |
| read_image | FHD HALF RGB EXR ZIPS file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 64.518 | 64.289 | 15.6 | 62.168 | 67.345 | 0.4 | yes |
| read_image | FHD HALF RGB EXR DWAA file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 25.153 | 24.606 | 40.6 | 24.229 | 25.684 | 1.0 | yes |
| read_image | FHD HALF RGB EXR DWAB file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 26.260 | 26.185 | 38.2 | 25.825 | 26.795 | 1.0 | yes |
| read_image | FHD HALF RGB EXR RLE file, unchanged, source-fixed GPU lane, temporary-file I/O included | 177.179 | 162.049 | 6.2 | 157.153 | 219.767 | 0.2 | yes |
| read_image | FHD HALF RGB EXR PXR24 file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 330.270 | 330.448 | 3.0 | 299.542 | 364.100 | 0.1 | yes |
| read_image | FHD HALF RGB EXR B44 file, unchanged, source-fixed GPU lane, temporary-file I/O included | 83.478 | 71.782 | 13.9 | 69.215 | 128.423 | 0.3 | yes |
| read_image | FHD HALF RGB EXR B44A file, unchanged, source-fixed GPU lane, temporary-file I/O included | 114.376 | 102.995 | 9.7 | 100.099 | 158.392 | 0.2 | yes |
| read_image | FHD HALF RGB EXR PIZ file, unchanged, source-fixed GPU lane, temporary-file I/O included | 35.363 | 35.055 | 28.5 | 34.428 | 36.907 | 0.7 | yes |
| read_image | FHD uint8 RGB JPEG 2000 file, unchanged, temporary-file I/O included | 25.686 | 25.634 | 39.0 | 25.286 | 26.283 | 0.5 | yes |
| read_image | FHD uint8 RGB WebP file, unchanged, temporary-file I/O included | 41.755 | 41.687 | 24.0 | 40.910 | 42.854 | 0.3 | yes |
| read_image | FHD uint8 RGB BMP file, unchanged, temporary-file I/O included | 17.658 | 17.614 | 56.8 | 17.088 | 18.354 | 0.7 | yes |
| read_image | FHD uint8 RGB PNM file, unchanged, temporary-file I/O included | 19.311 | 19.223 | 52.0 | 18.751 | 19.969 | 0.6 | yes |
| read_image | FHD uint8 RGB TGA file, unchanged, temporary-file I/O and CPU RLE included | 12.703 | 12.707 | 78.7 | 12.418 | 12.952 | 1.0 | yes |
| read_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 135.287 | 134.518 | 7.4 | 132.113 | 139.446 | 0.2 | yes |
| read_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU unpack included | 5.682 | 5.623 | 177.9 | 5.293 | 6.054 | 5.9 | yes |
| write_image | FHD uint8 RGB PNG file, compression_level=4, temporary-file I/O included | 201.841 | 201.835 | 5.0 | 200.549 | 202.962 | 0.1 | yes |
| write_image | FHD uint8 RGB JPEG file, quality=95, temporary-file I/O included | 43.952 | 44.052 | 22.7 | 41.218 | 46.573 | 0.3 | yes |
| write_image | FHD uint8 RGB TIFF file, temporary-file I/O included | 61.084 | 60.589 | 16.5 | 57.591 | 65.679 | 0.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 39.184 | 38.921 | 25.7 | 37.615 | 41.187 | 1.0 | yes |
| write_image | FHD fp32 RGB to EXR NONE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 8.601 | 8.605 | 116.2 | 6.738 | 10.120 | 4.3 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 32.525 | 32.234 | 31.0 | 31.469 | 34.430 | 1.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIPS/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 88.079 | 89.132 | 11.2 | 72.819 | 104.309 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAA/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 88.976 | 87.287 | 11.5 | 83.391 | 98.145 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAB/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 34.520 | 33.830 | 29.6 | 33.359 | 37.662 | 1.1 | yes |
| write_image | FHD fp32 RGB to EXR RLE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 17.582 | 17.822 | 56.1 | 15.868 | 19.408 | 2.1 | yes |
| write_image | FHD fp32 RGB to EXR PXR24/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 42.954 | 42.980 | 23.3 | 41.347 | 44.596 | 0.9 | yes |
| write_image | FHD fp16 RGB to EXR B44/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 6.505 | 6.336 | 157.8 | 5.316 | 7.497 | 3.9 | yes |
| write_image | FHD fp16 RGB to EXR B44A/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 8.002 | 7.812 | 128.0 | 7.181 | 9.945 | 3.2 | yes |
| write_image | FHD fp16 RGB to EXR PIZ/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 80.984 | 80.759 | 12.4 | 79.917 | 83.056 | 0.3 | yes |
| write_image | FHD uint8 RGB JPEG 2000 file, lossless, temporary-file I/O included | 67.973 | 67.954 | 14.7 | 65.852 | 70.524 | 0.2 | yes |
| write_image | FHD uint8 RGB WebP file, lossless, temporary-file I/O included | 336.292 | 327.946 | 3.0 | 321.356 | 360.944 | 0.0 | yes |
| write_image | FHD uint8 RGB BMP file, temporary-file I/O included | 29.626 | 29.418 | 34.0 | 27.438 | 34.840 | 0.4 | yes |
| write_image | FHD uint8 RGB PNM file, temporary-file I/O included | 29.795 | 29.808 | 33.5 | 28.007 | 31.338 | 0.4 | yes |
| write_image | FHD uint8 RGB TGA file, temporary-file I/O and CPU RLE included | 54.016 | 53.814 | 18.6 | 53.072 | 55.090 | 0.2 | yes |
| write_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 300.864 | 298.743 | 3.3 | 293.265 | 311.728 | 0.1 | yes |
| write_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU packing included | 10.945 | 10.777 | 92.8 | 10.517 | 11.302 | 3.1 | yes |
| read_header | FHD uint8 RGB PNG header, temporary-file I/O included | 2.212 | 2.198 | 454.9 | 2.111 | 2.338 | 0.0 | yes |
| decode_lut | 65-sample RGB Cube 1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.200 | 0.191 | 5245.7 | 0.186 | 0.243 | 0.0 |  |
| decode_lut | 65^3 RGB Cube 3D UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 153.160 | 154.029 | 6.5 | 150.088 | 156.122 | 0.0 | yes |
| decode_lut | 17^3 RGB headerless 3DL UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 15.696 | 15.780 | 63.4 | 15.099 | 16.417 | 0.0 | yes |
| decode_lut | 65-sample RGB SPI1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.204 | 0.190 | 5257.8 | 0.180 | 0.267 | 0.0 |  |
| decode_lut | 17^3 RGB SPI3D UTF-8 bytes, sniff, explicit-index parse, and host-to-device transfer included | 9.811 | 9.777 | 102.3 | 9.532 | 10.231 | 0.0 | yes |
| decode_image | FHD uint8 RGB PNG, unchanged, host bytes exchange included | 24.632 | 24.645 | 40.6 | 23.753 | 25.503 | 0.5 | yes |
| decode_image | FHD uint8 RGB JPEG, unchanged, host bytes exchange included | 40.651 | 40.687 | 24.6 | 38.936 | 42.189 | 0.3 | yes |
| decode_image | FHD uint8 RGB TIFF, unchanged, host bytes exchange included | 32.802 | 32.594 | 30.7 | 32.218 | 33.693 | 0.4 | yes |
| decode_image | FHD uint8 RGB JPEG 2000, unchanged, host bytes exchange included | 24.697 | 24.759 | 40.4 | 24.040 | 25.337 | 0.5 | yes |
| decode_image | FHD uint8 RGB WebP, unchanged, host bytes exchange included | 41.575 | 41.403 | 24.2 | 40.483 | 43.694 | 0.3 | yes |
| decode_image | FHD uint8 RGB BMP, unchanged, host bytes exchange included | 17.361 | 17.339 | 57.7 | 16.659 | 18.175 | 0.7 | yes |
| decode_image | FHD uint8 RGB PNM, unchanged, host bytes exchange included | 19.387 | 19.283 | 51.9 | 18.541 | 20.045 | 0.6 | yes |
| encode_image | FHD uint8 RGB PNG, compression_level=4, host bytes exchange included | 201.653 | 200.633 | 5.0 | 196.665 | 209.469 | 0.1 | yes |
| encode_image | FHD uint8 RGB JPEG, quality=95, host bytes exchange included | 39.636 | 39.603 | 25.3 | 37.979 | 41.651 | 0.3 | yes |
| encode_image | FHD uint8 RGB TIFF, host bytes exchange included | 55.464 | 55.711 | 17.9 | 52.192 | 58.252 | 0.2 | yes |
| encode_image | FHD uint8 RGB JPEG 2000, lossless, host bytes exchange included | 63.345 | 63.370 | 15.8 | 61.746 | 64.945 | 0.2 | yes |
| encode_image | FHD uint8 RGB WebP, lossless, host bytes exchange included | 323.127 | 323.674 | 3.1 | 313.824 | 334.365 | 0.0 | yes |
| encode_image | FHD uint8 RGB BMP, host bytes exchange included | 24.260 | 24.238 | 41.3 | 22.855 | 25.634 | 0.5 | yes |
| encode_image | FHD uint8 RGB PNM, host bytes exchange included | 25.872 | 25.853 | 38.7 | 24.146 | 27.658 | 0.5 | yes |

