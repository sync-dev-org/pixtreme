# Performance

This is the complete 204-case FHD measurement report from a single full-suite `uv run pytest -m performance` run
at pixtreme commit `4c9c074`. The run completed with 258 passed and 3,978 deselected in 3,070.63 seconds (51:10).
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
| NONE | native | GPU | 27.911 | 7.960 |
| RLE | GPU | GPU | 151.628 | 15.864 |
| ZIPS | custom CPU | GPU | 59.030 | 87.948 |
| ZIP | custom CPU | GPU | 30.591 | 30.193 |
| PIZ | GPU | GPU | 32.459 | 76.885 |
| PXR24 | custom CPU | GPU | 327.005 | 39.146 |
| B44 | GPU | GPU | 64.990 | 4.689 |
| B44A | GPU | GPU | 94.319 | 5.085 |
| DWAA | GPU | GPU | 22.691 | 80.179 |
| DWAB | GPU | GPU | 24.817 | 31.072 |

The default float32-frame write case omits `dtype`, stores ZIP-compressed HALF, and measured 37.434 ms. Reading that
HALF fixture unchanged through the fixed custom CPU ZIP lane measured 35.758 ms. These general default-path cases use
a different deterministic corpus from the compression rows above.

## Full results

| target | representative parameters | mean ms | median ms | fps | p5 ms | p95 ms | effective GB/s | > 1 ms |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| copy | FHD fp32 RGB read+write | 0.094 | 0.092 | 10920.0 | 0.090 | 0.108 | 543.4 |  |
| resize | 1920x1080 -> 960x540, interpolation=nearest | 0.066 | 0.064 | 15564.4 | 0.063 | 0.077 | 484.1 |  |
| resize | 1920x1080 -> 960x540, interpolation=bilinear | 0.088 | 0.083 | 12098.3 | 0.081 | 0.128 | 376.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=bicubic | 0.138 | 0.127 | 7869.2 | 0.123 | 0.179 | 244.8 |  |
| resize | 1920x1080 -> 960x540, interpolation=b-spline | 0.139 | 0.139 | 7204.6 | 0.123 | 0.172 | 224.1 |  |
| resize | 1920x1080 -> 960x540, interpolation=mitchell | 0.138 | 0.127 | 7875.4 | 0.123 | 0.180 | 245.0 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos2 | 0.134 | 0.127 | 7880.5 | 0.124 | 0.177 | 245.1 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos3 | 0.132 | 0.129 | 7723.3 | 0.127 | 0.146 | 240.2 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos4 | 0.209 | 0.205 | 4881.9 | 0.203 | 0.223 | 151.8 |  |
| resize | 1920x1080 -> 960x540, interpolation=area | 0.149 | 0.146 | 6849.4 | 0.144 | 0.163 | 213.0 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=nearest | 0.274 | 0.273 | 3666.9 | 0.270 | 0.287 | 456.2 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bilinear | 0.248 | 0.238 | 4200.5 | 0.233 | 0.280 | 522.6 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bicubic | 0.439 | 0.436 | 2294.5 | 0.432 | 0.448 | 285.5 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=b-spline | 0.441 | 0.439 | 2279.8 | 0.435 | 0.453 | 283.6 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=mitchell | 0.440 | 0.438 | 2281.2 | 0.436 | 0.450 | 283.8 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos2 | 0.440 | 0.438 | 2284.5 | 0.435 | 0.452 | 284.2 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos3 | 0.534 | 0.531 | 1883.7 | 0.526 | 0.545 | 234.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos4 | 0.901 | 0.899 | 1112.0 | 0.891 | 0.917 | 138.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=area | 0.641 | 0.638 | 1567.1 | 0.636 | 0.653 | 195.0 |  |
| warp_affine | FHD fp32 RGB, centered 1.01x scale + 5deg rotation, auto lanczos4, constant 0 | 2.974 | 2.971 | 336.6 | 2.955 | 3.003 | 16.8 | yes |
| stack | 2x FHD fp32 RGB, direction=vertical, adapt=False | 0.193 | 0.189 | 5304.7 | 0.186 | 0.210 | 528.0 |  |
| shuffle | single FHD fp32 Frame BGR reorder, adapt=False | 0.298 | 0.296 | 3381.5 | 0.286 | 0.313 | 168.3 |  |
| shuffle | FHD fp32 RGBA from 2 Frames + constant, adapt=False | 0.396 | 0.394 | 2539.9 | 0.390 | 0.410 | 147.5 |  |
| shuffle | 2 FHD fp32 RGB Frames, sRGB/srgb source adapted to ACEScg/linear | 0.384 | 0.381 | 2626.7 | 0.375 | 0.399 | 261.4 |  |
| merge | FHD background + transformed 960x540 foreground, bilinear, normal | 0.455 | 0.449 | 2225.9 | 0.433 | 0.494 | 124.6 |  |
| gaussian_blur | sigma=1 | 0.531 | 0.528 | 1892.9 | 0.524 | 0.544 | 94.2 |  |
| gaussian_blur | sigma=2 | 0.620 | 0.617 | 1620.9 | 0.611 | 0.633 | 80.7 |  |
| gaussian_blur | sigma=4 | 0.854 | 0.852 | 1173.4 | 0.845 | 0.868 | 58.4 |  |
| unsharp_mask | sigma=2, amount=1 | 0.724 | 0.723 | 1384.0 | 0.715 | 0.738 | 68.9 |  |
| box_blur | size=3 | 0.450 | 0.448 | 2234.1 | 0.440 | 0.467 | 111.2 |  |
| box_blur | size=9 | 0.511 | 0.503 | 1987.6 | 0.496 | 0.552 | 98.9 |  |
| median_blur | size=3 | 0.326 | 0.324 | 3084.9 | 0.318 | 0.339 | 153.5 |  |
| median_blur | size=5 | 0.712 | 0.711 | 1406.9 | 0.697 | 0.725 | 70.0 |  |
| median_blur | size=7 | 0.964 | 0.963 | 1038.0 | 0.947 | 0.985 | 51.7 |  |
| bilateral_blur | sigma_space=1, sigma_value=0.1 | 0.376 | 0.373 | 2679.6 | 0.369 | 0.389 | 133.4 |  |
| bilateral_blur | sigma_space=2, sigma_value=0.1 | 0.828 | 0.826 | 1210.0 | 0.819 | 0.844 | 60.2 |  |
| convolve_box | size=(1,31), normalize=True | 0.528 | 0.526 | 1902.1 | 0.521 | 0.542 | 94.7 |  |
| erosion | radius=5, shape=disk | 0.312 | 0.309 | 3240.5 | 0.304 | 0.324 | 161.3 |  |
| dilation | radius=5, shape=disk | 0.312 | 0.308 | 3244.6 | 0.305 | 0.328 | 161.5 |  |
| opening | radius=5, shape=disk | 0.572 | 0.570 | 1753.5 | 0.563 | 0.587 | 87.3 |  |
| closing | radius=5, shape=disk | 0.569 | 0.567 | 1764.8 | 0.563 | 0.582 | 87.8 |  |
| morphological_gradient | radius=5, shape=disk | 0.334 | 0.330 | 3031.0 | 0.324 | 0.347 | 150.8 |  |
| white_tophat | radius=5, shape=disk | 0.578 | 0.574 | 1740.8 | 0.568 | 0.593 | 86.6 |  |
| black_tophat | radius=5, shape=disk | 0.575 | 0.572 | 1748.0 | 0.566 | 0.589 | 87.0 |  |
| sobel | direction=x | 0.462 | 0.460 | 2171.9 | 0.454 | 0.477 | 108.1 |  |
| sobel | direction=y | 0.463 | 0.462 | 2162.5 | 0.452 | 0.481 | 107.6 |  |
| sobel | direction=magnitude | 0.476 | 0.473 | 2115.2 | 0.463 | 0.492 | 105.3 |  |
| laplacian | kernel=3x3 | 0.267 | 0.266 | 3759.9 | 0.260 | 0.275 | 187.1 |  |
| canny | threshold_low=0.5, threshold_high=1.0, border=mirror | 2.100 | 2.077 | 481.4 | 2.047 | 2.206 | 24.0 | yes |
| sharpen | amount=1, border=mirror | 0.443 | 0.442 | 2262.0 | 0.439 | 0.452 | 112.6 |  |
| difference_of_gaussians | sigma1=1, sigma2=2 | 1.208 | 1.204 | 830.4 | 1.193 | 1.240 | 41.3 | yes |
| corner_harris | FHD fp32 RGB, block_size=3, k=0.04, border=mirror | 0.548 | 0.544 | 1839.8 | 0.535 | 0.582 | 61.0 |  |
| match_template | FHD fp32 RGB + 64x64 fp32 RGB, method=ccoeff_normed | 12.568 | 12.544 | 79.7 | 12.529 | 12.605 | 2.6 | yes |
| psnr | FHD fp32 RGB reference/candidate, data_range=1.0 default | 0.283 | 0.261 | 3825.9 | 0.252 | 0.358 | 190.4 |  |
| ssim | FHD fp32 RGB reference/candidate, data_range=1.0 default | 1.980 | 1.977 | 505.9 | 1.972 | 1.998 | 25.2 | yes |
| ssim_map | FHD fp32 RGB reference/candidate, data_range=1.0 default | 1.960 | 1.960 | 510.2 | 1.943 | 1.977 | 29.6 | yes |
| equalize_histogram | domain=(0,1), bins=1024 | 0.846 | 0.845 | 1184.1 | 0.834 | 0.863 | 58.9 |  |
| clahe | clip_limit=2, tiles_y=8, tiles_x=8, domain=(0,1), bins=1024 | 2.801 | 2.798 | 357.4 | 2.793 | 2.819 | 17.8 | yes |
| directional_blur | angle=30, length=8 | 0.494 | 0.488 | 2050.5 | 0.480 | 0.539 | 102.0 |  |
| directional_blur | angle=30, length=32 | 1.505 | 1.504 | 664.9 | 1.498 | 1.517 | 33.1 | yes |
| directional_blur | angle=30, length=128 | 7.181 | 7.171 | 139.5 | 7.091 | 7.305 | 6.9 | yes |
| zoom_blur | amount=0.05 | 1.817 | 1.811 | 552.3 | 1.794 | 1.840 | 27.5 | yes |
| zoom_blur | amount=0.2 | 8.775 | 8.766 | 114.1 | 8.640 | 8.932 | 5.7 | yes |
| spin_blur | angle=2 | 1.188 | 1.177 | 849.8 | 1.169 | 1.224 | 42.3 | yes |
| spin_blur | angle=10 | 7.771 | 7.755 | 128.9 | 7.629 | 7.960 | 6.4 | yes |
| vector_blur | uniform \|v\|=8, shutter=centered | 0.765 | 0.762 | 1312.1 | 0.758 | 0.777 | 87.1 |  |
| vector_blur | uniform \|v\|=32, shutter=centered | 1.720 | 1.719 | 581.7 | 1.711 | 1.733 | 38.6 | yes |
| vector_blur | uniform \|v\|=128, shutter=centered | 7.071 | 7.059 | 141.7 | 6.989 | 7.196 | 9.4 | yes |
| vector_blur | rotation field, corner \|v\|=32, shutter=centered | 1.093 | 1.091 | 916.2 | 1.080 | 1.110 | 60.8 | yes |
| lens_blur | circle radius=4 | 0.678 | 0.678 | 1475.8 | 0.672 | 0.688 | 73.4 |  |
| lens_blur | circle radius=8 | 1.494 | 1.492 | 670.4 | 1.487 | 1.503 | 33.4 | yes |
| lens_blur | circle radius=16 | 1.504 | 1.501 | 666.1 | 1.497 | 1.516 | 33.2 | yes |
| lens_blur | circle radius=32 | 1.304 | 1.299 | 769.6 | 1.296 | 1.339 | 38.3 | yes |
| lens_blur | blades=6, radius=16 | 1.513 | 1.502 | 665.6 | 1.498 | 1.518 | 33.1 | yes |
| lens_blur | blades=6, radius=32 | 1.313 | 1.308 | 764.6 | 1.298 | 1.345 | 38.1 | yes |
| line | diagonal thickness=4, aa=distance | 0.162 | 0.153 | 6541.2 | 0.148 | 0.193 | 325.5 |  |
| polyline | 5 points, closed, thickness=6, aa=distance | 0.193 | 0.196 | 5097.9 | 0.173 | 0.221 | 253.7 |  |
| rectangle | 1280x720 fill, corner_radius=48, aa=distance | 0.194 | 0.185 | 5419.3 | 0.174 | 0.238 | 269.7 |  |
| circle | fill radius=320, aa=supersample | 0.164 | 0.160 | 6236.9 | 0.159 | 0.179 | 310.4 |  |
| ellipse | radii=(520,260), rotation=25, thickness=8 | 0.154 | 0.150 | 6679.6 | 0.148 | 0.169 | 332.4 |  |
| polygon | 8-point concave fill, aa=distance | 0.214 | 0.207 | 4831.1 | 0.204 | 0.235 | 240.4 |  |
| text | single-line CJK, size=64, one outline, supersample=False | 0.322 | 0.302 | 3312.8 | 0.291 | 0.399 | 164.9 |  |
| text | single-line CJK, size=64, one outline, supersample=True | 0.422 | 0.355 | 2820.0 | 0.337 | 0.606 | 140.3 |  |
| ramp | FHD linear RGB | 0.116 | 0.113 | 8867.8 | 0.111 | 0.132 | 220.7 |  |
| grid | FHD cell=(64,64), line_width=2, aa=distance | 0.117 | 0.114 | 8806.9 | 0.112 | 0.134 | 219.1 |  |
| checkerboard | FHD cell=(64,64), aa=distance | 0.118 | 0.114 | 8777.8 | 0.112 | 0.136 | 218.4 |  |
| color_bars | FHD ARIB STD-B28 normalized | 0.085 | 0.077 | 12914.7 | 0.074 | 0.110 | 321.4 |  |
| fractal_noise | FHD scale=64, octaves=4 | 2.069 | 2.068 | 483.6 | 2.060 | 2.080 | 4.0 | yes |
| turbulent_noise | FHD scale=64, octaves=4 | 2.076 | 2.077 | 481.5 | 2.062 | 2.084 | 4.0 | yes |
| grain | FHD intensity=0.1, size=1, RGB | 3.452 | 3.451 | 289.8 | 3.443 | 3.465 | 7.2 | yes |
| from_array | CHW + affine scale=255 -> float32 HWC | 0.132 | 0.129 | 7771.6 | 0.127 | 0.150 | 386.8 |  |
| from_array | CHW uint16, bit_depth=10 -> float32 HWC | 0.113 | 0.111 | 9047.9 | 0.108 | 0.130 | 337.7 |  |
| to_array | BGR + NCHW + float16 + affine | 0.110 | 0.107 | 9379.4 | 0.105 | 0.127 | 350.1 |  |
| to_array | bit_depth=10 -> uint16 HWC | 0.102 | 0.099 | 10149.0 | 0.096 | 0.119 | 378.8 |  |
| rgb_to_rgb | ACEScg linear -> sRGB srgb | 0.124 | 0.122 | 8227.1 | 0.120 | 0.140 | 409.4 |  |
| rgb_to_rgb | ACES 1.3 analytic -> sRGB srgb | 0.149 | 0.146 | 6859.3 | 0.144 | 0.165 | 341.4 |  |
| rgb_to_rgb | ACES 2.0 analytic -> sRGB srgb | 0.147 | 0.143 | 6985.8 | 0.141 | 0.165 | 347.7 |  |
| rgb_to_rgb | BT.2408 direct mapping -> Rec.2020 pq | 0.135 | 0.129 | 7727.1 | 0.127 | 0.155 | 384.5 |  |
| chromatic_adaptation | FHD fp32 RGB, D50 input -> D60 output, CAT02 | 0.136 | 0.133 | 7544.3 | 0.130 | 0.153 | 375.5 |  |
| white_balance | FHD fp32 RGB, Temperature=5000 K, Tint=0 Duv, CAT02 | 0.150 | 0.145 | 6874.5 | 0.143 | 0.170 | 342.1 |  |
| white_point_simulation | FHD fp32 RGB, D65 input display -> D93 output display | 0.129 | 0.124 | 8035.0 | 0.122 | 0.147 | 399.9 |  |
| rgb_to_ycbcr | RGB -> YCbCr, matrix=native | 0.137 | 0.132 | 7550.1 | 0.130 | 0.153 | 375.7 |  |
| rgb_to_hsv | RGB -> HSV, label-driven scene values | 0.109 | 0.105 | 9500.2 | 0.104 | 0.123 | 472.8 |  |
| hsv_to_rgb | HSV six sectors, S=[0,1], V=[0,2] -> RGB | 0.109 | 0.106 | 9430.4 | 0.104 | 0.122 | 469.3 |  |
| ycbcr_to_rgb | YCbCr -> RGB, matrix=bt709 | 0.135 | 0.131 | 7605.8 | 0.129 | 0.152 | 378.5 |  |
| rgb_to_grayscale | RGB -> Y, matrix=native | 0.114 | 0.110 | 9086.4 | 0.108 | 0.131 | 301.5 |  |
| gamma_to_linear | gamma=Gamma-2.6 claim -> linear | 0.143 | 0.147 | 6784.3 | 0.127 | 0.164 | 337.6 |  |
| linear_to_gamma | linear -> gamma=Gamma-2.6 | 0.133 | 0.130 | 7714.5 | 0.128 | 0.148 | 383.9 |  |
| ycbcr_to_ycbcr | YCbCr bt709 -> native rematrix | 0.136 | 0.132 | 7580.5 | 0.130 | 0.153 | 377.3 |  |
| full_to_legal | full -> legal, bit_depth=10 | 0.130 | 0.133 | 7491.0 | 0.111 | 0.149 | 372.8 |  |
| legal_to_full | legal -> full, bit_depth=10 | 0.123 | 0.117 | 8583.4 | 0.111 | 0.142 | 427.2 |  |
| quantize | float32 -> uint8, bit_depth=8 | 0.088 | 0.083 | 11976.2 | 0.082 | 0.104 | 372.5 |  |
| dequantize | uint8 -> float32, bit_depth=8 | 0.085 | 0.083 | 12115.9 | 0.081 | 0.099 | 376.9 |  |
| cast_dtype | float32 -> float16 | 0.090 | 0.087 | 11431.6 | 0.086 | 0.103 | 426.7 |  |
| recode_dtype | uint8 -> float32 | 0.086 | 0.083 | 11984.1 | 0.082 | 0.101 | 372.8 |  |
| recode_dtype | float32 -> uint8 | 0.094 | 0.085 | 11818.0 | 0.081 | 0.132 | 367.6 |  |
| from_uyvy422 | legal range | 0.089 | 0.086 | 11600.5 | 0.085 | 0.103 | 336.8 |  |
| from_v210 | legal range | 0.090 | 0.087 | 11472.4 | 0.086 | 0.104 | 348.9 |  |
| from_nv12 | legal range, siting=left, interpolation=bilinear | 0.088 | 0.085 | 11774.5 | 0.083 | 0.104 | 329.6 |  |
| from_p010 | legal range, siting=left, interpolation=bilinear | 0.091 | 0.088 | 11301.2 | 0.087 | 0.107 | 351.5 |  |
| from_yuv420p | legal range, interpolation=bilinear | 0.088 | 0.085 | 11756.8 | 0.083 | 0.103 | 329.1 |  |
| from_yuv422p | legal range | 0.095 | 0.093 | 10802.2 | 0.091 | 0.111 | 358.4 |  |
| from_yuv444p | 10-bit legal range | 0.097 | 0.094 | 10633.7 | 0.092 | 0.112 | 396.9 |  |
| from_yuva444p | 12-bit legal range | 0.114 | 0.110 | 9050.2 | 0.109 | 0.129 | 450.4 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=trilinear | 0.339 | 0.335 | 2984.7 | 0.333 | 0.351 | 148.5 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=tetrahedral | 0.337 | 0.334 | 2993.1 | 0.331 | 0.349 | 149.0 |  |
| apply_lut | FHD fp32 RGB, 65-sample 1D LUT, interpolation=linear | 0.337 | 0.335 | 2987.6 | 0.330 | 0.350 | 148.7 |  |
| to_uyvy422 | FHD area, legal | 0.083 | 0.081 | 12351.8 | 0.080 | 0.098 | 358.6 |  |
| to_v210 | FHD area, legal, 128-byte rows | 0.144 | 0.141 | 7088.4 | 0.138 | 0.160 | 215.6 |  |
| to_nv12 | FHD area, legal, siting=left | 0.131 | 0.126 | 7912.6 | 0.123 | 0.148 | 221.5 |  |
| to_p010 | FHD area, legal, siting=left | 0.132 | 0.128 | 7814.8 | 0.126 | 0.147 | 243.1 |  |
| to_yuv420p | 8-bit area, legal, siting=left | 0.131 | 0.126 | 7926.4 | 0.124 | 0.146 | 221.9 |  |
| to_yuv422p | 10-bit area, legal | 0.115 | 0.112 | 8921.3 | 0.110 | 0.131 | 296.0 |  |
| to_yuv444p | 10-bit legal | 0.092 | 0.088 | 11367.1 | 0.087 | 0.107 | 424.3 |  |
| to_yuva444p | 12-bit legal, alpha full | 0.107 | 0.105 | 9540.2 | 0.103 | 0.123 | 474.8 |  |
| read_lut | 65^3 RGB .cube file, parse, float4 packing, and host-to-device transfer included | 128.406 | 127.373 | 7.9 | 126.546 | 128.462 | 0.1 | yes |
| read_lut | 65-sample RGB Cube 1D file, parse and host-to-device transfer included | 0.226 | 0.223 | 4483.1 | 0.205 | 0.261 | 0.0 |  |
| read_lut | 17^3 RGB headerless 3DL file, parse, packing, and host-to-device transfer included | 7.966 | 7.942 | 125.9 | 7.714 | 8.135 | 0.0 | yes |
| read_lut | 65-sample RGB SPI1D file, parse and host-to-device transfer included | 0.189 | 0.181 | 5520.0 | 0.178 | 0.223 | 0.0 |  |
| read_lut | 17^3 RGB SPI3D file, explicit-index parse, packing, and host-to-device transfer included | 8.277 | 8.263 | 121.0 | 8.035 | 8.535 | 0.0 | yes |
| write_lut | 65-sample RGB Lut1D, device-to-host transfer and Cube file write included | 0.271 | 0.262 | 3815.8 | 0.233 | 0.306 | 0.0 |  |
| write_lut | 65^3 RGB Lut, device-to-host transfer and Cube file write included | 304.647 | 304.436 | 3.3 | 302.891 | 307.496 | 0.0 | yes |
| read_image | FHD uint8 RGB PNG file, unchanged, temporary-file I/O included | 28.262 | 28.236 | 35.4 | 27.467 | 29.163 | 0.4 | yes |
| read_image | FHD uint8 RGB JPEG file, unchanged, temporary-file I/O included | 36.728 | 36.902 | 27.1 | 34.974 | 37.845 | 0.3 | yes |
| read_image | FHD uint8 RGB TIFF file, unchanged, temporary-file I/O included | 36.053 | 36.035 | 27.8 | 34.935 | 37.295 | 0.3 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 35.795 | 35.758 | 28.0 | 35.080 | 36.695 | 0.7 | yes |
| read_image | FHD HALF RGB EXR NONE file, unchanged, source-fixed native lane, temporary-file I/O included | 28.027 | 27.911 | 35.8 | 27.605 | 28.579 | 0.9 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 31.034 | 30.591 | 32.7 | 29.841 | 31.210 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIPS file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 60.148 | 59.030 | 16.9 | 57.386 | 61.445 | 0.4 | yes |
| read_image | FHD HALF RGB EXR DWAA file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 22.736 | 22.691 | 44.1 | 22.460 | 23.038 | 1.1 | yes |
| read_image | FHD HALF RGB EXR DWAB file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 25.345 | 24.817 | 40.3 | 24.453 | 27.005 | 1.0 | yes |
| read_image | FHD HALF RGB EXR RLE file, unchanged, source-fixed GPU lane, temporary-file I/O included | 164.600 | 151.628 | 6.6 | 148.398 | 198.649 | 0.2 | yes |
| read_image | FHD HALF RGB EXR PXR24 file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 311.166 | 327.005 | 3.1 | 282.214 | 338.057 | 0.1 | yes |
| read_image | FHD HALF RGB EXR B44 file, unchanged, source-fixed GPU lane, temporary-file I/O included | 74.570 | 64.990 | 15.4 | 63.281 | 111.502 | 0.4 | yes |
| read_image | FHD HALF RGB EXR B44A file, unchanged, source-fixed GPU lane, temporary-file I/O included | 105.436 | 94.319 | 10.6 | 93.654 | 141.840 | 0.3 | yes |
| read_image | FHD HALF RGB EXR PIZ file, unchanged, source-fixed GPU lane, temporary-file I/O included | 32.472 | 32.459 | 30.8 | 32.280 | 32.782 | 0.8 | yes |
| read_image | FHD uint8 RGB JPEG 2000 file, unchanged, temporary-file I/O included | 24.159 | 24.128 | 41.4 | 23.798 | 24.550 | 0.5 | yes |
| read_image | FHD uint8 RGB WebP file, unchanged, temporary-file I/O included | 42.209 | 41.940 | 23.8 | 40.339 | 44.229 | 0.3 | yes |
| read_image | FHD uint8 RGB BMP file, unchanged, temporary-file I/O included | 21.148 | 21.103 | 47.4 | 20.136 | 22.176 | 0.6 | yes |
| read_image | FHD uint8 RGB PNM file, unchanged, temporary-file I/O included | 22.772 | 22.664 | 44.1 | 21.934 | 23.707 | 0.5 | yes |
| read_image | FHD uint8 RGB TGA file, unchanged, temporary-file I/O and CPU RLE included | 11.381 | 11.374 | 87.9 | 11.192 | 11.646 | 1.1 | yes |
| read_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 123.165 | 123.027 | 8.1 | 122.423 | 123.921 | 0.3 | yes |
| read_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU unpack included | 4.854 | 4.734 | 211.2 | 4.520 | 5.491 | 7.0 | yes |
| write_image | FHD uint8 RGB PNG file, compression_level=4, temporary-file I/O included | 191.389 | 191.132 | 5.2 | 188.088 | 194.537 | 0.1 | yes |
| write_image | FHD uint8 RGB JPEG file, quality=95, temporary-file I/O included | 40.242 | 39.961 | 25.0 | 38.566 | 42.843 | 0.3 | yes |
| write_image | FHD uint8 RGB TIFF file, temporary-file I/O included | 53.168 | 52.748 | 19.0 | 49.974 | 56.954 | 0.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 37.445 | 37.434 | 26.7 | 36.711 | 38.142 | 1.0 | yes |
| write_image | FHD fp32 RGB to EXR NONE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 8.327 | 7.960 | 125.6 | 5.935 | 10.367 | 4.7 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 30.449 | 30.193 | 33.1 | 29.187 | 32.913 | 1.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIPS/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 102.259 | 87.948 | 11.4 | 83.796 | 133.617 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAA/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 82.913 | 80.179 | 12.5 | 76.024 | 93.987 | 0.5 | yes |
| write_image | FHD fp32 RGB to EXR DWAB/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 32.017 | 31.072 | 32.2 | 30.216 | 38.759 | 1.2 | yes |
| write_image | FHD fp32 RGB to EXR RLE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 15.722 | 15.864 | 63.0 | 14.017 | 16.930 | 2.4 | yes |
| write_image | FHD fp32 RGB to EXR PXR24/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 39.391 | 39.146 | 25.5 | 38.610 | 40.669 | 1.0 | yes |
| write_image | FHD fp16 RGB to EXR B44/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 5.132 | 4.689 | 213.3 | 4.179 | 7.191 | 5.3 | yes |
| write_image | FHD fp16 RGB to EXR B44A/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 5.524 | 5.085 | 196.7 | 4.343 | 7.113 | 4.9 | yes |
| write_image | FHD fp16 RGB to EXR PIZ/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 76.965 | 76.885 | 13.0 | 76.601 | 77.660 | 0.3 | yes |
| write_image | FHD uint8 RGB JPEG 2000 file, lossless, temporary-file I/O included | 60.666 | 60.804 | 16.4 | 59.185 | 62.358 | 0.2 | yes |
| write_image | FHD uint8 RGB WebP file, lossless, temporary-file I/O included | 283.622 | 280.763 | 3.6 | 277.983 | 295.686 | 0.0 | yes |
| write_image | FHD uint8 RGB BMP file, temporary-file I/O included | 23.901 | 23.783 | 42.0 | 22.447 | 25.078 | 0.5 | yes |
| write_image | FHD uint8 RGB PNM file, temporary-file I/O included | 25.104 | 24.976 | 40.0 | 23.783 | 26.292 | 0.5 | yes |
| write_image | FHD uint8 RGB TGA file, temporary-file I/O and CPU RLE included | 49.957 | 49.850 | 20.1 | 49.106 | 50.536 | 0.2 | yes |
| write_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 271.884 | 271.828 | 3.7 | 270.170 | 273.259 | 0.1 | yes |
| write_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU packing included | 9.615 | 9.445 | 105.9 | 9.014 | 10.043 | 3.5 | yes |
| read_header | FHD uint8 RGB PNG header, temporary-file I/O included | 2.021 | 2.013 | 496.7 | 1.981 | 2.098 | 0.0 | yes |
| decode_lut | 65-sample RGB Cube 1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.170 | 0.165 | 6077.5 | 0.161 | 0.197 | 0.0 |  |
| decode_lut | 65^3 RGB Cube 3D UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 141.562 | 140.606 | 7.1 | 139.594 | 146.818 | 0.1 | yes |
| decode_lut | 17^3 RGB headerless 3DL UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 14.098 | 14.026 | 71.3 | 13.780 | 14.681 | 0.0 | yes |
| decode_lut | 65-sample RGB SPI1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.169 | 0.163 | 6150.1 | 0.160 | 0.201 | 0.0 |  |
| decode_lut | 17^3 RGB SPI3D UTF-8 bytes, sniff, explicit-index parse, and host-to-device transfer included | 9.078 | 9.020 | 110.9 | 8.822 | 9.624 | 0.0 | yes |
| decode_image | FHD uint8 RGB PNG, unchanged, host bytes exchange included | 26.893 | 26.836 | 37.3 | 25.919 | 27.938 | 0.5 | yes |
| decode_image | FHD uint8 RGB JPEG, unchanged, host bytes exchange included | 36.983 | 36.980 | 27.0 | 36.454 | 37.644 | 0.3 | yes |
| decode_image | FHD uint8 RGB TIFF, unchanged, host bytes exchange included | 35.948 | 35.929 | 27.8 | 34.972 | 36.936 | 0.3 | yes |
| decode_image | FHD uint8 RGB JPEG 2000, unchanged, host bytes exchange included | 23.041 | 22.998 | 43.5 | 22.751 | 23.367 | 0.5 | yes |
| decode_image | FHD uint8 RGB WebP, unchanged, host bytes exchange included | 41.484 | 41.148 | 24.3 | 39.356 | 44.096 | 0.3 | yes |
| decode_image | FHD uint8 RGB BMP, unchanged, host bytes exchange included | 20.747 | 20.679 | 48.4 | 19.689 | 22.019 | 0.6 | yes |
| decode_image | FHD uint8 RGB PNM, unchanged, host bytes exchange included | 21.892 | 21.740 | 46.0 | 20.725 | 23.174 | 0.6 | yes |
| encode_image | FHD uint8 RGB PNG, compression_level=4, host bytes exchange included | 188.544 | 187.833 | 5.3 | 185.610 | 193.048 | 0.1 | yes |
| encode_image | FHD uint8 RGB JPEG, quality=95, host bytes exchange included | 37.496 | 37.401 | 26.7 | 35.573 | 39.296 | 0.3 | yes |
| encode_image | FHD uint8 RGB TIFF, host bytes exchange included | 52.412 | 51.343 | 19.5 | 48.844 | 55.989 | 0.2 | yes |
| encode_image | FHD uint8 RGB JPEG 2000, lossless, host bytes exchange included | 57.881 | 57.166 | 17.5 | 55.954 | 60.885 | 0.2 | yes |
| encode_image | FHD uint8 RGB WebP, lossless, host bytes exchange included | 281.891 | 279.923 | 3.6 | 276.639 | 289.014 | 0.0 | yes |
| encode_image | FHD uint8 RGB BMP, host bytes exchange included | 20.542 | 20.560 | 48.6 | 19.295 | 21.634 | 0.6 | yes |
| encode_image | FHD uint8 RGB PNM, host bytes exchange included | 21.809 | 21.692 | 46.1 | 20.536 | 23.190 | 0.6 | yes |
