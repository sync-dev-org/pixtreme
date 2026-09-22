# Performance

This is the complete 210-case FHD measurement report from a single full-suite `uv run pytest -m performance` run
at pixtreme commit `260fe6e`. The run completed with 267 passed and 5,346 deselected in 3,337.74 seconds
(0:55:38).
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
| NONE | native | GPU | 31.150 | 8.636 |
| RLE | GPU | GPU | 171.050 | 17.439 |
| ZIPS | custom CPU | GPU | 68.132 | 91.344 |
| ZIP | custom CPU | GPU | 35.358 | 35.517 |
| PIZ | GPU | GPU | 36.527 | 89.619 |
| PXR24 | custom CPU | GPU | 347.039 | 49.000 |
| B44 | GPU | GPU | 71.817 | 6.994 |
| B44A | GPU | GPU | 106.062 | 8.575 |
| DWAA | GPU | GPU | 28.228 | 97.956 |
| DWAB | GPU | GPU | 28.935 | 37.577 |

The default float32-frame write case omits `dtype`, stores ZIP-compressed HALF, and measured 40.572 ms. Reading that
HALF fixture unchanged through the fixed custom CPU ZIP lane measured 40.758 ms. These general default-path cases use
a different deterministic corpus from the compression rows above.

## Full results

| target | representative parameters | mean ms | median ms | fps | p5 ms | p95 ms | effective GB/s | > 1 ms |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| copy | FHD fp32 RGB read+write | 0.107 | 0.101 | 9871.7 | 0.099 | 0.128 | 491.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=nearest | 0.079 | 0.074 | 13579.6 | 0.071 | 0.104 | 422.4 |  |
| resize | 1920x1080 -> 960x540, interpolation=bilinear | 0.102 | 0.094 | 10586.5 | 0.090 | 0.126 | 329.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=bicubic | 0.146 | 0.141 | 7081.7 | 0.138 | 0.171 | 220.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=b-spline | 0.148 | 0.144 | 6959.9 | 0.139 | 0.170 | 216.5 |  |
| resize | 1920x1080 -> 960x540, interpolation=mitchell | 0.153 | 0.145 | 6889.9 | 0.139 | 0.184 | 214.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos2 | 0.149 | 0.143 | 6997.9 | 0.139 | 0.175 | 217.7 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos3 | 0.154 | 0.148 | 6774.4 | 0.140 | 0.182 | 210.7 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos4 | 0.234 | 0.228 | 4382.1 | 0.223 | 0.259 | 136.3 |  |
| resize | 1920x1080 -> 960x540, interpolation=area | 0.173 | 0.167 | 5990.2 | 0.159 | 0.207 | 186.3 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=nearest | 0.308 | 0.305 | 3276.6 | 0.298 | 0.327 | 407.7 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bilinear | 0.270 | 0.267 | 3740.4 | 0.261 | 0.288 | 465.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bicubic | 0.482 | 0.479 | 2086.1 | 0.473 | 0.496 | 259.5 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=b-spline | 0.484 | 0.481 | 2079.0 | 0.474 | 0.502 | 258.7 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=mitchell | 0.484 | 0.481 | 2079.8 | 0.475 | 0.501 | 258.8 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos2 | 0.485 | 0.482 | 2076.0 | 0.476 | 0.505 | 258.3 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos3 | 0.582 | 0.579 | 1726.9 | 0.570 | 0.604 | 214.9 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos4 | 0.983 | 0.980 | 1020.2 | 0.969 | 1.009 | 126.9 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=area | 0.696 | 0.694 | 1441.6 | 0.683 | 0.722 | 179.4 |  |
| warp_affine | FHD fp32 RGB, centered 1.01x scale + 5deg rotation, auto lanczos4, constant 0 | 3.262 | 3.248 | 307.9 | 3.214 | 3.359 | 15.3 | yes |
| stack | 2x FHD fp32 RGB, direction=vertical, adapt=False | 0.218 | 0.213 | 4704.2 | 0.207 | 0.244 | 468.2 |  |
| shuffle | single FHD fp32 Frame BGR reorder, adapt=False | 0.138 | 0.132 | 7550.6 | 0.129 | 0.161 | 375.8 |  |
| shuffle | FHD fp32 RGBA from 2 Frames + constant, adapt=False | 0.199 | 0.192 | 5213.8 | 0.187 | 0.231 | 302.7 |  |
| shuffle | 2 FHD fp32 RGB Frames, sRGB/sRGB source adapted to ACEScg/linear | 0.281 | 0.275 | 3641.7 | 0.268 | 0.316 | 362.5 |  |
| merge | FHD background + transformed 960x540 foreground, bilinear, normal | 0.488 | 0.474 | 2111.4 | 0.457 | 0.559 | 118.2 |  |
| gaussian_blur | sigma=1 | 0.581 | 0.578 | 1729.2 | 0.571 | 0.598 | 86.1 |  |
| gaussian_blur | sigma=2 | 0.679 | 0.677 | 1476.2 | 0.665 | 0.700 | 73.5 |  |
| gaussian_blur | sigma=4 | 0.930 | 0.928 | 1077.3 | 0.918 | 0.949 | 53.6 |  |
| unsharp_mask | sigma=2, amount=1 | 0.789 | 0.786 | 1272.6 | 0.778 | 0.809 | 63.3 |  |
| box_blur | size=3 | 0.495 | 0.488 | 2051.0 | 0.476 | 0.535 | 102.1 |  |
| box_blur | size=9 | 0.553 | 0.551 | 1815.3 | 0.540 | 0.580 | 90.3 |  |
| median_blur | size=3 | 0.358 | 0.356 | 2806.1 | 0.340 | 0.390 | 139.6 |  |
| median_blur | size=5 | 0.769 | 0.766 | 1305.7 | 0.745 | 0.805 | 65.0 |  |
| median_blur | size=7 | 1.048 | 1.045 | 957.3 | 1.018 | 1.085 | 47.6 | yes |
| bilateral_blur | sigma_space=1, sigma_value=0.1 | 0.414 | 0.411 | 2434.3 | 0.401 | 0.438 | 121.1 |  |
| bilateral_blur | sigma_space=2, sigma_value=0.1 | 0.907 | 0.905 | 1105.6 | 0.891 | 0.929 | 55.0 |  |
| convolve_box | size=(1,31), normalize=True | 0.573 | 0.571 | 1750.2 | 0.560 | 0.592 | 87.1 |  |
| erosion | radius=5, shape=disk | 0.342 | 0.337 | 2969.0 | 0.332 | 0.362 | 147.8 |  |
| dilation | radius=5, shape=disk | 0.344 | 0.336 | 2978.7 | 0.329 | 0.385 | 148.2 |  |
| opening | radius=5, shape=disk | 0.633 | 0.630 | 1587.7 | 0.611 | 0.680 | 79.0 |  |
| closing | radius=5, shape=disk | 0.627 | 0.624 | 1603.7 | 0.607 | 0.677 | 79.8 |  |
| morphological_gradient | radius=5, shape=disk | 0.368 | 0.362 | 2763.8 | 0.354 | 0.419 | 137.5 |  |
| white_tophat | radius=5, shape=disk | 0.628 | 0.626 | 1598.6 | 0.620 | 0.643 | 79.6 |  |
| black_tophat | radius=5, shape=disk | 0.626 | 0.625 | 1600.5 | 0.615 | 0.644 | 79.7 |  |
| sobel | direction=x | 0.500 | 0.497 | 2013.0 | 0.489 | 0.518 | 100.2 |  |
| sobel | direction=y | 0.501 | 0.499 | 2005.7 | 0.490 | 0.519 | 99.8 |  |
| sobel | direction=magnitude | 0.510 | 0.507 | 1970.7 | 0.498 | 0.526 | 98.1 |  |
| laplacian | kernel=3x3 | 0.295 | 0.290 | 3444.7 | 0.286 | 0.313 | 171.4 |  |
| canny | threshold_low=0.5, threshold_high=1.0, border=mirror | 2.387 | 2.385 | 419.4 | 2.231 | 2.548 | 20.9 | yes |
| sharpen | amount=1, border=mirror | 0.487 | 0.482 | 2074.1 | 0.478 | 0.506 | 103.2 |  |
| difference_of_gaussians | sigma1=1, sigma2=2 | 1.317 | 1.315 | 760.5 | 1.302 | 1.337 | 37.8 | yes |
| corner_harris | FHD fp32 RGB, block_size=3, k=0.04, border=mirror | 0.592 | 0.589 | 1696.8 | 0.577 | 0.610 | 56.3 |  |
| match_template | FHD fp32 RGB + 64x64 fp32 RGB, method=ccoeff_normed | 13.689 | 13.676 | 73.1 | 13.574 | 13.829 | 2.4 | yes |
| psnr | FHD fp32 RGB reference/candidate, data_range=1.0 default | 0.344 | 0.322 | 3108.2 | 0.301 | 0.444 | 154.7 |  |
| ssim | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.161 | 2.158 | 463.4 | 2.142 | 2.190 | 23.1 | yes |
| ssim_map | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.146 | 2.142 | 466.8 | 2.123 | 2.187 | 27.0 | yes |
| equalize_histogram | domain=(0,1), bins=1024 | 0.917 | 0.913 | 1094.7 | 0.908 | 0.941 | 54.5 |  |
| clahe | clip_limit=2, tiles_y=8, tiles_x=8, domain=(0,1), bins=1024 | 3.071 | 3.064 | 326.4 | 3.053 | 3.114 | 16.2 | yes |
| directional_blur | angle=30, length=8 | 0.543 | 0.537 | 1863.6 | 0.524 | 0.588 | 92.7 |  |
| directional_blur | angle=30, length=32 | 1.647 | 1.647 | 607.1 | 1.627 | 1.670 | 30.2 | yes |
| directional_blur | angle=30, length=128 | 7.843 | 7.832 | 127.7 | 7.736 | 7.992 | 6.4 | yes |
| zoom_blur | amount=0.05 | 1.971 | 1.969 | 508.0 | 1.947 | 2.004 | 25.3 | yes |
| zoom_blur | amount=0.2 | 9.540 | 9.529 | 104.9 | 9.390 | 9.736 | 5.2 | yes |
| spin_blur | angle=2 | 1.294 | 1.292 | 774.0 | 1.278 | 1.318 | 38.5 | yes |
| spin_blur | angle=10 | 8.435 | 8.418 | 118.8 | 8.288 | 8.630 | 5.9 | yes |
| vector_blur | uniform \|v\|=8, shutter=centered | 0.838 | 0.836 | 1195.7 | 0.824 | 0.857 | 79.3 |  |
| vector_blur | uniform \|v\|=32, shutter=centered | 1.878 | 1.877 | 532.9 | 1.862 | 1.898 | 35.4 | yes |
| vector_blur | uniform \|v\|=128, shutter=centered | 7.712 | 7.701 | 129.9 | 7.595 | 7.861 | 8.6 | yes |
| vector_blur | rotation field, corner \|v\|=32, shutter=centered | 1.197 | 1.194 | 837.3 | 1.177 | 1.223 | 55.6 | yes |
| lens_blur | circle radius=4 | 0.747 | 0.745 | 1342.5 | 0.734 | 0.766 | 66.8 |  |
| lens_blur | circle radius=8 | 1.633 | 1.630 | 613.5 | 1.618 | 1.654 | 30.5 | yes |
| lens_blur | circle radius=16 | 1.643 | 1.640 | 609.7 | 1.628 | 1.662 | 30.3 | yes |
| lens_blur | circle radius=32 | 1.425 | 1.423 | 702.6 | 1.412 | 1.442 | 35.0 | yes |
| lens_blur | blades=6, radius=16 | 1.643 | 1.640 | 609.7 | 1.628 | 1.662 | 30.3 | yes |
| lens_blur | blades=6, radius=32 | 1.427 | 1.424 | 702.2 | 1.412 | 1.449 | 34.9 | yes |
| line | diagonal thickness=4, aa=distance | 0.189 | 0.190 | 5261.2 | 0.168 | 0.222 | 261.8 |  |
| polyline | 5 points, closed, thickness=6, aa=distance | 0.207 | 0.200 | 5002.8 | 0.196 | 0.239 | 249.0 |  |
| rectangle | 1280x720 fill, corner_radius=48, aa=distance | 0.206 | 0.199 | 5028.6 | 0.195 | 0.232 | 250.3 |  |
| circle | fill radius=320, aa=supersample | 0.188 | 0.182 | 5508.1 | 0.178 | 0.214 | 274.1 |  |
| ellipse | radii=(520,260), rotation=25, thickness=8 | 0.176 | 0.170 | 5878.5 | 0.167 | 0.204 | 292.6 |  |
| polygon | 8-point concave fill, aa=distance | 0.238 | 0.232 | 4312.9 | 0.228 | 0.266 | 214.6 |  |
| text | single-line CJK, size=64, one outline, supersample=False | 0.393 | 0.379 | 2641.3 | 0.352 | 0.473 | 131.4 |  |
| text | single-line CJK, size=64, one outline, supersample=True | 0.409 | 0.401 | 2496.5 | 0.373 | 0.468 | 124.2 |  |
| ramp | FHD linear RGB | 0.158 | 0.162 | 6160.3 | 0.135 | 0.190 | 153.3 |  |
| grid | FHD cell=(64,64), line_width=2, aa=distance | 0.144 | 0.137 | 7276.2 | 0.131 | 0.174 | 181.1 |  |
| checkerboard | FHD cell=(64,64), aa=distance | 0.139 | 0.133 | 7499.1 | 0.130 | 0.164 | 186.6 |  |
| color_bars | FHD ARIB STD-B28 normalized | 0.095 | 0.091 | 10969.6 | 0.085 | 0.118 | 273.0 |  |
| fractal_noise | FHD scale=64, octaves=4 | 2.266 | 2.263 | 441.9 | 2.250 | 2.293 | 3.7 | yes |
| turbulent_noise | FHD scale=64, octaves=4 | 2.274 | 2.266 | 441.2 | 2.250 | 2.334 | 3.7 | yes |
| grain | FHD intensity=0.1, size=1, RGB | 3.768 | 3.764 | 265.7 | 3.749 | 3.803 | 6.6 | yes |
| from_array | CHW + affine scale=255 -> float32 HWC | 0.150 | 0.144 | 6941.1 | 0.140 | 0.178 | 345.4 |  |
| from_array | CHW uint16, bit_depth=10 -> float32 HWC | 0.129 | 0.123 | 8112.2 | 0.120 | 0.156 | 302.8 |  |
| to_array | BGR + NCHW + float16 + affine | 0.133 | 0.123 | 8134.1 | 0.117 | 0.185 | 303.6 |  |
| to_array | bit_depth=10 -> uint16 HWC | 0.118 | 0.111 | 8991.2 | 0.107 | 0.147 | 335.6 |  |
| rgb_to_rgb | ACEScg linear -> sRGB sRGB | 0.147 | 0.141 | 7106.3 | 0.137 | 0.174 | 353.7 |  |
| rgb_to_rgb | ACES 1.3 analytic -> sRGB sRGB | 0.183 | 0.187 | 5334.7 | 0.161 | 0.210 | 265.5 |  |
| rgb_to_rgb | ACES 2.0 analytic -> sRGB sRGB | 0.184 | 0.184 | 5434.2 | 0.156 | 0.214 | 270.4 |  |
| rgb_to_rgb | BT.2408 direct mapping -> Rec.2020 pq | 0.173 | 0.173 | 5793.1 | 0.145 | 0.202 | 288.3 |  |
| chromatic_adaptation | FHD fp32 RGB, D50 input -> D60 output, CAT02 | 0.176 | 0.180 | 5565.8 | 0.151 | 0.201 | 277.0 |  |
| grade | FHD fp32 RGB, per-channel Lift / Gamma / Gain | 0.243 | 0.235 | 4261.0 | 0.225 | 0.288 | 212.1 |  |
| white_balance | FHD fp32 RGB, Temperature=5000 K, Tint=0 Duv, CAT02 | 0.176 | 0.170 | 5899.7 | 0.165 | 0.205 | 293.6 |  |
| white_point_simulation | FHD fp32 RGB, D65 input display -> D93 output display | 0.152 | 0.146 | 6862.1 | 0.141 | 0.177 | 341.5 |  |
| rgb_to_ycbcr | RGB -> YCbCr, matrix=native | 0.160 | 0.153 | 6545.8 | 0.148 | 0.190 | 325.8 |  |
| rgb_to_hsv | RGB -> HSV, label-driven scene values | 0.129 | 0.121 | 8276.1 | 0.115 | 0.159 | 411.9 |  |
| hsv_to_rgb | HSV six sectors, S=[0,1], V=[0,2] -> RGB | 0.125 | 0.120 | 8363.3 | 0.116 | 0.147 | 416.2 |  |
| ycbcr_to_rgb | YCbCr -> RGB, matrix=bt709 | 0.156 | 0.150 | 6681.8 | 0.146 | 0.184 | 332.5 |  |
| rgb_to_grayscale | RGB -> Y, matrix=native | 0.136 | 0.131 | 7640.6 | 0.124 | 0.163 | 253.5 |  |
| gamma_to_linear | gamma=Gamma-2.6 claim -> linear | 0.154 | 0.148 | 6741.3 | 0.144 | 0.178 | 335.5 |  |
| linear_to_gamma | linear -> gamma=Gamma-2.6 | 0.154 | 0.149 | 6729.1 | 0.144 | 0.179 | 334.9 |  |
| ycbcr_to_ycbcr | YCbCr bt709 -> native rematrix | 0.159 | 0.153 | 6521.0 | 0.149 | 0.185 | 324.5 |  |
| full_to_legal | full -> legal, bit_depth=10 | 0.134 | 0.129 | 7755.0 | 0.125 | 0.160 | 385.9 |  |
| legal_to_full | legal -> full, bit_depth=10 | 0.133 | 0.128 | 7815.6 | 0.125 | 0.157 | 389.0 |  |
| quantize | float32 -> uint8, bit_depth=8 | 0.109 | 0.111 | 8979.9 | 0.089 | 0.142 | 279.3 |  |
| dequantize | uint8 -> float32, bit_depth=8 | 0.115 | 0.113 | 8881.9 | 0.089 | 0.145 | 276.3 |  |
| cast_dtype | float32 -> float16 | 0.107 | 0.100 | 9979.0 | 0.096 | 0.132 | 372.5 |  |
| recode_dtype | uint8 -> float32 | 0.100 | 0.093 | 10714.8 | 0.089 | 0.124 | 333.3 |  |
| recode_dtype | float32 -> uint8 | 0.106 | 0.113 | 8876.3 | 0.086 | 0.132 | 276.1 |  |
| from_uyvy422 | legal range | 0.107 | 0.101 | 9936.5 | 0.093 | 0.132 | 288.5 |  |
| from_v210 | legal range | 0.111 | 0.102 | 9798.2 | 0.096 | 0.159 | 298.0 |  |
| from_nv12 | legal range, siting=left, interpolation=bilinear | 0.101 | 0.096 | 10415.6 | 0.093 | 0.118 | 291.6 |  |
| from_p010 | legal range, siting=left, interpolation=bilinear | 0.104 | 0.099 | 10083.8 | 0.097 | 0.123 | 313.6 |  |
| from_p216 | FHD 16-bit legal, interpolation=bilinear | 0.109 | 0.104 | 9613.4 | 0.101 | 0.132 | 319.0 |  |
| from_yuv420p | legal range, interpolation=bilinear | 0.102 | 0.097 | 10349.9 | 0.094 | 0.126 | 289.7 |  |
| from_yuv422p | legal range | 0.108 | 0.103 | 9673.8 | 0.100 | 0.132 | 321.0 |  |
| from_yuv444p | 10-bit legal range | 0.111 | 0.105 | 9495.0 | 0.102 | 0.134 | 354.4 |  |
| from_yuva444p | 12-bit legal range | 0.129 | 0.124 | 8074.9 | 0.120 | 0.153 | 401.9 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=trilinear | 0.372 | 0.367 | 2721.3 | 0.363 | 0.389 | 135.4 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=tetrahedral | 0.371 | 0.367 | 2726.8 | 0.362 | 0.389 | 135.7 |  |
| apply_lut | FHD fp32 RGB, 65-sample 1D LUT, interpolation=linear | 0.375 | 0.369 | 2706.7 | 0.364 | 0.398 | 134.7 |  |
| apply_lut | FHD fp32 RGB, 65^3 baked LUT, interpolation=None | 0.366 | 0.361 | 2767.4 | 0.357 | 0.389 | 137.7 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, 65-sample nonidentity shaper, interpolation=None | 0.367 | 0.363 | 2758.4 | 0.357 | 0.390 | 137.3 |  |
| to_uyvy422 | FHD area, legal | 0.095 | 0.091 | 11042.5 | 0.088 | 0.118 | 320.6 |  |
| to_v210 | FHD area, legal, 128-byte rows | 0.163 | 0.159 | 6295.6 | 0.151 | 0.188 | 191.5 |  |
| to_nv12 | FHD area, legal, siting=left | 0.148 | 0.144 | 6942.5 | 0.136 | 0.173 | 194.3 |  |
| to_p010 | FHD area, legal, siting=left | 0.156 | 0.160 | 6266.1 | 0.136 | 0.187 | 194.9 |  |
| to_p216 | FHD 16-bit area, legal | 0.275 | 0.272 | 3679.0 | 0.261 | 0.299 | 122.1 |  |
| to_yuv420p | 8-bit area, legal, siting=left | 0.148 | 0.141 | 7083.7 | 0.134 | 0.176 | 198.3 |  |
| to_yuv422p | 10-bit area, legal | 0.131 | 0.126 | 7959.3 | 0.123 | 0.156 | 264.1 |  |
| to_yuv444p | 10-bit legal | 0.104 | 0.099 | 10059.9 | 0.096 | 0.129 | 375.5 |  |
| to_yuva444p | 12-bit legal, alpha full | 0.122 | 0.117 | 8563.1 | 0.114 | 0.143 | 426.2 |  |
| read_lut | 65^3 RGB .cube file, parse, float4 packing, and host-to-device transfer included | 143.370 | 142.140 | 7.0 | 140.751 | 147.065 | 0.1 | yes |
| read_lut | 65-sample RGB Cube 1D file, parse and host-to-device transfer included | 0.255 | 0.247 | 4053.4 | 0.242 | 0.293 | 0.0 |  |
| read_lut | 17^3 RGB headerless 3DL file, parse, packing, and host-to-device transfer included | 9.148 | 9.108 | 109.8 | 8.706 | 9.685 | 0.0 | yes |
| read_lut | 65-sample RGB SPI1D file, parse and host-to-device transfer included | 0.223 | 0.213 | 4701.4 | 0.207 | 0.266 | 0.0 |  |
| read_lut | 17^3 RGB SPI3D file, explicit-index parse, packing, and host-to-device transfer included | 9.362 | 9.277 | 107.8 | 8.992 | 9.910 | 0.0 | yes |
| write_lut | 65-sample RGB Lut1D, device-to-host transfer and Cube file write included | 0.275 | 0.256 | 3904.3 | 0.242 | 0.348 | 0.0 |  |
| write_lut | 65^3 RGB Lut, device-to-host transfer and Cube file write included | 338.835 | 338.215 | 3.0 | 335.800 | 345.481 | 0.0 | yes |
| read_image | FHD uint8 RGB PNG file, unchanged, temporary-file I/O included | 26.331 | 26.303 | 38.0 | 25.595 | 27.139 | 0.5 | yes |
| read_image | FHD uint8 RGB JPEG file, unchanged, temporary-file I/O included | 39.685 | 39.595 | 25.3 | 38.852 | 40.945 | 0.3 | yes |
| read_image | FHD uint8 RGB TIFF file, unchanged, temporary-file I/O included | 34.084 | 34.012 | 29.4 | 33.437 | 35.035 | 0.4 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 40.772 | 40.758 | 24.5 | 39.715 | 41.751 | 0.6 | yes |
| read_image | FHD HALF RGB EXR NONE file, unchanged, source-fixed native lane, temporary-file I/O included | 31.244 | 31.150 | 32.1 | 30.700 | 32.060 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 35.416 | 35.358 | 28.3 | 34.365 | 36.480 | 0.7 | yes |
| read_image | FHD HALF RGB EXR ZIPS file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 69.174 | 68.132 | 14.7 | 65.208 | 70.519 | 0.4 | yes |
| read_image | FHD HALF RGB EXR DWAA file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 27.545 | 28.228 | 35.4 | 25.345 | 29.413 | 0.9 | yes |
| read_image | FHD HALF RGB EXR DWAB file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 28.931 | 28.935 | 34.6 | 27.831 | 30.092 | 0.9 | yes |
| read_image | FHD HALF RGB EXR RLE file, unchanged, source-fixed GPU lane, temporary-file I/O included | 188.840 | 171.050 | 5.8 | 163.611 | 239.552 | 0.1 | yes |
| read_image | FHD HALF RGB EXR PXR24 file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 350.382 | 347.039 | 2.9 | 312.986 | 392.348 | 0.1 | yes |
| read_image | FHD HALF RGB EXR B44 file, unchanged, source-fixed GPU lane, temporary-file I/O included | 84.025 | 71.817 | 13.9 | 70.397 | 137.745 | 0.3 | yes |
| read_image | FHD HALF RGB EXR B44A file, unchanged, source-fixed GPU lane, temporary-file I/O included | 121.223 | 106.062 | 9.4 | 103.976 | 174.478 | 0.2 | yes |
| read_image | FHD HALF RGB EXR PIZ file, unchanged, source-fixed GPU lane, temporary-file I/O included | 36.605 | 36.527 | 27.4 | 35.820 | 37.495 | 0.7 | yes |
| read_image | FHD uint8 RGB JPEG 2000 file, unchanged, temporary-file I/O included | 26.693 | 26.771 | 37.4 | 25.912 | 27.341 | 0.5 | yes |
| read_image | FHD uint8 RGB WebP file, unchanged, temporary-file I/O included | 42.673 | 42.428 | 23.6 | 41.588 | 44.020 | 0.3 | yes |
| read_image | FHD uint8 RGB BMP file, unchanged, temporary-file I/O included | 17.978 | 17.909 | 55.8 | 17.270 | 18.930 | 0.7 | yes |
| read_image | FHD uint8 RGB PNM file, unchanged, temporary-file I/O included | 19.522 | 19.473 | 51.4 | 18.880 | 20.477 | 0.6 | yes |
| read_image | FHD uint8 RGB TGA file, unchanged, temporary-file I/O and CPU RLE included | 12.717 | 12.693 | 78.8 | 12.464 | 13.016 | 1.0 | yes |
| read_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 135.779 | 135.303 | 7.4 | 134.274 | 137.991 | 0.2 | yes |
| read_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU unpack included | 5.421 | 5.329 | 187.7 | 5.047 | 5.971 | 6.2 | yes |
| write_image | FHD uint8 RGB PNG file, compression_level=4, temporary-file I/O included | 208.632 | 208.120 | 4.8 | 206.641 | 211.241 | 0.1 | yes |
| write_image | FHD uint8 RGB JPEG file, quality=95, temporary-file I/O included | 43.921 | 43.792 | 22.8 | 41.863 | 45.858 | 0.3 | yes |
| write_image | FHD uint8 RGB TIFF file, temporary-file I/O included | 66.946 | 65.641 | 15.2 | 61.849 | 77.690 | 0.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 40.558 | 40.572 | 24.6 | 39.190 | 42.574 | 0.9 | yes |
| write_exr_channels | FHD HALF RGB + UINT ID to EXR ZIP, native mixed dtypes, temporary-file I/O included | 39.814 | 39.456 | 25.3 | 37.691 | 42.533 | 1.1 | yes |
| write_image | FHD fp32 RGB to EXR NONE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 9.898 | 8.636 | 115.8 | 6.821 | 23.191 | 4.3 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 35.501 | 35.517 | 28.2 | 34.582 | 36.549 | 1.1 | yes |
| write_image | FHD fp32 RGB to EXR ZIPS/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 88.596 | 91.344 | 10.9 | 74.821 | 101.590 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAA/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 98.686 | 97.956 | 10.2 | 96.073 | 102.609 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAB/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 37.819 | 37.577 | 26.6 | 36.186 | 39.941 | 1.0 | yes |
| write_image | FHD fp32 RGB to EXR RLE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 17.582 | 17.439 | 57.3 | 15.689 | 19.766 | 2.1 | yes |
| write_image | FHD fp32 RGB to EXR PXR24/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 49.056 | 49.000 | 20.4 | 47.516 | 50.839 | 0.8 | yes |
| write_image | FHD fp16 RGB to EXR B44/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 7.430 | 6.994 | 143.0 | 5.611 | 9.305 | 3.6 | yes |
| write_image | FHD fp16 RGB to EXR B44A/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 8.887 | 8.575 | 116.6 | 7.703 | 10.312 | 2.9 | yes |
| write_image | FHD fp16 RGB to EXR PIZ/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 89.619 | 89.619 | 11.2 | 88.488 | 90.919 | 0.3 | yes |
| write_image | FHD uint8 RGB JPEG 2000 file, lossless, temporary-file I/O included | 66.967 | 66.482 | 15.0 | 64.959 | 70.409 | 0.2 | yes |
| write_image | FHD uint8 RGB WebP file, lossless, temporary-file I/O included | 339.147 | 338.825 | 3.0 | 332.446 | 350.288 | 0.0 | yes |
| write_image | FHD uint8 RGB BMP file, temporary-file I/O included | 28.537 | 28.426 | 35.2 | 26.184 | 30.529 | 0.4 | yes |
| write_image | FHD uint8 RGB PNM file, temporary-file I/O included | 29.737 | 29.666 | 33.7 | 27.800 | 31.774 | 0.4 | yes |
| write_image | FHD uint8 RGB TGA file, temporary-file I/O and CPU RLE included | 56.015 | 55.918 | 17.9 | 55.226 | 57.089 | 0.2 | yes |
| write_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 306.368 | 305.810 | 3.3 | 304.391 | 310.072 | 0.1 | yes |
| write_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU packing included | 11.582 | 10.921 | 91.6 | 10.717 | 17.514 | 3.0 | yes |
| read_header | FHD uint8 RGB PNG header, temporary-file I/O included | 2.180 | 2.167 | 461.4 | 2.127 | 2.267 | 0.0 | yes |
| decode_lut | 65-sample RGB Cube 1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.205 | 0.197 | 5070.2 | 0.194 | 0.240 | 0.0 |  |
| decode_lut | 65^3 RGB Cube 3D UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 156.571 | 156.026 | 6.4 | 154.898 | 159.015 | 0.0 | yes |
| decode_lut | 17^3 RGB headerless 3DL UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included | 16.415 | 16.284 | 61.4 | 15.891 | 17.391 | 0.0 | yes |
| decode_lut | 65-sample RGB SPI1D UTF-8 bytes, sniff, parse, and host-to-device transfer included | 0.255 | 0.273 | 3659.4 | 0.184 | 0.318 | 0.0 |  |
| decode_lut | 17^3 RGB SPI3D UTF-8 bytes, sniff, explicit-index parse, and host-to-device transfer included | 10.572 | 10.481 | 95.4 | 10.060 | 11.306 | 0.0 | yes |
| decode_image | FHD uint8 RGB PNG, unchanged, host bytes exchange included | 24.775 | 24.662 | 40.5 | 23.657 | 25.960 | 0.5 | yes |
| decode_image | FHD uint8 RGB JPEG, unchanged, host bytes exchange included | 38.156 | 38.183 | 26.2 | 36.816 | 39.546 | 0.3 | yes |
| decode_image | FHD uint8 RGB TIFF, unchanged, host bytes exchange included | 33.780 | 33.805 | 29.6 | 32.928 | 34.730 | 0.4 | yes |
| decode_image | FHD uint8 RGB JPEG 2000, unchanged, host bytes exchange included | 25.887 | 25.828 | 38.7 | 25.239 | 26.608 | 0.5 | yes |
| decode_image | FHD uint8 RGB WebP, unchanged, host bytes exchange included | 42.893 | 42.668 | 23.4 | 41.428 | 44.504 | 0.3 | yes |
| decode_image | FHD uint8 RGB BMP, unchanged, host bytes exchange included | 17.828 | 17.814 | 56.1 | 17.324 | 18.359 | 0.7 | yes |
| decode_image | FHD uint8 RGB PNM, unchanged, host bytes exchange included | 19.384 | 19.111 | 52.3 | 18.468 | 20.448 | 0.7 | yes |
| encode_image | FHD uint8 RGB PNG, compression_level=4, host bytes exchange included | 205.912 | 205.928 | 4.9 | 204.092 | 207.917 | 0.1 | yes |
| encode_image | FHD uint8 RGB JPEG, quality=95, host bytes exchange included | 41.168 | 41.238 | 24.2 | 39.076 | 42.917 | 0.3 | yes |
| encode_image | FHD uint8 RGB TIFF, host bytes exchange included | 55.033 | 54.597 | 18.3 | 51.494 | 58.785 | 0.2 | yes |
| encode_image | FHD uint8 RGB JPEG 2000, lossless, host bytes exchange included | 64.383 | 64.177 | 15.6 | 61.810 | 67.616 | 0.2 | yes |
| encode_image | FHD uint8 RGB WebP, lossless, host bytes exchange included | 354.290 | 354.332 | 2.8 | 328.851 | 380.058 | 0.0 | yes |
| encode_image | FHD uint8 RGB BMP, host bytes exchange included | 24.074 | 23.774 | 42.1 | 21.837 | 27.142 | 0.5 | yes |
| encode_image | FHD uint8 RGB PNM, host bytes exchange included | 26.783 | 26.843 | 37.3 | 24.809 | 28.856 | 0.5 | yes |
