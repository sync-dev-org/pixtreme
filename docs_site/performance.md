# Performance

This is the complete 190-case FHD measurement report. The original 162 non-EXR values are copied without numerical
changes from the fresh full-suite run at pixtreme commit `9bc236d`; the two TGA file-boundary values were measured with
the same harness and conditions at commit `17ba4a5`, the two HDR file-boundary values at commit `2c5a54e`, and the two
DPX file-boundary values at commit `9d1c25e`. The 18 unchanged EXR public-boundary values were measured with the same
harness and conditions at commit `713edde`, after the final source-fixed routing and HALF write default were in place.
The DWAA, DWAB, RLE, and PXR24 read values were remeasured at commit `18c5ffa` after every EXR registry read fixture was
standardized on HALF storage. They characterize this hardware and workload; they are not performance guarantees for
other systems.

## Measurement conditions

- **GPU:** NVIDIA RTX A6000, driver 596.72. GPU 0 was dedicated to the run.
- **Environment:** WSL2 on Linux 6.18, CUDA runtime 12.9, CuPy 14.1.1, and Python 3.12.
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
| NONE | native | GPU | 30.507 | 7.546 |
| RLE | GPU | GPU | 171.123 | 17.213 |
| ZIPS | custom CPU | GPU | 63.145 | 88.328 |
| ZIP | custom CPU | GPU | 32.826 | 31.036 |
| PIZ | GPU | GPU | 33.598 | 71.584 |
| PXR24 | custom CPU | GPU | 305.966 | 41.802 |
| B44 | GPU | GPU | 70.301 | 5.175 |
| B44A | GPU | GPU | 91.025 | 5.050 |
| DWAA | GPU | GPU | 25.249 | 92.127 |
| DWAB | GPU | GPU | 28.637 | 36.316 |

The default float32-frame write case omits `dtype`, stores ZIP-compressed HALF, and measured 37.042 ms. Reading that
HALF fixture unchanged through the fixed custom CPU ZIP lane measured 30.925 ms. These general default-path cases use
a different deterministic corpus from the compression rows above.

### Registry coverage decision

This feature adds no public GPU pixel operation or boundary operation. The existing registry already covers
`read_image` and `write_image`, including a read/write pair for every EXR compression token, so no case was added. The
existing EXR cases were remeasured to follow the final routing and HALF default.

## Full results

| target | representative parameters | mean ms | median ms | fps | p5 ms | p95 ms | effective GB/s | > 1 ms |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| copy | FHD fp32 RGB read+write | 0.101 | 0.098 | 10240.3 | 0.092 | 0.117 | 509.6 |  |
| resize | 1920x1080 -> 960x540, interpolation=nearest | 0.069 | 0.067 | 15027.1 | 0.065 | 0.083 | 467.4 |  |
| resize | 1920x1080 -> 960x540, interpolation=bilinear | 0.090 | 0.085 | 11781.6 | 0.082 | 0.107 | 366.5 |  |
| resize | 1920x1080 -> 960x540, interpolation=bicubic | 0.133 | 0.129 | 7722.2 | 0.127 | 0.149 | 240.2 |  |
| resize | 1920x1080 -> 960x540, interpolation=b-spline | 0.133 | 0.130 | 7671.0 | 0.128 | 0.148 | 238.6 |  |
| resize | 1920x1080 -> 960x540, interpolation=mitchell | 0.133 | 0.130 | 7679.3 | 0.128 | 0.147 | 238.9 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos2 | 0.133 | 0.131 | 7654.6 | 0.128 | 0.148 | 238.1 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos3 | 0.136 | 0.133 | 7505.1 | 0.131 | 0.151 | 233.4 |  |
| resize | 1920x1080 -> 960x540, interpolation=lanczos4 | 0.215 | 0.209 | 4791.4 | 0.205 | 0.235 | 149.0 |  |
| resize | 1920x1080 -> 960x540, interpolation=area | 0.159 | 0.152 | 6594.4 | 0.146 | 0.187 | 205.1 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=nearest | 0.280 | 0.279 | 3588.8 | 0.276 | 0.290 | 446.5 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bilinear | 0.245 | 0.243 | 4111.2 | 0.240 | 0.254 | 511.5 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=bicubic | 0.444 | 0.442 | 2262.2 | 0.439 | 0.455 | 281.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=b-spline | 0.450 | 0.444 | 2252.9 | 0.439 | 0.485 | 280.3 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=mitchell | 0.457 | 0.444 | 2251.7 | 0.439 | 0.491 | 280.1 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos2 | 0.451 | 0.445 | 2245.4 | 0.441 | 0.484 | 279.4 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos3 | 0.546 | 0.536 | 1864.3 | 0.530 | 0.581 | 231.9 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=lanczos4 | 0.924 | 0.914 | 1093.7 | 0.904 | 0.960 | 136.1 |  |
| resize | 1920x1080 -> 3840x2160, interpolation=area | 0.650 | 0.650 | 1539.5 | 0.637 | 0.679 | 191.5 |  |
| warp_affine | FHD fp32 RGB, centered 1.01x scale + 5deg rotation, auto lanczos4, constant 0 | 3.063 | 3.043 | 328.7 | 3.023 | 3.144 | 16.4 | yes |
| stack | 2x FHD fp32 RGB, direction=vertical, adapt=False | 0.197 | 0.193 | 5191.1 | 0.190 | 0.215 | 516.7 |  |
| shuffle | single FHD fp32 Frame BGR reorder, adapt=False | 0.303 | 0.302 | 3306.9 | 0.288 | 0.313 | 164.6 |  |
| shuffle | FHD fp32 RGBA from 2 Frames + constant, adapt=False | 0.407 | 0.404 | 2476.2 | 0.399 | 0.418 | 143.8 |  |
| shuffle | 2 FHD fp32 RGB Frames, sRGB/srgb source adapted to ACEScg/linear | 0.409 | 0.392 | 2554.1 | 0.380 | 0.453 | 254.2 |  |
| merge | FHD background + transformed 960x540 foreground, bilinear, normal | 0.475 | 0.466 | 2146.6 | 0.454 | 0.515 | 120.2 |  |
| gaussian_blur | sigma=1 | 0.547 | 0.540 | 1851.4 | 0.533 | 0.581 | 92.1 |  |
| gaussian_blur | sigma=2 | 0.638 | 0.634 | 1577.4 | 0.623 | 0.668 | 78.5 |  |
| gaussian_blur | sigma=4 | 0.873 | 0.868 | 1151.5 | 0.865 | 0.888 | 57.3 |  |
| unsharp_mask | sigma=2, amount=1 | 0.743 | 0.738 | 1354.3 | 0.733 | 0.760 | 67.4 |  |
| box_blur | size=3 | 0.460 | 0.457 | 2190.2 | 0.447 | 0.495 | 109.0 |  |
| box_blur | size=9 | 0.528 | 0.522 | 1914.6 | 0.504 | 0.568 | 95.3 |  |
| median_blur | size=3 | 0.331 | 0.327 | 3058.0 | 0.320 | 0.352 | 152.2 |  |
| median_blur | size=5 | 0.720 | 0.717 | 1395.4 | 0.701 | 0.753 | 69.4 |  |
| median_blur | size=7 | 0.988 | 0.985 | 1015.6 | 0.959 | 1.029 | 50.5 |  |
| bilateral_blur | sigma_space=1, sigma_value=0.1 | 0.383 | 0.381 | 2626.1 | 0.378 | 0.396 | 130.7 |  |
| bilateral_blur | sigma_space=2, sigma_value=0.1 | 0.847 | 0.845 | 1182.8 | 0.837 | 0.861 | 58.9 |  |
| convolve_box | size=(1,31), normalize=True | 0.545 | 0.543 | 1840.5 | 0.540 | 0.556 | 91.6 |  |
| erosion | radius=5, shape=disk | 8.423 | 8.376 | 119.4 | 8.341 | 8.665 | 5.9 | yes |
| dilation | radius=5, shape=disk | 8.357 | 8.350 | 119.8 | 8.340 | 8.404 | 6.0 | yes |
| opening | radius=5, shape=disk | 16.720 | 16.668 | 60.0 | 16.499 | 16.887 | 3.0 | yes |
| closing | radius=5, shape=disk | 16.481 | 16.627 | 60.1 | 16.190 | 16.786 | 3.0 | yes |
| morphological_gradient | radius=5, shape=disk | 16.899 | 16.877 | 59.3 | 16.008 | 17.140 | 2.9 | yes |
| white_tophat | radius=5, shape=disk | 16.470 | 16.607 | 60.2 | 15.996 | 16.697 | 3.0 | yes |
| black_tophat | radius=5, shape=disk | 16.667 | 16.646 | 60.1 | 15.865 | 17.040 | 3.0 | yes |
| sobel | direction=x | 0.442 | 0.436 | 2294.4 | 0.430 | 0.461 | 114.2 |  |
| sobel | direction=y | 0.460 | 0.458 | 2185.2 | 0.455 | 0.471 | 108.8 |  |
| sobel | direction=magnitude | 0.477 | 0.481 | 2079.7 | 0.462 | 0.486 | 103.5 |  |
| laplacian | kernel=3x3 | 0.276 | 0.271 | 3686.7 | 0.266 | 0.288 | 183.5 |  |
| canny | threshold_low=0.5, threshold_high=1.0, border=mirror | 2.264 | 2.244 | 445.7 | 2.067 | 2.461 | 22.2 | yes |
| sharpen | amount=1, border=mirror | 0.465 | 0.457 | 2188.5 | 0.444 | 0.506 | 108.9 |  |
| difference_of_gaussians | sigma1=1, sigma2=2 | 1.265 | 1.261 | 792.9 | 1.250 | 1.284 | 39.5 | yes |
| corner_harris | FHD fp32 RGB, block_size=3, k=0.04, border=mirror | 0.563 | 0.559 | 1788.0 | 0.554 | 0.581 | 59.3 |  |
| match_template | FHD fp32 RGB + 64x64 fp32 RGB, method=ccoeff_normed | 24.131 | 24.280 | 41.2 | 23.519 | 24.421 | 1.3 | yes |
| psnr | FHD fp32 RGB reference/candidate, data_range=1.0 default | 0.333 | 0.294 | 3403.0 | 0.277 | 0.575 | 169.4 |  |
| ssim | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.052 | 2.041 | 490.1 | 2.027 | 2.097 | 24.4 | yes |
| ssim_map | FHD fp32 RGB reference/candidate, data_range=1.0 default | 2.007 | 1.998 | 500.6 | 1.990 | 2.042 | 29.0 | yes |
| equalize_histogram | domain=(0,1), bins=1024 | 0.857 | 0.848 | 1178.9 | 0.844 | 0.901 | 58.7 |  |
| clahe | clip_limit=2, tiles_y=8, tiles_x=8, domain=(0,1), bins=1024 | 2.858 | 2.860 | 349.7 | 2.844 | 2.872 | 17.4 | yes |
| directional_blur | angle=30, length=8 | 0.495 | 0.493 | 2029.7 | 0.488 | 0.508 | 101.0 |  |
| directional_blur | angle=30, length=32 | 1.544 | 1.538 | 650.3 | 1.524 | 1.569 | 32.4 | yes |
| directional_blur | angle=30, length=128 | 10.876 | 10.769 | 92.9 | 10.610 | 11.266 | 4.6 | yes |
| zoom_blur | amount=0.05 | 1.900 | 1.896 | 527.4 | 1.854 | 1.967 | 26.2 | yes |
| zoom_blur | amount=0.2 | 12.024 | 12.023 | 83.2 | 11.768 | 12.280 | 4.1 | yes |
| spin_blur | angle=2 | 1.201 | 1.203 | 831.2 | 1.184 | 1.217 | 41.4 | yes |
| spin_blur | angle=10 | 10.330 | 10.307 | 97.0 | 10.012 | 10.770 | 4.8 | yes |
| vector_blur | uniform \|v\|=8, shutter=centered | 0.816 | 0.811 | 1233.7 | 0.791 | 0.841 | 81.9 |  |
| vector_blur | uniform \|v\|=32, shutter=centered | 1.747 | 1.758 | 568.7 | 1.713 | 1.776 | 37.7 | yes |
| vector_blur | uniform \|v\|=128, shutter=centered | 10.330 | 10.326 | 96.8 | 10.227 | 10.430 | 6.4 | yes |
| vector_blur | rotation field, corner \|v\|=32, shutter=centered | 1.109 | 1.110 | 900.8 | 1.093 | 1.128 | 59.8 | yes |
| lens_blur | circle radius=4 | 0.707 | 0.707 | 1414.3 | 0.692 | 0.720 | 70.4 |  |
| lens_blur | circle radius=8 | 1.542 | 1.538 | 650.3 | 1.527 | 1.565 | 32.4 | yes |
| lens_blur | circle radius=16 | 1.588 | 1.594 | 627.2 | 1.538 | 1.615 | 31.2 | yes |
| lens_blur | circle radius=32 | 1.365 | 1.382 | 723.8 | 1.326 | 1.393 | 36.0 | yes |
| lens_blur | blades=6, radius=16 | 1.531 | 1.531 | 653.0 | 1.493 | 1.577 | 32.5 | yes |
| lens_blur | blades=6, radius=32 | 1.320 | 1.308 | 764.5 | 1.294 | 1.352 | 38.0 | yes |
| line | diagonal thickness=4, aa=distance | 0.165 | 0.167 | 5991.2 | 0.146 | 0.201 | 298.2 |  |
| polyline | 5 points, closed, thickness=6, aa=distance | 0.196 | 0.198 | 5055.9 | 0.174 | 0.225 | 251.6 |  |
| rectangle | 1280x720 fill, corner_radius=48, aa=distance | 0.192 | 0.193 | 5172.9 | 0.169 | 0.224 | 257.4 |  |
| circle | fill radius=320, aa=supersample | 0.181 | 0.182 | 5504.8 | 0.158 | 0.211 | 274.0 |  |
| ellipse | radii=(520,260), rotation=25, thickness=8 | 0.166 | 0.167 | 5984.6 | 0.149 | 0.207 | 297.8 |  |
| polygon | 8-point concave fill, aa=distance | 0.230 | 0.225 | 4436.2 | 0.208 | 0.270 | 220.8 |  |
| text | single-line CJK, size=64, one outline, supersample=False | 0.405 | 0.399 | 2508.9 | 0.332 | 0.494 | 124.9 |  |
| text | single-line CJK, size=64, one outline, supersample=True | 0.359 | 0.353 | 2836.7 | 0.337 | 0.396 | 141.2 |  |
| ramp | FHD linear RGB | 0.116 | 0.113 | 8869.3 | 0.111 | 0.133 | 220.7 |  |
| grid | FHD cell=(64,64), line_width=2, aa=distance | 0.118 | 0.113 | 8872.1 | 0.111 | 0.143 | 220.8 |  |
| checkerboard | FHD cell=(64,64), aa=distance | 0.120 | 0.115 | 8678.3 | 0.113 | 0.141 | 215.9 |  |
| color_bars | FHD ARIB STD-B28 normalized | 0.083 | 0.079 | 12689.9 | 0.076 | 0.099 | 315.8 |  |
| fractal_noise | FHD scale=64, octaves=4 | 4.212 | 4.217 | 237.1 | 4.145 | 4.235 | 2.0 | yes |
| turbulent_noise | FHD scale=64, octaves=4 | 4.224 | 4.220 | 237.0 | 4.205 | 4.275 | 2.0 | yes |
| grain | FHD intensity=0.1, size=1, RGB | 12.228 | 12.217 | 81.9 | 11.722 | 13.170 | 2.0 | yes |
| from_array | CHW + affine scale=255 -> float32 HWC | 0.132 | 0.128 | 7813.6 | 0.124 | 0.150 | 388.9 |  |
| from_array | CHW uint16, bit_depth=10 -> float32 HWC | 0.117 | 0.114 | 8787.2 | 0.112 | 0.135 | 328.0 |  |
| px.io.to_array | BGR + NCHW + float16 + affine | 0.117 | 0.109 | 9182.3 | 0.105 | 0.145 | 342.7 |  |
| px.io.to_array | bit_depth=10 -> uint16 HWC | 0.119 | 0.122 | 8177.5 | 0.099 | 0.146 | 305.2 |  |
| rgb_to_rgb | ACEScg linear -> sRGB srgb | 0.136 | 0.124 | 8095.9 | 0.119 | 0.192 | 402.9 |  |
| rgb_to_rgb | ACES 1.3 analytic -> sRGB srgb | 0.200 | 0.191 | 5249.0 | 0.177 | 0.250 | 261.2 |  |
| rgb_to_rgb | ACES 2.0 analytic -> sRGB srgb | 1.358 | 1.378 | 725.8 | 1.311 | 1.393 | 36.1 | yes |
| rgb_to_rgb | ACES 2.0 LUT -> sRGB srgb | 0.132 | 0.129 | 7734.0 | 0.125 | 0.150 | 384.9 |  |
| rgb_to_rgb | BT.2408 direct mapping -> Rec.2020 pq | 0.132 | 0.129 | 7742.6 | 0.127 | 0.149 | 385.3 |  |
| rgb_to_ycbcr | RGB -> YCbCr, matrix=native | 0.160 | 0.157 | 6389.2 | 0.133 | 0.206 | 318.0 |  |
| rgb_to_hsv | RGB -> HSV, label-driven scene values | 0.149 | 0.170 | 5892.0 | 0.105 | 0.180 | 293.2 |  |
| hsv_to_rgb | HSV six sectors, S=[0,1], V=[0,2] -> RGB | 0.114 | 0.108 | 9296.2 | 0.105 | 0.143 | 462.6 |  |
| ycbcr_to_rgb | YCbCr -> RGB, matrix=bt709 | 0.141 | 0.136 | 7337.4 | 0.134 | 0.159 | 365.2 |  |
| rgb_to_grayscale | RGB -> Y, matrix=native | 0.118 | 0.114 | 8749.8 | 0.111 | 0.137 | 290.3 |  |
| gamma_to_linear | gamma=2.6 claim -> linear | 0.155 | 0.155 | 6437.4 | 0.132 | 0.203 | 320.4 |  |
| linear_to_gamma | linear -> gamma=2.6 | 0.147 | 0.137 | 7305.3 | 0.132 | 0.183 | 363.6 |  |
| ycbcr_to_ycbcr | YCbCr bt709 -> native rematrix | 0.141 | 0.137 | 7277.3 | 0.135 | 0.158 | 362.2 |  |
| full_to_legal | full -> legal, bit_depth=10 | 0.123 | 0.119 | 8395.9 | 0.116 | 0.139 | 417.8 |  |
| legal_to_full | legal -> full, bit_depth=10 | 0.126 | 0.119 | 8368.8 | 0.116 | 0.143 | 416.5 |  |
| quantize | float32 -> uint8, bit_depth=8 | 0.100 | 0.097 | 10343.2 | 0.095 | 0.117 | 321.7 |  |
| dequantize | uint8 -> float32, bit_depth=8 | 0.108 | 0.104 | 9631.3 | 0.102 | 0.127 | 299.6 |  |
| cast_dtype | float32 -> float16 | 0.093 | 0.091 | 10996.3 | 0.090 | 0.108 | 410.4 |  |
| recode_dtype | uint8 -> float32 | 0.114 | 0.109 | 9191.2 | 0.107 | 0.139 | 285.9 |  |
| recode_dtype | float32 -> uint8 | 0.133 | 0.124 | 8081.9 | 0.099 | 0.177 | 251.4 |  |
| from_uyvy422 | legal range | 0.121 | 0.106 | 9444.8 | 0.086 | 0.174 | 274.2 |  |
| from_v210 | legal range | 0.105 | 0.109 | 9184.4 | 0.086 | 0.125 | 279.3 |  |
| from_nv12 | legal range, siting=left, interpolation=bilinear | 0.093 | 0.087 | 11514.5 | 0.084 | 0.111 | 322.3 |  |
| from_p010 | legal range, siting=left, interpolation=bilinear | 0.095 | 0.089 | 11278.8 | 0.087 | 0.113 | 350.8 |  |
| from_yuv420p | legal range, interpolation=bilinear | 0.094 | 0.085 | 11738.6 | 0.083 | 0.111 | 328.6 |  |
| from_yuv422p | legal range | 0.112 | 0.113 | 8846.2 | 0.092 | 0.141 | 293.5 |  |
| from_yuv444p | 10-bit legal range | 0.100 | 0.094 | 10694.3 | 0.092 | 0.118 | 399.2 |  |
| from_yuva444p | 12-bit legal range | 0.131 | 0.114 | 8759.7 | 0.110 | 0.182 | 435.9 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=trilinear | 0.299 | 0.297 | 3363.1 | 0.294 | 0.314 | 167.4 |  |
| apply_lut | FHD fp32 RGB, 65^3 LUT, interpolation=tetrahedral | 0.179 | 0.175 | 5720.3 | 0.164 | 0.204 | 284.7 |  |
| px.io.to_uyvy422 | FHD area, legal | 0.108 | 0.101 | 9920.4 | 0.081 | 0.153 | 288.0 |  |
| px.io.to_v210 | FHD area, legal, 128-byte rows | 0.174 | 0.164 | 6102.8 | 0.142 | 0.215 | 185.6 |  |
| px.io.to_nv12 | FHD area, legal, siting=left | 0.150 | 0.135 | 7410.8 | 0.126 | 0.196 | 207.5 |  |
| px.io.to_p010 | FHD area, legal, siting=left | 0.156 | 0.148 | 6761.4 | 0.128 | 0.198 | 210.3 |  |
| px.io.to_yuv420p | 8-bit area, legal, siting=left | 0.155 | 0.148 | 6774.9 | 0.126 | 0.208 | 189.7 |  |
| px.io.to_yuv422p | 10-bit area, legal | 0.128 | 0.133 | 7527.9 | 0.113 | 0.153 | 249.8 |  |
| px.io.to_yuv444p | 10-bit legal | 0.116 | 0.107 | 9363.0 | 0.089 | 0.162 | 349.5 |  |
| px.io.to_yuva444p | 12-bit legal, alpha full | 0.123 | 0.113 | 8876.7 | 0.105 | 0.175 | 441.8 |  |
| read_lut | 65^3 RGB .cube file, parse, float4 packing, and host-to-device transfer included | 111.387 | 111.459 | 9.0 | 110.436 | 112.231 | 0.1 | yes |
| read_image | FHD uint8 RGB PNG file, unchanged, temporary-file I/O included | 24.939 | 24.937 | 40.1 | 24.010 | 25.653 | 0.5 | yes |
| read_image | FHD uint8 RGB JPEG file, unchanged, temporary-file I/O included | 35.099 | 35.088 | 28.5 | 34.587 | 35.822 | 0.4 | yes |
| read_image | FHD uint8 RGB TIFF file, unchanged, temporary-file I/O included | 31.932 | 31.896 | 31.4 | 31.616 | 32.292 | 0.4 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 31.960 | 30.925 | 32.3 | 29.255 | 35.430 | 0.8 | yes |
| read_image | FHD HALF RGB EXR NONE file, unchanged, source-fixed native lane, temporary-file I/O included | 31.146 | 30.507 | 32.8 | 30.246 | 31.079 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 33.069 | 32.826 | 30.5 | 31.996 | 33.847 | 0.8 | yes |
| read_image | FHD HALF RGB EXR ZIPS file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 63.551 | 63.145 | 15.8 | 61.423 | 64.884 | 0.4 | yes |
| read_image | FHD HALF RGB EXR DWAA file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 25.820 | 25.249 | 39.6 | 24.762 | 26.886 | 1.0 | yes |
| read_image | FHD HALF RGB EXR DWAB file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 28.502 | 28.637 | 34.9 | 26.213 | 30.009 | 0.9 | yes |
| read_image | FHD HALF RGB EXR RLE file, unchanged, source-fixed GPU lane, temporary-file I/O included | 177.058 | 171.123 | 5.8 | 169.152 | 194.852 | 0.1 | yes |
| read_image | FHD HALF RGB EXR PXR24 file, unchanged, source-fixed custom CPU lane, temporary-file I/O included | 306.220 | 305.966 | 3.3 | 293.368 | 320.083 | 0.1 | yes |
| read_image | FHD HALF RGB EXR B44 file, unchanged, source-fixed GPU lane, temporary-file I/O included | 74.756 | 70.301 | 14.2 | 68.446 | 93.049 | 0.4 | yes |
| read_image | FHD HALF RGB EXR B44A file, unchanged, source-fixed GPU lane, temporary-file I/O included | 95.768 | 91.025 | 11.0 | 90.160 | 113.199 | 0.3 | yes |
| read_image | FHD HALF RGB EXR PIZ file, unchanged, source-fixed GPU lane, temporary-file I/O included | 33.946 | 33.598 | 29.8 | 33.114 | 34.947 | 0.7 | yes |
| read_image | FHD uint8 RGB JPEG 2000 file, unchanged, temporary-file I/O included | 25.184 | 25.194 | 39.7 | 24.657 | 25.751 | 0.5 | yes |
| read_image | FHD uint8 RGB WebP file, unchanged, temporary-file I/O included | 41.128 | 41.004 | 24.4 | 40.105 | 42.625 | 0.3 | yes |
| read_image | FHD uint8 RGB BMP file, unchanged, temporary-file I/O included | 17.386 | 17.328 | 57.7 | 16.735 | 17.983 | 0.7 | yes |
| read_image | FHD uint8 RGB PNM file, unchanged, temporary-file I/O included | 18.649 | 18.610 | 53.7 | 18.151 | 19.163 | 0.7 | yes |
| read_image | FHD uint8 RGB TGA file, unchanged, temporary-file I/O and CPU RLE included | 9.424 | 9.389 | 106.5 | 9.179 | 9.710 | 1.3 | yes |
| read_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 127.068 | 126.732 | 7.9 | 126.032 | 128.264 | 0.3 | yes |
| read_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU unpack included | 3.229 | 3.219 | 310.6 | 3.035 | 3.459 | 10.3 | yes |
| write_image | FHD uint8 RGB PNG file, compression_level=4, temporary-file I/O included | 196.107 | 196.090 | 5.1 | 194.849 | 197.293 | 0.1 | yes |
| write_image | FHD uint8 RGB JPEG file, quality=95, temporary-file I/O included | 39.903 | 39.546 | 25.3 | 38.099 | 42.379 | 0.3 | yes |
| write_image | FHD uint8 RGB TIFF file, temporary-file I/O included | 62.541 | 60.787 | 16.5 | 59.480 | 73.821 | 0.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 37.380 | 37.042 | 27.0 | 36.499 | 38.747 | 1.0 | yes |
| write_image | FHD fp32 RGB to EXR NONE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 7.731 | 7.546 | 132.5 | 6.177 | 8.968 | 4.9 | yes |
| write_image | FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 31.249 | 31.036 | 32.2 | 29.912 | 32.846 | 1.2 | yes |
| write_image | FHD fp32 RGB to EXR ZIPS/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 88.381 | 88.328 | 11.3 | 68.142 | 106.073 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAA/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 91.481 | 92.127 | 10.9 | 88.410 | 94.934 | 0.4 | yes |
| write_image | FHD fp32 RGB to EXR DWAB/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included | 36.345 | 36.316 | 27.5 | 33.545 | 38.660 | 1.0 | yes |
| write_image | FHD fp32 RGB to EXR RLE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 16.855 | 17.213 | 58.1 | 15.357 | 18.522 | 2.2 | yes |
| write_image | FHD fp32 RGB to EXR PXR24/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 41.569 | 41.802 | 23.9 | 40.237 | 42.896 | 0.9 | yes |
| write_image | FHD fp16 RGB to EXR B44/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 5.904 | 5.175 | 193.2 | 4.778 | 8.092 | 4.8 | yes |
| write_image | FHD fp16 RGB to EXR B44A/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 5.789 | 5.050 | 198.0 | 4.559 | 7.572 | 4.9 | yes |
| write_image | FHD fp16 RGB to EXR PIZ/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included | 71.552 | 71.584 | 14.0 | 70.771 | 72.427 | 0.3 | yes |
| write_image | FHD uint8 RGB JPEG 2000 file, lossless, temporary-file I/O included | 62.768 | 62.973 | 15.9 | 60.374 | 64.539 | 0.2 | yes |
| write_image | FHD uint8 RGB WebP file, lossless, temporary-file I/O included | 310.879 | 311.120 | 3.2 | 302.956 | 319.749 | 0.0 | yes |
| write_image | FHD uint8 RGB BMP file, temporary-file I/O included | 25.623 | 25.539 | 39.2 | 24.487 | 26.716 | 0.5 | yes |
| write_image | FHD uint8 RGB PNM file, temporary-file I/O included | 26.918 | 26.738 | 37.4 | 25.676 | 28.123 | 0.5 | yes |
| write_image | FHD uint8 RGB TGA file, temporary-file I/O and CPU RLE included | 52.046 | 52.010 | 19.2 | 50.325 | 53.795 | 0.2 | yes |
| write_image | FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included | 279.685 | 279.272 | 3.6 | 277.519 | 282.207 | 0.1 | yes |
| write_image | FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU packing included | 7.522 | 7.409 | 135.0 | 6.791 | 8.141 | 4.5 | yes |
| read_header | FHD uint8 RGB PNG header, temporary-file I/O included | 2.100 | 2.095 | 477.3 | 2.064 | 2.145 | 0.0 | yes |
| decode_image | FHD uint8 RGB PNG, unchanged, host bytes exchange included | 23.434 | 23.422 | 42.7 | 22.965 | 23.874 | 0.5 | yes |
| decode_image | FHD uint8 RGB JPEG, unchanged, host bytes exchange included | 35.214 | 35.265 | 28.4 | 34.343 | 35.928 | 0.4 | yes |
| decode_image | FHD uint8 RGB TIFF, unchanged, host bytes exchange included | 31.523 | 31.431 | 31.8 | 30.972 | 32.059 | 0.4 | yes |
| decode_image | FHD uint8 RGB JPEG 2000, unchanged, host bytes exchange included | 23.459 | 23.441 | 42.7 | 23.092 | 23.903 | 0.5 | yes |
| decode_image | FHD uint8 RGB WebP, unchanged, host bytes exchange included | 39.647 | 39.315 | 25.4 | 38.556 | 41.061 | 0.3 | yes |
| decode_image | FHD uint8 RGB BMP, unchanged, host bytes exchange included | 16.470 | 16.438 | 60.8 | 15.976 | 16.998 | 0.8 | yes |
| decode_image | FHD uint8 RGB PNM, unchanged, host bytes exchange included | 17.856 | 17.852 | 56.0 | 17.504 | 18.225 | 0.7 | yes |
| encode_image | FHD uint8 RGB PNG, compression_level=4, host bytes exchange included | 194.341 | 194.331 | 5.1 | 193.207 | 195.509 | 0.1 | yes |
| encode_image | FHD uint8 RGB JPEG, quality=95, host bytes exchange included | 37.361 | 37.393 | 26.7 | 35.987 | 38.604 | 0.3 | yes |
| encode_image | FHD uint8 RGB TIFF, host bytes exchange included | 49.555 | 49.449 | 20.2 | 47.458 | 51.851 | 0.3 | yes |
| encode_image | FHD uint8 RGB JPEG 2000, lossless, host bytes exchange included | 58.731 | 58.667 | 17.0 | 57.717 | 59.687 | 0.2 | yes |
| encode_image | FHD uint8 RGB WebP, lossless, host bytes exchange included | 299.695 | 300.318 | 3.3 | 295.584 | 302.509 | 0.0 | yes |
| encode_image | FHD uint8 RGB BMP, host bytes exchange included | 20.123 | 20.068 | 49.8 | 19.557 | 20.871 | 0.6 | yes |
| encode_image | FHD uint8 RGB PNM, host bytes exchange included | 23.825 | 23.889 | 41.9 | 21.857 | 25.509 | 0.5 | yes |

1 ms exceedances: 90
