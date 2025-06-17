# pixtreme

pixtreme is a high-performance image processing library for Python, leveraging CuPy and CUDA for GPU-accelerated computations, enabling real-time video processing with advanced features.

## Features

- **GPU-Accelerated Processing**: Optimized with CuPy's CUDA RawKernel for extremely fast video analysis
- **Comprehensive Color Space Support**: BGR, RGB, HSV, YCbCr, YUV (4:2:0, 4:2:2), ACES color spaces
- **Advanced Image Transforms**: Affine transformation, resize with 11 interpolation methods
- **Face Processing Suite**: Detection, embedding extraction, and face swapping capabilities
- **Super Resolution**: Multiple backend support (ONNX, PyTorch, TensorRT)
- **3D LUT Processing**: Trilinear and tetrahedral interpolation with both CuPy and NumPy implementations
- **Professional Color Management**: ACES color space transformations for film/VFX workflows
- **Flexible Data Type Support**: Seamless conversion between uint8, uint16, float16, and float32
- **DLPack Integration**: Zero-copy tensor sharing between frameworks

## Requirements

- Python >= 3.10
- CUDA Toolkit 12.x
- System dependencies:
  ```
  basicsr-fixed >= 1.4.2
  cupy-cuda12x >= 13.4.1
  numpy >= 2.2.6
  nvidia-nvimgcodec-cu12-stubs >= 0.5.0.13
  nvidia-nvimgcodec-cu12[all] >= 0.5.0.13
  onnx >= 1.18.0
  onnxconverter-common >= 1.13.0
  onnxruntime-gpu >= 1.22.0
  opencv-python >= 4.11.0.86
  openexr >= 3.3.3
  pydantic >= 2.11.4
  spandrel >= 0.4.1
  spandrel_extra_arches >= 0.2.0
  tensorrt >= 10.11.0.33
  tensorrt_stubs >= 10.11.0.33
  torch >= 2.4
  torchvision >= 0.19
  ```

## Installation

### Standard Installation
```bash
pip install pixtreme
```

### With PyTorch Support
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install pixtreme
```

### Development Installation
```bash
git clone https://github.com/minamikik/pixtreme.git
cd pixtreme

# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python pin 3.12  # supports 3.10 - 3.13
uv sync --extra dev
```

## Complete API Reference

### Color Space Conversions

#### Basic Color Conversions
- `bgr_to_rgb(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray` - Convert BGR to RGB
- `rgb_to_bgr(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray` - Convert RGB to BGR
- `bgr_to_grayscale(image: cp.ndarray) -> cp.ndarray` - Convert BGR to grayscale
- `rgb_to_grayscale(image: cp.ndarray) -> cp.ndarray` - Convert RGB to grayscale using Rec.709 coefficients

#### HSV Color Space
- `bgr_to_hsv(image: cp.ndarray) -> cp.ndarray` - Convert BGR to HSV
- `hsv_to_bgr(image: cp.ndarray) -> cp.ndarray` - Convert HSV to BGR
- `rgb_to_hsv(image: cp.ndarray) -> cp.ndarray` - Convert RGB to HSV
- `hsv_to_rgb(image: cp.ndarray) -> cp.ndarray` - Convert HSV to RGB

#### YCbCr Color Space
- `bgr_to_ycbcr(image: cp.ndarray) -> cp.ndarray` - Convert BGR to YCbCr
- `rgb_to_ycbcr(image: cp.ndarray) -> cp.ndarray` - Convert RGB to YCbCr (Rec.709, legal range)
- `ycbcr_to_bgr(image: cp.ndarray) -> cp.ndarray` - Convert YCbCr to BGR
- `ycbcr_to_rgb(image: cp.ndarray) -> cp.ndarray` - Convert YCbCr to RGB (10-bit precision)
- `ycbcr_to_grayscale(image: cp.ndarray) -> cp.ndarray` - Extract Y channel as grayscale

#### YUV Format Conversions
- `yuv420p_to_ycbcr444(yuv420_data: cp.ndarray, width: int, height: int, interpolation: int = 1) -> cp.ndarray` - Convert YUV 4:2:0 planar to YCbCr 4:4:4
- `yuv422p10le_to_ycbcr444(ycbcr422_data: cp.ndarray, width: int, height: int) -> cp.ndarray` - Convert 10-bit YUV 4:2:2 to YCbCr 4:4:4
- `uyvy422_to_ycbcr444(uyvy_data: cp.ndarray, height: int, width: int) -> cp.ndarray` - Convert UYVY 4:2:2 packed format
- `ndi_uyvy422_to_ycbcr444(uyvy_data: cp.ndarray) -> cp.ndarray` - NDI-specific UYVY conversion

#### ACES Color Space Transformations
- `aces2065_1_to_acescct(image: cp.ndarray) -> cp.ndarray` - ACES2065-1 to ACEScct (log encoding)
- `aces2065_1_to_acescg(image: cp.ndarray) -> cp.ndarray` - ACES2065-1 to ACEScg (linear)
- `aces2065_1_to_rec709(image: cp.ndarray) -> cp.ndarray` - ACES2065-1 to Rec.709
- `acescct_to_aces2065_1(image: cp.ndarray) -> cp.ndarray` - ACEScct to ACES2065-1
- `acescg_to_aces2065_1(image: cp.ndarray) -> cp.ndarray` - ACEScg to ACES2065-1
- `rec709_to_aces2065_1(image: cp.ndarray) -> cp.ndarray` - Rec.709 to ACES2065-1

### 3D LUT Operations

- `read_lut(file_path: str, use_cache: bool = True, cache_dir: str = "cache") -> cp.ndarray` - Read .cube format 3D LUT files
- `apply_lut(frame_rgb: cp.ndarray, lut: cp.ndarray, interpolation: int = 0) -> cp.ndarray` - Apply 3D LUT with interpolation
  - `interpolation=0`: Trilinear interpolation
  - `interpolation=1`: Tetrahedral interpolation

### Image Transformations

#### Resize with Advanced Interpolation
- `resize(src: cp.ndarray | list[cp.ndarray], dsize: tuple[int, int] | None = None, fx: float | None = None, fy: float | None = None, interpolation: int = INTER_AUTO) -> cp.ndarray | list[cp.ndarray]`

Interpolation constants:
- `INTER_NEAREST` (0) - Nearest neighbor
- `INTER_LINEAR` (1) - Bilinear interpolation
- `INTER_CUBIC` (2) - Bicubic interpolation
- `INTER_AREA` (3) - Area-based resampling
- `INTER_LANCZOS4` (4) - Lanczos interpolation over 8x8 neighborhood
- `INTER_AUTO` (5) - Auto-select based on image size
- `INTER_MITCHELL` (6) - Mitchell-Netravali cubic filter
- `INTER_B_SPLINE` (7) - B-spline interpolation
- `INTER_CATMULL_ROM` (8) - Catmull-Rom spline
- `INTER_LANCZOS2` (9) - Lanczos over 4x4 neighborhood
- `INTER_LANCZOS3` (10) - Lanczos over 6x6 neighborhood

#### Geometric Transformations
- `affine_transform(src: cp.ndarray, M: cp.ndarray, dsize: tuple, flags: int = INTER_AUTO) -> cp.ndarray` - Apply affine transformation
- `crop_from_kps(image: cp.ndarray, kps: cp.ndarray, size: int = 512) -> tuple[cp.ndarray, cp.ndarray]` - Crop image based on keypoints
- `get_inverse_matrix(M: cp.ndarray) -> cp.ndarray` - Calculate inverse transformation matrix

#### Morphological Operations
- `erode(image: cp.ndarray, kernel_size: int, kernel: cp.ndarray | None = None, border_value: float = 0.0) -> cp.ndarray` - Morphological erosion

### Face Processing

#### Detection and Analysis
- `FaceDetection` - Face detection using ONNX models
  - Constructor: `FaceDetection(model_path: str | None = None, model_bytes: bytes | None = None, device_id: int = 0, device: str = "cuda")`
  - Supports CUDA, TensorRT, and CPU backends

- `FaceEmbedding` - Extract face embeddings for recognition
  - Constructor: `FaceEmbedding(model_path: str | None = None, model_bytes: bytes | None = None, device: str = "cuda")`

- `FaceSwap` - Swap faces between images
  - Constructor: `FaceSwap(model_path: str | None = None, model_bytes: bytes | None = None, device_id: int = 0, device: str = "cuda")`

- `PasteBack` - Seamlessly paste processed faces back
- `paste_back(target_image: cp.ndarray, paste_image: cp.ndarray, M: cp.ndarray, mask: cp.ndarray = None) -> cp.ndarray` - Function interface for paste back

- `Face` - Data class for face information (Pydantic model)

### Image Filtering

- `GaussianBlur` - Gaussian blur filter class
- `gaussian_blur(image: cp.ndarray, kernel_size: int, sigma: float, kernel: cp.ndarray | None = None) -> cp.ndarray` - Apply Gaussian blur

### Super Resolution / Upscaling

- `OnnxUpscaler` - ONNX Runtime-based upscaling
  - Constructor: `OnnxUpscaler(model_path: str | None = None, model_bytes: bytes | None = None, device_id: int = 0, provider_options: list | None = None)`

- `TorchUpscaler` - PyTorch-based upscaling
  - Constructor: `TorchUpscaler(model_path: str | None = None, model_bytes: bytes | None = None, device: str = "cuda")`

- `TrtUpscaler` - TensorRT-optimized upscaling
  - Constructor: `TrtUpscaler(model_path: str | None = None, model_bytes: bytes | None = None, device_id: int = 0)`

#### Model Conversion Utilities
- `torch_to_onnx(model_path: str, onnx_path: str, input_shape: tuple = (1, 3, 1080, 1920), opset_version: int = 20, precision: str = "fp32", dynamic_axes: dict | None = None, device: str = "cuda")` - Convert PyTorch models to ONNX
- `onnx_to_trt(onnx_path: str, engine_path: str, precision: str = "fp16", workspace: int = 1024 << 20)` - Convert ONNX models to TensorRT

### Data Type Conversions

- `to_uint8(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray` - Convert to 8-bit unsigned integer
- `to_uint16(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray` - Convert to 16-bit unsigned integer
- `to_float16(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray` - Convert to 16-bit float
- `to_float32(image: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray` - Convert to 32-bit float
- `to_dtype(image: np.ndarray | cp.ndarray, dtype: str) -> np.ndarray | cp.ndarray` - Convert to specified dtype

### I/O Operations

- `imread(input_path: str, is_rgb: bool = False, is_nvimgcodec: bool = False) -> cp.ndarray` - Read image file to GPU memory
- `imwrite(output_path: str, image: cp.ndarray | np.ndarray, param: int = -1, is_rgb: bool = False) -> None` - Write GPU image to file
- `imshow(title: str, image: np.ndarray | cp.ndarray | nvimgcodec.Image, scale: float = 1.0, is_rgb: bool = False) -> None` - Display GPU image
- `waitkey(delay: int) -> int` - Wait for keyboard input
- `destroy_all_windows() -> None` - Close all OpenCV windows

### Utility Functions

- `to_blob(image: cp.ndarray, scalefactor: float = 1.0, size: tuple | None = None, mean: tuple = (0, 0, 0), swapRB: bool = False, fp16: bool = False) -> cp.ndarray` - Convert to blob format for neural networks
- `to_cupy(image: np.ndarray | torch.Tensor | nvimgcodec.Image) -> cp.ndarray` - Convert various formats to CuPy array
- `to_numpy(image: cp.ndarray | torch.Tensor | nvimgcodec.Image) -> np.ndarray` - Convert to NumPy array
- `to_tensor(image: np.ndarray | cp.ndarray | nvimgcodec.Image, device: str | torch.device | None = None) -> torch.Tensor` - Convert to PyTorch tensor (requires PyTorch)

### Device Management

- `get_device_id() -> int` - Get current CUDA device ID
- `get_device_count() -> int` - Get number of available CUDA devices
- `Device` - CUDA device context manager

## Usage Examples

### Basic Image Processing
```python
import cupy as cp
import pixtreme as px

# Read and convert image
frame_bgr = px.imread("image.jpg")
frame_bgr = px.to_float32(frame_bgr)
frame_rgb = px.bgr_to_rgb(frame_bgr)

# Color space conversions
frame_ycbcr = px.rgb_to_ycbcr(frame_rgb)
frame_hsv = px.rgb_to_hsv(frame_rgb)

# Advanced resize with Mitchell filter
frame_resized = px.resize(frame_rgb, (1920, 1080), interpolation=px.INTER_MITCHELL)

# Apply Gaussian blur
frame_blurred = px.gaussian_blur(frame_resized, 15, 5.0)

# Convert back and save
frame_bgr_out = px.rgb_to_bgr(frame_blurred)
frame_bgr_out = px.to_uint8(frame_bgr_out)
px.imwrite("output.jpg", frame_bgr_out)
```

### Face Processing
```python
import pixtreme as px

# Initialize face processing
detector = px.FaceDetection(model_path="models/detection.onnx")
embedder = px.FaceEmbedding(model_path="models/embedding.onnx")
swapper = px.FaceSwap(model_path="models/swap.onnx")

# Process faces
image = px.imread("portrait.jpg")
faces = detector.get(image)
embeddings = embedder.get(face)
swapped = swapper.get(target_face, source_face)
```

### Super Resolution
```python
import pixtreme as px

# ONNX upscaling
upscaler = px.OnnxUpscaler(model_path="models/esrgan.onnx")
image_hr = upscaler.upscale(image_lr)

# TensorRT optimized upscaling
trt_upscaler = px.TrtUpscaler(model_path="models/esrgan.trt")
image_hr_fast = trt_upscaler.upscale(image_lr)
```

### Professional Color Grading with ACES
```python
import pixtreme as px

# Load image in Rec.709
image = px.imread("footage.png")
image = px.to_float32(image)

# Convert to ACES color space
image_aces = px.rec709_to_aces2065_1(image)

# Work in ACEScg (linear)
image_acescg = px.aces2065_1_to_acescg(image_aces)

# Apply color grading...

# Convert back to Rec.709
image_graded = px.acescg_to_aces2065_1(image_acescg)
image_final = px.aces2065_1_to_rec709(image_graded)
```

### 3D LUT Application
```python
import pixtreme as px

# Load LUT
lut = px.read_lut("color_grade.cube")

# Apply with tetrahedral interpolation
image_graded = px.apply_lut(image_rgb, lut, interpolation=1)
```

### Multi-GPU Processing
```python
import pixtreme as px

# Process on specific GPU
with px.Device(1):
    image = px.imread("large_image.jpg")
    processed = px.resize(image, (4096, 2160))
    
# Get device info
device_count = px.get_device_count()
current_device = px.get_device_id()
```

## Performance Notes

- All color conversion operations use optimized CUDA kernels
- Supports both legal range (16-235) and full range (0-255) for video processing
- 10-bit precision support for professional video workflows
- Zero-copy tensor sharing via DLPack for framework interoperability
- Batch processing support for multiple images

## License

pixtreme is distributed under the MIT License (see [LICENSE](LICENSE)).

### Included Components

* **ACES LUTs**
  - © 2014 Academy of Motion Picture Arts and Sciences
  - Licensed under "License Terms for ACES Components"
  - See [third_party/aces/LICENSE_ACES.txt](third_party/aces/LICENSE_ACES.txt) for details

## Authors

minamik (@minamikik)

## Acknowledgments

sync.dev

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
