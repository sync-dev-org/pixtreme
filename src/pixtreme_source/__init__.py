import cupy as cp

from .color.aces_transform import (
    aces2065_1_to_acescct,
    aces2065_1_to_acescg,
    aces2065_1_to_rec709,
    acescct_to_aces2065_1,
    acescg_to_aces2065_1,
    rec709_to_aces2065_1,
)
from .color.apply_lut import apply_lut
from .color.bgr import bgr_to_rgb, rgb_to_bgr
from .color.grayscale import bgr_to_grayscale, rgb_to_grayscale
from .color.hsv import bgr_to_hsv, hsv_to_bgr, hsv_to_rgb, rgb_to_hsv
from .color.read_lut import read_lut
from .color.uyvy422 import uyvy422_to_ycbcr444, uyvy422_to_ycbcr444_cp
from .color.uyvy422_ndi import ndi_uyvy422_to_ycbcr444_cp
from .color.ycbcr import (
    bgr_to_ycbcr,
    rgb_to_ycbcr,
    ycbcr_full_to_legal,
    ycbcr_legal_to_full,
    ycbcr_to_bgr,
    ycbcr_to_grayscale,
    ycbcr_to_rgb,
)
from .color.yuv420 import yuv420p_to_ycbcr444, yuv420p_to_ycbcr444_cp
from .color.yuv422p10le import yuv422p10le_to_ycbcr444, yuv422p10le_to_ycbcr444_cp
from .face.detection import FaceDetection
from .face.embedding import FaceEmbedding
from .face.paste import PasteBack, paste_back
from .face.schema import Face
from .face.swap import FaceSwap
from .filter.gaussian import GaussianBlur, gaussian_blur
from .io.imread import imread
from .io.imshow import destroy_all_windows, imshow, waitkey
from .io.imwrite import imwrite
from .transform.affine import affine_transform, crop_from_kps, get_inverse_matrix
from .transform.erode import erode
from .transform.resize import resize
from .transform.schema import (
    INTER_AREA,
    INTER_AUTO,
    INTER_B_SPLINE,
    INTER_CATMULL_ROM,
    INTER_CUBIC,
    INTER_LANCZOS2,
    INTER_LANCZOS3,
    INTER_LANCZOS4,
    INTER_LINEAR,
    INTER_MITCHELL,
    INTER_NEAREST,
)
from .upscale.convert import onnx_to_trt, torch_to_onnx
from .upscale.core.onnx import OnnxUpscaler
from .upscale.core.torch import TorchUpscaler
from .upscale.core.trt import TrtUpscaler
from .utils.blob import to_blob
from .utils.dlpack import to_cupy, to_numpy, to_tensor
from .utils.dtypes import to_dtype, to_float16, to_float32, to_float64, to_uint8, to_uint16

get_device_id = cp.cuda.device.get_device_id
get_device_count = cp.cuda.runtime.getDeviceCount
Device = cp.cuda.Device

__all__ = [
    "aces2065_1_to_acescct",
    "aces2065_1_to_acescg",
    "aces2065_1_to_rec709",
    "acescct_to_aces2065_1",
    "acescg_to_aces2065_1",
    "rec709_to_aces2065_1",
    "bgr_to_rgb",
    "rgb_to_bgr",
    "bgr_to_grayscale",
    "rgb_to_grayscale",
    "bgr_to_hsv",
    "hsv_to_bgr",
    "hsv_to_rgb",
    "rgb_to_hsv",
    "bgr_to_ycbcr",
    "rgb_to_ycbcr",
    "ycbcr_full_to_legal",
    "ycbcr_legal_to_full",
    "ycbcr_to_bgr",
    "ycbcr_to_rgb",
    "ycbcr_to_grayscale",
    "affine_transform",
    "crop_from_kps",
    "get_inverse_matrix",
    "erode",
    "OnnxUpscaler",
    "TorchUpscaler",
    "TrtUpscaler",
    "resize",
    "INTER_NEAREST",
    "INTER_LINEAR",
    "INTER_CUBIC",
    "INTER_AREA",
    "INTER_AUTO",
    "INTER_LANCZOS4",
    "INTER_MITCHELL",
    "INTER_B_SPLINE",
    "INTER_CATMULL_ROM",
    "INTER_LANCZOS2",
    "INTER_LANCZOS3",
    "Face",
    "FaceDetection",
    "FaceEmbedding",
    "FaceSwap",
    "PasteBack",
    "paste_back",
    "GaussianBlur",
    "gaussian_blur",
    "apply_lut",
    "read_lut",
    "yuv420p_to_ycbcr444_cp",
    "yuv420p_to_ycbcr444",
    "yuv422p10le_to_ycbcr444_cp",
    "yuv422p10le_to_ycbcr444",
    "uyvy422_to_ycbcr444_cp",
    "uyvy422_to_ycbcr444",
    "ndi_uyvy422_to_ycbcr444_cp",
    "to_blob",
    "to_cupy",
    "to_numpy",
    "to_tensor",
    "to_dtype",
    "to_float16",
    "to_float32",
    "to_float64",
    "to_uint8",
    "to_uint16",
    "imread",
    "imshow",
    "imwrite",
    "destroy_all_windows",
    "waitkey",
    "torch_to_onnx",
    "onnx_to_trt",
]
