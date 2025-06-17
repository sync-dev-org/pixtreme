import cupy as cp

from .bgr import bgr_to_rgb, rgb_to_bgr

# Rec.709 full-range to legal-range conversion constants
SCALE_Y_F2L = 876.0 / 1023.0  # 0.856941
SCALE_C_F2L = 896.0 / 1023.0  # 0.875122
OFFS_Y_L = 64.0 / 1023.0  # 0.062561
OFFS_C_CENTER = 0.5  # 512 code centre
SCALE_Y_L2F = 1023.0 / 876.0  # 1.167303
SCALE_C_L2F = 1023.0 / 896.0  # 1.142188


def ycbcr_to_grayscale(image: cp.ndarray) -> cp.ndarray:
    """
    YCbCr to Grayscale conversion

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in YCbCr 4:4:4 format.

    Returns
    -------
    image_gray : cp.ndarray
        Grayscale frame. Shape 3D array (height, width, 3) in RGB 4:4:4 format.
    """
    image_ycbcr = image
    image_y = image_ycbcr[:, :, 0]
    image_gray = cp.dstack((image_y, image_y, image_y))
    return image_gray


def bgr_to_ycbcr(image: cp.ndarray) -> cp.ndarray:
    image_bgr = image
    image_rgb = bgr_to_rgb(image_bgr)
    return rgb_to_ycbcr(image_rgb)


def rgb_to_ycbcr(image: cp.ndarray) -> cp.ndarray:
    """
    Convert RGB to YCbCr

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in RGB format.

    Returns
    -------
    image_ycbcr : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in YCbCr format.
    """
    image_rgb = image

    image_rgb = cp.clip(image_rgb, 0, 1.0)
    height, width, _ = image_rgb.shape
    image_ycbcr = cp.empty_like(image_rgb)
    grid_size = ((width + 31) // 32, (height + 31) // 32)
    block_size = (32, 32)
    rgb_to_ycbcr_kernel(grid_size, block_size, (image_rgb, image_ycbcr, width, height))
    return image_ycbcr


rgb_to_ycbcr_kernel_code = """
extern "C" __global__
void rgb_to_ycbcr_kernel(const float* rgb, float* ycbcr, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float r = rgb[idx];
    float g = rgb[idx + 1];
    float b = rgb[idx + 2];

    // --- Rec.709 full-range -------------------------------- ★
    float y_f  = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    float cb_f = (b - y_f)/1.8556f + 0.5f;   // = (B'-Y')/(2*(1-K_b))+0.5
    float cr_f = (r - y_f)/1.5748f + 0.5f;   // = (R'-Y')/(2*(1-K_r))+0.5

    float y_component  = y_f;
    float cb_component = cb_f;
    float cr_component = cr_f;

    ycbcr[idx] = y_component;
    ycbcr[idx + 1] = cb_component;
    ycbcr[idx + 2] = cr_component;
}
"""
rgb_to_ycbcr_kernel = cp.RawKernel(code=rgb_to_ycbcr_kernel_code, name="rgb_to_ycbcr_kernel")


def ycbcr_to_bgr(image: cp.ndarray) -> cp.ndarray:
    image_ybcr = image
    image_rgb = ycbcr_to_rgb(image_ybcr)
    return rgb_to_bgr(image_rgb)


def ycbcr_to_rgb(image: cp.ndarray) -> cp.ndarray:
    """
    Convert YCbCr to RGB

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in YCbCr format.

    Returns
    -------
    frame_rgb : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in RGB format.
    """
    image_ycbcr = image

    height, width, _ = image_ycbcr.shape
    image_rgb = cp.empty_like(image_ycbcr)
    grid_size = ((width + 31) // 32, (height + 31) // 32)
    block_size = (32, 32)
    ycbcr_to_rgb_kernel(grid_size, block_size, (image_ycbcr, image_rgb, width, height))
    return image_rgb


ycbcr_to_rgb_kernel_code = """
extern "C" __global__
void ycbcr_to_rgb_kernel(const float* ycbcr, float* rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    float y_component  = ycbcr[idx];
    float cb_component = ycbcr[idx + 1] - 0.5f;
    float cr_component = ycbcr[idx + 2] - 0.5f;

    float r = y_component + 1.5748f * cr_component;
    float g = y_component - 0.1873f * cb_component - 0.4681f * cr_component;
    float b = y_component + 1.8556f * cb_component;

    rgb[idx] = r;
    rgb[idx + 1] = g;
    rgb[idx + 2] = b;
}
"""
ycbcr_to_rgb_kernel = cp.RawKernel(code=ycbcr_to_rgb_kernel_code, name="ycbcr_to_rgb_kernel")


def ycbcr_full_to_legal(image: cp.ndarray) -> cp.ndarray:
    """
    Convert YCbCr full-range to legal-range

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in YCbCr full-range format.

    Returns
    -------
    image_ycbcr_legal : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in YCbCr legal-range format.
    """
    image_ycbcr_full = image

    height, width, _ = image_ycbcr_full.shape
    image_ycbcr_legal = cp.empty_like(image_ycbcr_full)
    grid_size = ((width + 31) // 32, (height + 31) // 32)
    block_size = (32, 32)
    ycbcr_full_to_legal_kernel(grid_size, block_size, (image_ycbcr_full, image_ycbcr_legal, width, height))
    return image_ycbcr_legal


ycbcr_full_to_legal_kernel_code = """
extern "C" __global__
void ycbcr_full_to_legal(const float* in, float* out,
                         int w, int h) {{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;

    int idx = (y*w + x)*3;
    float Y  = in[idx    ];
    float Cb = in[idx + 1];
    float Cr = in[idx + 2];

    Y  = Y  * {SCALE_Y_F2L:f} + {OFFS_Y_L:f};
    Cb = (Cb - {OFFS_C_CENTER:f})*{SCALE_C_F2L:f} + {OFFS_C_CENTER:f};
    Cr = (Cr - {OFFS_C_CENTER:f})*{SCALE_C_F2L:f} + {OFFS_C_CENTER:f};

    out[idx    ] = Y;
    out[idx + 1] = Cb;
    out[idx + 2] = Cr;
}}
"""
ycbcr_full_to_legal_kernel = cp.RawKernel(code=ycbcr_full_to_legal_kernel_code, name="ycbcr_full_to_legal")


def ycbcr_legal_to_full(image: cp.ndarray) -> cp.ndarray:
    """
    Convert YCbCr legal-range to full-range

    Parameters
    ----------
    image : cp.ndarray
        Input frame. Shape 3D array (height, width, 3) in YCbCr legal-range format.

    Returns
    -------
    image_ycbcr_full : cp.ndarray
        Output frame. Shape 3D array (height, width, 3) in YCbCr full-range format.
    """
    image_ycbcr_legal = image

    height, width, _ = image_ycbcr_legal.shape
    image_ycbcr_full = cp.empty_like(image_ycbcr_legal)
    grid_size = ((width + 31) // 32, (height + 31) // 32)
    block_size = (32, 32)
    ycbcr_legal_to_full_kernel(grid_size, block_size, (image_ycbcr_legal, image_ycbcr_full, width, height))
    return image_ycbcr_full


ycbcr_legal_to_full_kernel_code = """
extern "C" __global__
void ycbcr_legal_to_full(const float* in, float* out,
                         int w, int h) {{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;

    int idx = (y*w + x)*3;
    float Y  = in[idx    ];
    float Cb = in[idx + 1];
    float Cr = in[idx + 2];

    Y  = (Y  - {OFFS_Y_L:f}) * {SCALE_Y_L2F:f};
    Cb = (Cb - {OFFS_C_CENTER:f}) * {SCALE_C_L2F:f} + {OFFS_C_CENTER:f};
    Cr = (Cr - {OFFS_C_CENTER:f}) * {SCALE_C_L2F:f} + {OFFS_C_CENTER:f};

    out[idx    ] = Y;
    out[idx + 1] = Cb;
    out[idx + 2] = Cr;
}}
"""
ycbcr_legal_to_full_kernel = cp.RawKernel(code=ycbcr_legal_to_full_kernel_code, name="ycbcr_legal_to_full")
ycbcr_legal_to_full_kernel = cp.RawKernel(code=ycbcr_legal_to_full_kernel_code, name="ycbcr_legal_to_full")
