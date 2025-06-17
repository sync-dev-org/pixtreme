import cupy as cp

from .bgr import bgr_to_rgb, rgb_to_bgr


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

    float y_component = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float cb_component = -0.114572f * r - 0.385428f * g + 0.5f * b + 0.5f;
    float cr_component = 0.5f * r - 0.454153f * g - 0.045847f * b + 0.5f;

    // to legal range
    y_component = (y_component * 219.0f + 16.0f) / 255.0f;
    cb_component = (cb_component * 224.0f + 16.0f) / 255.0f;
    cr_component = (cr_component * 224.0f + 16.0f) / 255.0f;

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
    float y_component = ycbcr[idx];

    // 10bit precision
    float cb_component = ycbcr[idx + 1] - 0.5004887f;
    float cr_component = ycbcr[idx + 2] - 0.5004887f;
    const float under_offset_16 = 0.0625610f;

    // 8bit precision
    //float cb_component = ycbcr[idx + 1] - 0.5019607f;
    //float cr_component = ycbcr[idx + 2] - 0.5019607f;
    //const float under_offset_16 = 0.0627450f;

    // 709
    //float r = y_component + 1.5748037f * cr_component;
    //float g = y_component - 0.1873261f * cb_component - 0.4681249f * cr_component;
    //float b = y_component + 1.8555993f * cb_component;

    // 709 legal
    float r = 1.1643835 * (y_component - under_offset_16) + 1.5960267f * cr_component;
    float g = 1.1643835 * (y_component - under_offset_16) - 0.3917622f * cb_component - 0.8129676 * cr_component;
    float b = 1.1643835 * (y_component - under_offset_16) + 2.0172321f * cb_component;

    // 601
    //float r = y_component + 1.402f * cr_component;
    //float g = y_component - 0.344136f * cb_component - 0.714136 * cr_component;
    //float b = y_component + 1.772f * cb_component;


    rgb[idx] = r;
    rgb[idx + 1] = g;
    rgb[idx + 2] = b;
}
"""
ycbcr_to_rgb_kernel = cp.RawKernel(code=ycbcr_to_rgb_kernel_code, name="ycbcr_to_rgb_kernel")
