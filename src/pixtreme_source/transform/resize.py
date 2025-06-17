import cupy as cp

from ..utils.dtypes import to_float32
from .interporation.area import area_kernel
from .interporation.bicubic import bicubic_kernel
from .interporation.bilinear import bilinear_kernel
from .interporation.lanczos import lanczos_kernel
from .interporation.mitchell import mitchell_kernel
from .interporation.nearest import nearest_kernel
from .schema import (
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


def resize(
    src: cp.ndarray | list[cp.ndarray],
    dsize: tuple[int, int] | None = None,
    fx: float | None = None,
    fy: float | None = None,
    interpolation: int = INTER_AUTO,
) -> cp.ndarray | list[cp.ndarray]:
    """
    Resize the input image or list of images to the specified size.

    Parameters
    ----------
    src : np.ndarray | cp.ndarray | list[np.ndarray | cp.ndarray]
        The input image or list of images in RGB or any channels format.
    dsize : tuple[int, int] | None (optional)
        The output image size. The format is (width, height). by default None.
    fx : float | None (optional)
        The scaling factor along the horizontal axis. by default None.
    fy : float | None (optional)
        The scaling factor along the vertical axis. by default None.
    interpolation : int (optional)
        The interpolation method to use. by default 1, options are: 0 for nearest neighbor, 1 for bilinear, 2 for bicubic, 3 for area, 4 for Lanczos4.

    Returns
    -------
    np.ndarray | cp.ndarray | list[np.ndarray | cp.ndarray]
        The resized image or list of resized images. The shape is (height, width, channels). dtype is float32.

    """
    if isinstance(src, list):
        return [_resize(image, dsize, fx, fy, interpolation) for image in src]

    return _resize(src, dsize, fx, fy, interpolation)


def _resize(
    src: cp.ndarray,
    dsize: tuple[int, int] | None = None,
    fx: float | None = None,
    fy: float | None = None,
    interpolation: int = INTER_AUTO,
) -> cp.ndarray:
    """
    Resize the input image to the specified size.

    Parameters
    ----------
    image : cp.ndarray
        The input image in RGB or any channels format.
    dsize : tuple[int, int] | None (optional)
        The output image size. The format is (width, height). by default None.
    fx : float | None (optional)
        The scaling factor along the horizontal axis. by default None.
    fy : float | None (optional)
        The scaling factor along the vertical axis. by default None.
    interpolation : int (optional)
        The interpolation method to use. by default 1, options are: 0 for nearest neighbor, 1 for bilinear, 2 for bicubic, 3 for area, 4 for Lanczos4.

    Returns
    -------
    image_resized : cp.ndarray
        The resized image. The shape is (height, width, channels). dtype is float32.

    """

    input_image: cp.ndarray = src

    if input_image.dtype != cp.float32:
        input_image: cp.ndarray = to_float32(input_image)

    input_height: int = input_image.shape[0]
    input_width: int = input_image.shape[1]

    if dsize is not None:
        output_height: int = dsize[1]
        output_width: int = dsize[0]
    elif fx is not None and fy is not None:
        output_height: int = int(input_image.shape[1] * fx)
        output_width: int = int(input_image.shape[0] * fy)
    else:
        raise ValueError("Either dsize or fx and fy must be specified.")

    if len(input_image.shape) == 3:
        channels: int = input_image.shape[2]
    else:
        height_in, width_in = input_image.shape
        channels: int = 1

    if interpolation == INTER_AUTO:
        if output_height < input_height or output_width < input_width:
            _interpolation: int = INTER_AREA
        else:
            _interpolation: int = INTER_LANCZOS4
    else:
        _interpolation: int = interpolation

    # input_image = cp.ascontiguousarray(input_image)
    output_image: cp.ndarray = cp.empty((output_height, output_width, channels), dtype=cp.float32)
    block_size: tuple[int, int] = (16, 16)
    grid_size: tuple[int, int] = (
        (output_width + block_size[0] - 1) // block_size[0],
        (output_height + block_size[1] - 1) // block_size[1],
    )

    if _interpolation == INTER_NEAREST:
        nearest_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, channels),
        )
    elif _interpolation == INTER_LINEAR:
        bilinear_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, channels),
        )
    elif _interpolation == INTER_CUBIC:
        bicubic_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, channels),
        )
    elif _interpolation == INTER_AREA:
        area_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, channels),
        )
    elif _interpolation == INTER_MITCHELL:
        b: cp.float32 = cp.float32(1 / 3)
        c: cp.float32 = cp.float32(1 / 3)
        mitchell_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, channels, b, c),
        )
    elif _interpolation == INTER_CATMULL_ROM:
        b: cp.float32 = cp.float32(0)
        c: cp.float32 = cp.float32(0.5)
        mitchell_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, channels, b, c),
        )
    elif _interpolation == INTER_B_SPLINE:
        b: cp.float32 = cp.float32(1)
        c: cp.float32 = cp.float32(0)
        mitchell_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, channels, b, c),
        )
    elif _interpolation == INTER_LANCZOS2:
        a: int = 2
        lanczos_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, a, channels),
        )
    elif _interpolation == INTER_LANCZOS3:
        a: int = 3
        lanczos_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, a, channels),
        )
    elif _interpolation == INTER_LANCZOS4:
        a: int = 4
        lanczos_kernel(
            grid_size,
            block_size,
            (input_image, output_image, input_width, input_height, output_width, output_height, a, channels),
        )

    output_image: cp.ndarray = cp.nan_to_num(output_image)
    output_image: cp.ndarray = output_image.clip(0, 1)

    return output_image
