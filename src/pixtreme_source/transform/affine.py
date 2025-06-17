import cupy as cp
import numpy as np

from .interporation.area import area_affine_kernel
from .interporation.bicubic import bicubic_affine_kernel
from .interporation.bilinear import bilinear_affine_kernel
from .interporation.lanczos import lanczos_affine_kernel
from .interporation.mitchell import mitchell_affine_kernel
from .interporation.nearest import nearest_affine_kernel
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


def affine_transform(src: cp.ndarray, M: cp.ndarray, dsize: tuple, flags: int = INTER_AUTO) -> cp.ndarray:
    """
    Apply an affine transformation to the input image. Using CUDA.

    Parameters
    ----------
    src : cp.ndarray
        The image in BGR format.
    M : cp.ndarray
        The transformation matrix. The input matrix. 2 x 3.
    dst_shape : cp.ndarray
        The shape of the destination image (height, width, channels).

    Returns
    -------
    cp.ndarray
        The transformed image in BGR format.
    """
    # if M is 3 x3, convert it to 2 x 3
    if M.shape[0] == 3:
        M = M[:2, :]

    # Check if M is a cupy array, if not convert it
    if flags == INTER_AUTO:
        scale_x = cp.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        scale_y = cp.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
        # logger.debug("INTER_AUTO")
        # logger.debug(f"scale_x: {scale_x}, scale_y: {scale_y}")
        if scale_x > 1.0 or scale_y > 1.0:
            interpolation = INTER_MITCHELL
            # logger.debug("Interpolation: INTER_MITCHELL")
        else:
            interpolation = INTER_AREA
            # logger.debug("Interpolation: INTER_AREA")
    else:
        interpolation = flags

    dst_h, dst_w = dsize
    output_image = cp.zeros((dst_h, dst_w, 3), dtype=cp.float32)
    block = (16, 16, 1)
    grid = ((dst_w + block[0] - 1) // block[0], (dst_h + block[1] - 1) // block[1])

    inv_M = get_inverse_matrix(M)
    inv_M_flat = cp.asarray(inv_M, dtype=cp.float32).ravel()

    if interpolation == INTER_LINEAR:
        bilinear_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w))
    elif interpolation == INTER_NEAREST:
        nearest_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w))
    elif interpolation == INTER_AREA:
        area_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w))
    elif interpolation == INTER_CUBIC:
        bicubic_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w))
    elif interpolation == INTER_MITCHELL:
        B = cp.float32(1 / 3)
        C = cp.float32(1 / 3)
        mitchell_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w, B, C))
    elif interpolation == INTER_CATMULL_ROM:
        B = cp.float32(0)
        C = cp.float32(0.5)
        mitchell_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w, B, C))
    elif interpolation == INTER_B_SPLINE:
        B = cp.float32(1)
        C = cp.float32(0)
        mitchell_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w, B, C))
    elif interpolation == INTER_LANCZOS2:
        A = 2
        lanczos_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w, A))
    elif interpolation == INTER_LANCZOS3:
        A = 3
        lanczos_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w, A))
    elif interpolation == INTER_LANCZOS4:
        A = 4
        lanczos_affine_kernel(grid, block, (src, output_image, inv_M_flat, src.shape[0], src.shape[1], dst_h, dst_w, A))

    else:
        raise ValueError(f"Unsupported interpolation: {interpolation}")

    return output_image


def get_inverse_matrix(M: cp.ndarray) -> cp.ndarray:
    """
    Get the inverse of the affine matrix.

    Parameters
    ----------
    M : Union[np.ndarray, cp.ndarray]
        The input matrix. 2 x 3.

    Returns
    -------
    Union[np.ndarray, cp.ndarray]
        The inverse matrix. 2 x 3.
    """
    if M.shape[0] == 3:
        M = M[:2, :]

    if isinstance(M, np.ndarray):
        M_3x3 = np.concatenate([M, np.array([[0, 0, 1]], dtype=M.dtype)], axis=0)
        inverse_M = np.linalg.inv(M_3x3)
    elif isinstance(M, cp.ndarray):
        M_3x3 = cp.concatenate([M, cp.array([[0, 0, 1]], dtype=M.dtype)], axis=0)
        inverse_M = cp.linalg.inv(M_3x3)
    else:
        raise ValueError(f"Unsupported type: {type(M)}")
    return inverse_M[:2, :]


def crop_from_kps(
    image: cp.ndarray,
    kps: cp.ndarray,
    size: int = 512,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Get the affine matrix from keypoints.

    Parameters
    ----------
    image : cp.ndarray
        The frame, RGB format. height, width, channels.
    size : int
        The size of the output image.
    kps : cp.ndarray
        The source keypoints. left eye(top-left), right eye(top-right), nose(center), mouth left(bottom-left), mouth right(bottom-right)

    Returns
    -------
    cp.ndarray
        The affine matrix. 2 x 3.
    """
    dst = cp.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=cp.float32,
    )

    # Offset
    if size % 112 == 0:
        ratio = float(size) / 112.0
        diff_x = 0
    else:
        ratio = float(size) / 128.0
        diff_x = int(8.0 * ratio)
    dst = dst * ratio
    dst[:, 0] += diff_x

    # Centralize points
    src_mean = cp.mean(kps, axis=0)
    dst_mean = cp.mean(dst, axis=0)
    src_centered = kps - src_mean
    dst_centered = dst - dst_mean

    # Scale adjustment
    src_dists = cp.linalg.norm(src_centered, axis=1)
    dst_dists = cp.linalg.norm(dst_centered, axis=1)
    scale = cp.sum(dst_dists) / cp.sum(src_dists)

    # Rotation
    U, _, VT = cp.linalg.svd(cp.dot(dst_centered.T, src_centered))
    R = cp.dot(U, VT)

    if cp.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = cp.dot(U, VT)

    # Translation
    T = dst_mean - scale * cp.dot(R, src_mean)

    # Construct transformation matrix
    M = cp.zeros((2, 3))
    M[0:2, 0:2] = scale * R
    M[:, 2] = T

    # Crop
    # inverse_M = get_inverse_matrix(M)
    cropped_image = affine_transform(image, M, (size, size))

    return cropped_image, M
