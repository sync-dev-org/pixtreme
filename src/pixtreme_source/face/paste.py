from typing import Union

import cupy as cp
import numpy as np

from ..filter.gaussian import GaussianBlur
from ..transform.affine import affine_transform, get_inverse_matrix
from ..utils.dlpack import to_cupy, to_numpy
from ..utils.dtypes import to_float32


class PasteBack:
    def __init__(self):
        self.gaussian_blur = GaussianBlur()
        self.mask = None
        self.size = None

    def create_mask(self, size: tuple):
        padding_v = size[0] // 5
        padding_h = size[1] // 7

        white_plate = cp.ones((size[0], size[1], 3), dtype=cp.float32)
        white_plate[:padding_v, :] = 0
        white_plate[-padding_v:, :] = 0
        white_plate[:, :padding_h] = 0
        white_plate[:, -padding_h:] = 0
        return self.gaussian_blur.get(white_plate, int(padding_h), float(padding_h))

    def get(
        self,
        target_image: Union[np.ndarray, cp.ndarray],
        paste_image: Union[np.ndarray, cp.ndarray],
        M,
    ) -> Union[np.ndarray, cp.ndarray]:
        if isinstance(target_image, np.ndarray):
            is_np = True
            target_image = to_cupy(target_image)
        else:
            is_np = False
        if isinstance(paste_image, np.ndarray):
            paste_image = to_cupy(paste_image)
        if isinstance(M, np.ndarray):
            M = cp.asarray(M)

        target_image = to_float32(target_image)
        paste_image = to_float32(paste_image)

        if self.size != paste_image.shape[:2]:
            self.size = paste_image.shape[:2]
            self.mask = self.create_mask(self.size)

        merged_image = paste_back(target_image, paste_image, M, self.mask)

        if is_np:
            merged_image = to_numpy(merged_image)

        return merged_image


def paste_back(
    target_image: cp.ndarray,
    paste_image: cp.ndarray,
    M: cp.ndarray,
    mask: cp.ndarray = None,
) -> cp.ndarray:
    if mask is None:
        mask = cp.ones((paste_image.shape[0], paste_image.shape[1], 3), dtype=cp.float32)

    IM = get_inverse_matrix(M)
    paste_image = affine_transform(paste_image, IM, (target_image.shape[0], target_image.shape[1]))
    paste_mask = affine_transform(mask, IM, (target_image.shape[0], target_image.shape[1]))
    merged_image = paste_mask * paste_image + (1 - paste_mask) * target_image
    return merged_image
