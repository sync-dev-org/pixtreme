"""Real-GPU end-to-end tests for Frame's DLPack boundaries."""

from __future__ import annotations

import numpy as np
import pytest

import pixtreme as px


def test_frame_accepts_cuda_dlpack_zero_copy_and_rejects_cpu_producers() -> None:
    """v1-boundary-api acceptance 2: only CUDA DLPack producers enter Frame, with pointer-preserving import."""
    import torch

    assert torch.cuda.is_available(), "v1-frame-core acceptance 4 requires the repository's NVIDIA GPU environment"
    cuda_tensor = torch.arange(2 * 3 * 3, dtype=torch.float32, device="cuda").reshape(2, 3, 3)

    result = px.io.from_array(cuda_tensor, colorspace="sRGB", gamma="sRGB", channels="RGB")

    assert result.data.data.ptr == cuda_tensor.data_ptr()
    with pytest.raises(ValueError, match="CUDA"):
        px.io.from_array(torch.zeros((1, 1, 3)), colorspace="sRGB", gamma="sRGB", channels="RGB")
    with pytest.raises(ValueError, match="CUDA"):
        px.io.from_array(np.zeros((1, 1, 3), dtype=np.float32), colorspace="sRGB", gamma="sRGB", channels="RGB")


def test_torch_consumes_frame_dlpack_zero_copy() -> None:
    """v1-boundary-api acceptance 17: torch.from_dlpack(Frame) preserves the CuPy allocation pointer."""
    import cupy as cp
    import torch

    assert torch.cuda.is_available(), "v1-frame-core acceptance 12 requires the repository's NVIDIA GPU environment"
    data = cp.arange(2 * 3 * 3, dtype=cp.float32).reshape(2, 3, 3)
    source = px.io.from_array(data, colorspace="sRGB", gamma="sRGB", channels="RGB")

    consumed = torch.from_dlpack(source)

    assert consumed.data_ptr() == data.data.ptr
    assert tuple(consumed.shape) == source.shape
