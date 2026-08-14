"""Test-only CUDA noise primitive harness; production code must not import this module."""

from functools import lru_cache

import cupy as cp

from pixtreme._generate.noise import _NOISE_KERNEL_SOURCE


@lru_cache(maxsize=1)
def _hash_kernel() -> cp.RawKernel:
    return cp.RawKernel(_NOISE_KERNEL_SOURCE, "pixtreme_noise_hash")
