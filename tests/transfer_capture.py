"""Observe CuPy array transfers made while a public operation runs."""

from dataclasses import dataclass, field

import cupy as cp
import numpy as np
import pytest


@dataclass(repr=False)
class ArrayTransfers:
    host_to_device: list[np.ndarray] = field(default_factory=list)
    device_to_host: list[cp.ndarray] = field(default_factory=list)

    @staticmethod
    def _is_pixel_data(array: np.ndarray | cp.ndarray) -> bool:
        # The tested images contain at least nine elements; their control vectors contain at most eight.
        return array.size > 8

    @property
    def pixel_host_to_device(self) -> list[np.ndarray]:
        return [array for array in self.host_to_device if self._is_pixel_data(array)]

    @property
    def pixel_device_to_host(self) -> list[cp.ndarray]:
        return [array for array in self.device_to_host if self._is_pixel_data(array)]


def capture_array_transfers(monkeypatch: pytest.MonkeyPatch) -> ArrayTransfers:
    """Record cp.asarray, cp.asnumpy, cupy.ndarray.get, and cupy.ndarray.set transfers."""
    transfers = ArrayTransfers()
    original_asarray = cp.asarray
    original_asnumpy = cp.asnumpy
    original_get = cp.ndarray.get
    original_set = cp.ndarray.set
    asarray_depth = 0
    asnumpy_depth = 0

    def capture_asarray(value: object, *args: object, **kwargs: object) -> cp.ndarray:
        nonlocal asarray_depth
        if isinstance(value, np.ndarray):
            transfers.host_to_device.append(value)
        asarray_depth += 1
        try:
            return original_asarray(value, *args, **kwargs)
        finally:
            asarray_depth -= 1

    def capture_asnumpy(value: cp.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        nonlocal asnumpy_depth
        if isinstance(value, cp.ndarray):
            transfers.device_to_host.append(value)
        asnumpy_depth += 1
        try:
            return original_asnumpy(value, *args, **kwargs)
        finally:
            asnumpy_depth -= 1

    def capture_get(value: cp.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        if asnumpy_depth == 0:
            transfers.device_to_host.append(value)
        return original_get(value, *args, **kwargs)

    def capture_set(value: cp.ndarray, host: np.ndarray, *args: object, **kwargs: object) -> None:
        if asarray_depth == 0:
            transfers.host_to_device.append(host)
        original_set(value, host, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", capture_asarray)
    monkeypatch.setattr(cp, "asnumpy", capture_asnumpy)
    monkeypatch.setattr(cp.ndarray, "get", capture_get)
    monkeypatch.setattr(cp.ndarray, "set", capture_set)
    return transfers
