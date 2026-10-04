"""GPU usage: importing allocates no GPU resource, and reading moves pixels to GPU without CPU round trips."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from transfer_capture import capture_array_transfers

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.req("REQ-PIX-018")
def test_import_keeps_the_cuda_primary_context_inactive() -> None:
    """Importing pixtreme leaves the CUDA primary context inactive and allocates no GPU resource.

    The subprocess asks the CUDA driver for device 0's primary-context state before and after import. The probe
    initializes the driver but never retains or creates a context, so an import-time device allocation activates the
    state and is observed without relying on CuPy's allocator as the oracle.
    """
    script = r"""
import ctypes

driver = ctypes.CDLL("libcuda.so.1")
assert driver.cuInit(0) == 0
device = ctypes.c_int()
assert driver.cuDeviceGet(ctypes.byref(device), 0) == 0

def primary_context_active():
    flags = ctypes.c_uint()
    active = ctypes.c_int()
    result = driver.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active))
    assert result == 0
    return active.value

assert primary_context_active() == 0
import pixtreme
assert primary_context_active() == 0
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.req("REQ-PIX-018")
def test_tga_read_transfers_pixel_payload_to_gpu_once_without_returning_pixels_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TGA reading transfers its pixel payload to GPU once and returns no pixels to CPU.

    Pixel data transfers are counted; short control-data transfers independent of image size are excluded.
    """
    payload = bytes(range(36))
    header = struct.pack("<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, 4, 3, 24, 0x20)
    path = tmp_path / "transfer.tga"
    path.write_bytes(header + payload)
    transfers = capture_array_transfers(monkeypatch)

    frame = px.io.read_image(path)

    assert frame.shape == (3, 4, 3)
    assert len(transfers.pixel_host_to_device) == 1
    assert (transfers.pixel_host_to_device[0].dtype, transfers.pixel_host_to_device[0].shape) == (
        np.dtype(np.uint8),
        (len(payload),),
    )
    assert len(transfers.pixel_device_to_host) == 0
