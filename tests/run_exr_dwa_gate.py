"""Run all DWAA/DWAB fp16/fp32 gate cases once in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from exr_dwa_performance import (
    DWA_COMPRESSIONS,
    DWA_DTYPES,
    build_dwa_performance_inputs,
    device_identity,
    inspect_dwa_fixture,
    measure_dwa_gate_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-order", choices=("cpu-first", "gpu-first"), default="cpu-first")
    arguments = parser.parse_args()
    backend_orders = (
        {"read": ("cpu", "custom_cpu", "gpu"), "write": ("cpu", "gpu")}
        if arguments.backend_order == "cpu-first"
        else {"read": ("gpu", "custom_cpu", "cpu"), "write": ("gpu", "cpu")}
    )
    with tempfile.TemporaryDirectory(prefix="pixtreme-dwa-gate-") as directory_name:
        inputs = build_dwa_performance_inputs(Path(directory_name))
        fixtures = [
            asdict(inspect_dwa_fixture(inputs.read_path(compression, dtype), compression, dtype))
            for compression in DWA_COMPRESSIONS
            for dtype in DWA_DTYPES
        ]
        measurements = [
            asdict(
                measure_dwa_gate_case(
                    inputs,
                    compression,
                    dtype,
                    direction,
                    backend_order=backend_orders[direction],
                )
            )
            for compression in DWA_COMPRESSIONS
            for dtype in DWA_DTYPES
            for direction in ("read", "write")
        ]
        payload = {
            "backend_order": arguments.backend_order,
            "backend_orders": backend_orders,
            "device": asdict(device_identity()),
            "fixtures": fixtures,
            "measurements": measurements,
        }
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
