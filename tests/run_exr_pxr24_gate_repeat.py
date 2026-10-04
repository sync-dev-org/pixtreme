"""Run one isolated PXR24 EXR gate direction in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from exr_gate_measurement import _gate_source_frames, device_identity
from exr_pxr24_performance import build_pxr24_performance_inputs, inspect_pxr24_gate_fixture, measure_pxr24_gate_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("read", "write"))
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pixtreme-pxr24-repeat-") as directory_name:
        frame, _ = _gate_source_frames()
        inputs = build_pxr24_performance_inputs(Path(directory_name), frame)
        inspection = inspect_pxr24_gate_fixture(inputs.read_path(), "pxr24")
        measurement = measure_pxr24_gate_case(inputs, "pxr24", arguments.direction)
        print(
            json.dumps(
                {
                    "device": asdict(device_identity()),
                    "fixture": asdict(inspection),
                    "measurement": asdict(measurement),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
