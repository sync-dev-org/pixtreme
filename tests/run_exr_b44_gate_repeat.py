"""Run one isolated B44 or B44A EXR gate direction in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from exr_b44_performance import build_b44_performance_inputs, inspect_b44_gate_fixture, measure_b44_gate_case
from exr_gate_measurement import _gate_source_frames, device_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compression", choices=("b44", "b44a"))
    parser.add_argument("direction", choices=("read", "write"))
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pixtreme-b44-repeat-") as directory_name:
        _, frame = _gate_source_frames()
        inputs = build_b44_performance_inputs(Path(directory_name), frame)
        inspection = inspect_b44_gate_fixture(inputs.read_path(arguments.compression), arguments.compression)
        measurement = measure_b44_gate_case(inputs, arguments.compression, arguments.direction)
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
