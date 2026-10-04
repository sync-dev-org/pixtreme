"""Run one isolated RLE EXR gate direction in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from exr_gate_measurement import _gate_source_frames, device_identity
from exr_rle_performance import build_rle_performance_inputs, inspect_rle_gate_fixture, measure_rle_gate_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("read", "write"))
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pixtreme-rle-repeat-") as directory_name:
        frame, _ = _gate_source_frames()
        inputs = build_rle_performance_inputs(Path(directory_name), frame)
        inspection = inspect_rle_gate_fixture(inputs.read_path(), "rle")
        measurement = measure_rle_gate_case(inputs, "rle", arguments.direction)
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
