"""Run one isolated PIZ gate direction in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from exr_piz_performance import (
    build_piz_performance_inputs,
    device_identity,
    inspect_piz_gate_fixture,
    measure_piz_gate_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("read", "write"))
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pixtreme-piz-repeat-") as directory_name:
        inputs = build_piz_performance_inputs(Path(directory_name))
        inspection = inspect_piz_gate_fixture(inputs.read_path("fp16"), "fp16")
        measurement = measure_piz_gate_case(inputs, arguments.direction, dtype="fp16")
        payload = {
            "device": asdict(device_identity()),
            "fixture": asdict(inspection),
            "measurement": asdict(measurement),
        }
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
