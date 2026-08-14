"""Run one isolated Phase 3 EXR gate pair in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from exr_phase3_performance import (
    PHASE3_COMPRESSIONS,
    build_phase3_performance_inputs,
    device_identity,
    inspect_phase3_gate_fixture,
    measure_phase3_gate_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compression", choices=PHASE3_COMPRESSIONS)
    parser.add_argument("direction", choices=("read", "write"))
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pixtreme-phase3-repeat-") as directory_name:
        inputs = build_phase3_performance_inputs(Path(directory_name))
        inspection = inspect_phase3_gate_fixture(inputs.read_path(arguments.compression), arguments.compression)
        measurement = measure_phase3_gate_case(inputs, arguments.compression, arguments.direction)
        payload = {
            "device": asdict(device_identity()),
            "fixture": asdict(inspection),
            "measurement": asdict(measurement),
        }
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
