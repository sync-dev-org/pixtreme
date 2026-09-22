"""Regenerate AC-11-10/11 fixtures using pinned PyOpenColorIO, without pixtreme.

Usage: uv run python tests/generate_lut_shaper_oracle.py [output_directory]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lut_shaper_oracle import CASES, fixture_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path(__file__).parent / "data" / "lut_shaper")
    args = parser.parse_args()
    for edge, curve in CASES:
        directory = args.output / f"{edge}-{curve}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in fixture_bytes(edge, curve).items():
            (directory / name).write_bytes(content)


if __name__ == "__main__":
    main()
