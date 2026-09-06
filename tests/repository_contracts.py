"""Shared helpers for tests that inspect repository-only files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def require_repo_file(relative_path: str) -> Path:
    """Return one repository-only file or skip when a distribution omits it."""
    path = ROOT / relative_path
    if not path.is_file():
        kind = "tooling" if relative_path.startswith("tools/") else "documentation"
        pytest.skip(f"repo-only {kind} contract: {relative_path} is absent from this distribution")
    return path


def latest_changelog_section(markdown: str) -> str:
    """Return the first level-two section, regardless of its heading name."""
    headings = tuple(re.finditer(r"^## .+$", markdown, flags=re.MULTILINE))
    if not headings:
        raise ValueError("changelog has no level-two section")
    end = headings[1].start() if len(headings) > 1 else len(markdown)
    return markdown[headings[0].start() : end].rstrip()
