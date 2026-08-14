"""Shared three-element (why / what / how) actionable error message helper."""

from __future__ import annotations


def _actionable_error(*, why: str, what: str, how: str) -> str:
    return f"why={why}; what={what}; how={how}"
