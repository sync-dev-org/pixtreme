"""Specification tests for the shared actionable error message helper."""

from __future__ import annotations

import pytest

from pixtreme._core.errors import _actionable_error


@pytest.mark.req("REQ-PIX-017")
def test_actionable_error_joins_why_what_how_in_fixed_order() -> None:
    """Public errors carry fixed-order why=...; what=...; how=... slots in one message."""
    message = _actionable_error(why="left", what="middle", how="right")
    assert message == "why=left; what=middle; how=right"
