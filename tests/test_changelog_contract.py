"""Contracts for repository changelog inspection."""

from __future__ import annotations


def test_latest_changelog_section_accepts_development_and_release_headings() -> None:
    """REQ-TEST-004 and REQ-TEST-008; GitHub #29: latest changelog selection ignores the heading name."""
    from repository_contracts import latest_changelog_section

    development = """# Changelog

## Unreleased

current development claims

## 1.3.0 - 2026-09-03

old release claims
"""
    released = """# Changelog

## 1.3.0 - 2026-09-03

current release claims

## 1.2.1 - 2026-08-01

old release claims
"""

    development_section = latest_changelog_section(development)
    released_section = latest_changelog_section(released)

    assert "Unreleased" in development_section
    assert "current development claims" in development_section
    assert "old release claims" not in development_section
    assert "1.3.0 - 2026-09-03" in released_section
    assert "current release claims" in released_section
    assert "old release claims" not in released_section
