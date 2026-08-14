"""Specification tests for feature-acceptance traceability."""

from __future__ import annotations

from pathlib import Path

import pytest
from acceptance_traceability import (
    TraceabilityFailure,
    check_repository,
    docs_canon_available,
    format_failures,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fixture_repository(
    root: Path,
    *,
    sheets: dict[str, str],
    tests: dict[str, str],
    requirements: str = "**REQ-TEST-001: source** requirement text\n",
) -> Path:
    _write(root, "docs/requirements.md", requirements)
    for slug, acceptance_section in sheets.items():
        _write(
            root,
            f"docs/features/{slug}.md",
            f"# feature: {slug}\n\n## 受入条件\n\n{acceptance_section.rstrip()}\n",
        )
    for name, source in tests.items():
        _write(root, f"tests/{name}", source)
    return root


def _categories(failures: tuple[TraceabilityFailure, ...]) -> set[str]:
    return {failure.category for failure in failures}


def test_checker_reports_an_unreferenced_default_acceptance(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 3 and 16: an unresolved default clause fails."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. requires a test\n2. [trace:manual] external gate"},
        tests={
            "test_alpha.py": (
                "def test_manual_support():\n"
                '    """v1-alpha acceptance 2: an annotated clause may still be referenced."""\n'
                "    pass\n"
            )
        },
    )

    failures = check_repository(root)

    assert any(
        failure.category == "acceptance-unreferenced" and failure.subject == "v1-alpha" and failure.acceptance == 1
        for failure in failures
    )


def test_checker_reports_duplicate_acceptance_numbers(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 2 and 16: duplicate identities fail closed."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. first body\n1. second body"},
        tests={
            "test_alpha.py": (
                'def test_alpha():\n    """v1-alpha acceptance 1: the first identity has a terminal."""\n    pass\n'
            )
        },
    )

    failures = check_repository(root)

    assert any(
        failure.category == "acceptance-number-duplicate" and failure.subject == "v1-alpha" and failure.acceptance == 1
        for failure in failures
    )


@pytest.mark.parametrize(
    ("body", "category"),
    (
        ("[trace:review] unsupported class", "inline-class-unknown"),
        ("[trace:manual] [trace:visual] two classes", "inline-class-duplicate"),
        ("ordinary body [trace:manual]", "inline-class-misplaced"),
        ("[trace:manual malformed class", "inline-class-malformed"),
    ),
)
def test_checker_rejects_invalid_inline_classes(tmp_path: Path, body: str, category: str) -> None:
    """v1-acceptance-traceability acceptance 4 and 16: invalid class syntax fails closed."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": f"1. {body}"},
        tests={
            "test_alpha.py": (
                "def test_alpha():\n"
                '    """v1-alpha acceptance 1: invalid classification cannot hide this reference."""\n'
                "    pass\n"
            )
        },
    )

    assert category in _categories(check_repository(root))


def test_checker_does_not_treat_inline_code_examples_as_trace_classes(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 4 and 16: code literals are not inline annotations."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. the literal `[trace:manual]` documents the grammar"},
        tests={
            "test_alpha.py": (
                "def test_alpha():\n"
                '    """v1-alpha acceptance 1: the acceptance has a normal test terminal."""\n'
                "    pass\n"
            )
        },
    )

    assert check_repository(root) == ()


@pytest.mark.parametrize(
    ("sheets", "category"),
    (
        ({"v1-alpha": "1. [trace:superseded-by:v1-missing] moved"}, "supersede-target-missing"),
        ({"v1-alpha": "1. [trace:superseded-by:v1-alpha] moved"}, "supersede-target-self"),
    ),
)
def test_checker_rejects_invalid_supersede_targets(
    tmp_path: Path,
    sheets: dict[str, str],
    category: str,
) -> None:
    """v1-acceptance-traceability acceptance 5 and 16: supersede targets resolve to another sheet."""
    root = _fixture_repository(tmp_path, sheets=sheets, tests={})

    assert category in _categories(check_repository(root))


def test_checker_reports_every_orphan_feature_and_requirement_reference(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 7-8 and 16: valid siblings do not mask orphan references."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. [trace:manual] external gate"},
        tests={
            "test_orphans.py": (
                'def test_orphans():\n    """v1-missing acceptance 9 and REQ-GHOST-999 are unresolved."""\n    pass\n'
            )
        },
    )

    categories = _categories(check_repository(root))

    assert "feature-sheet-missing" in categories
    assert "requirement-missing" in categories
    assert "derivation-terminal-missing" in categories


def test_checker_reports_an_out_of_range_feature_acceptance(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 7 and 16: references bind to an existing number."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. [trace:manual] external gate"},
        tests={
            "test_alpha.py": (
                'def test_alpha():\n    """v1-alpha acceptance 2: this number does not exist."""\n    pass\n'
            )
        },
    )

    failures = check_repository(root)

    assert any(
        failure.category == "feature-acceptance-missing" and failure.subject == "test_alpha" and failure.acceptance == 2
        for failure in failures
    )


def test_checker_reports_a_test_without_a_derivation_terminal(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 9 and 16: issue prose is not a terminal."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. [trace:manual] external gate"},
        tests={"test_untraced.py": ('def test_untraced():\n    """I-999 is only an issue reference."""\n    pass\n')},
    )

    failures = check_repository(root)

    assert any(
        failure.category == "derivation-terminal-missing" and failure.subject == "test_untraced" for failure in failures
    )


def test_checker_rejects_each_one_sided_characterization_marker(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 9 and 16: characterization needs both name and marker."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. [trace:manual] external gate"},
        tests={
            "test_characterization.py": (
                "def test_name_only_characterization():\n"
                '    """The name alone is insufficient."""\n'
                "    pass\n\n"
                "def test_marker_only():\n"
                '    """characterization: the marker alone is insufficient."""\n'
                "    pass\n"
            )
        },
    )

    failures = check_repository(root)
    incomplete = {failure.subject for failure in failures if failure.category == "characterization-incomplete"}
    missing = {failure.subject for failure in failures if failure.category == "derivation-terminal-missing"}

    assert incomplete == {"test_marker_only", "test_name_only_characterization"}
    assert missing == incomplete


def test_checker_expands_selectors_and_stops_before_natural_language(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 6 and 16: selectors expand and natural prose terminates them."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": ("1. first\n2. second\n3. third\n4. [trace:manual] external gate\n5. fifth")},
        tests={
            "test_alpha.py": (
                'def test_alpha():\n    """v1-alpha acceptance 1-3, and 5 and verifies metadata."""\n    pass\n'
            )
        },
    )

    assert check_repository(root) == ()


@pytest.mark.parametrize("selector", ("1-bogus", "1x", "1--2"))
def test_checker_rejects_text_attached_to_a_selector_atom(tmp_path: Path, selector: str) -> None:
    """v1-acceptance-traceability acceptance 6 and 16: malformed selector suffixes fail closed."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. requires a canonical terminal"},
        tests={
            "test_alpha.py": (
                "def test_alpha():\n"
                f'    """v1-alpha acceptance {selector}: malformed text cannot satisfy the terminal."""\n'
                "    pass\n"
            )
        },
    )

    failures = check_repository(root)
    categories = _categories(failures)

    assert "selector-syntax-invalid" in categories
    assert "derivation-terminal-missing" in categories
    assert "acceptance-unreferenced" in categories


def test_checker_expands_multiple_feature_clauses_independently(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 7 and 16: clauses in one docstring resolve independently."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": "1. alpha", "v1-beta": "2. beta"},
        tests={
            "test_shared.py": (
                "def test_shared():\n"
                '    """v1-alpha acceptance 1; v1-beta acceptance 2: both contracts apply."""\n'
                "    pass\n"
            )
        },
    )

    assert check_repository(root) == ()


def test_checker_rejects_a_descending_selector_range(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 6 and 16: range endpoints cannot descend."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-alpha": ("1. [trace:manual] first\n2. [trace:manual] second\n3. [trace:manual] third")},
        tests={
            "test_alpha.py": (
                'def test_alpha():\n    """v1-alpha acceptance 3-1: descending ranges are invalid."""\n    pass\n'
            )
        },
    )

    categories = _categories(check_repository(root))

    assert "selector-range-descending" in categories
    assert "derivation-terminal-missing" in categories


def test_checker_fails_when_a_sheet_has_no_acceptance_section(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 1 and 10: missing sheet structure fails closed."""
    _write(tmp_path, "docs/requirements.md", "**REQ-TEST-001: source** requirement text\n")
    _write(tmp_path, "docs/features/v1-alpha.md", "# feature: alpha\n")

    assert "acceptance-section-missing" in _categories(check_repository(tmp_path))


def test_failure_rendering_is_stable_and_complete(tmp_path: Path) -> None:
    """v1-acceptance-traceability acceptance 10 and 16: failures expose stable identifying fields."""
    root = _fixture_repository(
        tmp_path,
        sheets={"v1-zeta": "2. unreferenced", "v1-alpha": "1. unreferenced"},
        tests={},
    )

    failures = check_repository(root)
    rendered = format_failures(failures)

    assert [failure.path for failure in failures] == sorted(failure.path for failure in failures)
    assert "category=acceptance-unreferenced" in rendered
    assert "path=docs/features/v1-alpha.md" in rendered
    assert "subject=v1-alpha" in rendered
    assert "acceptance=1" in rendered
    assert "observed=" in rendered


@pytest.mark.skipif(
    not docs_canon_available(ROOT),
    reason="REQ-TEST-008: repository traceability requires the docs canon, which distributions may omit",
)
def test_repository_acceptance_traceability_is_closed() -> None:
    """v1-acceptance-traceability acceptance 10-12: the repository corpus resolves bidirectionally."""
    failures = check_repository(ROOT)

    assert not failures, format_failures(failures)
