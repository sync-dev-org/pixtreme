"""Static traceability checker for the repository test corpus."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

_REQUIREMENT_RE = re.compile(r"\bREQ-[A-Z]+-[0-9]{3}\b")
_SHEET_NAME_RE = re.compile(r"v1-[a-z0-9]+(?:-[a-z0-9]+)*")
_ACCEPTANCE_LINE_RE = re.compile(r"^([0-9]+)\. (.*)$")
_TRACE_RE = re.compile(r"\[trace:([^\]\r\n]*)\]")
_FIXED_TRACE_RE = re.compile(r"^\[trace:([^\]\r\n]+)\](?:\s+|$)")
_INLINE_CODE_RE = re.compile(r"`[^`\r\n]*`")
_SIMPLE_TRACE_CLASSES = frozenset({"visual", "manual", "performance"})
_SELECTOR_ATOM = r"[0-9]+(?:-[0-9]+)?"
_FEATURE_CLAUSE_RE = re.compile(
    rf"\b(?P<slug>v1-[a-z0-9]+(?:-[a-z0-9]+)*) acceptance "
    rf"(?P<selector>{_SELECTOR_ATOM}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+){_SELECTOR_ATOM})*)"
)
_SELECTOR_ATOM_RE = re.compile(_SELECTOR_ATOM)
_CHARACTERIZATION_RE = re.compile(r"(?:^|\s)characterization:(?:\s|$)")
_SELECTOR_TERMINATORS = frozenset(":;,.!?)]}")

_TestNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class TraceabilityFailure:
    """One deterministic, fully identified traceability violation."""

    category: str
    path: str
    subject: str
    acceptance: int | None
    observed: str

    def sort_key(self) -> tuple[str, str, int, str, str]:
        """Return the path-and-identity-first ordering required by the feature contract."""
        return (
            self.path,
            self.subject,
            -1 if self.acceptance is None else self.acceptance,
            self.category,
            self.observed,
        )

    def render(self) -> str:
        """Render every required field without relying on object repr details."""
        acceptance = "-" if self.acceptance is None else str(self.acceptance)
        return (
            f"category={self.category}; path={self.path}; subject={self.subject}; "
            f"acceptance={acceptance}; observed={self.observed}"
        )


@dataclass(frozen=True)
class _Acceptance:
    path: str
    slug: str
    number: int
    trace_class: str | None
    supersede_target: str | None


def docs_canon_available(root: Path) -> bool:
    """Return whether a tree contains the repo-only inputs required by the checker."""
    return (root / "docs" / "requirements.md").is_file() and (root / "docs" / "features").is_dir()


def format_failures(failures: Sequence[TraceabilityFailure]) -> str:
    """Format failures in their already deterministic checker order."""
    if not failures:
        return ""
    return "acceptance traceability failed:\n" + "\n".join(failure.render() for failure in failures)


def _failure(
    category: str,
    path: str,
    subject: str,
    acceptance: int | None,
    observed: str,
) -> TraceabilityFailure:
    return TraceabilityFailure(category, path, subject, acceptance, observed)


def _section_bounds(lines: list[str]) -> tuple[int, int] | None:
    starts = [index for index, line in enumerate(lines) if line == "## 受入条件"]
    if not starts:
        return None
    start = starts[0] + 1
    end = next((index for index in range(start, len(lines)) if lines[index].startswith("## ")), len(lines))
    return start, end


def _parse_trace_class(
    *,
    body: str,
    path: str,
    slug: str,
    number: int,
    line_number: int,
) -> tuple[str | None, str | None, list[TraceabilityFailure]]:
    failures: list[TraceabilityFailure] = []
    annotation_text = _INLINE_CODE_RE.sub("", body)
    markers = _TRACE_RE.findall(annotation_text)
    fixed = _FIXED_TRACE_RE.match(body)
    observed = f"line {line_number}: {body}"

    if "[trace:" in annotation_text and not markers:
        failures.append(_failure("inline-class-malformed", path, slug, number, observed))
        return None, None, failures
    if not markers:
        return None, None, failures
    if len(markers) > 1:
        failures.append(_failure("inline-class-duplicate", path, slug, number, observed))
        return None, None, failures
    marker = markers[0]
    if fixed is None or fixed.group(1) != marker:
        failures.append(_failure("inline-class-misplaced", path, slug, number, observed))
        return None, None, failures
    if marker in _SIMPLE_TRACE_CLASSES:
        return marker, None, failures

    prefix = "superseded-by:"
    if marker.startswith(prefix):
        target = marker.removeprefix(prefix)
        if _SHEET_NAME_RE.fullmatch(target) is not None:
            return "superseded-by", target, failures
    failures.append(_failure("inline-class-unknown", path, slug, number, observed))
    return None, None, failures


def _parse_sheet(
    root: Path,
    sheet_path: Path,
) -> tuple[dict[int, _Acceptance], list[TraceabilityFailure]]:
    relative = sheet_path.relative_to(root).as_posix()
    slug = sheet_path.stem
    failures: list[TraceabilityFailure] = []
    try:
        lines = sheet_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        failures.append(_failure("sheet-unreadable", relative, slug, None, type(error).__name__))
        return {}, failures

    section = _section_bounds(lines)
    if section is None:
        failures.append(
            _failure("acceptance-section-missing", relative, slug, None, "missing exact heading: ## 受入条件")
        )
        return {}, failures

    start, end = section
    acceptances: dict[int, _Acceptance] = {}
    current_number: int | None = None
    for index in range(start, end):
        line = lines[index]
        match = _ACCEPTANCE_LINE_RE.match(line)
        if match is None:
            if "[trace:" in _INLINE_CODE_RE.sub("", line):
                failures.append(
                    _failure(
                        "inline-class-misplaced",
                        relative,
                        slug,
                        current_number,
                        f"line {index + 1}: {line}",
                    )
                )
            continue

        number = int(match.group(1))
        body = match.group(2)
        current_number = number
        if number <= 0:
            failures.append(_failure("acceptance-number-invalid", relative, slug, number, f"line {index + 1}: {line}"))
            continue
        if number in acceptances:
            failures.append(
                _failure("acceptance-number-duplicate", relative, slug, number, f"line {index + 1}: {line}")
            )
            continue

        trace_class, target, trace_failures = _parse_trace_class(
            body=body,
            path=relative,
            slug=slug,
            number=number,
            line_number=index + 1,
        )
        failures.extend(trace_failures)
        acceptances[number] = _Acceptance(relative, slug, number, trace_class, target)

    if not acceptances:
        failures.append(
            _failure("acceptance-number-missing", relative, slug, None, "section has no positive indent-free N. entry")
        )
    return acceptances, failures


def _iter_test_nodes(module: ast.Module) -> Iterator[tuple[str, _TestNode]]:
    def visit(statements: list[ast.stmt], class_names: tuple[str, ...]) -> Iterator[tuple[str, _TestNode]]:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if statement.name.startswith("test_"):
                    yield ".".join((*class_names, statement.name)), statement
            elif isinstance(statement, ast.ClassDef):
                yield from visit(statement.body, (*class_names, statement.name))

    yield from visit(module.body, ())


def _first_paragraph(docstring: str) -> str:
    return re.split(r"\n\s*\n", docstring.strip(), maxsplit=1)[0]


def _invalid_selector_suffix(text: str, selector_end: int) -> str | None:
    if selector_end >= len(text):
        return None
    first = text[selector_end]
    if first.isspace() or first in _SELECTOR_TERMINATORS:
        return None

    suffix_end = selector_end + 1
    while suffix_end < len(text):
        character = text[suffix_end]
        if character.isspace() or character in _SELECTOR_TERMINATORS:
            break
        suffix_end += 1
    return text[selector_end:suffix_end]


def _expand_selector(
    selector: str,
    *,
    path: str,
    subject: str,
    slug: str,
) -> tuple[tuple[int, ...], list[TraceabilityFailure]]:
    numbers: list[int] = []
    failures: list[TraceabilityFailure] = []
    for atom_match in _SELECTOR_ATOM_RE.finditer(selector):
        atom = atom_match.group(0)
        if "-" not in atom:
            number = int(atom)
            if number <= 0:
                failures.append(_failure("selector-number-invalid", path, subject, number, f"{slug} {atom}"))
            else:
                numbers.append(number)
            continue

        start_text, end_text = atom.split("-", maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        if start <= 0 or end <= 0:
            failures.append(_failure("selector-number-invalid", path, subject, start, f"{slug} {atom}"))
        elif end < start:
            failures.append(_failure("selector-range-descending", path, subject, start, f"{slug} {atom}"))
        else:
            numbers.extend(range(start, end + 1))
    return tuple(numbers), failures


def _check_test_node(
    *,
    path: str,
    subject: str,
    node: _TestNode,
    requirements: frozenset[str],
    sheets: dict[str, dict[int, _Acceptance]],
    referenced_acceptances: set[tuple[str, int]],
) -> list[TraceabilityFailure]:
    failures: list[TraceabilityFailure] = []
    docstring = ast.get_docstring(node, clean=True)
    text = "" if docstring is None else docstring
    has_terminal = False

    for requirement in sorted(set(_REQUIREMENT_RE.findall(text))):
        if requirement in requirements:
            has_terminal = True
        else:
            failures.append(_failure("requirement-missing", path, subject, None, requirement))

    for clause in _FEATURE_CLAUSE_RE.finditer(text):
        slug = clause.group("slug")
        selector = clause.group("selector")
        invalid_suffix = _invalid_selector_suffix(text, clause.end("selector"))
        if invalid_suffix is not None:
            failures.append(
                _failure(
                    "selector-syntax-invalid",
                    path,
                    subject,
                    None,
                    f"{slug} acceptance {selector}{invalid_suffix}",
                )
            )
            continue
        numbers, selector_failures = _expand_selector(
            selector,
            path=path,
            subject=subject,
            slug=slug,
        )
        failures.extend(selector_failures)
        sheet = sheets.get(slug)
        for number in numbers:
            if sheet is None:
                failures.append(_failure("feature-sheet-missing", path, subject, number, slug))
            elif number not in sheet:
                failures.append(
                    _failure("feature-acceptance-missing", path, subject, number, f"{slug} acceptance {number}")
                )
            else:
                referenced_acceptances.add((slug, number))
                has_terminal = True

    name_marker = node.name.endswith("_characterization")
    doc_marker = bool(docstring and _CHARACTERIZATION_RE.search(_first_paragraph(docstring)))
    if name_marker and doc_marker:
        has_terminal = True
    elif name_marker or doc_marker:
        failures.append(
            _failure(
                "characterization-incomplete",
                path,
                subject,
                None,
                f"name_marker={name_marker}; doc_marker={doc_marker}",
            )
        )

    if not has_terminal:
        observed = "docstring missing" if docstring is None else "no valid REQ, feature acceptance, or characterization"
        failures.append(_failure("derivation-terminal-missing", path, subject, None, observed))
    return failures


def _check_test_file(
    root: Path,
    test_path: Path,
    *,
    requirements: frozenset[str],
    sheets: dict[str, dict[int, _Acceptance]],
    referenced_acceptances: set[tuple[str, int]],
) -> list[TraceabilityFailure]:
    relative = test_path.relative_to(root).as_posix()
    try:
        source = test_path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=relative)
    except (OSError, UnicodeError, SyntaxError) as error:
        return [_failure("test-source-unreadable", relative, "<module>", None, type(error).__name__)]

    failures: list[TraceabilityFailure] = []
    for subject, node in _iter_test_nodes(module):
        failures.extend(
            _check_test_node(
                path=relative,
                subject=subject,
                node=node,
                requirements=requirements,
                sheets=sheets,
                referenced_acceptances=referenced_acceptances,
            )
        )
    return failures


def check_repository(root: Path) -> tuple[TraceabilityFailure, ...]:
    """Check every feature sheet and test source without importing repository modules."""
    root = root.resolve()
    failures: list[TraceabilityFailure] = []
    requirements_path = root / "docs" / "requirements.md"
    try:
        requirements = frozenset(_REQUIREMENT_RE.findall(requirements_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError) as error:
        requirements = frozenset()
        failures.append(
            _failure(
                "requirements-unreadable",
                "docs/requirements.md",
                "requirements",
                None,
                type(error).__name__,
            )
        )

    features_dir = root / "docs" / "features"
    sheet_paths = sorted(features_dir.glob("v1-*.md"), key=lambda path: path.as_posix())
    if not sheet_paths:
        failures.append(
            _failure("feature-sheets-missing", "docs/features", "features", None, "no docs/features/v1-*.md files")
        )

    sheets: dict[str, dict[int, _Acceptance]] = {}
    for sheet_path in sheet_paths:
        acceptances, sheet_failures = _parse_sheet(root, sheet_path)
        sheets[sheet_path.stem] = acceptances
        failures.extend(sheet_failures)

    for slug, acceptances in sheets.items():
        for acceptance in acceptances.values():
            target = acceptance.supersede_target
            if target is None:
                continue
            if target == slug:
                failures.append(
                    _failure(
                        "supersede-target-self",
                        acceptance.path,
                        slug,
                        acceptance.number,
                        target,
                    )
                )
            elif target not in sheets:
                failures.append(
                    _failure(
                        "supersede-target-missing",
                        acceptance.path,
                        slug,
                        acceptance.number,
                        target,
                    )
                )

    referenced_acceptances: set[tuple[str, int]] = set()
    tests_dir = root / "tests"
    test_paths = sorted(tests_dir.rglob("test_*.py"), key=lambda path: path.as_posix()) if tests_dir.is_dir() else []
    for test_path in test_paths:
        failures.extend(
            _check_test_file(
                root,
                test_path,
                requirements=requirements,
                sheets=sheets,
                referenced_acceptances=referenced_acceptances,
            )
        )

    for slug in sorted(sheets):
        for number, acceptance in sorted(sheets[slug].items()):
            if acceptance.trace_class is None and (slug, number) not in referenced_acceptances:
                failures.append(
                    _failure(
                        "acceptance-unreferenced",
                        acceptance.path,
                        slug,
                        number,
                        "default test class has no canonical feature backreference",
                    )
                )

    return tuple(sorted(failures, key=TraceabilityFailure.sort_key))
