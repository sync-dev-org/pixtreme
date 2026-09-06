"""Structural contracts for repo-only test inputs."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _path_segments(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return (*_path_segments(node.left), *_path_segments(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
        return (
            *_path_segments(node.func.value),
            *(segment for argument in node.args for segment in _path_segments(argument)),
        )
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return tuple(segment for segment in node.value.replace("\\", "/").split("/") if segment)
    return ()


def _names(node: ast.AST) -> set[str]:
    return {candidate.id for candidate in ast.walk(node) if isinstance(candidate, ast.Name)}


def _is_path_construction(node: ast.AST) -> bool:
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)) or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath"
    )


def _is_nested_path_construction(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
        return True
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "joinpath":
        call = parents.get(parent)
        return isinstance(call, ast.Call) and call.func is parent
    return False


def _path_root(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _path_root(node.left)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
        return _path_root(node.func.value)
    return node


def _is_temporary_fixture_rooted(node: ast.AST) -> bool:
    return bool({"tmp_path", "tmp_path_factory"}.intersection(_names(_path_root(node))))


def _is_guarded(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    candidate = node
    while candidate in parents:
        candidate = parents[candidate]
        if isinstance(candidate, ast.Call) and isinstance(candidate.func, ast.Name):
            return candidate.func.id == "require_repo_file"
    return False


def _is_existence_only_path(node: ast.AST, tree: ast.Module, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if not isinstance(parent, ast.Assign) or len(parent.targets) != 1 or not isinstance(parent.targets[0], ast.Name):
        return False
    name = parent.targets[0].id
    uses = (
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Load) and candidate.id == name
    )
    return all(
        isinstance(parents.get(candidate), ast.Attribute)
        and parents[candidate].value is candidate  # type: ignore[union-attr]
        and parents[candidate].attr in {"exists", "is_dir", "is_file"}  # type: ignore[union-attr]
        for candidate in uses
    )


def _repo_only_path_offenders(tree: ast.Module) -> list[int]:
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not _is_path_construction(node):
            continue
        if _is_nested_path_construction(node, parents):
            continue
        segments = _path_segments(node)
        if not {"docs", "tools"}.intersection(segments):
            continue
        if (
            _is_temporary_fixture_rooted(node)
            or _is_guarded(node, parents)
            or _is_existence_only_path(node, tree, parents)
        ):
            continue
        offenders.append(node.lineno)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"Path", "open"} or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if first.value == "docs" or first.value == "tools" or first.value.startswith(("docs/", "tools/")):
            if not _is_guarded(node, parents):
                offenders.append(node.lineno)

    return sorted(set(offenders))


@pytest.mark.parametrize(
    ("source", "expected_lines"),
    [
        ('ROOT.joinpath("docs", "future.md").read_text()', (1,)),
        ('(ROOT / "docs" / tmp_path.name).read_text()', (1,)),
        ('(ROOT / "tools" / "future.py").read_text()', (1,)),
        ('Path("docs/future.md").read_text()', (1,)),
        ('open("tools/future.py")', (1,)),
        ('tmp_path.joinpath("docs", "artifact.md").write_text("data")', ()),
        ('(tmp_path / "tools" / "artifact.py").write_text("data")', ()),
        ('require_repo_file(ROOT.joinpath("docs", "future.md")).read_text()', ()),
        ('candidate = ROOT.joinpath("docs", "future.md")\nassert candidate.exists()', ()),
    ],
    ids=[
        "joinpath-docs",
        "repo-root-with-tmp-path-tail",
        "slash-tools",
        "path-literal",
        "open-literal",
        "tmp-path-joinpath-root",
        "tmp-path-slash-root",
        "guarded-joinpath",
        "existence-only-joinpath",
    ],
)
def test_repo_only_path_detector_contract(source: str, expected_lines: tuple[int, ...]) -> None:
    """REQ-TEST-003 and REQ-TEST-008; GitHub #29: structural detector positive and negative fixtures."""
    tree = ast.parse(source, filename="test_virtual_repo_path.py")

    assert tuple(_repo_only_path_offenders(tree)) == expected_lines


def test_repo_only_paths_use_the_central_guard_contract() -> None:
    """REQ-TEST-003 and REQ-TEST-008; GitHub #29: repo-only paths use the common guard."""
    offenders: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if path == Path(__file__):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(f"{path.relative_to(ROOT)}:{line_number}" for line_number in _repo_only_path_offenders(tree))

    assert not offenders, f"repo-only paths bypass require_repo_file: {', '.join(offenders)}"
