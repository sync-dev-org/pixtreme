"""Specification tests for the public mirror snapshot script."""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from repository_contracts import require_repo_file

ROOT = Path(__file__).resolve().parents[1]
_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class _RepositoryFixture:
    dev: Path
    public: Path
    source_sha: str
    public_tip: str | None
    included: dict[str, bytes]


def _run(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=text,
        timeout=_TIMEOUT_SECONDS,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {command!r}\nstdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )
    return completed


def _git(repo: Path, *args: str) -> str:
    completed = _run(["git", "-C", str(repo), *args], cwd=repo)
    assert isinstance(completed.stdout, str)
    return completed.stdout.strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    completed = _run(["git", "-C", str(repo), *args], cwd=repo, text=False)
    assert isinstance(completed.stdout, bytes)
    return completed.stdout


def _write(root: Path, relative: str, content: bytes, *, executable: bool = False) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def _init_repository(path: Path, *, bare: bool = False) -> None:
    path.mkdir(parents=True)
    arguments = ["git", "init", "--initial-branch=main"]
    if bare:
        arguments.append("--bare")
    _run(arguments, cwd=path)
    if not bare:
        _git(path, "config", "user.name", "Mirror Test")
        _git(path, "config", "user.email", "mirror-test@example.invalid")


def _make_repository(tmp_path: Path, *, public_has_main: bool) -> _RepositoryFixture:
    dev = tmp_path / "dev"
    public = tmp_path / "public.git"
    _init_repository(dev)
    _init_repository(public, bare=True)

    included = {
        "README.md": b"# public surface\n",
        "src/pixtreme/data.bin": b"\x00public\xffblob\n",
        "bin/pixtreme-tool": b"#!/bin/sh\nexit 0\n",
        "docs/tokens.md": b"# Public tokens\n",
        "docs-notes.txt": b"not the docs directory\n",
        ".nfo/visible.txt": b"not the .nf directory\n",
        ".githubx/visible.txt": b"not the .github directory\n",
        "AGENTS.md.bak": b"not the excluded root file\n",
        "toolshed.txt": b"not the tools directory\n",
    }
    excluded_contents = {
        ".nf/docs/legacy/requirements.md": b"private canon\n",
        ".nf/docs/legacy/features/v1-example.md": b"private feature\n",
        ".nf/issues/I-1.md": b"private runtime ledger\n",
        ".github/ISSUE_TEMPLATE/task.md": b"private issue template\n",
        ".github/workflows/dev.yml": b"private development workflow\n",
        "AGENTS.md": b"private agent instructions\n",
        "CLAUDE.md": b"private engine instructions\n",
        "tools/internal.py": b"private release tooling\n",
    }
    for relative, content in included.items():
        _write(dev, relative, content, executable=relative == "bin/pixtreme-tool")
    for relative, content in excluded_contents.items():
        _write(dev, relative, content)
    _git(dev, "add", *sorted((*included, *excluded_contents)))
    _git(dev, "commit", "-m", "fixture: source snapshot")
    source_sha = _git(dev, "rev-parse", "HEAD")
    _git(dev, "remote", "add", "public", str(public))

    public_tip: str | None = None
    if public_has_main:
        seed = tmp_path / "seed"
        _init_repository(seed)
        _write(seed, "legacy.txt", b"legacy public tree\n")
        _git(seed, "add", "legacy.txt")
        _git(seed, "commit", "-m", "seed public history")
        _git(seed, "remote", "add", "origin", str(public))
        _git(seed, "push", "origin", "main")
        public_tip = _git(public, "rev-parse", "refs/heads/main")

    return _RepositoryFixture(
        dev=dev,
        public=public,
        source_sha=source_sha,
        public_tip=public_tip,
        included=included,
    )


def _run_script(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    script = require_repo_file("tools/mirror_public.py")
    completed = _run([sys.executable, str(script), *arguments], cwd=repository)
    assert isinstance(completed, subprocess.CompletedProcess)
    assert isinstance(completed.stdout, str)
    return completed


@pytest.fixture
def repository(tmp_path: Path) -> _RepositoryFixture:
    return _make_repository(tmp_path, public_has_main=True)


def test_dry_run_manifest_is_deterministic_and_remote_is_unchanged(repository: _RepositoryFixture) -> None:
    """A default HEAD dry run reports a stable manifest without changing remote refs."""
    assert repository.public_tip is not None
    refs_before = _git(repository.public, "show-ref")

    first = _run_script(repository.dev)
    second = _run_script(repository.dev)

    assert first.stdout == second.stdout
    assert "mode: dry-run" in first.stdout
    assert f"source-commit: {repository.source_sha}" in first.stdout
    assert f"manifest-count: {len(repository.included)}" in first.stdout
    assert "excluded docs/:" not in first.stdout
    assert "excluded .nf/: 3" in first.stdout
    assert "excluded .github/: 2" in first.stdout
    assert "excluded AGENTS.md: 1" in first.stdout
    assert "excluded CLAUDE.md: 1" in first.stdout
    assert "excluded tools/: 1" in first.stdout
    assert f"snapshot-message: mirror: {repository.source_sha[:12]}" in first.stdout
    assert _git(repository.public, "show-ref") == refs_before


def test_push_uses_the_selected_ref_and_preserves_the_public_blobs(repository: _RepositoryFixture) -> None:
    """Push snapshots the selected ref onto public main with only included source blobs."""
    assert repository.public_tip is not None
    _write(repository.dev, "late.txt", b"must not enter the selected snapshot\n")
    _git(repository.dev, "add", "late.txt")
    _git(repository.dev, "commit", "-m", "fixture: later source state")

    completed = _run_script(repository.dev, "--ref", repository.source_sha, "--push")

    public_tip = _git(repository.public, "rev-parse", "refs/heads/main")
    topology = _git(repository.public, "rev-list", "--parents", "-n", "1", public_tip).split()
    public_paths = tuple(_git(repository.public, "ls-tree", "-r", "--name-only", public_tip).splitlines())
    assert completed.stdout.startswith("mode: push\n")
    assert topology == [public_tip, repository.public_tip]
    assert _git(repository.public, "show", "-s", "--format=%B", public_tip) == (f"mirror: {repository.source_sha[:12]}")
    assert public_paths == tuple(sorted(repository.included))
    assert "late.txt" not in public_paths
    for relative, expected in repository.included.items():
        assert _git_bytes(repository.public, "show", f"{public_tip}:{relative}") == expected
        assert _git(repository.public, "ls-tree", public_tip, relative) == _git(
            repository.dev, "ls-tree", repository.source_sha, relative
        )
        assert _git(repository.public, "rev-parse", f"{public_tip}:{relative}") == _git(
            repository.dev, "rev-parse", f"{repository.source_sha}:{relative}"
        )


def test_push_uses_the_remote_pushurl_without_mutating_the_fetch_endpoint(
    repository: _RepositoryFixture, tmp_path: Path
) -> None:
    """Push writes to the configured push URL without changing the fetch endpoint."""
    assert repository.public_tip is not None
    push_endpoint = tmp_path / "push.git"
    _init_repository(push_endpoint, bare=True)
    fetch_refs_before = _git(repository.public, "show-ref")
    _git(repository.dev, "remote", "set-url", "--push", "public", str(push_endpoint))

    _run_script(repository.dev, "--push")

    assert _git(repository.public, "show-ref") == fetch_refs_before
    push_tip = _git(push_endpoint, "rev-parse", "refs/heads/main")
    assert _git(push_endpoint, "rev-list", "--parents", "-n", "1", push_tip).split() == [
        push_tip,
        repository.public_tip,
    ]


def test_push_resolves_a_relative_local_remote_from_the_development_repository(
    repository: _RepositoryFixture,
) -> None:
    """Push resolves a relative local remote from the development repository."""
    assert repository.public_tip is not None
    _git(repository.dev, "remote", "set-url", "public", "../public.git")

    _run_script(repository.dev, "--push")

    public_tip = _git(repository.public, "rev-parse", "refs/heads/main")
    assert public_tip != repository.public_tip
    assert _git(repository.public, "rev-list", "--parents", "-n", "1", public_tip).split() == [
        public_tip,
        repository.public_tip,
    ]


def test_init_push_creates_a_root_snapshot(tmp_path: Path) -> None:
    """An initial push creates public main as a snapshot without a parent commit."""
    repository = _make_repository(tmp_path, public_has_main=False)

    _run_script(repository.dev, "--push", "--init")

    public_tip = _git(repository.public, "rev-parse", "refs/heads/main")
    assert _git(repository.public, "rev-list", "--parents", "-n", "1", public_tip).split() == [public_tip]


def test_tag_pushes_an_annotated_release_tag(repository: _RepositoryFixture) -> None:
    """A tagged push publishes an annotated tag and release commit message."""
    tag = "v9.9.9"
    remote = "mirror"
    _git(repository.dev, "remote", "add", remote, str(repository.public))

    _run_script(repository.dev, "--remote", remote, "--push", "--tag", tag)

    public_tip = _git(repository.public, "rev-parse", "refs/heads/main")
    tag_ref = f"refs/tags/{tag}"
    assert _git(repository.public, "cat-file", "-t", tag_ref) == "tag"
    assert _git(repository.public, "rev-parse", f"{tag_ref}^{{commit}}") == public_tip
    assert _git(repository.public, "show", "-s", "--format=%B", public_tip) == f"release: {tag}"


def test_script_imports_only_python_standard_library_modules() -> None:
    """The mirror script imports only Python standard library modules."""
    script = require_repo_file("tools/mirror_public.py")
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            imported.add(node.module.partition(".")[0])

    allowed = set(sys.stdlib_module_names) | {"__future__"}
    assert imported <= allowed
    assert {"argparse", "subprocess"} <= imported
