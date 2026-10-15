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


def _run_script(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    script = require_repo_file("tools/mirror_public.py")
    completed = _run([sys.executable, str(script), *arguments], cwd=repository, check=check)
    assert isinstance(completed, subprocess.CompletedProcess)
    assert isinstance(completed.stdout, str)
    return completed


@pytest.fixture
def repository(tmp_path: Path) -> _RepositoryFixture:
    return _make_repository(tmp_path, public_has_main=True)


def _add_development_origin(repository: _RepositoryFixture) -> Path:
    origin = repository.dev.parent / "origin.git"
    _init_repository(origin, bare=True)
    _git(repository.dev, "remote", "add", "origin", str(origin))
    _git(repository.dev, "push", "origin", "main")
    return origin


@pytest.mark.parametrize("push", [False, True], ids=["dry-run", "push"])
def test_development_origin_is_rejected_without_changing_refs(repository: _RepositoryFixture, push: bool) -> None:
    """Issue #80: using origin for a mirror fails and preserves development, origin and public refs."""
    origin = _add_development_origin(repository)
    repositories = (repository.dev, origin, repository.public)
    refs_before = tuple(_git(repo, "show-ref") for repo in repositories)
    arguments = ["--push"] if push else []

    completed = _run_script(repository.dev, "--remote", "origin", *arguments, check=False)

    assert tuple(_git(repo, "show-ref") for repo in repositories) == refs_before
    assert completed.returncode == 2
    assert "development remote 'origin'" in completed.stderr
    assert "public" in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize("push", [False, True], ids=["dry-run", "push"])
@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("origin_url", ["fetch", "push"])
@pytest.mark.parametrize("mirror_url", ["fetch", "push"])
def test_remote_alias_sharing_an_origin_url_is_rejected_without_changing_refs(
    tmp_path: Path, mirror_url: str, origin_url: str, relative: bool, push: bool
) -> None:
    """Issue #80: matching any resolved fetch/push URL rejects an alias and preserves all refs and tags."""
    repository = _make_repository(tmp_path, public_has_main=False)
    origin = _add_development_origin(repository)
    origin_push = tmp_path / "origin-push.git"
    _init_repository(origin_push, bare=True)
    _git(repository.dev, "push", str(origin_push), "main")
    _git(repository.dev, "remote", "set-url", "--push", "origin", str(origin_push))
    _git(repository.dev, "push", "public", "main")

    remote = "mirror"
    _git(repository.dev, "remote", "add", remote, str(repository.public))
    shared_endpoint = origin if origin_url == "fetch" else origin_push
    shared_url = f"../{shared_endpoint.name}" if relative else str(shared_endpoint)
    url_arguments = ["--push"] if mirror_url == "push" else []
    _git(repository.dev, "remote", "set-url", *url_arguments, remote, shared_url)
    # Keep the other URL distinct so each case isolates one URL pairing.
    if mirror_url == "fetch":
        _git(repository.dev, "remote", "set-url", "--push", remote, str(repository.public))

    tag = "v9.9.9"
    _git(repository.dev, "tag", tag, repository.source_sha)
    repositories = (repository.dev, origin, origin_push, repository.public)
    refs_before = tuple(_git(repo, "show-ref") for repo in repositories)
    arguments = ["--push"] if push else []

    completed = _run_script(repository.dev, "--remote", remote, "--tag", tag, *arguments, check=False)

    assert tuple(_git(repo, "show-ref") for repo in repositories) == refs_before
    assert completed.returncode == 2
    assert remote in completed.stderr
    assert "development remote 'origin'" in completed.stderr
    assert "public" in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize("push", [False, True], ids=["dry-run", "push"])
def test_origin_url_is_rejected_before_accessing_an_unavailable_remote(
    repository: _RepositoryFixture, push: bool
) -> None:
    """Issue #80: a shared origin URL reports the destination error before attempting to read remote main."""
    unavailable = repository.dev.parent / "unavailable.git"
    _git(repository.dev, "remote", "add", "origin", str(unavailable))
    _git(repository.dev, "remote", "add", "mirror", "../unavailable.git")
    repositories = (repository.dev, repository.public)
    refs_before = tuple(_git(repo, "show-ref") for repo in repositories)
    arguments = ["--push"] if push else []

    completed = _run_script(repository.dev, "--remote", "mirror", "--init", *arguments, check=False)

    assert completed.returncode == 2
    assert "development remote 'origin'" in completed.stderr
    assert tuple(_git(repo, "show-ref") for repo in repositories) == refs_before
    assert not unavailable.exists()


@pytest.mark.parametrize("push", [False, True], ids=["dry-run", "push"])
def test_public_mirror_succeeds_when_a_distinct_development_origin_exists(
    repository: _RepositoryFixture, push: bool
) -> None:
    """Issue #80: a distinct public remote still supports dry runs and tagged pushes when origin exists."""
    origin = _add_development_origin(repository)
    tag = "v9.9.9"
    _git(repository.dev, "tag", tag, repository.source_sha)
    development_repositories = (repository.dev, origin)
    development_refs_before = tuple(_git(repo, "show-ref") for repo in development_repositories)
    public_refs_before = _git(repository.public, "show-ref")
    arguments = ["--push"] if push else []

    completed = _run_script(repository.dev, "--tag", tag, *arguments)

    assert completed.returncode == 0
    assert f"source-commit: {repository.source_sha}" in completed.stdout
    assert f"manifest-count: {len(repository.included)}" in completed.stdout
    assert tuple(_git(repo, "show-ref") for repo in development_repositories) == development_refs_before
    if push:
        public_tip = _git(repository.public, "rev-parse", "refs/heads/main")
        assert _git(repository.public, "rev-list", "--parents", "-n", "1", public_tip).split() == [
            public_tip,
            repository.public_tip,
        ]
        assert tuple(_git(repository.public, "ls-tree", "-r", "--name-only", public_tip).splitlines()) == tuple(
            sorted(repository.included)
        )
        assert _git(repository.public, "cat-file", "-t", f"refs/tags/{tag}") == "tag"
        assert _git(repository.public, "rev-parse", f"refs/tags/{tag}^{{commit}}") == public_tip
    else:
        assert _git(repository.public, "show-ref") == public_refs_before


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
    """Issue #77: a ref matching an annotated development tag publishes a release tag and message."""
    tag = "v9.9.9"
    remote = "mirror"
    _git(repository.dev, "tag", "-a", tag, "-m", "fixture: development release", repository.source_sha)
    _git(repository.dev, "remote", "add", remote, str(repository.public))

    _run_script(repository.dev, "--remote", remote, "--push", "--tag", tag, "--ref", repository.source_sha)

    public_tip = _git(repository.public, "rev-parse", "refs/heads/main")
    tag_ref = f"refs/tags/{tag}"
    assert _git(repository.public, "cat-file", "-t", tag_ref) == "tag"
    assert _git(repository.public, "rev-parse", f"{tag_ref}^{{commit}}") == public_tip
    assert _git(repository.public, "show", "-s", "--format=%B", public_tip) == f"release: {tag}"


@pytest.mark.parametrize("push", [False, True], ids=["dry-run", "push"])
@pytest.mark.parametrize("annotated", [False, True], ids=["lightweight", "annotated"])
def test_tag_selects_the_development_tag_after_head_advances(
    repository: _RepositoryFixture, push: bool, annotated: bool
) -> None:
    """Issue #77: a release snapshots the development tag's commit even after HEAD advances."""
    tag = "v9.9.9"
    tag_arguments = ["-a", "-m", "fixture: development release"] if annotated else []
    _git(repository.dev, "tag", *tag_arguments, tag, repository.source_sha)
    _write(repository.dev, "README.md", b"# later development state\n")
    _write(repository.dev, "late.txt", b"must not enter the release snapshot\n")
    _git(repository.dev, "add", "README.md", "late.txt")
    _git(repository.dev, "commit", "-m", "fixture: after development release")
    refs_before = _git(repository.public, "show-ref")
    arguments = ["--push"] if push else []

    completed = _run_script(repository.dev, "--tag", tag, *arguments)

    assert f"source-commit: {repository.source_sha}\n" in completed.stdout
    assert f"manifest-count: {len(repository.included)}\n" in completed.stdout
    if push:
        public_tip = _git(repository.public, "rev-parse", "refs/heads/main")
        assert _git(repository.public, "rev-parse", f"refs/tags/{tag}^{{commit}}") == public_tip
        assert tuple(_git(repository.public, "ls-tree", "-r", "--name-only", public_tip).splitlines()) == tuple(
            sorted(repository.included)
        )
        assert _git_bytes(repository.public, "show", f"{public_tip}:README.md") == repository.included["README.md"]
    else:
        assert _git(repository.public, "show-ref") == refs_before


@pytest.mark.parametrize("push", [False, True], ids=["dry-run", "push"])
@pytest.mark.parametrize("same_named_branch", [False, True], ids=["absent", "branch-only"])
def test_missing_development_tag_is_rejected_without_changing_public_refs(
    repository: _RepositoryFixture, push: bool, same_named_branch: bool
) -> None:
    """Issue #77: a missing development tag fails without publishing, even if a branch has that name."""
    tag = "v9.9.9"
    if same_named_branch:
        _git(repository.dev, "branch", tag)
    refs_before = _git(repository.public, "show-ref")
    arguments = ["--push"] if push else []

    completed = _run_script(repository.dev, "--tag", tag, *arguments, check=False)

    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert _git(repository.public, "show-ref") == refs_before


@pytest.mark.parametrize("push", [False, True], ids=["dry-run", "push"])
def test_tag_rejects_an_explicit_ref_at_a_different_commit_without_changing_public_refs(
    repository: _RepositoryFixture, push: bool
) -> None:
    """Issue #77: an explicit HEAD differing from the development tag fails without publishing."""
    tag = "v9.9.9"
    _git(repository.dev, "tag", tag, repository.source_sha)
    _write(repository.dev, "late.txt", b"a different source commit\n")
    _git(repository.dev, "add", "late.txt")
    _git(repository.dev, "commit", "-m", "fixture: after development release")
    refs_before = _git(repository.public, "show-ref")
    arguments = ["--push"] if push else []

    completed = _run_script(repository.dev, "--tag", tag, "--ref", "HEAD", *arguments, check=False)

    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "--ref" in completed.stderr
    assert tag in completed.stderr
    assert _git(repository.public, "show-ref") == refs_before


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
