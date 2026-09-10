"""Specification tests for the installed font catalog."""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import get_args

import pytest
from repository_contracts import require_repo_file

import pixtreme as px

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
_ACTIONABLE = r"^why=.*; what=.*; how=.*$"


def _write_distribution(root: Path, name: str, entry_points: tuple[tuple[str, str], ...]) -> None:
    metadata = root / f"{name.replace('-', '_')}-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: {name}\nVersion: 1.0\n",
        encoding="utf-8",
    )
    declarations = "\n".join(f"{entry_name} = {value}" for entry_name, value in entry_points)
    (metadata / "entry_points.txt").write_text(
        f"[pixtreme.fonts]\n{declarations}\n",
        encoding="utf-8",
    )


def _write_module(root: Path, name: str, source: str) -> None:
    (root / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")


def _run_isolated(
    *roots: Path,
    code: str,
    before_pixtreme_import: str = "",
    pixtreme_import: str = "import pixtreme as px",
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    assert roots
    runtime_metadata = roots[0] / "pixtreme-0.dist-info"
    runtime_metadata.mkdir(exist_ok=True)
    (runtime_metadata / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: pixtreme\nVersion: {px.__version__}\n",
        encoding="utf-8",
    )
    root_literals = ", ".join(repr(str(root)) for root in roots)
    prelude = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        import cupy as _cupy
        import cupyx as _cupyx
        import freetype as _freetype
        import numpy as _numpy
        import annotated_types as _annotated_types
        import pydantic.fields as _pydantic_fields
        from pydantic import BaseModel as _BaseModel
        from pydantic import ConfigDict as _ConfigDict
        from pydantic import ValidationInfo as _ValidationInfo
        from pydantic import field_validator as _field_validator
        import uharfbuzz as _uharfbuzz

        _source_root = Path({str(_SOURCE_ROOT)!r}).resolve()
        _stdlib_paths = [
            item for item in sys.path
            if "site-packages" not in item
            and "dist-packages" not in item
            and "repositories/pixtreme" not in item
            and Path(item or ".").resolve() != _source_root
        ]
        _provider_roots = [{root_literals}]
        sys.path[:] = [*_provider_roots, str(_source_root), *_stdlib_paths]
        """
    )
    prelude += "\n" + textwrap.dedent(before_pixtreme_import)
    prelude += "\n" + textwrap.dedent(pixtreme_import)
    prelude += textwrap.dedent(
        """

        import pixtreme as px
        assert Path(px.__file__).resolve().is_relative_to(_source_root), px.__file__
        """
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    return subprocess.run(
        [sys.executable, "-I", "-c", prelude + "\n" + textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )


def _assert_success(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 0, completed.stderr


def test_fonts_public_surface_is_exact() -> None:
    """v1-fonts-module acceptance 1-2: fonts has one root owner, two exact functions, and no token alias."""
    assert px.__all__[-2:] == ("fonts", "__version__")
    assert px.fonts.__all__ == ("font_path", "available")
    assert {name for name in vars(px.fonts) if not name.startswith("_")} == {"font_path", "available"}

    font_path_signature = inspect.signature(px.fonts.font_path, eval_str=True)
    assert tuple(font_path_signature.parameters) == ("name",)
    assert font_path_signature.parameters["name"].annotation is str
    assert font_path_signature.return_annotation is Path
    assert inspect.signature(px.fonts.available, eval_str=True).return_annotation == tuple[str, ...]
    assert get_args(px.core.TextFont) == ("sans", "mono")

    for forbidden in ("load", "catalog", "register", "unregister", "refresh"):
        assert not hasattr(px.fonts, forbidden)


def test_bundled_catalog_is_exact_without_providers(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 3: a provider-free process exposes the two readable bundled font paths."""
    completed = _run_isolated(
        tmp_path,
        code="""
        import os
        from pathlib import Path

        assert px.fonts.available() == ("sans", "mono")
        paths = {name: px.fonts.font_path(name) for name in px.fonts.available()}
        assert all(isinstance(path, Path) and path.is_absolute() for path in paths.values())
        assert all(path.is_file() and os.access(path, os.R_OK) for path in paths.values())
        assert "NotoSansCJKjp-VF.otf" == paths["sans"].name
        assert "NotoSansMonoCJKjp-VF.otf" == paths["mono"].name
        """,
    )
    _assert_success(completed)


@pytest.mark.parametrize(
    "pixtreme_import",
    ("import pixtreme as px", "import pixtreme.fonts"),
    ids=("root", "fonts-module"),
)
def test_import_invalid_lookup_and_bundled_lookup_do_not_load_providers(tmp_path: Path, pixtreme_import: str) -> None:
    """v1-fonts-module acceptance 4 and 6-7: validation and bundled lookup precede lazy provider loading."""
    marker = tmp_path / "imported"
    font = tmp_path / "extension.otf"
    font.write_bytes(b"font-placeholder")
    _write_module(
        tmp_path,
        "example_lazy_provider",
        f"""
        from pathlib import Path
        Path({str(marker)!r}).write_text("loaded", encoding="utf-8")
        FONT_FILES = {{"extension": Path({str(font)!r})}}
        """,
    )
    _write_distribution(tmp_path, "example-lazy", (("catalog", "example_lazy_provider:FONT_FILES"),))

    completed = _run_isolated(
        tmp_path,
        before_pixtreme_import="""
        import os

        _provider_metadata_reads = []
        _provider_metadata_prefixes = tuple(f"{Path(root).resolve()}{os.sep}" for root in _provider_roots)

        def _observe_provider_metadata(event, args):
            if event != "open" or not args:
                return
            try:
                opened = os.fsdecode(args[0])
            except (TypeError, ValueError):
                return
            frame = sys._getframe(1)
            while frame is not None:
                if frame.f_globals.get("__name__") == "pixtreme._fonts":
                    if opened.endswith("entry_points.txt") and opened.startswith(_provider_metadata_prefixes):
                        _provider_metadata_reads.append(opened)
                    return
                frame = frame.f_back

        sys.addaudithook(_observe_provider_metadata)
        """,
        pixtreme_import=pixtreme_import,
        code=f"""
        import re
        from pathlib import Path

        marker = Path({str(marker)!r})
        assert _provider_metadata_reads == [], ("import metadata read", _provider_metadata_reads)
        assert not marker.exists(), "provider loaded during import"
        assert "example_lazy_provider" not in sys.modules, "provider module present during import"
        for value in (None, 1, "", " sans", "sans ", "\\x00name", "name\\x00", "\tname"):
            try:
                px.fonts.font_path(value)
            except ValueError as error:
                assert re.fullmatch({_ACTIONABLE!r}, str(error))
            else:
                raise AssertionError(value)
            assert _provider_metadata_reads == [], ("invalid lookup metadata read", value, _provider_metadata_reads)
            assert not marker.exists(), ("provider loaded during invalid lookup", value)
        assert px.fonts.font_path("sans").name == "NotoSansCJKjp-VF.otf"
        assert px.fonts.font_path("mono").name == "NotoSansMonoCJKjp-VF.otf"
        assert _provider_metadata_reads == [], ("bundled lookup metadata read", _provider_metadata_reads)
        assert not marker.exists(), "provider loaded during bundled lookup"
        assert "example_lazy_provider" not in sys.modules, "provider module present after bundled lookup"
        assert px.fonts.available() == ("sans", "mono", "extension")
        assert _provider_metadata_reads, "negative control did not observe real metadata discovery"
        assert marker.read_text(encoding="utf-8") == "loaded"
        assert "example_lazy_provider" in sys.modules
        """,
    )
    _assert_success(completed)


def test_provider_catalog_merges_orders_and_snapshots_paths(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 5-7: real metadata merges declarations into an immutable path snapshot."""
    first = tmp_path / "first.otf"
    second = tmp_path / "second.otf"
    target = tmp_path / "target.otf"
    link = tmp_path / "linked.otf"
    for path in (first, second, target):
        path.write_bytes(path.name.encode())
    link.symlink_to(target)

    _write_module(
        tmp_path,
        "example_merge_a",
        f"""
        from pathlib import Path
        FONT_FILES = {{"zeta": Path({str(first)!r}), "é": Path({str(second)!r})}}
        """,
    )
    _write_module(
        tmp_path,
        "example_merge_b",
        f"""
        from pathlib import Path
        FONT_FILES = {{"Alpha": Path({str(link)!r})}}
        """,
    )
    _write_distribution(
        tmp_path,
        "example-merge",
        (("z-declaration", "example_merge_a:FONT_FILES"), ("a-declaration", "example_merge_b:FONT_FILES")),
    )

    completed = _run_isolated(
        tmp_path,
        code=f"""
        import os
        from pathlib import Path
        expected = ("sans", "mono", "Alpha", "é", "zeta")
        assert px.fonts.available() == expected
        assert px.fonts.font_path("Alpha") == Path({str(link)!r})
        assert px.fonts.font_path("Alpha").is_symlink()
        import example_merge_a
        example_merge_a.FONT_FILES["later"] = Path({str(target)!r})
        Path({str(first)!r}).unlink()
        assert px.fonts.available() == expected
        assert px.fonts.font_path("zeta") == Path({str(first)!r})
        assert not px.fonts.font_path("zeta").exists()
        assert "later" not in px.fonts.available()
        """,
    )
    _assert_success(completed)


def test_success_snapshot_ignores_hot_install_and_uninstall_until_fresh_process(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 7: metadata changes affect only a fresh-process success snapshot."""
    provider_root = tmp_path / "providers"
    staged_root = tmp_path / "staged"
    provider_root.mkdir()
    initial_font = tmp_path / "initial.otf"
    added_font = tmp_path / "added.otf"
    initial_font.write_bytes(b"initial-font")
    added_font.write_bytes(b"added-font")
    _write_module(
        provider_root,
        "example_initial_provider",
        f"from pathlib import Path\nFONT_FILES = {{'initial': Path({str(initial_font)!r})}}",
    )
    _write_module(
        provider_root,
        "example_added_provider",
        f"from pathlib import Path\nFONT_FILES = {{'added': Path({str(added_font)!r})}}",
    )
    _write_distribution(
        provider_root,
        "example-initial",
        (("catalog", "example_initial_provider:FONT_FILES"),),
    )
    _write_distribution(
        staged_root,
        "example-added",
        (("catalog", "example_added_provider:FONT_FILES"),),
    )
    initial_metadata = provider_root / "example_initial-1.0.dist-info"
    staged_metadata = staged_root / "example_added-1.0.dist-info"
    added_metadata = provider_root / staged_metadata.name

    completed = _run_isolated(
        provider_root,
        code=f"""
        import shutil
        from pathlib import Path

        expected = ("sans", "mono", "initial")
        assert px.fonts.available() == expected
        assert px.fonts.font_path("initial") == Path({str(initial_font)!r})

        shutil.rmtree(Path({str(initial_metadata)!r}))
        Path({str(staged_metadata)!r}).replace(Path({str(added_metadata)!r}))

        assert px.fonts.available() == expected
        assert px.fonts.font_path("initial") == Path({str(initial_font)!r})
        try:
            px.fonts.font_path("added")
        except ValueError:
            pass
        else:
            raise AssertionError("hot-installed metadata changed the cached catalog")
        """,
    )
    _assert_success(completed)

    fresh = _run_isolated(
        provider_root,
        code=f"""
        from pathlib import Path

        assert px.fonts.available() == ("sans", "mono", "added")
        assert px.fonts.font_path("added") == Path({str(added_font)!r})
        try:
            px.fonts.font_path("initial")
        except ValueError:
            pass
        else:
            raise AssertionError("fresh process retained uninstalled metadata")
        """,
    )
    _assert_success(fresh)


@pytest.mark.parametrize(
    ("module_source", "fragment", "has_cause"),
    (
        ("raise LookupError('load boom')", "could not be loaded", True),
        ("FONT_FILES = ()", "must be a non-empty Mapping", False),
        ("FONT_FILES = {}", "must be a non-empty Mapping", False),
        (
            """
            from collections.abc import Mapping
            class Broken(Mapping):
                def __getitem__(self, key): raise KeyError(key)
                def __iter__(self): return iter(())
                def __len__(self): return 1
                def items(self): raise OSError("iteration boom")
            FONT_FILES = Broken()
            """,
            "could not be iterated",
            True,
        ),
        ("from pathlib import Path\nFONT_FILES = {' bad': Path('/tmp/x')}", "invalid font name", False),
        ("FONT_FILES = {'valid': '/tmp/x'}", "must be pathlib.Path", False),
        ("from pathlib import Path\nFONT_FILES = {'valid': Path('relative.otf')}", "must be absolute", False),
    ),
)
def test_invalid_provider_declarations_fail_closed_and_cache(
    tmp_path: Path, module_source: str, fragment: str, has_cause: bool
) -> None:
    """v1-fonts-module acceptance 6 and 11: provider declaration failures are actionable cached RuntimeErrors."""
    marker = tmp_path / "loads"
    source = (
        f"from pathlib import Path\nPath({str(marker)!r}).open('a', encoding='utf-8').write('x')\n"
        + textwrap.dedent(module_source)
    )
    _write_module(tmp_path, "example_broken_provider", source)
    _write_distribution(tmp_path, "example-broken", (("catalog", "example_broken_provider:FONT_FILES"),))

    completed = _run_isolated(
        tmp_path,
        code=f"""
        import re
        from pathlib import Path

        messages = []
        causes = []
        for _ in range(2):
            try:
                px.fonts.available()
            except RuntimeError as error:
                messages.append(str(error))
                causes.append(error.__cause__)
            else:
                raise AssertionError("provider failure was not fail-closed")
        assert messages[0] == messages[1]
        assert re.fullmatch({_ACTIONABLE!r}, messages[0])
        assert {fragment!r} in messages[0]
        assert all((cause is not None) is {has_cause!r} for cause in causes)
        assert Path({str(marker)!r}).read_text(encoding="utf-8") == "x"
        """,
    )
    _assert_success(completed)


@pytest.mark.parametrize("kind", ("missing", "directory", "unreadable"))
def test_invalid_provider_paths_fail_closed(tmp_path: Path, kind: str) -> None:
    """v1-fonts-module acceptance 5 and 11: provider paths must be absolute readable regular files."""
    candidate = tmp_path / "candidate.otf"
    if kind == "directory":
        candidate.mkdir()
    elif kind == "unreadable":
        candidate.write_bytes(b"font-placeholder")
        candidate.chmod(0)

    _write_module(
        tmp_path,
        "example_path_provider",
        f"from pathlib import Path\nFONT_FILES = {{'extension': Path({str(candidate)!r})}}",
    )
    _write_distribution(tmp_path, "example-path", (("catalog", "example_path_provider:FONT_FILES"),))
    try:
        completed = _run_isolated(
            tmp_path,
            code=f"""
            import re
            try:
                px.fonts.available()
            except RuntimeError as error:
                assert re.fullmatch({_ACTIONABLE!r}, str(error))
                assert "regular readable file" in str(error)
            else:
                raise AssertionError("invalid path was accepted")
            """,
        )
        _assert_success(completed)
    finally:
        if candidate.exists() and kind == "unreadable":
            candidate.chmod(0o600)


def test_metadata_enumeration_failure_is_chained_and_cached(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 11: metadata enumeration failure is an actionable cached RuntimeError."""
    completed = _run_isolated(
        tmp_path,
        code=f"""
        import re
        import pixtreme._fonts as implementation

        calls = 0
        def broken_entry_points(*, group):
            global calls
            calls += 1
            assert group == "pixtreme.fonts"
            raise OSError("metadata boom")
        implementation._metadata.entry_points = broken_entry_points
        messages = []
        for _ in range(2):
            try:
                px.fonts.available()
            except RuntimeError as error:
                assert isinstance(error.__cause__, OSError)
                assert re.fullmatch({_ACTIONABLE!r}, str(error))
                messages.append(str(error))
            else:
                raise AssertionError("metadata failure was not reported")
        assert messages[0] == messages[1]
        assert calls == 1
        """,
    )
    _assert_success(completed)


def test_failure_snapshot_ignores_path_repair_until_fresh_process(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 11: path repair recovers only in a fresh process after cached failure."""
    marker = tmp_path / "loads"
    repaired_font = tmp_path / "repaired.otf"
    _write_module(
        tmp_path,
        "example_repair_provider",
        f"""
        from pathlib import Path
        Path({str(marker)!r}).open("a", encoding="utf-8").write("x")
        FONT_FILES = {{"repaired": Path({str(repaired_font)!r})}}
        """,
    )
    _write_distribution(tmp_path, "example-repair", (("catalog", "example_repair_provider:FONT_FILES"),))

    completed = _run_isolated(
        tmp_path,
        code=f"""
        from pathlib import Path

        messages = []
        try:
            px.fonts.available()
        except RuntimeError as error:
            messages.append(str(error))
        else:
            raise AssertionError("missing provider path was accepted")

        Path({str(repaired_font)!r}).write_bytes(b"repaired-font")
        try:
            px.fonts.available()
        except RuntimeError as error:
            messages.append(str(error))
        else:
            raise AssertionError("cached failure retried after path repair")

        assert messages[0] == messages[1]
        assert Path({str(marker)!r}).read_text(encoding="utf-8") == "x"
        """,
    )
    _assert_success(completed)

    fresh = _run_isolated(
        tmp_path,
        code=f"""
        from pathlib import Path

        assert px.fonts.available() == ("sans", "mono", "repaired")
        assert px.fonts.font_path("repaired") == Path({str(repaired_font)!r})
        assert Path({str(marker)!r}).read_text(encoding="utf-8") == "xx"
        """,
    )
    _assert_success(fresh)


def test_collisions_are_deterministic_across_discovery_order(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 10-11: all collisions fail atomically with stable diagnostics."""
    roots = (tmp_path / "z-root", tmp_path / "a-root")
    for root in roots:
        root.mkdir()
    font = tmp_path / "shared.otf"
    font.write_bytes(b"font-placeholder")
    _write_module(
        roots[0],
        "example_collision_z",
        f"from pathlib import Path\nFONT_FILES = {{'shared': Path({str(font)!r}), 'sans': Path({str(font)!r})}}",
    )
    _write_module(
        roots[1],
        "example_collision_a",
        f"from pathlib import Path\nFONT_FILES = {{'shared': Path({str(font)!r})}}",
    )
    _write_distribution(roots[0], "z-example", (("catalog-z", "example_collision_z:FONT_FILES"),))
    _write_distribution(roots[1], "a-example", (("catalog-a", "example_collision_a:FONT_FILES"),))

    messages: list[str] = []
    for ordered_roots in (roots, roots[::-1]):
        completed = _run_isolated(
            *ordered_roots,
            code="""
            try:
                px.fonts.available()
            except RuntimeError as error:
                print(str(error))
                assert px.fonts.font_path("sans").name == "NotoSansCJKjp-VF.otf"
            else:
                raise AssertionError("collisions were accepted")
            """,
        )
        _assert_success(completed)
        messages.append(completed.stdout.strip())
    assert messages[0] == messages[1]
    assert messages[0].index("sans") < messages[0].index("shared")
    assert "a-example" in messages[0] and "z-example" in messages[0]
    assert re.fullmatch(_ACTIONABLE, messages[0])


@pytest.mark.parametrize("fails", (False, True), ids=("success", "failure"))
def test_concurrent_first_discovery_has_one_atomic_result(tmp_path: Path, fails: bool) -> None:
    """v1-fonts-module acceptance 8: concurrent first discovery loads once and exposes one atomic result."""
    marker = tmp_path / "loads"
    font = tmp_path / "concurrent.otf"
    font.write_bytes(b"font-placeholder")
    outcome = "raise LookupError('concurrent boom')" if fails else f"FONT_FILES = {{'threaded': Path({str(font)!r})}}"
    _write_module(
        tmp_path,
        "example_thread_provider",
        f"""
        import time
        from pathlib import Path
        Path({str(marker)!r}).open("a", encoding="utf-8").write("x")
        time.sleep(0.1)
        {outcome}
        """,
    )
    _write_distribution(tmp_path, "example-thread", (("catalog", "example_thread_provider:FONT_FILES"),))

    completed = _run_isolated(
        tmp_path,
        code=f"""
        import threading
        from pathlib import Path

        barrier = threading.Barrier(8)
        results = []
        def call():
            barrier.wait()
            try:
                results.append(("ok", px.fonts.available()))
            except RuntimeError as error:
                results.append(("error", str(error)))
        threads = [threading.Thread(target=call) for _ in range(8)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(10)
        assert all(not thread.is_alive() for thread in threads)
        assert len(results) == 8
        assert len(set(results)) == 1
        assert results[0][0] == {"error" if fails else "ok"!r}
        assert Path({str(marker)!r}).read_text(encoding="utf-8") == "x"
        """,
        timeout=30,
    )
    _assert_success(completed)


def test_same_thread_reentry_poison_is_not_recoverable_by_provider(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 9: same-thread provider reentry fails without hanging and poisons the snapshot."""
    marker = tmp_path / "loads"
    inner = tmp_path / "inner-error"
    font = tmp_path / "reentry.otf"
    font.write_bytes(b"font-placeholder")
    _write_module(
        tmp_path,
        "example_reentrant_provider",
        f"""
        from pathlib import Path
        import pixtreme as px
        Path({str(marker)!r}).open("a", encoding="utf-8").write("x")
        try:
            px.fonts.available()
        except RuntimeError as error:
            Path({str(inner)!r}).write_text(str(error), encoding="utf-8")
        FONT_FILES = {{"reentrant": Path({str(font)!r})}}
        """,
    )
    _write_distribution(tmp_path, "example-reentrant", (("catalog", "example_reentrant_provider:FONT_FILES"),))

    completed = _run_isolated(
        tmp_path,
        code=f"""
        from pathlib import Path
        messages = []
        for _ in range(2):
            try:
                px.fonts.available()
            except RuntimeError as error:
                messages.append(str(error))
            else:
                raise AssertionError("provider recovered from forbidden reentry")
        assert messages[0] == messages[1] == Path({str(inner)!r}).read_text(encoding="utf-8")
        assert Path({str(marker)!r}).read_text(encoding="utf-8") == "x"
        """,
        timeout=10,
    )
    _assert_success(completed)


def test_unknown_name_error_is_neutral_and_catalog_failure_takes_precedence(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 12: unknown names are neutral ValueErrors only for a healthy catalog."""
    font = tmp_path / "known.otf"
    font.write_bytes(b"font-placeholder")
    _write_module(
        tmp_path,
        "example_known_provider",
        f"from pathlib import Path\nFONT_FILES = {{'known': Path({str(font)!r})}}",
    )
    _write_distribution(tmp_path, "example-known", (("catalog", "example_known_provider:FONT_FILES"),))
    completed = _run_isolated(
        tmp_path,
        code=f"""
        import re
        try:
            px.fonts.font_path("unknown")
        except ValueError as error:
            message = str(error)
            assert re.fullmatch({_ACTIONABLE!r}, message)
            assert "unknown" in message
            assert repr(("sans", "mono", "known")) in message
            assert "font package" in message
            assert "px.draw.Font.from_file" in message
            for forbidden in ("example-known", "example_known_provider", "pip install", "github.com", "https://"):
                assert forbidden not in message
        else:
            raise AssertionError("unknown name was accepted")
        """,
    )
    _assert_success(completed)


def test_unknown_lookup_preserves_cached_broken_catalog_runtime_error(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 12: a broken catalog takes precedence over unknown-name ValueError."""
    marker = tmp_path / "loads"
    _write_module(
        tmp_path,
        "example_unknown_failure",
        f"""
        from pathlib import Path
        Path({str(marker)!r}).open("a", encoding="utf-8").write("x")
        raise LookupError("catalog boom")
        """,
    )
    _write_distribution(tmp_path, "example-unknown-failure", (("catalog", "example_unknown_failure:FONT_FILES"),))

    completed = _run_isolated(
        tmp_path,
        code=f"""
        from pathlib import Path

        observed = []
        for call in (px.fonts.available, lambda: px.fonts.font_path("unknown")):
            try:
                call()
            except Exception as error:
                observed.append((type(error), str(error)))
            else:
                raise AssertionError("broken catalog was not reported")

        assert observed[0][0] is RuntimeError
        assert observed[1][0] is RuntimeError
        assert observed[0][1] == observed[1][1]
        assert Path({str(marker)!r}).read_text(encoding="utf-8") == "x"
        """,
    )
    _assert_success(completed)


def test_bundled_integrity_failure_is_distinct_from_provider_failure(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 4: broken bundled package data raises its own actionable RuntimeError."""
    completed = _run_isolated(
        tmp_path,
        code=f"""
        import re
        from pathlib import Path
        import pixtreme._fonts as implementation

        implementation._BUNDLED_FONT_PATHS["sans"] = Path({str(tmp_path / "missing.otf")!r})
        try:
            px.fonts.font_path("sans")
        except RuntimeError as error:
            assert re.fullmatch({_ACTIONABLE!r}, str(error))
            assert "bundled font package data" in str(error)
            assert error.__cause__ is None
        else:
            raise AssertionError("broken package data was accepted")
        """,
    )
    _assert_success(completed)


def test_broken_provider_does_not_change_draw_token_results(tmp_path: Path) -> None:
    """v1-fonts-module acceptance 13: draw tokens stay bit-identical and independent from provider failure."""
    _write_module(tmp_path, "example_draw_failure", "raise LookupError('broken provider')")
    _write_distribution(tmp_path, "example-draw", (("catalog", "example_draw_failure:FONT_FILES"),))
    completed = _run_isolated(
        tmp_path,
        code="""
        import cupy as cp
        import freetype
        import uharfbuzz

        source = px.core.Frame(
            data=cp.zeros((64, 128, 1), dtype=cp.float32),
            colorspace="Rec.709",
            gamma="linear",
            channels=("Y",),
            matrix=None,
        )
        kwargs = {"text": "font", "position": (4.0, 44.0), "size": 24.0, "color": (1.0,), "font": "sans"}
        before = px.draw.text(source, **kwargs)
        try:
            px.fonts.available()
        except RuntimeError:
            pass
        else:
            raise AssertionError("broken provider was accepted")
        assert px.fonts.font_path("sans").name == "NotoSansCJKjp-VF.otf"
        after = px.draw.text(source, **kwargs)
        assert cp.array_equal(before.data, after.data)
        """,
        timeout=60,
    )
    _assert_success(completed)


def test_fonts_canonical_docs_and_docstrings_are_self_contained() -> None:
    """v1-fonts-module acceptance 1 and 14-15; v1-grade acceptance 1: public docs expose one exact catalog surface."""
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    readme = require_repo_file("README.md").read_text(encoding="utf-8")
    tokens = (_ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    font_path_doc = inspect.getdoc(px.fonts.font_path) or ""
    available_doc = inspect.getdoc(px.fonts.available) or ""
    draw_doc = inspect.getdoc(px.draw.text) or ""
    from_file_doc = inspect.getdoc(px.draw.Font.from_file) or ""

    for fragment in ("REQ-PKG-005", "`fonts`", "| 2 |", "公開 operation は計 98 関数"):
        assert fragment in requirements
    root_module_counts = re.findall(r"\b(\d+)\s+(?:focused(?: operation)?\s+)?modules\b", readme)
    assert root_module_counts
    assert set(root_module_counts) == {"14"}
    assert "14 focused modules, including `core`" in readme
    for fragment in (
        "px.fonts.font_path",
        "px.fonts.available",
        "pixtreme.fonts",
        "Mapping[str, pathlib.Path]",
        "sans",
        "mono",
        "case-sensitive",
        "process",
        "fresh process",
        "stale",
        "filesystem",
        "RuntimeError",
        "ValueError",
    ):
        assert fragment in tokens
    for fragment in ("exact", "case-sensitive", "path", "snapshot", "stale", "ValueError", "RuntimeError"):
        assert fragment.lower() in font_path_doc.lower()
    for fragment in ("sans", "mono", "code-point", "entry point", "process", "thread", "RuntimeError"):
        assert fragment.lower() in available_doc.lower()
    assert "px.fonts.font_path" in draw_doc
    assert "px.fonts.font_path" in from_file_doc
