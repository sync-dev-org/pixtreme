"""Installed font catalog implementation."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata as _metadata
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NoReturn

from pixtreme._core.errors import _actionable_error

_ENTRY_POINT_GROUP = "pixtreme.fonts"
_BUNDLED_FONT_NAMES = ("sans", "mono")
_BUNDLED_FONT_PATHS = {
    "sans": Path(__file__).parent / "data" / "fonts" / "NotoSansCJKjp-VF.otf",
    "mono": Path(__file__).parent / "data" / "fonts" / "NotoSansMonoCJKjp-VF.otf",
}


@dataclass(frozen=True, slots=True)
class _FailureSnapshot:
    message: str
    cause: BaseException | None

    def raise_error(self) -> NoReturn:
        error = RuntimeError(self.message)
        if self.cause is not None:
            raise error from self.cause
        raise error


@dataclass(frozen=True, slots=True)
class _Declaration:
    distribution_name: str
    entry_point_name: str
    entry_point_value: str
    entry_point: _metadata.EntryPoint

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.distribution_name, self.entry_point_name, self.entry_point_value)

    @property
    def descriptor(self) -> str:
        return (
            f"distribution={self.distribution_name!r}, entry_point={self.entry_point_name!r}, "
            f"value={self.entry_point_value!r}"
        )


_CatalogState = Literal["uninitialized", "initializing", "success", "failure"]
_catalog_condition = threading.Condition(threading.RLock())
_catalog_state: _CatalogState = "uninitialized"
_initializing_thread: int | None = None
_extension_snapshot: Mapping[str, Path] = MappingProxyType({})
_failure_snapshot: _FailureSnapshot | None = None
_reentrant_failure: _FailureSnapshot | None = None


def _failure_from(error: RuntimeError) -> _FailureSnapshot:
    return _FailureSnapshot(message=str(error), cause=error.__cause__)


def _validate_name(name: object, *, provider: _Declaration | None = None) -> str:
    if isinstance(name, str) and name != "" and name.strip() == name and "\x00" not in name:
        return name
    if provider is None:
        raise ValueError(
            _actionable_error(
                why="font name must be a non-empty exact case-sensitive str without surrounding whitespace or NUL",
                what=f"received name={name!r}",
                how="pass a non-empty str equal to str.strip(), without NUL; names are exact and case-sensitive",
            )
        )
    raise RuntimeError(
        _actionable_error(
            why="font provider declared an invalid font name",
            what=f"{provider.descriptor}, name={name!r}",
            how="declare every key as a non-empty str equal to str.strip(), without NUL",
        )
    )


def _bundled_font_path(name: str) -> Path:
    path = _BUNDLED_FONT_PATHS[name]
    try:
        absolute = path.is_absolute()
        exists = path.exists()
        regular = path.is_file()
        readable = os.access(path, os.R_OK)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            _actionable_error(
                why="bundled font package data state could not be inspected",
                what=f"name={name!r}, path={path!r}, state={type(error).__name__}: {error}",
                how="repair or reinstall pixtreme so its bundled font package data is a readable regular file",
            )
        ) from error
    if not (absolute and regular and readable):
        raise RuntimeError(
            _actionable_error(
                why="bundled font package data must be an absolute readable regular file",
                what=(
                    f"name={name!r}, path={path!r}, absolute={absolute}, exists={exists}, "
                    f"is_file={regular}, readable={readable}"
                ),
                how="repair or reinstall pixtreme so its bundled font package data is a readable regular file",
            )
        )
    return path


def _declaration(entry_point: _metadata.EntryPoint) -> _Declaration:
    try:
        distribution = entry_point.dist
        distribution_name = distribution.metadata.get("Name") if distribution is not None else None
        entry_point_name = entry_point.name
        entry_point_value = entry_point.value
    except Exception as error:
        raise RuntimeError(
            _actionable_error(
                why="font provider metadata could not be inspected",
                what=f"entry_point={entry_point!r}, state={type(error).__name__}: {error}",
                how=f"repair the installed distribution metadata for entry point group {_ENTRY_POINT_GROUP!r}",
            )
        ) from error
    if not isinstance(distribution_name, str) or distribution_name == "":
        distribution_name = "<unknown>"
    if not isinstance(entry_point_name, str) or not isinstance(entry_point_value, str):
        raise RuntimeError(
            _actionable_error(
                why="font provider entry point name and value must be strings",
                what=(
                    f"distribution={distribution_name!r}, entry_point={entry_point_name!r}, value={entry_point_value!r}"
                ),
                how=f"repair the installed entry point declaration in group {_ENTRY_POINT_GROUP!r}",
            )
        )
    return _Declaration(distribution_name, entry_point_name, entry_point_value, entry_point)


def _raise_reentrant_failure() -> None:
    with _catalog_condition:
        failure = _reentrant_failure
    if failure is not None:
        failure.raise_error()


def _provider_items(declaration: _Declaration) -> tuple[tuple[object, object], ...]:
    try:
        loaded = declaration.entry_point.load()
    except Exception as error:
        _raise_reentrant_failure()
        raise RuntimeError(
            _actionable_error(
                why="font provider entry point could not be loaded",
                what=f"{declaration.descriptor}, state={type(error).__name__}: {error}",
                how="repair or uninstall the failing font provider and start a new Python process",
            )
        ) from error
    _raise_reentrant_failure()
    if not isinstance(loaded, Mapping):
        raise RuntimeError(
            _actionable_error(
                why="font provider object must be a non-empty Mapping",
                what=f"{declaration.descriptor}, observed_type={type(loaded).__module__}.{type(loaded).__qualname__}",
                how="expose a non-empty Mapping[str, pathlib.Path] constant from the entry point",
            )
        )
    try:
        raw_items = tuple(loaded.items())
    except Exception as error:
        raise RuntimeError(
            _actionable_error(
                why="font provider mapping could not be iterated",
                what=f"{declaration.descriptor}, state={type(error).__name__}: {error}",
                how="expose a stable non-empty Mapping[str, pathlib.Path] constant",
            )
        ) from error
    if not raw_items:
        raise RuntimeError(
            _actionable_error(
                why="font provider object must be a non-empty Mapping",
                what=f"{declaration.descriptor}, observed_items=()",
                how="declare at least one font name and absolute pathlib.Path",
            )
        )
    result: list[tuple[object, object]] = []
    for item in raw_items:
        try:
            name, path = item
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                _actionable_error(
                    why="font provider mapping items must be key and path pairs",
                    what=f"{declaration.descriptor}, observed_item={item!r}",
                    how="expose a Mapping[str, pathlib.Path] constant",
                )
            ) from error
        result.append((name, path))
    return tuple(result)


def _provider_path(name: str, path: object, declaration: _Declaration) -> Path:
    if not isinstance(path, Path):
        raise RuntimeError(
            _actionable_error(
                why="font provider path must be pathlib.Path",
                what=(
                    f"{declaration.descriptor}, name={name!r}, "
                    f"observed_type={type(path).__module__}.{type(path).__qualname__}, path={path!r}"
                ),
                how="declare an absolute pathlib.Path to a readable regular file",
            )
        )
    if not path.is_absolute():
        raise RuntimeError(
            _actionable_error(
                why="font provider path must be absolute",
                what=f"{declaration.descriptor}, name={name!r}, path={path!r}",
                how="declare an absolute pathlib.Path to a readable regular file",
            )
        )
    try:
        valid = path.is_file() and os.access(path, os.R_OK)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            _actionable_error(
                why="font provider path state could not be inspected",
                what=(f"{declaration.descriptor}, name={name!r}, path={path!r}, state={type(error).__name__}: {error}"),
                how="declare an absolute pathlib.Path to a readable regular file",
            )
        ) from error
    if not valid:
        raise RuntimeError(
            _actionable_error(
                why="font provider path must identify a regular readable file at discovery time",
                what=f"{declaration.descriptor}, name={name!r}, path={path!r}",
                how="install or repair the provider asset and start a new Python process",
            )
        )
    return path


def _discover_extension_catalog() -> dict[str, Path]:
    try:
        entry_points = tuple(_metadata.entry_points(group=_ENTRY_POINT_GROUP))
    except Exception as error:
        raise RuntimeError(
            _actionable_error(
                why="installed font provider metadata could not be enumerated",
                what=f"group={_ENTRY_POINT_GROUP!r}, state={type(error).__name__}: {error}",
                how="repair the Python installation metadata and start a new Python process",
            )
        ) from error

    declarations = sorted((_declaration(entry_point) for entry_point in entry_points), key=lambda item: item.key)
    candidates: dict[str, list[tuple[str, Path]]] = {}
    for declaration in declarations:
        for raw_name, raw_path in _provider_items(declaration):
            name = _validate_name(raw_name, provider=declaration)
            path = _provider_path(name, raw_path, declaration)
            candidates.setdefault(name, []).append((declaration.descriptor, path))

    collisions = tuple(
        (name, tuple(sorted(descriptor for descriptor, _path in providers)))
        for name, providers in sorted(candidates.items())
        if name in _BUNDLED_FONT_NAMES or len(providers) > 1
    )
    if collisions:
        raise RuntimeError(
            _actionable_error(
                why="font provider registrations conflict with reserved or duplicate names",
                what=f"collisions={collisions!r}",
                how="make all provider font names disjoint, avoid 'sans' and 'mono', then start a new Python process",
            )
        )
    return {name: providers[0][1] for name, providers in candidates.items()}


def _extension_catalog() -> Mapping[str, Path]:
    global _catalog_state, _extension_snapshot, _failure_snapshot, _initializing_thread, _reentrant_failure

    current_thread = threading.get_ident()
    with _catalog_condition:
        while True:
            if _catalog_state == "success":
                return _extension_snapshot
            if _catalog_state == "failure":
                assert _failure_snapshot is not None
                _failure_snapshot.raise_error()
            if _catalog_state == "uninitialized":
                _catalog_state = "initializing"
                _initializing_thread = current_thread
                _reentrant_failure = None
                break
            if _initializing_thread == current_thread:
                failure = _FailureSnapshot(
                    message=_actionable_error(
                        why="font provider re-entered extension catalog discovery on the initializing thread",
                        what=f"group={_ENTRY_POINT_GROUP!r}, thread_id={current_thread}",
                        how="make provider import expose its mapping without calling px.fonts, then start a new process",
                    ),
                    cause=None,
                )
                _reentrant_failure = failure
                failure.raise_error()
            _catalog_condition.wait()

    try:
        discovered = _discover_extension_catalog()
        _raise_reentrant_failure()
    except RuntimeError as error:
        failure = _failure_from(error)
        with _catalog_condition:
            if _reentrant_failure is not None:
                failure = _reentrant_failure
            _failure_snapshot = failure
            _catalog_state = "failure"
            _initializing_thread = None
            _catalog_condition.notify_all()
        failure.raise_error()
    except Exception as error:
        failure = _FailureSnapshot(
            message=_actionable_error(
                why="installed font catalog discovery failed",
                what=f"group={_ENTRY_POINT_GROUP!r}, state={type(error).__name__}: {error}",
                how="repair or uninstall the failing font provider and start a new Python process",
            ),
            cause=error,
        )
        with _catalog_condition:
            _failure_snapshot = failure
            _catalog_state = "failure"
            _initializing_thread = None
            _catalog_condition.notify_all()
        failure.raise_error()

    with _catalog_condition:
        _extension_snapshot = MappingProxyType(dict(discovered))
        _catalog_state = "success"
        _initializing_thread = None
        _catalog_condition.notify_all()
        return _extension_snapshot


def available() -> tuple[str, ...]:
    """Return bundled and installed font names from one process-lifetime snapshot.

    ``sans`` and ``mono`` are first in that fixed order; installed names follow
    in Python code-point order. The first call discovers and imports entry points
    from group ``pixtreme.fonts`` atomically across threads. Both success and
    failure are cached until a fresh process, and provider failures raise
    actionable ``RuntimeError`` rather than exposing a partial catalog.
    """
    for name in _BUNDLED_FONT_NAMES:
        _bundled_font_path(name)
    extension = _extension_catalog()
    return (*_BUNDLED_FONT_NAMES, *sorted(extension))


def font_path(name: str) -> Path:
    """Return the filesystem path snapshot registered under one exact font name.

    Names are non-empty, exact, case-sensitive strings without surrounding
    whitespace or NUL. Bundled ``sans`` and ``mono`` paths bypass installed
    provider discovery. Other names trigger the process-lifetime entry-point
    snapshot; its provider-supplied ``Path`` is returned without resolving it or
    rechecking it after discovery, so it can become stale. Invalid or unknown
    names raise actionable ``ValueError``; bundled-data or provider-catalog
    failures raise actionable ``RuntimeError``.
    """
    checked_name = _validate_name(name)
    if checked_name in _BUNDLED_FONT_NAMES:
        return _bundled_font_path(checked_name)
    extension = _extension_catalog()
    try:
        return extension[checked_name]
    except KeyError:
        names = (*_BUNDLED_FONT_NAMES, *sorted(extension))
        raise ValueError(
            _actionable_error(
                why="requested font name is not registered",
                what=f"name={checked_name!r}, available={names!r}",
                how=(
                    "install a font package and retry in a new Python process, or pass an existing font file path "
                    "to px.draw.Font.from_file"
                ),
            )
        ) from None
