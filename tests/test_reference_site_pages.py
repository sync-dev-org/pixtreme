"""Structural parity between the public-module contract and the API reference site pages."""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_CONTRACT_PATH = _TESTS_DIR / "test_public_namespace_contract.py"
_REFERENCE_DIR = _TESTS_DIR.parent / "docs_site" / "reference"


def _contract_root_modules() -> tuple[str, ...]:
    tree = ast.parse(_CONTRACT_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ROOT_MODULES":
                    return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"ROOT_MODULES assignment not found in {_CONTRACT_PATH}")


def test_reference_pages_match_public_module_contract_characterization() -> None:
    """characterization: freeze the one-page-per-public-module structure of the reference site.

    The mkdocs reference site renders one page per public module through a module-level
    mkdocstrings directive, so member-level drift is impossible while the page set matches
    the public-module contract; only adding or removing a public module requires a page
    change, and nothing else pins that. Correctness is not independently established
    because no requirement or feature sheet specifies the reference site yet. Supersede
    this pin with a canonical clause when a docs feature sheet or REQ covers the site.
    """
    modules = _contract_root_modules()
    pages = {path.stem for path in _REFERENCE_DIR.glob("*.md")}
    assert pages == set(modules), (
        f"reference pages and ROOT_MODULES diverge: "
        f"missing_pages={sorted(set(modules) - pages)}; extra_pages={sorted(pages - set(modules))}"
    )
    for module in modules:
        page_text = (_REFERENCE_DIR / f"{module}.md").read_text(encoding="utf-8")
        assert f"::: pixtreme.{module}" in page_text, (
            f"docs_site/reference/{module}.md lacks its mkdocstrings directive"
        )
