"""Documentation contract for embedded ICC metadata and its five vocabulary additions."""

from __future__ import annotations

import inspect

from repository_contracts import latest_changelog_section, require_repo_file

import pixtreme as px


def test_icc_token_reference_requirements_features_docstrings_and_changelog_are_synchronized() -> None:
    """v1-chroma-siting-h273 acceptance 10; v1-io-icc acceptance 22:
    every canonical and public prose projection describes the implemented contract.
    """
    tokens = require_repo_file("docs_site/tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = latest_changelog_section(require_repo_file("CHANGELOG.md").read_text(encoding="utf-8"))

    for claim in ("199 canonical tokens", "29 Colorspace", "36 Gamma"):
        assert claim in tokens
        assert claim in requirements
    for token in ("Gamma-1.8", "Adobe-RGB", "ProPhoto-RGB"):
        assert f"`{token}`" in tokens
        assert token in changelog
    for claim in (
        "563 / 256",
        "1 / 32",
        "1 / 512",
        "0.2100",
        "0.7100",
        "0.1596",
        "0.8404",
        "16 MiB",
        "cICP > iCCP > sRGB > gAMA",
        "ICC_PROFILE",
        "InterColorProfile",
        "ICCP",
        'raw["ICC"]',
        "mappable",
    ):
        assert claim in tokens

    arch = requirements.split("**REQ-ARCH-003", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    api_color = requirements.split("**REQ-API-003", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    api_io = requirements.split("**REQ-API-005", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "Gamma-1.8" in api_color
    assert "ICC" in api_io and "file 明示" in api_io and "写像不能" in api_io
    assert "permanent alias" in arch

    for relative_path in (
        "docs/features/v1-io.md",
        "docs/features/v1-io-formats.md",
        "docs/features/v1-io-orientation.md",
    ):
        feature = require_repo_file(relative_path).read_text(encoding="utf-8")
        assert "v1-io-icc" in feature
        assert "ICC" in feature

    for operation in (
        px.io.read_image,
        px.io.decode_image,
        px.io.read_header,
        px.color.rgb_to_rgb,
        px.color.gamma_to_linear,
        px.color.linear_to_gamma,
    ):
        docstring = inspect.getdoc(operation)
        assert docstring is not None
        assert "Adobe-RGB" in docstring
        assert "ProPhoto-RGB" in docstring
    header_docstring = inspect.getdoc(px.io.ImageHeader)
    assert header_docstring is not None
    assert "ICC" in header_docstring and "mappable" in header_docstring

    added = changelog.split("### Added", maxsplit=1)[1].split("### Changed", maxsplit=1)[0]
    changed = changelog.split("### Changed", maxsplit=1)[1]
    assert "ICC" in added and "five" in added
    assert "metadata" in changed and "pixel" in changed and "color" in changed
