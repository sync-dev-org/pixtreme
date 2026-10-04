"""Documentation contract tests for the public token reference."""

from __future__ import annotations

import pytest


@pytest.mark.req("REQ-PIX-008")
@pytest.mark.req("REQ-PIX-017")
def test_legal_to_full_docstring_contains_the_reverse_composition_recipe() -> None:
    """The public legal-to-full documentation shows color conversion, range conversion, and the reverse color
    step in order."""
    import inspect

    docstring = inspect.getdoc(__import__("pixtreme").values.legal_to_full)
    assert docstring is not None
    first_transform = docstring.index("px.color.rgb_to_ycbcr")
    range_conversion = docstring.index("px.values.legal_to_full", first_transform)
    second_transform = docstring.index("px.color.ycbcr_to_rgb", range_conversion)
    assert first_transform < range_conversion < second_transform
    for required in ('matrix="BT.709"', "bit_depth=8"):
        assert required in docstring
