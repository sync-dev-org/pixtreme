"""Documentation contract tests for the public token reference."""

from __future__ import annotations


def test_legal_to_full_docstring_contains_the_reverse_composition_recipe() -> None:
    """v1-color-semantics acceptance 34: API docs use the directional color pair in the repair recipe."""
    import inspect

    docstring = inspect.getdoc(__import__("pixtreme").values.legal_to_full)
    assert docstring is not None
    first_transform = docstring.index("px.color.rgb_to_ycbcr")
    range_conversion = docstring.index("px.values.legal_to_full", first_transform)
    second_transform = docstring.index("px.color.ycbcr_to_rgb", range_conversion)
    assert first_transform < range_conversion < second_transform
    for required in ('matrix="BT.709"', "bit_depth=8"):
        assert required in docstring
