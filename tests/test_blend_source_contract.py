"""Contracts for the shared CUDA blend-value substrate."""

from __future__ import annotations

import ast
import inspect
import re


def _normalized_assignment(source: str, left_pattern: str) -> str:
    match = re.search(rf"{left_pattern}\s*=\s*(.*?);", source, flags=re.DOTALL)
    assert match is not None
    return " ".join(match.group(1).split())


def _normalized_returns(source: str) -> tuple[str, ...]:
    return tuple(" ".join(match.split()) for match in re.findall(r"\breturn\s+(.*?);", source, flags=re.DOTALL))


def test_blend_code_tables_preserve_canonical_indices_and_draw_subset() -> None:
    """REQ-TEST-001 and REQ-TEST-003: canonical blend indices stay explicit while draw keeps its public token subset."""
    from pixtreme._core.blend import _BLEND_CODES, _BLEND_TOKENS, _DRAW_BLEND_CODES, _DRAW_BLEND_TOKENS

    assert _BLEND_TOKENS == (
        "normal",
        "lighten",
        "add",
        "screen",
        "darken",
        "multiply",
        "difference",
        "overlay",
        "hardlight",
        "softlight",
    )
    assert _BLEND_CODES == {token: index for index, token in enumerate(_BLEND_TOKENS)}
    assert _DRAW_BLEND_TOKENS == ("normal", "add", "multiply", "screen")
    assert _DRAW_BLEND_CODES == {"normal": 0, "add": 2, "multiply": 5, "screen": 3}


def test_blend_consumers_embed_the_canonical_device_fragment_once() -> None:
    """REQ-TEST-003 structure contract: each runtime kernel embeds one shared full-table blend-value fragment."""
    import pixtreme._composite.merge as composite_merge
    import pixtreme._draw.shapes as shapes
    import pixtreme._draw.text as text
    from pixtreme._core.blend import _BLEND_DEVICE_SOURCE

    for source in (
        shapes._DRAW_KERNEL_SOURCE,
        text._TEXT_COMPOSITE_KERNEL_SOURCE,
        composite_merge._COMPOSITE_KERNEL_SOURCE,
    ):
        assert source.count(_BLEND_DEVICE_SOURCE) == 1
        assert source.count("pixtreme_blend(") == 2


def test_blend_consumers_bind_kernel_sources_to_shared_fragment_dataflow() -> None:
    """REQ-TEST-003 structure contract: every consumer kernel assignment dataflow includes the shared fragment."""
    import pixtreme._composite.merge as composite_merge
    import pixtreme._draw.shapes as shapes
    import pixtreme._draw.text as text

    modules_and_assignments = (
        (shapes, "_DRAW_KERNEL_SOURCE"),
        (text, "_TEXT_COMPOSITE_KERNEL_SOURCE"),
        (composite_merge, "_COMPOSITE_KERNEL_SOURCE"),
    )
    for module, assignment_name in modules_and_assignments:
        module_source = inspect.getsource(module)
        tree = ast.parse(module_source)
        kernel_assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets)
        )
        shared_loads = [
            node
            for node in ast.walk(kernel_assignment.value)
            if isinstance(node, ast.Name) and node.id == "_BLEND_DEVICE_SOURCE" and isinstance(node.ctx, ast.Load)
        ]
        assert len(shared_loads) == 1


def test_blend_fragment_and_alpha_skeleton_expression_order_characterization() -> None:
    """REQ-TEST-003 characterization: freeze the current normalized fp32 blend and source-over expression order.
    Correctness is not independently established because the oracle is extracted from the current CUDA source rather
    than a numeric reference. Remove these pins when a specification-backed numeric oracle replaces them, or update
    them only for an intentional numeric-contract change that independently re-derives the expected expressions."""
    import pixtreme._composite.merge as composite_merge
    import pixtreme._draw.shapes as shapes
    import pixtreme._draw.text as text
    from pixtreme._core.blend import _BLEND_DEVICE_SOURCE

    assert _normalized_returns(_BLEND_DEVICE_SOURCE) == (
        "source",
        "fmaxf(background, source)",
        "background + source",
        "1.0f - (1.0f - background) * (1.0f - source)",
        "fminf(background, source)",
        "background * source",
        "fabsf(background - source)",
        "background <= 0.5f ? 2.0f * background * source : 1.0f - 2.0f * (1.0f - background) * (1.0f - source)",
        "source <= 0.5f ? 2.0f * background * source : 1.0f - 2.0f * (1.0f - background) * (1.0f - source)",
        "background - (1.0f - 2.0f * source) * background * (1.0f - background)",
        "background + (2.0f * source - 1.0f) * (d - background)",
    )
    assert _normalized_assignment(_BLEND_DEVICE_SOURCE, r"\bd") == (
        "background <= 0.25f ? ((16.0f * background - 12.0f) * background + 4.0f) * background : sqrtf(background)"
    )

    for draw_source in (shapes._DRAW_KERNEL_SOURCE, text._TEXT_COMPOSITE_KERNEL_SOURCE):
        assert _normalized_assignment(draw_source, r"\bblend_value") == (
            "pixtreme_blend(destination, source_color, blend)"
        )
        assert _normalized_assignment(draw_source, r"output\s*\[\s*output_offset\s*\+\s*channel\s*\]") == (
            "destination * (1.0f - alpha) + blend_value * alpha"
        )

    composite_source = composite_merge._COMPOSITE_KERNEL_SOURCE
    assert _normalized_assignment(composite_source, r"\bblend_value") == (
        "pixtreme_blend(background_color, source_color, blend)"
    )
    assert _normalized_assignment(composite_source, r"\bcomposite_source") == (
        "(1.0f - background_alpha) * source_color + background_alpha * blend_value"
    )
    assert _normalized_assignment(composite_source, r"\boutput_premultiplied") == (
        "effective_alpha * composite_source + background_alpha * (1.0f - effective_alpha) * background_color"
    )
