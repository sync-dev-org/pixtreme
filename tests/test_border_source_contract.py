"""Contracts for consumers of the shared CUDA border substrate."""

from __future__ import annotations

import ast
import inspect
import re

from pixtreme._core.border import _BORDER_PREAMBLE


def test_warp_affine_kernel_source_dataflow_composes_canonical_border_preamble() -> None:
    """REQ-TEST-001 and REQ-TEST-003 bind warp kernel construction to the canonical border preamble while warp owns
    coordinate geometry; pre-consolidation warp helper names remain absent as a scoped legacy regression."""
    import pixtreme._transform.warp_affine as warp_affine

    module_source = inspect.getsource(warp_affine)
    tree = ast.parse(module_source)
    kernel_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_WARP_AFFINE_KERNEL_SOURCE" for target in node.targets)
    )

    border_loads = [
        node
        for node in ast.walk(kernel_assignment.value)
        if isinstance(node, ast.Name) and node.id == "_BORDER_PREAMBLE" and isinstance(node.ctx, ast.Load)
    ]
    assert len(border_loads) == 1
    assert warp_affine._WARP_AFFINE_KERNEL_SOURCE.count(_BORDER_PREAMBLE) == 1
    assert warp_affine._WARP_AFFINE_KERNEL_SOURCE.count("__device__ long long pixtreme_positive_modulo(") == 1
    assert warp_affine._WARP_AFFINE_KERNEL_SOURCE.count("__device__ long long pixtreme_border_index(") == 1
    assert warp_affine._WARP_AFFINE_KERNEL_SOURCE.count("__device__ float pixtreme_border_sample(") == 1
    consumer_source = warp_affine._WARP_AFFINE_KERNEL_SOURCE.replace(_BORDER_PREAMBLE, "", 1)
    border_calls = set(re.findall(r"\b(pixtreme_(?:positive_modulo|border_index|border_sample))\s*\(", consumer_source))
    assert border_calls == {"pixtreme_border_sample"}
    legacy_calls = {"pixtreme_warp_positive_modulo", "pixtreme_warp_border_index", "pixtreme_warp_sample"}
    assert not legacy_calls & set(re.findall(r"\b(pixtreme_\w+)\s*\(", consumer_source))
