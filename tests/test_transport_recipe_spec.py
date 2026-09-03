"""Documentation and CUDA-ordering contracts for the hardware-codec recipe."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "docs_site" / "transport.md"
FEATURE = ROOT / "docs" / "features" / "v1-transport-recipe.md"


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _recipe_python_block(containing: str) -> ast.Module:
    recipe = RECIPE.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", recipe, flags=re.DOTALL)
    return ast.parse(next(block for block in blocks if containing in block))


def _if_chain(node: ast.If) -> tuple[list[ast.If], list[ast.stmt]]:
    branches = [node]
    while len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        node = node.orelse[0]
        branches.append(node)
    return branches, node.orelse


def test_transport_recipe_documents_decoder_routes_without_private_integration_names() -> None:
    """v1-transport-recipe acceptance 1 and 6: the public decoder recipe names only reusable boundaries."""
    recipe = RECIPE.read_text(encoding="utf-8")

    for text in ("px.io.from_nv12", "px.io.from_p010", "px.io.from_array", "P016", "P010"):
        assert text in recipe
    assert "not a substitute for" in recipe
    for private_name in ("pixtreme-transport", "pixtreme-infer", "streamflow"):
        assert private_name not in recipe


def test_transport_recipe_rejects_unproved_decoder_and_encoder_formats() -> None:
    """v1-transport-recipe acceptance 1 and 3: only proved raw-surface formats reach named boundaries."""
    decoder = _recipe_python_block("decoded.wait_on")
    decoder_route = next(
        node
        for node in ast.walk(decoder)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "decoded.format == 'nv12'"
    )
    decoder_branches, decoder_fallback = _if_chain(decoder_route)
    assert [ast.unparse(branch.test) for branch in decoder_branches] == [
        "decoded.format == 'nv12'",
        "decoded.format == 'p010'",
        "decoded.format == 'unpacked'",
    ]
    assert len(decoder_fallback) == 1
    assert isinstance(decoder_fallback[0], ast.Raise)

    encoder = _recipe_python_block("encoder.submit")
    encoder_route = next(
        node
        for node in ast.walk(encoder)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "output_signal.bit_depth == 8"
    )
    encoder_branches, encoder_fallback = _if_chain(encoder_route)
    assert [ast.unparse(branch.test) for branch in encoder_branches] == [
        "output_signal.bit_depth == 8",
        "output_signal.bit_depth == 10",
    ]
    assert len(encoder_fallback) == 1
    assert isinstance(encoder_fallback[0], ast.Raise)


def test_transport_recipe_documents_signal_mapping_and_encoder_round_trip() -> None:
    """v1-transport-recipe acceptance 2, 3, and 7: signal claims and the encode route stay explicit."""
    recipe = RECIPE.read_text(encoding="utf-8")
    normalized_recipe = " ".join(recipe.split())
    feature = FEATURE.read_text(encoding="utf-8")

    for text in (
        "matrix_coefficients",
        "chroma_sample_loc_type",
        "video_full_range_flag",
        "matrix=",
        "siting=",
        "range=",
        "px.color.rgb_to_ycbcr",
        "px.io.to_nv12",
        "px.io.to_p010",
        "Do not infer chroma siting from",
    ):
        assert text in recipe
    for instruction in (
        "Configure the encoder's VUI or container signal separately",
        "Matching function arguments alone cannot update encoded bitstream signalling",
    ):
        assert instruction in normalized_recipe
    for requirement in (
        "REQ-ARCH-002",
        "REQ-DEC-004",
        "REQ-DEC-005",
        "REQ-DEC-006",
        "REQ-DEC-007",
        "REQ-ENC-002",
        "REQ-ENC-003",
        "REQ-ENC-004",
    ):
        assert requirement in feature
    assert "gap" in feature
    assert "chroma sample location" in feature


def test_transport_recipe_stream_contract_matches_boundary_docstrings_and_source() -> None:
    """v1-transport-recipe acceptance 4 and 5: recipe, docstrings, and launch sites agree on ordering."""
    recipe = RECIPE.read_text(encoding="utf-8")
    for text in (
        "current CuPy stream",
        "consumer stream",
        "CUDA event",
        "stream synchronization",
        "does not perform an implicit host synchronization",
    ):
        assert text in recipe

    for function in (
        px.io.from_array,
        px.io.to_array,
        px.io.from_nv12,
        px.io.from_p010,
        px.io.to_nv12,
        px.io.to_p010,
    ):
        docstring = inspect.getdoc(function)
        assert docstring is not None
        assert "current CuPy stream" in docstring
        assert "host synchronization" in docstring

    sampling_path = ROOT / "src" / "pixtreme" / "_io" / "wire" / "sampling.py"
    for name in ("_from_subsampled", "_to_subsampled"):
        function = _function(sampling_path, name)
        kernel_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "kernel"
        ]
        assert len(kernel_calls) == 1
        assert all(keyword.arg != "stream" for keyword in kernel_calls[0].keywords)
        assert not any(isinstance(node, ast.Attribute) and node.attr == "synchronize" for node in ast.walk(function))


def test_from_array_passes_the_current_consumer_stream_to_dlpack() -> None:
    """v1-transport-recipe acceptance 4: DLPack import performs the producer-consumer stream handshake."""
    import cupy as cp

    class RecordingProducer:
        def __init__(self, array: cp.ndarray) -> None:
            self.array = array
            self.streams: list[int | None] = []

        def __dlpack_device__(self) -> tuple[int, int]:
            return self.array.__dlpack_device__()

        def __dlpack__(self, *, stream: int | None = None) -> object:
            self.streams.append(stream)
            return self.array.__dlpack__(stream=stream)

    source = cp.zeros((2, 2, 3), dtype=cp.float32)
    producer = RecordingProducer(source)
    consumer = cp.cuda.Stream(non_blocking=True)

    with consumer:
        frame = px.io.from_array(producer, colorspace="sRGB", gamma="sRGB", channels="RGB")

    assert producer.streams == [consumer.ptr]
    assert frame.data.data.ptr == source.data.ptr
