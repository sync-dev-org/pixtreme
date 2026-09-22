"""Source documentation and real-GPU execution contracts for P216, independent of helper names."""

from __future__ import annotations

import inspect
import re

import cupy as cp
import numpy as np
import pytest
from p216_oracle import FROM_FILTERS, TO_FILTERS, asymmetric_codes, from_reference, to_reference
from test_to_format_spec import _frame

import pixtreme as px


@pytest.mark.parametrize("direction", ("from", "to"))
def test_p216_docstrings_expose_the_full_source_contract(direction: str) -> None:
    """v1-p216-wire-format acceptance 16: structural source-docstring contract for every invisible boundary obligation.

    Check contract vocabulary and equations, not a particular sentence or private module layout.
    Human review still establishes the meaning and completeness of the prose.
    """
    function = getattr(px.io, f"{direction}_p216")
    doc = " ".join((inspect.getdoc(function) or "").lower().replace("`", "").split())
    patterns = {
        "layout": r"y.{0,100}interleav.{0,60}cb.{0,20}cr|y.{0,100}cb.{0,20}cr.{0,60}interleav",
        "shape": r"2\s*\*\s*(?:w(?:idth)?\s*\*\s*h(?:eight)?|h(?:eight)?\s*\*\s*w(?:idth)?)",
        "dtype": r"uint16",
        "working dtype": r"float32|fp32",
        "active bits": r"16.{0,20}(?:bit|effective)|(?:bit|effective).{0,20}16",
        "contiguity": r"c-contiguous",
        "legal and full": r"legal.*full|full.*legal",
        "luma extent": r"56064",
        "legal offset": r"4096",
        "chroma extent": r"57344",
        "chroma center": r"32768",
        "full extent": r"65535",
        "rounding": r"half.{0,12}away.{0,12}zero",
        "container clip": r"clip.{0,100}(?:65535|container)|(?:65535|container).{0,100}clip",
        "co-siting": r"co-sited",
        "even width": r"even.{0,25}width|width.{0,25}even",
        "vertical full resolution": r"vertical.{0,45}(?:full|1:1|same row|no|unfilter)",
        "filter default": r"default.{0,60}" + ("bilinear" if direction == "from" else "area"),
        "constant chroma guarantee": r"constant.{0,40}chroma|chroma.{0,40}constant",
        "nearest guarantee": r"nearest.{0,100}(?:round.trip|bit)|(?:round.trip|bit).{0,100}nearest",
        "general subsampling loss": r"(?:general|arbitrary).{0,100}(?:loss|not|non|no |irrevers)",
        "metadata": r"colorspace.*gamma.*matrix",
        "current stream": r"current.{0,30}(?:cupy|cp).{0,30}stream|current.{0,30}stream",
        "asynchronous host": r"asynchron|host.{0,30}(?:no|without).{0,30}synchron|no.{0,30}host.{0,30}synchron",
        "producer ordering": r"producer.{0,100}(?:order|stream|event|wait)|(?:order|event|wait).{0,100}producer",
        "consumer ordering": r"consumer.{0,100}(?:order|stream|event|wait)|(?:order|event|wait).{0,100}consumer",
        "input lifetime": r"(?:input|buf|frame).{0,100}(?:alive|lifetime|valid until|completion)",
        "output ownership": r"(?:return|output|result|frame).{0,100}(?:own|private|independent|new alloc)",
    }
    for axis, pattern in patterns.items():
        assert re.search(pattern, doc), f"{function.__name__}: missing {axis} contract"
    for token in FROM_FILTERS if direction == "from" else TO_FILTERS:
        assert token in doc, f"{function.__name__}: missing {token} subset member"


class _AllocationTrace(cp.cuda.MemoryHook):
    def __init__(self) -> None:
        self.sizes: list[int] = []

    def malloc_preprocess(self, **kwargs: int) -> None:
        self.sizes.append(kwargs["size"])


@pytest.mark.parametrize(
    "direction,interpolation", [("from", token) for token in FROM_FILTERS] + [("to", token) for token in TO_FILTERS]
)
def test_p216_uses_one_kernel_and_one_output_allocation_on_the_current_stream(
    direction: str, interpolation: str
) -> None:
    """v1-p216-wire-format acceptance 9: real CUDA graph and allocator trace detect extra passes, buffers and host copies.

    Structural execution contract: warm compilation first, then capture on a non-default stream.
    A single KERNEL node (no memcpy/host node), one output-sized allocation, and oracle output
    bind the public operation without naming factories, private helpers or kernel entry points.
    Stream capture also rejects synchronous device-to-host reads. No timing threshold is used.
    """
    function = getattr(px.io, f"{direction}_p216")
    codes = asymmetric_codes()
    values = from_reference(codes, 3, 6, "full", "nearest").astype(np.float32)
    source = cp.asarray(codes) if direction == "from" else _frame(values)
    kwargs = {"interpolation": interpolation, "range": "full"}
    if direction == "from":
        kwargs.update(width=6, height=3)
    warm = function(source, **kwargs)
    cp.cuda.get_current_stream().synchronize()
    # Keep the warm result alive so output allocation cannot silently reuse its live storage.
    warm_data = warm.data if direction == "from" else warm
    expected_bytes = 3 * 6 * (12 if direction == "from" else 4)
    trace = _AllocationTrace()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        try:
            with trace:
                result = function(source, **kwargs)
        finally:
            graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    dot = graph.debug_dot_str(cp.cuda.runtime.cudaGraphDebugDotFlagsVerbose)
    node_types = re.findall(r'label="\{([A-Z][A-Z_ ]*)\n', dot)
    assert node_types == ["KERNEL"], dot
    assert trace.sizes == [expected_bytes], trace.sizes
    result_data = result.data if direction == "from" else result
    assert result_data.data.ptr != warm_data.data.ptr
    if direction == "from":
        np.testing.assert_allclose(
            result_data.get(), from_reference(codes, 3, 6, "full", interpolation), rtol=0, atol=3e-6
        )
    else:
        expected, q64 = to_reference(values, "full", interpolation)
        actual = result_data.get()
        differences = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
        assert differences.max() <= 1
        changed = differences != 0
        boundary = np.clip(np.floor(q64), 0, 65534) + 0.5
        assert np.all(np.abs(q64[changed] - boundary[changed]) <= 0.125)


@pytest.mark.parametrize("direction", ("from", "to"))
def test_p216_preserves_the_cause_if_a_backend_failure_is_converted(direction: str) -> None:
    """v1-p216-wire-format acceptance 17: an allocator boundary failure is propagated or retained as the explicit cause."""
    function = getattr(px.io, f"{direction}_p216")
    source = cp.asarray(asymmetric_codes()) if direction == "from" else _frame(np.zeros((3, 6, 3), dtype=np.float32))
    kwargs = {"width": 6, "height": 3} if direction == "from" else {}
    failure = RuntimeError("P216 test allocator failure")

    def failed_allocation(size: int) -> cp.cuda.MemoryPointer:
        raise failure

    with cp.cuda.using_allocator(failed_allocation), pytest.raises(Exception) as error:
        function(source, **kwargs)
    if error.value is not failure:
        assert error.value.__cause__ is failure
        assert re.fullmatch(r"why=.+; what=.+; how=.+", str(error.value))
