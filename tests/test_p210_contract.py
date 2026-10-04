"""Source documentation and real-GPU execution contracts for P210, independent of helper names."""

from __future__ import annotations

import inspect
import re

import cupy as cp
import numpy as np
import pytest
from p210_oracle import FROM_FILTERS, TO_FILTERS, assert_near_tie_codes, asymmetric_words, from_reference, to_reference
from test_to_format_spec import _frame

import pixtreme as px


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize("direction", ("from", "to"))
def test_p210_docstrings_expose_the_full_source_contract(direction: str) -> None:
    """The public P210 conversion documentation states its buffer layout, range mapping, sampling, and error
    contracts.
    """
    function = getattr(px.io, f"{direction}_p210")
    doc = " ".join((inspect.getdoc(function) or "").lower().replace("`", "").split())
    patterns = {
        "layout": r"y.{0,100}interleav.{0,60}cb.{0,20}cr|y.{0,100}cb.{0,20}cr.{0,60}interleav",
        "shape": r"2\s*\*\s*(?:w(?:idth)?\s*\*\s*h(?:eight)?|h(?:eight)?\s*\*\s*w(?:idth)?)",
        "dtype": r"uint16",
        "working dtype": r"float32|fp32",
        "active bits": r"10.{0,20}(?:bit|effective)|(?:bit|effective).{0,20}10",
        "MSB alignment": r"msb|most significant|upper 10",
        "padding ignored or zeroed": r"(?:padding|lower 6).{0,100}(?:ignor|zero)|(?:ignor|zero).{0,100}(?:padding|lower 6)",
        "contiguity": r"c-contiguous",
        "legal and full": r"legal.*full|full.*legal",
        "luma extent": r"876",
        "legal offset": r"64",
        "chroma extent": r"896",
        "chroma center": r"512",
        "full extent": r"1023",
        "rounding": r"half.{0,12}away.{0,12}zero",
        "code clip": r"clip.{0,100}1023|1023.{0,100}clip",
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


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-018")
@pytest.mark.parametrize(
    "direction,interpolation", [("from", token) for token in FROM_FILTERS] + [("to", token) for token in TO_FILTERS]
)
def test_p210_uses_one_kernel_and_one_output_allocation_on_the_current_stream(
    direction: str, interpolation: str
) -> None:
    """P210 conversion performs one GPU kernel pass and one output allocation on the current stream."""
    function = getattr(px.io, f"{direction}_p210")
    codes = asymmetric_words()
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
        assert_near_tie_codes(actual, expected, q64)


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-017")
@pytest.mark.parametrize("direction", ("from", "to"))
def test_p210_preserves_the_cause_if_a_backend_failure_is_converted(direction: str) -> None:
    """P210 conversion propagates backend allocation errors or retains them as the cause of a public error."""
    function = getattr(px.io, f"{direction}_p210")
    source = cp.asarray(asymmetric_words()) if direction == "from" else _frame(np.zeros((3, 6, 3), dtype=np.float32))
    kwargs = {"width": 6, "height": 3} if direction == "from" else {}
    failure = RuntimeError("P210 test allocator failure")

    def failed_allocation(size: int) -> cp.cuda.MemoryPointer:
        raise failure

    with cp.cuda.using_allocator(failed_allocation), pytest.raises(Exception) as error:
        function(source, **kwargs)
    if error.value is not failure:
        assert error.value.__cause__ is failure
        assert re.fullmatch(r"why=.+; what=.+; how=.+", str(error.value))


@pytest.mark.req("REQ-PIX-009")
@pytest.mark.req("REQ-PIX-018")
@pytest.mark.parametrize("direction", ("from", "to"))
def test_p210_nondefault_producer_and_waiting_consumer_observe_ordered_private_result(direction: str) -> None:
    """P210 conversion respects a nondefault producer stream and returns private data to a waiting consumer stream."""
    codes = asymmetric_words()
    values = from_reference(codes, 3, 6, "full", "nearest").astype(np.float32)
    if direction == "from":
        source_data = cp.empty_like(cp.asarray(codes))
        pending = cp.asarray(codes)
        source = source_data
        expected = from_reference(codes, 3, 6, "full", "nearest")
    else:
        source = _frame(np.zeros_like(values))
        source_data = source.data
        pending = cp.asarray(values)
        expected, _ = to_reference(values, "full", "nearest")
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    finished = cp.cuda.Event()
    with producer:
        cp.copyto(source_data, pending)
        result = (
            px.io.from_p210(source, width=6, height=3, range="full", interpolation="nearest")
            if direction == "from"
            else px.io.to_p210(source, range="full", interpolation="nearest")
        )
        finished.record(producer)
    result_data = result.data if direction == "from" else result
    with consumer:
        consumer.wait_event(finished)
        observed = result_data.copy()
    consumer.synchronize()
    assert result_data.data.ptr != source_data.data.ptr
    np.testing.assert_array_equal(source_data.get(), codes if direction == "from" else values)
    if direction == "from":
        np.testing.assert_allclose(observed.get(), expected, rtol=0, atol=2e-7)
    else:
        assert_near_tie_codes(observed.get(), expected, to_reference(values, "full", "nearest")[1])
    source_data.fill(0)
    np.testing.assert_array_equal(result_data.get(), observed.get())
