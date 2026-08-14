"""Internal plumbing structure contracts."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from pixtreme._channel import shuffle
from pixtreme._filter import common as filter_common
from pixtreme._io.wire import sampling, yuv444p, yuva444p


def test_dead_internal_parameters_and_cache_partitions_are_absent() -> None:
    """REQ-TEST-002 and REQ-TEST-003: intentional structure contract asserting only the absence of
    removed dead internal plumbing; surviving parameter names and order remain free to
    evolve without touching this contract."""
    absent_parameters = {
        shuffle._mismatch_error: {"adapt"},
        sampling._from_subsampled: {"interpolation", "siting"},
        sampling._to_subsampled: {"interpolation", "siting"},
        sampling._to_planar_444_kernel_source: {"bit_depth"},
    }
    for function, removed in absent_parameters.items():
        parameters = set(inspect.signature(function).parameters)
        regained = sorted(parameters & removed)
        assert not regained, f"{function.__qualname__} regained dead parameters: {regained}"

    # The planar 444 export kernel source does not vary by bit depth: the factories are
    # argument-free so a single cached kernel instance serves all bit depths per module.
    assert tuple(inspect.signature(yuv444p._to_kernel).parameters) == ()
    assert tuple(inspect.signature(yuva444p._to_kernel).parameters) == ()


def test_separable_axis_launchers_share_planning_but_keep_named_abi_wrappers() -> None:
    """REQ-TEST-001 and REQ-TEST-003: shared planning selects execution at the memory limit and forwards the ABI."""
    calls: list[tuple[str, tuple[int, ...], tuple[int, int], tuple[object, ...], int]] = []

    def shared_kernel(
        grid: tuple[int, ...],
        block: tuple[int, int],
        arguments: tuple[object, ...],
        *,
        shared_mem: int,
    ) -> None:
        calls.append(("shared", grid, block, arguments, shared_mem))

    def global_kernel(
        grid: tuple[int, ...],
        block: tuple[int, int],
        arguments: tuple[object, ...],
        *,
        shared_mem: int,
    ) -> None:
        calls.append(("global", grid, block, arguments, shared_mem))

    def build_arguments(
        shape_arguments: tuple[np.int64, np.int64, np.int64],
        radius: np.int64,
        border: np.int32,
        border_value: np.float32,
    ) -> tuple[object, ...]:
        return (*shape_arguments, radius, border, border_value)

    frame = SimpleNamespace(width=10, height=5, channels=("R", "G", "B"))
    for radius in (250, 251):
        filter_common._launch_separable_axis(
            shared_kernel,
            global_kernel,
            frame=frame,
            radius=radius,
            border="mirror",
            border_value=0.25,
            horizontal=True,
            argument_builder=build_arguments,
        )

    expected_prefix = ((1, 1), (32, 8))
    assert calls == [
        (
            "shared",
            *expected_prefix,
            (np.int64(10), np.int64(5), np.int64(3), np.int64(250), np.int32(0), np.float32(0.25)),
            49_024,
        ),
        (
            "global",
            *expected_prefix,
            (np.int64(10), np.int64(5), np.int64(3), np.int64(251), np.int32(0), np.float32(0.25)),
            0,
        ),
    ]

    box_source = inspect.getsource(filter_common._launch_box_axis)
    gaussian_source = inspect.getsource(filter_common._launch_gaussian_axis)
    assert "_launch_separable_axis(" in box_source
    assert "np.float32(scale)" in box_source
    assert "_launch_separable_axis(" in gaussian_source
    assert "weights" in gaussian_source
