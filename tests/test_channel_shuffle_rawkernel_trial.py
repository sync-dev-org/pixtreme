"""Characterization tests for the channel shuffle RawKernel trial."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cupy as cp
import numpy as np
import pytest

import pixtreme as px


def _frame(
    values: Any,
    *,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    channels: tuple[str, ...] = ("R", "G", "B"),
) -> px.core.Frame:
    return px.io.from_array(
        cp.asarray(values, dtype=cp.float32),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
    )


def _legacy_route_data(
    sources: Sequence[cp.ndarray],
    routes: Sequence[tuple[int, int] | np.float32],
) -> cp.ndarray:
    """Retain the pre-trial slice copy / fill composition as an independent oracle."""
    height, width = sources[0].shape[:2]
    output = cp.empty((height, width, len(routes)), dtype=cp.float32)
    for output_index, route in enumerate(routes):
        if isinstance(route, tuple):
            source_index, channel_index = route
            output[..., output_index] = sources[source_index][..., channel_index]
        else:
            output[..., output_index].fill(route)
    return output


def _assert_bit_equal(actual: cp.ndarray, expected: cp.ndarray) -> None:
    cp.testing.assert_array_equal(actual.view(cp.uint32), expected.view(cp.uint32))


@pytest.mark.req("REQ-PIX-001")
@pytest.mark.req("REQ-PIX-003")
def test_shuffle_rawkernel_trial_matches_legacy_reorder_multi_fill_and_adapt_bits_characterization() -> None:
    """characterization: Channel shuffle keeps bit-exact reorder, multi-Frame fill, and adaptation results."""
    first_bits = np.asarray(
        [
            [[0x80000000, 0x3F000000, 0x7FC00001], [0xBF800000, 0x3F800000, 0x7FC01234]],
            [[0x00000000, 0x40000000, 0x40400000], [0x3E800000, 0xC0200000, 0x40800000]],
        ],
        dtype=np.uint32,
    )
    second_bits = np.asarray(
        [
            [[0x3DCCCCCD, 0x3E4CCCCD, 0x3E99999A], [0x3ECCCCCD, 0x3F000000, 0x3F19999A]],
            [[0x3F333333, 0x3F4CCCCD, 0x3F666666], [0x3F800000, 0x3F8CCCCD, 0x3F99999A]],
        ],
        dtype=np.uint32,
    )
    first = _frame(first_bits.view(np.float32))
    second = _frame(second_bits.view(np.float32))

    reordered = px.channel.shuffle(B=(first, "B"), G=(first, "G"), R=(first, "R"))
    _assert_bit_equal(reordered.data, _legacy_route_data((first.data,), ((0, 2), (0, 1), (0, 0))))

    fill = np.float32(-1.25)
    combined = px.channel.shuffle(R=(first, "R"), G=(first, "G"), B=(second, "B"), A=fill)
    _assert_bit_equal(
        combined.data,
        _legacy_route_data((first.data, second.data), ((0, 0), (0, 1), (1, 2), fill)),
    )

    encoded = _frame(
        np.asarray(
            [
                [[0.02, 0.30, 0.90], [0.80, 0.10, 0.04]],
                [[0.12, 0.40, 0.70], [0.60, 0.20, 0.08]],
            ],
            dtype=np.float32,
        ),
        colorspace="sRGB",
        gamma="sRGB",
    )
    adapted = px.color.rgb_to_rgb(encoded, output_colorspace=first.colorspace, output_gamma=first.gamma)
    adapt_result = px.channel.shuffle(adapt=True, R=(first, "R"), G=(encoded, "G"), B=(encoded, "B"))
    _assert_bit_equal(
        adapt_result.data,
        _legacy_route_data((first.data, adapted.data), ((0, 0), (1, 1), (1, 2))),
    )
