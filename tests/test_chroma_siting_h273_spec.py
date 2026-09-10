"""Specification and compatibility tests for all H.273 4:2:0 chroma locations."""

from __future__ import annotations

import hashlib
import inspect
from typing import get_args

import numpy as np
import pytest

import pixtreme as px

SITING_OFFSETS = {
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "topleft": (0.0, 0.0),
    "top": (0.5, 0.0),
    "bottomleft": (0.0, 1.0),
    "bottom": (0.5, 1.0),
}


def _frame(values: np.ndarray) -> px.core.Frame:
    import cupy as cp

    return px.core.Frame(
        data=cp.asarray(np.ascontiguousarray(values)),
        colorspace="Rec.709",
        gamma="Rec.709",
        channels=("Y", "Cb", "Cr"),
    )


def _characterization_from_inputs() -> dict[str, object]:
    import cupy as cp

    height, width = 4, 6
    y8 = ((np.arange(height * width, dtype=np.uint16) * 7 + 11) % 256).astype(np.uint8)
    cb8 = np.asarray((17, 59, 101, 149, 197, 239), dtype=np.uint8)
    cr8 = np.asarray((241, 199, 151, 103, 61, 19), dtype=np.uint8)
    uv8 = np.stack((cb8, cr8), axis=1).reshape(-1)
    p010_uv = np.stack((cb8.astype(np.uint16) * 4, cr8.astype(np.uint16) * 4), axis=1).reshape(-1)
    return {
        "from_nv12": cp.asarray(np.concatenate((y8, uv8))),
        "from_p010": cp.asarray(np.concatenate((y8.astype(np.uint16) * 4, p010_uv)) << 6),
        "from_yuv420p": cp.asarray(np.concatenate((y8, cb8, cr8))),
    }


def _characterization_frame() -> px.core.Frame:
    height, width = 4, 6
    yy, xx = np.indices((height, width), dtype=np.float32)
    values = np.stack(
        (
            np.mod(xx * 17 + yy * 29 + 11, 251) / np.float32(255),
            np.mod(xx * 43 + yy * 13 + 37, 251) / np.float32(255),
            np.mod(xx * 7 + yy * 53 + 83, 251) / np.float32(255),
        ),
        axis=2,
    ).astype(np.float32)
    return _frame(values)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def test_chroma_siting_literal_and_normalization_follow_the_h273_table() -> None:
    """v1-chroma-siting-h273 acceptance 1 and 2: the independent six-position table drives the closed token."""
    from pixtreme._core.validation import _normalized_closed_token

    assert get_args(px.core.ChromaSiting) == tuple(SITING_OFFSETS)
    variants = {
        "left": ("LEFT",),
        "center": ("Center",),
        "topleft": ("TOP-LEFT", "Top Left", "top_left", "top.left"),
        "top": ("TOP",),
        "bottomleft": ("BOTTOM-LEFT", "Bottom Left", "bottom_left", "bottom.left"),
        "bottom": ("BOTTOM",),
    }
    for canonical, accepted_variants in variants.items():
        for variant in accepted_variants:
            assert _normalized_closed_token(variant, axis="siting", accepted=tuple(SITING_OFFSETS)) == canonical
    keys = tuple(
        token.replace(" ", "").replace(".", "").replace("-", "").replace("_", "").casefold() for token in SITING_OFFSETS
    )
    assert len(keys) == len(set(keys)) == 6


def test_six_420_functions_retain_their_static_signatures() -> None:
    """v1-chroma-siting-h273 acceptance 3: only the ChromaSiting Literal value set expands."""
    expected = {
        "from_nv12": (
            ("buf", "width", "height", "colorspace", "gamma", "matrix", "range", "siting", "interpolation"),
            {"range": "legal", "siting": "left", "interpolation": "bilinear"},
        ),
        "from_p010": (
            ("buf", "width", "height", "colorspace", "gamma", "matrix", "range", "siting", "interpolation"),
            {"range": "legal", "siting": "left", "interpolation": "bilinear"},
        ),
        "from_yuv420p": (
            (
                "buf",
                "width",
                "height",
                "bit_depth",
                "colorspace",
                "gamma",
                "matrix",
                "range",
                "siting",
                "interpolation",
            ),
            {"bit_depth": 8, "range": "legal", "siting": "left", "interpolation": "bilinear"},
        ),
        "to_nv12": (
            ("frame", "range", "siting", "interpolation"),
            {"range": "legal", "siting": "left", "interpolation": "area"},
        ),
        "to_p010": (
            ("frame", "range", "siting", "interpolation"),
            {"range": "legal", "siting": "left", "interpolation": "area"},
        ),
        "to_yuv420p": (
            ("frame", "bit_depth", "range", "siting", "interpolation"),
            {"bit_depth": 8, "range": "legal", "siting": "left", "interpolation": "area"},
        ),
    }
    for name, (parameter_names, defaults) in expected.items():
        parameters = inspect.signature(getattr(px.io, name)).parameters
        assert tuple(parameters) == parameter_names
        assert parameters["siting"].annotation == "ChromaSiting"
        for parameter_name, default in defaults.items():
            assert parameters[parameter_name].default == default


@pytest.mark.parametrize("rejected", ("diagonal", 3, ("top", "bottom"), "", " .-_ "))
def test_420_functions_reject_non_singular_or_unknown_siting_before_pixels(rejected: object) -> None:
    """v1-chroma-siting-h273 acceptance 9: all six boundaries fail fast with raw actionable errors."""
    import cupy as cp

    from_cases = {
        "from_nv12": cp.zeros(6, dtype=cp.uint8),
        "from_p010": cp.zeros(6, dtype=cp.uint16),
        "from_yuv420p": cp.zeros(6, dtype=cp.uint8),
    }
    values = np.zeros((2, 2, 3), dtype=np.float32)
    for name, source in from_cases.items():
        with pytest.raises(ValueError) as error:
            getattr(px.io, name)(source, width=2, height=2, siting=rejected)
        message = str(error.value)
        assert message.index("why=") < message.index("what=") < message.index("how=")
        assert repr(rejected) in message
        assert repr(tuple(SITING_OFFSETS)) in message
    for name in ("to_nv12", "to_p010", "to_yuv420p"):
        with pytest.raises(ValueError) as error:
            getattr(px.io, name)(_frame(values), siting=rejected)
        message = str(error.value)
        assert message.index("why=") < message.index("what=") < message.index("how=")
        assert repr(rejected) in message
        assert repr(tuple(SITING_OFFSETS)) in message


def test_existing_three_siting_paths_remain_bit_exact_characterization() -> None:
    """characterization: freezes base d16ccc7 output because numerical correctness belongs to independent oracles.

    v1-chroma-siting-h273 acceptance 8: retire only if a later specification intentionally breaks 4:2:0 compatibility.
    The digests were generated from base commit d16ccc7 with this test's fixed inputs, full range, and bicubic filter.
    """
    expected = {
        "from_nv12:omitted": "5f24d5b0ce862d3bc7d9d11a6071d83ed7b2637adb3439135a0d84dc52b8bf0d",
        "from_nv12:left": "5f24d5b0ce862d3bc7d9d11a6071d83ed7b2637adb3439135a0d84dc52b8bf0d",
        "from_nv12:center": "2317883a6661b397899d4984473c3f4a945c7f06f0c996da1dd0ded9678822ac",
        "from_nv12:topleft": "269c659c4370f7627ff04873a7b596d09320652b19e027f58013f449ed4f4615",
        "from_p010:omitted": "6e3d416283d0536ec4d2e09f3f4092580df3867ac2b79d0a2aba2cfb0c58be0e",
        "from_p010:left": "6e3d416283d0536ec4d2e09f3f4092580df3867ac2b79d0a2aba2cfb0c58be0e",
        "from_p010:center": "40b1b8e45da87c07c8c5ea3b445508cc73e0062984c1bcc44f6284de7828b021",
        "from_p010:topleft": "824fb95c180c484a2f91ce1c7d7eb431fb331eb08cbc7ece6d8e8e9c5ac62d8e",
        "from_yuv420p:omitted": "5f24d5b0ce862d3bc7d9d11a6071d83ed7b2637adb3439135a0d84dc52b8bf0d",
        "from_yuv420p:left": "5f24d5b0ce862d3bc7d9d11a6071d83ed7b2637adb3439135a0d84dc52b8bf0d",
        "from_yuv420p:center": "2317883a6661b397899d4984473c3f4a945c7f06f0c996da1dd0ded9678822ac",
        "from_yuv420p:topleft": "269c659c4370f7627ff04873a7b596d09320652b19e027f58013f449ed4f4615",
        "to_nv12:omitted": "5f1c097ba91a3f7d91c3fc059d5d128b728e5d9f894cb70f85f1d9369457f166",
        "to_nv12:left": "5f1c097ba91a3f7d91c3fc059d5d128b728e5d9f894cb70f85f1d9369457f166",
        "to_nv12:center": "7a8559bb2844611271a3c099c94c4c1f1cc5991174fd437553a60bd9153e3399",
        "to_nv12:topleft": "70a100446f269fb39b3fad351ea296869f7e098a3d5c3715a5d323dc54cf9c0e",
        "to_p010:omitted": "3326c56610ecc9afaad3a620efe7b1eb5817e3272ff8d6601bf30d08777b21b7",
        "to_p010:left": "3326c56610ecc9afaad3a620efe7b1eb5817e3272ff8d6601bf30d08777b21b7",
        "to_p010:center": "2458e45a8e20739030808ac873d54d8196a669b53032ba725502af5f4c3e5ab6",
        "to_p010:topleft": "ee77154f9f2516ca106153b8fa35586994fc7682777bdd2bbb14beeef4cd742d",
        "to_yuv420p:omitted": "0b309eabb19938e4309289c795ea85c32be9d9d8c22363ba80a9d340eed71895",
        "to_yuv420p:left": "0b309eabb19938e4309289c795ea85c32be9d9d8c22363ba80a9d340eed71895",
        "to_yuv420p:center": "c033c10afb00e43607f0de69f2675ca799e25a724fb768c768134c8e58945b3c",
        "to_yuv420p:topleft": "0a6ff9627ca2d0d4da9efe4f0a0ac6127dcb6426e89b556b7817d226b5792db6",
    }
    actual: dict[str, str] = {}
    for name, source in _characterization_from_inputs().items():
        function = getattr(px.io, name)
        for siting in (None, "left", "center", "topleft"):
            kwargs = {"width": 6, "height": 4, "range": "full", "interpolation": "bicubic"}
            if siting is not None:
                kwargs["siting"] = siting
            actual[f"{name}:{siting or 'omitted'}"] = _digest(function(source, **kwargs).data.get())

    frame = _characterization_frame()
    for name in ("to_nv12", "to_p010", "to_yuv420p"):
        function = getattr(px.io, name)
        for siting in (None, "left", "center", "topleft"):
            kwargs = {"range": "full", "interpolation": "bicubic"}
            if siting is not None:
                kwargs["siting"] = siting
            actual[f"{name}:{siting or 'omitted'}"] = _digest(function(frame, **kwargs).get())

    assert actual == expected


def test_six_420_docstrings_explain_the_single_progressive_frame_offset_contract() -> None:
    """v1-chroma-siting-h273 acceptance 12: all six public boundaries expose the same LLM-readable contract."""
    h273_mapping = (
        (0, "left", "(0, 0.5)"),
        (1, "center", "(0.5, 0.5)"),
        (2, "topleft", "(0, 0)"),
        (3, "top", "(0.5, 0)"),
        (4, "bottomleft", "(0, 1)"),
        (5, "bottom", "(0.5, 1)"),
    )
    contract_fragments = (
        "H.273 mappings are",
        "default is ``left``",
        "import and export use the same frame offset",
        "progressive frame",
        "does not interpret field signalling",
    )
    for name in ("from_nv12", "from_p010", "from_yuv420p", "to_nv12", "to_p010", "to_yuv420p"):
        docstring = inspect.getdoc(getattr(px.io, name))
        assert docstring is not None
        normalized = " ".join(docstring.split())
        for type_number, token, offset in h273_mapping:
            fragment = f"type {type_number} ``{token}={offset}``"
            assert fragment in normalized, (name, fragment)
        for fragment in contract_fragments:
            assert fragment in normalized, (name, fragment)
