"""Specification tests for S-Log3 and ARRI-LogC4 signed-domain extensions."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

import cupy as cp
import numpy as np
import pytest

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]

_SLOG3_CUT = np.float64(0.01125)
_SLOG3_CODE_CUT = np.float64(171.2102946929) / np.float64(1023.0)

_LOGC4_A = (np.float64(2.0) ** np.float64(18.0) - np.float64(16.0)) / np.float64(117.45)
_LOGC4_B = (np.float64(1023.0) - np.float64(95.0)) / np.float64(1023.0)
_LOGC4_C = np.float64(95.0) / np.float64(1023.0)
_LOGC4_S = (
    np.float64(7.0)
    * np.log(np.float64(2.0))
    * np.float64(2.0) ** (np.float64(7.0) - np.float64(14.0) * _LOGC4_C / _LOGC4_B)
) / (_LOGC4_A * _LOGC4_B)
_LOGC4_T = (
    np.float64(2.0) ** (np.float64(14.0) * (-_LOGC4_C / _LOGC4_B) + np.float64(6.0)) - np.float64(64.0)
) / _LOGC4_A


def _piecewise(
    values: np.ndarray, cut: float, lower: Callable[[np.ndarray], np.ndarray], upper: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    lower_mask = values < cut
    result[lower_mask] = lower(values[lower_mask])
    result[~lower_mask] = upper(values[~lower_mask])
    return result


def _slog3_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        _SLOG3_CUT,
        lambda x: (
            (x * (np.float64(171.2102946929) - np.float64(95.0)) / _SLOG3_CUT + np.float64(95.0)) / np.float64(1023.0)
        ),
        lambda x: (
            (np.float64(420.0) + np.log10((x + np.float64(0.01)) / np.float64(0.19)) * np.float64(261.5))
            / np.float64(1023.0)
        ),
    )


def _slog3_decode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        _SLOG3_CODE_CUT,
        lambda e: (
            (e * np.float64(1023.0) - np.float64(95.0)) * _SLOG3_CUT / (np.float64(171.2102946929) - np.float64(95.0))
        ),
        lambda e: (
            np.float64(10.0) ** ((e * np.float64(1023.0) - np.float64(420.0)) / np.float64(261.5)) * np.float64(0.19)
            - np.float64(0.01)
        ),
    )


def _logc4_encode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        _LOGC4_T,
        lambda x: (x - _LOGC4_T) / _LOGC4_S,
        lambda x: (
            ((np.log2(_LOGC4_A * x + np.float64(64.0)) - np.float64(6.0)) / np.float64(14.0)) * _LOGC4_B + _LOGC4_C
        ),
    )


def _logc4_decode(values: np.ndarray) -> np.ndarray:
    return _piecewise(
        values,
        np.float64(0.0),
        lambda e: e * _LOGC4_S + _LOGC4_T,
        lambda e: (
            (np.float64(2.0) ** (np.float64(14.0) * (e - _LOGC4_C) / _LOGC4_B + np.float64(6.0)) - np.float64(64.0))
            / _LOGC4_A
        ),
    )


def _frame(
    values: np.ndarray,
    *,
    gamma: str = "linear",
    auxiliary: bool = False,
) -> px.core.Frame:
    rgb = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 3, axis=1)
    if auxiliary:
        z = np.arange(values.size, dtype=np.float32)[:, None] + np.float32(16.0)
        data = np.concatenate((z, rgb[:, [2, 0, 1]]), axis=1)
        channels: tuple[str, ...] = ("Z", "B", "R", "G")
    else:
        data = rgb
        channels = ("R", "G", "B")
    return px.io.from_array(
        cp.asarray(data[None, :, :]),
        colorspace="ACEScg",
        gamma=gamma,
        channels=channels,
        matrix="native",
    )


def _red_values(frame: px.core.Frame) -> np.ndarray:
    red = frame.channels.index("R")
    return px.io.to_array(frame).get()[0, :, red]


def test_slog3_encode_directly_applies_the_vendor_lower_branch_and_anchors() -> None:
    """v1-log-negative-extension acceptance 1 and 5: S-Log3 encode matches the independent vendor oracle."""
    cut = np.float32(_SLOG3_CUT)
    values = np.asarray(
        (
            -0.25,
            -0.01,
            0.0,
            np.nextafter(cut, np.float32(-np.inf)),
            cut,
            np.nextafter(cut, np.float32(np.inf)),
            0.18,
            0.9,
            1.0,
            1.5,
        ),
        dtype=np.float32,
    )

    actual = _red_values(px.color.linear_to_gamma(_frame(values), gamma="S-Log3"))
    expected = _slog3_encode(values.astype(np.float64))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-6)
    np.testing.assert_array_equal(
        np.rint(actual[[2, 6, 7]] * np.float32(1023.0)).astype(np.int64),
        np.asarray((95, 420, 598), dtype=np.int64),
    )


def test_slog3_decode_directly_applies_the_vendor_lower_branch_through_the_cut() -> None:
    """v1-log-negative-extension acceptance 2 and 5: S-Log3 decode matches the independent branch oracle."""
    cut = np.float32(_SLOG3_CODE_CUT)
    values = np.asarray(
        (
            -0.25,
            -0.01,
            0.0,
            95.0 / 1023.0,
            np.nextafter(cut, np.float32(-np.inf)),
            cut,
            np.nextafter(cut, np.float32(np.inf)),
            420.0 / 1023.0,
            598.0 / 1023.0,
            1.0,
        ),
        dtype=np.float32,
    )

    actual = _red_values(px.color.gamma_to_linear(_frame(values, gamma="S-Log3")))
    expected = _slog3_decode(values.astype(np.float64))

    # 1e-5 covers approximately 2.6 float32 ULPs at the largest decoded value (38.4) after CUDA powf.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_logc4_encode_directly_selects_log_or_linear_at_the_negative_cut() -> None:
    """v1-log-negative-extension acceptance 3 and 5; v1-red-tokens acceptance 68: retain ARRI-LogC4 encode."""
    cut = np.float32(_LOGC4_T)
    values = np.asarray(
        (
            -0.25,
            np.nextafter(cut, np.float32(-np.inf)),
            cut,
            np.nextafter(cut, np.float32(np.inf)),
            -0.01,
            0.0,
            0.18,
            1.0,
            1.5,
        ),
        dtype=np.float32,
    )

    actual = _red_values(px.color.linear_to_gamma(_frame(values), gamma="ARRI-LogC4"))
    expected = _logc4_encode(values.astype(np.float64))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(
        actual[[5, 6, 7]],
        np.asarray((95.0 / 1023.0, 0.2783958365, 0.4275193648), dtype=np.float64),
        rtol=0.0,
        atol=5e-6,
    )


def test_logc4_decode_uses_the_linear_branch_for_every_negative_code() -> None:
    """v1-log-negative-extension acceptance 4 and 5; v1-red-tokens acceptance 68: retain ARRI-LogC4 decode."""
    values = np.asarray(
        (
            -0.5,
            -0.1,
            np.nextafter(np.float32(0.0), np.float32(-np.inf)),
            0.0,
            np.nextafter(np.float32(0.0), np.float32(np.inf)),
            95.0 / 1023.0,
            0.2783958365,
            0.4275193648,
            1.0,
        ),
        dtype=np.float32,
    )

    actual = _red_values(px.color.gamma_to_linear(_frame(values, gamma="ARRI-LogC4")))
    expected = _logc4_decode(values.astype(np.float64))

    # 2e-5 is below one float32 ULP at the largest decoded value (469.8) after CUDA powf.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-5)


@pytest.mark.parametrize(
    ("gamma", "encode", "decode", "linear"),
    (
        ("S-Log3", _slog3_encode, _slog3_decode, (-0.25, -0.01, 0.0, 0.01125, 0.18, 1.0, 1.5)),
        ("ARRI-LogC4", _logc4_encode, _logc4_decode, (-0.25, float(_LOGC4_T), -0.01, 0.0, 0.18, 1.0, 1.5)),
    ),
)
def test_log_transfers_round_trip_both_directions_against_independent_oracles(
    gamma: str,
    encode: Callable[[np.ndarray], np.ndarray],
    decode: Callable[[np.ndarray], np.ndarray],
    linear: tuple[float, ...],
) -> None:
    """v1-log-negative-extension acceptance 5; v1-red-tokens acceptance 68: renamed transfers round-trip."""
    linear_values = np.asarray(linear, dtype=np.float32)
    encoded_values = encode(linear_values.astype(np.float64)).astype(np.float32)

    encoded = px.color.linear_to_gamma(_frame(linear_values), gamma=gamma)
    restored_linear = px.color.gamma_to_linear(encoded, gamma=gamma)
    decoded = px.color.gamma_to_linear(_frame(encoded_values, gamma=gamma), gamma=gamma)
    restored_encoded = px.color.linear_to_gamma(decoded, gamma=gamma)

    np.testing.assert_allclose(_red_values(encoded), encode(linear_values.astype(np.float64)), rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(_red_values(restored_linear), linear_values, rtol=0.0, atol=8e-6)
    np.testing.assert_allclose(_red_values(decoded), decode(encoded_values.astype(np.float64)), rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(_red_values(restored_encoded), encoded_values, rtol=0.0, atol=8e-6)


@pytest.mark.parametrize("gamma", ("S-Log3", "ARRI-LogC4"))
def test_all_public_transfer_paths_bind_vendor_results_and_preserve_frame_contract(gamma: str) -> None:
    """v1-log-negative-extension acceptance 6; v1-red-tokens acceptance 68: paths preserve renamed labels."""
    values = np.asarray((-0.25, -0.01, 0.0, 0.18, 1.0, 1.5), dtype=np.float32)
    source = _frame(values, auxiliary=True)
    source_before = source.data.copy()

    one_way_encoded = px.color.linear_to_gamma(source, gamma=gamma)
    fused_encoded = px.color.rgb_to_rgb(source, output_gamma=gamma)
    one_way_decoded = px.color.gamma_to_linear(one_way_encoded, gamma=gamma)
    fused_decoded = px.color.rgb_to_rgb(one_way_encoded, output_gamma="linear")

    assert cp.array_equal(one_way_encoded.data, fused_encoded.data)
    assert cp.array_equal(one_way_decoded.data, fused_decoded.data)
    assert cp.array_equal(one_way_encoded.data[..., 0], source.data[..., 0])
    assert cp.array_equal(one_way_decoded.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, source_before)
    assert source.gamma == "linear"
    assert source.matrix == "native"
    assert one_way_encoded.gamma == gamma
    assert one_way_decoded.gamma == "linear"
    assert one_way_encoded.matrix is one_way_decoded.matrix is None
    assert one_way_encoded.channels == one_way_decoded.channels == source.channels
    assert one_way_encoded.data.dtype == one_way_decoded.data.dtype == cp.float32
    assert one_way_encoded is not source
    assert one_way_encoded.data.data.ptr != source.data.data.ptr


@pytest.mark.parametrize(
    ("gamma", "linear", "encode_bits", "encoded", "decode_bits"),
    (
        (
            "S-Log3",
            (0.0, 0.011249999515712261, 0.011250000447034836, 0.01125000137835741, 0.18000000715255737, 1.0, 1.5),
            (1035874188, 1043030190, 1043030190, 1043030190, 1053963405, 1058575679, 1059324708),
            (
                0.0,
                0.16736097633838654,
                0.16736099123954773,
                0.16736100614070892,
                0.09286412596702576,
                0.41055718064308167,
                0.5845552086830139,
                1.0,
            ),
            (3160785829, 1010323946, 1010323947, 1010323950, 789333745, 1043878379, 1063689578, 1108979464),
        ),
        (
            "ARRI-LogC4",
            (0.0, 0.18000000715255737, 1.0, 1.5),
            (1035874188, 1049528807, 1054532562, 1055775089),
            (0.0, 0.09286412596702576, 0.2783958315849304, 0.4275193512439728, 1.0),
            (3163810883, 0, 1043878379, 1065353203, 1139467878),
        ),
    ),
)
# Baseline provenance: the integer fixtures below were regenerated from
# 38b949d0bcf892f63e6383070ce77e667f95d945 and matched exactly. Extract that commit, then run the probe:
#   BASE_FIXTURE_DIR=$(mktemp -d); git archive 38b949d0bcf892f63e6383070ce77e667f95d945 | tar -x -C "$BASE_FIXTURE_DIR"
#   UV_PROJECT_ENVIRONMENT=/home/mia/repositories/pixtreme/.venv PYTHONPATH="$BASE_FIXTURE_DIR/src" uv run --no-sync python -c \
#   'import runpy,numpy as np,pixtreme as px;m=runpy.run_path("tests/test_log_negative_extension_spec.py");f=m["_frame"];r=m["_red_values"];rows=m["test_corrected_transfers_keep_pre_correction_nonnegative_bits_characterization"].pytestmark[0].args[1];[(print(g,r(px.color.linear_to_gamma(f(np.asarray(x,dtype=np.float32)),gamma=g)).view(np.uint32).tolist()),print(g,r(px.color.gamma_to_linear(f(np.asarray(e,dtype=np.float32),gamma=g),gamma=g)).view(np.uint32).tolist()))for g,x,_,e,_ in rows]'
def test_corrected_transfers_keep_pre_correction_nonnegative_bits_characterization(
    gamma: str,
    linear: tuple[float, ...],
    encode_bits: tuple[int, ...],
    encoded: tuple[float, ...],
    decode_bits: tuple[int, ...],
) -> None:
    """characterization: v1-log-negative-extension acceptance 6; v1-red-tokens acceptance 68 freezes GPU bits.

    The vendor formulas establish correctness independently. Keep this snapshot until the explicit nonnegative
    bit-identity contract changes or the CUDA arithmetic implementation is intentionally replaced.
    """
    actual_encode = _red_values(
        px.color.linear_to_gamma(_frame(np.asarray(linear, dtype=np.float32)), gamma=gamma)
    ).view(np.uint32)
    actual_decode = _red_values(
        px.color.gamma_to_linear(_frame(np.asarray(encoded, dtype=np.float32), gamma=gamma), gamma=gamma)
    ).view(np.uint32)

    np.testing.assert_array_equal(actual_encode, np.asarray(encode_bits, dtype=np.uint32))
    np.testing.assert_array_equal(actual_decode, np.asarray(decode_bits, dtype=np.uint32))


_UNCHANGED_TRANSFER_BITS = {
    "linear": (
        (3196059648, 0, 1043878380, 1065353216, 1069547520),
        (3196059648, 0, 1043878380, 1065353216, 1069547520),
    ),
    "sRGB": ((3226384466, 0, 1055667935, 1065353215, 1066982086), (3164504977, 0, 1021242176, 1065353216, 1075994818)),
    "Rec.709": (
        (3213885440, 0, 1053911414, 1065353216, 1067198555),
        (3177418297, 0, 1027778413, 1065353216, 1075003731),
    ),
    "BT.1886": (
        (3205475542, 0, 1056610176, 1065353216, 1066897169),
        (3172141195, 0, 1015393358, 1065353216, 1076452091),
    ),
    "PQ": (
        (3210348844, 893662952, 1062265261, 1065353216, 1065706239),
        (3121028420, 0, 959860078, 1065353216, 1134365847),
    ),
    "HLG": ((3208450448, 0, 1059856297, 1065353216, 1065973579), (3165301419, 0, 1009840764, 1065353217, 1098914311)),
    "Cineon": (
        (3204351021, 1035874182, 1055532491, 1059810011, 1060668676),
        (3168382053, 3149474432, 1009755528, 1096308956, 1143702814),
    ),
    "Gamma-2.2": (
        (3204993860, 0, 1055577349, 1065353216, 1067050896),
        (3175219967, 0, 1018977343, 1065353216, 1075587576),
    ),
    "Gamma-2.4": (
        (3205475542, 0, 1056610176, 1065353216, 1066897169),
        (3172141195, 0, 1015393358, 1065353216, 1076452091),
    ),
    "Gamma-2.6": (
        (3205903348, 0, 1057251335, 1065353216, 1066768924),
        (3168722025, 0, 1010678284, 1065353216, 1077389631),
    ),
}


def test_out_of_scope_transfers_keep_pre_correction_bits_characterization() -> None:
    """characterization: v1-log-negative-extension acceptance 7 freezes every out-of-scope transfer bit pattern.

    Existing public-formula tests establish correctness. Keep this snapshot until one of those transfer contracts or
    its CUDA arithmetic is intentionally changed.
    """
    values = np.asarray((-0.25, 0.0, 0.18, 1.0, 1.5), dtype=np.float32)
    for gamma, (encode_bits, decode_bits) in _UNCHANGED_TRANSFER_BITS.items():
        actual_encode = _red_values(px.color.linear_to_gamma(_frame(values), gamma=gamma)).view(np.uint32)
        actual_decode = _red_values(px.color.gamma_to_linear(_frame(values, gamma=gamma), gamma=gamma)).view(np.uint32)
        np.testing.assert_array_equal(actual_encode, np.asarray(encode_bits, dtype=np.uint32))
        np.testing.assert_array_equal(actual_decode, np.asarray(decode_bits, dtype=np.uint32))


def test_public_docs_describe_vendor_piecewise_signed_extensions() -> None:
    """v1-log-negative-extension acceptance 8; v1-sony-tokens acceptance 12; v1-red-tokens acceptance 72.

    Public docs state signed branches with the renamed ARRI token.
    """
    token_reference = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    sony_tokens = (ROOT / "docs" / "features" / "v1-sony-tokens.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    docstrings = {
        function.__name__: inspect.getdoc(function) or ""
        for function in (px.color.gamma_to_linear, px.color.linear_to_gamma, px.color.rgb_to_rgb)
    }
    token_rows = {
        token: next(line for line in token_reference.splitlines() if line.startswith(f"| `{token}` |"))
        for token in ("S-Log3", "ARRI-LogC4")
    }

    surfaces = {
        "docs_site/tokens.md S-Log3 row": (
            token_rows["S-Log3"],
            (
                "S-Log3 applies the Sony piecewise formula directly to signed inputs",
                "the lower linear branch extends below zero",
                "maps linear 0 to `95 / 1023`",
                "does not use sign/magnitude mirroring",
            ),
        ),
        "docs_site/tokens.md ARRI-LogC4 row": (
            token_rows["ARRI-LogC4"],
            (
                "ARRI-LogC4 applies the ARRI piecewise formula directly to signed inputs",
                "the log branch covers `x >= t`",
                "negative encoded values decode linearly without sign/magnitude mirroring",
            ),
        ),
        "docs/features/v1-sony-tokens.md": (
            sony_tokens,
            (
                "S-Log / S-Log2 / S-Log3 は入力の符号へ piecewise 式を直接適用する",
                "lower linear branch を負側へ直接適用する定義域外延長を共有する",
                "非負域の画素値は変わらない",
            ),
        ),
        "CHANGELOG.md": (
            changelog,
            (
                "S-Log3 applies the Sony piecewise formula directly to signed inputs",
                "ARRI-LogC4 applies the ARRI piecewise formula directly to signed inputs",
                "Results for nonnegative inputs remain float32 bit-identical",
            ),
        ),
        **{
            f"{name} docstring": (
                docstring,
                (
                    "S-Log / S-Log2 / S-Log3 apply their lower linear branches directly to signed inputs",
                    "Established S-Log3 and ARRI-LogC4 results for nonnegative inputs remain float32 bit-identical",
                ),
            )
            for name, docstring in docstrings.items()
        },
    }
    old_mirror_claims = (
        "Apply the standard formula to nonnegative magnitude and reflect the negative side with preserved sign",
        "`S-Log3` は 非負 magnitude に標準式を適用して符号を戻す既定の延長",
        "`S-Log3` の sign/magnitude mirror",
    )

    for surface, (text, required_claims) in surfaces.items():
        normalized = " ".join(text.split())
        for claim in required_claims:
            assert claim in normalized, f"{surface} is missing {claim!r}"
        for old_claim in old_mirror_claims:
            assert old_claim not in normalized, f"{surface} retains obsolete mirror classification {old_claim!r}"
