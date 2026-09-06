"""Specification tests for explicit white-point and Temperature / Tint white balance."""

from __future__ import annotations

import importlib.metadata
import inspect
import math
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest
from repository_contracts import require_repo_file

import pixtreme as px

_CAT_ORACLE_NAMES = {
    "Bradford": "Bradford",
    "CAT02": "CAT02",
    "CAT16": "CAT16",
    "von-Kries": "Von Kries",
}

# Adobe DNG SDK 1.7.1 dng_temperature.cpp kTempTable. The 325-mired
# coordinate intentionally retains the SDK's 0.24702 record.
_DNG_TEMPERATURE_TABLE = (
    (0.0, 0.18006, 0.26352, -0.24341),
    (10.0, 0.18066, 0.26589, -0.25479),
    (20.0, 0.18133, 0.26846, -0.26876),
    (30.0, 0.18208, 0.27119, -0.28539),
    (40.0, 0.18293, 0.27407, -0.30470),
    (50.0, 0.18388, 0.27709, -0.32675),
    (60.0, 0.18494, 0.28021, -0.35156),
    (70.0, 0.18611, 0.28342, -0.37915),
    (80.0, 0.18740, 0.28668, -0.40955),
    (90.0, 0.18880, 0.28997, -0.44278),
    (100.0, 0.19032, 0.29326, -0.47888),
    (125.0, 0.19462, 0.30141, -0.58204),
    (150.0, 0.19962, 0.30921, -0.70471),
    (175.0, 0.20525, 0.31647, -0.84901),
    (200.0, 0.21142, 0.32312, -1.0182),
    (225.0, 0.21807, 0.32909, -1.2168),
    (250.0, 0.22511, 0.33439, -1.4512),
    (275.0, 0.23247, 0.33904, -1.7298),
    (300.0, 0.24010, 0.34308, -2.0637),
    (325.0, 0.24702, 0.34655, -2.4681),
    (350.0, 0.25591, 0.34951, -2.9641),
    (375.0, 0.26400, 0.35200, -3.5814),
    (400.0, 0.27218, 0.35407, -4.3633),
    (425.0, 0.28039, 0.35577, -5.3762),
    (450.0, 0.28863, 0.35714, -6.7262),
    (475.0, 0.29685, 0.35823, -8.5955),
    (500.0, 0.30505, 0.35907, -11.324),
    (525.0, 0.31320, 0.35968, -15.628),
    (550.0, 0.32129, 0.36011, -23.325),
    (575.0, 0.32931, 0.36038, -40.770),
    (600.0, 0.33724, 0.36051, -116.45),
)

_SRGB_RGB_TO_XYZ = np.asarray(
    (
        (0.4123907992659595, 0.35758433938387796, 0.1804807884018343),
        (0.21263900587151036, 0.7151686787677559, 0.07219231536073371),
        (0.01933081871559182, 0.11919477979462599, 0.9505321522496607),
    ),
    dtype=np.float64,
)


def _assert_actionable(error: pytest.ExceptionInfo[ValueError]) -> None:
    message = str(error.value)
    assert message.startswith("why=")
    assert "; what=" in message
    assert "; how=" in message
    assert message.index("why=") < message.index("; what=") < message.index("; how=")


def _frame(
    values: Any,
    *,
    colorspace: str = "sRGB",
    gamma: str = "linear",
    channels: str | tuple[str, ...] = "RGB",
    dtype: Any = np.float32,
) -> px.core.Frame:
    import cupy as cp

    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    return px.io.from_array(cp.asarray(array), colorspace=colorspace, gamma=gamma, channels=channels)


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _independent_temperature_to_xy(temperature: float, tint: float = 0.0) -> tuple[float, float]:
    r = np.float64(1_000_000.0) / np.float64(temperature)
    upper_index = next(
        (index for index, record in enumerate(_DNG_TEMPERATURE_TABLE) if np.float64(record[0]) >= r),
        len(_DNG_TEMPERATURE_TABLE) - 1,
    )
    lower_index = max(0, upper_index - 1)
    lower = _DNG_TEMPERATURE_TABLE[lower_index]
    upper = _DNG_TEMPERATURE_TABLE[upper_index]
    if lower_index == upper_index:
        fraction = np.float64(0.0)
    else:
        fraction = (r - np.float64(lower[0])) / (np.float64(upper[0]) - np.float64(lower[0]))
    u = np.float64(lower[1]) + fraction * (np.float64(upper[1]) - np.float64(lower[1]))
    v = np.float64(lower[2]) + fraction * (np.float64(upper[2]) - np.float64(lower[2]))
    slope = np.float64(lower[3]) + fraction * (np.float64(upper[3]) - np.float64(lower[3]))
    length = np.sqrt(np.float64(1.0) + slope * slope)
    u_tinted = u - np.float64(tint) / length
    v_tinted = v - slope * np.float64(tint) / length
    denominator = np.float64(2.0) * u_tinted - np.float64(8.0) * v_tinted + np.float64(4.0)
    return (
        float(np.float64(3.0) * u_tinted / denominator),
        float(np.float64(2.0) * v_tinted / denominator),
    )


def _xy_to_uv(xy: tuple[float, float]) -> np.ndarray:
    x, y = (np.float64(value) for value in xy)
    denominator = np.float64(1.5) - x + np.float64(6.0) * y
    return np.asarray((np.float64(2.0) * x / denominator, np.float64(3.0) * y / denominator))


def _decode_srgb(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    linear = values <= np.float64(0.04045)
    result[linear] = values[linear] / np.float64(12.92)
    result[~linear] = np.power((values[~linear] + np.float64(0.055)) / np.float64(1.055), np.float64(2.4))
    return result


def _encode_srgb(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    linear = values <= np.float64(0.0031308)
    result[linear] = np.float64(12.92) * values[linear]
    result[~linear] = np.float64(1.055) * np.power(values[~linear], np.float64(1.0) / np.float64(2.4)) - np.float64(
        0.055
    )
    return result


def test_public_surface_signatures_alias_counts_and_docs_are_synchronized() -> None:
    """v1-white-balance acceptance 1; v1-white-point-simulation acceptance 1:
    API, Literal, operation counts, requirements, and token docs agree. GitHub #29.
    """
    expected_tokens = ("Bradford", "CAT02", "CAT16", "von-Kries")
    assert get_args(px.core.ChromaticAdaptation) == expected_tokens
    assert px.color.__all__[-3:-1] == ("chromatic_adaptation", "white_balance")
    assert len(px.color.__all__) == 15

    chromatic_signature = inspect.signature(px.color.chromatic_adaptation)
    assert tuple(chromatic_signature.parameters) == ("frame", "input_white", "output_white", "cat")
    assert chromatic_signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        chromatic_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("input_white", "output_white", "cat")
    )
    assert chromatic_signature.parameters["input_white"].default is inspect.Parameter.empty
    assert chromatic_signature.parameters["output_white"].default is inspect.Parameter.empty
    assert chromatic_signature.parameters["cat"].default == "CAT02"

    balance_signature = inspect.signature(px.color.white_balance)
    assert tuple(balance_signature.parameters) == ("frame", "temperature", "tint", "cat")
    assert balance_signature.parameters["frame"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        balance_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("temperature", "tint", "cat")
    )
    assert balance_signature.parameters["temperature"].default is inspect.Parameter.empty
    assert balance_signature.parameters["tint"].default == 0.0
    assert balance_signature.parameters["cat"].default == "CAT02"

    root = Path(__file__).resolve().parents[1]
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    tokens = (root / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    assert "| `color` |" in requirements and "| 15 |" in next(
        line for line in requirements.splitlines() if line.startswith("| `color` |")
    )
    assert "公開 operation は計 94 関数" in requirements
    section = tokens.split("## chromatic adaptation\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    assert (
        tuple(line.split("|")[1].strip().strip("`") for line in section.splitlines() if line.startswith("| `"))
        == expected_tokens
    )


def _import_colour_oracle():
    """Import the colour oracle without leaking its NumPy print-option mutation.

    colour 0.4.7 runs ``np.set_printoptions(legacy="1.13")`` at import time. That
    process-global state truncates ``str(np.float32(...))`` shortest round-trip
    formatting, which the Cube LUT serializer relies on for bit-exact output, so the
    snapshot taken before the import is always restored afterwards.
    """
    import warnings

    saved = np.get_printoptions()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import colour
            import colour.adaptation  # noqa: F401
    finally:
        np.set_printoptions(**saved)
    return colour


@pytest.mark.parametrize(("token", "oracle_name"), tuple(_CAT_ORACLE_NAMES.items()))
def test_every_cat_matrix_matches_colour_science_047(token: str, oracle_name: str) -> None:
    """v1-white-balance acceptance 2: every CAT uses the exact named colour-science 0.4.7 matrix oracle."""
    colour = _import_colour_oracle()
    matrix_chromatic_adaptation_VonKries = colour.adaptation.matrix_chromatic_adaptation_VonKries

    from pixtreme._color.white_balance import _chromatic_adaptation_matrix

    assert colour.__version__ == "0.4.7"
    input_white = (0.34567, 0.35850)
    output_white = (0.31270, 0.32900)
    expected = matrix_chromatic_adaptation_VonKries(
        _xy_to_xyz(input_white),
        _xy_to_xyz(output_white),
        transform=oracle_name,
    )
    actual = _chromatic_adaptation_matrix(input_white, output_white, token)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)


def test_default_cat_is_cat02_and_cat_tokens_remain_distinct() -> None:
    """v1-white-balance acceptance 2: omission selects CAT02 without normalizing the four CAT paths together."""
    source = _frame((0.31, 0.47, 0.82))
    kwargs = {"input_white": (0.34567, 0.35850), "output_white": (0.31270, 0.32900)}
    default = px.color.chromatic_adaptation(source, **kwargs)
    explicit = px.color.chromatic_adaptation(source, cat="CAT02", **kwargs)
    assert np.array_equal(px.io.to_array(default).get(), px.io.to_array(explicit).get())
    outputs = {
        token: px.io.to_array(px.color.chromatic_adaptation(source, cat=token, **kwargs)).get().tobytes()
        for token in _CAT_ORACLE_NAMES
    }
    assert len(set(outputs.values())) == len(_CAT_ORACLE_NAMES)


@pytest.mark.parametrize(
    ("input_white", "output_white", "cat"),
    (
        ((0.3,), (0.3127, 0.3290), "CAT02"),
        ((0.3, 0.3, 0.4), (0.3127, 0.3290), "CAT02"),
        ((True, 0.3), (0.3127, 0.3290), "CAT02"),
        (("0.3", 0.3), (0.3127, 0.3290), "CAT02"),
        ((math.nan, 0.3), (0.3127, 0.3290), "CAT02"),
        ((math.inf, 0.3), (0.3127, 0.3290), "CAT02"),
        ((0.0, 0.3), (0.3127, 0.3290), "CAT02"),
        ((0.3, 0.0), (0.3127, 0.3290), "CAT02"),
        ((0.6, 0.4), (0.3127, 0.3290), "CAT02"),
        ((0.3, 0.3), (0.3127, 0.3290), "CAT02"),
        ((0.3, 0.3), (0.3127, 0.3290), None),
    ),
)
def test_chromatic_adaptation_rejects_invalid_public_arguments_before_pixel_processing(
    monkeypatch: pytest.MonkeyPatch,
    input_white: object,
    output_white: object,
    cat: object,
) -> None:
    """v1-white-balance acceptance 3: malformed xy and CAT values fail actionably before launching a pixel pass."""
    import pixtreme._color.white_balance as implementation

    def forbidden_transform(*args: object, **kwargs: object) -> object:
        raise AssertionError("pixel processing must not start for invalid arguments")

    monkeypatch.setattr(implementation, "_transform_data", forbidden_transform)
    with pytest.raises(ValueError) as error:
        px.color.chromatic_adaptation(
            _frame((0.2, 0.3, 0.4)),
            input_white=input_white,  # type: ignore[arg-type]
            output_white=output_white,  # type: ignore[arg-type]
            cat=cat,  # type: ignore[arg-type]
        )
    _assert_actionable(error)


def test_chromatic_adaptation_rejects_numerically_zero_cat_response() -> None:
    """v1-white-balance acceptance 3: a valid xy that creates a zero CAT cone response is rejected."""
    y = np.float64(0.1)
    x = (np.float64(0.1624) - np.float64(0.5920) * y) / np.float64(0.8952)
    assert x > 0.0 and y > 0.0 and x + y < 1.0
    with pytest.raises(ValueError) as error:
        px.color.chromatic_adaptation(
            _frame((0.2, 0.3, 0.4)),
            input_white=(float(x), float(y)),
            output_white=(0.3127, 0.3290),
            cat="CAT02",
        )
    _assert_actionable(error)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    (
        (px.color.chromatic_adaptation, {"input_white": (0.34567, 0.35850), "output_white": (0.3127, 0.3290)}),
        (px.color.white_balance, {"temperature": 5000.0}),
    ),
)
@pytest.mark.parametrize(
    "invalid_frame",
    (
        object(),
        pytest.param("float16", id="float16"),
        pytest.param("missing-rgb", id="missing-rgb"),
    ),
)
def test_both_apis_fail_fast_for_non_frame_dtype_and_rgb_contract(
    operation: Callable[..., px.core.Frame], kwargs: dict[str, object], invalid_frame: object
) -> None:
    """v1-white-balance acceptance 4: both APIs require a float32 Frame with exactly one R, G, and B label."""
    if invalid_frame == "float16":
        invalid_frame = _frame((0.2, 0.3, 0.4), dtype=np.float16)
    elif invalid_frame == "missing-rgb":
        invalid_frame = _frame((0.2, 0.3), channels=("R", "G"))
    with pytest.raises(ValueError) as error:
        operation(invalid_frame, **kwargs)  # type: ignore[arg-type]
    _assert_actionable(error)
    assert "Frame" in str(error.value) or "float32" in str(error.value) or "R" in str(error.value)


def test_adaptation_is_one_pass_label_driven_private_and_metadata_preserving(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-white-balance acceptance 5: one fused pass transforms labelled RGB and preserves private auxiliary data."""
    import cupy as cp

    import pixtreme._color.white_balance as implementation

    values = np.asarray(
        [[[-0.25, 0.60, 17.0, 0.15, 1.40, -3.0], [1.25, -0.10, 23.0, 0.70, 0.05, 8.0]]],
        dtype=np.float32,
    )
    source = _frame(values, gamma="sRGB", channels=("A", "B", "custom", "R", "G", "Z"))
    source_snapshot = source.data.copy()
    metadata_snapshot = (source.colorspace, source.gamma, source.channels, source.matrix)
    original_transform = implementation._transform_data
    calls = 0

    def counted_transform(*args: object, **kwargs: object) -> cp.ndarray:
        nonlocal calls
        calls += 1
        return original_transform(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(implementation, "_transform_data", counted_transform)
    output = px.color.chromatic_adaptation(
        source,
        input_white=(0.34567, 0.35850),
        output_white=(0.31270, 0.32900),
        cat="CAT16",
    )

    assert calls == 1
    assert output is not source
    assert not cp.shares_memory(output.data, source.data)
    assert output.data.flags.c_contiguous
    assert output.data.dtype == cp.float32
    assert (output.colorspace, output.gamma, output.channels, output.matrix) == (
        source.colorspace,
        source.gamma,
        source.channels,
        None,
    )
    assert cp.array_equal(output.data[..., (0, 2, 5)], source.data[..., (0, 2, 5)])
    assert cp.array_equal(source.data, source_snapshot)
    assert (source.colorspace, source.gamma, source.channels, source.matrix) == metadata_snapshot


@pytest.mark.parametrize("cat", tuple(_CAT_ORACLE_NAMES))
def test_equal_white_points_return_bit_preserving_private_copy(cat: str) -> None:
    """v1-white-balance acceptance 6: equal whites return a bit-identical all-channel private copy for every CAT."""
    import cupy as cp

    source = _frame(
        np.asarray([[[-0.0, math.nan, 1.25, -2.0], [math.inf, -math.inf, 0.5, 7.0]]], dtype=np.float32),
        channels=("B", "A", "R", "G"),
    )
    output = px.color.chromatic_adaptation(
        source,
        input_white=(0.3127, 0.3290),
        output_white=(0.3127, 0.3290),
        cat=cat,
    )
    assert not cp.shares_memory(output.data, source.data)
    assert output.data.view(cp.uint32).tobytes() == source.data.view(cp.uint32).tobytes()
    assert output.matrix is None


@pytest.mark.parametrize("gamma", ("linear", "sRGB"))
def test_scene_values_match_independent_host_matrix_and_reverse_round_trip(gamma: str) -> None:
    """v1-white-balance acceptance 7: scene values match an independent host oracle without clipping and round-trip."""
    matrix_chromatic_adaptation_VonKries = _import_colour_oracle().adaptation.matrix_chromatic_adaptation_VonKries

    input_white = (0.34567, 0.35850)
    output_white = (0.31270, 0.32900)
    values = np.asarray(
        [[[-0.20, 0.35, 1.40], [1.75, 0.08, 0.60], [0.04, 1.20, 0.22]]],
        dtype=np.float32,
    )
    encoded_values = _encode_srgb(values.astype(np.float64)).astype(np.float32) if gamma == "sRGB" else values
    source = _frame(encoded_values, gamma=gamma)

    cat_matrix = matrix_chromatic_adaptation_VonKries(
        _xy_to_xyz(input_white),
        _xy_to_xyz(output_white),
        transform="CAT02",
    )
    rgb_matrix = np.linalg.inv(_SRGB_RGB_TO_XYZ) @ cat_matrix @ _SRGB_RGB_TO_XYZ
    decoded = _decode_srgb(encoded_values.astype(np.float64)) if gamma == "sRGB" else encoded_values.astype(np.float64)
    linear_expected = np.einsum("ij,...j->...i", rgb_matrix, decoded)
    expected = _encode_srgb(linear_expected) if gamma == "sRGB" else linear_expected

    actual = px.color.chromatic_adaptation(
        source,
        input_white=input_white,
        output_white=output_white,
    )
    tolerance = 8e-5 if gamma == "sRGB" else 1e-5
    np.testing.assert_allclose(px.io.to_array(actual).get(), expected, rtol=tolerance, atol=tolerance)
    assert np.min(px.io.to_array(actual).get()) < 0.0
    assert np.max(px.io.to_array(actual).get()) > 1.0

    restored = px.color.chromatic_adaptation(
        actual,
        input_white=output_white,
        output_white=input_white,
    )
    np.testing.assert_allclose(
        px.io.to_array(restored).get(),
        encoded_values,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    ("mired", "tint"),
    ((600.0, 0.0), (325.0, 0.0), (287.5, 0.0), (287.5, 0.0137), (1e-6, -0.0042)),
)
def test_temperature_table_records_off_grid_interpolation_and_uv_conversion(mired: float, tint: float) -> None:
    """v1-white-balance acceptance 8-9: DNG records, off-grid interpolation, Duv direction, and uv-to-xy are exact."""
    from pixtreme._color.white_balance import _DNG_TEMPERATURE_TABLE as production_table
    from pixtreme._color.white_balance import _temperature_to_xy

    assert production_table == _DNG_TEMPERATURE_TABLE
    temperature = 1_000_000.0 / mired
    expected = _independent_temperature_to_xy(temperature, tint)
    actual = _temperature_to_xy(temperature, tint)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("temperature", (True, math.nan, math.inf, -math.inf, 0.0, 1600.0))
def test_temperature_validation_rejects_invalid_or_below_table_values(temperature: object) -> None:
    """v1-white-balance acceptance 8: Temperature is finite, non-bool, and bounded only by the 600-mired endpoint."""
    with pytest.raises(ValueError) as error:
        px.color.white_balance(_frame((0.2, 0.3, 0.4)), temperature=temperature)  # type: ignore[arg-type]
    _assert_actionable(error)


def test_temperature_accepts_every_finite_value_through_float64_max() -> None:
    """v1-white-balance acceptance 8: the 0-mired endpoint imposes no finite upper Temperature bound."""
    output = px.color.white_balance(_frame((0.2, 0.3, 0.4)), temperature=np.finfo(np.float64).max)
    assert np.isfinite(px.io.to_array(output).get()).all()


@pytest.mark.parametrize("tint", (True, math.nan, math.inf, -math.inf, 1.0))
def test_tint_validation_rejects_nonfinite_and_out_of_chromaticity_without_clamping(tint: object) -> None:
    """v1-white-balance acceptance 9: Tint is finite raw Duv and invalid derived xy is rejected rather than clamped."""
    with pytest.raises(ValueError) as error:
        px.color.white_balance(_frame((0.2, 0.3, 0.4)), temperature=5000.0, tint=tint)  # type: ignore[arg-type]
    _assert_actionable(error)


def test_tint_is_signed_raw_duv_with_positive_green_side() -> None:
    """v1-white-balance acceptance 9: positive and negative Tint move symmetrically by raw Duv on the documented line."""
    from pixtreme._color.white_balance import _temperature_to_xy

    magnitude = np.float64(0.01)
    zero_uv = _xy_to_uv(_temperature_to_xy(5000.0, 0.0))
    positive_delta = _xy_to_uv(_temperature_to_xy(5000.0, float(magnitude))) - zero_uv
    negative_delta = _xy_to_uv(_temperature_to_xy(5000.0, float(-magnitude))) - zero_uv
    np.testing.assert_allclose(positive_delta, -negative_delta, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(np.linalg.norm(positive_delta), magnitude, rtol=0.0, atol=2e-16)
    assert positive_delta[0] < 0.0 < positive_delta[1]


@pytest.mark.parametrize(
    ("temperature", "tint", "cat"),
    ((2800.0, -0.008, "Bradford"), (6500.0, 0.0, "CAT02"), (12000.0, 0.011, "von-Kries")),
)
def test_white_balance_is_bit_identical_to_explicit_low_level_call(temperature: float, tint: float, cat: str) -> None:
    """v1-white-balance acceptance 10: convenience mapping executes the same low-level input/output/CAT kernel."""
    source = _frame(
        np.asarray([[[-0.1, 0.4, 1.2], [0.7, 0.2, 0.05]]], dtype=np.float32),
        colorspace="sRGB",
        gamma="sRGB",
    )
    input_white = _independent_temperature_to_xy(temperature, tint)
    direct = px.color.chromatic_adaptation(
        source,
        input_white=input_white,
        output_white=(0.3127, 0.3290),
        cat=cat,
    )
    convenience = px.color.white_balance(source, temperature=temperature, tint=tint, cat=cat)
    assert px.io.to_array(convenience).get().tobytes() == px.io.to_array(direct).get().tobytes()


def test_temperature_and_tint_have_source_illuminant_correction_semantics() -> None:
    """v1-white-balance acceptance 11: hotter sources warm output and positive green Tint corrects toward magenta."""
    source = _frame((0.18, 0.18, 0.18))
    cold = px.io.to_array(px.color.white_balance(source, temperature=2800.0)).get()[0, 0]
    nominal = px.io.to_array(px.color.white_balance(source, temperature=6500.0)).get()[0, 0]
    hot = px.io.to_array(px.color.white_balance(source, temperature=12000.0)).get()[0, 0]
    positive_tint = px.io.to_array(px.color.white_balance(source, temperature=6500.0, tint=0.01)).get()[0, 0]
    negative_tint = px.io.to_array(px.color.white_balance(source, temperature=6500.0, tint=-0.01)).get()[0, 0]

    assert hot[0] / hot[2] > nominal[0] / nominal[2] > cold[0] / cold[2]
    assert abs(float(nominal[0] - nominal[2])) < 0.01
    assert positive_tint[1] / np.mean(positive_tint[[0, 2]]) < nominal[1] / np.mean(nominal[[0, 2]])
    assert negative_tint[1] / np.mean(negative_tint[[0, 2]]) > nominal[1] / np.mean(nominal[[0, 2]])


def test_calls_are_bit_deterministic_across_order_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-white-balance acceptance 12: output has no time, environment, call-order, metadata, or filesystem state."""
    source = _frame(
        np.asarray([[[-0.2, 0.3, 1.5], [0.8, 0.1, 0.6]]], dtype=np.float32),
        colorspace="ACEScg",
        gamma="linear",
    )
    kwargs = {"temperature": 7312.5, "tint": -0.00625, "cat": "von-Kries"}
    first = px.io.to_array(px.color.white_balance(source, **kwargs)).get().tobytes()
    monkeypatch.setenv("PIXTREME_WHITE_BALANCE_TEST_SENTINEL", "must-not-be-read")
    px.color.white_balance(source, temperature=2400.0, tint=0.015, cat="Bradford")
    second = px.io.to_array(px.color.white_balance(source, **kwargs)).get().tobytes()
    assert first == second


def test_adaptation_host_memoization_is_bounded_lru_and_recomputes_bit_exactly() -> None:
    """v1-white-balance acceptance 18: resolved adaptation matrices use a 128-entry LRU with bit-exact eviction."""
    import pixtreme._color.white_balance as implementation

    cached = implementation._compose_adaptation_rgb_matrix
    cached.cache_clear()
    key = ("sRGB", (0.3127, 0.3290), (0.3457, 0.3585), "CAT02")
    cold = cached(*key).tobytes()
    assert cached.cache_info().maxsize == 128
    assert cached(*key).tobytes() == cold
    assert cached.cache_info().hits == 1

    for index in range(128):
        input_white = (float(np.float64(0.29) + np.float64(index) * np.float64(1e-5)), 0.33)
        cached("sRGB", input_white, (0.3457, 0.3585), "CAT02")

    assert cached.cache_info().currsize == 128
    misses_before_revisit = cached.cache_info().misses
    recomputed = cached(*key)
    assert cached.cache_info().misses == misses_before_revisit + 1
    assert recomputed.tobytes() == cold == cached.__wrapped__(*key).tobytes()


def test_adaptation_memoization_identity_uses_every_resolved_binary64_value() -> None:
    """v1-white-balance acceptance 19: tokens and Temperature/Tint normalize to exact resolved binary64 cache keys."""
    import pixtreme._color.white_balance as implementation

    cached = implementation._compose_adaptation_rgb_matrix
    cached.cache_clear()
    source = _frame((0.2, 0.3, 0.4), colorspace="sRGB")
    px.color.chromatic_adaptation(source, input_white="D65", output_white="D50", cat="CAT02")
    token_stats = cached.cache_info()
    px.color.chromatic_adaptation(source, input_white=(0.3127, 0.3290), output_white=(0.3457, 0.3585), cat="CAT02")
    direct_stats = cached.cache_info()
    assert direct_stats.hits == token_stats.hits + 1
    assert direct_stats.misses == token_stats.misses

    temperature = 7312.5
    tint = -0.00625
    resolved_input = implementation._temperature_to_xy(temperature, tint)
    px.color.white_balance(source, temperature=temperature, tint=tint, cat="von-Kries")
    balance_stats = cached.cache_info()
    px.color.chromatic_adaptation(
        source,
        input_white=resolved_input,
        output_white="D65",
        cat="von-Kries",
    )
    assert cached.cache_info().hits == balance_stats.hits + 1

    input_white = (0.3127, 0.3290)
    output_white = (0.3457, 0.3585)
    cached.cache_clear()
    cached("sRGB", input_white, output_white, "CAT02")
    misses = cached.cache_info().misses
    variants = (
        ("ACEScg", input_white, output_white, "CAT02"),
        ("sRGB", (float(np.nextafter(input_white[0], np.inf)), input_white[1]), output_white, "CAT02"),
        ("sRGB", (input_white[0], float(np.nextafter(input_white[1], np.inf))), output_white, "CAT02"),
        ("sRGB", input_white, (float(np.nextafter(output_white[0], np.inf)), output_white[1]), "CAT02"),
        ("sRGB", input_white, (output_white[0], float(np.nextafter(output_white[1], np.inf))), "CAT02"),
        ("sRGB", input_white, output_white, "CAT16"),
    )
    for variant in variants:
        cached(*variant)
    assert cached.cache_info().misses == misses + len(variants)


def test_adaptation_cache_states_and_uncached_composition_are_publicly_bit_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-white-balance acceptance 20: miss, hit, interposition, eviction, and uncached calls are bit identical."""
    import pixtreme._color.white_balance as implementation

    cached = implementation._compose_adaptation_rgb_matrix
    cached.cache_clear()
    source = _frame(
        np.asarray([[[-0.2, 0.3, 1.5], [0.8, 0.1, 0.6]]], dtype=np.float32),
        colorspace="ACEScg",
        gamma="linear",
    )
    temperature = 7312.5
    tint = -0.00625
    cat = "von-Kries"
    input_white = implementation._temperature_to_xy(temperature, tint)
    output_white = (0.32168, 0.33767)

    def balance_bits() -> bytes:
        return (
            px.io.to_array(px.color.white_balance(source, temperature=temperature, tint=tint, cat=cat)).get().tobytes()
        )

    def direct_bits() -> bytes:
        return (
            px.io.to_array(
                px.color.chromatic_adaptation(
                    source,
                    input_white=input_white,
                    output_white=output_white,
                    cat=cat,
                )
            )
            .get()
            .tobytes()
        )

    cold = balance_bits()
    hit = balance_bits()
    direct = direct_bits()
    monkeypatch.setenv("PIXTREME_WHITE_BALANCE_CACHE_SENTINEL", "ignored")
    px.color.chromatic_adaptation(source, input_white="D50", output_white="ACES", cat="Bradford")
    interposed = balance_bits()
    for index in range(128):
        other_input = (float(np.float64(0.29) + np.float64(index) * np.float64(1e-5)), 0.33)
        cached("ACEScg", other_input, output_white, cat)
    evicted = balance_bits()
    cached_matrix = cached("ACEScg", input_white, output_white, cat).tobytes()
    uncached_matrix = cached.__wrapped__("ACEScg", input_white, output_white, cat).tobytes()

    monkeypatch.setattr(implementation, "_compose_adaptation_rgb_matrix", cached.__wrapped__)
    uncached_balance = balance_bits()
    uncached_direct = direct_bits()
    assert cold == hit == direct == interposed == evicted == uncached_balance == uncached_direct
    assert cached_matrix == uncached_matrix


def test_colour_science_is_exact_pinned_test_only_and_absent_from_runtime_metadata() -> None:
    """v1-white-balance acceptance 13: colour-science 0.4.7 is an exact-pinned development-only oracle."""
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "colour-science==0.4.7" in pyproject["dependency-groups"]["dev"]
    assert not any(requirement.startswith("colour-science") for requirement in pyproject["project"]["dependencies"])
    runtime_requirements = importlib.metadata.requires("pixtreme") or ()
    assert not any(requirement.lower().startswith("colour-science") for requirement in runtime_requirements)

    production_sources = "\n".join(path.read_text(encoding="utf-8") for path in (root / "src").rglob("*.py"))
    assert "import colour" not in production_sources
    assert "from colour" not in production_sources


def test_colour_oracle_import_leaves_numpy_print_state_for_bit_exact_lut_serialization() -> None:
    """v1-white-balance acceptance 13: the colour oracle import leaves process-global NumPy state unchanged.

    colour 0.4.7 sets NumPy legacy print options at import time. If that state leaks,
    ``str(np.float32(...))`` loses shortest round-trip formatting and the Cube LUT
    serializer emits values that no longer restore bit-exactly, in every test order
    where this module imports the oracle before a LUT round-trip test runs.
    """
    before = np.get_printoptions()
    _import_colour_oracle()
    assert np.get_printoptions() == before
    one_ulp_below_one = np.nextafter(np.float32(1.0), np.float32(0.0))
    assert str(one_ulp_below_one) == "0.99999994"
    assert np.float32(str(one_ulp_below_one)) == one_ulp_below_one
