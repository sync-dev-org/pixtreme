"""Specification tests for CPU-only ICC matrix/TRC parsing and numerical mapping."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from icc_test_utils import (
    chad_tag,
    curv_tag,
    duplicate_tag_record,
    icc_profile,
    mutate_tag_record,
    para_tag,
    trc_payload,
)
from PIL import ImageCms


def _map(profile: bytes, *, compatible: bool = True) -> tuple[dict[str, object], str | None, str | None, bool | None]:
    from pixtreme._io.icc import _icc_color_info

    info = _icc_color_info(profile, compatible=compatible)
    return info.raw, info.colorspace, info.gamma, info.mappable


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
@pytest.mark.parametrize("version", (2, 4))
@pytest.mark.parametrize("profile_class", (b"mntr", b"scnr", b"spac"))
def test_valid_matrix_trc_profiles_map_independently_of_version_and_supported_class(
    version: int, profile_class: bytes
) -> None:
    """Supported ICC versions and profile classes map identical device-to-PCS tags to the same color metadata."""
    profile = icc_profile(version=version, profile_class=profile_class, colorspace="sRGB", gamma="sRGB")

    assert _map(profile) == ({"ICC": profile}, "sRGB", "sRGB", True)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
@pytest.mark.parametrize(
    "profile",
    (
        icc_profile(version=3),
        icc_profile(profile_class=b"prtr"),
        icc_profile(data_space=b"CMYK"),
        icc_profile(pcs=b"Lab "),
        icc_profile(signature=b"nope"),
        icc_profile(omit_tags=(b"rTRC",)),
        icc_profile(extra_tags={b"A2B0": b"mft1" + b"\x00" * 44}),
    ),
    ids=("version", "class", "space", "pcs", "signature", "incomplete", "hybrid"),
)
def test_unsupported_profile_structures_preserve_exact_raw_but_map_neither_component(profile: bytes) -> None:
    """Unsupported ICC structures preserve exact raw profile bytes without assigning color-space or gamma metadata."""
    assert _map(profile) == ({"ICC": profile}, None, None, False)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_profile_size_tag_table_alignment_overlap_and_duplicate_signature_are_validated() -> None:
    """ICC profile parsing validates declared size, tag table bounds, alignment, overlap, and duplicate signatures."""
    valid = icc_profile()
    wrong_size = bytearray(valid)
    struct.pack_into(">I", wrong_size, 0, len(valid) - 1)
    misaligned = mutate_tag_record(valid, b"rXYZ", offset_delta=1)
    outside = mutate_tag_record(valid, b"rXYZ", size_delta=len(valid))
    overlap = mutate_tag_record(valid, b"rXYZ", offset_delta=4)
    duplicate = duplicate_tag_record(valid, b"rXYZ")

    for profile in (bytes(wrong_size), misaligned, outside, overlap, duplicate):
        assert _map(profile) == ({"ICC": profile}, None, None, False)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_exact_tag_payload_sharing_is_accepted_for_the_three_identical_trcs() -> None:
    """ICC parsing accepts three identical transfer curves that share the same tag payload offset and size."""
    profile = icc_profile(colorspace="Adobe-RGB", gamma="Adobe-RGB")
    count = struct.unpack_from(">I", profile, 128)[0]
    records = {
        profile[132 + 12 * index : 136 + 12 * index]: struct.unpack_from(">II", profile, 136 + 12 * index)
        for index in range(count)
    }

    assert records[b"rTRC"] == records[b"gTRC"] == records[b"bTRC"]
    assert _map(profile) == ({"ICC": profile}, "Adobe-RGB", "Adobe-RGB", True)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_numerically_identical_colorspaces_use_only_the_fixed_icc_priorities() -> None:
    """ICC mapping applies fixed priorities when numerically identical definitions name sRGB, Rec.709, or Sony gamut
    variants.
    """
    srgb = icc_profile(colorspace="sRGB", gamma="linear")
    sgamut = icc_profile(colorspace="S-Gamut", gamma="linear")

    assert _map(srgb)[1] == "sRGB"
    assert _map(sgamut)[1] == "S-Gamut"


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_chad_v2_and_v4_recovery_follow_the_fixed_source_white_rules() -> None:
    """ICC color-space recovery follows the specified source-white rules for chad, v2 wtpt, and D50-native v4
    profiles.
    """
    chad = icc_profile(version=4, colorspace="Adobe-RGB", gamma="Adobe-RGB", with_chad=True)
    v2_without_chad = icc_profile(version=2, colorspace="Adobe-RGB", gamma="Adobe-RGB", with_chad=False)
    v4_d50 = icc_profile(version=4, colorspace="ProPhoto-RGB", gamma="ProPhoto-RGB", with_chad=False)
    v4_d65 = icc_profile(version=4, colorspace="Adobe-RGB", gamma="Adobe-RGB", with_chad=False)

    assert _map(chad)[1:] == ("Adobe-RGB", "Adobe-RGB", True)
    assert _map(v2_without_chad)[1:] == ("Adobe-RGB", "Adobe-RGB", True)
    assert _map(v4_d50)[1:] == ("ProPhoto-RGB", "ProPhoto-RGB", True)
    assert _map(v4_d65)[1:] == (None, "Gamma-2.2", False)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_singular_chad_and_xyz_scale_guard_only_remove_colorspace_mapping() -> None:
    """An invalid ICC color matrix prevents color-space mapping while preserving an independently valid gamma
    mapping.
    """
    singular = icc_profile(
        colorspace="Adobe-RGB",
        gamma="Adobe-RGB",
        extra_tags={b"chad": chad_tag(np.zeros((3, 3), dtype=np.float64))},
    )
    scaled = icc_profile(colorspace="Adobe-RGB", gamma="Adobe-RGB", colorant_scale=0.5)

    for profile in (singular, scaled):
        assert _map(profile) == ({"ICC": profile}, None, "Gamma-2.2", False)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
@pytest.mark.parametrize(
    ("colorspace", "gamma", "expected"),
    (
        ("sRGB", "linear", "linear"),
        ("sRGB", "Gamma-1.8", "Gamma-1.8"),
        ("sRGB", "Gamma-2.2", "Gamma-2.2"),
        ("sRGB", "Gamma-2.4", "Gamma-2.4"),
        ("Adobe-RGB", "Adobe-RGB", "Adobe-RGB"),
        ("ProPhoto-RGB", "ProPhoto-RGB", "ProPhoto-RGB"),
        ("sRGB", "sRGB", "sRGB"),
        ("sRGB", "Rec.709", "Rec.709"),
    ),
)
def test_curv_and_para_realized_curves_map_only_to_the_closed_target_set(
    colorspace: str, gamma: str, expected: str
) -> None:
    """ICC curv and para transfer curves map only when their realized values match a supported gamma name."""
    profile = icc_profile(colorspace=colorspace, gamma=gamma)

    assert _map(profile)[2] == expected


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
@pytest.mark.parametrize(
    "payload",
    (
        para_tag(0, (2.4,)),
        para_tag(1, (2.4, 1.0, 0.0)),
        para_tag(2, (2.4, 1.0, 0.0, 0.0)),
        para_tag(3, (2.4, 1.0, 0.0, 0.0, 0.0)),
        para_tag(4, (2.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ),
)
def test_all_para_types_accept_equivalent_degenerate_pure_power_forms(payload: bytes) -> None:
    """All ICC parametric curve types recognize equivalent pure-power forms from their realized values."""
    profile = icc_profile(gamma="Gamma-2.4", trcs=(payload,) * 3)

    assert _map(profile)[2] == "Gamma-2.4"


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_sampled_curve_uses_knots_grid_and_branch_points_to_reject_sparse_false_positive() -> None:
    """ICC sampled-curve mapping checks knots, a dense grid, and branch points so sparse matches do not select a
    gamma name.
    """
    x = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    dense = curv_tag(samples=x**2.2)
    sparse = curv_tag(samples=np.asarray((0.0, 0.5, 1.0), dtype=np.float64) ** 2.2)

    dense_profile = icc_profile(gamma="Gamma-2.2", trcs=(dense,) * 3)
    sparse_profile = icc_profile(gamma="Gamma-2.2", trcs=(sparse,) * 3)

    assert _map(dense_profile)[2] == "Gamma-2.2"
    assert _map(sparse_profile)[2] is None


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_adobe_count_one_tie_break_depends_only_on_recovered_colorspace() -> None:
    """An ICC one-sample Adobe transfer curve uses only the recovered color space to break its gamma-name tie."""
    adobe_curve = trc_payload("Adobe-RGB")
    adobe = icc_profile(colorspace="Adobe-RGB", trcs=(adobe_curve,) * 3)
    srgb = icc_profile(colorspace="sRGB", trcs=(adobe_curve,) * 3)
    para_adobe = icc_profile(colorspace="sRGB", trcs=(para_tag(0, (563.0 / 256.0,)),) * 3)

    assert _map(adobe)[2] == "Adobe-RGB"
    assert _map(srgb)[2] == "Gamma-2.2"
    assert _map(para_adobe)[2] == "Adobe-RGB"


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_channel_mismatch_and_container_incompatibility_preserve_partial_contract() -> None:
    """ICC transfer-curve mismatch and non-RGB containers preserve raw bytes and any independently valid metadata
    component.
    """
    profile = icc_profile(
        colorspace="Adobe-RGB",
        trcs=(trc_payload("sRGB"), trc_payload("Rec.709"), trc_payload("sRGB")),
    )

    assert _map(profile) == ({"ICC": profile}, "Adobe-RGB", None, False)
    assert _map(profile, compatible=False) == ({"ICC": profile}, None, None, False)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_description_header_illuminant_and_reserved_values_do_not_affect_mapping() -> None:
    """ICC description and non-color header fields do not change the selected color-space or gamma names."""
    profile = bytearray(icc_profile(extra_tags={b"desc": b"desc" + b"\x00" * 28}))
    profile[68:80] = b"\xff" * 12
    profile[100:128] = b"ignored reserved field bytes!"[:28]
    struct.pack_into(">I", profile, 0, len(profile))
    value = bytes(profile)

    assert _map(value) == ({"ICC": value}, "sRGB", "sRGB", True)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_malformed_trc_type_and_count_leave_valid_colorspace_available() -> None:
    """Malformed ICC transfer-curve data blocks gamma mapping while leaving a valid color-space mapping available."""
    malformed = b"para\x00\x00\x00\x00" + struct.pack(">HH", 5, 0) + b"\x00" * 28
    profile = icc_profile(colorspace="P3-D65", trcs=(malformed,) * 3)

    assert _map(profile) == ({"ICC": profile}, "P3-D65", None, False)


@pytest.mark.req("REQ-PIX-002")
@pytest.mark.req("REQ-PIX-007")
def test_lcms_generated_v4_srgb_profile_maps_without_becoming_a_runtime_dependency() -> None:
    """An independently generated ICC v4 sRGB profile maps to sRGB metadata without requiring lcms2 at runtime."""
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    assert _map(profile) == ({"ICC": profile}, "sRGB", "sRGB", True)
