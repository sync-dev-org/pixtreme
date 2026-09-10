"""Independent checks for the deterministic ICC profile corpus built during each test run."""

from __future__ import annotations

import struct
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
from icc_test_utils import BRADFORD, D50, curv_tag, icc_profile, para_tag, rgb_to_xyz, xy_to_xyz, xyz_tag
from PIL import Image, ImageCms

from pixtreme._io.icc import _icc_color_info


def _tag_record(profile: bytes, signature: bytes) -> tuple[int, int] | None:
    if len(profile) < 132:
        return None
    count = struct.unpack_from(">I", profile, 128)[0]
    if 132 + 12 * count > len(profile):
        return None
    for index in range(count):
        record = 132 + 12 * index
        name, offset, size = struct.unpack_from(">4sII", profile, record)
        if name == signature:
            return offset, size
    return None


def _tags(profile: bytes) -> dict[bytes, bytes]:
    count = struct.unpack_from(">I", profile, 128)[0]
    result = {}
    for index in range(count):
        record = 132 + 12 * index
        signature, offset, size = struct.unpack_from(">4sII", profile, record)
        result[signature] = profile[offset : offset + size]
    return result


def _fixed_values(payload: bytes, count: int) -> np.ndarray:
    return np.asarray(struct.unpack_from(f">{count}i", payload, 8), dtype=np.float64) / 65536.0


def _source_chromaticities(profile: bytes) -> np.ndarray:
    tags = _tags(profile)
    matrix = np.column_stack(tuple(_fixed_values(tags[name], 3) for name in (b"rXYZ", b"gXYZ", b"bXYZ")))
    if b"chad" in tags:
        inverse = np.linalg.inv(_fixed_values(tags[b"chad"], 9).reshape(3, 3))
        matrix = inverse @ matrix
        white = inverse @ D50
    elif profile[8] == 2:
        white = _fixed_values(tags[b"wtpt"], 3)
        source_cones = BRADFORD @ white
        target_cones = BRADFORD @ D50
        adaptation = np.linalg.inv(BRADFORD) @ np.diag(target_cones / source_cones) @ BRADFORD
        matrix = np.linalg.inv(adaptation) @ matrix
    else:
        white = D50
    xyz = tuple(matrix[:, index] for index in range(3)) + (white,)
    return np.asarray(tuple((value[0] / value.sum(), value[1] / value.sum()) for value in xyz), dtype=np.float64)


def _decode(values: np.ndarray, gamma: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if gamma == "sRGB":
        return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    if gamma == "Adobe-RGB":
        return values ** (563.0 / 256.0)
    if gamma == "Gamma-1.8":
        return values**1.8
    if gamma == "Gamma-2.4":
        return values**2.4
    if gamma == "ProPhoto-RGB":
        return np.where(values < 1.0 / 32.0, values / 16.0, values**1.8)
    raise ValueError(gamma)


def _encode_srgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.0031308, 12.92 * values, 1.055 * np.maximum(values, 0.0) ** (1.0 / 2.4) - 0.055)


def _bradford_conversion(source: object, target: object) -> np.ndarray:
    source_primaries = tuple(tuple(float(value) for value in row) for row in source.primaries)
    source_white = tuple(float(value) for value in source.whitepoint)
    target_primaries = tuple(tuple(float(value) for value in row) for row in target.primaries)
    target_white = tuple(float(value) for value in target.whitepoint)
    source_cones = BRADFORD @ xy_to_xyz(source_white)
    target_cones = BRADFORD @ xy_to_xyz(target_white)
    adaptation = np.linalg.inv(BRADFORD) @ np.diag(target_cones / source_cones) @ BRADFORD
    return (
        np.linalg.inv(rgb_to_xyz(target_primaries, target_white))
        @ adaptation
        @ rgb_to_xyz(source_primaries, source_white)
    )


def _description_tag(value: str) -> bytes:
    ascii_value = value.encode("ascii") + b"\x00"
    return (
        b"desc\x00\x00\x00\x00"
        + struct.pack(">I", len(ascii_value))
        + ascii_value
        + struct.pack(">IIHB", 0, 0, 0, 0)
        + b"\x00" * 67
    )


def _flow_tags(description: str) -> dict[bytes, bytes]:
    return {
        b"bkpt": xyz_tag((0.0, 0.0, 0.0)),
        b"cprt": b"text\x00\x00\x00\x00hand-built test profile\x00",
        b"desc": _description_tag(description),
    }


def _srgb_curve(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _corpus() -> tuple[tuple[str, bytes, str, str, str], ...]:
    sampled_x = np.linspace(0.0, 1.0, 1024, dtype=np.float64)
    sampled_srgb = curv_tag(samples=_srgb_curve(sampled_x))
    v2_srgb = icc_profile(
        colorspace="sRGB",
        version=2,
        with_chad=False,
        trcs=(sampled_srgb,) * 3,
        extra_tags=_flow_tags("sRGB v2 sampled"),
    )
    adobe = icc_profile(
        colorspace="Adobe-RGB",
        gamma="Adobe-RGB",
        version=2,
        with_chad=False,
        extra_tags=_flow_tags("Adobe RGB (1998) v2 count-one"),
    )
    prophoto_flow = icc_profile(
        colorspace="ProPhoto-RGB",
        gamma="Gamma-1.8",
        version=2,
        with_chad=False,
        extra_tags=_flow_tags("ROMM RGB v2 count-one"),
    )
    lcms_srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    display_p3_v4 = icc_profile(
        colorspace="P3-D65",
        gamma="sRGB",
        version=4,
        with_chad=True,
        extra_tags=_flow_tags("Display P3 v4 parametric"),
    )
    display_p3_v2 = icc_profile(
        colorspace="P3-D65",
        version=2,
        with_chad=False,
        trcs=(sampled_srgb,) * 3,
        extra_tags=_flow_tags("Display P3 v2 sampled"),
    )
    prophoto_toe = icc_profile(
        colorspace="ProPhoto-RGB",
        version=4,
        with_chad=True,
        trcs=(para_tag(3, (1.8, 1.0, 0.0, 1.0 / 16.0, 1.0 / 32.0)),) * 3,
        extra_tags=_flow_tags("ProPhoto RGB ISO toe"),
    )
    rec2020 = icc_profile(
        colorspace="Rec.2020",
        gamma="Gamma-2.4",
        version=4,
        with_chad=True,
        extra_tags=_flow_tags("Rec.2020 pure 2.4"),
    )
    return (
        ("sRGB v2 curv table", v2_srgb, "sRGB", "sRGB", "sRGB"),
        ("Adobe RGB v2 curv count 1", adobe, "Adobe-RGB", "Adobe-RGB", "Adobe RGB (1998)"),
        ("ProPhoto v2 curv count 1", prophoto_flow, "ProPhoto-RGB", "Gamma-1.8", "ProPhoto RGB"),
        ("lcms2 sRGB v4 para", lcms_srgb, "sRGB", "sRGB", "sRGB"),
        ("Display P3 v4 para", display_p3_v4, "P3-D65", "sRGB", "Display P3"),
        ("Display P3 v2 curv table", display_p3_v2, "P3-D65", "sRGB", "Display P3"),
        ("ProPhoto v4 para toe", prophoto_toe, "ProPhoto-RGB", "ProPhoto-RGB", "ProPhoto RGB"),
        ("Rec.2020 v4 para 2.4", rec2020, "Rec.2020", "Gamma-2.4", "ITU-R BT.2020"),
    )


def test_deterministic_icc_corpus_maps_and_matches_colour_lcms_and_composite_oracles() -> None:
    """v1-io-icc acceptance 14 and 21: generated v2/v4 curv/para profiles satisfy every independent oracle."""
    corpus = _corpus()
    assert len(corpus) == 8
    assert len(corpus[3][1]) == 588
    assert not (Path(__file__).resolve().parent / "data" / "icc").exists()

    versions: list[int] = []
    curve_types: list[bytes] = []
    levels = np.arange(0, 256, 17, dtype=np.uint8)
    cube = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1).reshape(64, 64, 3)
    destination = ImageCms.createProfile("sRGB")
    print_options = np.get_printoptions()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import colour
    finally:
        np.set_printoptions(**print_options)

    for label, profile, expected_colorspace, expected_gamma, colour_name in corpus:
        versions.append(profile[8])
        tags = _tags(profile)
        curve_types.append(tags[b"rTRC"][:4])
        color = _icc_color_info(profile, compatible=True)
        assert (color.colorspace, color.gamma, color.mappable) == (expected_colorspace, expected_gamma, True), label

        reference = colour.RGB_COLOURSPACES[colour_name]
        expected_xy = np.asarray((*reference.primaries, reference.whitepoint), dtype=np.float64)
        np.testing.assert_allclose(_source_chromaticities(profile), expected_xy, rtol=0.0, atol=1e-4)

        opened = ImageCms.getOpenProfile(BytesIO(profile))
        for signature, attribute in (
            (b"rXYZ", "red_colorant"),
            (b"gXYZ", "green_colorant"),
            (b"bXYZ", "blue_colorant"),
            (b"wtpt", "media_white_point"),
        ):
            lcms_xyz = np.asarray(getattr(opened.profile, attribute)[0], dtype=np.float64)
            np.testing.assert_allclose(lcms_xyz, _fixed_values(tags[signature], 3), rtol=0.0, atol=1.0 / 65536.0)

        transform = ImageCms.buildTransformFromOpenProfiles(
            opened,
            destination,
            "RGB",
            "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            flags=0,
        )
        lcms_output = np.asarray(ImageCms.applyTransform(Image.fromarray(cube, mode="RGB"), transform), dtype=np.int16)
        target = colour.RGB_COLOURSPACES["sRGB"]
        linear = _decode(cube.astype(np.float64) / 255.0, expected_gamma)
        converted = linear @ _bradford_conversion(reference, target).T
        expected = np.floor(np.clip(_encode_srgb(converted), 0.0, 1.0) * 255.0 + 0.5).astype(np.int16)
        assert int(np.max(np.abs(lcms_output - expected))) <= 1, label

        description_record = _tag_record(profile, b"desc") or _tag_record(profile, b"mluc")
        assert description_record is not None
        description_offset, description_size = description_record
        changed = bytearray(profile)
        payload_start = min(description_offset + 8, description_offset + description_size)
        changed[payload_start : description_offset + description_size] = b"\x00" * (
            description_offset + description_size - payload_start
        )
        changed_color = _icc_color_info(bytes(changed), compatible=True)
        assert (changed_color.colorspace, changed_color.gamma, changed_color.mappable) == (
            expected_colorspace,
            expected_gamma,
            True,
        )

    assert versions.count(2) == versions.count(4) == 4
    assert curve_types.count(b"curv") == curve_types.count(b"para") == 4
