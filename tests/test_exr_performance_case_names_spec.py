"""EXR performance case IDs used by the development measurement report."""

from __future__ import annotations

from test_performance_spec import _PERFORMANCE_CASES, _exr_case_compression_direction


def test_every_exr_performance_case_names_its_compression_and_direction() -> None:
    """Each EXR performance case identifies the compression and read or write route it measures."""
    expected = {
        "file-exr-none-read": ("none", "read"),
        "file-exr-zip-read": ("zip", "read"),
        "file-exr-zips-read": ("zips", "read"),
        "file-exr-dwaa-read": ("dwaa", "read"),
        "file-exr-dwab-read": ("dwab", "read"),
        "file-exr-rle-read": ("rle", "read"),
        "file-exr-pxr24-read": ("pxr24", "read"),
        "file-exr-b44-read": ("b44", "read"),
        "file-exr-b44a-read": ("b44a", "read"),
        "file-exr-piz-read": ("piz", "read"),
        "file-exr-none-write": ("none", "write"),
        "file-exr-zip-write": ("zip", "write"),
        "file-exr-zips-write": ("zips", "write"),
        "file-exr-dwaa-write": ("dwaa", "write"),
        "file-exr-dwab-write": ("dwab", "write"),
        "file-exr-rle-write": ("rle", "write"),
        "file-exr-pxr24-write": ("pxr24", "write"),
        "file-exr-b44-write": ("b44", "write"),
        "file-exr-b44a-write": ("b44a", "write"),
        "file-exr-piz-write": ("piz", "write"),
        "file-read-exr": ("zip", "read"),
        "file-write-exr": ("zip", "write"),
        "file-exr-mixed-dtype-write-zip": ("zip", "write"),
    }
    actual = {case.case_id for case in _PERFORMANCE_CASES if "exr" in case.case_id}
    assert actual == set(expected)
    for case_id, identity in expected.items():
        assert _exr_case_compression_direction(case_id) == identity
