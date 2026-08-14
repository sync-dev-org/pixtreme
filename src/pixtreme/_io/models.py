"""Image header value models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _ImagePart(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    channels: dict[str, str]
    deep: bool = False


class _ImageColorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw: dict[str, object]
    colorspace: str | None
    gamma: str | None
    mappable: bool | None


class ImageHeader(BaseModel):
    """Describe an image file without decoding its pixels.

    Headers produced by ``read_header`` carry a ``format`` of ``"PNG"``,
    ``"JPEG"``, ``"TIFF"``, ``"EXR"``, ``"JPEG2000"``, ``"WEBP"``,
    ``"BMP"``, ``"PNM"``, ``"TGA"``, ``"HDR"``, or ``"DPX"``, with ``width`` and ``height``
    describing the stored image dimensions. ``parts`` contains each part name, a mapping
    from channel label to stored dtype, and a per-part ``deep`` flag. Raster files have one unnamed part,
    while EXR files may expose multiple parts and use the first part's data
    window for the dimensions.

    ``color`` retains the parsed raw color attributes together with any mapped
    ``colorspace`` and ``gamma`` tokens and a mappability result. Instances are
    frozen Pydantic models; validation enforces the field structure and types,
    not semantic value ranges — the constraints above describe what
    ``read_header`` produces, not what the constructor rejects.
    """

    model_config = ConfigDict(frozen=True)

    format: str
    width: int
    height: int
    parts: tuple[_ImagePart, ...]
    color: _ImageColorInfo
