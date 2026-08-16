"""Generate deterministic user-font fixtures from the bundled Noto asset."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from fontTools import subset
from fontTools.designspaceLib import AxisDescriptor, DesignSpaceDocument, SourceDescriptor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTCollection, TTFont
from fontTools.ttLib.tables import otTables
from fontTools.varLib import build
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "src" / "pixtreme" / "data" / "fonts" / "NotoSansCJKjp-VF.otf"
DEFAULT_OUTPUT = ROOT / "tests" / "data" / "draw_text_user_font"
FIXTURE_TEXT = " AVfioffice\u9aa8"


def _subset_variable(source: Path) -> TTFont:
    font = TTFont(source, recalcTimestamp=False)
    options = subset.Options()
    options.layout_features = "*"
    options.name_IDs = ["*"]
    options.name_legacy = True
    options.name_languages = [0x0409]
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text=FIXTURE_TEXT)
    subsetter.subset(font)
    return font


def _static_instance(variable: TTFont, weight: float) -> TTFont:
    return instantiateVariableFont(
        copy.deepcopy(variable),
        {"wght": weight},
        inplace=True,
        optimize=True,
    )


def _append_static_width_axis_to_var_store(var_store: Any) -> None:
    if var_store.VarRegionList.RegionAxisCount != 1:
        raise ValueError("fixture source variation stores must contain exactly the wght axis")
    region_list = var_store.VarRegionList
    region_list.RegionAxisCount += 1
    for region in region_list.Region:
        if len(region.VarRegionAxis) != 1:
            raise ValueError("fixture source variation regions must contain exactly the wght axis")
        axis = otTables.VarRegionAxis()
        axis.StartCoord = 0.0
        axis.PeakCoord = 0.0
        axis.EndCoord = 0.0
        region.VarRegionAxis.append(axis)


def _walk_var_stores(value: Any, *, seen: set[int]) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, otTables.VarStore):
        _append_static_width_axis_to_var_store(value)
    if not hasattr(value, "iterSubTables"):
        return
    for child in value.iterSubTables():
        _walk_var_stores(child.value, seen=seen)


def _scaled_width(source: TTFont, scale: float) -> TTFont:
    scaled = copy.deepcopy(source)
    top_dict = scaled["CFF2"].cff.topDictIndex[0]
    glyph_set = scaled.getGlyphSet()
    for glyph_name in scaled.getGlyphOrder():
        original = top_dict.CharStrings[glyph_name]
        pen = T2CharStringPen(None, glyph_set, CFF2=True)
        glyph_set[glyph_name].draw(TransformPen(pen, (scale, 0.0, 0.0, 1.0, 0.0, 0.0)))
        top_dict.CharStrings[glyph_name] = pen.getCharString(original.private, original.globalSubrs)
    for glyph_name, (advance, left_side_bearing) in tuple(scaled["hmtx"].metrics.items()):
        scaled["hmtx"].metrics[glyph_name] = (
            round(advance * scale),
            round(left_side_bearing * scale),
        )
    return scaled


def _axis(name: str, tag: str, minimum: float, default: float, maximum: float) -> AxisDescriptor:
    axis = AxisDescriptor()
    axis.name = name
    axis.tag = tag
    axis.minimum = minimum
    axis.default = default
    axis.maximum = maximum
    return axis


def _source(name: str, font: TTFont, *, weight: float, width: float) -> SourceDescriptor:
    source = SourceDescriptor()
    source.name = name
    source.font = font
    source.location = {"weight": weight, "width": width}
    return source


def _two_axis_variable(source: TTFont) -> TTFont:
    weight_axis = next((axis for axis in source["fvar"].axes if axis.axisTag == "wght"), None)
    if weight_axis is None or len(source["fvar"].axes) != 1:
        raise ValueError("fixture source must contain exactly one wght variation axis")

    default_weight = float(weight_axis.defaultValue)
    default = _static_instance(source, default_weight)
    designspace = DesignSpaceDocument()
    designspace.addAxis(
        _axis(
            "weight",
            "wght",
            float(weight_axis.minValue),
            default_weight,
            float(weight_axis.maxValue),
        )
    )
    designspace.addAxis(_axis("width", "wdth", 75.0, 100.0, 125.0))
    designspace.addSource(_source("default", default, weight=default_weight, width=100.0))
    designspace.addSource(
        _source(
            "maximum-weight",
            _static_instance(source, float(weight_axis.maxValue)),
            weight=float(weight_axis.maxValue),
            width=100.0,
        )
    )
    designspace.addSource(_source("narrow", _scaled_width(default, 0.75), weight=default_weight, width=75.0))
    designspace.addSource(_source("wide", _scaled_width(default, 1.25), weight=default_weight, width=125.0))

    copied_tables = ("BASE", "VORG", "VVAR", "GDEF", "GPOS", "GSUB")
    variable, _model, _masters = build(designspace, optimize=True, exclude=list(copied_tables))
    for tag in copied_tables:
        if tag not in source:
            continue
        table = copy.deepcopy(source[tag])
        compiled = getattr(table, "table", None)
        if compiled is not None:
            _walk_var_stores(compiled, seen=set())
        variable[tag] = table
    variable.recalcBBoxes = False
    return variable


def generate(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    base = _subset_variable(source)

    variable = _two_axis_variable(base)
    variable.save(output / "noto-user-variable.otf", reorderTables=True)

    static_400 = _static_instance(base, 400.0)
    static_400.save(output / "noto-user-static.otf", reorderTables=True)

    collection = TTCollection()
    collection.fonts = [static_400, _static_instance(base, 700.0)]
    collection.save(output / "noto-user-collection.ttc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
