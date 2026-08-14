"""pixtreme: GPU-first image processing library built on CUDA and CuPy."""

from importlib.metadata import version as _distribution_version

from pixtreme import channel as channel
from pixtreme import color as color
from pixtreme import composite as composite
from pixtreme import core as core
from pixtreme import draw as draw
from pixtreme import feature as feature
from pixtreme import filter as filter
from pixtreme import generate as generate
from pixtreme import io as io
from pixtreme import metrics as metrics
from pixtreme import morphology as morphology
from pixtreme import transform as transform
from pixtreme import values as values

__version__ = _distribution_version("pixtreme")

__all__ = (
    "core",
    "io",
    "color",
    "filter",
    "transform",
    "draw",
    "generate",
    "morphology",
    "metrics",
    "feature",
    "values",
    "channel",
    "composite",
    "__version__",
)
