"""pixtreme: GPU-first image processing library built on CUDA and CuPy."""

from importlib import import_module as _import_module
from importlib.metadata import version as _distribution_version
from types import ModuleType as _ModuleType

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

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

_COMPANION_DISTRIBUTIONS = {
    "infer": "pixtreme-infer",
    "transport": "pixtreme-transport",
}

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


def __getattr__(name: str) -> _ModuleType:
    distribution = _COMPANION_DISTRIBUTIONS.get(name)
    if distribution is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = f"{__name__}.{name}"
    try:
        module = _import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise ImportError(
            f"why=optional companion module {module_name!r} is not installed; "
            f"what=module {module_name!r} was not found; how=pip install {distribution}"
        ) from error

    globals().pop(name, None)
    return module
