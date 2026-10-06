"""Bundled and installed font-file discovery."""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from pixtreme._fonts import available, font_path

__all__ = ("font_path", "available")
