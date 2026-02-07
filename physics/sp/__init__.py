"""
Single-particle physics module.

This subpackage contains utilities for single-particle Hamiltonians,
tight-binding models, and related spectral properties.
"""

from . import utils
from . import dos
from . import spectrum

__all__ = ["utils", "dos", "spectrum"]
