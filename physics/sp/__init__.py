"""
Single-particle physics module.

This subpackage contains utilities for single-particle Hamiltonians,
tight-binding models, and related spectral properties.
"""

from . import correlation_matrix

# Optional submodules — import only if they exist
try:
    from . import utils
except ImportError:
    pass
try:
    from . import dos
except ImportError:
    pass
try:
    from . import spectrum
except ImportError:
    pass

__all__ = ["correlation_matrix", "utils", "dos", "spectrum"]
