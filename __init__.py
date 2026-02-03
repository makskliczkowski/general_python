"""
General Python utilities for Quantum EigenSolver.

This module provides common utilities, lattice definitions, algebra tools,
and mathematical functions used throughout the QES package.

Submodules:
-----------
- algebra   : Algebraic structures, eigensolvers, and backend operations
- common    : Common utilities (logging, file I/O, plotting)
- lattices  : Lattice definitions and k-space utilities
- maths     : Mathematical functions and numerical utilities
- physics   : Physical constants and quantum mechanics tools
- ml        : Machine learning utilities

Author      : Maksymilian Kliczkowski
Date        : 2025-02-02
"""

import sys
import importlib

__version__ = "1.1.0"

# Submodules that can be lazy-loaded
_SUBMODULES = {
    'algebra',
    'common',
    'lattices',
    'maths',
    'physics',
    'ml',
}

def __getattr__(name: str):
    """Lazy-load submodules on demand."""
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', package=__name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """Return list of available attributes."""
    return sorted(list(globals().keys()) + list(_SUBMODULES))

# ----------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------