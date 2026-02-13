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

# Aliases
_ALIASES = {
    'random': 'algebra.ran_wrapper',
}

# Try to import submodules eagerly for CI compatibility
# In CI environments, __getattr__ may not work reliably
try:
    from . import algebra
    from . import common
    from . import lattices
    from . import maths
    from . import physics
    from . import ml
except ImportError:
    # Fallback to lazy loading if eager import fails (e.g., missing optional dependencies)
    pass

def __getattr__(name: str):
    """Lazy-load submodules on demand."""
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', package=__name__)
    if name in _ALIASES:
        return importlib.import_module(f'.{_ALIASES[name]}', package=__name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def list_capabilities():
    """List available capabilities."""
    return sorted(list(_SUBMODULES) + list(_ALIASES.keys()))

def __dir__():
    """Return list of available attributes."""
    return sorted(list(globals().keys()) + list(_SUBMODULES) + list(_ALIASES.keys()) + ['list_capabilities'])

def list_available_modules():
    """Return list of available submodules."""
    return sorted(list(_SUBMODULES))

def get_module_description(module_name: str) -> str:
    """Return a brief description of the module."""
    descriptions = {
        'algebra': "Algebraic structures, eigensolvers, and backend operations.",
        'common': "Common utilities (logging, file I/O, plotting).",
        'lattices': "Lattice definitions and k-space utilities.",
        'maths': "Mathematical functions and numerical utilities.",
        'physics': "Physical constants and quantum mechanics tools.",
        'ml': "Machine learning utilities.",
    }
    return descriptions.get(module_name, "No description available.")

# ----------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------