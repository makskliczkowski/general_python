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
Version     : 1.1.0
Changelog   :
- 1.1.0: Added lazy loading for submodules and common exports, improved documentation, and added capability listing functions.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import algebra, common, lattices, maths, ml, physics
    from .algebra import ran_wrapper as random
    from .common import (
        LazyDataEntry,
        LazyHDF5Entry,
        LazyJsonEntry,
        LazyNpzEntry,
        LazyPickleEntry,
        PlotData,
        ResultSet,
        dtype_to_name,
        filter_results,
        load_results,
    )

    gp_algebra  = algebra
    gp_common   = common
    gp_lattices = lattices
    gp_maths    = maths
    gp_physics  = physics
    gp_ml       = ml

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
    'random'        : 'algebra.ran_wrapper',
    'gp_algebra'    : 'algebra',
    'gp_common'     : 'common',
    'gp_lattices'   : 'lattices',
    'gp_maths'      : 'maths',
    'gp_physics'    : 'physics',
    'gp_ml'         : 'ml',
}

_COMMON_EXPORTS = {
    'dtype_to_name',
    'load_results',
    'filter_results',
    'ResultSet',
    'PlotData',
    'LazyDataEntry',
    'LazyHDF5Entry',
    'LazyNpzEntry',
    'LazyPickleEntry',
    'LazyJsonEntry',
}

def __getattr__(name: str):
    """Lazy-load submodules on demand."""
    if name in _SUBMODULES:
        module = importlib.import_module(f'.{name}', package=__name__)
        globals()[name] = module
        return module
    if name in _ALIASES:
        module = importlib.import_module(f'.{_ALIASES[name]}', package=__name__)
        globals()[name] = module
        return module
    if name in _COMMON_EXPORTS:
        common = importlib.import_module('.common', package=__name__)
        value = getattr(common, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def list_capabilities():
    """List available capabilities."""
    return sorted(list(_SUBMODULES) + list(_ALIASES.keys()) + list(_COMMON_EXPORTS))

def __dir__():
    """Return list of available attributes."""
    return sorted(
        list(globals().keys())
        + list(_SUBMODULES)
        + list(_ALIASES.keys())
        + list(_COMMON_EXPORTS)
        + ['list_capabilities']
    )

def list_available_modules():
    """Return list of available submodules."""
    return sorted(list(_SUBMODULES))

def get_module_description(module_name: str) -> str:
    """Return a brief description of the module."""
    descriptions = {
        'algebra'   : "Algebraic structures, eigensolvers, and backend operations.",
        'common'    : "Common utilities (logging, file I/O, plotting).",
        'lattices'  : "Lattice definitions and k-space utilities.",
        'maths'     : "Mathematical functions and numerical utilities.",
        'physics'   : "Physical constants and quantum mechanics tools.",
        'ml'        : "Machine learning utilities.",
    }
    return descriptions.get(module_name, "No description available.")

__all__ = sorted(
    list(_SUBMODULES)
    + list(_ALIASES.keys())
    + list(_COMMON_EXPORTS)
    + ['dtype_to_name', 'get_module_description', 'list_available_modules', 'list_capabilities']
)

# ----------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------
