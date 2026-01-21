# general_python/__init__.py

"""
General Python Utilities - A comprehensive Python library for scientific computing.

This package provides utilities for scientific computing, particularly focused on 
quantum physics simulations and numerical methods. It consolidates commonly used 
functionalities into a unified, easy-to-use package with support for both NumPy 
and JAX backends.

Modules:
--------
- algebra   : Advanced linear algebra operations, solvers, and backend management
- common    : Common utilities for file handling, plotting, logging, and data management  
- lattices  : Tools for creating and manipulating lattice geometries
- maths     : Mathematical utilities, statistics, and random number generators
- ml        : Machine learning utilities with neural networks and training tools
- physics   : Quantum physics utilities including density matrices and entropy calculations

Examples:
---------
>>> import general_python as gp
>>> from general_python.algebra import utils
>>> backend = utils.get_global_backend()
>>> 
>>> from general_python.lattices import square
>>> lattice = square.SquareLattice(4, 4)

File    : general_python/__init__.py
Version : 0.1.0
Author  : Maksymilian Kliczkowski
License : MIT
"""

import sys
import importlib
from typing import TYPE_CHECKING
from .common.lazy import LazyImporter

# Package metadata
__version__         = "0.1.0"
__author__          = "Maksymilian Kliczkowski"
__email__           = "maksymilian.kliczkowski@pwr.edu.pl"
__license__         = "MIT"

# Description used by general_python.registry
MODULE_DESCRIPTION  = "Shared scientific utilities: algebra backends, logging, lattices, maths, ML, physics."

# -----------------------------------------------------------------------------------------------
# Lazy Import Configuration
# -----------------------------------------------------------------------------------------------

# Mapping of attribute names to (module_relative_path, attribute_name_in_module)
# If attribute_name_in_module is None, the module itself is imported.
_LAZY_IMPORTS = {
    # Subpackages
    "algebra"           : (".algebra",              None),
    "common"            : (".common",               None),
    "lattices"          : (".lattices",             None),
    "maths"             : (".maths",                None),
    "ml"                : (".ml",                   None),
    "physics"           : (".physics",              None),
    
    # Algebra conveniences
    "random"            : (".algebra.ran_wrapper",  None),
    "random_matrices"   : (".algebra.ran_matrices", None),
}

# For type checking, import modules explicitly to support static analysis
if TYPE_CHECKING:
    from . import algebra
    from . import common
    from . import lattices
    from . import maths
    from . import ml
    from . import physics
    from .algebra import ran_wrapper as random
    from .algebra import ran_matrices as random_matrices

# Initialize LazyImporter
_importer = LazyImporter(__name__, _LAZY_IMPORTS)

def __getattr__(name: str):
    """
    Module-level __getattr__ for lazy imports (PEP 562).
    """
    return _importer.lazy_import(name)


def __dir__():
    """
    Support for autocompletion of lazy attributes.
    """
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


# List of available modules (public API)
__all__ = list(_LAZY_IMPORTS.keys()) + [
    "get_module_description",
    "list_available_modules",
    "list_capabilities"
]

# -----------------------------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------------------------

def get_module_description(module_name):
    """
    Get the description of a specific module in the general_python package.
    
    Parameters
    ----------
    module_name : str
        The name of the module.
    
    Returns
    -------
    str
        The description of the module.
    """
    descriptions = {
        "algebra"   : "Advanced linear algebra operations, solvers, and backend management with NumPy/JAX support.",
        "common"    : "Common utilities for file handling, plotting, logging, binary operations, and data management.",
        "lattices"  : "Tools for creating and manipulating lattice geometries (square, hexagonal, honeycomb).",
        "maths"     : "Mathematical utilities, statistical functions, and high-quality random number generators.",
        "ml"        : "Machine learning utilities with neural networks, training tools, and optimization.",
        "physics"   : "Quantum physics utilities including density matrices, entropy calculations, and operators."
    }
    return descriptions.get(module_name, "Module not found.")

def list_available_modules():
    """
    List all available submodules in the general_python package.
    
    Returns
    -------
    list
        List of available submodule names.
    """
    return sorted([k for k, v in _LAZY_IMPORTS.items() if v[1] is None])

def list_capabilities():
    """Summarize core capabilities across subpackages.

    Returns a dictionary with keys: random, random_matrices, modules.
    """
    caps = {}
    try:
        rw              = _importer.lazy_import("random")
        caps["random"]  = rw.list_capabilities()
    except Exception:
        caps["random"]  = {}
    try:
        rm                      = _importer.lazy_import("random_matrices")
        caps["random_matrices"] = rm.list_capabilities()
    except Exception:
        caps["random_matrices"] = {}
    caps["modules"] = list_available_modules()
    return caps

# ---------------------------------------------------------------------
#! EOF
# ---------------------------------------------------------------------