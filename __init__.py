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

File    : QES/general_python/__init__.py
Version : 0.1.0
Author  : Maksymilian Kliczkowski
License : MIT
"""

import sys
import importlib

# Package metadata
__version__         = "0.1.0"
__author__          = "Maksymilian Kliczkowski"
__email__           = "maksymilian.kliczkowski@pwr.edu.pl"
__license__         = "MIT"

# Description used by QES.registry
MODULE_DESCRIPTION  = "Shared scientific utilities: algebra backends, logging, lattices, maths, ML, physics."

# List of available modules (not imported by default)
__all__             = ["algebra", "common", "lattices", "maths", "ml", "physics", "random", "random_matrices"]

# Description of modules
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

# List available modules
def list_available_modules():
    """
    List all available modules in the general_python package.
    
    Returns
    -------
    list
        List of available module names.
    """
    return __all__

# Lazy import subpackages on attribute access (PEP 562)
def __getattr__(name):  # pragma: no cover - simple indirection
    # Convenience aliases
    if name == "random":
        return importlib.import_module(".algebra.ran_wrapper", __name__)
    if name == "random_matrices":
        return importlib.import_module(".algebra.ran_matrices", __name__)
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)


def list_capabilities():
    """Summarize core capabilities across subpackages.

    Returns a dictionary with keys: random, random_matrices, modules.
    """
    caps = {}
    try:
        rw = __getattr__("random")
        caps["random"] = rw.list_capabilities()
    except Exception:
        caps["random"] = {}
    try:
        rm = __getattr__("random_matrices")
        caps["random_matrices"] = rm.list_capabilities()
    except Exception:
        caps["random_matrices"] = {}
    caps["modules"] = list_available_modules()
    return caps

# ---------------------------------------------------------------------

# # Import all modules for documentation and access
# from . import algebra
# from . import common
# from . import lattices
# from . import maths
# try:
#     from . import ml
# except ImportError:
#     # ML module might fail during documentation builds due to sklearn dependencies
#     pass
# from . import physics

# ---------------------------------------------------------------------
#! EOF
# ---------------------------------------------------------------------