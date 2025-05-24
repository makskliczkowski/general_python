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

Version: 0.1.0
Author: Maksymilian Kliczkowski
License: MIT
"""

import sys
import importlib

# Package metadata
__version__ = "0.1.0"
__author__  = "Maksymilian Kliczkowski"
__email__   = "maksymilian.kliczkowski@pwr.edu.pl"
__license__ = "MIT"

# List of available modules (not imported by default)
__all__     = ["algebra", "common", "lattices", "maths", "ml", "physics"]

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

# ---------------------------------------------------------------------

# Import all modules for documentation and access
from . import algebra
from . import common
from . import lattices
from . import maths
from . import ml
from . import physics

# ---------------------------------------------------------------------