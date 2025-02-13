"""
This module contains the following submodules:
"""

# Import the required modules common to all submodules
from . import common
from .common.plot import MatrixPrinter, Plotter, PlotterSave
from .common.flog import Logger

# Lattice operations
from . import lattices

# Mathematical operations
from . import maths

# Algebraic operations
from . import algebra
# module level imports
from .algebra import _JAX_AVAILABLE, _KEY, DEFAULT_BACKEND, get_backend, maybe_jit

# list of submodules
__all__ = ["common", "lattices", "maths", "algebra"]

# description of the modules to be displayed
def get_module_description(module_name):
    """
    Get the description of a specific module in the general_python package.
    
    Parameters:
    - module_name (str): The name of the module.
    
    Returns:
    - str: The description of the module.
    """
    descriptions = {
        "common"    : "Provides common functionalities used in any Python project.",
        "lattices"  : "Provides functionalities for creating and managing lattices.",
        "maths"     : "Provides mathematical utilities and functions.",
        "algebra"   : "Provides functionalities for algebraic operations." 
    }
    return descriptions.get(module_name, "Module not found.")

# list of available modules
def list_available_modules():
    """
    List all available modules in the general_python package.
    
    Returns:
    - list: A list of available module names.
    """
    return ["common", "lattices", "maths", "algebra"]

# ----------------------------------------------------------------------------------------------