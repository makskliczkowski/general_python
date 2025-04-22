# general_python/__init__.py is

"""
This module contains the following submodules:
- common    : Provides common functionalities used in any Python project.
- lattices  : Provides functionalities for creating and managing lattices.
- maths     : Provides mathematical utilities and functions.
- algebra   : Provides functionalities for algebraic operations.
- _G_LOGGER : Global logger for the package.
"""

import sys
import importlib

# List of available modules (not imported by default)
__all__     = ["common", "lattices", "maths", "algebra"]

# Description of modules
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

# List available modules
def list_available_modules():
    """List all available modules in the general_python package."""
    return __all__

# ---------------------------------------------------------------------

# Assign lazy modules

from . import algebra
from . import common
from . import lattices
from . import maths

# ---------------------------------------------------------------------