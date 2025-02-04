"""
This module contains the following submodules:
"""

from . import common
from . import lattices

# list of submodules
__all__ = ["common", "lattices"]

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
        "lattices"  : "Provides functionalities for creating and managing lattices."
    }
    return descriptions.get(module_name, "Module not found.")

# ----------------------------------------------------------------------------------------------
# __init__.py
