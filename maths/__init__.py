"""
This module contains mathematical utilities and functions.
"""

from . import __math__ as MathMod
from . import __random__ as RandomMod
from . import statistics as StatisticsMod

__all__ = ["MathMod", "RandomMod", "StatisticsMod"]

# Importing modules from the maths package

def get_module_description(module_name):
    """
    Get the description of a specific module in the maths package.
    
    Parameters:
    - module_name (str): The name of the module.
    
    Returns:
    - str: The description of the module.
    """
    descriptions = {
        "MathMod": "Provides general mathematical functions and utilities.",
        "RandomMod": "Provides functions for random number generation.",
        "StatisticsMod": "Provides statistical functions and utilities."
    }
    return descriptions.get(module_name, "Module not found.")

def list_available_modules():
    """
    List all available modules in the maths package.
    
    Returns:
    - list: A list of available module names.
    """
    return ["MathMod", "RandomMod", "StatisticsMod"]

# Example usage
# print(get_module_description("MathMod"))
# print(list_available_modules())
