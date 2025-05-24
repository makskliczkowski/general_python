"""
Mathematics Module for General Python Utilities.

This module provides comprehensive mathematical utilities including general math functions,
high-quality random number generators, and statistical analysis tools.

Modules:
--------
- math_utils : General mathematical functions and utilities
- __random__ : High-quality pseudorandom number generators (e.g., Xoshiro256)
- statistics : Statistical functions and data analysis utilities

Examples:
---------
>>> from general_python.maths import math_utils as MathMod
>>> from general_python.maths import __random__ as RandomMod
>>> from general_python.maths import statistics as StatisticsMod
>>> 
>>> # Use high-quality random number generator
>>> rng = RandomMod.Xoshiro256(seed=42)
>>> values = [rng.random() for _ in range(5)]
>>> 
>>> # Statistical analysis
>>> mean_val = StatisticsMod.mean(values)

Author: Maksymilian Kliczkowski
License: MIT
"""

from . import math_utils as MathMod
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
