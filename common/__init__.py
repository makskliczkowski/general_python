# //general_python/common/__init__.py

"""
Common Utilities Module

This module contains the following submodules:
- binary: Handles binary data operations.
- directories: Manages directory operations.
- plot: Provides visualization utilities.
- datah: Handles data operations.
- hdf5_lib: Reads and writes HDF5 files.
- flog: Provides logging functionalities.
"""

import sys

####################################################################################################

# Lazily import submodules
from . import binary
from . import directories
from . import plot
from . import datah
from . import hdf5_lib
from . import flog

####################################################################################################

def get_global_logger():
    """
    Lazily loads and returns the global logger instance.

    Returns:
    - Logger instance from flog module.
    """
    return flog.get_global_logger()

def get_module_description(module_name):
    """
    Get the description of a specific module in the common package.

    Parameters:
    - module_name (str): The name of the module.

    Returns:
    - str: The description of the module.
    """
    descriptions = {
        "binary"        : "Handles binary data operations.",
        "directories"   : "Provides the Directories class for managing directory structures.",
        "plot"          : "Provides visualization utilities (Plotter, PlotterSave, MatrixPrinter).",
        "datah"         : "Handles various data operations and transformations.",
        "hdf5_lib"      : "Manages reading and writing HDF5 files.",
        "flog"          : "Provides logging functionalities for structured output."
    }
    return descriptions.get(module_name, "Module not found.")

def list_available_modules():
    """
    List all available modules in the common package.

    Returns:
    - list: A list of available module names.
    """
    return list(get_module_description("").keys())

# Example usage
# print(get_module_description("__directories__"))
# print(list_available_modules())
# print(Directories)
# - Directories class is not imported directly
# print(Plotter)
# print(PlotterSave)
# print(MatrixPrinter)
# - MatrixPrinter class is not imported directly