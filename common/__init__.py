"""
Common utilities for file handling, plotting, logging, binary operations, and data management.

This module provides essential utility functions and classes for general-purpose tasks including:

**File and Directory Management:**
- Directory operations and path management
- File I/O utilities and data handling

**Plotting and Visualization:**
- Advanced plotting utilities with customizable styles
- Matrix printing and data visualization tools
- Color schemes, line styles, and marker collections

**Data Processing:**
- Data handling utilities for scientific computing
- Matrix operations and data transformation
- Statistical analysis tools

**Binary Operations:**
- Binary data manipulation and conversion utilities
- Bit-level operations for quantum computing applications

**Logging and Monitoring:**
- Advanced logging systems with multiple output formats
- Performance monitoring and debugging tools

Example:
    >>> from general_python.common import Plotter, DataHandler, Directories
    >>> plotter = Plotter()
    >>> data_handler = DataHandler()
    >>> dirs = Directories()
"""

# Importing modules from the common package
from .directories import Directories

#! Plotting and visualization utilities
from .plot import (
    Plotter, PlotterSave, MatrixPrinter,
    colorsCycle, colorsCycleBright, colorsCycleDark, colorsList,
    linestylesCycle, linestylesCycleExtended, linestylesList,
    markersCycle, markersList
)
from .datah import DataHandler, complement_indices, indices_from_mask
from .hdf5man import HDF5Manager
# Description of the modules
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


####################################################################################################

# # Lazily import submodules
# from . import binary
# from . import directories
# from . import hdf5_lib
# from general_python.common.flog import get_global_logger

####################################################################################################

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

######################################################################################################

