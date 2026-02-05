"""Common utilities for data, plotting, logging, and files.

Purpose
-------
Provide shared helpers used across QES and general_python for plotting, IO,
logging, and small data utilities. Imports are lazy to keep startup time low.

Input/output contracts
----------------------
- File and directory helpers accept path-like strings and return validated paths
  or file handles.
- Plotting helpers expect numeric arrays with shapes compatible with matplotlib.
- Logging utilities return logger instances configured for QES conventions.

Dtype and shape expectations
----------------------------
- Plotting helpers expect float or complex arrays with standard 1D or 2D shapes.
- Binary utilities operate on integer scalars or 1D integer arrays.

Numerical stability notes
-------------------------
- Formatting and plotting should not be used for high-precision validation;
  use numeric assertions in analysis code instead.

Determinism notes
-----------------
- Most helpers are deterministic given fixed inputs.
- Log timestamps and file system ordering can vary across runs and platforms.
"""

import  importlib
from    typing import TYPE_CHECKING

# For static type checking (IDE support) without runtime import
if TYPE_CHECKING:
    from .directories   import Directories
    from .plot          import (
                            Plotter, PlotterSave, MatrixPrinter,
                            colorsCycle, colorsCycleBright, colorsCycleDark, colorsList,
                            linestylesCycle, linestylesCycleExtended, linestylesList,
                            markersCycle, markersList
                        )
    from .datah         import DataHandler
    from .binary        import (
                            ctz64, popcount64,
                            mask_from_indices, indices_from_mask,
                            complement_mask, complement_indices
                        )
    from .hdf5man       import HDF5Manager
    from .flog          import Logger, get_global_logger
    from .memory        import log_memory_status, check_memory_for_operation
    from .timer         import Timer

# Lazy loading registry
_LAZY_IMPORTS = {
    # directories
    'Directories'               : ('.directories', 'Directories'),
    # plot
    'Plotter'                   : ('.plot', 'Plotter'),
    'PlotterSave'               : ('.plot', 'PlotterSave'),
    'MatrixPrinter'             : ('.plot', 'MatrixPrinter'),
    'colorsCycle'               : ('.plot', 'colorsCycle'),
    'colorsCycleBright'         : ('.plot', 'colorsCycleBright'),
    'colorsCycleDark'           : ('.plot', 'colorsCycleDark'),
    'colorsList'                : ('.plot', 'colorsList'),
    'linestylesCycle'           : ('.plot', 'linestylesCycle'),
    'linestylesCycleExtended'   : ('.plot', 'linestylesCycleExtended'),
    'linestylesList'            : ('.plot', 'linestylesList'),
    'markersCycle'              : ('.plot', 'markersCycle'),
    'markersList'               : ('.plot', 'markersList'),
    # datah
    'DataHandler'               : ('.datah', 'DataHandler'),
    # binary - Numba-safe bit operations
    'ctz64'                     : ('.binary', 'ctz64'),
    'popcount64'                : ('.binary', 'popcount64'),
    'mask_from_indices'         : ('.binary', 'mask_from_indices'),
    'indices_from_mask'         : ('.binary', 'indices_from_mask'),
    'complement_mask'           : ('.binary', 'complement_mask'),
    'complement_indices'        : ('.binary', 'complement_indices'),  # alias
    # hdf5
    'HDF5Manager'               : ('.hdf5man', 'HDF5Manager'),
    # logging
    'Logger'                    : ('.flog', 'Logger'),
    'get_global_logger'         : ('.flog', 'get_global_logger'),
    # memory
    'log_memory_status'         : ('.memory', 'log_memory_status'),
    'check_memory_for_operation': ('.memory', 'check_memory_for_operation'),
    # timer
    'Timer'                     : ('.timer', 'Timer'),
}

# Cache for loaded modules
_LOADED = {}

def __getattr__(name: str):
    """Lazy import handler - loads modules only when accessed."""
    if name in _LAZY_IMPORTS:
        if name not in _LOADED:
            module_path, attr_name  = _LAZY_IMPORTS[name]
            module                  = importlib.import_module(module_path, package=__name__)
            _LOADED[name]           = getattr(module, attr_name)
        return _LOADED[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """List available attributes for autocompletion."""
    return list(_LAZY_IMPORTS.keys()) + ['get_module_description', 'list_available_modules']

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
        "hdf5man"       : "Manages reading and writing HDF5 files.",
        "flog"          : "Provides logging functionalities for structured output.",
        "memory"        : "Memory monitoring and management utilities.",
        "timer"         : "Context manager and utilities for performance timing."
    }
    return descriptions.get(module_name, "Module not found.")

def list_available_modules():
    """
    List all available modules in the common package.

    Returns:
    - list: A list of available module names.
    """
    return ["binary", "directories", "plot", "datah", "hdf5man", "flog", "memory", "timer"]

# Expose all lazy-loadable names for `from common import *`
__all__ = list(_LAZY_IMPORTS.keys()) + ['get_module_description', 'list_available_modules']

####################################################################################################
#! EOF
####################################################################################################
