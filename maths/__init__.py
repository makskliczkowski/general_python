"""Mathematical utilities for general_python.

Purpose
-------
Provide mathematical helpers, random-matrix routines, and statistics utilities
used throughout QES and related tooling.

Input/output contracts
----------------------
- Functions accept numeric scalars or arrays and return scalars or arrays with
  shapes consistent with the inputs.
- Random-matrix utilities expect explicit sizes and optional RNG controls.

Dtype and shape expectations
----------------------------
- Operations preserve input dtype when practical (float or complex).
- Random-matrix helpers return arrays with the requested shape.

Numerical stability notes
-------------------------
- Statistical estimators and polynomial transforms can be sensitive to scale.
  Normalize inputs when working with large dynamic ranges.

Determinism notes
-----------------
- Random-matrix utilities are deterministic only when an explicit RNG seed or
  key is provided.
"""

import  sys
import  importlib
from    typing import TYPE_CHECKING, List

# Description used by general_python.registry
MODULE_DESCRIPTION = (
    "Mathematical utilities, random number generators, and statistical analysis tools."
)

# -----------------------------------------------------------------------------------------------
# Lazy Import Configuration
# -----------------------------------------------------------------------------------------------

# Mapping of attribute names to (module_relative_path, attribute_name_in_module)
_LAZY_IMPORTS = {
    # Modules
    "math_utils"    : (".math_utils",   None),
    "random"        : (".random",       None),
    "statistics"    : (".statistics",   None),
    
    # Aliases for backward compatibility or convenience (optional)
    "MathMod"       : (".math_utils",   None),
    "RandomMod"     : (".random",       None),
    "StatisticsMod" : (".statistics",   None),
}

# Cache for lazily loaded modules/attributes
_LAZY_CACHE = {}

if TYPE_CHECKING:
    from . import math_utils
    from . import random
    from . import statistics
    # Aliases
    from . import math_utils as MathMod
    from . import random as RandomMod
    from . import statistics as StatisticsMod

# -----------------------------------------------------------------------------------------------
# Lazy Import Implementation
# -----------------------------------------------------------------------------------------------

def _lazy_import(name: str):
    """
    Lazily import a module or attribute based on _LAZY_IMPORTS configuration.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_path, attr_name = _LAZY_IMPORTS[name]
    
    try:
        module = importlib.import_module(module_path, package=__name__)
        
        if attr_name is None:
            result = module
        else:
            result = getattr(module, attr_name)
        
        _LAZY_CACHE[name] = result
        return result
    except ImportError as e:
        raise ImportError(f"Failed to import lazy module '{name}' from '{module_path}': {e}") from e

def __getattr__(name: str):
    return _lazy_import(name)

def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))

def get_module_description(module_name: str) -> str:
    """Return a description for a maths submodule."""
    _descriptions = {
        "math_utils": "General mathematical functions and utilities.",
        "random": "High-quality pseudorandom number generators and CUE matrices.",
        "statistics": "Statistical functions and data analysis utilities.",
    }
    # Handle aliases
    real_name = _LAZY_IMPORTS.get(module_name, (None, None))[0]
    if real_name:
        real_name = real_name.lstrip(".")
        return _descriptions.get(real_name, "Module not found.")
    return "Module not found."

def list_available_modules() -> List[str]:
    """Return the list of available maths submodules."""
    return sorted([k for k in _LAZY_IMPORTS.keys() if not k.endswith("Mod")])

__all__ = list(_LAZY_IMPORTS.keys()) + ["get_module_description", "list_available_modules"]

# -------------------------------------------------------------------------
#! EOF
# -------------------------------------------------------------------------
