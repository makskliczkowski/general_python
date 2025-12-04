"""
Utilities submodule for algebra operations.

This module provides specialized mathematical utilities like
Pfaffian and Hafnian calculations with support for JAX backends.

Uses lazy imports to minimize startup overhead.
"""

from typing import TYPE_CHECKING
import importlib

# -----------------------------------------------------------------------------------------------
# Lazy Import Configuration
# -----------------------------------------------------------------------------------------------

_LAZY_IMPORTS = {
    'pfaffian'      : ('.pfaffian', None),
    'pfaffian_jax'  : ('.pfaffian_jax', None),
    'hafnian'       : ('.hafnian', None),
    'hafnian_jax'   : ('.hafnian_jax', None),
}

_LAZY_CACHE = {}

# For type checking only
if TYPE_CHECKING:
    from . import pfaffian
    from . import pfaffian_jax
    from . import hafnian
    from . import hafnian_jax


def __getattr__(name: str):
    """
    Module-level __getattr__ for lazy imports.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_path, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_path, package=__name__)
    
    if attr_name is None:
        result = module
    else:
        result = getattr(module, attr_name)
    
    _LAZY_CACHE[name] = result
    return result


__all__ = [
    'pfaffian',
    'pfaffian_jax',
    'hafnian',
    'hafnian_jax',
]