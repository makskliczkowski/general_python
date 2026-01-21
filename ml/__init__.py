"""
Machine Learning Module for General Python Utilities.

This module provides machine learning utilities with support for both JAX and NumPy backends,
including neural network implementations, training utilities, optimizers, and loss functions.

---------------------------------
File            : general_python/ml/__init__.py
Author          : Maksymilian Kliczkowski
License         : MIT
---------------------------------
"""

import importlib

# Import main ML modules
try:
    from . import networks
    from . import schedulers
except ImportError as e:
    raise Exception(f"Could not import {e}")

# Lazy import aliases for common submodules

_LAZY_MODULES = {
    'activation_functions'      : 'general_python.ml.net_impl.activation_functions',
    'interface_net_flax'        : 'general_python.ml.net_impl.interface_net_flax',
    'net_general'               : 'general_python.ml.net_impl.net_general',
    'net_simple'                : 'general_python.ml.net_impl.net_simple',
    'networks'                  : 'general_python.ml.networks',
    'schedulers'                : 'general_python.ml.schedulers',
    'training_phases'           : 'general_python.ml.training_phases',
    # Add more aliases as needed
}

def __getattr__(name):
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = list(_LAZY_MODULES.keys()) + [
    'networks',
    'schedulers', 
    '__general__',
    '__loss__',
    'net_impl'
]

# --------------------------------------------------------------
# Direct import of net_impl for easier access
# --------------------------------------------------------------

from . import net_impl

# Define what's available when importing with "from general_python.ml import *"
__all__ = [
    'networks',
    'schedulers', 
    '__general__',
    '__loss__',
    'net_impl'
]

__version__     = '0.1.0'
__author__      = 'Maksymilian Kliczkowski'
__email__       = 'maksymilian.kliczkowski@pwr.edu.pl'

# --------------------------------------------------------------
# End of File
# --------------------------------------------------------------