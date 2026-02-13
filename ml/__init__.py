"""Machine-learning entry points for neural-network workflows.

The package collects model registries, scheduler utilities, and concrete network
implementations used in supervised and variational experiments.

Purpose
-------
Use this namespace to obtain model constructors and training-time utilities
without importing every backend-specific implementation eagerly.

Input/output contracts
----------------------
Public factories typically accept model identifiers, shape metadata (for
example ``input_shape=(n_features,)``), and optional dtype and seed arguments.
Returned objects are model instances or callables compatible with the training
helpers in :mod:`general_python.ml.training_phases`.

dtype and shape expectations
----------------------------
Input batches are conventionally rank-2 arrays with shape ``(batch, features)``
unless a model documents an image or sequence layout. For stable optimization,
``float32`` is the practical default on accelerators, while ``float64`` may be
required for high-precision experiments.

Numerical stability and determinism
-----------------------------------
Training trajectories depend on initialization, optimizer state, and operation
ordering. For reproducibility, fix random seeds, keep backend/device constant,
and avoid mixing precision policies within one experiment.
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

# ------------------------------------------------------------------------------
# Direct import of net_impl for easier access
# ------------------------------------------------------------------------------

from . import net_impl

# Define what's available when importing with "from general_python.ml import *"
__all__ = [
    'networks',
    'schedulers',
    'net_impl'
]

__version__     = '0.1.0'
__author__      = 'Maksymilian Kliczkowski'
__email__       = 'maksymilian.kliczkowski@pwr.edu.pl'

# ------------------------------------------------------------------------------
# End of File
# ------------------------------------------------------------------------------
