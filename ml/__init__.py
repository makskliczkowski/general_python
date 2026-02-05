"""Machine learning utilities for QES workflows.

Purpose
-------
Provide network factories, schedulers, and training helpers used in NQS/VMC
pipelines with a focus on quantum-physics data.

Input/output contracts
----------------------
- Network factories expect ``input_shape`` and backend-specific dtype settings.
- Training helpers expect batched arrays and return loss scalars or metrics.
- Flax-based models return parameter PyTrees and pure forward functions.

Dtype and shape expectations
----------------------------
- Inputs are typically shape ``(batch, features)`` or ``(batch, sites, ...)``.
- Complex dtypes are common for wavefunction models; keep dtype consistent
  across preprocessing and model evaluation.

Numerical stability notes
-------------------------
- Log-amplitude and normalization paths can underflow for large systems; use
  float64 where possible and monitor gradient norms.
- Schedulers and optimizers assume finite gradients; clip or rescale if needed.

Determinism notes
-----------------
- JAX-based models require explicit PRNG keys for reproducible initialization.
- Parallel training can introduce nondeterministic reductions on some backends.
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
