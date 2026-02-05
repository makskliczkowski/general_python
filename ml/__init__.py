"""Machine-learning entry points for neural-network workflows.

This package exposes network factories, schedulers, and low-level implementations
used in variational and supervised experiments.

Backend expectations
--------------------
Most training-oriented models are JAX/Flax-first. NumPy may be used for
pre/post-processing, but model forward/backward passes generally expect JAX
arrays when using Flax-based networks.

Input/output contracts
----------------------
Network factories consume model identifiers plus shape metadata (for example
``input_shape=(n_features,)``), dtype controls, and optional seeds. Outputs are
model objects compatible with project training utilities.

Determinism and numerical notes
-------------------------------
Determinism depends on seeded PRNG keys and backend execution settings. On
accelerators, reduction order and mixed precision can produce small numerical
variations versus CPU runs.
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
