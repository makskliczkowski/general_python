'''
Utility helpers for the network implementation layer.

This package includes initializers, activation helpers, backend-specific
utilities, and small state-representation helpers used by the general network
wrappers. 

The JAX wrappers can now describe input conventions explicitly
through ``input_is_spin`` and ``input_value`` instead of duplicating local
representation conversions. This is used 
in the Variational Monte Carlo ansatze and the MLP wrapper, while the RBM and CNN
backbones stay agnostic to the meaning of the samples they process.

-----------------------------------------------------------------
File        : general_python/ml/net_impl/utils/__init__.py
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
date        : 2025-11-01
------------------------------------------------------------------
'''

try:
    from . import net_init
    from . import net_init_jax as jaxpy
    from . import net_init_np as numpy
    from . import net_state_repr_jax
    from . import net_wrapper_utils
except ImportError as e:
    raise ImportError("Failed to import ML utility modules. Ensure general_python package is correctly installed.") from e

__all__ = [
    'net_init',
    'jaxpy',
    'numpy',
    'net_state_repr_jax',
    'net_wrapper_utils',
]

__version__     = '0.1.0'
__author__      = 'Maksymilian Kliczkowski'
__email__       = 'maksymilian.kliczkowski@pwr.edu.pl'
__status__      = 'Development'
__license__     = 'MIT'
__copyright__   = 'Copyright (c) 2025 Maksymilian Kliczkowski'
__description__ = 'General Python Machine Learning Framework Utilities using JAX and NumPy'

# ----------------------------------------------------------------
#! End of file
# ---------------------------------------------------------------
