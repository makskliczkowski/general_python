'''
Utility helpers for the network implementation layer.

This package includes initializers, activation helpers, backend-specific
utilities, and small wrapper helpers used by the general network layer.

-----------------------------------------------------------------
File        : general_python/ml/net_impl/utils/__init__.py
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
date        : 2025-11-01
------------------------------------------------------------------
'''

try:
    from . import net_init
    from . import net_init_jax  as jaxpy
    from . import net_init_np   as numpy
    from . import net_state_repr_jax
    from . import net_wrapper_utils
    from .net_init_jax import get_initializer
except ImportError as e:
    raise ImportError("Failed to import ML utility modules. Ensure general_python package is correctly installed.") from e

__all__ = [
    'net_init',
    'jaxpy',
    'numpy',
    'net_state_repr_jax',
    'net_wrapper_utils',
    'get_initializer'
]

__version__     = '1.0.0'
__author__      = 'Maksymilian Kliczkowski'
__email__       = 'maksymilian.kliczkowski@pwr.edu.pl'
__status__      = 'Development'
__license__     = 'MIT'
__copyright__   = 'Copyright (c) 2025 Maksymilian Kliczkowski'
__description__ = 'General Python Machine Learning Framework Utilities using JAX and NumPy'

# ----------------------------------------------------------------
#! End of file
# ---------------------------------------------------------------
