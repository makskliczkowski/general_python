'''
This module provides utility functions for the general Python machine learning framework.

It includes functions for initializing weights, creating activation functions, and handling
activation function parameters. The module supports both JAX and NumPy backends, allowing for
flexible use in different environments.

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
except ImportError as e:
    raise ImportError("Failed to import net_init, net_init_jax, or net_init_np modules. Ensure general_python package is correctly installed.") from e

__all__ = [
    'net_init',
    'jaxpy',
    'numpy',
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