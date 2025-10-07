'''
This module provides utility functions for the general Python machine learning framework.

It includes functions for initializing weights, creating activation functions, and handling
activation function parameters. The module supports both JAX and NumPy backends, allowing for
flexible use in different environments.
'''

from .... import ml.net_impl.utils.net_init as net_init
from .... import ml.net_impl.utils.net_init_jax as jaxpy
from .... import ml.net_impl.utils.net_init_np as numpy

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
