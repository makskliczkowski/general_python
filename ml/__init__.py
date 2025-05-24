"""
Machine Learning Module for General Python Utilities.

This module provides machine learning utilities with support for both JAX and NumPy backends,
including neural network implementations, training utilities, optimizers, and loss functions.

Modules:
--------
- networks : Neural network implementations with flexible backends
- schedulers : Learning rate schedulers and optimization utilities  
- __general__ : General ML parameters and training utilities
- __loss__ : Loss functions for various ML tasks
- net_impl : Low-level network implementation utilities

Examples:
---------
>>> from general_python.ml import networks
>>> from general_python.ml import __general__ as ml_general
>>> 
>>> # Create ML parameters
>>> params = ml_general.MLParams(
...     epo=100, batch=32, lr=0.001, reg={}, loss='mse',
...     fNum=10, shape=(10,), optimizer='adam'
... )

Author: Maksymilian Kliczkowski
License: MIT
"""

# Import main ML modules
from . import networks
from . import schedulers
from . import __general__
from . import __loss__
from . import net_impl

# Define what's available when importing with "from general_python.ml import *"
__all__ = [
    'networks',
    'schedulers', 
    '__general__',
    '__loss__',
    'net_impl'
]

__version__ = '0.1.0'
__author__ = 'Maksymilian Kliczkowski'
__email__ = 'maksymilian.kliczkowski@pwr.edu.pl'
