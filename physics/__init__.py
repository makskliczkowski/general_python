"""
Physics Module for General Python Utilities.

This module provides utilities for quantum physics simulations and calculations,
including density matrix operations, entropy calculations, eigenstate analysis,
and quantum operator manipulations.

Modules:
--------
- density_matrix : Density matrix operations and utilities
- density_matrix_jax : JAX-optimized density matrix operations
- eigenlevels : Eigenstate and energy level analysis
- entropy : Entropy calculations for quantum systems
- entropy_jax : JAX-optimized entropy calculations
- sp : Single particle physics utilities
- __operators__ : Quantum operator definitions and operations

Examples:
---------
>>> from general_python.physics import density_matrix
>>> from general_python.physics import entropy
>>> 
>>> # Calculate entropy of a quantum state
>>> rho = density_matrix.create_density_matrix(psi)
>>> s = entropy.von_neumann_entropy(rho)

Author: Maksymilian Kliczkowski
License: MIT
"""

# Import main physics modules
from . import sp as single_particle
from . import density_matrix
from . import eigenlevels
from . import entropy

# Try to import JAX versions if available
try:
    from . import density_matrix_jax
    from . import entropy_jax
    _jax_available = True
except ImportError:
    _jax_available = False

# Define what's available when importing with "from general_python.physics import *"
__all__ = [
    'single_particle',
    'density_matrix', 
    'eigenlevels',
    'entropy',
]

if _jax_available:
    __all__.extend(['density_matrix_jax', 'entropy_jax'])

__version__ = '0.1.0'
__author__  = 'Maksymilian Kliczkowski'
__email__   = 'maksymilian.kliczkowski@pwr.edu.pl'