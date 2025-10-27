"""
Physics Module for General Python Utilities.

This module provides comprehensive utilities for quantum physics simulations and calculations,
organized by physical concepts and applications.

Organization:
-------------

**Basic Quantum States & Properties:**
- density_matrix                : Density matrix operations and utilities
    - density_matrix_jax        : JAX-optimized density matrix operations
- eigenlevels                   : Eigenstate and energy level analysis
- entropy                       : Entropy calculations for quantum systems (von Neumann, Rényi, etc.)
    - entropy_jax               : JAX-optimized entropy calculations
- operators                     : Quantum operator definitions and spectral analysis

**Statistical Analysis:**
- statistical                   : Windowing, averaging, time series analysis for quantum data
    - Finite window averages and moving statistics
    - Local density of states (LDOS) and strength functions
    - Energy window utilities for spectral analysis

**Thermal Physics:**
- thermal                       : Thermal quantum physics and statistical mechanics
    - Partition functions and Boltzmann weights
    - Thermal averages and expectation values
    - Thermodynamic quantities (free energy, heat capacity, entropy)
    - Magnetic and charge susceptibilities

**Spectral Functions (subpackage):**
- spectral.dos                  : Density of states (histogram and Gaussian-broadened)
- spectral.greens               : Green's functions G(\Omega) and Fourier transforms
- spectral.spectral_function    : Spectral functions A(k,\Omega) = -Im[G]/\pi

**Response Functions (subpackage):**
- response.structure_factor     : Dynamic structure factor S(q,\Omega) for spins (optimized!)
- response.susceptibility       : Magnetic and charge susceptibilities χ(q,\Omega)

**Specialized:**
- sp                            : Single particle physics utilities

Examples:
---------
>>> # Entropy calculation
>>> from general_python.physics import density_matrix, entropy
>>> rho = density_matrix.create_density_matrix(psi)
>>> s   = entropy.von_neumann_entropy(rho)

>>> # Thermal physics
>>> from general_python.physics import thermal
>>> Z       = thermal.partition_function(energies, beta=1.0)
>>> U       = thermal.internal_energy(energies, beta=1.0)
>>> C_V     = thermal.heat_capacity(energies, beta=1.0)

>>> # Spectral functions
>>> from general_python.physics.spectral import dos, greens, spectral_function
>>> dos_hist, edges = dos.dos_histogram(energies, bins=50)
>>> G               = greens.greens_function_eigenbasis(omega, eigenvalues, eigenvectors)
>>> A               = spectral_function.spectral_function(G)

>>> # Response functions
>>> from general_python.physics.response import structure_factor, susceptibility
>>> S_q_omega = structure_factor.structure_factor_spin(gs, eigvals, eigvecs, S_q, omega_grid)
>>> chi       = susceptibility.magnetic_susceptibility(eigvals, eigvecs, M_q, omega_grid)

>>> # Statistical analysis
>>> from general_python.physics import statistical
>>> smooth_data = statistical.moving_average(noisy_data, window_size=10)
>>> ldos_vals   = statistical.ldos(energies, overlaps)

File    : QES/general_python/physics/__init__.py
Version : 0.1.0
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
License : MIT
"""

# Import main physics modules
from . import sp as single_particle
from . import density_matrix
from . import eigenlevels
from . import entropy
from . import operators

# New modules
from . import statistical
from . import thermal

# Subpackages
from . import spectral
from . import response

# Try to import JAX versions if available
try:
    from . import density_matrix_jax
    from . import entropy_jax
    _jax_available = True
except ImportError:
    _jax_available = False

# Define what's available when importing with "from general_python.physics import *"
__all__ = [
    # Basic quantum states
    'single_particle',
    'density_matrix', 
    'eigenlevels',
    'entropy',
    'operators',
    
    # Statistical & thermal
    'statistical',
    'thermal',
    
    # Subpackages
    'spectral',
    'response',
]

if _jax_available:
    __all__.extend(['density_matrix_jax', 'entropy_jax'])


# Convenience function to list available capabilities
def list_capabilities():
    """
    List available physics capabilities and modules.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'modules'         : list of available top-level modules
        - 'spectral'        : list of spectral submodules
        - 'response'        : list of response submodules
        - 'jax_available'   : whether JAX acceleration is available
    """
    capabilities = {
        'modules': [
            'density_matrix',
            'eigenlevels',
            'entropy',
            'operators',
            'statistical',
            'thermal',
            'single_particle'
        ],
        'spectral': [
            'dos',
            'greens',
            'spectral_function'
        ],
        'response': [
            'structure_factor',
            'susceptibility'
        ],
        'jax_available': _jax_available
    }
    
    return capabilities


__version__ = '0.1.0'
__author__  = 'Maksymilian Kliczkowski'
__email__   = 'maksymilian.kliczkowski@pwr.edu.pl'

# ------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------