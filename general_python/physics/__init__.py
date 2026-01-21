r"""
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
- response.susceptibility       : Magnetic and charge susceptibilities chi(q,\Omega)

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
>>> from general_python.physics.spectral import greens_function, spectral_function
>>> from general_python.physics.spectral.spectral_backend import greens_function_quadratic
>>> G = greens_function_quadratic(omega, eigenvalues, eigenvectors)
>>> A = spectral_function.spectral_function(greens_function=G)

>>> # Response functions
>>> from general_python.physics.response import structure_factor, susceptibility
>>> S_q_omega = structure_factor.structure_factor_spin(gs, eigvals, eigvecs, S_q, omega_grid)
>>> chi       = susceptibility.magnetic_susceptibility(eigvals, eigvecs, M_q, omega_grid)

>>> # Statistical analysis
>>> from general_python.physics import statistical
>>> smooth_data = statistical.moving_average(noisy_data, window_size=10)
>>> ldos_vals   = statistical.ldos(energies, overlaps)

File    : general_python/physics/__init__.py
Version : 0.1.0
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
License : MIT
"""

import sys
import importlib
from typing import TYPE_CHECKING

# -----------------------------------------------------------------------------------------------
# Lazy Import Configuration
# -----------------------------------------------------------------------------------------------

# Mapping of attribute names to (module_relative_path, attribute_name_in_module)
_LAZY_IMPORTS = {
    # Basic modules
    "density_matrix"    : (".density_matrix",       None),
    "eigenlevels"       : (".eigenlevels",          None),
    "entropy"           : (".entropy",              None),
    "operators"         : (".operators",            None),
    
    # Statistical & thermal
    "statistical"       : (".statistical",          None),
    "thermal"           : (".thermal",              None),
    
    # Subpackages
    "spectral"          : (".spectral",             None),
    "response"          : (".response",             None),
    
    # Specialized
    "single_particle"   : (".sp",                   None),
    "sp"                : (".sp",                   None), # Alias
    
    # JAX versions (will fail on access if JAX not installed, handled in caller/user code)
    "density_matrix_jax": (".density_matrix_jax",   None),
    "entropy_jax"       : (".entropy_jax",          None),
}

# Cache for lazily loaded modules/attributes
_LAZY_CACHE = {}

if TYPE_CHECKING:
    from . import density_matrix
    from . import eigenlevels
    from . import entropy
    from . import operators
    from . import statistical
    from . import thermal
    from . import spectral
    from . import response
    from . import sp as single_particle
    from . import sp
    try:
        from . import density_matrix_jax
        from . import entropy_jax
    except ImportError:
        pass

# -----------------------------------------------------------------------------------------------
# Lazy Import Implementation
# -----------------------------------------------------------------------------------------------

def _lazy_import(name: str):
    """
    Lazily import a module or attribute based on _LAZY_IMPORTS configuration.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_path, attr_name = _LAZY_IMPORTS[name]
    
    try:
        module = importlib.import_module(module_path, package=__name__)
        
        if attr_name is None:
            result = module
        else:
            result = getattr(module, attr_name)
        
        _LAZY_CACHE[name] = result
        return result
    except ImportError as e:
        raise ImportError(f"Failed to import lazy module '{name}' from '{module_path}': {e}") from e

def __getattr__(name: str):
    return _lazy_import(name)

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))

__all__ = list(_LAZY_IMPORTS.keys()) + ["list_capabilities"]

# -----------------------------------------------------------------------------------------------
# Capabilities
# -----------------------------------------------------------------------------------------------

def list_capabilities():
    """
    List available physics capabilities and modules.
    """
    # Check JAX availability dynamically
    try:
        import jax
        jax_available = True
    except ImportError:
        jax_available = False

    capabilities = {
        'modules': [
            'density_matrix', 'eigenlevels', 'entropy', 
            'operators', 'statistical', 'thermal', 'single_particle'
        ],
        'spectral': ['dos', 'greens', 'spectral_function'],
        'response': ['structure_factor', 'susceptibility'],
        'jax_available': jax_available
    }
    return capabilities

__version__ = '0.1.0'
__author__  = 'Maksymilian Kliczkowski'
__email__   = 'maksymilian.kliczkowski@pwr.edu.pl'

# ------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------