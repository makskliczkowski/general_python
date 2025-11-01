"""
Convenience wrappers for spectral function calculations.

This module provides convenient access to spectral functions by delegating to
the unified physics backend (../backend.py). All actual implementations are
centralized in spectral_backend.py to avoid code duplication.

For detailed documentation, see spectral_backend.py or backend.py.

-------------------------------------------------------------------------------
File        : general_python/physics/spectral/spectral_function.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
-------------------------------------------------------------------------------
"""

from typing import Optional, Union
import numpy as np

try:
    from ...algebra.utils import JAX_AVAILABLE, Array
except ImportError:
    JAX_AVAILABLE   = False
    Array           = np.ndarray

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# Import from centralized backend to avoid duplication
try:
    from .. import backend as physics_backend
except ImportError:
    physics_backend = None

# =============================================================================
# Spectral Function from Green's Function
# =============================================================================

def spectral_function(greens_function: Array, operator: Optional[Array] = None) -> Array:
    """
    Compute spectral function from Green's function.
    
    Delegates to backend.spectral_function for actual implementation.
    See backend.py or spectral_backend.py for details.
    """
    if physics_backend is None:
        raise ImportError("Physics backend not available")
    return physics_backend.spectral_function(greens_function, operator, backend="default")

def spectral_function_diagonal(omega: float, eigenvalues: Array, eta: float = 0.01) -> Array:
    """
    Compute diagonal spectral function directly from eigenvalues.
    
    Delegates to backend.spectral_function_diagonal for actual implementation.
    See backend.py or spectral_backend.py for details.
    """
    if physics_backend is None:
        raise ImportError("Physics backend not available")
    return physics_backend.spectral_function_diagonal(omega, eigenvalues, eta, backend="default")

def spectral_function_multi_omega(omegas: Array, eigenvalues: Array, eigenvectors: Optional[Array] = None,
        eta: float = 0.01, diagonal_only: bool = False) -> Array:
    """
    Compute spectral function for multiple frequencies.
    
    Delegates to backend.spectral_function_multi_omega for actual implementation.
    See backend.py or spectral_backend.py for details.
    """
    if physics_backend is None:
        raise ImportError("Physics backend not available")
    return physics_backend.spectral_function_multi_omega(omegas, eigenvalues, eigenvectors, eta, diagonal_only, backend="default")

def spectral_function_k_resolved(
        omegas          : Array,
        k_points        : Array,
        eigenvalues_k   : Array,
        eta             : float = 0.01) -> Array:
    """
    Compute momentum-resolved spectral function A(k,\Omega).
    
    Delegates to backend.spectral_function_k_resolved for actual implementation.
    See backend.py or spectral_backend.py for details.
    """
    if physics_backend is None:
        raise ImportError("Physics backend not available")
    return physics_backend.spectral_function_k_resolved(omegas, k_points, eigenvalues_k, eta, backend="default")

def integrated_spectral_weight(
        spectral_function   : Array,
        omega_grid          : Array,
        omega_min           : Optional[float] = None,
        omega_max           : Optional[float] = None) -> Union[float, Array]:
    """
    Compute integrated spectral weight over an energy window.
    
    Delegates to backend.integrated_spectral_weight for actual implementation.
    See backend.py or spectral_backend.py for details.
    """
    if physics_backend is None:
        raise ImportError("Physics backend not available")
    return physics_backend.integrated_spectral_weight(spectral_function, omega_grid, omega_min, omega_max, backend="default")

def find_spectral_peaks(
        spectral_function   : Array,
        omega_grid          : Array,
        threshold           : float = 0.1,
        min_distance        : int = 5) -> Array:
    """
    Find peaks in spectral function.
    
    Delegates to backend.find_spectral_peaks for actual implementation.
    See backend.py or spectral_backend.py for details.
    """
    if physics_backend is None:
        raise ImportError("Physics backend not available")
    return physics_backend.find_spectral_peaks(spectral_function, omega_grid, threshold, min_distance, backend="default")

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Basic spectral functions
    'spectral_function',
    'spectral_function_diagonal',
    'spectral_function_multi_omega',
    
    # Momentum-resolved
    'spectral_function_k_resolved',
    
    # Analysis
    'integrated_spectral_weight',
    'find_spectral_peaks',
]

# ============================================================================
#! End of file
# ============================================================================