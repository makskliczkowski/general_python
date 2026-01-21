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

# Import from spectral backend
try:
    from .spectral_backend import (
        spectral_function as _spectral_function_backend,
        spectral_function_k_resolved as _spectral_function_k_resolved_backend,
        integrated_spectral_weight as _integrated_spectral_weight_backend,
        find_spectral_peaks as _find_spectral_peaks_backend,
        greens_function_manybody_finite_T as _greens_function_manybody_finite_T,
        greens_function_quadratic_finite_T as _greens_function_quadratic_finite_T,
        greens_function_lanczos_finite_T as _greens_function_lanczos_finite_T,
    )
except ImportError:
    raise ImportError("spectral_backend module not available")

# =============================================================================
# Spectral Function Calculations
# =============================================================================

def spectral_function(
        greens_function     : Optional[Array]   = None,
        omega               : Optional[Union[float, Array]] = None,
        eta                 : float             = 0.01,
        *,
        # For quadratic systems (if greens_function is None)
        eigenvalues         : Optional[Array]   = None,
        eigenvectors        : Optional[Array]   = None,
        operator_a          : Optional[Array]   = None,
        operator_b          : Optional[Array]   = None,
        occupations         : Optional[Array]   = None,
        basis_transform     : bool              = True,
        # For many-body systems (if greens_function is None)
        mb_states           : Optional[Union[Array, int]] = None,
        # Other options
        backend             : str               = "default",
        kind                : str               = "retarded") -> Array:
    r"""
    Compute spectral function.
    
    If greens_function is provided:
        A(\omega) = -(1/\pi) Im[G(\omega)]
    
    Otherwise, compute from eigenvalues/eigenvectors:
        - Quadratic systems: via greens_function_quadratic
        - Many-body systems: via greens_function_manybody
    
    Parameters
    ----------
    greens_function : array-like, optional
        Pre-computed Green's function. If provided, spectral function is
        computed directly as A = -(1/pi) Im[G].
    omega : float or array-like, optional
        Frequency or frequencies (required if greens_function is None).
    eta : float, optional
        Broadening parameter (default: 0.01).
    eigenvalues : array-like, optional
        System eigenvalues (required if greens_function is None).
    eigenvectors : array-like, optional
        System eigenvectors (for quadratic systems).
    operator_a : array-like, optional
        Operator for Green's function calculation.
    operator_b : array-like, optional
        Second operator (if different from operator_a).
    occupations : array-like, optional
        Occupation numbers (for quadratic systems).
    basis_transform : bool, optional
        Whether to transform operators to eigenbasis (default: True).
    mb_states : array-like or int, optional
        Many-body states for Lehmann representation.
    backend : str, optional
        Numerical backend (default: "default").
    kind : str, optional
        Type of Green's function (default: "retarded").
        
    Returns
    -------
    Array
        Spectral function A(\omega), always real and non-negative.
        
    Examples
    --------
    >>> # From pre-computed Green's function
    >>> A = spectral_function(greens_function=G)
    >>> 
    >>> # From eigenvalues (quadratic system)
    >>> A = spectral_function(omega=omegas, eigenvalues=E, eigenvectors=U, eta=0.01)
    >>> 
    >>> # From eigenvalues (many-body system)  
    >>> A = spectral_function(omega=omegas, eigenvalues=E, operator_a=O, mb_states=states)
    """
    return _spectral_function_backend(
        greens_function     = greens_function,
        backend             = backend,
        eta                 = eta,
        kind                = kind,
        omega               = omega,
        q_eigenvalues       = eigenvalues,
        q_eigenvectors      = eigenvectors,
        q_operator_a        = operator_a,
        q_operator_b        = operator_b,
        q_occupations       = occupations,
        q_basis_transform   = basis_transform,
        mb_eigenvalues      = eigenvalues if mb_states is not None else None,
        mb_operator_a       = operator_a if mb_states is not None else None,
        mb_operator_b       = operator_b if mb_states is not None else None,
        mb_states           = mb_states,
    )

def spectral_function_k_resolved(
        omegas          : Array,
        k_points        : Array,
        eigenvalues_k   : Array,
        eta             : float = 0.01,
        backend         : str = "default") -> Array:
    r"""
    Compute momentum-resolved spectral function A(k,\omega).
    
    See spectral_backend.py for full documentation.
    """
    return _spectral_function_k_resolved_backend(omegas, k_points, eigenvalues_k, eta, backend)

def integrated_spectral_weight(
        spectral_function   : Array,
        omega_grid          : Array,
        omega_min           : Optional[float] = None,
        omega_max           : Optional[float] = None,
        backend             : str = "default") -> Union[float, Array]:
    """
    Compute integrated spectral weight over an energy window.
    
    See spectral_backend.py for full documentation.
    """
    return _integrated_spectral_weight_backend(spectral_function, omega_grid, omega_min, omega_max, backend)

def find_spectral_peaks(
        spectral_function   : Array,
        omega_grid          : Array,
        threshold           : float = 0.1,
        min_distance        : int = 5,
        backend             : str = "default") -> Array:
    """
    Find peaks in spectral function.
    
    See spectral_backend.py for full documentation.
    """
    return _find_spectral_peaks_backend(spectral_function, omega_grid, threshold, min_distance, backend)

# =============================================================================
# Finite-Temperature Green's Functions
# =============================================================================

def greens_function_manybody_finite_T(
        omega           : Union[float, Array],
        eigenvalues     : Array,
        operator_a      : Array,
        eta             : float = 0.01,
        *,
        beta            : float = 1.0,
        operator_b      : Optional[Array] = None,
        backend         : str = "default",
        kind            : str = "retarded",
        lehmann_full    : bool = False) -> Array:
    """
    Finite-temperature many-body Green's function.
    
    See spectral_backend.greens_function_manybody_finite_T for full documentation.
    
    Parameters
    ----------
    omega : float or array
        Frequency grid.
    eigenvalues : array
        Many-body eigenvalues.
    operator_a : array
        Operator A.
    eta : float, optional
        Broadening (default: 0.01).
    beta : float, optional
        Inverse temperature (default: 1.0).
    operator_b : array, optional
        Operator B (default: A^dagger).
    backend : str, optional
        Numerical backend (default: "default").
    kind : str, optional
        "retarded" or "advanced" (default: "retarded").
    lehmann_full : bool, optional
        Use (p_m - p_n) factor. Default False ensures positive spectral functions.
        
    Returns
    -------
    Array
        Finite-temperature Green's function.
    """
    return _greens_function_manybody_finite_T(
        omega, eigenvalues, operator_a, eta,
        beta=beta, operator_b=operator_b, backend=backend,
        kind=kind, lehmann_full=lehmann_full
    )

def greens_function_quadratic_finite_T(
        omega           : float,
        eigenvalues     : Array,
        eigenvectors    : Optional[Array] = None,
        eta             : float = 0.01,
        *,
        operator_a      : Optional[Array] = None,
        operator_b      : Optional[Array] = None,
        beta            : float = 1.0,
        mu              : float = 0.0,
        basis_transform : bool = True,
        backend         : str = "default") -> Array:
    """
    Finite-temperature quadratic (non-interacting) Green's function.
    
    See spectral_backend.greens_function_quadratic_finite_T for full documentation.
    
    Parameters
    ----------
    omega : float
        Frequency.
    eigenvalues : array
        Single-particle eigenvalues.
    eigenvectors : array, optional
        Single-particle eigenvectors.
    eta : float, optional
        Broadening (default: 0.01).
    operator_a : array, optional
        Operator A.
    operator_b : array, optional
        Operator B (default: A^dagger).
    beta : float, optional
        Inverse temperature (default: 1.0).
    mu : float, optional
        Chemical potential (default: 0.0).
    basis_transform : bool, optional
        Transform to eigenbasis (default: True).
    backend : str, optional
        Numerical backend (default: "default").
        
    Returns
    -------
    Array
        Finite-temperature quadratic Green's function.
    """
    return _greens_function_quadratic_finite_T(
        omega, eigenvalues, eigenvectors, eta,
        operator_a=operator_a, operator_b=operator_b,
        beta=beta, mu=mu, basis_transform=basis_transform,
        backend=backend
    )

def greens_function_lanczos_finite_T(
        omega           : Union[float, Array],
        hamiltonian     : Array,
        operator_a      : Array,
        eta             : float = 0.01,
        *,
        beta            : float = 1.0,
        operator_b      : Optional[Array] = None,
        n_random        : int = 50,
        max_krylov      : int = 100,
        backend         : str = "default",
        kind            : str = "retarded",
        lehmann_full    : bool = False,
        seed            : Optional[int] = None) -> Array:
    """
    Finite-Temperature Lanczos Method (FTLM) for many-body Green's functions.
    
    Prelovsek's method: Phys. Rev. B 49, 5065 (1994).
    
    See spectral_backend.greens_function_lanczos_finite_T for full documentation.
    
    Parameters
    ----------
    omega : float or array
        Frequency grid.
    hamiltonian : array
        Full many-body Hamiltonian.
    operator_a : array
        Operator A.
    eta : float, optional
        Broadening (default: 0.01).
    beta : float, optional
        Inverse temperature (default: 1.0).
    operator_b : array, optional
        Operator B (default: A^dagger).
    n_random : int, optional
        Number of random states for stochastic trace (default: 50).
    max_krylov : int, optional
        Maximum Krylov dimension (default: 100).
    backend : str, optional
        Numerical backend (default: "default").
    kind : str, optional
        "retarded" or "advanced" (default: "retarded").
    lehmann_full : bool, optional
        Use (p_m - p_n) factor. Default False ensures positive spectral functions.
    seed : int, optional
        Random seed (default: None).
        
    Returns
    -------
    Array
        Finite-temperature Green's function from FTLM.
    """
    return _greens_function_lanczos_finite_T(
        omega, hamiltonian, operator_a, eta,
        beta=beta, operator_b=operator_b,
        n_random=n_random, max_krylov=max_krylov,
        backend=backend, kind=kind, lehmann_full=lehmann_full,
        seed=seed
    )

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Basic spectral functions
    'spectral_function',
    'spectral_function_k_resolved',
    
    # Analysis
    'integrated_spectral_weight',
    'find_spectral_peaks',
    
    # Finite-temperature Green's functions
    'greens_function_manybody_finite_T',
    'greens_function_quadratic_finite_T',
    'greens_function_lanczos_finite_T',
]

# ============================================================================
#! End of file
# ============================================================================