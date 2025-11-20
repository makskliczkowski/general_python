r"""
Green's functions and Fourier transforms for quantum systems.

This module provides convenient wrappers for Green's function calculations,
delegating to the unified spectral_backend.py for all implementations.

-------------------------------------------------------------------------------
File        : Python/QES/general_python/physics/spectral/greens_function.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
-------------------------------------------------------------------------------
"""

from typing import Optional, Union
import numpy as np

try:
    from ...algebra.utils import JAX_AVAILABLE, Array
except ImportError:
    raise ImportError("algebra.utils module not available")

if JAX_AVAILABLE:
    import jax.numpy as jnp
else:
    jnp = np

# Import from spectral backend
try:
    from .spectral_backend import (
        greens_function_diagonal,
        greens_function_quadratic,
        greens_function_quadratic_finite_T,
        greens_function_manybody,
        greens_function_manybody_finite_T,
        greens_function_lanczos,
        greens_function_lanczos_finite_T,
    )
except ImportError:
    raise ImportError("spectral_backend module not available")

# =============================================================================
# Green's Functions - All delegate to spectral_backend
# =============================================================================

# Single-particle Green's functions (quadratic systems)
# Already imported above: greens_function_quadratic, greens_function_quadratic_finite_T

# Many-body Green's functions
# Already imported above: greens_function_manybody

# Lanczos Green's functions  
# Already imported above: greens_function_lanczos

# =============================================================================
# Fourier Transforms (Utilities)
# =============================================================================
#
# Note: For lattice-based systems with translation symmetry, prefer using
# the Lattice methods for optimal performance and consistency:
#   - lattice.kspace_from_realspace(G_real)  : Real-space -> k-space
#   - lattice.realspace_from_kspace(G_k)     : k-space -> Real-space
# These methods properly handle sublattices, basis transformations, and
# preserve the spectral properties exactly.
# =============================================================================

def fourier_transform_matrix(greens_function: Array) -> Array:
    r"""
    Fourier transform Green's function using FFT.
    
    G(t) = FFT[G(\Omega)]
    
    Note
    ----
    For lattice systems, consider using lattice.kspace_from_realspace()
    or lattice.realspace_from_kspace() for proper handling of sublattices.
    
    Parameters
    ----------
    greens_function : array-like, shape (..., N)
        Green's function in frequency domain.
        
    Returns
    -------
    Array, complex
        Green's function in time domain.
    """
    return np.fft.fft(greens_function, axis=-1)

def fourier_transform_with_dft(greens_function: Array, dft_matrix: Array) -> Array:
    r"""
    Fourier transform using pre-computed DFT matrix.
    
    G_FT = DFT @ G
    """
    dft_matrix      = np.asarray(dft_matrix, dtype=complex)
    greens_function = np.asarray(greens_function, dtype=complex)
    return dft_matrix @ greens_function

def fourier_transform_diagonal(greens_function: Array, dft_matrix: Array) -> Array:
    r"""
    Fourier transform and extract diagonal elements.
    
    G_diag(k) = diag(DFT @ G @ DFT\dag)
    """
    dft_matrix      = np.asarray(dft_matrix, dtype=complex)
    greens_function = np.asarray(greens_function, dtype=complex)
    temp    = dft_matrix @ greens_function
    result  = np.sum(temp * dft_matrix.conj(), axis=1)
    return result

def fourier_transform_lattice(greens_function: Array, lattice_k_vectors: Array, lattice_r_vectors: Array) -> complex:
    r"""
    Fourier transform from real space to a specific k-point on a lattice.
    
    G(k) = \sum_{i,j} G_{ij} exp(i k\cdot (r_i - r_j))
    
    Note
    ----
    For full lattice transformations with proper sublattice handling,
    use lattice.kspace_from_realspace(G_real) instead. This provides:
      - Correct treatment of basis sites
      - Efficient FFT-based computation
      - Shape (Lx, Ly, Lz, Nb, Nb) output with momentum grid
    
    This function is for single k-point evaluation or custom use cases.
    """
    greens_function = np.asarray(greens_function, dtype=complex)
    k               = np.asarray(lattice_k_vectors)
    r_vectors       = np.asarray(lattice_r_vectors)
    N               = len(r_vectors)
    result          = 0.0 + 0.0j
    for i in range(N):
        r_i = r_vectors[i]
        for j in range(N):
            r_j = r_vectors[j]
            phase = np.exp(1j * np.dot(k, r_i - r_j))
            result += greens_function[i, j] * phase
    return result

def fourier_transform_lattice_translational(greens_function: Array, lattice_k_vectors: Array, lattice_r_vectors: Array) -> Array:
    r"""
    Fourier transform assuming translational invariance: G(k) diagonal in k-space.
    """
    lattice_k_vectors = np.asarray(lattice_k_vectors)
    K = len(lattice_k_vectors)
    result = np.zeros(K, dtype=complex)
    for idx, k in enumerate(lattice_k_vectors):
        result[idx] = fourier_transform_lattice(greens_function, k, lattice_r_vectors)
    return result

# =============================================================================
# Utilities
# =============================================================================

def local_dos_from_greens(greens_diagonal: Array) -> Array:
    r"""
    Compute local density of states from diagonal Green's function.
    
    LDOS(\Omega) = -(1/\pi) Im[G_ii(\Omega)]
    """
    return -np.imag(greens_diagonal) / np.pi

def trace_greens(greens_function: Array) -> complex:
    r"""
    Compute trace of Green's function: Tr[G(\Omega)].
    """
    return np.trace(greens_function)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Green's functions (from spectral_backend)
    'greens_function_diagonal',
    'greens_function_quadratic',
    'greens_function_quadratic_finite_T',
    'greens_function_manybody',
    'greens_function_manybody_finite_T',
    'greens_function_lanczos',
    'greens_function_lanczos_finite_T',
    
    # Fourier transforms
    'fourier_transform_matrix',
    'fourier_transform_with_dft',
    'fourier_transform_diagonal',
    'fourier_transform_lattice',
    'fourier_transform_lattice_translational',
    
    # Utilities
    'local_dos_from_greens',
    'trace_greens',
]

# ============================================================================
#! End of file
# ============================================================================
