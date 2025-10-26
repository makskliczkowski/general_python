"""
Green's functions and Fourier transforms for quantum systems.

Provides time-resolved (retarded) Green's functions G(\Omega) for
noninteracting systems, with support for:
- Dense and sparse Hamiltonians
- Eigenvalue/eigenvector representation
- Real-space and momentum-space Fourier transforms
- Lattice-aware transformations
- Utilities for local DOS and traces

File        : Python/QES/general_python/physics/spectral/greens.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
"""

from typing import Optional, Union, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

from ...algebra.utils import JAX_AVAILABLE, Array

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Time-Resolved Green's Functions G(\Omega)
# =============================================================================

def greens_function_dense(omega: float, hamiltonian: Array, eta: float = 0.01) -> Array:
    """
    Compute retarded Green's function G(\Omega) = (\Omega + iη - H)^(-1) for dense Hamiltonian.
    
    Parameters
    ----------
    omega : float
        Frequency \Omega.
    hamiltonian : array-like, shape (N, N)
        Hamiltonian matrix H.
    eta : float, optional
        Broadening parameter η > 0 (default: 0.01).
        
    Returns
    -------
    Array, shape (N, N), complex
        Green's function matrix G(\Omega).
        
    Notes
    -----
    Uses LU decomposition via scipy.linalg.solve for numerical stability.
    For Hermitian H, could use solve with assume_a='her' for better performance.
    """
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    N           = hamiltonian.shape[0]
    
    # Construct (\Omega + iη)I - H
    matrix      = (omega + 1j * eta) * np.eye(N, dtype=complex) - hamiltonian
    
    # Solve: G = matrix^(-1)
    identity    = np.eye(N, dtype=complex)
    greens      = la.solve(matrix, identity, assume_a='gen')

    return greens

def greens_function_sparse(omega: float, hamiltonian: sp.spmatrix, eta: float = 0.01) -> Array:
    """
    Compute retarded Green's function G(\Omega) for sparse Hamiltonian.
    
    Parameters
    ----------
    omega : float
        Frequency \Omega.
    hamiltonian : scipy.sparse matrix, shape (N, N)
        Sparse Hamiltonian matrix H.
    eta : float, optional
        Broadening parameter η > 0 (default: 0.01).
        
    Returns
    -------
    Array, shape (N, N), complex
        Green's function matrix G(\Omega) (returned as dense array).
        
    Notes
    -----
    Uses sparse linear solver. Returns dense array because G is typically dense
    even when H is sparse.
    """
    N = hamiltonian.shape[0]
    
    # Construct (\Omega + iη)I - H as sparse matrix
    identity_sparse = sp.eye(N, dtype=complex, format='csr')
    matrix          = (omega + 1j * eta) * identity_sparse - hamiltonian
    
    # Solve: G = matrix^(-1)
    identity_dense  = np.eye(N, dtype=complex)
    greens          = sp.linalg.spsolve(matrix, identity_dense)

    # spsolve returns 1D array if RHS is 1D, 2D if RHS is 2D
    if greens.ndim == 1:
        greens = greens.reshape(-1, 1)
    
    return greens

def greens_function_eigenbasis(omega: float, eigenvalues: Array, eigenvectors: Array, eta: float = 0.01) -> Array:
    """
    Compute Green's function using eigenvalue decomposition.
    
    G(\Omega) = U * diag(1/(\Omega + iη - E_n)) * U\dag
    
    where H = U * diag(E_n) * U\dag.
    
    Parameters
    ----------
    omega : float
        Frequency \Omega.
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n of Hamiltonian.
    eigenvectors : array-like, shape (N, N)
        Eigenvectors U (columns are eigenstates).
    eta : float, optional
        Broadening parameter η > 0 (default: 0.01).
        
    Returns
    -------
    Array, shape (N, N), complex
        Green's function matrix G(\Omega).
        
    Notes
    -----
    Most efficient method when eigendecomposition is already available.
    Avoids solving linear system.
    """
    eigenvalues     = np.asarray(eigenvalues)
    eigenvectors    = np.asarray(eigenvectors, dtype=complex)

    # Diagonal matrix elements: 1/(\Omega + iη - E_n)
    diag_inv        = 1.0 / (omega + 1j * eta - eigenvalues)
    
    # G = U * diag_inv * U\dag
    U_scaled        = eigenvectors * diag_inv[np.newaxis, :]
    greens          = U_scaled @ eigenvectors.conj().T
    
    return greens

def greens_function_diagonal(omega: float, eigenvalues: Array, eta: float = 0.01) -> Array:
    """
    Compute diagonal Green's function in energy eigenbasis.
    
    G_nn(\Omega) = 1/(\Omega + iη - E_n)
    
    Parameters
    ----------
    omega : float
        Frequency \Omega.
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n.
    eta : float, optional
        Broadening parameter η > 0 (default: 0.01).
        
    Returns
    -------
    Array, shape (N,), complex
        Diagonal elements of Green's function.
        
    Notes
    -----
    Useful for computing local DOS or when only diagonal elements are needed.
    """
    eigenvalues = np.asarray(eigenvalues)
    return 1.0 / (omega + 1j * eta - eigenvalues)

# =============================================================================
# Multi-Frequency Green's Functions
# =============================================================================

def greens_function_multi_omega(omegas: Array, eigenvalues: Array, eigenvectors: Array, eta: float = 0.01, return_diagonal_only: bool = False) -> Array:
    """
    Compute Green's function for multiple frequencies.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Array of frequencies.
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n.
    eigenvectors : array-like, shape (N, N)
        Eigenvectors U.
    eta : float, optional
        Broadening parameter (default: 0.01).
    return_diagonal_only : bool, optional
        If True, return only diagonal elements (default: False).
        
    Returns
    -------
    Array
        If return_diagonal_only=False: shape (n_omega, N, N), full G(\Omega) for each \Omega.
        If return_diagonal_only=True: shape (n_omega, N), diagonal G_nn(\Omega).
        
    Notes
    -----
    Vectorized over frequencies for efficiency.
    """
    omegas          = np.asarray(omegas)
    eigenvalues     = np.asarray(eigenvalues)
    eigenvectors    = np.asarray(eigenvectors, dtype=complex)

    n_omega         = len(omegas)
    N               = len(eigenvalues)

    if return_diagonal_only:
        # Shape: (n_omega, N)
        greens = np.zeros((n_omega, N), dtype=complex)
        for i, omega in enumerate(omegas):
            greens[i] = greens_function_diagonal(omega, eigenvalues, eta)
    else:
        # Shape: (n_omega, N, N)
        greens = np.zeros((n_omega, N, N), dtype=complex)
        for i, omega in enumerate(omegas):
            greens[i] = greens_function_eigenbasis(omega, eigenvalues, eigenvectors, eta)
    
    return greens

# =============================================================================
# Fourier Transforms
# =============================================================================

def fourier_transform_matrix(greens_function: Array) -> Array:
    """
    Fourier transform Green's function using FFT.
    
    G(t) = FFT[G(\Omega)]
    
    Parameters
    ----------
    greens_function : array-like, shape (..., N)
        Green's function in frequency domain.
        
    Returns
    -------
    Array, complex
        Green's function in time domain.
        
    Notes
    -----
    Uses NumPy's FFT along last axis.
    """
    return np.fft.fft(greens_function, axis=-1)

def fourier_transform_with_dft(greens_function: Array, dft_matrix: Array) -> Array:
    """
    Fourier transform using pre-computed DFT matrix.
    
    G_FT = DFT @ G
    
    Parameters
    ----------
    greens_function : array-like, shape (N, M)
        Green's function matrix.
    dft_matrix : array-like, shape (K, N)
        Discrete Fourier transform matrix.
        
    Returns
    -------
    Array, shape (K, M), complex
        Fourier-transformed Green's function.
        
    Notes
    -----
    Useful when custom DFT matrix is needed (e.g., for specific k-points).
    """
    
    dft_matrix      = np.asarray(dft_matrix, dtype=complex)
    greens_function = np.asarray(greens_function, dtype=complex)
    
    return dft_matrix @ greens_function

def fourier_transform_diagonal(greens_function: Array, dft_matrix: Array) -> Array:
    """
    Fourier transform and extract diagonal elements.
    
    G_diag(k) = diag(DFT @ G @ DFT\dag)
    
    Parameters
    ----------
    greens_function : array-like, shape (N, N)
        Green's function matrix.
    dft_matrix : array-like, shape (K, N)
        DFT matrix.
        
    Returns
    -------
    Array, shape (K,), complex
        Diagonal elements in Fourier space.
    """
    dft_matrix      = np.asarray(dft_matrix, dtype=complex)
    greens_function = np.asarray(greens_function, dtype=complex)
    
    # Compute: diag(DFT @ G @ DFT\dag)
    temp    = dft_matrix @ greens_function
    result  = np.sum(temp * dft_matrix.conj(), axis=1)
    
    return result

def fourier_transform_lattice(greens_function: Array, lattice_k_vectors: Array, lattice_r_vectors: Array) -> complex:
    """
    Fourier transform from real space to a specific k-point on a lattice.
    
    G(k) = \sum _{i,j} G_{ij} exp(i k·(r_i - r_j))
    
    Parameters
    ----------
    greens_function : array-like, shape (N, N)
        Green's function in real space.
    lattice_k_vectors : array-like, shape (d,)
        k-vector for Fourier transform (d = spatial dimension).
    lattice_r_vectors : array-like, shape (N, d)
        Real-space lattice vectors r_i.
        
    Returns
    -------
    complex
        G(k) at the specified k-point.
        
    Notes
    -----
    This is a lattice-aware Fourier transform. For general use, consider
    vectorizing over multiple k-points.
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
    """
    Fourier transform assuming translational invariance: G(k) diagonal in k-space.
    
    G(k) = \sum _{i,j} G_{ij} exp(i k·(r_i - r_j)) for all k
    
    Returns diagonal matrix in k-space.
    
    Parameters
    ----------
    greens_function : array-like, shape (N, N)
        Green's function in real space (assumed translationally invariant).
    lattice_k_vectors : array-like, shape (K, d)
        Array of k-vectors.
    lattice_r_vectors : array-like, shape (N, d)
        Real-space lattice vectors.
        
    Returns
    -------
    Array, shape (K,), complex
        G(k) for each k-point (diagonal elements only).
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
    """
    Compute local density of states from diagonal Green's function.
    
    LDOS(\Omega) = -(1/π) Im[G_ii(\Omega)]
    
    Parameters
    ----------
    greens_diagonal : array-like, complex
        Diagonal elements of Green's function.
        
    Returns
    -------
    Array, real
        Local density of states.
    """
    return -np.imag(greens_diagonal) / np.pi

def trace_greens(greens_function: Array) -> complex:
    """
    Compute trace of Green's function: Tr[G(\Omega)].
    
    Parameters
    ----------
    greens_function : array-like, shape (N, N)
        Green's function matrix.
        
    Returns
    -------
    complex
        Trace of G.
    """
    return np.trace(greens_function)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Single frequency
    'greens_function_dense',
    'greens_function_sparse',
    'greens_function_eigenbasis',
    'greens_function_diagonal',
    
    # Multi-frequency
    'greens_function_multi_omega',
    
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
