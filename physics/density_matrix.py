'''
file    : QES/general_python/physics/density_matrix.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl

This module contains functions for manipulating and analyzing density matrices in quantum mechanics.
'''

import numpy as np
try:
    import numba
except ImportError as e:
    numba   = None
    
from scipy.linalg import svd, eigh
from typing import Union, List, Callable, Tuple

from general_python.algebra.utils import Array

###############################################################################
#! Helper functions
###############################################################################

def _split_dims(state : Array, size_a: int, L: int | None) -> Tuple[int,int]:
    """
    Splits the dimensions of a quantum state vector into two subsystems.
    Args:
        state (Array):
            The quantum state vector to be split.
        size_a (int):
            The number of qubits (or log2 dimension) in subsystem A.
        L (int | None):
            The total number of qubits (or log2 dimension) in the system. 
            If None, it is inferred from the size of `state`.
    Returns:
        Tuple[int, int]:
            A tuple (dimA, dimB) where dimA is the dimension of subsystem A,
            and dimB is the dimension of subsystem B.
    Raises:
        ValueError:
            If the size of `state` is incompatible with the specified `size_a` and `L`.
    """

    if L is None:
        L = int(np.log2(state.size))
        
    dimA    = 1 << size_a
    dimB    = 1 << (L - size_a)
    if state.size != dimA * dimB:
        raise ValueError("state length incompatible with size_a and L")
    return dimA, dimB

#! Numpy

def rho_numpy(state : Array, dimA: int, dimB: int) -> Array:
    """
    Computes the reduced density matrix ρ from a pure state vector using NumPy.
    Given a state vector representing a bipartite quantum system of dimensions (dimA, dimB),
    this function reshapes the state into a matrix of shape (dimA, dimB) using column-major
    order (Fortran order), and then computes the density matrix ρ = ψ ψ†.
    Args:
        state (Array):
            The state vector of the composite quantum system.
        dimA (int):
            Dimension of subsystem A.
        dimB (int):
            Dimension of subsystem B.
    Returns:
        Array: The density matrix ρ of shape (dimA, dimA).
    """
    
    # reshape as (dimA, dimB) and call BLAS gemm ρ = ψ ψ†
    psi = state.reshape(dimA, dimB, order="F") # column‑major
    return psi @ psi.conj().T

def rho_mask(state : np.ndarray,
        extract_a   : Callable[[int], int],
        extract_b   : Callable[[int], int],
        size_a      : int,
        size        : int,
        tol         : float = 1e-12):
    """
    Constructs a reshaped wavefunction (psi) from a given quantum state vector by mapping indices using provided extraction functions.

    This function is typically used to prepare a state for partial trace or reduced density matrix calculations by reshaping the state vector into a 2D array according to subsystem partitions.

    Args:
        state (np.ndarray):
            The input quantum state vector (1D complex array).
        extract_a (Callable[[int], int]):
            Function to extract the subsystem-A index from a basis index.
        extract_b (Callable[[int], int]):
            Function to extract the subsystem-B index from a basis index.
        size_a (int):
            Number of qubits (or bits) in subsystem A.
        size (int):
            Total number of qubits (or bits) in the system.
        tol (float, optional):
            Amplitude threshold below which values are ignored (default: 1e-12).

    Returns:
        np.ndarray:
            A 2D array of shape (2**size_a, 2**(size - size_a)) representing the reshaped wavefunction.
    """
    size_b  = size - size_a
    dA      = 1 << size_a
    dB      = 1 << size_b
    psi     = np.zeros((dA, dB), dtype=state.dtype)
    for i in range(state.size):
        amp             = state[i]
        if np.abs(amp) <= tol:        
            continue
        iA              = extract_a(i)
        iB              = extract_b(i)
        psi[iA, iB]    += amp
    return psi

def schmidt_numpy( state : Array,
                    dimA    : int,
                    dimB    : int,
                    eig     : bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Schmidt decomposition of a bipartite quantum state using NumPy.
    Parameters
    ----------
    state : Array
        The input state vector representing the bipartite quantum system.
    dimA : int
        Dimension of subsystem A.
    dimB : int
        Dimension of subsystem B.
    eig : bool
        If True, use eigenvalue decomposition of the reduced density matrix; 
        if False, use singular value decomposition (SVD).
    Returns
    -------
    vals : np.ndarray
        The Schmidt coefficients (squared singular values or eigenvalues), sorted in descending order.
    vecs : np.ndarray
        The corresponding Schmidt vectors, columns ordered according to descending Schmidt coefficients.
    Notes
    -----
    - The input state is reshaped according to Fortran order ("F") to match the bipartite structure.
    - For `eig=True`, the function computes the eigenvalues and eigenvectors of the reduced density matrix of the smaller subsystem.
    - For `eig=False`, the function computes the singular value decomposition (SVD) of the reshaped state.
    """
    
    # do the same to obtain the Schmidt values
    psi = state.reshape(dimA, dimB, order="F")
    if eig:
        # use the smaller Hermitian matrix
        if dimA <= dimB:
            rho         = psi @ psi.conj().T
            vals, vecs  = np.linalg.eigh(rho)
        else:
            sigma   = psi.conj().T @ psi
            vals, V = np.linalg.eigh(sigma)
            vecs    = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
    else:
        vecs, s, _  = np.linalg.svd(psi, full_matrices=False)
        vals        = s * s
    return vals[::-1], vecs[:, ::-1]

#! Numba
if numba:
    
    # @numba.njit
    def rho_numba(state: Array, dimA: int, dimB: int) -> Array:
        psi = state.reshape(dimA, dimB, order="F")
        return psi @ psi.conj().T

    # @numba.njit(cache=True)
    def rho_numba_mask(state    : np.ndarray,
                    order       : tuple,
                    size_a      : int) -> Array:
        """
        Reshapes and reorders a quantum state vector for subsystem partitioning.
        This function takes a 1D quantum state vector and reshapes it into a multi-dimensional array,
        then transposes and flattens it to produce a 2D array suitable for partial trace or reduced
        density matrix calculations. The partitioning is determined by the `order` and `size_a` arguments.
        Args:
            state (np.ndarray):
                The input quantum state vector as a 1D complex-valued array of length 2**N, where N is the number of qubits.
            order (tuple):
                A tuple specifying the new order of qubits after partitioning. The first `size_a` elements correspond to subsystem A.
            size_a (int):
                The number of qubits in subsystem A.
        Returns:
            Array:
                A 2D array of shape (2**size_a, 2**(N - size_a)), where N = len(order), representing the reshaped and reordered wavefunction.
        """
        psi_nd      = state.reshape((2, ) * len(order), order='F') # no copy, reshape to (2, 2, ..., 2)
        dA          = 1 << size_a
        return psi_nd.transpose(order).reshape(dA, -1)
        
                    # extract_a   : Callable[[int], int],
                    # extract_b   : Callable[[int], int],
                    # size_a      : int,
                    # size        : int,
                    # tol         : float = 1e-14):
        # """
        # Constructs a reshaped wavefunction (psi) from a given quantum state vector by mapping indices using provided extraction functions.

        # This function is typically used to prepare a state for partial trace or reduced density matrix calculations by reshaping the state vector into a 2D array according to subsystem partitions.

        # Args:
        #     state (np.ndarray):
        #         The input quantum state vector (1D complex array).
        #     extract_a (Callable[[int], int]):
        #         Function to extract the subsystem-A index from a basis index.
        #     extract_b (Callable[[int], int]):
        #         Function to extract the subsystem-B index from a basis index.
        #     size_a (int):
        #         Number of qubits (or bits) in subsystem A.
        #     size (int):
        #         Total number of qubits (or bits) in the system.
        #     tol (float, optional):
        #         Amplitude threshold below which values are ignored (default: 1e-12).

        # Returns:
        #     np.ndarray:
        #         A 2D array of shape (2**size_a, 2**(size - size_a)) representing the reshaped wavefunction.
        # """
        
        # psi_nd  = state.reshape((2,) * size) # no copy, reshape to (2, 2, ..., 2)
        
        
        
        # size_b  = size - size_a
        # dA      = 1 << size_a
        # dB      = 1 << size_b
        # psi     = np.zeros((dA, dB), dtype=state.dtype)
        # for i in numba.prange(state.size):
        #     i               = np.int64(i)
        #     amp             = state[i]
        #     if np.abs(amp) <= tol:        
        #         continue
        #     iA              = extract_a(i)
        #     iB              = extract_b(i)
        #     psi[iA, iB]    += amp
        # return psi

    # @numba.njit(cache=True, fastmath=True)
    def schmidt_numba(  psi     : Array,
                        dimA    : int,
                        dimB    : int,
                        eig     : bool) -> Tuple[np.ndarray, np.ndarray]:
        if eig:
            if dimA <= dimB:
                rho         = psi @ psi.conj().T
                vals, vecs  = np.linalg.eigh(rho)
            else:
                sigma       = psi.conj().T @ psi
                vals, V     = np.linalg.eigh(sigma)
                vecs        = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
        else:
            vecs, s, _      = np.linalg.svd(psi, full_matrices=False)
            vals            = s * s
        return vals[::-1], vecs[:, ::-1]

    @numba.njit(cache=True, fastmath=True)
    def schmidt_numba_mask(psi      : np.ndarray,
                        order       : tuple,
                        size_a      : int,
                        eig         : bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Schmidt decomposition of a bipartite quantum state using Numba.
        Parameters
        ----------
        state : np.ndarray
            The input state vector representing the bipartite quantum system.
        order : tuple
            A tuple specifying the new order of qubits after partitioning. The first `size_a` elements correspond to subsystem A.
        size_a : int
            Dimension of subsystem A.
        eig : bool
            If True, use eigenvalue decomposition of the reduced density matrix; 
            if False, use singular value decomposition (SVD).
        Returns
        -------
        vals : np.ndarray
            The Schmidt coefficients (squared singular values or eigenvalues), sorted in descending order.
        vecs : np.ndarray
            The corresponding Schmidt vectors, columns ordered according to descending Schmidt coefficients.
        """
        
        if eig:
            if size_a <= len(order) - size_a:
                rho         = psi @ psi.conj().T
                vals, vecs  = np.linalg.eigh(rho)
            else:
                sigma       = psi.conj().T @ psi
                vals, V     = np.linalg.eigh(sigma)
                vecs        = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
        else:
            vecs, s, _      = np.linalg.svd(psi, full_matrices=False)
            vals            = np.square(s)
        return vals[::-1], vecs[:, ::-1]
    
    # @numba.njit
    # def schmidt_numba_mask(state    : np.ndarray,
    #                     extract_a   : Callable[[int], int],
    #                     extract_b   : Callable[[int], int],
    #                     size_a      : int,
    #                     size        : int,
    #                     eig         : bool) -> Tuple[np.ndarray, np.ndarray]:
    #     psi = rho_numba_mask(state, extract_a, extract_b, size_a, size)
    #     if eig:
    #         if size_a <= size - size_a:
    #             rho         = psi @ psi.conj().T
    #             vals, vecs  = np.linalg.eigh(rho)
    #         else:
    #             sigma       = psi.conj().T @ psi
    #             vals, V     = np.linalg.eigh(sigma)
    #             vecs        = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
    #     else:
    #         vecs, s, _      = np.linalg.svd(psi, full_matrices=True)
    #         vals            = s * s
    #     return vals[::-1], vecs[:, ::-1]
    
else:
    rho_numba = rho_numba_mask = schmidt_numba = None

# -----------------------------------------------------------------------------

#! JAX if available
try:
    import general_python.physics.density_matrix_jax as jnp
except ImportError:
    jnp = np