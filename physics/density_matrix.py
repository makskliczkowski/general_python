'''
file    : QES/general_python/physics/density_matrix.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl

This module contains functions for manipulating and analyzing density matrices in quantum mechanics.
'''

import numpy as np
from scipy.linalg import svd, eigh

###############################################################################

def r_dens_mat( state   : np.ndarray,
                A_size  : int,
                L       : int | None = None) -> np.ndarray:
    """
    Computes the reduced density matrix of subsystem A from a pure state vector.

    Given a pure state vector representing a bipartite quantum system (A ⊗ B),
    this function computes the reduced density matrix for subsystem A by tracing out subsystem B.

    Args:
        state (np.ndarray):
            The state vector of the composite system, assumed to be a 1D complex numpy array.
        A_size (int):
            The number of qubits (or log2 of the dimension) in subsystem A.
        L (int | None, optional):
            The total number of qubits in the system.
            If None, it is inferred from the size of `state`.

    Returns:
        np.ndarray:
            The reduced density matrix of subsystem A as a 2D numpy array.

    Raises:
        ValueError:
            If the size of `state` is incompatible with `A_size` and `L`.

    Notes:
        - The input state is assumed to be a pure state (not a density matrix).
        - The function does not copy the input state when reshaping.
    """

    if L is None:
        L = int(np.log2(state.size))
    dimA = 1 << A_size
    dimB = 1 << (L - A_size)

    if state.size != dimA * dimB:
        raise ValueError("state length incompatible with A_size and L")

    psi     = state.reshape(dimA, dimB) # view, no copy
    rho     = psi @ psi.conj().T        # BLAS gemm
    return rho

def r_dens_mat_schmidt(state        : np.ndarray,
                        La          : int,
                        L           : int | None = None,
                        use_eigh    : bool = True) -> np.ndarray:
    """
    Compute the eigenvalues and eigenvectors (or singular values) of the reduced density matrix
    for a bipartitioned quantum state using the Schmidt decomposition.

    Parameters
    ----------
    state : np.ndarray
        The input state vector representing the pure quantum state of the system.
    La : int
        The number of qubits (or sites) in subsystem A.
    L : int or None, optional
        The total number of qubits (or sites) in the system. If None, it is inferred from the size of `state`.
    use_eigh : bool, default=True
        If True, use eigenvalue decomposition of the reduced density matrix.
        If False, use singular value decomposition (SVD) of the reshaped state.

    Returns
    -------
    eigvals : np.ndarray
        The eigenvalues (or squared singular values) of the reduced density matrix.
    U : np.ndarray
        The eigenvectors (or left singular vectors) corresponding to the reduced density matrix.

    Raises
    ------
    ValueError
        If the size of `state` is incompatible with the specified `La` and `L`.
    """

    if L is None:
        L = int(np.log2(state.size))

    dimA = 1 << La
    dimB = 1 << (L - La)

    if state.size != dimA * dimB:
        raise ValueError("state length incompatible with La and L")

    psi     = state.reshape(dimA, dimB)
    if use_eigh:
        if dimA <= dimB:
            rho         = psi @ psi.conj().T
            eigvals, U  = eigh(rho)
        else:
            rho         = psi.conj().T @ psi
            eigvals, U  = eigh(rho)
    else:
        U, s, _     = svd(psi, full_matrices=False)
        eigvals     = s * s
    return eigvals, U

# -----------------------------------------------------------------------------

#! JAX if available
try:
    import general_python.physics.density_matrix_jax as jnp
except ImportError:
    jnp = None

# '''
# Given the specific number of states in linear combination create random coefficients
# '''
# def genRandomStateCoefficients(gamma : int):
#     if gamma == 1:
#         return np.ones(1)
#     # generate random Haar matrix
#     Uhaar           =       random_matrix.CUE((gamma, gamma))
#     # multiply the states by the Haar random
#     coefficients    =       np.matmul(Uhaar, np.ones(gamma))
#     coefficients    /=      np.sqrt(np.vdot(coefficients, coefficients))
#     return coefficients