r"""
general_python/physics/sp/correlation_matrix.py
=====================

This module provides functions to construct the single-particle correlation matrix 
C_ij = ⟨ψ|cᵢ ^\dag  cⱼ|ψ⟩ for various types of quantum many-body states:

Functions
---------
- `corr_single`: Computes the correlation matrix for a single Slater determinant 
    specified by an occupation bit-string.
- `corr_superposition`: Computes the correlation matrix for a linear combination 
    of Slater determinants, |ψ⟩ = Σ_k αₖ |mₖ⟩.
- `corr_from_state_vector`: Computes the correlation matrix for an arbitrary 
    many-body state given as a state vector (general but less efficient).

- Ls: Total number of one-particle orbitals or lattice sites.
- La: Size of the spatial subsystem A (first La sites are considered part of A).
- W: Ls \times Ls unitary matrix that diagonalizes the quadratic Hamiltonian.
- W_A: Ls \times La matrix containing the first La columns of W (rows: orbitals, columns: sites in A).

For a single Slater determinant with occupation vector `n` (shape (Ls,)), the reduced 
correlation matrix C_A is given by:

    C_A = W_A ^\dag  · diag(n) · W_A
    # raw multiplication (Eq. (3) PHYSICAL REVIEW LETTERS 125, 180604 (2020))

References
----------
"""

import numpy as np
from typing import Sequence, Tuple, Optional

import numpy as np
import numba

from general_python.algebra.utils import JAX_AVAILABLE, Array

__all__ = [
    "corr_single",
    "corr_superposition",
    "corr_state",
]

#################################### 
#! 1)  Single Slater determinant
###################################

# @numba.njit(cache = True)
def corr_single(
    W_A                 : np.ndarray,   # shape (Ls, La)
    occ                 : np.ndarray,   # boolean / 0-1 vector, shape (Ls,)
    subtract_identity   : bool = True,
    W_A_CT              : Optional[np.ndarray] = None,
    raw                 : bool = True) -> np.ndarray:
    """
    Correlation matrix C_A of a single Slater determinant.

    Parameters
    ----------
    W_A  : ndarray (Ls, La)
        Rows = orbitals q, columns = real-space sites i∈A.
    occ  : ndarray (Ls,) of {0,1}
        Occupation numbers n_q of the determinant.
    subtract_identity : bool, default True
        If True, return `C_A - I` (frequently needed for ent-H calcs).

    Returns
    -------
    C    : ndarray (La, La)
    """
    # Ensure W_A_CT is provided or compute it as the conjugate transpose of W_A
    if W_A_CT is None:
        W_A_CT = W_A.conj().T  # (La, Ls)

    # Use occupation vector directly for raw mode, otherwise map to ±1 for entanglement Hamiltonian
    if raw:
        indices     =       occ.astype(bool)
        Wp          =       W_A[indices, :] 
        W_prime     =       W_A_CT[:, indices]
        C           =       2.0 * np.matmul(W_prime, Wp)
    else:
        prefactors  = 2 * occ - 1 # maps 0→-1, 1→+1
        # Efficiently compute C = W_A† · diag(prefactors) · W_A without explicit diag
        # (La, Ls) * (Ls,) → (La, Ls), then @ (Ls, La) → (La, La)
        C           = (W_A_CT * prefactors) @ W_A

    if subtract_identity:
        np.fill_diagonal(C, C.diagonal() - 1.0)

    return C

#################################### 
#! 2)  Multiple Slater determinants
###################################

def corr_superposition(
    W_A                 : np.ndarray,               # (Ls, La)
    occ_list            : Sequence[np.ndarray],     # list/tuple of bool arrays, each (Ls,)
    coeff               : np.ndarray | None = None,
    *,
    WA_CT               : Optional[np.ndarray] = None,
    subtract_identity   : bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the correlation matrix for a quantum state that is a superposition of Slater determinants.

    Given a state:
        |ψ⟩ = Σ_k α_k |m_k⟩,    with normalization ⟨ψ|ψ⟩ = 1

    This function calculates the single-particle correlation matrix C for |ψ⟩, taking into account both diagonal (within the same determinant) and off-diagonal (between determinants differing by exactly two orbitals) contributions.

    - All determinants in `occ_list` must conserve particle number and have the same number of particles (N).
    - Off-diagonal contributions are included only for pairs of determinants that differ in exactly two orbital occupations.

    W_A : np.ndarray, shape (Ls, La)
        Single-particle orbital transformation matrix restricted to a subsystem.
    occ_list : Sequence[np.ndarray]
        List or tuple of boolean or {0,1} arrays, each of shape (Ls,), representing occupation vectors for each determinant.
    coeff : np.ndarray or None, optional
        1D complex array of coefficients (α_k) for the superposition, of length len(occ_list).
        If None, coefficients are drawn Haar-randomly.
        If True, subtracts the identity from the diagonal of the correlation matrix.

    C : np.ndarray, shape (La, La)
        The computed correlation matrix for the superposition state.
    coeff : np.ndarray
        The array of coefficients actually used in the calculation (either provided or generated).

    Notes
    -----
    - For off-diagonal terms, the implementation follows Eq. (5) from PRL 125, 180604 (2020).
    - The function raises ValueError if `occ_list` is empty or if any occupation vector does not match the expected length.
    """
    gamma = len(occ_list)
    if gamma == 0:
        raise ValueError("occ_list is empty")

    Ls = W_A.shape[0]
    for occ in occ_list:
        if occ.size != Ls:
            raise ValueError("Every occupation vector must have length Ls")

    # ------------------------------------------------------------------------------------------------
    #!  Equal part (m = n) …  sum_k |α_k|² · W_A† diag(n_k) W_A
    # ------------------------------------------------------------------------------------------------
    if coeff is None:
        coeff = np.random.normal(size = 1.0) / np.sqrt(gamma)

    # allocate me
    C = np.zeros((W_A.shape[1], W_A.shape[1]), dtype=W_A.dtype)

    # add diagonal part
    for ck, occ in zip(coeff, occ_list):
        C += (abs(ck) ** 2) * corr_single(W_A, occ, subtract_identity=False)

    # ------------------------------------------------------------------------------------------------
    #  Unequal part  (m ≠ n)  … only when dets differ in exactly 2 orbitals
    #  Use Eq. (5) from PRL 125 180604
    # ------------------------------------------------------------------------------------------------
    La      = W_A.shape[1]
    if WA_CT is None:
        WA_CT = W_A.conj().T                # (La, Ls)

    # go through the occupations
    for a, occ_a in enumerate(occ_list):
        for b in range(a + 1, gamma):
            occ_b   = occ_list[b]
            diff    = occ_a ^ occ_b         # XOR: where they differ
            if diff.sum() != 2:
                continue                    # only 2‑orbital difference contributes

            q1, q2 = np.nonzero(diff)[0]    # the two differing orbitals
            # The ± sign from fermionic exchange:
            #   if |m⟩ has q1 occupied and |n⟩ has q1 empty  → annihilate at q1
            #   otherwise the opposite.
            # We encode this in the prefactor below.

            sign =  1.0
            if occ_a[q2]: # annihilation order flips sign if starting point occupied
                sign = -1.0

            coef = 2.0 * sign * np.conjugate(coeff[a]) * coeff[b]

            #! Outer product  (La,1)·(1,La) fused by BLAS as gemm
            C += coef * np.outer(WA_CT[:, q1], W_A[q2, :])

            # and its Hermitian conjugate (because we skipped the term with indices
            # swapped in the loop)
            C += np.conjugate(coef) * np.outer(WA_CT[:, q2], W_A[q1, :])

    #! Identity
    if subtract_identity:
        np.fill_diagonal(C, C.diagonal() - 1.0)
    
    return C, coeff

# ---------------------------------------------------------------------------
#! 3)  Generic many‑body state
# ---------------------------------------------------------------------------

def corr_from_state_vector(psi: np.ndarray) -> np.ndarray:
    """
    Build  C_ij  directly from the full wave‑function |ψ⟩ in the
    computational basis   {|n₀ … n_{L‑1}⟩}.

    Complexity  O(L² 2ᴸ)  – use only for **small L** (≤ 16) or debugging.

    Parameters
    ----------
    psi : ndarray, shape (2**L,)
        Amplitudes in binary order,  n = Σ_j n_j 2^j.

    Returns
    -------
    C   : ndarray (L, L)
    """
    nbasis = psi.size
    L = int(np.log2(nbasis))
    if 1 << L != nbasis:
        raise ValueError("psi length is not a power of two")

    C = np.zeros((L, L), dtype=np.complex128)
    # diagonal ⟨n_j⟩ is easy and cheap
    idx = np.arange(nbasis, dtype=np.uint64)
    occ = ((idx[:, None] >> np.arange(L)) & 1).astype(np.uint8)  # (nbasis, L)
    probs = np.abs(psi)**2
    C[np.diag_indices(L)] = probs @ occ      # sum_n |ψ_n|² n_j

    # off‑diagonals
    for i in range(L):
        for j in range(i + 1, L):
            # we need pairs (n, n') that differ by removing a fermion at j and
            # adding at i
            mask_i = 1 << i
            mask_j = 1 << j
            for n in range(nbasis):
                if (n & mask_j) and not (n & mask_i):          # n_j = 1, n_i = 0
                    n_prime = n ^ mask_j ^ mask_i              # remove j, add i
                    phase   = (-1) ** (bin(n & ((1 << j) - 1)).count("1")
                                      - bin(n & ((1 << i) - 1)).count("1"))
                    C[i, j] += np.conjugate(psi[n]) * psi[n_prime] * phase
                    C[j, i] = np.conjugate(C[i, j])

    np.fill_diagonal(C, C.diagonal() - 1.0)   # conventional −I shift
    return C

###########################################
#! JAX
###########################################

if JAX_AVAILABLE:
    from functools import partial
    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnums=(2,))
    def corr_single_jax(
        W_A                 : jnp.ndarray,  # (Ls, La)
        occ                 : jnp.ndarray,  # (Ls,)
        subtract_identity   : bool = True) -> jnp.ndarray:
        """
        JAX kernel – one line of XLA‑fused linear algebra.
        """
        occ_f   = occ.astype(W_A.dtype)
        C       = (jnp.conjugate(W_A).T * occ_f) @ W_A
        if subtract_identity:
            C   = C - jnp.eye(W_A.shape[1], dtype=W_A.dtype)
        return C

    #######################################

#! End