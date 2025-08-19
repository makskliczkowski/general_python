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
    of Slater determinants, |ψ⟩ = Σ_k aₖ |mₖ⟩.
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
from typing import Literal, Sequence, Tuple, Optional, Literal

import numpy as np
import numba
from torch import mode

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
    raw                 : bool = True,
    mode                : Literal["slater", "bdg-normal", "bdg-full"] = "slater",
    stacked_uv          : bool = False # if True and mode starts with "bdg", W_A is stacked [U; V] with shape (2*Ls, La)
    ) -> np.ndarray:
    r"""
    Correlation matrix C_A of a single Slater determinant.

    Parameters
    ----------
    W_A  : ndarray (Ls, La)
        - mode = "slater":
            ndarray (Ls, La), rows = orbitals q, cols = sites i∈A.
        - mode starts with "bdg":
            either a tuple (U_A, V_A) of shape (Ls, La) each, or a stacked ndarray (2*Ls, La) if `stacked_uv=True`
            with first Ls rows = U_A and last Ls rows = V_A.
    occ : ndarray (Ls,)
        - "slater": 
            occupations n_q \in {0,1}.
        - "bdg-*": quasiparticle occupations f_q \in [0,1] (T=0 vacuum -> f=0).
    subtract_identity : bool
        For "slater" and "bdg-normal":
            subtract I on the (La\times La) normal block.
        For "bdg-full":
            subtract block-diag(I, I) on the 2La \times 2La Nambu matrix.
    W_A_CT : Optional precomputed conjugate transpose(s) to avoid recomputation.
        - "slater":
            W_A_CT = W_A.conj().T (La, Ls).
        - "bdg-*":
            ignored unless you pass a tuple: then you may pass a tuple (U_A_CT, V_A_CT).
    raw : bool
        "slater" fast path:
            if True, uses selection by boolean mask and computes 2·(W_occ^\dag W_occ).
        If False, uses `pref = 2·occ - 1` trick.
    mode : {"slater","bdg-normal","bdg-full"}
        - "slater"     : C = ⟨c_i^\dag c_j⟩ (La\times La).
        - "bdg-normal" : N = ⟨c_i^\dag c_j⟩ for BdG (La\times La).
        - "bdg-full"   : Nambu G =
                        [[ ⟨c^\dag c⟩, ⟨c^\dag c^\dag⟩ ],
                        [ ⟨c   c⟩, ⟨c   c^\dag⟩ ]]  of shape (2La\times 2La).
    stacked_uv : bool
        If True in BdG modes, interpret W_A as vertically stacked [U_A; V_A].

    Returns
    -------
    ndarray
        - "slater" / "bdg-normal": (La, La)
        - "bdg-full"             : (2*La, 2*La)

    Notes (BdG, zero/finite T):
    ---------------------------
    Let c = U a + V a^\dag, with U,V ∈ C^{La\times Ls} (we use U_A,V_A as (Ls,La) row-major in orbitals; see below).
    For diagonal quasiparticle occupations f = diag(f_q), the standard equal-time correlations are
        N ≡ ⟨c^\dag c⟩  = U f U^\dag + V (I - f) V^\dag,
        F ≡ ⟨c   c⟩  = U (I - f) V^T + V f U^T,
        Ṅ ≡ ⟨c   c^\dag⟩ = U (I - f) U^\dag + V f V^\dag = I - N.
    Implementation uses row-major (orbitals q) storage: U_A has shape (Ls, La), so
        N = (U_A^\dag · f · U_A) + (V_A^\dag · (1-f) · V_A),
        computed efficiently without forming diag(f) via column-wise scaling of U_A^\dag and V_A^\dag.
    """
    if mode == "slater":
        if W_A_CT is None:
            W_A_CT        = W_A.conj().T                   # (La, Ls)
        La                = W_A.shape[1]
        occ_bool          = np.asarray(occ, dtype=bool)
        num_occupied      = int(np.sum(occ_bool))

        if num_occupied == 0:
            C             = np.zeros((La, La), dtype=W_A.dtype)
        elif raw:
            W_A_occ       = W_A[occ_bool, :]               # (nocc, La)
            C             = W_A_occ.conj().T @ W_A_occ     # (La, La)
            C            *= 2.0                            # spin-unpolarized
        else:
            pref          = (2.0 * np.asarray(occ) - 1.0).astype(W_A.real.dtype, copy=False)
            # (La, Ls) * (Ls,) -> broadcast weights across columns, then @ (Ls, La)
            C             = (W_A_CT * pref) @ W_A

        if subtract_identity:
            np.fill_diagonal(C, np.diag(C) - 1.0)
        return C

    # BdG utilities (parse inputs)
    # Accept: (U_A, V_A) or stacked [U;V]
    if isinstance(W_A, tuple):
        U_A, V_A            = W_A
        if (W_A_CT is not None) and isinstance(W_A_CT, tuple):
            U_A_CT, V_A_CT  = W_A_CT
        else:
            U_A_CT          = U_A.conj().T # (La, Ls)
            V_A_CT          = V_A.conj().T
    else:
        if not stacked_uv:
            raise ValueError("For BdG modes, pass (U_A, V_A) tuple or set stacked_uv=True with stacked [U; V].")
        LsLa              = W_A.shape
        if len(LsLa) != 2 or (LsLa[0] % 2 != 0):
            raise ValueError("Stacked BdG array must have shape (2*Ls, La).")
        Ls                = LsLa[0] // 2
        U_A               = W_A[:Ls, :]
        V_A               = W_A[Ls:, :]
        U_A_CT            = U_A.conj().T
        V_A_CT            = V_A.conj().T

    La                    = U_A.shape[1]
    f                     = np.asarray(occ, dtype=U_A.real.dtype)          # quasiparticle occupations, length Ls
    if f.ndim != 1 or f.shape[0] != U_A.shape[0]:
        raise ValueError("`occ` for BdG must be a 1D array of length Ls (number of quasiparticle modes).")
    one_minus_f           = 1.0 - f

    # Efficient weighted Gram forms: (A^\dag * weights) @ A
    # N  = U f U^\dag + V (1-f) V^\dag  -> using (A_CT * w) @ A, where A_CT = A.conj().T and w broadcasts over columns.
    N                     = (U_A_CT * f)         @ U_A \
                          + (V_A_CT * one_minus_f) @ V_A

    if mode == "bdg-normal":
        if subtract_identity:
            np.fill_diagonal(N, np.diag(N) - 1.0)
        return N

    # mode == "bdg-full": build full Nambu matrix
    # F     = U (1-f) V^T + V f U^T
    # Ñ     = I - N
    F                     = (U_A_CT * one_minus_f).T @ V_A  +  (V_A_CT * f).T @ U_A  # (La, La)
    N_tilde               = np.eye(La, dtype=N.dtype) - N

    # Assemble G =
    # [ [ N,       F\dag ],
    #   [ F,     Ñ   ] ]
    G_upper_left          = N
    G_upper_right         = F.conj().T
    G_lower_left          = F
    G_lower_right         = N_tilde

    G                     = np.empty((2*La, 2*La), dtype=N.dtype)
    G[:La,     :La    ]   = G_upper_left
    G[:La,     La:    ]   = G_upper_right
    G[La:,     :La    ]   = G_lower_left
    G[La:,     La:    ]   = G_lower_right

    if subtract_identity:
        # subtract block-diag(I, I)
        idx               = np.arange(2*La)
        G[idx, idx]      -= 1.0

    return G

#################################### 
#! 2)  Multiple Slater determinants
####################################

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
        |ψ⟩ = Σ_k a_k |m_k⟩,    with normalization ⟨ψ|ψ⟩ = 1

    This function calculates the single-particle correlation matrix C for |ψ⟩, taking into account both diagonal (within the same determinant) and off-diagonal (between determinants differing by exactly two orbitals) contributions.

    - All determinants in `occ_list` must conserve particle number and have the same number of particles (N).
    - Off-diagonal contributions are included only for pairs of determinants that differ in exactly two orbital occupations.

    W_A : np.ndarray, shape (Ls, La)
        Single-particle orbital transformation matrix restricted to a subsystem.
    occ_list : Sequence[np.ndarray]
        List or tuple of boolean or {0,1} arrays, each of shape (Ls,), representing occupation vectors for each determinant.
    coeff : np.ndarray or None, optional
        1D complex array of coefficients (a_k) for the superposition, of length len(occ_list).
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
    #!  Equal part (m = n) …  sum_k |a_k|² · W_A\dag diag(n_k) W_A
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
                continue                    # only 2-orbital difference contributes

            q1, q2 = np.nonzero(diff)[0]    # the two differing orbitals
            # The ± sign from fermionic exchange:
            #   if |m⟩ has q1 occupied and |n⟩ has q1 empty  -> annihilate at q1
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
#! 3)  Generic many-body state
# ---------------------------------------------------------------------------

_MODE_SLATER     = 0
_MODE_BDG_N      = 1
_MODE_BDG_FULL   = 2

@numba.njit(inline="always")
def _popcount_u64(x: np.uint64) -> int:
    '''
    Quick population count for 64-bit unsigned integers.
    '''
    cnt = 0
    while x:
        x   &= x - np.uint64(1)
        cnt += 1
    return cnt

@numba.njit(inline="always")
def _apply_annihilation(idx: int, j: int, Ns: int) -> (int, int):
    """
    Return (new_idx, sign). If annihilation gives 0, return (0,0).
    """
    mask = 1 << (Ns - 1 - j)
    if idx & mask:  
        # occupied
        new_idx     = idx ^ mask
        # number of fermions to the left of j. The convension 
        # is to apply the operator to the first position only
        parity      = _popcount_u64(idx >> (Ns - j))
        sign        = -1 if parity & 1 else 1
        return new_idx, sign
    return 0, 0

@numba.njit(inline="always")
def _apply_creation(idx: int, i: int, Ns: int) -> (int, int):
    """
    Return (new_idx, sign). If creation gives 0, return (0,0).
    """
    mask = 1 << (Ns - 1 - i)
    if not (idx & mask): # empty
        new_idx = idx ^ mask
        parity  = _popcount_u64(idx >> (Ns - i))
        sign    = -1 if parity & 1 else 1
        return new_idx, sign
    return 0, 0

@numba.njit
def _corr_from_statevector_jit(psi                  : np.ndarray,
                            ns                      : int,
                            mode_code               : int,
                            subtract_identity       : bool,
                            spin_unpolarized_double : bool):
    """
    Compute the correlation matrix from a state vector.
    
    Para
    """
    
    dim = psi.size
    if dim != (1 << ns):
        raise ValueError("Many-body state length not 2^ns")

    # non-zero elements of the many-body state
    nz      = np.nonzero(psi)[0]

    # N_ij = <c_i^\dag  c_j>
    Nmat    = np.zeros((ns, ns), dtype=psi.dtype)
    for j in range(ns):
        for i in range(ns):
            acc = 0.0
            for kk in range(nz.size):
                idx         = int(nz[kk])
                amp         = psi[idx]

                # first apply anihilation
                idx1, s1    = _apply_annihilation(idx, j, ns)
                if s1 == 0:
                    continue

                # secondly apply creation
                idx2, s2    = _apply_creation(idx1, i, ns)
                if s2 == 0:
                    continue
                acc        += (s1 * s2) * np.conj(psi[idx2]) * amp
            Nmat[i, j] = acc

    # optional \times 2 for spin-unpolarized convention to match your Slater path
    if spin_unpolarized_double:
        for r in range(ns):
            for c in range(ns):
                Nmat[r, c] *= 2.0

    if mode_code == _MODE_BDG_N:
        if subtract_identity:
            for k in range(ns):
                Nmat[k, k] -= 1.0
        return Nmat

    if mode_code == _MODE_SLATER:
        if subtract_identity:
            for k in range(ns):
                Nmat[k, k] -= 1.0
            return Nmat
        else:
            return Nmat

    # F_ij = <c_i c_j>  (IMPORTANT: two ANNIHILATIONS, not creations)
    # The rest will follow from conjugation
    Fmat = np.zeros((ns, ns), dtype=psi.dtype)
    for j in range(ns):
        for i in range(ns):
            acc = 0.0
            for kk in range(nz.size):
                idx         = int(nz[kk])
                amp         = psi[idx]

                # apply first annihilation
                idx1, s1    = _apply_annihilation(idx, j, ns)
                if s1 == 0:
                    continue
                
                # apply second annihilation
                idx2, s2    = _apply_annihilation(idx1, i, ns)
                if s2 == 0:
                    continue
                
                acc        += (s1 * s2) * np.conj(psi[idx2]) * amp
            Fmat[i, j] = acc

    # build N_tilde = I - N (note: no extra \times 2 here; already matched above)
    Ntilde  = np.empty_like(Nmat)
    for r in range(ns):
        for c in range(ns):
            Ntilde[r, c] = (-Nmat[r, c])
    
    for k in range(ns):
        Ntilde[k, k] += 1.0

    G = np.empty((2*ns, 2*ns), dtype=psi.dtype)
    
    # TL: N
    for r in range(ns):
        for c in range(ns):
            G[r, c] = Nmat[r, c]
    # TR: F\dag 
    for r in range(ns):
        for c in range(ns):
            G[r, ns + c] = np.conj(Fmat[c, r])
    # BL: F
    for r in range(ns):
        for c in range(ns):
            G[ns + r, c] = Fmat[r, c]
    # BR: Ntilde
    for r in range(ns):
        for c in range(ns):
            G[ns + r, ns + c] = Ntilde[r, c]

    if subtract_identity:
        for k in range(2*ns):
            G[k, k] -= 1.0

    return G

# ---------------- user-facing wrapper ---------------- #

def corr_from_statevector(psi               : np.ndarray,
                        ns                  : int,
                        mode                : Literal["slater","bdg-normal","bdg-full"] = "slater",
                        subtract_identity   : bool = True,
                        spin_unpolarized    : bool = True) -> np.ndarray:
    """
    ψ-based correlators matching corr_single's conventions.

    Parameters
    ----------
    psi:
        (2**Ns,) - state vector for spinless particles
    Ns: int
        Number of single-particle states
    mode: 
        "slater" | "bdg-normal" | "bdg-full"
    subtract_identity: 
        subtract I (or block-diag(I,I) for bdg-full)
    spin_unpolarized: 
        if True, multiplies the NORMAL blocks by 2.0 to match Slater path.

    Returns
    -------
    (Ns,Ns) or (2Ns,2Ns)
    """
    if psi.ndim != 1 or psi.size != (1 << ns):
        raise ValueError("Many-body state must be 1D of length 2**Ns")
    
    # if psi.dtype not in (np.complex128, np.complex64):
    #     psi = psi.astype(np.complex128, copy=False)

    if mode == "slater":
        mcode = _MODE_SLATER
    elif mode == "bdg-normal":
        mcode = _MODE_BDG_N
    elif mode == "bdg-full":
        mcode = _MODE_BDG_FULL
    else:
        raise ValueError("The mode must be 'slater', 'bdg-normal', or 'bdg-full'")

    return _corr_from_statevector_jit(psi, int(ns), int(mcode), bool(subtract_identity), bool(spin_unpolarized))

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
        JAX kernel – one line of XLA-fused linear algebra.
        """
        occ_f   = occ.astype(W_A.dtype)
        C       = (jnp.conjugate(W_A).T * occ_f) @ W_A
        if subtract_identity:
            C   = C - jnp.eye(W_A.shape[1], dtype=W_A.dtype)
        return C

    #######################################

###########################################
#! End