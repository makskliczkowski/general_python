r"""
general_python/physics/sp/correlation_matrix.py
=====================

This module provides functions to construct the single-particle correlation matrix 
C_ij = <\psi |c_i ^\dag  c_j|\psi > for various types of quantum many-body states:

Functions
---------
- `corr_single`: Computes the correlation matrix for a single Slater determinant 
    specified by an occupation bit-string.
- `corr_superposition`: Computes the correlation matrix for a linear combination 
    of Slater determinants, |\psi > = \sum _k a_k |m_k>.
- `corr_from_state_vector`: Computes the correlation matrix for an arbitrary 
    many-body state given as a state vector (general but less efficient).

- Ls: Total number of one-particle orbitals or lattice sites.
- La: Size of the spatial subsystem A (first La sites are considered part of A).
- W: Ls \times Ls unitary matrix that diagonalizes the quadratic Hamiltonian.
- W_A: Ls \times La matrix containing the first La columns of W (rows: orbitals, columns: sites in A).

For a single Slater determinant with occupation vector `n` (shape (Ls,)), the reduced 
correlation matrix C_A is given by:

    C_A = W_A ^\dag  \cdot  diag(n) \cdot  W_A
    # raw multiplication (Eq. (3) PHYSICAL REVIEW LETTERS 125, 180604 (2020))

References
----------
"""

import numpy as np
from typing import Literal, Sequence, Tuple, Optional, Literal

import numpy as np
import numba
from torch import mode

try:
    from ...algebra.utils import JAX_AVAILABLE, Array
except ImportError:
    raise ImportError("Utilities package is required to use correlation_matrix module.")

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
            ndarray (Ls, La), rows = orbitals q, cols = sites i\inA.
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
            if True, uses selection by boolean mask and computes 2\cdot (W_occ^\dag W_occ).
        If False, uses `pref = 2\cdot occ - 1` trick.
    mode : {"slater","bdg-normal","bdg-full"}
        - "slater"     : C = <c_i^\dag c_j> (La\times La).
        - "bdg-normal" : N = <c_i^\dag c_j> for BdG (La\times La).
        - "bdg-full"   : Nambu G =
                        [[ <c^\dag c>, <c^\dag c^\dag> ],
                        [ <c   c>, <c   c^\dag> ]]  of shape (2La\times 2La).
    stacked_uv : bool
        If True in BdG modes, interpret W_A as vertically stacked [U_A; V_A].

    Returns
    -------
    ndarray
        - "slater" / "bdg-normal": (La, La)
        - "bdg-full"             : (2*La, 2*La)

    Notes (BdG, zero/finite T):
    ---------------------------
    Let c = U a + V a^\dag, with U,V \in C^{La\times Ls} (we use U_A,V_A as (Ls,La) row-major in orbitals; see below).
    For diagonal quasiparticle occupations f = diag(f_q), the standard equal-time correlations are
        N ≡ <c^\dag c>  = U f U^\dag + V (I - f) V^\dag,
        F ≡ <c   c>  = U (I - f) V^T + V f U^T,
        Ṅ ≡ <c   c^\dag> = U (I - f) U^\dag + V f V^\dag = I - N.
    Implementation uses row-major (orbitals q) storage: U_A has shape (Ls, La), so
        N = (U_A^\dag \cdot  f \cdot  U_A) + (V_A^\dag \cdot  (1-f) \cdot  V_A),
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

def corr_full(
    W                   : Array,            # Slater    : (Ls, L)
                                            # BdG       : (U, V) each (Ls, L), or stacked (2*Ls, L) with stacked_uv=True
    occ                 : Array,            # Slater    : n_q in {0,1};  BdG: f_q in [0,1]
    *,
    subtract_identity   : bool              = True,
    W_CT                : Optional[Array]   = None,
    raw                 : bool              = True,
    mode                : Literal["slater", "bdg-normal", "bdg-full"] = "slater",
    stacked_uv          : bool              = False) -> Array:
    r"""
    Full-system correlation matrix (no subsystem restriction).

    Parameters
    ----------
    W : 
        - "slater":
            single-particle orbitals (rows=orbitals q, cols=sites i), shape (Ls, L).
        - "bdg-*":
            either a tuple (U, V) with each (Ls, L), or stacked [U; V] of shape (2*Ls, L) if `stacked_uv=True`.
    occ :
        - "slater":
            occupations n_q \in {0,1}.
        - "bdg-*":
            quasiparticle occupations f_q \in [0,1].
    subtract_identity :
        - "slater"/"bdg-normal":
            subtract I on the normal block.
        - "bdg-full":
            subtract block-diag(I, I) on the 2L times 2L Nambu matrix.
    raw :
        - "slater" fast path using selection by boolean mask and computes 2\cdot (W_occ^\dag W_occ).
        If False, uses the (2\cdot occ-1) trick.
    mode : {"slater","bdg-normal","bdg-full"}
        - "slater"     : returns C = <c^\dag c>, shape (L, L).
        - "bdg-normal" : returns N = <c^\dag c> for BdG, shape (L, L).
        - "bdg-full"   : returns Nambu G =
                        [[ <c^\dag c>, <c^\dag c^\dag> ],
                        [ <c   c>, <c   c^\dag> ]] of shape (2L, 2L).
    stacked_uv :
        If True in BdG modes, interpret W as vertically stacked [U; V].

    Notes
    -----
    - For "slater", we keep the spin-unpolarized convention C = 2\cdot W_occ^\dag W_occ (you can drop the factor 2 if not needed).
    - For BdG with diagonal quasiparticle occupations f = diag(f_q):
        N = U f U^\dag + V (I - f) V^\dag,
        F = U (I - f) V^T + V f U^T,
        Ñ = I - N.
    """
    if mode == "slater":
        if W_CT is None:
            W_CT = W.conj().T # (N, Ns)
        
        Ls, L       = W.shape
        occ_bool    = np.asarray(occ, dtype=bool)
        nocc        = int(np.sum(occ_bool))
        if nocc == 0:
            C = np.zeros((L, L), dtype=W.dtype)
        elif raw:
            W_occ   = W[occ_bool, :]            # (nocc, L)
            C       = W_occ.conj().T @ W_occ    # (L, L)
            C      *= 2.0                       # spin-unpolarized doubling
        else:
            pref    = (2.0 * np.asarray(occ) - 1.0).astype(np.float64, copy=False)  # weights over orbitals
            C       = (W_CT * pref) @ W

        if subtract_identity:
            np.fill_diagonal(C, np.diag(C) - 1.0)
        return C

    # --- BdG parse ---
    if isinstance(W, tuple):
        U, V = W
        U_CT = U.conj().T
        V_CT = V.conj().T
    else:
        if not stacked_uv:
            raise ValueError("For BdG modes, pass (U, V) tuple or set stacked_uv=True with stacked [U; V].")
        sh = W.shape
        if len(sh) != 2 or (sh[0] % 2 != 0):
            raise ValueError("Stacked BdG array must have shape (2*Ls, L).")
        Ls = sh[0] // 2
        U = W[:Ls, :]
        V = W[Ls:, :]
        U_CT = U.conj().T
        V_CT = V.conj().T

    L = U.shape[1]
    f = np.asarray(occ, dtype=np.float64)  # (Ls,)
    if f.ndim != 1 or f.shape[0] != U.shape[0]:
        raise ValueError("`occ` for BdG must be a 1D array of length Ls (number of quasiparticle modes).")
    one_minus_f = 1.0 - f

    # N = U f U^\dag + V (1-f) V^\dag   via weighted Gram without forming diag(f)
    N = (U_CT * f) @ U + (V_CT * one_minus_f) @ V

    if mode == "bdg-normal":
        if subtract_identity:
            np.fill_diagonal(N, np.diag(N) - 1.0)
        return N

    # bdg-full: build G with F and Ñ = I - N
    F = (U_CT * one_minus_f).T @ V + (V_CT * f).T @ U   # (L, L)
    N_tilde = np.eye(L, dtype=N.dtype) - N

    G = np.empty((2*L, 2*L), dtype=N.dtype)
    G[:L, :L]   = N
    G[:L, L:]   = F.conj().T
    G[L:, :L]   = F
    G[L:, L:]   = N_tilde

    if subtract_identity:
        idx = np.arange(2*L)
        G[idx, idx] -= 1.0
    return G

@numba.njit
def corr_single2_slater_wick(corr: np.ndarray, ns: int, j: int = 0, l: int = 0):
    '''
    Compute the Wick contraction for a single Slater determinant.
    Mathematically:
    
    C_wick[i,k] = -C[i,k] C[j,l] + C[i,l] C[j,k]
    where C is the single-particle correlation matrix.
    
    or 
    $$ 
    C_{wick}^{(i,k)} = -C^{(i,k)} C^{(j,l)} + C^{(i,l)} C^{(j,k)}
    $$
    
    for i,k,j,l = 0,...,ns-1. This is useful for computing two-body correlation functions:
    <c_i^\dag c_j^\dag c_l c_k> = C_wick[i,k] = -C[i,k] C[j,l] + C[i,l] C[j,k]
    '''
    
    C_wick = np.zeros_like(corr)
    for k in range(ns):
        for i in range(ns):
            C_wick[i, k] += -corr[i, k] * corr[j, l] + corr[i, l] * corr[j, k]
    return C_wick

#################################### 
#! 2)  Multiple Slater determinants
####################################

def _haar_complex_unit_vector(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return a complex Haar-random unit vector of length n."""
    rng     = np.random.default_rng() if rng is None else rng
    z       = rng.normal(size=n) + 1j * rng.normal(size=n)
    z      /= np.linalg.norm(z)
    return z

@numba.njit(parallel=True, fastmath=True)
def _accumulate(C, occ_list, coeff, W_A, W_A_CT):
    gamma = len(occ_list)
    La = W_A.shape[1]
    for a in range(gamma):
        occ_a = occ_list[a]
        for b in range(a+1, gamma):
            diff = occ_a ^ occ_list[b]
            if diff.sum() != 2:
                continue
            q = np.nonzero(diff)[0]
            q1, q2 = q[0], q[1]
            sign = 1.0
            if occ_a[q2]:
                sign = -1.0
            coef = 2.0 * sign * np.conj(coeff[a]) * coeff[b]
            ua = W_A_CT[:, q1]
            vb = W_A[q2, :]
            for i in range(La):
                for j in range(La):
                    C[i,j] += coef * ua[i]*vb[j] + np.conj(coef) * W_A_CT[i,q2]*W_A[q1,j]

def corr_superposition(
    W_A                 : np.ndarray,                   # (Ls, La)
    occ_list            : Sequence[np.ndarray],         # list/tuple of bool arrays, each (Ls,)
    coeff               : Optional[np.ndarray] = None,  # 1D complex array of coefficients (a_k) for the superposition, length len(occ_list)
    *,
    W_A_CT              : Optional[np.ndarray] = None,  # (La, Ls)
    subtract_identity   : bool = True,
    raw                 : bool = True,
    mode                : Literal["slater", "bdg-normal", "bdg-full"] = "slater",
    stacked_uv          : bool = False  # if True and mode starts with "bdg", W_A is stacked [U; V] with shape (2*Ls, La)
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correlation matrix for a superposition of Slater determinants:
        |psi> = sum_k a_k |m_k>,  ||psi|| = 1.

    Matches the interface/conventions of `corr_single`:
    - spin-unpolarized factor of 2 on normal blocks,
    - subtract_identity behavior identical to `corr_single`,
    - accepts W_A_CT precompute,
    - same (mode, raw, stacked_uv) signature; only "slater" is implemented here.

    Returns
    -------
    C : ndarray
        (La, La) normal correlation matrix in "slater" mode.
    coeff : ndarray
        The (normalized) coefficient vector actually used.
    """
    
    if mode != "slater":
        raise NotImplementedError("corr_superposition currently supports mode='slater' only.")

    gamma = len(occ_list)
    if gamma == 0:
        raise ValueError("occ_list is empty")

    Ls, La          = W_A.shape[0], W_A.shape[1]
    occ_bool_list   = []
    for k, occ in enumerate(occ_list):
        occ = np.asarray(occ, dtype=bool)
        if occ.ndim != 1 or occ.size != Ls:
            raise ValueError(f"Occupation vector #{k} must be 1D of length Ls={Ls}.")
        occ_bool_list.append(occ)
        
    # Coefficients: use user-supplied, else draw Haar-random complex and normalize.
    if coeff is None:
        coeff = _haar_complex_unit_vector(gamma)
    else:
        if coeff.ndim != 1 or coeff.size != gamma:
            raise ValueError(f"coeff must be 1D of length gamma={gamma}.")
        # Normalize to ensure <psi|psi>=1 (does not change physics of C)
        norm = np.linalg.norm(coeff)
        if norm == 0:
            raise ValueError("coeff must not be the zero vector.")
        coeff = coeff / norm    
        
    # precompute W_A_CT if not provided
    if W_A_CT is None:
        W_A_CT = W_A.conj().T  # (La, Ls)

    # allocate me
    C = np.zeros((W_A.shape[1], W_A.shape[1]), dtype=W_A.dtype)

    # --------------------------------------------------------------------------------
    # Diagonal (m = n): sum_k |a_k|^2 * C[occ_k], with the same fast-path as corr_single
    # --------------------------------------------------------------------------------
    cache = {}
    for ck, occ in zip(coeff, occ_bool_list):
        key = occ.tobytes()
        if key not in cache:
            # Fast path: C_occ = 2 * (W_occ)\dag (W_occ)
            if np.any(occ):
                W_occ = W_A[occ, :] # (nocc, La)
                C_occ = (W_occ.conj().T @ W_occ) * 2.0
            else:
                C_occ = np.zeros((La, La), dtype=W_A.dtype)
            cache[key] = C_occ
        C += (abs(ck) ** 2) * cache[key]

    # --------------------------------------------------------------------------------
    # Off-diagonal (m \neq  n): only when configurations differ by exactly one hop
    #   i.e., XOR has Hamming weight 2. Let q_from be occupied in m, empty in n,
    #   and q_to be empty in m, occupied in n. Then the contribution is:
    #       2 * a_m* a_n * sgn * |phi_{q_to}><phi_{q_from}|  + h.c.
    #   where sgn = (-1)^(# occupied between q_from and q_to in m).
    # --------------------------------------------------------------------------------
    
    # go through the occupations
    for a, occ_a in enumerate(occ_list):
        for b in range(a + 1, gamma):
            occ_b   = occ_list[b]
            diff    = occ_a ^ occ_b         # XOR: where they differ
            if diff.sum() != 2:
                continue                    # only 2-orbital difference contributes

            # Identify source/destination indices for a -> b
            # q_from: occupied in a, empty in b
            # q_to  : empty in a, occupied in b
            q1, q2 = np.nonzero(diff)[0]    # the two differing orbitals
            
            # The +/-  sign from fermionic exchange:
            #   if |m> has q1 occupied and |n> has q1 empty  -> annihilate at q1
            #   otherwise the opposite.
            # We encode this in the prefactor below.

            sign =  1.0
            if occ_a[q2]: # annihilation order flips sign if starting point occupied
                sign = -1.0

            coef = 2.0 * sign * np.conjugate(coeff[a]) * coeff[b]

            #! Outer product  (La,1)\cdot (1,La) fused by BLAS as gemm
            C += coef * np.outer(W_A_CT[:, q1], W_A[q2, :])

            # and its Hermitian conjugate (because we skipped the term with indices
            # swapped in the loop)
            C += np.conjugate(coef) * np.outer(W_A_CT[:, q2], W_A[q1, :])

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

# -----------------------------------------

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

@numba.njit
def _corr_from_statevector2_slater_jit(psi  : np.ndarray,
                                ns          : int,
                                j           : int = 0,
                                l           : int = 0):
    r"""
    Compute the correlation matrix from a state vector.
    This is:
    <c_i^\dag c_j^\dag c_k c_l> - fixing j and l

    Parameters
    ----------
    psi: np.ndarray
        The state vector.
    ns: int
        The number of single-particle states.
    j: int
        The index of the second creation operator.
    l: int
        The index of the second annihilation operator.

    Returns
    -------
    np.ndarray
        The correlation matrix.
    """
    
    dim = psi.size
    if dim != (1 << ns):
        raise ValueError("Many-body state length not 2^ns")

    if ns > 64:
        raise ValueError("ns must be <= 64 for many-body states")

    # non-zero elements of the many-body state
    nz              = np.nonzero(psi)[0]

    # N_ij = <c_i^\dag c_j^\dag c_k c_l>
    Nmat            = np.zeros((ns, ns), dtype=psi.dtype)

    # annihilation
    for kk in range(nz.size):
        idx         = np.uint64(nz[kk])
        amp         = psi[int(idx)]
        idx1, s1    = _apply_annihilation(idx, l, ns)
        if s1 == 0:
            continue
        
        # Loop over k that are occupied in idx1 (second annihilation)
        for k in range(ns):

            # cannot annihilate at l
            if k == l:
                continue
            
            idx2, s2 = _apply_annihilation(idx1, k, ns)
            if s2 == 0:
                continue

            idx3, s3 = _apply_creation(idx2, j, ns)
            if s3 == 0:
                continue

            # Loop over i that are empty in idx3 (second creation)
            for i in range(ns):

                # cannot create twice at j
                if i == j:
                    continue

                idx4, s4 = _apply_creation(idx3, i, ns)
                if s4 == 0:
                    continue

                contrib     = (s1 * s2 * s3 * s4) * np.conj(psi[int(idx4)]) * amp
                Nmat[i,k]  += contrib

    return Nmat

# -----------------------------------------

def corr_from_statevector(psi               : np.ndarray,
                        ns                  : int,
                        mode                : Literal["slater","bdg-normal","bdg-full"] = "slater",
                        subtract_identity   : bool = True,
                        spin_unpolarized    : bool = True,
                        order               : int  = 2,
                        **kwargs) -> np.ndarray:
    """
    \psi -based correlators matching corr_single's conventions.

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
    
    if order == 4 and mode == "slater":
        # special case for 4-point Wick contractions
        return _corr_from_statevector2_slater_jit(psi, ns, j=kwargs.get("j", 0), l=kwargs.get("l", 0))
    elif order != 2:
        raise ValueError("order must be 2 or 4 for corr_from_statevector")
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
        JAX kernel - one line of XLA-fused linear algebra.
        """
        occ_f   = occ.astype(W_A.dtype)
        C       = (jnp.conjugate(W_A).T * occ_f) @ W_A
        if subtract_identity:
            C   = C - jnp.eye(W_A.shape[1], dtype=W_A.dtype)
        return C

    #######################################

###########################################
#! End