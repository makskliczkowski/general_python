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


----------------
File            : general_python/physics/sp/correlation_matrix.py
Author          : Maksymilian Kliczkowski
----------------
"""

import numpy as np
from typing import Literal, Sequence, Tuple, Optional, Dict, Literal, TYPE_CHECKING

import numpy as np
import numba

if TYPE_CHECKING:
    from ...algebra.utils import Array

#################################### 
#! Numba-optimized kernels for correlation matrix computation
###################################

@numba.njit(cache=True, fastmath=True)
def _corr_single_slater_raw_kernel(W_A: np.ndarray, occ_bool: np.ndarray, C: np.ndarray) -> None:
    r"""
    Numba kernel for fast Slater correlation: C = 2 * W_occ^\dag @ W_occ.
    
    Parameters
    ----------
    W_A : np.ndarray
        Eigenvector matrix (Ls, La).
    occ_bool : np.ndarray
        Boolean occupation mask (Ls,).
    C : np.ndarray
        Output correlation matrix (La, La), modified in-place.
    """
    Ls, La  = W_A.shape
    # Extract occupied rows - manual extraction for Numba
    nocc    = 0
    for i in range(Ls):
        if occ_bool[i]:
            nocc += 1
    
    if nocc == 0:
        # C already zeros
        return
    
    # Build W_occ manually (nocc, La)
    W_occ   = np.empty((nocc, La), dtype=W_A.dtype)
    idx     = 0
    for i in range(Ls):
        if occ_bool[i]:
            for j in range(La):
                W_occ[idx, j] = W_A[i, j]
            idx += 1
    
    # C = 2 * W_occ.conj().T @ W_occ - optimized with explicit loop
    # Exploit Hermitian symmetry: only compute upper triangle
    for i in range(La):
        for j in range(i, La):  # Upper triangle only
            s = 0.0 + 0.0j
            for k in range(nocc):
                s += np.conj(W_occ[k, i]) * W_occ[k, j]
            C[i, j] = 2.0 * s
            if i != j:
                C[j, i] = np.conj(C[i, j])  # Fill lower triangle by Hermitian conjugate

@numba.njit(cache=True, fastmath=True, parallel=True)
def _corr_superposition_diagonal_kernel(
    W_A: np.ndarray,
    occ_list_packed: np.ndarray,    # (gamma, Ls) as uint8
    coeffs_abs_sq: np.ndarray,      # (gamma,) real
    C: np.ndarray                   # (La, La) output
) -> None:
    r"""
    Numba kernel for diagonal contributions to superposition correlation matrix.
    
    Computes: C += sum_k |a_k|^2 * 2 * W_occ[k]^\dag @ W_occ[k]
    """
    gamma, Ls   = occ_list_packed.shape
    La          = W_A.shape[1]
    
    for k in numba.prange(gamma):
        # Count occupied
        nocc = 0
        for i in range(Ls):
            if occ_list_packed[k, i] != 0:
                nocc += 1
        
        if nocc == 0:
            continue
        
        # Extract occupied rows
        W_occ = np.empty((nocc, La), dtype=W_A.dtype)
        idx = 0
        for i in range(Ls):
            if occ_list_packed[k, i] != 0:
                for j in range(La):
                    W_occ[idx, j] = W_A[i, j]
                idx += 1
        
        # Compute C_k = W_occ^\dag @ W_occ with Hermitian symmetry
        weight = 2.0 * coeffs_abs_sq[k]
        for i in range(La):
            for j in range(i, La):  # Upper triangle only
                s = 0.0 + 0.0j
                for m in range(nocc):
                    s += np.conj(W_occ[m, i]) * W_occ[m, j]
                # Atomic add for thread safety
                val      = weight * s
                C[i, j] += val
                if i != j:
                    C[j, i] += np.conj(val)  # Hermitian symmetry

@numba.njit(cache=True, fastmath=True)
def _find_single_hop_pairs(occ_a: np.ndarray, occ_b: np.ndarray) -> Tuple[int, int, int]:
    """
    Find if two occupations differ by exactly one hop.
    
    Returns
    -------
    Tuple[int, int, int]
        (status, q_from, q_to) where:
        - status: 0 if not single hop, 1 if valid single hop
        - q_from: orbital occupied in a, empty in b
        - q_to: orbital empty in a, occupied in b
    """
    Ls          = occ_a.shape[0]
    diff_count  = 0
    q1          = -1
    q2          = -1
    
    for i in range(Ls):
        if occ_a[i] != occ_b[i]:
            diff_count += 1
            if diff_count == 1:
                q1 = i
            elif diff_count == 2:
                q2 = i
            else:
                return (0, -1, -1)  # More than 2 differences
    
    if diff_count != 2:
        return (0, -1, -1)
    
    # Determine which is q_from (occupied in a) and q_to (occupied in b)
    if occ_a[q1] != 0 and occ_a[q2] == 0:
        return (1, q1, q2)
    elif occ_a[q2] != 0 and occ_a[q1] == 0:
        return (1, q2, q1)
    else:
        return (0, -1, -1)

#################################### 
#! 1) Single Slater determinant
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
            # Use optimized Numba kernel for raw mode
            C               = np.zeros((La, La), dtype=W_A.dtype)
            _corr_single_slater_raw_kernel(W_A, occ_bool, C)
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
    W                   : 'Array',          # Slater    : (Ls, L)
                                            # BdG       : (U, V) each (Ls, L), or stacked (2*Ls, L) with stacked_uv=True
    occ                 : 'Array',          # Slater    : n_q in {0,1};  BdG: f_q in [0,1]
    *,
    subtract_identity   : bool              = True,
    W_CT                : Optional['Array'] = None,
    raw                 : bool              = True,
    mode                : Literal["slater", "bdg-normal", "bdg-full"] = "slater",
    stacked_uv          : bool              = False) -> 'Array':
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
            W_CT = W.conj().T # (L, Ls)
        
        Ls, L       = W.shape
        occ_bool    = np.asarray(occ, dtype=bool)
        nocc        = int(np.sum(occ_bool))
        if nocc == 0:
            C = np.zeros((L, L), dtype=W.dtype)
        elif raw:
            # Use optimized Numba kernel
            C       = np.zeros((L, L), dtype=W.dtype)
            _corr_single_slater_raw_kernel(W, occ_bool, C)
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

#################################### 
#! 1a) Wick contractions for single Slater determinant
####################################

def wick_distances(C_wick           : np.ndarray,
                   C_mb             : np.ndarray,
                   *,
                   tmp              : Optional[np.ndarray] = None,
                   eps              : float = 1e-12,
                   gate             : Optional[float] = None,
                   s0               : Optional[float] = None) -> Dict[str, float]:
    r"""
    Compute a set of normalized distances between two complex/real matrices
    (typically: Wick-predicted vs many-body 4-point slice).

    Distances returned
    ------------------
    - diff_fro2                 : ||C_wick - C_mb||_F^2
    - fro2_wick                 : ||C_wick||_F^2
    - fro2_mb                   : ||C_mb||_F^2
    - rel_fro2_sym              : ||\Delta||_F^2 / (||C_wick||_F^2 + ||C_mb||_F^2 + eps)
    - frac_diff_energy          : ||\Delta||_F^2 / (||\Delta||_F^2 + ||C_wick||_F^2 + ||C_mb||_F^2 + eps)
                                 (always in [0,1])
    - rel_fro_sym               : ||\Delta||_F / (||C_wick||_F + ||C_mb||_F + eps)
    - cosine_dist               : 1 - |<C_wick, C_mb>| / (||C_wick||_F ||C_mb||_F + eps)
                                 (in [0,1])
    - angle_dist                : arccos( |<.|.>| / (||.|| ||.|| + eps) ) / pi
                                 (in [0,1])

    Optional gating / weighting
    ---------------------------
    - gate: if provided and 0.5*(||C_wick||_F^2+||C_mb||_F^2) < gate, distances are set to 0.0
            (to ignore near-zero signals).
    - s0:   if provided, apply smooth weight w = S/(S+s0) with S = 0.5*(||C_wick||_F^2+||C_mb||_F^2)
            to the main scale-invariant distances (rel_fro2_sym, frac_diff_energy, rel_fro_sym, cosine_dist, angle_dist).
    """
    
    if C_wick.shape != C_mb.shape:
        raise ValueError("C_wick and C_mb must have the same shape.")
    if C_wick.ndim != 2:
        raise ValueError("C_wick and C_mb must be 2D arrays.")
    if C_wick.dtype != C_mb.dtype:
        # allow different dtypes but will upcast on subtract; better to keep consistent
        pass

    if tmp is None or tmp.shape != C_wick.shape:
        tmp = np.empty_like(C_wick)

    # squared Frobenius norm of the difference
    np.subtract(C_wick, C_mb, out=tmp)
    K2 = float(np.vdot(tmp, tmp).real)              # ||C_wick - C_mb||_F^2
    W2  = float(np.vdot(C_wick, C_wick).real)       # Wick squared Frobenius norm
    M2  = float(np.vdot(C_mb,   C_mb).real)         # Many-body squared Frobenius norm
    S   = 0.5 * (W2 + M2)                           # signal measure

    #! Safety gate
    if gate is not None and S < float(gate):
        return {
            "wick/diff_fro2"         : 0.0,
            "wick/fro2_wick"         : W2,
            "wick/fro2_mb"           : M2,
            "wick/rel_fro2_sym"      : 0.0,
            "wick/frac_diff_energy"  : 0.0,
            "wick/rel_fro_sym"       : 0.0,
            "wick/cosine_dist"       : 0.0,
            "wick/angle_dist"        : 0.0,
            "wick/signal"            : S,
            "wick/weight"            : 0.0 if s0 is None else 0.0,
        }

    # symmetric relative (squared) Frobenius error
    rel_fro2_sym                = K2 / (W2 + M2 + eps)

    # bounded "fraction of energy in the difference"
    frac_diff_energy            = K2 / (K2 + W2 + M2 + eps)

    # symmetric relative Frobenius error (non-squared)
    rel_fro_sym                 = np.sqrt(K2) / (np.sqrt(W2) + np.sqrt(M2) + eps)

    # cosine distance (pattern mismatch)
    dot                         = np.vdot(C_wick, C_mb)
    denom                       = (np.sqrt(W2 * M2) + eps)
    cos_sim                     = float(np.abs(dot) / denom)
    if cos_sim > 1.0:           cos_sim = 1.0  # numerical clamp
    cosine_dist                 = 1.0 - cos_sim

    # normalized angle distance in [0,1]
    angle_dist                  = float(np.arccos(cos_sim) / np.pi)
    weight                      = 1.0
    
    if s0 is not None:
        s0                      = float(s0)
        weight                  = float(S / (S + s0))
        rel_fro2_sym           *= weight
        frac_diff_energy       *= weight
        rel_fro_sym            *= weight
        cosine_dist            *= weight
        angle_dist             *= weight

    return {
            "wick/diff_fro2"        : K2,
            "wick/fro2_wick"        : W2,
            "wick/fro2_mb"          : M2,
            "wick/rel_fro2_sym"     : rel_fro2_sym,
            "wick/frac_diff_energy" : frac_diff_energy,
            "wick/rel_fro_sym"      : rel_fro_sym,
            "wick/cosine_dist"      : cosine_dist,
            "wick/angle_dist"       : angle_dist,
            "wick/signal"           : S,
            "wick/weight"           : weight,
        }

@numba.njit(inline="always")
def wick_2body_scalar(C: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    # <c_i^\dag c_j^\dag c_l c_k> = C[i,k]*C[j,l] - C[i,l]*C[j,k]
    return C[i, k] * C[j, l] - C[i, l] * C[j, k]

@numba.njit
def wick_2body(C: np.ndarray, j: int, l: int, out: np.ndarray):
    ns      = C.shape[0]
    C_jl    = C[j, l]
    for i in range(ns):
        C_il = C[i, l]
        for k in range(ns):
            out[i, k] = C[i, k] * C_jl - C_il * C[j, k]

def corr_4_from_wick(C: np.ndarray, j: int = 0, l: int = 0, *, C_wick: Optional[np.ndarray] = None) -> np.ndarray:
    r'''
    Compute the Wick contraction for a single Slater determinant. This computes 4-point correlation 
    functions of the form <c_i^\dag c_j^\dag c_l c_k> using Wick's theorem.
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
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square 2D array.")
    ns = C.shape[0]
    if not (0 <= j < ns and 0 <= l < ns):
        raise ValueError("j and l must be within [0, ns).")

    if C_wick is None:
        C_wick = np.empty_like(C)
    else:
        if C_wick.shape != C.shape or C_wick.dtype != C.dtype:
            raise ValueError("C_wick must match C in shape and dtype.")
        if not C_wick.flags.c_contiguous:
            C_wick = np.ascontiguousarray(C_wick)

    wick_2body(C, j, l, C_wick)
    return C_wick

#################################### 
#! 2) Multiple Slater determinants
####################################

def _haar_complex_unit_vector(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return a complex Haar-random unit vector of length n."""
    rng     = np.random.default_rng() if rng is None else rng
    z       = rng.normal(size=n) + 1j * rng.normal(size=n)
    z      /= np.linalg.norm(z)
    return z

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
    
    # Use optimized Numba kernel for diagonal contributions
    # Pack occupations into uint8 array for Numba compatibility
    occ_list_packed = np.array([occ.astype(np.uint8) for occ in occ_bool_list], dtype=np.uint8)
    coeffs_abs_sq   = np.abs(coeff)**2
    
    _corr_superposition_diagonal_kernel(W_A, occ_list_packed, coeffs_abs_sq, C)

    # --------------------------------------------------------------------------------
    # Off-diagonal (m \neq  n): only when configurations differ by exactly one hop
    #   i.e., XOR has Hamming weight 2. Let q_from be occupied in m,rmm empty in n,
    #   and q_to be empty in m, occupied in n. Then the contribution is:
    #       2 * a_m* a_n * sgn * |phi_{q_to}><phi_{q_from}|  + h.c.
    #   where sgn = (-1)^(# occupied between q_from and q_to in m).
    # --------------------------------------------------------------------------------
    
    # go through the occupations - optimized with Numba helper
    for a in range(gamma):
        occ_a = occ_list_packed[a]
        for b in range(a + 1, gamma):
            # Use Numba helper to check for single-hop
            occ_b                   = occ_list_packed[b]
            status, q_from, q_to    = _find_single_hop_pairs(occ_a, occ_b)
            if status == 0:
                continue

            # The +/-  sign from fermionic exchange
            sign = -1.0 if occ_a[q_to] != 0 else 1.0
            coef = 2.0 * sign * np.conjugate(coeff[a]) * coeff[b]

            #! Outer product  (La,1)\cdot (1,La) fused by BLAS as gemm
            C   += coef * np.outer(W_A_CT[:, q_from], W_A[q_to, :])

            # and its Hermitian conjugate (because we skipped the term with indices
            # swapped in the loop)
            C   += np.conjugate(coef) * np.outer(W_A_CT[:, q_to], W_A[q_from, :])

    #! Identity
    if subtract_identity:
        np.fill_diagonal(C, C.diagonal() - 1.0)
    
    return C, coeff

# ---------------------------------------------------------------------------
#! 3) Generic many-body state
# ---------------------------------------------------------------------------

_MODE_SLATER     = 0
_MODE_BDG_N      = 1
_MODE_BDG_FULL   = 2

# -----------------------------------------
#! Endian conversion utilities
# -----------------------------------------

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
def _orb_to_bitpos(ns: int, orb: int) -> int:
    ''' Convert orbital index to bit position (MSB0).'''
    # MSB0 convention: orb 0 -> bit ns-1, orb ns-1 -> bit 0
    return ns - 1 - orb

@numba.njit(inline="always")
def _bitpos_to_orb(ns: int, bitpos: int) -> int:
    ''' Convert bit position (MSB0) to orbital index.'''
    return ns - 1 - bitpos

@numba.njit(inline="always")
def _mask_at(ns: int, orb: int) -> np.uint64:
    ''' Return mask with 1 at orbital `orb` (MSB0) for `ns` orbitals.'''
    return np.uint64(1) << np.uint64(_orb_to_bitpos(ns, orb))

# -----------------------------------------

@numba.njit(inline="always")
def _apply_annihilation(idx: np.uint64, orb: int, ns: int) -> (int, int):
    r"""
    Apply c_orb on |idx>. Return (new_idx, sign). If result is 0, return (0,0).
    """
    m = _mask_at(ns, orb)
    if idx & m:
        new_idx = idx ^ m
        # fermions to the left (more significant bits): shift by (bitpos+1) = ns - orb
        parity  = _popcount_u64(idx >> np.uint64(ns - orb))
        sign    = -1 if (parity & 1) else 1
        return new_idx, sign
    return np.uint64(0), 0

@numba.njit(inline="always")
def _apply_creation(idx: np.uint64, orb: int, ns: int) -> tuple[np.uint64, int]:
    r"""
    Apply c_orb^\dag on |idx>. Return (new_idx, sign). If result is 0, return (0,0).
    """
    m = _mask_at(ns, orb)
    if not (idx & m):
        new_idx = idx ^ m
        parity  = _popcount_u64(idx >> np.uint64(ns - orb))
        sign    = -1 if (parity & 1) else 1
        return new_idx, sign
    return np.uint64(0), 0

# -----------------------------------------

@numba.njit
def corr_from_slater(psi                    : np.ndarray,
                    ns                      : int,
                    spin_unpolarized_double : bool,
                    subtract_identity       : bool,
                    out                     : Optional[np.ndarray] = None
                    ) -> np.ndarray:
    r"""
    Compute N[i,j] = <c_i^\dag c_j> for a generic many-body state psi (spinless, MSB0 indexing).
    """
    
    if psi.ndim != 1:
        raise ValueError("psi must be 1D")
    
    if ns <= 0 or ns > 63:
        # 2^ns statevector is anyway infeasible for large ns; 63 is a safe bit limit.
        raise ValueError("ns must be in [1, 63]")

    dim         = psi.size
    if dim != (1 << ns):
        raise ValueError("psi length must be 2^ns")

    full_mask   = (np.uint64(1) << np.uint64(ns)) - np.uint64(1)

    nz          = np.nonzero(psi)[0]
    N           = np.zeros((ns, ns), dtype=psi.dtype) if out is None else out

    # loop over non-zero amplitudes
    for t in range(nz.size):
        idx0        = np.uint64(nz[t])
        amp0        = psi[int(idx0)]

        # occupied orbitals in idx0:
        occ_bits    = idx0 & full_mask

        while occ_bits:
            lsb         = occ_bits & -occ_bits
            bitpos      = int(np.log2(lsb))
            j           = _bitpos_to_orb(ns, bitpos)

            # apply annihilation and get new idx1
            idx1, s1    = _apply_annihilation(idx0, j, ns)
            if s1 != 0:
                # empty orbitals in idx1:
                emp_bits        = (~idx1) & full_mask

                while emp_bits:
                    lsb2        = emp_bits & -emp_bits
                    bitpos2     = int(np.log2(lsb2))
                    i           = _bitpos_to_orb(ns, bitpos2)
                    
                    # apply creation
                    idx2, s2    = _apply_creation(idx1, i, ns)
                    if s2 != 0:
                        N[i, j] += (s1 * s2) * np.conj(psi[int(idx2)]) * amp0

                    emp_bits    &= emp_bits - np.uint64(1)

            occ_bits &= occ_bits - np.uint64(1)

    if spin_unpolarized_double:
        N *= 2.0

    if subtract_identity:
        for k in range(ns):
            N[k, k] -= 1.0

    return N

@numba.njit
def corr_from_bdg(psi                   : np.ndarray,
                ns                      : int,
                spin_unpolarized_double : bool,
                subtract_identity       : bool,
                out                     : Optional[np.ndarray] = None
                ) -> np.ndarray:
    r"""
    Return G (2ns,2ns) with blocks:
      TL = N = <c^\dag c>
      TR = F^\dag
      BL = F = <c c>
      BR = I - N  (number-conserving convention)
    """
    
    if psi.ndim != 1:
        raise ValueError("psi must be 1D")
    if ns <= 0 or ns > 63:
        raise ValueError("ns must be in [1, 63]")
    if psi.size != (1 << ns):
        raise ValueError("psi length must be 2^ns")

    full_mask = (np.uint64(1) << np.uint64(ns)) - np.uint64(1)
    nz        = np.nonzero(psi)[0]

    N = np.zeros((ns, ns), dtype=psi.dtype)
    F = np.zeros((ns, ns), dtype=psi.dtype)

    # --- N[i,j] = <c_i^\dag c_j>
    for t in range(nz.size):
        idx0 = np.uint64(nz[t])
        amp0 = psi[int(idx0)]

        occ_bits = idx0 & full_mask
        while occ_bits:
            lsb     = occ_bits & -occ_bits
            bitpos  = int(np.log2(lsb))
            j       = _bitpos_to_orb(ns, bitpos)

            idx1, s1 = _apply_annihilation(idx0, j, ns)
            if s1 != 0:
                emp_bits = (~idx1) & full_mask
                while emp_bits:
                    lsb2    = emp_bits & -emp_bits
                    bitpos2 = int(np.log2(lsb2))
                    i       = _bitpos_to_orb(ns, bitpos2)

                    idx2, s2 = _apply_creation(idx1, i, ns)
                    if s2 != 0:
                        N[i, j] += (s1 * s2) * np.conj(psi[int(idx2)]) * amp0

                    emp_bits &= emp_bits - np.uint64(1)

            occ_bits &= occ_bits - np.uint64(1)

    if spin_unpolarized_double:
        N *= 2.0

    # --- F[i,j] = <c_i c_j> (two annihilations)
    for t in range(nz.size):
        idx0 = np.uint64(nz[t])
        amp0 = psi[int(idx0)]

        occ_bits_j = idx0 & full_mask
        while occ_bits_j:
            lsbj    = occ_bits_j & -occ_bits_j
            bitposj = int(np.log2(lsbj))
            j       = _bitpos_to_orb(ns, bitposj)

            idx1, s1 = _apply_annihilation(idx0, j, ns)
            if s1 != 0:
                occ_bits_i = idx1 & full_mask
                while occ_bits_i:
                    lsbi    = occ_bits_i & -occ_bits_i
                    bitposi = int(np.log2(lsbi))
                    i       = _bitpos_to_orb(ns, bitposi)

                    idx2, s2 = _apply_annihilation(idx1, i, ns)
                    if s2 != 0:
                        F[i, j] += (s1 * s2) * np.conj(psi[int(idx2)]) * amp0

                    occ_bits_i &= occ_bits_i - np.uint64(1)

            occ_bits_j &= occ_bits_j - np.uint64(1)

    # build G
    G = np.empty((2*ns, 2*ns), dtype=psi.dtype)

    # TL: N
    for r in range(ns):
        for c in range(ns):
            G[r, c] = N[r, c]

    # TR: F^\dag
    for r in range(ns):
        for c in range(ns):
            G[r, ns + c] = np.conj(F[c, r])

    # BL: F
    for r in range(ns):
        for c in range(ns):
            G[ns + r, c] = F[r, c]

    # BR: I - N
    for r in range(ns):
        for c in range(ns):
            G[ns + r, ns + c] = -N[r, c]
    for k in range(ns):
        G[ns + k, ns + k] += 1.0

    if subtract_identity:
        for k in range(2*ns):
            G[k, k] -= 1.0

    return G

@numba.njit
def corr_4_from_slater(psi: np.ndarray, ns: int, j: int = 0, l: int = 0, *, out: Optional[np.ndarray] = None) -> np.ndarray:
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
    
    if psi.ndim != 1:
        raise ValueError("psi must be 1D")
    
    if ns <= 0 or ns > 63:
        raise ValueError("ns must be in [1, 63] for uint64 bitmasks")
    
    if psi.size != (1 << ns):
        raise ValueError("Many-body state length must be 2^ns")
    
    if j < 0 or j >= ns or l < 0 or l >= ns:
        raise ValueError("j,l out of range")

    full_mask   = (np.uint64(1) << np.uint64(ns)) - np.uint64(1)
    nz          = np.nonzero(psi)[0]
    out         = np.zeros((ns, ns), dtype=psi.dtype) if out is None else out

    for t in range(nz.size):
        idx0    = np.uint64(nz[t])
        amp0    = psi[int(idx0)]

        # apply c_l
        idx1, s1 = _apply_annihilation(idx0, l, ns)
        if s1 == 0:
            continue

        # iterate only occupied orbitals k in idx1, excluding l
        occ_bits = idx1 & full_mask
        while occ_bits:
            lsb     = occ_bits & -occ_bits
            bitpos  = int(np.log2(lsb))
            k_orb   = ns - 1 - bitpos   # MSB0: bitpos -> orbital

            occ_bits &= occ_bits - np.uint64(1)

            if k_orb == l:
                continue

            idx2, s2    = _apply_annihilation(idx1, k_orb, ns)
            if s2 == 0:
                continue

            # apply c_j^\dag (fixed)
            idx3, s3    = _apply_creation(idx2, j, ns)
            if s3 == 0:
                continue

            pref = (s1 * s2 * s3) * amp0

            # iterate only empty orbitals i in idx3, excluding j
            emp_bits    = (~idx3) & full_mask
            while emp_bits:
                lsb2        = emp_bits & -emp_bits
                bitpos2     = int(np.log2(lsb2))
                i_orb       = ns - 1 - bitpos2

                emp_bits   &= emp_bits - np.uint64(1)

                if i_orb == j:
                    continue

                idx4, s4    = _apply_creation(idx3, i_orb, ns)
                if s4 == 0:
                    continue

                out[i_orb, k_orb] += (pref * s4) * np.conj(psi[int(idx4)])

    return out
    
# -----------------------------------------

def corr_from_statevector(psi               : np.ndarray,
                        ns                  : int,
                        mode                : Literal["slater","bdg-normal","bdg-full"] = "slater",
                        subtract_identity   : bool = True,
                        spin_unpolarized    : bool = True,
                        order               : int  = 2,
                        out                 : Optional[np.ndarray] = None,
                        **kwargs) -> np.ndarray | None:
    r"""
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
        return corr_4_from_slater(psi, ns, out=out, **kwargs)
    
    elif order == 2:
        if mcode == _MODE_SLATER:
            return corr_from_slater(psi, ns, spin_unpolarized, subtract_identity, out=out)
        elif mcode in (_MODE_BDG_N, _MODE_BDG_FULL):
            return corr_from_bdg(psi, ns, spin_unpolarized, subtract_identity, out=out)
        
    elif order != 2:
        raise NotImplementedError("Only order=2 and order=4 (slater) are implemented.")

    return out

###########################################
#! JAX
###########################################

try:
    import  jax
    import  jax.numpy as jnp
    from    functools import partial

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

except ImportError:
    # No JAX available
    pass

###########################################
#! End
###########################################