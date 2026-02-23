"""
UDT (Unitary-Diagonal-Triangular) decomposition for DQMC stability.

Provides methods for decomposing matrix products into UDT form to maintain
numerical stability in low-temperature (large beta) simulations.

Based on the algorithm by Loh et al., doi:10.1016/j.laa.2010.06.023
"""

from typing import NamedTuple, Any, Tuple
import numpy as np

try:
    from ..utils import get_backend, JAX_AVAILABLE
except ImportError:
    raise ImportError("UDT utilities require the 'utils' module from the algebra package.")

# -----------------------------------------------------------------------------------------------

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax.scipy as jsp
else:
    jnp = None
    jsp = None

class UDTState(NamedTuple):
    """
    Represents the state of a UDT decomposition: M = U @ diag(D) @ T.
    U is Unitary (Orthogonal), D is Diagonal, T is Upper Triangular.
    """
    U: Any
    D: Any
    T: Any

def _safe_diag_for_inverse(diag: Any, be: Any, tiny: float = 1e-30) -> Any:
    """
    Return a numerically safe diagonal for row scaling.

    Keeps the original phase/sign when |diag_i| is very small.
    """
    mag     = be.abs(diag)
    one     = be.ones_like(diag)
    denom   = be.where(mag > tiny, mag, one)
    phase   = diag / denom
    phase   = be.where(mag > tiny, phase, one)
    return be.where(mag > tiny, diag, tiny * phase)

def _loh_split(diag: Any, be: Any) -> Tuple[Any, Any]:
    """
    Loh split for a diagonal vector D:
      D = D_big * D_small, where |D_big| >= 1 and |D_small| <= 1.

    Returns
    -------
    (D_big_inv, D_small):
        D_big_inv = 1 / D_big, used in stable solves.
    """
    mag         = be.abs(diag)
    one         = be.ones_like(diag)
    large       = mag > 1.0
    d_big_inv   = be.where(large, 1.0 / diag, one)
    d_small     = be.where(large, one, diag)
    return d_big_inv, d_small

def udt_decompose(M: Any, backend: str = "default") -> UDTState:
    """
    Decompose matrix M into U @ diag(D) @ T.
    Uses QR decomposition.
    """
    be      = get_backend(backend)
    
    Q, R    = be.linalg.qr(M)
    # Keep signed/complex diagonal (do not use abs), as required by Loh split.
    D       = be.diag(R)
    D_safe  = _safe_diag_for_inverse(D, be)
    T       = be.diag(1.0 / D_safe) @ R
    
    return UDTState(U=Q, D=D, T=T)

def udt_fact_mult(Ml: Any, state: UDTState, backend: str = "default") -> UDTState:
    """
    Multiply UDT state by matrix Ml from the left: Ml @ (U @ diag(D) @ T).
    Returns new UDTState.
    """
    be              = get_backend(backend)
    
    # Ml @ U @ diag(D) @ T = (Ml @ U @ diag(D)) @ T
    M_tmp           = (Ml @ state.U) * state.D[None, :]
    
    Q_new, R_new    = be.linalg.qr(M_tmp)
    D_new           = be.diag(R_new)
    D_safe          = _safe_diag_for_inverse(D_new, be)
    
    T_new           = (be.diag(1.0 / D_safe) @ R_new) @ state.T
    
    return UDTState(U=Q_new, D=D_new, T=T_new)

def udt_inv_1p(state: UDTState, backend: str = "default") -> Any:
    """
    Compute (I + U @ diag(D) @ T)^{-1} stably using Loh's trick.
    """
    be = get_backend(backend)
    
    # C++ parity (UDT_QR::inv1P): solve((Db * U^H + Ds * T), Db * U^H)
    # where Db is inverse max-scale and Ds is min-scale from Loh split.
    db_inv, ds = _loh_split(state.D, be)
    u_h = state.U.conj().T
    mat = db_inv[:, None] * u_h + ds[:, None] * state.T
    rhs = db_inv[:, None] * u_h
    
    return be.linalg.solve(mat, rhs)

def udt_inv_sum(state_a: UDTState, state_b: UDTState, backend: str = "default") -> Any:
    """
    Compute (Ua @ diag(Da) @ Ta + Ub @ diag(Db) @ Tb)^{-1} stably.
    Based on the algorithm in doi:10.1016/j.laa.2010.06.023
    """
    be = get_backend(backend)
    
    # Let Ma = Ua Da Ta, Mb = Ub Db Tb
    # (Ma + Mb)^{-1} = [Ua Da Ta + Ub Db Tb]^{-1}
    # We use Loh's strategy: Da = Dmax_a Dmin_a, Db = Dmax_b Dmin_b
    
    abs_da = be.abs(state_a.D)
    abs_db = be.abs(state_b.D)
    one_a = be.ones_like(state_a.D)
    one_b = be.ones_like(state_b.D)

    Dmax_a = be.where(abs_da > 1.0, state_a.D, one_a)
    Dmin_a = be.where(abs_da > 1.0, one_a, state_a.D)
    
    Dmax_b = be.where(abs_db > 1.0, state_b.D, one_b)
    Dmin_b = be.where(abs_db > 1.0, one_b, state_b.D)
    
    invTb = be.linalg.inv(state_b.T)
    matL = (state_a.T @ invTb) * (Dmin_a / Dmax_b)[None, :]
    
    matR = (state_a.U.conj().T @ state_b.U) * (Dmin_b / Dmax_a)[None, :]
    
    sum_mat = matL + matR
    
    res = be.linalg.solve(sum_mat, state_a.U.conj().T / Dmax_a[:, None])
    res = invTb @ res / Dmax_b[:, None]
    
    return res

# -----------------------------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------------------------
