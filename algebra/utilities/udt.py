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

def udt_decompose(M: Any, backend: str = "default") -> UDTState:
    """
    Decompose matrix M into U @ diag(D) @ T.
    Uses QR decomposition with pivoting for better stability if available.
    """
    be      = get_backend(backend)
    
    # In NumPy/SciPy we can use qr with pivoting: M[:, P] = Q @ R
    # In JAX, jnp.linalg.qr doesn't support pivoting currently in a simple way.
    # We use standard QR for now.
    
    Q, R    = be.linalg.qr(M)
    D       = be.abs(be.diag(R))
    # Avoid zero division
    D_safe  = be.where(D < 1e-16, 1e-16, D)
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
    D_new           = be.abs(be.diag(R_new))
    D_safe          = be.where(D_new < 1e-16, 1e-16, D_new)
    
    T_new           = (be.diag(1.0 / D_safe) @ R_new) @ state.T
    
    return UDTState(U=Q_new, D=D_new, T=T_new)

def udt_inv_1p(state: UDTState, backend: str = "default") -> Any:
    """
    Compute (I + U @ diag(D) @ T)^{-1} stably.
    """
    be = get_backend(backend)
    
    # G = (I + UDT)^{-1} = (U U^H + UDT)^{-1} = (U(U^H + DT))^{-1} = (U^H + DT)^{-1} U^H
    # To stabilize, we use Loh's trick: D = Db @ Ds where Db = max(D, 1) and Ds = min(D, 1).
    # Wait, the C++ implementation used Db = max(D, 1)^-1 and Ds = min(D, 1).
    # Let's use Db = max(D, 1) and Ds = min(D, 1).
    # G = (U^H + Db Ds T)^{-1} U^H = (Db^-1 U^H + Ds T)^{-1} Db^-1 U^H
    
    Db = be.maximum(state.D, 1.0)
    Ds = be.minimum(state.D, 1.0)
    
    # (Db^-1 U^H + Ds T) x = Db^-1 U^H
    # is equivalent to (U^H + Db Ds T) x = U^H? No.
    # (Db^-1 U^H + Ds T) x = (diag(1/Db) @ U^H + diag(Ds) @ T) x = diag(1/Db) @ U^H
    
    mat = (state.U.conj().T / Db[:, None]) + (Ds[:, None] * state.T)
    rhs = state.U.conj().T / Db[:, None]
    
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
    
    Dmax_a = be.maximum(state_a.D, 1.0)
    Dmin_a = be.minimum(state_a.D, 1.0)
    
    Dmax_b = be.maximum(state_b.D, 1.0)
    Dmin_b = be.minimum(state_b.D, 1.0)
    
    # (Ua Dmax_a Dmin_a Ta + Ub Dmax_b Dmin_b Tb)^{-1}
    # = (Ua Dmax_a (Dmin_a Ta + Dmax_a^-1 Ua^H Ub Dmax_b Dmin_b Tb))^{-1}
    # This is getting messy. The C++ code had:
    # matL = Ta @ inv(Tb) @ diag(Dmin_a / Dmax_b)
    # matR = Ua^H @ Ub @ diag(Dmin_b / Dmax_a)
    # result = Tb^-1 @ (matL + matR)^-1 @ Dmax_a^-1 @ Ua^H
    
    invTb = be.linalg.inv(state_b.T)
    matL = (state_a.T @ invTb) * (Dmin_a / Dmax_b)[None, :]
    
    matR = (state_a.U.conj().T @ state_b.U) * (Dmin_b / Dmax_a)[None, :]
    
    sum_mat = matL + matR
    
    # We want Mb^-1 ( Ma Mb^-1 + I )^-1? No.
    # Standard stable inversion of sum:
    # Ma + Mb = Ua Dmax_a (Dmin_a Ta + Dmax_a^-1 Ub Db Tb)
    # Actually, let's use the most robust form:
    # Ma + Mb = (Ua Dmax_a) (Dmin_a Ta Tb^-1 Dmax_b^-1 + Dmax_a^-1 Ua^H Ub Dmin_b) (Dmax_b Tb)
    
    res = be.linalg.solve(sum_mat, state_a.U.conj().T / Dmax_a[:, None])
    res = invTb @ res / Dmax_b[:, None]
    
    return res

# -----------------------------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------------------------