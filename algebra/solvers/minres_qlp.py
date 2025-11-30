r'''
Solves symmetric (possibly singular) linear systems:

$$
(A - \sigma I)x = b,
$$

or least-squares problems $ \min ||(A-\sigma I)x - b||_2 $. 
It finds the minimum length solution
$ x $ (the minimum $||x||_2$ ) among all least-squares solutions.

Based on the algorithm by Choi, Paige, and Saunders.

Mathematical Sketch (Simplified):
---------------------------------
MINRES applies the Lanczos process to $ A' = A - \sigma I $ 
(implicitly using preconditioner M if provided, on $ M^{-1}A' $)
to generate an orthonormal basis

$$
V_k = [v_1, ..., v_k]
$$

for the Krylov subspace 

$$
K_k(M^{-1}A', M^{-1}r_0),
$$

and a symmetric tridiagonal matrix

$$
T_k = V_k^T (M^{-1}A') V_k.
$$

The residual norm $ ||r_k|| = ||b - A'x_k|| $ is minimized over 
$ x_k \in x_0 + K_k $.
This involves solving the tridiagonal system 
$ T_k y_k = \beta_1 e_1 $ implicitly
using QR factorization via Givens rotations 
($ Q_k T_k = R_k $).
$ ||r_k|| = |(\beta_1 Q_k e_1)_{k+1}| = |\phi_k| $.
The solution is updated as $ x_k = x_0 + V_k y_k $.

MINRES-QLP extends this by applying 
right Givens rotations ($ R_k P_k = L_k $)
to get an LQ factorization of $ T_k $. This allows stable computation of the
minimum length solution 

$$
x_k = x_0 + W_k t_k 
$$

where $ W_k = V_k P_k^T $ and
$ L_k t_k = \beta_1 Q_k e_1 $.
This is particularly useful for singular or
ill-conditioned systems.

References:
-----------
    - Paige, C. C., & Saunders, M. A. (1975). Solution of sparse indefinite systems
        of linear equations. SIAM Journal on Numerical Analysis, 12(4), 617-629. (MINRES)
    - Choi, S.-C. T., Paige, C. C., & Saunders, M. A. (2011). MINRES-QLP: A
        Krylov subspace method for indefinite or singular symmetric systems.
        SIAM Journal on Scientific Computing, 33(4), 1810-1836. (MINRES-QLP)
        
-----------------------------------------------------------------------------
file:       general_python/algebra/solvers/minres_qlp.py
author:     Maksymilian Kliczkowski
desc:       Native MINRES-QLP solver implementation for symmetric systems.
            Supports both JAX and NumPy backends.
date:       2025-02-01
-----------------------------------------------------------------------------
'''

from typing import Optional, Callable, Tuple, List, Union, Any, NamedTuple, Type
import numpy as np
import enum
import inspect

# Base Solver classes and types
try:
    from ..solver import Solver, SolverResult, SolverError, SolverErrorMsg, SolverType, Array, MatVecFunc, StaticSolverFunc, _sym_ortho
    from ..preconditioners import Preconditioner, PreconitionerApplyFun
except ImportError:
    raise ImportError("QES package is required to use MINRES-QLP solver.")

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    import jax.lax as lax
    JAX_AVAILABLE   = True
except ImportError:
    JAX_AVAILABLE   = False
    jax             = None
    jnp             = np
    lax             = None

# Numba import (currently MINRES-QLP NumPy version is not Numba-compiled due to complexity)
try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    _NUMBA_AVAILABLE = False

# ##############################################################################
#! Constants & Flags
# ##############################################################################

_MACHEPS        = np.finfo(np.float64).eps
_TINY           = np.sqrt(_MACHEPS) # Use sqrt(macheps) as a general small threshold
_MIN_NORM       = 1e-15             # Threshold for norms/denominators near zero

# Constants for QLP path switching and termination
_MAX_X_NORM    = 1.0e+15            # Increased from original
_MAX_A_COND    = 1.0e+15            # Condition number limit
_TRANS_A_COND  = 1.0e+7             # Threshold to switch to QLP updates

class MinresQLPFlag(enum.IntEnum):
    ''' 
    Convergence flags for MINRES-QLP solver.
    '''
    PROCESSING          = -2        # Still processing
    BETA_ZERO           = -1        # beta_k = 0. Solution is an eigenvector.
    X_ZERO              = 0         # Found zero solution x=0 because initial residual was zero.
    RESID_RTOL          = 1         # Converged: ||r_k|| / ||b|| <= tol
    RESID_AR            = 2         # Converged: ||A r_k|| / (||A|| ||r_k||) <= tol (approx)
    RESID_EPS           = 3         # Converged: ||r_k|| <= eps_r = max(eps*||b||, eps*||A||*||x||) (approx)
    RESID_EPS_AR        = 4         # Converged: Both RESID_EPS and RESID_AR criteria met.
    EIGEN               = 5         # Converged: Found eigenvector (related to BETA_ZERO).
    X_NORM_LIMIT        = 6         # Exceeded ||x|| limit (MAXXNORM). Solution may diverge.
    A_COND_LIMIT        = 7         # Exceeded Acond limit. System ill-conditioned.
    MAX_ITER            = 8         # Reached max iterations without convergence.
    SINGULAR            = 9         # System appears singular (gamma_k near zero in QLP).
    PRECOND_INDEF       = 10        # Preconditioner detected as indefinite or singular.

# ##############################################################################
#! Core MINRES-QLP Logic
# ##############################################################################

class MinresQLPState(NamedTuple):
    """
    Full state matching C++ MINRES_QLP_s protected members.
    """
    k: int
    
    # Solution
    x_k: Any            # Current solution vector
    x_km1: Any          # Previous solution (needed for QLP reconstruction)
    
    # Search Directions (W)
    w_k: Any
    w_km1: Any
    w_km2: Any

    # Lanczos Vectors
    v_k: Any            # Current normalized Lanczos vector (v)
    z_k: Any            # Current residual direction (z)
    z_km1: Any
    z_km2: Any

    # Scalars - Lanczos
    beta_k: float       # beta_{k+1} (current)
    beta_km1: float     # beta_k (previous)
    beta_last: float    # beta_{k-1}
    
    # Scalars - Left Rotation Q_{k-1} & Q_k
    c_k_1: float; s_k_1: float  # c_{k}, s_{k}
    delta_k: float      # delta_{k+1} (stored for next iter)
    phi_k: float        # phi_k (RHS element)
    eps_k: float        # epsilon_k
    eps_kp1: float      # epsilon_{k+1}

    # Scalars - Right Rotations P
    c_k_2: float; s_k_2: float  # P_{k-2, k} (prev)
    c_k_3: float; s_k_3: float  # P_{k-1, k} (curr)
    
    # Tridiagonal / LQ Factors
    gamma_k: float
    gamma_km1: float; gamma_km2: float; gamma_km3: float
    
    theta_k: float
    theta_km1: float; theta_km2: float
    
    eta_k: float; eta_km1: float; eta_km2: float
    
    tau_k: float; tau_km1: float; tau_km2: float
    
    # Solution Update Scalars (Mu)
    mu_k: float; mu_km1: float; mu_km2: float; mu_km3: float; mu_km4: float
    
    # QLP Specific History
    qlp_iter: int
    gamma_qlp_k: float; gamma_qlp_km1: float
    theta_qlp_k: float
    mu_qlp_k: float; mu_qlp_km1: float

    # Norms & Status
    xl2norm_k: float    # ||x_{k-2}||
    xnorm_k: float      # ||x_k||
    a_norm: float
    a_cond: float
    gamma_min: float
    gamma_min_km1: float; gamma_min_km2: float
    
    relres: float
    flag: int

# ##############################################################################
#! Single MINRES-QLP Iteration Step
# ##############################################################################

def _minres_qlp_step(state          : MinresQLPState, 
                    matvec          : Callable, 
                    precond_apply   : Callable, 
                    tol             : float, 
                    maxiter         : int, 
                    backend         : Any) -> MinresQLPState:
    """
    Single iteration of MINRES-QLP matching C++ implementation exactly.
    """
    s = state
    
    # ==========================================================================
    # 1. PRECONDITIONED LANCZOS
    # ==========================================================================
    # C++ line 89-90: _beta_km1 = _beta_k; v = z_k / _beta_km1;
    beta_last = s.beta_km1
    beta_km1 = s.beta_k
    
    # Normalize (with protection)
    inv_beta = backend.where(beta_km1 == 0, 1.0, 1.0 / beta_km1)
    v = s.z_k * inv_beta
    
    # C++ line 91: z_k = this->matVecFun_(v, this->reg_);
    z_k = matvec(v)
    
    # C++ line 92: if (k > 0) z_k -= z_km2 * _beta_km1 / _beta_last;
    if s.k > 0:
        safe_beta_last = backend.where(beta_last == 0, 1.0, beta_last)
        z_k = z_k - s.z_km2 * (beta_km1 / safe_beta_last)
    
    # C++ line 93: _alpha = arma::cdot(z_k, v);
    alpha = backend.real(backend.vdot(z_k, v))
    
    # C++ line 94: z_k -= z_km1 * _alpha / _beta_km1;
    z_k = z_k - s.z_km1 * (alpha * inv_beta)
    
    # C++ line 95: z_km2 = z_km1; z_km1 = z_k;
    z_km2_next = s.z_km1
    z_km1_next = z_k
    
    # C++ lines 98-108: Apply preconditioner and compute beta_{k+1}
    if precond_apply:
        z_k_precond = precond_apply(z_k)
        beta_k_sq = backend.real(backend.vdot(z_k, z_k_precond))
        beta_k = backend.sqrt(backend.maximum(beta_k_sq, 0.0))
    else:
        z_k_precond = z_k
        beta_k = backend.linalg.norm(z_k)
    
    # C++ line 116: _pnorm_rho_k = std::sqrt(algebra::norm(_beta_last, _alpha, _beta_k));
    pnorm_rho_k = backend.sqrt(beta_last**2 + alpha**2 + beta_k**2)
    
    # ==========================================================================
    # 2. LEFT ROTATION Q_{k-1} (Previous Reflection)
    # ==========================================================================
    # C++ lines 125-131
    dbar = s.delta_k
    delta = (s.c_k_1 * dbar) + (s.s_k_1 * alpha)
    eps_k = s.eps_kp1
    eps_kp1 = s.s_k_1 * beta_k
    gammabar = (s.s_k_1 * dbar) - (s.c_k_1 * alpha)
    delta_k = -s.c_k_1 * beta_k
    deltaqlp = delta  # Saved for MINRES update
    
    # ==========================================================================
    # 3. CURRENT LEFT REFLECTION Q_k
    # ==========================================================================
    # C++ lines 139-146
    gamma_km3 = s.gamma_km2
    gamma_km2 = s.gamma_km1
    gamma_km1_prev = s.gamma_k
    
    c_k_1, s_k_1, gamma_k = _sym_ortho(gammabar, beta_k, 
                                        "jax" if hasattr(backend, 'jit') else "default")
    
    gamma_k_tmp = gamma_k  # C++ line 145: temporary value before P2/P3
    
    tau_km2 = s.tau_km1
    tau_km1 = s.tau_k
    tau_k = c_k_1 * s.phi_k
    phi_k = s_k_1 * s.phi_k
    
    # ==========================================================================
    # 4. PREVIOUS RIGHT REFLECTION P_{k-2,k}
    # ==========================================================================
    # C++ lines 154-165 (only if k > 1)
    if s.k > 1:
        theta_km2 = s.theta_km1
        eta_km2 = s.eta_km1
        eta_km1 = s.eta_k
        
        # C++ line 157: _delta_k_tmp = (_s_k_2 * _theta_k) - (_c_k_2 * _delta);
        delta_tmp = (s.s_k_2 * s.theta_k) - (s.c_k_2 * delta)
        # C++ line 159: _theta_km1 = (_c_k_2 * _theta_k) + (_s_k_2 * _delta);
        theta_km1 = (s.c_k_2 * s.theta_k) + (s.s_k_2 * delta)
        # C++ line 161: _delta = _delta_k_tmp;
        delta = delta_tmp
        # C++ line 163: _eta_k = _s_k_2 * _gamma_k;
        eta_k = s.s_k_2 * gamma_k
        # C++ line 165: _gamma_k = (-_c_k_2) * _gamma_k;
        gamma_k = -s.c_k_2 * gamma_k
    else:
        theta_km2 = s.theta_km2
        theta_km1 = s.theta_km1
        eta_k = s.eta_k
        eta_km1 = s.eta_km1
        eta_km2 = s.eta_km2
    
    # ==========================================================================
    # 5. CURRENT RIGHT REFLECTION P_{k-1,k}
    # ==========================================================================
    # C++ lines 172-178 (only if k > 0)
    if s.k > 0:
        # C++ line 174: sym_ortho(_gamma_km1, _delta, _c_k_3, _s_k_3, _gamma_km1);
        c_k_3, s_k_3, gamma_km1 = _sym_ortho(gamma_km1_prev, delta,
                                              "jax" if hasattr(backend, 'jit') else "default")
        # C++ line 175: _theta_k = _s_k_3 * _gamma_k;
        theta_k = s_k_3 * gamma_k
        # C++ line 177: _gamma_k = (-_c_k_3) * _gamma_k;
        gamma_k = -c_k_3 * gamma_k
    else:
        c_k_3 = s.c_k_3
        s_k_3 = s.s_k_3
        gamma_km1 = gamma_km1_prev
        theta_k = 0.0  # Initialize for k=0
    
    # ==========================================================================
    # 6. UPDATE MU & XNORM
    # ==========================================================================
    # C++ lines 185-207
    mu_km4 = s.mu_km3
    mu_km3 = s.mu_km2
    
    # C++ line 187: if (k > 1) _mu_km2 = ...
    if s.k > 1:
        safe_gamma_km2 = backend.where(backend.abs(gamma_km2) < _MIN_NORM, _TINY, gamma_km2)
        mu_km2 = (tau_km2 - eta_km2 * mu_km4 - theta_km2 * mu_km3) / safe_gamma_km2
    else:
        mu_km2 = s.mu_km2
    
    # C++ line 188: if (k > 0) _mu_km1 = ...
    if s.k > 0:
        safe_gamma_km1 = backend.where(backend.abs(gamma_km1) < _MIN_NORM, _TINY, gamma_km1)
        mu_km1 = (tau_km1 - eta_km1 * mu_km3 - theta_km1 * mu_km2) / safe_gamma_km1
    else:
        mu_km1 = s.mu_km1
    
    # C++ lines 190-203: Compute mu_k with singularity checks
    xnorm_tmp = backend.sqrt(s.xl2norm_k**2 + mu_km2**2 + mu_km1**2)
    
    if backend.abs(gamma_k) > _MIN_NORM and xnorm_tmp < _MAX_X_NORM:
        mu_k = (tau_k - eta_k * mu_km2 - theta_k * mu_km1) / gamma_k
        xnorm_k_test = backend.sqrt(xnorm_tmp**2 + mu_k**2)
        if xnorm_k_test > _MAX_X_NORM:
            mu_k = 0.0
            flag = 6  # MAXXNORM
        else:
            flag = s.flag
    else:
        mu_k = 0.0
        flag = 5  # SINGULAR
    
    # C++ lines 205-206: Update norms
    xl2norm_k = backend.sqrt(s.xl2norm_k**2 + mu_km2**2)
    xnorm_k = backend.sqrt(xl2norm_k**2 + mu_km1**2 + mu_k**2)
    
    # ==========================================================================
    # 7. UPDATE VECTORS (W & X) - QLP SWITCHING
    # ==========================================================================
    # C++ lines 213-274
    
    # Check condition: C++ line 215
    cond_minres = (backend.real(s.a_cond) < _TRANS_A_COND and 
                   flag == s.flag and 
                   s.qlp_iter == 0)
    
    if cond_minres:
        # MINRES PATH (C++ lines 215-221)
        w_km2 = s.w_km1
        w_km1 = s.w_k
        safe_gamma_tmp = backend.where(gamma_k_tmp == 0, _TINY, gamma_k_tmp)
        w_k = (v - eps_k * w_km2 - deltaqlp * w_km1) / safe_gamma_tmp
        
        if xnorm_k < _MAX_X_NORM:
            x_k = s.x_k + tau_k * w_k
        else:
            x_k = s.x_k
            flag = 6  # MAXXNORM
        
        x_km1 = s.x_km1
        qlp_iter = 0
        
    else:
        # MINRES-QLP PATH (C++ lines 223-274)
        qlp_iter = s.qlp_iter + 1
        
        # C++ lines 226-235: Transition step reconstruction
        if qlp_iter == 1:
            if s.k > 0:
                if s.k > 2:
                    w_km2 = s.gamma_km3 * s.w_km2 + s.theta_km2 * s.w_km1 + s.eta_km1 * s.w_k
                else:
                    w_km2 = s.w_km2
                    
                if s.k > 1:
                    w_km1 = s.gamma_qlp_km1 * s.w_km1 + s.theta_qlp_k * s.w_k
                else:
                    w_km1 = s.w_km1
                    
                w_k = s.gamma_qlp_k * s.w_k
                x_km1 = s.x_k - w_km1 * s.mu_qlp_km1 - w_k * s.mu_qlp_k
            else:
                w_km2 = s.w_km2
                w_km1 = s.w_km1
                w_k = s.w_k
                x_km1 = s.x_km1
        else:
            w_km2 = s.w_km1
            x_km1 = s.x_km1
            w_km1 = s.w_km1  # Will be updated below
            w_k = s.w_k      # Will be updated below
        
        # C++ lines 237-268: Standard QLP recurrence
        if s.k == 0:
            w_km1 = v * s_k_3
            w_k = -v * c_k_3
        elif s.k == 1:
            w_km1 = s.w_k * c_k_3 + v * s_k_3
            w_k = s.w_k * s_k_3 - v * c_k_3
        else:
            # C++ lines 264-268
            w_km1_temp = w_k  # Save old w_k
            w_k = w_km2 * s.s_k_2 - v * s.c_k_2
            w_km2 = w_km2 * s.c_k_2 + v * s.s_k_2
            v_temp = w_km1_temp * c_k_3 + w_k * s_k_3
            w_k = w_km1_temp * s_k_3 - w_k * c_k_3
            w_km1 = v_temp
        
        # C++ lines 270-271: Update solution
        x_km1 = x_km1 + w_km2 * mu_km2
        x_k = x_km1 + w_km1 * mu_km1 + w_k * mu_k
    
    # ==========================================================================
    # 8. PREPARE NEXT P2 ROTATION
    # ==========================================================================
    # C++ lines 278-281
    gamma_k_tmp_for_p2 = gamma_km1  # Save before next sym_ortho
    c_k_2, s_k_2, gamma_km1_updated = _sym_ortho(gamma_km1, eps_kp1,
                                                   "jax" if hasattr(backend, 'jit') else "default")
    
    # ==========================================================================
    # 9. STORE QUANTITIES FOR NEXT ITERATION (QLP TRANSFER)
    # ==========================================================================
    # C++ lines 287-292
    gammaqlp_km1 = gamma_k_tmp  # From stage 3
    thetaqlp_k = theta_k
    gammaqlp_k = gamma_k        # After stage 5
    muqlp_km1 = mu_km1
    muqlp_k = mu_k
    
    # ==========================================================================
    # 10. ESTIMATE NORMS & CONVERGENCE
    # ==========================================================================
    # C++ lines 298-336
    abs_gamma = backend.abs(gamma_k)
    a_norm = backend.maximum(s.a_norm, backend.abs(gamma_km1))
    a_norm = backend.maximum(a_norm, abs_gamma)
    a_norm = backend.maximum(a_norm, pnorm_rho_k)
    
    if s.k == 0:
        gamma_min = abs_gamma
        gamma_min_km1 = abs_gamma
        gamma_min_km2 = abs_gamma
    else:
        gamma_min_km2 = s.gamma_min_km1
        gamma_min_km1 = s.gamma_min
        gamma_min = backend.minimum(gamma_min_km2, backend.abs(gamma_km1))
        gamma_min = backend.minimum(gamma_min, abs_gamma)
    
    safe_gamma_min = backend.where(gamma_min < _MIN_NORM, _MIN_NORM, gamma_min)
    a_cond = a_norm / safe_gamma_min
    
    # C++ lines 310-311: Update residual norm
    if flag != 5:  # Not SINGULAR
        rnorm = backend.abs(phi_k)
    else:
        rnorm = s.relres
    
    relres = rnorm / (a_norm * xnorm_k + s.beta_k + _TINY)
    
    # C++ lines 321-330: Convergence checks
    if flag == s.flag or flag == 5:  # PROCESSING or SINGULAR
        if s.k >= maxiter - 1:
            flag = 4  # MAXITER
        elif a_cond > _MAX_A_COND:
            flag = 7  # ACOND
        elif relres <= tol:
            flag = 1  # SOLUTION_RTOL
    
    # ==========================================================================
    # 11. PACK STATE FOR NEXT ITERATION
    # ==========================================================================
    return MinresQLPState(
        k=s.k + 1,
        
        x_k=x_k,
        x_km1=x_km1,
        
        w_k=w_k,
        w_km1=w_km1,
        w_km2=w_km2,
        
        v_k=v,
        z_k=z_k_precond,
        z_km1=z_km1_next,
        z_km2=z_km2_next,
        
        beta_k=beta_k,
        beta_km1=beta_km1,
        beta_last=beta_last,
        
        c_k_1=c_k_1,
        s_k_1=s_k_1,
        delta_k=delta_k,
        phi_k=phi_k,
        eps_k=eps_k,
        eps_kp1=eps_kp1,
        
        c_k_2=c_k_2,
        s_k_2=s_k_2,
        c_k_3=c_k_3,
        s_k_3=s_k_3,
        
        gamma_k=gamma_k,        # After P2/P3 - will be gamma_km1 next iter
        gamma_km1=gamma_km1_updated,  # After P2 preparation
        gamma_km2=gamma_km1_prev,     # Previous gamma_k
        gamma_km3=gamma_km2,
        
        theta_k=theta_k,
        theta_km1=theta_km1,
        theta_km2=theta_km2,
        
        eta_k=eta_k,
        eta_km1=eta_km1,
        eta_km2=eta_km2,
        
        tau_k=tau_k,
        tau_km1=tau_km1,
        tau_km2=tau_km2,
        
        mu_k=mu_k,
        mu_km1=mu_km1,
        mu_km2=mu_km2,
        mu_km3=mu_km3,
        mu_km4=mu_km4,
        
        qlp_iter=qlp_iter,
        gamma_qlp_k=gammaqlp_k,
        gamma_qlp_km1=gammaqlp_km1,
        theta_qlp_k=thetaqlp_k,
        mu_qlp_k=muqlp_k,
        mu_qlp_km1=muqlp_km1,
        
        xl2norm_k=xl2norm_k,
        xnorm_k=xnorm_k,
        a_norm=a_norm,
        a_cond=a_cond,
        gamma_min=gamma_min,
        gamma_min_km1=gamma_min_km1,
        gamma_min_km2=gamma_min_km2,
        
        relres=relres,
        flag=flag
    )

# -----------------------------------------------------------------------------
#! Core MINRES-QLP Logic Functions
# -----------------------------------------------------------------------------

def _minres_qlp_logic_numpy(
    matvec          : Callable,
    b               : np.ndarray,
    x0              : np.ndarray,
    tol             : float,
    maxiter         : int,
    precond_apply   : Optional[Callable] = None,
    sigma           : float = 0.0) -> tuple:
    """
    Pure NumPy MINRES-QLP implementation.
    
    Returns:
        tuple: (x, converged, iterations, residual)
    """
    xp = np
    
    # Apply shift to matvec: A' = A - sigma*I
    def _shifted_matvec(v):
        return matvec(v) - sigma * v
    
    # Initial Residual
    r0 = b - _shifted_matvec(x0)
    
    # Apply Preconditioner to r0
    if precond_apply:
        z0 = precond_apply(r0)
        beta0 = xp.sqrt(xp.real(xp.vdot(r0, z0)))
    else:
        z0 = r0
        beta0 = xp.linalg.norm(r0)
    
    # Initial State
    state = MinresQLPState(
        k=0,
        x_k=x0, x_km1=x0,
        w_k=xp.zeros_like(x0), w_km1=xp.zeros_like(x0), w_km2=xp.zeros_like(x0),
        v_k=xp.zeros_like(x0),
        z_k=z0, z_km1=xp.zeros_like(x0), z_km2=xp.zeros_like(x0),
        
        beta_k=beta0, beta_km1=0.0, beta_last=0.0,
        
        c_k_1=1.0, s_k_1=0.0, delta_k=0.0, phi_k=beta0,
        eps_k=0.0, eps_kp1=0.0,
        
        c_k_2=1.0, s_k_2=0.0, c_k_3=1.0, s_k_3=0.0,
        
        gamma_k=0.0, gamma_km1=0.0, gamma_km2=0.0, gamma_km3=0.0,
        theta_k=0.0, theta_km1=0.0, theta_km2=0.0,
        eta_k=0.0, eta_km1=0.0, eta_km2=0.0,
        tau_k=0.0, tau_km1=0.0, tau_km2=0.0,
        
        mu_k=0.0, mu_km1=0.0, mu_km2=0.0, mu_km3=0.0, mu_km4=0.0,
        
        qlp_iter=0,
        gamma_qlp_k=0.0, gamma_qlp_km1=0.0, theta_qlp_k=0.0,
        mu_qlp_k=0.0, mu_qlp_km1=0.0,
        
        xl2norm_k=0.0, xnorm_k=0.0,
        a_norm=0.0, a_cond=1.0,
        gamma_min=1.0e30, gamma_min_km1=1.0e30, gamma_min_km2=1.0e30,
        
        relres=1.0,
        flag=int(MinresQLPFlag.PROCESSING)
    )
    
    # Iteration loop
    while (state.flag == int(MinresQLPFlag.PROCESSING)) and (state.k < maxiter):
        state = _minres_qlp_step(state, _shifted_matvec, precond_apply, tol, maxiter, xp)
    
    converged = (state.flag > 0 and state.flag < 8)
    return state.x_k, converged, state.k, state.relres * beta0

if JAX_AVAILABLE:

    def _minres_qlp_logic_jax(
        matvec          : Callable,
        b               : Any,
        x0              : Any,
        tol             : float,
        maxiter         : int,
        precond_apply   : Optional[Callable] = None,
        sigma           : float = 0.0) -> tuple:
        """
        JAX-compatible MINRES-QLP implementation with while_loop.
        
        Returns:
            tuple: (x, converged, iterations, residual)
        """
        
        # Apply shift to matvec: A' = A - sigma*I
        def _shifted_matvec(v):
            return matvec(v) - sigma * v
        
        # Initial Residual
        r0              = b - _shifted_matvec(x0)
        # Handle preconditioner
        safe_precond    = precond_apply if precond_apply is not None else (lambda x: x)
        
        # Apply Preconditioner to r0
        if precond_apply:
            z0          = safe_precond(r0)
            beta0       = jnp.sqrt(jnp.real(jnp.vdot(r0, z0)))
        else:
            z0          = r0
            beta0       = jnp.linalg.norm(r0)
        
        # Initial State
        init_state = MinresQLPState(
                        k=0,
                        x_k=x0, x_km1=x0,
                        w_k=jnp.zeros_like(x0), w_km1=jnp.zeros_like(x0), w_km2=jnp.zeros_like(x0),
                        v_k=jnp.zeros_like(x0),
                        z_k=z0, z_km1=jnp.zeros_like(x0), z_km2=jnp.zeros_like(x0),
                        
                        beta_k=beta0, beta_km1=0.0, beta_last=0.0,
                        
                        c_k_1=1.0, s_k_1=0.0, delta_k=0.0, phi_k=beta0,
                        eps_k=0.0, eps_kp1=0.0,
                        
                        c_k_2=1.0, s_k_2=0.0, c_k_3=1.0, s_k_3=0.0,
                        
                        gamma_k=0.0, gamma_km1=0.0, gamma_km2=0.0, gamma_km3=0.0,
                        theta_k=0.0, theta_km1=0.0, theta_km2=0.0,
                        eta_k=0.0, eta_km1=0.0, eta_km2=0.0,
                        tau_k=0.0, tau_km1=0.0, tau_km2=0.0,
                        
                        mu_k=0.0, mu_km1=0.0, mu_km2=0.0, mu_km3=0.0, mu_km4=0.0,
                        
                        qlp_iter=0,
                        gamma_qlp_k=0.0, gamma_qlp_km1=0.0, theta_qlp_k=0.0,
                        mu_qlp_k=0.0, mu_qlp_km1=0.0,
                        
                        xl2norm_k=0.0, xnorm_k=0.0,
                        a_norm=0.0, a_cond=1.0,
                        gamma_min=1.0e30, gamma_min_km1=1.0e30, gamma_min_km2=1.0e30,
                        
                        relres=1.0,
                        flag=int(MinresQLPFlag.PROCESSING)
                    )
        
        def cond_fun(state: MinresQLPState) -> bool:
            is_processing   = state.flag == int(MinresQLPFlag.PROCESSING)
            is_within_limit = state.k < maxiter
            return jnp.logical_and(is_processing, is_within_limit)
        
        def body_fun(state: MinresQLPState) -> MinresQLPState:
            return _minres_qlp_step(state, _shifted_matvec, safe_precond, tol, maxiter, jnp)
        
        # Compile with while_loop
        final_state = lax.while_loop(cond_fun, body_fun, init_state)
        converged   = jnp.logical_and(final_state.flag > 0, final_state.flag < 8)
        return final_state.x_k, converged, final_state.k, final_state.relres * beta0

# Compile JAX version if available
_minres_qlp_logic_jax_compiled = None
if JAX_AVAILABLE:
    _minres_qlp_logic_jax_compiled = jax.jit(_minres_qlp_logic_jax, static_argnums=(3, 4, 6))

# -----------------------------------------------------------------------------
#! MinresQLP Solver Class
# -----------------------------------------------------------------------------

class MinresQLPSolver(Solver):
    r'''
    Minimum Residual method with QLP stabilization for symmetric systems.
    '''
    _solver_type = SolverType.MINRES_QLP
    _symmetric   = True

    @staticmethod
    def get_solver_func(
            backend_module      : Any,
            use_matvec          : bool = True,
            use_fisher          : bool = False,
            use_matrix          : bool = False,
            sigma               : Optional[float] = None) -> StaticSolverFunc:
        """
        Returns the backend-specific compiled/optimized MINRES-QLP function.

        Args:
            backend_module (Any):
                The numerical backend (`numpy` or `jax.numpy`).
            use_matvec (bool):
                Whether to use matvec interface (always True for MINRES-QLP).
            use_fisher (bool):
                Whether to construct Fisher information matrix.
            use_matrix (bool):
                Whether to use dense matrix interface.
            sigma (Optional[float]):
                Shift parameter for (A - sigma*I).

        Returns:
            StaticSolverFunc:
                The core MINRES-QLP function for the backend.
        """
        # JAX Backend
        if backend_module is jnp:
            if _minres_qlp_logic_jax_compiled is None:
                raise ImportError("JAX not installed but JAX backend requested.")
            func = _minres_qlp_logic_jax_compiled
            
        # NumPy Backend
        elif backend_module is np:
            func = _minres_qlp_logic_numpy
            
        else:
            raise ValueError(f"Unsupported backend: {backend_module}")

        # Wrap the core logic (e.g., Fisher construction) using the base class helper
        return Solver._solver_wrap_compiled(backend_module, func, use_matvec, use_fisher, use_matrix, sigma)
        
    @staticmethod
    def solve(
        matvec          : MatVecFunc,
        b               : Array,
        x0              : Array,
        *,
        tol             : float,
        maxiter         : int,
        precond_apply   : Optional[Callable[[Array], Array]] = None,
        backend_module  : Any = np,
        sigma           : Optional[float] = None,
        **kwargs        : Any) -> SolverResult:
        """
        Static MINRES-QLP execution: Determines the appropriate backend function and executes it.

        Args:
            matvec (MatVecFunc): 
                Function performing the matrix-vector product $ v \\mapsto Av $.
            b (Array): 
                Right-hand side vector $ b $.
            x0 (Array): 
                Initial guess $ x_0 $.
            tol (float): 
                Relative tolerance $ \\epsilon_{rel} $.
            maxiter (int): 
                Maximum number of iterations.
            precond_apply (Optional[Callable[[Array], Array]]): 
                Function performing the preconditioning step $ r \\mapsto M^{-1}r $.
            backend_module (Any): 
                Backend module (`numpy` or `jax.numpy`).
            sigma (Optional[float]):
                Shift parameter (default: 0.0).
            **kwargs (Any): 
                Additional arguments.

        Returns:
            SolverResult: 
                A named tuple containing the solution, convergence status, 
                iteration count, and residual norm.
        """
        try:
            # Get the compiled solver function
            solver_func = MinresQLPSolver.get_solver_func(backend_module, use_matvec=True, sigma=sigma if sigma is not None else 0.0)
            
            # Run the solver
            return Solver.run_solver_func(backend_module, solver_func, 
                matvec=matvec, b=b, x0=x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply)
            
        except Exception as e:
            raise RuntimeError(f"MINRES-QLP execution failed: {e}") from e
        
# -------------------------------------------------------------------------
#! EOF
# -------------------------------------------------------------------------