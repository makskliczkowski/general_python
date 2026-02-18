r'''
MINRES-QLP Solver for Symmetric (Possibly Singular) Linear Systems
===================================================================

Solves symmetric (possibly singular) linear systems:

$$
(A - \sigma I)x = b,
$$

or least-squares problems $ \min ||(A-\sigma I)x - b||_2 $. 
It finds the minimum length solution $ x $ (minimum $||x||_2$) among all least-squares solutions.

Based on the algorithm by Choi, Paige, and Saunders (2011).

Mathematical Sketch:
--------------------
MINRES applies the Lanczos process to $ A' = A - \sigma I $ to generate an orthonormal 
basis $ V_k $ for the Krylov subspace and a symmetric tridiagonal matrix $ T_k $.
The residual norm $ ||r_k|| $ is minimized over $ x_k \in x_0 + K_k $.

MINRES-QLP extends this by applying right Givens rotations to obtain an LQ factorization,
enabling stable computation of the minimum-length solution for singular/ill-conditioned systems.

References:
-----------
    - Choi, S.-C. T., Paige, C. C., & Saunders, M. A. (2011). MINRES-QLP: A
        Krylov subspace method for indefinite or singular symmetric systems.
        SIAM Journal on Scientific Computing, 33(4), 1810-1836.
        
-----------------------------------------------------------------------------
file:       general_python/algebra/solvers/minres_qlp.py
author:     Maksymilian Kliczkowski
desc:       Native MINRES-QLP solver - JAX & NumPy backends.
date:       2025-02-01
-----------------------------------------------------------------------------
'''

from    typing import Optional, Callable, Any
import  numpy as np
import  enum

# Base Solver classes and types
try:
    from ..solver           import Solver, SolverResult, SolverType, Array, MatVecFunc, StaticSolverFunc
    from ..preconditioners  import PreconitionerApplyFun
except ImportError:
    raise ImportError("general_python package is required to use MINRES-QLP solver.")

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    import jax.lax as lax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = np
    lax = None

# ##############################################################################
#! Constants & Flags (matching C++ implementation)
# ##############################################################################

_MACHEPS        = np.finfo(np.float64).eps
_TINY           = 1e-14                 # Threshold for near-zero values
_MIN_NORM       = 1e-14                 # Minimum norm threshold
_MAX_X_NORM     = 1.0e+7                # Maximum solution norm (MAXXNORM)
_MAX_A_COND     = 1.0e+15               # Maximum condition number (CONLIM)
_TRANS_A_COND   = 1.0e+7                # Threshold to switch to QLP (TRANSCOND)

class MinresQLPFlag(enum.IntEnum):
    '''Convergence flags for MINRES-QLP solver (matching C++ MINRES_QLP_FLAGS).'''
    PROCESSING      = -2    # Still processing
    BETA_ZERO       = -1    # beta_k = 0. Solution is eigenvector.
    X_ZERO          = 0     # Found x=0 (initial residual was zero)
    RESID_RTOL      = 1     # Converged: ||r|| / ||b|| <= tol
    RESID_AR        = 2     # Converged: ||Ar|| / (||A|| ||r||) <= tol
    RESID_EPS       = 3     # Converged: ||r|| <= eps
    RESID_EPS_AR    = 4     # Converged: Both eps criteria met
    EIGEN           = 5     # Found eigenvector
    X_NORM_LIMIT    = 6     # Exceeded ||x|| limit (MAXXNORM)
    A_COND_LIMIT    = 7     # Exceeded Acond limit
    MAX_ITER        = 8     # Reached max iterations
    SINGULAR        = 9     # System appears singular
    PRECOND_INDEF   = 10    # Preconditioner is indefinite

# ##############################################################################
#! Symmetric Orthogonalization (Givens Rotation)
# ##############################################################################

def _sym_ortho_np(a: float, b: float):
    """
    Compute Givens rotation parameters (c, s, r) such that:
        [c  s] [a]   [r]
        [-s c] [b] = [0]
    
    Returns (c, s, r) where r = sqrt(a^2 + b^2).
    NumPy version.
    """
    abs_a = abs(a)
    abs_b = abs(b)
    
    if abs_b < _TINY:
        c = np.sign(a) if a != 0 else 1.0
        return c, 0.0, abs_a
    elif abs_a < _TINY:
        return 0.0, np.sign(b), abs_b
    else:
        r = np.sqrt(a*a + b*b)
        return a/r, b/r, r

# ##############################################################################
#! Pure NumPy MINRES-QLP Implementation (Optimized, No Complex State)
# ##############################################################################

def _minres_qlp_logic_numpy(
    matvec          : Callable,
    b               : np.ndarray,
    x0              : np.ndarray,
    tol             : float,
    maxiter         : int,
    precond_apply   : Optional[Callable] = None,
    sigma           : float = 0.0,
    **kwargs
) -> SolverResult:
    """
    Pure NumPy MINRES-QLP implementation.
    Check:
    - https://web.stanford.edu/group/SOL/software/minresqlp/
    - https://epubs.siam.org/doi/10.1137/100804959
    
    Parameters:
        matvec:
            Matrix-vector product function (already includes -sigma shift if needed)
        b: 
            Right-hand side vector
        x0: 
            Initial guess
        tol:
            Relative tolerance
        maxiter:
            Maximum iterations
        precond_apply: 
            Optional preconditioner M^{-1}
        sigma: 
            Shift parameter (applied as A - sigma*I)
        
    Returns:
        SolverResult with solution, convergence status, iterations, residual
    """
    n = len(b)
    if maxiter == 0:
        maxiter = n
    
    # Apply shift if needed
    if sigma != 0.0:
        _matvec = lambda v: matvec(v) - sigma * v
    else:
        _matvec = matvec
    
    has_precond = precond_apply is not None
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    # Lanczos vectors
    z_km2   = np.zeros(n, dtype=b.dtype)
    
    # Initialize with residual: r0 = b - A*x0
    r0      = b - _matvec(x0)
    z_km1   = r0.copy()
    z_k     = precond_apply(z_km1) if has_precond else z_km1.copy()
    
    # Initial beta
    beta1   = np.real(np.vdot(r0, z_k))
    
    # Check preconditioner definiteness
    flag    = int(MinresQLPFlag.PROCESSING)
    if has_precond:
        if beta1 < 0:
            return SolverResult(x0, False, 0, np.linalg.norm(r0))
        elif abs(beta1) < _MIN_NORM:
            return SolverResult(x0, True, 0, 0.0)
        beta1 = np.sqrt(beta1)
    else:
        beta1 = np.sqrt(beta1)
    
    beta_k          = beta1
    beta_km1        = 0.0
    beta_last       = 0.0
    phi_k           = beta_k
    
    # Solution vectors
    x_k             = x0.copy()
    x_km1           = x_k.copy()
    
    # Search directions
    w_k             = np.zeros(n, dtype=b.dtype)
    w_km1           = np.zeros(n, dtype=b.dtype)
    w_km2           = np.zeros(n, dtype=b.dtype)
    
    # Rotation coefficients (Initial: c=-1, s=0 per C++ line 68-69)
    c_k_1, s_k_1    = -1.0, 0.0
    c_k_2, s_k_2    = -1.0, 0.0
    c_k_3, s_k_3    = -1.0, 0.0
    delta_k         = 0.0
    eps_kp1         = 0.0
    
    # Tridiagonal elements
    gamma_k         = gamma_km1 = gamma_km2 = gamma_km3 = 0.0
    theta_k         = theta_km1 = theta_km2 = 0.0
    eta_k           = eta_km1 = eta_km2 = 0.0
    tau_k           = tau_km1 = tau_km2 = 0.0
    
    # Solution coefficients
    mu_k            = mu_km1 = mu_km2 = mu_km3 = mu_km4 = 0.0
    
    # Norms
    xl2norm_k       = 0.0
    xnorm_k         = 0.0
    a_norm          = 0.0
    a_cond          = 1.0
    gamma_min       = 1e30
    
    # QLP history
    qlp_iter        = 0
    gamma_qlp_k     = gamma_qlp_km1 = 0.0
    theta_qlp_k     = 0.0
    mu_qlp_k        = mu_qlp_km1 = 0.0
    
    relres          = 1.0
    rnorm           = beta_k
    k               = 0
    
    # =========================================================================
    # MAIN ITERATION LOOP (C++ lines 126-362)
    # =========================================================================
    
    while flag == int(MinresQLPFlag.PROCESSING) and k < maxiter:
        
        # ---------------------------------------------------------------------
        # 1. PRECONDITIONED LANCZOS
        # ---------------------------------------------------------------------
        beta_last   = beta_km1
        beta_km1    = beta_k
        
        # Normalize
        if abs(beta_km1) < _TINY:
            break
        # Matrix-vector product
        v   = z_k / beta_km1
        z_k = _matvec(v)
        
        # Lanczos orthogonalization
        if k > 0 and abs(beta_last) > _TINY:
            z_k = z_k - z_km2 * (beta_km1 / beta_last)
        
        alpha   = np.real(np.vdot(z_k, v))
        z_k     = z_k - z_km1 * (alpha / beta_km1)
        
        # Shift Lanczos vectors
        z_km2   = z_km1.copy()
        z_km1   = z_k.copy()
        
        # Apply preconditioner and compute new beta
        if has_precond:
            z_k = precond_apply(z_k)
            beta_k_sq = np.real(np.vdot(z_km1, z_k))
            if beta_k_sq > 0:
                beta_k = np.sqrt(beta_k_sq)
            else:
                flag = int(MinresQLPFlag.PRECOND_INDEF)
                break
        else:
            beta_k = np.linalg.norm(z_k)
        
        # Check for early termination
        if k == 0 and abs(beta_k) < _MIN_NORM:
            if abs(alpha) < _MIN_NORM:
                flag = int(MinresQLPFlag.X_ZERO)
                break
            else:
                flag = int(MinresQLPFlag.BETA_ZERO)
                x_k = b / alpha
                break
        
        pnorm_rho_k = np.sqrt(beta_last**2 + alpha**2 + beta_k**2)
        
        # ---------------------------------------------------------------------
        # 2. PREVIOUS LEFT REFLECTION Q_{k-1} - for QLP
        # ---------------------------------------------------------------------
        dbar        = delta_k
        delta       = c_k_1 * dbar + s_k_1 * alpha
        eps_k       = eps_kp1
        eps_kp1     = s_k_1 * beta_k
        gammabar    = s_k_1 * dbar - c_k_1 * alpha
        delta_k     = -c_k_1 * beta_k
        deltaqlp    = delta
        
        # ---------------------------------------------------------------------
        # 3. CURRENT LEFT REFLECTION Q_k - for QLP 
        # ---------------------------------------------------------------------
        gamma_km3   = gamma_km2
        gamma_km2   = gamma_km1
        gamma_km1   = gamma_k
        
        c_k_1, s_k_1, gamma_k = _sym_ortho_np(gammabar, beta_k)
        gamma_k_tmp = gamma_k
        
        tau_km2     = tau_km1
        tau_km1     = tau_k
        tau_k       = c_k_1 * phi_k
        phi_k       = s_k_1 * phi_k
        
        # ---------------------------------------------------------------------
        # 4. PREVIOUS RIGHT REFLECTION P_{k-2,k}
        # ---------------------------------------------------------------------
        if k > 1:
            theta_km2   = theta_km1
            eta_km2     = eta_km1
            eta_km1     = eta_k
            
            delta_tmp   = s_k_2 * theta_k - c_k_2 * delta
            theta_km1   = c_k_2 * theta_k + s_k_2 * delta
            delta       = delta_tmp
            eta_k       = s_k_2 * gamma_k
            gamma_k     = -c_k_2 * gamma_k
        
        # ---------------------------------------------------------------------
        # 5. CURRENT RIGHT REFLECTION P_{k-1,k}
        # ---------------------------------------------------------------------
        if k > 0:
            c_k_3, s_k_3, gamma_km1 = _sym_ortho_np(gamma_km1, delta)
            theta_k                 = s_k_3 * gamma_k
            gamma_k                 = -c_k_3 * gamma_k
        
        # ---------------------------------------------------------------------
        # 6. UPDATE MU & XNORM (C++ lines 237-263)
        # ---------------------------------------------------------------------
        mu_km4 = mu_km3
        mu_km3 = mu_km2
        
        if k > 1 and abs(gamma_km2) > _MIN_NORM:
            mu_km2 = (tau_km2 - eta_km2 * mu_km4 - theta_km2 * mu_km3) / gamma_km2
        if k > 0 and abs(gamma_km1) > _MIN_NORM:
            mu_km1 = (tau_km1 - eta_km1 * mu_km3 - theta_km1 * mu_km2) / gamma_km1
        
        xnorm_tmp = np.sqrt(xl2norm_k**2 + mu_km2**2 + mu_km1**2)
        
        if abs(gamma_k) > _MIN_NORM and xnorm_tmp < _MAX_X_NORM:
            mu_k = (tau_k - eta_k * mu_km2 - theta_k * mu_km1) / gamma_k
            if np.sqrt(xnorm_tmp**2 + mu_k**2) > _MAX_X_NORM:
                mu_k = 0.0
                flag = int(MinresQLPFlag.X_NORM_LIMIT)
        else:
            mu_k = 0.0
            if flag == int(MinresQLPFlag.PROCESSING):
                flag = int(MinresQLPFlag.SINGULAR)
        
        xl2norm_k   = np.sqrt(xl2norm_k**2 + mu_km2**2)
        xnorm_k     = np.sqrt(xl2norm_k**2 + mu_km1**2 + mu_k**2)
        
        # ---------------------------------------------------------------------
        # 7. UPDATE W & X (MINRES vs QLP path)
        # ---------------------------------------------------------------------
        if np.real(a_cond) < _TRANS_A_COND and flag == int(MinresQLPFlag.PROCESSING) and qlp_iter == 0:
            # MINRES updates
            w_km2 = w_km1.copy()
            w_km1 = w_k.copy()
            if abs(gamma_k_tmp) > _TINY:
                w_k     = (v - eps_k * w_km2 - deltaqlp * w_km1) / gamma_k_tmp
            
            if xnorm_k < _MAX_X_NORM:
                x_k     = x_k + tau_k * w_k
            else:
                flag    = int(MinresQLPFlag.X_NORM_LIMIT)
        else:
            # MINRES-QLP updates
            qlp_iter += 1
            
            if qlp_iter == 1:
                if k > 0:
                    if k > 2:
                        w_km2 = gamma_km3 * w_km2 + theta_km2 * w_km1 + eta_km1 * w_k
                    if k > 1:
                        w_km1 = gamma_qlp_km1 * w_km1 + theta_qlp_k * w_k
                    w_k = gamma_qlp_k * w_k
                    x_km1 = x_k - w_km1 * mu_qlp_km1 - w_k * mu_qlp_k
            
            w_km2_old = w_km2.copy()
            
            if k == 0:
                w_km1       = v * s_k_3
                w_k         = -v * c_k_3
            elif k == 1:
                w_km1       = w_k * c_k_3 + v * s_k_3
                w_k         = w_k * s_k_3 - v * c_k_3
            else:
                w_km1_old   = w_k.copy()
                w_k         = w_km2_old * s_k_2 - v * c_k_2
                w_km2       = w_km2_old * c_k_2 + v * s_k_2
                v_tmp       = w_km1_old * c_k_3 + w_k * s_k_3
                w_k         = w_km1_old * s_k_3 - w_k * c_k_3
                w_km1       = v_tmp
            
            x_km1   = x_km1 + w_km2 * mu_km2
            x_k     = x_km1 + w_km1 * mu_km1 + w_k * mu_k
        
        # ---------------------------------------------------------------------
        # 8. PREPARE NEXT P2 ROTATION
        # ---------------------------------------------------------------------
        c_k_2, s_k_2, gamma_km1 = _sym_ortho_np(gamma_km1, eps_kp1)
        
        # ---------------------------------------------------------------------
        # 9. STORE QLP QUANTITIES
        # ---------------------------------------------------------------------
        gamma_qlp_km1   = gamma_k_tmp
        theta_qlp_k     = theta_k
        gamma_qlp_k     = gamma_k
        mu_qlp_km1      = mu_km1
        mu_qlp_k        = mu_k
        
        # ---------------------------------------------------------------------
        # 10. ESTIMATE NORMS & CHECK CONVERGENCE
        # ---------------------------------------------------------------------
        abs_gamma   = abs(gamma_k)
        a_norm      = max(a_norm, abs(gamma_km1), abs_gamma, pnorm_rho_k)
        
        if k == 0:
            gamma_min = abs_gamma
        else:
            gamma_min = min(gamma_min, abs(gamma_km1), abs_gamma)
        
        if gamma_min > _MIN_NORM:
            a_cond = a_norm / gamma_min
        
        if flag == int(MinresQLPFlag.PROCESSING) or flag == int(MinresQLPFlag.SINGULAR):
            rnorm = abs(phi_k)
        
        relres = rnorm / (a_norm * xnorm_k + beta1 + _TINY)
        
        # Convergence checks
        if flag == int(MinresQLPFlag.PROCESSING):
            if k >= maxiter - 1:
                flag = int(MinresQLPFlag.MAX_ITER)
            elif a_cond > _MAX_A_COND:
                flag = int(MinresQLPFlag.A_COND_LIMIT)
            elif relres <= tol:
                flag = int(MinresQLPFlag.RESID_RTOL)
        
        k += 1
    
    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    
    converged       = (flag > 0 and flag < 8)
    final_residual  = rnorm
    
    return SolverResult(
        x               =   x_k, 
        converged       =   converged, 
        iterations      =   k, 
        residual_norm   =  final_residual
    )

# ##############################################################################
#! JAX MINRES-QLP Implementation (JIT-compatible with lax.while_loop)
# ##############################################################################

if JAX_AVAILABLE:
    
    def _sym_ortho_jax(a, b):
        """
        JAX-compatible symmetric orthogonalization (Givens rotation).
        
        Computes c, s, r such that:
            [c  s] [a]   [r]
            [-s c] [b] = [0]
        
        For MINRES-QLP, inputs a, b are always real scalars (from Lanczos).
        Returns (c, s, r) where r = sqrt(a^2 + b^2) is always real.
        
        Uses lax.cond for JIT-compatible branching with consistent dtypes.
        """
        abs_a = jnp.abs(a)
        abs_b = jnp.abs(b)
        
        # Get the real dtype for consistent scalar creation
        # For complex inputs, we want real outputs (c, s are rotation angles, r is norm)
        input_dtype = jnp.result_type(a, b)
        # For rotation matrix elements, use real dtype
        real_dtype  = jnp.finfo(input_dtype).dtype if jnp.issubdtype(input_dtype, jnp.complexfloating) else input_dtype
        
        zero        = jnp.zeros((), dtype=real_dtype)
        one         = jnp.ones((), dtype=real_dtype)
        
        # Cast abs values to real dtype for consistent output
        abs_a_real  = abs_a.astype(real_dtype)
        abs_b_real  = abs_b.astype(real_dtype)
        a_real      = jnp.real(a).astype(real_dtype) if jnp.issubdtype(input_dtype, jnp.complexfloating) else a.astype(real_dtype)
        b_real      = jnp.real(b).astype(real_dtype) if jnp.issubdtype(input_dtype, jnp.complexfloating) else b.astype(real_dtype)
        
        def case_b_zero():
            # b ≈ 0: rotation is identity or sign flip
            c = jnp.where(a_real != 0, jnp.sign(a_real), one)
            return c, zero, abs_a_real
        
        def case_a_zero():
            # a ≈ 0: rotation is 90 degrees
            return zero, jnp.sign(b_real), abs_b_real
        
        def case_general():
            # General case: compute Givens rotation
            r = jnp.sqrt(a_real*a_real + b_real*b_real)
            return a_real/r, b_real/r, r
        
        return lax.cond(
            abs_b < _TINY,
            case_b_zero,
            lambda: lax.cond(abs_a < _TINY, case_a_zero, case_general)
        )

    def _minres_qlp_logic_jax(
        matvec          : Callable,
        b               : Any,
        x0              : Any,
        tol             : float,
        maxiter         : int,
        precond_apply   : Optional[Callable] = None,
        **kwargs) -> SolverResult:
        """
        JAX-compatible MINRES-QLP with lax.while_loop for JIT compilation.
        
        Uses a flat tuple state to avoid NamedTuple overhead and ensure
        all branches use lax.cond for proper JIT tracing.
        
        MINRES-QLP works on symmetric (real) or Hermitian (complex) systems.
        For complex inputs:
        - Vectors (x, w, z) remain complex
        - Scalars (rotation angles, norms, etc.) are always real
        """
        n = b.shape[0]
        if maxiter == 0:
            maxiter = n
        
        # === DTYPE SETUP ===
        # Vector dtype matches input (can be complex)
        # Scalar dtype is always real (for rotation angles, norms, etc.)
        vec_dtype       = jnp.result_type(b)
        scalar_dtype    = jnp.finfo(vec_dtype).dtype if jnp.issubdtype(vec_dtype, jnp.complexfloating) else vec_dtype
        
        # Real scalar constants for rotation angles, norms, etc.
        _zero           = jnp.zeros((), dtype=scalar_dtype)
        _one            = jnp.ones((), dtype=scalar_dtype)
        _neg_one        = -_one
        _large          = jnp.array(1e30, dtype=scalar_dtype)
        
        # === PRECONDITIONER SETUP ===
        precond_fn      = precond_apply if precond_apply is not None else (lambda x: x)
        has_precond     = precond_apply is not None
        
        # === INITIAL RESIDUAL AND BETA ===
        # z_km1 = r0, z_k = M^{-1}r0
        # Calculate initial residual: r0 = b - A*x0
        Ax0             = matvec(x0)
        r0              = b - Ax0
        
        z_km1_init      = r0
        z_k_init        = precond_fn(r0)
        # beta0 = sqrt(<r0, M^{-1}r0>) is always real
        beta0_sq        = jnp.real(jnp.vdot(r0, z_k_init))
        beta0           = jnp.sqrt(jnp.maximum(beta0_sq, _zero))
        
        # Zero vector for initialization (matches input dtype)
        zeros           = jnp.zeros_like(x0)
        
        # === STATE LAYOUT ===
        # Vectors (indices 1-8): match input dtype (can be complex)
        # Scalars (indices 9-52): always real dtype
        # State indices (for clarity):
        # 0: k (iteration counter, int)
        # 1-2: x_k, x_km1 (solution vectors)
        # 3-5: w_k, w_km1, w_km2 (auxiliary vectors for solution update)
        # 6-8: z_k, z_km1, z_km2 (Lanczos vectors)
        # 9-13: beta_k, beta_km1, beta_last, phi_k, beta1 (Lanczos betas and residual)
        # 14-17: c_k_1, s_k_1, delta_k, eps_kp1 (left rotation and tridiagonal elements)
        # 18-21: c_k_2, s_k_2, c_k_3, s_k_3 (right rotations for QLP)
        # 22-25: gamma_k, gamma_km1, gamma_km2, gamma_km3 (diagonal elements after rotations)
        # 26-28: theta_k, theta_km1, theta_km2 (off-diagonal elements)
        # 29-31: eta_k, eta_km1, eta_km2 (fill-in elements)
        # 32-34: tau_k, tau_km1, tau_km2 (right-hand side after rotations)
        # 35-39: mu_k, mu_km1, mu_km2, mu_km3, mu_km4 (solution coefficients)
        # 40: qlp_iter (iteration at which QLP started, int)
        # 41-43: gamma_qlp_k, gamma_qlp_km1, theta_qlp_k (QLP-specific values)
        # 44-45: mu_qlp_k, mu_qlp_km1 (QLP solution coefficients)
        # 46-48: xl2norm_k, xnorm_k, a_norm (norms for convergence)
        # 49-50: a_cond, gamma_min (condition number estimates)
        # 51-52: relres, flag
        
        init_state = (
            0,                                              # 0: k
            x0, x0,                                         # 1-2: x_k, x_km1
            zeros, zeros, zeros,                            # 3-5: w_k, w_km1, w_km2
            z_k_init, z_km1_init, zeros,                    # 6-8: z_k, z_km1, z_km2
            beta0, _zero, _zero, beta0, beta0,              # 9-13: beta_k, beta_km1, beta_last, phi_k, beta1
            _neg_one, _zero, _zero, _zero,                  # 14-17: c_k_1, s_k_1, delta_k, eps_kp1
            _neg_one, _zero, _neg_one, _zero,               # 18-21: c_k_2, s_k_2, c_k_3, s_k_3
            _zero, _zero, _zero, _zero,                     # 22-25: gamma_k, gamma_km1, gamma_km2, gamma_km3
            _zero, _zero, _zero,                            # 26-28: theta_k, theta_km1, theta_km2
            _zero, _zero, _zero,                            # 29-31: eta_k, eta_km1, eta_km2
            _zero, _zero, _zero,                            # 32-34: tau_k, tau_km1, tau_km2
            _zero, _zero, _zero, _zero, _zero,              # 35-39: mu_k, mu_km1, mu_km2, mu_km3, mu_km4
            0,                                              # 40: qlp_iter
            _zero, _zero, _zero,                            # 41-43: gamma_qlp_k, gamma_qlp_km1, theta_qlp_k
            _zero, _zero,                                   # 44-45: mu_qlp_k, mu_qlp_km1
            _zero, _zero, _zero,                            # 46-48: xl2norm_k, xnorm_k, a_norm
            _one, _large,                                   # 49-50: a_cond, gamma_min
            _one, int(MinresQLPFlag.PROCESSING)             # 51-52: relres, flag
        )
        
        def cond_fun(state):
            flag    = state[52]
            k       = state[0]
            return jnp.logical_and(
                flag == int(MinresQLPFlag.PROCESSING),
                k < maxiter
            )
        
        def body_fun(state):
            # Get scalar dtype for consistent scalar creation inside body
            # All rotation angles, norms, etc. are real even for complex inputs
            _zero_s = jnp.zeros((), dtype=scalar_dtype)
            
            # Unpack state
            (k, x_k, x_km1, w_k, w_km1, w_km2, z_k, z_km1, z_km2,
             beta_k, beta_km1, beta_last, phi_k, beta1,
             c_k_1, s_k_1, delta_k, eps_kp1,
             c_k_2, s_k_2, c_k_3, s_k_3,
             gamma_k, gamma_km1, gamma_km2, gamma_km3,
             theta_k, theta_km1, theta_km2,
             eta_k, eta_km1, eta_km2,
             tau_k, tau_km1, tau_km2,
             mu_k, mu_km1, mu_km2, mu_km3, mu_km4,
             qlp_iter,
             gamma_qlp_k, gamma_qlp_km1, theta_qlp_k,
             mu_qlp_k, mu_qlp_km1,
             xl2norm_k, xnorm_k, a_norm,
             a_cond, gamma_min,
             relres, flag) = state
            
            # =================================================================
            # 1. PRECONDITIONED LANCZOS STEP
            # Extends the Krylov subspace K_k(A, b) to K_{k+1}(A, b)
            # Computes: v = z_k / beta_k, then z_{k+1} via three-term recurrence
            # =================================================================
            beta_last_new = beta_km1
            beta_km1_new = beta_k
            
            _one_s = jnp.ones((), dtype=scalar_dtype)
            inv_beta = jnp.where(jnp.abs(beta_km1_new) < _TINY, _one_s, _one_s / beta_km1_new)
            v = z_k * inv_beta
            
            z_k_new = matvec(v)
            
            # Orthogonalize against z_{k-2}
            safe_beta_last = jnp.where(jnp.abs(beta_last_new) < _TINY, _one_s, beta_last_new)
            z_k_new = lax.cond(
                k > 0,
                lambda: z_k_new - z_km2 * (beta_km1_new / safe_beta_last),
                lambda: z_k_new
            )
            
            alpha = jnp.real(jnp.vdot(z_k_new, v))
            z_k_new = z_k_new - z_km1 * (alpha * inv_beta)
            
            z_km2_new = z_km1
            z_km1_new = z_k_new
            
            # Apply preconditioner
            z_k_precond = precond_fn(z_k_new)
            
            beta_k_sq = lax.cond(
                has_precond,
                lambda: jnp.real(jnp.vdot(z_km1_new, z_k_precond)),
                lambda: jnp.real(jnp.vdot(z_k_new, z_k_new))
            )
            beta_k_new = jnp.sqrt(jnp.maximum(beta_k_sq, _zero_s))
            
            pnorm_rho_k = jnp.sqrt(beta_last_new**2 + alpha**2 + beta_k_new**2)
            
            # =================================================================
            # 2. PREVIOUS LEFT REFLECTION Q_{k-1}
            # Applies the previous Givens rotation to the new column of T_k
            # Updates delta, epsilon, and gammabar for the QR factorization
            # =================================================================
            dbar = delta_k
            delta_new = c_k_1 * dbar + s_k_1 * alpha
            eps_k = eps_kp1
            eps_kp1_new = s_k_1 * beta_k_new
            gammabar = s_k_1 * dbar - c_k_1 * alpha
            delta_k_new = -c_k_1 * beta_k_new
            deltaqlp = delta_new
            
            # =================================================================
            # 3. CURRENT LEFT REFLECTION Q_k
            # Computes new Givens rotation to eliminate beta_{k+1}
            # gamma_k is the new diagonal element after rotation
            # Updates tau (transformed RHS) and phi (residual norm component)
            # =================================================================
            gamma_km3_new = gamma_km2
            gamma_km2_new = gamma_km1
            gamma_km1_prev = gamma_k
            
            c_k_1_new, s_k_1_new, gamma_k_new = _sym_ortho_jax(gammabar, beta_k_new)
            gamma_k_tmp = gamma_k_new
            
            tau_km2_new = tau_km1
            tau_km1_new = tau_k
            tau_k_new = c_k_1_new * phi_k
            phi_k_new = s_k_1_new * phi_k
            
            # =================================================================
            # 4. PREVIOUS RIGHT REFLECTION P_{k-2,k}
            # QLP phase: applies right reflections for better conditioning
            # Transforms the upper triangular factor for stability
            # Updates theta and eta elements
            # =================================================================
            def apply_p2():
                _theta_km2 = theta_km1
                _eta_km2 = eta_km1
                _eta_km1 = eta_k
                _delta_tmp = s_k_2 * theta_k - c_k_2 * delta_new
                _theta_km1 = c_k_2 * theta_k + s_k_2 * delta_new
                _eta_k = s_k_2 * gamma_k_new
                _gamma_k = -c_k_2 * gamma_k_new
                return _theta_km2, _theta_km1, _eta_km2, _eta_km1, _eta_k, _delta_tmp, _gamma_k
            
            def skip_p2():
                return theta_km2, theta_km1, eta_km2, eta_km1, eta_k, delta_new, gamma_k_new
            
            (theta_km2_new, theta_km1_new, eta_km2_new, eta_km1_new, 
             eta_k_new, delta_after_p2, gamma_k_after_p2) = lax.cond(k > 1, apply_p2, skip_p2)
            
            # =================================================================
            # 5. CURRENT RIGHT REFLECTION P_{k-1,k}
            # Completes the QLP factorization for current iteration
            # Produces final gamma_k used for solution update
            # =================================================================
            def apply_p3():
                _c, _s, _gkm1 = _sym_ortho_jax(gamma_km1_prev, delta_after_p2)
                _theta = _s * gamma_k_after_p2
                _gk = -_c * gamma_k_after_p2
                return _c, _s, _gkm1, _theta, _gk
            
            def skip_p3():
                return c_k_3, s_k_3, gamma_km1_prev, _zero_s, gamma_k_after_p2
            
            c_k_3_new, s_k_3_new, gamma_km1_new, theta_k_new, gamma_k_final = lax.cond(
                k > 0, apply_p3, skip_p3
            )
            
            # =================================================================
            # 6. UPDATE MU & XNORM
            # Computes mu coefficients via back-substitution
            # mu_k are the solution coefficients in the Lanczos basis
            # Monitors solution norm for stability/convergence
            # =================================================================
            mu_km4_new = mu_km3
            mu_km3_new = mu_km2
            
            safe_gamma_km2 = jnp.where(jnp.abs(gamma_km2_new) < _MIN_NORM, _TINY, gamma_km2_new)
            mu_km2_new = lax.cond(
                k > 1,
                lambda: (tau_km2_new - eta_km2_new * mu_km4_new - theta_km2_new * mu_km3_new) / safe_gamma_km2,
                lambda: mu_km2
            )
            
            safe_gamma_km1 = jnp.where(jnp.abs(gamma_km1_new) < _MIN_NORM, _TINY, gamma_km1_new)
            mu_km1_new = lax.cond(
                k > 0,
                lambda: (tau_km1_new - eta_km1_new * mu_km3_new - theta_km1_new * mu_km2_new) / safe_gamma_km1,
                lambda: mu_km1
            )
            
            xnorm_tmp = jnp.sqrt(xl2norm_k**2 + mu_km2_new**2 + mu_km1_new**2)
            
            safe_gamma_k = jnp.where(jnp.abs(gamma_k_final) < _MIN_NORM, _TINY, gamma_k_final)
            mu_k_candidate = (tau_k_new - eta_k_new * mu_km2_new - theta_k_new * mu_km1_new) / safe_gamma_k
            
            xnorm_test = jnp.sqrt(xnorm_tmp**2 + mu_k_candidate**2)
            
            # Determine mu_k and flag based on conditions
            mu_k_new = jnp.where(
                jnp.logical_and(jnp.abs(gamma_k_final) > _MIN_NORM, xnorm_tmp < _MAX_X_NORM),
                jnp.where(xnorm_test > _MAX_X_NORM, _zero_s, mu_k_candidate),
                _zero_s
            )
            
            flag_new = jnp.where(
                jnp.abs(gamma_k_final) <= _MIN_NORM,
                int(MinresQLPFlag.SINGULAR),
                jnp.where(
                    jnp.logical_and(xnorm_tmp < _MAX_X_NORM, xnorm_test > _MAX_X_NORM),
                    int(MinresQLPFlag.X_NORM_LIMIT),
                    flag
                )
            )
            
            xl2norm_k_new = jnp.sqrt(xl2norm_k**2 + mu_km2_new**2)
            xnorm_k_new = jnp.sqrt(xl2norm_k_new**2 + mu_km1_new**2 + mu_k_new**2)
            
            # =================================================================
            # 7. UPDATE W & X (MINRES vs QLP PHASE)
            # MINRES phase: direct update using w vectors (faster, less stable)
            # QLP phase: update using mu coefficients (slower, more stable)
            # Switches to QLP when condition number estimate exceeds threshold
            # =================================================================
            use_minres = jnp.logical_and(
                jnp.logical_and(a_cond < _TRANS_A_COND, flag_new == int(MinresQLPFlag.PROCESSING)),
                qlp_iter == 0
            )
            
            def minres_update():
                _w_km2 = w_km1
                _w_km1 = w_k
                safe_gamma_tmp = jnp.where(jnp.abs(gamma_k_tmp) < _TINY, _TINY, gamma_k_tmp)
                _w_k = (v - eps_k * _w_km2 - deltaqlp * _w_km1) / safe_gamma_tmp
                _x_k = jnp.where(xnorm_k_new < _MAX_X_NORM, x_k + tau_k_new * _w_k, x_k)
                _x_km1 = x_km1
                _qlp = 0
                return _w_k, _w_km1, _w_km2, _x_k, _x_km1, _qlp
            
            def qlp_update():
                _qlp = qlp_iter + 1
                
                def qlp_k0():
                    _w_km1 = v * s_k_3_new
                    _w_k = -v * c_k_3_new
                    return w_km2, _w_km1, _w_k, x_km1
                
                def qlp_k1():
                    _w_km1 = w_k * c_k_3_new + v * s_k_3_new
                    _w_k = w_k * s_k_3_new - v * c_k_3_new
                    return w_km2, _w_km1, _w_k, x_km1
                
                def qlp_kge2():
                    _w_km1_old = w_k
                    _w_km2_tmp = w_km2
                    _w_k_tmp = _w_km2_tmp * s_k_2 - v * c_k_2
                    _w_km2_new = _w_km2_tmp * c_k_2 + v * s_k_2
                    _v_tmp = _w_km1_old * c_k_3_new + _w_k_tmp * s_k_3_new
                    _w_k_new = _w_km1_old * s_k_3_new - _w_k_tmp * c_k_3_new
                    _w_km1_new = _v_tmp
                    return _w_km2_new, _w_km1_new, _w_k_new, x_km1
                
                _w_km2, _w_km1, _w_k, _x_km1_base = lax.cond(
                    k == 0, qlp_k0,
                    lambda: lax.cond(k == 1, qlp_k1, qlp_kge2)
                )
                
                _x_km1 = _x_km1_base + _w_km2 * mu_km2_new
                _x_k = _x_km1 + _w_km1 * mu_km1_new + _w_k * mu_k_new
                
                return _w_k, _w_km1, _w_km2, _x_k, _x_km1, _qlp
            
            w_k_new, w_km1_new, w_km2_new, x_k_new, x_km1_new, qlp_iter_new = lax.cond(
                use_minres, minres_update, qlp_update
            )
            
            # =================================================================
            # 8. PREPARE NEXT P2 ROTATION
            # =================================================================
            c_k_2_new, s_k_2_new, gamma_km1_final = _sym_ortho_jax(gamma_km1_new, eps_kp1_new)
            
            # =================================================================
            # 9. STORE QLP QUANTITIES
            # =================================================================
            gamma_qlp_km1_new = gamma_k_tmp
            theta_qlp_k_new = theta_k_new
            gamma_qlp_k_new = gamma_k_final
            mu_qlp_km1_new = mu_km1_new
            mu_qlp_k_new = mu_k_new
            
            # =================================================================
            # 10. ESTIMATE NORMS & CHECK CONVERGENCE\n            # Computes estimates of ||A|| and cond(A) from Lanczos elements\n            # Checks convergence: relative residual <= tolerance\n            # Also checks for ill-conditioning and iteration limits\n            # =================================================================
            abs_gamma = jnp.abs(gamma_k_final)
            a_norm_new = jnp.maximum(a_norm, jnp.maximum(jnp.abs(gamma_km1_new), jnp.maximum(abs_gamma, pnorm_rho_k)))
            
            gamma_min_new = lax.cond(
                k == 0,
                lambda: abs_gamma,
                lambda: jnp.minimum(gamma_min, jnp.minimum(jnp.abs(gamma_km1_new), abs_gamma))
            )
            
            safe_gamma_min = jnp.where(gamma_min_new < _MIN_NORM, _MIN_NORM, gamma_min_new)
            a_cond_new = a_norm_new / safe_gamma_min
            
            rnorm = jnp.where(
                jnp.logical_or(flag_new == int(MinresQLPFlag.PROCESSING), 
                              flag_new == int(MinresQLPFlag.SINGULAR)),
                jnp.abs(phi_k_new),
                relres * beta1
            )
            relres_new = rnorm / (a_norm_new * xnorm_k_new + beta1 + _TINY)
            
            # Convergence checks
            flag_final = jnp.where(
                flag_new == int(MinresQLPFlag.PROCESSING),
                jnp.where(
                    k >= maxiter - 1,
                    int(MinresQLPFlag.MAX_ITER),
                    jnp.where(
                        a_cond_new > _MAX_A_COND,
                        int(MinresQLPFlag.A_COND_LIMIT),
                        jnp.where(
                            relres_new <= tol,
                            int(MinresQLPFlag.RESID_RTOL),
                            int(MinresQLPFlag.PROCESSING)
                        )
                    )
                ),
                flag_new
            )
            
            # Pack new state
            return (
                k + 1, x_k_new, x_km1_new, w_k_new, w_km1_new, w_km2_new,
                z_k_precond, z_km1_new, z_km2_new,
                beta_k_new, beta_km1_new, beta_last_new, phi_k_new, beta1,
                c_k_1_new, s_k_1_new, delta_k_new, eps_kp1_new,
                c_k_2_new, s_k_2_new, c_k_3_new, s_k_3_new,
                gamma_k_final, gamma_km1_final, gamma_km2_new, gamma_km3_new,
                theta_k_new, theta_km1_new, theta_km2_new,
                eta_k_new, eta_km1_new, eta_km2_new,
                tau_k_new, tau_km1_new, tau_km2_new,
                mu_k_new, mu_km1_new, mu_km2_new, mu_km3_new, mu_km4_new,
                qlp_iter_new,
                gamma_qlp_k_new, gamma_qlp_km1_new, theta_qlp_k_new,
                mu_qlp_k_new, mu_qlp_km1_new,
                xl2norm_k_new, xnorm_k_new, a_norm_new,
                a_cond_new, gamma_min_new,
                relres_new, flag_final
            )
        
        # Run the while loop with the condition and body functions
        final_state     = lax.while_loop(cond_fun, body_fun, init_state)
        
        k_final         = final_state[0]
        x_final         = final_state[1]
        beta1_final     = final_state[13]
        relres_final    = final_state[51]
        flag_final      = final_state[52]
        
        converged       = jnp.logical_and(flag_final > 0, flag_final < 8)
        residual        = relres_final * beta1_final
        
        return SolverResult(
            x               =   x_final,
            converged       =   converged,
            iterations      =   k_final,
            residual_norm   =   residual
        )

    # Compile with static args: 
    # matvec (0), tol (3), maxiter (4), precond_apply (5)
    # NOTE: sigma is NOT in the signature - it should be incorporated into matvec
    _minres_qlp_logic_jax_compiled = jax.jit(_minres_qlp_logic_jax, static_argnums=(0, 3, 4, 5))

else:
    _minres_qlp_logic_jax_compiled = None

# ##############################################################################
#! MinresQLP Solver Class
# ##############################################################################

class MinresQLPSolver(Solver):
    """
    MINRES-QLP solver for symmetric (possibly indefinite/singular) linear systems.
    
    Solves (A - sigma*I)x = b using the MINRES-QLP algorithm, which is stable
    for indefinite and singular systems.
    
    Examples
    --------
    >>> # Direct static solve
    >>> result = MinresQLPSolver.solve(matvec, b, x0, tol=1e-8, maxiter=1000, backend_module=np)
    
    >>> # Get compiled solver function for repeated use
    >>> solver_func = MinresQLPSolver.get_solver_func(jnp, use_fisher=True, sigma=1e-4)
    >>> result = solver_func(S, Sp, b, x0, tol=1e-8, maxiter=500)
    """
    _solver_type    = SolverType.MINRES_QLP
    _symmetric      = True

    @staticmethod
    def get_solver_func(
        backend_module  : Any,
        use_matvec      : bool              = True,
        use_fisher      : bool              = False,
        use_matrix      : bool              = False,
        sigma           : Optional[float]   = None,
        **kwargs
    ) -> StaticSolverFunc:
        """
        Returns the backend-specific compiled/optimized MINRES-QLP function.

        Args:
            backend_module: Numerical backend (numpy or jax.numpy)
            use_matvec: Use matvec interface (default True)
            use_fisher: Construct Fisher/Gram matrix from S, Sp
            use_matrix: Use dense matrix A directly
            sigma: Shift parameter for (A - sigma*I)

        Returns:
            Compiled solver function
        """
        if backend_module is jnp and JAX_AVAILABLE:
            if _minres_qlp_logic_jax_compiled is None:
                raise ImportError("JAX not available but JAX backend requested.")
            func = _minres_qlp_logic_jax
        elif backend_module is np:
            func = _minres_qlp_logic_numpy
        else:
            raise ValueError(f"Unsupported backend: {backend_module}")

        return Solver._solver_wrap_compiled(backend_module, func, use_matvec, use_fisher, use_matrix, sigma)

    @staticmethod
    def solve(
        matvec          : MatVecFunc,
        b               : Array,
        x0              : Array,
        *,
        tol             : float,
        maxiter         : int,
        precond_apply   : Optional[Callable[[Array], Array]]    = None,
        backend_module  : Any                                   = jnp,
        sigma           : Optional[float]                       = None,
        **kwargs
    ) -> SolverResult:
        """
        Solve symmetric linear system using MINRES-QLP.

        Args:
            matvec: Matrix-vector product function v -> Av
            b: Right-hand side vector
            x0: Initial guess
            tol: Relative tolerance
            maxiter: Maximum iterations
            precond_apply: Optional preconditioner M^{-1}
            backend_module: Backend (numpy or jax.numpy)
            sigma: Shift for (A - sigma*I)x = b

        Returns:
            SolverResult with solution, convergence status, iterations, residual
        """
        try:
            # Note: sigma should be incorporated into matvec by caller
            # We pass it as kwarg just in case, but JAX version ignores it
            sigma_val = 0.0 if sigma is None else float(sigma)
            
            if backend_module is jnp:
                if _minres_qlp_logic_jax_compiled is None:
                    raise ImportError("JAX not available")
                # Don't pass sigma as positional - it's in kwargs and ignored
                # (shift should be in matvec)
                return _minres_qlp_logic_jax_compiled(
                    matvec, b, x0, tol, maxiter, precond_apply
                )
            else:
                return _minres_qlp_logic_numpy(
                    matvec, b, x0, tol, maxiter, precond_apply, sigma_val
                )
        except Exception as e:
            raise RuntimeError(f"MINRES-QLP execution failed: {e}") from e

# ##############################################################################
#! EOF
# ##############################################################################
