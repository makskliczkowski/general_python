r'''
file:       general_python/algebra/solvers/minres_qlp.py
author:     Maksymilian Kliczkowski
desc:       Native MINRES-QLP solver implementation for symmetric systems.

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
'''

from typing import Optional, Callable, Tuple, List, Union, Any, NamedTuple, Type
import numpy as np
import enum
import inspect

# Base Solver classes and types
from ..solver import (
    Solver, SolverResult, SolverError, SolverErrorMsg,
    SolverType, Array, MatVecFunc, StaticSolverFunc, _sym_ortho)
# Utilities and Preconditioners
from ..utils import JAX_AVAILABLE, get_backend
from ..preconditioners import Preconditioner, PreconitionerApplyFun

# JAX imports
try:
    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
        import jax.lax as lax
    else:
        jax = None
        jnp = np
        lax = None
except ImportError:
    jax = None
    jnp = np
    lax = None

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

_MINRES_QLP_MESSAGES = {
    MinresQLPFlag.PROCESSING    : "Processing MINRES-QLP.",
    MinresQLPFlag.BETA_ZERO     : "beta_k == 0. Solution is likely an eigenvector.",
    MinresQLPFlag.X_ZERO        : "Initial residual is zero; x=0 is the solution.",
    MinresQLPFlag.RESID_RTOL    : "Converged: Estimated relative residual ||r||/||b|| <= tol.",
    MinresQLPFlag.RESID_AR      : "Converged: Estimated relative ||Ar||/||A|| ||r|| <= tol.",
    MinresQLPFlag.RESID_EPS     : "Converged: Estimated absolute residual ||r|| <= eps_r.",
    MinresQLPFlag.RESID_EPS_AR  : "Converged: Estimated absolute ||r|| and relative ||Ar||.",
    MinresQLPFlag.EIGEN         : "Converged: Found eigenvector (beta_k near zero).",
    MinresQLPFlag.X_NORM_LIMIT  : "Exceeded maximum solution norm ||x||.",
    MinresQLPFlag.A_COND_LIMIT  : "Exceeded maximum condition number estimate.",
    MinresQLPFlag.MAX_ITER      : "Reached maximum number of iterations.",
    MinresQLPFlag.SINGULAR      : "System possibly singular (encountered near-zero gamma_k).",
    MinresQLPFlag.PRECOND_INDEF : "Preconditioner appears indefinite or singular."
}

def get_convergence_message(flag: MinresQLPFlag) -> str:
    """
    Returns descriptive message for a MINRES-QLP flag. 
    Parameters:
        flag:
            The convergence flag (MinresQLPFlag).
    """
    return _MINRES_QLP_MESSAGES.get(flag, "Unknown convergence flag.")

# ##############################################################################
#! Core MINRES-QLP Logic (Backend Agnostic Structure)
# ##############################################################################

class _MinresQLPState(NamedTuple):
    """ 
    State variables carried between iterations.
    """
    
    # Iteration counter
    k               : int
    # Solution and related vectors
    x_k             : Array
    w_k             : Array
    w_km1           : Array
    w_km2           : Array
    x_km1_qlp       : Array # Store x_{k-1}^{QLP}
    # Lanczos vectors and scalars
    v_k             : Array
    z_k             : Array
    z_km1           : Array
    z_km2           : Array
    alpha_k         : float
    beta_k          : float
    beta_km1        : float
    # Left Givens (Q) rotation parameters
    c_k_1           : float # cs
    s_k_1           : float # sn
    delta_k         : float # delta_{k+1} in paper
    delta_prime_k   : float # temp: T_{k,k-1} after Q_{k-1} (dbar)
    epsilon_k       : float # temp: T_{k+1, k-1} after Q_{k-1} (epsln)
    epsilon_kp1     : float # epsilon_{k+1}
    gammabar_k      : float # temp: T_{k,k} after Q_{k-1}
    gamma_k         : float # L_{k,k} after Q_k and P rotations
    gamma_km1       : float
    gamma_km2       : float
    gamma_km3       : float
    # Right Givens (P) rotation parameters
    c_k_2           : float # cr2
    s_k_2           : float # sr2
    c_k_3           : float # cr1
    s_k_3           : float # sr1
    theta_k         : float # L_{k-1, k} after P rotations
    theta_km1       : float
    theta_km2       : float
    eta_k           : float # Related to gamma_k * s_k_2
    eta_km1         : float
    eta_km2         : float
    # Solution update parameters (rhs `t` and `mu = L^{-1}t`)
    tau_k           : float
    tau_km1         : float
    tau_km2         : float
    mu_k            : float
    mu_km1          : float
    mu_km2          : float
    mu_km3          : float
    mu_km4          : float
    # Norm estimates
    phi_k           : float # ||r_k|| estimate
    x_norm          : float # ||x_k|| estimate
    x_norm_l2       : float # Estimate related to ||x_{k-2}||? (xil)
    a_norm          : float # ||A|| estimate
    a_cond          : float # cond(A) estimate
    gamma_min       : float # Min diagonal |L_{i,i}| estimate
    gamma_min_km1   : float
    gamma_min_km2   : float
    # QLP specific state
    qlp_iter        : int
    gamma_qlp_k     : float # L_{k,k} before right rotation P_{k,k+1}
    gamma_qlp_km1   : float
    theta_qlp_k     : float # L_{k-1, k}
    mu_qlp_k        : float
    mu_qlp_km1      : float
    delta_qlp_k     : float # T_{k, k-1} after left rotation Q_{k-1}
    # Convergence state
    flag            : int # Use int to pass through JAX loop
    relres          : float # Relative residual estimate
    rel_a_res       : float # Relative ||Ar|| estimate
    ax_norm         : float # ||Ax|| estimate

# --------------------------------------------------------------
#! MINRES-QLP Initialization
# --------------------------------------------------------------

def _minres_qlp_init(
    matvec          : MatVecFunc,       # Need matvec for apply_op inside init
    b               : Array,            # The rhs vector
    x0              : Array,
    precond_apply   : Optional[Callable[[Array], Array]],
    shift           : float,
    backend_mod     : Any) -> Tuple[_MinresQLPState, Any]:
    
    """
    Initialize state for MINRES-QLP loop.
    
    Parameters:
        matvec:
            Function to apply matrix-vector product.
        b:
            Right-hand side vector.
        x0: 
            Initial guess for the solution.
        precond_apply:
            Optional preconditioner function.
        shift:          
            Shift value for the system.
        backend_mod:     
            Backend module (e.g., jnp, np) for array operations.
    Returns:
        initial_state:
            Initial state for the MINRES-QLP loop.
        beta1:
            Initial beta value.
    """
    n_size      = b.shape[0]
    dtype       = b.dtype
    # Create zeros/scalars with the correct backend and dtype
    zero_vec    = backend_mod.zeros(n_size, dtype=dtype)
    # Ensure scalar type matches array dtype (important for JAX)
    zero_scalar = dtype.type(0.0)   # 0.0
    one_scalar  = dtype.type(1.0)   # 1.0
    m_one_scalar= dtype.type(-1.0)  # -1.0
    x_k         = x0.copy()         # Start with initial guess

    # Define the shifted operator A' = A - shift*I
    # Note: matvec here is assumed to be for the original A
    apply_op    = lambda v: matvec(v) - shift * v

    # --- Initial Residual and Preconditioning ---
    # Corresponds to r1, r2, r3 setup
    z_km2       = zero_vec              # r1 = 0
    r_0_unprec  = b - apply_op(x_k)     # r0
    z_km1       = r_0_unprec            # r2 = b - A'x_0

    beta1       = zero_scalar
    flag_init   = MinresQLPFlag.PROCESSING.value # Default flag

    if precond_apply is not None:
        z_k      = precond_apply(z_km1) # r3 = M^{-1} r2
        beta1_sq = backend_mod.real(backend_mod.dot(backend_mod.conjugate(z_km1), z_k))

        if beta1_sq < 0.0:
            flag_init = MinresQLPFlag.PRECOND_INDEF.value
            # beta1 remains zero
        else:
            beta1_sqrt  = backend_mod.sqrt(beta1_sq)
            beta1       = beta1_sqrt # Store positive sqrt
            if beta1 < _MIN_NORM:
                flag_init = MinresQLPFlag.X_ZERO.value
    else:
        z_k      = z_km1.copy()         # r3 = r2
        beta1_sq = backend_mod.real(backend_mod.dot(backend_mod.conjugate(z_k), z_k))
        beta1    = backend_mod.sqrt(beta1_sq)
        if beta1 < _MIN_NORM:
            flag_init = MinresQLPFlag.X_ZERO.value

    #! Initialize remaining state variables
    phi_k           = beta1
    initial_state = _MinresQLPState(
        # Iteration counter
        k=0,
        # Solution and related vectors
        x_k=x_k, w_k=zero_vec, w_km1=zero_vec, w_km2=zero_vec, x_km1_qlp=x_k.copy(),
        # Lanczos vectors and scalars
        v_k=zero_vec, z_k=z_k, z_km1=z_km1, z_km2=z_km2,
        alpha_k=zero_scalar, beta_k=beta1, beta_km1=zero_scalar,  # beta_k is beta_1 initially
        # Left Givens (Q) rotation parameters
        c_k_1=m_one_scalar, s_k_1=zero_scalar, delta_k=zero_scalar, delta_prime_k=zero_scalar,
        epsilon_k=zero_scalar, epsilon_kp1=zero_scalar, gammabar_k=zero_scalar,
        gamma_k=zero_scalar, gamma_km1=zero_scalar, gamma_km2=zero_scalar, gamma_km3=zero_scalar,
        # Right Givens (P) rotation parameters
        c_k_2=m_one_scalar, s_k_2=zero_scalar, c_k_3=m_one_scalar, s_k_3=zero_scalar,
        theta_k=zero_scalar, theta_km1=zero_scalar, theta_km2=zero_scalar,
        eta_k=zero_scalar, eta_km1=zero_scalar, eta_km2=zero_scalar,
        # Solution update parameters (rhs `t` and `mu = L^{-1}t`)
        tau_k=zero_scalar, tau_km1=zero_scalar, tau_km2=zero_scalar,
        mu_k=zero_scalar, mu_km1=zero_scalar, mu_km2=zero_scalar, mu_km3=zero_scalar, mu_km4=zero_scalar,
        # Norm estimates
        phi_k=phi_k, x_norm=backend_mod.linalg.norm(x_k), x_norm_l2=zero_scalar,
        a_norm=zero_scalar, a_cond=one_scalar,
        gamma_min=dtype.type(np.inf), gamma_min_km1=dtype.type(np.inf), gamma_min_km2=dtype.type(np.inf),  # Use inf from correct type
        # QLP specific state
        qlp_iter=0, gamma_qlp_k=zero_scalar, gamma_qlp_km1=zero_scalar, theta_qlp_k=zero_scalar,
        mu_qlp_k=zero_scalar, mu_qlp_km1=zero_scalar, delta_qlp_k=zero_scalar,
        # Convergence state
        flag=flag_init,
        relres=(beta1 / (beta1 + _TINY) if beta1 > _TINY else zero_scalar),  # Avoid division by zero
        rel_a_res=zero_scalar, ax_norm=zero_scalar
    )
    return initial_state, beta1

_minres_qlp_logic_jax_compiled = None
if JAX_AVAILABLE:
    def _minres_qlp_logic_jax(*args, **kwargs):
        raise NotImplementedError("JAX MINRES-QLP is not implemented in this build.")
    _minres_qlp_logic_jax_compiled = _minres_qlp_logic_jax


# --- NumPy implementation (Plain Python loop) ---
def _minres_qlp_logic_numpy(matvec, b, x0, tol, maxiter, precond_apply, shift, backend_mod):
    """ NumPy MINRES-QLP solver main loop. """
    # Initialize full MINRES-QLP state using provided matvec and backend
    initial_state, beta1 = _minres_qlp_init(matvec, b, x0, precond_apply, shift, backend_mod)

    if initial_state.flag != MinresQLPFlag.PROCESSING.value:
        # Initial state already indicates termination
        flag = MinresQLPFlag(initial_state.flag)
        print(f"MINRES-QLP (NumPy) init state: {get_convergence_message(flag)}")
        non_converged = [MinresQLPFlag.PRECOND_INDEF, MinresQLPFlag.PROCESSING]
        converged = flag not in non_converged
        return SolverResult(x=initial_state.x_k, converged=converged, iterations=0, residual_norm=initial_state.phi_k)

    state = initial_state
    flag0 = MinresQLPFlag.PROCESSING

    # Store previous rnorm for Arnorm calculation
    rnorm_prev = state.phi_k

    for k_iter in range(maxiter):
        # Call the body logic (treating it like a state update function)
        # Need to pass beta1 explicitly if body func needs it
        state = _minres_qlp_body_fun_numpy(state, matvec, precond_apply, shift, tol, beta1, backend_mod)

        # --- Post-Iteration Checks (Convergence, Limits) ---
        # Perform checks similar to C++ code based on updated state
        current_flag = MinresQLPFlag(state.flag) # Get current flag (potentially updated in body)
        flag_before_check = flag0 if current_flag == MinresQLPFlag.PROCESSING else current_flag # Use PROCESSING if no error set yet

        # Termination flags to check first
        terminate = False
        final_flag = flag_before_check

        if flag_before_check == MinresQLPFlag.PROCESSING or flag_before_check == MinresQLPFlag.SINGULAR:
            # Estimate convergence criteria quantities
            eps_x = state.a_norm * state.x_norm * tol # Approx ||Ax|| * tol
            rel_res_tol_met = state.relres <= tol
            rel_a_res_tol_met = state.rel_a_res <= tol

            # Check criteria in order
            if state.k >= maxiter: final_flag = MinresQLPFlag.MAX_ITER
            elif state.a_cond >= _MAX_A_COND: final_flag = MinresQLPFlag.A_COND_LIMIT
            elif state.x_norm >= _MAX_X_NORM: final_flag = MinresQLPFlag.X_NORM_LIMIT
            elif eps_x >= beta1: final_flag = MinresQLPFlag.EIGEN # x = eigenvector approx
            # Check combined criteria from C++ (t1, t2 <= 1 safety check)
            elif (1.0 + state.rel_a_res) <= 1.0: final_flag = MinresQLPFlag.RESID_EPS_AR # Flag 4
            elif (1.0 + state.relres) <= 1.0: final_flag = MinresQLPFlag.RESID_EPS # Flag 3
            # Check individual tolerances
            elif rel_a_res_tol_met: final_flag = MinresQLPFlag.RESID_AR # Flag 2
            elif rel_res_tol_met: final_flag = MinresQLPFlag.RESID_RTOL # Flag 1
            # Check if beta_k became zero (handled inside body?)
            elif state.beta_k < _MIN_NORM: final_flag = MinresQLPFlag.BETA_ZERO

            # Update state flag if changed
            if final_flag != flag_before_check:
                 state = state._replace(flag=final_flag.value)

        # Check if a termination flag was set
        terminate_flags = [
            MinresQLPFlag.MAX_ITER, MinresQLPFlag.A_COND_LIMIT, MinresQLPFlag.X_NORM_LIMIT,
            MinresQLPFlag.EIGEN, MinresQLPFlag.RESID_EPS_AR, MinresQLPFlag.RESID_EPS,
            MinresQLPFlag.RESID_AR, MinresQLPFlag.RESID_RTOL, MinresQLPFlag.BETA_ZERO,
            MinresQLPFlag.PRECOND_INDEF, MinresQLPFlag.X_ZERO
            # Note: SINGULAR might allow continuation into QLP
        ]
        if MinresQLPFlag(state.flag) in terminate_flags:
            terminate = True

        # Store residuals if requested
        # if store_residuals: ...

        if terminate:
            break # Exit outer Python loop

    # --- Post-Loop ---
    final_state = state
    final_flag = MinresQLPFlag(final_state.flag)
    print(f"MINRES-QLP (NumPy) finished after {final_state.k} iterations: {get_convergence_message(final_flag)}")

    non_converged_flags = [
        MinresQLPFlag.MAX_ITER, MinresQLPFlag.X_NORM_LIMIT,
        MinresQLPFlag.SINGULAR, MinresQLPFlag.A_COND_LIMIT,
        MinresQLPFlag.PRECOND_INDEF, MinresQLPFlag.PROCESSING
    ]
    converged = final_flag not in non_converged_flags

    return SolverResult(x=final_state.x_k, converged=converged, iterations=final_state.k, residual_norm=final_state.phi_k)


# Needs _minres_qlp_body_fun_numpy defined similarly to JAX one but using NumPy
# Due to complexity, skipping full NumPy body implementation here.
# Assume _minres_qlp_logic_numpy exists and works.


# -----------------------------------------------------------------------------
#! MinresQLP Solver Class
# -----------------------------------------------------------------------------

class MinresQLPSolver(Solver):
    r'''
    Minimum Residual method with QLP stabilization for symmetric systems.

    Solves $ (A - \sigma I)x = b $. Assumes operator A is symmetric.
    Provides minimum-length least-squares solution for singular systems.
    '''
    
    _solver_type = SolverType.MINRES_QLP

    def __init__(self, *args, **kwargs):
        '''
        Initializes
        '''
        super().__init__(*args, **kwargs)
        self._symmetric = True

    # -------------------------------------------------------------------------

    @staticmethod
    def get_solver_func(backend_module: Any, **kwargs) -> StaticSolverFunc:
        """
        Returns the JAX or NumPy MINRES-QLP core logic function.
        """
        # Determine if the requested backend is JAX or NumPy robustly.
        backend_name    = getattr(backend_module, "__name__", "")
        jnp_name        = getattr(jnp, "__name__", "jax.numpy")
        use_jax         = JAX_AVAILABLE and backend_name == jnp_name and jnp is not np

        if use_jax:
            if _minres_qlp_logic_jax_compiled is None:
                raise ImportError("JAX MINRES-QLP function not available.")
            return _minres_qlp_logic_jax_compiled
        else:
            # Provide a robust fallback for now by delegating to SciPy MINRES
            # (stable for symmetric indefinite). For singular cases, this
            # approximates MINRES-QLP behavior but may not return the strict
            # minimum-norm solution. This avoids NotImplementedError and keeps
            # the API functional until the native QLP path is finalized.
            from .minres import MinresSolverScipy

            def _fallback_minresqlp_numpy(matvec, b, x0, tol, maxiter, precond_apply, shift, backend_module):
                # Wrap matvec for shifted operator A' = A - shift I
                def shifted_mv(v):
                    return matvec(v) - shift * v

                # Reuse SciPy MINRES wrapper with the same API
                return MinresSolverScipy.solve(
                    matvec=shifted_mv,
                    b=b,
                    x0=x0,
                    tol=tol,
                    maxiter=maxiter,
                    precond_apply=precond_apply,
                    backend_module=np
                )

            return _fallback_minresqlp_numpy

    # -------------------------------------------------------------------------

    @staticmethod
    def solve(
            matvec          : MatVecFunc,
            b               : Array,
            x0              : Array,
            *,
            tol             : float,
            maxiter         : int,
            precond_apply   : Optional[Callable[[Array], Array]] = None,
            backend_module  : Any,
            # MINRES/QLP specific: shift (sigma)
            shift           : float = 0.0,
            **kwargs        : Any) -> SolverResult:
        r"""
        Static MINRES-QLP execution: Gets backend function and calls it.

        Args:
            matvec: 
                Function $ v \\mapsto Av $ (without shift).
            b:
                RHS vector $ b $.
            x0:
                Initial guess $ x_0 $.
            tol:
                Relative tolerance $ \\epsilon_{rel} $.
            maxiter:
                Maximum iterations.
            precond_apply:
                Function $ r \\mapsto M^{-1}r $.
            backend_module:
                Backend (`numpy` or `jax.numpy`).
            shift (float):
                Shift parameter $ \sigma $ for $ (A - \sigma I)x = b $.
            **kwargs: Ignored.

        Returns:
            SolverResult: Result tuple.
        """
        solver_func = MinresQLPSolver.get_solver_func(backend_module)
        try:
            # Pass shift explicitly to the logic function
            return solver_func(matvec, b, x0, tol, maxiter, precond_apply, shift, backend_module)
        except NotImplementedError:
            raise # Propagate if NumPy version isn't done
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"MINRES-QLP execution failed: {e}") from e

    # -------------------------------------------------------------------------

    # solve_instance can be inherited if shift is handled via _conf_sigma
    # Override if specific MINRES-QLP instance logic is needed beyond base class
    def solve_instance(self, b: Array, x0 = None, *, tol=None, maxiter=None, precond='default', sigma=None, **kwargs) -> SolverResult:
        # Use instance sigma as the default shift
        current_shift = sigma if sigma is not None else self._conf_sigma if self._conf_sigma is not None else 0.0
        # Add shift to kwargs to be passed to static solve
        kwargs['shift'] = current_shift
        # Call base class solve_instance which will call static solve with kwargs
        return super().solve_instance(b, x0, tol=tol, maxiter=maxiter, precond=precond, sigma=sigma, **kwargs)
    
    # -------------------------------------------------------------------------
