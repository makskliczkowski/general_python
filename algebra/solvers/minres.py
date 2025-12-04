'''
file        : general_python/algebra/solvers/minres.py
author      : Maksymilian Kliczkowski

Implements the Minimum Residual (MINRES) iterative algorithm for solving linear systems
Ax = b where A is symmetric (but not necessarily positive definite).

MINRES uses a Lanczos process combined with Givens rotations to minimize the residual
norm ||Ax - b||_2 over the Krylov subspace.

Mathematical Background:
-----------------------
Given a symmetric matrix A, right-hand side b, initial guess x0, and preconditioner M:

1. Lanczos process generates orthonormal basis vectors v_i for Krylov subspace
2. Givens rotations maintain the QR factorization of the tridiagonal matrix
3. Solution x_k minimizes ||Ax - b||_2 over span{v_1, ..., v_k}

References:
-----------
- Paige, C. C., & Saunders, M. A. (1975). Solution of Sparse Indefinite Systems of Linear Equations.
  SIAM Journal on Numerical Analysis, 12(4), 617-629.
- Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM.
'''

import numpy as np
from typing import Optional, Callable, Any

try:
    from ..solver import (
        SolverType, Solver, SolverError, SolverErrorMsg,
        Array, MatVecFunc, StaticSolverFunc, SolverResult
    )
    from ..utils import JAX_AVAILABLE
    from ..preconditioners import Preconditioner, PreconitionerApplyFun
    from .backend_ops import BackendOps, get_backend_ops
except ImportError as e:
    raise ImportError("Failed to import necessary modules from the solver package. Ensure QES package is correctly installed.") from e

# -----------------------------------------------------------------------------
#! Backend imports
# -----------------------------------------------------------------------------

try:
    import scipy.sparse.linalg as spsla
except ImportError:  # pragma: no cover - scipy might not be installed
    spsla = None
    pass
if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        import jax.scipy.sparse.linalg as jspsla
    except Exception:   # pragma: no cover - jax path optional
        jnp     = np
        jspsla  = None
        jax     = None
else:                   # pragma: no cover - default path
    jax = None
    jnp = np
    jspsla = None

# -----------------------------------------------------------------------------
#! MINRES Solver Class using SciPy
# -----------------------------------------------------------------------------

class MinresSolverScipy(Solver):
    """
    Wrapper for SciPy/JAX MINRES implementations.
    Uses scipy.sparse.linalg.minres for NumPy backend,
    and jax.scipy.sparse.linalg.minres if available for JAX.
    """
    _solver_type = SolverType.SCIPY_MINRES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._symmetric = True

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
        """
        Return a backend-specific callable that runs MINRES.
        The returned function is wrapped via Solver._solver_wrap_compiled to handle
        matvec/matrix/Fisher mode dispatching.
        
        Parameters:
        -----------
            backend_module:
                The numerical backend (np or jnp).
            use_matvec:
                If True, expects a matvec function directly.
            use_fisher:
                If True, expects (s, s_p) Fisher decomposition.
            use_matrix:
                If True, expects a dense matrix A.
            sigma:
                Optional regularization (diagonal shift).
                
        Returns:
        --------
            StaticSolverFunc: A wrapped solver function.
        """
        # Check for JAX backend - must verify JAX is actually available
        # (when JAX is unavailable, jnp is aliased to np)
        is_jax_backend = JAX_AVAILABLE and (backend_module is not np)
        
        if is_jax_backend and jspsla is not None and hasattr(jspsla, 'minres'):
            # JAX path - jax.scipy.sparse.linalg.minres
            def solver_func(matvec, b, x0, tol, maxiter, precond_apply):
                x, info = jspsla.minres(matvec, b, x0=x0, tol=tol, maxiter=maxiter, M=precond_apply)
                # info==0 success; >0 no convergence; <0 illegal input
                conv = (info == 0)
                return SolverResult(x=x, converged=bool(conv), iterations=maxiter, residual_norm=None)
            return Solver._solver_wrap_compiled(backend_module, solver_func, use_matvec, use_fisher, use_matrix, sigma)

        elif spsla is not None:
            # NumPy/SciPy path - scipy.sparse.linalg.minres
            def solver_func(matvec, b, x0, tol, maxiter, precond_apply):
                # Wrap callable into LinearOperator expected by SciPy
                n   = b.shape[0]
                Aop = spsla.LinearOperator((n, n), matvec=matvec, dtype=b.dtype)
                Mop = None
                if precond_apply is not None:
                    Mop = spsla.LinearOperator((n, n), matvec=precond_apply, dtype=b.dtype)
                # Capture iteration count via callback
                it = {'k': 0}
                def _cb(_xk):
                    it['k'] += 1
                x, info = spsla.minres(Aop, b, x0=x0, rtol=tol, maxiter=maxiter, M=Mop, callback=_cb)
                conv    = (info == 0)
                return SolverResult(x=x, converged=bool(conv), iterations=it['k'], residual_norm=None)
            return Solver._solver_wrap_compiled(backend_module, solver_func, use_matvec, use_fisher, use_matrix, sigma)
        else:
            raise SolverError(SolverErrorMsg.METHOD_NOT_IMPL, f"MINRES SciPy/JAX not available for backend {backend_module}")

    @staticmethod
    def solve(
            matvec          : MatVecFunc,
            b               : Array,
            x0              : Array,
            *,
            tol             : float,
            maxiter         : int,
            precond_apply   : Optional[Callable[[Array], Array]]    = None,
            backend_module  : Any                                   = np,
            a               : Optional[Array]                       = None,
            s               : Optional[Array]                       = None,
            s_p             : Optional[Array]                       = None,
            **kwargs        : Any) -> 'SolverResult':
        """
        Static solve method for MINRES matching the Solver interface.
        
        Dispatches to the appropriate mode based on provided inputs:
        - If `matvec` is provided: uses matvec mode
        - If `s` and `s_p` are provided: uses Fisher mode (NQS)
        - If `a` is provided: uses matrix mode
        
        Parameters:
        -----------
            matvec: Matrix-vector product function A @ v
            b: Right-hand side vector
            x0: Initial guess
            tol: Convergence tolerance
            maxiter: Maximum iterations
            precond_apply: Optional preconditioner M^{-1}
            backend_module: NumPy or JAX module
            a: Optional dense matrix (alternative to matvec)
            s, s_p: Optional Gram matrix factors (alternative to matvec)
            **kwargs: Additional arguments (e.g., sigma for regularization)
            
        Returns:
        --------
            SolverResult with solution, convergence status, iterations, residual_norm
        """
        try:
            # Determine mode
            use_matvec = (matvec is not None)
            use_fisher = (s is not None and s_p is not None)
            use_matrix = (a is not None and not use_fisher)
            sigma = kwargs.get('sigma', None)
            
            # Get the wrapped solver function
            solver_func = MinresSolverScipy.get_solver_func(
                backend_module = backend_module,
                use_matvec     = use_matvec and not use_fisher and not use_matrix,
                use_fisher     = use_fisher,
                use_matrix     = use_matrix,
                sigma          = sigma
            )
            
            # Dispatch based on mode
            if use_fisher:
                # Fisher mode: pass (s, s_p, b, x0, tol, maxiter, precond_apply)
                return solver_func(s, s_p, b, x0, tol, maxiter, precond_apply, sigma=sigma)
            elif use_matrix:
                # Matrix mode: pass (a, b, x0, tol, maxiter, precond_apply)
                return solver_func(a, b, x0, tol, maxiter, precond_apply, sigma=sigma)
            elif use_matvec:
                # Matvec mode: pass (matvec, b, x0, tol, maxiter, precond_apply)
                return solver_func(matvec, b, x0, tol, maxiter, precond_apply)
            else:
                raise SolverError(SolverErrorMsg.MATVEC_FUNC_NOT_SET, 
                                  "Must provide matvec, matrix A, or Fisher (s, s_p)")
                
        except SolverError:
            raise
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"MINRES execution failed: {e}") from e

# -----------------------------------------------------------------------------
# Native MINRES Solver Class
# -----------------------------------------------------------------------------

def _minres_logic_numpy(matvec          : MatVecFunc,
                        b               : np.ndarray,
                        x0              : np.ndarray,
                        tol             : float,
                        maxiter         : int,
                        precond_apply   : Optional[Callable[[np.ndarray], np.ndarray]] = None
                        ) -> SolverResult:
    """
    Core MINRES algorithm implementation using NumPy with backend_ops.
    
    Solves Ax = b for symmetric A (not necessarily positive definite).
    Uses Lanczos process + Givens rotations to minimize residual norm.
    
    Parameters:
    ------------
        matvec:
            Function computing A @ v
        b: 
            Right-hand side vector
        x0: 
            Initial guess
        tol: 
            Convergence tolerance (relative residual)
        maxiter: 
            Maximum iterations
        precond_apply: 
            Optional preconditioner M^{-1}

    Returns:
        SolverResult with solution, convergence status, iterations, residual_norm
    """
    ops     = get_backend_ops('numpy')
    n       = b.shape[0]
    
    # Apply preconditioner helper
    def apply_precond(vec):
        return precond_apply(vec) if precond_apply is not None else vec
    
    # Initialize
    x       = x0.copy()
    r       = b - matvec(x)  # Initial residual
    y       = apply_precond(r)

    beta1   = np.sqrt(np.dot(r, y))

    if beta1 == 0:
        # Already converged
        return SolverResult(x=x, converged=True, iterations=0, residual_norm=0.0)
    
    bnorm   = ops.norm(b)
    if bnorm == 0:
        bnorm = 1.0
    
    # Normalize to get first Lanczos vector
    v_new   = y / beta1
    
    # Initialize solution direction vectors (previous two)
    w       = np.zeros(n)
    w_prev  = np.zeros(n)

    # Initialize Givens rotation parameters
    s_prev = 0.0
    c_prev = 1.0
    
    # Initialize residual components
    eta_bar = beta1
    residual_norm = beta1
    
    # Lanczos vectors (we only need to keep 3 at a time)
    v_old_old   = np.zeros(n)
    v_old       = np.zeros(n)
    v           = v_new

    # Tri-diagonal elements
    beta_old    = 0.0
    converged   = False
    
    for k in range(maxiter):
        # Lanczos step: v_{k+1} = A v_k - alpha_k v_k - beta_k v_{k-1}
        y       = apply_precond(matvec(v))
        
        # alpha_k = v_k^T A v_k
        alpha   = np.dot(v, y)
        
        # v_new = A v_k - alpha_k v_k - beta_k v_{k-1}
        y = y - (beta_old / beta1 if k > 0 else 0.0) * v_old
        y = y - (alpha / beta1) * v
        
        # Re-orthogonalization (full)
        y_beta = np.dot(y, apply_precond(y))
        beta = np.sqrt(y_beta) * beta1
        
        # QR factorization via Givens rotations
        # Apply previous rotation
        delta_k     = c_prev * alpha - s_prev * beta_old
        gamma_k     = s_prev * alpha + c_prev * beta_old
        epsilon_k   = c_prev * beta
        delta_kp1   = -s_prev * beta

        # Compute new Givens rotation
        rho_k       = np.sqrt(gamma_k**2 + beta**2)
        c           = gamma_k / rho_k if rho_k > 0 else 1.0
        s           = beta / rho_k if rho_k > 0 else 0.0

        # Update solution direction
        w_new       = (v - delta_k * w_prev - epsilon_k * w) / rho_k
        x           = x + c * eta_bar * w_new

        # Update residual estimate
        eta_bar     = -s * eta_bar
        residual_norm = np.abs(eta_bar)
        
        # Check convergence
        if residual_norm / bnorm <= tol:
            converged = True
            break
        
        # Prepare for next iteration
        if beta < 1e-14:  # Breakdown
            break
        
        # Shift vectors
        beta1_new       = beta / beta1
        v_old_old       = v_old.copy()
        v_old           = v.copy()
        v               = y / beta1_new
    
        w_prev          = w.copy()
        w               = w_new.copy()

        beta_old       = beta
        beta1          = beta1_new

        s_prev, c_prev = s, c
    
    return SolverResult(
        x=x,
        converged=converged,
        iterations=k + 1 if k < maxiter else maxiter,
        residual_norm=float(residual_norm)
    )

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.lax as lax
    
    def _minres_logic_jax(matvec          : MatVecFunc,
                          b               : jnp.ndarray,
                          x0              : jnp.ndarray,
                          tol             : float,
                          maxiter         : int,
                          precond_apply   : Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
                          ) -> SolverResult:
        """
        Core MINRES algorithm using JAX with jax.lax.fori_loop for JIT compilation.
        This implementation uses a functional loop compatible with JAX JIT.
        
        Note: All operations must be JAX-compatible - no Python bool/int/float 
        conversions on traced values.
        """
        n       = b.shape[0]
        dtype   = b.dtype  # Preserve input dtype (may be complex)
        
        # Apply preconditioner helper
        def apply_precond(vec):
            return precond_apply(vec) if precond_apply is not None else vec
        
        # Initialize
        r           = b - matvec(x0)
        y           = apply_precond(r)
        
        # Use real part for beta since it should be real for symmetric A
        beta1       = jnp.sqrt(jnp.real(jnp.vdot(y, r)))
        
        # Safe division for initial v
        beta1_safe  = jnp.where(beta1 == 0, 1.0, beta1)
        v           = y / beta1_safe
        
        # Compute b norm for relative residual
        bnorm       = jnp.linalg.norm(b)
        bnorm       = jnp.where(bnorm == 0, 1.0, bnorm)
        
        # Initial state - use correct dtype for all vectors
        # Use real dtype for scalars that should be real
        real_dtype = jnp.float64 if dtype == jnp.complex128 else jnp.float32
        
        state = {
            'x'             : x0.astype(dtype),
            'v'             : v.astype(dtype),
            'v_old'         : jnp.zeros(n, dtype=dtype),
            'w_old1'        : jnp.zeros(n, dtype=dtype),
            'w_old2'        : jnp.zeros(n, dtype=dtype),
            'alpha_old'     : jnp.array(0.0, dtype=real_dtype),
            'beta'          : jnp.array(beta1, dtype=real_dtype),
            'c'             : jnp.array(1.0, dtype=real_dtype),
            's'             : jnp.array(0.0, dtype=real_dtype),
            'phi_bar'       : jnp.array(beta1, dtype=real_dtype),
            'residual_norm' : jnp.array(jnp.abs(beta1), dtype=real_dtype),
            'iterations'    : jnp.array(0, dtype=jnp.int32)
        }
        
        def body_fun(k, state):
            # Lanczos step
            Av          = matvec(state['v'])
            Av          = apply_precond(Av)
            # alpha should be real for symmetric A
            alpha       = jnp.real(jnp.vdot(state['v'], Av))

            # Next Lanczos direction
            v_new       = Av - alpha * state['v'] - state['beta'] * state['v_old']
            y_new       = apply_precond(v_new)
            beta_new    = jnp.sqrt(jnp.real(jnp.vdot(v_new, y_new)))
            beta_new_safe = jnp.where(beta_new == 0, 1.0, beta_new)
            v_next      = y_new / beta_new_safe

            # Apply previous Givens rotation to get delta
            delta       = state['c'] * state['alpha_old'] + state['s'] * alpha
            
            # Compute new Givens rotation parameters
            gamma       = state['s'] * state['alpha_old'] - state['c'] * alpha
            epsilon     = state['s'] * beta_new
            
            # rho_bar for the current rotation
            rho_bar     = -state['c'] * beta_new
            
            # Compute rho (for normalization)
            rho         = jnp.sqrt(gamma**2 + beta_new**2)
            rho_safe    = jnp.where(rho == 0, 1.0, rho)
            
            # New Givens rotation
            c_new       = gamma / rho_safe
            s_new       = beta_new / rho_safe
            
            # Update solution direction
            w_new       = (state['v'] - delta * state['w_old1'] - epsilon * state['w_old2']) / rho_safe
            
            # Update solution
            x_new       = state['x'] + (state['c'] * state['phi_bar']) * w_new
            
            # Update residual estimate
            phi_bar_new     = state['s'] * state['phi_bar']
            residual_norm_new = jnp.abs(phi_bar_new)
            
            return {
                'x'             : x_new,
                'v'             : v_next,
                'v_old'         : state['v'],
                'w_old1'        : w_new,
                'w_old2'        : state['w_old1'],
                'alpha_old'     : jnp.array(alpha, dtype=state['alpha_old'].dtype),
                'beta'          : jnp.array(beta_new, dtype=state['beta'].dtype),
                'c'             : jnp.array(c_new, dtype=state['c'].dtype),
                's'             : jnp.array(s_new, dtype=state['s'].dtype),
                'phi_bar'       : jnp.array(phi_bar_new, dtype=state['phi_bar'].dtype),
                'residual_norm' : jnp.array(residual_norm_new, dtype=state['residual_norm'].dtype),
                'iterations'    : jnp.array(k + 1, dtype=jnp.int32)
            }
        
        final_state = lax.fori_loop(0, maxiter, body_fun, state)
        
        # Check convergence based on relative residual
        rel_res     = final_state['residual_norm'] / bnorm
        converged   = rel_res <= tol
        
        # Return SolverResult - values will be concrete after JIT execution
        return SolverResult(
            x               = final_state['x'],
            converged       = converged,
            iterations      = final_state['iterations'],
            residual_norm   = final_state['residual_norm']
        )

else:
    _minres_logic_jax = None

class MinresSolver(Solver):
    '''
    Native Minimum Residual (MINRES) Solver for symmetric matrices.
    Uses Lanczos process with Givens rotations to minimize residual norm.
    
    **Note**: This implementation is a work-in-progress and requires further debugging.
    For production use, prefer `MinresSolverScipy` which wraps the battle-tested
    SciPy implementation.
    
    TODO: Fix Lanczos orthogonalization and Givens rotation updates to ensure
    correct residual minimization.
    '''
    _solver_type = SolverType.MINRES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._symmetric = True

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
        """
        Return a backend-specific MINRES solver function.
        Uses native implementation with backend_ops.
        
        Note: Do NOT JIT the inner solver_func here - let _solver_wrap_compiled
        handle the JIT compilation with proper static_argnames for matvec.
        """
        is_jax = (backend_module is not np) and JAX_AVAILABLE
        
        if is_jax and _minres_logic_jax is not None:
            def solver_func(matvec, b, x0, tol, maxiter, precond_apply):
                return _minres_logic_jax(matvec, b, x0, tol, maxiter, precond_apply)
            return Solver._solver_wrap_compiled(backend_module, solver_func, use_matvec, use_fisher, use_matrix, sigma)
        else:
            def solver_func(matvec, b, x0, tol, maxiter, precond_apply):
                return _minres_logic_numpy(matvec, b, x0, tol, maxiter, precond_apply)
            return Solver._solver_wrap_compiled(backend_module, solver_func, use_matvec, use_fisher, use_matrix, sigma)

    @staticmethod
    def solve(
            matvec          : MatVecFunc,
            b               : Array,
            x0              : Array,
            *,
            tol             : float,
            maxiter         : int,
            precond_apply   : Optional[Callable[[Array], Array]]    = None,
            backend_module  : Any                                   = np,
            a               : Optional[Array]                       = None,
            s               : Optional[Array]                       = None,
            s_p             : Optional[Array]                       = None,
            **kwargs        : Any) -> 'SolverResult':
        """
        Static solve method for MINRES matching the Solver interface.
        
        Dispatches to the appropriate mode based on provided inputs:
        - If `matvec` is provided: uses matvec mode
        - If `s` and `s_p` are provided: uses Fisher mode (NQS)
        - If `a` is provided: uses matrix mode
        
        Parameters:
        -----------
            matvec: Matrix-vector product function A @ v
            b: Right-hand side vector
            x0: Initial guess
            tol: Convergence tolerance
            maxiter: Maximum iterations
            precond_apply: Optional preconditioner M^{-1}
            backend_module: NumPy or JAX module
            a: Optional dense matrix (alternative to matvec)
            s, s_p: Optional Gram matrix factors (alternative to matvec)
            **kwargs: Additional arguments (e.g., sigma for regularization)
            
        Returns:
        --------
            SolverResult with solution, convergence status, iterations, residual_norm
        """
        try:
            # Determine mode
            use_matvec  = (matvec is not None)
            use_fisher  = (s is not None and s_p is not None)
            use_matrix  = (a is not None and not use_fisher)
            sigma       = kwargs.get('sigma', None)
            
            # Get the wrapped solver function
            solver_func = MinresSolver.get_solver_func(
                backend_module = backend_module,
                use_matvec     = use_matvec and not use_fisher and not use_matrix,
                use_fisher     = use_fisher,
                use_matrix     = use_matrix,
                sigma          = sigma
            )
            
            # Dispatch based on mode
            if use_fisher:
                # Fisher mode: pass (s, s_p, b, x0, tol, maxiter, precond_apply)
                return solver_func(s, s_p, b, x0, tol, maxiter, precond_apply, sigma=sigma)
            elif use_matrix:
                # Matrix mode: pass (a, b, x0, tol, maxiter, precond_apply)
                return solver_func(a, b, x0, tol, maxiter, precond_apply, sigma=sigma)
            elif use_matvec:
                # Matvec mode: pass (matvec, b, x0, tol, maxiter, precond_apply)
                return solver_func(matvec, b, x0, tol, maxiter, precond_apply)
            else:
                raise SolverError(SolverErrorMsg.MATVEC_FUNC_NOT_SET, "Must provide matvec, matrix A, or Fisher (s, s_p)")
                
        except SolverError:
            raise
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"MINRES execution failed: {e}") from e

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------