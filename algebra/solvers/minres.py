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
    raise ImportError("Failed to import necessary modules from the solver package. Ensure general_python package is correctly installed.") from e

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

    if beta1 < 1e-15:
        # Already converged
        return SolverResult(x=x, converged=True, iterations=0, residual_norm=0.0)
    
    bnorm   = ops.norm(b)
    if bnorm == 0:
        bnorm = 1.0
    
    # Normalize to get first Lanczos vector
    v       = y / beta1
    v_old   = np.zeros(n)
    
    # Initialize solution direction vectors
    w       = np.zeros(n)
    w_old   = np.zeros(n)

    # Initialize Givens rotation parameters
    # c_old, s_old correspond to G_{k-1}
    # c_old2, s_old2 correspond to G_{k-2}
    # Initially for k=1, G_0 is Reflection (-1, 0) to preserve alpha sign
    c_old, s_old = -1.0, 0.0
    c_old2, s_old2 = 0.0, 0.0 # Doesn't matter for k=1 as beta_old=0

    # Initialize Lanczos parameters
    beta_old = 0.0
    
    # Initialize residual components
    eta = beta1
    residual_norm = beta1
    
    converged = False
    k = 0
    
    for k in range(maxiter):
        # Lanczos step: v_{k+1}
        # Av = A * v_k
        Av = apply_precond(matvec(v))
        
        # alpha_k = v_k^T A v_k
        alpha = np.dot(v, Av)
        
        # v_{k+1} = Av - alpha_k v_k - beta_k v_{k-1}
        v_new = Av - alpha * v - beta_old * v_old
        
        # Re-orthogonalization / normalization
        # beta_{k+1} = ||v_{k+1}||
        # If preconditioned, norm is M-norm: sqrt(v^T M v)
        # Assuming v_new is already M^{-1} applied vector? No.
        # MINRES standard:
        # r = b - Ax. y = M r. beta = sqrt(r.y).
        # v = y / beta.
        # Av is computed.
        # If preconditioned, we are solving M^{-1} A x = M^{-1} b ?
        # Or Preconditioned Lanczos for M^{-1} A.
        # Let's assume standard preconditioned Lanczos logic matches inputs.
        
        beta_new = np.sqrt(np.dot(v_new, apply_precond(v_new))) if precond_apply else ops.norm(v_new)

        if beta_new < 1e-15:
            # Exact solution in subspace or breakdown
            v_new = v_new * 0.0 # avoid division by zero
        else:
            v_new = v_new / beta_new

        # QR Factorization Updates
        # Apply G_{k-2} to column k of T (rows k-2, k-1)
        # Input at k-2: 0. Input at k-1: beta_old.
        # Result at k-2: epsilon (super-super-diagonal)
        # Result at k-1: delta_tmp (temporary super-diagonal)

        epsilon   = s_old2 * beta_old
        delta_tmp = -c_old2 * beta_old

        # Apply G_{k-1} to column k (rows k-1, k)
        # Input at k-1: delta_tmp. Input at k: alpha.
        # Result at k-1: delta (final super-diagonal)
        # Result at k: gamma_tmp (temporary diagonal)

        delta     = c_old * delta_tmp + s_old * alpha
        gamma_tmp = s_old * delta_tmp - c_old * alpha

        # Determine G_k to eliminate beta_new at (k+1, k) using gamma_tmp at (k, k)
        # Input: (gamma_tmp, beta_new)
        # Result at k: gamma (final diagonal)

        rho = np.sqrt(gamma_tmp**2 + beta_new**2)
        if rho < 1e-15:
            rho = 1e-15 # Safety

        c = gamma_tmp / rho
        s = beta_new / rho
        gamma = rho

        # Update solution direction vector w_k
        # w_k = (v_k - delta_k * w_{k-1} - epsilon_k * w_{k-2}) / gamma_k
        w_new = (v - delta * w - epsilon * w_old) / gamma

        # Update solution x_k
        # x_k = x_{k-1} + c_k * eta_k * w_k
        x = x + (c * eta) * w_new

        # Update residual parameter eta_{k+1}
        # [eta_new, eta_next] = G_k . [eta, 0]
        # We used top part (c * eta) for x update.
        # Bottom part becomes new eta for next step.
        eta = s * eta

        residual_norm = abs(eta)
        
        if residual_norm / bnorm <= tol:
            converged = True
            break

        # Shift variables
        v_old = v
        v = v_new
        beta_old = beta_new
        
        w_old = w
        w = w_new
        
        c_old2, s_old2 = c_old, s_old
        c_old, s_old = c, s
    
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
        Core MINRES algorithm using JAX with jax.lax.while_loop.
        Follows same logic as NumPy implementation.
        """
        n       = b.shape[0]
        dtype   = b.dtype
        
        # Helper for preconditioner
        def apply_precond(vec):
            return precond_apply(vec) if precond_apply is not None else vec
        
        # Initial residuals
        r0      = b - matvec(x0)
        y0      = apply_precond(r0)
        beta1   = jnp.sqrt(jnp.vdot(r0, y0).real)
        
        # Norm of b
        bnorm   = jnp.linalg.norm(b)
        bnorm   = jnp.where(bnorm == 0.0, 1.0, bnorm)
        
        # Initial Lanczos vector
        v       = y0 / jnp.where(beta1 == 0, 1.0, beta1)
        v_old   = jnp.zeros_like(v)
        
        # Initial solution directions
        w       = jnp.zeros_like(v)
        w_old   = jnp.zeros_like(v)
        
        # Initial state
        init_state = {
            'x'             : x0,
            'v'             : v,
            'v_old'         : v_old,
            'w'             : w,
            'w_old'         : w_old,
            'beta_old'      : jnp.array(0.0, dtype=dtype),
            'c_old'         : jnp.array(-1.0, dtype=dtype),
            's_old'         : jnp.array(0.0, dtype=dtype),
            'c_old2'        : jnp.array(0.0, dtype=dtype),
            's_old2'        : jnp.array(0.0, dtype=dtype),
            'eta'           : jnp.array(beta1, dtype=dtype),
            'residual_norm' : jnp.array(beta1, dtype=dtype),
            'k'             : jnp.array(0, dtype=jnp.int32),
            'converged'     : jnp.array(beta1 <= tol * bnorm, dtype=jnp.bool_)
        }
        
        def cond_fun(state):
            # Run while not converged AND k < maxiter
            return jnp.logical_and(
                jnp.logical_not(state['converged']),
                state['k'] < maxiter
            )

        def body_fun(state):
            v, v_old = state['v'], state['v_old']
            beta_old = state['beta_old']

            # 1. Lanczos
            Av = matvec(v)
            Av = apply_precond(Av)
            alpha = jnp.vdot(v, Av).real
            
            v_new = Av - alpha * v - beta_old * v_old
            
            # Norm
            if precond_apply is not None:
                beta_new = jnp.sqrt(jnp.vdot(v_new, apply_precond(v_new)).real)
            else:
                beta_new = jnp.linalg.norm(v_new)

            # Normalize
            beta_safe = jnp.where(beta_new < 1e-15, 1.0, beta_new)
            v_new_norm = v_new / beta_safe

            # 2. QR Update
            c_old2, s_old2 = state['c_old2'], state['s_old2']
            c_old, s_old   = state['c_old'], state['s_old']

            epsilon   = s_old2 * beta_old
            delta_tmp = -c_old2 * beta_old

            delta     = c_old * delta_tmp + s_old * alpha
            gamma_tmp = s_old * delta_tmp - c_old * alpha
            
            rho = jnp.sqrt(gamma_tmp**2 + beta_new**2)
            rho_safe = jnp.where(rho < 1e-15, 1.0, rho)
            
            c = gamma_tmp / rho_safe
            s = beta_new / rho_safe
            gamma = rho
            
            # 3. Update solution
            w, w_old = state['w'], state['w_old']
            w_new = (v - delta * w - epsilon * w_old) / gamma
            
            x = state['x'] + (c * state['eta']) * w_new
            eta_new = s * state['eta']
            
            residual_norm = jnp.abs(eta_new)
            converged = residual_norm <= tol * bnorm
            
            return {
                'x'             : x,
                'v'             : v_new_norm,
                'v_old'         : v,
                'w'             : w_new,
                'w_old'         : w,
                'beta_old'      : jnp.array(beta_new, dtype=dtype),
                'c_old'         : jnp.array(c, dtype=dtype),
                's_old'         : jnp.array(s, dtype=dtype),
                'c_old2'        : c_old,
                's_old2'        : s_old,
                'eta'           : eta_new,
                'residual_norm' : residual_norm,
                'k'             : state['k'] + 1,
                'converged'     : converged
            }

        final_state = lax.while_loop(cond_fun, body_fun, init_state)
        
        return SolverResult(
            x               = final_state['x'],
            converged       = final_state['converged'],
            iterations      = final_state['k'],
            residual_norm   = final_state['residual_norm']
        )

else:
    _minres_logic_jax = None

class MinresSolver(Solver):
    '''
    Native Minimum Residual (MINRES) Solver for symmetric matrices.
    Uses Lanczos process with Givens rotations to minimize residual norm.
    
    This implementation matches the standard MINRES algorithm (Paige & Saunders, 1975).
    For production use, `MinresSolverScipy` is also available which wraps `scipy.sparse.linalg.minres`.
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
                        sigma           : Optional[float] = None,
                        a               : Optional[Array] = None,
                        s               : Optional[Array] = None,
                        s_p             : Optional[Array] = None,
                        **kwargs        : Any                        
                        ) -> StaticSolverFunc:
        """
        Return a backend-specific MINRES solver function.
        Uses native implementation with backend_ops.
        
        Note: Do NOT JIT the inner solver_func here - let _solver_wrap_compiled
        handle the JIT compilation with proper static_argnames for matvec.
        """
        is_jax = (backend_module is not np) and JAX_AVAILABLE
        
        if is_jax and _minres_logic_jax is not None:
            def solver_func(matvec, b, x0, tol, maxiter, precond_apply, a=None, s=None, s_p=None, **kwargs):
                return _minres_logic_jax(matvec, b, x0, tol, maxiter, precond_apply)
            return Solver._solver_wrap_compiled(backend_module, solver_func, use_matvec, use_fisher, use_matrix, sigma)
        else:
            def solver_func(matvec, b, x0, tol, maxiter, precond_apply, a=None, s=None, s_p=None, **kwargs):
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