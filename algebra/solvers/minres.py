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
        The returned function must match the signature used by Solver.run_solver_func.
        """
        if backend_module is jnp and jspsla is not None and hasattr(jspsla, 'minres'):  # JAX path (if available)
            @jax.jit
            def solver_func(matvec, b, x0, tol, maxiter, precond_apply):
                x, info = jspsla.minres(matvec, b, x0=x0, tol=tol, maxiter=maxiter, M=precond_apply)
                # info==0 success; >0 no convergence; <0 illegal input
                conv = (info == 0)
                return SolverResult(x=x, converged=bool(conv), iterations=maxiter, residual_norm=None)
            return Solver._solver_wrap_compiled(backend_module, solver_func, use_matvec, use_fisher, use_matrix, sigma)

        elif backend_module is np and spsla is not None:
            def solver_func(matvec, b, x0, tol, maxiter, precond_apply):
                # Wrap callable into LinearOperator expected by SciPy
                n = b.shape[0]
                Aop = spsla.LinearOperator((n, n), matvec=matvec, dtype=b.dtype)
                Mop = None
                if precond_apply is not None:
                    Mop = spsla.LinearOperator((n, n), matvec=precond_apply, dtype=b.dtype)
                # capture iteration count via callback
                it              = {'k': 0}
                def _cb(_xk):   it['k'] += 1
                x, info         = spsla.minres(Aop, b, x0=x0, rtol=tol, maxiter=maxiter, M=Mop, callback=_cb)
                conv            = (info == 0)
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
        try:
            solver_func = MinresSolverScipy.get_solver_func(
                backend_module = backend_module,
                use_matvec     = (matvec is not None),
                use_fisher     = (s is not None),
                use_matrix     = (a is not None),
                sigma          = kwargs.get('sigma', None)
            )
            return Solver.run_solver_func(
                bck     = backend_module,
                func    = solver_func,
                matvec  = matvec,
                a       = a,
                s       = s,
                s_p     = s_p,
                b       = b,
                x0      = x0,
                tol     = tol,
                maxiter = maxiter,
                precond_apply = precond_apply
            )
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
        """
        ops = get_backend_ops('jax')
        n   = b.shape[0]
        
        # Apply preconditioner helper
        def apply_precond(vec):
            return precond_apply(vec) if precond_apply is not None else vec
        
        # Initialize
        x       = x0
        r       = b - matvec(x)
        y       = apply_precond(r)
        beta1   = jnp.sqrt(jnp.dot(y, r))
        
        # Early exit if already converged
        def early_check():
            return SolverResult(x=x, converged=True, iterations=0, residual_norm=0.0)
        
        def run_iterations():
            v = y / beta1
            
            # Initial state
            state = {
                'x'             : x,
                'v'             : v,
                'v_old'         : jnp.zeros(n),
                'w_old1'        : jnp.zeros(n),
                'w_old2'        : jnp.zeros(n),
                'alpha_old'     : 0.0,
                'beta'          : beta1,
                'c'             : 1.0,
                's'             : 0.0,
                'phi_bar'       : beta1,
                'residual_norm' : jnp.abs(beta1),
                'converged'     : False,
                'iterations'    : 0
            }
            
            bnorm = jnp.linalg.norm(b)
            bnorm = jnp.where(bnorm == 0, 1.0, bnorm)
            
            def body_fun(k, state):
                # Lanczos step
                Av          = matvec(state['v'])
                Av          = apply_precond(Av)
                alpha       = jnp.dot(state['v'], Av)

                # Next Lanczos direction
                v_new       = Av - alpha * state['v'] - state['beta'] * state['v_old']
                y_new       = apply_precond(v_new)
                beta_new    = jnp.sqrt(jnp.dot(v_new, y_new))
                v_next      = jnp.where(beta_new > 0, y_new / beta_new, y_new)

                # Apply Givens rotation
                delta       = state['s'] * alpha - state['c'] * state['beta']
                gamma       = state['c'] * alpha + state['s'] * state['beta']

                # Compute new Givens rotation (using NumPy version in ops)
                # For JAX, we need a pure implementation
                c_new       = jnp.where(beta_new == 0, 1.0, jnp.where(jnp.abs(beta_new) > jnp.abs(gamma),
                                   gamma / jnp.sqrt(gamma**2 + beta_new**2),
                                   jnp.sign(gamma) / jnp.sqrt(1 + (beta_new/gamma)**2)))
                s_new       = jnp.where(beta_new == 0, 0.0, jnp.where(jnp.abs(beta_new) > jnp.abs(gamma),
                                   beta_new / jnp.sqrt(gamma**2 + beta_new**2),
                                   c_new * beta_new / gamma))
                rho         = jnp.sqrt(gamma**2 + beta_new**2)
                
                # Update solution
                w_new       = (state['v'] - delta * state['w_old1'] - state['alpha_old'] * state['w_old2']) / rho
                x_new       = state['x'] + (c_new * state['phi_bar']) * w_new
                
                # Update residual estimate
                phi_bar_new         = -s_new * state['phi_bar']
                residual_norm_new   = jnp.abs(phi_bar_new)
                
                # Check convergence
                rel_res             = residual_norm_new / bnorm
                converged_new       = rel_res <= tol
                
                return {
                    'x'             : x_new,
                    'v'             : v_next,
                    'v_old'         : state['v'],
                    'w_old1'        : w_new,
                    'w_old2'        : state['w_old1'],
                    'alpha_old'     : alpha,
                    'beta'          : beta_new,
                    'c'             : c_new,
                    's'             : s_new,
                    'phi_bar'       : phi_bar_new,
                    'residual_norm' : residual_norm_new,
                    'converged'     : converged_new,
                    'iterations'    : k + 1
                }
            
            final_state = lax.fori_loop(0, maxiter, body_fun, state)
            
            return SolverResult(
                x               = final_state['x'],
                converged       = bool(final_state['converged']),
                iterations      = int(final_state['iterations']),
                residual_norm   = float(final_state['residual_norm'])
            )
        
        return lax.cond(beta1 == 0, early_check, run_iterations)

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
        """
        is_jax = (backend_module is not np) and JAX_AVAILABLE
        
        if is_jax and _minres_logic_jax is not None:
            @jax.jit
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
        
        Args:
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
            SolverResult with solution, convergence status, iterations, residual_norm
        """
        try:
            solver_func = MinresSolver.get_solver_func(
                backend_module = backend_module,
                use_matvec     = (matvec is not None),
                use_fisher     = (s is not None),
                use_matrix     = (a is not None),
                sigma          = kwargs.get('sigma', None)
            )
            return Solver.run_solver_func(
                bck     = backend_module,
                func    = solver_func,
                matvec  = matvec,
                a       = a,
                s       = s,
                s_p     = s_p,
                b       = b,
                x0      = x0,
                tol     = tol,
                maxiter = maxiter,
                precond_apply = precond_apply
            )
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"MINRES execution failed: {e}") from e

    