r'''
file:       general_python/algebra/solvers/cg_solver.py
author:     Maksymilian Kliczkowski

Implements the Conjugate Gradient (CG) iterative algorithm for solving linear systems
of equations :math:`Ax = b`, where the matrix :math:`A` is symmetric and positive-definite (SPD).
This file provides:
    1. A concrete `CgSolver` class inheriting from the base `Solver`, utilizing
        backend-specific implementations (NumPy/Numba or JAX JIT).
    2. A `CgSolverScipy` class wrapping `scipy.sparse.linalg.cg`.
    3. Helper functions containing the core CG logic for different backends.

Mathematical Formulation (Preconditioned CG):
-------------------------------------------
Given an SPD matrix :math:`A`, a right-hand side vector :math:`b`, an initial guess
:math:`x_0`, and an SPD preconditioner :math:`M \approx A`:

1.  Initialize:
    *   :math:`r_0 = b - Ax_0` (initial residual)
    *   :math:`z_0 = M^{-1}r_0` (apply preconditioner)
    *   :math:`p_0 = z_0` (initial search direction)
    *   :math:`\rho_0 = r_0^T z_0`

2.  Iterate :math:`k = 0, 1, 2, \dots` until convergence:
    *   :math:`\mathbf{v}_k = A p_k`
    *   :math:`\alpha_k = \rho_k / (p_k^T \mathbf{v}_k)` (step length)
    *   :math:`x_{k+1} = x_k + \alpha_k p_k`               (update solution)
    *   :math:`r_{k+1} = r_k - \alpha_k \mathbf{v}_k`     (update residual)
    *   Check convergence: e.g., :math:`\|r_{k+1}\|_2 < \epsilon \|b\|_2`
    *   :math:`z_{k+1} = M^{-1} r_{k+1}`                    (apply preconditioner)
    *   :math:`\rho_{k+1} = r_{k+1}^T z_{k+1}`
    *   :math:`\beta_k = \rho_{k+1} / \rho_k`            (update coefficient)
    *   :math:`p_{k+1} = z_{k+1} + \beta_k p_k`            (update search direction)
    *   :math:`\rho_k = \rho_{k+1}`

If no preconditioner is used (:math:`M = I`), then :math:`z_k = r_k`.

References:
-----------
    - Hestenes, M. R., & Stiefel, E. (1952). Methods of Conjugate Gradients for
        Solving Linear Systems. Journal of Research of the National Bureau of Standards, 49(6), 409.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM. Chapter 6.
    - Shewchuk, J. R. (1994). An Introduction to the Conjugate Gradient Method
        Without the Agonizing Pain. Carnegie Mellon University Technical Report CS-94-125.
'''

import os
import numpy as np
from typing import Optional, Callable, Any, Tuple
from functools import partial

try:
    from ..solver           import Solver, SolverResult, SolverError, SolverErrorMsg, SolverType, Array, MatVecFunc, StaticSolverFunc
    from ..preconditioners  import Preconditioner, PreconitionerApplyFun
except ImportError:
    raise ImportError("Could not import base solver or preconditioner classes. Check the module structure.")

try:
    import jax
    import jax.numpy    as jnp
    import jax.lax      as lax
    JAX_AVAILABLE       = True
except ImportError:
    JAX_AVAILABLE       = False
    jax                 = None
    jnp                 = None
    lax                 = None

# --- 

try:
    import numba
    _NUMBA_AVAILABLE    = True
except ImportError:
    numba = None
    _NUMBA_AVAILABLE    = False

# Import SciPy sparse if needed by CgSolverScipy
import scipy.sparse.linalg as spsla

# -----------------------------------------------------------------------------
#! Helper functions for CG logic (NumPy)
# -----------------------------------------------------------------------------

def _cg_logic_numpy(matvec          : MatVecFunc,
                    b               : np.ndarray,
                    x0              : np.ndarray,
                    tol             : float,
                    maxiter         : int,
                    precond_apply   : Optional[Callable[[np.ndarray], np.ndarray]] = None
                    ) -> SolverResult:
    """
    Core Conjugate Gradient (CG) algorithm implementation using NumPy.

    Solves the linear system $ Ax = b $ for a symmetric positive definite matrix A.
    If a preconditioner $ M \\approx A $ is provided, it solves $ M^{-1}Ax = M^{-1}b $
    implicitly by modifying the search direction updates.

    Args:
        matvec (MatVecFunc):
            Function performing the matrix-vector product $ v \\mapsto Av $.
        b (np.ndarray):
            Right-hand side vector $ b $.
        x0 (np.ndarray):
            Initial guess $ x_0 $.
        tol (float):
            Relative tolerance $ \\epsilon_{rel} $. Convergence if $ ||r_k|| / ||b|| < \\epsilon_{rel} $.
        maxiter (int):
            Maximum number of iterations.
        precond_apply (Optional[Callable[[np.ndarray], np.ndarray]]):
            Function performing the preconditioning step $ r \\mapsto M^{-1}r $.

    Returns:
        SolverResult:
            Named tuple containing the solution $ x_k $, convergence status,
            iteration count, and final residual norm $ ||r_k|| $.
    """
    
    x           = x0.copy()             # Work on a copy
    r           = b - matvec(x)         # Get the residual of the initial guess 
    res_norm_sq = np.real(np.vdot(r, r))  # Use vdot for proper complex norm
    norm_b_sq   = np.real(np.vdot(b, b))  # Use vdot for proper complex norm
    
    # Use absolute tolerance if b is zero, otherwise relative
    tol_crit_sq = (tol**2) * norm_b_sq if norm_b_sq > 1e-15 else tol**2
    iterations  = 0
    
    # this is already a result
    if res_norm_sq < tol_crit_sq:
        return SolverResult(x=x, converged=True, iterations=iterations, residual_norm=np.sqrt(res_norm_sq))
    
    if precond_apply:
        z       = precond_apply(r)      # z_0 = M^{-1} r_0
        if not isinstance(z, np.ndarray) or z.shape != r.shape:
            raise SolverError(SolverErrorMsg.PRECOND_INVALID,
                f"Preconditioner output invalid shape/type: {z.shape} / {type(z)}")
    else:
        z       = r                     # z_0 = r_0
    p           = z.copy()              # p_0 = z_0
    rs_old      = np.vdot(r, z)          # rho_k = <r_k, z_k> (initially rho_0)

    for i in range(maxiter):
        iterations  = i + 1             # check the iterations of the algorithm
        ap          = matvec(p)         # v_k = A p_k
        alpha_denom = np.vdot(p, ap)    # <p_k, v_k>

        if np.abs(alpha_denom) <= 1e-15:
            print(f"Warning (CG NumPy): Potential breakdown/non-SPD at iter {iterations}. Denom={alpha_denom:.4e}")
            res_norm_sq = np.real(np.vdot(r, r))
            converged   = res_norm_sq < tol_crit_sq
            return SolverResult(x=x, converged=converged, iterations=iterations, residual_norm=np.sqrt(res_norm_sq))

        alpha       = rs_old / alpha_denom  # alpha_k
        x          += alpha * p             # x_{k+1}
        r          -= alpha * ap            # r_{k+1}
        res_norm_sq = np.real(np.vdot(r, r))

        if res_norm_sq < tol_crit_sq:
            return SolverResult(x=x, converged=True, iterations=iterations, residual_norm=np.sqrt(res_norm_sq))

        if precond_apply:
            z       = precond_apply(r)      # z_{k+1} = M^{-1} r_{k+1}
            if not isinstance(z, np.ndarray) or z.shape != r.shape:
                raise SolverError(SolverErrorMsg.PRECOND_INVALID,
                    f"Preconditioner output invalid shape/type: {z.shape} / {type(z)}")
        else:
            z       = r                     # z_{k+1} = r_{k+1}
        rs_new      = np.vdot(r, z)         # rho_{k+1} = <r_{k+1}, z_{k+1}>

        if np.abs(rs_old) < 1e-15:
            print(f"Warning (CG NumPy): rs_old near zero at iter {iterations}, potential breakdown.")
            converged = res_norm_sq < tol_crit_sq
            return SolverResult(x=x, converged=converged, iterations=iterations, residual_norm=np.sqrt(res_norm_sq))

        beta        = rs_new / rs_old   # beta_k
        p           = z + beta * p      # p_{k+1}
        rs_old      = rs_new            # Update rho_k for next iteration

    # Max iterations reached
    final_residual_norm = np.sqrt(res_norm_sq)
    converged           = res_norm_sq < tol_crit_sq
    if not converged:
        print(f"Warning (CG NumPy): Did not converge within {maxiter} iterations.")
    return SolverResult(x=x, converged=converged, iterations=iterations, residual_norm=final_residual_norm)

_cg_logic_numpy_numba_no_precond_impl   = None
_cg_logic_numpy_numba_precond_impl      = None

#! NUMBA CG Logic
if _NUMBA_AVAILABLE:
    
    @numba.njit(cache=True, error_model='numpy', fastmath=True)
    def _cg_logic_numpy_numba_no_precond_impl(
            matvec_nb       : Callable[[np.ndarray], np.ndarray],
            b_nb            : np.ndarray,
            x0_nb           : np.ndarray,
            tol_nb          : float,
            maxiter_nb      : int) -> Tuple[np.ndarray, bool, int, float]:
        """
        Numba-compiled CG core (no preconditioner). 
        
        Parameters:
            matvec_nb Callable[[np.ndarray], np.ndarray]:
                Function performing the matrix-vector product $ v \\mapsto Av $.
            b_nb np.ndarray:
                Right-hand side vector $ b $.
            x0_nb np.ndarray:
                Initial guess $ x_0 $.
            tol_nb float:
                Relative tolerance $ \\epsilon_{rel} $. Convergence if $ ||r_k|| / ||b|| < \\epsilon_{rel} $.
            maxiter_nb int:
                Maximum number of iterations.
        Returns: 
            Tuple[np.ndarray, bool, int, float]:
                SolverResult:
                    Named tuple containing the solution $ x_k $, convergence status,
                    iteration count, and final residual norm $ ||r_k|| $.       
        Note:
            See _cg_logic_numpy for math.
        """
        
        x           = x0_nb.copy()
        r           = b_nb - matvec_nb(x)
        p           = r.copy()
        rs_old      = np.vdot(r, r)
        norm_b_sq   = np.real(np.vdot(b_nb, b_nb))
        tol_crit_sq = (tol_nb**2) * norm_b_sq if norm_b_sq > 0 else tol_nb**2
        iterations  = 0

        def fallback(r, tol_crit_sq):
            """ Fallback for when rs_old is near zero. """
            final_res_norm_sq   = np.real(np.vdot(r, r))
            converged           = final_res_norm_sq < tol_crit_sq
            return final_res_norm_sq, converged
        
        # check the convergence already
        rs_old_real = np.real(rs_old)
        if rs_old_real < tol_crit_sq:
            return x, True, 0, np.sqrt(rs_old_real)

        # iterate over the maximum number of iterations
        # Note: Numba does not support Python's `break` statement in loops
        # so we use a fallback function to handle the case where rs_old is near zero
        # and we cannot compute the denominator for alpha.
        for i in range(maxiter_nb):
            iterations  = i + 1
            Ap          = matvec_nb(p)
            alpha_denom = np.vdot(p, Ap)
            
            # Check for potential breakdown (denominator near zero)
            # This is a fallback for when rs_old is near zero
            # and we cannot compute the denominator for alpha.
            if np.abs(alpha_denom) < 1e-15:
                rr, converged = fallback(r, tol_crit_sq)
                return x, converged, iterations, np.sqrt(rr)
            
            alpha       = rs_old / alpha_denom
            x          += alpha * p
            r          -= alpha * Ap
            rs_new      = np.vdot(r, r)
            rs_new_real = np.real(rs_new)
            
            # Check for convergence
            if rs_new_real < tol_crit_sq:
                return x, True, iterations, np.sqrt(rs_new_real)
            
            beta        = rs_new / rs_old
            p           = r + beta * p
            rs_old      = rs_new
            
            # Check for potential breakdown (rs_old near zero)
            # This is a fallback for when rs_old is near zero
            # and we cannot compute the denominator for alpha.
            if np.abs(rs_old) < 1e-15:
                return x, rs_new_real < tol_crit_sq, iterations, np.sqrt(rs_new_real)

        final_res_norm_sq, converged = fallback(r, tol_crit_sq)
        return x, converged, iterations, np.sqrt(final_res_norm_sq)

    @numba.njit(cache=True, error_model='numpy', fastmath=True)
    def _cg_logic_numpy_numba_precond_impl(
            matvec_nb       : Callable[[np.ndarray], np.ndarray],
            b_nb            : np.ndarray,
            x0_nb           : np.ndarray,
            tol_nb          : float,
            maxiter_nb      : int,
            precond_apply_nb: Callable[[np.ndarray], np.ndarray] # Required
            ) -> Tuple[np.ndarray, bool, int, float]:
        """ 
        Numba-compiled CG core (WITH preconditioner).
        Parameters:
            matvec_nb Callable[[np.ndarray], np.ndarray]:
                Function performing the matrix-vector product $ v \\mapsto Av $.
            b_nb np.ndarray:
                Right-hand side vector $ b $.
            x0_nb np.ndarray:
                Initial guess $ x_0 $.
            tol_nb float:
                Relative tolerance $ \\epsilon_{rel} $. Convergence if $ ||r_k|| / ||b|| < \\epsilon_{rel} $.
            maxiter_nb int:
                Maximum number of iterations.
            precond_apply_nb Callable[[np.ndarray], np.ndarray]:
                Function performing the preconditioning step $ r \\mapsto M^{-1}r $.
        Returns:
            Tuple[np.ndarray, bool, int, float]:
                SolverResult:
                    Named tuple containing the solution $ x_k $, convergence status,
                    iteration count, and final residual norm $ ||r_k|| $.
        """
        
        # Initialize variables
        x           = x0_nb.copy()
        r           = b_nb - matvec_nb(x)
        z           = precond_apply_nb(r)
        p           = z.copy()
        rs_old      = np.vdot(r, z)
        norm_b_sq   = np.real(np.vdot(b_nb, b_nb))
        tol_crit_sq = (tol_nb**2) * norm_b_sq if norm_b_sq > 0 else tol_nb**2
        iterations  = 0
        res_norm_sq = np.real(np.vdot(r, r))

        if res_norm_sq < tol_crit_sq:
            return x, True, 0, np.sqrt(res_norm_sq)

        # iterate over the maximum number of iterations
        for i in range(maxiter_nb):
            iterations  = i + 1
            Ap          = matvec_nb(p)
            alpha_denom = np.vdot(p, Ap)
            
            # Check for potential breakdown (denominator near zero)
            if np.abs(alpha_denom) < 1e-15:
                break
            
            alpha       = rs_old / alpha_denom
            x          += alpha * p
            r          -= alpha * Ap
            res_norm_sq = np.real(np.vdot(r, r))

            # Check for convergence
            if res_norm_sq < tol_crit_sq:
                return x, True, iterations, np.sqrt(res_norm_sq)

            z           = precond_apply_nb(r)
            rs_new      = np.vdot(r, z)

            # Check for potential breakdown (rs_old near zero)
            if np.abs(rs_old) < 1e-15:
                break
            
            beta        = rs_new / rs_old
            p           = z + beta * p
            rs_old      = rs_new

        final_res_norm_sq   = np.real(np.vdot(r,r))
        converged           = final_res_norm_sq < tol_crit_sq
        return x, converged, iterations, np.sqrt(final_res_norm_sq)

    # Wrapper that calls the appropriate Numba function
    def _cg_logic_numpy_compiled_wrapper(matvec,
                                        b,
                                        x0,
                                        tol             : float,
                                        maxiter         : int,
                                        precond_apply   : Callable[[np.ndarray], np.ndarray] = None):
        """ 
        Wrapper for Numba CG: Calls no_precond or precond version. 
        Parameters:
            matvec Callable[[np.ndarray], np.ndarray]:
                Function performing the matrix-vector product $ v \\mapsto Av $.
            b np.ndarray:
                Right-hand side vector $ b $.
            x0 np.ndarray:
                Initial guess $ x_0 $.
            tol float:
                Relative tolerance $ \\epsilon_{rel} $. Convergence if $ ||r_k|| / ||b|| < \\epsilon_{rel} $.
            maxiter int:
                Maximum number of iterations.
            precond_apply Callable[[np.ndarray], np.ndarray]:
                Function performing the preconditioning step $ r \\mapsto M^{-1}r $.
        """
        # Check if matvec is a callable (function) - if so, use plain Python version
        if callable(matvec):
            # Matrix-free mode with Python function - can't use Numba
            return _cg_logic_numpy(matvec, b, x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply)
        
        # Otherwise use Numba-compiled version
        dtype   = b.dtype
        b_nb    = b.astype(dtype, copy=False)
        x0_nb   = x0.astype(dtype, copy=False)

        if precond_apply is not None:
            # Try calling the preconditioned Numba version
            # Note: This assumes precond_apply itself is Numba-compatible!
            # If it fails, it will raise an exception. Consider adding try-except here
            # to fallback to plain numpy if Numba compilation fails for the combination.
            # print("Attempting Numba CG with preconditioner...")
            x, conv, it, res = _cg_logic_numpy_numba_precond_impl(
                matvec, b_nb, x0_nb, float(tol), int(maxiter), precond_apply)
        else:
            # print("Running Numba CG without preconditioner...")
            x, conv, it, res = _cg_logic_numpy_numba_no_precond_impl(
                matvec, b_nb, x0_nb, float(tol), int(maxiter))

        return SolverResult(x=x, converged=conv, iterations=it, residual_norm=res)
    
    # Compile the Numba functions
    _cg_logic_numpy_compiled = _cg_logic_numpy_compiled_wrapper
else:
    _cg_logic_numpy_compiled = _cg_logic_numpy

#! JAX CG Logic
_cg_logic_jax_compiled                  = None

if JAX_AVAILABLE:
    
    # Define the core JAX logic function (can include preconditioning logic)
    def _cg_logic_jax_core(
            matvec          : MatVecFunc,
            b               : jnp.ndarray,
            x0              : jnp.ndarray,
            tol             : float,
            maxiter         : int,
            precond_apply   : Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None) -> SolverResult:
        """
        Core CG algorithm implementation using JAX. See _cg_logic_numpy for math.
        
        Args:
            matvec (MatVecFunc):
                Function performing the matrix-vector product $ v \\mapsto Av $.
            b (jnp.ndarray):
                Right-hand side vector $ b $.
            x0 (jnp.ndarray):
                Initial guess $ x_0 $. 
            tol (float):
                Relative tolerance $ \\epsilon_{rel} $. Convergence if $ ||r_k|| / ||b|| < \\epsilon_{rel} $.
            maxiter (int):
                Maximum number of iterations.
            precond_apply (Optional[Callable[[jnp.ndarray], jnp.ndarray]]):
                Function performing the preconditioning step $ r \\mapsto M^{-1}r $.
        Returns:
            SolverResult:
                Named tuple containing the solution $ x_k $, convergence status,
                iteration count, and final residual norm $ ||r_k|| $.
        """
        
        # Ensure precond_apply is a function or identity
        precond_fn  = precond_apply if precond_apply is not None else lambda x: x
        
        # Initial state
        r0          = b - matvec(x0)
        z0          = precond_fn(r0)
        p0          = z0
        # Use vdot for Hermitian inner product <r, z>
        rho0        = jnp.real(jnp.vdot(r0, z0))

        # Norms for convergence check
        norm_b      = jnp.linalg.norm(b)
        tol_val     = tol * jnp.where(norm_b == 0, 1.0, norm_b)
        
        # Loop State: (iteration, x, r, p, rho, residual_norm)
        # We compute initial residual norm
        res_norm0   = jnp.linalg.norm(r0)
        init_val    = (0, x0, r0, p0, rho0, res_norm0)

        def cond_fun(state):
            """ 
            Loop condition: iter < maxiter and ||r||^2 >= tol_crit^2
            
            Allows for early exit if the residual norm is already below the tolerance.
            Requires the state to be unpacked.
            """
            i, _, _, _, _, res_norm = state
            return (i < maxiter) & (res_norm > tol_val)

        def body_fun(val):
            """ 
            One CG iteration (see math in _cg_logic_numpy). 
            Runs the CG algorithm body logic.
            Note: 
                The preconditioner is applied conditionally using lax.cond.
            """
            i, x, r, p, rho, _  = val
            
            # Matrix-vector product (Expensive step)
            Ap                  = matvec(p)
            
            # Alpha calculation
            # <p, Ap> must be real for Hermitian A, but take real to be safe/stable
            pAp                 = jnp.real(jnp.vdot(p, Ap))
            
            # Protect against division by zero (breakdown)
            alpha               = rho / jnp.where(pAp == 0, 1.0, pAp)
            
            # Updates
            x_new               = x + alpha * p
            r_new               = r - alpha * Ap
            
            # Preconditioning
            z_new               = precond_fn(r_new)
            
            # Beta calculation
            rho_new             = jnp.real(jnp.vdot(r_new, z_new))
            beta                = rho_new / jnp.where(rho == 0, 1.0, rho)
            
            # Search direction update
            p_new               = z_new + beta * p
            
            # Residual norm for convergence check
            res_norm_new        = jnp.linalg.norm(r_new)
            
            return (i + 1, x_new, r_new, p_new, rho_new, res_norm_new)

        final_val = lax.while_loop(cond_fun, body_fun, init_val)
        iter_final, x_final, _, _, _, res_norm_final = final_val
        converged = res_norm_final <= tol_val
        
        return SolverResult(
            x               =   x_final, 
            converged       =   converged, 
            iterations      =   iter_final, 
            residual_norm   =   res_norm_final
        )

    # Compile once. 
    # 0: matvec (function), 5: precond_apply (function/None) -> Static
    _cg_logic_jax_compiled = jax.jit(_cg_logic_jax_core, static_argnums=(0, 5))
    
else:
    _cg_logic_jax_compiled = None

# -----------------------------------------------------------------------------
#! Concrete CgSolver Implementation
# -----------------------------------------------------------------------------

class CgSolver(Solver):
    '''
    Conjugate Gradient (CG) solver for symmetric positive definite linear systems.

    Implements the static `solve` and `get_solver_func` methods required by
    the `Solver` base class, dispatching to backend-specific implementations
    (NumPy/Numba or JAX JIT). Assumes the operator A (defined by `matvec`)
    is symmetric positive definite.
    '''
    _solver_type    = SolverType.CG
    _symmetric      = True

    # --------------------------------------------------
    #! Static Methods Override
    # --------------------------------------------------

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
        """
        Returns the backend-specific compiled/optimized CG function.

        Args:
            backend_module (Any):
                The numerical backend (`numpy` or `jax.numpy`).

        Returns:
            StaticSolverFunc:
                The core CG function for the backend.
        """
        
        if backend_module is jnp:
            if _cg_logic_jax_compiled is None:
                raise ImportError("JAX not installed but JAX backend requested.")
            func = _cg_logic_jax_compiled
            
        # NumPy Backend
        elif backend_module is np:
            # If we have a pure matvec function (not Fisher/Matrix data), we can try Numba
            # Note: Numba requires the matvec function itself to be jittable.
            # For simplicity/robustness, we stick to the pure python wrapper or specific logic
            # unless explicitly optimizing standard matrix cases.
            func = _cg_logic_numpy
            
            if _NUMBA_AVAILABLE and not (use_fisher or use_matrix):
                # If user provides a generic matvec, optimization is tricky 
                # because we can't jit arbitrary python callbacks easily in Numba 
                # without objmode.
                pass

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
            precond_apply   : Optional[Callable[[Array], Array]]    = None,
            backend_module  : Any                                   = np,
            sigma           : Optional[float]                       = None,
            **kwargs        : Any) -> SolverResult:
        """
        Static CG execution: Determines the appropriate backend function and executes it.

        Args:
            matvec          (MatVecFunc): 
                Function performing the matrix-vector product $ v \\mapsto Av $.
            b               (Array): 
                Right-hand side vector $ b $.
            x0              (Array): 
                Initial guess $ x_0 $.
            tol             (float): 
                Relative tolerance $ \\epsilon_{rel} $.
            maxiter         (int): 
                Maximum number of iterations.
            precond_apply   (Optional[Callable[[Array], Array]]): 
                Function performing the preconditioning step $ r \\mapsto M^{-1}r $.
            backend_module  (Any): 
                Backend module (`numpy` or `jax.numpy`).
            a               (Optional[Array]): 
                            Optional array for matrix solution...
            s               (Optional[Array]): 
                Optional array for Fisher-based matvec construction. If it 
                is provided alone, it is assumed to be matrix A already...
            s_p             (Optional[Array]): 
                Optional array for Fisher-based matvec construction.
            **kwargs        (Any): 
                Additional arguments (e.g., `sigma`).

        Returns:
            SolverResult: 
            A named tuple containing the solution, convergence status, iteration count, and residual norm.
        """
        try:
            # Instance-less call logic
            solver_func     = CgSolver.get_solver_func(
                backend_module, 
                use_matvec  = True, 
                sigma       = sigma
            )
            
            return Solver.run_solver_func(backend_module, solver_func, matvec=matvec, 
                b=b, x0=x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply)
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"CG execution failed: {e}") from e

# -----------------------------------------------------------------------------
#! Conjugate Gradient Solver Class Wrapper for SciPy
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    import jax.scipy.sparse.linalg as jax_sla
    
    def _static_jax_cg(matvec, b, x0, tol, maxiter, M):
        # JAX CG signature: (A, b, x0, tol, atol, maxiter, M)
        x, info = jax_sla.cg(matvec, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
        return x, info

class CgSolverScipy(Solver):
    '''
    Wrapper for SciPy's Conjugate Gradient solver (`scipy.sparse.linalg.cg` or jax.scipy.cg).
    '''
    _solver_type    = SolverType.SCIPY_CG
    _symmetric      = True

    # --------------------------------------------------
    #! Static Methods Override
    # --------------------------------------------------

    @staticmethod
    def get_solver_func(backend_module      : Any,
                        use_matvec          : bool = True,
                        use_fisher          : bool = False,
                        use_matrix          : bool = False,
                        sigma               : Optional[float] = None, **kwargs) -> StaticSolverFunc:
        """ Returns the backend-specific SciPy CG wrapper function. """
        
        if backend_module is jnp:

            def _wrapper(matvec, b, x0, tol, maxiter, precond_apply, **ignored_kwargs):
                x, info = _static_jax_cg(matvec, b, x0, tol, maxiter, precond_apply)
                # Helper to package result into your custom dataclass
                return SolverResult(x, True, maxiter, None) 
            func = _wrapper
            
        elif backend_module is np:
            
            def _scipy_wrapper(matvec, b, x0, tol, maxiter, precond_apply, **ignored_kwargs):
                # Wrap matvec in LinearOperator for SciPy
                n       = b.shape[0]
                A_op    = spsla.LinearOperator((n, n), matvec=matvec, dtype=b.dtype)
                M_op    = spsla.LinearOperator((n, n), matvec=precond_apply, dtype=b.dtype) if precond_apply else None
                x, info = spsla.cg(A_op, b, x0=x0, tol=tol, maxiter=maxiter, M=M_op)
                return SolverResult(x, info==0, maxiter, None)
            
            func = _scipy_wrapper
        
        else:
            raise ValueError(f"Unsupported backend: {backend_module}")

        return Solver._solver_wrap_compiled(backend_module, func, use_matvec, use_fisher, use_matrix, sigma)
    
    # --------------------------------------------------
    
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
            **kwargs        : Any) -> SolverResult:
        """
        Static CG execution: Determines the appropriate backend function and executes it.

        Args:
            matvec          (MatVecFunc): 
                Function performing the matrix-vector product $ v \\mapsto Av $.
            b               (Array): 
                Right-hand side vector $ b $.
            x0              (Array): 
                Initial guess $ x_0 $.
            tol             (float): 
                Relative tolerance $ \\epsilon_{rel} $.
            maxiter         (int): 
                Maximum number of iterations.
            precond_apply   (Optional[Callable[[Array], Array]]): 
                Function performing the preconditioning step $ r \\mapsto M^{-1}r $.
            backend_module  (Any): 
                Backend module (`numpy` or `jax.numpy`).
            a               (Optional[Array]): 
                            Optional array for matrix solution...
            s               (Optional[Array]): 
                Optional array for Fisher-based matvec construction. If it 
                is provided alone, it is assumed to be matrix A already...
            s_p             (Optional[Array]): 
                Optional array for Fisher-based matvec construction.
            **kwargs        (Any): 
                Additional arguments (e.g., `sigma`).

        Returns:
            SolverResult: 
            A named tuple containing the solution, convergence status, iteration count, and residual norm.
        """
        try:
            
            func = CgSolverScipy.get_solver_func(backend_module, use_matvec=True)
            return Solver.run_solver_func(backend_module, func, matvec=matvec, b=b, x0=x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply)
        
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"CG execution failed: {e}") from e

    # --------------------------------------------------

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------
