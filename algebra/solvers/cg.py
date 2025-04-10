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

from typing import Optional, Callable, Union, Any, Tuple, Type
import numpy as np

from general_python.algebra.solver import (Solver, SolverResult, SolverError, SolverErrorMsg,
    SolverType, Array, MatVecFunc, StaticSolverFunc)
from general_python.algebra.preconditioners import Preconditioner, PreconitionerApplyFun

# Import backend specifics and compilation tools
from general_python.algebra.utils import _JAX_AVAILABLE

if _JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        import jax.lax as lax
    except ImportError:
        pass
else:
    jax = None
    jnp = None
    lax = None

# --- 

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    _NUMBA_AVAILABLE = False

# Import SciPy sparse if needed by CgSolverScipy
import scipy.sparse.linalg as spsla
import scipy.linalg

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
    res_norm_sq = np.dot(r, r)
    norm_b_sq   = np.dot(b, b)
    
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
    rs_old      = np.dot(r, z)          # rho_k = r_k^T z_k (initially rho_0)

    for i in range(maxiter):
        iterations  = i + 1             # check the iterations of the algorithm
        ap          = matvec(p)         # v_k = A p_k
        alpha_denom = np.dot(p, ap)     # p_k^T v_k

        if alpha_denom <= 1e-15:
            print(f"Warning (CG NumPy): Potential breakdown/non-SPD at iter {iterations}. Denom={alpha_denom:.4e}")
            res_norm_sq = np.dot(r, r)
            converged   = res_norm_sq < tol_crit_sq
            return SolverResult(x=x, converged=converged, iterations=iterations, residual_norm=np.sqrt(res_norm_sq))

        alpha       = rs_old / alpha_denom  # alpha_k
        x          += alpha * p             # x_{k+1}
        r          -= alpha * ap            # r_{k+1}
        res_norm_sq = np.dot(r, r)

        if res_norm_sq < tol_crit_sq:
            return SolverResult(x=x, converged=True, iterations=iterations, residual_norm=np.sqrt(res_norm_sq))

        if precond_apply:
            z       = precond_apply(r)      # z_{k+1} = M^{-1} r_{k+1}
            if not isinstance(z, np.ndarray) or z.shape != r.shape:
                raise SolverError(SolverErrorMsg.PRECOND_INVALID,
                    f"Preconditioner output invalid shape/type: {z.shape} / {type(z)}")
        else:
            z       = r                     # z_{k+1} = r_{k+1}
        rs_new      = np.dot(r, z)          # rho_{k+1} = r_{k+1}^T z_{k+1}

        if abs(rs_old) < 1e-15:
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
        rs_old      = np.dot(r, r)
        norm_b_sq   = np.dot(b_nb, b_nb)
        tol_crit_sq = (tol_nb**2) * norm_b_sq if norm_b_sq > 0 else tol_nb**2
        iterations  = 0

        def fallback(r, tol_crit_sq):
            """ Fallback for when rs_old is near zero. """
            final_res_norm_sq   = np.dot(r, r)
            converged           = final_res_norm_sq < tol_crit_sq
            return final_res_norm_sq, converged
        
        # check the convergence already
        if rs_old < tol_crit_sq:
            return x, True, 0, np.sqrt(rs_old)

        # iterate over the maximum number of iterations
        # Note: Numba does not support Python's `break` statement in loops
        # so we use a fallback function to handle the case where rs_old is near zero
        # and we cannot compute the denominator for alpha.
        for i in range(maxiter_nb):
            iterations  = i + 1
            Ap          = matvec_nb(p)
            alpha_denom = np.dot(p, Ap)
            
            # Check for potential breakdown (denominator near zero)
            # This is a fallback for when rs_old is near zero
            # and we cannot compute the denominator for alpha.
            if np.abs(alpha_denom) < 1e-15:
                rr, converged = fallback(r, tol_crit_sq)
                return x, converged, iterations, np.sqrt(rr)
            
            alpha       = rs_old / alpha_denom
            x          += alpha * p
            r          -= alpha * Ap
            rs_new      = np.dot(r, r)
            
            # Check for convergence
            if rs_new < tol_crit_sq:
                return x, True, iterations, np.sqrt(rs_new)
            
            beta        = rs_new / rs_old
            p           = r + beta * p
            rs_old      = rs_new
            
            # Check for potential breakdown (rs_old near zero)
            # This is a fallback for when rs_old is near zero
            # and we cannot compute the denominator for alpha.
            if np.abs(rs_old) < 1e-15:
                return x, rs_new < tol_crit_sq, iterations, np.sqrt(rs_new)

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
        rs_old      = np.dot(r, z)
        norm_b_sq   = np.dot(b_nb, b_nb)
        tol_crit_sq = (tol_nb**2) * norm_b_sq if norm_b_sq > 0 else tol_nb**2
        iterations  = 0
        res_norm_sq = np.dot(r, r)

        if res_norm_sq < tol_crit_sq:
            return x, True, 0, np.sqrt(res_norm_sq)

        # iterate over the maximum number of iterations
        for i in range(maxiter_nb):
            iterations  = i + 1
            Ap          = matvec_nb(p)
            alpha_denom = np.dot(p, Ap)
            
            # Check for potential breakdown (denominator near zero)
            if np.abs(alpha_denom) < 1e-15:
                break
            
            alpha       = rs_old / alpha_denom
            x          += alpha * p
            r          -= alpha * Ap
            res_norm_sq = np.dot(r, r)

            # Check for convergence
            if res_norm_sq < tol_crit_sq:
                return x, True, iterations, np.sqrt(res_norm_sq)

            z           = precond_apply_nb(r)
            rs_new      = np.dot(r, z)

            # Check for potential breakdown (rs_old near zero)
            if np.abs(rs_old) < 1e-15:
                break
            
            beta        = rs_new / rs_old
            p           = z + beta * p
            rs_old      = rs_new

        final_res_norm_sq   = np.dot(r,r)
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
if _JAX_AVAILABLE:
    
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
        
        # Initialize variables
        x           = x0
        r           = b - matvec(x)
        norm_b_sq   = jnp.dot(b, b)
        tol_crit_sq = (tol**2) * jnp.where(norm_b_sq == 0.0, 1.0, norm_b_sq)

        # Use lax.cond for conditional preconditioning application
        # Define identity function for the 'else' branch
        def no_precond(vec):
            return vec
        precond_app = no_precond if precond_apply is None else precond_apply
        z           = precond_app(r)# Argument to the chosen function
        p           = z             # p_0 = z_0
        rs_old      = jnp.dot(r, z) # rho_0
        res_norm_sq = jnp.dot(r, r) # ||r_k||^2
        
        # Initial State: (iter, x, r, p, rho_k, res_norm_sq)
        initial_state = (0, x, r, p, rs_old, res_norm_sq)

        def cond_fun(state):
            """ 
            Loop condition: iter < maxiter and ||r||^2 >= tol_crit^2
            
            Allows for early exit if the residual norm is already below the tolerance.
            Requires the state to be unpacked.
            """
            i, _, _, _, _, current_res_norm_sq = state
            return jnp.logical_and(i < maxiter, current_res_norm_sq >= tol_crit_sq)

        def body_fun(state):
            """ 
            One CG iteration (see math in _cg_logic_numpy). 
            Runs the CG algorithm body logic.
            Note: 
                The preconditioner is applied conditionally using lax.cond.
            """
            i, x_i, r_i, p_i, rs_old_i, _ = state
            Ap                  = matvec(p_i)
            alpha_denom         = jnp.dot(p_i, Ap)
            safe_denom          = jnp.where(jnp.abs(alpha_denom) < 1e-15, 1.0, alpha_denom)
            alpha               = rs_old_i / safe_denom
            x_next              = x_i + alpha * p_i
            r_next              = r_i - alpha * Ap

            # Apply preconditioner conditionally inside body
            z_next              = precond_app(r_next) if precond_apply is not None else r_next
            rs_new              = jnp.dot(r_next, z_next)
            safe_rs_old         = jnp.where(jnp.abs(rs_old_i) < 1e-15, 1.0, rs_old_i)
            beta                = rs_new / safe_rs_old
            p_next              = z_next + beta * p_i
            res_norm_sq_next    = jnp.dot(r_next, r_next)
            return (i + 1, x_next, r_next, p_next, rs_new, res_norm_sq_next)

        # Run the JAX loop
        final_state = lax.while_loop(cond_fun, body_fun, initial_state)

        # Unpack final results
        iter_run, x_final, _, _, _, final_res_norm_sq = final_state
        conv        = final_res_norm_sq < tol_crit_sq
        final_res   = jnp.sqrt(final_res_norm_sq)
        return SolverResult(x=x_final, converged=conv, iterations=iter_run, residual_norm=final_res)

    # ---

    # JIT matvec and precond_apply statically if they dont change
    _cg_logic_jax_compiled = jax.jit(_cg_logic_jax_core, static_argnums=(0, 5))
    
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
    _solver_type = SolverType.CG

    def __init__(self, *args, **kwargs):
        '''
        Initializes...
        '''
        super().__init__(*args, **kwargs)

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
        
        func: Callable = None
        if backend_module is jnp:
            if _cg_logic_jax_compiled is None:
                print("Warning: Jax compiled function not available, returning Plain jax")
                func = _cg_logic_jax_core
            func = _cg_logic_jax_compiled
        elif backend_module is np:
            if _cg_logic_numpy_compiled is None:
                print("Warning: Numba CG function not available, returning plain Python version.")
                func = _cg_logic_numpy
            func = _cg_logic_numpy_compiled
        else:
            raise ValueError(f"Unsupported backend module for CG: {backend_module}")
        return Solver._solver_wrap_compiled(backend_module, func,
                                use_matvec, use_fisher, use_matrix, sigma)
        
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
            # Decide the appropriate solver function based on the backend and inputs
            solver_func = CgSolver.decide_solver_func(
                backend_module = backend_module,
                matvec         = matvec,
                a              = a,
                s              = s,
                s_p            = s_p,
                sigma          = kwargs.get('sigma', None)
            )
            # Execute the solver function
            return Solver.run_solver_func(  bck     = backend_module,
                                            func    = solver_func,
                                            a       = a,
                                            s       = s,
                                            s_p     = s_p,
                                            b       = b, 
                                            x0      = x0,
                                            tol     = tol,
                                            maxiter = maxiter,
                                            precond_apply = precond_apply)
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"CG execution failed: {e}") from e

# -----------------------------------------------------------------------------
#! Conjugate Gradient Solver Class Wrapper for SciPy
# -----------------------------------------------------------------------------

class CgSolverScipy(Solver):
    '''
    Wrapper for SciPy's Conjugate Gradient solver (`scipy.sparse.linalg.cg`).

    Uses instance configuration but overrides `solve_instance` to call SciPy.
    Does *not* implement the static `solve`/`get_solver_func` interface.
    Requires NumPy backend.
    '''
    _solver_type = SolverType.SCIPY_CG

    def __init__(self,
                backend         : str                             = 'numpy', # Force numpy
                dtype           : Optional[Type]                  = None,
                eps             : float                           = 1e-8,
                maxiter         : int                             = 1000,
                default_precond : Optional[Preconditioner]        = None,
                a               : Optional[Array]                 = None,
                s               : Optional[Array]                 = None,
                sp              : Optional[Array]                 = None,
                matvec_func     : Optional[MatVecFunc]            = None,
                sigma           : Optional[float]                 = None,
                is_gram         : bool                            = False):
        
        if backend != 'numpy': print(f"Warning: {self.__class__.__name__} uses SciPy, forcing backend to 'numpy'.")
        super().__init__(backend='numpy', dtype=dtype, eps=eps, maxiter=maxiter,
                        default_precond=default_precond, a=a, s=s, sp=sp,
                        matvec_func=matvec_func, sigma=sigma, is_gram=is_gram)
        self._symmetric = True

    # --------------------------------------------------
    #! Static Methods Override
    # --------------------------------------------------

    @staticmethod
    def get_solver_func(backend_module: Any) -> StaticSolverFunc:
        """
        Returns a wrapper function that calls the static `CgSolverScipy.solve`.

        Args:
            backend_module (Any): Must be `numpy`.

        Returns:
            StaticSolverFunc: A callable matching the required signature.
        """
        if backend_module is not np:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH,
                            f"{SolverType.SCIPY_CG.name} requires NumPy backend.")

        # Return a lambda that captures the static solve method
        return lambda matvec, b, x0, tol, maxiter, precond_apply, backend_mod, **kwargs: \
                    CgSolverScipy.solve(matvec=matvec, b=b, x0=x0, tol=tol, maxiter=maxiter,
                                          precond_apply=precond_apply, backend_module=backend_mod, **kwargs)

    @staticmethod
    def solve(
            # === Core Problem Definition ===
            matvec          : MatVecFunc,
            b               : Array,
            x0              : Array,
            # === Solver Parameters ===
            *,
            tol             : float,
            maxiter         : int,
            # === Optional Preconditioner ===
            precond_apply   : Optional[Callable[[Array], Array]] = None,
            # === Backend Specification ===
            backend_module  : Any,
            # === Solver Specific Arguments ===
            **kwargs        : Any # Pass SciPy specific args (atol, callback) here
            ) -> SolverResult:
        """
        Static solve implementation calling `scipy.sparse.linalg.cg`.

        Args: See base class `solve` docstring.

        Returns: SolverResult.
        """
        if backend_module is not np:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH,
                    f"{SolverType.SCIPY_CG.name} requires NumPy backend.")

        # Prepare Arguments for SciPy
        M_op    = None
        n_size  = b.shape[0]
        if precond_apply is not None:
            if not callable(precond_apply):
                raise SolverError(SolverErrorMsg.PRECOND_INVALID,
                        "precond_apply must be callable.")
            M_op = spsla.LinearOperator((n_size, n_size), matvec=precond_apply)

        # Ensure NumPy arrays (SciPy works best with NumPy arrays)
        # Assume matvec is already NumPy compatible if backend_module is np
        b_np    = np.asarray(b)
        x0_np   = np.asarray(x0) # x0 is guaranteed not None by signature? No, x0 is required now.
        # Add dimension checks? Assume caller provides correct shapes.
        if x0_np.shape != b_np.shape:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH,
                    f"Shape mismatch: b={b_np.shape}, x0={x0_np.shape}")


        # Setup Callback for Iteration Count
        iter_count = [0]
        def scipy_callback(xk):
            iter_count[0] += 1
        
        # User can override callback via kwargs
        kwargs.setdefault('callback', scipy_callback)
        # SciPy uses 'tol' (relative), ensure 'atol' is handled if needed
        kwargs.setdefault('atol', 'legacy')

        # Call SciPy CG
        try:
            print(f"({CgSolverScipy.__name__}) Calling static scipy.sparse.linalg.cg...")
            x_sol, exit_code = spsla.cg(
                                        matvec,     # Pass the Python callable
                                        b_np,       # RHS vector
                                        x0=x0_np,   # Initial guess
                                        tol=tol,
                                        maxiter=maxiter,
                                        M=M_op,
                                        **kwargs    # Pass atol, callback etc.
                                        )

            converged       = (exit_code == 0)
            iterations_run  = iter_count[0]
            # Calculate final residual norm if needed
            final_residual  = b_np - matvec(x_sol)
            final_res_norm  = np.linalg.norm(final_residual)
            return SolverResult(x=x_sol, converged=converged, iterations=iterations_run, residual_norm=final_res_norm)
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED,
                    f"Static SciPy CG call failed: {e}") from e

    # --------------------------------------------------
    #! Instance Methods Override
    # --------------------------------------------------
    
    def solve_instance(self,
                    b               : Array,
                    x0              : Optional[Array]   = None,
                    *,
                    tol             : Optional[float]   = None,
                    maxiter         : Optional[int]     = None,
                    precond         : Union[Preconditioner, Callable[[Array], Array], None] = 'default',
                    sigma           : Optional[float]   = None,
                    **kwargs) -> SolverResult:
        """
        Overrides base method to solve $ Ax = b $ using `scipy.sparse.linalg.cg`.
        Uses instance configuration to set up defaults and call static solve.

        Args: See previous docstring.

        Returns: SolverResult.
        """
        if self._backend is not np:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH, f"{self.__class__.__name__} requires NumPy backend.")

        current_tol                     = tol if tol is not None else self._default_eps
        current_maxiter                 = maxiter if maxiter is not None else self._default_maxiter
        current_sigma                   = sigma if sigma is not None else self._conf_sigma

        # Get matvec func (compile=False)
        matvec_func                     = self._check_matvec_solve(current_sigma, np, compile_matvec=False, **kwargs)
        # Get precond apply func
        precond_apply_func              = self._check_precond_solve(precond)

        # Prepare b and x0 (handle optional x0)
        b_np                            = np.asarray(b, dtype=self._dtype)
        x0_np                           = np.zeros_like(b_np) if x0 is None else np.asarray(x0, dtype=self._dtype)
        if x0_np.shape != b_np.shape:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH, f"Shape mismatch: b={b_np.shape}, x0={x0_np.shape}")

        # Call the static solve method of this class
        print(f"({self.__class__.__name__}) Calling static solve via instance method...")
        result = CgSolverScipy.solve(
            matvec          = matvec_func,
            b               = b_np,
            x0              = x0_np,
            tol             = current_tol,
            maxiter         = current_maxiter,
            precond_apply   = precond_apply_func,
            backend_module  = np, # Explicitly numpy
            **kwargs
        )

        # Store results
        self._last_solution             = result.x
        self._last_converged            = result.converged
        self._last_iterations           = result.iterations
        self._last_residual_norm        = result.residual_norm

        print(f"({self.__class__.__name__}) Instance solve finished. Converged: {result.converged}, Iter: {result.iterations}, ResNorm: {result.residual_norm:.4e}")
        return result

    # --------------------------------------------------

# -----------------------------------------------------------------------------