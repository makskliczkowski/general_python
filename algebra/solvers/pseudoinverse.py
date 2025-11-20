r'''
file:       general_python/algebra/solvers/pseudo_inverse.py
author:     Maksymilian Kliczkowski

Implements a solver for linear systems $ Ax = b $ using the Moore-Penrose
pseudoinverse $ A^+ $. The solution obtained is the minimum norm solution
among all vectors $ x $ that minimize the least-squares residual $ ||Ax - b||_2 $.

Mathematical Formulation:
------------------------
The Moore-Penrose pseudoinverse $ A^+ $ of a matrix $ A $ is unique and satisfies:
1. $ A A^+ A = A $
2. $ A^+ A A^+ = A^+ $
3. $ (A A^+)^* = A A^+ $ (conjugate transpose)
4. $ (A^+ A)^* = A^+ A $

The minimum norm least-squares solution to $ Ax = b $ is given by:
$ x = A^+ b $

If a regularization (or shift) $ \sigma $ is applied, the solver computes:
$ x = (A + \sigma I)^+ b $
where $ I $ is the identity matrix. This can help with ill-conditioned matrices.

Computationally, $ A^+ $ is typically found via the Singular Value Decomposition (SVD).
If $ A = U \Sigma V^H $ is the SVD of A, then $ A^+ = V \Sigma^+ U^H $, where
$ \Sigma^+ $ is obtained by taking the reciprocal of the non-zero singular values
in $ \Sigma $ and transposing.

References:
-----------
    - Penrose, R. (1955). A generalized inverse for matrices. Proc. Cambridge Philos. Soc.
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). JHU Press. Chapter 5.
'''

from typing import Optional, Callable, Union, Any, Type, Tuple
from functools import partial
import numpy as np
import inspect

# Assuming these are correctly imported from the solver module
from ..solver import (
    Solver, SolverResult, SolverError, SolverErrorMsg,
    SolverType, Array, MatVecFunc, StaticSolverFunc
)
# Preconditioner import might not be strictly needed here
from ..preconditioners import Preconditioner

# Backend/Compilation tools
from ..utils import JAX_AVAILABLE, get_backend
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = np

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    _NUMBA_AVAILABLE = False


# -----------------------------------------------------------------------------
#! Core Logic for Pseudo-Inverse Solve
# -----------------------------------------------------------------------------

def _pinv_solve_logic(
    A               : Array,
    b               : Array,
    sigma           : Optional[float],
    pinv_rtol       : Optional[float],              # rcond for pinv
    backend_module  : Any) -> Tuple[Array, float]:  # Returns solution and residual norm
    """
    Core logic for solving Ax=b using pinv.
    
    Parameters:
        A (Array):
            Coefficient matrix.
        b (Array):
            Right-hand side vector.
        sigma (Optional[float]):
            Regularization term.
        pinv_rtol (Optional[float]):
            Tolerance for pinv.
        backend_module (Any):
            Backend module (numpy or jax.numpy).
    Returns:
        Tuple[Array, float]:
            Solution vector and final residual norm.
    Raises:
        SolverError:
            If matrix is singular or other errors occur.
    """
    
    try:
        if sigma is not None and sigma != 0.0:
            a_eff = A + sigma * backend_module.eye(A.shape[0], dtype=A.dtype)
        else:
            a_eff = A

        #! Compute Pseudo-inverse and Solution
        pinv_kwargs = {}
        if pinv_rtol is not None:
            # NumPy/SciPy use rcond, JAX uses rtol. Map to commnon language.
            if hasattr(backend_module.linalg, 'pinv'):
                sig = inspect.signature(backend_module.linalg.pinv)
                if 'rcond' in sig.parameters:
                    pinv_kwargs['rcond'] = pinv_rtol
                elif 'rtol' in sig.parameters:
                    pinv_kwargs['rtol'] = pinv_rtol

        print(f"({PseudoInverseSolver.__name__}) Calling {backend_module.__name__}.linalg.pinv...")
        a_pinv          = backend_module.linalg.pinv(a_eff, **pinv_kwargs)
        x_sol           = backend_module.dot(a_pinv, b)
        final_residual  = b - backend_module.dot(a_eff, x_sol)
        final_res_norm  = backend_module.linalg.norm(final_residual)
        return x_sol, final_res_norm

    except np.linalg.LinAlgError as e:
        raise SolverError(SolverErrorMsg.MAT_SINGULAR, f"Pseudo-inverse failed (LinAlgError): {e}") from e
    except Exception as e:
        # Catch other backend-specific linalg errors
        if "LinAlgError" in str(type(e)) or "LAPACK" in str(e): # Basic check
            raise SolverError(SolverErrorMsg.MAT_SINGULAR, f"Pseudo-inverse failed (LinAlgError): {e}") from e
        raise SolverError(SolverErrorMsg.CONV_FAILED, f"Pseudo-inverse failed: {e}") from e

_pinv_solve_logic_numba_compiled = None
if _NUMBA_AVAILABLE:

    # @numba.njit # This WILL likely fail or force object mode
    def _pinv_solve_logic_numba_wrapper(A, b, sigma, pinv_rtol, backend_mod):
        return _pinv_solve_logic(A, b, sigma, pinv_rtol, np)
    _pinv_solve_logic_numba_compiled = _pinv_solve_logic_numba_wrapper

_pinv_solve_logic_jax_compiled = None
if JAX_AVAILABLE:
    
    import jax
    from jax import lax
    
    # JAX pinv IS jittable
    # We need to handle optional sigma and rtol correctly within JIT
    @jax.jit
    def _pinv_solve_logic_jax_core(A, b, sigma, pinv_rtol):
        """ JIT-compiled core logic for JAX pinv solve. """
        # Conditional effective matrix based on sigma
        a_eff = lax.cond(sigma is not None and sigma != 0.0,
            lambda A_, s_: A_ + s_ * jnp.eye(A_.shape[0], dtype=A_.dtype),
            lambda A_, s_: A_,
            A, sigma)
        
        # Conditional rtol for pinv
        # Note: jax.linalg.pinv uses rtol, default might be None or float
        pinv_kwargs = {}
        if pinv_rtol is not None:           # This check might happen outside JIT trace
            pinv_kwargs['rtol'] = pinv_rtol # Pass if provided

        a_pinv                      = jnp.linalg.pinv(a_eff, **pinv_kwargs)
        x_sol                       = jnp.dot(a_pinv, b)
        final_residual              = b - jnp.dot(a_eff, x_sol)
        final_res_norm              = jnp.linalg.norm(final_residual)
        return x_sol, final_res_norm

    _pinv_solve_logic_jax_compiled  = _pinv_solve_logic_jax_core

# -----------------------------------------------------------------------------
#! Pseudo-inverse Solver Class
# -----------------------------------------------------------------------------

class PseudoInverseSolver(Solver):
    r'''
    Solves $ Ax = b $ using the Moore-Penrose pseudoinverse $ x = A^+ b $.
    Provides the minimum norm least-squares solution.
    Supports regularization $ x = (A + \sigma I)^+ b $.
    '''
    _solver_type    = SolverType.PSEUDO_INVERSE

    def __init__(self,
                backend         : str                             = 'default',
                dtype           : Optional[Type]                  = None,
                eps             : float                           = 0,    # Tolerance for pinv (rcond/rtol)
                maxiter         : int                             = 1,    # Not iterative
                default_precond : Optional[Preconditioner]        = None, # Not used
                a               : Optional[Array]                 = None, # Matrix A needed
                s               : Optional[Array]                 = None, # Not used directly
                s_p             : Optional[Array]                 = None, # Not used directly
                matvec_func     : Optional[MatVecFunc]            = None, # Not used
                sigma           : Optional[float]                 = None, # Regularization
                is_gram         : bool                            = False # To form A
                ):
        # Store pinv tolerance in self._default_eps for consistency
        super().__init__(backend=backend, dtype=dtype, eps=eps, maxiter=maxiter,
                        default_precond=default_precond, a=a, s=s, s_p=s_p,
                        matvec_func=matvec_func, sigma=sigma, is_gram=is_gram)
        self._symmetric = False # Works for non-symmetric

    # --------------------------------------------------
    #! Instance Helpers
    # --------------------------------------------------

    def _form_gram_matrix(self) -> Array:
        """
        Forms the Gram matrix A = (Sp @ S) / N if the configuration is set for Gram matrix computation.
        Returns:
            Array: The computed Gram matrix.
        Raises:
            SolverError: If the required components for Gram matrix computation are not set.
        """
        if self._conf_s is not None and self._conf_sp is not None:
            print(f"({self.__class__.__name__}) Forming Gram matrix for pseudo-inverse.")
            n_size = self._conf_s.shape[0]
            norm_factor = float(n_size) if n_size > 0 else 1.0
            return (self._conf_sp @ self._conf_s) / norm_factor
        else:
            raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Required components for Gram matrix computation are not set.")

    # --------------------------------------------------
    #! Static Methods Implementation
    # --------------------------------------------------

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
        """
        Returns a backend-adapted function compatible with the common solver wrapper.

        The returned function conforms to the StaticSolverFunc signature used by
        Solver._solver_wrap_compiled and internally calls PseudoInverseSolver.solve,
        forwarding the matrix as kwarg 'A' (accepting either 'a' or 'A').

        Note: For NumPy, a Numba-compiled core may be used; for JAX, JIT is supported.
        """
        def func(matvec, b, x0, tol, maxiter, precond_apply, **kwargs):
            A_kw   = kwargs.get('A', kwargs.get('a', None))
            return PseudoInverseSolver.solve(
                matvec          = matvec,
                b               = b,
                x0              = x0,
                tol             = 0.0 if tol is None else tol,
                maxiter         = 1,
                precond_apply   = None,
                backend_module  = backend_module,
                A               = A_kw,
                sigma           = sigma
            )

        return Solver._solver_wrap_compiled(backend_module, func,
                                            use_matvec, use_fisher, use_matrix, sigma)

    # --------------------------------------------------

    @staticmethod
    def solve(
            matvec          : MatVecFunc,           # Ignored
            b               : Array,
            x0              : Array,                # Ignored
            *,
            tol             : float,                # Used as rcond/rtol for pinv
            maxiter         : int,                  # Ignored
            precond_apply   : Optional[Callable[[Array], Array]] = None, # Ignored
            backend_module  : Any,
            A               : Optional[Array] = None, # REQUIRED
            sigma           : Optional[float] = None, # Optional
            **kwargs        : Any
            ) -> SolverResult:
        """
        Static solve implementation using the backend's `linalg.pinv`.

        Args:
            matvec, x0, maxiter, precond_apply: Ignored.
            b (Array): RHS vector.
            tol (float): Tolerance (rcond/rtol) for `linalg.pinv`.
            backend_module (Any): Backend (numpy or jax.numpy).
            A (Array): **Required** kwarg. Matrix $ A $.
            sigma (float, optional): **Optional** kwarg. Regularization $ \\sigma $.
            **kwargs: Other ignored arguments.

        Returns:
            SolverResult: Contains the minimum norm least-squares solution.
        """
        if A is None:
            raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix 'A' required via kwargs.")
        if backend_module is None:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH, "Backend module required.")

        # Get the appropriate core logic function (compiled if possible)
        if backend_module is jnp:
            if _pinv_solve_logic_jax_compiled is None:
                raise ImportError("JAX pinv func error.")
            core_func = _pinv_solve_logic_jax_compiled
            pinv_rtol = tol # JAX uses rtol
        elif backend_module is np:
            core_func = _pinv_solve_logic # Use plain python logic directly for clarity
            pinv_rtol = tol # NumPy uses rcond, pass tol here
            if _pinv_solve_logic_numba_compiled:
                # print("Note: Using Numba wrapper for pinv (may run in object mode).")
                core_func = _pinv_solve_logic_numba_compiled
        else:
            raise ValueError(f"Unsupported backend module for PseudoInverseSolver: {backend_module}.")

        # Call the core logic
        x_sol, final_res_norm = core_func(
            A=backend_module.asarray(A), # Ensure backend array
            b=backend_module.asarray(b),
            sigma=sigma,
            pinv_rtol=pinv_rtol,
            # Pass backend_module if the core func needs it (Numba wrapper might)
            **({'backend_mod': backend_module} if backend_module is np else {})
        )

        # pinv conceptually "converges" in 1 step if successful
        return SolverResult(x=x_sol, converged=True, iterations=1, residual_norm=final_res_norm)

    # --------------------------------------------------
    #! Instance Methods Override
    # --------------------------------------------------

    def solve_instance(self,
                    b               : Array,
                    x0              : Optional[Array]   = None, # Ignored
                    *,
                    tol             : Optional[float]   = None, # Used as pinv tolerance
                    maxiter         : Optional[int]     = None, # Ignored
                    precond         : Union[Preconditioner, Callable[[Array], Array], None] = None, # Ignored
                    sigma           : Optional[float]   = None,
                    **kwargs) -> SolverResult:
        r"""
        Instance: Solves $ x = (A + \sigma I)^+ b $ using `linalg.pinv`.

        Uses matrix `A` and default `sigma` from init, overridden by args.
        `tol` argument overrides instance `eps` and is passed as `rtol`/`rcond` to `pinv`.
        Ignores `x0`, `maxiter`, `precond`.

        Args:
            b (Array):
                RHS vector $ b $.
            x0, maxiter, precond: Ignored.
            tol (Optional[float]):
                Tolerance for pinv, overrides instance `eps`.
            sigma (Optional[float]):
                Overrides instance default regularization $ \\sigma $.
            **kwargs: Passed to static `solve` (mostly for 'A').

        Returns:
            SolverResult
        """
        # 1. Determine Matrix A
        matrix_a = self._conf_a if self._conf_a is not None else kwargs.get('A', None)
        if matrix_a is None:
            if self._conf_is_gram:
                matrix_a = self._form_gram_matrix()
                matrix_a = self._backend.asarray(matrix_a)
            else:
                raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix A not configured or passed via kwargs.")

        # 2. Determine Sigma and Tolerance
        current_sigma                   = sigma if sigma is not None else self._conf_sigma
        current_tol                     = tol if tol is not None else self._default_eps # Use eps as pinv tol

        # 3. Call Static Solve
        result = PseudoInverseSolver.solve(
            matvec          = None,     # Ignored
            b               = b,
            x0              = None,     # Ignored
            tol             = current_tol, # Pass as pinv tolerance
            maxiter         = 1,        # Ignored
            precond_apply   = None,     # Ignored
            backend_module  = self._backend,
            A               = matrix_a,
            sigma           = current_sigma,
            **kwargs
        )

        # Store results
        self._last_solution             = result.x
        self._last_converged            = result.converged
        self._last_iterations           = result.iterations
        self._last_residual_norm        = result.residual_norm
        print(f"({self.__class__.__name__}) Instance solve finished.")
        return result

# -----------------------------------------------------------------------------
#! Example Tests
# -----------------------------------------------------------------------------
def test_pseudo_inverse_solver():
    print("--- Running PseudoInverseSolver Tests ---")

    # --- Test Data ---
    # Well-conditioned matrix
    A_well = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_vec = np.array([1.0, 2.0])
    x_exact_well = np.linalg.solve(A_well, b_vec) # Exact solution

    # Singular matrix (rank deficient)
    A_sing = np.array([[1.0, 1.0], [1.0, 1.0]])
    # b that has solution in column space: [c, c]^T
    b_in_col_space = np.array([2.0, 2.0])
    # b that is NOT in column space
    b_not_in_col_space = np.array([1.0, 2.0])
    # Min norm solution for Ax=b_in_col_space is [1.0, 1.0]^T
    x_min_norm_sol = np.array([1.0, 1.0])
    # Min norm least squares solution for Ax=b_not_in_col_space
    x_min_norm_ls_sol = np.linalg.pinv(A_sing) @ b_not_in_col_space # Calculate expected

    # Tolerance for checks
    test_tol = 1e-6

    # --- NumPy Test ---
    print("\n--- NumPy Backend Test ---")
    try:
        solver_np = PseudoInverseSolver(backend='numpy', a=A_well)
        result_np_well = solver_np.solve_instance(b_vec, tol=1e-14)
        print("NumPy Well-conditioned:")
        print(f"  Converged: {result_np_well.converged}")
        print(f"  Solution: {result_np_well.x}")
        print(f"  Residual Norm: {result_np_well.residual_norm:.2e}")
        assert result_np_well.converged
        assert np.allclose(result_np_well.x, x_exact_well, atol=test_tol)

        solver_np_sing = PseudoInverseSolver(backend='numpy', a=A_sing)
        result_np_sing_ls = solver_np_sing.solve_instance(b_not_in_col_space, tol=1e-14)
        print("NumPy Singular (Least Squares):")
        print(f"  Converged: {result_np_sing_ls.converged}")
        print(f"  Solution (min norm LS): {result_np_sing_ls.x}")
        print(f"  Residual Norm: {result_np_sing_ls.residual_norm:.2e}")
        assert result_np_sing_ls.converged # pinv should succeed
        assert np.allclose(result_np_sing_ls.x, x_min_norm_ls_sol, atol=test_tol)

        # Test static solve directly
        result_np_static = PseudoInverseSolver.solve(None, b_vec, None, tol=1e-14, maxiter=1, A=A_well, backend_module=np)
        print("NumPy Static Solve:")
        print(f"  Solution: {result_np_static.x}")
        assert np.allclose(result_np_static.x, x_exact_well, atol=test_tol)


    except Exception as e:
        print(f"NumPy test failed: {e}")

    # --- JAX Test ---
    if JAX_AVAILABLE:
        print("\n--- JAX Backend Test ---")
        try:
            A_well_jax = jnp.array(A_well)
            b_vec_jax = jnp.array(b_vec)
            A_sing_jax = jnp.array(A_sing)
            b_not_in_jax = jnp.array(b_not_in_col_space)

            # Test via instance
            solver_jax = PseudoInverseSolver(backend='jax', a=A_well_jax)
            result_jax_well = solver_jax.solve_instance(b_vec_jax, tol=1e-14)
            print("JAX Well-conditioned:")
            print(f"  Converged: {result_jax_well.converged}")
            print(f"  Solution: {result_jax_well.x}")
            print(f"  Residual Norm: {result_jax_well.residual_norm:.2e}")
            assert result_jax_well.converged
            assert np.allclose(np.array(result_jax_well.x), x_exact_well, atol=test_tol) # Compare with NumPy

            # Test singular via instance
            solver_jax_sing = PseudoInverseSolver(backend='jax', a=A_sing_jax)
            result_jax_sing_ls = solver_jax_sing.solve_instance(b_not_in_jax, tol=1e-14)
            print("JAX Singular (Least Squares):")
            print(f"  Converged: {result_jax_sing_ls.converged}")
            print(f"  Solution (min norm LS): {result_jax_sing_ls.x}")
            print(f"  Residual Norm: {result_jax_sing_ls.residual_norm:.2e}")
            assert result_jax_sing_ls.converged
            assert np.allclose(np.array(result_jax_sing_ls.x), x_min_norm_ls_sol, atol=test_tol)

            # Test static solve directly
            result_jax_static = PseudoInverseSolver.solve(None, b_vec_jax, None, tol=1e-14, maxiter=1, A=A_well_jax, backend_module=jnp)
            print("JAX Static Solve:")
            print(f"  Solution: {result_jax_static.x}")
            assert np.allclose(np.array(result_jax_static.x), x_exact_well, atol=test_tol)

            # Test getting the compiled function
            jax_pinv_solver_func = PseudoInverseSolver.get_solver_func(jnp)
            result_jax_compiled = jax_pinv_solver_func(None, b_vec_jax, None, 1e-14, 1, None, jnp, A=A_well_jax, sigma=None)
            print("JAX Compiled Func Solve:")
            print(f"  Solution: {result_jax_compiled.x}")
            assert np.allclose(np.array(result_jax_compiled.x), x_exact_well, atol=test_tol)


        except Exception as e:
            print(f"JAX test failed: {e}")
    else:
        print("\nJAX not available, skipping JAX tests.")

    print("\n--- Tests Complete ---")
    
# -----------------------------------------------------------------------------
