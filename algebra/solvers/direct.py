'''
file:       general_python/algebra/solvers/direct.py
author:     Maksymilian Kliczkowski

Implements direct solvers for linear systems $ Ax = b $.
These methods typically involve matrix factorization (like LU) or inversion
and are suitable for dense, well-conditioned matrices of moderate size.
They generally do not benefit from iterative preconditioners.
'''

from typing import Optional, Callable, Union, Any, NamedTuple, Type
import numpy as np
import inspect

from ..solver import (
    Solver, SolverResult, SolverError, SolverErrorMsg,
    SolverType, Array, MatVecFunc, StaticSolverFunc
)
from ..preconditioners import Preconditioner

from ..utils import JAX_AVAILABLE, get_backend
try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jax_spla
except ImportError:
    jax = None
    jnp = np

import scipy.linalg as spla

# -----------------------------------------------------------------------------
#! Direct Solver using Backend's linalg.solve
# -----------------------------------------------------------------------------

class DirectSolver(Solver):
    r'''
    Direct solver using the backend's `linalg.solve` function (e.g., `numpy.linalg.solve`
    or `jax.numpy.linalg.solve`).

    Solves $ (A + \sigma I)x = b $ directly. Recommended over explicit inversion.
    '''
    _solver_type    = SolverType.DIRECT # Or BACKEND_SOLVER

    def __init__(self,
                backend         : str                             = 'default',
                dtype           : Optional[Type]                  = None,
                eps             : float                           = 0, # Not used
                maxiter         : int                             = 1, # Not used
                default_precond : Optional[Preconditioner]        = None, # Not used
                a               : Optional[Array]                 = None,
                s               : Optional[Array]                 = None, # Not used directly
                sp              : Optional[Array]                 = None, # Not used directly
                matvec_func     : Optional[MatVecFunc]            = None, # Not used
                sigma           : Optional[float]                 = None,
                is_gram         : bool                            = False
                ):
        super().__init__(backend=backend, dtype=dtype, eps=eps, maxiter=maxiter,
                        default_precond=default_precond, a=a, s=s,
                        matvec_func=matvec_func, sigma=sigma, is_gram=is_gram)
        self._symmetric = False # Can work for non-symmetric

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
        Returns a lambda function wrapping the static `DirectSolver.solve`.

        Note: The returned function adheres to the StaticSolverFunc signature
        but ignores iterative parameters and requires matrix 'A' via kwargs.

        Args:
            backend_module (Any): The backend module (numpy or jax.numpy).

        Returns:
            StaticSolverFunc: A callable wrapper for the direct solve.
        """
        # This lambda matches the expected signature but calls the specific static solve
        return lambda matvec, b, x0, tol, maxiter, precond_apply, backend_mod, **kwargs: \
                    DirectSolver.solve(matvec=matvec, b=b, x0=x0, tol=tol, maxiter=maxiter,
                                         precond_apply=precond_apply, backend_module=backend_mod, **kwargs)

    @staticmethod
    def solve(
            # === Core Problem Definition ===
            matvec          : MatVecFunc,           # Ignored
            b               : Array,
            x0              : Array,                # Ignored
            # === Solver Parameters ===
            *,
            tol             : float,                # Ignored
            maxiter         : int,                  # Ignored
            # === Optional Preconditioner ===
            precond_apply   : Optional[Callable[[Array], Array]] = None, # Ignored
            # === Backend Specification ===
            backend_module  : Any,
            # === Solver Specific Arguments ===
            A               : Optional[Array] = None, # REQUIRED
            sigma           : Optional[float] = None, # Optional
            **kwargs        : Any) -> SolverResult:
        """
        Static solve implementation using the backend's `linalg.solve`.

        Args:
            matvec:
                Ignored.
            b:
                RHS vector.
            x0:
                Ignored.
            tol:
                Ignored.
            maxiter:
                Ignored.
            precond_apply:
                Ignored.
            backend_module:
                The backend (numpy or jax.numpy).
            A (Array):
                **Required** kwarg. The matrix $ A $.
            sigma (float, optional):
                **Optional** kwarg. Regularization $ \\sigma $.
            **kwargs: Other ignored arguments.

        Returns:
            SolverResult: Contains the solution. Iterations=1, converged=True.
        """
        if A is None:
            raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix 'A' required via kwargs.")
        if backend_module is None:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH,
                                        "Backend module required.")

        # find the matrices in a given backend
        a_be    = backend_module.asarray(A)
        b_be    = backend_module.asarray(b)

        if a_be.ndim != 2 or a_be.shape[0] != a_be.shape[1]:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH,
                    f"Matrix A must be square, got {a_be.shape}")
        if a_be.shape[0] != b_be.shape[0] or b_be.ndim != 1:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH,
                    f"Dimension mismatch: A={a_be.shape}, b={b_be.shape}")

        try:
            if sigma is not None and sigma != 0.0:
                aeff = a_be + sigma * backend_module.eye(a_be.shape[0], dtype=a_be.dtype)
            else:
                aeff = a_be

            print(f"({DirectSolver.__name__}) Calling static {backend_module.__name__}.linalg.solve...")
            x_sol = backend_module.linalg.solve(aeff, b_be)

            # Calculate residual norm (using effective matrix)
            final_residual = b_be - backend_module.dot(aeff, x_sol)
            final_res_norm = backend_module.linalg.norm(final_residual)

            return SolverResult(x=x_sol, converged=True, iterations=1, residual_norm=final_res_norm)

        except np.linalg.LinAlgError as e:
            # Specific check for NumPy's LinAlgError (covers singular etc.)
            raise SolverError(SolverErrorMsg.MAT_SINGULAR, f"Direct solve failed (LinAlgError): {e}") from e
        except Exception as e:
            # Catch other potential backend errors (like JAX's LinAlgError)
            if "LinAlgError" in str(type(e)): # Crude check for JAX error
                raise SolverError(SolverErrorMsg.MAT_SINGULAR, f"Direct solve failed (LinAlgError): {e}") from e
            else: # General failure
                raise SolverError(SolverErrorMsg.CONV_FAILED, f"Direct solve failed: {e}") from e

    # --------------------------------------------------
    #! Instance Methods Override
    # --------------------------------------------------

    def solve_instance(self,
                    b               : Array,
                    x0              : Optional[Array]   = None, # Ignored
                    *,
                    tol             : Optional[float]   = None, # Ignored
                    maxiter         : Optional[int]     = None, # Ignored
                    precond         : Union[Preconditioner, Callable[[Array], Array], None] = None, # Ignored
                    sigma           : Optional[float]   = None,
                    **kwargs) -> SolverResult:
        r"""
        Instance: Solves $ (A + \sigma I)x = b $ using backend's `linalg.solve`.

        Uses matrix `A` and default `sigma` from init, overridden by `sigma` arg.
        Ignores iterative parameters.

        Args:
            b (Array):
                RHS vector $ b $.
            x0, tol, maxiter, precond:
                Ignored.
            sigma (Optional[float])
                : Overrides instance default regularization $ \\sigma $.
            **kwargs: Passed to static `solve` (mostly for 'A' if not set in instance).

        Returns:
            SolverResult
        """
        # Determine Matrix A (priority: instance -> kwargs)
        matrix_a = self._conf_a if self._conf_a is not None else kwargs.get('A', None)
        if matrix_a is None:
            if self._conf_is_gram:
                matrix_a = self._form_gram_matrix()
                matrix_a = self._backend.asarray(matrix_a)
            else:
                raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix A not configured or passed via kwargs.")

        # Determine Sigma
        current_sigma = sigma if sigma is not None else self._conf_sigma

        # Call Static Solve
        result = DirectSolver.solve(
            matvec          = None,     # Ignored
            b               = b,
            x0              = None,     # Ignored
            tol             = 0,        # Ignored
            maxiter         = 1,        # Ignored
            precond_apply   = None,     # Ignored
            backend_module  = self._backend,
            A               = matrix_a, # Pass the determined matrix
            sigma           = current_sigma,
            **kwargs                    # Pass any remaining kwargs
        )

        # Store results
        self._last_solution             = result.x
        self._last_converged            = result.converged
        self._last_iterations           = result.iterations
        self._last_residual_norm        = result.residual_norm
        print(f"({self.__class__.__name__}) Instance solve finished.")
        return result

# -----------------------------------------------------------------------------
#! Direct Solver using Explicit Inversion
# -----------------------------------------------------------------------------

class DirectInvSolver(Solver):
    r'''
    Direct solver using explicit matrix inversion: $ x = (A + \sigma I)^{-1} b $.

    Note:
        Generally less numerically stable and potentially less efficient
        than using `linalg.solve`. Provided for completeness or specific use cases.
        Works with both NumPy and JAX backends.
    '''
    _solver_type    = SolverType.DIRECT # Could argue for a more specific type

    def __init__(self,
                backend         : str                             = 'default',
                dtype           : Optional[Type]                  = None,
                eps             : float                           = 0,
                maxiter         : int                             = 1,
                default_precond : Optional[Preconditioner]        = None,
                a               : Optional[Array]                 = None,
                s               : Optional[Array]                 = None,
                sp              : Optional[Array]                 = None,
                matvec_func     : Optional[MatVecFunc]            = None,
                sigma           : Optional[float]                 = None,
                is_gram         : bool                            = False
                ):
        super().__init__(backend=backend, dtype=dtype, eps=eps, maxiter=maxiter,
                        default_precond=default_precond, a=a, s=s, sp=sp,
                        matvec_func=matvec_func, sigma=sigma, is_gram=is_gram)
        self._symmetric = False

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
        """ 
        Returns a lambda function wrapping the static `DirectInvSolver.solve`. 
        """
        return lambda matvec, b, x0, tol, maxiter, precond_apply, backend_mod, **kwargs: \
                    DirectInvSolver.solve(matvec=matvec, b=b, x0=x0, tol=tol, maxiter=maxiter,
                                            precond_apply=precond_apply, backend_module=backend_mod, **kwargs)

    @staticmethod
    def solve(
            matvec          : MatVecFunc,               # Ignored
            b               : Array,
            x0              : Array,                    # Ignored
            *,
            tol             : float,                    # Ignored
            maxiter         : int,                      # Ignored
            precond_apply   : Optional[Callable[[Array], Array]] = None, # Ignored
            backend_module  : Any,
            A               : Optional[Array] = None,   # REQUIRED
            sigma           : Optional[float] = None,   # Optional
            **kwargs        : Any) -> SolverResult:
        r""" 
        Static solve using explicit matrix inversion $ (A + \sigma I)^{-1} b $.
        """
        if A is None:
            raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix 'A' required via kwargs.")
        if backend_module is None:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH, "Backend module required.")

        A_be    = backend_module.asarray(A)
        b_be    = backend_module.asarray(b)

        if A_be.ndim != 2 or A_be.shape[0] != A_be.shape[1]:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH,
                        f"Matrix A must be square, got {A_be.shape}")
        if A_be.shape[0] != b_be.shape[0] or b_be.ndim != 1:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH,
                        f"Dimension mismatch: A={A_be.shape}, b={b_be.shape}")

        try:
            if sigma is not None and sigma != 0.0:
                A_eff = A_be + sigma * backend_module.eye(A_be.shape[0], dtype=A_be.dtype)
            else:
                A_eff = A_be

            print(f"({DirectInvSolver.__name__}) Calling static {backend_module.__name__}.linalg.inv ...")
            A_inv           = backend_module.linalg.inv(A_eff)
            x_sol           = backend_module.dot(A_inv, b_be)

            final_residual  = b_be - backend_module.dot(A_eff, x_sol)
            final_res_norm  = backend_module.linalg.norm(final_residual)
            return SolverResult(x=x_sol, converged=True, iterations=1, residual_norm=final_res_norm)

        except np.linalg.LinAlgError as e:
            raise SolverError(SolverErrorMsg.MAT_SINGULAR,
                    f"Direct inverse failed (LinAlgError): {e}") from e
        except Exception as e:
            if "LinAlgError" in str(type(e)):
                raise SolverError(SolverErrorMsg.MAT_SINGULAR,
                    f"Direct inverse failed (LinAlgError): {e}") from e
            else:
                raise SolverError(SolverErrorMsg.CONV_FAILED,
                    f"Direct inverse failed: {e}") from e

    # -------------------------------------------------------------------------

    def solve_instance(self, b: Array, x0: Optional[Array] = None, *, tol=None, maxiter=None, precond=None, sigma=None, **kwargs) -> SolverResult:
        r"""
        Instance method: Solves $ x = (A + \sigma I)^{-1} b $.
        """
        matrix_a = self._conf_a if self._conf_a is not None else kwargs.get('A', None)
        
        if matrix_a is None:
            if self._conf_is_gram and self._conf_s is not None and self._conf_sp is not None:
                print(f"({self.__class__.__name__}) Forming Gram matrix for direct solve.")
                n_size      = self._conf_s.shape[0]
                norm_factor = float(n_size) if n_size > 0 else 1.0
                matrix_a    = (self._conf_sp @ self._conf_s) / norm_factor
                matrix_a    = self._backend.asarray(matrix_a) # Ensure backend
            else:
                raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix A not configured or passed via kwargs.")

        current_sigma                   = sigma if sigma is not None else self._conf_sigma
        result = DirectInvSolver.solve(matvec=None, b=b, x0=None, tol=0, maxiter=1, precond_apply=None,
                                       backend_module=self._backend, A=matrix_a, sigma=current_sigma, **kwargs)
        self._last_solution             = result.x
        self._last_converged            = result.converged
        self._last_iterations           = result.iterations
        self._last_residual_norm        = result.residual_norm
        print(f"({self.__class__.__name__}) Instance solve finished.")
        return result

    # -------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#! Direct Solver using SciPy's linalg.solve
# -----------------------------------------------------------------------------

class DirectScipy(Solver):
    ''' 
    Direct solver using `scipy.linalg.solve`. Requires NumPy backend. 
    '''
    _solver_type    = SolverType.SCIPY_DIRECT

    def __init__(self, *args, **kwargs): 
        # Reuse init logic from DirectInvSolver, force numpy
        # Ensure 'backend' kwarg is set to numpy or default is handled correctly
        kw_backend = kwargs.get('backend', 'default')
        if kw_backend != 'numpy':
            print(f"Warning: {self.__class__.__name__} uses SciPy, forcing backend to 'numpy'.")
            kwargs['backend'] = 'numpy'
        super().__init__(*args, **kwargs)
        self._symmetric = False

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
        """ Returns a lambda wrapping `DirectScipy.solve`. """
        if backend_module is not np:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH, f"{SolverType.SCIPY_DIRECT.name} requires NumPy.")
        return lambda matvec, b, x0, tol, maxiter, precond_apply, backend_mod, **kwargs: \
                    DirectScipy.solve(matvec=matvec, b=b, x0=x0, tol=tol, maxiter=maxiter,
                                        precond_apply=precond_apply, backend_module=backend_mod, **kwargs)

    # -------------------------------------------------------------------------

    @staticmethod
    def solve(matvec, b, x0, *, tol, maxiter, precond_apply=None, backend_module, A=None, sigma=None, **kwargs) -> SolverResult:
        r"""
        Static solve using `scipy.linalg.solve`.
        Args:
            matvec:
                Ignored.
            b:
                RHS vector.
            x0:
                Ignored.
            tol:
                Ignored.
            maxiter:
                Ignored.
            precond_apply:
                Ignored.
            backend_module:
                The backend module (should be NumPy).
            A (Array):
                **Required** kwarg. The matrix $ A $.
            sigma (float, optional):
                **Optional** kwarg. Regularization $ \\sigma $.
            **kwargs: Other ignored arguments.
        """
        if A is None:
            raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix 'A' required via kwargs.")
        if backend_module is not np:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH, f"{SolverType.SCIPY_DIRECT.name} requires NumPy.")

        A_np    = np.asarray(A); b_np = np.asarray(b)
        if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]: 
            raise SolverError(SolverErrorMsg.DIM_MISMATCH, f"A must be square, got {A_np.shape}")
        if A_np.shape[0] != b_np.shape[0] or b_np.ndim != 1: 
            raise SolverError(SolverErrorMsg.DIM_MISMATCH, f"Dimension mismatch: A={A_np.shape}, b={b_np.shape}")

        try:
            print(f"({DirectScipy.__name__}) Calling static scipy.linalg.solve...")
            A_eff           = A_np + sigma * np.eye(A_np.shape[0], dtype=A_np.dtype) if sigma else A_np
            solve_kwargs    = {k:v for k,v in kwargs.items() if k in inspect.signature(spla.solve).parameters}
            x_sol           = spla.solve(A_eff, b_np, **solve_kwargs)
            final_residual  = b_np - A_eff @ x_sol
            final_res_norm  = np.linalg.norm(final_residual)
            return SolverResult(x=x_sol, converged=True, iterations=1, residual_norm=final_res_norm)
        except spla.LinAlgError as e:
            raise SolverError(SolverErrorMsg.MAT_SINGULAR, f"SciPy direct solve failed (LinAlgError): {e}") from e
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"SciPy direct solve failed: {e}") from e

    def solve_instance(self, b: Array, x0=None, *, tol=None, maxiter=None, precond=None, sigma=None, **kwargs) -> SolverResult:
        r""" 
        Instance method: Solves $ (A + \sigma I)x = b $ using `scipy.linalg.solve`.
        """
        matrix_a = self._conf_a if self._conf_a is not None else kwargs.get('A', None)
        if matrix_a is None:
            if self._conf_is_gram and self._conf_s is not None and self._conf_sp is not None:
                print(f"({self.__class__.__name__}) Forming Gram matrix for SciPy direct solve.")
                n_size      = self._conf_s.shape[0]
                norm_factor = float(n_size) if n_size > 0 else 1.0
                matrix_a    = (self._conf_sp @ self._conf_s) / norm_factor
            else:
                raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix A not configured or passed.")
        matrix_a_np = np.asarray(matrix_a)  # Ensure NumPy

        current_sigma                   = sigma if sigma is not None else self._conf_sigma
        result                          = DirectScipy.solve(matvec=None, b=b, x0=None, tol=0, maxiter=1, precond_apply=None,
                                            backend_module=np, A=matrix_a_np, sigma=current_sigma, **kwargs)
        self._last_solution             = result.x
        self._last_converged            = result.converged
        self._last_iterations           = result.iterations
        self._last_residual_norm        = result.residual_norm
        print(f"({self.__class__.__name__}) Instance solve finished.")
        return result

# -----------------------------------------------------------------------------
#! Direct Solver using JAX-SciPy's linalg.solve
# -----------------------------------------------------------------------------

class DirectJaxScipy(Solver):
    r'''
    Direct solver using `jax.scipy.linalg.solve`. Requires JAX backend.

    Solves $ (A + \\sigma I)x = b $. Functionality depends on JAX's implementation.
    '''
    _solver_type    = SolverType.SCIPY_DIRECT # Reusing enum, maybe needs JAX_DIRECT?

    def __init__(self,
                backend         : str                             = 'jax', # Force jax
                dtype           : Optional[Type]                  = None,
                eps             : float                           = 0,
                maxiter         : int                             = 1,
                default_precond : Optional[Preconditioner]        = None,
                a               : Optional[Array]                 = None,
                s               : Optional[Array]                 = None,
                sp              : Optional[Array]                 = None,
                matvec_func     : Optional[MatVecFunc]            = None,
                sigma           : Optional[float]                 = None,
                is_gram         : bool                            = False
                ):
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for DirectJaxScipy.")
        if backend != 'jax': 
            print(f"Warning: {self.__class__.__name__} uses JAX SciPy, forcing backend to 'jax'.")
        super().__init__(backend='jax', dtype=dtype, eps=eps, maxiter=maxiter,
                        default_precond=default_precond, a=a, s=s, sp=sp,
                        matvec_func=matvec_func, sigma=sigma, is_gram=is_gram)
        self._symmetric = False

    # --------------------------------------------------
    #! Static Methods Implementation
    # --------------------------------------------------

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
        """ Returns a lambda function wrapping the static `DirectJaxScipy.solve`. """
        if backend_module is not jnp:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH, f"{SolverType.SCIPY_DIRECT.name} (JAX) requires JAX backend.")
        if jax_spla is None:
            raise ImportError("jax.scipy.linalg not available.")
        # Return lambda matching signature
        return lambda matvec, b, x0, tol, maxiter, precond_apply, backend_mod, **kwargs: \
                    DirectJaxScipy.solve(matvec=matvec, b=b, x0=x0, tol=tol, maxiter=maxiter,
                                           precond_apply=precond_apply, backend_module=backend_mod, **kwargs)

    # --------------------------------------------------

    @staticmethod
    def solve(matvec, b, x0, *, tol, maxiter, precond_apply=None, backend_module, A=None, sigma=None, **kwargs) -> SolverResult:
        r"""
        Static solve using `jax.scipy.linalg.solve`. 
        """
        if A is None:
            raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix 'A' required via kwargs.")
        if backend_module is not jnp:
            raise SolverError(SolverErrorMsg.BACKEND_MISMATCH, f"{DirectJaxScipy.__name__} requires JAX backend.")
        if jax_spla is None:
            raise ImportError("jax.scipy.linalg not available.")

        # JAX operates better if types are consistent from the start
        dtype   = b.dtype 
        A_jax   = jnp.asarray(A, dtype=dtype)
        b_jax   = jnp.asarray(b, dtype=dtype)

        if A_jax.ndim != 2 or A_jax.shape[0] != A_jax.shape[1]:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH,
                        f"Matrix A must be square, got {A_jax.shape}")
        if A_jax.shape[0] != b_jax.shape[0] or b_jax.ndim != 1:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH,
                        f"Dimension mismatch: A={A_jax.shape}, b={b_jax.shape}")

        try:
            if sigma is not None and sigma != 0.0:
                A_eff = A_jax + sigma * jnp.eye(A_jax.shape[0], dtype=dtype)
            else:
                A_eff = A_jax

            print(f"({DirectJaxScipy.__name__}) Calling static jax.scipy.linalg.solve...")
            # Pass relevant kwargs (check jax.scipy.linalg.solve signature)
            # Currently it takes `sym_pos` which corresponds to `assume_a='pos'`?
            # JAX solve might be less feature-rich than SciPy's regarding options.
            solve_kwargs = {}
            if 'sym_pos' in kwargs:
                solve_kwargs['sym_pos'] = kwargs['sym_pos']
            if 'lower' in kwargs: 
                solve_kwargs['lower'] = kwargs['lower']

            x_sol = jax_spla.solve(A_eff, b_jax, **solve_kwargs)

            # Ensure solution is computed if execution is deferred
            x_sol.block_until_ready()

            final_residual = b_jax - jnp.dot(A_eff, x_sol)
            final_res_norm = jnp.linalg.norm(final_residual)

            return SolverResult(x=x_sol, converged=True, iterations=1, residual_norm=final_res_norm)

        except Exception as e: # Catch JAX LinAlgError etc.
            if "LinAlgError" in str(type(e)):
                raise SolverError(SolverErrorMsg.MAT_SINGULAR, f"JAX direct solve failed (LinAlgError): {e}") from e
            raise SolverError(SolverErrorMsg.CONV_FAILED, f"JAX direct solve failed: {e}") from e

    # --------------------------------------------------

    def solve_instance(self, b: Array, x0=None, *, tol=None, maxiter=None, precond=None, sigma=None, **kwargs) -> SolverResult:
        r""" 
        Instance method: Solves $ (A + \sigma I)x = b $ using `jax.scipy.linalg.solve`.
        """
        matrix_a = self._conf_a if self._conf_a is not None else kwargs.get('A', None)
        if matrix_a is None:
            if self._conf_is_gram and self._conf_s is not None and self._conf_sp is not None:
                print(f"({self.__class__.__name__}) Forming Gram matrix for JAX direct solve.")
                n_size      = self._conf_s.shape[0]
                norm_factor = float(n_size) if n_size > 0 else 1.0
                matrix_a = (self._conf_sp @ self._conf_s) / norm_factor
            else: raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Matrix A not configured or passed.")
            
        # Ensure JAX array for static solve
        matrix_a_jax                    = jnp.asarray(matrix_a)

        current_sigma                   = sigma if sigma is not None else self._conf_sigma
        result                          = DirectJaxScipy.solve(matvec=None, b=b, x0=x0, tol=tol, maxiter=maxiter, precond_apply=precond,
                                            backend_module=jnp, A=matrix_a_jax, sigma=current_sigma, **kwargs)
        self._last_solution             = result.x; self._last_converged = result.converged;
        self._last_iterations           = result.iterations; self._last_residual_norm = result.residual_norm;
        print(f"({self.__class__.__name__}) Instance solve finished.")
        return result

# -----------------------------------------------------------------------------