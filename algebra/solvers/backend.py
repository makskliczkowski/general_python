r'''

Implements a unified BackendSolver that wraps native `linalg.solve` functions
from NumPy, SciPy, and JAX backends. This provides a simple, direct solver
interface that automatically dispatches to the appropriate backend.

The BackendSolver solves linear systems $ Ax = b $ using:
- `numpy.linalg.solve`      for NumPy arrays
- `scipy.linalg.solve`      for SciPy (with additional options like assume_a)
- `jax.numpy.linalg.solve`  for JAX arrays (JIT-compatible)

Mathematical Formulation:
-------------------------
Given a matrix $ A $ and right-hand side vector $ b $, solve:
    $ (A + \\sigma I) x = b $

where $ \\sigma $ is an optional regularization/shift parameter.

Usage:
------
    >>> from general_python.algebra.solvers.backend import BackendSolver
    >>> result = BackendSolver.solve(matvec=None, b=b, x0=None, A=A, backend_module=np)
    >>> x = result.x

References:
-----------
    - NumPy: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    - SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html
    - JAX: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.solve.html

-----------------------------------------------------
File            : general_python/algebra/solvers/backend.py
Author          : Maksymilian Kliczkowski
-----------------------------------------------------
'''

from typing import Optional, Callable, Any, Type
import numpy as np

try:
    from ..solver import (
        Solver, SolverResult, SolverError, SolverErrorMsg,
        SolverType, Array, MatVecFunc, StaticSolverFunc
    )
    from ..preconditioners import Preconditioner
except ImportError:
    raise ImportError("Could not import base solver classes. Check the module structure.")

# JAX imports
try:
    from ..utils import JAX_AVAILABLE
    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
    else:
        jax = None
        jnp = np
except ImportError:
    JAX_AVAILABLE   = False
    jax             = None
    jnp             = np

# SciPy imports
try:
    import scipy.linalg as spla
    SCIPY_AVAILABLE = True
except ImportError:
    spla = None
    SCIPY_AVAILABLE = False

# -----------------------------------------------------------------------------
#! Backend-specific solve wrappers
# -----------------------------------------------------------------------------

def _solve_numpy(A: np.ndarray, b: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    Solve Ax = b using numpy.linalg.solve.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n).
    b : np.ndarray
        Right-hand side vector of shape (n,) or (n, k).
    sigma : Optional[float]
        Regularization parameter. If provided, solves (A + sigma*I)x = b.
        
    Returns
    -------
    np.ndarray
        Solution vector x.
    """
    if sigma is not None and sigma != 0.0:
        A_eff = A + sigma * np.eye(A.shape[0], dtype=A.dtype)
    else:
        A_eff = A
    return np.linalg.solve(A_eff, b)

def _solve_scipy(A: np.ndarray, b: np.ndarray, sigma: Optional[float] = None,
                 assume_a: str = 'gen', check_finite: bool = True) -> np.ndarray:
    """
    Solve Ax = b using scipy.linalg.solve.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n).
    b : np.ndarray
        Right-hand side vector of shape (n,) or (n, k).
    sigma : Optional[float]
        Regularization parameter. If provided, solves (A + sigma*I)x = b.
    assume_a : str
        Matrix structure assumption for optimization:
        - 'gen': general matrix (default)
        - 'sym': symmetric
        - 'her': Hermitian
        - 'pos': positive definite
    check_finite : bool
        Whether to check that inputs contain only finite numbers.
        
    Returns
    -------
    np.ndarray
        Solution vector x.
    """
    if spla is None:
        raise ImportError("SciPy is required for scipy backend solver.")
    
    if sigma is not None and sigma != 0.0:
        A_eff = A + sigma * np.eye(A.shape[0], dtype=A.dtype)
    else:
        A_eff = A
    return spla.solve(A_eff, b, assume_a=assume_a, check_finite=check_finite)

if JAX_AVAILABLE:
    import jax.numpy as jnp
    
    @jax.jit
    def _solve_jax_core(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled JAX solve (no sigma for JIT compatibility)."""
        return jnp.linalg.solve(A, b)
    
    def _solve_jax(A: jnp.ndarray, b: jnp.ndarray, sigma: Optional[float] = None) -> jnp.ndarray:
        """
        Solve Ax = b using jax.numpy.linalg.solve.
        
        Parameters
        ----------
        A : jnp.ndarray
            Square matrix of shape (n, n).
        b : jnp.ndarray
            Right-hand side vector of shape (n,) or (n, k).
        sigma : Optional[float]
            Regularization parameter. If provided, solves (A + sigma*I)x = b.
            
        Returns
        -------
        jnp.ndarray
            Solution vector x.
        """
        if sigma is not None and sigma != 0.0:
            A_eff = A + sigma * jnp.eye(A.shape[0], dtype=A.dtype)
        else:
            A_eff = A
        return _solve_jax_core(A_eff, b)
else:
    _solve_jax = None

# -----------------------------------------------------------------------------
#! BackendSolver Class
# -----------------------------------------------------------------------------

class BackendSolver(Solver):
    r'''
    Unified direct solver using the appropriate backend's `linalg.solve`.
    
    Automatically dispatches to:
    - `numpy.linalg.solve` for NumPy backend
    - `scipy.linalg.solve` for SciPy backend (with additional options)
    - `jax.numpy.linalg.solve` for JAX backend (JIT-compiled)
    
    Solves $ (A + \sigma I)x = b $ directly. This is recommended over explicit 
    matrix inversion for numerical stability.
    
    Parameters
    ----------
    backend : str
        Backend to use: 'numpy', 'scipy', 'jax', or 'default'.
    sigma : Optional[float]
        Default regularization parameter.
    assume_a : str
        For SciPy backend: matrix structure assumption ('gen', 'sym', 'her', 'pos').
    **kwargs
        Additional arguments passed to base Solver.
        
    Examples
    --------
    >>> solver = BackendSolver(backend='numpy')
    >>> result = solver.solve_instance(b, A=A)
    >>> x = result.x
    
    >>> # Static usage
    >>> result = BackendSolver.solve(None, b, None, A=A, backend_module=np)
    '''
    
    _solver_type    = SolverType.BACKEND
    _symmetric      = False  # Can work for non-symmetric matrices
    
    def __init__(self,
                backend         : str                              = 'default',
                dtype           : Optional[Type]                   = None,
                eps             : float                            = 0,        # Not used
                maxiter         : int                              = 1,        # Not used
                default_precond : Optional[Preconditioner]         = None,     # Not used
                a               : Optional[Array]                  = None,
                s               : Optional[Array]                  = None,     # Not used directly
                sp              : Optional[Array]                  = None,     # Not used directly
                matvec_func     : Optional[MatVecFunc]             = None,     # Not used
                sigma           : Optional[float]                  = None,
                is_gram         : bool                             = False,
                assume_a        : str                              = 'gen',    # SciPy option
                **kwargs):
        
        super().__init__(
            backend         = backend, 
            dtype           = dtype, 
            eps             = eps, 
            maxiter         = maxiter,
            default_precond = default_precond, 
            a               = a, 
            s               = s,
            matvec_func     = matvec_func, 
            sigma           = sigma, 
            is_gram         = is_gram
        )
        
        self._assume_a = assume_a  # SciPy-specific option
        self._symmetric = False

    # --------------------------------------------------------------------------
    #! Static Methods Implementation
    # --------------------------------------------------------------------------

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
        """
        Returns a backend-adapted function compatible with the common solver wrapper.
        
        The returned function conforms to the StaticSolverFunc signature used by
        Solver._solver_wrap_compiled. Iterative params (tol, maxiter, precond) 
        are ignored by this direct method.
        
        Parameters
        ----------
        backend_module : Any
            numpy, jax.numpy, or 'scipy' indicator.
        use_matvec : bool
            Ignored for direct solver.
        use_fisher : bool
            Whether to construct matrix from Fisher form.
        use_matrix : bool
            Whether matrix is provided directly.
        sigma : Optional[float]
            Regularization parameter.
            
        Returns
        -------
        StaticSolverFunc
            The solver function.
        """
        
        def func(matvec, b, x0, tol, maxiter, precond_apply, **kwargs):
            # Accept both 'a' and 'A' from wrapper callers
            A_kw        = kwargs.get('A', kwargs.get('a', None))
            assume_a    = kwargs.get('assume_a', 'gen')
            
            return BackendSolver.solve(
                matvec          = matvec,       # Ignored
                b               = b,
                x0              = x0,           # Ignored
                tol             = 0.0,          # Ignored
                maxiter         = 1,            # Ignored
                precond_apply   = None,         # Ignored
                backend_module  = backend_module,
                A               = A_kw,
                sigma           = sigma,
                assume_a        = assume_a
            )
        
        return Solver._solver_wrap_compiled(
            backend_module, func, use_matvec, use_fisher, use_matrix, sigma
        )

    @staticmethod
    def solve(
            # === Core Problem Definition ===
            matvec          : MatVecFunc,               # Ignored
            b               : Array,
            x0              : Array,                    # Ignored
            # === Solver Parameters ===
            *,
            tol             : float = 0.0,              # Ignored
            maxiter         : int = 1,                  # Ignored
            # === Optional Preconditioner ===
            precond_apply   : Optional[Callable[[Array], Array]] = None,  # Ignored
            # === Backend Specification ===
            backend_module  : Any = None,
            # === Solver Specific Arguments ===
            A               : Optional[Array] = None,   # REQUIRED
            sigma           : Optional[float] = None,   # Optional regularization
            assume_a        : str = 'gen',              # SciPy option
            use_scipy       : bool = False,             # Force SciPy even with NumPy arrays
            **kwargs        : Any) -> SolverResult:
        r"""
        Static solve implementation using the appropriate backend's `linalg.solve`.
        
        Parameters
        ----------
        matvec : MatVecFunc
            Ignored for direct solver.
        b : Array
            Right-hand side vector $ b $.
        x0 : Array
            Ignored for direct solver.
        tol : float
            Ignored for direct solver.
        maxiter : int
            Ignored for direct solver.
        precond_apply : Optional[Callable]
            Ignored for direct solver.
        backend_module : Any
            The backend module (numpy, jax.numpy, or 'scipy').
        A : Array
            **Required**. The matrix $ A $.
        sigma : Optional[float]
            Regularization parameter $ \\sigma $. Solves $ (A + \\sigma I)x = b $.
        assume_a : str
            SciPy option: matrix structure assumption ('gen', 'sym', 'her', 'pos').
        use_scipy : bool
            If True, use scipy.linalg.solve even for NumPy arrays.
        **kwargs
            Additional ignored arguments.
            
        Returns
        -------
        SolverResult
            Named tuple containing the solution x, converged=True, iterations=1,
            and the residual norm.
            
        Raises
        ------
        SolverError
            If matrix A is not provided or solve fails.
        """
        # Validate inputs
        if A is None:
            raise SolverError(
                SolverErrorMsg.MAT_NOT_SET, 
                "Matrix 'A' required via kwargs."
            )
        
        if backend_module is None:
            raise SolverError(
                SolverErrorMsg.BACKEND_MISMATCH,
                "Backend module required."
            )
        
        # Convert to backend arrays
        A_be = backend_module.asarray(A)
        b_be = backend_module.asarray(b)
        
        # Validate dimensions
        if A_be.ndim != 2 or A_be.shape[0] != A_be.shape[1]:
            raise SolverError(
                SolverErrorMsg.DIM_MISMATCH,
                f"Matrix A must be square, got shape {A_be.shape}"
            )
        
        if A_be.shape[0] != b_be.shape[0]:
            raise SolverError(
                SolverErrorMsg.DIM_MISMATCH,
                f"Dimension mismatch: A={A_be.shape}, b={b_be.shape}"
            )
        
        try:
            # Dispatch to appropriate backend
            if backend_module is jnp and JAX_AVAILABLE:
                # JAX backend
                x_sol = _solve_jax(A_be, b_be, sigma=sigma)
                
            elif use_scipy and SCIPY_AVAILABLE:
                # Force SciPy
                x_sol = _solve_scipy(
                    np.asarray(A_be), np.asarray(b_be), 
                    sigma=sigma, assume_a=assume_a
                )
                x_sol = backend_module.asarray(x_sol)
                
            elif backend_module is np:
                # NumPy backend
                if SCIPY_AVAILABLE and assume_a != 'gen':
                    # Use SciPy if special matrix structure is specified
                    x_sol = _solve_scipy(A_be, b_be, sigma=sigma, assume_a=assume_a)
                else:
                    x_sol = _solve_numpy(A_be, b_be, sigma=sigma)
                    
            else:
                # Generic fallback - try numpy.linalg.solve
                x_sol = _solve_numpy(
                    np.asarray(A_be), np.asarray(b_be), sigma=sigma
                )
                x_sol = backend_module.asarray(x_sol)
            
            # Calculate residual norm
            if sigma is not None and sigma != 0.0:
                A_eff = A_be + sigma * backend_module.eye(A_be.shape[0], dtype=A_be.dtype)
            else:
                A_eff = A_be
                
            residual = b_be - backend_module.dot(A_eff, x_sol)
            res_norm = float(backend_module.linalg.norm(residual))
            
            return SolverResult(
                x               = x_sol, 
                converged       = True, 
                iterations      = 1, 
                residual_norm   = res_norm
            )
            
        except np.linalg.LinAlgError as e:
            raise SolverError(
                SolverErrorMsg.MAT_SINGULAR,
                f"Direct solve failed (LinAlgError): {e}"
            ) from e
            
        except Exception as e:
            if "LinAlgError" in str(type(e)):
                raise SolverError(
                    SolverErrorMsg.MAT_SINGULAR,
                    f"Direct solve failed (LinAlgError): {e}"
                ) from e
            else:
                raise SolverError(
                    SolverErrorMsg.CONV_FAILED,
                    f"Direct solve failed: {e}"
                ) from e

    # --------------------------------------------------------------------------
    #! Instance Methods Override
    # --------------------------------------------------------------------------

    def solve_instance(self,
                    b               : Array,
                    x0              : Optional[Array]   = None,  # Ignored
                    *,
                    tol             : Optional[float]   = None,  # Ignored
                    maxiter         : Optional[int]     = None,  # Ignored
                    precond         : Optional[Preconditioner] = None,  # Ignored
                    sigma           : Optional[float]   = None,
                    assume_a        : Optional[str]     = None,
                    **kwargs) -> SolverResult:
        r"""
        Instance method: Solves $ (A + \sigma I)x = b $ using backend's `linalg.solve`.
        
        Uses matrix `A` and default `sigma` from init, which can be overridden 
        by the `sigma` argument. Ignores iterative parameters.
        
        Parameters
        ----------
        b : Array
            Right-hand side vector $ b $.
        x0 : Optional[Array]
            Ignored.
        tol : Optional[float]
            Ignored.
        maxiter : Optional[int]
            Ignored.
        precond : Optional[Preconditioner]
            Ignored.
        sigma : Optional[float]
            Overrides instance default regularization $ \sigma $.
        assume_a : Optional[str]
            Overrides instance default matrix structure assumption.
        **kwargs
            Additional arguments. Can include 'A' to override instance matrix.
            
        Returns
        -------
        SolverResult
            Solution result.
        """
        # Determine Matrix A (priority: kwargs -> instance)
        matrix_a = kwargs.get('A', kwargs.get('a', self._conf_a))
        
        if matrix_a is None:
            if self._conf_is_gram:
                matrix_a = self._form_gram_matrix()
                matrix_a = self._backend.asarray(matrix_a)
            else:
                raise SolverError(
                    SolverErrorMsg.MAT_NOT_SET,
                    "Matrix A not configured or passed via kwargs."
                )
        
        # Determine parameters
        current_sigma = sigma if sigma is not None else self._conf_sigma
        current_assume_a = assume_a if assume_a is not None else self._assume_a
        
        # Call static solve
        result = BackendSolver.solve(
            matvec          = None,         # Ignored
            b               = b,
            x0              = None,         # Ignored
            tol             = 0,            # Ignored
            maxiter         = 1,            # Ignored
            precond_apply   = None,         # Ignored
            backend_module  = self._backend,
            A               = matrix_a,
            sigma           = current_sigma,
            assume_a        = current_assume_a,
            **kwargs
        )
        
        # Store results
        self._last_result = result
        return result

# -----------------------------------------------------------------------------
#! Convenience aliases
# -----------------------------------------------------------------------------

# Alias for backward compatibility with DirectSolver pattern
BackendDirectSolver = BackendSolver

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------
