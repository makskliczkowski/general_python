r'''
Linear System Solver Interface
==========================

Defines the abstract interface and helper structures for solving linear systems ($Ax = b$), potentially using left preconditioning ($M^{-1}Ax = M^{-1}b$) or right preconditioning ($A M^{-1} y = b,\ x = M^{-1}y$).  Supports different backends (NumPy, JAX) and promotes a static-method-based interface for solver algorithms, allowing for easier compilation (e.g., JIT).

---------------------------------------------------------
file            : general_python/algebra/solver.py
author          : Maksymilian Kliczkowski
license         : MIT
---------------------------------------------------------
'''

import numpy as np
import numba
import scipy as sp
from functools import partial
from typing import Optional, Callable, Union, Any, NamedTuple, Type, TypeAlias
from abc import ABC, abstractmethod
from enum import Enum, auto, unique                 # for enumerations

# -----------------------------------------------------------------------------

from ..algebra.utils import JAX_AVAILABLE, get_backend, maybe_jit, Array
from ..algebra.preconditioners import Preconditioner, PreconitionerApplyFun

try:
    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
    else:
        jax                 = None
        jnp                 = None
except ImportError:
    jax                     = None
    jnp                     = None

# -----------------------------------------------------------------------------
#! Type hints
# -----------------------------------------------------------------------------

MatVecFunc          = Callable[[Array], Array]
StaticSolverFunc    = Callable[..., 'SolverResult']

# -----------------------------------------------------------------------------

@unique
class SolverType(Enum):
    """
    Enumeration class for the different types of solvers.
    """
    
    # Default
    DIRECT          = auto() # Direct solver x = A^-1 b
    BACKEND         = auto() # Use the default backend solver (e.g., numpy.linalg.solve)
    PSEUDO_INVERSE  = auto() # Pseudo-inverse solver
    # Wrappers around backend/scipy solvers (might use instance methods)
    SCIPY_CG        = auto()
    SCIPY_MINRES    = auto()
    SCIPY_GMRES     = auto()
    SCIPY_DIRECT    = auto()
    # my solvers
    CG              = auto() # Conjugate gradient
    MINSR           = auto() # Minimum Stochastic Reconfiguration with Noise to Signal Ratio
    MINRES          = auto() # Minimum residual
    MINRES_QLP      = auto() # Minimum residual - using QLP
    GMRES           = auto() # Generalized Minimal Residual
    ARNOLDI         = auto() # Arnoldi iteration for building the Krylov/Ritz subspaces
    
# -----------------------------------------------------------------------------
#! Errors
# -----------------------------------------------------------------------------

class SolverErrorMsg(Enum):
    '''
    Enumeration class for solver error messages.
    '''
    MATVEC_FUNC_NOT_SET = 101
    MAT_NOT_SET         = 102
    DIM_MISMATCH        = 106
    METHOD_NOT_IMPL     = 109
    BACKEND_MISMATCH    = 111
    INVALID_INPUT       = 112
    COMPILATION_NA      = 113
    
    def __str__(self):
        return self.name.replace('_', ' ').title()

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: '{str(self)}'>"

class SolverError(Exception):
    '''
    Base class for exceptions in the solver module. 
    '''
    def __init__(self, code: SolverErrorMsg, message: Optional[str] = None):
        self.code       = code
        self.message    = message if message else str(code)
        super().__init__(self.message)

    def __str__(self):
        return f"[SolverError {self.code.name} ({self.code.value})]: {self.message}"

    def __repr__(self):
        return self.__str__()

class SolverResult(NamedTuple):
    '''
    Stores the result of a solver's static execution.

    Attributes:
        x (Array):
            The computed solution vector.
        converged (bool):
            Whether the solver reached the desired tolerance.
        iterations (int):
            The number of iterations performed.
        residual_norm (Optional[float]):
            The norm of the final residual (||b - Ax||).
    '''
    x               : Array
    converged       : bool
    iterations      : int
    residual_norm   : Optional[float]
    
    def __repr__(self):
        return (f"SolverResult(x=Array(shape={self.x.shape}, dtype={self.x.dtype}), "
                f"converged={self.converged}, iterations={self.iterations}, "
                f"residual_norm={self.residual_norm})")
        
    def __str__(self):
        return f'converged={self.converged}, iterations={self.iterations}, residual_norm={self.residual_norm}'

# -----------------------------------------------------------------------------
#! General Solver Abstract Base Class
# -----------------------------------------------------------------------------

class Solver(ABC):
    '''
    Abstract base class for linear system solvers

    Targets problems of the form $Ax = b$.

    Primarily defines the static interface `solve` that concrete algorithm
    implementations (like CG, MINRES) must provide.

    Also includes static helpers `create_matvec_from_*` to construct the
    matrix-vector product function and an optional instance method `solve_instance`
    for convenience when working with configured Solver objects.
    
    Normally, one should focus on using the implementation provided
    by the constructor-set function due to the optimized call. Nevertheless,
    instance based setups are convenient for the less-time consuming tasks.
    
    The solver can be initialized with a matrix A, a matrix-vector multiplication
    function, or a Fisher matrix S. The solver can also be initialized with
    a preconditioner M. 
    
    The solver can be used to solve the linear system
    Ax = b or M^{-1}Ax = M^{-1}b.
    '''
    _solver_type : Optional[SolverType] = None # To be set by concrete subclasses

    def __init__(self,
                backend         : str                             = 'default',
                dtype           : Optional[Type]                  = None,
                # Default parameters (mainly informational or for convenience wrappers)
                eps             : float                           = 1e-8,
                maxiter         : int                             = 1000,
                default_precond : Optional[Preconditioner]        = None,
                # Configuration for convenience instance setup (optional)
                matvec_func     : Optional[MatVecFunc]            = None,
                a               : Optional[Array]                 = None,
                s               : Optional[Array]                 = None,
                s_p             : Optional[Array]                 = None,
                sigma           : Optional[float]                 = None,
                is_gram         : bool                            = False,
                **kwargs
                ):
        '''
        Initializes solver metadata and optionally pre-configures for instance usage.

        Args:
            backend (str):
                Preferred backend ('numpy', 'jax'). Affects helpers.
            dtype (Type, optional):
                Default data type.
            eps (float):
                Default tolerance for convenience methods.
            maxiter (int):
                Default max iterations for convenience methods.
            default_precond (Preconditioner, optional):
                A default preconditioner instance.
            a (Array, optional):
                Explicit matrix A for instance setup.
            s (Array, optional):
                Matrix S for Fisher setup.
            s_p (Array, optional):
                Matrix Sp for Fisher setup.
            matvec_func (Callable, optional):
                Explicit matvec function.
            sigma (float, optional):
                Default regularization for setup helpers.
            is_gram (bool):
                If using Fisher setup (S, Sp).
                
        Example:
            >>> solver = CgSolver(backend='jax', dtype=jnp.float32, eps=1e-6, maxiter=500)
            >>> result = solver.solve_instance(b, x0)
        '''
        self._backend_str               : str
        self._backend                   : Any  # numpy-like module
        self._backend_sp                : Any  # scipy-like module (potential use)
        self._isjax                     : bool
        self._set_backend(backend)      # Set backend attributes

        self._dtype                     = dtype if dtype is not None else self._backend.float32

        # Store defaults / config
        self._default_eps               = eps
        self._default_maxiter           = maxiter
        self._default_precond           = default_precond   # Store the instance
        self._conf_a                    = a                 # Store the default A
        self._conf_s                    = s                 # Store the default decomposition of A
        self._conf_sp                   = s_p
        self._conf_matvec_func          = matvec_func       # Store the default matvec function
        self._conf_sigma                = sigma             # Store the default regularization -> DIAGONAL SHIFT
        self._conf_is_gram              = is_gram
        
        # Cached compiled solver function for instance use
        self._cached_solver_func        = None
        
        # Store results from last instance solve call
        self._last_solution             : Optional[Array]   = None
        self._last_converged            : Optional[bool]    = None
        self._last_iterations           : Optional[int]     = None
        self._last_residual_norm        : Optional[float]   = None

    # -------------------------------------------------------------------------

    def _set_backend(self, backend: str):
        """ 
        Internal method to set backend attributes.
        """
        new_backend_str                 = backend
        if new_backend_str == 'default' and JAX_AVAILABLE:
            new_backend_str = 'jax'
        else:
            new_backend_str = 'numpy'
        self._backend_str               = new_backend_str
        bck, (rnd, key), sp             = get_backend(self._backend_str, scipy=True, random=True)
        self._backend, self._backend_sp = bck, sp
        self._backend_rng               = rnd
        self._backend_rng_key           = key
        self._isjax                     = JAX_AVAILABLE and self._backend is not np

    # -------------------------------------------------------------------------

    @staticmethod
    def create_matvec_from_matrix_jax(a: Array, sigma: Optional[float] = None) -> MatVecFunc:
        """
        Static Helper:
            Creates matvec function `x -> (A + sigma*I) @ x`.
        
        Parameters:
        -----------
            a (Array):
                The matrix (dense or sparse compatible with JAX).
            sigma (float, optional):
                Optional regularization parameter.
        Returns:
        --------
            MatVecFunc:
                The matrix-vector product function.        
        """
        
        # Convert None to 0.0 outside the matvec to avoid tracer issues
        sigma_val = 0.0 if sigma is None else sigma
        
        def matvec(x): 
            return jnp.matmul(a, x) + sigma_val * x
        return matvec

    @staticmethod
    def create_matvec_from_matrix_np(a: Array, sigma: Optional[float] = None) -> MatVecFunc:
        """
        Static Helper:
            Creates matvec function `x -> (A + sigma*I) @ x`.
        Parameters:
            a (Array):
                The matrix (dense or sparse compatible with NumPy).
            sigma (float, optional):
                Optional regularization parameter.
        Returns:
            MatVecFunc:
                The matrix-vector product function.
        """
        def matvec(x): 
            return np.dot(a, x) + sigma * x
        return matvec
    
    @staticmethod
    def create_matvec_from_matrix(
            a               : Array,
            sigma           : Optional[float]   = None,
            backend_module  : Any               = np,
            compile_func    : bool              = False) -> MatVecFunc:
        """
        Static Helper:
            Creates matvec function `x -> (A + sigma*I) @ x`.

        Args:
            a (np.ndarray, jnp.ndarray):
                The matrix (dense or sparse compatible with backend).
            sigma (float):
                Optional regularization parameter.
            backend_module:
                The backend (e.g., np, jnp) to use for operations.

        Returns:
            Callable[[Array], Array]:
                The matrix-vector product function.
        """
        if backend_module == np:
            matvec = Solver.create_matvec_from_matrix_np(a, sigma=sigma)
            if compile_func:
                matvec = Solver._compile_helper_np(matvec)
        else:
            matvec = Solver.create_matvec_from_matrix_jax(a, sigma=sigma)
            if compile_func:
                matvec = Solver._compile_helper_jax(matvec)
        return matvec

    # -------------------------------------------------------------------------
    #! Static Helpers for Creating MatVec Functions from Fisher Information
    # -------------------------------------------------------------------------
    
    @staticmethod
    def create_matvec_from_fisher_jax(
            s               : Array,
            s_p             : Array,
            sigma           : Optional[float]   = None,
            create_full     : Optional[bool]    = False) -> MatVecFunc:
        """
        Creates a matrix-vector multiplication function (MatVecFunc) based on a Fisher information inspired formulation.
        This function constructs a custom matrix-vector product operator using the input arrays `s` and `s_p`.
        It first computes the normalization constant `n` as the number of rows of `s`.
        Depending on the flag `create_full`, it either:
            - Computes the full matrix as (s_p @ s) / n and passes it along with `sigma` to Solver.create_matvec_from_matrix_jax,
                or
            - Constructs a matvec function that, for a given vector `x`, computes:
                    1. s_dot_x = dot(s, x)
                    2. sp_dot_s_dot_x = dot(s_p, s_dot_x)
                    3. The output as sp_dot_s_dot_x / n, with an additional term sigma * x if sigma is not None and non-zero.
        The resulting matvec function is then compiled (presumably via JAX JIT) using Solver._compile_helper_jax.
        Parameters:
                s (Array): A JAX array representing the first component used in constructing the operator.
                s_p (Array): A JAX array representing the second component used alongside `s`.
                sigma (Optional[float], optional): A scalar value to be added to the diagonal (identity) part of the operation.
                                                                                            Defaults to None.
                create_full (Optional[bool], optional): A flag to determine whether to create a full matrix operator using
                                                                                                    Solver.create_matvec_from_matrix_jax. Defaults to False.
        Returns:
                MatVecFunc: A function that computes the matrix-vector product corresponding to the constructed operator.
                                        This operator is compiled using Solver._compile_helper_jax.
        """

        n = s.shape[0]
        if create_full:
            return Solver.create_matvec_from_matrix_jax(s_p @ s / n, sigma)
        
        # Always include sigma term - JAX will optimize away if sigma=0
        # Use 0.0 as default to avoid None checks inside traced code
        # The None check happens HERE (outside the matvec), at trace time
        sigma_val = 0.0 if sigma is None else sigma
        
        def matvec(x):
            # O(N_samples * N_params) associative order
            # sigma_val is captured from closure - already concrete (float or traced float, never None)
            return (jnp.matmul(s_p, jnp.matmul(s, x)) / n) + (sigma_val * x)
        
        return matvec

    @staticmethod
    def create_matvec_from_fisher_np(
            s               : np.ndarray,
            s_p             : np.ndarray,
            sigma           : Optional[float]   = None,
            create_full     : Optional[bool]    = False) -> MatVecFunc:
        """
        Creates a matrix-vector multiplication function (matvec) based on the Fisher information matrix.
        This function generates a matvec function that computes the product of a vector with a matrix 
        derived from the input arrays `s` and `s_p`. Optionally, a regularization term `sigma` can be 
        added to the diagonal of the matrix. The function can also return a full matrix-based matvec 
        if `create_full` is set to True.
        Args:
            s (np.ndarray):
                A 2D array representing the first operand in the Fisher information matrix computation.
            s_p (np.ndarray):
                A 2D array representing the second operand in the Fisher information matrix computation.
            sigma (Optional[float], optional):
                A regularization parameter added to the diagonal of the matrix. 
                Defaults to None, which means no regularization is applied.
            create_full (Optional[bool], optional): If True, creates a matvec function based on the full matrix 
                `s @ s_p / n`. Defaults to False.
        Returns:
            MatVecFunc: A function that performs matrix-vector multiplication with the derived matrix.
        """

        n = s.shape[0]
        if create_full:
            # Create full Gram matrix: S\dag S where s=S [n_samples, n_params], s_p=S\dag [n_params, n_samples]
            return Solver.create_matvec_from_matrix_np(s_p @ s / n, sigma)

        # Use 0.0 as default for consistency with JAX version
        sigma_val = 0.0 if sigma is None else sigma
        
        def matvec(x):
            # O(N_samples * N_params) associative order
            return (np.dot(s_p, np.dot(s, x)) / n) + (sigma_val * x)
        return matvec

    @staticmethod
    def create_matvec_from_fisher(
            s               : Array,
            s_p             : Array,
            n               : Optional[int]     = None,
            sigma           : Optional[float]   = None,
            backend_module  : Any               = np,
            create_full     : bool              = False,
            compile_func    : bool              = False) -> MatVecFunc:
        """
        Static Helper:
            Creates matvec function `x -> (Sp @ S / N + sigma*I) @ x`.

        Args:
            s:
                Matrix S.
            s_p:
                Matrix Sp (transpose/adjoint of S).
            n:
                Normalization factor (often number of samples/outputs). Defaults to S.shape[0].
            sigma:
                Optional regularization parameter.
            backend_module:
                The backend (e.g., np, jnp) to use for operations.

        Returns:
            Callable[[Array], Array]: The matrix-vector product function.
        """
        if s.ndim != 2 or s_p.ndim != 2:
            raise SolverError(SolverErrorMsg.INVALID_INPUT, "S and Sp must be 2D.")
        if s.shape[1] != s_p.shape[0] or s.shape[0] != s_p.shape[1]:
            raise SolverError(SolverErrorMsg.DIM_MISMATCH, f"Shape mismatch: S {s.shape}, Sp {s_p.shape}")

        if create_full:
            return Solver.create_matvec_from_matrix(s_p @ s / n, sigma, backend_module)
        
        norm_factor                     = float(n) if n is not None and n > 0 else float(s.shape[0])
        if norm_factor <= 0:
            norm_factor                 = 1.0 # Avoid division by zero/negative

        if backend_module == np:
            matvec = Solver.create_matvec_from_fisher_np(s, s_p, sigma, create_full=False)
            if compile_func:
                matvec = Solver._compile_helper_np(matvec)
        else:
            matvec = Solver.create_matvec_from_fisher_jax(s, s_p, sigma, create_full=False)
            if compile_func:
                matvec = Solver._compile_helper_jax(matvec)
        return matvec

    # -------------------------------------------------------------------------
    #! Static Interface to Get the Compiled Solver Function
    # -------------------------------------------------------------------------

    @staticmethod
    def _solver_wrap_compiled(backend_module: Any,
                            callable_comp   : StaticSolverFunc,
                            use_matvec      : bool = True,
                            use_fisher      : bool = False,
                            use_matrix      : bool = False,
                            sigma           : Optional[float] = None):
        '''
        Wraps a compiled solver function to accept matrices or Fisher components
        instead of just a matvec function.

        Normally (use_matvec=True) expects callable_comp signature like:
            matvec          : MatVecFunc,
            b               : Array,
            x0              : Array,
            # Solver Parameters
            *,              # Enforce keyword arguments
            tol             : float,
            maxiter         : int,
            # Optional Preconditioner, This is the function r -> M^{-1}r
            precond_apply   : Optional[Callable[[Array], Array]] = None,
            **kwargs

        If use_matvec=False, the returned wrapper will accept:
          - if use_fisher=True  : s, s_p, b, x0, *, tol, maxiter, precond_apply, **kwargs
          - if use_fisher=False : a, b, x0, *, tol, maxiter, precond_apply, **kwargs
          
        Note: sigma can be overridden at runtime by passing sigma=value to the returned
        wrapper function. This avoids recompilation when diag_shift changes during training.
        '''
        
        if backend_module is jnp:
            # Static args: tolerances and preconditioner function (since functions must be static)
            static_argnames = ('maxiter', 'precond_apply')
            default_sigma   = 0.0 if sigma is None else float(sigma)
        
            # Dense Matrix A (Dynamic A, dynamic sigma)
            if use_matrix:
                def wrapper_logic(a, b, x0, tol, maxiter, precond_apply=None, sigma=None, snr_tol=None):
                    effective_sigma = default_sigma if sigma is None else sigma
                    x0_val          = jnp.zeros_like(b) if x0 is None else x0
                    M               = None
                    
                    def matvec(v):
                        return jnp.matmul(a, v) + effective_sigma * v
                    
                    if precond_apply is not None:
                        def M(r):
                            # Pass dynamic 'a' to the preconditioner so it can 
                            # re-compute diagonal/factors inside the JIT trace.
                            return precond_apply(r, a)

                    return callable_comp(matvec, b, x0_val, tol, maxiter, M, a=a, sigma=effective_sigma) # IF THE SOLVER NEEDS a, sigma

                return jax.jit(wrapper_logic, static_argnames=static_argnames)
        
            # NQS / Fisher (Dynamic S, Sp, and now dynamic sigma!)
            # This allows S/Sp AND sigma to change every step without re-compiling.
            elif use_fisher:
                # For Fisher/Gram mode, preconditioners from get_apply_gram() expect (r, s, sp).
                # We always wrap to curry s, s_p since we're in Gram mode.
                # The wrapping happens inside the traced function to capture dynamic s, s_p.
                
                def wrapper_logic(s, s_p, b, x0, tol, maxiter, precond_apply=None, sigma=None, snr_tol=None):
                    # Use runtime sigma if provided, otherwise fall back to default
                    effective_sigma = default_sigma if sigma is None else sigma
                    x0_val          = jnp.zeros_like(b) if x0 is None else x0
                    n_samples       = s.shape[0] if s.shape[0] > s.shape[1] else s.shape[1]
                    
                    def matvec(v):
                        inter       = jnp.matmul(s, v)
                        final       = jnp.matmul(s_p, inter)
                        return (final / n_samples) + (effective_sigma * v)
                    
                    # Wrap preconditioner to curry s, s_p for Gram-mode preconditioners
                    # In Fisher mode, precond_apply from get_apply_gram() expects (r, s, sp)
                    M = None
                    if precond_apply is not None:
                        def M(r):
                            # Pass dynamic 's' and 's_p' to the preconditioner.
                            # _setup_gram_kernel will run HERE,
                            # extracting the diagonal (s_p @ s) inside the JIT trace efficiently.
                            return precond_apply(r, s, s_p)
                        
                    return callable_comp(matvec, b, x0_val, tol, maxiter, M, s=s, s_p=s_p, sigma=effective_sigma) # IF THE SOLVER NEEDS s, s_p, sigma
                
                return jax.jit(wrapper_logic, static_argnames=static_argnames)
            
            else: 
                def wrapper_logic(matvec, b, x0, tol, maxiter, precond_apply=None, sigma=None, snr_tol=None):
                    x0_val          = jnp.zeros_like(b) if x0 is None else x0

                    effective_sigma = default_sigma if sigma is None else sigma

                    def mv(v):
                        return matvec(v) + effective_sigma * v

                    return callable_comp(mv, b, x0_val, tol, maxiter, precond_apply)
                
                # Add 'matvec' to static args
                return jax.jit(wrapper_logic, static_argnames=static_argnames + ('matvec',))
            
        else:
            # Note: For NumPy, we just create Python wrappers. 
            # Actual compilation (Numba) happens inside callable_comp if supported.
            default_sigma = 0.0 if sigma is None else float(sigma)
            
            if use_matrix:
                def wrapper_np(a, b, x0, tol, maxiter, precond_apply=None, sigma=None, **kwargs):
                    effective_sigma = default_sigma if sigma is None else sigma
                    x0_val          = np.zeros_like(b) if x0 is None else x0
                    def mv(v):      return a @ v + effective_sigma * v
                    
                    return callable_comp(mv, b, x0_val, tol=tol, maxiter=maxiter, precond_apply=precond_apply)
                return wrapper_np
            
            elif use_fisher:
                def wrapper_np(s, s_p, b, x0, tol, maxiter, precond_apply=None, sigma=None, **kwargs):
                    effective_sigma = default_sigma if sigma is None else sigma
                    # In standard NumPy, efficient matvec creation is simple lambda
                    n = s.shape[0]
                    def mv(v): 
                        return (s_p @ (s @ v)) / n + effective_sigma * v
                    
                    x0_val = np.zeros_like(b) if x0 is None else x0
                    
                    # Wrap preconditioner to curry s, s_p for Gram-mode preconditioners
                    # In Fisher mode, precond_apply from get_apply_gram() expects (r, s, sp)
                    wrapped_precond = None
                    if precond_apply is not None:
                        def wrapped_precond(r):
                            return precond_apply(r, s, s_p)
                    
                    return callable_comp(mv, b, x0_val, tol=tol, maxiter=maxiter, precond_apply=wrapped_precond)
                return wrapper_np

            else:
                # Wrapper for matvec case to handle sigma and x0
                def wrapper_np(matvec, b, x0, tol, maxiter, precond_apply=None, sigma=None, **kwargs):
                    x0_val          = np.zeros_like(b) if x0 is None else x0
                    effective_sigma = default_sigma if sigma is None else sigma

                    if effective_sigma != 0:
                        def mv(v): return matvec(v) + effective_sigma * v
                        mv_to_use = mv
                    else:
                        mv_to_use = matvec

                    return callable_comp(mv_to_use, b, x0_val, tol, maxiter, precond_apply)
                return wrapper_np
    
    # -------------------------------------------------------------------------
    #! Static Solve Interface (Core Requirement)
    # -------------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def solve(
            # Core Problem Definition
            matvec          : MatVecFunc,
            b               : Array,
            x0              : Array,
            # Solver Parameters
            *,              # Enforce keyword arguments
            tol             : float,
            maxiter         : int,
            # Optional Preconditioner, This is the function r -> M^{-1}r
            precond_apply   : Optional[Callable[[Array], Array]] = None,
            # Backend Specification
            backend_module  : Any,
            # Solver Specific Arguments
            **kwargs        : Any
            ) -> SolverResult:
        """
        Abstract Static:
            Solves the linear system Ax = b using a specific algorithm.

        Requires all inputs explicitly. Concrete implementations (e.g., `CgSolver.solve`)
        contain the actual algorithm for the specified backend.

        Parameters:
        -----------
            matvec : Callable[[Array], Array]
                Function implementing the matrix-vector product A @ x.
                It must accept a vector of shape (N,) and return a vector of shape (N,).
                Must be compatible with `backend_module` (NumPy or JAX).
            b : Array
                Right-hand side vector of shape (N,). Must be a `backend_module` array.
            x0 : Array
                Initial guess vector of shape (N,). Must be a `backend_module` array.
            tol : float
                Relative convergence tolerance (||Ax - b|| / ||b||).
            maxiter : int
                Maximum number of iterations allowed.
            precond_apply : Callable[[Array], Array], optional
                Function applying the preconditioner M^{-1}.
                Takes a vector `r` of shape (N,) and returns `M^{-1}r` of shape (N,).
                Must be compatible with `backend_module`.
            backend_module : module
                The numerical backend module to use for array operations (e.g., `numpy` or `jax.numpy`).
                This allows the solver logic to be backend-agnostic.
            **kwargs : Any
                Additional solver-specific keyword arguments (e.g., `restart` for GMRES).

        Returns:
            SolverResult:
                A named tuple containing:
                - `x` (Array): The computed solution vector of shape (N,).
                - `converged` (bool): True if the solver reached the desired tolerance.
                - `iterations` (int): The number of iterations performed.
                - `residual_norm` (float): The norm of the final residual (||b - Ax||).

        Raises:
            NotImplementedError:
                If a subclass hasn't implemented this method.
            SolverError:
                If convergence fails catastrophically or inputs are invalid.
        """
        raise NotImplementedError(str(SolverErrorMsg.METHOD_NOT_IMPL))

    @staticmethod
    @abstractmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None, **kwargs) -> StaticSolverFunc:
        """
        Abstract Static:
            Retrieves the solver function, which may be JIT-compiled (with JAX),
            Numba-compiled, or a plain Python function based on the provided backend_module.
        
        Args:
            backend_module:
            The numerical backend (e.g., numpy or jax.numpy) used for any necessary compilation.
        
        Returns:
            StaticSolverFunc:
            A callable with the signature:
                (matvec, b, x0, tol, maxiter, precond_apply, **kwargs) -> SolverResult
        
        Note:
            The backend_module helps in tailoring the solver function for the appropriate numerical library.
        """
        raise NotImplementedError(str(SolverErrorMsg.METHOD_NOT_IMPL))
    
    # -------------------------------------------------------------------------
    #! Static Helpers for Creating MatVec Functions
    # -------------------------------------------------------------------------

    @staticmethod
    def _compile_helper_jax(func: Callable):
        '''
        Internal helper to apply JIT.
        '''
        return jax.jit(func)
    
    @staticmethod
    def _compile_helper_np(func: Callable):
        '''
        Internal helper Numba.
        '''
        return numba.njit(func)
    
    # -------------------------------------------------------------------------
    #! Static Helpers for Creating MatVec Functions
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _compile_helper(func: Callable, backend_module: Any) -> Callable:
        """
        Internal helper to apply JIT or Numba based on backend.
        Parameters:
            func (Callable):
                a function to compile
            backend_module:
                np or jax...
        Returns:
            A compiled function.
        """
        
        # get the function name for convenience
        func_name = getattr(func, '__name__', '<anonymous_lambda>')
        
        if JAX_AVAILABLE and backend_module is jnp:
            if jax is None:
                raise SolverError(SolverErrorMsg.COMPILATION_NA, "JAX not available for JIT.")
            print(f"(Solver) JIT compiling function {func_name}...")
            return Solver._compile_helper_jax(func)
        # if the module is numpy, use Numba
        elif backend_module is np:
            if numba is None:
                print(f"Warning: Numba not available, cannot compile function {func_name} for NumPy.")
                return func
            print(f"(Solver) Numba compiling function {func_name}...")
            
            try:
                return Solver._compile_helper_np(func)
            except numba.NumbaError as e:
                print(f"Warning: Numba compilation failed for {func_name}: {e}. Returning original function.")
                return func
            except Exception as e:
                print(f"Warning: Exception occurred in {func_name}: {e}. Returning original function.")
                return func
        else:
            return func # Unknown backend
    
    # -------------------------------------------------------------------------
    #! Convenience Instance Method (Wrapper around Static Solve)
    # -------------------------------------------------------------------------

    def _form_gram_matrix(self) -> Array:
        """
        Forms the Gram matrix A = (Sp @ S) / N if the configuration is set for Gram matrix computation.
        
        Returns:
        --------
            Array: The computed Gram matrix.
        Raises:
            SolverError: If the required components for Gram matrix computation are not set.
        """
        if self._conf_s is not None and self._conf_sp is not None:
            print(f"({self.__class__.__name__}) Forming Gram matrix A = (Sp @ S) / N.")
            n_size = self._conf_s.shape[0]
            if n_size > 0:
                norm_factor = float(n_size) if n_size > 0 else 1.0
                return (self._conf_sp @ self._conf_s) / norm_factor
            
            # otherwise, return without division
            return (self._conf_sp @ self._conf_s)
        else:
            raise SolverError(SolverErrorMsg.MAT_NOT_SET, "Required components for Gram matrix computation are not set.")

    def _check_precond_solve(self, precond):
        """
        Validates and processes the provided preconditioner for the solver.
        This method checks the type and compatibility of the given preconditioner
        and returns a callable function to apply the preconditioner if valid.
        
        ---
        Args:
            precond (Union[Preconditioner, Callable, None, str]):
                The preconditioner to be used.
                    - If 'default', it is treated as None.
                    - If an instance of `Preconditioner`, it ensures compatibility with the solver's backend.
                    - If callable, it is assumed to be a valid preconditioner function.
                    - If None, no preconditioner is applied.        
        Returns:
            Optional[Callable[[Array], Array]]:
                A callable function to apply the preconditioner,
                or None if no valid preconditioner is provided.
        Raises:
            TypeError:
                If the provided preconditioner is not of a valid type.
        """

        # Initialize the preconditioner apply function
        precond_apply_func: Optional[Callable[[Array], Array]] = None
        # Can be Instance, Callable, None, 'default'
        actual_precond_source = precond

        # check if is 'default' string
        if actual_precond_source == 'default':
            actual_precond_source = None

        # Check type and process accordingly
        if isinstance(actual_precond_source, Preconditioner):
            # Ensure preconditioner uses the same backend
            if actual_precond_source.backend_str != self.backend_str:
                print(f"Warning: Preconditioner backend '{actual_precond_source.backend_str}' "
                    f"differs from solver backend '{self.backend_str}'. Resetting preconditioner.")
                actual_precond_source.reset_backend(self._backend)
                
            # Use the instance's compiled apply function
            precond_apply_func = actual_precond_source
        elif callable(actual_precond_source):
            # Assume it's the correct r -> M^{-1}r function
            precond_apply_func = actual_precond_source
            
        elif actual_precond_source is not None:
            raise TypeError(f"Invalid 'precond' type: {type(actual_precond_source)}. Expected Preconditioner, Callable, None, or 'default'.")
        
        return precond_apply_func
    

    # -------------------------------------------------------------------------
    #! Instance Method to Run the Solver
    # -------------------------------------------------------------------------

    def solve_instance(self,
                    b               : Array,
                    x0              : Optional[Array]   = None,
                    *,
                    # Overrides for this call
                    tol             : Optional[float]   = None,
                    maxiter         : Optional[int]     = None,
                    precond         : Union[Preconditioner, Callable[[Array], Array], None] = 'default',
                    sigma           : Optional[float]   = None,
                    compile_matvec  : bool              = False,
                    # Kwargs for the static solver
                    **kwargs) -> SolverResult:
        """
        Convenience instance method to run the solver.

        Sets up `matvec` and `precond_apply` based on instance configuration
        (if provided during __init__) or arguments, then calls the static `solve`
        method of this solver's class. Stores the result in instance attributes.

        Args:
            b (Array):
                Right-hand side vector.
            x0 (Optional[Array]):
                Initial guess. Defaults to zeros.
            tol (Optional[float]):
                Tolerance override. Uses instance default if None.
            maxiter (Optional[int]):
                Max iterations override. Uses instance default if None.
            precond (Union[Preconditioner, Callable, None, str]):
                Preconditioner for this solve.
                    - If `Preconditioner` instance:
                        Uses its `__call__` method.
                    - If `Callable`:
                        Assumes it's `r -> M^{-1}r` and uses it directly.
                    - If None:
                        No preconditioning.
                    - If 'default':
                        Uses None...
            sigma (Optional[float]):
                Regularization for `matvec` creation *if* `matvec`
                is not already defined for the instance. Uses
                instance default `_conf_sigma` if None.
            **kwargs:
                Additional arguments passed directly to the static `solve`.

        Returns:
            SolverResult:
                Result from the static solve method.
        """
        
        # Determine Mode (Priority: Matvec -> Matrix -> Fisher)
        # This order is important. If user provided matvec_func, we use it.
        # If user provided S/Sp (NQS), we use Fisher mode.
        
        use_matvec_func = (self._conf_matvec_func is not None)                  # Matvec function case    
        use_matrix      = (self._conf_a is not None and not self._conf_is_gram) # Dense A case
        use_fisher      = (self._conf_s is not None)                            # NQS case
        
        # Retrieve or Create Cached Kernel
        # We only create the kernel ONCE per configuration type.
        if self._cached_solver_func is None:
            self._cached_solver_func = self.get_solver_func(
                                        self._backend,
                                        use_matvec  =   use_matvec_func,
                                        use_fisher  =   use_fisher,
                                        use_matrix  =   use_matrix,
                                        sigma       =   self._conf_sigma
                                    )

        # Prepare Arguments
        tol         = tol       or self._default_eps
        maxiter     = maxiter   or self._default_maxiter
        precond     = precond   or self._default_precond 
        
        # Preconditioner
        precond_fn  = None
        if isinstance(precond, Preconditioner):
            precond_fn = precond.get_apply()
        elif callable(precond):
            precond_fn = precond

        # Dispatch based on Mode
        
        # Case A: General Matvec
        if use_matvec_func:
            result = self._cached_solver_func(self._conf_matvec_func, b, x0, tol, maxiter, precond_fn, sigma=sigma)
            
        # Case B: NQS Fisher (Dynamic S, Sp)
        elif use_fisher:
            # Allow S/Sp to be overridden by kwargs (critical for NQS loop)
            s_dyn   = kwargs.get('s', self._conf_s)
            sp_dyn  = kwargs.get('s_p', self._conf_sp)
            
            # Auto-conjugate if sp is missing
            if sp_dyn is None and s_dyn is not None:
                sp_dyn = self._backend.conjugate(s_dyn.T)

            result = self._cached_solver_func(s_dyn, sp_dyn, b, x0, tol, maxiter, precond_fn)
            
        # Case C: Dense Matrix
        elif use_matrix:
            a_dyn   = kwargs.get('a', self._conf_a)
            result  = self._cached_solver_func(a_dyn, b, x0, tol, maxiter, precond_fn)
            
        else:
            raise SolverError(SolverErrorMsg.INVALID_INPUT, "Solver not configured. Provide matvec_func, matrix A, or Fisher S/Sp.")

        # Store last result
        self._last_solution         = result.x
        self._last_converged        = result.converged
        self._last_iterations       = result.iterations
        self._last_residual_norm    = result.residual_norm
        
        return result

    # -------------------------------------------------------------------------
    #! Properties for Last Result
    # -------------------------------------------------------------------------
    
    @property
    def solution(self) -> Optional[Array]:      return self._last_solution
    @property
    def converged(self) -> Optional[bool]:      return self._last_converged
    @property
    def iterations(self) -> Optional[int]:      return self._last_iterations
    @property
    def residual_norm(self) -> Optional[float]: return self._last_residual_norm

    # -------------------------------------------------------------------------
    #! Properties for Configuration (Read-only access)
    # -------------------------------------------------------------------------
    
    @property
    def backend_str(self) -> str:               return self._backend_str
    @property
    def dtype(self) -> Type:                    return self._dtype
    @property
    def default_eps(self) -> float:             return self._default_eps
    @property
    def default_maxiter(self) -> int:           return self._default_maxiter

    # -------------------------------------------------------------------------

    def __repr__(self) -> str:                  return f"{self.__class__.__name__}(type={self._solver_type.name if self._solver_type else 'Unknown'}')"
    def __str__(self) -> str:                   return self.__repr__()

# -----------------------------------------------------------------------------

@maybe_jit
def _sym_ortho(a, b, backend):
    '''
    Performs a stable symmetric Householder (Givens) reflection for complex numbers.

    Parameters
    ----------
    a : scalar
        The first element of the two-vector [a; b].
    b : scalar
        The second element of the two-vector [a; b].
    backend : module
        The numerical backend module (numpy or jax.numpy).

    Returns
    -------
    c, s, r : scalar
        The rotation coefficients and the norm.
    '''
    _absa   = backend.abs(a)
    _absb   = backend.abs(b)
    if b == 0:
        return 1, 0, a # c = 1, s = 0, r = a
    elif a == 0:
        return 0, 1, b # c = 0, s = 1, r = b
    elif _absb > _absa:
        tau = a / b
        c   = 1 / backend.sqrt(1 + tau * tau)
        s   = backend.sign(b) * c
        r   = b / s
        return c, s, r
    # |a| >= |b|
    tau = b / a
    c   = backend.sign(a) / backend.sqrt(1 + tau * tau)
    s   = c * tau
    r   = a / c
    return c, s, r

def sym_ortho(a, b, backend: str = "default"):
    """
    Stable symmetric Householder (Givens) reflection.
    
    Computes parameters c, s, r such that:
    
        [ c  s ] [ a ] = [ r ]
        [ s -c ] [ b ]   [ 0 ]
    
    For real inputs, r = sqrt(a^2 + b^2) is nonnegative.
    For complex inputs, r preserves the phase of a (if b==0) or b (if a==0),
    and the reflectors are computed in a stable manner.
    
    Parameters
    ----------
    a : scalar (real or complex)
        The first element of the two-vector [a; b].
    b : scalar (real or complex)
        The second element of the two-vector [a; b].
    backend : str, optional (default "default")
        Specifies which backend to use. If set to "jax", the function uses
        jax.numpy and is jitted for speed.
    
    Returns
    -------
    (c, s, r) : tuple of scalars
        The computed reflection parameters satisfying:
            c = a / r   and   s = b / r,
        with r = sqrt(a^2 + b^2) for real numbers (or the appropriately phased value for complex).

    Numerical stability
    -------------------
    This function avoids overflow and underflow by scaling by the larger magnitude component
    (either `|a|` or `|b|`). This ensures that the intermediate calculations of `tau`
    and the hypotenuse do not exceed floating-point range limits unnecessarily.
    """
    # Select the numerical backend: jax.numpy if "jax" is chosen; otherwise, NumPy.
    
    # select backend
    backend = get_backend(backend)

    # Promote types if necessary for the backend
    # Example: Ensure floats for division/sqrt
    is_complex = isinstance(a, complex) or isinstance(b, complex) or \
                (hasattr(backend, 'iscomplexobj') and (backend.iscomplexobj(a) or backend.iscomplexobj(b)))

    if not is_complex and isinstance(a, (int, float)) and isinstance(b, (int, float)):
        a = float(a)
        b = float(b)
        
    # Add promotion to complex type if needed by backend
    elif is_complex:
        a = backend.astype(a, backend.complex128) # Or appropriate complex type
        b = backend.astype(b, backend.complex128)

    return _sym_ortho(a, b, backend)

# -----------------------------------------------------------------------------
