'''
file:       general_python/algebra/solver.py
author:     Maksymilian Kliczkowski

Defines the abstract interface and helper structures for solving linear systems

$$
Ax = b,
$$

potentially using preconditioning

$$
M^{-1}Ax = M^{-1}b \rm {[left]},
$$

or

$$
A M^{-1} y = b, x = M^{-1}y \rm {[right]},
$$ 

Supports different backends (NumPy, JAX) and promotes a static-method-based
interface for solver algorithms, allowing for easier compilation (e.g., JIT).
'''

import numpy as np
import numba
import scipy as sp
from functools import partial
from typing import Optional, Callable, Union, Any, NamedTuple, Type, TypeAlias
from abc import ABC, abstractmethod
from enum import Enum, auto, unique                 # for enumerations

# -----------------------------------------------------------------------------

from general_python.algebra.utils import _JAX_AVAILABLE, get_backend, maybe_jit
from general_python.algebra.preconditioners import Preconditioner, PreconitionerApplyFun, preconditioner_idn

try:
    if _JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
        Array : TypeAlias = Union[np.ndarray, jnp.ndarray]
    else:
        jax     = None
        jnp     = None
        Array : TypeAlias = np.ndarray
except ImportError:
    jax     = None
    jnp     = None
    Array : TypeAlias = np.ndarray

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
    BACKEND_SOLVER  = auto() # Use the default backend solver
    PSEUDO_INVERSE  = auto() # Pseudo-inverse solver
    # Wrappers around backend/scipy solvers (might use instance methods)
    SCIPY_CG        = auto()
    SCIPY_MINRES    = auto()
    SCIPY_GMRES     = auto()
    SCIPY_DIRECT    = auto()
    # my solvers
    CG              = auto() # Conjugate gradient
    MINRES          = auto() # Minimum residual
    MINRES_QLP      = auto() # Minimum residual - using QLP
    GMRES           = auto() # Generalized Minimal Residual
    ARNOLDI         = auto() # Arnoldi iteration for building the Krylov subspaces
    
# -----------------------------------------------------------------------------
#! Errors
# -----------------------------------------------------------------------------

class SolverErrorMsg(Enum):
    '''
    Enumeration class for solver error messages.
    '''
    MATVEC_FUNC_NOT_SET = 101
    MAT_NOT_SET         = 102
    MAT_S_NOT_SET       = 103
    MAT_SP_NOT_SET      = 104
    CONV_FAILED         = 105
    DIM_MISMATCH        = 106
    PSEUDOINV_FAILED    = 107
    MAT_SINGULAR        = 108
    METHOD_NOT_IMPL     = 109
    PRECOND_INVALID     = 110
    BACKEND_MISMATCH    = 111
    INVALID_INPUT       = 112
    COMPILATION_NA      = 113
    
    def __str__(self):
        # Simple default string conversion
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

# -----------------------------------------------------------------------------
#! General Solver Abstract Base Class
# -----------------------------------------------------------------------------

class Solver(ABC):
    '''
    Abstract base class for linear system solvers
    
    $$
    Ax = b.
    $$
    
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
                a               : Optional[Array]                 = None,
                s               : Optional[Array]                 = None,
                s_p             : Optional[Array]                 = None,
                matvec_func     : Optional[MatVecFunc]            = None,
                sigma           : Optional[float]                 = None,
                is_gram         : bool                            = False
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
        self._conf_sigma                = sigma
        self._conf_is_gram              = is_gram

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
        if new_backend_str == 'default' and _JAX_AVAILABLE:
            new_backend_str = 'jax'
        else:
            new_backend_str = 'numpy'
        self._backend_str               = new_backend_str
        self._backend, self._backend_sp = get_backend(self._backend_str, scipy=True)
        self._isjax                     = _JAX_AVAILABLE and self._backend is not np

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

        Args:
            matvec:
                Function implementing the matrix-vector product A @ x.
                Must be compatible with `backend_module`.
            b:
                Right-hand side vector (as `backend_module` array).
            x0:
                Initial guess vector (as `backend_module` array).
            tol:
                Relative convergence tolerance (||Ax - b|| / ||b||).
            maxiter:
                Maximum number of iterations.
            precond_apply:
                Function applying the preconditioner M^{-1} (optional).
                Takes `r` and returns `M^{-1}r`. Must be compatible
                with `backend_module`.
            backend_module:
                The numerical backend module (e.g., `numpy` or `jax.numpy`).
            **kwargs:
                Additional solver-specific keyword arguments.

        Returns:
            SolverResult:
                Named tuple with solution, convergence status, iterations, residual norm.

        Raises:
            NotImplementedError:
                If a subclass hasn't implemented this method.
            SolverError:
                If convergence fails or inputs are invalid within implementation.
        """
        raise NotImplementedError(str(SolverErrorMsg.METHOD_NOT_IMPL))

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
          - if use_fisher=True: s, s_p, b, x0, *, tol, maxiter, precond_apply, **kwargs
          - if use_fisher=False: a, b, x0, *, tol, maxiter, precond_apply, **kwargs
        '''
        if use_matvec:
            # Assume callable_comp is already appropriately jitted or handled
            return callable_comp

        # --- Define which arguments to the wrapper MUST be static for JAX ---
        # - Functions passed as arguments: 'precond_apply'
        # - Control parameters often treated as static: 'tol', 'maxiter'
        # - We also need to make kwargs static if we pass them through **kwargs
        #   to callable_comp *within* the jitted function, as JAX cannot trace
        #   arbitrary dictionary contents. A common pattern is to require kwargs
        #   to be hashable/pytrees or handle them outside the JIT boundary if possible.
        #   For simplicity here, let's assume callable_comp's JIT handles its own kwargs,
        #   or that kwargs passed to wrapper are not essential for the core JIT compilation
        #   of the matvec creation and the call itself. The crucial ones are tol, maxiter, precond_apply.
        #
        # Note on 'sigma', 'use_matrix', 'callable_comp': These are values from the outer scope
        # used *inside* the wrapper. JAX's JIT typically handles these closures correctly,
        # capturing their values when the wrapper is defined and jitted. They don't need
        # to be listed in static_argnames unless they were *arguments* to the wrapper itself.

        static_argnames_tuple = ('tol', 'maxiter', 'precond_apply')

        if use_fisher:
            if backend_module == np:

                # @np.njit !TODO: should apply?
                def wrapper_np_fisher(s, s_p, b, x0, *, tol, maxiter, precond_apply=None, **kwargs):
                    # Create the matvec function (or matrix if use_matrix=True implies that)
                    matvec = Solver.create_matvec_from_fisher_np(s, s_p, sigma=sigma, create_full=use_matrix)
                    return callable_comp(matvec, b, x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply, **kwargs)
                
                # Numba requires default for optional function args
                wrapper_np_fisher.__defaults__  = (None,) # For precond_apply
                _the_wrapper                    = wrapper_np_fisher
            else:
                # JAX version
                @partial(jax.jit, static_argnames=static_argnames_tuple)
                def wrapper_jax_fisher(s, s_p, b, x0, *, tol, maxiter, precond_apply=None, **kwargs):
                    # Create the matvec function (or matrix)
                    matvec = Solver.create_matvec_from_fisher_jax(s, s_p, sigma=sigma, create_full=use_matrix)
                    return callable_comp(matvec, b, x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply, **kwargs)
                _the_wrapper = wrapper_jax_fisher
                
        else:
            if backend_module == np:
                # @np.njit
                def wrapper_np_matrix(a, b, x0, *, tol, maxiter, precond_apply=None, **kwargs):
                    matvec = Solver.create_matvec_from_matrix_np(a, sigma=sigma)
                    return callable_comp(matvec, b, x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply, **kwargs)
                wrapper_np_matrix.__defaults__ = (None,)
                _the_wrapper = wrapper_np_matrix
            else:
                # JAX version
                @partial(jax.jit, static_argnames=static_argnames_tuple)
                def wrapper_jax_matrix(a, b, x0, *, tol, maxiter, precond_apply=None, **kwargs):
                    matvec = Solver.create_matvec_from_matrix_jax(a, sigma=sigma)
                    return callable_comp(matvec, b, x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply, **kwargs)
                _the_wrapper = wrapper_jax_matrix

        # Add a default for precond_apply=None in the signature for clarity
        # (though JAX handles Optional better than Numba typically)
        _the_wrapper.__defaults__ = (None,) * len(static_argnames_tuple) # Set defaults for static args if needed, though keyword-only args handle None ok
        return _the_wrapper
    
    @staticmethod
    def decide_solver_func(backend_module,
                        matvec          : Optional[Union[Callable, bool]]    = True,
                        a               : Optional[Union[Callable, bool]]    = None,
                        s               : Optional[Union[Array, bool]]       = None,
                        s_p             : Optional[Union[Array, bool]]       = None,
                        sigma           : Optional[Union[float, bool]]       = None):
        """
        Determines and returns the appropriate solver function based on the provided arguments.

        The function chooses the solver function by evaluating the inputs in a prioritized manner:
        1. If 'matvec' is provided and evaluates to True, it selects the solver function that uses a matrix-vector product.
        2. If 'matvec' is not True and 's' is provided:
            - If 's_p' is also provided (i.e., not None), it selects the solver function corresponding to the combination of 's' and 's_p'.
            - Otherwise, it selects the solver function based solely on 's'.
        3. If none of these conditions are met, a RuntimeError is raised, indicating that the solver function could not be determined.

        Parameters:
            backend_module:
                The module providing backend solver functions.
            matvec (Optional[Union[Callable, bool]]):
                A callable or flag indicating usage of a matrix-vector product based solver if set to a truthy value.
            a (Optional[Union[Array, bool]]):
                An arr
            s (Optional[Union[Array, bool]]):
                An array or flag representing a solver parameter used when 'matvec' is not activated.
            s_p (Optional[Union[Array, bool]]):
                An additional parameter used in conjunction with 's' to determine the specific solver function.
            sigma (Optional[Union[float, bool]]):
                An auxiliary parameter that may be required by some solver functions.

        Returns:
            The solver function as determined by the input parameters.

        Raises:
            RuntimeError: If neither 'matvec' nor 's' is provided (or valid), thereby making it impossible to decide on a solver function.
        """

        if matvec:
            return Solver.get_solver_func(backend_module, True, False, False, sigma)
        
        if a is not None:
            return Solver.get_solver_func(backend_module, False, False, True, sigma)
        elif s is not None:
            if s_p is not None:
                return Solver.get_solver_func(backend_module, False, True, False, sigma)
            else:
                return Solver.get_solver_func(backend_module, False, True, True, sigma)
        else:
            raise RuntimeError("Cannot parse the solver function, all options are None...")
    
    @staticmethod
    def run_solver_func(
                        bck     : Any,
                        func    : Callable,
                        *args,
                        matvec  : Optional[Callable] = None,
                        a       : Optional[Array]    = None,
                        s       : Optional[Array]    = None,
                        s_p     : Optional[Array]    = None,
                        **kwargs) -> SolverResult:
        """
        Executes a solver function based on the provided arguments.

        ---
        Parameters:
            bck (Any):
                A backend or context object used for operations.
            func (Callable):
                The solver function to be executed.
            matvec (Optional[Callable], optional):
                A callable representing a matrix-vector operation. 
                If provided, it will be passed to `func`. Defaults to None.
            a (Optional[Array], optional):
                An array to be passed to `func` if `matvec` is not provided. 
                Defaults to None.
            s (Optional[Array], optional):
                An array to be passed to `func` along with `s_p` if both 
                `matvec` and `a` are not provided. Defaults to None.
            s_p (Optional[Array], optional):
                A secondary array to be used with `s`. If not provided, 
                it is computed as the conjugate transpose of `s` using the `bck` object. Defaults to None.
            *args:
                Additional positional arguments to be passed to `func`.
            **kwargs:
                Additional keyword arguments to be passed to `func`.

        Returns:
            SolverResult: The result of the solver function execution.

        Raises:
            RuntimeError: If none of the required arguments (`matvec`, `a`, or `s`) are provided.
        """
        if matvec is not None:
            return func(matvec, *args, **kwargs)
        if a is not None:
            return func(a, *args, **kwargs)
        if s is not None:
            if s_p is None:
                s_p = bck.conjugate(s_p.T)
            return func(s, s_p, *args, **kwargs)
        raise RuntimeError("Cannot run the solver function. Wrong arguments")
    
    @staticmethod
    @abstractmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool = True,
                        use_fisher      : bool = False,
                        use_matrix      : bool = False,
                        sigma           : Optional[float] = None) -> StaticSolverFunc:
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
        
        # if the module is jax, use JIT
        if _JAX_AVAILABLE and backend_module is jnp:
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
    
    @staticmethod
    def create_matvec_from_matrix_jax(
            a               : Array,
            sigma           : Optional[float]   = None) -> MatVecFunc:
        """
        Static Helper:
            Creates matvec function `x -> (A + sigma*I) @ x`.
        """

        mat_op = jnp.asarray(a)
        if sigma is not None and sigma != 0:
            #! Should Precompute A + sigma*I (potential memory cost for large A)?
            effective_a = mat_op + sigma * jnp.eye(mat_op.shape[0], dtype=mat_op.dtype)
            def matvec(x: Array) -> Array:
                '''
                Multiply a vector x by a matrix A
                '''
                return jnp.dot(effective_a, jnp.asarray(x))
            return Solver._compile_helper_jax(matvec)
        else:
            def matvec(x) -> Array:
                return jnp.dot(a, x)
            return Solver._compile_helper_jax(matvec)

    @staticmethod
    def create_matvec_from_matrix_np(
            a               : Array,
            sigma           : Optional[float]   = None) -> MatVecFunc:
        """
        Static Helper:
            Creates matvec function `x -> (A + sigma*I) @ x`.
        """

        mat_op = np.asarray(a)
        if sigma is not None and sigma != 0:
            #! Should Precompute A + sigma*I (potential memory cost for large A)?
            effective_a = mat_op + sigma * jnp.eye(mat_op.shape[0], dtype=mat_op.dtype)
            def matvec(x: Array) -> Array:
                '''
                Multiply a vector x by a matrix A
                '''
                return np.dot(effective_a, x)
            return Solver._compile_helper_np(matvec)
        else:
            def matvec(x) -> Array:
                return np.dot(a, x)
            return Solver._compile_helper_np(matvec)
    
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

        # take the module
        mat_op = backend_module.asarray(a)

        if sigma is not None and sigma != 0:
            #! Should Precompute A + sigma*I (potential memory cost for large A)?
            # effective_a = mat_op + sigma * backend_module.eye(mat_op.shape[0], dtype=mat_op.dtype)
            def matvec(x: Array) -> Array:
                '''
                Multiply a vector x by a matrix A
                '''
                effective_a = mat_op + sigma * backend_module.eye(mat_op.shape[0], dtype=mat_op.dtype)
                return backend_module.dot(effective_a, backend_module.asarray(x))
            
            print(f"Created matvec from matrix with sigma={sigma} using {backend_module.__name__}")
        else:
            def matvec(x: Array) -> Array:
                return backend_module.dot(mat_op, backend_module.asarray(x))
            print(f"Created matvec from matrix (no regularization) using {backend_module.__name__}")
        return Solver._compile_helper(matvec, backend_module) if compile_func else matvec
    
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
            - Computes the full matrix as (s @ s_p) / n and passes it along with `sigma` to Solver.create_matvec_from_matrix_jax,
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
            return Solver.create_matvec_from_matrix_jax(s @ s_p / n, sigma)

        s_op        = jnp.asarray(s)
        sp_op       = jnp.asarray(s_p)
        if sigma is not None and sigma != 0:
            def matvec(x: Array) -> Array:
                x_arr                   = jnp.asarray(x)
                s_dot_x                 = jnp.dot(s_op, x_arr)
                sp_dot_s_dot_x          = jnp.dot(sp_op, s_dot_x)
                return sp_dot_s_dot_x / n + sigma * x_arr
        else:
            def matvec(x: Array) -> Array:
                x_arr                   = jnp.asarray(x)
                s_dot_x                 = jnp.dot(s_op, x_arr)
                sp_dot_s_dot_x          = jnp.dot(sp_op, s_dot_x)
                return sp_dot_s_dot_x / n
        return Solver._compile_helper_jax(matvec)

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
            return Solver.create_matvec_from_matrix_np(s @ s_p / n, sigma)

        s_op        = np.asarray(s)
        sp_op       = np.asarray(s_p)
        if sigma is not None and sigma != 0:
            def matvec(x: np.ndarray) -> np.ndarray:
                x_arr                   = np.asarray(x)
                s_dot_x                 = np.dot(s_op, x_arr)
                sp_dot_s_dot_x          = np.dot(sp_op, s_dot_x)
                return sp_dot_s_dot_x / n + sigma * x_arr
        else:
            def matvec(x: np.ndarray) -> np.ndarray:
                x_arr                   = np.asarray(x)
                s_dot_x                 = np.dot(s_op, x_arr)
                sp_dot_s_dot_x          = np.dot(sp_op, s_dot_x)
                return sp_dot_s_dot_x / n
        return Solver._compile_helper_np(matvec)
    
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

        s_op                            = backend_module.asarray(s)
        sp_op                           = backend_module.asarray(s_p)
        if sigma is not None and sigma != 0:
            def matvec(x: Array) -> Array:
                x_arr                   = backend_module.asarray(x)
                s_dot_x                 = backend_module.dot(s_op, x_arr)
                sp_dot_s_dot_x          = backend_module.dot(sp_op, s_dot_x)
                return sp_dot_s_dot_x / norm_factor + sigma * x_arr
            print(f"Created Fisher matvec (Sp S / {norm_factor:.1f}) with sigma={sigma} using {backend_module.__name__}")
        else:
            def matvec(x: Array) -> Array:
                x_arr                   = backend_module.asarray(x)
                s_dot_x                 = backend_module.dot(s_op, x_arr)
                sp_dot_s_dot_x          = backend_module.dot(sp_op, s_dot_x)
                return sp_dot_s_dot_x / norm_factor
            print(f"Created Fisher matvec (Sp S / {norm_factor:.1f}) (no regularization) using {backend_module.__name__}")
        return Solver._compile_helper(matvec, backend_module) if compile_func else matvec

    # -------------------------------------------------------------------------
    #! Convenience Instance Method (Wrapper around Static Solve)
    # -------------------------------------------------------------------------

    def _form_gram_matrix(self) -> Array:
        """
        Forms the Gram matrix A = (Sp @ S) / N if the configuration is set for Gram matrix computation.
        Returns:
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

        
        precond_apply_func: Optional[Callable[[Array], Array]] = None
        # Can be Instance, Callable, None, 'default'
        actual_precond_source           = precond

        # check if is string
        if actual_precond_source == 'default':
            actual_precond_source       = None

        if isinstance(actual_precond_source, Preconditioner):
            # Ensure preconditioner uses the same backend
            if actual_precond_source.backend_str != self.backend_str:
                print(f"Warning: Preconditioner backend '{actual_precond_source.backend_str}' "
                    f"differs from solver backend '{self.backend_str}'. Resetting preconditioner.")
                actual_precond_source.reset_backend(self._backend)
            # Use the instance's compiled apply function
            precond_apply_func          = actual_precond_source
        elif callable(actual_precond_source):
            # Assume it's the correct r -> M^{-1}r function
            precond_apply_func          = actual_precond_source
        elif actual_precond_source is not None:
            raise TypeError(f"Invalid 'precond' type: {type(actual_precond_source)}. Expected Preconditioner, Callable, None, or 'default'.")
        return precond_apply_func
    
    def _check_matvec_solve(self,
                            current_sigma       : float,
                            current_backend_mod : Any,
                            compile_matvec      : bool,
                            **kwargs) -> MatVecFunc:
        """ 
        Internal: Determines the matvec function based on instance config.
        If a matrix is provided, it creates a matvec function from it.
        
        If a Fisher matrix is provided, it creates a matvec function from it.
        If a matvec function is already set, it uses that.
        """
        if self._conf_matvec_func is not None:
            matvec_func                 = self._conf_matvec_func
            # TODO: Consider if recompilation should happen if sigma differs?
        elif self._conf_a is not None and not self._conf_is_gram:
            matvec_func                 = self.create_matvec_from_matrix(
                                                self._conf_a, current_sigma, current_backend_mod, compile_func=compile_matvec)
        elif self._conf_s is not None and self._conf_sp is not None and self._conf_is_gram:
            create_full                 = kwargs.get("create_full", False)
            n_guess                     = self._conf_s.shape[0]
            matvec_func                 = self.create_matvec_from_fisher(
                                                self._conf_s, self._conf_sp,
                                                n_guess, current_sigma, current_backend_mod,
                                                compile_func=compile_matvec, create_full=create_full)
        else:
            raise SolverError(SolverErrorMsg.MATVEC_FUNC_NOT_SET, "Instance needs matvec func or matrices.")
        return matvec_func

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
        # Determine Parameters for this Call
        current_tol                     = tol if tol is not None else self._default_eps
        current_maxiter                 = maxiter if maxiter is not None else self._default_maxiter
        current_backend_mod             = self._backend # Use instance's backend

        # Determine MatVec Function
        current_sigma                   = sigma if sigma is not None else self._conf_sigma
        matvec_func : MatVecFunc        = self._check_matvec_solve(current_sigma,
                                                                    current_backend_mod,
                                                                    compile_matvec=compile_matvec,
                                                                    **kwargs)
        

        # Determine Preconditioner Apply Function
        precond_apply_func              = self._check_precond_solve(precond)
        
        # Prepare b and x0
        b_be                            = current_backend_mod.asarray(b, dtype=self._dtype)
        if x0 is None:
            x0_be                       = current_backend_mod.zeros_like(b_be)
        else:
            x0_be                       = current_backend_mod.asarray(x0, dtype=self._dtype)
            if x0_be.shape != b_be.shape:
                raise SolverError(SolverErrorMsg.DIM_MISMATCH, f"Shape mismatch: b={b_be.shape}, x0={x0_be.shape}")

        # Call the Static Solve Method of the Concrete Class
        print(f"({self.__class__.__name__}) Calling static solve with backend={self.backend_str}, tol={current_tol}, maxiter={current_maxiter}...")
        result = self.__class__.solve(
            matvec          = matvec_func,
            b               = b_be,
            x0              = x0_be,
            tol             = current_tol,
            maxiter         = current_maxiter,
            precond_apply   = precond_apply_func,
            backend_module  = current_backend_mod,
            **kwargs
        )

        # Store results in instance
        self._last_solution             = result.x
        self._last_converged            = result.converged
        self._last_iterations           = result.iterations
        self._last_residual_norm        = result.residual_norm

        print(f"({self.__class__.__name__}) Instance solve finished. Converged: {result.converged}, Iterations: {result.iterations}, Residual Norm: {result.residual_norm:.4e}")
        return result

    # -------------------------------------------------------------------------
    #! Properties for Last Result
    # -------------------------------------------------------------------------
    
    @property
    def solution(self) -> Optional[Array]:
        ''' What is the last solution? '''
        return self._last_solution
    
    @property
    def converged(self) -> Optional[bool]:
        ''' Is it converged solution? '''
        return self._last_converged
    
    @property
    def iterations(self) -> Optional[int]:
        ''' How many iterations? '''
        return self._last_iterations
    
    @property
    def residual_norm(self) -> Optional[float]:
        ''' What is the quality of the last result? '''
        return self._last_residual_norm

    # -------------------------------------------------------------------------
    #! Properties for Configuration (Read-only access)
    # -------------------------------------------------------------------------
    
    @property
    def backend_str(self) -> str:
        ''' Default backend string '''
        return self._backend_str
    
    @property
    def dtype(self) -> Type:
        ''' Default type of the arrays '''
        return self._dtype
    
    @property
    def default_eps(self) -> float:
        ''' Default error epsilon '''
        return self._default_eps
    
    @property
    def default_maxiter(self) -> int:
        ''' Default maximal number of iterations '''
        return self._default_maxiter

    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        ''' Returns the name and configuration of the solver. '''
        return f"{self.__class__.__name__}(type={self._solver_type.name if self._solver_type else 'Unknown'}, backend='{self.backend_str}')"

    def __str__(self) -> str:
        ''' Returns the name of the solver. '''
        return self.__repr__()

# -----------------------------------------------------------------------------

@maybe_jit
def _sym_ortho(a, b, backend):
    '''
    Performs a stable symmetric Householder (Givens) reflection for complex numbers.
    Parameters:
    a : scalar
        The first element of the two-vector [a; b].
    b : scalar
        The second element of the two-vector [a; b].
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
    
    For real inputs, r = sqrt(a² + b²) is nonnegative.
    For complex inputs, r preserves the phase of a (if b==0) or b (if a==0),
    and the reflectors are computed in a stable manner.
    
    Parameters:
        a : scalar (real or complex)
            The first element of the two-vector [a; b].
        b : scalar (real or complex)
            The second element of the two-vector [a; b].
        backend : str, optional (default "default")
            Specifies which backend to use. If set to "jax", the function uses
            jax.numpy and is jitted for speed.
    
    Returns:
        (c, s, r) : tuple of scalars
            The computed reflection parameters satisfying:
                c = a / r   and   s = b / r,
            with r = sqrt(a² + b²) for real numbers (or the appropriately phased value for complex).
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
