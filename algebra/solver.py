import numpy as np
import numba
import scipy as sp
from typing import Optional, Callable, Union, Any
from abc import ABC, abstractmethod
from enum import Enum, auto, unique                 # for enumerations
from typing import NamedTuple                       # Alternative to dataclass

# -----------------------------------------------------------------------------

from general_python.algebra.utils import _JAX_AVAILABLE, maybe_jit, get_backend
from general_python.algebra.preconditioners import Preconditioner, PreconitionerApplyFun

try:
    if _JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
        Array   = Union[np.ndarray, jnp.ndarray]
    else:
        jax     = None
        jnp     = None
        Array   = np.ndarray
except ImportError:
    jax     = None
    jnp     = None
    Array   = np.ndarray

# -----------------------------------------------------------------------------

@unique
class SolverType(Enum):
    """
    Enumeration class for the different types of solvers.
    """
    DIRECT          = auto() # Direct solver x = A^-1 b
    BACKEND_SOLVER  = auto() # Use the default backend solver
    PSEUDO_INVERSE  = auto() # Pseudo-inverse solver
    
    # scipy based solvers
    SCIPY_DIRECT    = auto() # Direct solver - using scipy (or jax equivalent)
    SCIPY_CJ        = auto() # Conjugate gradient - using scipy (or jax equivalent)
    SCIPY_MINRES    = auto() # Minimum residual - using scipy (or jax equivalent)
    # my solvers
    CJ              = auto() # Conjugate gradient
    MINRES          = auto() # Minimum residual
    MINRES_QLP      = auto() # Minimum residual - using QLP
    ARNOLDI         = auto() # Arnoldi iteration (basis generation, might be used by other solvers like GMRES)

# -----------------------------------------------------------------------------

_SOL_TYPE_ERROR             = f"Unknown solver type: must be one of {', '.join([s.name for s in SolverType])}"

# -----------------------------------------------------------------------------
#! Errors
# -----------------------------------------------------------------------------

SOL_ERR_MSG_MATMULT_NOT_SET = "Matrix-vector multiplication function not set."
SOL_ERR_MSG_MAT_NOT_SET     = "Matrix A not set."
SOL_ERR_MSG_MAT_S_NOT_SET   = "Matrix S not set."
SOL_ERR_MSG_MAT_SP_NOT_SET  = "Matrix S^T not set."
SOL_ERR_MSG_CONV_FAILED     = "Convergence failed after maximum number of iterations."
SOL_ERR_MSG_DIM_MISMATCH    = "Dimension mismatch in the linear system."
SOL_ERR_MSG_PSEUDOINV       = "Pseudo-inverse solver failed to converge."
SOL_ERR_MSG_MAT_SINGULAR    = "Matrix is singular."
SOL_ERR_MSG_METHOD_NOT_IMPL = "The subclass has to implement this method..."

class SolverErrorMsg(Enum):
    '''
    Enumeration class for solver error messages.
    '''
    MATMULT_NOT_SET = 101
    MAT_NOT_SET     = 102
    MAT_S_NOT_SET   = 103
    MAT_SP_NOT_SET  = 104
    CONV_FAILED     = 105
    DIM_MISMATCH    = 106
    PSEUDOINV       = 107
    MAT_SINGULAR    = 108
    METH_NOT_IMPL   = 109
    
    def __str__(self):
        match (self):
            case SolverErrorMsg.MATMULT_NOT_SET:
                return SOL_ERR_MSG_MATMULT_NOT_SET
            case SolverErrorMsg.MAT_NOT_SET:
                return SOL_ERR_MSG_MAT_NOT_SET
            case SolverErrorMsg.MAT_S_NOT_SET:
                return SOL_ERR_MSG_MAT_S_NOT_SET
            case SolverErrorMsg.MAT_SP_NOT_SET:
                return SOL_ERR_MSG_MAT_SP_NOT_SET
            case SolverErrorMsg.CONV_FAILED:
                return SOL_ERR_MSG_CONV_FAILED
            case SolverErrorMsg.DIM_MISMATCH:
                return SOL_ERR_MSG_DIM_MISMATCH
            case SolverErrorMsg.PSEUDOINV:
                return SOL_ERR_MSG_PSEUDOINV
            case SolverErrorMsg.MAT_SINGULAR:
                return SOL_ERR_MSG_MAT_SINGULAR
            case SolverErrorMsg.METH_NOT_IMPL
                return SOL_ERR_MSG_METHOD_NOT_IMPL

    def __repr__(self):
        return self.__str__()

class SolverError(Exception):
    '''
    Base class for exceptions in the solver module. It 
    shall handle the messages and codes of the errors.
    Attributes:
        message : str
            The error message.
        code : SolverErrorMsg
            The error code.
    '''

    def __init__(self, code: SolverErrorMsg):
        ''' Initialize the solver error with the given message and code. '''
        self.message    = str(code)
        self.code       = code
        super().__init__(self.message)
    
    def __str__(self):
        if self.code is not None:
            return f"[Error {self.code}]: {self.message}"
        return self.message
    
    def __repr__(self):
        return self.__str__()

class SolverResult(NamedTuple):
    '''
    Named tuple to store the result of a solver.
    
    Attributes:
        x : np.ndarray
            The solution vector.
        converged : bool
            Whether the solver converged.
        iterations : int
            The number of iterations performed.
        error : float
            The error of the solution.
    '''
    x          : Array                  # The solution vector
    converged  : bool                   # Whether the solver converged
    iterations : int                    # The number of iterations performed
    error      : float                  # The error of the solution
    res_norm   : Optional[float] = None # The residual norm (if available)

# -----------------------------------------------------------------------------
#! General Solver Class
# -----------------------------------------------------------------------------

MatVecFun           = Callable[[Array, float], Array]
SolverFun           = Callable[[MatVecFun, Array, Array, float, int, PreconitionerApplyFun], SolverResult]

# numpy backend functions
if True:
    
    # ---
    # just return x man
    def mat_vec_function_idn(_: Any):
        '''
        Standard interface example for the matrix-vector multiplication function
        $$
        Ax = b
        $$
        
        Parameters:
            a : np.ndarray
                The matrix A.
        Returns:
            Function -> np.ndarray
                The result of A @ x.
        '''
        def wrap(x):
            return x
        return wrap
    
    # ---
    # standard Matrix-vector multiplication
    def mat_vec_function(a: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Matrix-vector multiplication function.
        
        $$
        Ax = b
        $$
        
        Parameters:
        a : np.ndarray
            The matrix A.
        sigma : Optional[float]
            add identity to the matrix with a parameter $\sigma$
            to the matrix \tilde{A} = A + \sigma * \hat{I}
        Returns:
            Function -> np.ndarray
                The result of A @ x.
        """
        if sigma is not None:
            def wrap(x):
                return a @ x + sigma * x
            return wrap
        def wrap(x):
            return a @ x
        return wrap
    
    # ---
    # matrix-vector multiplication with a preconditioner
    def mat_vec_function_precond(a      : np.ndarray,
                                p_apply : PreconitionerApplyFun,
                                sigma   : Optional[float] = None) -> np.ndarray:
        """
        Matrix-vector multiplication function with a preconditioner.
        
        $$
        A x + M^{-1}Ax,
        $$
        where M is the preconditioner.
        
        Parameters:
        a : np.ndarray
            The matrix A.
        p_apply : PreconitionerApplyFun
            The function to apply the preconditioner.
        sigma : Optional[float]
            The 
        Returns:
            Function -> np.ndarray
                The result of A @ x + M^{-1}Ax
        """
        if sigma is not None:
            def wrap(x):
                r = a @ x
                return r + p_apply(r) + sigma * x
            return wrap
        def wrap(x):
            r = a @ x
            return r + p_apply(r)
        return wrap
    
    # ---
    # matrix-vector multiplication function with a Fisher type matrix
    # (provided S and S+, where A = S^+S)
    def mat_vec_function_fisher(smat    : np.ndarray,
                                spmat   : np.ndarray,
                                sigma   : Optional[float] = None) -> np.ndarray:
        """
        Matrix-vector multiplication function for a Fisher type matrix.
        
        $$
        Ax = b, where A = S^\dag S / n
        $$
        
        Parameters:
        smat : np.ndarray
            Part of the Fisher matrix
        spmat : np.ndarray
            The conjugate transpose of the Fisher matrix
        sigma : 
        Returns:
            Function -> np.ndarray
                The result of A @ x, where A = S^\dag S / n
        """
        if sigma is not None:
            def wrap(x):
                n               = smat.shape[1]
                intermediate    = smat @ x
                return spmat @ intermediate / n + sigma * x
            return wrap
        def wrap(x):
            n               = smat.shape[1]
            intermediate    = smat @ x
            return spmat @ intermediate / n
        return wrap
    
    # ---
    # matrix-vector multiplication function with a Fisher type matrix using a preconditioner
    def mat_vec_function_fisher_precond(smat    : np.ndarray,
                                        spmat   : np.ndarray,
                                        p_apply : PreconitionerApplyFun,
                                        sigma   : Optional[float] = None) -> np.ndarray:
        """
        Matrix-vector multiplication function for a Fisher type matrix with a preconditioner.
        
        $$
        Ax = b, where A = S^\dag S / n
        $$
        
        Parameters:
        smat : np.ndarray
            Part of the Fisher matrix
        spmat : np.ndarray
            The conjugate transpose of the Fisher matrix
        p_apply : PreconitionerApplyFun
            The function to apply the preconditioner.
        x : np.ndarray
            The vector x.
        Returns:
            Function -> np.ndarray
                The result of A @ x, where A = S^\dag S / n
        """
        
        if sigma is not None:
            def wrap(x):
                n               = smat.shape[1]
                intermediate    = smat @ x
                r               = spmat @ intermediate / n
                return r + p_apply(r) + sigma * x
            return wrap
        def wrap(x):
            n               = smat.shape[1]
            intermediate    = smat @ x
            r               = spmat @ intermediate / n
            return r + p_apply(r)
        return wrap
    
# -----------------------------------------------------------------------------



class Solver(ABC):
    '''
    Abstract base class for all solvers of linear systems of type Ax = b
    or with a preconditioner M, M^{-1}Ax = M^{-1}b.
    
    The solver can be initialized with a matrix A, a matrix-vector multiplication
    function, or a Fisher matrix S. The solver can also be initialized with
    a preconditioner M. The solver can be used to solve the linear system
    Ax = b or M^{-1}Ax = M^{-1}b.
    
    There are several methods to initialize the solver:
    - init_from_matrix(a, b, x0=None, sigma=None): Initializes the solver from a matrix A.
    - init_from_fisher(s, sp, b, x0=None, sigma=None): Initializes the solver from a Fisher matrix S.
    - init_from_function(func, b, x0=None): Initializes the solver from a matrix-vector multiplication function.
    
    The implmentation of the solver shall provide static (potentially compiled methods
    for the backend) and dynamic methods (Python functions) to solve the linear system.
    
    It follows the usage of numpy, scipy solvers and jax solvers.
    
    Subclasses implement specific algorithms (e.g., CG, MINRES) as static methods.
    The primary interface is the `solve` static method.
    '''

    # -------------------------------------------------------------------------
    
    _solver_type    : Optional[SolverType] = None
    
    # -------------------------------------------------------------------------
    
    def __init__(self,
                backend                         = 'default',
                size        : int               = 1,
                dtype                           = None,
                eps         : float             = 1e-10,
                maxiter     : int               = 1000,
                reg         : float             = None,
                precond     : Preconditioner    = None,
                restart     : bool              = False,
                maxrestarts : int               = 1,
                matvecfun   : Optional[MatVecFun] = None,
                solverfun   : Optional[SolverFun] = None):
        '''
        Initializes the solver with the given backend and size.
        
        Parameters:
        backend : str, optional
            The backend to be used for computation. Default is 'default'.
        size : int, optional
            The size of the linear system. Default is 1.
        dtype : data type, optional
            The data type of the matrix and vectors. Default is None.
        eps : float, optional
            The convergence tolerance. Default is 1e-10.
        maxiter : int, optional
            The maximum number of iterations for the solver. Default is 1000.
        reg : float, optional
            Regularization parameter. Default is None.
        precond : Preconditioner, optional
            Preconditioner to be used. Default is None.
        restart : bool, optional
            Whether to enable restarts. Default is False.
        maxrestarts : int, optional
            Maximum number of restarts. Default is 1.
        matvecfun : MatVecFun, optional
            The matrix-vector multiplication function. Default is None.
        solverfun : SolverFun, optional
            The solver function. Default is None.
        '''
        
        # setup backend first - to be handled by classes
        self._backend_str                       = backend
        self._backend, self._backend_sp         = get_backend(backend, scipy=True)
        self._isjax                             = _JAX_AVAILABLE and self._backend != np
        self._dtype                             = dtype if dtype is not None else self._backend_sp.float32
        
        # flags
        self._symmetric                         = False
        self._gram                              = False
        self._converged                         = False
        
        # size of the matrix
        self._n                                 = size
        self._iter                              = 0
        self._maxiter                           = maxiter
        self._eps                               = eps
        self._reg                               = reg
        
        # preconditioner - for better solutions
        self._preconditioner                    = precond
        
        # matrix-vector multiplication function (as a linear operator)
        self._mat_vec_func: Optional[MatVecFun] = matvecfun
        self._solve_func: SolverFun             = self._wrap_function(self.solve) if solverfun is None else solverfun
        self._precond_func: Optional[PreconitionerApplyFun] = None
        # restarts
        self._restart                           = restart
        self._maxrestarts                       = maxrestarts
        self._restarts                          = 0
        
        # solution
        self._solution: Optional[Array]         = None
        self._a                                 = None # matrix, only used when mat_vec_mult is not set
        self._s                                 = None # matrix S, only used when mat_vec_mult is not set
        self._sp                                = None # matrix S^T, only used when mat_vec_mult is not set
    
    # -------------------------------------------------------------------------
    
    def check_mat_or_matvec(self, needs_matrix = False):
        '''
        Check if either the matrix or matrix-vector multiplication function is set.
        
        Note:
        
        This is a helper function for the derived classes and only some classes
        use it.
        
        Parameters:
        needs_matrix : bool, optional
            Whether the matrix is needed. Default is False.
        Raises:
            SolverError : If neither the matrix nor the matrix-vector multiplication function is set.
        '''
        
        if self._a is None and needs_matrix:
            if self._s is not None and self._sp is not None:
                self._a = self._sp @ self._s / self._n
                return
            else:
                raise SolverError(SolverErrorMsg.MAT_NOT_SET)
            
        if not needs_matrix:
            return
        
        # check if the matrix-vector multiplication function is set
        if self._mat_vec_func is None and self._a is not None:
            self.init_from_matrix(self._a, np.zeros(self._n))
        elif self._mat_vec_func is None and self._sp is not None and self._s is not None:
            self.init_from_fisher(self._s, self._sp, np.zeros(self._n))
        elif self._mat_vec_func is None:
            raise SolverError(SolverErrorMsg.MATMULT_NOT_SET)
    
    # -------------------------------------------------------------------------
    
    @staticmethod
    @abstractmethod
    def solve(
            # Core Problem Definition
            matvec          : MatVecFun,
            # Problem Definition
            b               : Array,
            x0              : Array,
            # Solver Parameters
            *,              # Enforce keyword arguments for parameters
            tol             : float,
            maxiter         : int,
            # Optional Preconditioner
            precond_apply   : Optional[PreconitionerApplyFun] = None,
            # Solver Specific Arguments
            **kwargs        : Any) -> SolverResult:
        """
        Solves the linear system Ax = b using a specific algorithm.
        
        Note:
            If the preconditioner is provided, the solver aims to 
            improve the solution stability and convergence by solving 
            
        This is a static method requiring all inputs to be provided explicitly.

        Args:
            matvec (Callable):
                The matrix-vector product function (A @ x). Must be compatible
                with the specified backend.
            b:
                The right-hand side vector. Must be a numpy or jnp array.
            x0:
                The initial guess vector. Must be a `backend_module` array.
            tol:
                Relative convergence tolerance (||Ax - b|| / ||b||).
            maxiter:
                Maximum number of iterations.
            precond_apply:
                A function to apply the preconditioner M^-1 (optional).

        Returns:
            A SolverResult object containing the solution, convergence status,
            iterations run, and final residual norm.

        Raises:
            SolverError:
                If convergence fails or inputs are invalid.
            NotImplementedError:
                If a subclass hasn't implemented this method.
        """
        raise NotImplementedError(SolverErrorMsg.METH_NOT_IMPL)
    
    def get_apply_p(self) -> PreconitionerApplyFun:
        ''' Returns the preconditioner application function '''
        return self._precond_func
    
    def get_apply(self) -> MatVecFun:
        '''
        Returns the preconditioner apply function.
        '''
        return self._mat_vec_func
    
    def get_solve(self) -> SolverFun:
        '''
        Returns the solver function.
        '''
        return self._solve_func
    
    # -------------------------------------------------------------------------
    #! Getters and properties
    # -------------------------------------------------------------------------
    
    @property
    def backend_str(self) -> str:
        '''
        Returns the backend used for computation.
        '''
        return self._backend_str
    
    @property
    def dtype(self) -> np.dtype:
        '''
        Returns the data type used for computation.
        '''
        return self._dtype
    
    @property
    def size(self) -> int:
        '''
        Returns the size of the linear system.
        '''
        return self._n
    
    @property
    def maxiter(self) -> int:
        '''
        Returns the maximum number of iterations.
        '''
        return self._maxiter
    
    @property
    def eps(self) -> float:
        '''
        Returns the convergence tolerance.
        '''
        return self._eps
    
    @property
    def reg(self) -> float:
        '''
        Returns the regularization parameter.
        '''
        return self._reg
    
    @property
    def preconditioner(self) -> Optional[Preconditioner]:
        '''
        Returns the preconditioner used.
        '''
        return self._preconditioner
    
    @property
    def restart(self) -> bool:
        '''
        Returns whether restarts are enabled.
        '''
        return self._restart
    
    @property
    def maxrestarts(self) -> int:
        '''
        Returns the maximum number of restarts.
        '''
        return self._maxrestarts
    
    @property
    def restarts(self) -> int:
        '''
        Returns the number of restarts.
        '''
        return self._restarts
    
    @property
    def solution(self) -> Optional[np.ndarray]:
        '''
        Returns the solution of the linear system.
        '''
        return self._solution
    
    @property
    def matrix(self) -> Optional[np.ndarray]:
        '''
        Returns the matrix A used for the linear system.
        '''
        return self._a
    
    @property
    def s(self) -> Optional[np.ndarray]:
        '''
        Returns the matrix S used for the linear system.
        '''
        return self._s
    
    @property
    def sp(self) -> Optional[np.ndarray]:
        '''
        Returns the matrix S^T used for the linear system.
        '''
        return self._sp
    
    @property
    def symmetric(self) -> bool:
        '''
        Returns whether the matrix is symmetric.
        '''
        return self._symmetric
    
    @property
    def gram(self) -> bool:
        '''
        Returns whether the matrix is the Gram matrix.
        '''
        return self._gram
    
    @property
    def converged(self) -> bool:
        '''
        Returns whether the solver has converged.
        '''
        return self._converged
    
    @property
    def sigma(self) -> Optional[float]:
        '''
        Returns the regularization parameter.
        '''
        return self._reg
    
    def get_restarts(self) -> int:
        '''
        Returns the number of restarts.
        '''
        return self._restarts
    
    def get_maxrestarts(self) -> int:
        '''
        Returns the maximum number of restarts.
        '''
        return self._maxrestarts
    
    def get_maxiter(self) -> int:
        '''
        Returns the maximum number of iterations.
        '''
        return self._maxiter
    
    # -------------------------------------------------------------------------
    #! Setters
    # -------------------------------------------------------------------------
    
    @backend_str.setter
    def backend_str(self, backend: str) -> None:
        '''
        Sets the backend for computation.
        '''
        self._backend_str = backend
        self._backend, self._backend_sp = get_backend(backend, scipy=True)
        
    @eps.setter
    def eps(self, eps: float) -> None:
        '''
        Sets the convergence tolerance.
        '''
        self._eps = eps
        
    @maxiter.setter
    def maxiter(self, maxiter: int) -> None:
        '''
        Sets the maximum number of iterations.
        '''
        self._maxiter = maxiter
        
    @reg.setter
    def reg(self, reg: float) -> None:
        '''
        Sets the regularization parameter.
        '''
        self._reg = reg
        
    @preconditioner.setter
    def preconditioner(self, precond: Preconditioner) -> None:
        '''
        Sets the preconditioner.
        '''
        self._preconditioner = precond

    @matrix.setter
    def matrix(self, a: np.ndarray) -> None:
        '''
        Sets the matrix A used for the linear system.
        '''
        self._a = a
        self._n = a.shape[1]
        
    @s.setter
    def s(self, s: np.ndarray) -> None:
        '''
        Sets the matrix S used for the linear system.
        '''
        self._s = s
        
    @sp.setter
    def sp(self, sp: np.ndarray) -> None:
        '''
        Sets the matrix S^T used for the linear system.
        '''
        self._sp = sp
        
    @solution.setter
    def solution(self, x: np.ndarray) -> None:
        '''
        Sets the solution of the linear system.
        '''
        self._solution = x
    
    def set_matrix_vector(self, matvec: MatVecFun) -> None:
        '''
        Sets the matrix-vector multiplication function.
        It additionally compiles it.
        
        Parameters:
            matvec: Callable
                matrix vector multiplication
        '''
        self._mat_vec_func = self._wrap_function(matvec)
    
    def set_preconditioner(self, precond: Preconditioner) -> None:
        '''
        Sets the preconditioner.
        Parameters:
            precond: Preconditioner
                the preconditioner for the linear solve
        '''
        self._preconditioner    = precond
        self._precond_func      = precond.get_apply()
    
    def set_preconditioner_sigma(self, sigma: float) -> None:
        '''
        Sets the regularization parameter for the preconditioner.
        '''
        if self._preconditioner is not None:
            self._preconditioner.sigma = sigma
        
    def set_matrix(self, a: np.ndarray) -> None:
        '''
        Sets the matrix A used for the linear system.
        '''
        self._a = a
        self._n = a.shape[1]
    
    def next_restart(self) -> None:
        '''
        Increments the number of restarts.
        '''
        self._restarts += 1
    
    def check_restart_up(self) -> bool:
        '''
        Checks whether the maximum number of restarts has been reached.
        '''
        return self._restarts > self._maxrestarts
    
    def increase_reg(self, factor = 1.1):
        '''
        Increases the regularization parameter.
        '''
        if self._reg is not None:
            self._reg *= factor
    
    # -------------------------------------------------------------------------

    def _wrap_function(self, func: MatVecFun) -> Callable:
        """
        Wrap the given function with either 
        jax.jit or numba.njit
        
        Parameters:
            func : MatVecFun
                The function to be wrapped.
        Returns:
            A compiled function...
        """
        if _JAX_AVAILABLE and self._isjax:
            return jax.jit(func)
        else:
            return numba.njit(func)

    # -------------------------------------------------------------------------
    #! Abstract methods that use only the right-hand side vector b (and optionally x0)
    # -------------------------------------------------------------------------

    def init(self, b: Array, x0: Optional[Array] = None) -> None:
        """
        Abstract initialization routine that is called once the
        matrix-vector multiplication function has been set.
        This method must initialize any solver-specific data using b (and x0 if provided).
        
        Parameters:
        b : array-like
            The right-hand side vector.
        
        x0 : array-like, optional
            Initial guess for the solution.
        """
        self._n = b.shape[0]
        
        if x0 is not None:
            self._solution = x0
        else:
            self._solution = self._backend.zeros(self._n, dtype=self._dtype)
    
    def solve_class(self,
                    b       : Array,
                    x0      : Optional[Array] = None,
                    precond : Optional[Preconditioner] = None,
                    **kwargs) -> SolverResult:
        """
        Solves a linear system of equations using the specified solver function.
        Parameters:
            b (Array):
                The right-hand side vector of the linear system.
            x0 (Optional[Array], optional):
                The initial guess for the solution. Defaults to None.
            precond (Optional[Preconditioner], optional):
                A preconditioner to improve convergence. Defaults to None.
            **kwargs:
                Additional keyword arguments passed to the solver function.
        Returns:
            SolverResult: The result of the solver, including the solution and metadata.
        Notes:
            - If a preconditioner is provided, it will be set before solving the system.
            - The solver function is defined by `self._solve_func` and operates on the matrix-vector product function `self._mat_vec_func`.
        """
        
        if precond is not None:
            self.set_preconditioner(precond)
        return self._solve_func(self._mat_vec_func, b, x0,
                    self._eps, self._maxiter, self._precond_func, **kwargs)
    
    # -------------------------------------------------------------------------
    #! specific initialization routines
    # -------------------------------------------------------------------------
    
    def init_from_matrix(self, a, sigma: Optional[float] = None) -> None:
        """
        Initialize the solver from a dense or sparse matrix A.
        The operation performed is: A*x + sigma*x.
        Parameters:
        a : array-like
            The matrix A.
        """
        
        # reference to the matrix - do not copy!
        self._a     = a
        # self._s     = None  # this should be set when using the Fisher matrix
        # self._sp    = None  # this should be set when using the Fisher matrix
        self._n     = a.shape[1]
        if self._precond_func is not None:
            func = mat_vec_function_precond(self._a, self._precond_func, sigma)
        else:
            func = mat_vec_function(self._a, sigma)
        self.set_matrix_vector(func)
        
    def init_from_fisher(self,
                        s       : Array,
                        sp      : Array,
                        sigma   : float = None,
                        set_a   : bool  = False) -> None:
        '''
        Initialize the solver from the Fisher matrix.
        The operation performed is for matrix A = S^\dag * S + sigma * I. but the matrix is not stored.
        Parameters:
        s : array-like
            The matrix S.
        sp : array-like
            The matrix S^\dag
        sigma : float, optional
            Regularization parameter. Default is None. 
        '''
        self._gram  = True
        self._n     = s.shape[1]
        self._s     = s     # this should be set when using the Fisher matrix - do not copy!
        self._sp    = sp    # this should be set when using the Fisher matrix - do not copy!
        if set_a:
            return self.init_from_matrix(self._sp @ self._s / self._n, sigma)
        if self._precond_func is not None:
            func = mat_vec_function_fisher_precond(self._s, self._sp,
                                                self._precond_func, sigma)
        else:
            func = mat_vec_function_fisher(self._s, self._sp, sigma)
        self._a     = None
    
    def init_from_function(self,
                        func    : Callable,
                        n       : int) -> None:
        '''
        Initialize the solver from a matrix-vector multiplication function.
        Parameters:
        func : callable
            The matrix-vector multiplication function.
        b : array-like
            The right-hand side vector.
        x0 : array-like, optional
            Initial guess for the solution.
        '''
        self._mat_vec_func  = self._wrap_function(func)
        self._n             = n
        self._a             = None
        self._s             = None
        self._sp            = None
        
    # -------------------------------------------------------------------------
    #! specific solve routines
    # -------------------------------------------------------------------------
    
    def solve_from_matrix(self,
                        a,
                        b,
                        x0      : Optional[np.ndarray] = None,
                        precond : Optional[Preconditioner] = None) -> np.ndarray:
        '''
        Solve the linear system Ax = b with the given matrix A.
        Parameters:
        a : array-like
            The matrix A.
        b : array-like
            The right-hand side vector.
        x0 : array-like, optional
            Initial guess for the solution.
        precond : Preconditioner, optional
            Preconditioner to be used. Default is None.
        Returns:
        array-like
            The solution x.
        '''
        self.init_from_matrix(a, b, x0)
        return self.solve(b, x0, precond)
    
    def solve_from_fisher(self,
                        s,
                        sp,
                        b,
                        x0      : Optional[np.ndarray] = None,
                        sigma   : float = None,
                        precond : Optional[Preconditioner] = None) -> np.ndarray:
        '''
        Solve the linear system Ax = b with the given Fisher matrix.
        Parameters:
        s : array-like
            The matrix S.
        sp : array-like
            The matrix S^T.
        b : array-like
            The right-hand side vector.
        x0 : array-like, optional
            Initial guess for the solution.
        sigma : float, optional
            Regularization parameter. Default is None.
        precond : Preconditioner, optional
            Preconditioner to be used. Default is None.
        Returns:
        array-like
            The solution x.
        '''
        self.init_from_fisher(s, sp, b, x0, sigma)
        return self.solve(b, x0, precond)
    
    def solve_from_function(self,
                        func    : Callable,
                        b,
                        x0      : Optional[np.ndarray] = None,
                        precond : Optional[Preconditioner] = None) -> np.ndarray:
        '''
        Solve the linear system Ax = b with the given matrix-vector multiplication function.
        Parameters:
        func : callable
            The matrix-vector multiplication function.
        b : array-like
            The right-hand side vector.
        x0 : array-like, optional
            Initial guess for the solution.
        precond : Preconditioner, optional
            Preconditioner to be used. Default is None.
        Returns:
        array-like
            The solution x.
        '''
        self.init_from_function(func, b, x0)
        return self.solve(b, x0, precond)
    
    # -------------------------------------------------------------------------

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

    # Determine if the inputs are complex.
    # is_complex = backend.iscomplexobj(a) or backend.iscomplexobj(b)
    
    return _sym_ortho(a, b, backend)

# -----------------------------------------------------------------------------
