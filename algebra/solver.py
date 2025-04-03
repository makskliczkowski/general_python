import numpy as np
import scipy as sp
from abc import ABC, abstractmethod
from enum import Enum, auto, unique                 # for enumerations

# -----------------------------------------------------------------------------

from typing import Optional, Callable
from general_python.algebra.utils import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend
from general_python.algebra.preconditioners import Preconditioner

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

_SOL_TYPE_ERROR     = f"Unknown solver type: must be one of {', '.join([s.name for s in SolverType])}"

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

    def __repr__(self):
        return self.__str__()
class SolverError(Exception):
    '''
    Base class for exceptions in the solver module.
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

# -----------------------------------------------------------------------------
#! General Solver Class
# -----------------------------------------------------------------------------

class Solver(ABC):
    '''
    Abstract base class for all solvers of linear systems of type Ax = b
    or with a preconditioner M, M^{-1}Ax = M^{-1}b.
    '''

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
                maxrestarts : int               = 1):
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
        '''
        
        # setup backend first - to be handled by classes
        self._solver_type                       = 0
        self._backend_str                       = backend
        self._backend, self._backend_sp         = get_backend(backend, scipy=True)
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
        self._mat_vec_mult: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
        self._solver_func                       = lambda b, x0, precond: x0
        
        # restarts
        self._restart                           = restart
        self._maxrestarts                       = maxrestarts
        self._restarts                          = 0
        
        # solution
        self._solution: Optional[np.ndarray]    = None
        self._a                                 = None # matrix, only used when mat_vec_mult is not set
        self._s                                 = None # matrix S, only used when mat_vec_mult is not set
        self._sp                                = None # matrix S^T, only used when mat_vec_mult is not set
    
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
        if self._mat_vec_mult is None and self._a is not None:
            self.init_from_matrix(self._a, np.zeros(self._n))
        elif self._mat_vec_mult is None and self._sp is not None and self._s is not None:
            self.init_from_fisher(self._s, self._sp, np.zeros(self._n))
        elif self._mat_vec_mult is None:
            raise SolverError(SolverErrorMsg.MATMULT_NOT_SET)
    
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
        
    def set_preconditioner(self, precond: Preconditioner) -> None:
        '''
        Sets the preconditioner.
        '''
        self._preconditioner = precond
    
    def set_preconditioner_sigma(self, sigma: float) -> None:
        '''
        Sets the regularization parameter for the preconditioner.
        '''
        if self._preconditioner is not None:
            self._preconditioner.sigma(sigma)
        
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

    def _wrap_function(self, func: Callable) -> Callable:
        """
        Wrap the given function with jit if the chosen backend is JAX.
        Parameters:
        func : callable
            The function to be wrapped.
        """
        if self._backend_str.lower() == 'jax' or self._backend_str.lower() == 'jnp':
            return maybe_jit(func)
        else:
            return func

    # -------------------------------------------------------------------------
    #! Abstract methods that use only the right-hand side vector b (and optionally x0)
    # -------------------------------------------------------------------------

    def init(self, b: np.ndarray, x0: Optional[np.ndarray] = None) -> None:
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
    
    @abstractmethod
    def solve(self, b: np.ndarray, x0: Optional[np.ndarray] = None,
                precond: Optional[Preconditioner] = None) -> np.ndarray:
        """
        Abstract solve routine. Subclasses must use the pre-set matrix-vector multiplication
        function (self._mat_vec_mult) and right-hand side vector b to compute the solution.
        """
        
        if precond is not None:
            self._preconditioner = precond
            
    # -------------------------------------------------------------------------
    #! specific initialization routines
    # -------------------------------------------------------------------------
    
    def init_from_matrix(self,
                        a,
                        b      : np.ndarray,
                        x0     : Optional[np.ndarray] = None,
                        sigma  : float = None) -> None:
        """
        Initialize the solver from a dense or sparse matrix A.
        The operation performed is: A*x + sigma*x.
        Parameters:
        a : array-like
            The matrix A.
        b : array-like
            The right-hand side vector.
        x0 : array-like, optional
            Initial guess for the solution.        
        """
        
        # reference to the matrix - do not copy!
        self._a = a
        

        self._s  = None  # this should be set when using the Fisher matrix
        self._sp = None  # this should be set when using the Fisher matrix
        

        if sigma is not None:
            def mv(x, sig: float = sigma):
                return self._a.dot(x) + sig * x
            self._mat_vec_mult = self._wrap_function(mv)
        else:
            self._mat_vec_mult = self._wrap_function(self._a.dot)

        self._n = b.shape[0]
        self.init(b, x0)
        
    def init_from_fisher(self,
                        s,
                        sp,
                        b,
                        x0      : Optional[np.ndarray]  = None,
                        sigma   : float                 = None,
                        set_a   : bool                  = False) -> None:
        '''
        Initialize the solver from the Fisher matrix.
        The operation performed is for matrix A = S^T * S + sigma * I. but the matrix is not stored.
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
        '''
        self._gram  = True
        self._n     = b.shape[0]
        self._s     = s     # this should be set when using the Fisher matrix - do not copy!
        self._sp    = sp    # this should be set when using the Fisher matrix - do not copy!
        if set_a:
            self._a = self._sp @ self._s / self._n
            
        if sigma is not None:
            def mv(x, sig: float = sigma):
                intermediate = s.dot(x)
                return sp.dot(intermediate) / self._n + sig * x
            self._mat_vec_mult = self._wrap_function(mv)
        else:
            def mv(x):
                intermediate = s.dot(x)
                return sp.dot(intermediate) / self._n
            self._mat_vec_mult = self._wrap_function(mv)
        self.init(b, x0)
        
    def init_from_function(self,
                        func    : Callable,
                        b,
                        x0      : Optional[np.ndarray] = None) -> None:
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
        self._mat_vec_mult  = self._wrap_function(func)
        self._n             = b.shape[0]
        self._a             = None
        self._s             = None
        self._sp            = None
        self.init(b, x0)
        
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
