from typing import Optional, Callable
import numpy as np

from general_python.algebra.solver import SolverType, Solver, SolverError, SolverErrorMsg
from general_python.algebra.utils import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend as __backend
from general_python.algebra.preconditioners import Preconditioner

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import jax.lax as lax
    from jax import jit
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


# -----------------------------------------------------------------------------
# Conjugate Gradient Solver Class for symmetric positive definite matrices using SciPy
# -----------------------------------------------------------------------------

class CgSolverScipy(Solver):
    '''
    Conjugate Gradient Solver for symmetric positive definite matrices using SciPy.
    '''

    def __init__(self, backend='default', size = 1, dtype=None, eps = 1e-10, maxiter = 1000, reg = None, precond = None, restart = False, maxrestarts = 1):
        super().__init__(backend, size, dtype, eps, maxiter, reg, precond, restart, maxrestarts)
        self._symmetric     = True
        self._solver_type   = SolverType.SCIPY_CJ

    # -------------------------------------------------------------------------

    def solve(self, b, x0: Optional[np.ndarray] = None, precond: Optional[Preconditioner] = None):
        '''
        Solve the linear system Ax = b.
        Parameters:
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
        
        self.check_mat_or_matvec()
        
        m = None
        if precond is not None:
            self.set_preconditioner(precond)
            if self.reg is not None:
                self.set_preconditioner_sigma(self.reg)
            m = self._backend_sp.sparse.linalg.LinearOperator((self.size, self.size), matvec=precond.__call__)
            
        try:             
            return self._backend_sp.sparse.linalg.cg(self._mat_vec_mult, b, x0=x0, rtol=self._eps, maxiter=self._maxiter, M=m)
        except Exception as e:
            raise SolverError(SolverErrorMsg.MATRIX_INVERSION_FAILED) from e
        return None
    
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------
# Conjugate Gradient Solver Class for symmetric positive definite matrices not using SciPy
# -----------------------------------------------------------------------------

@jit
def solve_no_precond_jit(mat_vec_mult: Callable,
        b,
        x0          : Optional[jnp.ndarray] = None,
        _eps        : float = 1e-10,
        _maxiter    : int = 100):
    """
    Solve a linear system using the Conjugate Gradient (CG) algorithm with JIT compilation.
    Parameters:
        mat_vec_mult : Callable
            A function implementing the matrix-vector multiplication for the system matrix.
        b : array_like
            The right-hand side vector of the linear system.
        x0 : Optional[np.ndarray], default None
            The initial guess for the solution. If None, a zero vector with the same shape as b is used.
        _eps : float, default 1e-10
            The convergence tolerance. The algorithm stops when the squared residual is below this value.
        _maxiter : int, default 100
            The maximum number of iterations allowed.
    Returns:
        tuple
            A tuple containing:
            - x_final : np.ndarray
                The computed solution vector.
            - converged : bool
                A flag indicating whether the solution converged (True) or the maximum number of iterations was reached (False).
    Notes:
        This function uses JAX's JIT compilation and automatic vectorization (via lax.while_loop) to speed up the CG iteration.
    """

    x       = jnp.zeros_like(b) if x0 is None else x0
    r       = b - mat_vec_mult(x)
    rs_old  = jnp.dot(r, r)
    # The state is a 5-tuple: (iteration, x, r, p, rs_old)
    state   = (0, x, r, r, rs_old)
    
    def cond(state):
        '''
        Condition for the loop to stop iterating.
        '''
        i, _, _, _, rs_old = state
        # Continue if iteration count is below _maxiter and residual norm not converged.
        return jnp.logical_and(i < _maxiter, rs_old >= _eps)
    
    def body(state):
        '''
        The body of the loop. Implements the CG algorithm.
        '''
        i, x, r, p, rs_old  = state
        ap                  = mat_vec_mult(p)
        alpha               = rs_old / jnp.dot(p, ap)
        x_new               = x + alpha * p
        r_new               = r - alpha * ap
        rs_new              = jnp.dot(r_new, r_new)
        p_new               = r_new + (rs_new / rs_old) * p
        return (i + 1, x_new, r_new, p_new, rs_new)
    
    state_final                         = lax.while_loop(cond, body, state)
    _, x_final, r_final, _, rs_final    = state_final
    converged                           = rs_final < _eps
    return x_final, converged

@jit
def solve_precond_jit(mat_vec_mult : Callable,
        b,
        x0          : Optional[jnp.ndarray] = None,
        precond     : Optional[Preconditioner] = None,
        _eps        : float = 1e-10,
        _maxiter    : int   = 100):
    '''
    Solve a linear system using the Conjugate Gradient (CG) algorithm with JIT compilation and preconditioning.
    Parameters:        
        mat_vec_mult : Callable
            A function implementing the matrix-vector multiplication for the system matrix.
        b : array_like
            The right-hand side vector of the linear system.
        x0 : Optional[np.ndarray], default None
            The initial guess for the solution. If None, a zero vector with the same shape as b is used.
        precond : Optional[Preconditioner], default None
            The preconditioner to be used for the CG algorithm.
        _eps : float, default 1e-10
            The convergence tolerance. The algorithm stops when the squared residual is below this value.
        _maxiter : int, default 100
            The maximum number of iterations allowed.
    Returns:
        tuple
            A tuple containing:
            - x_final : np.ndarray
                The computed solution vector.
            - converged : bool
                A flag indicating whether the solution converged (True) or the maximum number of iterations was reached (False).
    '''
    
    x       = jnp.zeros_like(b) if x0 is None else x0
    r       = b - mat_vec_mult(x)
    z       = precond(r)
    p       = z
    rs_old  = jnp.dot(r, z)
    # The state is a 5-tuple: (iteration, x, r, p, rs_old)
    state   = (0, x, r, r, rs_old)
    
    def cond(state):
        '''
        Condition for the loop to stop iterating.
        '''
        i, _, r, _, _ = state
        return jnp.logical_and(i < _maxiter, jnp.dot(r, r) >= _eps)
    
    def body(state):
        '''
        The body of the loop. Implements the CG algorithm.
        '''
        i, x, r, p, rs_old  = state
        ap                  = mat_vec_mult(p)
        alpha               = rs_old / jnp.dot(p, ap)
        x_new               = x + alpha * p
        r_new               = r - alpha * ap
        z_new               = precond(r_new)
        rs_new              = jnp.dot(r_new, z_new)
        p_new               = z_new + (rs_new / rs_old) * p
        return (i + 1, x_new, r_new, p_new, rs_new)
    
    state_final                         = lax.while_loop(cond, body, state)
    _, x_final, r_final, _, rs_final    = state_final
    converged                           = jnp.dot(r_final, r_final) < _eps
    return x_final, converged

# -----------------------------------------------------------------------------

class CgSolver(Solver):
    '''
    A Conjugate Gradient (CG) solver for linear systems that supports both preconditioned 
    and non-preconditioned formulations. This solver can leverage JAX's JIT compilation 
    for improved performance or fall back to a traditional NumPy-based implementation.
    '''
    
    def __init__(self, backend='default', size = 1, dtype=None, eps = 1e-10, maxiter = 1000, reg = None, precond = None, restart = False, maxrestarts = 1):
        super().__init__(backend, size, dtype, eps, maxiter, reg, precond, restart, maxrestarts)
        self._symmetric     = True
        self._solver_type   = SolverType.CJ
    
    # -------------------------------------------------------------------------
    
    @staticmethod
    def solve_no_precond(
            mat_vec_mult : Callable,
            b,
            x0          : Optional[np.ndarray] = None,
            _eps        : float = 1e-10,
            _maxiter    : int   = 100):
        '''
        Core function to solve the linear system Ax = b through the Conjugate Gradient method 
        using JAX. This function is JIT-compiled and does not support precond.
        '''
        x       = np.zeros_like(b) if x0 is None else x0
        r       = b - mat_vec_mult(x)
        rs_old  = np.dot(r, r)
        
        # check convergence already
        if np.abs(rs_old) < _eps:
            return x, True
        
        p       = r
        for _ in range(_maxiter):
            ap      = mat_vec_mult(p)
            alpha   = rs_old / np.dot(p, ap)
            x       += alpha * p
            r       -= alpha * ap
            rs_new  = np.dot(r, r)
            
            if np.abs(rs_new) < _eps:
                return x, True
            
            p       = r + (rs_new / rs_old) * p
            rs_old  = rs_new
        
        return x, False
    
    # -------------------------------------------------------------------------
    
    @staticmethod
    def solve_precond(mat_vec_mult : Callable,
            b           : np.ndarray,
            x0          : Optional[np.ndarray]      = None, 
            precond     : Optional[Preconditioner]  = None,
            _eps        : float = 1e-10,
            _maxiter    : int   = 100):
        '''
        Core function to solve the linear system Ax = b through the Conjugate Gradient method via SciPy.
        '''
        x       = np.zeros_like(b) if x0 is None else x0
        r       = b - mat_vec_mult(x)
        z       = precond(r)
        p       = z
        rs_old  = np.dot(r, z)
        
        # iterate over the maximum number of iterations
        for _ in range(_maxiter):
            ap      = mat_vec_mult(p)
            alpha   = rs_old / np.dot(p, ap) 
            x       += alpha * p
            r       -= alpha * ap
            
            if np.abs(np.dot(r, r)) < _eps:
                return x, True
            
            z       = precond(r)
            rs_new  = np.dot(r, z)
            p       = z + (rs_new / rs_old) * p
            rs_old  = rs_new
        return x, False
    
    # -------------------------------------------------------------------------
    
    def solve(self, b, x0: Optional[np.ndarray] = None, precond: Optional[Preconditioner] = None):
        '''
        Solve the linear system Ax = b.
        Parameters:
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
        
        self.check_mat_or_matvec()
        
        _take_jax = (_JAX_AVAILABLE and (self._backend_str == 'jax' or self._backend_str == 'jnp')) or \
                    (self._backend_str == 'default' and _JAX_AVAILABLE)
        
        try:
            if precond is not None:
                self.set_preconditioner_sigma(self.reg)
                if _take_jax:
                    self._solution, self._converged = solve_precond_jit(self._mat_vec_mult, b, x0, precond)
                    return self._solution
                else:
                    self._solution, self._converged = self.solve_precond(self._mat_vec_mult, b, x0, precond)
                    return self._solution
            else:
                if _take_jax:
                    self._solution, self._converged = solve_no_precond_jit(self._mat_vec_mult, b, x0)
                    return self._solution
                else:
                    self._solution, self._converged = self.solve_no_precond(self._mat_vec_mult, b, x0)
                    return self._solution
        except Exception as e:
            raise SolverError(SolverErrorMsg.CONV_FAILED) from e
        return None
    
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------