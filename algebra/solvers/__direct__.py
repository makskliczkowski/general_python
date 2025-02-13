from typing import Optional, Callable
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .. import solver
from solver import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend as __backend, _KEY, Solver, Preconditioner, SolverError, SolverErrorMsg

# -----------------------------------------------------------------------------
# Direct Solver Class
# -----------------------------------------------------------------------------

class DirectSolver(Solver):
    '''
    Direct solver class for linear systems of type Ax = b.
    '''

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
        
        super().solve(b, x0, precond)
        
        self.check_mat_or_matvec(needs_matrix=True)
        
        # solve the linear system
        try:
            if self.sigma is not None:
                ainv = self._backend.linalg.inv(self.matrix + self.sigma * self._backend.eye(self.size, self.size))
                self.solution(ainv @ b)
            else:
                ainv = self._backend.linalg.inv(self.matrix)
                self.solution(ainv @ b)
            self.converged(True)
            return self.solution
        except Exception as e:
            raise SolverError(SolverErrorMsg.MATRIX_INVERSION_FAILED_SINGULAR) from e
        return None
    
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

class DirectScipy(Solver):
    '''
    Direct solver class for linear systems of type Ax = b. Using built-in
    scipy.linalg.solve function.
    '''
    
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
        
        # initialize the solver
        super().solve(b, x0, precond)
        
        self.check_mat_or_matvec(needs_matrix=True)
        
        # solve the linear system
        try:
            if self.sigma() is not None:
                self.solution(self._backend_sp.linalg.solve(self.matrix + self.sigma * self._backend_sp.eye(self.size, self.size), b))
            else:
                self.solution(self._backend_sp.linalg.solve(self.matrix, b))
            self._converged(True)
            return self.solution
        except Exception as e:
            raise SolverError(SolverErrorMsg.MATRIX_INVERSION_FAILED) from e
        return None
    
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

class DirectBackend(Solver):
    """
    Direct solver class for linear systems of type Ax = b.
    """

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
        
        # initialize the solver
        super().solve(b, x0, precond)
        
        self.check_mat_or_matvec(needs_matrix=True)
            
        # solve the linear system
        try:
            if self.sigma() is not None:
                self.solution(self._backend.linalg.solve(self.matrix + self._backend.eye(self.size, self.size), b))
            else:
                self.solution(self._backend.linalg.solve(self.matrix, b))
                
            # convergence check
            self._converged(True)
            
            return self.solution
        except Exception as e:
            raise SolverError(SolverErrorMsg.MATRIX_INVERSION_FAILED) from e
        return None
    
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------
