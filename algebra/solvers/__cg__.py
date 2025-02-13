from typing import Optional, Callable
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .. import solver
from solver import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend as __backend, _KEY, Solver, Preconditioner, SolverError, SolverErrorMsg

# -----------------------------------------------------------------------------
# Conjugate Gradient Solver Class for symmetric positive definite matrices
# -----------------------------------------------------------------------------

class CGSolver_scipy(Solver):
    '''
    Conjugate Gradient solver class for symmetric positive definite matrices directly from SciPy.
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
        
        self.check_mat_or_matvec()
        
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