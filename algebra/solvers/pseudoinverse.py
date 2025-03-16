from typing import Optional, Callable
import numpy as np

from general_python.algebra.solver import SolverType, Solver, SolverError, SolverErrorMsg
from general_python.algebra.utils import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend as __backend
from general_python.algebra.preconditioners import Preconditioner

# -----------------------------------------------------------------------------
# Pseudo-inverse Solver Class - uses the Moore-Penrose pseudo-inverse to solve
# linear systems of type Ax = b.
# -----------------------------------------------------------------------------

class PseudoInverseSolver(Solver):
    '''
    Pseudo-inverse solver class for linear systems of type Ax = b.
    '''
    
    # -------------------------------------------------------------------------
    
    @maybe_jit
    def _solve_pinv(self, a, b, backend = 'default'):
        '''
        Solve the linear system Ax = b using the pseudo-inverse.
        '''
        try:
            apinv           =   self._backend.linalg.pinv(a, rtol=self._reg)
            self._solution  =   apinv @ b
            self._converged =   True
            return self._solution
        except Exception as e:
            self._converged =   False
            raise SolverError(SolverErrorMsg.MAT_SINGULAR) from e
    
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
    
        # Solve the linear system using the pseudo-inverse.
        if not self._restart:
            try:
                return self._solve_pinv(self._a, b)
            except SolverError as e:
                print(str(e))
        else:
            while self._restarts <= self.maxrestarts and not self.converged:
                try:
                    return self._solve_pinv(self._a, b)
                except SolverError as e:
                    print(e)                # print the error message  
                    self.next_restart()     # increment the restart counter
                    self.increase_reg()     # increase the regularization parameter
                    if self.check_restart_up():
                        break
        return None
    
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------