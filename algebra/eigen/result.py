"""
Eigenvalue Solver Result Types

Standardized result containers for eigenvalue computations.
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, NamedTuple
from numpy.typing import NDArray

class EigenSolver:
    """
    Marker class for eigenvalue solver types.
    """
    
    @staticmethod
    def _is_hermitian(A, tol=1e-12):
        """Check if A is symmetric/Hermitian, works for dense and sparse."""
        if sp.issparse(A):
            diff = A - A.T.conjugate()
            # For sparse: use the largest absolute entry in the difference
            return diff.nnz == 0 or np.all(np.abs(diff.data) < tol)
        else:
            return np.allclose(A, A.T.conj(), atol=tol)
    
    def is_dense_solver() -> bool:
        """Indicate if the solver is for dense matrices."""
        return False
    
    def is_sparse_solver() -> bool:
        """Indicate if the solver is for sparse matrices."""
        return False
    
    def is_iterative_solver() -> bool:
        """Indicate if the solver is iterative."""
        return False
    
    # ----------------------------------------------------------------------------
    
    def solve(self, *args, **kwargs) -> 'EigenResult':
        """
        Solve the eigenvalue problem.

        Returns:
            EigenResult: Standardized result container.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
# ---------------------------------------------------------------------------------

class EigenResult(NamedTuple):
    r"""
    Standardized result from eigenvalue solvers.
    
    Attributes:
        eigenvalues:
            Computed eigenvalues (sorted by magnitude or value)
        eigenvectors: 
            Corresponding eigenvectors as columns
        subspacevectors: 
            Basis vectors of the subspace used (for iterative methods)
        iterations: 
            Number of iterations performed (None for direct methods)
        converged: 
            Whether the solver converged successfully
        residual_norms: 
            Residual norms ||A v - \lambda v|| for each eigenpair (optional)
    """
    eigenvalues     : NDArray
    eigenvectors    : NDArray
    subspacevectors : Optional[NDArray] = None
    iterations      : Optional[int]     = None
    converged       : bool              = True
    residual_norms  : Optional[NDArray] = None

    def __repr__(self):
        n_eigs      = len(self.eigenvalues) if self.eigenvalues is not None else 0
        iter_str    = f"{self.iterations}" if self.iterations is not None else "N/A"
        return (f"EigenResult(n_eigenvalues={n_eigs}, "
                f"converged={self.converged}, iterations={iter_str})")

    def __str__(self):
        return f'converged={self.converged}, iterations={self.iterations}'
    
# ------------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------------