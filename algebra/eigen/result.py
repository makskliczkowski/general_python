"""
Eigenvalue Solver Result Types

Standardized result containers for eigenvalue computations.
"""

import numpy as np
from typing import Optional, NamedTuple
from numpy.typing import NDArray


class EigenResult(NamedTuple):
    """
    Standardized result from eigenvalue solvers.
    
    Attributes:
        eigenvalues:
            Computed eigenvalues (sorted by magnitude or value)
        eigenvectors: 
            Corresponding eigenvectors as columns
        iterations: 
            Number of iterations performed (None for direct methods)
        converged: 
            Whether the solver converged successfully
        residual_norms: 
            Residual norms ||A v - λ v|| for each eigenpair (optional)
    """
    eigenvalues     : NDArray
    eigenvectors    : NDArray
    iterations      : Optional[int] = None
    converged       : bool = True
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