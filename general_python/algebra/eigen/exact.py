"""
Exact Diagonalization (Full Eigenvalue Decomposition)

Provides wrappers for full eigenvalue decomposition using NumPy, SciPy, and JAX.
These methods compute all eigenvalues and eigenvectors, suitable for small to
medium-sized matrices where iterative methods are not needed.

Key Features:
    - Full eigenvalue decomposition (all eigenvalues + eigenvectors)
    - Specialized solvers for symmetric/Hermitian matrices
    - General solvers for non-symmetric matrices  
    - Backend support: NumPy, SciPy (sparse), JAX
    - Returns standardized EigenResult

Mathematical Background:
    For symmetric/Hermitian A: A = Q Λ Q^T (or Q^H for Hermitian)
    For general A: A = V Λ V^(-1) (not necessarily orthogonal)

References:
    - Golub & Van Loan, "Matrix Computations" (4th ed.), Chapter 8
    - NumPy linalg documentation
"""

from typing import Optional, Literal, Union
from numpy.typing import NDArray
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from scipy import linalg as scipy_linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .result import EigenResult, EigenSolver

# ----------------------------------------------------------------------------------------
#! Exact Eigensolver Classes
# ----------------------------------------------------------------------------------------

class ExactEigensolver:
    """
    Full eigenvalue decomposition using NumPy or JAX.
    
    Computes all eigenvalues and eigenvectors of a matrix using direct methods.
    For symmetric/Hermitian matrices, uses specialized algorithms that guarantee
    real eigenvalues and orthonormal eigenvectors.
    
    Args:
        hermitian: Whether matrix is symmetric/Hermitian (default: True)
        sort: How to sort eigenvalues ('ascending', 'descending', None)
        backend: 'numpy' or 'jax' (default: 'numpy')
    
    Example:
        >>> A = np.array([[4, -1], [-1, 3]])
        >>> solver = ExactEigensolver(hermitian=True, sort='ascending')
        >>> result = solver.solve(A)
        >>> print(f"All eigenvalues: {result.eigenvalues}")
    """

    def __init__(self, hermitian: bool = True, sort: Optional[Literal['ascending', 'descending']] = 'ascending', backend: Literal['numpy', 'jax'] = 'numpy'):
        self.hermitian  = hermitian
        self.sort       = sort
        self.backend    = backend

        if backend == 'jax' and not JAX_AVAILABLE:
            raise ImportError("JAX backend requested but JAX is not installed")
    
    def solve(self, A: NDArray) -> EigenResult:
        """
        Solve for all eigenvalues and eigenvectors.
        
        Args:
            A: Matrix to diagonalize
        
        Returns:
            EigenResult with all eigenvalues and eigenvectors
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")
        
        n = A.shape[0]
        
        # Dispatch to backend
        if self.backend == 'numpy':
            return self._solve_numpy(A, n)
        elif self.backend == 'jax':
            return self._solve_jax(A, n)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _solve_numpy(self, A: NDArray, n: int) -> EigenResult:
        """NumPy implementation."""
        
        if self.hermitian:
            # Symmetric/Hermitian: eigenvalues are always real
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            eigenvalues = eigenvalues.real  # Remove tiny imaginary parts
        else:
            # General case: eigenvalues may be complex
            eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Sort if requested
        if self.sort == 'ascending':
            if np.iscomplexobj(eigenvalues):
                # Sort by magnitude for complex eigenvalues
                idx = np.argsort(np.abs(eigenvalues))
            else:
                idx = np.argsort(eigenvalues)
            eigenvalues     = eigenvalues[idx]
            eigenvectors    = eigenvectors[:, idx]
        elif self.sort == 'descending':
            if np.iscomplexobj(eigenvalues):
                idx = np.argsort(np.abs(eigenvalues))[::-1]
            else:
                idx = np.argsort(eigenvalues)[::-1]
            eigenvalues     = eigenvalues[idx]
            eigenvectors    = eigenvectors[:, idx]
        
        # Compute residuals
        residual_norms = np.array([
            np.linalg.norm(A @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
            for i in range(n)
        ])
        
        # Full ED always converges
        return EigenResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            iterations=1,  # Direct method
            converged=True,
            residual_norms=residual_norms
        )
    
    def _solve_jax(self, A: NDArray, n: int) -> EigenResult:
        """JAX implementation."""
        
        A_jax = jnp.array(A)
        
        if self.hermitian:
            eigenvalues, eigenvectors = jnp.linalg.eigh(A_jax)
            eigenvalues = jnp.real(eigenvalues)
        else:
            eigenvalues, eigenvectors = jnp.linalg.eig(A_jax)
        
        # Sort if requested
        if self.sort == 'ascending':
            if jnp.iscomplexobj(eigenvalues):
                idx = jnp.argsort(jnp.abs(eigenvalues))
            else:
                idx = jnp.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        elif self.sort == 'descending':
            if jnp.iscomplexobj(eigenvalues):
                idx = jnp.argsort(jnp.abs(eigenvalues))[::-1]
            else:
                idx = jnp.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        
        # Compute residuals
        residual_norms = jnp.array([
            jnp.linalg.norm(A_jax @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
            for i in range(n)
        ])
        
        return EigenResult(
            eigenvalues=np.array(eigenvalues),
            eigenvectors=np.array(eigenvectors),
            iterations=1,
            converged=True,
            residual_norms=np.array(residual_norms)
        )

# ----------------------------------------------------------------------------------------
#! SciPy Exact Eigensolver Class
# ----------------------------------------------------------------------------------------

class ExactEigensolverScipy(EigenSolver):
    """
    Full eigenvalue decomposition using SciPy.
    
    Provides access to SciPy's LAPACK-based eigenvalue solvers, which may
    offer better performance or additional options compared to NumPy.
    
    Args:
        hermitian: Whether matrix is symmetric/Hermitian (default: True)
        sort: How to sort eigenvalues ('ascending', 'descending', None)
        driver: LAPACK driver ('ev', 'evd', 'evr', 'evx') - only for eigh
            - 'ev': Standard divide-and-conquer
            - 'evd': Divide-and-conquer (default)
            - 'evr': MRRR algorithm  
            - 'evx': Expert driver
    
    Example:
        >>> A = create_symmetric_matrix(100, 100)
        >>> solver = ExactEigensolverScipy(hermitian=True)
        >>> result = solver.solve(A)
    """
    
    def __init__(self, hermitian: bool = True, sort: Optional[Literal['ascending', 'descending']] = 'ascending', driver: Optional[str] = None):
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for ExactEigensolverScipy")
        
        self.hermitian  = hermitian
        self.sort       = sort
        self.driver     = driver

    def solve(self, A: NDArray) -> EigenResult:
        """
        Solve for all eigenvalues and eigenvectors.
        
        Args:
            A: Matrix to diagonalize
        
        Returns:
            EigenResult with all eigenvalues and eigenvectors
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")
        
        n = A.shape[0]
        
        if self.hermitian:
            # Use eigh for symmetric/Hermitian
            eigenvalues, eigenvectors = scipy_linalg.eigh(A, driver=self.driver)
            eigenvalues = eigenvalues.real
        else:
            # Use eig for general matrices
            eigenvalues, eigenvectors = scipy_linalg.eig(A)
        
        # Sort if requested
        if self.sort == 'ascending':
            if np.iscomplexobj(eigenvalues):
                idx = np.argsort(np.abs(eigenvalues))
            else:
                idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        elif self.sort == 'descending':
            if np.iscomplexobj(eigenvalues):
                idx = np.argsort(np.abs(eigenvalues))[::-1]
            else:
                idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        
        # Compute residuals
        residual_norms = np.array([
            np.linalg.norm(A @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
            for i in range(n)
        ])
        
        return EigenResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            iterations=1,
            converged=True,
            residual_norms=residual_norms
        )

# ----------------------------------------------------------------------------------------
#! Convenience Function
# ----------------------------------------------------------------------------------------

def full_diagonalization(A          : NDArray, 
                        hermitian   : bool = True,
                        sort        : Optional[Literal['ascending', 'descending']] = 'ascending',
                        backend     : Literal['numpy', 'scipy', 'jax'] = 'numpy') -> EigenResult:
    """
    Convenience function for full eigenvalue decomposition.
    
    Automatically chooses the appropriate solver based on backend preference.
    
    Args:
        A: Matrix to diagonalize
        hermitian: Whether matrix is symmetric/Hermitian
        sort: How to sort eigenvalues
        backend: Which backend to use
    
    Returns:
        EigenResult with all eigenvalues and eigenvectors
    
    Example:
        >>> A = np.random.randn(50, 50)
        >>> A = 0.5 * (A + A.T)  # Make symmetric
        >>> result = full_diagonalization(A, hermitian=True)
        >>> print(f"Ground state energy: {result.eigenvalues[0]}")
    """
    if backend == 'scipy':
        solver = ExactEigensolverScipy(hermitian=hermitian, sort=sort)
    elif backend == 'jax':
        solver = ExactEigensolver(hermitian=hermitian, sort=sort, backend='jax')
    else:  # numpy
        solver = ExactEigensolver(hermitian=hermitian, sort=sort, backend='numpy')
    
    return solver.solve(A)

# ----------------------------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------------------------