"""
Unified Eigenvalue Solver Interface

Provides a factory function to choose the appropriate eigenvalue solver based on
problem characteristics (matrix size, symmetry, number of eigenvalues needed).

This module simplifies the selection of eigenvalue solvers by automatically
choosing the most appropriate method based on user requirements.

Typical Usage:
    >>> # Small matrix - use exact diagonalization
    >>> result = choose_eigensolver('exact', A, hermitian=True)
    
    >>> # Large sparse symmetric matrix - use Lanczos for extremal eigenvalues
    >>> result = choose_eigensolver('lanczos', A, k=10, which='smallest')
    
    >>> # Large non-symmetric matrix - use Arnoldi
    >>> result = choose_eigensolver('arnoldi', A, k=10, which='LM')
    
    >>> # Multiple eigenvalues with clustering - use Block Lanczos
    >>> result = choose_eigensolver('block_lanczos', A, k=20, block_size=5)
    
File        : Python/QES/general_python/algebra/eigen/factory.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 2025-10-15
"""

from typing import Optional, Callable, Literal, Union
import numpy as np
from numpy.typing import NDArray

try:
    from .result import EigenResult
    from .exact import ExactEigensolver, ExactEigensolverScipy
    from .lanczos import LanczosEigensolver, LanczosEigensolverScipy
    from .arnoldi import ArnoldiEigensolver, ArnoldiEigensolverScipy
    from .block_lanczos import BlockLanczosEigensolver, BlockLanczosEigensolverScipy
except ImportError as e:
    raise ImportError("Failed to import eigen solvers. Ensure all dependencies are installed.") from e

# ----------------------------------------------------------------------------------------
#! Unified Eigenvalue Solver Factory Function
# ----------------------------------------------------------------------------------------

def choose_eigensolver(
    method          : Literal['exact', 'lanczos', 'arnoldi', 'block_lanczos', 'auto'] = 'auto',
    A               : Optional[NDArray] = None,
    matvec          : Optional[Callable[[NDArray], NDArray]] = None,
    n               : Optional[int] = None,
    k               : Optional[int] = 6,
    hermitian       : bool = True,
    which           : Union[str, Literal['smallest', 'largest', 'both']] = 'smallest',
    backend         : Literal['numpy', 'scipy', 'jax'] = 'numpy',
    use_scipy       : bool = False,
    **kwargs) -> EigenResult:
    """
    Unified interface for eigenvalue solvers.
    
    Automatically selects the most appropriate eigenvalue solver based on the
    method specified or problem characteristics (when method='auto').
    
    Parameters:
    -----------
        method: Which solver to use
            - 'exact'           : Full diagonalization (all eigenvalues)
            - 'lanczos'         : Lanczos iteration (symmetric/Hermitian)
            - 'arnoldi'         : Arnoldi iteration (general matrices)
            - 'block_lanczos'   : Block Lanczos (multiple eigenpairs)
            - 'auto'            : Automatically choose based on problem size
        A : 
            Matrix to diagonalize (optional if matvec provided)
        matvec :
            Matrix-vector product function (optional if A provided)
        n : 
            Dimension of problem (required if matvec provided without A)
        k : 
            Number of eigenvalues to compute (ignored for 'exact')
        hermitian: 
            Whether matrix is symmetric/Hermitian
        which: 
            Which eigenvalues to find
                - For Lanczos: 'smallest', 'largest', 'both'
                - For Arnoldi: 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
        backend: 
            Backend to use ('numpy', 'scipy', 'jax')
        use_scipy: 
            Prefer SciPy wrappers when available
        **kwargs: Additional arguments passed to the solver
    
    Returns:
        EigenResult with eigenvalues and eigenvectors
    
    Examples:
        >>> # Exact diagonalization
        >>> A = np.random.randn(100, 100)
        >>> A = 0.5 * (A + A.T)
        >>> result = choose_eigensolver('exact', A)
        
        >>> # Lanczos for large sparse symmetric
        >>> result = choose_eigensolver('lanczos', A, k=10, which='smallest')
        
        >>> # Auto-select based on size
        >>> result = choose_eigensolver('auto', A, k=10, hermitian=True)
    """
    
    # Determine dimension
    if A is not None:
        n = A.shape[0]
    elif n is None:
        raise ValueError("Must provide n when using matvec")
    
    # Auto-select method based on problem size
    if method == 'auto':
        if n <= 500:
            # Small problem - use exact diagonalization
            method = 'exact'
        elif hermitian:
            # Large symmetric - use Lanczos
            if k is not None and k > 1 and (k >= 10 or n > 5000):
                method = 'block_lanczos'  # Many eigenvalues or very large
            else:
                method = 'lanczos'
        else:
            # Large non-symmetric - use Arnoldi
            method = 'arnoldi'
    
    # Choose solver based on method
    if method == 'exact':
        # Exact diagonalization - compute all eigenvalues
        if use_scipy or backend == 'scipy':
            solver = ExactEigensolverScipy(hermitian=hermitian, **kwargs)
        elif backend == 'jax':
            solver = ExactEigensolver(hermitian=hermitian, backend='jax', **kwargs)
        else:
            solver = ExactEigensolver(hermitian=hermitian, backend='numpy', **kwargs)
        
        if A is None:
            raise ValueError("Exact diagonalization requires explicit matrix A")
        
        return solver.solve(A)
    
    elif method == 'lanczos':
        # Lanczos for symmetric/Hermitian
        if not hermitian:
            raise ValueError("Lanczos requires hermitian=True. Use 'arnoldi' for non-symmetric matrices")
        
        if use_scipy or backend == 'scipy':
            # Map 'smallest'/'largest' to SciPy's 'SA'/'LA'
            scipy_which = {'smallest': 'SA', 'largest': 'LA', 'both': 'BE'}.get(which, 'SA')
            solver = LanczosEigensolverScipy(k=k, which=scipy_which, **kwargs)
        elif backend == 'jax':
            solver = LanczosEigensolver(k=k, which=which, backend='jax', **kwargs)
        else:
            solver = LanczosEigensolver(k=k, which=which, backend='numpy', **kwargs)
        
        return solver.solve(A=A, matvec=matvec, n=n)
    
    elif method == 'arnoldi':
        # Arnoldi for general matrices
        if use_scipy or backend == 'scipy':
            solver = ArnoldiEigensolverScipy(k=k, which=which, **kwargs)
        elif backend == 'jax':
            solver = ArnoldiEigensolver(k=k, which=which, backend='jax', **kwargs)
        else:
            solver = ArnoldiEigensolver(k=k, which=which, backend='numpy', **kwargs)
        
        return solver.solve(A=A, matvec=matvec, n=n)
    
    elif method == 'block_lanczos':
        # Block Lanczos for multiple eigenpairs
        if not hermitian:
            raise ValueError("Block Lanczos requires hermitian=True")
        
        if use_scipy or backend == 'scipy':
            # Map 'smallest'/'largest' to boolean for LOBPCG
            largest = (which == 'largest')
            solver = BlockLanczosEigensolverScipy(k=k, largest=largest, **kwargs)
        elif backend == 'jax':
            solver = BlockLanczosEigensolver(k=k, which=which, backend='jax', **kwargs)
        else:
            solver = BlockLanczosEigensolver(k=k, which=which, backend='numpy', **kwargs)
        
        return solver.solve(A=A, matvec=matvec, n=n)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'exact', 'lanczos', 'arnoldi', 'block_lanczos', 'auto'")


def decide_method(n         : int,
                k           : Optional[int] = None,
                hermitian   : bool = True,
                memory_mb   : Optional[float] = None) -> str:
    """
    Decide which eigenvalue method to use based on problem characteristics.
    
    Parameters:
    -----------
        n: 
            Dimension of the matrix
        k: 
            Number of eigenvalues needed (None = all eigenvalues)
        hermitian: 
            Whether matrix is symmetric/Hermitian
        memory_mb: 
            Available memory in MB (optional)

    Returns:
        Recommended method name: 'exact', 'lanczos', 'arnoldi', or 'block_lanczos'
    
    Example:
        >>> method = decide_method(n=10000, k=10, hermitian=True)
        >>> print(f"Recommended method: {method}")
        Recommended method: lanczos
    """
    
    # Memory requirement for full diagonalization
    # Dense matrix: n^2 * 8 bytes (float64) + working space
    full_memory_mb = (n * n * 8 / 1024 / 1024) * 2.5  # Factor for working space
    
    # If k is None or close to n, need all eigenvalues
    if k is None or k > n * 0.5:
        # Need most/all eigenvalues
        if n <= 500:
            return 'exact'
        elif memory_mb is not None and full_memory_mb > memory_mb:
            # Not enough memory for full diagonalization
            if hermitian:
                return 'block_lanczos'  # Use iterative with large block
            else:
                return 'arnoldi'
        else:
            return 'exact'
    
    # Need only k << n eigenvalues
    if n <= 500:
        # Small enough for exact
        return 'exact'
    elif k == 1:
        # Single eigenvalue - standard Lanczos/Arnoldi
        return 'lanczos' if hermitian else 'arnoldi'
    elif k >= 10 or n > 5000:
        # Many eigenvalues or very large matrix - use block method
        return 'block_lanczos' if hermitian else 'arnoldi'
    else:
        # Moderate case - standard iterative
        return 'lanczos' if hermitian else 'arnoldi'

# ----------------------------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------------------------