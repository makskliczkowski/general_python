"""
Unified Eigenvalue Solver Interface

Provides a factory function to choose the appropriate eigenvalue solver based on
problem characteristics (matrix size, symmetry, number of eigenvalues needed).

This module simplifies the selection of eigenvalue solvers by automatically
choosing the most appropriate method based on user requirements.

----------------------------------------------
File        : Python/QES/general_python/algebra/eigen/factory.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 2025-10-26
----------------------------------------------
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable, Literal, Union, Any, Dict

# ----------------------------------------------------------------------------------------

try:
    from .result            import EigenResult, EigenSolver
    from .exact             import ExactEigensolver, ExactEigensolverScipy
    from .lanczos           import LanczosEigensolver, LanczosEigensolverScipy
    from .arnoldi           import ArnoldiEigensolver, ArnoldiEigensolverScipy
    from .block_lanczos     import BlockLanczosEigensolver, BlockLanczosEigensolverScipy
except ImportError as e:
    raise ImportError("Failed to import eigen solvers. Ensure all dependencies are installed.") from e

# Optional scipy imports
try:
    import scipy.linalg as scipy_linalg
    import scipy.sparse.linalg as scipy_sparse_linalg
    SCIPY_AVAILABLE         = True
except ImportError:
    SCIPY_AVAILABLE         = False
    scipy_linalg            = None
    scipy_sparse_linalg     = None

# Optional JAX imports
try:
    import jax.numpy as jnp
    import jax.scipy.linalg as jax_scipy_linalg
    JAX_AVAILABLE           = True
except ImportError:
    JAX_AVAILABLE           = False
    jax                     = None
    jnp                     = None
    jax_scipy_linalg        = None

# ----------------------------------------------------------------------------------------
#! Unified Eigenvalue Solver Factory Function
# ----------------------------------------------------------------------------------------

def choose_eigensolver(
        method          : Literal['exact', 'lanczos', 'arnoldi', 'block_lanczos', 'shift-invert', 
                                'scipy-eigh', 'scipy-eig', 'scipy-eigs', 'lobpcg', 
                                'jax-eigh', 'auto'] = 'auto',
        A               : Optional[NDArray]                                     = None,
        matvec          : Optional[Callable[[NDArray], NDArray]]                = None,
        n               : Optional[int]                                         = None,
        k               : Optional[int]                                         = 6,
        hermitian       : bool                                                  = True,
        which           : Union[str, Literal['smallest', 'largest', 'both']]    = 'smallest',
        backend         : Literal['numpy', 'scipy', 'jax']                      = 'numpy',
        use_scipy       : bool                                                  = False,
        B               : Optional[NDArray]                                     = None,
        dtype           : Optional[np.dtype]                                    = None,
        **kwargs) -> EigenResult:
    r"""
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
            - 'shift-invert'    : Shift-invert for interior eigenvalues
            - 'scipy-eigh'      : SciPy dense Hermitian solver
            - 'scipy-eig'       : SciPy dense general solver
            - 'scipy-eigs'      : SciPy sparse general solver
            - 'lobpcg'          : Locally Optimal Block Preconditioned CG
            - 'jax-eigh'        : JAX Hermitian solver (GPU-accelerated)
            - 'auto'            : Automatically choose based on problem size
        A : 
            Matrix to diagonalize (optional if matvec provided)
        matvec :
            Matrix-vector product function (optional if A provided)
        n : 
            Dimension of problem (required if matvec provided without A)
        k : 
            Number of eigenvalues to compute (ignored for 'exact', 'scipy-eigh', 'scipy-eig')
        hermitian: 
            Whether matrix is symmetric/Hermitian
        which: 
            Which eigenvalues to find
                - For Lanczos: 'smallest', 'largest', 'both'
                - For Arnoldi/scipy-eigs: 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
                - For shift-invert: eigenvalues nearest to sigma
                - For lobpcg: 'smallest', 'largest'
        backend: 
            Backend to use ('numpy', 'scipy', 'jax')
        use_scipy: 
            Prefer SciPy wrappers when available
        B :
            Second matrix for generalized eigenvalue problem A @ v = \lambda @ B @ v
        **kwargs: Additional arguments passed to the solver
            - sigma             : float - Shift value for shift-invert (default: 0.0)
            - tol               : float - Convergence tolerance
            - max_iter          : int - Maximum iterations
            - M                 : Preconditioner for lobpcg (callable or matrix)
            - subset_by_index   : tuple - (lo, hi) for scipy-eigh subset
            - subset_by_value   : tuple - (lo, hi) for scipy-eigh subset

    Returns:
        EigenResult with eigenvalues and eigenvectors
    
    Examples:
        >>> # Exact diagonalization
        >>> A = np.random.randn(100, 100)
        >>> A = 0.5 * (A + A.T)
        >>> result = choose_eigensolver('exact', A)
        
        >>> # Lanczos for large sparse symmetric
        >>> result = choose_eigensolver('lanczos', A, k=10, which='smallest')
        
        >>> # JAX GPU-accelerated
        >>> result = choose_eigensolver('jax-eigh', A)
        
        >>> # LOBPCG with preconditioner
        >>> M = scipy.sparse.diags([1.0/np.diag(A)])
        >>> result = choose_eigensolver('lobpcg', A, k=10, M=M)
        
        >>> # Generalized eigenvalue problem
        >>> result = choose_eigensolver('scipy-eigh', A, B=B)
        
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
            
    # ----------------------------------------------
    #! handle matrix-vector product or explicit matrix A
    # ----------------------------------------------
    
    if A is None and matvec is None:
        raise ValueError("Must provide either matrix A or matvec function")
    
    if A is None and n is None:
        raise ValueError("Must provide dimension n when using matvec without A")
    
    if A is not None and matvec is None:
        # Define matvec from A
        def matvec(x: NDArray) -> NDArray:
            return A @ x
            
    # ----------------------------------------------    
    
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
        return solver.solve(A=A)
    
    # ----------------------------------------------    
    elif method == 'lanczos':
        # Lanczos for symmetric/Hermitian
        # Note: Use backend='numpy' (native) for access to Krylov basis and tridiagonal matrix
        # The scipy wrapper is faster but doesn't expose Lanczos internals
        if not hermitian:
            raise ValueError("Lanczos requires hermitian=True. Use 'arnoldi' for non-symmetric matrices")
        
        if use_scipy or backend == 'scipy':
            # Map 'smallest'/'largest' to SciPy's 'SA'/'LA'
            scipy_which     = {'smallest': 'SA', 'largest': 'LA', 'both': 'BE'}.get(which, 'SA')
            # Extract parameters that go to __init__ vs solve()
            init_kwargs     = {k: v for k, v in kwargs.items() if k in ['tol', 'hilbert', 'maxiter', 'v0', 'seed']}
            solver          = LanczosEigensolverScipy(k=k, which=scipy_which, dtype=dtype, **init_kwargs)
            solve_kwargs    = {k: v for k, v in kwargs.items() if k not in ['tol', 'hilbert', 'maxiter', 'v0', 'seed', 'reorthogonalize']}
            result          = solver.solve(A=A, matvec=matvec, n=n, k=k, dtype=dtype, **solve_kwargs)
            return result
        elif backend == 'jax':
            solver = LanczosEigensolver(k=k, which=which, backend='jax', **kwargs)
        else:
            # Native numpy implementation - exposes Krylov basis and tridiagonal matrix
            solver = LanczosEigensolver(k=k, which=which, backend='numpy', **kwargs)
        
        return solver.solve(A=A, matvec=matvec, n=n, k=k, dtype=dtype,
                max_iter=kwargs.get('max_iter', None), reorthogonalize=kwargs.get('reorthogonalize', True))
    
    # ----------------------------------------------    
    elif method == 'arnoldi':
        # Arnoldi for general matrices
        if use_scipy or backend == 'scipy':
            solver = ArnoldiEigensolverScipy(k=k, which=which, **kwargs)
        elif backend == 'jax':
            solver = ArnoldiEigensolver(k=k, which=which, backend='jax', **kwargs)
        else:
            solver = ArnoldiEigensolver(k=k, which=which, backend='numpy', **kwargs)
        
        return solver.solve(A=A, matvec=matvec, n=n)
    
    # ----------------------------------------------
    elif method == 'shift-invert':
        from scipy.sparse.linalg import LinearOperator, gmres, minres
        from scipy import sparse
        
        def get_spectral_shift_operator(A, sigma, n):
            """
            Returns a LinearOperator that approximates (A - sigma*I)^-1
            using an iterative solver (low memory) instead of LU (high memory).
            """
            
            def matvec(v):
                # We want to solve: (A - sigma*I) x = v
                # for Hermitian A, use minres. For non-Hermitian, use gmres.
                # tol controls accuracy of the 'inversion'. 
                # It doesn't need to be machine precision for early iterations.
                x, info     = minres(A - sigma * sparse.eye(n), v, tol=1e-10)
                if info != 0: print(f"Warning: Linear solver did not converge (info={info})")
                return x
            return LinearOperator((n, n), matvec=matvec, dtype=A.dtype)
        
        # Shift-invert mode for interior eigenvalues
        if not hermitian:
            raise ValueError("Shift-invert requires hermitian=True. Use 'arnoldi' for non-symmetric matrices")
        
        # Use scipy's eigsh with shift-invert mode
        try:
            from scipy.sparse.linalg import eigsh as scipy_eigsh
            
            if A is None:
                raise ValueError("Shift-invert requires explicit matrix A")
            
            # eigsh with sigma uses shift-invert mode
            which_map       = { 'smallest': 'LM', 'largest': 'SM', 'both': 'LM' }
            which_si        = which_map.get(which, 'LM')                            # Use 'LM' for shift-invert
            OP              = get_spectral_shift_operator(A, sigma, n)              # Create shift-invert operator
            sigma           = kwargs.get('sigma', 0.0)                              # Shift value
            evals, evecs    = scipy_eigsh(A, k=k, sigma=sigma, which=which_si, tol=kwargs.get('tol', 0), maxiter=kwargs.get('max_iter', None))
            true_evals      = sigma + 1.0 / evals
                                        
            # Sort by eigenvalue
            idx             = np.argsort(true_evals)
            true_evals      = true_evals[idx]
            evecs           = evecs[:, idx]

            return EigenResult(
                eigenvalues     = true_evals,
                eigenvectors    = evecs,
                iterations      = None,
                converged       = True,
                residual_norms  = None
            )
            
        except ImportError:
            raise ImportError("Shift-invert mode requires scipy.sparse.linalg.eigsh")
        
    # ----------------------------------------------    
    elif method == 'block_lanczos':
        # Block Lanczos for multiple eigenpairs
        if not hermitian:
            raise ValueError("Block Lanczos requires hermitian=True")
        
        if use_scipy or backend == 'scipy':
            # Map 'smallest'/'largest' to boolean for LOBPCG
            largest = (which == 'largest')
            solver  = BlockLanczosEigensolverScipy(k=k, largest=largest, **kwargs)
        elif backend == 'jax':
            solver  = BlockLanczosEigensolver(k=k, which=which, backend='jax', **kwargs)
        else:
            solver  = BlockLanczosEigensolver(k=k, which=which, backend='numpy', **kwargs)
            
        return solver.solve(A=A, matvec=matvec, n=n, max_iter=kwargs.get('max_iter', None), k=k, X0=kwargs.get('X0', None))
    # ----------------------------------------------
    
    elif method == 'scipy-eigh':
        # SciPy dense Hermitian/symmetric eigenvalue solver
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy-eigh requires scipy to be installed")
        if A is None:
            raise ValueError("scipy-eigh requires explicit matrix A")
        if not hermitian:
            raise ValueError("scipy-eigh requires hermitian=True. Use 'scipy-eig' for general matrices")
        
        # Convert to dense array if sparse
        A_dense     = A.toarray() if hasattr(A, 'toarray') else np.asarray(A)
        
        # Handle generalized eigenvalue problem
        b_mat       = None
        if B is not None:
            b_mat   = B.toarray() if hasattr(B, 'toarray') else np.asarray(B)
        
        # Extract subset parameters
        subset_by_index = kwargs.get('subset_by_index', None)
        subset_by_value = kwargs.get('subset_by_value', None)
        
        # Call scipy.linalg.eigh
        eigenvalues, eigenvectors = scipy_linalg.eigh(
            A_dense, b=b_mat,
            subset_by_index=subset_by_index,
            subset_by_value=subset_by_value
        )
        
        # For dense solver, return full spectrum (ignore k)
        return EigenResult(
            eigenvalues     =   eigenvalues,
            eigenvectors    =   eigenvectors,
            iterations      =   None,
            converged       =   True,
            residual_norms  =   None
        )
    # ----------------------------------------------    
    
    elif method == 'scipy-eig':
        # SciPy dense general eigenvalue solver
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy-eig requires scipy to be installed")
        if A is None:
            raise ValueError("scipy-eig requires explicit matrix A")
        
        # Convert to dense array if sparse
        A_dense     = A.toarray() if hasattr(A, 'toarray') else np.asarray(A)
        
        # Handle generalized eigenvalue problem
        b_mat       = None
        if B is not None:
            b_mat   = B.toarray() if hasattr(B, 'toarray') else np.asarray(B)

        # Call scipy.linalg.eig
        eigenvalues, eigenvectors = scipy_linalg.eig(A_dense, b=b_mat)
        
        # Sort by real part or magnitude
        if np.iscomplexobj(eigenvalues):
            idx = np.argsort(np.abs(eigenvalues))
        else:
            idx = np.argsort(eigenvalues)
        
        eigenvalues     = eigenvalues[idx]
        eigenvectors    = eigenvectors[:, idx]
        
        # For dense solver, return full spectrum (ignore k)
        return EigenResult(
            eigenvalues     = eigenvalues,
            eigenvectors    = eigenvectors,
            iterations      = None,
            converged       = True,
            residual_norms  = None
        )    
    # ----------------------------------------------
    
    elif method == 'scipy-eigs':
        # SciPy sparse general eigenvalue solver (for non-Hermitian)
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy-eigs requires scipy to be installed")
        if A is None:
            raise ValueError("scipy-eigs requires explicit matrix A")
        
        # Use scipy.sparse.linalg.eigs
        eigenvalues, eigenvectors = scipy_sparse_linalg.eigs(
            A, k=k, which=which,
            tol=kwargs.get('tol', 0),
            maxiter=kwargs.get('max_iter', None),
            v0=kwargs.get('v0', None)
        )
        
        # Sort by magnitude or real part
        if np.iscomplexobj(eigenvalues):
            idx = np.argsort(np.abs(eigenvalues))
        else:
            idx = np.argsort(eigenvalues)

        eigenvalues     = eigenvalues[idx]
        eigenvectors    = eigenvectors[:, idx]

        return EigenResult(
            eigenvalues     = eigenvalues,
            eigenvectors    = eigenvectors,
            iterations      = None,
            converged       = True,
            residual_norms  = None
        )
    # ----------------------------------------------    
    
    elif method == 'lobpcg':
        # Locally Optimal Block Preconditioned Conjugate Gradient
        if not SCIPY_AVAILABLE:
            raise ImportError("lobpcg requires scipy to be installed")
        if A is None:
            raise ValueError("lobpcg requires explicit matrix A")
        if not hermitian:
            raise ValueError("lobpcg requires hermitian=True")
        
        from scipy.sparse import sparse, issparse
        from scipy.sparse.linalg import lobpcg
        
        # Generate random initial vectors
        X           = np.random.randn(n if n is not None else A.shape[0], k)    # Initial guess
        X, _        = np.linalg.qr(X)                                           # force orthonormality
        
        # Extract preconditioner
        M           = kwargs.get('M', None)
        
        if issparse(A) and M is None:
            # Extract diagonal
            diag_A      = A.diagonal()
            # Shift to avoid zero division if necessary
            M_diag      = 1.0 / (np.abs(diag_A) + 1.0) 
            M           = sparse.diags([M_diag], [0])
        
        # Handle generalized eigenvalue problem
        b_mat       = B.toarray() if (B is not None and hasattr(B, 'toarray')) else B
    
        # Map which to largest parameter
        largest     = (which in ['largest', 'LA', 'LM'])
        tol         = kwargs.get('tol', 1e-8)
        max_iter    = kwargs.get('max_iter', 1000)
        
        # Call scipy.sparse.linalg.lobpcg
        eigenvalues, eigenvectors   = lobpcg(
            A, 
            X, 
            M       = M, 
            tol     = tol, 
            maxiter = max_iter, 
            largest = largest, 
        )
        
        # Sort eigenvalues
        idx             = np.argsort(eigenvalues)
        if largest:     idx = idx[::-1]
        
        eigenvalues     = eigenvalues[idx]
        eigenvectors    = eigenvectors[:, idx]

        return EigenResult(
            eigenvalues     = eigenvalues,
            eigenvectors    = eigenvectors,
            iterations      = max_iter,
            converged       = True,
            residual_norms  = None
        )
        
    # ----------------------------------------------
    elif method == 'jax-eigh':
        # JAX Hermitian eigenvalue solver (GPU-accelerated)
        if not JAX_AVAILABLE:
            raise ImportError("jax-eigh requires JAX to be installed")
        if A is None:
            raise ValueError("jax-eigh requires explicit matrix A")
        if not hermitian:
            raise ValueError("jax-eigh requires hermitian=True")
        
        # Convert to JAX array
        A_jax           = jnp.asarray(A.toarray() if hasattr(A, 'toarray') else A)
        
        # Call JAX eigh
        eigenvalues, eigenvectors   = jax_scipy_linalg.eigh(A_jax)
        
        # Convert back to numpy
        eigenvalues     = np.array(eigenvalues)
        eigenvectors    = np.array(eigenvectors)

        # Dense JAX solver: return full spectrum (ignore k) to match dense SciPy behavior
        
        return EigenResult(
            eigenvalues     = eigenvalues,
            eigenvectors    = eigenvectors,
            iterations      = None,
            converged       = True,
            residual_norms  = None
        )
    # ----------------------------------------------    
    else:
        # Unknown method
        raise ValueError(
            f"Unknown method: {method}. Choose from: "
            "'exact', 'lanczos', 'arnoldi', 'block_lanczos', 'shift-invert', "
            "'scipy-eigh', 'scipy-eig', 'scipy-eigs', 'lobpcg', 'jax-eigh', 'auto'"
        )

# --------------------------------------------------

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
#! MEMORY UTILITY FUNCTIONS
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
#! Memory Utilities
# ----------------------------------------------------------------------------------------

def estimate_hilbert_memory_gb(hilbert_dim: int, dtype=np.complex128) -> Dict[str, float]:
    """
    Estimate memory requirements for various operations.
    
    Returns dict with memory estimates in GB for:
    - single_vector: one state vector
    - hamiltonian_dense: full dense Hamiltonian matrix
    - hamiltonian_sparse: estimated sparse Hamiltonian (assuming ~ns*nh non-zeros)
    - eigenvectors_k: k eigenvector storage
    """
    bytes_per_element   = np.dtype(dtype).itemsize
    single_vec_bytes    = hilbert_dim * bytes_per_element
    dense_ham_bytes     = hilbert_dim * hilbert_dim * bytes_per_element
    
    return {
        'single_vector'         : single_vec_bytes  / (1024**3),
        'hamiltonian_dense'     : dense_ham_bytes   / (1024**3),
        'per_lanczos_vector'    : single_vec_bytes  / (1024**3),
    }

# ----------------------------------------------------------------------------------------
#! End of File
# ----------------------------------------------------------------------------------------
