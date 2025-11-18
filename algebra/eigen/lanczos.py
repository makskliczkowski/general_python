r"""
Lanczos Eigenvalue Solver

Implements the Lanczos algorithm for finding extremal eigenvalues and eigenvectors
of large sparse symmetric/Hermitian matrices.

The Lanczos method builds an orthonormal Krylov subspace and reduces the matrix
to tridiagonal form, then computes eigenvalues of the much smaller tridiagonal matrix.

Key Features:
    - Finds k smallest or largest eigenvalues
    - Memory-efficient: Only stores k+1 vectors at a time
    - Supports reorthogonalization for numerical stability
    - Backend support: NumPy and JAX
    - SciPy wrapper for production use

Mathematical Background:
    1. Starting from
        $$
        v_1,
        $$
        build Krylov subspace
        $$
        K_m(A, v_1 ) = span{v_1 , Av_1 , A^2v_1 , ...}
        $$
    2. Orthogonalize to get V = [v_1 , v_2 , ..., v_m ]
    3. Results in: 
        $$
        A V = V T + \beta_m v_{m+1} e_m^T
        $$
        where T is tridiagonal with diagonal $\alpha$ and off-diagonal $\beta$

References:
[1]...
    
File        : QES/general_python/algebra/eigen/lanczos.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 2025-10-20
"""

from typing import Optional, Callable, Tuple, Literal
import numpy as np
from numpy.typing import NDArray

# ----------------------------------------------------------------------------------------
#! Backend imports
# ----------------------------------------------------------------------------------------

try:
    from ..utils import JAX_AVAILABLE, jax, jnp, jcfg
    # If utils didn't load JAX (because PREFER_JAX was False), try direct import
    if not JAX_AVAILABLE:
        try:
            import jax
            import jax.numpy as jnp
            from jax import config as jcfg
            JAX_AVAILABLE   = True
            # Enable 64-bit precision
            jcfg.update("jax_enable_x64", True)
        except ImportError:
            JAX_AVAILABLE   = False
            jax             = None
            jnp             = None
            jcfg            = None
    elif JAX_AVAILABLE and jcfg:
        # JAX was loaded by utils, ensure x64 is enabled
        jcfg.update("jax_enable_x64", True)    
except ImportError:
    JAX_AVAILABLE   = False
    jax             = None
    jnp             = None
    jcfg            = None

try:
    from scipy.sparse.linalg import eigsh
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from .result import EigenResult, EigenSolver
except ImportError:
    raise ImportError("EigenResult class not found. Ensure 'result.py' is available.")

# ----------------------------------------------------------------------------------------
#! LanczosEigensolver
# ----------------------------------------------------------------------------------------

class LanczosEigensolver(EigenSolver):
    r"""
    Lanczos algorithm for symmetric/Hermitian eigenvalue problems.
    
    Computes k extremal eigenvalues and corresponding eigenvectors of a
    symmetric (or Hermitian) matrix A using the Lanczos iteration.
    
    The algorithm is memory-efficient and particularly effective for finding
    a few eigenvalues of very large sparse matrices.
    
    Parameters:
    -----------
        k: 
            Number of eigenvalues to compute
        which: 
            Which eigenvalues to compute ('smallest', 'largest', 'both')
        max_iter: 
            Maximum number of Lanczos iterations (default: min(2*k+1, n))
        tol: 
            Convergence tolerance for eigenvalues (default: 1e-10)
        reorthogonalize: 
            Whether to reorthogonalize vectors (default: True)
        reorth_tol: 
            Tolerance for reorthogonalization (default: 1e-12)
        backend: 
            'numpy' or 'jax' (default: 'numpy')
    
    Example:
        >>> A = create_symmetric_matrix(1000, 1000)
        >>> solver = LanczosEigensolver(k=5, which='smallest')
        >>> result = solver.solve(A)
        >>> print(f"Smallest eigenvalues: {result.eigenvalues}")
    """
    
    def __init__(self,
                k               : int = 6,
                which           : Literal['smallest', 'largest', 'both'] = 'smallest',
                max_iter        : Optional[int] = None,
                tol             : float = 1e-10,
                reorthogonalize : bool = True,
                reorth_tol      : float = 1e-12,
                backend         : Literal['numpy', 'jax'] = 'numpy'):
        '''
        Initialize Lanczos eigensolver.
        
        Parameters
        ----------
        k: int
            Number of eigenvalues to compute
        which: {'smallest', 'largest', 'both'}
            Which eigenvalues to compute
        max_iter: int, optional
            Maximum number of Lanczos iterations
        tol: float
            Convergence tolerance
        reorthogonalize: bool
            Whether to reorthogonalize vectors
        reorth_tol: float
            Tolerance for reorthogonalization
        backend: {'numpy', 'jax'}
            Backend to use for computations
        '''
        
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        self.k                  = k
        self.which              = which
        self.max_iter           = max_iter
        self.tol                = tol
        self.reorthogonalize    = reorthogonalize
        self.reorth_tol         = reorth_tol
        self.backend            = backend

        if backend == 'jax' and not JAX_AVAILABLE:
            raise ImportError("JAX backend requested but JAX is not installed")
    
    # ------------------------------------------------------------------------------------
    #! solve
    # ------------------------------------------------------------------------------------
    
    def solve(self,
            A                   : Optional[NDArray]                         = None,
            matvec              : Optional[Callable[[NDArray], NDArray]]    = None,
            v0                  : Optional[NDArray]                         = None,
            n                   : Optional[int]                             = None,
            *,
            k                   : int                                       = None,
            which               : Literal['smallest', 'largest', 'both']    = None,
            max_iter            : Optional[int]                             = None,
            reorthogonalize     : bool                                      = True,
            reorth_tol          : float                                     = 1e-12) -> EigenResult:
        """
        Solve for eigenvalues and eigenvectors.
        
        Parameters:
        -----------
            A: 
                Matrix (if provided, matvec = lambda x: A @ x)
            matvec: 
                Matrix-vector product function (if A not provided)
            v0: 
                Initial vector (random if None)
            n: 
                Dimension of the problem (required if matvec provided without A)
            max_iter:
                Maximum number of Lanczos iterations
            tol:
                Convergence tolerance
            reorthogonalize:
                Whether to reorthogonalize vectors
            reorth_tol:
                Tolerance for reorthogonalization
        Returns:
            EigenResult with eigenvalues, eigenvectors, and convergence info, including:
                - eigenvalues: Computed eigenvalues
                - eigenvectors: Computed eigenvectors
                - converged: Boolean indicating convergence
                - iterations: Number of iterations performed
                - residual_norms: Residual norms for each eigenpair
                - subspacevectors: Krylov basis vectors used in computation
        """
        
        # Determine dimension and matvec function
        if A is not None:
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(f"A must be square, got shape {A.shape}")
            
            n       = A.shape[0]
            _matvec = lambda x: A @ x
            
            # Check symmetry
            if not EigenSolver._is_hermitian(A):
                raise ValueError("A must be symmetric or Hermitian for Lanczos")
            
        elif matvec is not None:
            if n is None:
                raise ValueError("n (dimension) must be provided when using matvec")
            _matvec = matvec
            
        else:
            raise ValueError("Either A or matvec must be provided")
        
        # Determine max iterations
        # Default: Use min(n, max(50, 3*k)) for better convergence
        k               = k if k is not None else self.k
        self.max_iter   = max_iter if max_iter is not None else self.max_iter
        max_iter        = self.max_iter if self.max_iter is not None else min(n, max(50, 3*k))
        max_iter        = min(max_iter, n)  # Cannot exceed dimension
        which           = self.which if which is None else which
        
        # If using NumPy backend with a matvec and SciPy is available, prefer robust eigsh
        # for matrix-free accuracy and performance.
        if self.backend == 'numpy' and matvec is not None and SCIPY_AVAILABLE:
            from scipy.sparse.linalg import LinearOperator, eigsh as _eigsh
            which_map = {
                'smallest': 'SA',
                'largest' : 'LA',
                'both'    : 'BE',
            }
            which_scipy     = which_map.get(which, 'SA')
            Aop             = LinearOperator((n, n), matvec=_matvec)
            # Allow ARPACK to choose iterations for robust convergence
            evals, evecs    = _eigsh(Aop, k=k, which=which_scipy, tol=self.tol, maxiter=None, v0=v0)
            # Sort according to requested ordering for consistency
            if which == 'smallest':
                order = np.argsort(evals)[:k]
            elif which == 'largest':
                order = np.argsort(evals)[-k:][::-1]
            elif which == 'both':
                # eigsh with 'BE' already returns both ends; keep as-is
                order = np.arange(len(evals))
            else:
                order = np.argsort(evals)[:k]
            evals           = evals[order]
            evecs           = evecs[:, order]
            return EigenResult(
                eigenvalues     = evals,
                eigenvectors    = evecs,
                subspacevectors = evecs,  # no Krylov basis from eigsh; return eigenvectors as subspace
                iterations      = None,
                converged       = True,
                residual_norms  = None,
            )

        # Use appropriate backend
        if self.backend == 'numpy':
            return LanczosEigensolver._lanczos_numpy(_matvec, n, v0, max_iter, 
                                        k               = k,
                                        which           = which,
                                        reorthogonalize = reorthogonalize,
                                        reorth_tol      = reorth_tol)
        else:  # jax
            return LanczosEigensolver._lanczos_jax(_matvec, n, v0, max_iter,
                                        k               = k,
                                        which           = which,
                                        reorthogonalize = reorthogonalize,
                                        reorth_tol      = reorth_tol)
    
    # ------------------------------------------------------------------------------------
    
    @staticmethod
    def _lanczos_numpy(
                    matvec          : Callable[[NDArray], NDArray],
                    n               : int,
                    v0              : Optional[NDArray],
                    max_iter        : int,
                    k               : int,
                    *,
                    which           : Literal['smallest', 'largest', 'both'] = 'smallest',
                    reorthogonalize : bool  = True,
                    reorth_tol      : float = 1e-12,
                    ) -> EigenResult:
        r"""
        NumPy implementation of Lanczos iteration.
        
        Parameters:
        -----------
            matvec: 
                Matrix-vector product function
            n: 
                Dimension of the problem
            v0: 
                Initial vector
            max_iter: 
                Maximum number of Lanczos iterations
        Returns:
            EigenResult with eigenvalues, eigenvectors, and convergence info
        """
        
        # Initialize starting vector
        probe           = np.ones(n)
        probe_res       = matvec(probe)
        needs_complex   = np.iscomplexobj(probe_res)

        # Initialize starting vector with correct dtype
        if v0 is None:
            v0 = np.random.randn(n)
            if needs_complex:
                v0 = v0.astype(np.complex128) + 1j * np.random.randn(n)
        else:
            v0 = np.asarray(v0)
            if v0.shape != (n,):
                raise ValueError("v0 has incompatible shape")
            # If operator is complex but v0 is real, promote it
            if needs_complex and not np.iscomplexobj(v0):
                v0 = v0.astype(np.complex128)

        dtype       = np.complex128 if needs_complex else np.float64
        v0          = v0.astype(dtype, copy=False)
        v0         /= np.linalg.norm(v0)
        
        # Storage for Krylov basis (columns are basis vectors)
        V           = np.zeros((n, max_iter + 1), dtype=dtype)
        V[:, 0]     = v0
        
        # Tridiagonal matrix elements
        alpha       = np.zeros(max_iter, dtype=float)   # always real
        beta        = np.zeros(max_iter, dtype=float)

        # Lanczos iteration
        v_prev      = np.zeros(n, dtype=v0.dtype)
        beta_prev   = 0.0
        
        for j in range(max_iter):
            # Apply matrix
            w           = matvec(V[:, j])
            
            # Orthogonalize against current vector
            alpha[j]    = np.real(np.vdot(V[:, j], w))
            w          -= alpha[j] * V[:, j]
            if j > 0:   w -= beta_prev * v_prev
                
            # Reorthogonalization (symmetric case): restrict to last two vectors to preserve 3-term recurrence
            if reorthogonalize:
                i0 = j - 1 if j > 0 else 0
                for i in (i0, j):
                    proj = np.vdot(V[:, i], w)
                    w   -= proj * V[:, i]
            
            # Compute beta and normalize - next basis vector
            beta[j] = np.linalg.norm(w)

            # Check for breakdown (lucky breakdown - exact invariant subspace)
            if beta[j] < reorth_tol * 1e-2:
                max_iter    = j + 1
                alpha       = alpha[:max_iter]
                beta        = beta[:max_iter]
                V           = V[:, :max_iter]
                break
            
            # Normalize new vector
            if j < max_iter - 1:
                V[:, j + 1] = w / beta[j]
                v_prev      = V[:, j].copy()
                beta_prev   = beta[j]
        else:
            V = V[:, :max_iter]

        # Orthonormalize basis and form Rayleigh–Ritz matrix H = V^H A V
        V, _                = np.linalg.qr(V)
        m                   = V.shape[1]
        AV                  = np.empty((n, m), dtype=dtype)
        for i in range(m):
            AV[:, i]        = matvec(V[:, i])
        
        # Construct Rayleigh–Ritz matrix
        H                   = V.T.conj() @ AV
        H                   = 0.5 * (H + H.T.conj())
        evals_H, evecs_H    = np.linalg.eigh(H)
        
        # Select desired eigenvalues
        indices             = LanczosEigensolver._select_eigenvalues(evals_H, k, which)
        selected_evals      = evals_H[indices]
        selected_evecs_H    = evecs_H[:, indices]
        # Ritz vectors in original space
        eigenvectors        = V @ selected_evecs_H
        # Refine eigenvalues via Rayleigh quotients in the original space
        selected_evals      = np.array([
                                    np.vdot(eigenvectors[:, i], matvec(eigenvectors[:, i])).real
                                    for i in range(eigenvectors.shape[1])
                                ])

        # Compute residual norms: ||A v - \lambda v||
        residual_norms      = np.zeros(len(selected_evals))
        for i, (lam, vec) in enumerate(zip(selected_evals, eigenvectors.T)):
            residual            = matvec(vec) - lam * vec
            residual_norms[i]   = np.linalg.norm(residual)

        # Check convergence
        converged           = np.all(residual_norms < reorth_tol)

        return EigenResult(
            eigenvalues     = selected_evals,
            eigenvectors    = eigenvectors,
            subspacevectors = V,
            iterations      = max_iter,
            converged       = converged,
            residual_norms  = residual_norms,
            # Lanczos coefficients specifically
            lanczos_alpha   = alpha,
            lanczos_beta    = beta,
            krylov_basis    = V
        )
    
    # ------------------------------------------------------------------------------------
    
    @staticmethod
    def _lanczos_jax(matvec         : Callable[[NDArray], NDArray],
                    n               : int,
                    v0              : Optional[NDArray],
                    max_iter        : int,
                    k               : int,
                    *,
                    which           : Literal['smallest', 'largest', 'both'] = 'smallest',
                    reorthogonalize : bool  = True,
                    reorth_tol      : float = 1e-12,
                    ) -> EigenResult:
        """JAX implementation of Lanczos iteration."""
        
        # Initialize starting vector
        if v0 is None:
            key         = jax.random.PRNGKey(42)
            v0          = jax.random.normal(key, (n,))
            # Check if complex
            test_vec    = matvec(jnp.ones(n, dtype=jnp.complex64))
            if jnp.iscomplexobj(test_vec):
                key2    = jax.random.PRNGKey(43)
                v0      = v0 + 1j * jax.random.normal(key2, (n,))
        
        v0              = v0 / jnp.linalg.norm(v0)
        
        # Storage for Krylov basis
        V               = jnp.zeros((n, max_iter + 1), dtype=v0.dtype)
        V               = V.at[:, 0].set(v0)

        # Tridiagonal matrix elements
        alpha           = jnp.zeros(max_iter, dtype=jnp.float64)
        beta            = jnp.zeros(max_iter, dtype=jnp.float64)

        # Lanczos iteration (note: JAX prefers functional style)
        v_prev          = jnp.zeros(n, dtype=v0.dtype)
        beta_prev       = 0.0

        actual_iters    = max_iter
        
        for j in range(max_iter):
            # Apply matrix
            w       = matvec(V[:, j])
            
            # Orthogonalize against current vector
            alpha_j = jnp.real(jnp.vdot(V[:, j], w))
            alpha   = alpha.at[j].set(alpha_j)
            w       = w - alpha_j * V[:, j] - beta_prev * v_prev
            
            # Reorthogonalization (symmetric case): restrict to last two vectors
            if reorthogonalize:
                for i in range(max(0, j - 1), j + 1):
                    proj = jnp.vdot(V[:, i], w)
                    w    = w - proj * V[:, i]
            
            # Compute beta and normalize
            beta_j = jnp.linalg.norm(w)
            beta   = beta.at[j].set(beta_j)
            
            # Check for breakdown
            if beta_j < reorth_tol * 1e-2:
                actual_iters = j + 1
                break
            
            # Normalize new vector
            if j < max_iter - 1:
                V           = V.at[:, j + 1].set(w / beta_j)
                v_prev      = V[:, j]
                beta_prev   = beta_j
        
        # Trim if early termination
        V                   = V[:, :actual_iters]
        alpha               = alpha[:actual_iters]
        beta                = beta[:actual_iters]

        # Orthonormalize basis and form Rayleigh–Ritz matrix H = V^H A V
        V, _                = jnp.linalg.qr(V)
        AV                  = jnp.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
        H                   = V.T.conj() @ AV
        H                   = 0.5 * (H + H.T.conj())
        evals_H, evecs_H    = jnp.linalg.eigh(H)
        indices             = LanczosEigensolver._select_eigenvalues(np.array(evals_H), k, which)
        selected_evals      = evals_H[indices]
        selected_evecs_H    = evecs_H[:, indices]
        eigenvectors        = V @ selected_evecs_H
        # Refine eigenvalues via Rayleigh quotients in the original space
        selected_evals      = jnp.array([
                                    jnp.vdot(eigenvectors[:, i], matvec(eigenvectors[:, i])).real
                                    for i in range(eigenvectors.shape[1])
                                ])
        
        # Compute residual norms
        residual_norms      = jnp.array([
                                            jnp.linalg.norm(matvec(eigenvectors[:, i]) - selected_evals[i] * eigenvectors[:, i])
                                            for i in range(len(selected_evals))
                                        ])

        converged           = jnp.all(residual_norms < reorth_tol)
        
        return EigenResult(
            eigenvalues         = np.array(selected_evals),
            eigenvectors        = np.array(eigenvectors),
            subspacevectors     = np.array(V),
            iterations          = actual_iters,
            converged           = bool(converged),
            residual_norms      = np.array(residual_norms),
            # Lanczos coefficients specifically
            lanczos_alpha       = np.array(alpha),
            lanczos_beta        = np.array(beta),
            krylov_basis        = np.array(V)
        )
    
    # ------------------------------------------------------------------------------------
    #! Helper Methods
    # ------------------------------------------------------------------------------------
    
    @staticmethod
    def _construct_tridiagonal(alpha: NDArray, beta: NDArray) -> NDArray:
        """Construct tridiagonal matrix from Lanczos coefficients."""
        m       = len(alpha)
        T       = np.diag(alpha)
        if m > 1:
            T  += np.diag(beta[:-1], k=1) + np.diag(beta[:-1], k=-1)
        return T
    
    @staticmethod
    def _select_eigenvalues(eigenvalues: NDArray, k: int, which: str) -> NDArray:
        """Select k eigenvalues according to 'which' criterion."""
        
        if which == 'smallest':
            indices = np.argsort(eigenvalues)[:k]
        elif which == 'largest':
            indices = np.argsort(eigenvalues)[-k:][::-1]
        elif which == 'both':
            # Half smallest, half largest
            k_half          = k // 2
            smallest_idx    = np.argsort(eigenvalues)[:k_half]
            largest_idx     = np.argsort(eigenvalues)[-(k - k_half):][::-1]
            indices         = np.concatenate([smallest_idx, largest_idx])
        else:
            raise ValueError(f"Invalid which='{which}'. Must be 'smallest', 'largest', or 'both'")
        
        return indices

# ----------------------------------------------------------------------------------------
#! LanczosEigensolverScipy
# ----------------------------------------------------------------------------------------

class LanczosEigensolverScipy(EigenSolver):
    """
    SciPy wrapper for Lanczos eigenvalue solver.
    
    Uses scipy.sparse.linalg.eigsh which provides a robust, production-ready
    implementation of the Lanczos algorithm with advanced features.
    
    This is recommended for production use. The native LanczosEigensolver
    is provided for educational purposes and JAX compatibility.

    Parameters:
    -----------
        k: 
            Number of eigenvalues to compute
        which: 
            Which eigenvalues ('SM'=smallest magnitude, 'LM'=largest magnitude,
            'SA'=smallest algebraic, 'LA'=largest algebraic, 'BE'=both ends)
        tol: 
            Convergence tolerance (default: 0, uses SciPy default)
        maxiter: 
            Maximum iterations (default: None, uses SciPy default)
        v0: 
            Initial vector (default: None, random)

    Example:
        >>> A = create_sparse_matrix(10000, 10000)
        >>> solver = LanczosEigensolverScipy(k=10, which='SA')
        >>> result = solver.solve(A)
    """
    
    def __init__(self,
                k       : int = 6,
                which   : Literal['SM', 'LM', 'SA', 'LA', 'BE'] = 'SA',
                tol     : float = 0.0,
                maxiter : Optional[int] = None,
                v0      : Optional[NDArray] = None,
                seed    : Optional[int]     = None):
        """
        Initialize SciPy Lanczos eigensolver.
        
        Note: This wrapper does NOT expose the Krylov basis or tridiagonal matrix.
        For access to Lanczos internals (alpha, beta, Krylov basis), use the native
        LanczosEigensolver instead.
        
        Parameters
        ----------
            k: int
                Number of eigenvalues to compute
            which: Literal['SM', 'LM', 'SA', 'LA', 'BE']
                Which eigenvalues to compute
            tol: float
                Convergence tolerance
            maxiter: Optional[int]
                Maximum number of iterations
            v0: Optional[NDArray]
                Initial vector (if None, a random vector with seed is used)
            seed: Optional[int]
                Random seed for reproducibility when v0 is None
        """
        
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for LanczosEigensolverScipy")
        
        self.k          = k
        self.which      = which
        self.tol        = tol
        self.maxiter    = maxiter
        self.v0         = v0
        self.seed       = seed

    def solve(self, 
            A         : Optional[NDArray] = None,
            matvec    : Optional[Callable[[NDArray], NDArray]] = None,
            n         : Optional[int] = None,
            *,
            k         : int = None,
            which     : Literal['SM', 'LM', 'SA', 'LA', 'BE'] = 'SA',
            tol       : float               = 0.0,
            maxiter   : Optional[int]       = None,
            v0        : Optional[NDArray]   = None,
            seed      : Optional[int]       = None
            ) -> EigenResult:
        """
        Solve for eigenvalues using SciPy's eigsh.
        
        WARNING: This wrapper does NOT return Krylov basis or tridiagonal matrix.
        Use LanczosEigensolver (native implementation) if you need:
        - lanczos_alpha, lanczos_beta (tridiagonal matrix)
        - krylov_basis (orthonormal Krylov vectors)
        
        Parameters:
        -----------
            A: 
                Matrix or LinearOperator
            matvec: 
                Matrix-vector product function (if A not provided)
            n: 
                Dimension (required if matvec provided)
            seed:
                Random seed for reproducibility (only used if v0 is None)

        Returns:
            EigenResult with eigenvalues and eigenvectors (but NO Krylov basis)
        """
        from scipy.sparse.linalg import LinearOperator
        
        # Determine dimension
        if A is None and matvec is not None:
            if n is None:
                raise ValueError("n must be provided when using matvec")
            dim = n
        elif A is not None:
            dim = A.shape[0] if hasattr(A, 'shape') else n
        else:
            raise ValueError("Either A or matvec must be provided")
        
        # Create initial vector with reproducible random seed
        use_seed = seed if seed is not None else self.seed
        use_v0   = v0 if v0 is not None else self.v0
        
        if use_v0 is None and use_seed is not None:
            rng     = np.random.RandomState(use_seed)
            use_v0  = rng.randn(dim)
            use_v0 /= np.linalg.norm(use_v0)
        
        # Create LinearOperator if matvec provided
        if A is None and matvec is not None:
            A = LinearOperator((dim, dim), matvec=matvec)
        
        # Call SciPy eigsh
        try:
            eigenvalues, eigenvectors = eigsh(
                A,
                k                   = self.k if k is None else k,
                which               = self.which if which is None else which,
                tol                 = self.tol if tol == 0.0 else tol,
                maxiter             = self.maxiter if maxiter is None else maxiter,
                v0                  = use_v0,
                return_eigenvectors = True
            )
            
            converged   = True
            iterations  = None  # SciPy doesn't return iteration count directly

        except Exception as e:
            # Handle convergence failure
            raise RuntimeError(f"SciPy eigsh failed: {e}")
        
        return EigenResult(
            eigenvalues     =   eigenvalues,
            eigenvectors    =   eigenvectors,
            iterations      =   iterations,
            converged       =   converged,
            residual_norms  =   None,
            # SciPy eigsh does not expose Krylov basis or tridiagonal matrix
            lanczos_alpha   =   None,
            lanczos_beta    =   None,
            krylov_basis    =   None
        )
        
# ------------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------------