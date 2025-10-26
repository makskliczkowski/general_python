"""
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
    Starting from v₁, build Krylov subspace K_m(A, v₁) = span{v₁, Av₁, A²v₁, ...}
    Orthogonalize to get V = [v₁, v₂, ..., v_m]
    Results in: A V = V T + \beta_m v_{m+1} e_m^T
    where T is tridiagonal with diagonal \alpha and off-diagonal \beta

References:
    - Golub & Van Loan, "Matrix Computations" (4th ed.), Algorithm 10.1.1
    - Saad, "Numerical Methods for Large Eigenvalue Problems"
    
File        : QES/general_python/algebra/eigen/lanczos.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 2024-06-15
"""

from typing import Optional, Callable, Tuple, Literal
import numpy as np
from numpy.typing import NDArray

# ----------------------------------------------------------------------------------------
#! Backend imports
# ----------------------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE   = True
except ImportError:
    JAX_AVAILABLE   = False

try:
    from scipy.sparse.linalg import eigsh
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from .result import EigenResult
except ImportError:
    raise ImportError("EigenResult class not found. Ensure 'result.py' is available.")

# ----------------------------------------------------------------------------------------
#! LanczosEigensolver
# ----------------------------------------------------------------------------------------

class LanczosEigensolver:
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
            A       : Optional[NDArray]                         = None,
            matvec  : Optional[Callable[[NDArray], NDArray]]    = None,
            v0      : Optional[NDArray]                         = None,
            n       : Optional[int]                             = None,
            *,
            k                   : int                           = None,
            which               : Literal['smallest', 'largest', 'both'] = 'smallest',
            max_iter            : Optional[int]                 = None,
            tol                 : float                         = 1e-10,
            reorthogonalize     : bool                          = True,
            reorth_tol          : float                         = 1e-12) -> EigenResult:
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
            if not np.allclose(A, A.T.conj()):
                raise ValueError("A must be symmetric or Hermitian for Lanczos")
            
        elif matvec is not None:
            if n is None:
                raise ValueError("n (dimension) must be provided when using matvec")
            _matvec = matvec
            
        else:
            raise ValueError("Either A or matvec must be provided")
        
        # Determine max iterations
        # Default: Use min(n, max(50, 3*k)) for better convergence
        if self.max_iter is not None:
            max_iter = self.max_iter
        else:
            max_iter = min(n, max(50, 3 * self.k))
        max_iter = min(max_iter, n)  # Cannot exceed dimension
        
        # Set other parameters
        self.tol                = tol if tol is not None else self.tol
        self.reorthogonalize    = reorthogonalize if reorthogonalize is not None else self.reorthogonalize
        self.reorth_tol         = reorth_tol if reorth_tol is not None else self.tol
        self.k                  = k if k is not None else self.k
        self.which              = which if which is not None else self.which

        # Use appropriate backend
        if self.backend == 'numpy':
            return self._lanczos_numpy(_matvec, n, v0, max_iter, k=self.k, reorthogonalize=self.reorthogonalize, reorth_tol=self.reorth_tol)
        else:  # jax
            return self._lanczos_jax(_matvec, n, v0, max_iter, k=self.k, reorthogonalize=self.reorthogonalize, reorth_tol=self.reorth_tol)
    
    # ------------------------------------------------------------------------------------
    
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
        """
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
        if v0 is None:
            v0 = np.random.randn(n)
            if np.iscomplexobj(matvec(np.ones(n, dtype=complex))):
                v0 = v0 + 1j * np.random.randn(n)
        
        v0          = v0 / np.linalg.norm(v0) # start from this vector
        
        # Storage for Krylov basis (columns are basis vectors)
        V           = np.zeros((n, max_iter + 1), dtype=v0.dtype)
        V[:, 0]     = v0
        
        # Tridiagonal matrix elements
        alpha       = np.zeros(max_iter, dtype=np.float64)
        beta        = np.zeros(max_iter, dtype=np.float64)
        
        # Lanczos iteration
        v_prev      = np.zeros(n, dtype=v0.dtype)
        beta_prev   = 0.0
        
        for j in range(max_iter):
            # Apply matrix
            w           = matvec(V[:, j])
            
            # Orthogonalize against current vector
            alpha[j]    = np.real(np.vdot(V[:, j], w))
            w          -= alpha[j] * V[:, j] + beta_prev * v_prev
            
            # Reorthogonalization if requested
            if reorthogonalize:
                for i in range(j + 1):
                    proj    = np.vdot(V[:, i], w)
                    # Always perform reorthogonalization for stability
                    w  -= proj * V[:, i]
            
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
        
        # Globally orthonormalize V and perform Rayleigh–Ritz on H = V^T A V
        V, _                = np.linalg.qr(V)
        try:
            AV              = matvec(V)
        except Exception:
            AV              = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
        H                   = V.T.conj() @ AV
        H                   = 0.5 * (H + H.T.conj())
        evals_H, evecs_H    = np.linalg.eigh(H)
        # Select desired eigenvalues
        indices             = LanczosEigensolver._select_eigenvalues(evals_H, k, which)
        selected_evals      = evals_H[indices]
        selected_evecs_H    = evecs_H[:, indices]
        # Eigenvectors in original space
        eigenvectors        = V @ selected_evecs_H
        # Enforce orthonormality of returned eigenvectors and refresh eigenvalues
        Q, _                = np.linalg.qr(eigenvectors)
        eigenvectors        = Q
        # Update eigenvalues via Rayleigh quotients
        try:
            AQ              = matvec(eigenvectors)
        except Exception:
            AQ              = np.column_stack([matvec(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])])
        selected_evals      = np.array([np.vdot(eigenvectors[:, i], AQ[:, i]).real for i in range(eigenvectors.shape[1])])
        
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
            residual_norms  = residual_norms
        )
    
    # ------------------------------------------------------------------------------------
    
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
            
            # Reorthogonalization if requested
            if reorthogonalize:
                for i in range(j + 1):
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

        # Orthonormalize V and perform Rayleigh–Ritz on H = V^T A V
        V, _                = jnp.linalg.qr(V)
        # Build AV column-wise if needed
        try:
            AV              = matvec(V)
        except Exception:
            AV              = jnp.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
        H                   = V.T.conj() @ AV
        H                   = 0.5 * (H + H.T.conj())
        evals_H, evecs_H    = jnp.linalg.eigh(H)
        indices             = LanczosEigensolver._select_eigenvalues(np.array(evals_H), k, which)
        selected_evals      = evals_H[indices]
        selected_evecs_H    = evecs_H[:, indices]
        eigenvectors        = V @ selected_evecs_H
        # Enforce orthonormality via QR and refresh eigenvalues with Rayleigh quotients
        Q, _                = jnp.linalg.qr(eigenvectors)
        eigenvectors        = Q
        try:
            AQ              = matvec(eigenvectors)
        except Exception:
            AQ              = jnp.column_stack([matvec(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])])
        selected_evals      = jnp.array([jnp.vdot(eigenvectors[:, i], AQ[:, i]).real for i in range(eigenvectors.shape[1])])
        
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
            residual_norms      = np.array(residual_norms)
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

class LanczosEigensolverScipy:
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
                v0      : Optional[NDArray] = None):
        """
        Initialize SciPy Lanczos eigensolver.
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
                Initial vector
        """
        
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for LanczosEigensolverScipy")
        
        self.k          = k
        self.which      = which
        self.tol        = tol
        self.maxiter    = maxiter
        self.v0         = v0

    def solve(self, 
              A         : Optional[NDArray] = None,
              matvec    : Optional[Callable[[NDArray], NDArray]] = None,
              n         : Optional[int] = None) -> EigenResult:
        """
        Solve for eigenvalues using SciPy's eigsh.
        
        Parameters:
        -----------
            A: 
                Matrix or LinearOperator
            matvec: 
                Matrix-vector product function (if A not provided)
            n: 
                Dimension (required if matvec provided)

        Returns:
            EigenResult with eigenvalues and eigenvectors
        """
        from scipy.sparse.linalg import LinearOperator
        
        # Create LinearOperator if matvec provided
        if A is None and matvec is not None:
            if n is None:
                raise ValueError("n must be provided when using matvec")
            A = LinearOperator((n, n), matvec=matvec)
        elif A is None:
            raise ValueError("Either A or matvec must be provided")
        
        # Call SciPy eigsh
        try:
            eigenvalues, eigenvectors = eigsh(
                A,
                k       = self.k,
                which   = self.which,
                tol     = self.tol,
                maxiter = self.maxiter,
                v0      = self.v0,
                return_eigenvectors=True
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
            residual_norms  =   None
        )
        
# ------------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------------