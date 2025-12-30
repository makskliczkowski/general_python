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

from typing import Optional, Callable, Tuple, Literal, TYPE_CHECKING, Dict, Union
from numpy.typing import NDArray
import numpy as np

# ----------------------------------------------------------------------------------------
#! Backend imports
# ----------------------------------------------------------------------------------------

try:
    from ..utils import JAX_AVAILABLE, jax, jnp, jcfg
    
    if TYPE_CHECKING:
        from ...common.flog import Logger
    
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
#! Scalable Lanczos Parameters
# ----------------------------------------------------------------------------------------

def get_lanczos_parameters(hilbert_dim          : int, 
                        ns                      : int,
                        requested_k             : Optional[int]         = None,
                        requested_max_iter      : Optional[int]         = None,
                        convergence_factor      : float                 = 2.5,
                        logger                  : Optional['Logger']    = None
                        ) -> Dict[str, int]:
    """
    Compute optimal Lanczos parameters that scale with system size.
    
    The key insight is that for convergence:
    - k (number of eigenvalues) should be enough to capture the pbhysics. We don't need much more than the low-energy spectrum.
        - usually k ~ 10-200 for typical systems.
    - max_iter should be >> k for good convergence (typically 2-5x k)
    - Reorthogonalization is ESSENTIAL for numerical stability. Always enable it.
    
    For state-of-the-art sizes (N_s ~ 32-48 spins, dim ~ 10^9 - 10^14):
    - We want k ~ 20-100    low-energy states
    - max_iter  ~ 200-500   depending on gap structure and the degeneracy
    
    Rules of thumb:
    - min(k)    = 10 (to capture ground state + some excitations reliably)
    - max(k)    = min(5000, dim/10) (don't exceed 10% of Hilbert space)
    - max_iter >= convergence_factor * k (typically 2-5x, higher for gapless systems)
    
    Args:
        hilbert_dim:
            Dimension of Hilbert space
        ns: 
            Number of sites
        requested_k: 
            User-requested number of eigenvalues (None for auto)
        requested_max_iter: 
            User-requested max iterations (None for auto)
        target_gap_resolution: 
            Target relative resolution for energy gaps
        convergence_factor: 
            max_iter / k ratio (higher for harder problems)
    
    Returns:
        Dict with 'k', 'max_iter', 'tol', 'reorthogonalize' parameters
    """
    
    # Base k scaling: starts at ~20 for small systems, grows logarithmically
    # For physics: want enough states to see the spectrum structure
    base_k          = max(20, int(8 * np.log2(max(ns, 4))))  # ~20 for 4 sites, ~60 for 32 sites
    max_k_fraction  = 0.1  # Never compute more than 10% of states
    max_k_absolute  = 2000  # Hard cap for memory
    max_k           = min(max_k_absolute, int(hilbert_dim * max_k_fraction))
    
    # Determine actual k
    if requested_k is not None:
        k = min(requested_k, max_k, hilbert_dim - 1)
    else:
        k = min(base_k, max_k, hilbert_dim - 1)
    
    # Ensure minimum k for reliable ground state
    k               = max(k, min(10, hilbert_dim - 1))
    
    # max_iter scaling: needs to be larger than k for convergence
    # For gapped systems: 2-3x k is often enough
    # For gapless/critical: may need 5-10x k
    # Scale with system size as larger systems tend to have denser spectra
    
    size_factor     = 1.0 + 0.5 * np.log2(max(ns / 8, 1))           # Increases with system size
    base_max_iter   = int(k * convergence_factor * size_factor)     # Base scaling
    
    # Additional buffer for large systems
    if hilbert_dim > 1e6:
        base_max_iter = int(base_max_iter * 1.5)
    if hilbert_dim > 1e9:
        base_max_iter = int(base_max_iter * 2.0)
    
    # Cap max_iter at Hilbert dimension and reasonable compute time
    max_iter_cap = min(hilbert_dim, 10000)                          # 10000 iterations is usually overkill
    
    if requested_max_iter is not None:
        max_iter = min(requested_max_iter, max_iter_cap)
    else:
        max_iter = min(base_max_iter, max_iter_cap)
    
    # Ensure max_iter > k
    max_iter = max(max_iter, k + 50)
    
    # Tolerance: tighter for smaller systems, relaxed for huge systems
    if hilbert_dim < 1e4:
        tol = 1e-12
    elif hilbert_dim < 1e6:
        tol = 1e-10
    elif hilbert_dim < 1e9:
        tol = 1e-8
    else:
        tol = 1e-6  # For truly massive systems, accept slightly lower precision
    
    params = {
        'k'                 : k,
        'max_iter'          : max_iter,
        'tol'               : tol,
        'reorthogonalize'   : True, # ALWAYS reorthogonalize for numerical stability
    }
    if logger:
        logger.info(f"Lanczos parameters for dim={hilbert_dim:.2e}, ns={ns}:", lvl=1, color='green')
        logger.info(f"k={k}, max_iter={max_iter}, tol={tol:.0e}", lvl=2, color='green')
    
    return params

def get_lanczos_memory_estimate_gb(hilbert_dim: int, max_iter: int, dtype=np.complex128) -> float:
    """
    Estimate memory needed for Lanczos iteration.
    
    Lanczos stores:
    - ~3 current vectors (v, v_old, w) during iteration
    - If full reorthogonalization: all max_iter Krylov vectors
    - Tridiagonal matrix: negligible
    
    For SciPy's eigsh with full reorthogonalization, it stores the Lanczos basis.
    """
    
    bytes_per_element   = np.dtype(dtype).itemsize
    vec_size            = hilbert_dim * bytes_per_element
    
    # Basic iteration: 3 vectors
    basic_memory        = 3 * vec_size
    
    # Full reorthogonalization: store all Krylov vectors
    krylov_memory       = max_iter * vec_size
    
    # Eigenvector storage (at end)
    # Usually much smaller than Krylov basis
    
    total_bytes         = basic_memory + krylov_memory
    return total_bytes / (1024**3)

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
            dtype               : Optional[np.dtype]                        = None,
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
        
        if v0 is not None:
            v0 = np.asarray(v0)
            if dtype is not None:
                v0 = v0.astype(dtype)
        
        # Determine dimension and matvec function
        if A is not None:
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(f"A must be square, got shape {A.shape}")
            
            n       = A.shape[0]
            def _matvec(x: NDArray) -> NDArray:
                return A @ x
            
            # Check symmetry
            if not EigenSolver._is_hermitian(A):
                raise ValueError("A must be symmetric or Hermitian for Lanczos")
            
        elif matvec is not None:
            if n is None:
                raise ValueError("n (dimension) must be provided when using matvec")
            _matvec = matvec
            
        else:
            raise ValueError("Either A or matvec must be provided")
        
        # ------------------------
        
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
                Dimension of the problem - required - original matrix size
            v0: 
                Initial vector
            max_iter: 
                Maximum number of Lanczos iterations - required - maximum number of Lanczos iterations
        Returns:
            EigenResult with eigenvalues, eigenvectors, and convergence info
        """
        
        # Initialize starting vector
        probe           = np.zeros(n)
        probe[0]        = 1.0
        probe_res       = matvec(probe)
        needs_complex   = np.iscomplexobj(probe_res)
        dtype           = np.complex128 if needs_complex else np.float64
        
        # Initialize starting vector with correct dtype
        if v0 is None:
            v0  = np.random.randn(n) + (1j * np.random.randn(n) if dtype == np.complex128 else 0)
            
        v0              = np.asarray(v0).astype(dtype)
        v0             /= np.linalg.norm(v0)
        
        # Storage for Krylov basis (columns are basis vectors)
        V               = np.zeros((n, max_iter + 1), dtype=dtype)
        V[:, 0]         = v0
        
        # Tridiagonal matrix elements
        alpha           = np.zeros(max_iter, dtype=float)   # always real
        beta            = np.zeros(max_iter - 1, dtype=float)

        # Lanczos iteration
        w               = matvec(v0) # First matvec
        
        # Initial projection alpha_0 = v0* . A . v0
        alpha[0]        = np.real(np.vdot(v0, w))
        w               = w - alpha[0] * v0             # Orthogonalize w against v0 to start the next step
        m_steps         = max_iter
        
        for j in range(1, max_iter):
            
            # Compute beta_{j-1} = || w ||
            beta_val    = np.linalg.norm(w)
            
            # Lucky breakdown check (invariant subspace found)
            if beta_val < 1e-12:
                m_steps = j
                break
                
            beta[j-1]   = beta_val
            
            # Normalize to get next basis vector v_j
            v_next      = w / beta_val
            V[:, j]     = v_next
            
            # Apply Matrix: w = A * v_j
            w           = matvec(v_next)
            
            # Standard Lanczos Orthogonalization (3-term recurrence)
            # w = A v_j - beta_{j-1} v_{j-1}
            w           = w - beta[j-1] * V[:, j-1]
            
            # Calculate alpha_j = v_j* . A . v_j
            alpha_val   = np.real(np.vdot(v_next, w))
            alpha[j]    = alpha_val
            
            # Orthogonalize w against v_j
            # w = w - alpha_j v_j
            w           = w - alpha_val * v_next
            
            # FULL REORTHOGONALIZATION
            # We must ensure w is orthogonal to ALL previous V[:, 0...j]
            if reorthogonalize:
                # Gram-Schmidt against all previous vectors
                overlaps    = V[:, :j+1].conj().T @ w           # Using Matrix-Vector mul for speed: overlaps = V[:, :j+1].H @ w
                w           = w - V[:, :j+1] @ overlaps         # Subtract projections: w = w - V @ overlaps

        # Truncate if we stopped early
        alpha   = alpha[:m_steps]
        beta    = beta[:m_steps-1]
        V       = V[:, :m_steps]

        # Solve Tridiagonal Problem
        # ----------------------------
        # Instead of calculating V.T @ A @ V (expensive), we use the T matrix we just built.
        # T is symmetric tridiagonal: diag=alpha, off_diag=beta
        
        # scipy.linalg.eigh_tridiagonal is extremely fast O(m^2)
        import scipy.linalg
        evals_T, evecs_T = scipy.linalg.eigh_tridiagonal(alpha, beta)
        
        # Extract Results
        # ------------------
        # Select eigenvalues based on 'which'
        if which == 'smallest':
            indices         = np.argsort(evals_T)[:k]
        elif which == 'largest':
            indices         = np.argsort(evals_T)[-k:][::-1] # Reverse for largest first
        else: # both
            idx_s           = np.argsort(evals_T)[:k]
            idx_l           = np.argsort(evals_T)[-k:]
            indices         = np.unique(np.concatenate((idx_s, idx_l)))

        selected_evals      = evals_T[indices]
        selected_evecs_T    = evecs_T[:, indices]
        
        # Compute Ritz vectors (lift back to full space)
        # eigenvectors = V @ y
        eigenvectors        = V @ selected_evecs_T
        
        # Convergence Check (Residual Norms)
        # -------------------------------------
        # Using the Lanczos relation: || A x - lambda x || = |beta_m * y_m|
        # where y_m is the last component of the tridiagonal eigenvector.
        # This avoids doing extra matvecs A@x for checking convergence!
        
        # Last component of the Ritz eigenvectors in the T-basis
        bottom_elements     = selected_evecs_T[-1, :] 
        
        # Residual = |beta_{last} * bottom_element|
        # Note: if we stopped early, use the last computed beta
        last_beta           = beta[-1] if len(beta) > 0 else 0.0    # Handle 1-step case
        residual_norms      = np.abs(last_beta * bottom_elements)   # Residual norms for each eigenpair
        converged           = np.all(residual_norms < 1e-8)         # Relaxed tol for iterative

        return EigenResult(
            eigenvalues     = selected_evals,
            eigenvectors    = eigenvectors,
            subspacevectors = V,
            iterations      = m_steps,
            converged       = converged,
            residual_norms  = residual_norms,
            # Lanczos coefficients specifically
            krylov_basis    = V,
            lanczos_alpha   = alpha,
            lanczos_beta    = beta
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
                dtype   : Optional[np.dtype]= None,
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
            seed      : Optional[int]       = None,
            dtype     : Optional[np.dtype]  = None,
            **kwargs
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
        from scipy.sparse.linalg import LinearOperator, ArpackError
        
        # Determine Dimension and Dtype
        if A is None and matvec is not None:
            if n is None:
                raise ValueError("n must be provided when using matvec")
            dim         = n
            op_dtype    = dtype if dtype is not None else np.complex128        
        elif A is not None:
            dim         = A.shape[0] if hasattr(A, 'shape') else n
            op_dtype    = A.dtype if hasattr(A, 'dtype')    else (dtype or np.complex128)
        else:
            raise ValueError("Either A or matvec must be provided")
        
        # Setup Random Seed
        current_seed    = seed if seed is not None else self.seed
        # Resolve Parameters
        k               = self.k if k is None else k
        which           = self.which if which is None else which
        maxiter         = self.maxiter if maxiter is None else maxiter
        tol             = self.tol if tol == 0.0 else tol
        
        # Extract special args
        sigma           = kwargs.get('sigma', None)
        ncv             = kwargs.get('ncv', None)

        # NCV Logic
        if ncv is None:
            # Standard rule: 2*k + 1 is efficient.
            # 20 is a safe minimum floor for convergence stability.
            desired_ncv = max(2 * k + 1, 20)
            ncv         = min(dim, desired_ncv)
            
            # ARPACK Constraint: ncv must be > k
            if ncv <= k:
                ncv = min(dim, k + 1)
                if ncv <= k:
                    raise ValueError(f"k={k} is too large for dim={dim}. ARPACK requires k < ncv <= n.")
        
        # Shift-Invert Validation
        if sigma is not None and A is None:
            if 'OPinv' not in kwargs:
                raise ValueError("Matrix-Free Shift-Invert mode ('sigma') requires 'OPinv' argument. SciPy cannot invert your function automatically.")

        # Create LinearOperator
        if A is None and matvec is not None:
            A_op = LinearOperator((dim, dim), matvec=matvec, dtype=op_dtype)
        else:
            A_op = A

        # Sanity Check: NaNs in MatVec
        # Generate a test probe to ensure the operator is healthy
        try:
            check_rng = np.random.RandomState(42)
            check_vec = check_rng.randn(dim).astype(op_dtype)
            if np.issubdtype(op_dtype, np.complexfloating):
                check_vec += 1j * check_rng.randn(dim).astype(op_dtype)
            
            check_res = A_op @ check_vec
            if not np.all(np.isfinite(check_res)):
                raise ValueError("Hamiltonian/Operator produces NaNs or Infs! Check your model parameters or potential division by zero.")
        except Exception as e:
            if "NaN" in str(e) or "Inf" in str(e):
                raise
            # If matvec fails for other reasons, let eigsh handle/report it later or warn
            # print(f"Warning: Pre-check matvec failed: {e}")
        
        # Prepare call args, removing those we handle explicitly
        call_kwargs = kwargs.copy()
        call_kwargs.pop('sigma', None)
        call_kwargs.pop('ncv', None)
        call_kwargs.pop('max_iter', None)
        call_kwargs.pop('v0', None) 
        call_kwargs.pop('tol', None)
        call_kwargs.pop('which', None)
        call_kwargs.pop('k', None)
        call_kwargs.pop('return_eigenvectors', None)
        call_kwargs.pop('maxiter', None)
        call_kwargs.pop('hilbert', None)

        # Retries
        max_retries = 3
        last_error  = None
        
        for attempt in range(max_retries):
            try:
                # Initialize v0 for this attempt
                if attempt == 0 and v0 is not None:
                    use_v0 = np.asarray(v0, dtype=op_dtype)
                else:
                    # Generate new random seed for retry
                    retry_seed = (current_seed + attempt) if current_seed is not None else None
                    rng_retry  = np.random.RandomState(retry_seed)
                    
                    if np.issubdtype(op_dtype, np.complexfloating):
                        use_v0 = (rng_retry.randn(dim) + 1j * rng_retry.randn(dim)).astype(op_dtype)
                    else:
                        use_v0 = rng_retry.randn(dim).astype(op_dtype)
                    use_v0 /= np.linalg.norm(use_v0)

                eigenvalues, eigenvectors = eigsh(
                    A_op,
                    k                   = k,
                    which               = which,
                    tol                 = tol,
                    maxiter             = maxiter,
                    v0                  = use_v0,
                    return_eigenvectors = True,
                    sigma               = sigma,
                    ncv                 = ncv,
                    **call_kwargs
                )
                
                # Check for NaNs/Infs in results
                if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
                    raise RuntimeError("eigsh returned Non-finite values (NaN/Inf).")

                # Success
                if attempt > 0:
                    # If we recovered from a failure, let the user know
                    print(f"Warning: Lanczos converged after {attempt} retries.")

                order = np.argsort(eigenvalues)
                if which in ['LM', 'LA']: 
                     if which == 'LA': order = order[::-1]
                     pass
                     
                return EigenResult(
                    eigenvalues     =   eigenvalues[order],
                    eigenvectors    =   eigenvectors[:, order],
                    iterations      =   -1, # Unknown with scipy eigsh
                    converged       =   True,
                    residual_norms  =   None,
                    lanczos_alpha   =   None,
                    lanczos_beta    =   None,
                    krylov_basis    =   None
                )

            except (ArpackError, RuntimeError) as e:
                last_error = e
                # Check for ZLASCL or ARPACK specific errors to decide if retry is worth it
                err_str = str(e)
                if "ZLASCL" in err_str or "ARPACK error" in err_str or "did not converge" in err_str or "Non-finite" in err_str:
                    # Transient numerical issue, try new vector
                    if attempt < max_retries - 1:
                        continue
                raise RuntimeError(f"SciPy eigsh failed after {max_retries} attempts. Last error: {e}")

        # Should be unreachable due to raise
        raise RuntimeError(f"SciPy eigsh failed: {last_error}")

# ------------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------------