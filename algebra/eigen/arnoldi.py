"""
Arnoldi Eigenvalue Solver

Implements the Arnoldi iteration for finding eigenvalues and eigenvectors
of general (non-symmetric) matrices.

The Arnoldi method generalizes the Lanczos algorithm to non-symmetric matrices,
building an orthonormal Krylov subspace and reducing the matrix to upper Hessenberg form.

Key Features:
    - Finds k eigenvalues of general matrices
    - Handles non-symmetric and complex matrices
    - Modified Gram-Schmidt with reorthogonalization
    - Backend support: NumPy and JAX
    - SciPy wrapper for production use

Mathematical Background:
    Starting from v_1 , build Krylov subspace K_m(A, v_1 ) = span{v_1 , Av_1 , A^2v_1 , ...}
    Orthogonalize using Modified Gram-Schmidt to get V = [v_1 , v_2 , ..., v_m]
    Results in: A V = V H + h_{m+1,m} v_{m+1} e_m^T
    where H is upper Hessenberg

References:
    [1] ...
    
----------------------------------------
----------------------------------------

"""

from typing import Optional, Callable, Tuple, Literal, Union
import numpy as np
from numpy.typing import NDArray

# Backend imports via global utils, with fallback to direct JAX import
try:
    from ..utils import JAX_AVAILABLE, jax, jnp, jcfg
    # If utils didn't enable JAX or x64, try to ensure availability here
    if not JAX_AVAILABLE:
        try:
            import jax                      # type: ignore
            import jax.numpy as jnp         # type: ignore
            from jax import config as jcfg  # type: ignore
            JAX_AVAILABLE   = True
            # Prefer 64-bit for numerical accuracy
            jcfg.update("jax_enable_x64", True)
        except ImportError:
            JAX_AVAILABLE   = False
            jax             = None
            jnp             = None
            jcfg            = None
    elif JAX_AVAILABLE and jcfg:
        try:
            jcfg.update("jax_enable_x64", True)
        except Exception:
            pass
except ImportError:
        JAX_AVAILABLE       = False
        jax                 = None
        jnp                 = None
        jcfg                = None

# Import scipy methods
try:
    from scipy.sparse.linalg import eigs
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from .result import EigenResult, EigenSolver
except ImportError:
    raise 

class ArnoldiEigensolver(EigenSolver):
    """
    Arnoldi iteration for general (non-symmetric) eigenvalue problems.
    
    Computes k eigenvalues and corresponding eigenvectors of a general matrix A
    using the Arnoldi iteration with Modified Gram-Schmidt orthogonalization.
    
    Unlike Lanczos (which requires symmetry), Arnoldi works for any matrix but
    produces an upper Hessenberg matrix instead of tridiagonal.
    
    Args:
        k: Number of eigenvalues to compute
        which: Which eigenvalues to compute
               'LM' = largest magnitude
               'SM' = smallest magnitude
               'LR' = largest real part
               'SR' = smallest real part
               'LI' = largest imaginary part
               'SI' = smallest imaginary part
        max_iter: Maximum number of Arnoldi iterations (default: min(2*k+1, n))
        tol: Convergence tolerance for eigenvalues (default: 1e-10)
        reorthogonalize: Whether to reorthogonalize vectors (default: True)
        reorth_tol: Tolerance for reorthogonalization (default: 1e-12)
        backend: 'numpy' or 'jax' (default: 'numpy')
    
    Example:
        >>> A = create_nonsymmetric_matrix(1000, 1000)
        >>> solver = ArnoldiEigensolver(k=5, which='LM')
        >>> result = solver.solve(A)
        >>> print(f"Eigenvalues: {result.eigenvalues}")
    """
    
    def __init__(
        self,
        k: int = 6,
        which: Literal['LM', 'SM', 'LR', 'SR', 'LI', 'SI'] = 'LM',
        max_iter: Optional[int] = None,
        tol: float = 1e-10,
        reorthogonalize: bool = True,
        reorth_tol: float = 1e-12,
        backend: Literal['numpy', 'jax'] = 'numpy'
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        self.k = k
        self.which = which
        self.max_iter = max_iter
        self.tol = tol
        self.reorthogonalize = reorthogonalize
        self.reorth_tol = reorth_tol
        self.backend = backend
        
        if backend == 'jax' and not JAX_AVAILABLE:
            raise ImportError("JAX backend requested but JAX is not installed")
    
    def solve(
        self,
        A: Optional[NDArray] = None,
        matvec: Optional[Callable[[NDArray], NDArray]] = None,
        v0: Optional[NDArray] = None,
        n: Optional[int] = None,
        return_krylov: bool = False
    ) -> Union[EigenResult, Tuple[EigenResult, NDArray]]:
        """
        Solve for eigenvalues and eigenvectors.
        
        Args:
            A: Matrix (if provided, matvec = lambda x: A @ x)
            matvec: Matrix-vector product function (if A not provided)
            v0: Initial vector (random if None)
            n: Dimension of the problem (required if matvec provided without A)
            return_krylov: Whether to return Krylov basis vectors
        
        Returns:
            EigenResult with eigenvalues, eigenvectors, and convergence info
            If return_krylov=True, also returns Krylov basis matrix V
        """
        # Determine dimension and matvec function
        if A is not None:
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(f"A must be square, got shape {A.shape}")
            n = A.shape[0]
            _matvec = lambda x: A @ x
        elif matvec is not None:
            if n is None:
                raise ValueError("n (dimension) must be provided when using matvec")
            _matvec = matvec
        else:
            raise ValueError("Either A or matvec must be provided")
        
        # Determine max iterations
        max_iter = self.max_iter if self.max_iter is not None else min(2 * self.k + 1, n)
        max_iter = min(max_iter, n)  # Cannot exceed dimension
        
        # Use appropriate backend
        if self.backend == 'numpy':
            return ArnoldiEigensolver._arnoldi_numpy(
                _matvec,
                n,
                v0,
                max_iter,
                return_krylov,
                k=self.k,
                which=self.which,
                reorthogonalize=self.reorthogonalize,
                reorth_tol=self.reorth_tol,
                tol=self.tol,
            )
        else:  # jax
            return ArnoldiEigensolver._arnoldi_jax(
                _matvec,
                n,
                v0,
                max_iter,
                return_krylov,
                k=self.k,
                which=self.which,
                reorthogonalize=self.reorthogonalize,
                reorth_tol=self.reorth_tol,
                tol=self.tol,
            )
    
    @staticmethod
    def _arnoldi_numpy(
        matvec: Callable[[NDArray], NDArray],
        n: int,
        v0: Optional[NDArray],
        max_iter: int,
        return_krylov: bool,
        *,
        k: int,
        which: Literal['LM', 'SM', 'LR', 'SR', 'LI', 'SI'],
        reorthogonalize: bool,
        reorth_tol: float,
        tol: float,
    ) -> Union[EigenResult, Tuple[EigenResult, NDArray]]:
        """NumPy implementation of Arnoldi iteration."""

        # Initialize starting vector
        if v0 is None:
            v0 = np.random.randn(n)
            # Probe complexity: only promote to complex if imaginary part is non-zero
            test_result = matvec(np.ones(n, dtype=complex))
            if np.iscomplexobj(test_result) and np.max(np.abs(np.imag(test_result))) > 1e-14:
                v0 = v0 + 1j * np.random.randn(n)

        v0 = v0 / np.linalg.norm(v0)
        dtype = v0.dtype

        # Storage for Arnoldi basis (Krylov vectors)
        V = np.zeros((n, max_iter + 1), dtype=dtype)
        V[:, 0] = v0

        # Upper Hessenberg matrix
        H = np.zeros((max_iter + 1, max_iter), dtype=np.complex128 if np.iscomplexobj(dtype) else np.float64)

        # Arnoldi iteration with Modified Gram-Schmidt
        for j in range(max_iter):
            # Apply matrix
            w = matvec(V[:, j])

            # Modified Gram-Schmidt orthogonalization
            for i in range(j + 1):
                H[i, j] = np.vdot(V[:, i], w)
                w = w - H[i, j] * V[:, i]

            # Reorthogonalization if requested (full reorthogonalization)
            if reorthogonalize:
                for i in range(j + 1):
                    correction = np.vdot(V[:, i], w)
                    if np.abs(correction) > reorth_tol:
                        H[i, j] += correction
                        w = w - correction * V[:, i]

            # Compute norm
            H[j + 1, j] = np.linalg.norm(w)

            # Check for breakdown (happy breakdown - exact invariant subspace)
            if H[j + 1, j] < tol * 1e-2:
                max_iter = j + 1
                H = H[:max_iter, :max_iter]
                V = V[:, :max_iter]
                break

            # Normalize new vector
            if j < max_iter - 1:
                V[:, j + 1] = w / H[j + 1, j]
        else:
            V = V[:, :max_iter]
            H = H[:max_iter, :max_iter]

        # Solve Hessenberg eigenvalue problem
        evals_H, evecs_H = np.linalg.eig(H)

        # Select desired eigenvalues
        indices = ArnoldiEigensolver._select_eigenvalues(evals_H, k, which)
        selected_evals = evals_H[indices]
        selected_evecs_H = evecs_H[:, indices]

        # Transform eigenvectors back to original space (Ritz vectors)
        # eigenvector of A ≈ V @ eigenvector of H
        eigenvectors = V @ selected_evecs_H
        # Normalize columns for numerical stability
        norms = np.linalg.norm(eigenvectors, axis=0)
        norms[norms == 0] = 1.0
        eigenvectors = eigenvectors / norms

        # Compute residual norms: ||A v - \lambda v||
        residual_norms = np.zeros(len(selected_evals), dtype=float)
        for i, (lam, vec) in enumerate(zip(selected_evals, eigenvectors.T)):
            residual = matvec(vec) - lam * vec
            residual_norms[i] = np.linalg.norm(residual)

        # Check convergence
        converged = np.all(residual_norms < tol)

        result = EigenResult(
            eigenvalues=selected_evals,
            eigenvectors=eigenvectors,
            subspacevectors=V,
            iterations=max_iter,
            converged=converged,
            residual_norms=residual_norms
        )

        if return_krylov:
            return result, V
        return result
    
    @staticmethod
    def _arnoldi_jax(
        matvec: Callable[[NDArray], NDArray],
        n: int,
        v0: Optional[NDArray],
        max_iter: int,
        return_krylov: bool,
        *,
        k: int,
        which: Literal['LM', 'SM', 'LR', 'SR', 'LI', 'SI'],
        reorthogonalize: bool,
        reorth_tol: float,
        tol: float,
    ) -> Union[EigenResult, Tuple[EigenResult, NDArray]]:
        """JAX implementation of Arnoldi iteration."""
        
        # Initialize starting vector
        if v0 is None:
            key = jax.random.PRNGKey(42)
            v0 = jax.random.normal(key, (n,))
            # Check if complex (promote only if imaginary part present)
            test_vec = matvec(jnp.ones(n, dtype=jnp.complex64))
            if jnp.iscomplexobj(test_vec) and jnp.max(jnp.abs(jnp.imag(test_vec))) > 1e-14:
                key2 = jax.random.PRNGKey(43)
                v0 = v0 + 1j * jax.random.normal(key2, (n,))
        
        v0 = v0 / jnp.linalg.norm(v0)
        dtype = v0.dtype
        
        # Storage for Arnoldi basis
        V = jnp.zeros((n, max_iter + 1), dtype=dtype)
        V = V.at[:, 0].set(v0)
        
        # Upper Hessenberg matrix
        H = jnp.zeros((max_iter + 1, max_iter), dtype=jnp.complex128 if jnp.iscomplexobj(dtype) else jnp.float64)
        
        actual_iters = max_iter
        
        # Arnoldi iteration
        for j in range(max_iter):
            w = matvec(V[:, j])
            
            # Modified Gram-Schmidt
            for i in range(j + 1):
                h_ij = jnp.vdot(V[:, i], w)
                H = H.at[i, j].set(h_ij)
                w = w - h_ij * V[:, i]
            
            # Reorthogonalization
            if reorthogonalize:
                for i in range(j + 1):
                    correction = jnp.vdot(V[:, i], w)
                    w = jnp.where(jnp.abs(correction) > reorth_tol,
                                  w - correction * V[:, i], w)
                    H = H.at[i, j].set(H[i, j] + jnp.where(jnp.abs(correction) > reorth_tol,
                                                            correction, 0.0))
            
            h_norm = jnp.linalg.norm(w)
            H = H.at[j + 1, j].set(h_norm)
            
            # Check for breakdown
            if h_norm < tol * 1e-2:
                actual_iters = j + 1
                break
            
            if j < max_iter - 1:
                V = V.at[:, j + 1].set(w / h_norm)
        
        # Trim if early termination
        V = V[:, :actual_iters]
        H = H[:actual_iters, :actual_iters]
        
        # Solve Hessenberg eigenvalue problem
        evals_H, evecs_H = jnp.linalg.eig(H)
        
        # Select desired eigenvalues
        indices = ArnoldiEigensolver._select_eigenvalues(np.array(evals_H), k, which)
        selected_evals = evals_H[indices]
        selected_evecs_H = evecs_H[:, indices]
        
        # Transform eigenvectors back
        eigenvectors = V @ selected_evecs_H
        # Normalize columns
        col_norms = jnp.linalg.norm(eigenvectors, axis=0)
        col_norms = jnp.where(col_norms == 0, 1.0, col_norms)
        eigenvectors = eigenvectors / col_norms
        
        # Compute residual norms
        residual_norms = jnp.array([
            jnp.linalg.norm(matvec(eigenvectors[:, i]) - selected_evals[i] * eigenvectors[:, i])
            for i in range(len(selected_evals))
        ])
        
        converged = jnp.all(residual_norms < tol)
        
        result = EigenResult(
            eigenvalues=np.array(selected_evals),
            eigenvectors=np.array(eigenvectors),
            subspacevectors=np.array(V),
            iterations=actual_iters,
            converged=bool(converged),
            residual_norms=np.array(residual_norms)
        )
        
        if return_krylov:
            return result, np.array(V)
        return result
    
    @staticmethod
    def _select_eigenvalues(eigenvalues: NDArray, k: int, which: str) -> NDArray:
        """Select k eigenvalues according to 'which' criterion."""
        if which == 'LM':  # Largest magnitude
            indices = np.argsort(np.abs(eigenvalues))[-k:][::-1]
        elif which == 'SM':  # Smallest magnitude
            indices = np.argsort(np.abs(eigenvalues))[:k]
        elif which == 'LR':  # Largest real part
            indices = np.argsort(np.real(eigenvalues))[-k:][::-1]
        elif which == 'SR':  # Smallest real part
            indices = np.argsort(np.real(eigenvalues))[:k]
        elif which == 'LI':  # Largest imaginary part
            indices = np.argsort(np.imag(eigenvalues))[-k:][::-1]
        elif which == 'SI':  # Smallest imaginary part
            indices = np.argsort(np.imag(eigenvalues))[:k]
        else:
            raise ValueError(f"Invalid which='{which}'. Must be 'LM', 'SM', 'LR', 'SR', 'LI', or 'SI'")
        
        return indices
    
    @staticmethod
    def ritz_vector_to_original(
        ritz_vector: NDArray,
        krylov_basis: NDArray
    ) -> NDArray:
        """
        Transform Ritz vector (eigenvector of H) back to original basis.
        
        Args:
            ritz_vector: Eigenvector of Hessenberg matrix H (length m)
            krylov_basis: Krylov basis matrix V (n \times m)
        
        Returns:
            Eigenvector in original space (length n)
        """
        if krylov_basis.shape[1] != len(ritz_vector):
            raise ValueError("Dimension mismatch: Krylov basis columns must match Ritz vector size")
        
        # Normalize Ritz vector
        ritz_vector = ritz_vector / np.linalg.norm(ritz_vector)
        
        # Transform: v = V @ s, where s is the Ritz vector
        return krylov_basis @ ritz_vector


class ArnoldiEigensolverScipy:
    """
    SciPy wrapper for Arnoldi eigenvalue solver.
    
    Uses scipy.sparse.linalg.eigs which provides a robust, production-ready
    implementation of the implicitly restarted Arnoldi method (ARPACK).
    
    This is recommended for production use. The native ArnoldiEigensolver
    is provided for educational purposes and JAX compatibility.
    
    Args:
        k: Number of eigenvalues to compute
        which: Which eigenvalues ('LM', 'SM', 'LR', 'SR', 'LI', 'SI')
        tol: Convergence tolerance (default: 0, uses SciPy default)
        maxiter: Maximum iterations (default: None, uses SciPy default)
        v0: Initial vector (default: None, random)
    
    Example:
        >>> A = create_sparse_nonsymmetric_matrix(10000, 10000)
        >>> solver = ArnoldiEigensolverScipy(k=10, which='LM')
        >>> result = solver.solve(A)
    """
    
    def __init__(
        self,
        k: int = 6,
        which: Literal['LM', 'SM', 'LR', 'SR', 'LI', 'SI'] = 'LM',
        tol: float = 0.0,
        maxiter: Optional[int] = None,
        v0: Optional[NDArray] = None
    ):
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for ArnoldiEigensolverScipy")
        
        self.k = k
        self.which = which
        self.tol = tol
        self.maxiter = maxiter
        self.v0 = v0
    
    def solve(
        self,
        A: Optional[NDArray] = None,
        matvec: Optional[Callable[[NDArray], NDArray]] = None,
        n: Optional[int] = None
    ) -> EigenResult:
        """
        Solve for eigenvalues using SciPy's eigs (ARPACK).
        
        Args:
            A: Matrix or LinearOperator
            matvec: Matrix-vector product function (if A not provided)
            n: Dimension (required if matvec provided)
        
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
        
        # Call SciPy eigs
        try:
            eigenvalues, eigenvectors = eigs(
                A,
                k=self.k,
                which=self.which,
                tol=self.tol,
                maxiter=self.maxiter,
                v0=self.v0,
                return_eigenvectors=True
            )
            
            converged = True
            iterations = None  # SciPy doesn't return iteration count directly
            
        except Exception as e:
            raise RuntimeError(f"SciPy eigs failed: {e}")
        
        return EigenResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            iterations=iterations,
            converged=converged,
            residual_norms=None
        )
