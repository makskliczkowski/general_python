r"""
Block Lanczos Eigenvalue Solver

Implements the Block Lanczos algorithm for finding multiple extremal eigenvalues
and eigenvectors of large sparse symmetric/Hermitian matrices simultaneously.

The Block Lanczos method extends the standard Lanczos algorithm by working with
blocks of vectors instead of single vectors. This is particularly effective for:
    - Finding clustered or degenerate eigenvalues
    - Improving convergence when multiple eigenvalues are needed
    - Better numerical stability through block orthogonalization

Key Features:
    - Find p eigenvalues per block iteration
    - Block Krylov subspace construction
    - QR-based block orthogonalization
    - Deflation of converged eigenpairs
    - Backend support: NumPy and JAX
    - SciPy wrapper

Mathematical Background:
    Starting from block V_1 \in  ℝⁿˣᵖ, build block Krylov subspace
    1. K_m(A, V_1) = span{V_1, AV_1, A^2V_1, ...}
    2. Block orthogonalize to get [V_1, V_2, ..., V_m]
    3. Results in: A[V_1,...,V_m] = [V_1,...,V_m]T + V_{m+1}B_m^T
       where T is block tridiagonal

References:
    - Golub & Ye, "An Inverse Free Preconditioned Krylov Subspace Method"
    - Knyazev, "Toward the Optimal Preconditioned Eigensolver" (LOBPCG)
    - Saad, "Numerical Methods for Large Eigenvalue Problems"

File        : general_python/algebra/eigen/block_lanczos.py
"""

import numpy as np
from typing import Optional, Callable, Literal, Tuple
from numpy.typing import NDArray

# -----------------------------

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE   = True
except ImportError:
    JAX_AVAILABLE   = False

try:
    from scipy.sparse.linalg import lobpcg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# -----------------------------
from .result import EigenResult
# -----------------------------

# -----------------------------
#! Algorithm
# -----------------------------
class BlockLanczosEigensolver:
    """
    Block Lanczos algorithm for symmetric/Hermitian eigenvalue problems.
    
    Computes k extremal eigenvalues using blocks of p vectors per iteration.
    Particularly effective for degenerate or clustered eigenvalues.
    
    Args:
        k               : Number of eigenvalues to compute
        block_size      : Number of vectors per block (default: min(k, 3))
        which           : Which eigenvalues to compute ('smallest', 'largest')
        max_iter        : Maximum number of block iterations (default: min(50, n//block_size))
        tol             : Convergence tolerance for eigenvalues (default: 1e-10)
        reorthogonalize : Whether to reorthogonalize blocks (default: True)
        backend         : 'numpy' or 'jax' (default: 'numpy')
    
    Example:
        >>> A       = create_symmetric_matrix(1000, 1000)
        >>> solver  = BlockLanczosEigensolver(k=10, block_size=3, which='smallest')
        >>> result  = solver.solve(A)
        >>> print(f"First 10 eigenvalues: {result.eigenvalues}")
    """
    
    def __init__(self,
                k               : int           = 6,
                block_size      : Optional[int] = None,
                which           : Literal['smallest', 'largest'] = 'smallest',
                max_iter        : Optional[int] = None,
                tol             : float         = 1e-10,
                reorthogonalize : bool          = True,
                backend         : Literal['numpy', 'jax'] = 'numpy'):
        '''
        Block Lanczos eigensolver initialization.
        '''
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        self.k                  = k
        self.block_size         = block_size if block_size is not None else min(k, 3)
        self.which              = which
        self.max_iter           = max_iter
        self.tol                = tol
        self.reorthogonalize    = reorthogonalize
        self.backend            = backend

        if backend == 'jax' and not JAX_AVAILABLE:
            raise ImportError("JAX backend requested but JAX is not installed")

        # Lazy-compiled JAX core (class-shared). Built on first use.
        if not hasattr(BlockLanczosEigensolver, "_JAX_CORE"):
            BlockLanczosEigensolver._JAX_CORE = None
    
    # -----------------------------

    def solve(self,
            A           : Optional[NDArray] = None,
            matvec      : Optional[Callable[[NDArray], NDArray]] = None,
            V0          : Optional[NDArray] = None,
            n           : Optional[int] = None,
            *,
            # Optional overrides for instance config
            k           : Optional[int] = None,
            block_size  : Optional[int] = None,
            which       : Optional[Literal['smallest','largest']] = None,
            max_iter    : Optional[int] = None,
            tol         : Optional[float] = None,
            reorthogonalize: Optional[bool] = None,
            # Optional basis transforms: to/from computational basis
            to_basis    : Optional[Callable[[NDArray], NDArray]] = None,
            from_basis  : Optional[Callable[[NDArray], NDArray]] = None) -> EigenResult:
        """
        Solve for eigenvalues and eigenvectors using Block Lanczos.
        
        Parameters:
        -----------
            A: 
                Symmetric/Hermitian matrix (optional if matvec provided)
            matvec: 
                Matrix-vector product function (optional if A provided)
            V0: 
                Initial block of vectors, shape (n, block_size) (optional, random if None)
            n: 
                Dimension of the problem (required if matvec provided without A)

        Returns:
            EigenResult with k eigenvalues and eigenvectors
        """
                # Apply overrides
        eff_k           = k if k is not None else self.k
        eff_block       = block_size if block_size is not None else self.block_size
        eff_which       = which if which is not None else self.which
        eff_tol         = tol if tol is not None else self.tol
        eff_reorth      = self.reorthogonalize if reorthogonalize is None else reorthogonalize

        # Set up matrix-vector product(s)
        jax_matvec      = None
        
        if A is not None:
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(f"A must be square, got shape {A.shape}")
            n = A.shape[0]
            
            # Check symmetry
            if not np.allclose(A, A.T.conj(), rtol=1e-10, atol=1e-12):
                raise ValueError("A must be symmetric or Hermitian for Block Lanczos")
            
            # Block matvec for matrix A
            def _matvec(V):
                if V.ndim == 1:
                    return A @ V
                else:
                    return A @ V
            # Wrap with basis transforms if provided (run in computational basis)
            if to_basis is not None and from_basis is not None:
                
                def _matvec(V):
                    if V.ndim == 1:
                        v_orig = to_basis(V)
                        w_orig = A @ v_orig
                        return from_basis(w_orig)
                    else:
                        W_cols = []
                        for i in range(V.shape[1]):
                            v_orig = to_basis(V[:, i])
                            w_orig = A @ v_orig
                            W_cols.append(from_basis(w_orig))
                        return np.column_stack(W_cols)
                    
            if self.backend == 'jax':
                # Only enable compiled path when no Python transforms are present
                if to_basis is None and from_basis is None:
                    A_jnp   = jnp.asarray(A)
                    def _jax_matvec(V):
                        Vj  = V
                        return A_jnp @ Vj
                    jax_matvec = _jax_matvec

        elif matvec is not None:
            
            if n is None:
                raise ValueError("n (dimension) must be provided when using matvec")
            # Extend matvec to handle blocks (in original basis)
            def _matvec(V):
                if V.ndim == 1:
                    return matvec(V)
                else:
                    return np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
            # Note: Compiled JAX path requires a JAX-native matvec; if a Python
            # matvec is provided, we will fall back to the NumPy implementation.
        else:
            raise ValueError("Either A or matvec must be provided")
        
        # Determine max iterations
        if max_iter is not None:
            eff_max_iter = max_iter
        else:
            eff_max_iter = self.max_iter if self.max_iter is not None else min(50, n // eff_block)
        
        # Dispatch to backend
        if self.backend == 'numpy':
            result = self._block_lanczos_numpy(_matvec, n, V0, eff_max_iter,
                                               k=eff_k, block_size=eff_block,
                                               which=eff_which, tol=eff_tol,
                                               reorthogonalize=eff_reorth)
            # Transform eigenvectors back to original basis if provided
            if from_basis is not None:
                ev = result.eigenvectors
                ev_orig = np.column_stack([from_basis(ev[:, i]) for i in range(ev.shape[1])])
                result = EigenResult(result.eigenvalues, ev_orig, result.iterations, result.converged, result.residual_norms)
            return result
        else:  # jax
            if not JAX_AVAILABLE:
                # Fallback if JAX missing
                return self._block_lanczos_numpy(_matvec, n, V0, eff_max_iter,
                                                k=eff_k, block_size=eff_block,
                                                which=eff_which, tol=eff_tol,
                                                reorthogonalize=eff_reorth)

            # Only compiled path when A is provided and no Python transforms
            if jax_matvec is None:
                # Use NumPy path when only a Python matvec is provided
                return self._block_lanczos_numpy(_matvec, n, V0, eff_max_iter,
                                                k=eff_k, block_size=eff_block,
                                                which=eff_which, tol=eff_tol,
                                                reorthogonalize=eff_reorth)

            # Build or fetch compiled core
            core = BlockLanczosEigensolver._get_or_build_jax_core()

            # Prepare inputs
            p = int(eff_block)
            if V0 is None:
                V0_np = np.random.randn(n, p)
                V0_j = jnp.asarray(V0_np)
            else:
                V0_j = jnp.asarray(V0)

            A_jnp = jnp.asarray(A)
            which_flag = 0 if eff_which == 'smallest' else 1
            sel_evals, evecs, aiter, res_norms, conv, Vtrim = core(
                A_jnp,
                V0_j,
                int(n), int(p), int(eff_k), float(eff_tol), int(eff_max_iter), int(which_flag), bool(eff_reorth)
            )

            # Convert to NumPy and (optionally) map back to original basis (not supported for compiled path)
            return EigenResult(
                eigenvalues     =   np.asarray(sel_evals),
                eigenvectors    =   np.asarray(evecs),
                subspacevectors =   np.asarray(Vtrim),
                iterations      =   int(np.asarray(aiter)),
                converged       =   bool(np.asarray(conv)),
                residual_norms  =   np.asarray(res_norms)
            )
    
    # -----------------------------
    #! NUMPY
    # -----------------------------

    def _block_lanczos_numpy(
        self,
        matvec      : Callable[[NDArray], NDArray],
        n           : int,
        V0          : Optional[NDArray],
        max_iter    : int,
        *,
        k           : int,
        block_size  : int,
        which       : Literal['smallest','largest'],
        tol         : float,
        reorthogonalize: bool) -> EigenResult:
        """NumPy implementation of Block Lanczos iteration."""
        
        p           = block_size
        
        # Initialize starting block
        if V0 is None:
            # Make complex if needed
            V0      = np.random.randn(n, p)
            test    = matvec(np.ones((n, 1), dtype=complex))
            if np.iscomplexobj(test):
                V0  = V0 + 1j * np.random.randn(n, p)
        
        if V0.shape != (n, p):
            raise ValueError(f"V0 must have shape ({n}, {p}), got {V0.shape}")
        
        # Orthonormalize starting block using QR
        V0, R0      = np.linalg.qr(V0)
        
        # Storage for block Krylov basis
        # We store blocks as columns in a large matrix (n x (max_iter * p))
        max_dim     = min(max_iter * p, n)
        V           = np.zeros((n, max_dim), dtype=V0.dtype)
        V[:, :p]    = V0
        
        # Block tridiagonal matrix elements
        # In block Lanczos  : A V_j = V_{j-1} B_{j-1}^T + V_j A_j + V_{j+1} B_j
        # where A_j are pxp diagonal blocks and B_j are pxp off-diagonal blocks
        alpha_blocks    = []    # Diagonal blocks A_j
        beta_blocks     = []    # Off-diagonal blocks B_j
        
        # Block Lanczos iteration
        V_prev          = np.zeros((n, p), dtype=V0.dtype)
        B_prev          = np.zeros((p, p), dtype=V0.dtype)

        actual_iter = 0
        
        for j in range(max_iter):
            # Get current block
            V_j         = V[:, j*p:(j+1)*p]
            
            # Apply matrix to block: W = A V_j
            W           = matvec(V_j)
            
            # Compute diagonal block: A_j = V_j^T @ A @ V_j = V_j^T @ W
            A_j         = V_j.T @ W
            # Symmetrize (should already be symmetric for symmetric A, but enforce it)
            A_j         = 0.5 * (A_j + A_j.T.conj())
            alpha_blocks.append(A_j)
            
            # Three-term recurrence: W = A V_j - V_{j-1} B_{j-1}^T - V_j A_j
            W           = W - V_j @ A_j
            if j > 0:
                W       = W - V_prev @ B_prev.T

            # Reorthogonalization if requested (important for numerical stability)
            if reorthogonalize:
                # Perform two passes of block classical Gram-Schmidt for robustness
                for _ in range(2):
                    for i in range(j + 1):
                        V_i     = V[:, i*p:(i+1)*p]
                        proj    = V_i.T @ W
                        W       = W - V_i @ proj

            # QR factorization: W = V_{j+1} @ B_j
            # where V_{j+1} is orthonormal and B_j is upper triangular
            try:
                V_next, B_j     = np.linalg.qr(W)
            except np.linalg.LinAlgError:
                # Breakdown - W has no full rank
                actual_iter     = j + 1
                break
            
            # Check for breakdown (B_j very small means we found an invariant subspace)
            if np.linalg.norm(B_j, 'fro') < tol * 1e-2:
                actual_iter     = j + 1
                break
            
            beta_blocks.append(B_j)
            
            # Store next block
            if (j + 2) * p <= max_dim:
                V[:, (j+1)*p:(j+2)*p]   = V_next
                V_prev                  = V_j.copy()
                B_prev                  = B_j.copy()
                actual_iter             = j + 2
            else:
                actual_iter             = j + 1
                break
        
        # Trim to actual size
        m   = actual_iter * p
        V   = V[:, :m]
        
        # As a final safeguard, globally orthonormalize the basis columns
        V, _ = np.linalg.qr(V)

        # Rayleigh-Ritz projection on the constructed subspace V
        # This is numerically more robust than relying on the idealized
        # block-tridiagonal T assembly and fixes divergence seen previously.
        try:
            AV = matvec(V)  # supports both 1D and 2D in our _matvec wrapper
        except Exception:
            # Fallback: apply column by column
            AV = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])

        # Enforce Hermitian structure to damp numerical asymmetry
        H   = V.T.conj() @ AV
        H   = 0.5 * (H + H.T.conj())

        # With the final global QR, V columns are orthonormal (G ≈ I),
        # so we can solve the standard Hermitian eigenproblem for H.
        evals_H, evecs_H = np.linalg.eigh(H)

        # Select desired eigenvalues/eigenvectors
        if which == 'smallest':
            indices = np.argsort(evals_H)[:k]
        elif which == 'largest':
            indices = np.argsort(evals_H)[-k:][::-1]
        else:
            raise ValueError(f"Unknown which: {self.which}")

        selected_evals      = evals_H[indices]
        selected_evecs_H    = evecs_H[:, indices]

        # Ritz vectors in the original space
        eigenvectors        = V @ selected_evecs_H
        # Normalize columns for stability
        norms = np.linalg.norm(eigenvectors, axis=0)
        norms[norms == 0] = 1.0
        eigenvectors = eigenvectors / norms
        
        # Compute residual norms
        residual_norms      = np.zeros(len(selected_evals))
        for i, (lam, vec) in enumerate(zip(selected_evals, eigenvectors.T)):
            # Compute residual: ||A*v - lambda*v||
            # matvec might expect 1D or 2D, so handle both
            try:
                Av = matvec(vec)
            except:
                # If 1D doesn't work, try 2D
                Av = matvec(vec.reshape(-1, 1)).flatten()
            residual = Av - lam * vec
            residual_norms[i] = np.linalg.norm(residual)
        
        converged = np.all(residual_norms < tol)
        
        return EigenResult(
            eigenvalues     =   selected_evals,
            eigenvectors    =   eigenvectors,
            subspacevectors =   V,
            iterations      =   actual_iter,
            converged       =   converged,
            residual_norms  =   residual_norms
        )
    
    # -----------------------------
    #! JAX core (compiled, no self)
    # -----------------------------
    
    @classmethod
    def _get_or_build_jax_core(cls):
        ''' '''
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available")
        if cls._JAX_CORE is not None:
            return cls._JAX_CORE

        def core(A_jnp, V0_in, n_in, p_in, k_in, tol_in, max_iter_in, which_flag, reorth):
            V0n, _  = jnp.linalg.qr(V0_in)
            max_dim = min(max_iter_in * p_in, n_in)
            Vmat    = jnp.zeros((n_in, max_dim), dtype=V0n.dtype)
            Vmat    = Vmat.at[:, :p_in].set(V0n)
            Vprev   = jnp.zeros((n_in, p_in), dtype=V0n.dtype)
            Bprev   = jnp.zeros((p_in, p_in), dtype=V0n.dtype)

            def body_fun(j, state):
                Vmat, Vprev, Bprev, actual_iter, active = state

                def do_step(args):
                    Vmat, Vprev, Bprev, actual_iter, _ain = args
                    start   = j * p_in
                    Vj      = jax.lax.dynamic_slice(Vmat, (0, start), (n_in, p_in))
                    W       = A_jnp @ Vj
                    Aj      = Vj.T.conj() @ W
                    Aj      = 0.5 * (Aj + Aj.T.conj())
                    W       = W - Vj @ Aj
                    W       = jax.lax.cond(j > 0, lambda w_: w_ - Vprev @ Bprev.T, lambda w_: w_, W)

                    def reorth_pass(W_in):
                        def inner(i, wcur):
                            Vi      = jax.lax.dynamic_slice(Vmat, (0, i*p_in), (n_in, p_in))
                            proj    = Vi.T.conj() @ wcur
                            return wcur - Vi @ proj
                        return jax.lax.fori_loop(0, j+1, inner, W_in)

                    W = jax.lax.cond(reorth, reorth_pass, lambda x: x, W)
                    W = jax.lax.cond(reorth, reorth_pass, lambda x: x, W)

                    Vnext, Bj   = jnp.linalg.qr(W)
                    fro         = jnp.linalg.norm(Bj, 'fro')
                    threshold   = tol_in * 1e-2
                    cont        = (fro >= threshold) & ((j + 1) * p_in < max_dim)
                    Vmat_upd    = jax.lax.cond(cont, lambda Vm: jax.lax.dynamic_update_slice(Vm, Vnext, (0, (j+1)*p_in)), lambda Vm: Vm, Vmat)
                    Vprev_upd   = jax.lax.cond(cont, lambda _: Vj, lambda _: Vprev, operand=None)
                    Bprev_upd   = jax.lax.cond(cont, lambda _: Bj, lambda _: Bprev, operand=None)
                    aiter_upd   = jax.lax.cond(cont, lambda _: actual_iter + 1, lambda _: actual_iter, operand=None)
                    active_upd  = cont
                    return (Vmat_upd, Vprev_upd, Bprev_upd, aiter_upd, active_upd)

                return jax.lax.cond(active, do_step, lambda args: args, (Vmat, Vprev, Bprev, actual_iter, active))

            init_state      = (Vmat, Vprev, Bprev, 1, True)
            Vmat, Vprev, Bprev, aiter, active = jax.lax.fori_loop(0, max_iter_in, body_fun, init_state)
            m               = jnp.minimum(aiter * p_in, max_dim)
            Vtrim           = Vmat
            Vtrim, _        = jnp.linalg.qr(Vtrim)
            AV              = A_jnp @ Vtrim
            H               = Vtrim.T.conj() @ AV
            H               = 0.5 * (H + H.T.conj())
            evals, evecsH   = jnp.linalg.eigh(H)
            order           = jnp.argsort(evals)
            idx             = jax.lax.cond(which_flag == 0, lambda _: order[:k_in], lambda _: jnp.flip(order[-k_in:]), operand=None)
            sel_evals       = evals[idx]
            sel_evecsH      = evecsH[:, idx]
            evecs           = Vtrim @ sel_evecsH
            def resnorm(v, lam):
                r = (A_jnp @ v) - lam * v
                return jnp.linalg.norm(r)
            
            res_vmap        = jax.vmap(resnorm, in_axes=(1, 0))
            res_norms       = res_vmap(evecs, sel_evals)
            conv            = jnp.all(res_norms < tol_in)
            return sel_evals, evecs, aiter, res_norms, conv, Vtrim

        cls._JAX_CORE = jax.jit(core, static_argnums=(2,3,4,6,7,8))
        return cls._JAX_CORE
    # -------------------------------
    @staticmethod
    def _construct_block_tridiagonal(
                alpha_blocks        : list,
                beta_blocks         : list,
                p                   : int) -> NDArray:
        """Construct block tridiagonal matrix from Block Lanczos coefficients."""
        m_blocks    = len(alpha_blocks)
        m           = m_blocks * p
        T           = np.zeros((m, m), dtype=alpha_blocks[0].dtype)
        
        for i in range(m_blocks):
            # Diagonal block
            T[i*p:(i+1)*p, i*p:(i+1)*p] = alpha_blocks[i]
            
            # Off-diagonal blocks
            if i < m_blocks - 1:
                T[i*p:(i+1)*p, (i+1)*p:(i+2)*p] = beta_blocks[i]
                T[(i+1)*p:(i+2)*p, i*p:(i+1)*p] = beta_blocks[i].T.conj()
        
        return T
# ------------------------------
#! Scipy implementation
# ------------------------------

class BlockLanczosEigensolverScipy:
    """
    Block eigenvalue solver using SciPy's LOBPCG.
    
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) is a robust
    block iterative method for finding extremal eigenvalues of symmetric matrices.
    
    Args:
        k: Number of eigenvalues to compute
        largest: If True, find largest eigenvalues; if False, find smallest
        tol: Convergence tolerance (default: 1e-10)
        maxiter: Maximum number of iterations (default: None = automatic)
        precond_apply: Preconditioner function M^{-1} v (optional)
    
    Example:
        >>> A = create_large_sparse_matrix(10000, 10000)
        >>> solver = BlockLanczosEigensolverScipy(k=10, largest=False)
        >>> result = solver.solve(A)
    """
    
    def __init__(
                self,
                k               : int = 6,
                largest         : bool = False,
                tol             : float = 1e-10,
                maxiter         : Optional[int] = None,
                precond_apply   : Optional[Callable] = None):
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for BlockLanczosEigensolverScipy")
        
        self.k              = k
        self.largest        = largest
        self.tol            = tol
        self.maxiter        = maxiter
        self.precond_apply  = precond_apply
    
    def solve(
        self,
        A           : Optional[NDArray] = None,
        matvec      : Optional[Callable[[NDArray], NDArray]] = None,
        X0          : Optional[NDArray] = None,
        n           : Optional[int] = None,
        *,
        k           : int = None
        
        ) -> EigenResult:
        """
        Solve using LOBPCG.
        
        Args:
            A: Matrix or LinearOperator
            matvec: Matrix-vector product function (if A not provided)
            X0: Initial guess for eigenvectors, shape (n, k)
            n: Dimension (required if matvec provided)
        
        Returns:
            EigenResult with eigenvalues and eigenvectors
        """
        # Set up operator
        if A is not None:
            n = A.shape[0]
        elif matvec is not None and n is not None:
            # Create LinearOperator for LOBPCG
            from scipy.sparse.linalg import LinearOperator
            A = LinearOperator((n, n), matvec=matvec)
        else:
            raise ValueError("Must provide either A or (matvec and n)")
        
        # Initial guess
        if X0 is None:
            X0 = np.random.randn(n, self.k)
            if np.iscomplexobj(A @ np.ones((n, 1), dtype=complex)):
                X0 = X0 + 1j * np.random.randn(n, self.k)
        
        # Set up preconditioner
        if self.precond_apply is not None:
            from scipy.sparse.linalg import LinearOperator
            M = LinearOperator((n, n), matvec=self.precond_apply)
        else:
            M = None
        
        # Call LOBPCG
        eigenvalues, eigenvectors = lobpcg(
            A,
            X0,
            M=M,
            largest=self.largest,
            tol=self.tol,
            maxiter=self.maxiter
        )
        
        # Sort eigenvalues
        if self.largest:
            idx = np.argsort(eigenvalues)[::-1]
        else:
            idx = np.argsort(eigenvalues)
        
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute residuals
        residual_norms = np.array([
            np.linalg.norm(A @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
            for i in range(len(eigenvalues))
        ])
        
        converged = np.all(residual_norms < self.tol)
        
        return EigenResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            iterations=self.maxiter if self.maxiter else 20,  # LOBPCG doesn't return iterations
            converged=converged,
            residual_norms=residual_norms
        )

# ------------------------------------------
