"""
general_python/algebra/spectral_backend_krylov.py

Unified spectral function backend for computing spectral properties
of quantum systems from eigenvalues, eigenvectors, or Lanczos coefficients.

Type Safety:
    - Explicit handling of complex spectral functions
    - Proper dtype preservation (complex stays complex, real stays real)
    - No implicit casting of complex to real

----------------------------------------------------------------------------
File        : general_python/algebra/spectral_backend.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Description : Spectral function calculations using various backends.
----------------------------------------------------------------------------
"""

from typing import Optional, Union, Tuple, Literal, Any, Callable, List
import numpy as np
import numba as nb

from numpy.typing import NDArray
from scipy import sparse as sp

# Backend imports
try:
    from ....algebra.utils import JAX_AVAILABLE, get_backend
except ImportError:
    JAX_AVAILABLE = False
    get_backend = lambda x="default": np

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = None

# Type alias
Array = Union[np.ndarray, Any]  # Any allows JAX arrays

# ============================================================================
#? Lanczos Green's function calculations
# ============================================================================

def greens_function_lanczos(
        omega               : Union[float, Array],
        hamiltonian         : Optional[Array] = None,
        operator_a          : Optional[Array] = None,
        operator_b          : Optional[Array] = None,
        lanczos_eigenvalues : Optional[Array] = None, 
        lanczos_eigenvector : Optional[Array] = None, 
        ground_state_vec    : Optional[Array] = None,
        eta                 : float = 0.01,
        max_krylov          : int = 200,
        backend             : str = "default",
        kind                : str = "retarded",
        **kwargs) -> Array:
        
    be  = np
    w   = be.atleast_1d(be.asarray(omega, dtype=be.complex128))

    # Bi-Lanczos (Cross Correlation A != B dagger)
    if operator_b is not None:
        if hamiltonian is None or operator_a is None:
            raise ValueError("Bi-Lanczos requires hamiltonian, operator_a, and operator_b.")
        
        return _greens_lanczos_bilanczos(
            omega       =   w,
            H           =   hamiltonian,
            A_op        =   operator_a,
            B_op        =   operator_b,
            eta         =   eta,
            mb_states   =   kwargs.get('mb_states', 0),
            backend     =   backend,
            kind        =   kind,
            max_krylov  =   max_krylov
        )

    # Single-Chain Hermitian Lanczos (Auto Correlation) 
    else:
        # If pre-computed data provided (unlikely for S(q,w) which changes every q)
        if lanczos_eigenvalues is not None and lanczos_eigenvector is not None:
            t_evals, t_evecs, norm_weight = lanczos_eigenvalues, lanczos_eigenvector, 1.0
        
        # Compute Lanczos from scratch starting at S_q|GS>
        else:
            if hamiltonian is None or operator_a is None or ground_state_vec is None:
                raise ValueError("Lanczos needs hamiltonian, operator_a, and ground_state_vec.")

            # 1. Start Vector |phi> = A |GS>
            psi0                            = np.asarray(ground_state_vec).flatten()
            if hasattr(operator_a, "dot"):  phi = operator_a.dot(psi0)
            else:                           phi = operator_a @ psi0
                
            # 2. Run Recursion
            t_evals, t_evecs, norm_weight = _run_hermitian_lanczos(hamiltonian, phi, max_krylov)
            
            # Handle zero spectral weight
            if t_evals is None:             return np.zeros_like(w, dtype=complex)

            # 3. Shift Energies (T_evals are total energies, we need excitation energies)
            if sp.issparse(hamiltonian):
                E0 = np.real(np.vdot(psi0, hamiltonian.dot(psi0)))
            else:
                E0 = np.real(np.vdot(psi0, hamiltonian @ psi0))
            t_evals = t_evals - E0
            
        # 4. Compute GF from T-matrix data
        G_norm = _greens_lanczos_single_chain(
            omega               = w,
            lanczos_eigenvalues = t_evals,
            lanczos_eigenvector = t_evecs,
            eta                 = eta,
            mb_states           = 0,
            backend             = backend,
            kind                = kind
        )
        
        return G_norm * (norm_weight**2)

def greens_function_lanczos_vectors(
        omega       : np.ndarray,
        hamiltonian : np.ndarray,       # can be Matrix or function (matrix vector product)
        v_right     : np.ndarray,
        v_left      : np.ndarray,
        E0          : float,
        eta         : float = 0.05,
        max_krylov  : int = 200) -> np.ndarray:
    """
    Computes G(w) = <v_left | (w - (H-E0))^{-1} | v_right>
    using Bi-Lanczos iteration and continued fraction expansion.
    
    No operator matrices required.
    """
    w           = np.atleast_1d(np.asarray(omega, dtype=complex))
    
    #! Normalize Starting Vectors
    # We are effectively computing <w0 | v0>
    # Note: v_left input is usually (A^dag|0>). 
    # In the math <L| ... |R>, <L| is the conjugate of v_left.
    
    overlap_0   = np.vdot(v_left, v_right)
    
    if np.abs(overlap_0) < 1e-14:
        return np.zeros_like(w)

    norm_v      = np.sqrt(np.abs(overlap_0))
    
    # Initialize Lanczos vectors
    # v correspond to |ket>, w correspond to <bra| (stored as ket)
    v           = v_right / norm_v
    w_vec       = v_left * (overlap_0 / np.abs(overlap_0)) / norm_v
    
    #! Bi-Lanczos Iteration
    alphas      = []
    betas       = []
    gammas      = []
    v_prev      = np.zeros_like(v)
    w_prev      = np.zeros_like(w_vec)
    
    # Define H application (handle Sparse or Dense)
    if sp.issparse(hamiltonian):
        apply_H     = lambda x: hamiltonian.dot(x)
        apply_H_dag = lambda x: hamiltonian.conj().T.dot(x)
    elif callable(hamiltonian):
        apply_H     = hamiltonian
        apply_H_dag = hamiltonian   # Assume H is Hermitian if function
    else:
        apply_H     = lambda x: hamiltonian @ x
        apply_H_dag = lambda x: hamiltonian.conj().T @ x

    for j in range(max_krylov):
        # Apply H
        Hv          = apply_H(v)
        Hw          = apply_H_dag(w_vec)
        
        alpha       = np.vdot(w_vec, Hv)
        alphas.append(alpha)
        
        r           = Hv - alpha * v - (gammas[-1] * v_prev if j > 0 else 0.0)
        l           = Hw - alpha.conj() * w_vec - (betas[-1].conj() * w_prev if j > 0 else 0.0)
        delta       = np.vdot(l, r)
        
        if np.abs(delta) < 1e-14: break
        
        beta_next   = np.sqrt(np.abs(delta))
        gamma_next  = delta / beta_next
        
        betas.append(beta_next)
        gammas.append(gamma_next)
        
        v_prev      = v
        w_prev      = w_vec
        v           = r / beta_next
        w_vec       = l / gamma_next.conj()

    #! Continued Fraction
    z_vals          = w + E0 + 1j * eta
    next_term       = np.zeros_like(z_vals)
    
    # Backward recurrence
    m = len(alphas)
    for k in range(m - 2, -1, -1):
        numer       = betas[k] * gammas[k]
        denom       = z_vals - alphas[k+1] - next_term
        next_term   = numer / denom
        
    G_final = overlap_0 / (z_vals - alphas[0] - next_term)
    
    return G_final

# ============================================================================
# Lanczos Helpers
# ============================================================================

def _run_hermitian_lanczos(H, v0, max_krylov, tol=1e-12):
    """
    Runs Hermitian Lanczos starting from v0 to get T-matrix eigenvalues/vectors.
    """
    beta_0 = np.linalg.norm(v0)
    if beta_0 < tol: return None, None, 0.0
    
    v           = v0 / beta_0
    v_prev      = np.zeros_like(v)

    alpha_list  = []
    beta_list   = []

    for j in range(max_krylov):
        # Apply H
        w       = H.dot(v) if sp.issparse(H) else H @ v
        
        # Orthogonalize
        alpha   = np.real(np.vdot(v, w))
        w       = w - alpha * v - (beta_list[-1] * v_prev if j > 0 else 0.0)

        # Reorthogonalize (Simple)
        proj    = np.vdot(v, w)
        w       -= proj * v

        beta    = np.linalg.norm(w)
        alpha_list.append(alpha)
        
        if beta < tol: break # Invariant subspace
            
        beta_list.append(beta)
        v_prev  = v
        v       = w / beta
        
    # Diag T
    m           = len(alpha_list)
    T           = np.diag(alpha_list) + np.diag(beta_list[:m-1], k=1) + np.diag(beta_list[:m-1], k=-1)
    return (*np.linalg.eigh(T), beta_0)

def _greens_lanczos_single_chain(omega, lanczos_eigenvalues, lanczos_eigenvector, eta=0.01, *, mb_states=None, backend="default", kind="retarded"):
    """Vectorized sum over poles of the T-matrix."""
    be      = np
    evals   = be.asarray(lanczos_eigenvalues, dtype=be.complex128)
    U_L     = be.asarray(lanczos_eigenvector, dtype=be.complex128)

    # Use first component of T-eigenvectors (projection onto starting vector)
    # Shape: (M,)
    weights = be.abs(U_L[0, :])**2 
    
    # Denominators: (N_omega, M)
    if kind == "retarded":  denom = omega[:, None] + 1j*eta - evals[None, :]
    else:                   denom = omega[:, None] - 1j*eta - evals[None, :]
        
    # Sum over M poles
    return be.sum(weights[None, :] / denom, axis=1)

def _greens_lanczos_bilanczos(
        omega               : Array,
        H                   : Array,
        A_op                : Array,
        B_op                : Array,
        eta                 : float = 0.01,
        *,
        mb_states           : Optional[Union[int, List[int]]] = 0,
        ground_state_vec    : Optional[Array] = None,
        backend             : str = "default",
        kind                : str = "retarded",
        max_krylov          : int = 200) -> Array:
    """
    General zero-T Green function G_AB using Bi-Lanczos iteration.
    
    Calculates: G(w) = <0| A (w - H)^-1 B |0>
    
    Algorithm:
    1. Start with |v_0> = B|0> and <w_0| = <0|A
    2. Enforce biorthogonality: <w_n | v_m> = delta_nm
    3. Construct non-Hermitian tridiagonal matrix elements (alpha, beta, gamma).
    4. Evaluate G(w) using continued fraction.
    """
    be = np # forcing numpy for the iterative part
    w_grid = be.atleast_1d(be.asarray(omega, dtype=be.complex128))
    
    # 1. Define Matrix-Vector Product helpers (Handle Sparse vs Dense)
    if sp.issparse(H):
        matvec_H = lambda x: H.dot(x)
        matvec_H_dag = lambda x: H.conj().T.dot(x)
    else:
        matvec_H = lambda x: H @ x
        matvec_H_dag = lambda x: H.conj().T @ x
        
    if sp.issparse(A_op): matvec_A_dag = lambda x: A_op.conj().T.dot(x)
    else: matvec_A_dag = lambda x: A_op.conj().T @ x
        
    if sp.issparse(B_op): matvec_B = lambda x: B_op.dot(x)
    else: matvec_B = lambda x: B_op @ x

    # 2. Handle Reference States
    if mb_states is None: mb_indices = [0]
    elif isinstance(mb_states, int): mb_indices = [mb_states]
    else: mb_indices = mb_states

    G_all = be.zeros((len(mb_indices), len(w_grid)), dtype=be.complex128)

    for idx, m0 in enumerate(mb_indices):
        
        # -------------------------------------------------------
        # A. Prepare Starting Vectors
        # -------------------------------------------------------
        # We need the actual Ground State Vector |psi_0>
        if ground_state_vec is not None:
            # If provided, use the vector (Correct Physics Path)
            psi0 = be.asarray(ground_state_vec).flatten()
        else:
            # Fallback: Dummy unit vector (Only if H is a matrix and we want row m0)
            psi0 = be.zeros(H.shape[0], dtype=be.complex128)
            psi0[m0] = 1.0

        # Ground State Energy (to shift poles)
        # E0 = <psi0 | H | psi0>
        # Note: Lanczos builds poles at Total Energy E_n. 
        # G(w) requires (w - (E_n - E_0)). We shift w -> w + E0 later.
        E0 = be.vdot(psi0, matvec_H(psi0)).real

        # Right Start: |phi_R> = B |psi_0>
        phi_R = matvec_B(psi0)
        
        # Left Start:  |phi_L> = A^dagger |psi_0>  (So that <phi_L| = <0|A)
        phi_L = matvec_A_dag(psi0)
        
        # Compute Static Overlap / Normalization
        # The Green's function is scaled by <phi_L | phi_R> = <0|AB|0>
        overlap_0 = be.vdot(phi_L, phi_R)
        
        if be.abs(overlap_0) < 1e-14:
            # Orthogonal operators -> Zero spectral weight
            continue

        # Normalize starting vectors for the iteration
        # We strictly enforce <w_0 | v_0> = 1 for the algorithm
        v = phi_R / be.sqrt(be.abs(overlap_0))
        w = phi_L * (overlap_0 / be.abs(overlap_0)) / be.sqrt(be.abs(overlap_0))
        
        # Sanity check: be.vdot(w, v) must be 1.0 now (or very close)

        # -------------------------------------------------------
        # B. Bi-Lanczos Iteration
        # -------------------------------------------------------
        
        alphas = [] # Diagonal
        betas  = [] # Off-diagonal (lower)
        gammas = [] # Off-diagonal (upper)
        
        v_prev = be.zeros_like(v)
        w_prev = be.zeros_like(w)
        
        # We rely on the recurrence:
        # H v_j = alpha_j v_j + beta_{j+1} v_{j+1} + gamma_j v_{j-1}
        # H^dag w_j = alpha_j^* w_j + gamma_{j+1}^* w_{j+1} + beta_j^* w_{j-1}
        
        current_krylov_dim = 0
        
        for j in range(max_krylov):
            current_krylov_dim += 1
            
            # 1. Apply Hamiltonian
            Hv = matvec_H(v)
            Hw = matvec_H_dag(w)
            
            # 2. Compute Alpha (Diagonal)
            # alpha_j = <w_j | H | v_j>
            alpha = be.vdot(w, Hv)
            alphas.append(alpha)
            
            # 3. Compute Residuals
            # r = H v - alpha v - gamma_{j} v_{prev}
            # l = H^dag w - alpha^* w - beta_{j}^* w_{prev}
            
            r = Hv - alpha * v
            l = Hw - alpha.conj() * w
            
            if j > 0:
                r -= gammas[-1] * v_prev
                l -= betas[-1].conj() * w_prev
            
            # 4. Compute Overlap of residuals (delta = <l|r>)
            delta = be.vdot(l, r)
            
            # Check breakdown
            if be.abs(delta) < 1e-14:
                # Lucky breakdown (invariant subspace found) or Unlucky (algorithm failure)
                # Usually we assume Lucky for physics applications and stop.
                break
                
            # 5. Determine Beta and Gamma
            # Choice: beta = gamma = sqrt(delta) is common
            # or beta = sqrt(|delta|), gamma = delta / beta
            
            beta_next = be.sqrt(be.abs(delta))
            gamma_next = delta / beta_next
            
            betas.append(beta_next)
            gammas.append(gamma_next)
            
            # 6. Update Vectors
            v_prev = v
            w_prev = w
            
            v = r / beta_next
            w = l / gamma_next.conj() # Note the conj because <w| is w^dagger
            
        # -------------------------------------------------------
        # C. Compute Continued Fraction
        # -------------------------------------------------------
        # G(z) = <0|AB|0> * [1 / (z - a0 - (b1*g1 / (z - a1 - ...)))]
        
        # Shift z by E0 because Lanczos generated poles at Total Energy, 
        # but we want excitation energy.
        # w_grid represents "omega". The poles should be at (E_n - E_0).
        # So z - (E_n - E_0) = (z + E_0) - E_n
        z_shifted = w_grid + E0 
        
        if kind == "retarded":
            z_vals = z_shifted + 1j * eta
        else:
            z_vals = z_shifted - 1j * eta
            
        # Backward Iteration
        # Start with 0 at the bottom of the fraction
        cf = be.zeros_like(z_vals)
        
        # Iterate from last recursion step up to 0
        for k in range(current_krylov_dim - 2, -1, -1):
            # numerator: beta_{k+1} * gamma_{k+1}
            numer = betas[k] * gammas[k]
            
            # denominator: z - alpha_{k+1} - (next_term)
            denom = z_vals - alphas[k+1] - cf
            
            cf = numer / denom
            
        # Final step for G0
        # G = overlap / (z - alpha_0 - cf)
        G_final = overlap_0 / (z_vals - alphas[0] - cf)
        
        G_all[idx, :] = G_final

    # Shape Handling
    if len(mb_indices) == 1 and len(w_grid) == 1: return G_all[0, 0]
    if len(mb_indices) == 1: return G_all[0, :]
    if len(w_grid) == 1: return G_all[:, 0]
    return G_all

# ============================================================================
#! Finite-Temperature Lanczos Green's Function
# ============================================================================

def greens_function_lanczos_finite_T(
        omega               : Union[float, Array],
        hamiltonian         : Array,
        operator_a          : Array,
        eta                 : float             = 0.01,
        *,
        beta                : float             = 1.0,
        operator_b          : Optional[Array]   = None,
        n_random            : int               = 50,
        max_krylov          : int               = 100,
        backend             : str               = "default",
        kind                : str               = "retarded",
        lehmann_full        : bool              = False,
        seed                : Optional[int]     = None) -> Array:
    r"""
    Finite-Temperature Lanczos Method (FTLM) for many-body Green's functions.

    Method of Jaklic and Prelovsek, Phys. Rev. B 49, 5065 (1994).

    Instead of exact diagonalization, this method approximates thermal traces
    using stochastic sampling with random states:

        Tr[e^{-\beta H} O] ~ (N_Hilbert / R) \sum_{r=1}^R <r|e^{-\beta H} O|r>

    For each random state |r>, a Lanczos run gives approximate eigenpairs
    {\epsilon_i^{(r)}, |\phi_i^{(r)}>} within a small Krylov subspace.

    The Green's function is then:

        G_{AB}(\omega) ~ (N_Hilbert / ZR) \sum_{r,i,j} e^{-\beta \epsilon_i^{(r)}}
                         <r|\phi_i^{(r)}><\phi_i^{(r)}|A^\dagger|\phi_j^{(r)}>
                         <\phi_j^{(r)}|B|r> / (\omega + i\eta - (\epsilon_j - \epsilon_i))

    This captures finite-T behavior with:
    - R ~ 50 random states
    - M ~ 100 Krylov dimension
    Much cheaper than full ED for large Hilbert spaces.

    Parameters
    ----------
    omega : float or array
        Frequency grid for Green's function.
    hamiltonian : array, shape (N_Hilbert, N_Hilbert)
        Full Hamiltonian matrix in the many-body basis.
    operator_a : array, shape (N_Hilbert, N_Hilbert)
        Operator A in the many-body basis.
    eta : float, optional
        Broadening parameter (default: 0.01).
    beta : float, optional
        Inverse temperature \beta = 1/T (default: 1.0).
    operator_b : array, optional
        Operator B. If None, uses A^\dagger (default: None).
    n_random : int, optional
        Number of random states R for stochastic trace (default: 50).
    max_krylov : int, optional
        Maximum Krylov dimension M for each Lanczos run (default: 100).
    backend : str, optional
        Numerical backend (default: "default").
    kind : str, optional
        Type of Green's function: "retarded" or "advanced" (default: "retarded").
    lehmann_full : bool, optional
        If True (default), include both +/- frequency terms (full Lehmann).
        If False, only positive frequency (single-pole, for spectral functions).
    seed : int, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    Array
        Finite-temperature Green's function G(\omega).
        Shape: (len(omega),) if omega is array, scalar if omega is scalar.

    Notes
    -----
    FTLM advantages:
    - Scales to much larger systems than full ED
    - Statistical error decreases as 1/sqrt(R)
    - Each Lanczos run is independent (parallelizable)

    **Note**: Current implementation is functional but may need tuning for optimal
    accuracy. For production use, test convergence with your specific system.
    Recommended: n_random >= 50, max_krylov >= 100.

    For very large systems (>10^6 states), consider TPQ (thermal pure quantum)
    method which uses a single |psi_T> = e^{-\beta H/2}|r> instead.

    References
    ----------
    Jaklic & Prelovsek, Phys. Rev. B 49, 5065 (1994)
    Prelovsek & Bonca, "Ground State and Finite Temperature Lanczos Methods"

    Examples
    --------
    >>> # FTLM at finite temperature
    >>> G_ftlm = greens_function_lanczos_finite_T(
    ...     omega=omega_grid, hamiltonian=H, operator_a=A_op,
    ...     beta=10.0, n_random=50, max_krylov=100, eta=0.05
    ... )
    """
    be = get_backend(backend)
    
    # Setup
    H = be.asarray(hamiltonian, dtype=be.complex128)
    A = be.asarray(operator_a, dtype=be.complex128)
    
    if operator_b is None:
        B = A.conj().T
    else:
        B = be.asarray(operator_b, dtype=be.complex128)
    
    N_Hilbert = H.shape[0]
    
    # Omega array
    if isinstance(omega, (int, float, complex)):
        w = be.asarray([omega], dtype=be.complex128)
        scalar_omega = True
    else:
        w = be.asarray(omega, dtype=be.complex128)
        scalar_omega = False
    
    # Initialize result
    G_total = be.zeros((len(w),), dtype=be.complex128)
    Z_total = 0.0  # Partition function accumulator
    
    # Random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Loop over random states
    for r_idx in range(n_random):
        # Generate random state |r>
        if be.__name__ == 'numpy':
            r_vec = np.random.randn(N_Hilbert) + 1j * np.random.randn(N_Hilbert)
            r_vec = r_vec / np.linalg.norm(r_vec)
        else:
            # For JAX, use numpy then convert
            r_vec = np.random.randn(N_Hilbert) + 1j * np.random.randn(N_Hilbert)
            r_vec = r_vec / np.linalg.norm(r_vec)
            r_vec = be.asarray(r_vec, dtype=be.complex128)
        
        # Lanczos tridiagonalization starting from |r>
        alpha_list = []
        beta_list = []
        V_lanczos = []  # Lanczos vectors
        
        v_prev = be.zeros(N_Hilbert, dtype=be.complex128)
        v = r_vec
        V_lanczos.append(v)
        
        for j in range(max_krylov):
            Hv = H @ v
            a = be.vdot(v, Hv).real  # Should be real for Hermitian H
            alpha_list.append(a)
            
            # Residual
            if j == 0:
                r_res = Hv - a * v
            else:
                r_res = Hv - a * v - beta_list[-1] * v_prev
            
            b = be.sqrt(be.vdot(r_res, r_res).real)
            
            if b < 1e-14:
                break
            
            beta_list.append(b)
            v_prev = v
            v = r_res / b
            V_lanczos.append(v)
        
        M = len(alpha_list)
        
        # Build tridiagonal matrix
        T = be.zeros((M, M), dtype=be.complex128)
        for i in range(M):
            T[i, i] = alpha_list[i]
            if i < M - 1:
                T[i, i+1] = beta_list[i]
                T[i+1, i] = beta_list[i]
        
        # Diagonalize T
        if be.__name__ == 'numpy':
            eps, U_T = np.linalg.eigh(np.asarray(T))
        else:
            # For JAX
            import jax.numpy as jnp
            eps, U_T = jnp.linalg.eigh(T)
        
        eps = be.asarray(eps, dtype=be.float64)
        U_T = be.asarray(U_T, dtype=be.complex128)
        
        # Compute Boltzmann weights
        # Shift for numerical stability
        eps_min = be.min(eps)
        exp_factors = be.exp(-beta * (eps - eps_min))
        Z_r = be.sum(exp_factors)
        Z_total += Z_r
        
        # Matrix elements in Lanczos basis
        # <r|\phi_i> = U_T[0, i] (first component, since |v_0> = |r>)
        overlap_r = U_T[0, :]  # <r|\phi_i>
        
        # Transform operators to Lanczos basis
        V_matrix = be.stack(V_lanczos[:M], axis=1)  # (N_Hilbert, M)
        
        A_lanczos = V_matrix.conj().T @ A @ V_matrix  # (M, M)
        B_lanczos = V_matrix.conj().T @ B @ V_matrix  # (M, M)
        
        # Transform to eigenbasis of T
        A_eig = U_T.conj().T @ A_lanczos @ U_T
        B_eig = U_T.conj().T @ B_lanczos @ U_T
        
        # Compute contribution to Green's function
        for i in range(M):
            weight_i = exp_factors[i] * be.abs(overlap_r[i])**2
            
            if weight_i < 1e-15:
                continue
            
            for j in range(M):
                # Overlap factors: <r|\phi_i> and <\phi_j|r>
                overlap_factor = overlap_r[i].conj() * overlap_r[j]
                
                # Matrix element <\phi_i|A|\phi_j>
                A_ij = A_eig[i, j]
                B_ji = B_eig[j, i]
                
                deltaE = eps[j] - eps[i]
                denom_pos = w + 1j * eta - deltaE
                
                # Positive frequency contribution
                G_total += weight_i * overlap_factor * A_ij * B_ji / denom_pos
                
                if lehmann_full:
                    # Negative frequency contribution (for full Lehmann)
                    denom_neg = w + 1j * eta + deltaE
                    G_total -= weight_i * overlap_factor * B_ji * A_ij / denom_neg
    
    # Normalize by partition function and number of random states
    G = (N_Hilbert / (n_random * Z_total)) * G_total
    
    # Handle kind
    if kind == "advanced":
        G = G.conj()
    elif kind != "retarded":
        raise ValueError(f"Unsupported kind '{kind}'. Use 'retarded' or 'advanced'.")
    
    # Return
    if scalar_omega:
        return G[0]
    return G

# ----------------------------------------------------------------------------
#! End of file
# ----------------------------------------------------------------------------