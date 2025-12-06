"""
QES/general_python/algebra/spectral_backend.py

Unified spectral function backend for computing spectral properties
of quantum systems from eigenvalues, eigenvectors, or Lanczos coefficients.

Features:
  - Spectral function from Green's function
  - Direct spectral function from eigenvalues (no full diagonalization)
  - Momentum-resolved spectral function for periodic systems
  - Green's function in eigenbasis or Lanczos basis
  - Fast spectral calculations from Lanczos coefficients (tridiagonal form)
  - Integrated spectral weight
  - Peak detection
  - Backend support: NumPy and JAX

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

from typing import Optional, Union, Tuple, Literal, Any, Callable
import numpy as np
import numba as nb

from numpy.typing import NDArray
from sympy import denom

# Backend imports
try:
    from ...algebra.utils                       import JAX_AVAILABLE, get_backend
    from .krylov.spectral_backend_krylov        import *
    from .quadratic.spectral_backend_quadratic  import *
except ImportError:
    JAX_AVAILABLE   = False
    get_backend     = lambda x="default": np

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = None

# Type alias
Array = Union[np.ndarray, Any] # Any allows JAX arrays

# =============================================================================
# Green's Function Calculations - 0 TEMPERATURE
# =============================================================================

def greens_function_diagonal(omega: Union[float, Array], eigenvalues: Array, eta: float = 0.01, backend: str = "default") -> Array:
    r"""
    Compute diagonal Green's function directly from eigenvalues.
    
    G_nn(omega) = 1 / (omega + ieta  - E_n)
    
    Parameters
    ----------
    omega : float or array
        Frequency omega (can be complex). If array, broadcasts over eigenvalues.
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n.
    eta : float, optional
        Broadening parameter eta  (default: 0.01).
    backend : str, optional
        Numerical backend: 'numpy' or 'jax' (default: 'default').
        
    Returns
    -------
    Array
        Diagonal Green's function elements.
        - If omega is scalar: shape (N,)
        - If omega is array of shape (M,): shape (N, M)
        
    Notes
    -----
    Efficient for diagonal systems or when only diagonal elements are needed.
    """
    be              = get_backend(backend)
    eigenvalues     = be.asarray(eigenvalues, dtype=be.complex128)
    eta_complex     = be.asarray(eta, dtype=be.complex128)
    
    # Handle scalar or array omega with proper broadcasting
    if isinstance(omega, (int, float, complex)):
        omega_complex = be.asarray(omega, dtype=be.complex128)
        return 1.0 / (omega_complex + 1j * eta_complex - eigenvalues)
    else:
        omega_complex = be.asarray(omega, dtype=be.complex128)
        # Broadcasting: (N, M) = eigenvalues[:, None] and omega[None, :]
        return 1.0 / (omega_complex[None, :] + 1j * eta_complex - eigenvalues[:, None])
    
def greens_function_manybody(
        omega           : Union[float, Array],
        eigenvalues     : Array,
        operator_a      : Array,
        eta             : float             = 0.01,
        *,
        mb_states       : Optional[Union[Array, int]] = None, # vector of indices is sufficient, as we have the matrix elements via operators
        operator_b      : Optional[Array]   = None,
        backend         : str               = "default",
        kind            : str               = "retarded",
        statistic       : str               = "boson") -> Array:
    r"""
    The Green's function in the many-body representation
    (for any two operators A and B):
    
    $$
    G_{AB}(\omega) = \sum_{m,n} \frac{<0|A|n><n|B|0>}{\omega + i\eta  - (E_n - E_0)} + \frac{<0|B|n><n|A|0>}{\omega + i\eta  + (E_n - E_0)}
    $$

    where |m> is the many-body state (usually the ground state), 
    |n> are excited states with energies E_n.
    
    Parameters
    ----------
    omega : float or array-like
        Frequency omega (can be array for multiple frequencies).
    eigenvalues : array-like, shape (N,)
        Many-body eigenvalues E_n.
    operator_a : array-like, shape (N, N)
        Many-body operator A in the eigenbasis.
    eta : float, optional
        Broadening parameter eta  (default: 0.01).
    mb_states : array-like or int, optional
        Many-body states |m> to consider (indices). If int, single state index.
        If None, defaults to ground state |0>.
    operator_b : array-like, shape (N, N), optional
        Many-body operator B in the eigenbasis. If None, uses A^†.
    backend : str, optional
        Numerical backend to use (default: "default").
    kind : str, optional
        Kind of Green's function to compute (default: "retarded").
        Supported kinds:
            - "retarded"        (default)
            - "advanced"
            - "greater"
            - "lesser"
            - "timeordered"
    statistic : str, optional
        Statistic of the operators (default: "boson").
        Supported statistics:
            - "boson"       -> affects sign in greater/lesser GF (commutator)
            - "fermion"     -> affects sign in time-ordered GF   (anticommutator)
            
    Examples
    --------
    >>> G = greens_function_manybody(
    ...     omega=omega_grid, eigenvalues=E_all, operator_a=A_op,
    ...     eta=0.05, mb_states=0, backend="jax"
    ... )
    >>> G
    """
    
    be = get_backend(backend)
    if operator_a.ndim == 1:
        # We assume reference is Ground State (E[0])
        # operator_a represents vectors projected onto eigenbasis:
        # trans_a[n] = <n | A_op^dagger | 0> 
        # trans_b[n] = <n | B_op | 0>
        
        trans_a     = operator_a
        trans_b     = operator_b if operator_b is not None else trans_a
        
        # Consistency check
        if len(trans_a) != len(eigenvalues):
            raise ValueError(f"Vector shape {trans_a.shape} must match eigenvalues {eigenvalues.shape}")

        # Weights W_n = <0|A|n><n|B|0>
        # Note: <0|A|n> = (<n|A^dagger|0>)* = trans_a[n].conj()
        numerator   = trans_a.conj() * trans_b
        
        # Energies relative to Ground State
        deltaE      = eigenvalues - eigenvalues[0]
        
        # Vectorized Sum (Particle Term Only)
        # S(q, w) at T=0 only has the particle excitation term.
        denom       = (omega[:, None] - deltaE[None, :]).astype(be.complex128)
        
        if kind == "retarded":   denom += 1j * eta
        elif kind == "advanced": denom -= 1j * eta
            
        # Sum over eigenstates (axis 1)
        return be.sum(numerator[None, :] / denom, axis=1)

    # Matrix Input (General Case)
    else:
        # Reference states
        if mb_states is None:
            mb = be.asarray([0], dtype=int)
        elif isinstance(mb_states, int):
            mb = be.asarray([mb_states], dtype=int)
        else:
            mb = be.asarray(mb_states, dtype=int)

        if operator_b is None:
            operator_b = operator_a.conj().T 

        A       = be.asarray(operator_a)
        B       = be.asarray(operator_b)
        G       = be.zeros((len(mb), len(omega)), dtype=be.complex128)
        zeta    = 1.0 if statistic == "boson" else -1.0
        
        for i, m in enumerate(mb):
            deltaE = eigenvalues - eigenvalues[m]
            
            # A_mn = <m|A|n>, B_nm = <n|B|m>
            A_mn        = A[m, :]
            B_nm        = B[:, m]

            # Term 1: Particle (w - deltaE)
            denom_pos   = omega[:, None] - deltaE[None, :]
            if kind == "retarded": denom_pos += 1j * eta
            elif kind == "advanced": denom_pos -= 1j * eta
            
            G[i, :] += be.sum((A_mn[None, :] * B_nm[None, :]) / denom_pos, axis=1)

            # Term 2: Hole (w + deltaE) - irrelevant for T=0 structure factor usually
            denom_neg = omega[:, None] + deltaE[None, :]
            if kind == "retarded": denom_neg += 1j * eta
            elif kind == "advanced": denom_neg -= 1j * eta

            G[i, :] -= zeta * be.sum((B_nm[None, :] * A_mn[None, :]) / denom_neg, axis=1)

        # Formatting output shape
        if len(mb) == 1 and len(omega) == 1:    return G[0, 0]
        if len(mb) == 1:                        return G[0, :]
        if len(omega) == 1:                     return G[:, 0]
        return G

def greens_function_manybody_finite_T(
        omega           : Union[float, Array],
        eigenvalues     : Array,
        operator_a      : Array,
        eta             : float             = 0.01,
        *,
        beta            : float             = 1.0,
        operator_b      : Optional[Array]   = None,
        backend         : str               = "default",
        kind            : str               = "retarded",
        lehmann_full    : bool              = False) -> Array:
    r"""
    Finite-temperature many-body Green's function in the Lehmann representation.

    At finite temperature T > 0, the system is in a thermal mixed state:
        \rho = e^{-\beta H} / Z,  Z = Tr[e^{-\beta H}]

    Two conventions are supported:

    1. Full retarded GF (lehmann_full=True):
        G_{AB}^R(\omega) = (1/Z) \sum_{m,n} (e^{-\beta E_m} - e^{-\beta E_n})
                                    <m|A|n><n|B|m> / (\omega + i\eta - (E_n - E_m))

       This is the standard finite-T retarded Green's function.
       At T=0, reduces to sum over n from ground state m=0 (NOT the same as
       zero-T Lehmann which has both +/- frequency terms).

    2. Single-pole (lehmann_full=False, default):
        G_{AB}(\omega) = (1/Z) \sum_{m,n} e^{-\beta E_m} 
                                <m|A|n><n|B|m> / (\omega + i\eta - (E_n - E_m))

       Only positive-frequency poles. Useful for spectral functions where
       you want A(\omega) = -Im[G]/pi to have simple pole structure.

    Note: The zero-T greens_function_manybody() uses a DIFFERENT formula
    with both +/- frequency terms (particle addition + removal), which is
    appropriate for T=0 but does not directly match finite-T conventions.

    Parameters
    ----------
    omega : float or array-like
        Frequency \omega (can be array for multiple frequencies).
    eigenvalues : array-like, shape (N,)
        Many-body eigenvalues E_n.
    operator_a : array-like, shape (N, N)
        Many-body operator A in the eigenbasis.
    eta : float, optional
        Broadening parameter \eta (default: 0.01).
    beta : float, optional
        Inverse temperature \beta = 1/T (default: 1.0).
    operator_b : array-like, shape (N, N), optional
        Many-body operator B in the eigenbasis. If None, uses A^\dagger.
    backend : str, optional
        Numerical backend to use (default: "default").
    kind : str, optional
        Kind of Green's function (default: "retarded").
        Supported: "retarded", "advanced"
    lehmann_full : bool, optional
        If True (default), include (p_m - p_n) factor (standard retarded GF).
        If False, only p_m factor (single-pole, for spectral functions).
        
        **Important**: For computing spectral functions A(omega) = -Im[G]/pi,
        use lehmann_full=False to ensure A(omega) >= 0. The full Lehmann form
        with (p_m - p_n) can give negative spectral weights at finite T.

    Returns
    -------
    Array
        Finite-temperature Green's function G_{AB}(\omega).
        Shape: (len(omega),) if omega is array, scalar if omega is scalar.

    Notes
    -----
    This is the exact finite-T formula requiring full eigenspectrum.
    For large systems, use finite-temperature Lanczos methods (FTLM) instead.

    Physics:
    - All states contribute weighted by Boltzmann factor e^{-\beta E_m}
    - Full Lehmann (p_m - p_n): Standard retarded GF for response functions
    - Single-pole (p_m only): For spectral densities A(\omega) >= 0
    - Partition function Z normalizes the thermal ensemble
    
    Recommendation:
    - Use lehmann_full=True for dynamic susceptibilities, correlation functions
    - Use lehmann_full=False for spectral functions (ensures positivity)

    Examples
    --------
    >>> # For spectral functions (recommended: single-pole)
    >>> G_spec = greens_function_manybody_finite_T(
    ...     omega=omega_grid, eigenvalues=E_all, operator_a=A_op,
    ...     eta=0.05, beta=10.0, lehmann_full=False
    ... )
    >>> A_omega = spectral_function(greens_function=G_spec)  # A(omega) >= 0
    >>> 
    >>> # For response/correlation functions (full Lehmann)
    >>> G_resp = greens_function_manybody_finite_T(
    ...     omega=omega_grid, eigenvalues=E_all, operator_a=A_op,
    ...     eta=0.05, beta=10.0, lehmann_full=True
    ... )
    """
    be  = get_backend(backend)
    E   = be.asarray(eigenvalues, dtype=be.complex128)

    # prepare omega array
    if isinstance(omega, (int, float, complex)):
        w = be.asarray([omega], dtype=be.complex128)
        scalar_omega = True
    else:
        w = be.asarray(omega, dtype=be.complex128)
        scalar_omega = False

    # operator B defaults to A^\dagger
    if operator_b is None:
        operator_b = operator_a.conj().T

    A = be.asarray(operator_a, dtype=be.complex128)
    B = be.asarray(operator_b, dtype=be.complex128)

    # Compute Boltzmann weights e^{-\beta E_m}
    # Shift by minimum energy for numerical stability
    E_min = be.min(E.real)
    exp_factors = be.exp(-beta * (E.real - E_min))
    Z = be.sum(exp_factors)  # Partition function

    # Thermal weights
    p_m = exp_factors / Z  # p_m = e^{-\beta E_m} / Z

    # Output array
    G = be.zeros((len(w),), dtype=be.complex128)

    # Double sum over m, n with thermal weights
    for m in range(len(E)):
        # Skip negligible thermal populations
        if p_m[m] < 1e-15:
            continue

        # Energy differences \Delta E = E_n - E_m
        deltaE = E - E[m]

        # Matrix elements
        A_mn = A[m, :]  # <m|A|n>
        B_nm = B[:, m]  # <n|B|m>

        if lehmann_full:
            # Full Lehmann: (p_m - p_n) / (omega + ieta - deltaE)
            # This is the correct retarded GF at finite T
            for n in range(len(E)):
                if be.abs(deltaE[n]) < 1e-15:  # Skip m=n (gives 0/0)
                    continue
                    
                weight_diff = p_m[m] - p_m[n]  # e^{-beta E_m} - e^{-beta E_n}
                
                if be.abs(weight_diff) < 1e-15:
                    continue
                
                denom = w + 1j * eta - deltaE[n]
                G += weight_diff * (A_mn[n] * B_nm[n]) / denom
        else:
            # Single-pole: only p_m term (for spectral functions)
            denom_pos = w[None, :] + 1j * eta - deltaE[:, None]
            G += p_m[m] * be.sum((A_mn[:, None] * B_nm[:, None]) / denom_pos, axis=0)

    # Handle different Green's function types
    if kind == "advanced":
        G = G.conj()
    elif kind != "retarded":
        raise ValueError(f"Unsupported kind '{kind}'. Use 'retarded' or 'advanced'.")

    # Return scalar if input was scalar
    if scalar_omega:
        return G[0]
    return G

# =============================================================================
# Spectral Function Calculations
# =============================================================================

def spectral_function(  greens_function     : Optional[Array]   = None,
                        backend             : str               = "default",
                        eta                 : float             = 0.01,
                        *, 
                        kind                : Literal["retarded", "advanced", "greater", "lesser", "timeordered"] = "retarded",
                        omega               : Optional[Union[float, Array]] = None,
                        # for quadratic, if greens_function is None
                        q_eigenvalues       : Optional[Array]   = None,
                        q_eigenvectors      : Optional[Array]   = None,
                        q_operator_a        : Optional[Array]   = None,
                        q_operator_b        : Optional[Array]   = None,
                        q_occupations       : Optional[Array]   = None,
                        q_basis_transform   : bool              = True,
                        # for many-body, if greens_function is None
                        mb_eigenvalues      : Optional[Array]   = None,
                        mb_operator_a       : Optional[Array]   = None,
                        mb_operator_b       : Optional[Array]   = None,
                        mb_states           : Optional[Union[Array, int]] = None,
                        #
                        return_greens       : bool              = True
                    ) -> Array:
    r"""
    Compute spectral function.
    
    If greens_function is provided:
        A(\omega) = -(1/\pi) Im[G(\omega)]
    
    Otherwise, compute Green's function first from eigenvalues/eigenvectors,
    then extract spectral function.
    
    Parameters
    ----------
    greens_function : array-like, dtype complex, optional
        Pre-computed Green's function G(\omega). If provided, spectral function
        is computed directly.
    backend : str, optional
        Numerical backend (default: 'default').
    eta : float, optional
        Broadening parameter (default: 0.01).
    kind : str, optional
        Type of Green's function (default: 'retarded').
    omega : float or array-like, optional
        Frequency or frequencies (required if greens_function is None).
    q_eigenvalues : array-like, optional
        Eigenvalues for quadratic system.
    q_eigenvectors : array-like, optional
        Eigenvectors for quadratic system.
    q_operator_a : array-like, optional
        Operator A for quadratic Green's function.
    q_operator_b : array-like, optional
        Operator B for quadratic Green's function.
    q_occupations : array-like, optional
        Occupation numbers for quadratic system.
    q_basis_transform : bool, optional
        Whether to transform to eigenbasis (default: True).
    mb_eigenvalues : array-like, optional
        Eigenvalues for many-body system.
    mb_operator_a : array-like, optional
        Operator A for many-body Green's function.
    mb_operator_b : array-like, optional
        Operator B for many-body Green's function.
    mb_states : array-like or int, optional
        Many-body states for Lehmann representation.
        
    Returns
    -------
    Array
        Spectral function A(\omega), same shape as input, dtype real.
        Always non-negative for physical systems.
        
    Notes
    -----
    The spectral function is computed as A(\omega) = -(1/\pi) Im[G(\omega)].
    
    Examples
    --------
    >>> # From pre-computed Green's function
    >>> A = spectral_function(greens_function=G)
    >>> 
    >>> # From quadratic system eigenvalues
    >>> A = spectral_function(omega=0.5, q_eigenvalues=E, q_eigenvectors=U, eta=0.01)
    """
    be  = get_backend(backend)
    
    # Case 1: Green's function provided directly
    if greens_function is not None:
        G = be.asarray(greens_function, dtype=be.complex128)
    
    # Case 2: Compute Green's function from eigenvalues
    else:
        if omega is None:
            raise ValueError("omega must be provided when greens_function is None.")
        
        # Determine which system type based on provided parameters
        # Priority: many-body if mb_states provided, else quadratic
        if mb_eigenvalues is not None:
            # Many-body system
            G = greens_function_manybody(
                omega               = omega,
                eigenvalues         = mb_eigenvalues,
                operator_a          = mb_operator_a,
                eta                 = eta,
                mb_states           = mb_states if mb_states is not None else 0,
                operator_b          = mb_operator_b,
                backend             = backend,
                kind                = kind
            )
        elif q_eigenvalues is not None:
            # Quadratic system
            G = greens_function_quadratic(
                omega               = omega,
                eigenvalues         = q_eigenvalues,
                eigenvectors        = q_eigenvectors,
                eta                 = eta,
                operator_a          = q_operator_a,
                operator_b          = q_operator_b,
                occupations         = q_occupations,
                basis_transform     = q_basis_transform,
                backend             = backend
            )
        else:
            raise ValueError("Must provide either greens_function or eigenvalues (q_eigenvalues or mb_eigenvalues).")
    
    # Compute spectral function: A(omega) = -(1/pi) Im[G(omega)]
    A = -be.imag(G) / np.pi

    if return_greens:
        return A, G
    return A

# =============================================================================
# Momentum-Resolved Spectral Function
# =============================================================================

def spectral_function_k_resolved(
        omegas          : Array,
        k_points        : Array,
        eigenvalues_k   : Array,
        eta             : float = 0.01,
        backend         : str = "default") -> Array:
    r"""
    Compute momentum-resolved spectral function A(k, omega).
    
    For systems with momentum as a good quantum number.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Frequencies.
    k_points : array-like, shape (n_k, d) or (n_k,)
        k-points in Brillouin zone.
    eigenvalues_k : array-like, shape (n_k,) or (n_k, n_bands)
        Energy dispersion E(k). If 2D, includes multiple bands.
    eta : float, optional
        Broadening parameter (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Shape (n_k, n_omega) if eigenvalues_k is 1D,
        Shape (n_k, n_bands, n_omega) if eigenvalues_k is 2D.
        Dtype: real
        
    Examples
    --------
    >>> # 1D tight-binding dispersion
    >>> k_points = np.linspace(-np.pi, np.pi, 100)
    >>> E_k = -2 * np.cos(k_points)  # single band
    >>> omegas = np.linspace(-3, 3, 200)
    >>> A_k_omega = spectral_function_k_resolved(omegas, k_points, E_k)
    >>> # Shape: (100, 200) - band structure mapped to spectral function
    >>> plt.imshow(A_k_omega, aspect='auto')
    """
    be              = get_backend(backend)
    omegas          = be.asarray(omegas)
    eigenvalues_k   = be.asarray(eigenvalues_k)
    
    n_omega         = len(omegas)
    
    if eigenvalues_k.ndim == 1:
        # Single band: shape (n_k,)
        n_k     = len(eigenvalues_k)
        A       = be.zeros((n_k, n_omega), dtype=be.float64)
        
        for k_idx in range(n_k):
            E_k     = eigenvalues_k[k_idx]
            A_k     = (eta / np.pi) / ((omegas - E_k)**2 + eta**2)
            if JAX_AVAILABLE and backend == "jax":
                A   = A.at[k_idx, :].set(A_k)
            else:
                A[k_idx, :] = A_k
    
    elif eigenvalues_k.ndim == 2:
        # Multiple bands: shape (n_k, n_bands)
        n_k, n_bands    = eigenvalues_k.shape
        A               = be.zeros((n_k, n_bands, n_omega), dtype=be.float64)
        
        for k_idx in range(n_k):
            for band in range(n_bands):
                E_k = eigenvalues_k[k_idx, band]
                A_k = (eta / np.pi) / ((omegas - E_k)**2 + eta**2)
                if JAX_AVAILABLE and backend == "jax":
                    A = A.at[k_idx, band, :].set(A_k)
                else:
                    A[k_idx, band, :] = A_k
    else:
        raise ValueError("eigenvalues_k must be 1D or 2D")
    
    return A

# =============================================================================
# Integrated Spectral Weight
# =============================================================================

def integrated_spectral_weight(
        spectral_function   : Array,
        omega_grid          : Array,
        omega_min           : Optional[float] = None,
        omega_max           : Optional[float] = None,
        backend             : str = "default") -> Union[float, Array]:
    r"""
    Compute integrated spectral weight over an energy window.
    
    W = int_{omega_min}^{omega_max} domega A(omega)
    
    Parameters
    ----------
    spectral_function : array-like
        Spectral function A(omega), shape (..., n_omega).
    omega_grid : array-like, shape (n_omega,)
        Frequency grid (should be uniformly or quasi-uniformly spaced).
    omega_min, omega_max : float, optional
        Integration limits. If None, integrates over full range.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    float or Array
        Integrated spectral weight. Returns scalar if input is 1D,
        otherwise returns array integrated along last axis.
        
    Notes
    -----
    Uses trapezoidal rule for integration.
    For a complete spectrum, integrated weight should be close to N
    (number of states).
    """
    be = get_backend(backend)
    spectral_function = be.asarray(spectral_function, dtype=be.float64)
    omega_grid = be.asarray(omega_grid, dtype=be.float64)
    
    # Determine integration window
    if omega_min is None:
        omega_min = be.asarray(omega_grid[0])
    else:
        omega_min = be.asarray(omega_min, dtype=be.float64)
    
    if omega_max is None:
        omega_max = be.asarray(omega_grid[-1])
    else:
        omega_max = be.asarray(omega_max, dtype=be.float64)
    
    # Find indices for integration window
    mask = (omega_grid >= omega_min) & (omega_grid <= omega_max)
    omega_window = omega_grid[mask]
    
    # Extract spectral function in window
    if spectral_function.ndim == 1:
        A_window = spectral_function[mask]
    elif spectral_function.ndim == 2:
        A_window = spectral_function[:, mask]
    else:
        A_window = spectral_function[..., mask]
    
    # Integrate using trapezoidal rule
    # Note: np.trapz works with JAX arrays too
    return np.trapz(A_window, omega_window, axis=-1)

# =============================================================================
# Peak Detection and Analysis
# =============================================================================

def find_spectral_peaks(
        spectral_function: Array,
        omega_grid: Array,
        threshold: float = 0.1,
        min_distance: int = 5,
        backend: str = "default"
) -> Array:
    r"""
    Find peaks in spectral function.
    
    Parameters
    ----------
    spectral_function : array-like, shape (n_omega,)
        1D spectral function.
    omega_grid : array-like, shape (n_omega,)
        Frequency grid corresponding to spectral_function.
    threshold : float, optional
        Minimum relative height for peak detection (default: 0.1).
        Peaks below threshold * max(A) are ignored.
    min_distance : int, optional
        Minimum distance between peaks in grid points (default: 5).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Frequencies omega where peaks are detected.
        
    Notes
    -----
    Uses simple local maximum detection. For more sophisticated peak finding,
    consider scipy.signal.find_peaks.
    
    Examples
    --------
    >>> omegas = np.linspace(-5, 5, 200)
    >>> evals = np.array([-2, -1, 1, 2])
    >>> A = spectral_function_multi_omega(omegas, evals, diagonal_only=True).sum(axis=1)
    >>> peaks = find_spectral_peaks(A, omegas, threshold=0.2)
    """
    be = get_backend(backend)
    spectral_function = be.asarray(spectral_function, dtype=be.float64)
    omega_grid = be.asarray(omega_grid, dtype=be.float64)
    
    # Find local maxima
    peaks = []
    A_max = be.max(spectral_function)
    threshold_val = threshold * A_max
    
    for i in range(min_distance, len(spectral_function) - min_distance):
        if (spectral_function[i] > threshold_val and
            be.all(spectral_function[i] >= spectral_function[i - min_distance:i]) and
            be.all(spectral_function[i] >= spectral_function[i + 1:i + min_distance + 1])):
            peaks.append(omega_grid[i])
    
    return be.asarray(peaks) if peaks else be.asarray([])

# =============================================================================
# Thermal Utilities (Shared with Susceptibility Module)
# =============================================================================

def thermal_weights(
        eigenvalues : Array,
        temperature : float = 0.0,
        backend     : str = "default"
) -> Array:
    r"""
    Compute thermal occupation weights (Boltzmann/Fermi/Bose factors).
    
    For temperature T > 0:
        ρ_n = exp(-β(E_n - E_0)) / Z, where β = 1/k_B T
    
    For temperature T = 0:
        ρ_0 = 1, all others = 0 (ground state only)
    
    Parameters
    ----------
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n. Must be real.
    temperature : float, optional
        Temperature in energy units (default: 0). 
        If 0, returns ground-state-only weights.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array, shape (N,), dtype float
        Normalized thermal weights (sum to 1).
        
    Examples
    --------
    >>> E = np.array([-2, -1, 0, 1, 2])
    >>> rho_T0 = thermal_weights(E, temperature=0.0)     # [1, 0, 0, 0, 0]
    >>> rho_T1 = thermal_weights(E, temperature=1.0)     # Boltzmann distribution
    """
    be = get_backend(backend)
    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    
    if temperature <= 0:
        # T=0: ground state only
        rho = be.zeros(len(eigenvalues), dtype=be.float64)
        rho_list = rho.tolist() if hasattr(rho, 'tolist') else list(rho)
        rho_list[0] = 1.0
        return be.asarray(rho_list, dtype=be.float64)
    else:
        # Finite temperature
        beta = 1.0 / temperature
        E_min = be.min(eigenvalues)
        rho = be.exp(-beta * (eigenvalues - E_min))
        Z = be.sum(rho)
        return rho / Z

# =============================================================================
# Operator-Projected Spectral Functions (Many-body)
# =============================================================================

def operator_spectral_function_lehmann(
        omega               : float,
        eigenvalues         : Array,
        eigenvectors        : Array,
        operator            : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0,
        backend             : str = "default"
) -> float:
    r"""
    Compute operator-projected spectral function using Lehmann representation.
    
    A_O(omega ) = \Sum _{m,n} (ρ_m - ρ_n) |<m|O|n>|² delta (omega  - (E_n - E_m))
    
    This represents the contribution of operator O to the spectrum:
    - Used for spin/charge/current response in many-body systems
    - Naturally accounts for finite temperature via ρ_n
    - Gives matrix elements <m|O|n> between ALL many-body eigenstates
    
    Parameters
    ----------
    omega : float
        Frequency omega .
    eigenvalues : array-like, shape (N,)
        Hamiltonian eigenvalues E_n.
    eigenvectors : array-like, shape (N, N)
        Hamiltonian eigenvectors (columns = eigenstates).
    operator : array-like, shape (N, N)
        Many-body operator O (e.g., magnetization, density).
    eta : float, optional
        Broadening parameter (default: 0.01). Controls peak width.
    temperature : float, optional
        Temperature in energy units (default: 0).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    float
        Operator-projected spectral function A_O(omega ).
        
    Notes
    -----
    At T=0, only ground-state contributions (|<n|O|0>|²) appear.
    At T>0, thermal factors ρ_m ≠ ρ_n activate all transitions.
    
    This is the foundation for computing susceptibilities:
    χ_OO(omega ) can be reconstructed from this for all operators.
    
    See Also
    --------
    operator_spectral_function_multi_omega : Vectorized over frequencies
    thermal_weights : Thermal occupation factors
    
    Examples
    --------
    >>> E, V = np.linalg.eigh(H)
    >>> S_z = compute_spin_operator()  # Your magnetization
    >>> A_sz = operator_spectral_function_lehmann(omega=0.5, 
    ...                                             eigenvalues=E, 
    ...                                             eigenvectors=V,
    ...                                             operator=S_z,
    ...                                             temperature=1.0)
    """
    be = get_backend(backend)
    
    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    eigenvectors = be.asarray(eigenvectors, dtype=be.complex128)
    operator = be.asarray(operator, dtype=be.complex128)
    
    # Ensure operator is 2D and properly shaped
    if operator.ndim == 0:
        # Scalar
        operator = be.eye(len(eigenvalues), dtype=be.complex128) * operator
    elif operator.ndim == 1:
        # 1D array - make it diagonal
        operator = be.diag(operator)
    
    N = len(eigenvalues)
    
    # Transform operator to eigenbasis: O_nm = <n|O|m>
    O_eigen = eigenvectors.conj().T @ operator @ eigenvectors
    
    # Thermal weights
    rho = thermal_weights(eigenvalues, temperature, backend)
    
    # Lorentzian broadening kernel
    def lorentzian(delta_E):
        return (eta / np.pi) / (delta_E**2 + eta**2)
    
    # Lehmann sum over all transitions
    A = 0.0
    for m in range(N):
        for n in range(N):
            if be.abs(rho[m] - rho[n]) < 1e-14:
                continue
            delta_E = omega - (eigenvalues[n] - eigenvalues[m])
            matrix_element_sq = be.abs(O_eigen[m, n])**2
            A += (rho[m] - rho[n]) * matrix_element_sq * lorentzian(delta_E)
    
    return float(be.real(A))

def operator_spectral_function_multi_omega(
        omegas              : Array,
        eigenvalues         : Array,
        eigenvectors        : Array,
        operator            : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0,
        backend             : str = "default"
) -> Array:
    r"""
    Compute operator-projected spectral function for multiple frequencies.
    
    Vectorized version of operator_spectral_function_lehmann.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Frequency grid.
    eigenvalues : array-like, shape (N,)
        Eigenvalues.
    eigenvectors : array-like, shape (N, N)
        Eigenvectors.
    operator : array-like, shape (N, N)
        Operator O.
    eta : float, optional
        Broadening (default: 0.01).
    temperature : float, optional
        Temperature (default: 0).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array, shape (n_omega,), dtype float
        Spectral function A_O(omega ) for each omega .
        
    Examples
    --------
    >>> omegas = np.linspace(-5, 5, 200)
    >>> A_sz = operator_spectral_function_multi_omega(omegas, E, V, S_z, temperature=1.0)
    >>> plt.plot(omegas, A_sz)
    """
    be = get_backend(backend)
    omegas = be.asarray(omegas)
    
    n_omega = len(omegas)
    A = be.zeros(n_omega, dtype=be.float64)
    
    for i, omega in enumerate(omegas):
        # Convert omega to Python float (avoid numpy scalar issues)
        omega_val = float(omega) if hasattr(omega, '__float__') else omega
        A_i = operator_spectral_function_lehmann(
            omega_val, eigenvalues, eigenvectors, operator,
            eta=eta, temperature=temperature, backend=backend
        )
        if JAX_AVAILABLE and backend == "jax":
            A = A.at[i].set(A_i)
        else:
            A[i] = A_i
    
    return A

# =============================================================================
# Bubble Susceptibility (Quadratic / Mean-Field)
# =============================================================================

def susceptibility_bubble(
        omega               : float,
        eigenvalues         : Array,
        vertex              : Optional[Array] = None,
        occupation          : Optional[Array] = None,
        eta                 : float = 0.01,
        backend             : str = "default"
) -> complex:
    r"""
    Compute bare susceptibility (Lindhard function) from single-particle spectrum.
    
    χ⁰(omega ) = \Sum _{m,n} (f_m - f_n) |V_{mn}|² / (omega  + i\eta - (E_n - E_m))
    
    This is the bubble diagram contribution, valid for quadratic or mean-field
    Hamiltonians where Wick's theorem holds.
    
    Parameters
    ----------
    omega : float or complex
        Frequency omega .
    eigenvalues : array-like, shape (N,)
        Single-particle energies E_n (not many-body eigenvalues!).
    vertex : array-like, shape (N, N), optional
        Vertex/coupling matrix V_{mn} (e.g., velocity for conductivity).
        If None, assumes identity (density-density response).
    occupation : array-like, shape (N,), optional
        Single-particle occupations f_n (Fermi/Bose factors).
        If None, assumes ground state (f_n = Θ(-E_n) for fermions at T=0).
    eta : float, optional
        Broadening (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    complex
        Bare susceptibility χ⁰(omega ) at this frequency.
        
    Notes
    -----
    The occupation f_n should be computed from the single-particle eigenenergies,
    NOT from the many-body Hamiltonian. Typical choices:
    
    - Fermions at T=0: f_n = Θ(-E_n) [filled orbitals below Fermi]
    - Fermions at T>0: f_n = 1/(1 + exp(β(E_n - μ)))
    - Bosons: f_n = 1/(exp(β E_n) - 1)
    
    For density-density response, vertex = I (identity).
    For charge/current response, vertex encodes the operator O_mn = <m|O|n>.
    
    See Also
    --------
    conductivity_kubo_bubble : Optical conductivity \sigma(omega ) from bubbles
    
    Examples
    --------
    >>> # Tight-binding chain: E_k = -2cos(k)
    >>> E_k = np.array([-2, -1.8, -1.5, 0, 1.5, 1.8, 2])
    >>> f = 1/(1 + np.exp(E_k/(0.1)))  # Fermi-Dirac at T=0.1
    >>> chi0 = susceptibility_bubble(omega=0.5, eigenvalues=E_k, occupation=f)
    """
    be = get_backend(backend)
    
    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    omega_complex = be.asarray(omega, dtype=be.complex128)
    eta_complex = be.asarray(eta, dtype=be.complex128)
    
    N = len(eigenvalues)
    
    if vertex is None:
        vertex = be.eye(N, dtype=be.complex128)
    else:
        vertex = be.asarray(vertex, dtype=be.complex128)
    
    if occupation is None:
        # T=0: all states below Fermi filled (filled Fermi sea)
        occupation = be.where(eigenvalues < 0, 1.0, 0.0)
    else:
        occupation = be.asarray(occupation, dtype=be.float64)
    
    # Bubble: χ⁰ = \Sum _{mn} (f_m - f_n) |V_{mn}|² / (omega  + i\eta - (E_n - E_m))
    chi = 0.0 + 0.0j
    for m in range(N):
        for n in range(N):
            if be.abs(occupation[m] - occupation[n]) < 1e-14:
                continue
            denom = omega_complex + 1j * eta_complex - (eigenvalues[n] - eigenvalues[m])
            V_mn_sq = be.abs(vertex[m, n])**2
            chi += (occupation[m] - occupation[n]) * V_mn_sq / denom
    
    return complex(chi)

def susceptibility_bubble_multi_omega(
        omegas              : Array,
        eigenvalues         : Array,
        vertex              : Optional[Array] = None,
        occupation          : Optional[Array] = None,
        eta                 : float = 0.01,
        backend             : str = "default"
) -> Array:
    r"""
    Compute bare susceptibility for multiple frequencies.
    
    Vectorized version of susceptibility_bubble.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Frequency grid.
    eigenvalues : array-like, shape (N,)
        Single-particle energies.
    vertex : array-like, shape (N, N), optional
        Vertex matrix. If None, uses identity.
    occupation : array-like, shape (N,), optional
        Occupations. If None, assumes T=0 filled Fermi sea.
    eta : float, optional
        Broadening (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array, shape (n_omega,), dtype complex
        χ⁰(omega ) for each omega .
    """
    be = get_backend(backend)
    omegas = be.asarray(omegas)
    
    n_omega = len(omegas)
    chi = be.zeros(n_omega, dtype=be.complex128)
    
    for i, omega in enumerate(omegas):
        chi_i = susceptibility_bubble(
            float(omega), eigenvalues, vertex, occupation,
            eta=eta, backend=backend
        )
        if JAX_AVAILABLE and backend == "jax":
            chi = chi.at[i].set(chi_i)
        else:
            chi[i] = chi_i
    
    return chi

# =============================================================================
# Kubo Conductivity (Quadratic Systems)
# =============================================================================

def conductivity_kubo_bubble(
        omega               : float,
        eigenvalues         : Array,
        velocity_matrix     : Array,
        occupation          : Optional[Array] = None,
        eta                 : float = 0.01,
        backend             : str = "default"
) -> complex:
    r"""
    Compute optical conductivity from Kubo-Greenwood formula (bubble diagram).
    
    \sigma(omega ) = (1/(2omega )) \Sum _{mn} (f_m - f_n) |<m|v|n>|² / (omega  + i\eta - (E_n - E_m))
    
    This is the single-particle bubble contribution, valid for non-interacting
    or mean-field Hamiltonians.
    
    Parameters
    ----------
    omega : float or complex
        Frequency omega  (should be real for physical conductivity).
    eigenvalues : array-like, shape (N,)
        Single-particle eigenenergies.
    velocity_matrix : array-like, shape (N, N)
        Velocity/momentum matrix elements v_mn = <m|v|n>.
        Typically: v_mn = ∂H/∂k in k-space, or related to hopping in real space.
    occupation : array-like, shape (N,), optional
        Single-particle occupations f_n.
        If None, assumes ground state (T=0).
    eta : float, optional
        Broadening/scattering rate (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    complex
        Conductivity \sigma(omega ).
        
    Notes
    -----
    The formula is:
        \sigma(omega ) = (1/(2omega )) χ⁰_{vv}(omega )
    
    Real part Re[\sigma(omega )] gives absorptive conductivity.
    Imaginary part Im[\sigma(omega )] gives reactive effects.
    
    For optical conductivity (intraband vs interband):
    - Intraband: f_m ≠ f_n within partially filled band
    - Interband: f_m ≠ f_n across band gap
    
    See Also
    --------
    susceptibility_bubble : General bubble susceptibility
    
    Examples
    --------
    >>> # Graphene-like: linear dispersion
    >>> E = np.array([-1, -0.5, 0, 0.5, 1])
    >>> v = np.array([[0, 1, 0, 0, 0],
    ...               [1, 0, 1, 0, 0],
    ...               [0, 1, 0, 1, 0],
    ...               [0, 0, 1, 0, 1],
    ...               [0, 0, 0, 1, 0]])
    >>> f = (E < 0).astype(float)  # filled below E=0
    >>> sigma = conductivity_kubo_bubble(omega=0.5, eigenvalues=E, 
    ...                                   velocity_matrix=v, occupation=f)
    """
    be = get_backend(backend)
    
    omega_complex = be.asarray(omega, dtype=be.complex128)
    
    # Compute χ⁰ from velocity vertex
    chi_vv = susceptibility_bubble(
        omega, eigenvalues, vertex=velocity_matrix,
        occupation=occupation, eta=eta, backend=backend
    )
    
    # \sigma(omega ) = (1/(2omega )) χ⁰_{vv}(omega )
    # Handle omega =0 carefully
    if be.abs(omega_complex) < 1e-14:
        return 0.0 + 0.0j
    
    sigma = chi_vv / (2.0 * omega_complex)
    return complex(sigma)

# =============================================================================
# Kramers-Kronig Relations
# =============================================================================

def kramers_kronig_transform(
        Im_chi           : Array,
        omega_grid       : Array,
        backend          : str = "default"
) -> Array:
    r"""
    Reconstruct Re[χ(omega )] from Im[χ(omega )] using Kramers-Kronig relations.
    
    Re[χ(omega )] = (2/pi) P int_0^∞ domega ' (omega ' Im[χ(omega ')] / (omega '² - omega ²))
    
    where P denotes principal value.
    
    Parameters
    ----------
    Im_chi : array-like, shape (n_omega,)
        Imaginary part of susceptibility (must be real-valued).
    omega_grid : array-like, shape (n_omega,)
        Frequency grid. Should be dense and include positive/negative omega  symmetrically
        for best accuracy.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array, shape (n_omega,)
        Real part Re[χ(omega )].
        
    Notes
    -----
    This uses a simple numerical Hilbert transform via integration.
    For high accuracy, use scipy.integrate.quad with analytic kernels.
    
    Assumes Im[χ(omega )] decays at large |omega |.
    
    See Also
    --------
    scipy.integrate.hilbert : More sophisticated Hilbert transform
    
    Examples
    --------
    >>> omegas = np.linspace(-10, 10, 500)
    >>> Im_chi = -0.1 / ((omegas - 1)**2 + 0.1**2)  # Lorentzian
    >>> Re_chi = kramers_kronig_transform(Im_chi, omegas)
    """
    from scipy import integrate
    
    be = get_backend(backend)
    Im_chi = np.asarray(Im_chi, dtype=np.float64)
    omega_grid = np.asarray(omega_grid, dtype=np.float64)
    
    n_omega = len(omega_grid)
    Re_chi = np.zeros(n_omega, dtype=np.float64)
    
    for i, omega in enumerate(omega_grid):
        # Avoid division by zero
        mask = np.abs(omega_grid - omega) > 1e-10
        omega_prime = omega_grid[mask]
        Im_chi_prime = Im_chi[mask]
        
        # Integrand: omega ' Im[χ(omega ')] / (omega '² - omega ²)
        integrand = omega_prime * Im_chi_prime / (omega_prime**2 - omega**2)
        
        # Integrate (could use more sophisticated method)
        integral = np.trapz(integrand, omega_prime)
        Re_chi[i] = (2.0 / np.pi) * integral
    
    return be.asarray(Re_chi, dtype=be.float64)


# =============================================================================
#! SpectralModule - Hamiltonian wrapper for spectral functions
# =============================================================================

class SpectralModule:
    """
    Spectral function module for Hamiltonians.
    
    Provides convenient access to Green's functions, spectral functions,
    and dynamical correlation functions. Supports both full ED and Lanczos.
    
    Usage
    -----
    >>> hamil.diagonalize()
    >>> spec = hamil.spectral
    >>> 
    >>> # Green's function at frequencies omega
    >>> omega = np.linspace(-5, 5, 100)
    >>> G = spec.greens_function(omega, operator_left, operator_right)
    >>> 
    >>> # Spectral function A(ω) = -Im[G(ω)]/π
    >>> A = spec.spectral_function(omega, operator)
    >>> 
    >>> # Dynamic structure factor (via Lanczos, no full diag needed)
    >>> S_q_omega = spec.dynamic_structure_factor(omega, S_q_operator)
    """
    
    def __init__(self, hamiltonian):
        """
        Initialize the spectral module.
        
        Parameters
        ----------
        hamiltonian : Hamiltonian
            The Hamiltonian instance.
        """
        self._hamil = hamiltonian
        
    def _check_diagonalized(self):
        """Check that Hamiltonian is diagonalized."""
        if self._hamil._eig_val is None or len(self._hamil._eig_val) == 0:
            raise RuntimeError("Hamiltonian must be diagonalized. Call hamil.diagonalize()")
    
    @property
    def energies(self) -> Array:
        """Get eigenvalues."""
        self._check_diagonalized()
        return self._hamil._eig_val
    
    @property
    def eigenstates(self) -> Array:
        """Get eigenstates."""
        self._check_diagonalized()
        return self._hamil._eig_vec
    
    @property
    def ground_state_energy(self) -> float:
        """Ground state energy E_0."""
        return self.energies[0]
    
    def greens_function(self,
                        omega: Array,
                        operator_left: Array = None,
                        operator_right: Array = None,
                        state_idx: int = 0,
                        eta: float = 0.05,
                        use_lanczos: bool = False,
                        max_krylov: int = 200) -> Array:
        """
        Compute Green's function G(ω) = <ψ|A† (ω - H + E_0 + iη)⁻¹ B|ψ>.
        
        Parameters
        ----------
        omega : Array
            Frequency points.
        operator_left : Array, optional
            Left operator A (in eigenbasis or Hilbert space).
            If None, uses identity.
        operator_right : Array, optional
            Right operator B. If None, uses operator_left.
        state_idx : int
            Reference state index (default: ground state).
        eta : float
            Broadening parameter.
        use_lanczos : bool
            Use Lanczos continued fraction (no full diagonalization needed).
        max_krylov : int
            Max Krylov dimension for Lanczos.
            
        Returns
        -------
        Array
            Green's function G(ω), shape (len(omega),).
        """
        omega = np.asarray(omega)
        
        if use_lanczos:
            from .krylov.spectral_backend_krylov import greens_function_lanczos_vectors
            
            psi = self.eigenstates[:, state_idx] if self._hamil._eig_vec is not None else None
            if psi is None:
                raise ValueError("Need at least one eigenstate for reference.")
            
            v_right = operator_right @ psi if operator_right is not None else psi
            v_left = operator_left @ psi if operator_left is not None else v_right
            
            H = self._hamil.hamil if self._hamil.hamil is not None else self._hamil.matvec_fun
            
            return greens_function_lanczos_vectors(
                omega, H, v_right, v_left,
                E0=self.ground_state_energy,
                eta=eta, max_krylov=max_krylov
            )
        else:
            self._check_diagonalized()
            
            # Transform operators to eigenbasis
            if operator_left is not None:
                trans_L = self.eigenstates.conj().T @ operator_left @ self.eigenstates[:, state_idx]
            else:
                trans_L = self.eigenstates[:, state_idx]
                
            if operator_right is not None:
                trans_R = self.eigenstates.conj().T @ operator_right @ self.eigenstates[:, state_idx]
            else:
                trans_R = trans_L
            
            return greens_function_manybody(
                omega, self.energies, trans_L, trans_R, eta=eta
            )
    
    def spectral_function(self,
                         omega: Array,
                         operator: Array = None,
                         state_idx: int = 0,
                         eta: float = 0.05,
                         use_lanczos: bool = False,
                         max_krylov: int = 200) -> Array:
        """
        Compute spectral function A(ω) = -Im[G(ω)]/π.
        
        Parameters
        ----------
        omega : Array
            Frequency points.
        operator : Array, optional
            Operator for spectral function. If None, uses diagonal.
        state_idx : int
            Reference state.
        eta : float
            Broadening.
        use_lanczos : bool
            Use Lanczos method.
        max_krylov : int
            Max Krylov dimension.
            
        Returns
        -------
        Array
            Spectral function A(ω).
        """
        G = self.greens_function(
            omega, operator, operator, state_idx,
            eta, use_lanczos, max_krylov
        )
        return -np.imag(G) / np.pi
    
    def dynamic_structure_factor(self,
                                omega: Array,
                                operator: Array,
                                state_idx: int = 0,
                                eta: float = 0.05,
                                use_lanczos: bool = True,
                                max_krylov: int = 200) -> Array:
        """
        Compute dynamic structure factor S(ω) for an operator.
        
        S(ω) = -Im[G(ω)]/π where G = <gs|O†(ω-H+E_0+iη)⁻¹O|gs>
        
        Parameters
        ----------
        omega : Array
            Frequency points.
        operator : Array
            Operator O (e.g., S_q for spin structure factor).
        state_idx : int
            Reference state (default: ground state).
        eta : float
            Broadening.
        use_lanczos : bool
            Use Lanczos (default: True, more memory efficient).
        max_krylov : int
            Max Krylov dimension.
            
        Returns
        -------
        Array
            Dynamic structure factor S(ω).
        """
        return self.spectral_function(
            omega, operator, state_idx, eta, use_lanczos, max_krylov
        )
    
    def susceptibility(self,
                      omega: Array,
                      operator_A: Array,
                      operator_B: Array = None,
                      eta: float = 0.05,
                      temperature: float = 0.0) -> Array:
        """
        Compute dynamical susceptibility χ_AB(ω).
        
        χ(ω) = i ∫ dt e^{iωt} <[A(t), B]> θ(t)
        
        Parameters
        ----------
        omega : Array
            Frequency points.
        operator_A : Array
            First operator.
        operator_B : Array, optional
            Second operator. If None, uses operator_A.
        eta : float
            Broadening.
        temperature : float
            Temperature (0 = ground state only).
            
        Returns
        -------
        Array
            Susceptibility χ(ω).
        """
        self._check_diagonalized()
        
        if operator_B is None:
            operator_B = operator_A
            
        if temperature == 0.0:
            # Zero temperature: ground state only
            return self.greens_function(
                omega, operator_A, operator_B, 
                state_idx=0, eta=eta, use_lanczos=False
            )
        else:
            # Finite temperature
            return greens_function_manybody_finite_T(
                omega, self.energies, operator_A, operator_B,
                temperature=temperature, eta=eta
            )
    
    def help(self):
        """Print help for spectral module."""
        print("""
        SpectralModule - Spectral functions for Hamiltonians
        =====================================================
        
        Requires: hamil.diagonalize() (or use_lanczos=True for some methods)
        
        Methods:
        --------
        greens_function(omega, A, B)        - G(ω) = <ψ|A†(ω-H+E₀+iη)⁻¹B|ψ>
        spectral_function(omega, O)         - A(ω) = -Im[G]/π
        dynamic_structure_factor(omega, O)  - S(ω) for dynamical correlations
        susceptibility(omega, A, B)         - χ_AB(ω) response function
        
        Example:
        --------
        >>> omega = np.linspace(-5, 5, 200)
        >>> 
        >>> # Spectral function from full ED
        >>> A_omega = hamil.spectral.spectral_function(omega, Sz_op, eta=0.1)
        >>> 
        >>> # Dynamic structure factor via Lanczos (no full diag)
        >>> S_q = hamil.spectral.dynamic_structure_factor(
        ...     omega, S_q_operator, use_lanczos=True, max_krylov=100
        ... )
        """)


def get_spectral_module(hamiltonian) -> SpectralModule:
    """Factory function to create spectral module."""
    return SpectralModule(hamiltonian)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Green's functions
    'greens_function_diagonal',
    'greens_function_quadratic',
    'greens_function_quadratic_finite_T',
    'greens_function_manybody',
    'greens_function_lanczos',
    # Spectral functions
    'spectral_function',
    'spectral_function_k_resolved',
    # Integrated quantities
    'integrated_spectral_weight',
    # Analysis
    'find_spectral_peaks',
    # Thermal utilities
    'thermal_weights',
    # Operator-projected spectral (many-body)
    'operator_spectral_function_lehmann',
    'operator_spectral_function_multi_omega',
    # Bubble susceptibilities (quadratic)
    'susceptibility_bubble',
    'susceptibility_bubble_multi_omega',
    # Kubo conductivity
    'conductivity_kubo_bubble',
    # Kramers-Kronig
    'kramers_kronig_transform',
    # Module wrapper
    'SpectralModule',
    'get_spectral_module',
]

# =============================================================================
#! EOF
# =============================================================================