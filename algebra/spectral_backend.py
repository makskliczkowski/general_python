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
from numpy.typing import NDArray

# Backend imports
try:
    from .utils import JAX_AVAILABLE, get_backend
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

# =============================================================================
# Green's Function Calculations
# =============================================================================

def greens_function_eigenbasis(
        omega           : float,
        eigenvalues     : Array,
        eigenvectors    : Optional[Array] = None,
        eta             : float = 0.01,
        backend         : str = "default") -> Array:
    r"""
    Compute Green's function in eigenbasis.
    
    If eigenvectors provided:
        G(omega) = Sum_n |v_n><v_n| / (omega + ieta  - E_n)
    
    If eigenvectors None (diagonal case):
        G(omega) = Sum_n 1 / (omega + ieta  - E_n)  (scalar Green's function)
    
    Parameters
    ----------
    omega : float
        Frequency omega (can be complex).
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n.
    eigenvectors : array-like, shape (N, N), optional
        Eigenvectors as columns. If None, returns diagonal Green's function.
    eta : float, optional
        Broadening parameter eta  (default: 0.01).
    backend : str, optional
        Numerical backend: 'numpy' or 'jax' (default: 'default').
        
    Returns
    -------
    Array
        If eigenvectors provided: Green's function matrix, shape (N, N), dtype complex
        If eigenvectors None: diagonal elements, shape (N,), dtype complex
        
    Notes
    -----
    For diagonal cases, computes:
        G_nn(omega) = 1 / (omega + ieta  - E_n)
    
    For full matrix (with eigenvectors):
        G_nm(omega) = Sum_n (V_n V_n^dagger )_{nm} / (omega + ieta  - E_n)
    """
    be              = get_backend(backend)
    eigenvalues     = be.asarray(eigenvalues)
    
    # Ensure complex type for Green's function
    omega_complex   = be.asarray(omega, dtype=be.complex128)
    eta_complex     = be.asarray(eta, dtype=be.complex128)
    
    # Denominator: omega + ieta  - E_n
    denom           = omega_complex + 1j * eta_complex - eigenvalues
    
    if eigenvectors is None:
        # Diagonal Green's function (no eigenvectors provided)
        return 1.0 / denom
    
    # Full Green's function: G = V (1/D) V^H
    eigenvectors    = be.asarray(eigenvectors, dtype=be.complex128)
    
    # Compute G = V @ diag(1/denom) @ V^H
    G = eigenvectors @ be.diag(1.0 / denom) @ eigenvectors.T.conj()
    
    return G

def greens_function_lanczos(
        omega       : float,
        alpha       : Array,
        beta        : Array,
        eta         : float = 0.01,
        backend     : str = "default") -> complex:
    r"""
    Compute Green's function from Lanczos tridiagonal matrix.
    
    Uses continued fraction representation of the Green's function
    with the tridiagonal Lanczos matrix.
    
    G(omega) = 1 / (omega + ieta  - alpha _0 - beta _0^2 / (omega + ieta  - alpha _1 - beta _1^2 / (...)))
    
    This is efficient for computing spectral function without full
    diagonalization of the original matrix.
    
    Parameters
    ----------
    omega : float or complex
        Frequency omega.
    alpha : array-like, shape (m,)
        Diagonal elements of tridiagonal Lanczos matrix.
    beta : array-like, shape (m-1,)
        Off-diagonal elements (superdiagonal).
    eta : float, optional
        Broadening parameter (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    complex
        Green's function value at frequency omega.
        
    Notes
    -----
    Uses backward recursion for numerical stability.
    
    References
    ----------
    Haydock, R. (1980). "The Recursive Solution of the Schrodinger Equation".
    Computer Physics Communications, 20(1), 11-16.
    """
    be              = get_backend(backend)
    alpha           = be.asarray(alpha, dtype=be.complex128)
    beta            = be.asarray(beta, dtype=be.complex128)

    omega_complex   = be.asarray(omega, dtype=be.complex128)
    eta_complex     = be.asarray(eta, dtype=be.complex128)

    m               = len(alpha)
    
    # Start from bottom and work backwards (continued fraction)
    # g_m = 1 / (omega + ieta  - alpha _{m-1})
    g               = 1.0 / (omega_complex + 1j * eta_complex - alpha[-1])
    
    # Recursively apply: g_j = 1 / (omega + ieta  - alpha _j - beta _j^2 * g_{j+1})
    for j in range(m - 2, -1, -1):
        beta_j = beta[j] if j < len(beta) else 0.0
        g      = 1.0 / (omega_complex + 1j * eta_complex - alpha[j] - beta_j**2 * g)
    
    return g

# =============================================================================
# Spectral Function Calculations
# =============================================================================

def spectral_function(greens_function: Array, backend: str = "default") -> Array:
    r"""
    Compute spectral function from Green's function.
    
    A(omega) = -(1/pi) Im[G(omega)]
    
    Parameters
    ----------
    greens_function : array-like, dtype complex
        Green's function G(omega).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Spectral function A(omega), same shape as greens_function, dtype real.
        
    Notes
    -----
    Spectral function is always real and non-negative for physical systems.
    """
    be  = get_backend(backend)
    G   = be.asarray(greens_function, dtype=be.complex128)
    
    # A(omega) = -(1/pi) Im[G(omega)]
    return -be.imag(G) / np.pi

def spectral_function_diagonal(
        omega       : float,
        eigenvalues : Array,
        eta         : float = 0.01,
        backend     : str = "default") -> Array:
    r"""
    Compute diagonal spectral function directly from eigenvalues.
    
    A_nn(omega) = (eta /pi) / [(omega - E_n)^2 + eta ^2]
    
    This is a Lorentzian distribution centered at each eigenvalue.
    
    Parameters
    ----------
    omega : float
        Frequency omega.
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n. Must be real for physical systems.
    eta : float, optional
        Broadening parameter (default: 0.01). Controls peak width.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Diagonal spectral function A_nn(omega), shape (N,), dtype real.
        
    Notes
    -----
    For diagonal systems or when considering only diagonal elements,
    computes the spectral function without the full matrix.
    Lorentzian:
        A(omega) = (eta /pi) / [(omega - E)^2 + eta ^2]
    """
    be          = get_backend(backend)
    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    omega_val   = be.asarray(omega, dtype=be.float64)
    eta_val     = be.asarray(eta, dtype=be.float64)

    # Lorentzian: (eta /pi) / [(omega - E_n)^2 + eta ^2]
    return (eta_val / np.pi) / ((omega_val - eigenvalues)**2 + eta_val**2)

def spectral_function_multi_omega(
        omegas          : Array,
        eigenvalues     : Array,
        eigenvectors    : Optional[Array] = None,
        eta             : float = 0.01,
        diagonal_only   : bool = False,
        backend         : str = "default") -> Array:
    r"""
    Compute spectral function for multiple frequencies.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Array of frequencies.
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n.
    eigenvectors : array-like, shape (N, N), optional
        Eigenvectors. Required if diagonal_only=False.
    eta : float, optional
        Broadening parameter (default: 0.01).
    diagonal_only : bool, optional
        If True, compute only diagonal elements. If False, compute full matrix.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        If diagonal_only=True: shape (n_omega, N), dtype real
        If diagonal_only=False: shape (n_omega, N, N), dtype real
        
    Examples
    --------
    >>> omegas = np.linspace(-5, 5, 100)
    >>> evals, evecs = np.linalg.eigh(H)
    >>> A_diag = spectral_function_multi_omega(omegas, evals, diagonal_only=True)
    >>> # Plot density of states (DOS): sum over all sites
    >>> dos = A_diag.sum(axis=1)
    >>> plt.plot(omegas, dos)
    """
    be = get_backend(backend)
    omegas = be.asarray(omegas)
    eigenvalues = be.asarray(eigenvalues)
    
    n_omega = len(omegas)
    N = len(eigenvalues)
    
    if diagonal_only:
        # Compute only diagonal A_nn(omega)
        A = be.zeros((n_omega, N), dtype=be.float64)
        for i, omega in enumerate(omegas):
            A_i = spectral_function_diagonal(omega, eigenvalues, eta, backend=backend)
            if JAX_AVAILABLE and backend == "jax":
                A = A.at[i, :].set(A_i)
            else:
                A[i, :] = A_i
    else:
        # Compute full matrix A(omega)
        if eigenvectors is None:
            raise ValueError("eigenvectors required when diagonal_only=False")
        
        eigenvectors = be.asarray(eigenvectors, dtype=be.complex128)
        A = be.zeros((n_omega, N, N), dtype=be.float64)
        
        for i, omega in enumerate(omegas):
            G = greens_function_eigenbasis(omega, eigenvalues, eigenvectors, eta, backend)
            A_i = spectral_function(G, backend)
            if JAX_AVAILABLE and backend == "jax":
                A = A.at[i, :, :].set(A_i)
            else:
                A[i, :, :] = A_i
    
    return A

def spectral_function_lanczos_multi_omega(
        omegas      : Array,
        alpha       : Array,
        beta        : Array,
        eta         : float = 0.01,
        backend     : str = "default") -> Array:
    r"""
    Compute spectral function from Lanczos coefficients for multiple frequencies.
    
    Uses tridiagonal Lanczos matrix representation for efficient calculation
    without full diagonalization.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Array of frequencies.
    alpha : array-like, shape (m,)
        Diagonal elements of Lanczos tridiagonal matrix.
    beta : array-like, shape (m-1,)
        Off-diagonal elements of Lanczos tridiagonal matrix.
    eta : float, optional
        Broadening parameter (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Spectral function values, shape (n_omega,), dtype real.
        
    Notes
    -----
    Very efficient for computing spectral function from Lanczos results.
    Avoids full diagonalization and eigenvector computation.
    
    Useful for:
      - Many-body systems with limited spectrum computed
      - Fast spectral scanning without storing full matrices
      - Continuing fractions evaluation
    """
    be = get_backend(backend)
    omegas = be.asarray(omegas)
    
    n_omega = len(omegas)
    A = be.zeros(n_omega, dtype=be.float64)
    
    for i, omega in enumerate(omegas):
        G = greens_function_lanczos(omega, alpha, beta, eta, backend)
        A_i = spectral_function(G, backend)
        if JAX_AVAILABLE and backend == "jax":
            A = A.at[i].set(A_i)
        else:
            A[i] = A_i
    
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
    
    A_O(omega ) = Σ_{m,n} (ρ_m - ρ_n) |<m|O|n>|² delta (omega  - (E_n - E_m))
    
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
    
    χ⁰(omega ) = Σ_{m,n} (f_m - f_n) |V_{mn}|² / (omega  + iη - (E_n - E_m))
    
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
    conductivity_kubo_bubble : Optical conductivity σ(omega ) from bubbles
    
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
    
    # Bubble: χ⁰ = Σ_{mn} (f_m - f_n) |V_{mn}|² / (omega  + iη - (E_n - E_m))
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
    
    σ(omega ) = (1/(2omega )) Σ_{mn} (f_m - f_n) |<m|v|n>|² / (omega  + iη - (E_n - E_m))
    
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
        Conductivity σ(omega ).
        
    Notes
    -----
    The formula is:
        σ(omega ) = (1/(2omega )) χ⁰_{vv}(omega )
    
    Real part Re[σ(omega )] gives absorptive conductivity.
    Imaginary part Im[σ(omega )] gives reactive effects.
    
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
    
    # σ(omega ) = (1/(2omega )) χ⁰_{vv}(omega )
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
    
    Re[χ(omega )] = (2/π) P int_0^∞ domega ' (omega ' Im[χ(omega ')] / (omega '² - omega ²))
    
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
# Exports
# =============================================================================

__all__ = [
    # Green's functions
    'greens_function_eigenbasis',
    'greens_function_lanczos',
    # Spectral functions
    'spectral_function',
    'spectral_function_diagonal',
    'spectral_function_multi_omega',
    'spectral_function_lanczos_multi_omega',
    # k-resolved spectral functions
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
]

# =============================================================================
#! EOF
# =============================================================================