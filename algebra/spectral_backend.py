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
        backend             : str = "default"
) -> Union[float, Array]:
    r"""
    Compute integrated spectral weight over an energy window.
    
    W = ∫_{omega_min}^{omega_max} domega A(omega)
    
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
]

# =============================================================================
#! EOF
# =============================================================================