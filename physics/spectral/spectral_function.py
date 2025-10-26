"""
general_python/physics/spectral/spectral_function.py

Spectral functions A(k,\Omega) for noninteracting quantum systems.

The spectral function is related to the imaginary part of the Green's function:
    A(k,\Omega) = -(1/π) Im[G(k,\Omega)]

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from typing import Optional, Union
import numpy as np

from ...algebra.utils import JAX_AVAILABLE, Array
from . import greens

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Spectral Function from Green's Function
# =============================================================================

def spectral_function(greens_function: Array) -> Array:
    """
    Compute spectral function from Green's function.
    
    A(\Omega) = -(1/π) Im[G(\Omega)]
    
    Parameters
    ----------
    greens_function : array-like, complex
        Green's function G(\Omega), any shape.
        
    Returns
    -------
    Array, real
        Spectral function A(\Omega), same shape as input.
        
    Examples
    --------
    >>> omega = 1.0
    >>> G = greens.greens_function_eigenbasis(omega, eigenvalues, eigenvectors, eta=0.01)
    >>> A = spectral_function(G)
    
    Notes
    -----
    The spectral function satisfies sum rule: ∫ d\Omega A(\Omega) = N (number of states).
    """
    return -np.imag(greens_function) / np.pi

def spectral_function_diagonal(omega: float, eigenvalues: Array, eta: float = 0.01) -> Array:
    """
    Compute diagonal spectral function directly from eigenvalues.
    
    A_nn(\Omega) = -(1/π) Im[1/(\Omega + iη - E_n)]
            = (η/π) / [(\Omega - E_n)^2 + η^2]
    
    Parameters
    ----------
    omega : float
        Frequency \Omega.
    eigenvalues : array-like
        Eigenvalues E_n.
    eta : float, optional
        Broadening parameter (default: 0.01).
        
    Returns
    -------
    Array
        Diagonal spectral function A_nn(\Omega).
        
    Notes
    -----
    This is a Lorentzian (Cauchy) distribution centered at each E_n.
    """
    eigenvalues = np.asarray(eigenvalues)
    
    # Lorentzian: (η/π) / [(\Omega - E_n)^2 + η^2]
    return (eta / np.pi) / ((omega - eigenvalues)**2 + eta**2)

def spectral_function_multi_omega(omegas: Array, eigenvalues: Array, eigenvectors: Optional[Array] = None,
        eta: float = 0.01, diagonal_only: bool = False) -> Array:
    """
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
        If True, compute only diagonal elements (default: False).
        
    Returns
    -------
    Array
        If diagonal_only=True: shape (n_omega, N)
        If diagonal_only=False: shape (n_omega, N, N)
        
    Examples
    --------
    >>> omegas = np.linspace(-5, 5, 100)
    >>> A_diag = spectral_function_multi_omega(omegas, eigenvalues, diagonal_only=True)
    >>> # Plot diagonal elements: LDOS
    >>> plt.plot(omegas, A_diag[:, site_index])
    """
    omegas = np.asarray(omegas)
    eigenvalues = np.asarray(eigenvalues)
    
    n_omega = len(omegas)
    N = len(eigenvalues)
    
    if diagonal_only:
        # Compute only diagonal A_nn(\Omega)
        A = np.zeros((n_omega, N), dtype=float)
        for i, omega in enumerate(omegas):
            A[i] = spectral_function_diagonal(omega, eigenvalues, eta)
    else:
        # Compute full matrix A(\Omega)
        if eigenvectors is None:
            raise ValueError("eigenvectors required when diagonal_only=False")
        
        eigenvectors = np.asarray(eigenvectors, dtype=complex)
        A = np.zeros((n_omega, N, N), dtype=float)
        
        for i, omega in enumerate(omegas):
            G = greens.greens_function_eigenbasis(omega, eigenvalues, eigenvectors, eta)
            A[i] = spectral_function(G)
    
    return A

# =============================================================================
# Momentum-Resolved Spectral Function
# =============================================================================

def spectral_function_k_resolved(
        omegas: Array,
        k_points: Array,
        eigenvalues_k: Array,
        eta: float = 0.01
) -> Array:
    """
    Compute momentum-resolved spectral function A(k,\Omega).
    
    For systems with momentum as a good quantum number, compute:
        A(k,\Omega) = (η/π) / [(\Omega - E(k))^2 + η^2]
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Frequencies.
    k_points : array-like, shape (n_k, d)
        k-points in Brillouin zone.
    eigenvalues_k : array-like, shape (n_k,) or (n_k, n_bands)
        Energy dispersion E(k). If 2D, includes multiple bands.
    eta : float, optional
        Broadening parameter (default: 0.01).
        
    Returns
    -------
    Array
        Shape (n_k, n_omega) if eigenvalues_k is 1D,
        Shape (n_k, n_bands, n_omega) if eigenvalues_k is 2D.
        
    Examples
    --------
    >>> # 1D tight-binding dispersion
    >>> k_points = np.linspace(-np.pi, np.pi, 100)
    >>> E_k = -2 * np.cos(k_points)  # dispersion
    >>> omegas = np.linspace(-3, 3, 200)
    >>> A_k_omega = spectral_function_k_resolved(omegas, k_points, E_k)
    >>> plt.imshow(A_k_omega, aspect='auto', extent=[omegas[0], omegas[-1], k_points[0], k_points[-1]])
    """
    omegas = np.asarray(omegas)
    eigenvalues_k = np.asarray(eigenvalues_k)
    
    n_omega = len(omegas)
    
    if eigenvalues_k.ndim == 1:
        # Single band: shape (n_k,)
        n_k = len(eigenvalues_k)
        A = np.zeros((n_k, n_omega), dtype=float)
        
        for k_idx in range(n_k):
            E_k = eigenvalues_k[k_idx]
            A[k_idx, :] = (eta / np.pi) / ((omegas - E_k)**2 + eta**2)
    
    elif eigenvalues_k.ndim == 2:
        # Multiple bands: shape (n_k, n_bands)
        n_k, n_bands = eigenvalues_k.shape
        A = np.zeros((n_k, n_bands, n_omega), dtype=float)
        
        for k_idx in range(n_k):
            for band in range(n_bands):
                E_k = eigenvalues_k[k_idx, band]
                A[k_idx, band, :] = (eta / np.pi) / ((omegas - E_k)**2 + eta**2)
    else:
        raise ValueError("eigenvalues_k must be 1D or 2D")
    
    return A


# =============================================================================
# Integrated Spectral Function
# =============================================================================

def integrated_spectral_weight(
        spectral_function: Array,
        omega_grid: Array,
        omega_min: Optional[float] = None,
        omega_max: Optional[float] = None
) -> Union[float, Array]:
    """
    Compute integrated spectral weight over an energy window.
    
    W = ∫_{\Omega_min}^{\Omega_max} d\Omega A(\Omega)
    
    Parameters
    ----------
    spectral_function : array-like
        Spectral function A(\Omega), shape (..., n_omega).
    omega_grid : array-like, shape (n_omega,)
        Frequency grid (must be uniformly spaced).
    omega_min, omega_max : float, optional
        Integration limits. If None, integrates over full range.
        
    Returns
    -------
    float or Array
        Integrated spectral weight. Returns scalar if input is 1D,
        otherwise returns array integrated along last axis.
        
    Notes
    -----
    Uses trapezoidal rule for integration.
    """
    spectral_function = np.asarray(spectral_function)
    omega_grid = np.asarray(omega_grid)
    
    # Determine integration window
    if omega_min is None:
        omega_min = omega_grid[0]
    if omega_max is None:
        omega_max = omega_grid[-1]
    
    # Find indices for integration window
    mask = (omega_grid >= omega_min) & (omega_grid <= omega_max)
    omega_window = omega_grid[mask]
    A_window = spectral_function[..., mask]
    
    # Integrate using trapezoidal rule
    return np.trapz(A_window, omega_window, axis=-1)


# =============================================================================
# Peak Finding and Analysis
# =============================================================================

def find_spectral_peaks(
        spectral_function: Array,
        omega_grid: Array,
        threshold: float = 0.1,
        min_distance: int = 5
) -> Array:
    """
    Find peaks in spectral function.
    
    Parameters
    ----------
    spectral_function : array-like, shape (n_omega,)
        1D spectral function.
    omega_grid : array-like, shape (n_omega,)
        Frequency grid.
    threshold : float, optional
        Minimum relative height for peak detection (default: 0.1).
    min_distance : int, optional
        Minimum distance between peaks in grid points (default: 5).
        
    Returns
    -------
    Array
        Frequencies of detected peaks.
        
    Notes
    -----
    Uses simple local maximum detection. For more sophisticated peak finding,
    consider scipy.signal.find_peaks.
    """
    from scipy.signal import find_peaks
    
    spectral_function = np.asarray(spectral_function)
    omega_grid = np.asarray(omega_grid)
    
    # Normalize for threshold
    A_max = np.max(spectral_function)
    height_threshold = threshold * A_max
    
    # Find peaks
    peak_indices, _ = find_peaks(
        spectral_function,
        height=height_threshold,
        distance=min_distance
    )
    
    return omega_grid[peak_indices]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Basic spectral functions
    'spectral_function',
    'spectral_function_diagonal',
    'spectral_function_multi_omega',
    
    # Momentum-resolved
    'spectral_function_k_resolved',
    
    # Analysis
    'integrated_spectral_weight',
    'find_spectral_peaks',
]

# ============================================================================
#! End of file
# ============================================================================