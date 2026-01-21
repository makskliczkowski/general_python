"""
general_python/physics/spectral/dos.py

Density of States (DOS) calculations for quantum systems.

Provides both histogram-based and Gaussian-broadened DOS for
noninteracting (single-particle) systems.

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy.stats import norm

from ...algebra.utils import JAX_AVAILABLE, Array

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Histogram-Based DOS
# =============================================================================

def dos_histogram(
        energies: Array,
        bins: Union[int, Array] = 100,
        range: Optional[Tuple[float, float]] = None,
        density: bool = False
) -> Tuple[Array, Array]:
    """
    Compute density of states using histogram binning.
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n, shape (N,).
    bins : int or array-like, optional
        Number of bins or array of bin edges (default: 100).
    range : tuple of float, optional
        (min, max) range for binning. If None, uses (E.min(), E.max()).
    density : bool, optional
        If True, normalize to probability density (default: False).
        
    Returns
    -------
    counts : Array
        Histogram counts (or density if density=True).
    bin_edges : Array
        Bin edges of length len(counts) + 1.
        
    Examples
    --------
    >>> energies = np.random.randn(1000)
    >>> dos, edges = dos_histogram(energies, bins=50)
    >>> bin_centers = (edges[:-1] + edges[1:]) / 2
    >>> plt.plot(bin_centers, dos)
    """
    energies = np.asarray(energies)
    
    counts, bin_edges = np.histogram(energies, bins=bins, range=range, density=density)
    
    return counts, bin_edges


if JAX_AVAILABLE:
    @partial(jax.jit, static_argnames=["nbins"])
    def dos_histogram_jax(
            energies: Array,
            nbins: int = 100,
            **kwargs
    ) -> Array:
        """
        JAX-optimized DOS via histogram binning.
        
        Parameters
        ----------
        energies : Array
            Eigenenergies.
        nbins : int
            Number of bins.
        **kwargs
            Additional arguments for jnp.histogram.
            
        Returns
        -------
        Array
            Histogram counts.
        """
        counts, _ = jnp.histogram(energies, bins=nbins, **kwargs)
        return counts
else:
    dos_histogram_jax = None


# =============================================================================
# Gaussian-Broadened DOS
# =============================================================================

def dos_gaussian(
        energies: Array,
        sigma: float = 0.01,
        energy_grid: Optional[Array] = None,
        n_points: int = 200
) -> Tuple[Array, Array]:
    """
    Compute density of states with Gaussian broadening.
    
    DOS(E) = \sum _n (1/\sqrt(2\pi\sigma ^2)) exp(-(E - E_n)^2/(2\sigma ^2))
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n, shape (N,).
    sigma : float, optional
        Standard deviation (broadening) of Gaussians (default: 0.01).
    energy_grid : array-like, optional
        Energy points at which to evaluate DOS. If None, creates a grid.
    n_points : int, optional
        Number of points in energy grid if energy_grid is None (default: 200).
        
    Returns
    -------
    dos : Array
        Density of states at each energy point.
    energy_grid : Array
        Energy points where DOS is evaluated.
        
    Examples
    --------
    >>> energies = np.linspace(-5, 5, 100)
    >>> dos, E_grid = dos_gaussian(energies, sigma=0.1)
    >>> plt.plot(E_grid, dos, label='DOS')
    
    Notes
    -----
    Gaussian broadening ensures smooth DOS and is commonly used in spectroscopy.
    Smaller sigma gives sharper features but requires denser sampling.
    """
    energies = np.asarray(energies)
    
    # Create energy grid if not provided
    if energy_grid is None:
        E_min = energies.min()
        E_max = energies.max()
        # Extend range slightly beyond eigenvalue range
        margin = 3 * sigma
        energy_grid = np.linspace(E_min - margin, E_max + margin, n_points)
    else:
        energy_grid = np.asarray(energy_grid)
    
    # Compute DOS by summing Gaussians
    dos = np.zeros_like(energy_grid, dtype=float)
    
    for E_n in energies:
        dos += norm.pdf(energy_grid, loc=E_n, scale=sigma)
    
    return dos, energy_grid


def dos_gaussian_fast(
        energies: Array,
        sigma: float = 0.01,
        energy_grid: Optional[Array] = None,
        n_points: int = 200
) -> Tuple[Array, Array]:
    """
    Fast vectorized Gaussian-broadened DOS.
    
    Uses broadcasting for better performance than dos_gaussian() on large datasets.
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n, shape (N,).
    sigma : float, optional
        Standard deviation (broadening) of Gaussians (default: 0.01).
    energy_grid : array-like, optional
        Energy points at which to evaluate DOS. If None, creates a grid.
    n_points : int, optional
        Number of points in energy grid (default: 200).
        
    Returns
    -------
    dos : Array
        Density of states at each energy point.
    energy_grid : Array
        Energy points where DOS is evaluated.
    """
    energies = np.asarray(energies)
    
    # Create energy grid if not provided
    if energy_grid is None:
        E_min = energies.min()
        E_max = energies.max()
        margin = 3 * sigma
        energy_grid = np.linspace(E_min - margin, E_max + margin, n_points)
    else:
        energy_grid = np.asarray(energy_grid)
    
    # Vectorized computation: broadcast (n_points, 1) vs (1, n_energies)
    prefactor = 1.0 / (sigma * np.sqrt(2 * np.pi))
    diff = energy_grid[:, np.newaxis] - energies[np.newaxis, :]
    dos = prefactor * np.sum(np.exp(-0.5 * (diff / sigma) ** 2), axis=1)
    
    return dos, energy_grid


# =============================================================================
# Integrated DOS
# =============================================================================

def integrated_dos(
        energies: Array,
        energy: float
) -> int:
    """
    Compute integrated density of states (IDOS) up to a given energy.
    
    IDOS(E) = number of eigenvalues ≤ E
    
    Parameters
    ----------
    energies : array-like
        Sorted eigenenergies E_n.
    energy : float
        Energy up to which to count states.
        
    Returns
    -------
    int
        Number of states with E_n ≤ energy.
        
    Examples
    --------
    >>> energies = np.linspace(-10, 10, 1000)
    >>> n_states = integrated_dos(energies, energy=0.0)
    >>> print(f"Number of states below E=0: {n_states}")
    """
    energies = np.asarray(energies)
    return np.sum(energies <= energy)


def idos_curve(
        energies: Array,
        energy_grid: Optional[Array] = None,
        n_points: int = 200
) -> Tuple[Array, Array]:
    """
    Compute integrated DOS curve over an energy range.
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    energy_grid : array-like, optional
        Energy points at which to evaluate IDOS. If None, creates a grid.
    n_points : int, optional
        Number of points in energy grid (default: 200).
        
    Returns
    -------
    idos : Array
        Integrated DOS at each energy point.
    energy_grid : Array
        Energy points where IDOS is evaluated.
        
    Examples
    --------
    >>> energies = np.random.randn(1000)
    >>> idos, E_grid = idos_curve(energies)
    >>> plt.plot(E_grid, idos, label='IDOS')
    """
    energies = np.asarray(energies)
    energies_sorted = np.sort(energies)
    
    if energy_grid is None:
        E_min = energies.min()
        E_max = energies.max()
        margin = 0.1 * (E_max - E_min)
        energy_grid = np.linspace(E_min - margin, E_max + margin, n_points)
    else:
        energy_grid = np.asarray(energy_grid)
    
    idos = np.array([integrated_dos(energies_sorted, E) for E in energy_grid])
    
    return idos, energy_grid


# =============================================================================
# DOS from Spectral Function
# =============================================================================

def dos_from_spectral(
        spectral_function: Array,
        k_points: int
) -> Array:
    """
    Compute DOS from spectral function A(k,\Omega) by integrating over k.
    
    DOS(\Omega) = int dk A(k,\Omega) / \Omega_BZ
    
    Parameters
    ----------
    spectral_function : array-like, shape (n_k, n_omega)
        Spectral function A(k,\Omega) on a k-point grid.
    k_points : int
        Number of k-points (for normalization).
        
    Returns
    -------
    Array
        DOS(\Omega), shape (n_omega,).
        
    Notes
    -----
    Assumes uniform k-point sampling. For non-uniform sampling,
    use appropriate k-point weights.
    """
    spectral_function = np.asarray(spectral_function)
    
    # Sum over k-points and normalize
    dos = np.sum(spectral_function, axis=0) / k_points
    
    return dos


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Histogram-based
    'dos_histogram',
    'dos_histogram_jax',
    
    # Gaussian-broadened
    'dos_gaussian',
    'dos_gaussian_fast',
    
    # Integrated DOS
    'integrated_dos',
    'idos_curve',
    
    # From spectral function
    'dos_from_spectral',
]
