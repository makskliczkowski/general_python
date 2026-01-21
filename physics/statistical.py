"""
general_python/physics/statistical.py

Statistical analysis utilities for quantum systems.

This module provides tools for:
- Finite window averages and time series analysis
- Local density of states (LDOS) and strength functions
- Spectral histograms and binning
- Windowed matrix element calculations
- Generic histogram and scatter analysis

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
import numba
import math

from ..algebra.utils import JAX_AVAILABLE, Array

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Window and Averaging Utilities
# =============================================================================

def moving_average(data: Array, window_size: int, mode: str = 'valid') -> Array:
    """
    Compute moving average with specified window size.
    
    Parameters
    ----------
    data : array-like
        Input data array.
    window_size : int
        Size of the moving window.
    mode : str, optional
        Convolution mode: 'valid' (default), 'same', or 'full'.
        
    Returns
    -------
    Array
        Moving average array.
        
    Examples
    --------
    >>> data = np.random.randn(100)
    >>> smooth = moving_average(data, window_size=5)
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode=mode)

def windowed_variance(data: Array, window_size: int, ddof: int = 1) -> Tuple[Array, Array]:
    """
    Compute moving mean and variance over a sliding window.
    
    Parameters
    ----------
    data : array-like
        Input data array.
    window_size : int
        Size of the moving window.
    ddof : int, optional
        Delta degrees of freedom for variance calculation (default: 1).
        
    Returns
    -------
    means : Array
        Moving mean.
    variances : Array
        Moving variance.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2 for variance")
    
    n = len(data)
    if n < window_size:
        raise ValueError("Data length must be >= window_size")
    
    means       = np.empty(n - window_size + 1)
    variances   = np.empty(n - window_size + 1)
    
    for i in range(n - window_size + 1):
        window          = data[i:i + window_size]
        means[i]        = np.mean(window)
        variances[i]    = np.var(window, ddof=ddof)
    
    return means, variances

# =============================================================================
# Exponential Moving Average
# =============================================================================

def exponential_moving_average(data: Array, alpha: float) -> Array:
    """
    Compute exponential moving average.
    
    EMA[i] = alpha * data[i] + (1 - alpha) * EMA[i-1]
    
    Parameters
    ----------
    data : array-like
        Input data array.
    alpha : float
        Smoothing factor in (0, 1]. Higher values give more weight to recent data.
        
    Returns
    -------
    Array
        Exponential moving average.
        
    Examples
    --------
    >>> data    = np.random.randn(100)
    >>> ema     = exponential_moving_average(data, alpha=0.3)
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    
    ema     = np.empty_like(data)
    ema[0]  = data[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    return ema

# =============================================================================
# Centered Window Extraction
# =============================================================================

def centered_window(data: Array, center_idx: int, window_size: int) -> Array:
    """
    Extract a centered window from data around a given index.
    
    Parameters
    ----------
    data : array-like
        Input data array.
    center_idx : int
        Center index of the window.
    window_size : int
        Total size of the window (must be odd for symmetric centering).
        
    Returns
    -------
    Array
        Windowed data array.
    """
    n           = len(data)
    half_window = window_size // 2
    start       = max(0, center_idx - half_window)
    end         = min(n, center_idx + half_window + 1)
    
    return data[start:end]

def window_mask(values: Array, center: float, width: float) -> Array:
    """
    Create a boolean mask for values within [center - width/2, center + width/2].
    
    Parameters
    ----------
    values : array-like
        Array of values.
    center : float
        Center of the window.
    width : float
        Width of the window.
        
    Returns
    -------
    Array (bool)
        Boolean mask where True indicates values within the window.
    """
    return np.abs(values - center) <= width / 2

def fractional_window(data: Array, fraction: float = 0.3, around: Optional[float] = None) -> Array:
    """
    Extract a fractional window of data centered around a value.
    
    Parameters
    ----------
    data : array-like
        Sorted input data.
    fraction : float
        Fraction of data to extract (0 < fraction <= 1).
    around : float, optional
        Value to center the window around. If None, uses median.
        
    Returns
    -------
    Array
        Windowed subset of data.
        
    Examples
    --------
    >>> energies = np.linspace(-10, 10, 1000)
    >>> window = fractional_window(energies, fraction=0.2, around=0.0)
    >>> # Returns ~200 energies centered near 0
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    
    data = np.asarray(data)
    if around is None:
        around = np.median(data)
    
    # Find index closest to 'around'
    center_idx = np.argmin(np.abs(data - around))
    
    # Calculate window size
    n = len(data)
    window_size = max(1, int(fraction * n))
    half_window = window_size // 2
    
    start = max(0, center_idx - half_window)
    end = min(n, center_idx + half_window)
    
    # Adjust if we're at boundaries
    if end - start < window_size and start == 0:
        end = min(n, window_size)
    elif end - start < window_size and end == n:
        start = max(0, n - window_size)
    
    return data[start:end]

# =============================================================================
# Energy Window Utilities for Spectral Analysis
# =============================================================================

@numba.njit(fastmath=True, cache=True)
def extract_indices_window(
        start           : int,
        stop            : int,
        eigvals         : np.ndarray,
        energy_target   : float = 0.0,
        bandwidth       : float = 1.0,
        energy_diff_cut : float = 0.015,
        whole_spectrum  : bool = False) -> Tuple[np.ndarray, int]:
    """
    Extract indices of eigenvalue pairs (i, j) where |(E_i + E_j)/2 - E_target| < tolerance.
    
    Optimized for computing matrix elements within energy windows, e.g., for structure
    factors, transition amplitudes, or response functions.
    
    Parameters
    ----------
    start, stop : int
        Index range to consider in eigvals.
    eigvals : ndarray
        Sorted eigenvalues (ascending or descending).
    energy_target : float
        Target energy for the window center.
    bandwidth : float
        Bandwidth scale factor.
    energy_diff_cut : float
        Relative tolerance: actual tolerance = bandwidth * energy_diff_cut.
    whole_spectrum : bool
        If True, skip windowing and return empty indices.
        
    Returns
    -------
    indices_alloc : ndarray of shape (N, 3)
        Each row: (i, j_start, j_end) where j in [j_start, j_end) satisfies the window.
    count : int
        Number of valid index triplets.
        
    Notes
    -----
    Assumes eigvals is sorted. The function efficiently finds pairs within the energy window
    by exploiting sorted order, avoiding O(N^2) naive search.
    """
    if whole_spectrum:
        return np.empty((0, 3), dtype=np.int64), 0
    
    if stop < start:
        start, stop = stop, start
    if stop > eigvals.shape[0]:
        stop = eigvals.shape[0]
    if start < 0:
        start = 0
    
    indices_alloc = np.zeros((stop - start, 3), dtype=np.int64)
    tol = bandwidth * energy_diff_cut
    
    # Pointers for j_lo and j_hi
    j_lo = stop - 1
    j_hi = stop - 1
    
    cnt = 0
    for i in range(start, stop):
        e_i = eigvals[i]
        # Window condition: |(E_i + E_j)/2 - E_target| < tol
        # => 2*E_target - tol < E_i + E_j < 2*E_target + tol
        low = 2.0 * (energy_target - tol) - e_i
        high = 2.0 * (energy_target + tol) - e_i
        
        # Advance j_hi to first eigvals[j] > high
        while j_hi >= 0 and eigvals[j_hi] >= high:
            j_hi -= 1
        
        # Advance j_lo to first eigvals[j] >= low
        j_lo = j_hi
        while j_lo > i and eigvals[j_lo] > low:
            j_lo -= 1
        
        if j_hi <= j_lo:
            break  # No more valid pairs
        
        indices_alloc[cnt, 0] = i
        indices_alloc[cnt, 1] = j_lo
        indices_alloc[cnt, 2] = j_hi + 1  # Exclusive end
        cnt                  += 1
    
    return indices_alloc[:cnt], cnt

# =============================================================================
# Local Density of States (LDOS)
# =============================================================================

if JAX_AVAILABLE:
    @partial(jax.jit, static_argnames=["degenerate", "tol"])
    def ldos_jax(
            energies: Array,
            overlaps: Array,
            degenerate: bool = False,
            tol: float = 1e-8
    ) -> Array:
        """
        JAX-optimized Local Density of States (LDOS).
        
        Parameters
        ----------
        energies : Array
            Eigenenergies.
        overlaps : Array
            Overlap amplitudes <n|\psi >.
        degenerate : bool
            If True, sum over nearly degenerate levels.
        tol : float
            Tolerance for degeneracy grouping.
            
        Returns
        -------
        Array
            LDOS for each energy index.
        """
        if not degenerate:
            return jnp.abs(overlaps) ** 2
        
        def _ldos_i(E_i):
            mask = jnp.abs(energies - E_i) < tol
            return jnp.sum(jnp.abs(overlaps) ** 2 * mask)
        
        return jax.vmap(_ldos_i)(energies)
else:
    ldos_jax = None


def ldos(
        energies: Array,
        overlaps: Array,
        degenerate: bool = False,
        tol: float = 1e-8
) -> Array:
    """
    Local Density of States (LDOS) or strength function.
    
    LDOS_i = |<i|\psi >|^2 (non-degenerate)
    LDOS_i = \sum _{j:|E_j - E_i|<tol} |<j|\psi >|^2 (degenerate)
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n, shape (N,).
    overlaps : array-like
        Overlap amplitudes <n|\psi >, shape (N,).
    degenerate : bool, optional
        Whether to sum over nearly degenerate levels (default: False).
    tol : float, optional
        Tolerance for degeneracy grouping (default: 1e-8).
        
    Returns
    -------
    Array
        LDOS for each energy index.
        
    Notes
    -----
    Use JAX version (ldos_jax) for better performance on GPU/TPU.
    """
    if JAX_AVAILABLE:
        return ldos_jax(energies, overlaps, degenerate, tol)
    
    if not degenerate:
        return np.abs(overlaps) ** 2
    
    N = energies.size
    ldos_arr = np.empty(N, dtype=float)
    for i in range(N):
        mask = np.abs(energies - energies[i]) < tol
        ldos_arr[i] = np.sum(np.abs(overlaps[mask]) ** 2)
    
    return ldos_arr


# =============================================================================
# Binning and Histogram Utilities
# =============================================================================

@numba.njit(cache=True, fastmath=True, inline='always')
def _bin_index(
        omega: float,
        bins: np.ndarray,
        bin0: float,
        inv_binw: float,
        uniform_bins: bool = False,
        uniform_log_bins: bool = False
) -> int:
    """
    Find bin index for a value omega.
    
    Supports uniform, uniform-log, and non-uniform bins.
    Returns 0 for underflow, len(bins)-1 for overflow.
    """
    nBins = bins.shape[0] - 1
    
    if uniform_bins:
        idx = int((omega - bin0) * inv_binw) + 1
        if omega < bins[0]:
            return 0
        elif omega >= bins[-1]:
            return nBins
        return idx
    
    elif uniform_log_bins:
        if omega <= 0.0:
            return 0
        t = math.log(omega) - bin0
        b = int(t * inv_binw) + 1
        if omega < bins[0]:
            return 0
        elif omega >= bins[-1]:
            return nBins
        return b
    
    # Non-uniform: binary search
    if omega < bins[0]:
        return 0
    elif omega >= bins[-1]:
        return nBins
    
    return np.searchsorted(bins, omega, side='right')


@numba.njit(fastmath=True, cache=True)
def _alloc_bin_info(
        uniform_bins: bool,
        uniform_log_bins: bool,
        bins: Optional[np.ndarray]
) -> Tuple[float, float, Tuple[bool, bool]]:
    """
    Pre-compute bin information for fast histogramming.
    
    Returns
    -------
    bin0 : float
        First bin edge (or log of first bin edge).
    inv_binw : float
        Inverse bin width for uniform bins.
    (is_uniform, is_log) : tuple of bool
        Flags indicating bin type.
    """
    if (not uniform_bins and not uniform_log_bins) or (bins is None) or (bins.shape[0] < 2):
        return 0.0, 0.0, (False, False)
    
    if uniform_bins:
        bin0 = bins[0]
        binw = bins[1] - bins[0]
        inv_binw = 1.0 / binw if binw > 0.0 else 0.0
        return bin0, inv_binw, (True, False)
    
    elif uniform_log_bins:
        log_bin0 = math.log(bins[0]) if bins[0] > 0.0 else -np.inf
        log_binw = math.log(bins[1]) - log_bin0
        bin0 = log_bin0
        inv_binw = 1.0 / log_binw if log_binw > 0.0 else 0.0
        return bin0, inv_binw, (False, True)
    
    else:
        return 0.0, 0.0, (False, False)


def create_bins(n_bins: int, range_min: float, range_max: float, log_scale: bool = False) -> np.ndarray:
    """
    Create bin edges for histogramming.
    
    Parameters
    ----------
    n_bins : int
        Number of bins.
    range_min, range_max : float
        Range of the bins.
    log_scale : bool, optional
        If True, create logarithmically spaced bins (default: False).
        
    Returns
    -------
    ndarray
        Bin edges of length n_bins + 1.
        
    Examples
    --------
    >>> bins = create_bins(50, 0.0, 10.0)
    >>> log_bins = create_bins(50, 1e-3, 10.0, log_scale=True)
    """
    if log_scale:
        if range_min <= 0:
            raise ValueError("For log scale, range_min must be > 0")
        return np.logspace(np.log10(range_min), np.log10(range_max), n_bins + 1)
    else:
        return np.linspace(range_min, range_max, n_bins + 1)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Window and averaging
    'moving_average',
    'windowed_variance',
    'exponential_moving_average',
    
    # Centered window
    'window_mask',
    'centered_window',
    'fractional_window',
    
    # Energy window utilities
    'extract_indices_window',
    
    # LDOS
    'ldos',
    'ldos_jax',
    
    # Binning
    'create_bins',
]

# =============================================================================
#! End of file
# =============================================================================