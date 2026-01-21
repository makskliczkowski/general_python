'''
Spectral function utilities for quantum physics.

General-purpose utilities for computing and manipulating spectral functions:
- Broadening of discrete spectra (Lorentzian, Gaussian)
- Data extraction and reshaping
- Normalization and sum rules

These functions are framework-agnostic and can be used with any
method that produces spectral data (ED, DMRG, QMC, etc.).

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-01-15
License             : MIT
----------------------------------------------------------------
'''

from typing import Tuple, Optional, Literal
import numpy as np

# ==============================================================================
# SPECTRAL BROADENING
# ==============================================================================

def compute_spectral_broadening(
        energies        : np.ndarray,
        weights         : np.ndarray,
        omega_grid      : np.ndarray,
        eta             : float = 0.1,
        kind            : Literal['lorentzian', 'gaussian', 'none'] = 'lorentzian'
    ) -> np.ndarray:
    """
    Apply spectral broadening to discrete delta peaks (ED spectra).
    
    Converts discrete excitation spectrum into smooth spectral function:
    A(w) = sum_n w_n * f(w - E_n)
    
    where f is Lorentzian or Gaussian kernel.
    
    Parameters
    ----------
    energies : (N,) array
        Excitation energies (delta peak positions)
    weights : (N,) array
        Spectral weights (delta peak heights)
    omega_grid : (Nw,) array
        Frequency/energy grid for output
    eta : float
        Broadening parameter (FWHM for Lorentzian, std for Gaussian)
    kind : str
        Broadening kernel: 'lorentzian', 'gaussian', or 'none'
        
    Returns
    -------
    spectrum : (Nw,) array
        Broadened spectral function
        
    Notes
    -----
    - Lorentzian: L(w) = (1/Pi) * (η / ((w-E)² + η²))
    - Gaussian: G(w) = (1/√(2Piη²)) * exp(-(w-E)²/(2η²))
    """
    energies    = np.asarray(energies, dtype=float).ravel()
    weights     = np.asarray(weights, dtype=float).ravel()
    omega_grid  = np.asarray(omega_grid, dtype=float).ravel()
    
    if len(energies) == 0 or len(weights) == 0:
        return np.zeros_like(omega_grid)
    
    if len(energies) != len(weights):
        raise ValueError(f"energies and weights must have same length: {len(energies)} vs {len(weights)}")
    
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    
    # Filter out invalid values
    valid_mask  = np.isfinite(energies) & np.isfinite(weights)
    energies    = energies[valid_mask]
    weights     = weights[valid_mask]
    
    if len(energies) == 0:
        return np.zeros_like(omega_grid)
    
    # Vectorized broadening: omega[Nw, None] - energies[None, N] -> (Nw, N)
    omega_diff  = omega_grid[:, None] - energies[None, :]
    
    if kind == 'lorentzian':
        # Lorentzian: L(w-E) = (1/Pi) * η / ((w-E)² + η²)
        kernel      = (eta / np.pi) / (omega_diff**2 + eta**2)
    elif kind == 'gaussian':
        # Gaussian: G(w-E) = (1/√(2Piη²)) * exp(-(w-E)²/(2η²))
        norm        = 1.0 / (np.sqrt(2 * np.pi) * eta)
        kernel      = norm * np.exp(-omega_diff**2 / (2 * eta**2))
    elif kind == 'none':
        # No broadening - just return histogram
        spectrum    = np.zeros_like(omega_grid)
        for E, w in zip(energies, weights):
            idx     = np.argmin(np.abs(omega_grid - E))
            spectrum[idx] += w
        return spectrum
    else:
        raise ValueError(f"Unknown broadening kind: {kind}")
    
    # Sum weighted kernels: A(w) = sum_n w_n * kernel(w - E_n)
    spectrum        = np.sum(weights[None, :] * kernel, axis=1)
    
    return spectrum

# ==============================================================================
# DATA EXTRACTION
# ==============================================================================

def extract_spectral_data(
        result,
        key                 : str,
        state_idx           : Optional[int]         = None,
        component           : Optional[str]         = None,
        reshape_to_komega   : bool                  = True,
        omega_key           : Optional[str]         = '/omega', 
        kvectors_key        : Optional[str]         = '/kvectors',
        kvectors            : Optional[np.ndarray]  = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract spectral function data from result object.
    
    Handles various common storage layouts and reshapes to canonical form:
    - omega: (Nw,)
    - k_vectors: (Nk, D)
    - data: (Nk, Nw)
    
    Parameters
    ----------
    result : Result object
        Result container with spectral data
    key : str
        Data key (e.g., 'akw', 'spectral/skw')
    state_idx : int, optional
        Which state to extract if multiple
    component : str, optional
        Which component (e.g., 'xx', 'zz') for multi-component data
    reshape_to_komega : bool
        If True, ensure output is (Nk, Nw); otherwise keep original shape
    omega_key : str, optional
        Suffix for omega data key (default: '/omega')
    kvectors_key : str, optional
        Key for k-vectors (default: '/kvectors')
    kvectors : array, optional
        Override k-vectors from result
        
    Returns
    -------
    omega : (Nw,) array
    k_vectors : (Nk, D) array
    data : (Nk, Nw) array
    """
    # Try to extract data
    data_raw    = result.get(key, None)
    if data_raw is None:
        raise ValueError(f"Key '{key}' not found in result")
    
    # Extract omega grid
    omega       = result.get(key + omega_key, None)
    if omega is None:
        omega   = result.get('omega', None)
    if omega is None:
        # Fallback: create uniform grid
        omega   = np.arange(data_raw.shape[-1])
    
    # Extract k-vectors (if available)
    k_vectors   = kvectors if kvectors is not None else result.get(kvectors_key, None)
    if k_vectors is None:
        k_vectors   = np.zeros((data_raw.shape[0], 3))
    
    # Convert data to array
    data        = np.asarray(data_raw, dtype=complex)
    
    # Handle state selection
    if state_idx is not None and data.ndim >= 3:
        data    = data[..., state_idx]
    
    # Reshape to (Nk, Nw) if requested
    if reshape_to_komega:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            # Flatten extra dimensions
            data = data.reshape(-1, data.shape[-1])
    
    # Take absolute value if complex
    if np.iscomplexobj(data):
        data = np.abs(data)
    
    omega       = np.asarray(omega, float)
    k_vectors   = np.asarray(k_vectors, float)
    
    return omega, k_vectors, data

# ==============================================================================
#! EOF
# ==============================================================================