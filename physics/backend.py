"""
general_python/physics/backend.py

Unified physics backend for spectral and thermal calculations.

This module provides a single consolidated interface for:
- Spectral properties (Green's functions, spectral functions, peak analysis)
- Thermal properties (partition functions, free energy, heat capacity, susceptibilities)
- Efficient numerical backend support (NumPy and JAX)

The backend is designed to avoid code duplication by wrapping the specialized
submodules (spectral, thermal) and providing consistent APIs across all features.

-------------------------------------------------------------------------------
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Version     : 1.0
-------------------------------------------------------------------------------
"""

from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

try:
    from ..algebra.utils import JAX_AVAILABLE, get_backend, Array
except ImportError:
    JAX_AVAILABLE   = False
    get_backend     = lambda x="default": np
    Array           = np.ndarray

# Import specialized modules (avoid circular imports by deferring as needed)
# These are imported lazily to prevent initialization issues

__all__ = [
    # Spectral functions (wrapped from spectral_backend.py)
    'greens_function_eigenbasis',
    'greens_function_lanczos',
    'spectral_function',
    'spectral_function_diagonal',
    'spectral_function_multi_omega',
    'spectral_function_k_resolved',
    'integrated_spectral_weight',
    'find_spectral_peaks',
    
    # Thermal functions (wrapped from thermal.py)
    'partition_function',
    'boltzmann_weights',
    'thermal_average_diagonal',
    'thermal_average_general',
    'free_energy',
    'internal_energy',
    'heat_capacity',
    'entropy_thermal',
    'magnetic_susceptibility',
    'charge_susceptibility',
    'specific_heat_from_moments',
    'susceptibility_from_moments',
    'thermal_scan',
    
    # High-level convenience functions
    'get_spectral_backend',
    'get_thermal_backend',
]

# =============================================================================
# Lazy Module Loading (Prevent Circular Imports)
# =============================================================================

_spectral_backend = None
_thermal_backend = None

def _load_spectral_backend():
    """Lazily load spectral backend module."""
    global _spectral_backend
    if _spectral_backend is None:
        try:
            from ..algebra import spectral_backend
            _spectral_backend = spectral_backend
        except ImportError as e:
            raise ImportError(f"Failed to load spectral backend: {e}") from e
    return _spectral_backend

def _load_thermal_backend():
    """Lazily load thermal backend module."""
    global _thermal_backend
    if _thermal_backend is None:
        try:
            from . import thermal
            _thermal_backend = thermal
        except ImportError as e:
            raise ImportError(f"Failed to load thermal backend: {e}") from e
    return _thermal_backend

# =============================================================================
# Spectral Function Wrappers (delegate to spectral_backend.py)
# =============================================================================

def greens_function_eigenbasis(
        omega: float,
        eigenvalues: Array,
        eigenvectors: Optional[Array] = None,
        eta: float = 0.01,
        backend: str = "default") -> Array:
    """Compute Green's function in eigenbasis. See spectral_backend.py for details."""
    spec = _load_spectral_backend()
    return spec.greens_function_eigenbasis(omega, eigenvalues, eigenvectors, eta, backend)

def greens_function_lanczos(
        omega: float,
        alpha: Array,
        beta: Array,
        eta: float = 0.01,
        backend: str = "default") -> complex:
    """Compute Green's function from Lanczos coefficients. See spectral_backend.py for details."""
    spec = _load_spectral_backend()
    return spec.greens_function_lanczos(omega, alpha, beta, eta, backend)

def spectral_function(
        greens_function: Array,
        operator: Optional[Array] = None,
        backend: str = "default") -> Array:
    """Compute spectral function from Green's function. See spectral_backend.py for details."""
    spec = _load_spectral_backend()
    return spec.spectral_function(greens_function, backend)

def spectral_function_diagonal(
        omega: float,
        eigenvalues: Array,
        eta: float = 0.01,
        backend: str = "default") -> Array:
    """Compute diagonal spectral function. See spectral_backend.py for details."""
    spec = _load_spectral_backend()
    return spec.spectral_function_diagonal(omega, eigenvalues, eta, backend)

def spectral_function_multi_omega(
        omegas: Array,
        eigenvalues: Array,
        eigenvectors: Optional[Array] = None,
        eta: float = 0.01,
        diagonal_only: bool = False,
        backend: str = "default") -> Array:
    """Compute spectral function for multiple frequencies. See spectral_backend.py for details."""
    spec = _load_spectral_backend()
    return spec.spectral_function_multi_omega(omegas, eigenvalues, eigenvectors, eta, diagonal_only, backend)

def spectral_function_k_resolved(
        omegas: Array,
        k_points: Array,
        eigenvalues_k: Array,
        eta: float = 0.01,
        backend: str = "default") -> Array:
    """Compute momentum-resolved spectral function. See spectral_backend.py for details."""
    spec = _load_spectral_backend()
    return spec.spectral_function_k_resolved(omegas, k_points, eigenvalues_k, eta, backend)

def integrated_spectral_weight(
        spectral_function: Array,
        omega_grid: Array,
        omega_min: Optional[float] = None,
        omega_max: Optional[float] = None,
        backend: str = "default") -> Union[float, Array]:
    """Compute integrated spectral weight. See spectral_backend.py for details."""
    spec = _load_spectral_backend()
    return spec.integrated_spectral_weight(spectral_function, omega_grid, omega_min, omega_max, backend)

def find_spectral_peaks(
        spectral_function: Array,
        omega_grid: Array,
        threshold: float = 0.1,
        min_distance: int = 5,
        backend: str = "default") -> Array:
    """Find peaks in spectral function. See spectral_backend.py for details."""
    spec = _load_spectral_backend()
    return spec.find_spectral_peaks(spectral_function, omega_grid, threshold, min_distance, backend)

# =============================================================================
# Thermal Function Wrappers (delegate to thermal.py)
# =============================================================================

def partition_function(energies: Array, beta: float) -> float:
    """Compute canonical partition function. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.partition_function(energies, beta)

def boltzmann_weights(energies: Array, beta: float, normalize: bool = True) -> Array:
    """Compute Boltzmann weights. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.boltzmann_weights(energies, beta, normalize)

def thermal_average_diagonal(
        energies: Array,
        observable_diagonal: Array,
        beta: float) -> Tuple[float, float]:
    """Compute thermal average of diagonal operator. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.thermal_average_diagonal(energies, observable_diagonal, beta)

def thermal_average_general(
        energies: Array,
        eigenvectors: Array,
        observable_matrix: Array,
        beta: float) -> Tuple[float, float]:
    """Compute thermal average of general operator. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.thermal_average_general(energies, eigenvectors, observable_matrix, beta)

def free_energy(energies: Array, beta: float) -> float:
    """Compute Helmholtz free energy. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.free_energy(energies, beta)

def internal_energy(energies: Array, beta: float) -> float:
    """Compute internal energy. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.internal_energy(energies, beta)

def heat_capacity(energies: Array, beta: float) -> float:
    """Compute heat capacity. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.heat_capacity(energies, beta)

def entropy_thermal(energies: Array, beta: float) -> float:
    """Compute thermal entropy. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.entropy_thermal(energies, beta)

def magnetic_susceptibility(
        energies: Array,
        magnetization_diagonal: Array,
        beta: float) -> float:
    """Compute magnetic susceptibility. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.magnetic_susceptibility(energies, magnetization_diagonal, beta)

def charge_susceptibility(
        energies: Array,
        charge_diagonal: Array,
        beta: float) -> float:
    """Compute charge susceptibility. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.charge_susceptibility(energies, charge_diagonal, beta)

def specific_heat_from_moments(
        avg_H: float,
        avg_H2: float,
        beta: float) -> float:
    """Compute specific heat from energy moments. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.specific_heat_from_moments(avg_H, avg_H2, beta)

def susceptibility_from_moments(
        avg_O: float,
        avg_O2: float,
        beta: float) -> float:
    """Compute susceptibility from moments. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.susceptibility_from_moments(avg_O, avg_O2, beta)

def thermal_scan(
        energies: Array,
        temperatures: Array,
        observables: Optional[Dict[str, Array]] = None) -> Dict[str, Array]:
    """Scan thermal quantities over temperature range. See thermal.py for details."""
    thermal = _load_thermal_backend()
    return thermal.thermal_scan(energies, temperatures, observables)

# =============================================================================
# Convenience Backends
# =============================================================================

def get_spectral_backend():
    """
    Get the spectral backend module.
    
    Returns
    -------
    module
        The spectral_backend module with all spectral functions.
        
    Examples
    --------
    >>> spec_backend = get_spectral_backend()
    >>> G = spec_backend.greens_function_eigenbasis(omega, evals, evecs)
    """
    return _load_spectral_backend()

def get_thermal_backend():
    """
    Get the thermal backend module.
    
    Returns
    -------
    module
        The thermal module with all thermal functions.
        
    Examples
    --------
    >>> thermal_backend = get_thermal_backend()
    >>> Z = thermal_backend.partition_function(energies, beta)
    """
    return _load_thermal_backend()

# =============================================================================
# End of file
# =============================================================================
