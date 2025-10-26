"""
general_python/physics/thermal.py

Thermal physics utilities for quantum systems.

Provides general-purpose functions for:
- Partition functions and statistical sums
- Thermal averages and expectation values
- Thermodynamic quantities (free energy, entropy, heat capacity)
- Magnetic and charge susceptibilities
- Boltzmann weights and probability distributions

Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
"""

from typing import Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp

try:
    from ..algebra.utils import JAX_AVAILABLE, Array
except ImportError:
    JAX_AVAILABLE   = False
    Array           = np.ndarray

# JAX-specific imports
if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Partition Function and Statistical Sums
# =============================================================================

def partition_function(energies: Array, beta: float) -> float:
    """
    Compute the canonical partition function Z(\beta) = \sum _n exp(-\beta E_n).
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Partition function Z(\beta).
        
    Examples
    --------
    >>> energies = np.array([0.0, 1.0, 2.0, 3.0])
    >>> Z = partition_function(energies, beta=1.0)
    """
    # Shift energies to avoid overflow
    energies    = np.asarray(energies)
    E_min       = np.min(energies)
    return np.sum(np.exp(-beta * (energies - E_min))) * np.exp(-beta * E_min)

def boltzmann_weights(energies: Array, beta: float, normalize: bool = True) -> Array:
    """
    Compute Boltzmann weights rho_n = exp(-\beta E_n) / Z.
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
    normalize : bool, optional
        If True, normalize by partition function Z (default: True).
        
    Returns
    -------
    Array
        Boltzmann weights (probabilities if normalized).
        
    Examples
    --------
    >>> energies = np.array([0.0, 1.0, 2.0])
    >>> rho = boltzmann_weights(energies, beta=1.0)
    >>> print(np.sum(rho))  # Should be 1.0
    """
    # Shift to avoid overflow
    energies    = np.asarray(energies)
    E_min       = np.min(energies)
    weights     = np.exp(-beta * (energies - E_min))
    
    if normalize:
        Z = np.sum(weights)
        if Z > 0:
            weights /= Z    
    return weights

# =============================================================================
# Thermal Averages
# =============================================================================

def thermal_average_diagonal(energies: Array, observable_diagonal: Array, beta: float) -> Tuple[float, float]:
    r"""
    Compute thermal average of an operator diagonal in the energy basis.
    
    <O>_\beta = \Tr[\rho O] / Z = \sum _n O_nn exp(-\beta E_n) / Z
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    observable_diagonal : array-like
        Diagonal matrix elements O_nn of the observable.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    average : float
        Thermal average <O>_\beta.
    partition_func : float
        Partition function Z(\beta).
        
    Examples
    --------
    >>> energies = np.array([0.0, 1.0, 2.0])
    >>> magnetization = np.array([1.0, 0.5, -0.5])
    >>> avg_M, Z = thermal_average_diagonal(energies, magnetization, beta=1.0)
    """
    energies = np.asarray(energies)
    observable_diagonal = np.asarray(observable_diagonal)
    
    if len(energies) != len(observable_diagonal):
        raise ValueError("energies and observable_diagonal must have the same length")
    
    # Shift energies to avoid overflow
    E_min = np.min(energies)
    exp_factors = np.exp(-beta * (energies - E_min))
    
    Z = np.sum(exp_factors)
    avg = np.sum(observable_diagonal * exp_factors)
    
    if Z > 0:
        avg /= Z
    else:
        avg = 0.0
    
    Z_full = Z * np.exp(-beta * E_min)
    
    return avg, Z_full

def thermal_average_general(energies: Array, eigenvectors: Array, observable_matrix: Union[Array, sp.spmatrix], beta: float) -> Tuple[float, float]:
    """
    Compute thermal average of a general operator.
    
    <O>_\beta = \sum _n <n|O|n> exp(-\beta E_n) / Z
    
    where |n> are energy eigenstates.
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    eigenvectors : array-like
        Matrix of eigenvectors (columns are eigenstates).
    observable_matrix : array-like or sparse matrix
        Operator matrix in the original basis.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    average : float
        Thermal average <O>_\beta.
    partition_func : float
        Partition function Z(\beta).
        
    Notes
    -----
    This function transforms the observable to the energy basis and computes
    the diagonal elements <n|O|n>.
    """
    energies        = np.asarray(energies)
    eigenvectors    = np.asarray(eigenvectors)
    
    # Transform observable to energy basis: O_diag = U\dag O U
    if sp.issparse(observable_matrix):
        O_transformed = eigenvectors.conj().T @ (observable_matrix @ eigenvectors)
    else:
        O_transformed = eigenvectors.conj().T @ np.asarray(observable_matrix) @ eigenvectors
    
    # Extract diagonal elements
    O_diagonal = np.real(np.diag(O_transformed))
    
    return thermal_average_diagonal(energies, O_diagonal, beta)

# =============================================================================
# Thermodynamic Quantities
# =============================================================================

def free_energy(energies: Array, beta: float) -> float:
    """
    Compute Helmholtz free energy F = -k_B T ln Z = -(1/\beta) ln Z.
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Free energy F.
        
    Notes
    -----
    We set k_B = 1 (natural units).
    """
    Z = partition_function(energies, beta)
    if Z > 0:
        return -np.log(Z) / beta
    else:
        return np.inf

def internal_energy(energies: Array, beta: float) -> float:
    """
    Compute internal energy U = <H> = \sum _n E_n exp(-\beta E_n) / Z.
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Internal energy U.
    """
    U, _ = thermal_average_diagonal(energies, energies, beta)
    return U

def heat_capacity(energies: Array, beta: float) -> float:
    """
    Compute heat capacity C_V = \beta^2 (<H^2> - <H>^2).
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Heat capacity C_V.
        
    Notes
    -----
    Uses the fluctuation-dissipation relation.
    """
    U, _    = thermal_average_diagonal(energies, energies, beta)
    U2, _   = thermal_average_diagonal(energies, energies**2, beta)
    
    return beta**2 * (U2 - U**2)

def entropy_thermal(energies: Array, beta: float) -> float:
    """
    Compute thermal entropy S = k_B (ln Z + \beta U) = \beta(U - F).
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Thermal entropy S.
        
    Notes
    -----
    We set k_B = 1 (natural units).
    """
    Z = partition_function(energies, beta)
    U = internal_energy(energies, beta)
    
    if Z > 0:
        return np.log(Z) + beta * U
    else:
        return 0.0

# =============================================================================
# Susceptibilities
# =============================================================================

def magnetic_susceptibility(energies: Array, magnetization_diagonal: Array, beta: float) -> float:
    r"""
    Compute magnetic susceptibility χ_M = \beta (<M^2> - <M>^2).
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    magnetization_diagonal : array-like
        Diagonal matrix elements M_nn of magnetization operator.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Magnetic susceptibility χ_M.
        
    Notes
    -----
    This is the linear response of magnetization to applied field.
    """
    M, _    = thermal_average_diagonal(energies, magnetization_diagonal, beta)
    M2, _   = thermal_average_diagonal(energies, magnetization_diagonal**2, beta)
    return beta * (M2 - M**2)

def charge_susceptibility(energies: Array, charge_diagonal: Array, beta: float) -> float:
    """
    Compute charge susceptibility χ_c = \beta (<N^2> - <N>^2).
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    charge_diagonal : array-like
        Diagonal matrix elements N_nn of particle number operator.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Charge susceptibility χ_c.
        
    Notes
    -----
    Related to compressibility via χ_c = \beta <(δN)^2>.
    """
    N, _    = thermal_average_diagonal(energies, charge_diagonal, beta)
    N2, _   = thermal_average_diagonal(energies, charge_diagonal**2, beta)
    return beta * (N2 - N**2)

# =============================================================================
# Specific Heat and Susceptibility from Moments
# =============================================================================

def specific_heat_from_moments(avg_H: float, avg_H2: float, beta: float) -> float:
    """
    Compute specific heat from energy moments: C_V = \beta^2 (<H^2> - <H>^2).
    
    Parameters
    ----------
    avg_H : float
        Average energy <H>.
    avg_H2 : float
        Average energy squared <H^2>.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Specific heat C_V.
    """
    return beta**2 * (avg_H2 - avg_H**2)

def susceptibility_from_moments(avg_O: float, avg_O2: float, beta: float) -> float:
    """
    Generic susceptibility from moments: χ = \beta (<O^2> - <O>^2).
    
    Parameters
    ----------
    avg_O : float
        Average observable <O>.
    avg_O2 : float
        Average observable squared <O^2>.
    beta : float
        Inverse temperature \beta = 1/(k_B T).
        
    Returns
    -------
    float
        Susceptibility χ.
    """
    return beta * (avg_O2 - avg_O**2)

# =============================================================================
# Temperature Scans
# =============================================================================

def thermal_scan(energies: Array, temperatures: Array, observables: Optional[dict] = None) -> dict:
    r"""
    Scan thermal quantities over a range of temperatures.
    
    Parameters
    ----------
    energies : array-like
        Eigenenergies E_n.
    temperatures : array-like
        Array of temperatures T.
    observables : dict, optional
        Dictionary of observable names to diagonal elements.
        Example: {'M_z': magnetization_diagonal, 'N': charge_diagonal}
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'T'       : temperatures
        - 'beta'    : inverse temperatures
        - 'F'       : free energies
        - 'U'       : internal energies
        - 'S'       : entropies
        - 'C_V'     : heat capacities
        - For each observable: average and susceptibility
        
    Examples
    --------
    >>> energies = np.array([0.0, 1.0, 2.0])
    >>> temps = np.linspace(0.1, 10.0, 100)
    >>> observables = {'M': np.array([1.0, 0.0, -1.0])}
    >>> results = thermal_scan(energies, temps, observables)
    >>> plt.plot(results['T'], results['C_V'])
    """
    temperatures = np.asarray(temperatures)
    energies = np.asarray(energies)
    
    n_temps = len(temperatures)
    results = {
        'T'     : temperatures,
        'beta'  : 1.0 / temperatures,
        'F'     : np.zeros(n_temps),
        'U'     : np.zeros(n_temps),
        'S'     : np.zeros(n_temps),
        'C_V'   : np.zeros(n_temps),
    }
    
    # Initialize observable arrays
    if observables is not None:
        for name in observables.keys():
            results[f'{name}_avg'] = np.zeros(n_temps)
            results[f'{name}_chi'] = np.zeros(n_temps)
    
    # Compute for each temperature
    for i, T in enumerate(temperatures):
        beta = 1.0 / T
        
        results['F'][i] = free_energy(energies, beta)
        results['U'][i] = internal_energy(energies, beta)
        results['S'][i] = entropy_thermal(energies, beta)
        results['C_V'][i] = heat_capacity(energies, beta)
        
        if observables is not None:
            for name, obs_diag in observables.items():
                avg, _ = thermal_average_diagonal(energies, obs_diag, beta)
                avg2, _ = thermal_average_diagonal(energies, obs_diag**2, beta)
                
                results[f'{name}_avg'][i] = avg
                results[f'{name}_chi'][i] = beta * (avg2 - avg**2)
    
    return results

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Partition function
    'partition_function',
    'boltzmann_weights',
    
    # Thermal averages
    'thermal_average_diagonal',
    'thermal_average_general',
    
    # Thermodynamic quantities
    'free_energy',
    'internal_energy',
    'heat_capacity',
    'entropy_thermal',
    
    # Susceptibilities
    'magnetic_susceptibility',
    'charge_susceptibility',
    'specific_heat_from_moments',
    'susceptibility_from_moments',
    
    # Scans
    'thermal_scan',
]

# =============================================================================
#! End of file
# =============================================================================