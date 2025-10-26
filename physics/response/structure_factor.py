"""
Fast dynamic structure factor S(q,\Omega) for spin systems.

The dynamic structure factor measures spin-spin correlations:
    S(q,\Omega) = \sum _f |<f|S^z_q|i>|^2 δ(\Omega - (E_f - E_i))

where S^z_q = \sum _j S^z_j exp(iq·r_j) is the Fourier-transformed spin operator.

File    : QES/general_python/physics/response/structure_factor.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
"""

from typing import Optional, Union, Tuple
import numpy as np
import numba

from ...algebra.utils import JAX_AVAILABLE, Array

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Dynamic Structure Factor S(q,\Omega) - Optimized Implementation
# =============================================================================

@numba.njit(parallel=True, fastmath=True, cache=True)
def _structure_factor_kernel(energies: np.ndarray, matrix_elements: np.ndarray,
        omega_bins: np.ndarray, eta: float) -> np.ndarray:
    """
    Fast Numba kernel for computing S(q,\Omega) with Lorentzian broadening.
    
    Parameters
    ----------
    energies : ndarray, shape (N,)
        Energy differences \Omega_fi = E_f - E_i.
    matrix_elements : ndarray, shape (N,)
        Matrix elements |<f|S_q|i>|^2.
    omega_bins : ndarray, shape (n_omega,)
        Frequency grid.
    eta : float
        Broadening parameter.
        
    Returns
    -------
    ndarray, shape (n_omega,)
        S(q,\Omega) on the frequency grid.
    """
    n_omega = omega_bins.shape[0]
    n_trans = energies.shape[0]
    
    S_q_omega = np.zeros(n_omega, dtype=np.float64)
    
    # Lorentzian broadening: (η/π) / [(\Omega - \Omega_fi)^2 + η^2]
    prefactor = eta / np.pi
    
    for i in numba.prange(n_omega):
        omega = omega_bins[i]
        for j in range(n_trans):
            delta_E = energies[j]
            diff = omega - delta_E
            lorentzian = prefactor / (diff * diff + eta * eta)
            S_q_omega[i] += matrix_elements[j] * lorentzian
    
    return S_q_omega


def structure_factor_spin(
        initial_state: Array,
        hamiltonian_eigvals: Array,
        hamiltonian_eigvecs: Array,
        spin_q_operator: Array,
        omega_grid: Array,
        eta: float = 0.01,
        temperature: Optional[float] = None
) -> Array:
    """
    Compute dynamic structure factor S(q,\Omega) for a spin system.
    
    S(q,\Omega) = \sum _f |<f|S_q|i>|^2 δ(\Omega - (E_f - E_i))
    
    At finite temperature:
    S(q,\Omega) = \sum _{i,f} ρ_i |<f|S_q|i>|^2 δ(\Omega - (E_f - E_i))
    
    where ρ_i = exp(-\betaE_i)/Z are Boltzmann weights.
    
    Parameters
    ----------
    initial_state : array-like, shape (N,)
        Initial state |i> (for T=0) or None (for thermal average).
    hamiltonian_eigvals : array-like, shape (N,)
        Hamiltonian eigenvalues E_n.
    hamiltonian_eigvecs : array-like, shape (N, N)
        Hamiltonian eigenvectors (columns are eigenstates).
    spin_q_operator : array-like, shape (N, N)
        Spin operator S_q = \sum _j S_j exp(iq·r_j) in real space basis.
    omega_grid : array-like, shape (n_omega,)
        Frequency grid for S(q,\Omega).
    eta : float, optional
        Lorentzian broadening parameter (default: 0.01).
    temperature : float, optional
        Temperature for thermal averaging. If None, uses T=0 (ground state).
        
    Returns
    -------
    Array, shape (n_omega,)
        Dynamic structure factor S(q,\Omega).
        
    Notes
    -----
    Optimized with Numba for performance. For large systems, consider
    using only relevant energy windows to reduce computation time.
    
    Examples
    --------
    >>> # Compute S(q,\Omega) for 1D spin chain at T=0
    >>> initial_state = hamiltonian_eigvecs[:, 0]  # Ground state
    >>> omega_grid = np.linspace(0, 5, 500)
    >>> S_q_omega = structure_factor_spin(
    ...     initial_state, eigvals, eigvecs, S_q, omega_grid, eta=0.05
    ... )
    >>> plt.plot(omega_grid, S_q_omega)
    """
    eigvals = np.asarray(hamiltonian_eigvals)
    eigvecs = np.asarray(hamiltonian_eigvecs, dtype=complex)
    S_q = np.asarray(spin_q_operator, dtype=complex)
    omega_grid = np.asarray(omega_grid)
    
    N = len(eigvals)
    
    # Transform S_q to energy eigenbasis
    S_q_eigenbasis = eigvecs.conj().T @ S_q @ eigvecs
    
    if temperature is None or temperature == 0:
        # T=0: single initial state
        if initial_state is None:
            # Use ground state
            initial_state = eigvecs[:, 0]
        
        initial_state = np.asarray(initial_state, dtype=complex)
        
        # Project initial state onto eigenbasis
        c_i = eigvecs.conj().T @ initial_state
        i_idx = np.argmax(np.abs(c_i))  # Dominant eigenstate
        E_i = eigvals[i_idx]
        
        # Compute matrix elements |<f|S_q|i>|^2
        matrix_elements = np.abs(S_q_eigenbasis[:, i_idx])**2
        
        # Energy differences
        energy_diffs = eigvals - E_i
        
        # Use fast kernel
        S_q_omega = _structure_factor_kernel(
            energy_diffs, matrix_elements, omega_grid, eta
        )
    
    else:
        # Finite temperature: thermal average
        beta = 1.0 / temperature
        
        # Boltzmann weights
        E_min = np.min(eigvals)
        weights = np.exp(-beta * (eigvals - E_min))
        Z = np.sum(weights)
        weights /= Z
        
        # Initialize S(q,\Omega)
        S_q_omega = np.zeros_like(omega_grid)
        
        # Sum over initial states weighted by Boltzmann factors
        for i_idx in range(N):
            if weights[i_idx] < 1e-10:
                continue  # Skip negligible weights
            
            E_i = eigvals[i_idx]
            rho_i = weights[i_idx]
            
            # Matrix elements |<f|S_q|i>|^2
            matrix_elements = np.abs(S_q_eigenbasis[:, i_idx])**2
            
            # Energy differences
            energy_diffs = eigvals - E_i
            
            # Add contribution from this initial state
            S_q_omega += rho_i * _structure_factor_kernel(
                energy_diffs, matrix_elements, omega_grid, eta
            )
    
    return S_q_omega


# =============================================================================
# Multi-q Structure Factor
# =============================================================================

def structure_factor_multi_q(
        initial_state: Array,
        hamiltonian_eigvals: Array,
        hamiltonian_eigvecs: Array,
        spin_operators_q: list,
        q_values: Array,
        omega_grid: Array,
        eta: float = 0.01,
        temperature: Optional[float] = None
) -> Array:
    """
    Compute S(q,\Omega) for multiple q-vectors.
    
    Parameters
    ----------
    initial_state : array-like or None
        Initial state (T=0) or None (thermal).
    hamiltonian_eigvals : array-like
        Eigenvalues.
    hamiltonian_eigvecs : array-like
        Eigenvectors.
    spin_operators_q : list of arrays
        List of S_q operators for each q-vector.
    q_values : array-like, shape (n_q, d)
        Array of q-vectors.
    omega_grid : array-like
        Frequency grid.
    eta : float, optional
        Broadening (default: 0.01).
    temperature : float, optional
        Temperature (default: None for T=0).
        
    Returns
    -------
    Array, shape (n_q, n_omega)
        S(q,\Omega) for each q-vector.
        
    Examples
    --------
    >>> # Compute S(q,\Omega) along a high-symmetry path
    >>> q_path = np.linspace(0, np.pi, 50)
    >>> S_q_list = [create_spin_q_operator(q, lattice) for q in q_path]
    >>> S_q_omega = structure_factor_multi_q(
    ...     gs, eigvals, eigvecs, S_q_list, q_path, omega_grid
    ... )
    >>> plt.imshow(S_q_omega, aspect='auto', extent=[omega_grid[0], omega_grid[-1], q_path[0], q_path[-1]])
    """
    n_q = len(spin_operators_q)
    n_omega = len(omega_grid)
    
    S_q_omega = np.zeros((n_q, n_omega), dtype=float)
    
    for i, S_q in enumerate(spin_operators_q):
        S_q_omega[i] = structure_factor_spin(
            initial_state,
            hamiltonian_eigvals,
            hamiltonian_eigvecs,
            S_q,
            omega_grid,
            eta=eta,
            temperature=temperature
        )
    
    return S_q_omega


# =============================================================================
# Sum Rules and Moments
# =============================================================================

def structure_factor_sum_rule(
        structure_factor: Array,
        omega_grid: Array
) -> float:
    """
    Compute sum rule: ∫ d\Omega S(q,\Omega) = <S\dag_q S_q>.
    
    Parameters
    ----------
    structure_factor : array-like
        S(q,\Omega).
    omega_grid : array-like
        Frequency grid (uniformly spaced).
        
    Returns
    -------
    float
        Integrated spectral weight.
    """
    return np.trapz(structure_factor, omega_grid)


def structure_factor_first_moment(
        structure_factor: Array,
        omega_grid: Array
) -> float:
    """
    Compute first moment: ∫ d\Omega \Omega S(q,\Omega) / ∫ d\Omega S(q,\Omega).
    
    This gives the average excitation energy at momentum q.
    
    Parameters
    ----------
    structure_factor : array-like
        S(q,\Omega).
    omega_grid : array-like
        Frequency grid.
        
    Returns
    -------
    float
        Average energy.
    """
    total_weight = np.trapz(structure_factor, omega_grid)
    first_moment = np.trapz(omega_grid * structure_factor, omega_grid)
    
    if total_weight > 0:
        return first_moment / total_weight
    else:
        return 0.0


# =============================================================================
# Utilities
# =============================================================================

def create_spin_q_operator_1d(
        q: float,
        lattice_positions: Array,
        local_spin_operators: Array
) -> Array:
    """
    Create Fourier-transformed spin operator S_q for 1D lattice.
    
    S_q = \sum _j S_j exp(iq·r_j)
    
    Parameters
    ----------
    q : float
        Momentum (1D).
    lattice_positions : array-like, shape (N,)
        Real-space positions r_j.
    local_spin_operators : array-like, shape (N_hilbert, N_hilbert)
        Local spin operators S_j (assumes they're block-diagonal in site basis).
        
    Returns
    -------
    Array
        S_q operator matrix.
        
    Notes
    -----
    For spin-1/2 systems, local_spin_operators would be σ^z/2 at each site.
    This is a simplified version; for general use, need proper tensor products.
    """
    lattice_positions = np.asarray(lattice_positions)
    S_q = np.zeros_like(local_spin_operators, dtype=complex)
    
    # This is a placeholder - actual implementation depends on Hilbert space structure
    # For a proper implementation, would need to construct S_q in the many-body basis
    raise NotImplementedError(
        "Full implementation requires knowledge of Hilbert space structure. "
        "See QES.Algebra.Operator module for proper operator construction."
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main functions
    'structure_factor_spin',
    'structure_factor_multi_q',
    
    # Analysis
    'structure_factor_sum_rule',
    'structure_factor_first_moment',
    
    # Utilities
    'create_spin_q_operator_1d',
]

# ============================================================================
#! End of file
# ============================================================================