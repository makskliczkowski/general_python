r"""
Magnetic and charge susceptibilities chi(q,\Omega) for quantum systems.

The dynamical susceptibility is the linear response function:
    chi(q,\Omega) = iintdt e^{i\Omegat} <[A_q(t), A\dag_{-q}(0)]>

Related to structure factor via fluctuation-dissipation theorem:
    S(q,\Omega) = -(1/\pi) Im[chi(q,\Omega)] / (1 - exp(-\beta\Omega))
    
This module provides functions to compute dynamical and static susceptibilities
using the Lehmann representation from Hamiltonian eigenvalues and eigenvectors.

----------------------------------------------------------------------------
File    : general_python/physics/response/susceptibility.py
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-11-01
----------------------------------------------------------------------------
"""

from typing import Optional, Union, Tuple
import numpy as np

try:
    from ...algebra.utils import JAX_AVAILABLE, Array
except ImportError:
    JAX_AVAILABLE   = False
    Array           = Union[np.ndarray, list, tuple]

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Dynamical Susceptibility chi(q,\Omega)
# =============================================================================

def susceptibility_lehmann(
        hamiltonian_eigvals     : Array,
        hamiltonian_eigvecs     : Array,
        operator_q              : Array,
        omega                   : float,
        eta                     : float = 0.01,
        temperature             : float = 0.0) -> complex:
    r"""
    Compute dynamical susceptibility using Lehmann representation.
    
    chi(q,\Omega) = \sum _{m,n} (rho _m - rho _n) <m|A_q|n><n|A\dag_q|m> / (\Omega - \Omega_nm + ieta )
    
    where \Omega_nm = E_n - E_m and rho _m are thermal occupation factors.
    
    Parameters
    ----------
    hamiltonian_eigvals : array-like
        Eigenvalues E_n.
    hamiltonian_eigvecs : array-like, shape (N, N)
        Eigenvectors (columns are eigenstates).
    operator_q : array-like, shape (N, N)
        Operator A_q (e.g., magnetization, charge density).
    omega : float
        Frequency \Omega.
    eta : float, optional
        Broadening parameter (default: 0.01).
    temperature : float, optional
        Temperature (default: 0 for T=0).
        
    Returns
    -------
    complex
        Dynamical susceptibility chi(q,\Omega).
        
    Notes
    -----
    At T=0, only ground state contributes: chi ~ \sum _n |<n|A|0>|^2 / (\Omega - \Omega_n0 + ieta ).
    """
    eigvals     = np.asarray(hamiltonian_eigvals)
    eigvecs     = np.asarray(hamiltonian_eigvecs, dtype=complex)
    A_q         = np.asarray(operator_q, dtype=complex)
    
    N           = len(eigvals)
    
    # Transform operator to eigenbasis
    A_q_eigen   = eigvecs.conj().T @ A_q @ eigvecs
    
    # Thermal weights
    if temperature > 0:
        beta    = 1.0 / temperature
        E_min   = np.min(eigvals)
        rho     = np.exp(-beta * (eigvals - E_min))
        Z       = np.sum(rho)
        rho    /= Z
    else:
        # T=0: only ground state occupied
        rho     = np.zeros(N)
        rho[0]  = 1.0

    # Lehmann representation
    chi = 0.0 + 0.0j
    
    for m in range(N):
        for n in range(N):
            if np.abs(rho[m] - rho[n]) < 1e-12:
                continue
            
            omega_nm        = eigvals[n] - eigvals[m]
            matrix_element  = A_q_eigen[m, n] * np.conj(A_q_eigen[m, n])
            
            chi += (rho[m] - rho[n]) * matrix_element / (omega - omega_nm + 1j * eta)
    
    return chi

def susceptibility_multi_omega(
        hamiltonian_eigvals : Array,
        hamiltonian_eigvecs : Array,
        operator_q          : Array,
        omega_grid          : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0) -> Array:
    r"""
    Compute chi(q,\Omega) for multiple frequencies.
    
    Parameters
    ----------
    hamiltonian_eigvals : array-like
        Eigenvalues.
    hamiltonian_eigvecs : array-like
        Eigenvectors.
    operator_q : array-like
        Operator A_q.
    omega_grid : array-like
        Frequency grid.
    eta : float, optional
        Broadening (default: 0.01).
    temperature : float, optional
        Temperature (default: 0).
        
    Returns
    -------
    Array, shape (n_omega,), complex
        chi(q,\Omega) for each frequency.
    """
    omega_grid  = np.asarray(omega_grid)
    n_omega     = len(omega_grid)
    
    chi = np.zeros(n_omega, dtype=complex)
    
    for i, omega in enumerate(omega_grid):
        chi[i] = susceptibility_lehmann(
            hamiltonian_eigvals,
            hamiltonian_eigvecs,
            operator_q,
            omega,
            eta=eta,
            temperature=temperature
        )
    
    return chi

# =============================================================================
# Static Susceptibility chi(q,\Omega=0)
# =============================================================================

def static_susceptibility(
        hamiltonian_eigvals : Array,
        hamiltonian_eigvecs : Array,
        operator_q          : Array,
        temperature         : float = 0.0) -> float:
    r"""
    Compute static (\Omega=0) susceptibility chi(q,0).
    
    chi(q,0) = \beta <(A_q - <A_q>)^2>  (fluctuation-dissipation)
    
    At T=0:
    chi(q,0) = 2 \sum _{n\neq 0} |<n|A_q|0>|^2 / (E_n - E_0)
    
    Parameters
    ----------
    hamiltonian_eigvals : array-like
        Eigenvalues.
    hamiltonian_eigvecs : array-like
        Eigenvectors.
    operator_q : array-like
        Operator A_q.
    temperature : float, optional
        Temperature (default: 0).
        
    Returns
    -------
    float
        Static susceptibility chi(q,0).
    """
    eigvals     = np.asarray(hamiltonian_eigvals)
    eigvecs     = np.asarray(hamiltonian_eigvecs, dtype=complex)
    A_q         = np.asarray(operator_q, dtype=complex)

    N           = len(eigvals)

    # Transform to eigenbasis
    A_q_eigen   = eigvecs.conj().T @ A_q @ eigvecs

    if temperature > 0:
        # Finite temperature: use fluctuation-dissipation
        beta    = 1.0 / temperature
        E_min   = np.min(eigvals)
        rho     = np.exp(-beta * (eigvals - E_min))
        Z       = np.sum(rho)
        rho     /= Z

        # <A> = \sum _n rho _n <n|A|n>
        A_avg   = np.sum(rho * np.real(np.diag(A_q_eigen)))

        # <A^2> = \sum _n rho _n <n|A^2|n>
        A2_eigen    = A_q_eigen @ A_q_eigen
        A2_avg      = np.sum(rho * np.real(np.diag(A2_eigen)))

        # chi = \beta (<A^2> - <A>^2)
        chi_static  = beta * (A2_avg - A_avg**2)
    
    else:
        # T=0: sum over excited states
        E_0         = eigvals[0]
        chi_static  = 0.0
        
        for n in range(1, N):
            matrix_element_sq   = np.abs(A_q_eigen[0, n])**2
            energy_diff         = eigvals[n] - E_0
            
            if energy_diff > 1e-12:
                chi_static += 2.0 * matrix_element_sq / energy_diff
    
    return chi_static

# =============================================================================
# Magnetic Susceptibility
# =============================================================================

def magnetic_susceptibility(
        hamiltonian_eigvals : Array,
        hamiltonian_eigvecs : Array,
        magnetization_q     : Array,
        omega_grid          : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0) -> Array:
    r"""
    Compute magnetic susceptibility chi_M(q,\Omega).
    
    This is the susceptibility_multi_omega with operator = magnetization.
    
    Parameters
    ----------
    hamiltonian_eigvals : array-like
        Eigenvalues.
    hamiltonian_eigvecs : array-like
        Eigenvectors.
    magnetization_q : array-like
        Magnetization operator M_q = \sum _j M_j exp(iq\cdot r_j).
    omega_grid : array-like
        Frequency grid.
    eta : float, optional
        Broadening (default: 0.01).
    temperature : float, optional
        Temperature (default: 0).
        
    Returns
    -------
    Array, complex
        chi_M(q,\Omega).
    """
    return susceptibility_multi_omega(
        hamiltonian_eigvals,
        hamiltonian_eigvecs,
        magnetization_q,
        omega_grid,
        eta=eta,
        temperature=temperature
    )

def charge_susceptibility(
        hamiltonian_eigvals : Array,
        hamiltonian_eigvecs : Array,
        density_q           : Array,
        omega_grid          : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0) -> Array:
    r"""
    Compute charge susceptibility chi_c(q,\Omega).
    
    This is the susceptibility_multi_omega with operator = charge density.
    
    Parameters
    ----------
    hamiltonian_eigvals : array-like
        Eigenvalues.
    hamiltonian_eigvecs : array-like
        Eigenvectors.
    density_q : array-like
        Charge density operator n_q = \sum _j n_j exp(iq\cdot r_j).
    omega_grid : array-like
        Frequency grid.
    eta : float, optional
        Broadening (default: 0.01).
    temperature : float, optional
        Temperature (default: 0).
        
    Returns
    -------
    Array, complex
        chi_c(q,\Omega).
    """
    return susceptibility_multi_omega(
        hamiltonian_eigvals,
        hamiltonian_eigvecs,
        density_q,
        omega_grid,
        eta=eta,
        temperature=temperature
    )

# =============================================================================
# Relation to Structure Factor
# =============================================================================

def susceptibility_to_structure_factor(
        chi         : Array,
        omega_grid  : Array,
        temperature : float = 0.0) -> Array:
    r"""
    Convert susceptibility to structure factor via fluctuation-dissipation theorem.
    
    S(q,\Omega) = -(1/\pi) Im[chi(q,\Omega)] / (1 - exp(-\beta\Omega))
    
    At T=0:
    S(q,\Omega) = -(1/\pi) Im[chi(q,\Omega)]  for \Omega > 0
    
    Parameters
    ----------
    chi : array-like, complex
        Dynamical susceptibility chi(q,\Omega).
    omega_grid : array-like
        Frequency grid.
    temperature : float, optional
        Temperature (default: 0).
        
    Returns
    -------
    Array, real
        Structure factor S(q,\Omega).
    """
    chi         = np.asarray(chi, dtype=complex)
    omega_grid  = np.asarray(omega_grid)
    
    if temperature > 0:
        beta                = 1.0 / temperature
        # Avoid division by zero at \Omega=0
        occupation_factor   = np.where(
                                np.abs(omega_grid) > 1e-12,
                                1.0 / (1.0 - np.exp(-beta * omega_grid)),
                                beta / 2.0  # Limit as \Omega->0: 1/(1-exp(-\beta\Omega)) -> \beta/2
                            )
        S_q_omega = -(1.0 / np.pi) * np.imag(chi) * occupation_factor
    else:
        # T=0: simple relation
        S_q_omega = -(1.0 / np.pi) * np.imag(chi)
        # S(q,\Omega) only defined for \Omega > 0 at T=0
        S_q_omega = np.where(omega_grid > 0, S_q_omega, 0.0)
    
    return S_q_omega

def structure_factor_to_susceptibility(
        S_q_omega       : Array,
        omega_grid      : Array,
        temperature     : float = 0.0) -> Array:
    r"""
    Convert structure factor to susceptibility (inverse of above).
    
    Im[chi(q,\Omega)] = -\pi S(q,\Omega) (1 - exp(-\beta\Omega))
    
    Parameters
    ----------
    S_q_omega : array-like
        Structure factor.
    omega_grid : array-like
        Frequency grid.
    temperature : float, optional
        Temperature (default: 0).
        
    Returns
    -------
    Array, complex
        Imaginary part of susceptibility.
        
    Notes
    -----
    This only gives Im[chi]. To get full chi, need Kramers-Kronig relations.
    """
    S_q_omega   = np.asarray(S_q_omega)
    omega_grid  = np.asarray(omega_grid)
    
    if temperature > 0:
        beta                = 1.0 / temperature
        occupation_factor   = 1.0 - np.exp(-beta * omega_grid)
    else:
        occupation_factor   = 1.0
    
    Im_chi = -np.pi * S_q_omega * occupation_factor
    
    return Im_chi

# =============================================================================
# Sum Rules
# =============================================================================

def susceptibility_sum_rule_check(
        chi                 : Array,
        omega_grid          : Array,
        operator_q          : Array,
        commutator_norm_sq  : float) -> Tuple[float, float]:
    r"""
    Check f-sum rule for susceptibility.
    
    \int d\Omega \Omega Im[chi(q,\Omega)] = -\pi/2 <[A_q, [H, A\dag_q]]>
    
    Parameters
    ----------
    chi : array-like, complex
        Susceptibility.
    omega_grid : array-like
        Frequency grid.
    operator_q : array-like
        Operator A_q (not used in simple version).
    commutator_norm_sq : float
        <[A_q, [H, A\dag_q]]>.
        
    Returns
    -------
    integral : float
        int d\Omega \Omega Im[chi(q,\Omega)].
    expected : float
        -\pi/2 <commutator>.
    """
    Im_chi = np.imag(chi)
    integral = np.trapz(omega_grid * Im_chi, omega_grid)
    expected = -np.pi / 2 * commutator_norm_sq
    
    return integral, expected

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Lehmann representation
    'susceptibility_lehmann',
    'susceptibility_multi_omega',
    
    # Static susceptibility
    'static_susceptibility',
    
    # Physical susceptibilities
    'magnetic_susceptibility',
    'charge_susceptibility',
    
    # Relation to structure factor
    'susceptibility_to_structure_factor',
    'structure_factor_to_susceptibility',
    
    # Sum rules
    'susceptibility_sum_rule_check',
]

# #############################################################################
#! End of file
# #############################################################################