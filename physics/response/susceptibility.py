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
    from ...algebra.utils import JAX_AVAILABLE, Array, get_backend
except ImportError:
    JAX_AVAILABLE   = False
    Array           = Union[np.ndarray, list, tuple]
    get_backend     = lambda x="default": np

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from functools import partial
else:
    jax = None
    jnp = np

# =============================================================================
# Internal Core Functions
# =============================================================================

def _compute_lehmann_components(
        hamiltonian_eigvals : Array,
        hamiltonian_eigvecs : Array,
        operator_q          : Array,
        temperature         : float,
        be                  : any) -> Tuple[Array, Array]:
    """
    Precompute frequency-independent components for Lehmann representation.
    """
    eigvals     = be.asarray(hamiltonian_eigvals)
    eigvecs     = be.asarray(hamiltonian_eigvecs, dtype=complex)
    A_q         = be.asarray(operator_q, dtype=complex)
    N           = len(eigvals)

    # Transform operator to eigenbasis
    A_q_eigen   = eigvecs.conj().T @ A_q @ eigvecs

    # Thermal weights
    if temperature > 0:
        beta    = 1.0 / temperature
        E_min   = be.min(eigvals)
        rho     = be.exp(-beta * (eigvals - E_min))
        Z       = be.sum(rho)
        rho    /= Z
    else:
        # T=0: only ground state occupied
        if be is jnp and jax is not None:
            rho = be.zeros(N).at[0].set(1.0)
        else:
            rho = np.zeros(N)
            rho[0] = 1.0
            rho = be.asarray(rho)

    # Precompute matrix elements and energy differences
    R           = rho[:, None] - rho[None, :]
    M           = be.abs(A_q_eigen)**2
    Omega_nm    = eigvals[None, :] - eigvals[:, None]

    from .structure_factor import SPARSE_MATRIX_THRESHOLD

    # Optimize by pruning negligible thermal weights and matrix elements.
    # This reduces array sizes and skips calculations for forbidden/negligible transitions.
    mask        = (be.abs(R) > 1e-12) & (M > SPARSE_MATRIX_THRESHOLD)
    weighted    = R[mask] * M[mask]
    Omega_flat  = Omega_nm[mask]

    return weighted, Omega_flat

# =============================================================================
# Dynamical Susceptibility chi(q,\Omega)
# =============================================================================

def susceptibility_lehmann(
        hamiltonian_eigvals     : Array,
        hamiltonian_eigvecs     : Array,
        operator_q              : Array,
        omega                   : float,
        eta                     : float = 0.01,
        temperature             : float = 0.0,
        backend                 : str = "default") -> complex:
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
    backend : str, optional
        Numerical backend to use (default: "default").
        
    Returns
    -------
    complex
        Dynamical susceptibility chi(q,\Omega).
    """
    be = get_backend(backend)
    weighted, Omega_flat = _compute_lehmann_components(
        hamiltonian_eigvals, hamiltonian_eigvecs, operator_q, temperature, be
    )
    
    chi = be.sum(weighted / (omega - Omega_flat + 1j * eta))
    
    return complex(chi)

def susceptibility_multi_omega(
        hamiltonian_eigvals : Array,
        hamiltonian_eigvecs : Array,
        operator_q          : Array,
        omega_grid          : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0,
        backend             : str = "default") -> Array:
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
    backend : str, optional
        Numerical backend to use (default: "default").
        
    Returns
    -------
    Array, shape (n_omega,), complex
        chi(q,\Omega) for each frequency.
    """
    be          = get_backend(backend)
    omega_grid  = be.asarray(omega_grid)
    n_omega     = len(omega_grid)
    
    # 1-3. Precompute frequency-independent components once
    weighted, Omega_flat = _compute_lehmann_components(
        hamiltonian_eigvals, hamiltonian_eigvecs, operator_q, temperature, be
    )
    
    num_trans   = len(Omega_flat)
    
    # 4. Vectorize over frequencies (with memory safety check)
    if n_omega * num_trans < 10**7:
        # Full vectorization: (n_omega, 1) - (1, num_trans)
        denom   = omega_grid[:, None] - Omega_flat[None, :] + 1j * eta
        chi     = be.sum(weighted[None, :] / denom, axis=1)
    else:
        # Loop over frequencies to save memory, still vectorized over transitions
        chi = be.zeros(n_omega, dtype=complex)
        for i in range(n_omega):
            val = be.sum(weighted / (omega_grid[i] - Omega_flat + 1j * eta))
            if be is jnp and jax is not None:
                chi = chi.at[i].set(val)
            else:
                chi[i] = val

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
        temperature         : float = 0.0,
        backend             : str = "default") -> Array:
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
    backend : str, optional
        Numerical backend to use (default: "default").
        
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
        temperature=temperature,
        backend=backend
    )

def charge_susceptibility(
        hamiltonian_eigvals : Array,
        hamiltonian_eigvecs : Array,
        density_q           : Array,
        omega_grid          : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0,
        backend             : str = "default") -> Array:
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
    backend : str, optional
        Numerical backend to use (default: "default").
        
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
        temperature=temperature,
        backend=backend
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
