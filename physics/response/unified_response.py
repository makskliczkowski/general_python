"""
general_python/physics/response/unified_response.py

Unified entry point for response function calculations.

Automatically selects the appropriate method based on system size and type:
  - Small systems (N ≤ 64): Full many-body exact diagonalization + Lehmann
  - Large systems: Quadratic/bubble approximation
  - Provides unified interface to physicist users

This module bridges the conceptual gap between many-body and single-particle
response functions, allowing seamless switching based on system capabilities.

----------------------------------------------------------------------------
Author  : Maksymilian Kliczkowski
Email   : maksymilian.kliczkowski@pwr.edu.pl
Date    : 2025-11-01
Version : 1.0
----------------------------------------------------------------------------
"""

from typing import Optional, Union, Tuple, Literal
import numpy as np
from numpy.typing import NDArray

# Type alias
Array = Union[np.ndarray, list, tuple]

# Import both response calculation backends
try:
    from ..spectral.spectral_backend import (
        operator_spectral_function_multi_omega,
        susceptibility_bubble_multi_omega,
        thermal_weights,
        conductivity_kubo_bubble,
    )
except ImportError:
    raise ImportError(
        "Failed to import spectral_backend. "
        "Ensure general_python.algebra is properly installed."
    )

try:
    from .susceptibility import susceptibility_multi_omega
except ImportError:
    raise ImportError(
        "Failed to import susceptibility module. "
        "Ensure general_python.physics.response is properly installed."
    )

# =============================================================================
# Unified Response Calculator
# =============================================================================

class UnifiedResponseFunction:
    """
    Unified interface for computing response functions.
    
    Automatically chooses between exact many-body (Lehmann) and mean-field
    (bubble) calculations based on system size and type.
    
    Examples
    --------
    >>> calc = UnifiedResponseFunction(H, eigenvalues, eigenvectors)
    >>> omegas = np.linspace(-5, 5, 200)
    >>> 
    >>> # Automatic method selection
    >>> chi = calc.compute_response(operator, omegas, temperature=0.5)
    """
    
    def __init__(
        self,
        hamiltonian_matrix  : Optional[Array] = None,
        eigenvalues         : Optional[Array] = None,
        eigenvectors        : Optional[Array] = None,
        is_quadratic        : bool = False,
        system_type         : Literal["many-body", "quadratic", "auto"] = "auto",):
        """
        Initialize response function calculator.
        
        Parameters
        ----------
        hamiltonian_matrix : array-like, optional
            Full Hamiltonian (for determining system size).
        eigenvalues : array-like
            Eigenvalues from diagonalization.
        eigenvectors : array-like
            Eigenvectors from diagonalization.
        is_quadratic : bool, optional
            Hint that system is quadratic (default: False).
        system_type : {"many-body", "quadratic", "auto"}, optional
            Explicitly specify system type. "auto" (default) determines from size.
        """
        self.H              = hamiltonian_matrix
        self.eigenvalues    = np.asarray(eigenvalues)
        self.eigenvectors   = np.asarray(eigenvectors)
        self.is_quadratic   = is_quadratic
        self.system_type    = system_type
        
        # Determine system regime
        self._determine_regime()
    
    def _determine_regime(self):
        """Determine if system is many-body or single-particle."""
        if self.system_type == "many-body":
            self.regime = "many-body"
        elif self.system_type == "quadratic":
            self.regime = "quadratic"
        elif self.system_type == "auto":
            # Heuristic: dimension indicates regime
            dim = len(self.eigenvalues)
            
            if self.is_quadratic or dim <= 128:
                self.regime = "many-body"
            else:
                self.regime = "quadratic"
        else:
            raise ValueError(f"Unknown system_type: {self.system_type}")
    
    def compute_response(
        self,
        operator        : Array,
        omega_grid      : Array,
        eta             : float = 0.01,
        temperature     : float = 0.0,
        kind            : Literal["spectral", "susceptibility", "auto"] = "auto",) -> Tuple[Array, str]:
        """
        Compute response function using automatic method selection.
        
        Parameters
        ----------
        operator : array-like
            Operator O for response χ_OO(omega ).
        omega_grid : array-like
            Frequency grid for evaluation.
        eta : float, optional
            Broadening parameter (default: 0.01).
        temperature : float, optional
            Temperature (default: 0).
        kind : {"spectral", "susceptibility", "auto"}, optional
            Type of response. "spectral" = |⟨m|O|n⟩|², 
            "susceptibility" = ⟨m|O|n⟩⟨n|O†|m⟩.
            
        Returns
        -------
        chi : array-like
            Response function χ(omega ) or A(omega ).
        method : str
            Which method was used ("many-body-lehmann", "quadratic-bubble", etc.)
        """
        operator    = np.asarray(operator)
        omega_grid  = np.asarray(omega_grid)
        
        if kind == "auto":
            # Default: spectral for many-body, susceptibility for quadratic
            kind = "spectral" if self.regime == "many-body" else "susceptibility"
        
        if self.regime == "many-body":
            # Use exact many-body Lehmann representation
            chi = operator_spectral_function_multi_omega(
                omega_grid,
                self.eigenvalues,
                self.eigenvectors,
                operator,
                eta=eta,
                temperature=temperature,
            )
            method = "many-body-lehmann"
        
        elif self.regime == "quadratic":
            # Use quadratic bubble
            chi = susceptibility_bubble_multi_omega(
                omega_grid,
                self.eigenvalues,
                vertex=operator,
                occupation=None,  # Assumes ground state
                eta=eta,
            )
            method = "quadratic-bubble"
        
        else:
            raise ValueError(f"Unknown regime: {self.regime}")
        
        return chi, method
    
    def compute_dos(self, eta: float = 0.01) -> Tuple[Array, str]:
        """
        Compute density of states (DOS).
        
        Parameters
        ----------
        eta : float, optional
            Broadening (default: 0.01).
            
        Returns
        -------
        dos : array-like
            Density of states values at eigenvalues.
        method : str
            Method used.
        """
        # Diagonal spectral function
        from ..spectral.spectral_backend import spectral_function_diagonal
        
        dos = spectral_function_diagonal(
                omega       =   self.eigenvalues,
                eigenvalues =   self.eigenvalues,
                eta         =   eta,
            )
        
        # Normalize
        dos = dos / np.trapz(dos, self.eigenvalues)
        
        return dos, "spectral-diagonal"
    
    def __repr__(self) -> str:
        return (
            f"UnifiedResponseFunction(regime='{self.regime}', "
            f"dim={len(self.eigenvalues)}, "
            f"quadratic={self.is_quadratic})"
        )

# =============================================================================
# Convenience Functions
# =============================================================================

def compute_response(
    eigenvalues     : Array,
    eigenvectors    : Array,
    operator        : Array,
    omega_grid      : Array,
    temperature     : float = 0.0,
    eta             : float = 0.01,
    system_type     : Literal["many-body", "quadratic", "auto"] = "auto",) -> Tuple[Array, str]:
    """
    Compute response function (one-call convenience).
    
    This is the simplest interface: pass eigenvalues/eigenvectors and operator,
    get response function back.
    
    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues from diagonalization.
    eigenvectors : array-like
        Eigenvectors from diagonalization.
    operator : array-like
        Operator O.
    omega_grid : array-like
        Frequency grid.
    temperature : float, optional
        Temperature (default: 0).
    eta : float, optional
        Broadening (default: 0.01).
    system_type : str, optional
        "many-body", "quadratic", or "auto" (default).
        
    Returns
    -------
    chi : array-like
        Response function.
    method : str
        Method used.
        
    Examples
    --------
    >>> E, V = np.linalg.eigh(H)
    >>> O = build_magnetization_operator()
    >>> omegas = np.linspace(-5, 5, 200)
    >>> chi, method = compute_response(E, V, O, omegas, temperature=0.1)
    >>> print(f"Used method: {method}")
    """
    calc = UnifiedResponseFunction(
        eigenvalues    =   eigenvalues,
        eigenvectors   =   eigenvectors,
        system_type    =   system_type,
    )
    return calc.compute_response(operator, omega_grid, eta=eta, temperature=temperature)

def compute_dos_direct(
    eigenvalues : Array,
    omega_grid  : Array,
    eta         : float = 0.01,) -> Array:
    """
    Compute density of states directly.
    
    Parameters
    ----------
    eigenvalues : array-like
        System eigenvalues.
    omega_grid : array-like
        Frequency grid for DOS.
    eta : float, optional
        Broadening (default: 0.01).
        
    Returns
    -------
    dos : array-like
        Density of states at each omega.
    """
    from ..spectral.spectral_backend import spectral_function_diagonal
    
    dos = np.zeros_like(omega_grid, dtype=float)
    
    for i, omega in enumerate(omega_grid):
        dos[i] = np.sum(
            spectral_function_diagonal(omega, eigenvalues, eta=eta)
        )
    
    return dos

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'UnifiedResponseFunction',
    'compute_response',
    'compute_dos_direct',
]

# #############################################################################
#! End of file
# #############################################################################
