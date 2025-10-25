"""
general_python/physics/spectral

Spectral functions and analysis for quantum systems.

This subpackage provides tools for:
- Density of states (DOS) calculations
- Green's functions and Fourier transforms
- Spectral functions A(k,\Omega)
- Response functions and correlations

Modules:
--------
- dos                   : Density of states (histogram and Gaussian-broadened)
- greens                : Time-resolved Green's functions and Fourier transforms
- spectral_function     : Spectral functions A(k,\Omega) = -Im[G]/π

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from . import dos
from . import greens
from . import spectral_function

__all__ = [
    'dos',
    'greens',
    'spectral_function',
]

# ============================================================================
#! End of file
# ============================================================================