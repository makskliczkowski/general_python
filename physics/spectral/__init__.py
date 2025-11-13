r"""
general_python/physics/spectral

Spectral functions and Green's functions for quantum systems.

This subpackage provides unified spectral analysis tools:
- Green's functions: quadratic, many-body, Lanczos
- Spectral functions A(\omega) = -(1/\pi) Im[G(\omega)]
- Density of states with various broadening methods
- Momentum-resolved spectral functions A(k,\omega)

Structure:
----------
- spectral_backend  : Core implementations (source of truth)
- greens_function   : Green's function wrappers and Fourier utilities
- spectral_function : Spectral function wrappers
- dos               : Density of states calculations

All implementations live in spectral_backend.py. Other modules provide
convenient wrappers and specialized utilities.

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

__all__ = [
    'spectral_backend',
    'greens_function', 
    'spectral_function',
    'dos',
]

from . import dos
from . import greens_function
from . import spectral_function

__all__ = [
    'dos',
    'greens_function',
    'spectral_function',
]

# ============================================================================
#! End of file
# ============================================================================