r"""
general_python/physics/response

Response functions and dynamical correlation functions for quantum systems.

This subpackage provides tools for:
- Dynamic structure factors S(q,\Omega) for spin systems
- Magnetic and charge susceptibilities chi(q,\Omega)
- Correlation functions and linear response

Modules:
--------
- structure_factor : Dynamic structure factor for spins (must be fast!)
- susceptibility   : Magnetic and charge susceptibilities

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from . import structure_factor
from . import susceptibility

__all__ = [
    'structure_factor',
    'susceptibility',
]

# ============================================================================
#! End of file
# ============================================================================