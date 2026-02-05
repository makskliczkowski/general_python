"""Physics utilities for quantum simulations.

Purpose
-------
Provide physics-specific analysis routines for spectra, thermodynamics, and
operator-based calculations used across QES workflows.

Input/output contracts
----------------------
- Density-matrix helpers accept state vectors or operators and return matrices
  of shape ``(dim, dim)``.
- Spectral and response utilities expect eigenvalues/eigenvectors and return
  arrays indexed by frequency grids.
- Thermal routines expect energies and inverse temperature ``beta`` and return
  scalar observables or arrays over parameter grids.

Dtype and shape expectations
----------------------------
- State vectors are typically shape ``(dim,)`` or ``(n_states, dim)``.
- Operators are shape ``(dim, dim)`` with real or complex dtype.
- Frequency-domain outputs are usually complex arrays of shape ``(n_omega, ...)``.

Numerical stability notes
-------------------------
- Near-degenerate spectra require careful broadening and resolution choices.
- Exponential weights in thermal sums can overflow; rescale energies when needed.

Determinism notes
-----------------
- Most routines are deterministic given fixed inputs.
- Optional JAX variants require explicit PRNG keys if random sampling is used.
"""

import sys
import importlib
from typing import TYPE_CHECKING

# -----------------------------------------------------------------------------------------------
# Lazy Import Configuration
# -----------------------------------------------------------------------------------------------

# Mapping of attribute names to (module_relative_path, attribute_name_in_module)
_LAZY_IMPORTS = {
    # Basic modules
    "density_matrix"    : (".density_matrix",       None),
    "eigenlevels"       : (".eigenlevels",          None),
    "entropy"           : (".entropy",              None),
    "operators"         : (".operators",            None),
    
    # Statistical & thermal
    "statistical"       : (".statistical",          None),
    "thermal"           : (".thermal",              None),
    
    # Subpackages
    "spectral"          : (".spectral",             None),
    "response"          : (".response",             None),
    
    # Specialized
    "single_particle"   : (".sp",                   None),
    "sp"                : (".sp",                   None), # Alias
    
    # JAX versions (will fail on access if JAX not installed, handled in caller/user code)
    "density_matrix_jax": (".density_matrix_jax",   None),
    "entropy_jax"       : (".entropy_jax",          None),
}

# Cache for lazily loaded modules/attributes
_LAZY_CACHE = {}

if TYPE_CHECKING:
    from . import density_matrix
    from . import eigenlevels
    from . import entropy
    from . import operators
    from . import statistical
    from . import thermal
    from . import spectral
    from . import response
    from . import sp as single_particle
    from . import sp
    try:
        from . import density_matrix_jax
        from . import entropy_jax
    except ImportError:
        pass

# -----------------------------------------------------------------------------------------------
# Lazy Import Implementation
# -----------------------------------------------------------------------------------------------

def _lazy_import(name: str):
    """
    Lazily import a module or attribute based on _LAZY_IMPORTS configuration.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_path, attr_name = _LAZY_IMPORTS[name]
    
    try:
        module = importlib.import_module(module_path, package=__name__)
        
        if attr_name is None:
            result = module
        else:
            result = getattr(module, attr_name)
        
        _LAZY_CACHE[name] = result
        return result
    except ImportError as e:
        raise ImportError(f"Failed to import lazy module '{name}' from '{module_path}': {e}") from e

def __getattr__(name: str):
    return _lazy_import(name)

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))

__all__ = list(_LAZY_IMPORTS.keys()) + ["list_capabilities"]

# -----------------------------------------------------------------------------------------------
# Capabilities
# -----------------------------------------------------------------------------------------------

def list_capabilities():
    """
    List available physics capabilities and modules.
    """
    # Check JAX availability dynamically
    try:
        import jax
        jax_available = True
    except ImportError:
        jax_available = False

    capabilities = {
        'modules': [
            'density_matrix', 'eigenlevels', 'entropy', 
            'operators', 'statistical', 'thermal', 'single_particle'
        ],
        'spectral': ['dos', 'greens', 'spectral_function'],
        'response': ['structure_factor', 'susceptibility'],
        'jax_available': jax_available
    }
    return capabilities

__version__ = '0.1.0'
__author__  = 'Maksymilian Kliczkowski'
__email__   = 'maksymilian.kliczkowski@pwr.edu.pl'

# ------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------
