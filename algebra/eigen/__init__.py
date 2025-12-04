"""
Eigenvalue Solvers Module

This module provides iterative eigenvalue solvers for large sparse matrices,
particularly useful for quantum mechanics and large-scale linear algebra problems.

Available Solvers:
    - Exact Diagonalization: Full eigenvalue decomposition for small-medium systems
    - Lanczos: For symmetric/Hermitian matrices (extremal eigenvalues)
    - Arnoldi: For general non-symmetric matrices
    - Block Lanczos: For multiple eigenpairs simultaneously

Factory Function:
    - choose_eigensolver: Unified interface for creating eigenvalue solvers
    - decide_method: Automatically choose method based on problem characteristics

Standard Result:
    - EigenResult: Standardized return type (eigenvalues, eigenvectors, iterations, converged)

This module uses lazy imports to minimize startup overhead.

-----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2025-12-01
Version         : 1.0
-----------------------------------------------------------
"""

from typing import TYPE_CHECKING
import importlib

# -----------------------------------------------------------------------------------------------
# Lazy Import Configuration
# -----------------------------------------------------------------------------------------------

_LAZY_IMPORTS = {
    # Exact diagonalization
    'ExactEigensolver'              : ('.exact', 'ExactEigensolver'),
    'ExactEigensolverScipy'         : ('.exact', 'ExactEigensolverScipy'),
    'full_diagonalization'          : ('.exact', 'full_diagonalization'),
    # Lanczos (symmetric/Hermitian)
    'LanczosEigensolver'            : ('.lanczos', 'LanczosEigensolver'),
    'LanczosEigensolverScipy'       : ('.lanczos', 'LanczosEigensolverScipy'),
    # Arnoldi (general matrices)
    'ArnoldiEigensolver'            : ('.arnoldi', 'ArnoldiEigensolver'),
    'ArnoldiEigensolverScipy'       : ('.arnoldi', 'ArnoldiEigensolverScipy'),
    # Block Lanczos
    'BlockLanczosEigensolver'       : ('.block_lanczos', 'BlockLanczosEigensolver'),
    'BlockLanczosEigensolverScipy'  : ('.block_lanczos', 'BlockLanczosEigensolverScipy'),
    # Factory interface
    'choose_eigensolver'            : ('.factory', 'choose_eigensolver'),
    'decide_method'                 : ('.factory', 'decide_method'),
    # Result type
    'EigenResult'                   : ('.result', 'EigenResult'),
}

_LAZY_CACHE = {}

# For type checking only
if TYPE_CHECKING:
    from .exact         import ExactEigensolver, ExactEigensolverScipy, full_diagonalization
    from .lanczos       import LanczosEigensolver, LanczosEigensolverScipy
    from .arnoldi       import ArnoldiEigensolver, ArnoldiEigensolverScipy
    from .block_lanczos import BlockLanczosEigensolver, BlockLanczosEigensolverScipy
    from .factory       import choose_eigensolver, decide_method
    from .result        import EigenResult

# -----------------------------------------------------------------------------------------------

def __getattr__(name: str):
    """
    Module-level __getattr__ for lazy imports.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_path, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_path, package=__name__)
    
    if attr_name is None:
        result = module
    else:
        result = getattr(module, attr_name)
    
    _LAZY_CACHE[name] = result
    return result


__all__ = [
    'EigenResult',
    # Exact diagonalization
    'ExactEigensolver',
    'ExactEigensolverScipy',
    'full_diagonalization',
    # Lanczos (symmetric/Hermitian)
    'LanczosEigensolver',
    'LanczosEigensolverScipy',
    # Arnoldi (general matrices)
    'ArnoldiEigensolver',
    'ArnoldiEigensolverScipy',
    # Block Lanczos
    'BlockLanczosEigensolver',
    'BlockLanczosEigensolverScipy',
    # Factory interface
    'choose_eigensolver',
    'decide_method',
]

# ------------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------------