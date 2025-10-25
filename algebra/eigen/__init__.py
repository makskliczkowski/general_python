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
"""

from .exact import ExactEigensolver, ExactEigensolverScipy, full_diagonalization
from .lanczos import LanczosEigensolver, LanczosEigensolverScipy
from .arnoldi import ArnoldiEigensolver, ArnoldiEigensolverScipy
from .block_lanczos import BlockLanczosEigensolver, BlockLanczosEigensolverScipy
from .factory import choose_eigensolver, decide_method
from .result import EigenResult

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