"""Algebra utilities for scientific computing.

Purpose
-------
Provide backend-agnostic linear algebra helpers, iterative solvers, and
random-matrix utilities used across QES and general_python workflows.

Input/output contracts
----------------------
- Solver factories expect arrays or linear operators that represent Ax=b.
- Preconditioners must map vectors of shape ``(n,)`` to ``(n,)`` and preserve
  the backend (NumPy or JAX) of the inputs.
- Random-matrix helpers return arrays with the requested shape and dtype.

Dtype and shape expectations
----------------------------
- Vectors are typically shape ``(n,)`` and matrices are shape ``(n, n)``.
- Complex dtypes are supported; callers should keep dtype consistent across a
  pipeline to avoid implicit promotion between float and complex.
- JAX backends use device arrays; NumPy backends use ``numpy.ndarray``.

Numerical stability notes
-------------------------
- Iterative solver convergence depends on matrix conditioning and tolerance.
- Preconditioners can improve stability but may change convergence paths; check
  residual norms explicitly for ill-conditioned problems.

Determinism notes
-----------------
- Deterministic results require explicit seed control in random-matrix helpers.
- JAX randomness requires passing PRNG keys; NumPy uses the global RNG state.
"""

from typing import TYPE_CHECKING
import importlib
from ..common.lazy import LazyImporter

# -----------------------------------------------------------------------------------------------
# Lazy Import Configuration
# -----------------------------------------------------------------------------------------------

# Mapping of attribute names to their module paths and actual attribute names
_LAZY_IMPORTS = {
    # Solver-related imports
    'SolverType'            : ('.solvers', 'SolverType'),
    'choose_solver'         : ('.solvers', 'choose_solver'),
    'get_backend_ops'       : ('.solvers.backend_ops', 'get_backend_ops'),
    'BackendOps'            : ('.solvers.backend_ops', 'BackendOps'),
    'default_ops'           : ('.solvers.backend_ops', 'default_ops'),
    'choose_precond'        : ('.preconditioners', 'choose_precond'),
    # Linalg module (heavy)
    'LinalgModule'          : ('.backend_linalg', None),  # None means import the whole module
    'backend_linalg'        : ('.backend_linalg', None),
    # Utility imports from common
    'MatrixPrinter'         : ('..common.plot', 'MatrixPrinter'),
    'get_logger'            : ('..common.flog', 'get_global_logger'),
    # Utils module exports (lazy - these are heavy)
    'backend_mgr'           : ('.utils', 'backend_mgr'),
    'get_backend'           : ('.utils', 'get_backend'),
    'get_global_backend'    : ('.utils', 'get_global_backend'),
    'ACTIVE_BACKEND_NAME'   : ('.utils', 'ACTIVE_BACKEND_NAME'),
    'ACTIVE_NP_MODULE'      : ('.utils', 'ACTIVE_NP_MODULE'),
    'ACTIVE_RANDOM'         : ('.utils', 'ACTIVE_RANDOM'),
    'ACTIVE_SCIPY_MODULE'   : ('.utils', 'ACTIVE_SCIPY_MODULE'),
    'ACTIVE_JIT'            : ('.utils', 'ACTIVE_JIT'),
    'ACTIVE_JAX_KEY'        : ('.utils', 'ACTIVE_JAX_KEY'),
    'ACTIVE_INT_TYPE'       : ('.utils', 'ACTIVE_INT_TYPE'),
    'ACTIVE_FLOAT_TYPE'     : ('.utils', 'ACTIVE_FLOAT_TYPE'),
    'ACTIVE_COMPLEX_TYPE'   : ('.utils', 'ACTIVE_COMPLEX_TYPE'),
    # Submodules (lazy)
    'solvers'               : ('.solvers', None),
    'preconditioners'       : ('.preconditioners', None),
    'ode'                   : ('.ode', None),
    'ran_wrapper'           : ('.ran_wrapper', None),
    'ran_matrices'          : ('.ran_matrices', None),
    'eigen'                 : ('.eigen', None),
    'utilities'             : ('.utilities', None),
    'utils'                 : ('.utils', None),
}

# For type checking, import types without runtime overhead
if TYPE_CHECKING:
    from .solvers               import SolverType, choose_solver
    from .solvers.backend_ops   import get_backend_ops, BackendOps, default_ops
    from .preconditioners       import choose_precond
    from .                      import backend_linalg as LinalgModule

# Initialize LazyImporter
_importer = LazyImporter(__name__, _LAZY_IMPORTS)

def __getattr__(name: str):
    """
    Module-level __getattr__ for lazy imports.

    This function is called when an attribute is not found in the module's namespace.
    It provides lazy loading for heavy dependencies.
    """
    return _importer.lazy_import(name)

# Helper function to get commonly used items with explicit imports
def _get_backend_and_logger():
    """Helper to get backend and logger lazily."""
    from .utils import get_backend
    get_logger = _importer.lazy_import('get_logger')
    return get_backend, get_logger

# --------------------------------------------------------------------------------------------------

__all__ = [
    # Lazy-loaded from utils
    "backend_mgr", "get_backend", "get_global_backend",
    # Global singletons (from utils)
    "ACTIVE_BACKEND_NAME", "ACTIVE_NP_MODULE", "ACTIVE_RANDOM",
    "ACTIVE_SCIPY_MODULE", "ACTIVE_JIT", "ACTIVE_JAX_KEY",
    "ACTIVE_INT_TYPE", "ACTIVE_FLOAT_TYPE", "ACTIVE_COMPLEX_TYPE",
    # Solver-related (lazy)
    "SolverType", "choose_solver",
    # Backend ops helpers (lazy)
    "get_backend_ops", "BackendOps", "default_ops",
    # Preconditioners (lazy)
    "choose_precond",
    # Submodules (lazy)
    "LinalgModule", "backend_linalg",
    "solvers", "preconditioners", "ode",
    "ran_wrapper", "ran_matrices",
    "eigen", "utilities", "utils",
]

# --------------------------------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------------------------------
