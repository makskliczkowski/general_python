"""
Algebraic operations and utilities for scientific computing.

This module provides a unified interface for linear algebra, solvers, and random number
generation that works seamlessly across different backends (NumPy and JAX).

Core Functionalities
--------------------
*   **Linear Solvers**: Iterative solvers (CG, MINRES, GMRES, etc.) that support both CPU (NumPy)
    and GPU/TPU (JAX) execution. See ``algebra.solvers``.
*   **Preconditioners**: Abstract and concrete preconditioners for iterative methods.
    See ``algebra.preconditioners``.
*   **Backend Abstraction**: Utilities to write backend-agnostic code. The module automatically
    dispatches to ``numpy`` or ``jax.numpy`` based on configuration or input types.
*   **Random Number Generation**: A unified wrapper around ``numpy.random`` and ``jax.random``
    to ensure reproducible scientific simulations. See ``algebra.ran_wrapper``.

Backend Agnosticism
-------------------
The library is designed to write code once and run it anywhere.
*   **NumPy**: Default backend for standard CPU execution.
*   **JAX**: Optional backend for high-performance computing, automatic differentiation,
    and JIT compilation.

Lazy Loading
------------
To minimize startup time, heavy dependencies (like JAX or large submodules) are imported
lazily. They are only loaded when you access them or when you explicitly configure
a backend that requires them.

Example
-------
.. code-block:: python

    from general_python.algebra import choose_solver

    # Create a Conjugate Gradient solver
    solver = choose_solver('cg', backend='jax', tol=1e-6)

    # Solve Ax = b
    result = solver.solve_instance(b, x0, a=A)

-----------------------------------------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maksymilian.kliczkowski@pwr.edu.pl
Date            : 2025-02-01
Version         : 1.1
Description     : General Algebra Module with Lazy Imports
-----------------------------------------------------------------------------------------------
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