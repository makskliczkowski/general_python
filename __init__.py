"""
General Python utilities for Quantum EigenSolver.

This module provides common utilities, lattice definitions, algebra tools,
and mathematical functions used throughout the QES package.

Submodules:
-----------
- algebra   : Algebraic structures, eigensolvers, and backend operations
- common    : Common utilities (logging, file I/O, plotting)
- lattices  : Lattice definitions and k-space utilities
- maths     : Mathematical functions and numerical utilities
- physics   : Physical constants and quantum mechanics tools
- ml        : Machine learning utilities

Author      : Maksymilian Kliczkowski
Date        : 2025-02-02
"""

__version__ = "1.1.0"

# Avoid eager imports to prevent circular dependencies
# Import submodules only when explicitly needed