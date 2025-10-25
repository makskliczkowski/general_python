'''
General tests for import behavior of general_python package.

Ensures that submodules are lazily imported and key exports are available.

Tests:
- Lazy loading of subpackages
- Key class/function exports
- Package metadata presence

File        : general_python/tests/test_imports.py
Author      : Maksymilian Kliczkowski
License     : MIT       
'''

import importlib
import types

# -------------------------------------------------------------------

def test_root_imports_lazy():
    import general_python as gp
    # Accessing attribute should trigger lazy import
    maths = gp.maths
    assert isinstance(maths, types.ModuleType)

# -------------------------------------------------------------------

def test_math_utils_exports():
    from general_python.maths.math_utils import CUE_QR, Fitter, FitterParams
    assert callable(CUE_QR)
    # basic sanity: returns unitary-like matrix for small n
    import numpy as np
    U = CUE_QR(3)
    # check shapes
    assert U.shape == (3, 3)
    # numerical unitarity check
    ident = U.conj().T @ U
    assert np.allclose(ident, np.eye(3), atol=1e-7)

# -------------------------------------------------------------------

def test_package_metadata():
    import general_python as gp
    assert hasattr(gp, "__version__")

# -------------------------------------------------------------------
#! End of file
# -------------------------------------------------------------------