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
    # math_utils should expose Fitter APIs but not random matrices
    from general_python.maths.math_utils import Fitter, FitterParams
    assert hasattr(Fitter, 'fit_linear')
    assert isinstance(FitterParams(None, None, None), FitterParams)


def test_cue_qr_from_ran_matrices():
    # CUE_QR now lives in algebra.ran_matrices
    import numpy as np
    from general_python.algebra.ran_matrices import CUE_QR
    assert callable(CUE_QR)
    
    U       = CUE_QR(3, simple=False)
    assert U.shape == (3, 3)
    
    ident   = U.conj().T @ U
    assert np.allclose(ident, np.eye(3), atol=1e-9)

# -------------------------------------------------------------------

def test_package_metadata():
    import general_python as gp
    assert hasattr(gp, "__version__")

# -------------------------------------------------------------------
#! End of file
# -------------------------------------------------------------------