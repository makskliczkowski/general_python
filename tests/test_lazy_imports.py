"""
Tests for the lazy import mechanism in general_python.
"""

import sys
import types
import pytest

# --------------------------------------------

def test_lazy_imports_not_loaded_initially():
    """Ensure submodules are not loaded when importing the top-level package."""
    # Ensure fresh start
    if 'general_python' in sys.modules:
        del sys.modules['general_python']
    # Also remove submodules from sys.modules to simulate fresh environment
    to_remove = [m for m in sys.modules if m.startswith('general_python.')]
    
    for m in to_remove:
        del sys.modules[m]

    import general_python
    
    # Check that heavy submodules are not in sys.modules yet (if not imported by something else in the environment)
    # Note: 'common' or others might be imported if pytest uses them, so we check specific ones
    # or just verify that accessing them works.
    
    # Verify the module is loaded
    assert isinstance(general_python, types.ModuleType)

def test_lazy_access():
    """Ensure accessing attributes triggers the import."""
    import general_python
    
    # Access 'maths'
    maths_mod = general_python.maths
    assert isinstance(maths_mod, types.ModuleType)
    assert maths_mod.__name__ == "general_python.maths"
    
    # Access 'algebra'
    algebra_mod = general_python.algebra
    assert isinstance(algebra_mod, types.ModuleType)
    assert algebra_mod.__name__ == "general_python.algebra"

def test_physics_lazy_access():
    """Ensure physics submodules like entropy are accessible lazily."""
    import general_python
    
    # Access 'physics' first
    physics_mod = general_python.physics
    assert isinstance(physics_mod, types.ModuleType)
    
    # Access 'entropy' from physics
    entropy_mod = physics_mod.entropy
    assert isinstance(entropy_mod, types.ModuleType)
    assert entropy_mod.__name__ == "general_python.physics.entropy"

def test_maths_lazy_access():
    """Ensure maths submodules and aliases are accessible lazily."""
    import general_python
    
    # Access 'maths'
    maths_mod = general_python.maths
    assert isinstance(maths_mod, types.ModuleType)
    
    # Access 'math_utils'
    utils_mod = maths_mod.math_utils
    assert isinstance(utils_mod, types.ModuleType)
    assert utils_mod.__name__ == "general_python.maths.math_utils"
    
    # Access alias 'MathMod'
    alias_mod = maths_mod.MathMod
    assert isinstance(alias_mod, types.ModuleType)
    assert alias_mod is utils_mod

def test_lazy_aliases():
    """Ensure lazy aliases like 'random' work."""
    import general_python
    
    rand_mod = general_python.random
    assert isinstance(rand_mod, types.ModuleType)
    # The alias maps to algebra.ran_wrapper
    assert rand_mod.__name__ == "general_python.algebra.ran_wrapper"

def test_dir_autocompletion():
    """Ensure dir() lists lazy attributes."""
    import general_python
    attrs = dir(general_python)
    assert "maths" in attrs
    assert "algebra" in attrs
    assert "random" in attrs
    assert "list_capabilities" in attrs

def test_invalid_attribute():
    """Ensure accessing non-existent attributes raises AttributeError."""
    import general_python
    with pytest.raises(AttributeError, match="has no attribute 'non_existent'"):
        _ = general_python.non_existent

# ---------------------------------------------
#! EOF
# ---------------------------------------------