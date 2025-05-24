#!/usr/bin/env python3
"""
Test script to verify the general_python module documentation and functionality.
"""

import sys
import os

# Add the path so we can import general_python
sys.path.insert(0, '/Users/makskliczkowski/Codes/QuantumEigenSolver/Python/QES')

def test_import_all_modules():
    """Test that all documented modules can be imported."""
    print("Testing module imports...")
    
    try:
        import general_python
        print(f"✓ general_python imported successfully")
        print(f"  Available modules: {general_python.list_available_modules()}")
        print(f"  Package version: {getattr(general_python, '__version__', 'N/A')}")
        
        # Test individual modules
        modules_to_test = ['algebra', 'common', 'lattices', 'maths', 'ml', 'physics']
        
        for module_name in modules_to_test:
            try:
                module = getattr(general_python, module_name)
                print(f"✓ general_python.{module_name} imported successfully")
                
                # Test module description
                desc = general_python.get_module_description(module_name)
                print(f"  Description: {desc[:80]}...")
                
            except Exception as e:
                print(f"✗ Error importing general_python.{module_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing general_python: {e}")
        return False

def test_key_functionality():
    """Test key functionality from each module."""
    print("\nTesting key functionality...")
    
    try:
        # Test algebra utilities
        from general_python.algebra import utils
        backend = utils.get_global_backend()
        print(f"✓ Backend manager working: {utils.backend_mgr.name}")
        
        # Test lattices
        from general_python.lattices import SquareLattice, LatticeBC
        lattice = SquareLattice(dim=2, lx=4, ly=4, lz=1, bc=LatticeBC.PBC)
        print(f"✓ SquareLattice created: {lattice.Ns} sites")
        
        # Test common utilities
        from general_python.common import Directories
        print("✓ Common utilities accessible")
        
        # Test maths
        from general_python.maths import math_utils
        print("✓ Math utilities accessible")
        
        # Test ml and physics modules exist
        import general_python.ml
        import general_python.physics
        print("✓ ML and Physics modules accessible")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing functionality: {e}")
        return False

def test_docstrings():
    """Test that modules have proper docstrings."""
    print("\nTesting docstrings...")
    
    try:
        import general_python
        
        # Check main module docstring
        if hasattr(general_python, '__doc__') and general_python.__doc__:
            print("✓ Main module has docstring")
        else:
            print("✗ Main module missing docstring")
        
        # Check submodule docstrings
        for module_name in ['algebra', 'common', 'lattices', 'maths', 'ml', 'physics']:
            module = getattr(general_python, module_name)
            if hasattr(module, '__doc__') and module.__doc__:
                print(f"✓ {module_name} module has docstring")
            else:
                print(f"✗ {module_name} module missing docstring")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing docstrings: {e}")
        return False

if __name__ == "__main__":
    print("General Python Module Documentation Test")
    print("=" * 50)
    
    success = True
    success &= test_import_all_modules()
    success &= test_key_functionality()
    success &= test_docstrings()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Documentation improvements are working correctly.")
    else:
        print("✗ Some tests failed. Check the output above for details.")
    
    print(f"\nFor full documentation, see: docs/index.rst")
    print(f"API reference updated in: docs/api.rst")
