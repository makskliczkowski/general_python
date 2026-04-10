#!/usr/bin/env python3
"""Smoke checks for the ``general_python`` documentation surface."""

import sys

import general_python as gp


def test_import_all_modules():
    """Verify that the documented top-level modules resolve correctly."""
    print("Testing module imports...")

    print(f"(v) general_python imported successfully")
    print(f"  Available modules: {gp.list_available_modules()}")
    print(f"  Package version: {getattr(gp, '__version__', 'N/A')}")

    modules_to_test = ["algebra", "common", "lattices", "maths", "ml", "physics"]

    for module_name in modules_to_test:
        module = getattr(gp, module_name)
        print(f"(v) general_python.{module_name} imported successfully")

        desc = gp.get_module_description(module_name)
        print(f"  Description: {desc[:80]}...")


def test_key_functionality():
    """Exercise a representative API from each major submodule."""
    print("\nTesting key functionality...")

    from general_python.algebra import utils
    utils.get_global_backend()
    print(f"(v) Backend manager working: {utils.backend_mgr.name}")

    from general_python.lattices import SquareLattice, LatticeBC

    lattice = SquareLattice(dim=2, lx=4, ly=4, lz=1, bc=LatticeBC.PBC)
    print(f"(v) SquareLattice created: {lattice.Ns} sites")

    from general_python.common import Directories

    print("(v) Common utilities accessible")

    from general_python.maths import math_utils

    print("(v) Math utilities accessible")

    import general_python.ml as ml
    import general_python.physics as physics

    print("(v) ML and Physics modules accessible")


def test_docstrings():
    """Ensure the documented modules expose non-empty docstrings."""
    print("\nTesting docstrings...")

    assert hasattr(gp, '__doc__') and gp.__doc__
    print("(v) Main module has docstring")

    for module_name in ["algebra", "common", "lattices", "maths", "ml", "physics"]:
        module = getattr(gp, module_name)
        assert hasattr(module, '__doc__') and module.__doc__
        print(f"(v) {module_name} module has docstring")


if __name__ == "__main__":
    print("General Python Module Documentation Test")
    print("=" * 50)

    try:
        test_import_all_modules()
        test_key_functionality()
        test_docstrings()
        print("\n" + "=" * 50)
        print("(v) All tests passed! Documentation improvements are working correctly.")
    except Exception as e:
        print(f"(x) Tests failed: {e}")
        sys.exit(1)

    print(f"\nFor full documentation, see: docs/index.rst")
    print(f"API reference updated in: docs/api.rst")
