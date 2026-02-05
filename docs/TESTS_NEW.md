# New Tests Documentation

This document describes the new tests added to the `general_python` library to ensure correctness and stability.

| Test File | Purpose | Failure Mode | Module Guarded |
| :--- | :--- | :--- | :--- |
| `algebra/tests/test_solvers_sanity.py` | Validates that `LanczosEigensolver` and `MinresQLPSolver` converge correctly on known physical problems (2D Laplacian) and handle different data types (float32, float64, complex128) deterministically. | Regressions in solver logic, numerical instability, or broken dtype support. Hardcoded tolerances in library may cause failures if precision drops. | `algebra.solvers`, `algebra.eigen` |
| `lattices/tests/test_lattice_invariants.py` | Verifies `SquareLattice` geometry, including neighbor finding (NN, NNN) under Periodic and Open Boundary Conditions, coordinate calculation, and site indexing. | Incorrect neighbor lists, broken boundary wrapping, or coordinate mapping errors. Catches bugs where neighbors are not correctly calculated or stored (e.g., `nnn` assignment bug). | `lattices` |
| `physics/tests/test_operator_invariants.py` | Tests the parsing logic in `Operators` class, specifically `resolveSite` and `resolve_operator`, ensuring edge cases like "L", "pi", and division are handled. | Failures in parsing operator strings, incorrect site index resolution, or crashes on valid syntax. | `physics.operators` |
| `maths/tests/test_utilities_edge_cases.py` | Checks edge cases and behavior of math utilities: `find_nearest_val` (returns index for arrays), power functions (`next_power`, `prev_power`), custom modulo functions, and `Fitter` basics. | Unexpected behavior in helper functions, regression in custom modulo logic, or `Fitter` breaking changes. | `maths` |

## Notes

*   **Lattices:** `test_next_nearest_neighbors` in `lattices/tests/test_lattice_invariants.py` is currently marked as `xfail` due to a known bug in `lattice.py` where `calculate_nnn` overwrites `self._nnn` with `None`.
*   **Algebra:** `test_lanczos_dtypes` in `algebra/tests/test_solvers_sanity.py` manually checks residual norms instead of relying on `result.converged` because the native Lanczos implementation has a hardcoded strict tolerance (`1e-8`) that `float32` cannot satisfy.
*   **Maths:** `find_nearest_val` returns the *index* (wrapped in array) for numpy inputs, which is counter-intuitive but currently asserted behavior. `mod_round` behaves like truncation for positive numbers.
