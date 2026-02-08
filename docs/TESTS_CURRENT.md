# Current Tests Inventory

This document lists the existing test modules in the repository, their validation scope, runtime classification, and determinism.

| Test Module | Validates | Runtime Class | Deterministic | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `algebra/eigen/tests/test_block_lanczos_comprehensive.py` | Block Lanczos eigensolver (NumPy/JAX) correctness and convergence. | Unit | Yes (seeds used) | Comprehensive coverage including block sizes and JAX support. |
| `algebra/eigen/tests/test_lanczos.py` | Lanczos eigensolver (NumPy/JAX/SciPy) correctness, convergence, and breakdown detection. | Unit | Yes (seeds used) | Checks tolerance handling (known issue with NumPy backend ignoring tolerance). |
| `algebra/solvers/tests/test_minres_qlp.py` | MINRES-QLP solver for symmetric indefinite systems. | Unit | Yes (seeds used) | Contains one skipped test (`test_minres_qlp_with_shift`) due to broken shift handling in library. |
| `algebra/tests/test_algebra.py` | Basic algebraic utilities import and function. | Unit | Yes | Minimal smoke test. |
| `algebra/tests/test_pfaffian_jax_ops.py` | JAX operations for Pfaffian computation. | Unit | Yes | Requires JAX. |
| `algebra/tests/test_solvers_sanity.py` | Integration tests for solvers on physical systems (2D Laplacian). | Integration | Yes (seeds used) | Verifies solvers work on realistic Hamiltonians. |
| `algebra/utilities/tests/test_pfaffian.py` | Pfaffian calculation utilities. | Unit | Yes | |
| `lattices/tests/test_lattice_invariants.py` | `SquareLattice` geometry, neighbors, and boundary conditions. | Unit | Yes | Contains one `xfail` test (`test_next_nearest_neighbors`) due to `calculate_nnn` bug in library. |
| `maths/tests/test_utilities_edge_cases.py` | Math utility functions (`find_nearest`, modular arithmetic, fitting). | Unit | Yes | Documents non-standard behavior of `find_nearest_val` (returns index) and `mod_round` (truncation). |
| `physics/tests/test_operator_invariants.py` | Operator string parsing and resolution (`Operators` class). | Unit | Yes | Checks parsing logic for site strings. |
| `test_documentation.py` | Docstring validity and import checks. | Unit | Yes | Meta-test. |
| `tests/test_activations.py` | Activation functions (tanh, log_cosh) for neural networks. | Unit | Yes | Checks JAX/NumPy consistency. |
| `tests/test_imports.py` | Package importability. | Unit | Yes | Meta-test. |
| `tests/test_lazy_imports.py` | Lazy import mechanism. | Unit | Yes | Meta-test. |

**Summary:**
- Most tests are unit tests and deterministic.
- Known issues:
  - `lattices/tests/test_lattice_invariants.py`: `test_next_nearest_neighbors` fails due to library bug.
  - `algebra/solvers/tests/test_minres_qlp.py`: `test_minres_qlp_with_shift` is skipped.
