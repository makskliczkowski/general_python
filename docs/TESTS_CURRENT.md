# Current Test Inventory

This document lists the existing test modules in the `general_python` codebase, their purpose, runtime class, and determinism status.

## Inventory

| Test Module | Purpose | Runtime Class | Deterministic |
| :--- | :--- | :--- | :--- |
| `algebra/tests/test_algebra.py` | Validates basic linear algebra operations (change of basis, tensor products). | Unit | Yes |
| `algebra/tests/test_solvers_sanity.py` | Checks convergence of Lanczos and MINRES solvers on known matrices (e.g., 2D Laplacian). | Integration | Yes |
| `algebra/tests/test_solver_edge_cases.py` | Tests solvers on singular, degenerate, and ill-conditioned matrices. | Integration | Yes |
| `algebra/tests/test_pfaffian_jax_ops.py` | Tests JAX-based Pfaffian operations. | Unit | Yes |
| `algebra/utilities/tests/test_pfaffian.py` | Tests general Pfaffian utilities. | Unit | Yes |
| `algebra/eigen/tests/test_lanczos.py` | Comprehensive tests for Lanczos eigensolver (NumPy/JAX, complex, matrix-free). | Integration | Yes |
| `algebra/eigen/tests/test_block_lanczos_comprehensive.py` | Tests Block Lanczos solver behavior. | Integration | Yes |
| `algebra/solvers/tests/test_minres_qlp.py` | Tests MINRES-QLP solver specifically. | Integration | Yes |
| `lattices/tests/test_lattice_new.py` | Validates neighbor finding and coordinate generation for Square Lattices. | Unit | Yes |
| `lattices/tests/test_lattice_invariants.py` | Tests invariant properties of lattice structures. | Unit | Yes |
| `physics/tests/test_operator_algebra_new.py` | Validates Pauli commutators and entropy/purity functions. | Unit | Yes |
| `physics/tests/test_operator_invariants.py` | Tests invariant properties of operators. | Unit | Yes |
| `maths/tests/test_math_utils_new.py` | Tests mathematical utilities (find_nearest_val, mod_round, etc.). | Unit | Yes |
| `maths/tests/test_utilities_edge_cases.py` | Tests utilities on edge inputs. | Unit | Yes |
| `tests/test_activations.py` | Tests ML activation functions (tanh, log_cosh) and RBM properties. | Unit | Yes |
| `tests/test_imports.py` | Verifies package import logic and availability of key modules. | Unit | Yes |
| `tests/test_lazy_imports.py` | Verifies lazy loading mechanism to ensure heavy dependencies are not loaded prematurely. | Unit | Yes |
| `test_documentation.py` | Validates that documentation strings and module imports work correctly. | Integration | Yes |

## Notes

- **Deterministic**: All tests currently appear to use fixed seeds (`np.random.seed`) or deterministic inputs.
- **Runtime Class**: Most tests are fast units tests. Solver tests (`algebra/eigen/`, `algebra/solvers/`) are integration tests that may take slightly longer but are generally fast enough for CI.
- **Redundancy**: There is some overlap between `algebra/tests/test_solvers_sanity.py` and `algebra/eigen/tests/test_lanczos.py`, but they cover different aspects (sanity check vs comprehensive suite). Both are valuable.
