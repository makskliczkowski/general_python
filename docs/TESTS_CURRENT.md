# Current Test Inventory

This document lists the existing tests in the `general_python` library, their purpose, and characteristics.

| Test File | Purpose | Type | Deterministic |
| :--- | :--- | :--- | :--- |
| `algebra/utilities/tests/test_pfaffian.py` | Unit tests for Pfaffian calculation and Sherman-Morrison skew updates using JAX. | Unit | Yes |
| `algebra/solvers/tests/test_minres_qlp.py` | Unit/Integration tests for MINRES-QLP solver (NumPy backend). Checks convergence on SPD/indefinite systems. | Unit/Integration | Yes (seeded) |
| `algebra/tests/test_pfaffian_jax_ops.py` | Unit tests for JAX implementation of Pfaffian operations and performance. | Unit | Yes (seeded) |
| `algebra/tests/test_algebra.py` | Unit tests for linear algebra utilities: change of basis, outer product, Kronecker product, ket-bra. | Unit | Yes |
| `algebra/eigen/tests/test_lanczos.py` | Comprehensive tests for Lanczos eigensolver (NumPy & JAX). Checks smallest/largest eigenvalues, matrix-free ops, eigenvectors. | Integration | Yes (seeded) |
| `algebra/eigen/tests/test_all_solvers.py` | High-level integration test checking all eigensolvers (Exact, Lanczos, Block Lanczos). | Integration | Yes (seeded) |
| `algebra/eigen/tests/test_block_lanczos_comprehensive.py` | Detailed tests for Block Lanczos solver with varying block sizes, backends, and matrix-free ops. | Integration | Yes (seeded) |
| `tests/test_imports.py` | Checks package structure, lazy loading of submodules, and key exports. | Integration | Yes |
| `tests/test_activations.py` | Unit tests for activation functions (tanh, etc.) in NumPy and JAX, and complex-value handling. | Unit | Yes |
| `tests/test_lazy_imports.py` | Verifies lazy import mechanism for heavy dependencies. | Integration | Yes |
| `test_documentation.py` | Meta-test ensuring documentation consistency or similar (root level). | Meta | Yes |

## Notes

*   Most tests in `algebra` subdirectories currently use `sys.path` insertion hacks to resolve the package, which should be replaced with proper `general_python` imports.
*   `test_documentation.py` checks for docstring validity and module imports.
*   Deterministic behavior is ensured via `np.random.seed` or `jax.random.PRNGKey` in most numerical tests.
