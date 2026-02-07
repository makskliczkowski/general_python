# Inventory of Current Tests

This document lists the existing test modules in the repository, their purpose, runtime classification, and determinism status.

## Algebra Tests

### `algebra/eigen/tests/test_block_lanczos_comprehensive.py`
*   **Validates**: Block Lanczos eigensolver functionality, including convergence and handling of degenerate eigenvalues.
*   **Class**: Integration / Slow
*   **Deterministic**: Yes (seeded)

### `algebra/eigen/tests/test_lanczos.py`
*   **Validates**: Basic Lanczos eigensolver functionality.
*   **Class**: Unit
*   **Deterministic**: Yes

### `algebra/solvers/tests/test_minres_qlp.py`
*   **Validates**: MINRES-QLP solver for linear systems Ax=b.
*   **Class**: Unit
*   **Deterministic**: Yes

### `algebra/tests/test_algebra.py`
*   **Validates**: Basic algebraic operations and backend utilities.
*   **Class**: Unit
*   **Deterministic**: Yes

### `algebra/tests/test_pfaffian_jax_ops.py`
*   **Validates**: Pfaffian calculation using JAX operations.
*   **Class**: Unit
*   **Deterministic**: Yes

### `algebra/tests/test_solvers_sanity.py`
*   **Validates**: Convergence of solvers (Lanczos, MINRES) on known matrices (e.g., Laplacian). Checks residuals and determinism.
*   **Class**: Integration / Slow
*   **Deterministic**: Yes

### `algebra/utilities/tests/test_pfaffian.py`
*   **Validates**: Pfaffian utility functions.
*   **Class**: Unit
*   **Deterministic**: Yes

## Lattices Tests

### `lattices/tests/test_lattice_invariants.py`
*   **Validates**: Lattice invariants such as neighbor finding, coordinate calculation, and boundary conditions for SquareLattice.
*   **Class**: Unit
*   **Deterministic**: Yes

## Physics Tests

### `physics/tests/test_operator_invariants.py`
*   **Validates**: Operator string resolution and basic operator invariants.
*   **Class**: Unit
*   **Deterministic**: Yes

## Maths Tests

### `maths/tests/test_utilities_edge_cases.py`
*   **Validates**: Edge cases for mathematical utility functions.
*   **Class**: Unit
*   **Deterministic**: Yes

## General Tests

### `test_documentation.py`
*   **Validates**: Documentation consistency and importability.
*   **Class**: Unit
*   **Deterministic**: Yes

### `tests/test_activations.py`
*   **Validates**: Activation functions (likely for ML components).
*   **Class**: Unit
*   **Deterministic**: Yes

### `tests/test_imports.py`
*   **Validates**: Package imports work correctly.
*   **Class**: Unit
*   **Deterministic**: Yes

### `tests/test_lazy_imports.py`
*   **Validates**: Lazy import mechanism.
*   **Class**: Unit
*   **Deterministic**: Yes
