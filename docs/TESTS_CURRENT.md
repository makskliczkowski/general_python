# Current Tests Inventory

This document lists the existing test modules in the codebase, their validation scope, runtime class, and determinism.

## Algebra

### `algebra/eigen/tests/test_block_lanczos_comprehensive.py`
- **Scope**: Comprehensive validation of `BlockLanczosEigensolver`. Tests various block sizes, $k$ values, basis transforms, and matrix-free operators. Includes JAX vs NumPy comparison.
- **Runtime Class**: Unit / Integration (can be slow depending on matrix size, but generally fast).
- **Determinism**: Deterministic (uses fixed random seeds).

### `algebra/eigen/tests/test_lanczos.py`
- **Scope**: Basic validation of `LanczosEigensolver`.
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.

### `algebra/solvers/tests/test_minres_qlp.py`
- **Scope**: Validation of `MinresQLPSolver` for symmetric/Hermitian systems. Tests basic solve, indefinite matrices, and initial guesses.
- **Runtime Class**: Unit.
- **Determinism**: Deterministic (uses fixed random seeds).
- **Notes**: Contains one `xfail` test (`test_minres_qlp_with_shift`) due to a known bug in shift parameter handling.

### `algebra/tests/test_algebra.py`
- **Scope**: General algebra utility tests.
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.

### `algebra/tests/test_pfaffian_jax_ops.py`
- **Scope**: JAX-based Pfaffian operations.
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.

### `algebra/tests/test_solver_edge_cases.py`
- **Scope**: Edge cases for solvers (e.g., small matrices).
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.

### `algebra/utilities/tests/test_pfaffian.py`
- **Scope**: Pfaffian calculation utilities.
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.

## Lattices

### `lattices/tests/test_lattice_invariants.py`
- **Scope**: Validation of `SquareLattice` geometry, neighbor finding (order=1, 2), and boundary conditions (PBC, OBC).
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.
- **Notes**: Contains a workaround for a known bug in `calculate_nnn` where `_nnn` is overwritten with `None`.

## Physics

### `physics/tests/test_operator_invariants.py`
- **Scope**: Validation of operator string parsing and site resolution.
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.

## Maths

### `maths/tests/test_utilities_edge_cases.py`
- **Scope**: Validation of math utilities (`find_nearest_val`, `mod_floor`, etc.) on edge inputs.
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.

## General / Integration

### `test_documentation.py`
- **Scope**: Verifies module imports and docstrings for documentation consistency.
- **Runtime Class**: System / Integration.
- **Determinism**: Deterministic.

### `tests/test_activations.py`
- **Scope**: Tests for activation functions.
- **Runtime Class**: Unit.
- **Determinism**: Deterministic.

### `tests/test_imports.py`
- **Scope**: Verifies that public modules can be imported correctly.
- **Runtime Class**: System.
- **Determinism**: Deterministic.

### `tests/test_lazy_imports.py`
- **Scope**: Verifies lazy import mechanisms.
- **Runtime Class**: System.
- **Determinism**: Deterministic.
