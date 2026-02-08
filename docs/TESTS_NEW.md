# New Tests Inventory

This document details the newly added tests to improve correctness coverage across the library.

## Algebra
**Module:** `algebra/tests/test_solver_edge_cases.py`
- **Purpose**: Verify solver behavior on edge cases and ill-conditioned systems.
- **Tests**:
  - `test_lanczos_singular_matrix`: Ensures `LanczosEigensolver` correctly finds 0 eigenvalue for singular matrices.
  - `test_lanczos_degenerate_eigenvalues`: Checks behavior with degenerate eigenvalues (multiplicity > 1).
  - `test_minres_ill_conditioned`: Verifies `MinresQLPSolver` can reduce residual even for high condition number matrices.

## Lattices
**Module:** `lattices/tests/test_lattice_new.py`
- **Purpose**: Expand lattice geometry coverage beyond standard 2D PBC.
- **Tests**:
  - `test_square_lattice_1d`: Validates 1D chain neighbor logic.
  - `test_square_lattice_3d`: Validates 3D cubic lattice neighbor logic.
  - `test_get_coordinates_shapes`: Ensures coordinate array shapes are consistent (N, 3).
  - `test_mbc_boundary`: verifies Mixed Boundary Conditions (MBC) behave as cylinder (PBC x, OBC y).

## Physics
**Module:** `physics/tests/test_operator_algebra_new.py`
- **Purpose**: Verify fundamental quantum mechanical invariants.
- **Tests**:
  - `test_pauli_commutators`: Checks that manually constructed Pauli matrices satisfy $[S_i, S_j] = i \epsilon_{ijk} S_k$.
  - `test_operators_parsing`: Verifies string resolution for operators.
  - `test_entropy_functions`: Checks `purity` and `vn_entropy` for pure and mixed states.

## Maths
**Module:** `maths/tests/test_math_utils_new.py`
- **Purpose**: Document and verify edge case behavior of math utilities.
- **Tests**:
  - `test_find_nearest_val_empty`: Ensures finding value in empty array raises `ValueError`.
  - `test_find_nearest_val_nan`: Checks behavior with `NaN`s in array.
  - `test_mod_round_negative`: Documents specific rounding behavior for negative inputs (tends towards zero/truncation logic).
