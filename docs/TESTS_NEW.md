# New Tests Documentation

This document describes the new tests added to the codebase to improve coverage of correctness invariants.

## Algebra

### `algebra/tests/test_solver_convergence.py`
- **Purpose**: Verify convergence of linear solvers and eigensolvers on dense and sparse matrices.
- **Coverage**:
  - `LanczosEigensolver`: Dense Hermitian (random), Sparse (2D Laplacian).
  - `MinresQLPSolver`: Dense SPD, Sparse SPD.
  - Residual norms explicitly checked against tolerance.
  - Deterministic behavior with fixed seeds.
  - Robustness across `float32` and `complex128` dtypes.
- **Guarded Modules**: `algebra.eigen.lanczos`, `algebra.solvers.minres_qlp`.

## Lattices

### `lattices/tests/test_lattice_boundaries.py`
- **Purpose**: Validate lattice connectivity and boundary conditions, especially for small lattices and edge cases.
- **Coverage**:
  - `SquareLattice` neighbors for 1x2, 2x1, 3x3 grids.
  - Boundary Conditions: `PBC` (Periodic), `OBC` (Open).
  - Explicit check for NNN (Next-Nearest Neighbor) logic using the workaround for a known bug in `calculate_nnn`.
- **Guarded Modules**: `lattices.lattice`, `lattices.square`.

## Physics

### `physics/tests/test_operator_properties.py`
- **Purpose**: Ensure physical correctness of operators and entropy calculations.
- **Coverage**:
  - Pauli matrix commutation relations (`[Si, Sj] = i e_ijk Sk`).
  - Entropy functions (`purity`, `vn_entropy`) on pure and mixed states.
  - Operator string parsing logic and edge cases (e.g., arithmetic in site indices).
- **Guarded Modules**: `physics.operators`, `physics.entropy`.

## Maths

### `maths/tests/test_math_utils_comprehensive.py`
- **Purpose**: Test mathematical utility functions with comprehensive edge cases.
- **Coverage**:
  - `find_nearest_val`: Behavior with empty arrays and NaNs. (Note: returns index, not value).
  - Modulo functions (`mod_floor`, `mod_round`, `mod_ceil`, `mod_euc`): Behavior with negative inputs, documenting non-standard implementations where applicable.
- **Guarded Modules**: `maths.math_utils`.
