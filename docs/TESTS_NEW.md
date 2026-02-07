# Inventory of New Tests

This document lists the new tests added to the repository to improve coverage and verify correctness invariants.

## Algebra Tests

### `algebra/tests/test_solvers_new.py`
*   **Purpose**: Verify correctness and stability of linear algebra solvers (`LanczosEigensolver`, `MinresQLPSolver`).
*   **Validates**:
    *   Convergence on known 1D/2D Laplacian matrices.
    *   Residual norm checks: $||Ax - \lambda x|| < \epsilon$ and $||Ax - b|| < \epsilon$.
    *   Deterministic behavior with fixed random seeds.
    *   Support for different dtypes (`float32`, `float64`, `complex128`).
*   **Guards**: `algebra.eigen`, `algebra.solvers`.

## Lattices Tests

### `lattices/tests/test_lattice_new.py`
*   **Purpose**: Verify lattice geometry, boundary conditions, and neighbor finding.
*   **Validates**:
    *   `SquareLattice` in 1D, 2D, and 3D configurations.
    *   Periodic Boundary Conditions (PBC) and Open Boundary Conditions (OBC) coordinate wrapping.
    *   Neighbor finding correctness for edge cases.
    *   Documents that `HexagonalLattice` and `TriangularLattice` are currently broken/stubbed (asserts failure).
*   **Guards**: `lattices.square`, `lattices.lattice`.

## Physics Tests

### `physics/tests/test_physics_new.py`
*   **Purpose**: Verify physical operator algebra and state properties.
*   **Validates**:
    *   Pauli matrix algebra: Hermiticity, traceless, commutation relations $[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k$.
    *   Density matrix properties: $\text{Tr}(\rho)=1$, $\rho^2=\rho$ (pure state), Hermiticity.
    *   Edge cases for `Operators.resolveSite` string parsing.
*   **Guards**: `physics.operators`, general quantum mechanics invariants.

## Maths Tests

### `maths/tests/test_maths_new.py`
*   **Purpose**: Verify mathematical utility functions and fitting routines.
*   **Validates**:
    *   `find_nearest_val` returns index (not value) and handles edge cases.
    *   `Fitter` classes (`fitLinear`, `fit_histogram`) work on synthetic data with noise.
    *   Modulo arithmetic functions (`mod_euc`, etc.) raise `ValueError` on zero divisor.
*   **Guards**: `maths.math_utils`.
