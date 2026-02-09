# New Tests Documentation

This document describes the new tests added to the codebase to improve correctness, coverage, and stability.

## New Test Files

### 1. `algebra/tests/test_solver_properties.py`

*   **Purpose**: Validates numerical properties of linear algebra solvers (`LanczosEigensolver`, `MinresQLPSolver`).
*   **Key Checks**:
    *   **Residual Norms**: Ensures solvers achieve requested tolerance for various dtypes (`float32`, `float64`, `complex64`, `complex128`).
    *   **Determinism**: Verifies that solvers produce identical results when provided with fixed seeds (addressing potential non-deterministic behavior in iterative methods).
    *   **Convergence**: Tests convergence on known matrices (e.g., 2D Laplacian) to ensure basic correctness.
*   **Guarded Modules**: `algebra.eigen`, `algebra.solvers`.

### 2. `lattices/tests/test_lattice_boundaries.py`

*   **Purpose**: explicit testing of boundary conditions and edge cases for Lattice classes.
*   **Key Checks**:
    *   **Small Lattices**: Validates behavior for degenerate cases like 1x1 and 2x1 lattices, which often break neighbor-finding logic.
    *   **Boundary Conditions**: strictly tests Periodic (PBC) vs Open (OBC) boundary logic for neighbor counting.
    *   **Consistency**: Verifies that generated coordinates match the neighbor topology (distance check).
*   **Guarded Modules**: `lattices.square`.

### 3. `physics/tests/test_physics_invariants.py`

*   **Purpose**: Enforces physical invariants in operator and state manipulations.
*   **Key Checks**:
    *   **Hermiticity**: Ensures constructed Pauli matrices and density matrices are Hermitian.
    *   **Trace/Normalization**: Verifies density matrices have unit trace and purity properties ($Tr(\rho^2) \le 1$).
    *   **Commutators**: Validates Jacobi identity for operator commutators.
*   **Guarded Modules**: `physics.operators`, `physics.entropy`.

### 4. `maths/tests/test_common_edge_cases.py`

*   **Purpose**: Tests mathematical utility functions against edge cases and potentially non-intuitive behaviors.
*   **Key Checks**:
    *   **Modulo Operations**: Documents and verifies the specific behavior of `mod_round` and `mod_floor` with negative divisors (e.g., `mod_round(5, -2) == -3`).
    *   **Next Power**: Tests `next_power` with inputs like 0 (raises Error) or non-integers.
    *   **Array Utils**: Tests `find_maximum_idx` with NumPy 2.x compatibility (handling `AxisError` location).
*   **Guarded Modules**: `maths.math_utils`.

## Improvements to Existing Tests

### `algebra/tests/test_algebra.py`
*   **Change**: Uncommented and enabled `outer`, `kron`, and `ket_bra` tests in the `run_all` suite.
*   **Impact**: Increases coverage of basic tensor operations which were previously skipped.

### `maths/tests/test_math_utils_new.py`
*   **Change**: Expanded parametrization for `mod_round` and `mod_floor` to explicitly cover and document negative input behaviors.
*   **Impact**: Prevents regression in utility functions that have non-standard rounding logic.
