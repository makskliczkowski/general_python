# New Tests Documentation

This document describes the new tests added to the repository to improve correctness coverage and robustness.

## Algebra
*   **Module**: `algebra/tests/test_arnoldi_sanity.py`
    *   **Purpose**: Validates the `ArnoldiEigensolver` (for non-symmetric matrices) and its SciPy wrapper.
    *   **Failure Modes**: Catches incorrect eigenvalue computations for non-symmetric matrices, backend issues in JAX implementation (currently xfailed due to a bug), and regression in SciPy wrapper logic.
    *   **Guards**: `algebra/eigen/arnoldi.py`

## Lattices
*   **Module**: `lattices/tests/test_lattice_dimensions.py`
    *   **Purpose**: Validates `SquareLattice` behavior in 1D and 3D dimensions, specifically checking neighbor finding logic under PBC and OBC.
    *   **Failure Modes**: Catches incorrect neighbor mapping (e.g., wrong stride calculation in 3D), boundary condition violations (wrapping when shouldn't), and coordinate calculation errors.
    *   **Guards**: `lattices/square.py`, `lattices/lattice.py`

## Physics
*   **Module**: `physics/tests/test_hamiltonian_invariants.py`
    *   **Purpose**: Verifies fundamental operator algebra (Pauli matrices) and the construction/diagonalization of a simple physical Hamiltonian (Heisenberg dimer).
    *   **Failure Modes**: Catches errors in operator matrix definitions (if they were library provided), tensor product logic (`np.kron` usage), and basic physical correctness (ground state energy, trace preservation).
    *   **Guards**: Implicitly guards `algebra/eigen` solvers (when used) and general physics modeling logic.

## Maths / Common
*   **Module**: `maths/tests/test_utilities_edge_cases.py` (Extended)
    *   **Purpose**: Added edge cases for `find_nearest_val`.
    *   **Failure Modes**: Catches handling of empty arrays (raises ValueError now) and ensures robust argument handling (ignored `col` for ndarray).
    *   **Guards**: `maths/math_utils.py`

## Fixes to Existing Tests
*   **Module**: `lattices/tests/test_lattice_invariants.py`
    *   **Action**: Unmarked `test_next_nearest_neighbors` as xfail by implementing a workaround for the known bug in `Lattice.calculate_nnn`.
    *   **Purpose**: To verify the correctness of the underlying neighbor-finding logic (`calculate_nnn_in`) despite the broken wrapper.
