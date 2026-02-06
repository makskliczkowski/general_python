# Current Test Inventory

This document lists the existing test modules in the repository, their purpose, runtime classification, and determinism status.

## Algebra

### Eigenvalue Solvers
*   **Module**: `algebra/eigen/tests/test_block_lanczos_comprehensive.py`
    *   **Validates**: `BlockLanczosEigensolver` correctness, complex Hermitian matrices, generalized eigenvalue problems, sparse matrix support, orthogonality, edge cases (small matrices), operator interface, restarting mechanism.
    *   **Class**: Integration / Slow
    *   **Deterministic**: Yes (uses fixed seeds).
*   **Module**: `algebra/eigen/tests/test_lanczos.py`
    *   **Validates**: `LanczosEigensolver` basic functionality, dense/sparse backends, extremal eigenvalues selection, convergence flags, max iteration limits, initial vector injection.
    *   **Class**: Unit
    *   **Deterministic**: Yes.

### Linear Solvers
*   **Module**: `algebra/solvers/tests/test_minres_qlp.py`
    *   **Validates**: `MinresQLPSolver` on small systems, random SPD matrices, sparse inputs, max iteration limits, singular systems behavior.
    *   **Class**: Unit
    *   **Deterministic**: Yes (uses fixed seeds).

### General / Sanity
*   **Module**: `algebra/tests/test_algebra.py`
    *   **Validates**: Basic algebra utilities (e.g., `random_hermitian`).
    *   **Class**: Unit
    *   **Deterministic**: Yes.
*   **Module**: `algebra/tests/test_solvers_sanity.py`
    *   **Validates**: High-level correctness of solvers (Lanczos, MINRES) on known physical problems (2D Laplacian), dtype coverage (float32/64, complex), and deterministic behavior across runs.
    *   **Class**: Integration
    *   **Deterministic**: Yes.

### Pfaffian / JAX
*   **Module**: `algebra/tests/test_pfaffian_jax_ops.py`
    *   **Validates**: JAX implementations of Pfaffian updates and Sherman-Morrison formulas.
    *   **Class**: Unit (requires JAX)
    *   **Deterministic**: Yes.
*   **Module**: `algebra/utilities/tests/test_pfaffian.py`
    *   **Validates**: Reference Pfaffian implementation (Pf^2 = det).
    *   **Class**: Unit
    *   **Deterministic**: Yes.

## Lattices
*   **Module**: `lattices/tests/test_lattice_invariants.py`
    *   **Validates**: `SquareLattice` neighbor finding (PBC/OBC), coordinate calculations, small lattice edge cases.
    *   **Class**: Unit
    *   **Deterministic**: Yes.
    *   **Note**: Contains a known failing test (`test_next_nearest_neighbors`) due to a bug in `Lattice.calculate_nnn`.

## Physics
*   **Module**: `physics/tests/test_operator_invariants.py`
    *   **Validates**: Pauli matrices algebra, fermion operator anti-commutation, operator Hermiticity, traces, norms, and simple Hamiltonian construction.
    *   **Class**: Unit
    *   **Deterministic**: Yes.

## Maths / Common
*   **Module**: `maths/tests/test_utilities_edge_cases.py`
    *   **Validates**: Math utilities (`find_nearest_val`, `mod_round`), statistical helpers (`Statistics`, `Fraction`).
    *   **Class**: Unit
    *   **Deterministic**: Yes.

## General / Infrastructure
*   **Module**: `tests/test_activations.py`
    *   **Validates**: Neural network activation functions (log_cosh), holomorphicity checks, gradients.
    *   **Class**: Unit
    *   **Deterministic**: Yes.
*   **Module**: `tests/test_imports.py`
    *   **Validates**: Package structure, importability of submodules, versioning.
    *   **Class**: Unit
    *   **Deterministic**: Yes.
*   **Module**: `tests/test_lazy_imports.py`
    *   **Validates**: Lazy loading mechanism functionality.
    *   **Class**: Unit
    *   **Deterministic**: Yes.
*   **Module**: `test_documentation.py`
    *   **Validates**: Presence of docstrings.
    *   **Class**: Meta
    *   **Deterministic**: Yes.
