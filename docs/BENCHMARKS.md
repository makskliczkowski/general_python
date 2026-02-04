# General Python Benchmarks

This directory contains a lightweight benchmark suite for the `general_python` library, focusing on representative scientific workloads.

## Overview

The benchmarks measure the performance of key components:
*   **Algebra**: Krylov subspace methods (Lanczos algorithm) on structured sparse matrices.
*   **Lattices**: Initialization and neighbor list generation for square lattices.
*   **Physics**: Construction and application of Heisenberg spin chain Hamiltonians.

## How to Run

To run the standard benchmarks (takes a few seconds):

```bash
python3 -m benchmarks.run
```

To run the heavy benchmarks (takes longer, scales to larger systems):

```bash
python3 -m benchmarks.run --heavy
```

### Machine Guidance

*   **Standard Mode**: Designed to run quickly (< 1 minute) on any modern laptop or CI environment.
*   **Heavy Mode**: May require more RAM (> 4GB) and compute time. Specifically, lattice benchmarks in heavy mode may allocate significant memory due to dense matrix operations in `SquareLattice`.

## What is Measured

### Algebra
*   **Benchmark**: `Lanczos Laplacian 1D`
*   **Metric**: Runtime (seconds) and Minimum Residual Norm (`||Av - λv||`).
*   **Workload**: Solving for the `k` smallest eigenvalues of a 1D Laplacian matrix of size `n`.
*   **Interpretation**: Measures the efficiency of the `LanczosEigensolver` and the underlying sparse matrix-vector multiplication. Lower runtime is better. Low residual (< 1e-10) indicates convergence.

### Lattices
*   **Benchmark**: `Lattice Init 2D`
*   **Metric**: Runtime (seconds) for initialization.
*   **Workload**: Creating a `SquareLattice` of size `Lx` x `Ly` with Periodic (PBC) or Open (OBC) boundary conditions. This involves calculating coordinate arrays, neighbor lists, and reciprocal vectors.
*   **Interpretation**: Measures the scaling of lattice setup. Note that `SquareLattice` allocates a full DFT matrix, so memory usage scales as `O(N^2)`, limiting the maximum feasible size.

### Physics
*   **Benchmark**: `Hamiltonian Build` / `Hamiltonian Apply`
*   **Metric**: Runtime (seconds).
*   **Workload**:
    1.  Constructing a dense Hamiltonian matrix for a Heisenberg spin chain of length `n_spins` (Hilbert space dimension `2^n`).
    2.  Applying this Hamiltonian to a random state vector (Matrix-Vector multiplication).
*   **Interpretation**: Measures the overhead of constructing physical operators and the raw performance of applying them. Construction scales exponentially with `n_spins`.

## Variance

Performance may vary based on:
*   **CPU**: Single-core performance (for Python loops) and multi-core performance (for BLAS/LAPACK operations in NumPy/SciPy).
*   **Memory**: Memory bandwidth affects large matrix operations.
*   **Background Load**: Run benchmarks on a quiet system for consistent results.
