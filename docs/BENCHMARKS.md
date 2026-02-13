# General Python Benchmarks

This directory contains a lightweight benchmark suite for the `general_python` library, focusing on representative scientific workloads.

## Overview

The benchmarks measure the performance of key components:
*   **Algebra**: Krylov subspace methods (Lanczos and Arnoldi algorithms) on structured sparse matrices (Laplacian 1D/2D and Convection-Diffusion).
*   **Lattices**: Initialization and neighbor list generation for square lattices.
*   **Physics**: Construction and application of Heisenberg spin chain Hamiltonians (Dense and Sparse).

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
    *   **Workload**: Solving for the `k` smallest eigenvalues of a symmetric 1D Laplacian matrix of size `n` using `LanczosEigensolver`.
*   **Benchmark**: `Lanczos Laplacian 2D`
    *   **Metric**: Runtime (seconds) and Minimum Residual Norm.
    *   **Workload**: Solving for the `k` smallest eigenvalues of a symmetric 2D Laplacian matrix of size `Nx` x `Ny` using `LanczosEigensolver`. The matrix is constructed via Kronecker sum.
*   **Benchmark**: `Arnoldi Conv-Diff`
    *   **Metric**: Runtime (seconds) and Minimum Residual Norm.
    *   **Workload**: Solving for the `k` largest magnitude eigenvalues of a non-symmetric Convection-Diffusion matrix of size `n` using `ArnoldiEigensolver`.
    *   **Interpretation**: Measures the efficiency of the eigensolvers on different matrix structures (symmetric vs non-symmetric).

### Lattices
*   **Benchmark**: `Lattice Init 2D`
    *   **Metric**: Runtime (seconds) for initialization.
    *   **Workload**: Creating a `SquareLattice` of size `Lx` x `Ly` with Periodic (PBC) or Open (OBC) boundary conditions. This involves calculating coordinate arrays, neighbor lists, and reciprocal vectors.
    *   **Interpretation**: Measures the scaling of lattice setup. Note that `SquareLattice` allocates a full DFT matrix, so memory usage scales as `O(N^2)`, limiting the maximum feasible size.

### Physics
*   **Benchmark**: `Hamiltonian Build` / `Hamiltonian Apply` (Dense & Sparse)
    *   **Metric**: Runtime (seconds).
    *   **Workload**:
        1.  Constructing a Hamiltonian matrix for a Heisenberg spin chain of length `n_spins`.
        2.  Applying this Hamiltonian to a random state vector (Matrix-Vector multiplication).
    *   **Scaling**:
        *   Dense: Up to `n=12` (~4096 states) standard.
        *   Sparse: Up to `n=18` (~262k states) in heavy mode.
    *   **Interpretation**: Measures the overhead of constructing physical operators and the performance of applying them. Sparse implementation allows scaling to much larger Hilbert spaces.

## Variance

Performance may vary based on:
*   **CPU**: Single-core performance (for Python loops in construction) and multi-core performance (for BLAS/LAPACK operations in NumPy/SciPy/MKL).
*   **Memory**: Memory bandwidth affects large matrix operations.
*   **Background Load**: Run benchmarks on a quiet system for consistent results.
