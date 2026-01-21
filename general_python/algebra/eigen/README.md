# Eigenvalue Algorithms

The `general_python.algebra.eigen` package implements Krylov-based eigensolvers that approximate extremal eigenpairs of large matrices without forming dense factorizations.  Every routine is backend-aware (NumPy or JAX) and accepts matrix-free linear operators.

## Problem Statements

- **Hermitian spectra**: Given $H = H^\dagger$, construct Lanczos bases $\mathcal{K}_k(H, v_0)$ with short-term recurrences $H v_j = \beta_{j-1} v_{j-1} + \alpha_j v_j + \beta_j v_{j+1}$, leading to tridiagonal $T_k$ whose Ritz pairs $(\theta_i, y_i)$ approximate $(\lambda_i, x_i)$ via $x_i \approx V_k y_i$.
- **General (non-Hermitian) spectra**: The Arnoldi iteration builds an orthonormal basis obeying $A V_k = V_k H_k + h_{k+1,k} v_{k+1} e_k^\top$, where $H_k$ is upper Hessenberg.  Ritz vectors $x_i \approx V_k y_i$ deliver approximate eigenpairs and residual $\|A x_i - \theta_i x_i\|_2 = |h_{k+1,k}|\, |e_k^\top y_i|$.
- **Block Krylov spaces**: Block-Lanczos propagates multiple start vectors $W \in \mathbb{C}^{n \times b}$ to capture clustered eigenvalues and degeneracies.  The block recurrence $A Q_j = Q_{j-1} B_j^\top + Q_j A_j + Q_{j+1} B_{j+1}$ yields block tridiagonal projected matrices solved in `block_lanczos.py`.

## Module Map

| Path | Algorithm | Notes |
|------|-----------|-------|
| `arnoldi.py`        | Standard Arnoldi iteration and residual monitoring for non-Hermitian matrices. | Exposes explicit `arnoldi_step` and Ritz extraction helpers. |
| `lanczos.py`        | Single-vector Lanczos for Hermitian matrices. | Implements reorthogonalisation toggles and Ritz filtering. |
| `block_lanczos.py`  | Block-Lanczos for structured degeneracies. | Supports orthonormal block basis construction and deflation. |
| `exact.py`          | Dense fallback using backend eigenvalue routines. | Handy for reference spectra and small validation tests. |
| `factory.py`        | Dispatch helpers returning solver callables according to matrix properties. | Integrates with `algebra.solver.Solver`. |
| `result.py`         | Data container storing Ritz values, vectors, and residual history. | Shared by all eigensolver frontends. |
| `tests/`            | Regression suites for Lanczos/Arnoldi implementations. | See `tests/test_lanczos.py`, `tests/debug_block_tridiag.py`. |

## Residual Control

- Ritz residual norms obey $\|A V_k y - \theta V_k y\|_2 = |h_{k+1,k}| |e_k^\top y|$ (Arnoldi) or $\beta_k |e_k^\top y|$ (Lanczos).  Stopping tolerances in each solver compare this value against user-provided absolute/relative thresholds.
- Reorthogonalisation strategies: selective orthogonalisation is available via callbacks; see `lanczos.py` documentation strings.
- Block algorithms compute Frobenius norms of block residuals $R_k = A Q_k - Q_k T_k - Q_{k+1} B_{k+1}$ and guard against loss of orthogonality with QR refreshes.

## Backend Behaviour

- All iterative routines consume a callable `matvec(v)` and optional `dtype`, ensuring compatibility with sparse matrices or JIT-compiled functions.
- When `backend="jax"`, generated bases live in device memory and can be used inside `jax.jit` compiled functions provided the basis size is static.

## Cross-References

- Algorithmic derivations and recurrence identities: `IMPLEMENTATION_SUMMARY.md`.
- Linear solver interoperability (e.g., MINRES warm starts): `../solvers/README.md`.
- Example pipelines illustrating spectral approximations: `../examples/README.md`.

## Copyright

Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.
