# Algebra Toolkit

The `general_python.algebra` package collects deterministic linear-algebra and random-matrix tools that form the numerical backbone of QES.  Every module is written to admit NumPy or JAX backends and focuses on well-defined mathematical problems.

## Core Problem Statements
- **Linear systems** — Solve $A x = b$ with optional left/right preconditioning $M^{-1} A x = M^{-1} b$ or $A M^{-1} y = b$, $x = M^{-1} y$.
- **Eigenproblems** — Approximate $(A - \lambda I) v = 0$ through Krylov bases $\mathcal{K}_k(A, v_0) = \mathrm{span}\{v_0, A v_0, \dots, A^{k-1} v_0\}$.
- **Matrix factorizations** — Compute Pfaffians and Hafnians of structured matrices:
  - Pfaffian for an antisymmetric $A \in \mathbb{C}^{2n \times 2n}$ satisfies $\mathrm{pf}(A)^2 = \det(A)$.
  - Hafnian for a symmetric $B \in \mathbb{C}^{2n \times 2n}$ is $\mathrm{haf}(B) = \sum_{M \in \mathcal{M}_{2n}} \prod_{(i,j) \in M} B_{i j}$ with the sum over perfect matchings $M$.
- **Random ensembles** — Sample matrices from Gaussian orthogonal/unitary ensembles (GOE/GUE), circular orthogonal/unitary ensembles (COE/CUE), etc., with well-defined Haar or Ginibre constructions.
- **Initial value problems** — Integrate $\dot{y}(t) = f(t, y)$ via explicit solver interfaces with adaptive backends.

## Package Layout
| Path | Mathematical focus | Notes |
|------|--------------------|-------|
| `linalg.py` | Dense basis transforms, tensor products, and inner/outer products. Implements $U A U^\dagger$ basis changes and Kronecker products consistent with backend dtype promotion. | Primary entry-point for deterministic dense operations. |
| `linalg_sparse.py` | Sparse Kronecker products and identity builders compatible with SciPy or `jax.experimental.sparse.BCOO`. | Guarantees sparsity preservation and index promotion in $A \otimes B$. |
| `ode.py` | Abstract initial-value solvers $y' = f(t, y)$ with NumPy/JAX stepping and SciPy integration fallbacks. | Normalizes RHS signatures and supports JIT compilation when available. |
| `solver.py` | Base classes for linear solvers, including residual bookkeeping and backend-aware matvec wrappers. | Emits `SolverResult` with $\|b - A x\|_2$ tracking. |
| `solvers/` | Concrete Krylov and direct algorithms: CG, MINRES, MINRES-QLP, GMRES, and pseudo-inverse evaluators. | Each solver minimizes a rigorously specified norm (e.g., MINRES minimizes $\|r_k\|_2$ at each iterate). See `solvers/README.md`. |
| `eigen/` | Arnoldi, Lanczos, and block-Lanczos eigensolvers plus result containers. | Targets Hermitian and general matrices; see `eigen/README.md` and `IMPLEMENTATION_SUMMARY.md` for recurrences. |
| `preconditioners.py` & `PRECONDITIONERS.md` | Implements $M^{-1}$ operators (identity, diagonal/Jacobi, incomplete Cholesky) with proofs of SPD preservation. | Used by both solvers and TDVP routines. |
| `ran_matrices.py`, `ran_wrapper.py` | Generates GOE/GUE/COE/CUE samples via QR-normalized Ginibre matrices and registers reproducible RNG factories. | Matches Haar-distribution normalization $\mathbb{E}[U_{i j} \overline{U_{k \ell}}] = \delta_{i k}\delta_{j \ell}/n$. |
| `utilities/` | Pfaffian and Hafnian implementations for even-dimensional structured matrices using scaling-stable decompositions. | Overview in `utilities/README.md`; `pfaffian_jax.py` mirrors the NumPy algorithm with JAX primitives. |
| `utils.py` | Backend negotiation (`get_backend`, `maybe_jit`) and array typing used across the package. | Centralizes NumPy/JAX feature detection. |

## Backend Guarantees
- Functions accept `backend="default"`, resolving to JAX if available, else NumPy.  Explicit `backend="np"` or `"jnp"` yields deterministic behavior.
- All public solvers accept matrix-free callables $v \mapsto A v$ and optional preconditioner callbacks $v \mapsto M^{-1} v$, enforcing shape compatibility before iteration.
- Random generators expose seedable registries so that $\mathbb{E}[A^\dagger A] = I$ holds across backends.

## Cross-References
- Detailed solver convergence theory: `solvers/README.md`.
- Eigenvalue algorithm roadmap: `eigen/README.md` (with derivations in `eigen/IMPLEMENTATION_SUMMARY.md`).
- Pfaffian/Hafnian numerics: `utilities/README.md`.
- Worked scripts demonstrating solver choices: `examples/README.md`.
- Preconditioner catalogue and proofs: `PRECONDITIONERS.md`.

## Copyright
Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.
