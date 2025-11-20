# Solver Examples

This directory contains practical examples showing how to use the linear algebra solvers.
Location: `QES/general_python/algebra/examples/`

## Available Examples

### Basic Usage

- **example_cg_basic.py**       - Conjugate Gradient solver for symmetric positive-definite systems
- **example_minres_scipy.py**   - MINRES solver for symmetric indefinite systems (production-ready SciPy wrapper)
- **example_direct.py**         - Direct solvers (LU decomposition, pseudo-inverse)

### Advanced Features

- **example_gram_form.py**              - GRAM form solver (S, Sp) - the most common pattern in TDVP/NQS
- **example_with_preconditioners.py**   - Using preconditioners to accelerate convergence
- **example_solver_comparison.py**      - Comparing different solvers and preconditioners

### API Patterns

- **example_matvec_form.py**            - Matrix-free solvers using matvec functions
- **example_solver_factory.py**         - Using the solver factory for flexible solver selection

## Quick Start

All examples can be run directly:

```bash
python example_cg_basic.py
```

Each example is self-contained and includes:

- Mathematical problem description
- Code with detailed comments
- Output interpretation

## Solver Selection Guide

| Problem Type | Recommended Solver | Notes |
|-------------|-------------------|-------|
| SPD, small-medium     | CG | Fast, memory efficient |
| SPD, large            | CG + preconditioner | Jacobi or Cholesky |
| Symmetric indefinite  | MINRES (SciPy) | Production-ready |
| GRAM form (TDVP/NQS)  | CG with GRAM | Most common pattern |
| Non-symmetric, small  | Direct | Exact solution |
| Rank-deficient        | Pseudo-inverse | Handles singular systems |

**SPD** = Symmetric Positive-Definite

## Mathematical Background

### Conjugate Gradient (CG)

Solves $Ax = b$ where $A$ is symmetric positive-definite.

- Iterative method
- Optimal Krylov subspace method
- Convergence rate: $O(\sqrt{\kappa})$ where $\kappa$ is condition number

### MINRES

Solves $Ax = b$ where $A$ is symmetric (possibly indefinite).

- Minimum residual method
- More robust than CG for indefinite systems
- Uses Lanczos iteration

### GRAM Form

Special form for TDVP/NQS: Solve $(S^\dagger S)p = S^\dagger r$ for $p$.

- $S$ is typically overlap matrix or Jacobian
- Common in quantum state optimization
- Equivalent to solving least-squares problem

## See Also

- `../README.md` - Algebra package overview and module map
- `solvers/README.md` - Solver implementation details
- `../eigen/README.md` - Krylov eigensolver summary
- `PRECONDITIONERS.md` - Preconditioner documentation
- `test_tdvp_nqs_compat.py` - TDVP/NQS compatibility tests
