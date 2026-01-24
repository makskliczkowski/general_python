# Algebra Module

The `algebra` module provides a comprehensive suite of tools for linear algebra, specifically designed to support both **NumPy** and **JAX** backends seamlessly. It is optimized for scientific computing tasks, including solving large sparse linear systems and eigenvalue problems.

## Key Features

### 1. Backend Agnostic Operations
- Automatically detects if JAX is available and prefers it for performance.
- Provides a unified interface through `general_python.algebra.utils` to access backend-specific functions (e.g., `xp.sin`, `xp.linalg.norm`).

### 2. Solvers (`algebra.solvers`)
A robust factory pattern (`choose_solver`) allows easy access to various iterative and direct solvers:
- **Krylov Subspace Methods**: Conjugate Gradient (CG), MINRES, MINRES-QLP.
- **Direct Solvers**: wrappers around `scipy.linalg.solve` and JAX equivalents.
- **Eigenvalue Solvers**: Interfaces for Arnoldi/Lanczos methods (via backend).

### 3. Preconditioners (`algebra.preconditioners`)
- **ILU**: Incomplete LU factorization (NumPy/SciPy only).
- **LinearOperator** support for custom preconditioners.

### 4. Random Matrices (`algebra.ran_matrices`)
- Utilities to generate random symmetric, Hermitian, or sparse matrices for testing and simulation.

## Usage Example

```python
from general_python.algebra.solvers import choose_solver, SolverType
import numpy as np

# Create a random symmetric matrix
A = np.random.rand(100, 100)
A = A + A.T + 100 * np.eye(100)  # Make it diagonally dominant (positive definite)
b = np.random.rand(100)

# Use CG solver
solver = choose_solver(SolverType.CG, A=A)
result = solver.solve(b)

print(f"Converged: {result.success}")
print(f"Solution norm: {np.linalg.norm(result.x)}")
```
