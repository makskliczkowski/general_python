# Linear Algebra Solvers

Comprehensive collection of iterative and direct linear system solvers for solving:

$$
Ax = b
$$

where $A$ is a matrix (dense or matrix-free), $b$ is the right-hand side vector, and $x$ is the solution vector.

## Table of Contents

1. [Overview](#overview)
2. [Available Solvers](#available-solvers)
3. [Mathematical Background](#mathematical-background)
4. [Usage Examples](#usage-examples)
5. [Preconditioners](#preconditioners)
6. [Backend Support](#backend-support)
7. [API Reference](#api-reference)

---

## Overview

This module provides both **direct** and **iterative** solvers for linear systems. Iterative solvers are particularly useful for large-scale problems where forming and factoring the full matrix is computationally expensive or infeasible.

### Key Features

- **Multiple solver algorithms**    : CG, MINRES, MINRES-QLP, Direct, Pseudo-Inverse
- **Dual backend support**          : NumPy and JAX for CPU/GPU acceleration
- **Preconditioner support**        : Identity, Jacobi, Incomplete Cholesky
- **Matrix-free interface**         : Solve without forming explicit matrices
- **Fisher/Gram matrix support**    : Efficient handling of $S = \Delta O^\dagger \Delta O$ forms

---

## Available Solvers

| Solver | Matrix Type | Best For | Convergence |
|--------|-------------|----------|-------------|
| **CG** | Symmetric Positive Definite  | Well-conditioned SPD systems | Fast for SPD |
| **MINRES** | Symmetric Indefinite     | Ill-conditioned symmetric systems | Moderate |
| **MINRES-QLP**                        | Symmetric (possibly singular) | Singular/badly scaled systems | Robust |
| **Direct**                            | General | Small-medium systems, high accuracy | Exact |
| **Pseudo-Inverse**                    | General (possibly singular) | Least-squares, minimum norm | Stable |

### Solver Selection Guide

```markdown
Is your matrix symmetric?
├─ YES: Is it positive definite?
│   ├─ YES: Use CG (fastest)
│   │   └─ CgSolver (native) or CgSolverScipy (wrapper)
│   └─ NO: Is it singular or badly scaled?
│       ├─ YES: Use MINRES-QLP (most robust) 🚧
│       └─ NO: Use MINRES
│           └─ MinresSolverScipy (recommended) (ok)
└─ NO: Use Direct or iterative GMRES (if available)
```

**Implementation Status**:

- (ok) **Stable**         : Production-ready, well-tested
- 🚧 **WIP**            : Work-in-progress, use alternative
- 🔍 **Needs Review**   : Requires validation/fixes

---

## Mathematical Background

### Conjugate Gradient (CG)

**Problem**     : Solve $Ax = b$ where $A$ is symmetric positive definite (SPD).
**Algorithm**   : CG minimizes the quadratic function:

$$
\phi(x) = \frac{1}{2}x^T A x - b^T x
$$

**Key Properties**:

- Converges in at most $n$ iterations (exact arithmetic)
- Each iteration requires one matrix-vector product
- Preconditioned version: $M^{-1}Ax = M^{-1}b$ where $M \approx A$

**Iteration**:

```math
Given x₀, compute r₀ = b - Ax₀
Set p₀ = r₀
For k = 0, 1, 2, ...
    \alpha_k       = (r_k ᵀr_k ) / (p_k ᵀAp_k )
    x_k ₊_1     = x_k  + \alpha_k p_k 
    r_k ₊_1     = r_k  - \alpha_k Ap_k 
    \beta_k       = (r_k ₊_1 ᵀr_k ₊_1 ) / (r_k ᵀr_k )
    p_k ₊_1     = r_k ₊_1  + \beta_k p_k 
```

**Convergence**: For SPD matrix $A$ with condition number $\kappa(A)$:

$$
\frac{\|e_k\|_A}{\|e_0\|_A} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k
$$

---

### MINRES

**Problem**     : Solve $Ax = b$ where $A$ is symmetric (possibly indefinite).
**Algorithm**   : Uses Lanczos process to build orthonormal basis $V_k$ for Krylov subspace:

$$
\mathcal{K}_k(A, r_0) = \text{span}\{r_0, Ar_0, A^2r_0, \ldots, A^{k-1}r_0\}
$$

**Lanczos Recurrence**:

$$
AV_k = V_k T_k + \beta_k v_{k+1} e_k^T
$$

where $T_k$ is a symmetric tridiagonal matrix:

$$
T_k = \begin{bmatrix}
\alpha_1 & \beta_1 &  &  \\
\beta_1 & \alpha_2 & \beta_2 &  \\
 & \beta_2 & \alpha_3 & \ddots \\
 &  & \ddots & \ddots
\end{bmatrix}
$$

**QR Factorization**: Apply Givens rotations to $T_k$:

$$
T_k = Q_k R_k
$$

**Minimization**: At iteration $k$, minimize:

$$
\|b - Ax_k\|_2 = \min_{x \in x_0 + \mathcal{K}_k} \|b - Ax\|_2
$$

**Properties**:

- Works for symmetric indefinite systems
- Minimizes residual norm at each iteration
- One matrix-vector product per iteration

---

### MINRES-QLP

**Problem**: Solve $Ax = b$ where $A$ is symmetric, possibly singular or badly scaled.

**Algorithm**: Extends MINRES with additional QLP factorization:

$$
T_k = Q_k L_k P_k^T
$$

where:

- $Q_k$: Orthogonal (left Givens rotations)
- $L_k$: Lower bidiagonal
- $P_k$: Orthogonal (right Givens rotations)

**Advantages over MINRES**:

- Computes minimum-length solution for singular systems
- More stable for ill-conditioned problems
- Better handles near-zero pivots

**Solution Update**:

$$
x_k = x_0 + V_k P_k^T t_k
$$

where $L_k t_k = \beta_1 Q_k e_1$.

**Convergence Flags**:

- `1`       : Converged on relative residual $\|r_k\| / \|b\| \leq$ tol
- `2`       : Converged on $\|Ar_k\| / (\|A\| \|r_k\|) \leq$ tol
- `3-4`     : Absolute residual convergence
- `5`       : Found eigenvector
- `6-7`     : Exceeded norm/condition limits
- `8`       : Max iterations
- `9`       : System appears singular

---

### Direct Solver

**Algorithm**: Uses backend's direct solver (LU decomposition for general, Cholesky for SPD).

**NumPy**: `numpy.linalg.solve(A, b)`
**JAX**: `jax.numpy.linalg.solve(A, b)`

**Complexity**: $O(n^3)$ for dense matrices, $O(n^{1.5})$ to $O(n^{2.4})$ for sparse.

**Use when**:

- Small to medium systems ($n < 10,000$)
- High accuracy required
- Matrix is well-conditioned
- Multiple solves with same matrix

---

### Pseudo-Inverse

**Algorithm**: Computes Moore-Penrose pseudo-inverse using SVD:

$$
A^+ = V\Sigma^+U^T
$$

where $\Sigma^+$ has reciprocal of non-zero singular values.

**Solution**: $x = A^+b$ minimizes both:

1. Residual norm: $\|Ax - b\|_2$
2. Solution norm: $\|x\|_2$ (among all minimizers)

**Use when**:

- System is singular or rank-deficient
- Least-squares solution needed
- Minimum norm solution required
- NQS time evolution

---

## Usage Examples

### Example 1: CG for SPD System

```python
from general_python.algebra.solvers import CgSolver
import numpy as np

# Create SPD matrix
A       = np.array([[4, 1], [1, 3]])
b       = np.array([1, 2])

# Create solver
solver = CgSolver(backend='numpy', eps=1e-8, maxiter=1000)

# Set matrix
solver.set_a(A)

# Solve
result  = solver.solve(b, x0=np.zeros(2))

print(f"Solution: {result.x}")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Residual: {result.residual_norm}")
```

### Example 2: MINRES with Preconditioner

```python
from general_python.algebra.solvers import MinresSolver
from general_python.algebra.preconditioners import JacobiPreconditioner
import numpy as np

# Create symmetric indefinite matrix
A       = np.array([[1, 2], [2, -1]])
b       = np.array([1, 1])

# Create Jacobi preconditioner
precond = JacobiPreconditioner()
precond.set_a(A)

# Create solver
solver  = MinresSolver(backend='numpy', eps=1e-8)
solver.set_a(A)

# Solve with preconditioning
result  = solver.solve(b, preconditioner=precond)
```

### Example 2b: SciPy MINRES via the solver factory

```python
from general_python.algebra.solvers import choose_solver, SolverType
import numpy as np

# Symmetric (possibly indefinite) system
A       = np.array([[2,1,0],[1,0,1],[0,1,2]], dtype=float)
b       = np.array([1.,2.,3.])

# Create SciPy-backed MINRES solver with consistent API
solver  = choose_solver(SolverType.SCIPY_MINRES, backend='numpy', eps=1e-8, maxiter=200)
solver.set_a(A)
res     = solver.solve(b)
print(res.converged, res.iterations)
```

### Example 3: Matrix-Free CG

```python
from general_python.algebra.solvers import CgSolver
import numpy as np

# Define matrix-vector product (e.g., for Laplacian)
def matvec(v):
    n = len(v)
    result = 2 * v
    result[:-1] -= v[1:]
    result[1:] -= v[:-1]
    return result

# Create solver with matrix-free operator
n = 100
solver = CgSolver(backend='numpy')
solver.set_matvec(matvec, n)

# Solve
b = np.ones(n)
result = solver.solve(b)
```

### Example 4: Fisher Matrix (Gram Form)

```python
from general_python.algebra.solvers import CgSolver
import numpy as np

# Fisher matrix S = delta O\dag delta O
# delta O: (n_samples, n_params) matrix of derivatives
n_samples, n_params = 1000, 50
DeltaO = np.random.randn(n_samples, n_params) + \
         1j * np.random.randn(n_samples, n_params)

# Create solver
solver = CgSolver(backend='numpy', is_gram=True)
solver.set_s(DeltaO)  # Sets S = delta O\dag delta O implicitly

# Solve Sx = b efficiently without forming S
b = np.random.randn(n_params) + 1j * np.random.randn(n_params)
result = solver.solve(b)
```

### Example 5: Singular System with MINRES-QLP

```python
from general_python.algebra.solvers import MinresQLPSolver
import numpy as np

# Create singular matrix
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [0, 0, 1]])
b = np.array([1, 2, 1])

# MINRES-QLP finds minimum-length solution
solver = MinresQLPSolver(backend='numpy', eps=1e-10)
solver.set_a(A)

result = solver.solve(b)
print(f"Minimum-length solution: {result.x}")
print(f"Convergence flag: {result.converged}")
```

### Example 6: JAX Backend with JIT

```python
from general_python.algebra.solvers import CgSolver
import jax.numpy as jnp
import jax

# Create solver with JAX backend
solver = CgSolver(backend='jax', eps=1e-8)

# Define matrix-vector product
@jax.jit
def matvec(v):
    A = jnp.array([[4., 1.], [1., 3.]])
    return A @ v

solver.set_matvec(matvec, 2)

# Solve (can be JIT-compiled)
b = jnp.array([1., 2.])
result = solver.solve(b)
```

---

## Preconditioners

Preconditioners transform the linear system to improve convergence:

$$
M^{-1}Ax = M^{-1}b
$$

where $M \approx A$ but $M^{-1}r$ is cheap to compute.

### Available Preconditioners

#### Identity (No Preconditioning)

```python
from general_python.algebra.preconditioners import IdentityPreconditioner
precond = IdentityPreconditioner()
```

#### Jacobi (Diagonal Scaling)

```python
from general_python.algebra.preconditioners import JacobiPreconditioner
precond = JacobiPreconditioner()
precond.set_a(A)  # M = diag(A)
```

#### Incomplete Cholesky

```python
from general_python.algebra.preconditioners import IncompleteCholeskyPreconditioner
precond = IncompleteCholeskyPreconditioner()
precond.set_a(A)  # M ≈ LL^T (incomplete)
```

### Choosing a Preconditioner

| Matrix Property | Recommended Preconditioner |
|----------------|---------------------------|
| Well-conditioned | Identity (none) |
| Diagonally dominant | Jacobi |
| SPD, large condition number | Incomplete Cholesky |
| General symmetric | Incomplete Cholesky/SSOR |

---

## Backend Support

### NumPy Backend

- Standard CPU computations
- Uses optimized BLAS/LAPACK
- Compatible with SciPy sparse matrices

```python
solver = CgSolver(backend='numpy')
```

### JAX Backend

- GPU/TPU acceleration
- Automatic differentiation support
- JIT compilation for performance

```python
solver = CgSolver(backend='jax')
```

### Automatic Selection

```python
solver = CgSolver(backend='default')  # Uses JAX if available, else NumPy
```

---

## API Reference

### Base Solver Class

```python
class Solver(ABC):
    def __init__(self, backend='default', eps=1e-8, maxiter=1000, ...):
        """Initialize solver with common parameters."""
        
    def set_a(self, a: Array):
        """Set matrix A."""
        
    def set_s(self, s: Array, sp: Optional[Array] = None):
        """Set Fisher/Gram matrix S = Sp @ S."""
        
    def set_matvec(self, func: Callable, size: int):
        """Set matrix-free operator."""
        
    @abstractmethod
    def solve(self, b: Array, x0: Optional[Array] = None, **kwargs) -> SolverResult:
        """Solve Ax = b."""
```

### SolverResult

```python
class SolverResult(NamedTuple):
    x: Array                     # Solution vector
    converged: bool              # Convergence flag
    iterations: int              # Number of iterations
    residual_norm: float         # Final ||r|| = ||b - Ax||
```

### Factory Function

```python
def choose_solver(solver_id, **kwargs) -> Solver:
    """
    Create solver from identifier.
    
    Args:
        solver_id: 'cg', 'minres', 'minres-qlp', 'direct', or SolverType enum
        **kwargs: Solver-specific arguments
        
    Returns:
        Configured solver instance
    """
```

---

## Performance Tips

1. **Use appropriate solver for matrix type**:
   - SPD -> CG (fastest)
   - Symmetric indefinite -> MINRES
   - Singular -> MINRES-QLP or Pseudo-Inverse

2. **Enable preconditioning** for ill-conditioned systems:

   ```python
   precond = JacobiPreconditioner()
   result = solver.solve(b, preconditioner=precond)
   ```

3. **Use JAX backend for GPU**:

   ```python
   solver = CgSolver(backend='jax')  # Automatic GPU if available
   ```

4. **Matrix-free for large systems**:

   ```python
   solver.set_matvec(matvec_func, n)  # Avoid forming full matrix
   ```

5. **Tune tolerance and max iterations**:

   ```python
   result = solver.solve(b, tol=1e-10, maxiter=5000)
   ```

---

## References

### Primary References

1. **Hestenes & Stiefel (1952)**: "Methods of Conjugate Gradients for Solving Linear Systems"
   - Original CG algorithm

2. **Paige & Saunders (1975)**: "Solution of Sparse Indefinite Systems of Linear Equations"
   - MINRES algorithm, SIAM J. Numer. Anal., 12(4), 617-629

3. **Choi, Paige & Saunders (2011)**: "MINRES-QLP: A Krylov Subspace Method for Indefinite or Singular Symmetric Systems"
   - MINRES-QLP algorithm, SIAM J. Sci. Comput., 33(4), 1810-1836

4. **Saad (2003)**: "Iterative Methods for Sparse Linear Systems" (2nd ed.), SIAM
   - Comprehensive reference for iterative methods

5. **Shewchuk (1994)**: "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
   - Excellent CG tutorial

### Additional Resources

- [Matrix Computations (Golub & Van Loan)](https://www.cs.cornell.edu/cv/GVL4/golubandvanloan.htm)
- [Templates for the Solution of Linear Systems (Barrett et al.)](https://www.netlib.org/templates/)
- [SciPy sparse.linalg documentation](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)

---

## License

MIT License - See main repository LICENSE file.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open an issue on the GitHub repository.
