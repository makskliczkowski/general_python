# Preconditioners for Iterative Solvers

Comprehensive guide to preconditioners for accelerating convergence of iterative linear system solvers.

## Table of Contents

1. [What are Preconditioners?](#what-are-preconditioners)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Available Preconditioners](#available-preconditioners)
4. [Usage Guide](#usage-guide)
5. [Performance Comparison](#performance-comparison)
6. [Implementation Details](#implementation-details)

---

## What are Preconditioners?

A **preconditioner** $M$ transforms a linear system $Ax = b$ into an equivalent system with better numerical properties, leading to faster convergence of iterative solvers.

### Left Preconditioning

$$
M^{-1}Ax = M^{-1}b
$$

The transformed system has the same solution but (ideally) better spectral properties.

### Right Preconditioning

$$
AM^{-1}y = b, \quad x = M^{-1}y
$$

### Split Preconditioning

$$
M_L^{-1}AM_R^{-1}y = M_L^{-1}b, \quad x = M_R^{-1}y
$$

---

## Mathematical Foundation

### Why Precondition?

The convergence rate of iterative methods depends on the **condition number**:

$$
\kappa(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}
$$

**Goal**: Find $M$ such that $\kappa(M^{-1}A) \ll \kappa(A)$.

### Ideal Preconditioner Properties

1. **Close approximation**: $M \approx A$
2. **Easy to apply**: $M^{-1}r$ is cheap to compute
3. **Improved spectrum**: $M^{-1}A$ has clustered eigenvalues near 1

### Convergence Improvement

For Conjugate Gradient with preconditioner $M$:

$$
\frac{\|e_k\|_M}{\|e_0\|_M} \leq 2\left(\frac{\sqrt{\kappa(M^{-1}A)}-1}{\sqrt{\kappa(M^{-1}A)}+1}\right)^k
$$

**Example**: If $\kappa(A) = 10^6$ and $M$ reduces it to $\kappa(M^{-1}A) = 10^2$:

- Without preconditioning:  ~1000 iterations
- With preconditioning:     ~10 iterations

---

## Available Preconditioners

### 1. Identity (No Preconditioning)

**Definition**: $M = I$

**Application**: $M^{-1}r = r$

**When to use**:

- Well-conditioned systems ($\kappa(A) < 100$)
- Testing and debugging
- Baseline comparison

**Pros**: No overhead, simple
**Cons**: No improvement in convergence

**Example**:

```python
from general_python.algebra.preconditioners import IdentityPreconditioner
precond = IdentityPreconditioner()
```

---

### 2. Jacobi (Diagonal Scaling)

**Definition**: $M = \text{diag}(A)$

**Application**:

$$
M^{-1}r = \begin{bmatrix} r_1/a_{11} \\ r_2/a_{22} \\ \vdots \\ r_n/a_{nn} \end{bmatrix}
$$

**When to use**:

- Diagonally dominant matrices
- Well-scaled problems
- First attempt at preconditioning

**Complexity**:

- Setup: $O(n)$
- Apply: $O(n)$

**For Gram matrices** $S = \Delta O^\dagger \Delta O$:

$$
M^{-1} = \text{diag}\left(\frac{1}{\|\Delta O_{\cdot,1}\|^2}, \ldots, \frac{1}{\|\Delta O_{\cdot,n}\|^2}\right)
$$

**Example**:

```python
from general_python.algebra.preconditioners import JacobiPreconditioner
import numpy as np

# Standard matrix
A       = np.array([[4, 1], [1, 3]])
precond = JacobiPreconditioner()
precond.set_a(A)  # M = diag([4, 3])

# Gram matrix form
DeltaO          = np.random.randn(1000, 50)             # (samples, params)
precond_gram    = JacobiPreconditioner(is_gram=True)    # Hello
precond_gram.set_s(DeltaO)                              # Efficient: doesn't form S
```

**Pros**:

- Very cheap (both setup and apply)
- Parallelizable
- Often effective for diagonally dominant matrices

**Cons**:

- Only accounts for diagonal
- Ineffective for strong off-diagonal coupling

---

### 3. Incomplete Cholesky

**Definition**: For SPD matrix $A$, find sparse lower triangular $L$ such that:

$$
A \approx LL^T
$$

with sparsity pattern matching (or similar to) $A$.

**Application**: Solve $LL^T z = r$ via forward/backward substitution

**When to use**:

- Symmetric positive definite matrices
- Strong off-diagonal coupling
- Sparse matrices

**Complexity**:

- Setup: $O(\text{nnz}(L))$ where $\text{nnz}$ = non-zeros
- Apply: $O(\text{nnz}(L))$ per solve

**Algorithm** (IC(0) - no fill-in):

```markdown
L = lower_triangular_part(A)
For k = 1 to n:
    L(k,k) = sqrt(L(k,k))
    For i = k+1 to n where L(i,k) \neq  0:
        L(i,k) = L(i,k) / L(k,k)
        For j = k+1 to i where L(i,j) \neq  0:
            L(i,j) = L(i,j) - L(i,k) * L(j,k)
```

**Example**:

```python
from general_python.algebra.preconditioners import IncompleteCholeskyPreconditioner
import numpy as np

# Create SPD matrix
A       = np.array([[4, 1, 0],
                    [1, 3, 1],
                    [0, 1, 2]])

precond = IncompleteCholeskyPreconditioner()
precond.set_a(A, sigma=0.01)  # Add regularization if needed

# For Gram form: S = Sp @ S
precond_gram = IncompleteCholeskyPreconditioner(is_gram=True)
precond_gram.set_s(DeltaO, sigma=0.01)
```

**Pros**:

- Exploits structure beyond diagonal
- Very effective for elliptic PDEs
- Natural for SPD systems

**Cons**:

- Requires Cholesky-compatible matrix
- May fail for ill-conditioned problems
- Setup cost higher than Jacobi

---

### 4. Symmetric Successive Over-Relaxation (SSOR)

**Definition**: Combines forward and backward Gauss-Seidel sweeps.

$$
M = (D + \omega L)D^{-1}(D + \omega U)
$$

where:

- $D = \text{diag}(A)$
- $L$ = strict lower triangular part of $A$
- $U$ = strict upper triangular part of $A$
- $\omega \in (0, 2)$ is relaxation parameter

**When to use**:

- Symmetric matrices (possibly indefinite)
- Alternative to Incomplete Cholesky
- $\omega \approx 1$ often works well

**Example**:

```python
from general_python.algebra.preconditioners import SSORPreconditioner

precond = SSORPreconditioner(omega=1.0)
precond.set_a(A)
```

---

## Usage Guide

### Basic Usage

```python
from general_python.algebra.solvers import CgSolver
from general_python.algebra.preconditioners import JacobiPreconditioner
import numpy as np

# Create matrix and RHS
A       = np.random.randn(100, 100)
A       = A @ A.T  # Make SPD
b       = np.random.randn(100)

# Setup preconditioner
precond = JacobiPreconditioner()
precond.set_a(A)

# Create solver
solver  = CgSolver(backend='numpy', eps=1e-8)
solver.set_a(A)

# Solve with preconditioning
result  = solver.solve(b, preconditioner=precond)

print(f"Converged: {result.converged}, Iterations: {result.iterations}")
```

### With Regularization

```python
# Add regularization: M ≈ (A + \sigma I)
sigma   = 0.01
precond = JacobiPreconditioner()
precond.set_a(A, sigma=sigma)

# Or set after initialization
precond.set_sigma(sigma)
```

### Gram Matrix Form

For Neural Quantum States with Fisher matrix $S = \Delta O^\dagger \Delta O$:

```python
from general_python.algebra.solvers import CgSolver
from general_python.algebra.preconditioners import JacobiPreconditioner
import numpy as np

# Gradient samples: (n_samples, n_params)
DeltaO  = np.random.randn(1000, 100) + 1j*np.random.randn(1000, 100)
force   = np.random.randn(100) + 1j*np.random.randn(100)

# Setup without forming S explicitly
precond = JacobiPreconditioner(is_gram=True)
precond.set_s(DeltaO)

solver  = CgSolver(backend='numpy', is_gram=True)
solver.set_s(DeltaO)

# Solve (S + \sigma I)x = force
result = solver.solve(force, preconditioner=precond)
```

### Backend Selection

```python
# NumPy backend
precond_np      = JacobiPreconditioner(backend='numpy')

# JAX backend (GPU-compatible)
precond_jax     = JacobiPreconditioner(backend='jax')

# Auto-select
precond_auto    = JacobiPreconditioner(backend='default')
```

---

## Performance Comparison

### Example: 2D Poisson Equation

For $-\nabla^2 u = f$ on $[0,1]^2$ with $n \times n$ grid:

TODO: Define the problem and test

### Rule of Thumb

| Matrix Property | Best Preconditioner | Expected Speedup |
|-----------------|---------------------|------------------|
| Diagonal dominant, well-scaled        | Jacobi | 2-5\times |
| SPD, sparse, structured               | Incomplete Cholesky | 5-20\times |
| General symmetric                     | SSOR | 2-10\times |
| Dense, small                          | None or Jacobi | 1-2\times |

---

## Implementation Details

### Base Preconditioner Class

All preconditioners inherit from abstract `Preconditioner` class:

```python
class Preconditioner(ABC):
    def set_a(self, a: Array, sigma: float = 0.0):
        """Setup from matrix A."""
        
    def set_s(self, s: Array, sp: Optional[Array] = None, sigma: float = 0.0):
        """Setup from Gram form S = Sp @ S."""
        
    def apply(self, r: Array) -> Array:
        """Apply M^{-1} to residual r."""
        
    def set_sigma(self, sigma: float):
        """Update regularization parameter."""
```

### Static vs Instance Methods

Preconditioners support both paradigms:

```python
from general_python.algebra.preconditioners import jacobi_apply_numpy

# Instance-based (convenient)
precond = JacobiPreconditioner()
precond.set_a(A)

# Static-based (for JIT compilation)
z       = precond.apply(r)
data    = {'diag_inv': 1.0 / np.diag(A)}
z       = jacobi_apply_numpy(r, data, sigma=0.0)
```

### JAX Compatibility

```python
import jax
import jax.numpy as jnp

# Create JAX-compatible preconditioner
precond = JacobiPreconditioner(backend='jax')
A_jax   = jnp.array(A)
precond.set_a(A_jax)

# Apply function is JAX-compatible
@jax.jit
def preconditioned_matvec(v):
    Av = A_jax @ v
    return precond.apply(Av)

# Can use in JAX transformations
grad_fn = jax.grad(lambda x: preconditioned_matvec(x).sum())
```

---

## Advanced Topics

### Choosing Regularization Parameter

For numerical stability, especially with Gram matrices:

$$
M \approx (A + \sigma I)
$$

**Guidelines**:

- Start with $\sigma = 10^{-8}$ to $10^{-4}$
- Increase if solver fails or preconditioner setup fails
- For NQS: often $\sigma \sim 10^{-4}$ to $10^{-2}$

### Incomplete Cholesky Variants

- **IC(0)**: No fill-in (sparsity preserved)
- **IC(k)**: Allow $k$ levels of fill-in
- **Modified IC**: Adjust diagonals to preserve row sums

### Multigrid Preconditioners

For structured grids (not yet implemented):

- Geometric multigrid
- Algebraic multigrid (AMG)
- Typically $O(\text{nnz}(A))$ complexity

---

## References

### Primary Sources

1. **Saad (2003)**: "Iterative Methods for Sparse Linear Systems"
   - Chapters 9-10 on preconditioning

2. **Barrett et al. (1994)**: "Templates for the Solution of Linear Systems"
   - Section on preconditioners

3. **Greenbaum (1997)**: "Iterative Methods for Solving Linear Systems"
   - Theoretical foundations

### Specific Preconditioners

1. **Gustafsson (1978)**: "A Class of First Order Factorization Methods"
   - Incomplete Cholesky, BIT 18:142-156

2. **Young (1971)**: "Iterative Solution of Large Linear Systems"
   - SSOR and related methods

---

## Troubleshooting

### Preconditioner Setup Fails

**Symptom**: Exception during `set_a()` or `set_s()`

**Solutions**:

1. Add regularization: `precond.set_a(A, sigma=1e-6)`
2. Try simpler preconditioner (Jacobi instead of IC)
3. Check matrix properties (SPD for Cholesky)

### No Improvement in Convergence

**Symptom**: Same iterations with and without preconditioner

**Solutions**:

1. Verify preconditioner is being used
2. Try different preconditioner type
3. Increase regularization
4. Check matrix scaling (rescale if needed)

### Solver Diverges with Preconditioner

**Symptom**: Solver fails when preconditioner is added

**Solutions**:

1. Ensure matrix and preconditioner are compatible (both SPD)
2. Reduce regularization (might be too large)
3. Use left preconditioning instead of right
4. Check for bugs in custom preconditioner

---

## Examples

See `examples/preconditioner_comparison.py` for comprehensive comparison on various problem types.

---

## License

MIT License - See repository LICENSE file.
