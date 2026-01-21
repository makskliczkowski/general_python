# Backend Integration with utils.py

## Overview

The `backend_ops.py` module provides solver-specific operations while integrating seamlessly with the existing `utils.py` backend infrastructure. This document explains how the two systems work together.

## Architecture

```bash
┌─────────────────────────────────────────────────────────────────┐
│                        utils.py                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            BackendManager (Global)                       │   │
│  │  - Manages NumPy/JAX detection and configuration         │   │
│  │  - Provides get_backend(), get_global_backend()          │   │
│  │  - Handles random number generation (RNG, PRNG keys)     │   │
│  │  - Controls JIT compilation                              │   │
│  │  - Sets default dtypes (int, float, complex)             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ imports
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     backend_ops.py                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              BackendOps Class                            │   │
│  │  - Wraps backend operations for solvers                  │   │
│  │  - Provides solver-specific ops (sym_ortho, etc.)        │   │
│  │  - Integrates with BackendManager for defaults           │   │
│  │  - Offers both instance and global access patterns       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ used by
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Solver Implementations                       │
│    cg.py, minres.py, minres_qlp.py, direct.py, etc.             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Integration Points

### 1. Import Structure

```python
# backend_ops.py imports from utils.py
from ..utils import (
    JAX_AVAILABLE,      # Global flag for JAX availability
    backend_mgr,        # Global BackendManager instance
    get_backend,        # Function to get backend modules
    Array,              # Type alias for arrays (np.ndarray | jnp.ndarray)
    is_jax_array        # Helper to detect JAX arrays
)
```

### 2. Default Backend Selection

When you create a `BackendOps` instance with `'default'`:

```python
ops = BackendOps('default')
```

It uses the **global backend manager's active backend**:

```python
if backend_lower in ['default', 'auto']:
    # Use the global backend manager's active backend
    self.backend_name   = backend_mgr.name
    self.backend        = backend_mgr.np
    self.backend_sp     = backend_mgr.scipy
    self.is_jax         = (self.backend_name == 'jax')
```

This ensures consistency across the entire codebase.

### 3. Global Instances

Pre-created instances for convenience:

```python
numpy_ops   = BackendOps.get_ops('numpy')    # Always NumPy
jax_ops     = BackendOps.get_ops('jax')      # JAX if available
default_ops = BackendOps.get_ops('default')  # Uses backend_mgr.name
```

## Usage Patterns in Solvers

### Pattern 1: Explicit Backend (Recommended)

```python
from ..solvers.backend_ops import get_backend_ops

class MinresSolver(Solver):
    def __init__(self, backend='numpy', **kwargs):
        self.backend_str    = backend
        self.ops            = get_backend_ops(backend)
    
    def solve(self, b, x0=None):
        # Use self.ops for all operations
        x                   = self.ops.zeros(len(b))
        norm_b              = self.ops.norm(b)
        # ... rest of algorithm
```

### Pattern 2: Auto-detect from Array

```python
from ..solvers.backend_ops import BackendOps

def minres(A, b, tol=1e-6):
    """Matrix-free MINRES solver."""
    # Automatically detect backend from input array
    ops     = BackendOps.from_array(b)
    x       = ops.zeros(len(b))
    # ... algorithm using ops
    return x
```

### Pattern 3: Use Global Default

```python
from ..solvers.backend_ops import get_backend_ops

class CgSolver(Solver):
    def __init__(self, **kwargs):
        # Use whatever backend is globally configured
        self.ops = get_backend_ops()  # Uses default_ops
```

## Comparison: utils.py vs backend_ops.py

### What `utils.py` Provides

- **Backend detection and initialization**  : Detects JAX, sets up imports
- **Global configuration**                  : Environment variables, seeds, dtypes
- **Random number generation**              : NumPy RNG, JAX PRNG keys
- **Backend switching**                     : `set_active_backend()`, `get_backend()`
- **JIT compilation**                       : `maybe_jit` decorator
- **Type system**                           : `distinguish_type()`, dtype registry

### What `backend_ops.py` Provides

- **Solver-specific operations**            : Unified interface for linear algebra
- **Special operations**                    : `sym_ortho()` for Givens rotations
- **Consistent API**                        : Same method names regardless of backend
- **Convenience methods**                   : `from_array()`, pre-created instances
- **Solver integration**                    : Designed for iterative solver needs

### Division of Responsibilities

| Feature | utils.py | backend_ops.py |
|---------|----------|----------------|
| Backend detection     | v (JAX_AVAILABLE)    | Uses from utils          |
| Backend switching     | v (BackendManager)   | Uses backend_mgr         |
| Random number gen     | v (RNG, PRNG keys)   | Not needed               |
| Linear algebra ops    | Basic (np, jnp)      | v (dot, norm, solve)     |
| Givens rotations      | (x)                    | v (sym_ortho)            |
| Solver-specific       | (x)                    | v (triangular_solve)     |
| Global state          | v (backend_mgr)      | Uses backend_mgr         |
| Array creation        | Basic                | v (zeros, ones)          |
| Type conversion       | v (distinguish_type) | Uses from utils          |

## Examples

### Example 1: Using Global Backend

```python
# In your main code, set the backend
from general_python.algebra.utils import backend_mgr

backend_mgr.set_active_backend('jax')

# In a solver
from ..solvers.backend_ops import get_backend_ops

ops = get_backend_ops()  # Automatically uses JAX
x   = ops.zeros(100)
# ops.backend_name == 'jax'
```

### Example 2: Override in Solver

```python
# Global backend is JAX
from general_python.algebra.utils import backend_mgr
backend_mgr.set_active_backend('jax')

# But use NumPy for a specific solver
from ..solvers import CgSolver

solver = CgSolver(backend='numpy')  # Explicitly use NumPy
# solver.ops.backend_name == 'numpy'
```

### Example 3: Random Numbers (utils.py)

For random number generation, use `utils.py`:

```python
from ..utils import get_backend, backend_mgr

# NumPy backend with random
np_mod, rng = get_backend('numpy', random=True, seed=42)
random_vec  = rng.normal(size=10)

# JAX backend with random (need to split keys!)
jax_np, (jax_rnd, key), jax_sp = get_backend('jax', random=True, scipy=True, seed=42)
key, subkey = jax_rnd.split(key)
random_vec  = jax_rnd.uniform(subkey, shape=(10,))
```

For solver operations (no randomness), use `backend_ops.py`:

```python
from ..solvers.backend_ops import get_backend_ops

ops     = get_backend_ops('jax')
x       = ops.zeros(10)
norm_x  = ops.norm(x)
```

## Best Practices

### 1. Import Strategy

```python
# For solvers - use backend_ops.py
from ..solvers.backend_ops import get_backend_ops, BackendOps

# For backend info/switching - use utils.py
from ..utils import backend_mgr, JAX_AVAILABLE, Array

# For random numbers - use utils.py
from ..utils import get_backend
```

### 2. Solver Implementation Pattern

```python
from ..solver import Solver, SolverResult
from ..solvers.backend_ops import get_backend_ops
from ..utils import Array

class MySolver(Solver):
    """My custom solver."""
    
    def __init__(self, backend='numpy', eps=1e-6, maxiter=1000):
        super().__init__(eps=eps, maxiter=maxiter)
        self.backend_str = backend
        self.ops = get_backend_ops(backend)
    
    def solve(self, b: Array, x0=None) -> SolverResult:
        """Solve the linear system."""
        n = len(b)
        
        # Use self.ops for all operations
        if x0 is None:
            x = self.ops.zeros(n, dtype=b.dtype)
        else:
            x = self.ops.copy(x0)
        
        # Algorithm...
        for i in range(self.maxiter):
            # Use matvec function from base class
            Ax = self.matvec(x)
            
            # Use ops for computations
            residual = self.ops.norm(b - Ax)
            
            if residual < self.eps:
                return SolverResult(
                    x               =   x,
                    converged       =   True,
                    iterations      =   i+1,
                    residual_norm   =   float(residual)
                )
        
        return SolverResult(x=x, converged=False, iterations=self.maxiter,
                           residual_norm=float(residual))
```

### 3. Type Hints

Use `Array` from `utils.py`:

```python
from ..utils import Array
from typing import Optional

def my_function(x: Array, tol: float = 1e-6) -> Array:
    """Works with both NumPy and JAX arrays."""
    ops = BackendOps.from_array(x)
    return ops.sqrt(x)
```

## Testing Integration

When writing tests:

```python
import pytest
from general_python.algebra.utils import JAX_AVAILABLE, backend_mgr
from general_python.algebra.solvers.backend_ops import get_backend_ops

@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if JAX_AVAILABLE else []))
def test_solver_both_backends(backend):
    """Test solver with both NumPy and JAX."""
    ops = get_backend_ops(backend)
    x   = ops.ones(10)
    assert ops.norm(x) == pytest.approx(np.sqrt(10))

def test_global_backend():
    """Test integration with global backend manager."""
    original = backend_mgr.name
    backend_mgr.set_active_backend('numpy')
    
    # Test with NumPy
    ops = get_backend_ops()
    assert ops.backend_name == 'numpy'
    
    # Test with JAX (if available)
    if JAX_AVAILABLE:
        backend_mgr.set_active_backend('jax')
        ops = get_backend_ops()
        assert ops.backend_name == 'jax'
    
    # Restore
    backend_mgr.set_active_backend(original)
```

## Summary

- **`utils.py`**        : Global backend management, configuration, random numbers
- **`backend_ops.py`**  : Solver-specific operations with clean API
- **Integration**       : `backend_ops.py` uses `utils.py` for defaults and configuration
- **Solvers**           : Use `backend_ops.py` for operations, `utils.py` for types/config
- **Pattern**           : Create `BackendOps` instance in solver `__init__`, use throughout

This architecture keeps the codebase modular while ensuring consistency between the general backend system and solver-specific needs.
