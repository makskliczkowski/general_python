# General Python Utilities – Scientific Computing Tools for QES

[![License](https://img.shields.io/badge/License-CC--BY--4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive library of reusable scientific computing utilities for quantum physics simulations, numerical linear algebra, and machine learning. Designed as the foundation for **QES** (Quantum EigenSolver), these tools are available throughout the package and can be used independently.

## Purpose

`general_python` consolidates commonly needed functionality in quantum many-body physics and computational science into a unified, **backend-agnostic** package. Whether you're working with NumPy on CPU or JAX on GPU, the same code works transparently.

## Core Modules

### 1. Algebra & Linear Algebra (`algebra/`)
Advanced numerical methods and sparse matrix operations:

- **Solvers**: Iterative methods for large sparse systems
  - Lanczos (eigenvalues of symmetric matrices)
  - Arnoldi (eigenvalues of general matrices)
  - MinresQLP (symmetric indefinite systems)
  - GMRES (general linear systems)
- **Backend Abstraction**: Seamless NumPy ↔ JAX switching
- **Preconditioners**: Improve iterative solver convergence
- **ODE Integration**: Runge-Kutta methods, exponential integrators
- **Matrix Utilities**: Sparse format conversions (COO, CSR, CSC)

**Example:**
```python
import QES
from QES.general_python.algebra import solvers
import numpy as np

# Sparse Hamiltonian (CSR format, 100×100)
A = np.random.rand(100, 100)
A = (A + A.T) / 2  # Symmetric
b = np.random.rand(100)

# Solve A·x = b using Lanczos preconditioned MINRES
solver = solvers.MinresQLPSolver(verbose=True)
x, info = solver.solve(A, b, tol=1e-8)
```

### 2. Physics & Quantum Tools (`physics/`)
Specialized quantum mechanical and statistical mechanics utilities:

#### Quantum States & Operations
- **Density Matrices**: Construction, purification, partial trace
- **Entropy Measures**:
  - Von Neumann entropy: $S(\rho) = -\text{Tr}(\rho \log \rho)$
  - Shannon entropy: $H(p) = -\sum p_i \log p_i$
  - Rényi entropy: $S_\alpha(\rho) = \frac{1}{1-\alpha} \log \text{Tr}(\rho^\alpha)$
- **Purity**: $\text{Tr}(\rho^2)$, distinguishes pure vs mixed states
- **Entanglement**: Entanglement entropy via SVD of subsystem density matrix
- **Correlations**: Two-point correlators, structure factors

#### Thermodynamic Properties
- **Boltzmann Statistics**: Partition function, thermal averages, free energy
- **Specific Heat**: Temperature-dependent heat capacity
- **Magnetic Susceptibility**: Response to external fields
- **Correlation Functions**: Static and dynamic response

#### Spectral Analysis
- **Spectral Functions**: Spectral weight, density of states
- **Response Theory**: Linear response, correlation-response relations
- **Green's Functions**: Matsubara and retarded formulations

**Example:**
```python
from QES.general_python.physics import entropy, correlations
import numpy as np

# Pure state (rank-1 density matrix)
psi = np.random.rand(64) + 1j * np.random.rand(64)
psi /= np.linalg.norm(psi)
rho = np.outer(psi.conj(), psi)

# Entanglement entropy of subsystem [0:4] in 6-spin system
S_vn = entropy.von_neumann_entropy(rho)
print(f"Purity: {np.trace(rho @ rho):.6f}")  # Should be ~1.0 for pure state

# Two-point spin correlation (Z-Z)
Sz = np.array([[1, 0], [0, -1]])  # Pauli-Z
corr_zz = correlations.spin_correlation_zz(psi, sites=(0, 3))
```

### 3. Lattice Geometry (`lattices/`)
Efficient lattice construction and topology handling:

#### Supported Geometries
| Lattice | Dimension | Neighbors | Use Cases |
|---------|-----------|-----------|-----------|
| Chain | 1D | 2 (nearest) | Spin chains, TFIM benchmarks |
| Square | 2D | 4 (nearest) | Standard 2D quantum systems |
| Hexagonal | 2D | 3 | Honeycomb materials, Kitaev models |
| Triangular | 2D | 6 | Frustrated magnetism |
| Honeycomb | 2D | 3 | Graphene-like structures |

#### Features
- **Neighbor Finding**: Efficient nearest/k-th neighbor queries
- **Boundary Conditions**: Periodic (PBC) and open (OBC) support
- **Lattice Vectors**: Reciprocal lattice, Brillouin zone
- **Visualization**: Plot lattice structure, Brillouin zones
- **Basis Modes**: Multi-orbital/sublattice structures

**Example:**
```python
from QES.general_python.lattices import SquareLattice

# 8×8 square lattice, periodic boundaries
lat = SquareLattice(lx=8, ly=8, bc='pbc')

print(f"Total sites: {lat.Ns}")  # 64
print(f"Lattice vectors: {lat.lattice_vectors}")

# Find neighbors of site 5
neighbors = lat.neighbors(site=5, distance=1)  # Up, down, left, right
print(f"Neighbors of site 5: {neighbors}")

# Visualize
lat.visualize(show_indices=True)
```

### 4. Machine Learning (`ml/`)
Tools for neural network implementations and training:

- **Network Layers**: Custom JAX/Flax modules for quantum networks
- **Optimizers**: Adam, SGD, RMSprop with learning rate scheduling
- **Loss Functions**: Custom physics-informed loss definitions
- **Sampling**: Mini-batch generation, data loaders
- **Visualization**: Loss curves, metric tracking

### 5. Mathematics (`maths/`)
General mathematical utilities beyond linear algebra:

- **Special Functions**: Bessel, Hermite, Legendre polynomials
- **Combinatorics**: Partitions, combinations, permutations
- **Interpolation**: Polynomial and cubic spline fitting
- **Integration**: Quadrature rules (Gauss-Legendre, etc.)
- **Optimization**: Scipy wrappers, golden section search

### 6. Random Number Generation (`algebra/ran_*`)
Reproducible high-quality pseudorandom sequences:

- **RNG Streams**: Seeded generators with independent streams
- **Random Matrices**: Gaussian Orthogonal Ensemble (GOE), Gaussian Unitary Ensemble (GUE)
- **Sampling**: Metropolis-Hastings, importance sampling helpers
- **Distributions**: Multivariate normal, exponential, etc.

**Example:**
```python
from QES.general_python.algebra import ran_wrapper
import numpy as np

# Create seeded RNG
rng = ran_wrapper.get_rng(seed=42)

# Reproducible samples
samples1 = rng.normal(size=1000)
rng_reset = ran_wrapper.get_rng(seed=42)
samples2 = rng_reset.normal(size=1000)

assert np.allclose(samples1, samples2)  # ✓
```

### 7. Common Utilities (`common/`)
General-purpose tools:

- **I/O**: HDF5 file handling, data serialization
- **Logging**: Configured logger for debugging
- **Caching**: Function result caching, memoization
- **Tools**: Bit manipulation, string utilities, system info
- **Configuration**: Settings management

---

## Backend Agnosticism

All modules support both NumPy and JAX transparently:

```python
import QES
from QES.general_python.algebra import utils

# Check active backend
backend = utils.get_global_backend()
print(f"Backend: {backend.__name__}")  # 'numpy' or 'jax'

# Get backend dynamically
xp = utils.get_backend("jax")  # Returns jax module
```

Switch backends via QES session manager:

```python
with QES.run(backend='jax'):
    # JAX code here
    from QES.general_python.physics import entropy
    S = entropy.von_neumann_entropy(rho)  # Uses JAX operations
```

---

## Installation

### Within QES
Already included. Install QES with:
```bash
pip install -e "QuantumEigenSolver/pyqusolver/Python/[all,dev]"
```

### Standalone (if extracted separately)
```bash
pip install -e ".[dev]"
```

---

## Quick Start Examples

### Example 1: Lattice Construction and Topology

```python
from QES.general_python.lattices import HexagonalLattice
import matplotlib.pyplot as plt

# Create 6×6 hexagonal lattice (Honeycomb/Graphene-like)
lat = HexagonalLattice(lx=6, ly=6, bc='pbc')

print(f"Sites: {lat.Ns}, Bonds: {len(lat.bond_list())}")

# Plot with bond structure
fig, ax = plt.subplots()
lat.visualize(ax=ax, draw_bonds=True)
plt.show()
```

### Example 2: Entanglement Entropy Calculation

```python
from QES.general_python.physics import entropy
import numpy as np

# Ground state of 8-spin system (from exact diagonalization)
psi = np.random.rand(256) + 1j * np.random.rand(256)
psi /= np.linalg.norm(psi)

# Bipartite entanglement entropy between [0:4] and [4:8]
S_ent = entropy.entanglement_entropy(psi, partition_a=list(range(4)), Ns=8)
print(f"Entanglement entropy: {S_ent:.4f} bits")

# For comparison: Log2(2^4) = 4 bits is maximum for 4 qubits
```

### Example 3: Sparse Matrix Solver

```python
from QES.general_python.algebra.solvers import LanczosSolver
from scipy.sparse import diags
import numpy as np

# Create sparse diagonal-tridiagonal matrix
n = 1000
diag = 2 * np.ones(n)
off_diag = -np.ones(n-1)
A = diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(n, n))

# Compute lowest 10 eigenvalues
solver = LanczosSolver(k=10)
evals, evecs = solver.solve(A)
print(f"Lowest eigenvalue: {evals[0]:.6f}")
```

---

## Documentation

### Submodule READMEs
- **[Algebra](algebra/README.md)** – Linear solvers, matrix operations
- **[Physics](physics/README.md)** – Quantum operators, thermodynamics, spectral analysis
- **[Lattices](lattices/README.md)** – Geometry, neighbor finding, visualization
- **[ML](ml/README.md)** – Neural networks, training loops

### Build Full Docs
```bash
cd docs
pip install -r requirements.txt
make html
# Open _build/html/index.html
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test general_python specifically
pytest tests/test_imports.py -v
```

---

## Contributing

Contributions welcome! Please:
1. Follow PEP 8 (enforced with Black)
2. Add docstrings to all functions
3. Include unit tests for new code
4. Ensure backward compatibility

---

## License

**CC-BY-4.0** – See root [LICENSE.md](../../../../LICENSE.md)