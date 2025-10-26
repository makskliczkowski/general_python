# General Python Utilities

[![Documentation Status](https://readthedocs.org/projects/general-python/badge/?version=latest)](https://general-python.readthedocs.io/en/latest/?badge=latest)

A comprehensive Python library providing utilities for scientific computing, particularly focused on quantum physics simulations and numerical methods. This library consolidates commonly used functionalities into a unified, easy-to-use package with support for both NumPy and JAX backends.

## Key Features

### 🧮 **Algebra & Linear Algebra**
- Advanced linear algebra operations with NumPy/JAX backend support
- Sparse matrix operations and solvers
- Eigenvalue/eigenvector computations
- Preconditioners for iterative solvers
- ODE solving utilities
- Mathematical and API summary: `algebra/README.md`

### 🎲 **Mathematics & Random Numbers**
- High-quality pseudorandom number generators
- Statistical functions and utilities
- Mathematical utilities and special functions
- Support for reproducible random sequences

### 🔗 **Lattice Structures**
- Tools for creating and manipulating lattice geometries
- Support for square, hexagonal, and honeycomb lattices
- Neighbor finding and lattice navigation
- Tenpy-inspired visualisation utilities for real/reciprocal space, Brillouin zones, and boundary conditions
- Jupyter demo (`Python/test/lattices/lattice_visualization_demo.ipynb`) showcasing these plotting helpers
- Common lattice operations for condensed matter physics

### 🧠 **Machine Learning**
- Neural network implementations with JAX/NumPy backends
- Training utilities and optimizers
- Loss functions and schedulers
- Keras integration utilities

### ⚛️ **Physics Utilities**
- Quantum state manipulations
- Density matrix operations
- Entropy calculations
- Eigenstate analysis
- Quantum operator utilities
- Mathematical and module map: `physics/README.md`

### 🛠️ **Common Utilities**
- File and directory management
- Data handling and HDF5 support
- Plotting and visualization tools
- Logging and debugging utilities
- Binary operations and bit manipulation
- Detailed module map: `common/README.md`

## Documentation Map

- [`algebra/README.md`](algebra/README.md) — scope of linear algebra, random ensembles, IVPs, and backend rules.
- [`algebra/solvers/README.md`](algebra/solvers/README.md) — iterative/direct solver catalogue and convergence criteria.
- [`algebra/eigen/README.md`](algebra/eigen/README.md) — Arnoldi, Lanczos, and block-Lanczos algorithms with residual controls.
- [`algebra/examples/README.md`](algebra/examples/README.md) — runnable scripts illustrating solver usage patterns.
- [`physics/README.md`](physics/README.md) — thermodynamics, spectral/response functions, and correlation matrix tooling.
- [`common/README.md`](common/README.md) — bitstring, filesystem, plotting, and diagnostic helpers shared across packages.
- [`common/embedded/README.md`](common/embedded/README.md) — low-level kernels that power the binary utilities.
- [`BACKEND_INTEGRATION.md`](BACKEND_INTEGRATION.md) — backend (NumPy/JAX) compatibility notes for solvers and eigensolvers.
- [`PRECONDITIONERS.md`](PRECONDITIONERS.md) — catalogue of supported preconditioners with mathematical guarantees.
- [`DOCUMENTATION_SUMMARY.md`](DOCUMENTATION_SUMMARY.md) — high-level overview of the entire `general_python` package.

## Installation

### From Source (Recommended for Development)

```bash
git clone <repository-url>
cd general_python
pip install -e .
```

The package uses a **src-layout** with symlinks for editable installs. The `src/general_python/` directory is auto-generated during installation and should not be committed to version control.

### From PyPI

```bash
pip install general-python-utils
```

## Quick Start

```python
import general_python as gp

# Use algebra utilities with automatic backend detection
from general_python.algebra import utils
backend = utils.get_global_backend()

# Create a lattice
from general_python.lattices import SquareLattice
lattice = SquareLattice(4, 4)

# Mathematical utilities
from general_python.maths import math_utils
result = math_utils.some_function()
```

## Documentation

Full documentation is available at [Read the Docs](https://general-python.readthedocs.io/).

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see the contributing guidelines in the documentation.
