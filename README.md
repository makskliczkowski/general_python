# General Python Utilities

[![Documentation Status](https://readthedocs.org/projects/general-python/badge/?version=latest)](https://general-python.readthedocs.io/en/latest/?badge=latest)
[![Python Versions](https://img.shields.io/pypi/pyversions/general-python-utils.svg)](https://pypi.org/project/general-python-utils/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python library providing utilities for scientific computing, particularly focused on quantum physics simulations, linear algebra, and numerical methods. This library consolidates commonly used functionalities into a unified, easy-to-use package with seamless support for both **NumPy** and **JAX** backends.

## Key Features

### Algebra & Linear Algebra
- **Backend Agnostic**: Seamlessly switch between NumPy and JAX backends.
- **Advanced Solvers**: Sparse matrix operations, eigenvalue/eigenvector computations (Arnoldi, Lanczos).
- **Optimization**: Preconditioners for iterative solvers and ODE solving utilities.
- *More details in [`algebra/README.md`](algebra/README.md)*.

### Physics & Quantum Tools
- **Quantum States**:   Density matrix operations, entropy calculations, and pure state manipulations.
- **Operators**:        Basis-aware operators and efficient observable calculations.
- **Thermodynamics**:   Statistical mechanics utilities and thermal property calculations.
- *More details in [`physics/README.md`](physics/README.md)*.

### Lattice Geometries
- **Topologies**:       Built-in support for Square, Hexagonal, Triangular, and Honeycomb lattices.
- **Navigation**:       Efficient neighbor finding and boundary condition handling (PBC/OBC).
- **Visualization**: Tools for plotting lattices, Brillouin zones, and reciprocal space.

### Machine Learning
- **Neural Networks**:  Implementations compatible with JAX/NumPy.
- **Training**:         Custom optimizers, loss functions, and schedulers.
- **Integration**:      Utilities for bridging with Keras and other frameworks.

### Mathematics & Random
- **RNG**:              High-quality, reproducible pseudorandom number generators.
- **Statistics**:       Statistical functions and special mathematical utilities.

### Common Utilities
- **IO**:               HDF5 support, efficient file/directory management.
- **Tools**:            Logging, debugging, and binary bit manipulation helpers.
- *More details in [`common/README.md`](common/README.md)*.

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### User Installation
Install the latest stable version directly from PyPI (if available) or from the source:

```bash
# From PyPI
pip install general-python-utils

# Or locally
pip install .
```

### Development Installation (Recommended)
For contributors or those who want to modify the source code, use an editable install:

```bash
git clone https://github.com/makskliczkowski/general_python.git
cd general_python
pip install -e ".[dev,docs,ml]"
```
This installs the package in editable mode along with development dependencies (pytest, black, flake8) and optional ML/documentation tools.

---

## Import Strategy & Best Practices

The package uses a **lazy import system** to minimize startup time and memory footprint. Submodules and heavy dependencies are only loaded when explicitly accessed.

### Recommended Usage

**Do not import deep trees.** Instead, access submodules through the top-level package or main subpackages.

**Good (Lazy & Clean):**
```python
import general_python as gp

# Access submodules lazily
solver  = gp.algebra.solvers.MinresQLPSolver
lattice = gp.lattices.SquareLattice(4, 4)
entropy = gp.physics.entropy.von_neumann_entropy(rho)
```

**Good (Explicit Imports):**
```python
from general_python.algebra import solvers
from general_python.physics import entropy

# Use specific functions
s = entropy.von_neumann_entropy(rho)
```

**Bad (Deep, Brittle Imports):**
```python
# Avoid importing from deep internal paths unless necessary
from general_python.algebra.solvers.minres_qlp  import MinresQLPSolver
from general_python.physics.entropy             import von_neumann_entropy
```

### Aliases and Shortcuts

The package provides several top-level aliases for convenience:

- **`gp.random`**           -> `general_python.algebra.ran_wrapper`
- **`gp.random_matrices`**  -> `general_python.algebra.ran_matrices`
- **`gp.physics.sp`**       -> `general_python.physics.single_particle`

### Backend Management

The backend (NumPy vs JAX) is managed centrally:

```python
from general_python.algebra import utils

# Check active backend
print(utils.ACTIVE_BACKEND_NAME)

# Get backend module dynamically
xp = utils.get_backend("jax") 
```

---

## Quick Start

```python
import general_python as gp

# 1. Automatic Backend Management (NumPy/JAX)
from general_python.algebra import utils
backend = utils.get_global_backend()
print(f"Using backend: {backend.name}")

# 2. Creating a Quantum Lattice
from general_python.lattices import SquareLattice
# Create a 4x4 square lattice with Periodic Boundary Conditions
lattice = SquareLattice(4, 4, bc='pbc')
print(f"Lattice sites: {lattice.Ns}")
```

---

## Testing

The project uses `pytest` for testing. To run the test suite:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=general_python
```

Ensure you have the development dependencies installed (`pip install -e ".[dev]"`).

---

## Documentation

Comprehensive documentation is hosted on Read the Docs. You can also build it locally:

```bash
cd docs
pip install -r requirements.txt
make html
```
Open `docs/_build/html/index.html` in your browser to view the local documentation.

---

## Contributing

Contributions are welcome! We follow standard open-source best practices.

1.  **Fork**    the repository.
2.  **Create**  a feature branch (`git checkout -b feature/amazing-feature`).
3.  **Commit**  your changes (`git commit -m 'Add amazing feature'`).
4.  **Lint**    your code using `black` and `flake8`.
5.  **Push**    to the branch (`git push origin feature/amazing-feature`).
6.  **Open**    a Pull Request.

Please ensure your code adheres to the project's style guidelines (Black formatting) and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.