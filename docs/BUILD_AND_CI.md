# Build and CI Instructions

This document outlines how to install, test, and build documentation for the `general_python` package.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Install for Usage
To install the package in your current environment:
```bash
pip install .
```

### Install for Development (Editable)
To install in editable mode (changes to source code are immediately reflected):
```bash
pip install -e .
```

### Optional Dependencies
The package provides several optional dependency groups (extras):
- `ml`: Machine learning utilities (TensorFlow, Scikit-learn, Pandas, JAX, Flax).
- `jax`: JAX-specific utilities (JAX, Flax).
- `docs`: Documentation building tools (Sphinx).
- `dev`: Development tools (pytest, Black, Flake8).

To install with specific extras:
```bash
pip install -e ".[ml,jax]"
```

To install all development dependencies:
```bash
pip install -e ".[dev,ml,jax,docs]"
```

## Running Tests

Tests are located in `tests/` and in `tests/` subdirectories within each module.

### Prerequisites
Install development and optional dependencies to run the full test suite:
```bash
pip install -e ".[dev,ml,jax]"
```
*Note: Some tests require `jax` or `h5py` (included in core dependencies) and will fail if these are missing.*

### Running with Pytest
Run all tests:
```bash
pytest
```

Run tests for a specific module:
```bash
pytest general_python/algebra/tests/
```

Run a specific test file:
```bash
pytest tests/test_imports.py
```

## Building Documentation

The documentation is built using Sphinx.

### Prerequisites
Install documentation dependencies:
```bash
pip install -e ".[docs]"
```

### Building HTML Docs
Using `make` (Unix/Linux/macOS):
```bash
cd docs
make html
```

Using `sphinx-build` directly:
```bash
sphinx-build -b html docs docs/_build/html
```

The generated HTML documentation will be available in `docs/_build/html/index.html`.

## CI Verification

For Continuous Integration (CI), the following steps are recommended:

1.  **Install**: `pip install .[dev,ml,jax,docs]`
2.  **Test**: `pytest tests/`
3.  **Doc Build**: `cd docs && make html`
4.  **Sanity Check**: `python3 test_documentation.py`
