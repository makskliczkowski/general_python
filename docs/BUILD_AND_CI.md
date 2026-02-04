# Build and CI Guide

This document provides instructions on how to install, test, and build documentation for `general_python`.

## Installation

### Dependencies

The project requires Python 3.8+.
Core dependencies: `numpy`, `scipy`, `matplotlib`, `numba`.

Optional dependencies:
- `ml`: Machine learning utilities (`tensorflow`, `jax`, `flax`, etc.)
- `jax`: JAX backend support (`jax`, `jaxlib`, `flax`)
- `docs`: Documentation building (`sphinx`, etc.)
- `dev`: Development tools (`pytest`, `black`, `flake8`)

### Installing for Development

To install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev,ml,jax,docs]"
```

For a minimal install:

```bash
pip install -e .
```

## Running Tests

Tests are located in the `tests/` directory.

To run all tests:

```bash
pytest tests/ -v
```

To run specific tests:

```bash
pytest tests/test_imports.py -v
```

## Building Documentation

The documentation is built using Sphinx.

1. Install documentation dependencies:
   ```bash
   pip install -e ".[docs]"
   # OR
   pip install -r docs/requirements.txt
   ```

2. Build the HTML documentation:
   ```bash
   cd docs
   make html
   ```

3. Open the documentation:
   ```bash
   # On Linux
   xdg-open _build/html/index.html
   # On macOS
   open _build/html/index.html
   ```

## Packaging

The project uses `setuptools` and `pyproject.toml` for packaging.
Source files are located in the root directory and included via `MANIFEST.in`.
