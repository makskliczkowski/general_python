# Build and CI

This document describes how to install, test, and build the `general_python` package.

## Installation

The package uses a `pyproject.toml` based build system.

### Editable Install (for development)
To install the package in editable mode (changes to source files are immediately reflected):

```bash
pip install -e .
```

### Regular Install
To install the package normally:

```bash
pip install .
```

## Dependencies

The project defines several optional dependency groups in `pyproject.toml`:

*   `ml`: Machine learning dependencies (tensorflow, scikit-learn, pandas, jax, flax).
*   `jax`: JAX specific dependencies (jax, jaxlib, flax).
*   `docs`: Documentation build dependencies (sphinx, sphinx-rtd-theme).
*   `dev`: Development tools (pytest, black, flake8).

To install with specific extras:

```bash
pip install .[dev,ml,jax]
```

## Testing

Tests are located in `tests/` and subdirectories of modules.

### Prerequisites
Install development and optional dependencies to run the full test suite:

```bash
pip install .[dev,ml,jax]
```

### Running Tests
Run all tests using `pytest`:

```bash
pytest tests/ -v
```

To run a specific test file:

```bash
pytest tests/test_imports.py -v
```

There is also a documentation integrity script:

```bash
python3 test_documentation.py
```

## Documentation

The documentation is built using Sphinx.

### Prerequisites
Install documentation dependencies:

```bash
pip install .[docs]
```
Note: You may also need `ml` or `jax` dependencies if `autodoc` imports modules that depend on them (e.g. `pandas` in `maths`, though it is guarded).

### Building Docs
Navigate to the `docs/` directory and run `make html`:

```bash
cd docs
make html
```

The output will be in `docs/_build/html/index.html`.

## CI Verification

For Continuous Integration, the following steps are recommended:

1.  **Install**: `pip install .[dev,ml,jax,docs]`
2.  **Test**: `pytest tests/`
3.  **Doc Build**: `cd docs && make html`
