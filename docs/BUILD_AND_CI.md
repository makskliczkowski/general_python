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

**Important:** For running tests, the package *must* be installed (either editable or regular) so that the `general_python` namespace is correctly resolved.

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

Tests are distributed across the codebase:
*   `tests/`: General integration and top-level tests.
*   `algebra/tests/`: Tests for algebra and solvers.
*   `lattices/tests/`: Tests for lattice structures.
*   `maths/tests/`: Tests for math utilities.
*   `physics/tests/`: Tests for physics modules.

### Prerequisites
Install development and optional dependencies to run the full test suite. JAX and ML dependencies are required for some tests (e.g., lattice invariants, network activations).

```bash
pip install -e ".[dev,ml,jax]"
```

### Running Tests
Run **all** tests using `pytest` from the root directory:

```bash
pytest
```

To run a specific test file:

```bash
pytest algebra/tests/test_solvers_sanity.py -v
```

There is also a documentation integrity script:

```bash
python3 test_documentation.py
```

### Troubleshooting
*   **ModuleNotFoundError: No module named 'general_python'**:
    *   Ensure you have installed the package using `pip install -e .`. The tests rely on importing `general_python` as an installed package.
*   **ModuleNotFoundError: No module named 'jax' / 'h5py'**:
    *   Ensure you installed the optional dependencies: `pip install -e ".[ml,jax]"`.

## Documentation

The documentation is built using Sphinx.

### Prerequisites
Install documentation dependencies:

```bash
pip install .[docs]
```
Note: You may also need `ml` or `jax` dependencies if `autodoc` imports modules that depend on them (e.g. `pandas` in `maths`). `docs/requirements.txt` is provided for ReadTheDocs environment.

### Building Docs
Navigate to the `docs/` directory and run `make html`:

```bash
cd docs
make html
```

The output will be in `docs/_build/html/index.html`.

## CI Verification

The project currently does not have an automated CI workflow (e.g., GitHub Actions).
For manual verification, the following steps are recommended:

1.  **Install**: `pip install -e ".[dev,ml,jax,docs]"`
2.  **Test**: `pytest` (runs all tests)
3.  **Doc Build**: `cd docs && make html`
