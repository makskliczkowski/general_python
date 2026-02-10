# Documentation Audit

## Current State

The documentation is built using **Sphinx** and hosted on **ReadTheDocs**. The source files are located in the `docs/` directory.

### Existing Documentation

*   **Core Pages (`.rst`)**:
    *   `index.rst`: Main entry point.
    *   `getting_started.rst`: Installation and basic usage.
    *   `design_principles.rst`: Architectural philosophy (Backend agnostic, Lazy loading).
    *   `installation.rst`: Detailed installation steps.
    *   `usage.rst`: Usage examples.
    *   `api.rst`: API reference structure.
    *   `contributing.rst`: Contribution guidelines.
    *   `license.rst`: License information.

*   **Markdown Guides (`.md`)**:
    *   `BENCHMARKS.md`: Performance benchmarks for Algebra, Lattices, and Physics modules.
    *   `BUILD_AND_CI.md`: Instructions for building the project and running CI.
    *   `FUTURE_BRANCHES.md`: Technical debt and future refactoring plans.
    *   `TESTS_CURRENT.md` & `TESTS_NEW.md`: Test inventory and new test documentation.

### Module Documentation

The source code (`algebra/`, `lattices/`, `maths/`, `ml/`, `physics/`, `common/`) generally contains high-quality docstrings, often using NumPy style.

*   **`algebra/`**: Good coverage. `solvers` and `utils` are well-documented.
*   **`lattices/`**: Extensive docstrings for `Lattice` classes (`Square`, `Hexagonal`, `Honeycomb`).
*   **`physics/`**: Detailed docstrings for `entropy`, `density_matrix`.
*   **`ml/`**: `networks` module has detailed factory documentation.
*   **`common/`**: `plot` module has extensive documentation on plotting utilities.

## Build Process

### Local Build

To build the documentation locally:

1.  Install documentation dependencies:
    ```bash
    pip install -e ".[docs]"
    ```
2.  Navigate to `docs/` and run `make`:
    ```bash
    cd docs
    make html
    ```
    Output is generated in `docs/_build/html/`.

### ReadTheDocs (RTD)

The project is configured for ReadTheDocs via `.readthedocs.yaml`.
*   **OS**: Ubuntu 24.04
*   **Python**: 3.12
*   **Configuration**: `docs/conf.py`

## Recent Improvements (Session Audit)

The following improvements have been identified and applied to enhance scientific clarity and robustness:

1.  **`ml/__init__.py`**:
    *   Replaced generic `Exception` with `ImportError` to provide clearer feedback when optional ML dependencies are missing.

2.  **`algebra/ran_matrices.py`**:
    *   Enhanced docstrings for `random_matrix` to clearly specify input/output contracts and supported ensembles.
    *   Clarified `CUE_QR` backend handling in docstrings.

3.  **`physics/operators.py`**:
    *   Added module-level docstring to explain the purpose of operator parsing and spectral statistics utilities.

4.  **`physics/density_matrix.py`**:
    *   Refined docstrings for `rho_numba_mask` and `schmidt_numba_mask` to explicitly state input dimensions (1D state vector) and output formats.

These changes ensure that the API contracts are explicit, especially regarding shapes, dtypes, and optional dependencies.
