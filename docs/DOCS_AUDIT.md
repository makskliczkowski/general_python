# Documentation Audit

## 1. Existing Documentation

The `docs/` directory contains the following Sphinx documentation source files:

*   **Core Guides:**
    *   `index.rst`: Main entry point.
    *   `introduction.rst`: Project overview.
    *   `getting_started.rst`: Installation, backend configuration, and testing.
    *   `design_principles.rst`: Architecture, backend agnosticism, lazy loading.
    *   `installation.rst`: Detailed installation steps.
    *   `usage.rst`: Basic usage examples.
    *   `api.rst`: API reference.
    *   `contributing.rst`: Contribution guidelines.
    *   `license.rst`: License information.

*   **Configuration:**
    *   `conf.py`: Sphinx configuration.
    *   `Makefile`: Build script for Make.
    *   `requirements.txt`: Dependencies for documentation build.

*   **Root Configuration:**
    *   `.readthedocs.yaml`: ReadTheDocs configuration file.

## 2. Missing or Outdated Information

*   **Module-Specific Details:**
    *   `physics`: The distinction between Numba-optimized CPU code and JAX-optimized GPU code needs to be clearer in the `density_matrix` and `entropy` modules.
    *   `lattices`: Detailed input/output contracts for factory functions like `choose_lattice` are minimal.
    *   `maths`: The `Statistics` class methods lack detailed docstrings regarding input shapes.
*   **Advanced Usage:**
    *   Guides for implementing custom Lattice classes or specialized Solvers are missing.
    *   Examples for mixed-backend usage (e.g., NumPy lattice with JAX solver) could be expanded.

## 3. Building Documentation

### Local Build

To build the documentation locally:

1.  **Install dependencies:**
    ```bash
    pip install ".[docs]"
    ```
    *Note: This installs `sphinx` and `sphinx-rtd-theme`.*

2.  **Build HTML:**
    ```bash
    cd docs
    make html
    ```

3.  **View:**
    Open `docs/_build/html/index.html` in your browser.

### ReadTheDocs (RTD)

The project is configured for RTD via `.readthedocs.yaml`.
*   **OS:** Ubuntu 22.04
*   **Python:** 3.10
*   **Configuration:** Uses `docs/conf.py`.
*   **Requirements:** Installs the package with `docs` extra.

## 4. Proposed Improvements (Current Task)

*   **Module Docstrings:** Enhance top-level docstrings for `algebra`, `lattices`, `maths`, `ml`, `physics`, and `common` to clarify purpose and scope.
*   **Function Docstrings:** Add detailed input/output contracts, shapes, and backend specifics to key functions in `algebra/solver.py`, `lattices/lattice.py`, `maths/statistics.py`, `ml/networks.py`, and `physics/density_matrix.py`.
*   **Alignment:** Ensure `getting_started.rst` and `design_principles.rst` accurately reflect the current lazy-loading architecture and backend selection mechanisms.
