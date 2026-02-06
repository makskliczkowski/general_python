# Documentation Audit

## Overview
This document assesses the state of the project's documentation, including Sphinx configuration, existing guides, and source code docstrings.

## Existing Documentation
- **Configuration**:
  - `docs/conf.py`: Configured for Sphinx with `sphinx_rtd_theme`.
  - `.readthedocs.yaml`: Configured for Python 3.12 on Ubuntu 24.04.
  - `docs/Makefile`: Standard Sphinx makefile.
- **Guides**:
  - `README.md`: Comprehensive project overview, installation, and module summaries.
  - `docs/getting_started.rst`: Covers installation (including `pyproject.toml` extras), backend selection, and testing.
  - `docs/design_principles.rst`: Outlines scientific contracts, backend agnosticism, and lazy loading.
  - `docs/api.rst`: Placeholder/Entry point for API docs.

## Build Instructions
### Local Build
Prerequisites: `pip install -e ".[docs]"`
```bash
cd docs
make html
# Output: _build/html/index.html
```

### ReadTheDocs (RTD)
Configuration is managed via `.readthedocs.yaml`. It installs the package with `[docs]` extras and builds using `docs/conf.py`.

## Dependencies Note
During verification, the following runtime dependencies were identified as critical for tests and documentation imports, though they might be optional in `pyproject.toml`:
- `h5py`: Required by `lattices` (via `common.hdf5man`) for import.
- `jax`, `flax`: Required for `ml` and parts of `algebra` tests.

## Codebase Docstring Audit

### General Findings
- **Inconsistent Styles**: The codebase mixes Google-style, NumPy-style, and custom/informal docstrings (e.g., `''' - param : ... '''`).
- **LaTeX Usage**: Some docstrings use LaTeX but may lack `r""` prefixes, risking invalid escape sequences.
- **Module-Level Docs**: Present in most `__init__.py` files, but quality varies.

### Module-Specific Audit

#### 1. `algebra/`
- **Status**: Generally good.
- **Issues**:
  - `ran_wrapper.py`: Uses custom docstring format. Needs standardization to NumPy/Google style.
  - `solver.py`: Mostly detailed, but some helper methods (e.g., `_sym_ortho`) have inconsistent formatting.

#### 2. `lattices/`
- **Status**: Good structure.
- **Issues**:
  - `square.py`: Methods like `calculate_nn_in` lack type information in arguments. Class docstrings could be more comprehensive regarding coordinate systems.

#### 3. `maths/`
- **Status**: Needs improvement.
- **Issues**:
  - `math_utils.py`: Uses non-standard, brief docstrings. `Fitter` class lacks detailed attribute descriptions.
  - `statistics.py`: Likely similar issues (inferred).

#### 4. `ml/`
- **Status**: High quality.
- **Highlights**: `networks.py` and `choose_network` have excellent, detailed docstrings.
- **Action**: Maintain this standard.

#### 5. `physics/`
- **Status**: Mixed.
- **Issues**:
  - `entropy.py`: Mix of good docstrings and brief/custom ones (e.g., `entro_random_gaussian`). Needs unification.

#### 6. `common/`
- **Status**: Functional but inconsistent.
- **Issues**:
  - `datah.py`: `DataHandler` methods need standardized docstrings.

## Action Plan
1. **Standardize Styles**: Convert custom docstrings to a consistent NumPy/Google style (supported by Napoleon extension).
2. **Enhance Contracts**: Ensure all public methods specify `Args` (with types/shapes) and `Returns`.
3. **Scientific Clarity**: Add notes on numerical stability and backend differences where applicable.
