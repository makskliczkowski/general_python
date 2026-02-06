# Future Structural Improvements

This document outlines potential structural improvements for the `general_python` package.

## Package Layout ("Flat" vs `src`)

### Current State
The project currently uses a "flat" layout where the root directory `.` is mapped to the package `general_python`.
Submodules like `algebra`, `maths`, `ml` are located in the root directory and become subpackages `general_python.algebra`, etc.

**Implications:**
*   `pyproject.toml` explicitly lists all subpackages to ensure they are discovered by `setuptools` (since `find_packages(where='.')` would find them as top-level packages, but we want them under `general_python`).
*   Namespace pollution in the root directory (mixing source code, config files, docs, scripts).
*   Editable installs require careful configuration (`package-dir = {"general_python": "."}`).
*   Requires `recursive-include` in `MANIFEST.in` to ensure source files are included.

### Proposal: Move to `src/` Layout
Refactor the repository to use the standard `src/` layout:

```text
general_python/
├── pyproject.toml
├── src/
│   └── general_python/
│       ├── __init__.py
│       ├── algebra/
│       ├── maths/
│       └── ...
├── tests/
├── docs/
└── ...
```

**Benefits:**
*   Cleaner root directory.
*   Standard auto-discovery of packages using `find_packages(where='src')`.
*   Avoids accidental import of the local folder as a package when running scripts from root (forces testing against installed package).
*   Simplifies `pyproject.toml` configuration.

## Missing `__init__.py` Files

Some subdirectories were found missing `__init__.py` files in previous versions (e.g. `common/embedded`, `physics/sp`).
These should be maintained to ensure proper subpackage traversal if they contain importable code.
Current audit shows they are present.

## Dependency Management

### Optional Dependencies
The `maths` module has a runtime dependency on `pandas` (guarded).
The `lattices` module has a hard dependency on `h5py` (via `common.hdf5man`), which was added to core dependencies during the audit.
Future work could refine these:
1.  **Split heavy modules**: Move `pandas`-dependent functions to `general_python.data_utils` or similar.
2.  **Lazy Loading**: Ensure heavy dependencies like `tensorflow` or `jax` are only imported when specific functions are called, not at module level (mostly handled by `LazyImporter` in `__init__.py` but check internal modules).

## CI/CD Pipeline

### Current State
There is no automated CI configuration (e.g. `.github/workflows`).
Documentation is built on ReadTheDocs (`.readthedocs.yaml` exists).

### Proposal: Add GitHub Actions
Add a `.github/workflows/ci.yml` to run tests on push/PR:
*   Checkout code.
*   Set up Python.
*   Install dependencies: `pip install .[dev,ml,jax]`.
*   Run tests: `pytest`.
*   Linting: `flake8` / `black`.

## Circular Imports and Lazy Loading
The project relies heavily on `LazyImporter` and `lazy_import` helpers in `__init__.py` files.
While this reduces startup time, it can hide import errors until runtime.
Regular verification scripts (like `test_documentation.py`) are essential to catch these issues early.
