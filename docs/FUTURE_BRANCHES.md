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

Some subdirectories were found missing `__init__.py` files, which prevented them from being treated as regular packages:
*   `general_python.common.embedded`
*   `general_python.physics.sp`

These have been patched by adding empty `__init__.py` files, but future modules should ensure `__init__.py` is present if they are intended to be importable packages.

## Optional Dependencies

The `maths` module has a hard dependency on `pandas` in `math_utils.py`.
This has been patched with a runtime guard, but structurally it might be better to:
1.  Move pandas-dependent functions to a separate submodule (e.g. `maths.statistics.dataframe_utils`).
2.  Or ensure `pandas` is a hard dependency if it is core to the library's function.

## Circular Imports and Lazy Loading
The project relies heavily on `LazyImporter` and `lazy_import` helpers in `__init__.py` files.
While this reduces startup time, it can hide import errors until runtime.
Regular verification scripts (like `test_documentation.py`) are essential to catch these issues early.

## Version Management

The version number is currently hardcoded in three places:
1.  `pyproject.toml`
2.  `__init__.py`
3.  `docs/conf.py`

This redundancy requires manual synchronization (e.g. bumping from 0.1.0 to 1.1.0).
**Proposal**: Use a single source of truth, such as reading `__init__.py` from `pyproject.toml` (dynamic metadata) or vice-versa.
