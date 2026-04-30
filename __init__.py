"""Shared scientific utilities used by QES.

``QES.general_python`` is a lightweight facade over the reusable utility
packages used by the algebra, model, NQS, and plotting layers. It lazy-loads
subpackages so ``import QES`` and ``import QES.general_python`` stay cheap.

Public subpackages
------------------
- ``algebra``: backend-neutral algebra helpers, random utilities, and solvers.
- ``common``: logging, dtype conversion, HDF5/result containers, plotting glue.
- ``lattices``: lattice definitions, regions, neighbor tables, and k-space paths.
- ``maths``: numerical helper functions.
- ``physics``: entropy, density-matrix, spectral, and statistical utilities.
- ``ml``: generic neural-network and optimizer utilities used by NQS.

----------------------------------------------------------------------------
Author      : Maksymilian Kliczkowski
Date        : 2025-02-02
Version     : 1.1.0
Changelog   :
- 1.1.0: Added lazy loading for submodules and common exports, improved documentation, and added capability listing functions.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .          import algebra, common, lattices, maths, ml, physics
    from .algebra   import ran_wrapper as random
    from .common    import (
                        LazyDataEntry,
                        LazyHDF5Entry,
                        LazyJsonEntry,
                        LazyNpzEntry,
                        LazyPickleEntry,
                        PlotData,
                        ResultSet,
                        dtype_to_name,
                        filter_results,
                        load_results,
                    )

__version__         = "1.1.0"
MODULE_DESCRIPTION  = "Shared scientific utilities for QES."

_SUBMODULES         = {
    "algebra",
    "common",
    "lattices",
    "maths",
    "physics",
    "ml",
}

_ALIASES            = {
    "random": "algebra.ran_wrapper",
}

_COMMON_EXPORTS = {
    "dtype_to_name",
    "load_results",
    "filter_results",
    "ResultSet",
    "PlotData",
    "LazyDataEntry",
    "LazyHDF5Entry",
    "LazyNpzEntry",
    "LazyPickleEntry",
    "LazyJsonEntry",
}

_LAZY_CACHE: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy-load submodules on demand."""
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", package=__name__)
        _LAZY_CACHE[name] = module
        return module
    if name in _ALIASES:
        module = importlib.import_module(f".{_ALIASES[name]}", package=__name__)
        _LAZY_CACHE[name] = module
        return module
    if name in _COMMON_EXPORTS:
        common = importlib.import_module(".common", package=__name__)
        value = getattr(common, name)
        _LAZY_CACHE[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def list_capabilities() -> list[str]:
    """List available capabilities."""
    return sorted(set(__all__))


def __dir__() -> list[str]:
    """Return the maintained general_python package surface."""
    return sorted(set(__all__))


def list_available_modules() -> list[str]:
    """Return list of available submodules."""
    return sorted(_SUBMODULES)


def get_module_description(module_name: str) -> str:
    """Return a brief description of the module."""
    descriptions = {
        "algebra"       : "Algebraic structures, eigensolvers, random utilities, and backend operations.",
        "common"        : "Common utilities for logging, file I/O, plotting, and result containers.",
        "lattices"      : "Lattice definitions, regions, neighbor tables, and k-space utilities.",
        "maths"         : "Mathematical functions and numerical utilities.",
        "physics"       : "Entropy, density-matrix, spectral, and statistical physics tools.",
        "ml"            : "Machine-learning utilities and generic network implementations.",
        "random"        : "Compatibility alias for QES.general_python.algebra.ran_wrapper.",
    }
    return descriptions.get(module_name, "No description available.")

__all__ = sorted(set(_SUBMODULES) | set(_ALIASES) | set(_COMMON_EXPORTS) | {
        "MODULE_DESCRIPTION",
        "__version__",
        "get_module_description",
        "list_available_modules",
        "list_capabilities",
    }
)

# ----------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------
