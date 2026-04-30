"""Reusable scientific Python utilities for numerical research.

``general_python`` collects backend-aware algebra, lattice, physics, plotting,
and machine-learning helpers behind a small lazy-loading facade. The package is
intended to be usable as an open library: importing the top level is cheap,
optional dependencies are loaded only when their subpackages are accessed, and
public helpers expose stable names for interactive work and documentation.

Public Subpackages
------------------
algebra
    Backend-neutral linear algebra, random utilities, ODE integrators, and
    iterative solvers.
common
    Logging, plotting, file IO, result containers, lazy data entries, and
    runtime helpers.
lattices
    Lattice definitions, neighbor tables, reciprocal-space paths, and region
    utilities.
maths
    Numerical helper functions, statistics, fitting, and random-matrix tools.
physics
    Density-matrix, entropy, operator, spectral, thermal, and statistical
    physics utilities.
ml
    Lightweight neural-network, scheduler, and training-phase utilities.

Notes
-----
The top-level module uses :func:`__getattr__` to load submodules on demand.
Prefer importing concrete APIs from their subpackages in library code, e.g.
``from general_python.algebra import choose_solver``.
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
    """Resolve a lazily exported submodule or compatibility alias.

    Parameters
    ----------
    name
        Public attribute requested from :mod:`general_python`.

    Returns
    -------
    Any
        Imported module or exported object.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the maintained public surface.
    """
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
    """Return the names exported by the top-level package."""
    return sorted(set(__all__))


def __dir__() -> list[str]:
    """Return the maintained general_python package surface."""
    return sorted(set(__all__))


def list_available_modules() -> list[str]:
    """Return lazily importable public subpackage names."""
    return sorted(_SUBMODULES)


def get_module_description(module_name: str) -> str:
    """Return a one-line description for a public submodule.

    Parameters
    ----------
    module_name
        Submodule or alias name, for example ``"algebra"`` or ``"random"``.
    """
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
