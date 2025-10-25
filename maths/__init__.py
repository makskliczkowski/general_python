"""
Mathematics utilities for the `general_python` toolkit.

The package aggregates mathematical helpers, high-quality random number
generators, and statistical analysis routines.  Submodules are imported lazily
to keep top-level imports lightweight and consistent with the modular QES
design.

Available submodule aliases
---------------------------
- ``MathMod``      -> ``general_python.maths.math_utils``
- ``RandomMod``    -> ``general_python.maths.random``
- ``StatisticsMod``-> ``general_python.maths.statistics``

Examples
--------
>>> from general_python.maths import MathMod
>>> rng_matrix = MathMod.random_unitary(4)

>>> from general_python.maths import RandomMod, StatisticsMod
>>> samples = RandomMod.normal(mean=0.0, std=1.0, size=10_000)
>>> StatisticsMod.Statistics.calculate_fluctuations(samples.reshape(100, -1), bin_size=5)

Author: Maksymilian Kliczkowski
License: MIT
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, List

# Description used by QES.registry
MODULE_DESCRIPTION = (
    "Mathematical utilities, random number generators, and statistical analysis tools."
)

# ---------------------------------------------------------------------------
# Lazy submodule registry

_ALIAS_TO_MODULE: Dict[str, str] = {
    "MathMod": "math_utils",
    "RandomMod": "random",
    "StatisticsMod": "statistics",
}

__all__: List[str] = [
    *tuple(_ALIAS_TO_MODULE.keys()),
    "get_module_description",
    "list_available_modules",
]

_DESCRIPTIONS: Dict[str, str] = {
    "MathMod": "Provides general mathematical functions and utilities.",
    "RandomMod": "High-quality pseudorandom number generators and CUE matrices.",
    "StatisticsMod": "Statistical functions and data analysis utilities.",
}

# ---------------------------------------------------------------------------

def _load_alias(name: str):
    """Import the requested maths submodule on first access."""
    module_path = _ALIAS_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = import_module(f".{module_path}", __name__)
    globals()[name] = module  # cache to avoid repeated imports
    return module


def __getattr__(name: str):
    """
    Provide lazy access to maths submodules using friendly aliases.

    Examples
    --------
    >>> from general_python.maths import MathMod
    >>> MathMod  # doctest: +SKIP
    <module 'general_python.maths.math_utils' ...>
    """
    return _load_alias(name)


def __dir__() -> List[str]:  # pragma: no cover - trivial shell helper
    return sorted(set(globals()) | set(_ALIAS_TO_MODULE))


def get_module_description(module_name: str) -> str:
    """
    Return a human-readable description for a maths submodule alias.

    Parameters
    ----------
    module_name : str
        Alias registered in :data:`_ALIAS_TO_MODULE`.
    """
    return _DESCRIPTIONS.get(module_name, "Module not found.")


def list_available_modules() -> List[str]:
    """
    Return the list of available maths submodule aliases.

    Returns
    -------
    list of str
        Sorted list of alias names (e.g., ``['MathMod', 'RandomMod', ...]``).
    """
    return sorted(_ALIAS_TO_MODULE.keys())

#! EOF
