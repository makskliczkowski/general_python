"""
NQS-oriented ansatz exports for ``general_python.ml.net_impl``.

The concrete implementations live in *general_python* modules under
``net_impl.networks``. This package provides the
cleaner namespace for wavefunction-specific architectures.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
--------------------------------
"""

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "ComplexAR"             : (".autoregressive", "ComplexAR"),
    "PairProduct"           : (".pair_product", "PairProduct"),
    "Jastrow"               : (".jastrow", "Jastrow"),
    "MPS"                   : (".mps", "MPS"),
    "AmplitudePhase"        : (".amplitude_phase", "AmplitudePhase"),
    "AnsatzApproxSymmetric" : (".approx_symmetric", "AnsatzApproxSymmetric"),
    "EquivariantGCNN"       : (".equivariant_gcnn", "EquivariantGCNN"),
}

_LAZY_CACHE = {}


def __getattr__(name: str) -> Any:
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name  = _LAZY_IMPORTS[name]
    module                  = importlib.import_module(module_path, package=__name__)
    value                   = getattr(module, attr_name)
    _LAZY_CACHE[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


__all__ = sorted(_LAZY_IMPORTS)

# ----------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------