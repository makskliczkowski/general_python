"""
Network implementations (lazy exports).

This module exposes all built-in ansatz classes while keeping import overhead low.
"""

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "FlaxSimpleNet"         : (".net_simple_flax", "FlaxSimpleNet"),
    "AnsatzApproxSymmetric" : (".net_approx_symmetric", "AnsatzApproxSymmetric"),
    "AnsatzStacked"         : (".net_stacked", "AnsatzStacked"),
    "RBM"                   : (".net_rbm", "RBM"),
    "CNN"                   : (".net_cnn", "CNN"),
    "ComplexAR"             : (".net_autoregressive", "ComplexAR"),
    "ResNet"                : (".net_res", "ResNet"),
    "PairProduct"           : (".net_pp", "PairProduct"),
    "MLP"                   : (".net_mlp", "MLP"),
    "GCNN"                  : (".net_gcnn", "GCNN"),
    "Jastrow"               : (".net_jastrow", "Jastrow"),
    "MPS"                   : (".net_mps", "MPS"),
    "Transformer"           : (".net_transformer", "Transformer"),
    "AmplitudePhase"        : (".net_amplitude_phase", "AmplitudePhase"),
}

_LAZY_CACHE = {}

# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        _LAZY_CACHE[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))

__all__ = sorted(_LAZY_IMPORTS.keys())

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------