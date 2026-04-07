"""
Network implementations (lazy exports).

This module exposes the concrete implementation modules while keeping import
overhead low. 

Reusable backbones stay here, while ansatz-oriented (Variational Monte Carlo) wrappers are
also available under ``general_python.ml.net_impl.ansatze``.
"""

import importlib
from typing import Any

_LAZY_IMPORTS = {
    # General-purpose networks
    "FlaxSimpleNet"         : (".net_simple_flax", "FlaxSimpleNet"),
    "RBM"                   : (".net_rbm", "RBM"),
    "CNN"                   : (".net_cnn", "CNN"),
    "ResNet"                : (".net_res", "ResNet"),
    "MLP"                   : (".net_mlp", "MLP"),
    "GCNN"                  : (".net_gcnn", "GCNN"),
    "Transformer"           : (".net_transformer", "Transformer"),
    # Ansatze that are designed for Variational Monte Carlo rather than being general-purpose networks.
    "AnsatzStacked"         : (".net_stacked", "AnsatzStacked"),
    "ComplexAR"             : ("..ansatze.autoregressive", "ComplexAR"),
    "PairProduct"           : ("..ansatze.pair_product", "PairProduct"),
    "Jastrow"               : ("..ansatze.jastrow", "Jastrow"),
    "MPS"                   : ("..ansatze.mps", "MPS"),
    "AmplitudePhase"        : ("..ansatze.amplitude_phase", "AmplitudePhase"),
    "AnsatzApproxSymmetric" : ("..ansatze.approx_symmetric", "AnsatzApproxSymmetric"),
    "EquivariantGCNN"       : ("..ansatze.equivariant_gcnn", "EquivariantGCNN"),
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
