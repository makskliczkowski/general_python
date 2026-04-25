"""
Network implementation subpackage.

Provides lazy access to reusable backbone wrappers, helper utilities, and a
separate ansatz namespace without importing JAX or Flax until needed.

The wrappers in this package stay general-purpose. They can be used directly
by higher-level solvers, while solver-specific fast paths are selected outside
this package. Input state conventions are configured explicitly through wrapper
arguments such as ``input_is_spin`` and ``input_value`` instead of relying on
hidden backend remaps in hot paths.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
--------------------------------
"""

import importlib

_LAZY_MODULES = {
    "activation_functions"  : ".activation_functions",
    "ansatze"               : ".ansatze",
    "interface_net_flax"    : ".interface_net_flax",
    "net_general"           : ".net_general",
    "net_simple"            : ".net_simple",
    "networks"              : ".networks",
    "utils"                 : ".utils",
}

_LAZY_ATTRS = {
    # General-purpose networks
    "RBM"                   : ".networks.net_rbm",
    "CNN"                   : ".networks.net_cnn",
    "ResNet"                : ".networks.net_res",
    "GCNN"                  : ".networks.net_gcnn",
    # Networks that are designed as ansatze for Variational Monte Carlo rather than being general...
    "ComplexAR"             : ".ansatze.autoregressive",
    "PairProduct"           : ".ansatze.pair_product",
    "Jastrow"               : ".ansatze.jastrow",
    "MPS"                   : ".ansatze.mps",
    "AmplitudePhase"        : ".ansatze.amplitude_phase",
    "AnsatzApproxSymmetric" : ".ansatze.approx_symmetric",
    "EquivariantGCNN"       : ".ansatze.equivariant_gcnn",
}

def __getattr__(name):
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name], package=__name__)
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], package=__name__)
        return getattr(module, name)
    if name == "choose_network":
        from ..networks import choose_network
        return choose_network
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_LAZY_MODULES.keys()) + list(_LAZY_ATTRS.keys()) + ["choose_network"]

# -------------------------------------
#! EOF
# -------------------------------------
