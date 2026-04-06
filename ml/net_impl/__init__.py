"""
Network implementation subpackage.

Provides lazy access to the general network wrappers and helper utilities
without importing JAX or Flax until needed.

The wrappers in this package stay general-purpose. They can be used directly
outside NQS, while NQS-specific fast paths are selected separately in the NQS
layer. Input state conventions are configured explicitly through wrapper
arguments such as ``input_spin`` and ``input_value`` instead of relying on
hidden backend remaps in hot paths.
"""

import importlib

_LAZY_MODULES = {
    "activation_functions"  : ".activation_functions",
    "interface_net_flax"    : ".interface_net_flax",
    "net_general"           : ".net_general",
    "net_simple"            : ".net_simple",
    "networks"              : ".networks",
    "utils"                 : ".utils",
}

_LAZY_ATTRS = {
    "RBM"                   : ".networks.net_rbm",
    "PairProduct"           : ".networks.net_pp",
    "CNN"                   : ".networks.net_cnn",
    "ResNet"                : ".networks.net_res",
    "ComplexAR"             : ".networks.net_autoregressive",
    "GCNN"                  : ".networks.net_gcnn",
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
