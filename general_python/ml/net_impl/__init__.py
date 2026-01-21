"""
Network implementation subpackage.

Provides lazy access to core helpers and common network classes without
importing JAX/Flax until needed.
"""

import importlib

_LAZY_MODULES = {
    "activation_functions"  : "general_python.ml.net_impl.activation_functions",
    "interface_net_flax"    : "general_python.ml.net_impl.interface_net_flax",
    "net_general"           : "general_python.ml.net_impl.net_general",
    "net_simple"            : "general_python.ml.net_impl.net_simple",
    "networks"              : "general_python.ml.net_impl.networks",
    "utils"                 : "general_python.ml.net_impl.utils",
}

_LAZY_ATTRS = {
    "RBM"                   : "general_python.ml.net_impl.networks.net_rbm",
    "PairProduct"           : "general_python.ml.net_impl.networks.net_pp",
    "CNN"                   : "general_python.ml.net_impl.networks.net_cnn",
    "ResNet"                : "general_python.ml.net_impl.networks.net_res",
    "ComplexAR"             : "general_python.ml.net_impl.networks.net_autoregressive",
}

def __getattr__(name):
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_LAZY_MODULES.keys()) + list(_LAZY_ATTRS.keys())

# -------------------------------------
#! EOF
# -------------------------------------