"""
Network implementation subpackage.

Provides lazy access to core helpers and common network classes without
importing JAX/Flax until needed.
"""

import importlib

_LAZY_MODULES = {
    "activation_functions"  : "QES.general_python.ml.net_impl.activation_functions",
    "interface_net_flax"    : "QES.general_python.ml.net_impl.interface_net_flax",
    "net_general"           : "QES.general_python.ml.net_impl.net_general",
    "net_simple"            : "QES.general_python.ml.net_impl.net_simple",
    "networks"              : "QES.general_python.ml.net_impl.networks",
    "utils"                 : "QES.general_python.ml.net_impl.utils",
}

_LAZY_ATTRS = {
    "RBM"                   : "QES.general_python.ml.net_impl.networks.net_rbm",
    "PairProduct"           : "QES.general_python.ml.net_impl.networks.net_pp",
    "CNN"                   : "QES.general_python.ml.net_impl.networks.net_cnn",
    "ResNet"                : "QES.general_python.ml.net_impl.networks.net_res",
    "ComplexAR"             : "QES.general_python.ml.net_impl.networks.net_autoregressive",
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