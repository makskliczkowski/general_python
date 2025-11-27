'''
This module has all the network implementations.

---------------------------------------------------------------
file    : general_python/ml/networks.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
---------------------------------------------------------------

'''
import importlib
from typing import Union, Optional, Any, Type, Dict, Tuple, TYPE_CHECKING
from enum import Enum, auto

try:
    from .net_impl.net_general          import GeneralNet, CallableNet
except ImportError as e:
    raise ImportError(f"Could not import GeneralNet or CallableNet. "
                      f"Ensure that all dependencies are installed. Original error: {e}")

# Type checking import only (does not trigger runtime import)
if TYPE_CHECKING:
    from .net_impl.interface_net_flax   import FlaxInterface
    from .net_impl.net_simple           import SimpleNet
    from flax import linen as nn

######################################################################

class Networks(str, Enum):
    """
    Enum class for available standard network architectures.
    Inherits from str to allow string comparison.
    """
    SIMPLE = 'simple'
    RBM    = 'rbm'
    CNN    = 'cnn'
    AR     = 'ar'
    
    def __str__(self):
        return self.value

######################################################################
# LAZY REGISTRY
# Maps 'string_key' -> ('module.path', 'ClassName')
######################################################################

_NETWORK_REGISTRY: Dict[str, Tuple[str, str]] = {
    'simple' : ('.net_impl.net_simple', 'SimpleNet'),
    'rbm'    : ('.net_impl.networks.net_rbm', 'RBM'),
    'cnn'    : ('.net_impl.networks.net_cnn', 'CNN'),
    'ar'     : ('.net_impl.networks.net_autoregressive', 'Autoregressive'),
    # Add future networks here without importing them!
}

def _lazy_load_class(key: str) -> Type[GeneralNet]:
    """Helper to import network classes only when requested."""
    if key not in _NETWORK_REGISTRY:
        raise ValueError(f"Network '{key}' is not registered in QES.")
    
    mod_path, cls_name  = _NETWORK_REGISTRY[key]
    try:
        # Relative import requires the package context
        # We assume this file is in QES.general_python.ml
        module          = importlib.import_module(mod_path, package='QES.general_python.ml')
        return getattr(module, cls_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to lazy load '{cls_name}' from '{mod_path}'.\nError: {e}")

######################################################################

def choose_network(network_type : Union[str, Networks, Type[Any], Any], 
                   input_shape  : Optional[tuple] = None,
                   backend      : str             = 'jax',
                   dtype        : Any             = None,
                   *args, **kwargs) -> GeneralNet:
    """
    Smart factory to instantiate a network.
    
    Capabilities:
    1. String/Enum lookup (Lazy Loaded).
    2. Auto-wrapping of raw Flax Modules.
    3. instantiation of custom GeneralNet classes.

    Parameters
    ----------
    network_type : Union[str, Networks, Type, Any]
        - String: 'rbm', 'simple'
        - Flax Class: MyFlaxModule (will be auto-wrapped)
        - GeneralNet Class: RBM (will be instantiated)
        - Instance: Returned as-is.
    """

    # 1. Handle Strings and Enums (Lazy Load Path)
    if isinstance(network_type, (str, Networks)):
        key     = str(network_type).lower()
        net_cls = _lazy_load_class(key)
        return net_cls(input_shape=input_shape, backend=backend, dtype=dtype, *args, **kwargs)

    # 2. Handle Existing Instances (Return as-is)
    if isinstance(network_type, GeneralNet):
        return network_type

    # 3. Handle Types/Classes
    if isinstance(network_type, type):
        
        # A. It is a subclass of GeneralNet (e.g. user imported RBM manually)
        if issubclass(network_type, GeneralNet):
            return network_type(input_shape=input_shape, backend=backend, dtype=dtype, *args, **kwargs)

        # B. It is a Flax Module (Auto-Wrap Logic)
        # We check this loosely to avoid importing flax if not needed
        is_flax = False
        try:
            import flax.linen as nn
            if issubclass(network_type, nn.Module):
                is_flax = True
        except ImportError:
            pass

        if is_flax:
            # Lazy import the interface wrapper
            from .net_impl.interface_net_flax import FlaxInterface
            
            # The network_type here IS the Flax Module class
            return FlaxInterface(
                net_module  =   network_type,
                net_args    =   args,       # Pass positional args to Flax __init__
                net_kwargs  =   kwargs,     # Pass kwargs to Flax __init__
                input_shape =   input_shape,
                backend     =   backend,
                dtype       =   dtype
            )

    # 4. Handle generic Callables (Factories)
    if callable(network_type):
        return CallableNet(input_shape=input_shape, backend=backend, dtype=dtype, *args, **kwargs)

    raise ValueError(f"Unknown network type: {type(network_type)}")

######################################################################
#! END OF FILE
######################################################################