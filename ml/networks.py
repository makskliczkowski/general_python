"""
QES.general_python.ml.networks
==============================

Network Factory and Registry.

This module provides a centralized factory function, `choose_network`, for
instantiating various neural network architectures used in the QES framework.
It uses a lazy-loading mechanism to improve startup performance.

Usage
-----
Import and use the factory to create a network. The factory takes the
network type and common parameters like `input_shape` and `dtype`.
Network-specific parameters are passed as keyword arguments.

    from QES.general_python.ml.networks import choose_network
    
    # Create an RBM using 'alpha' (hidden unit density)
    rbm_net = choose_network(
        'rbm',
        input_shape=(10,),
        alpha=2.0,          # Creates 2*10=20 hidden units
        dtype='complex64'
    )
    
    # Create a CNN for an 8x8 lattice
    cnn_net = choose_network(
        'cnn',
        input_shape=(64,),
        reshape_dims=(8, 8),
        features=[8, 16],
        kernel_sizes=[3, 3]
    )

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maksymilian.kliczkowski@pwr.edu.pl
Date            : 01.10.2025
Description     : Factory for creating neural network instances.
----------------------------------------------------------
"""
import importlib
import numpy as np
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
    'simple' : ('.net_impl.net_simple',                     'SimpleNet'),
    'rbm'    : ('.net_impl.networks.net_rbm',               'RBM'),
    'cnn'    : ('.net_impl.networks.net_cnn',               'CNN'),
    'ar'     : ('.net_impl.networks.net_autoregressive',    'ComplexAR'),
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
                   input_shape  : Optional[tuple]   = None,
                   backend      : str               = 'jax',
                   dtype        : Any               = None,
                   param_dtype  : Any               = None,
                   seed         : Optional[int]     = None,
                   **kwargs) -> GeneralNet:
    """
    Smart factory to instantiate a network.
    
    This factory can create networks by name (e.g., 'rbm', 'cnn'), wrap raw Flax modules,
    or instantiate custom `GeneralNet` subclasses. It handles network-specific arguments
    passed via `**kwargs` and provides conveniences like `alpha` for RBMs.

    Parameters
    ----------
    network_type : Union[str, Networks, Type, Any]
        - **String/Enum**       : 'rbm', 'cnn', 'simple', 'ar'. The factory will lazy-load and instantiate the corresponding class.
        - **Flax Module Class** : A raw `flax.linen.nn.Module` class. The factory will automatically wrap it in a `FlaxInterface`.
        - **GeneralNet Class**  : A class inheriting from `GeneralNet`. The factory will instantiate it.
        - **Instance**          : If an already-initialized network instance is passed, it is returned as-is.
    input_shape : Optional[tuple]
        The shape of the input to the network, e.g., `(n_spins,)`.
    backend : str
        The computational backend to use ('jax' or 'numpy'). Defaults to 'jax'.
    dtype : Any
        The data type for the network's computations (e.g., 'float32', 'complex64').
    param_dtype : Any
        The data type for the network's parameters. If `None`, defaults to `dtype`.
    seed : Optional[int]
        Random seed for network initialization, if applicable.
    **kwargs :
        Network-specific keyword arguments. See below for details on each network type.

    Using Custom Flax Modules
    -------------------------
    You can pass your own `flax.linen.nn.Module` class as the `network_type`.
    The factory will wrap it in a `QES.general_python.ml.net_impl.interface_net_flax.FlaxInterface`
    to make it compatible with the QES ecosystem.

    **Requirements for your custom module:**
    1.  It must be a valid `nn.Module`.
    2.  Its `__call__` method should accept a `(batch, n_visible)` JAX array as input.
    3.  It should return the log-amplitude of the wavefunction, typically with shape `(batch,)`.

    **Example:**
        >>> import flax.linen as nn
        >>> import jax.numpy as jnp
        >>>
        >>> class MyCustomNet(nn.Module):
        ...     @nn.compact
        ...     def __call__(self, s):
        ...         # A simple dense layer
        ...         x = nn.Dense(features=128)(s)
        ...         x = nn.relu(x)
        ...         log_psi = nn.Dense(features=1)(x)
        ...         return jnp.squeeze(log_psi, axis=-1)
        >>>
        >>> # The factory handles the wrapping
        >>> custom_net = choose_network(
        ...     MyCustomNet,
        ...     input_shape=(100,),
        ...     dtype='complex64' # Passed to the interface
        ... )
        >>> print(custom_net)


    Keyword Args (by network_type)
    ------------------------------
    
    **For 'rbm'**:
        - `alpha` (float)                               : Hidden unit density. `n_hidden` will be `int(alpha * n_visible)`.
        - `n_hidden` (int)                              : Number of hidden units. If `alpha` is also given, `alpha` takes precedence.
        - `bias` (bool)                                 : Whether to use a bias for the hidden layer. Default: `True`.
        - `visible_bias` (bool)                         : Whether to use a bias for the visible layer. Default: `True`.
    
    **For 'cnn'**:
        - `reshape_dims` (Tuple[int, ...])              : The spatial dimensions to reshape the 1D input into (e.g., `(8, 8)`).
        - `features` (Sequence[int])                    : Number of output channels for each convolutional layer.
        - `kernel_sizes` (Sequence[Union[int, Tuple]])  : Size of the kernel for each conv layer.
        - `strides` (Sequence[Union[int, Tuple]])       : Stride for each conv layer. Defaults to 1.
        - `output_shape` (Tuple[int, ...])              : Shape of the final output. Default: `(1,)`.
        
    **For 'simple'**:
        - `layers` (Tuple[int, ...])                    : A tuple defining the number of neurons in each hidden layer.
        - `output_shape` (Tuple[int, ...])              : Shape of the final output. Default: `(1,)`.
        - `act_fun` (Tuple[Union[str, Callable],...])   : Activation functions for each layer.
        
    **For 'ar' (Autoregressive)**:
        - `depth` (int)                                 : Number of layers in the model.
        - `num_hidden` (int)                            : Number of hidden units in each layer.
        - `rnn_type` (str)                              : Type of recurrent cell, if applicable (e.g., 'lstm', 'gru').

    Returns
    -------
    GeneralNet
        An initialized or wrapped network instance compatible with the QES framework.
    """

    # 1. Handle Strings and Enums (Lazy Load Path)
    if isinstance(network_type, (str, Networks)):
        key     = str(network_type).lower()
        
        # Argument Pre-processing for convenience
        if key == 'rbm':
            # Allow `alpha` or `hidden_density` to define `n_hidden`
            alpha = kwargs.pop('alpha', None) or kwargs.pop('hidden_density', None)
            
            if alpha is not None and 'n_hidden' not in kwargs:
                if not input_shape:
                    raise ValueError("`input_shape` must be provided when using `alpha` for RBM.")
                n_visible           = np.prod(input_shape)
                kwargs['n_hidden']  = int(alpha * n_visible)

        net_cls = _lazy_load_class(key)
        return net_cls(input_shape=input_shape, backend=backend, dtype=dtype, param_dtype=param_dtype, seed=seed, **kwargs)

    # 2. Handle Existing Instances (Return as-is)
    if isinstance(network_type, GeneralNet):
        return network_type

    # 3. Handle Types/Classes
    if isinstance(network_type, type):
        
        # It is a subclass of GeneralNet (e.g. user imported RBM manually)
        if issubclass(network_type, GeneralNet):
            return network_type(input_shape=input_shape, backend=backend, dtype=dtype, seed=seed, **kwargs)

        # It is a Flax Module (Auto-Wrap Logic)
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
            # We pass all kwargs to the FlaxInterface, which will pass them to the module
            all_kwargs = kwargs.copy()
            all_kwargs.pop('input_shape',   None)  # Remove if present in kwargs
            all_kwargs.pop('backend',       None)
            all_kwargs.pop('dtype',         None)
            all_kwargs.pop('param_dtype',   None)
            all_kwargs.update({
                'input_shape'   : input_shape,
                'backend'       : backend,
                'dtype'         : dtype,
                'param_dtype'   : param_dtype,
            })
            return FlaxInterface(
                net_module      =   network_type,
                net_kwargs      =   all_kwargs,
                input_shape     =   input_shape,
                backend         =   backend,
                dtype           =   dtype,
                param_dtype     =   param_dtype,
                seed            =   seed,
                in_activation   =   kwargs.get('in_activation', None)
            )

    # Handle generic Callables (Factories)
    if callable(network_type):
        return CallableNet(callable_fun=network_type, input_shape=input_shape, backend=backend, dtype=dtype, **kwargs)

    raise ValueError(f"Unknown network type: {type(network_type)}")

######################################################################
#! END OF FILE
######################################################################