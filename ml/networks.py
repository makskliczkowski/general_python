'''
file    : general_python/ml/net_simple.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl

This module has all the network implementations.
'''

from typing import Union, Optional, Any, Type
from enum import Enum, unique, auto
from general_python.ml.net_impl.net_general import GeneralNet, CallableNet
from general_python.ml.net_impl.net_simple import SimpleNet
from general_python.ml.net_impl.interface_net_flax import FlaxInterface


######################################################################

class Networks(Enum):
    """
    Enum class for different types of neural networks.
    """
    SIMPLE      = auto()

######################################################################

def choose_network(network_type : Union[str, Networks, Type[Any]], 
                input_shape     : Optional[tuple]   = None,
                backend         : Optional[str]     = None,
                dtype                               = None,
                *args, **kwargs) -> Any:
    """
    Choose and instantiate a network based on the provided type.

    Parameters
    ----------
    network_type : Union[str, Networks, Type[Any]]
        The network type to instantiate. This can be provided as:
            - A string (e.g., "simple")
            - A Networks enum member (e.g., Networks.SIMPLE)
            - A network class that is a subclass of GeneralNet.
            - A callable custom network factory.
            - An already instantiated network.
    input_shape : Optional[tuple], default=None
        The shape of the input to the network.
    backend : Optional[str], default=None
        The backend to be used (e.g., 'numpy' or 'jax').
    dtype : optional
        The data type for the network parameters.
    *args
        Additional positional arguments for network construction.
    **kwargs
        Additional keyword arguments for network construction.

    Returns
    -------
    Any
        An instance of the chosen network.

    Raises
    ------
    ValueError
        If the provided network type is unknown or invalid.
    """
    
    # If a string is provided, convert it to the corresponding Networks enum member.
    if isinstance(network_type, str):
        try:
            network_type_enum = Networks[network_type.upper()]
        except KeyError:
            raise ValueError(f"Unknown network type string: {network_type}")
        return choose_network(network_type_enum, *args, **kwargs)

    # If a Networks enum is provided, match to its corresponding implementation.
    if isinstance(network_type, Networks):
        if network_type == Networks.SIMPLE:
            return SimpleNet(input_shape=input_shape, backend=backend, dtype=dtype, *args, **kwargs)
        else:
            raise ValueError(f"Unknown network type enum: {network_type}")

    # If the provided network_type is already an instance of GeneralNet, return it.
    if isinstance(network_type, GeneralNet):
        return network_type

    # If a network class is provided.
    if isinstance(network_type, type):
        if issubclass(network_type, GeneralNet):
            return network_type(
                input_shape=input_shape, backend=backend, dtype=dtype, *args, **kwargs
            )
        else:
            return CallableNet(
                input_shape=input_shape, backend=backend, dtype=dtype, *args, **kwargs
            )

    # If network_type is callable but not a class (e.g., a custom network factory function).
    if callable(network_type):
        return CallableNet(
            input_shape=input_shape, backend=backend, dtype=dtype, *args, **kwargs
        )

    raise ValueError(f"Unknown network type: {network_type}")

######################################################################## 