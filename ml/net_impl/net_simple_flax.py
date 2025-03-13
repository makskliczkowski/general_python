"""
file    : general_python/ml/net_impl/net_simple_flax.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-03-10
"""

import numpy as np
import general_python.ml.net_impl.net_general as _net_general
from typing import Optional, Tuple, Callable

# import from general python module
from general_python.algebra.utils import _JAX_AVAILABLE

if not _JAX_AVAILABLE:
    raise ImportError("JAX is not available. Please install JAX to use this module.")

# jax imports
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn

# import the initializers
from general_python.ml.net_impl.activation_functions import get_activation
from general_python.ml.net_impl.utils.net_init_jax import complex_he_init, real_he_init
from general_python.ml.net_impl.interface_net_flax import FlaxInterface

##########################################################
#! EXAMPLE FLAX MODULE
##########################################################

class _FlaxNet(nn.Module):
    """
    Inner Flax module that implements a simple feedforward network.
    
    Attributes:
        layers      : Tuple[int, ...]       -- hidden layer sizes.
        output_dim  : int                   -- output dimension.
        bias        : bool                  -- whether to include bias.
        act_fun     : Tuple[Callable, ...]  -- activations to use (if too few, identity is used).
        dtype       : jnp.dtype             -- data type of the parameters.
    """
    layers          : Tuple[int, ...]
    output_dim      : int                   = 1
    bias            : bool                  = True
    act_fun         : Tuple[Callable, ...]  = (jax.nn.relu,)
    dtype           : jnp.dtype             = jnp.float32

    @nn.compact
    def __call__(self, x):
        s           = x
        num_layers  = len(self.layers)
        
        # Extend activation functions if not enough are provided.
        if len(self.act_fun) < num_layers:
            act_funs = self.act_fun + (lambda x: x,) * (num_layers - len(self.act_fun))
        else:
            act_funs = self.act_fun[:num_layers]
        
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            # Use complex He initialization for complex weights.
            kernel_init_fn = complex_he_init
        else:
            # Use real He initialization for real weights.
            kernel_init_fn = real_he_init
        
        # Hidden layers.
        for neurons, act in zip(self.layers, act_funs):
            s = nn.Dense(neurons, use_bias=self.bias, kernel_init=kernel_init_fn,
                        dtype=self.dtype, param_dtype=self.dtype)(s)
            s = act(s)
            
        # Final output layer.
        s = nn.Dense(self.output_dim, use_bias=self.bias, kernel_init=kernel_init_fn,
                    dtype=self.dtype, param_dtype=self.dtype)(s)
        return s

##########################################################
#! FLAX SIMPLE NET
##########################################################

class FlaxSimpleNet(FlaxInterface):
    """
    A Flax-based simple network that generates random (e.g. complex) outputs.
    
    This class derives from GeneralNet so that it shares the common interface
    (and backend logic) with other network implementations. Internally it uses
    a Flax module to manage parameters and function evaluation.
    
    Parameters:
        act_fun     : Tuple[Callable, ...]
            Activation functions to use after each hidden layer.
        input_shape : tuple
            Input shape.
        output_shape: tuple
            Output shape (if not a tuple, it is converted).
        layers      : tuple
            Hidden layer sizes.
        bias        : bool
            Whether to include bias terms.
        backend     : str
            Backend to use ('default', 'jax', etc.). (Here, only JAX/Flax is supported.)
        dtype       : Optional[np.dtype]
            Data type (use a complex type to get complex outputs).
    """
    
    def __init__(self,
                act_fun         : Optional[Tuple[Callable, ...]] = None,
                input_shape     : tuple = (10,),
                output_shape    : tuple = (1,),
                layers          : tuple = (1,),
                bias            : bool  = True,
                backend         : str   = 'jax',
                dtype           : Optional[np.dtype] = jnp.float32):
        # call the parent initializer.
        super().__init__(
            net_module  = _FlaxNet,
            net_args    = (layers, int(np.prod(output_shape))),
            net_kwargs  = {'bias': bias, 'act_fun': act_fun},
            input_shape = input_shape,
            backend     = backend,
            dtype       = dtype)

    #########################################################
    #! INFO
    #########################################################
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the FlaxSimpleNet object.
        """
        return f"FlaxSimpleNet(layers={self.net_kwargs['layers']}, input_shape={self.input_shape}, output_shape={self.output_shape}, dtype={self.dtype})"

    #########################################################
    
#############################################################

#! EXAMPLE USAGE
# from general_python.ml.net_impl.net_simple_flax import FlaxSimpleNet
# import jax.numpy as jnp

# # Define activation functions if needed; otherwise, the default is jax.nn.relu.
# act_functions = (jax.nn.relu, jax.nn.tanh)

# # Instantiate the network.
# net = FlaxSimpleNet(
#     act_fun      = act_functions,
#     input_shape  = (10,),
#     output_shape = (1,),
#     layers       = (20, 15),   # For instance, two hidden layers with 20 and 15 neurons.
#     bias         = True,
#     backend      = 'jax',
#     dtype        = jnp.float32
# )

# # Initialize parameters (this is done in __init__ but can be re-initialized if desired).
# params = net.get_params()

# # Evaluate the network on some input.
# import jax.numpy as jnp
# x = jnp.ones((5, 10))
# output = net.apply_jax(x)
# print(output)

def example():
    """
    Example usage of the FlaxSimpleNet class.
    
    This function demonstrates how to create an instance of the FlaxSimpleNet
    class and use it for forward propagation.
    """
    
    # Define activation functions if needed; otherwise, the default is jax.nn.relu.
    act_functions    = (jax.nn.relu, jax.nn.tanh)

    # Instantiate the network.
    net = FlaxSimpleNet(
        act_fun      = act_functions,
        input_shape  = (10,),
        output_shape = (1,),
        layers       = (20, 15),   # For instance, two hidden layers with 20 and 15 neurons.
        bias         = True,
        backend      = 'jax',
        dtype        = jnp.float32
    )

    # Initialize parameters (this is done in __init__ but can be re-initialized if desired).
    params = net.get_params()

    # Evaluate the network on some input.
    x       = jnp.ones((5, 10))
    output  = net.apply_jax(x)
    return output

######################################################################