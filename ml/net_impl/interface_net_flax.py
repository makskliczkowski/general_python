
"""
FlaxNetInterface - A Generic Flax-Based Network Interface

This module provides a wrapper class for Flax neural network modules that 
conforms to the GeneralNet interface, allowing for seamless integration with
other network implementations.

The FlaxNetInterface class initializes any Flax module with provided arguments,
manages parameter initialization, and provides a consistent API for network
evaluation using both JAX and NumPy (where applicable).

Example:
    # Create a simple MLP using Flax
    class SimpleMLP(nn.Module):
        features: tuple = (64, 32, 1)
        
        @nn.compact
        def __call__(self, x):
            for feat in self.features[:-1]:
                x = nn.relu(nn.Dense(feat)(x))
            x = nn.Dense(self.features[-1])(x)
            return x
            
    # Wrap it with the interface
    net = FlaxNetInterface(
        net_module=SimpleMLP,
        net_kwargs={'features': (64, 32, 1)},
        input_shape=(10,)
    )
    
    # Use the network
    params = net.get_params()
    output = net(jnp.ones((1, 10)))

Attributes:
    _flax_module    : The internal Flax module instance
    _parameters     : The network parameters (after initialization)
    _initialized    : Boolean flag indicating whether the network is initialized
    _apply_jax      : Function for network evaluation using JAX
    _apply_np       : Function for network evaluation using NumPy (not supported)

Notes:
    - This interface only supports the JAX backend, as it's designed for Flax modules
    - The network parameters are automatically initialized upon instantiation
    - The apply function is set to use the Flax module's apply method
    - The input shape is expected to be a tuple representing the dimensions of the input
    - The dtype is set to jnp.float32 by default, but can be changed
    - The network can be evaluated using the apply_jax method, which requires the network to be initialized
"""

import numpy as np
from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core import unfreeze, freeze
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

# from general_python utilities
from general_python.ml.net_impl.net_general import GeneralNet
from general_python.ml.net_impl.activation_functions import get_activation

########################################################################
#! GENERIC FLAX NETWORK INTERFACE
########################################################################

class FlaxInterface(GeneralNet):
    """
    A generic Flax-based network interface that wraps any Flax module.
    
    This class derives from GeneralNet so that it shares the common interface
    (and backend logic) with other network implementations. The Flax module
    is passed as an argument, along with its constructor parameters, allowing
    you to swap in any Flax-based architecture.
    
    Parameters:
        net_module  : nn.Module | Callable
            A Flax module class implementing the network.
        net_args    : tuple
            Positional arguments to pass to the network module constructor.
        net_kwargs  : dict
            Keyword arguments to pass to the network module constructor.
        input_shape : tuple
            Input shape.
        backend     : str
            Backend to use ('jax', etc.). (Here, only JAX/Flax is supported.)
        dtype       : Optional[np.dtype]
            Data type for the network parameters.
    """
    _TOL_HOLOMORPHIC         = 1e-14
    _ERR_JAX_NECESSARY       = "JAX backend is necessary for this module."
    _ERR_NET_NOT_INITIALIZED = "Network not initialized. Call init() first."
    
    def __init__(self,
                net_module    : nn.Module | Callable,
                net_args      : tuple = (),
                net_kwargs    : Optional[dict] = None,
                input_shape   : tuple = (10,),
                backend       : str   = 'jax',
                dtype         : Optional[jnp.dtype] = jnp.float32):
        # call the parent initializer.
        super().__init__(input_shape, backend, dtype)
        
        # check the dtype.
        if not isinstance(self.dtype, jnp.dtype):
            self._dtype = jnp.dtype(self._dtype)
        
        if self._backend != jnp:
            raise ValueError(self._ERR_JAX_NECESSARY)
            
        if net_kwargs is None:
            net_kwargs = {}
            
        # try to get the activation from the kwargs.
        self._act_fun               = net_kwargs.get('act_fun', None)
        self._handle_activations(net_kwargs)
        
        # Create the internal Flax module.
        self._flax_module           = net_module(*net_args, **net_kwargs, dtype=self._dtype)
        
        # Initialize parameters to None; will be set in init().
        self._parameters            = None
        self._initialized           = False
        
        # Set the callable functions for evaluation.
        self._apply_jax             = self._flax_module.apply
        self._apply_np              = None
        
        # important to set
        self._holomorphic           = None
        self._has_analitic_grad     = False
        self._use_jax               = True
        
        self._compiled_grad_fn      = None
        self._compiled_apply_fn     = None
        
        self.init()
        
    ########################################################
    #! INITIALIZATION
    ########################################################
    
    def _handle_activations(self, net_kwargs: dict) -> None:
        '''
        Handle the activation functions for the network.
        This method checks if the activation functions are provided'
        in the kwargs. If they are, it initializes them.
        If not, it sets the activation functions to identity.
        Note:
            - The activation functions are stored in a tuple.
            - If there are not enough activation functions provided,
        Params:
            net_kwargs: dict
                The keyword arguments for the network module.
        '''
        if self._act_fun is not None:
            # initialize the activation functions
            self.init_activation()
            # reset the activation in the kwargs.
            net_kwargs['act_fun'] = self._act_fun
    
    def init_activation(self) -> None:
        """
        Initialize activation functions for the network.
        This method sets the activation functions for each layer
        based on the provided list or defaults to identity.
        
        Note:
            - The activation functions are stored in a tuple.
            - If there are not enough activation functions provided,
                the rest will be filled with identity functions.
            - The activation functions are initialized using the 
                get_activation function.
        """
        
        # Store activation functions.
        self._act_fun = tuple(get_activation(act, self._parameters, self._backend_str)[0]
                        for act in self._act_fun)
        
        # if len(self._act_fun) < len(self.layers) + 1:
        #     # If there are not enough activation functions, fill the rest with identity.
        #     self._act_fun += tuple(get_activation("identity", self._backend_str)[0]
        #                 for _ in range(len(self.layers) + 1 - len(self._act_fun)))
    
    def init(self, key: Optional[jax.random.PRNGKey] = None):
        """
        Initialize the network parameters using Flax.
        Params:
            key: jax.random.PRNGKey
                Random key for initialization. If None, a default key is used.
        """
        if key is None:
            key = random.PRNGKey(0)
        
        # create a dummy input for initialization.
        try:
            dummy_input             = jnp.ones((1, self.input_dim), dtype=self._dtype)
            variables               = self._flax_module.init(key, dummy_input)
            self._parameters        = freeze(variables)['params']
        except Exception as e:
            self._initialized       = False
            raise ValueError(f"Failed to initialize the network: {e}")
        
        self._initialized       = True
        # set the internal parameters shapes and tree structure.
        self._param_shapes      = [(p.size, p.shape) for p in tree_flatten(self._parameters)[0]]
        self._param_tree_def    = jax.tree_util.tree_structure(self._parameters)
        self._param_num         = jnp.sum(jnp.array([p.size for p in tree_flatten(self._parameters)[0]]))
        
        # set the compiled functions.
        self._compiled_apply_fn = jax.jit(
            lambda p, x: self._flax_module.apply({'params': p}, x)
        )
        self._compiled_grad_fn  = jax.jit(
            lambda p, x: jax.jacrev(lambda _p: jnp.log(self._flax_module.apply({'params': _p}, x)))(p)
        )
        
        return self._parameters
    
    ########################################################
    #! SETTERS
    ########################################################
    
    def set_params(self, params: dict):
        """
        Set the network parameters. Now the parameters are
        unfrozen and set to the network.
        """
        self._parameters = params
    
    def get_params(self):
        """
        Get the current network parameters.
        """
        return self._parameters
        
    ########################################################
    #! EVALUATION
    ########################################################
    
    def apply_jax(self, x: 'array-like'):
        """
        Evaluate the network on input x using Flax (and JAX).
        Params:
            x: array-like
                Input data to evaluate the network.
        Returns:
            array-like
                Output of the network.
        """
        if not self.initialized:
            self.init()
        return self._flax_module.apply({'params': self._parameters}, x)
        
    def get_apply(self, use_jax = False) -> Tuple[Callable, dict]:
        """
        Return the apply function and current parameters.
        Params:
            use_jax: bool
                not used here as the flax module is always jax.
        Returns:
            Tuple[Callable, dict]
                The apply function and the current parameters.
        """
        if not self._use_jax:
            raise ValueError(self._ERR_JAX_NECESSARY)
        return self._compiled_apply_fn, self.get_params()
    
    def get_gradient(self, use_jax = False):
        '''
        Get the gradient of the network.
        Params:
            use_jax: bool
                not used here as the flax module is always jax.
        Returns:
            Tuple[Callable, dict]
                The apply function and the current parameters.
        '''
        if self._use_jax:
            return self._compiled_grad_fn
        raise ValueError(self._ERR_JAX_NECESSARY)
    
    #########################################################
    #! STRING REPRESENTATION
    #########################################################
    
    def __repr__(self) -> str:
        return (f"Intefrace[flax](input_dim={self.input_dim}, dtype={self._dtype}, "
                f"flax_module={self._flax_module.__class__.__name__})")
        
    def __str__(self) -> str:
        return (f"FlaxNetInterface(input_dim={self.input_dim}, dtype={self._dtype}, "
                f"flax_module={self._flax_module.__class__.__name__})")
        
    #########################################################
    #! CALL ME MAYBE
    #########################################################
    
    @jax.jit
    def __call__(self, x: 'array-like'):
        """
        Call the network on input x.
        This is a wrapper around the Flax module's apply method.
        Params:
            x: array-like
                Input data to evaluate the network.
        Returns:
            array-like
                Output of the network.
        """
        return self._compiled_apply_fn(self._parameters, x)

    #########################################################
    #! CHECK HOLOMORPHICITY
    #########################################################
    
    def check_holomorphic(self) -> bool:
        """
        Check if the network is holomorphic.
        This is done by checking if the gradients of the real
        and imaginary parts of the output are equal.
        If they are, the network is holomorphic.
        Returns:
            bool: True if the network is holomorphic, False otherwise.
        """
        
        # Check if the network is initialized.
        if not self._initialized:
            self.init()
        
        # Check if the network is already checked.
        if self._holomorphic is not None:
            return self._holomorphic
        
        # Check if the network is holomorphic.
        dummy_input     = jnp.ones((1, self.input_dim), dtype=self._dtype)

        # Flatten the parameters tree into a 1D array.
        def make_flat(x):
            leaves, _   = tree_flatten(x)
            return jnp.concatenate([p.ravel() for p in leaves])
        
        # Compute gradients of the real and imaginary parts.
        grads_real          = make_flat(jax.grad(lambda a,b: jnp.real(self._flax_module.apply(a,b)[0, 0]))(self._parameters, dummy_input)["params"])
        grads_imag          = make_flat(jax.grad(lambda a,b: jnp.imag(self._flax_module.apply(a,b)[0, 0]))(self._parameters, dummy_input)["params"] )

        # Flatten the gradients.
        flat_real           = make_flat(grads_real)
        flat_imag           = make_flat(grads_imag)
        
        norm_diff           = jnp.linalg.norm(flat_real - 1.j * flat_imag) / flat_real.shape[0]
        self._holomorphic   = jnp.isclose(norm_diff, 0.0, atol = self._TOL_HOLOMORPHIC)
        return self._holomorphic

    #########################################################
    
    @property
    def activations(self):
        """
        Get the activation functions for the network.
        Returns:
            tuple: A tuple of activation functions for each layer.
        """
        return self._activations
    
    #########################################################
    
#############################################################