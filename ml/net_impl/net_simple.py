"""
file    : general_python/ml/net_impl/net_simple.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-03-10

A simple network implementation that generates random complex outputs.

This class simulates a neural network by generating random complex numbers
for each input sample, regardless of the input values themselves.

Methods:
    apply(x): Applies the "network" to input data by generating random complex numbers.
    __call__(x): Wrapper for apply method to make the class instance callable.

Parameters:
    x: Input data array. Can be a single sample or a batch.

Returns:
    For a single input sample: A single complex random number.
    For a batch of inputs: An array of complex random numbers, one per sample.

Note:
    This implementation doesn't actually use the input values for computation;
    it simply generates random outputs based on the batch size.
"""

import numpy as np
import numba
from functools import partial
import general_python.ml.net_impl.net_general as _net_general
from typing import Optional, Tuple, Callable

# import from general python module
from general_python.algebra.utils import JAX_AVAILABLE, get_backend
from general_python.ml.net_impl.activation_functions import get_activation

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import random

##################################################################

# @numba.njit
def apply(params, x: 'array-like', activations) -> np.ndarray:
    """
    Apply the network to an input batch using NumPy
    
    Parameters:
        x : np.ndarray
            Input array of shape (batch, input_dim)
    
    Returns:
        np.ndarray: One output per sample.
    """
    
    # Apply each hidden layer.
    s           = x
    for (w, b), act in zip(params['layers'], activations):
        # Apply weights and bias.
        s = np.dot(s, w) + b
        s = act(s)
    return s

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(2,))
    def apply_jax(params, x: 'array-like', activations) -> jnp.ndarray:
        """
        Apply the network to an input batch using JAX
        
        Parameters:
            x : jnp.ndarray
                Input array of shape (batch, input_dim)
        
        Returns:
            jnp.ndarray: One output per sample.
        """
        
        # Apply each hidden layer.
        s           = x
        for (w, b), act in zip(params["layers"], activations):
            # Apply weights and bias.
            s = jnp.dot(s, w) + b
            s = act(s)
        return s

def apply_wrapper(activations, use_jax: bool):
    '''
    Wrapper function to apply the network using either NumPy or JAX.
    '''
    
    if use_jax and JAX_AVAILABLE:
        @jax.jit
        def apply_jax_in(params, x):
            """
            Apply the network to an input batch using JAX
            
            Parameters:
                x : jnp.ndarray
                    Input array of shape (batch, input_dim)
            
            Returns:
                jnp.ndarray: One output per sample.
            """
            return apply_jax(params, x, activations)
        return apply_jax_in
    
    def apply_in(params, x):
        """
        Apply the network to an input batch using NumPy
        
        Parameters:
            x : np.ndarray
                Input array of shape (batch, input_dim)
        
        Returns:
            np.ndarray: One output per sample.
        """
        return apply(params, x, activations)
    return apply_in

###################################################################
#! SIMPLE NETWORK
###################################################################

class SimpleNet(_net_general.GeneralNet):
    """
    A simple network implementation that generates random complex outputs.

    This class simulates a neural network by generating random outputs using random
    initializations of weights and biases. It can operate using either NumPy or JAX
    backends, depending on the provided backend argument.
    """

    def __init__(self,
                act_fun		: Optional[Tuple]   = None,
                input_shape	: tuple		        = (10,),
                output_shape: tuple		        = (1,),
                layers		: tuple		        = (1,), 
                bias		: bool		        = True, 
                backend		: str		        = 'default',
                dtype       : Optional[np.dtype]= np.float64,
                ):
        """ 
        Initialize a simple feed-forward network.

        Parameters:
            input_shape	: tuple
                            Dimension of the input.
            output_shape: tuple
                            Dimension of the output.          
            layers		: tuple
                            A tuple defining the number of neurons per hidden layer.
            bias		: bool
                            Whether to include bias terms.
            act_fun		: tuple
                            Activation functions for NumPy.
            jax_act_fun	: tuple
                            Activation functions for JAX.
            backend		: str
                            Backend to use ('default', 'numpy', or 'jax').
        """
        super().__init__(input_shape, backend, dtype)

        # Use the first element of input_shape as input_dim.
        self._output_shape      = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self._output_dim        = np.prod(self._output_shape)
        self.layers		        = layers
        self.bias		        = bias

        # Define architecture: hidden layers plus final layer of size 1.
        self._dims              = [self.input_dim] + list(layers) + [self._output_dim]
        self._parameters        = { "layers" : [], "params" : {} }
        self.act_fun            = act_fun
        self.init()
        
    ###########################
    #! INIT
    ###########################
    
    def init_activation(self) -> None:
        """
        Initialize activation functions for the network.
        """
        
        # Store activation functions.
        self.act_fun = tuple(get_activation(act, self._parameters, self._backend_str)[0] 
                        for act in self.act_fun)
        
        if len(self.act_fun) < len(self.layers) + 1:
            # If there are not enough activation functions, fill the rest with identity.
            self.act_fun += tuple(get_activation("identity", self._backend_str)[0]
                        for _ in range(len(self.layers) + 1 - len(self.act_fun)))
    
    def init_fun(self):
        """
        Initialize the network parameters.
        This method is a placeholder for any additional initialization logic
        that may be needed in the future.
        """
        if self._use_jax:
            # Convert parameters to JAX arrays.
            self._apply_jax = apply_wrapper(self.act_fun, True)
        self._apply_np = apply_wrapper(self.act_fun, False)
    
    def init(self, key = None) -> dict:
        """
        Initialize network parameters (weights and biases).
        
        Uses He initialization for weights (scaled random normal) and zeros for biases.
        For complex data types, initializes both real and imaginary parts.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key (for compatibility with other implementations)
        Returns
        -------
        dict
            Dictionary of initialized parameters with weights and biases
        """
        
        if self.initialized:
            # If already initialized, return existing parameters.
            return self._parameters
        
        self._parameters["layers"] = []
        
        for i in range(len(self._dims) - 1):
            # Initialize weights with He initialization scaling
            if np.issubdtype(self._dtype, np.complexfloating):
                w = np.random.randn(self._dims[i], self._dims[i+1])
                w = (w + 1j * np.random.randn(*w.shape)).astype(self._dtype)
            else:
                w = np.random.randn(self._dims[i], self._dims[i+1]).astype(self._dtype)
                
            # Scale weights by sqrt(2/fan_in)
            w *= np.sqrt(2.0 / self._dims[i])
            
            # Initialize biases as zeros if using bias
            if self.bias:
                b = np.random.random(self._dims[i+1])
                if np.issubdtype(self._dtype, np.complexfloating):
                    b = (b + 1j * np.random.random(self._dims[i+1])).astype(self._dtype)
            else:
                b = np.zeros(self._dims[i+1], dtype=self._dtype)
            self._parameters["layers"].append((w, b))

        # Convert to JAX arrays if using JAX backend
        if self._use_jax:
            self._parameters["layers"] = tuple(
                (jnp.array(W), jnp.array(b) if b is not None else None)
                for W, b in self._parameters["layers"])
        
        self.init_activation()
        self.init_fun()
        super().init(key)
        
        return self._parameters

    ##########################
    #! PROPERTIES
    ##########################
    
    @property
    def dtypes(self):
        '''
        Get the data types of the network parameters.
        '''
        if self._use_jax:
            return [self._dtype] * len(self._parameters["layers"])
        return [self._dtype] * len(self._parameters["layers"])
    
    @property
    def shapes(self):
        '''
        Get the shapes of the network parameters.
        '''
        if self._use_jax:
            return [W.shape for W, b in self._parameters["layers"]]
        return [W.shape for W, b in self._parameters["layers"]]
    
    @property
    def nparams(self):
        '''
        Get the number of parameters in the network.
        '''
        if self._use_jax:
            return [W.size for W, b in self._parameters["layers"]]
        return [W.size for W, b in self._parameters["layers"]]
    
    ##########################
    #! INFO
    ##########################
    
    def __repr__(self) -> str:
        """
        String representation of the network.
        
        Returns:
            str: Network architecture and parameters.
        """
        return f"SimpleNet(input_dim={self.input_dim}, layers={self.layers}, bias={self.bias}, backend={self.backend})"
    
    ##########################
    #! SETTERS
    ##########################

    def set_params(self, params: dict) -> None:
        """
        Set the parameters of the network.
        
        Parameters:
            params : dict
                Dictionary of parameters.
        """
        if not isinstance(params, dict) and isinstance(params, (list, tuple)):
            params = { "layers" : params }
        
        if "layers" not in params:
            raise KeyError("Parameters must contain 'layers' key.")
        self._parameters = params
    
    ###########################
    #! GETTERS
    ###########################

###########################################################################