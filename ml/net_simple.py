"""
file    : general_python/ml/net_simple.py
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
from typing import Optional, Tuple, Callable
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
from general_python.ml.activation_functions import get_activation

if _JAX_AVAILABLE:
		import jax.numpy as jnp
		from jax import random

##################################################################

class SimpleNet:
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
        # Get the backend.
        if isinstance(backend, str):
            self._backend_str	= backend
            self._backend		= get_backend(backend)
        else:
            self._backend		= get_backend('default')
            self._backend_str	= 'np' if backend == np else 'jax'

        # Use the first element of input_shape as input_dim.
        self._use_jax           = self.backend != np
        self._dtype             = dtype  
        self._input_dim	        = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        self._output_dim        = output_shape[0] if isinstance(output_shape, tuple) else output_shape
        self.layers		        = layers
        self.bias		        = bias

        # Define architecture: hidden layers plus final layer of size 1.
        dims                    = [self.input_dim] + list(layers) + [self._output_dim]
        self.params_np          = { "layers" : [] }
        self.init_weights(dims)
        
        # Initialize activation functions.
        self.init_activation(act_fun)
        
    ###########################
    
    def init_activation(self, act_fun: Optional[Tuple] = None) -> None:
        """
        Initialize activation functions for the network.
        
        Parameters:
            act_fun : tuple
                Activation functions for NumPy.
        """
        
        # Store activation functions.
        self.act_fun = tuple( get_activation(act, self._backend_str)[0]
								for act in act_fun )
        if len(self.act_fun) < len(self.layers) + 1:
            # If there are not enough activation functions, fill the rest with identity.
            self.act_fun += tuple(get_activation("identity", self._backend_str)[0] 
                                for _ in range(len(self.layers) + 1 - len(self.act_fun)))

    ###########################

    def init_weights(self, dims, key: Optional[random.PRNGKey] = None) -> None:
        """
        Initialize weights of the network using JAX.
        
        Parameters:
            key : jax.random.PRNGKey
                Random key for JAX random number generation.
        """

        # Initialize weights and biases using JAX.
        self.params_np["layers"] = []
        for i in range(len(dims) - 1):
            # Initialize weights with He-like scaling.
            if np.issubdtype(self._dtype, np.complexfloating):
                # Create complex weights: real and imaginary parts.
                W = (np.random.randn(dims[i], dims[i+1]) + 1j * np.random.randn(dims[i], dims[i+1]))
                W = W.astype(self._dtype)
            else:
                W = np.random.randn(dims[i], dims[i+1]).astype(self._dtype)
            # Scale weights.
            W   = W * np.sqrt(2.0 / dims[i])
            b   = np.zeros(dims[i+1], dtype=self._dtype) if self.bias else None
            self.params_np["layers"].append((W, b))

        if self._use_jax:
            # Convert parameters to JAX arrays.
            self.params_np["layers"] = tuple((jnp.array(W), jnp.array(b)) for W, b in self.params_np["layers"])

    ##########################
    #! PROPERTIES
    ##########################
    
    @property
    def input_dim(self) -> int:
        """
        Get the input dimension of the network.
        
        Returns:
            int: Input dimension.
        """
        return self._input_dim
        
    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the network parameters.
        
        Returns:
            np.dtype: Data type of the network parameters.
        """
        return self._dtype
    
    @property
    def backend(self) -> str:
        """
        Get the backend used by the network.
        
        Returns:
            str: Backend used ('numpy' or 'jax').
        """
        return self._backend_str
    
    ##########################

    def apply(self, x : np.ndarray) -> np.ndarray:
        """
        Apply the network to an input batch using NumPy
        
        Parameters:
            x : np.ndarray
                Input array of shape (batch, input_dim)
        
        Returns:
            np.ndarray: One output per sample.
        """
        
        # Apply each hidden layer.
        s = x
        for (W, b), act in zip(self.params_np["layers"], self.act_fun):
            s = self._backend.dot(s, W)
            if self.bias:
                s = s + b
            s = act(s)
        return s


    ##########################
    
    def __call__(self, parameters = None, x : np.ndarray = None) -> np.ndarray:
        """
        Apply the network to an input batch.
        
        Parameters:
            x : np.ndarray
                Input array of shape (batch, input_dim)
        
        Returns:
            np.ndarray: One output per sample.
        """
        if parameters is not None and x is not None:
            self.set_params(parameters)
            
        if x is None:
            return self.apply(parameters)
        
        return self.apply(x)
    
    ##########################
    
    def __repr__(self) -> str:
        """
        String representation of the network.
        
        Returns:
            str: Network architecture and parameters.
        """
        return f"SimpleNet(input_dim={self.input_dim}, layers={self.layers}, bias={self.bias}, backend={self.backend_str})"
    
    ##########################
    
    def __str__(self) -> str:
        """
        String representation of the network.
        
        Returns:
            str: Network architecture and parameters.
        """
        return self.__repr__()
    
    ##########################
    
    def set_params(self, params: dict) -> None:
        """
        Set the parameters of the network.
        
        Parameters:
            params : dict
                Dictionary of parameters.
        """
        self.params_np = params
        
    ##########################
    
    def get_params(self) -> dict:
        """
        Get the parameters of the network.
        
        Returns:
            dict: Dictionary of parameters.
        """
        return self.params_np
    
    ##########################

###########################################################################    
