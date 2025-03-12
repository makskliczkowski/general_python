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
import general_python.ml.net_impl.net_general as _net_general
from typing import Optional, Tuple, Callable

# import from general python module
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
from general_python.ml.net_impl.activation_functions import get_activation

if _JAX_AVAILABLE:
		import jax.numpy as jnp
		from jax import random

##################################################################

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
        self.params_np          = { "layers" : [] }
        self.init()
        
        # Initialize activation functions.
        self.init_activation(act_fun)
        
    ###########################
    #! INIT
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

    def init(self, key: Optional[random.PRNGKey] = None) -> dict:
        """
        Initialize network parameters (weights and biases).
        This method initializes the network's weights using He-like initialization
        (scaled random normal distribution) and zeros for biases. For complex 
        data types, it initializes both real and imaginary parts of the weights.
        Parameters
        ----------
        key : Optional[random.PRNGKey]
            JAX random key (not used in this implementation but kept for 
            compatibility with other network implementations)
        Returns
        -------
        dict
            Dictionary containing initialized parameters with a 'layers' key 
            that holds a list/tuple of weight-bias pairs for each layer.
            Each pair is a tuple of (W, b) where W is the weight matrix and 
            b is the bias vector (or None if bias is disabled).
        """


        # Initialize weights and biases using JAX.
        self.params_np["layers"] = []
        for i in range(len(self._dims) - 1):
            # Initialize weights with He-like scaling.
            if np.issubdtype(self._dtype, np.complexfloating):
                # Create complex weights: real and imaginary parts.
                W = (np.random.randn(self._dims[i], self._dims[i+1]) + 1j * np.random.randn(self._dims[i], self._dims[i+1]))
                W = W.astype(self._dtype)
            else:
                W = np.random.randn(self._dims[i], self._dims[i+1]).astype(self._dtype)
            # Scale weights.
            W   = W * np.sqrt(2.0 / self._dims[i])
            b   = np.zeros(self._dims[i+1], dtype=self._dtype) if self.bias else None
            self.params_np["layers"].append((W, b))

        if self._use_jax:
            # Convert parameters to JAX arrays.
            self.params_np["layers"] = tuple((jnp.array(W), jnp.array(b)) for W, b in self.params_np["layers"])
    
        return self.params_np
    
    ##########################
    #! PROPERTIES
    ##########################
    
    @property
    def dtypes(self):
        '''
        Get the data types of the network parameters.
        '''
        if self._use_jax:
            return [self._dtype] * len(self.params_np["layers"])
        return [self._dtype] * len(self.params_np["layers"])
    
    @property
    def shapes(self):
        '''
        Get the shapes of the network parameters.
        '''
        if self._use_jax:
            return [W.shape for W, b in self.params_np["layers"]]
        return [W.shape for W, b in self.params_np["layers"]]
    
    @property
    def nparams(self):
        '''
        Get the number of parameters in the network.
        '''
        if self._use_jax:
            return [W.size for W, b in self.params_np["layers"]]
        return [W.size for W, b in self.params_np["layers"]]
    
    ##########################
    #! APPLY
    # ########################
    
    def apply(self, x: 'array-like') -> np.ndarray:
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
        self.params_np = params
        
    ##########################
    #! GETTERS
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
