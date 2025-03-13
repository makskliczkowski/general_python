"""
file    : general_python/ml/net_simple.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-03-10
"""

import numpy as np
import numba
from typing import Optional, Tuple, Callable
from abc import ABC, abstractmethod
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import random
    
#########################################################

class GeneralNet(ABC):
    """
    A simple network implementation that generates random complex outputs.

    This class simulates a neural network by generating random outputs using random
    initializations of weights and biases. It can operate using either NumPy or JAX
    backends, depending on the provided backend argument.
    """

    def __init__(self,
                input_shape     : tuple,
                backend		    : str		        = 'default',
                dtype           : Optional[np.dtype]= np.float64):

        # Get the backend.
        if isinstance(backend, str):
            self._backend_str	= backend
            self._backend		= get_backend(backend)
            self._use_jax       = self._backend_str != 'np'
        else:
            self._backend		= get_backend(backend)
            self._backend_str	= 'np' if backend == np else 'jax'
            self._use_jax       = self._backend_str != 'np'
        
        # Set the type.
        self._dtype             = dtype

        # applies
        self._apply_jax         = lambda p, x: 1
        self._apply_np          = lambda p, x: 1
        
        # Set the input shape and dimension.
        self._input_shape       = input_shape if isinstance(input_shape, tuple) else (input_shape,)
        self._input_dim	        = np.prod(self._input_shape)
        self._output_shape      = None
        self._output_dim	    = None
        self._parameters        = None
        
        # helper functions
        self._holomorphic	    = None
        self._has_analitic_grad = True
        
        # initialization
        self._initialized       = False
        
    # ---------------------------------------------------
    #! INFO
    # ---------------------------------------------------
    
    def __str__(self) -> str:
        """
        String representation of the network.
        
        Returns:
            str: Network architecture and parameters.
        """
        return self.__repr__()

    # ---------------------------------------------------
    #! INIT 
    # ---------------------------------------------------
    
    def init(self, key = None):
        """
        Initialize the network parameters.
        
        Parameters:
            key (optional): Random key for initialization.
        """
        self._initialized = True
    
    # ---------------------------------------------------
    #! PROPERTIES
    # ---------------------------------------------------
    
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
    
    @property
    def input_shape(self) -> tuple:
        """
        Get the input shape of the network.
        
        Returns:
            tuple: Input shape of the network.
        """
        return self._input_shape
    
    @property
    def input_dim(self) -> int:
        """
        Get the input dimension of the network.
        
        Returns:
            int: Input dimension.
        """
        return self._input_dim
    
    @property
    def holomorphic(self) -> bool | None:
        """
        Check if the network is holomorphic.
        
        Returns:
            bool: True if the network is holomorphic, False otherwise. None if not set.
        """
        return self._holomorphic
    
    @holomorphic.setter
    def holomorphic(self, value: bool):
        """
        Set the holomorphic property of the network.
        
        Parameters:
            value (bool): True if the network is holomorphic, False otherwise.
        """
        if not isinstance(value, bool):
            raise ValueError("Holomorphic property must be a boolean.")
        self._holomorphic = value
        
    @property
    def dtypes(self):
        """
        Get the data types of the network parameters.
        
        Returns:
            Tuple[np.dtype, np.dtype]: Data types of the network parameters.
        """
        return [self._dtype]
    
    @property
    def shapes(self):
        """
        Get the shapes of the network parameters.
        
        Returns:
            Tuple[tuple, ...]: Shapes of the network parameters.
        """
        return [self._input_shape]
    
    @property
    def nparams(self):
        """
        Get the number of parameters in the
        network.    
        """
        return 0 # GeneralNet does not have parameters in the traditional sense.

    @property
    def output_shape(self) -> tuple:
        """
        Get the output shape of the network.
        
        Returns:
            tuple: Output shape of the network.
        """
        return self._output_shape
    
    @property
    def output_dim(self) -> int:
        """
        Get the output dimension of the network.
        
        Returns:
            int: Output dimension.
        """
        return self._output_dim
    
    @property
    def initialized(self) -> bool:
        """
        Check if the network is initialized.
        
        Returns:
            bool: True if the network is initialized, False otherwise.
        """
        return self._initialized

    # ---------------------------------------------------
    #! SETTERS
    # ---------------------------------------------------
    
    @abstractmethod
    def set_params(self, params: dict):
        """
        Set the network parameters.
        
        Parameters:
            params (dict): Network parameters.
        """
        pass
    
    # ---------------------------------------------------
    #! GETTERS
    # ---------------------------------------------------
    
    def get_params(self):
        """
        Get the network parameters.
        
        Returns:
            dict: Network parameters.
        """
        return self._parameters
    
    # ---------------------------------------------------
    #! METHODS
    # ---------------------------------------------------
    
    def apply_np(self, x: 'array-like'):
        """
        Apply the network to the input data.
        
        Parameters:
            x (array-like)          : Input data.
        
        Returns:
            array-like: Output of the network.
        """
        return self._apply_np(self.get_params(), x)
    
    def apply_jax(self, x: 'array-like'):
        """
        Apply the network to the input data using JAX.
        
        Parameters:
            x (array-like)          : Input data.
        
        Returns:
            array-like: Output of the network.
        """
        return self._apply_jax(self.get_params(), x)
    
    def apply(self, params, x: 'array-like', use_jax: bool = False):
        """
        Apply the network to the input data.
        
        Parameters:
            x (array-like)          : Input data.
            use_jax (bool): If True, use JAX backend. Default is False.
        
        Returns:
            array-like: Output of the network.
        """
        if params is not None:
            self.set_params(params)
            
        if self._use_jax or (use_jax and _JAX_AVAILABLE):
            return self.apply_jax(x)
        return self.apply_np(x)
    
    def get_apply(self, use_jax: bool = False) -> Callable:
        """
        Get the apply function of the network.
        Parameters:
            use_jax (bool): If True, use JAX backend. Default is False.
        If JAX is not available, it will use NumPy backend.
        Returns:
            Callable: Apply function of the network.
        """
        if self._use_jax or (use_jax and _JAX_AVAILABLE):
            return self._apply_jax, self.get_params()
        return self._apply_np, self.get_params()
    
    # ---------------------------------------------------
    
    def __call__(self, params = None, x: 'array-like' = None) -> 'array-like':
        """
        Apply the network to an input batch.
        
        Parameters:
            x (array-like)          : Input data.
            params (dict)           : Network parameters. If None, use the default parameters.
        
        Returns:
            array-like: Output of the network.
        """
        return self.apply(params=params, x=x, use_jax=self._use_jax)
    
    # ---------------------------------------------------

#########################################################
#! JUST FROM A CALLABLE - EASY FUNCTION
#########################################################

class CallableNet(GeneralNet):
    """
    A simple network implementation that generates random complex outputs.

    This class simulates a neural network by generating random outputs using random
    initializations of weights and biases. It can operate using either NumPy or JAX
    backends, depending on the provided backend argument.
    """

    def __init__(self,
                input_shape     : tuple,
                backend		    : str		        = 'default',
                dtype           : Optional[np.dtype]= np.float64,
                callable_fun    : Callable          = None):
        super().__init__(input_shape, backend, dtype)
        
        # Set the callable function.
        if callable_fun is None:
            raise ValueError("Callable function must be provided.")
        self._callable_fun  = callable_fun
        
        @numba.njit
        def apply_np_in(_, x):
            return self._callable_fun(x)
        
        @jax.jit
        def apply_jax_in(_, x):
            return self._callable_fun(x)
        
        # Set the apply functions.
        self._apply_np      = apply_np_in
        self._apply_jax     = apply_jax_in
        
        self.init()
        
    # ---------------------------------------------------
    #! INFO
    # ---------------------------------------------------
    
    def __repr__(self):
        return f"CallableNet(input_shape={self._input_shape}, backend={self._backend_str}, dtype={self._dtype})"
    
    # ---------------------------------------------------
    #! PROPERTIES
    # ---------------------------------------------------
    
    @property
    def callable_fun(self) -> Callable:
        """
        Get the callable function of the network.
        
        Returns:
            Callable: Callable function of the network.
        """
        return self._callable_fun
    
    @callable_fun.setter
    def callable_fun(self, value: Callable):
        """
        Set the callable function of the network.
        
        Parameters:
            value (Callable): Callable function.
        """
        if not callable(value):
            raise ValueError("Callable function must be a callable.")
        self._callable_fun = value
        
    @property
    
    def dtypes(self):
        """
        Get the data types of the network parameters.
        
        Returns:
            Tuple[np.dtype, np.dtype]: Data types of the network parameters.
        """
        return [self._dtype]
    
    @property
    def shapes(self):
        """
        Get the shapes of the network parameters.
        
        Returns:
            Tuple[tuple, ...]: Shapes of the network parameters.
        """
        return [self._input_shape]
    
    @property
    def nparams(self):
        """
        Get the number of parameters in the network.
        
        Returns:
            int: Number of parameters in the network.
        """
        return 0  # CallableNet does not have parameters in the traditional sense.
    
    # ---------------------------------------------------
    #! SETTERS
    # ---------------------------------------------------
    
    def set_params(self, params: dict):
        """
        Set the network parameters.
        
        Parameters:
            params (dict): Network parameters.
        """
        # This method is not applicable for CallableNet.
        pass
    
    # ---------------------------------------------------
    
#########################################################