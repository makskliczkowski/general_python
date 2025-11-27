"""

A general neural network implementation, which can be used as a template
for building more complex networks. This can be also a simple callable
network that generates random complex outputs.

---------------------------------------------------------------
file    : general_python/ml/net_simple.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-03-10
---------------------------------------------------------------
"""

import numpy as np
import numba
from typing import Optional, Tuple, Callable, List, Union
from abc import ABC, abstractmethod
from ...algebra.utils import JAX_AVAILABLE, get_backend, Array
from ...common.flog import get_global_logger, Logger

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jax     = None
    jnp     = None
    
#########################################################

class GeneralNet(ABC):
    """
    A simple network implementation that generates random complex outputs.

    This class simulates a neural network by generating random outputs using random
    initializations of weights and biases. It can operate using either NumPy or JAX
    backends, depending on the provided backend argument.
    """

    _dcol   = 'blue'            # Default color for the logger 

    def __init__(self,
                input_shape     : tuple,
                backend		    : str		         = 'default',
                dtype           : Optional[np.dtype] = np.float64,
                in_activation   : Optional[Callable] = None,
                **kwargs):

        self._name              = "GeneralNet"
        self._logger            = get_global_logger()
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
        
        # parameters
        self._parameters        = None
        self._param_shapes      = None
        self._param_num         = None
        self._param_tree_def    = None
        
        # helper functions
        self._holomorphic	    = None
        self._has_analytic_grad = True
        
        # initialization
        self._initialized       = False
        
        self._compiled_grad_fn  = None
        self._compiled_apply_fn = None

        # shapes for update
        self._shapes_for_update : Optional[List[Tuple[int, tuple]]] = None  # List of (num_real_comp, shape)
        
        # activation - modifies the input before the network
        self._in_activation     = in_activation
        self._net_module_class  = None
        self._net_args          = None
        self._seed              = None
        
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

    def log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'],
        lvl : int = 0, color : str = "white", append_msg = True):
        """
        Log the message.
        
        Args:
            msg (str) :
                The message to log.
            log (Union[int, str]) :
                The flag to log the message (default is 'info').
            lvl (int) :
                The level of the message.
            color (str) :
                The color of the message.
            append_msg (bool) :
                Flag to append the message.
        """
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        if append_msg:
            msg = f"[{self._name}] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log=log, lvl=lvl)

    # ---------------------------------------------------
    #! INIT 
    # ---------------------------------------------------
    
    def force_init(self, key = None):
        '''
        Force the network initialization...
        '''
        
        self._initialized = False
        return self._initialized
    
    def init(self, key = None):
        """
        Initialize the network parameters.
        
        Parameters:
            key (optional): Random key for initialization.
        """
        self._initialized = True
        return self.get_params()
    
    # ---------------------------------------------------
    #! PROPERTIES
    # ---------------------------------------------------
    
    @property
    def is_complex(self) -> bool:
        """
        Check if the network parameters are complex.
        
        Returns:
            bool: True if the network parameters are complex, False otherwise.
        """
        return np.issubdtype(self._dtype, np.complexfloating)
    
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
    
    # ---
    
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
    
    # ---
    
    @property
    def is_holomorphic(self) -> bool | None:
        """
        Check if the network is holomorphic.
        
        Returns:
            bool: True if the network is holomorphic, False otherwise. None if not set.
        """
        return self._holomorphic
    
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
        return self._param_shapes
    
    @property
    def tree_def(self):
        """
        Get the tree definition of the network parameters.
        
        Returns:
            dict: Tree definition of the network parameters.
        """
        return self._param_tree_def
    
    @property
    def nparams(self):
        """
        Get the number of parameters in the
        network.    
        """
        return self._param_num

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
    
    @property
    def has_analytic_grad(self) -> bool:
        """
        Check if the network has analytic gradients.
        
        Returns:
            bool: True if the network has analytic gradients, False otherwise.
        """
        return self._has_analytic_grad
    
    # ---
    
    @property
    def compiled_grad_fn(self):
        """
        Get the compiled gradient function.
        
        Returns:
            Callable: Compiled gradient function.
        """
        return self._compiled_grad_fn
    
    @property
    def compiled_apply_fn(self):
        """
        Get the compiled apply function.
        
        Returns:
            Callable: Compiled apply function.
        """
        return self._compiled_apply_fn
    
    @property
    def apply_fn(self):
        """
        Get the apply function.
        
        Returns:
            Callable: Apply function.
        """
        return self._compiled_apply_fn
    
    @property
    def grad_fn(self):
        """
        Get the gradient function.
        
        Returns:
            Callable: Gradient function.
        """
        return self._compiled_grad_fn
    
    # ---
    
    @property
    def shapes_for_update(self) -> Optional[List[Tuple[int, tuple]]]:
        """
        Get the shapes for update.
        
        Returns:
            Optional[List[Tuple[int, tuple]]]: Shapes for update.
        """
        return self._shapes_for_update

    # ---
    
    @property
    def net_module(self):
        """
        Get the network module.
        
        Returns:
            object: Network module.
        """
        return self._net_module_class

    @property
    def net_args(self):
        """
        Get the network arguments.
        
        Returns:
            object: Network arguments.
        """
        return self._net_args
    
    @property
    def seed(self):
        """
        Get the random seed.
        
        Returns:
            int: Random seed.
        """
        return self._seed

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
            
        if self._use_jax or (use_jax and JAX_AVAILABLE):
            return self.apply_jax(x)
        return self.apply_np(x)
    
    # ---------------------------------------------------
    #! GETTERS
    # ---------------------------------------------------
    
    def get_apply(self, use_jax: bool = False) -> Callable:
        """
        Get the apply function of the network.
        Parameters:
            use_jax (bool): If True, use JAX backend. Default is False.
        If JAX is not available, it will use NumPy backend.
        Returns:
            Callable: Apply function of the network.
        """
        if self._use_jax or (use_jax and JAX_AVAILABLE):
            return self._apply_jax, self.get_params()
        return self._apply_np, self.get_params()
    
    def get_gradient(self, use_jax: bool = False, analytic: bool = False) -> Callable:
        '''
        For the networks that may have an analytic gradient obtainable via 
        '''
        if self._use_jax or (use_jax and JAX_AVAILABLE):
            return self._compiled_grad_fn, self.get_params()
        return None
    
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
    #! CHECKER
    # ---------------------------------------------------
    
    def check_holomorphic(self):
        """
        Check if the network is holomorphic.
        
        Returns:
            bool: True if the network is holomorphic, False otherwise.
        """
        #     # Using numpy-based gradients.
        #     def make_flat(x):
        #         leaves, _ = flatten_func(x)
        #         return np.concatenate([p.ravel() for p in leaves])
            
        #     grads_real      = make_flat(np_grad(lambda a,b: anp.real(self.net.apply(a,b)))(self._weights, sample_state)["params"] )
        #     grads_imag      = make_flat(np_grad(lambda a,b: anp.imag(self.net.apply(a,b)))(self._weights, sample_state)["params"] )
            
        #     flat_real       = make_flat(grads_real)
        #     flat_imag       = make_flat(grads_imag)
            
        #     norm_diff       = np.linalg.norm(flat_real - 1.j * flat_imag) / flat_real.shape[0]
        #     return np.isclose(norm_diff, 0.0, atol= self._TOL_HOLOMORPHIC)
        self.holomorphic = True
        return True
    
    # ---------------------------------------------------
    #! ANALYTIC GRADIENT
    # ---------------------------------------------------
    
    @staticmethod
    def analytic_grad_np(params, x: 'array-like') -> 'array-like':
        """
        Compute the analytic gradient of the network.

        Parameters:
            params (dict)           : Network parameters.
            x (array-like)          : Input data.

        Returns:
            array-like: Analytic gradient of the network.
        """
        pass
    
    @staticmethod
    def analytic_grad_jax(params, x: 'array-like') -> 'array-like':
        """
        Compute the analytic gradient of the network using JAX.

        Parameters:
            params (dict)           : Network parameters.
            x (array-like)          : Input data.

        Returns:
            array-like: Analytic gradient of the network.
        """
        pass
    
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