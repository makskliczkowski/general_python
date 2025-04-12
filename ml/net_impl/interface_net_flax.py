
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
from typing import Tuple, Callable, Optional, Any

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core import unfreeze, freeze
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

# from general_python utilities
from general_python.ml.net_impl.net_general import GeneralNet
from general_python.ml.net_impl.activation_functions import get_activation
from general_python.algebra.utils import _JAX_AVAILABLE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE

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
                dtype         : Optional[jnp.dtype] = jnp.float32,
                seed          : int = 42):
        
        # check the dtype.
        self._dtype = dtype
        if not isinstance(self._dtype, jnp.dtype):
            self._dtype = jnp.dtype(self._dtype)
        
        try:
            # Convert common numpy dtypes or use JAX default
            if self._dtype is np.float32:
                jax_dtype = jnp.float32
            elif self._dtype is np.float64:
                jax_dtype = jnp.float64
            elif self._dtype is np.complex64:
                jax_dtype = jnp.complex64
            elif self._dtype is np.complex128:
                jax_dtype = jnp.complex128
            elif isinstance(self._dtype, jnp.dtype):
                jax_dtype = self._dtype
            else:
                jax_dtype = jnp.dtype(self._dtype)
        except TypeError:
            print(f"Warning: Could not convert dtype {dtype} to JAX dtype. Using default {DEFAULT_JP_FLOAT_TYPE}.")
            jax_dtype = DEFAULT_JP_FLOAT_TYPE
        self._dtype = jax_dtype
        
        #! Initialize the GeneralNet class.
        super().__init__(input_shape, backend, jax_dtype)
        
        #! Set the backend to JAX.
        if self._backend != jnp and self._backend != 'jax':
            raise ValueError(self._ERR_JAX_NECESSARY)
        if not _JAX_AVAILABLE:
            raise ImportError(self._ERR_JAX_NECESSARY + " JAX not installed.")
        
        
        #! Set the kwargs for the network module.
        if net_kwargs is None:
            net_kwargs = {}
        
        self._net_module_class      = net_module
        self._net_args              = net_args
        self._net_kwargs_in         = net_kwargs if net_kwargs is not None else {}
        self._seed                  = seed
        
        # try to get the activation from the kwargs.
        self._handle_activations(net_kwargs)
        net_kwargs.setdefault('dtype', self._dtype)
        
        # Create the internal Flax module.
        try:
            self._flax_module: nn.Module = self._net_module_class(*self._net_args, **net_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to instantiate Flax module {self._net_module_class.__name__} "
                            f"with args={self._net_args}, kwargs={net_kwargs}: {e}") from e
        
        # Initialize parameters to None; will be set in init().
        self._parameters            = None
        self._initialized           = False
        
        # Set the callable functions for evaluation.
        self._apply_jax             = self._flax_module.apply
        self._apply_np              = None
        
        # Important to set
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

        
        self._act_fun_specs             = net_kwargs.get('act_fun', None)
        if self._act_fun_specs:
            self._initialized_act_funs  = self._initialize_activations(self._act_fun_specs)
            net_kwargs['act_fun']       = self._initialized_act_funs
        else:
            self._initialized_act_funs  = () # No activations specified
    
    def _initialize_activations(self, act_fun_specs: Any) -> Tuple[Callable, ...]:
        """
        Converts activation specs (strings or callables) to callables. 
        This method ensures that the activation functions are in a tuple
        and initializes them using the get_activation utility.
        Params:
            act_fun_specs: Any
                Activation function specifications (strings or callables).
        Returns:
            Tuple[Callable, ...]
                A tuple of initialized activation functions.    
        """
        if not isinstance(act_fun_specs, (list, tuple)):
            act_fun_specs = (act_fun_specs,)

        initialized_funs = []
        for spec in act_fun_specs:
            if isinstance(spec, str):
                # Use get_activation utility
                act_fn, _       = get_activation(spec, backend=self._backend_str)
                
                # Check if the activation function is valid
                if act_fn is None:
                    raise ValueError(f"Unknown activation string: {spec}")
                
                # Append the activation function to the list
                initialized_funs.append(act_fn)
            elif callable(spec):
                initialized_funs.append(spec)
            else:
                raise TypeError(f"Activation spec must be string or callable, got {type(spec)}")
        return tuple(initialized_funs)
    
    def _compile_functions(self):
        """
        Compiles JITted versions of apply and gradient functions.
        This method uses JAX's jit to compile the functions for
        faster execution on the GPU/TPU.
        
        Note:
            This method should be called after the network is initialized.
        """
        if not self._initialized:
            raise RuntimeError(self._ERR_NET_NOT_INITIALIZED + " Cannot compile functions.")

        #! Compiled Apply Function
        # Takes params dict and input x
        @jax.jit
        def compiled_apply(p, x):
            return self._apply_jax({'params': p}, x)
        self._compiled_apply_fn = compiled_apply

        #! Compiled Gradient Function
        @jax.jit
        def compiled_log_psi_grad(p, x):
            #! Was used for testing the gradient of log(psi)
            # Define function whose gradient is needed: params -> log(psi)
            # def log_psi_of_params(_p):
            #     psi = self._apply_jax({'params': _p}, x)
            #     # Handle potential batch dimension in psi before log
            #     # Assuming psi is (batch,) or scalar if x was single instance
            #     # If psi can be zero or negative, log will produce NaN/Inf or complex results
            #     # Add stabilization if necessary, e.g., log(psi + epsilon) or handle complex log
            #     return jnp.log(psi + 1e-10) # Use complex-safe log with stabilization
            # grad_tree = jax.jacrev(log_psi_of_params)(p)

            # Calculate gradient w.r.t. first arg (_p)
            # jacrev returns PyTree matching _p structure
            psi         = self._apply_jax({'params': p}, x)
            grad_tree   = jax.jacrev(psi)(p)
            return grad_tree
        self._compiled_grad_fn = compiled_log_psi_grad
    
    def init(self, key: Optional[jax.random.PRNGKey] = None):
        """
        Initialize the network parameters using Flax.
        Params:
            key: jax.random.PRNGKey
                Random key for initialization. If None, a default key is used.
        """
        
        if self._initialized:
            print("Warning: Network already initialized. Re-initializing.")
        
        if key is None:
            key = random.PRNGKey(self._seed)
        
        # create a dummy input for initialization.
        try:
            print(f"DEBUG: Initializing Flax module. self.input_shape = {self.input_shape}")
            print(f"DEBUG: self.input_dim = {self.input_dim}")
            print(f"DEBUG: Creating dummy input with shape: {(1, self.input_dim)}, dtype: {self._dtype}")
            dummy_input             = jnp.ones((1, self.input_dim), dtype=self._dtype)
            variables               = self._flax_module.init(key, dummy_input)
            self._parameters        = variables['params'] # Store only params collection
            self._apply_jax         = self._flax_module.apply
            self._initialized       = True
            # print(f"Network initialized with parameters: {self._parameters}")
        except Exception as e:
            self._initialized       = False
            raise ValueError(f"Failed to initialize the network: {e}") from e
        
        # set the internal parameters shapes and tree structure.
        self._param_shapes      = [(p.size, p.shape) for p in tree_flatten(self._parameters)[0]]
        self._param_tree_def    = jax.tree_util.tree_structure(self._parameters)
        self._param_num         = jnp.sum(jnp.array([p.size for p in tree_flatten(self._parameters)[0]]))
        
        # set the compiled functions.
        self._compile_functions()
        return self._parameters
    
    ########################################################
    #! SETTERS
    ########################################################
    
    def set_params(self, params: dict):
        """
        Set the network parameters. Now the parameters are
        unfrozen and set to the network.
        """
        new_flat, new_tree = tree_flatten(params)
        if new_tree != self._param_tree_def:
            raise ValueError("New parameters have different tree structure.")
        self._params = params
    
    def get_params(self):
        """
        Get the current network parameters.
        """
        if not self._initialized:
            print("Warning: Network not initialized.")
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
    
    # @jax.jit
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
        # if not self.initialized:
            # raise ValueError(self._ERR_NET_NOT_INITIALIZED)
        return self._compiled_apply_fn(self._parameters, x.astype(self._dtype))

    #########################################################
    #! CHECK HOLOMORPHICITY
    #########################################################
    
    def _unflatten_params_vec(self, flat_vec, example_leaves):
        '''
        Unflatten a flat vector into a PyTree structure based on the shapes
        of the example leaves.
        Params:
            flat_vec: jnp.ndarray
                Flat vector to unflatten.
            example_leaves: list
                List of example leaves (shapes) to unflatten the vector.
        Returns:
            list
                Unflattened PyTree structure.
        '''
        # Reconstruct PyTree from flat vector based on shapes in example_leaves
        params_recon        = []
        current_index       = 0
        for example_leaf in example_leaves:
            size            = example_leaf.size
            chunk           = flat_vec[current_index : current_index + size]
            params_recon.append(chunk.reshape(example_leaf.shape))
            current_index   += size
        return params_recon
    
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
        
        if not jnp.issubdtype(self._dtype, jnp.complexfloating):
            print("Holomorphic check requires complex dtype.")
            self._holomorphic = False
            return False
        
        # Ensure parameters are complex too
        param_leaves, _     = tree_flatten(self._parameters)
        if not all(jnp.issubdtype(p.dtype, jnp.complexfloating) for p in param_leaves):
            print("Holomorphic check requires complex parameters.")
            self._holomorphic = False
            return False
        
        # Check if the network is holomorphic.
        dummy_input             = jnp.ones((1, self.input_dim), dtype=self._dtype)

        try:
            grad_real_tree      = jax.grad(lambda p: jnp.real(self._apply_jax({'params': p}, dummy_input)).sum())(self._parameters)
            grad_imag_tree      = jax.grad(lambda p: jnp.imag(self._apply_jax({'params': p}, dummy_input)).sum())(self._parameters)

            flat_real, _        = tree_flatten(grad_real_tree)
            flat_imag, _        = tree_flatten(grad_imag_tree)
            flat_real_vec       = jnp.concatenate([p.ravel() for p in flat_real])
            flat_imag_vec       = jnp.concatenate([p.ravel() for p in flat_imag])

            # Check Cauchy-Riemann: dRe/dx = dIm/dy, dRe/dy = -dIm/dx
            # Here, x, y are real/imag parts of params. Check d(Re(f))/dp_real vs d(Im(f))/dp_imag, etc.
            # A simpler proxy: check if grad(Re(f)) approx equals i * grad(Im(f)) for complex params
            # This requires careful handling of complex gradients.

            # Using the proxy norm || grad(Re) - i*grad(Im) || / ||grad(Re)|| ~= 0
            # This assumes parameters 'p' are complex.
            norm_diff           = jnp.linalg.norm(flat_real_vec - 1j * flat_imag_vec)
            norm_real           = jnp.linalg.norm(flat_real_vec)
            self._holomorphic   = jnp.isclose(norm_diff / (norm_real + 1e-12), 0.0, atol=self._TOL_HOLOMORPHIC)

        except Exception as e:
            print(f"Error during simplified holomorphic check: {e}")
            self._holomorphic = False
        print(f"Holomorphic check result: {self._holomorphic}")
        return self._holomorphic

    #########################################################
    
    @property
    def activations(self):
        """
        Get the activation functions for the network.
        Returns:
            tuple: A tuple of activation functions for each layer.
        """
        return self._initialized_act_funs
    
    #########################################################
    
#############################################################