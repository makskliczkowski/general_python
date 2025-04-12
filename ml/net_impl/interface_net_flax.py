#   file    : general_python/ml/net_impl/interface_net_flax.py
#   author  : Maksymilian Kliczkowski
#   date    : 2025-04-02

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
from typing import Tuple, Callable, Optional, Any, List

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core import unfreeze, freeze
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

# from general_python utilities
from general_python.ml.net_impl.net_general import GeneralNet
from general_python.ml.net_impl.activation_functions import get_activation
from general_python.algebra.utils import JAX_AVAILABLE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE

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
        
        self._name  = 'FlaxNetInterface'
        #! Check the dtype.
        self._dtype = self._dtype = self._resolve_dtype(dtype if dtype is not None else net_kwargs.get('dtype', DEFAULT_JP_FLOAT_TYPE))
        
        #! Initialize the GeneralNet class.
        super().__init__(input_shape, backend, self._dtype)

        #! Set the backend to JAX.
        if self._backend != jnp and self._backend != 'jax':
            raise ValueError(self._ERR_JAX_NECESSARY)
        if not JAX_AVAILABLE:
            raise ImportError(self._ERR_JAX_NECESSARY + " JAX not installed.")
        
        
        #! Set the kwargs for the network module.
        if net_kwargs is None:
            net_kwargs = {}
        
        self._seed                  = seed
        self._net_module_class      = net_module
        self._net_args              = net_args
        self._initialized           = False
        
        #! Ensure dtype consistency in kwargs passed to the module
        net_kwargs_processed        = net_kwargs.copy()
        net_kwargs_processed.setdefault('dtype', self._dtype)
        self._handle_activations(net_kwargs_processed)
        
        #! Create the internal Flax module.
        try:
            self._flax_module: nn.Module = self._net_module_class(*self._net_args, **net_kwargs_processed)
        except Exception as e:
            raise ValueError(f"Failed to instantiate Flax module {self._net_module_class.__name__} "
                            f"with args={self._net_args}, kwargs={net_kwargs_processed}: {e}") from e
        
        # Initialize parameters to None; will be set in init().
        self._parameters            : Optional[Any]     = None                  # PyTree of parameters
        self._param_tree_def        : Optional[Any]     = None                  # PyTree definition
        self._param_shapes_orig     : Optional[List[Tuple[int, tuple]]] = None  # List of (size, shape)
        self._shapes_for_update     : Optional[List[Tuple[int, tuple]]] = None  # List of (num_real_comp, shape)
        self._param_num             : Optional[int]     = None                  # Total number of parameters
        self._iscpx                 : Optional[bool]    = None                  # Is the model complex overall?
        self._holomorphic           : Optional[bool]    = None                  # Is the model holomorphic?
        self._initialized           : bool              = False                 # Is the model initialized?
        self._apply_jax_handle      : Optional[Callable]= None                  # Handle to module's apply
        self._compiled_apply_fn     : Optional[Callable]= None                  # JITted apply
        self._has_analytic_grad     : bool              = False                 # Does the model have analytic gradients?
        self.init()
        
    ########################################################
    #! INITIALIZATION
    ########################################################
    
    def _resolve_dtype(self, dtype_spec: Any) -> jnp.dtype:
        """
        Converts various dtype specifications to a JAX dtype.
        This method handles common numpy dtypes and attempts
        to convert other types (like strings) to JAX dtypes.
        """
        try:
            if dtype_spec is np.float32:            return jnp.float32
            if dtype_spec is np.float64:            return jnp.float64
            if dtype_spec is np.complex64:          return jnp.complex64
            if dtype_spec is np.complex128:         return jnp.complex128
            if isinstance(dtype_spec, jnp.dtype):   return dtype_spec
            # Attempt direct conversion for other cases (like strings 'float32')
            return jnp.dtype(dtype_spec)
        except TypeError:
            self.log(f"Warning: Could not convert dtype spec {dtype_spec} to JAX dtype. Using default {DEFAULT_JP_FLOAT_TYPE}.",
                    log = 'warning', lvl = 1, color = self._dcol)
            return DEFAULT_JP_FLOAT_TYPE
    
    def _compile_functions(self):
        """
        Compiles JITted version of the apply function.
        This method uses JAX's JIT compilation to optimize the apply function
        """
        if not self._initialized or self._apply_jax_handle is None:
            raise RuntimeError(self._ERR_NET_NOT_INITIALIZED + " Cannot compile functions.")
        
        apply_handle = self._apply_jax_handle
        @jax.jit
        def compiled_apply(p, x):
            return apply_handle({'params': p}, x)
        self._compiled_apply_fn = compiled_apply
    
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
    
    def init(self, key: Optional[jax.random.PRNGKey] = None):
        """
        Initialize the network parameters using Flax.
        Params:
            key: jax.random.PRNGKey
                Random key for initialization. If None, a default key is used.
        """
        
        if self._initialized:
            self.log(f"Warning: Network {self.__class__} already initialized. Re-initializing.",
                    log = 'warning', lvl = 1, color = self._dcol)

        if key is None:
            key = random.PRNGKey(self._seed)

        try:
            dummy_input     = jnp.ones((1, self.input_dim), dtype=self._dtype)
            variables       = self._flax_module.init(key, dummy_input)

            if 'params' not in variables:
                self.log(f"Warning: 'params' key not found in Flax variables: {variables.keys()}. Assuming variables ARE the params.",
                        log = 'warning', lvl = 1, color = self._dcol)
                self._parameters = variables
            else:
                self._parameters = variables['params']
                
            # Get the apply function handle
            self._apply_jax_handle = self._flax_module.apply

        except Exception as e:
            self._initialized = False
            raise ValueError(f"Failed to initialize the network with shape {(1, self.input_dim)} and dtype {self._dtype}: {e}") from e

        #! Calculate and Store Metadata
        if self._parameters is None:
            raise RuntimeError("Parameters are None after Flax initialization.")

        flat_params, self._param_tree_def = tree_flatten(self._parameters)
        if not self._param_tree_def or not flat_params:
            raise TypeError("Initialized parameters did not yield a valid PyTree structure or leaves.")

        self._param_shapes_orig     = []
        self._shapes_for_update     = []
        any_complex                 = False
        total_params                = 0

        #! Check if the parameters are complex
        for x in flat_params:
            is_leaf_complex         = jnp.iscomplexobj(x)
            num_real_components     = x.size * 2 if is_leaf_complex else x.size
            self._param_shapes_orig.append((x.size, x.shape))
            self._shapes_for_update.append((num_real_components, x.shape))
            total_params           += x.size
            if is_leaf_complex:
                any_complex = True

        self._param_num             = total_params
        self._iscpx                 = any_complex
        # Confirm overall dtype from first parameter leaf
        self._dtype                 = flat_params[0].dtype
        # -----------------------------------

        self._initialized           = True
        self._compile_functions()   # Compile apply function
        self._holomorphic           = None # Reset holomorphic check status
        self.check_holomorphic()    # Perform check after init

        self.log(f"FlaxInterface initialized: dtype={self.dtype}, is_complex={self._iscpx}, nparams={self._param_num}, is_holomorphic={self.is_holomorphic}",
                log = 'info', lvl = 1, color = self._dcol)
        return self._parameters
    
    ########################################################
    #! SETTERS
    ########################################################
    
    def set_params(self, params: Any):
        """
        Set the network parameters, checking tree structure.
        """
        if not self._initialized:
            raise RuntimeError(self._ERR_NET_NOT_INITIALIZED)
        
        new_flat, new_tree = tree_flatten(params)
        if new_tree != self._param_tree_def:
            raise ValueError("New parameters have different tree structure.")
        self._parameters = params   # assume correct type is provided
    
    def get_params(self):
        """
        Get the current network parameters.
        """
        if not self._initialized: 
            raise RuntimeError(self._ERR_NET_NOT_INITIALIZED)
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
        
    def get_apply(self, use_jax = True) -> Tuple[Callable, dict]:
        """
        Return the apply function and current parameters.
        Params:
            use_jax: bool
                not used here as the flax module is always jax.
        Returns:
            Tuple[Callable, dict]
                The apply function and the current parameters.
        """
        if not use_jax:
            raise ValueError(self._ERR_JAX_NECESSARY)
        if not self._initialized or self._compiled_apply_fn is None:
            raise RuntimeError("Network or compiled apply function not initialized.")
        
        # Return the *compiled* apply function handle
        return self._compiled_apply_fn, self.get_params()
    
    def get_gradient(self, use_jax = True):
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
        r"""
        Checks if the network output is approximately holomorphic w.r.t. complex parameters.

        Uses Wirtinger calculus and `jax.grad` to numerically verify the Cauchy-Riemann
        equations, which are necessary and sufficient conditions for holomorphicity.

        Theory
        ------
        Let the complex network parameters be represented by a PyTree :math:`p`, where each
        leaf can be viewed as a collection of complex variables :math:`p_k = p_{k,R} + i p_{k,I}`.
        Let the complex output of the network for a fixed input :math:`x` be :math:`f(p) = u(p) + i v(p)`,
        where :math:`u` and :math:`v` are the real and imaginary parts of the output, respectively,
        viewed as functions of the real components :math:`p_{k,R}, p_{k,I}`.

        A function :math:`f` is **holomorphic** (or complex differentiable) with respect to
        its complex parameters :math:`p` if it satisfies the Cauchy-Riemann equations.
        In terms of Wirtinger derivatives, this is equivalent to the condition [1]:

        .. math::
            \frac{\partial f}{\partial p^*} = 0

        where :math:`\frac{\partial}{\partial p^*} = \frac{1}{2} \left( \frac{\partial}{\partial p_R} + i \frac{\partial}{\partial p_I} \right)`
        is the Wirtinger derivative with respect to the complex conjugate :math:`p^*`.

        Alternatively, using the standard gradient operator :math:`\nabla_p` which treats
        :math:`p` as a vector in a higher-dimensional real space (mapping :math:`p_k` to :math:`(p_{k,R}, p_{k,I})`),
        and defining the complex gradient [2] as:

        .. math::
            \nabla_p h = \frac{\partial h}{\partial p_R} + i \frac{\partial h}{\partial p_I}

        The Wirtinger derivatives relate to this complex gradient as:
        :math:`\frac{\partial f}{\partial p} = \frac{1}{2} (\nabla_p f)^*` and
        :math:`\frac{\partial f}{\partial p^*} = \frac{1}{2} (\nabla_p \bar{f})^* = \frac{1}{2} (\nabla_p (u-iv))^*`

        The condition :math:`\frac{\partial f}{\partial p^*} = 0` is also equivalent to the
        standard Cauchy-Riemann equations:
        1. :math:`\frac{\partial u}{\partial p_R} = \frac{\partial v}{\partial p_I}`
        2. :math:`\frac{\partial u}{\partial p_I} = - \frac{\partial v}{\partial p_R}`

        These two equations can be compactly written using the complex gradients:

        .. math::
            \nabla_p u = i \nabla_p v

        Implementation Check
        --------------------
        This function numerically verifies the condition :math:`\nabla_p u \approx i \nabla_p v`.
        It computes:
        - :math:`\text{grad}_u = \nabla_p u = \nabla_p (\text{Re}[f(p)])`
        - :math:`\text{grad}_v = \nabla_p v = \nabla_p (\text{Im}[f(p)])`
            using `jax.grad`. Note that `jax.grad` applied to a real-valued function
            of complex variables computes exactly this complex gradient :math:`\nabla_p h`.
        It then checks if the norm of the difference is relatively small:

        .. math::
            \frac{\| \text{grad}_u - i \cdot \text{grad}_v \|}{\| \text{grad}_u \| + \epsilon} \approx 0

        where the norms and subtraction are calculated element-wise across the flattened parameter PyTrees.

        References
        ----------
        [1] Wirtinger, W. "Zur formalen Theorie der Funktionen von mehr komplexen Veränderlichen."
            Mathematische Annalen 97.1 (1927): 357-375.
        [2] Sorber, L., Barel, M. V., & Lathauwer, L. D. "Unconstrained optimization of real functions
            in complex variables." SIAM Journal on Optimization 22.3 (2012): 879-898. (Section 2)

        Returns
        -------
            bool: True if the check indicates holomorphicity within tolerance `_TOL_HOLOMORPHIC`, False otherwise.
        """
        
        # Check if the network is initialized.
        if not self.initialized:
            
            # Attempt initialization if not done yet
            self.log("Warning: check_holomorphic called before explicit init(). Attempting initialization.",
                    log='warning', lvl=2, color=self._dcol)
            try:
                self.init()
            except Exception as e:
                self.log(f"Initialization failed in check_holomorphic: {e}", log='error', lvl=1, color='red')
                self._holomorphic = False
                return False
        
        # Check if the network is already checked.
        if self._holomorphic is not None:
            return self._holomorphic
        
        if not self.is_complex:
            self.log("Holomorphic check skipped: Parameters are real.",
                    log='warning', lvl=2, color=self._dcol)
            self._holomorphic = False
            return False
        
        # Ensure parameters are complex too
        param_leaves, _ = tree_flatten(self._parameters)
        if not all(jnp.issubdtype(p.dtype, jnp.complexfloating) for p in param_leaves):
            self.log("Holomorphic check requires complex parameters.", log='error', lvl=1, color='red')
            self._holomorphic = False
            return False
        
        # Check if the network is holomorphic.
        dummy_input             = jnp.ones((1, self.input_dim), dtype=self.dtype)
        current_params          = self.get_params()

        try:
            # Define the function Re[f(p)] for differentiation
            apply_handle = self._apply_jax_handle
            if apply_handle is None: raise RuntimeError("Internal apply handle not set.")

            def real_part_of_output(p_tree):
                output = apply_handle({'params': p_tree}, dummy_input)
                return jnp.real(output).sum() # Sum for scalar output needed by grad

            def imag_part_of_output(p_tree):
                output = apply_handle({'params': p_tree}, dummy_input)
                return jnp.imag(output).sum() # Sum for scalar output needed by grad
            
            # grad_u_tree = ∇p(Re[f]) = (∂u/∂pR) + i(∂u/∂pI)
            grad_u_tree         = jax.grad(real_part_of_output)(current_params)
            # grad_v_tree = ∇p(Im[f]) = (∂v/∂pR) + i(∂v/∂pI)
            grad_v_tree         = jax.grad(imag_part_of_output)(current_params)

            #! Check CR Condition: grad_u ≈ i * grad_v
            # Calculate the difference: diff = grad_u - i * grad_v
            # This should be close to zero if CR equations hold.
            diff_tree           = tree_map(lambda u, v: u - 1j * v, grad_u_tree, grad_v_tree)

            # Check Cauchy-Riemann: dRe/dx = dIm/dy, dRe/dy = -dIm/dx
            # Here, x, y are real/imag parts of params. Check d(Re(f))/dp_real vs d(Im(f))/dp_imag, etc.
            # A simpler proxy: check if grad(Re(f)) approx equals i * grad(Im(f)) for complex params
            # This requires careful handling of complex gradients.
            # Flatten the difference and one of the gradients (e.g., grad_u) for norm comparison
            diff_leaves, _      = tree_flatten(diff_tree)
            grad_u_leaves, _    = tree_flatten(grad_u_tree)

            if not grad_u_leaves: # Handle empty parameters/gradients
                self.log("Warning: Holomorphic check encountered empty gradient trees.",
                        log='warning', lvl=2, color=self._dcol)
                self._holomorphic   = True
            else:
                # Concatenate leaves into flat vectors
                flat_diff_vec       = jnp.concatenate([leaf.ravel() for leaf in diff_leaves])
                flat_grad_u_vec     = jnp.concatenate([leaf.ravel() for leaf in grad_u_leaves])

                # Calculate norms
                norm_diff           = jnp.linalg.norm(flat_diff_vec)
                norm_grad_u         = jnp.linalg.norm(flat_grad_u_vec)

                # Check if the relative difference is small
                # Avoid division by zero if norm_grad_u is zero
                relative_diff       = norm_diff / (norm_grad_u + 1e-12) # Add epsilon for stability
                self._holomorphic   = jnp.allclose(relative_diff, 0.0, atol=self._TOL_HOLOMORPHIC)
        except Exception as e:
            self.log(f"Error during Cauchy-Riemann check: {e}", lvl=0, log='error', color='red')
            self._holomorphic = False # Default to False on error

        self.log(f"Holomorphic check result (||∇Re[f] - i*∇Im[f]|| / ||∇Re[f]|| ≈ 0): {self._holomorphic}",
                log='info', lvl=1, color=self._dcol)
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
    
    @property
    def is_holomorphic(self):
        """
        Check if the network is holomorphic.
        Returns:
            bool: True if the network is holomorphic, False otherwise.
        """
        return self._holomorphic
    
    @property
    def is_complex(self):
        """
        Check if the network is complex.
        Returns:
            bool: True if the network is complex, False otherwise.
        """
        return self._iscpx
    
    @property
    def backend(self):
        """
        Get the backend used by the network.
        Returns:
            str: The backend used by the network.
        """
        return self._backend_str
    
    @property
    def dtype(self):
        """
        Get the data type of the network parameters.
        Returns:
            jnp.dtype: The data type of the network parameters.
        """
        return self._dtype
    
    @property
    def input_dim(self):
        """
        Get the input dimension of the network.
        Returns:
            int: The input dimension of the network.
        """
        return self._input_dim
    
    #########################################################
    
#############################################################