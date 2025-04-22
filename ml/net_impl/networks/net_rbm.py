"""
file    : QES/general_python/ml/net_impl/networks/net_rbm.py
author  : Maksymilian Kliczkowski
date    : 2025-04-01

Flax implementations of Restricted Boltzmann Machines (RBMs) using the
FlaxInterface wrapper for compatibility with the QES framework.
"""

import numpy as np
from typing import Tuple, Callable, Optional, Any
from functools import partial

try:
    # Base Interface (essential)
    from general_python.ml.net_impl.interface_net_flax import FlaxInterface
    from general_python.ml.net_impl.activation_functions import log_cosh_jnp
    from general_python.ml.net_impl.utils.net_init_jax import cplx_variance_scaling, lecun_normal
    from general_python.algebra.utils import JAX_AVAILABLE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE
except ImportError as e:
    print(f"Error importing QES base modules: {e}")
    class FlaxInterface:
        pass
    JAX_AVAILABLE = False

# --- JAX / Flax Imports ---
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
else:
    raise ImportError("JAX is not available. Please install JAX to use this module.")


##########################################################
#! INNER FLAX RBM MODULE DEFINITION
##########################################################

class _FlaxRBM(nn.Module):
    """
    Inner Flax module for a Restricted Boltzmann Machine.

    Designed to be wrapped by `FlaxInterface`. Calculates the log-amplitude
    log(psi(s)) based on RBM formula.

    Attributes:
        n_hidden (int):
            Number of hidden units.
        bias (bool):
            Whether to include bias terms for hidden units.
        input_activation (Optional[Callable]):
            Activation applied to the input *before* the Dense layer.
            Commonly None or (lambda x: 2*x - 1) for spins {0,1} -> {-1,1}.
        param_dtype (jnp.dtype):
            Data type for the RBM parameters (weights, bias).
            Determines if it's a real or complex RBM.
        dtype (jnp.dtype):
            Data type for intermediate computations (usually matches param_dtype).
            Note: `act_fun` attribute from _FlaxNet is NOT used here.
    """
    n_hidden        : int
    bias            : bool                  = True                  # Bias for hidden units is common in RBMs
    input_activation: Optional[Callable]    = None                  # e.g., lambda x: 2*x-1
    visible_bias    : Optional[bool]        = True
    param_dtype     : jnp.dtype             = DEFAULT_JP_CPX_TYPE   # Default to complex
    dtype           : jnp.dtype             = DEFAULT_JP_CPX_TYPE   # Default to complex
    islog           : bool                  = True                  # Logarithmic form of the wavefunction
    
    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        r"""
        Calculates log(psi(s)) for the RBM.

        Formula: 
        
        ::math:: `log(psi(s)) = sum_j log(cosh( W_j * v + b_j )) + a * v`
        
        where v is the visible layer state (input `s`),
        W are weights,
        b is hidden bias,
        a is visible bias (often omitted or fixed).
        This implementation includes hidden bias `b` if `self.bias` is True,
        and omits the visible bias `a` for simplicity (common in VMC).

        Args:
            s (jax.Array): Input configuration(s) with shape (batch, n_visible).

        Returns:
            jax.Array: Log-amplitude(s) log(psi(s)) with shape (batch,).
        """
        # if not _JAX_AVAILABLE:
            # raise ImportError("Flax module requires JAX.")

        # Check if the input is complex
        complex_dtype       = jnp.issubdtype(self.param_dtype, jnp.complexfloating)
        
        # Apply input activation (e.g., map {0,1} to {-1,1})
        visible_state       = s.astype(self.dtype)
        if self.input_activation is not None:
            visible_state   = self.input_activation(visible_state)

        # Flatten input if necessary (RBM Dense layer expects rank 2: [batch, features])
        if visible_state.ndim > 2:
            visible_state   = visible_state.reshape(visible_state.shape[0], -1)

        #! Define Initializer based on parameter dtype
        if complex_dtype:
            kernel_init_fn  = cplx_variance_scaling(1.0, 'fan_in', 'normal', dtype=self.param_dtype)
            bias_init_fn    = jax.nn.initializers.zeros
        else:
            kernel_init_fn  = lecun_normal(dtype=self.param_dtype)
            bias_init_fn    = jax.nn.initializers.zeros
        
        #! Visible bias term
        n_visible           = visible_state.shape[-1]
        if self.visible_bias:
            visible_bias    = self.param("visible_bias", bias_init_fn,
                                (n_visible,), self.param_dtype)
        else:
            visible_bias    = None

        # Define the Dense layer (Visible to Hidden weights W and hidden bias b_h)
        # Input features    = number of visible units (s.shape[-1])
        # Output features   = number of hidden units
        dense_layer = nn.Dense(
            features    = self.n_hidden,
            name        = "VisibleToHidden",                # Name of the layer
            use_bias    = self.bias,                        # Include bias if True
            kernel_init = kernel_init_fn,                   # Weight initialization
            bias_init   = bias_init_fn,                     # Bias initialization
            dtype       = self.dtype,                       # Computation dtype
            param_dtype = self.param_dtype                  # Parameter dtype
        )

        # Calculate hidden layer activations (theta_j = W_j * v + b_j)
        theta           = dense_layer(visible_state)        # Shape (batch, n_hidden)

        # Apply log(cosh) activation to hidden activations
        log_cosh_theta  = log_cosh_jnp(theta)               # Shape (batch, n_hidden)
        
        # Sum over hidden units to get log(psi) for each batch element
        log_psi         = jnp.sum(log_cosh_theta, axis=-1)  # Shape (batch,)
        if self.visible_bias:
            log_psi     += jnp.sum(visible_state * visible_bias, axis=-1)
            
        # Add visible bias if needed (not included in this implementation)
        if self.islog:
            return log_psi.reshape(-1)
        return jnp.exp(log_psi).reshape(-1)                 # Ensure output is 1D (batch,)

##########################################################
#! RBM WRAPPER CLASSES USING FlaxInterface
##########################################################

class RBM(FlaxInterface):
    """
    Restricted Boltzmann Machine (RBM) based on FlaxInterface.

    Supports both real and complex RBMs by configuring the data types.

    Parameters:
        input_shape (tuple): 
            Input shape (excluding batch dimension), e.g., (n_sites,).
        n_hidden (int): 
            Number of hidden units. Default is 2.
        bias (bool): 
            Whether to include hidden bias terms. Default True.
        map_to_pm1 (bool): 
            If True, map input {0,1} to {-1,1} before processing. Default True.
        dtype (jnp.dtype): 
            Data type for computations. Default float32 for real RBM, complex64 for complex RBM.
        param_dtype (Optional[jnp.dtype]): 
            Data type for parameters. Defaults to dtype.
        seed (int): 
            Seed for parameter initialization. Default 0.
        _is_cpx (bool): 
            Whether the RBM is complex. Default False.
    """
    def __init__(self,
                input_shape    : tuple,
                n_hidden       : int            = 2,
                bias           : bool           = True,
                visible_bias   : bool           = True,
                in_activation  : bool           = False,                   # Flag to map {0,1} -> {-1,1}
                dtype          : Any            = DEFAULT_JP_FLOAT_TYPE,   # Default float for real RBM
                param_dtype    : Optional[Any]  = None,
                seed           : int            = 0,
                **kwargs):

        if not JAX_AVAILABLE:
            raise ImportError("RBM requires JAX.")

        # Determine dtypes
        final_dtype             = jnp.dtype(dtype)
        final_param_dtype       = jnp.dtype(param_dtype) if param_dtype is not None else final_dtype
        self._is_cpx            = jnp.issubdtype(final_param_dtype, jnp.complexfloating)

        # Basic type compatibility check
        is_final_cpx = jnp.issubdtype(final_dtype, jnp.complexfloating)
        self._is_cpx = jnp.issubdtype(final_param_dtype, jnp.complexfloating)
        if is_final_cpx != self._is_cpx:
            self.log(f"Warning: RBM dtype ({final_dtype}) and param_dtype ({final_param_dtype}) differ in complexity. "
                    "Ensure this is intended.", log='warning', lvl=1, color='yellow')
    
        # Define input activation based on map_to_pm1 flag
        # input_activation        = (lambda x: 2 * x - 1) if map_to_pm1 else None
        self._in_activation     = in_activation
        # Prepare kwargs for _FlaxRBM
        net_kwargs = {
            'n_hidden'        : n_hidden,
            'bias'            : bias,
            'visible_bias'    : visible_bias,
            'input_activation': None,
            'param_dtype'     : final_param_dtype,
            'dtype'           : final_dtype,
            **kwargs
            
        }

        # Initialize using FlaxInterface parent
        super().__init__(
            net_module  = _FlaxRBM,     # The inner Flax module CLASS
            net_args    = (),           # No positional args for _FlaxRBM
            net_kwargs  = net_kwargs,   # Keyword args for _FlaxRBM
            input_shape = input_shape,
            backend     = 'jax',        # Force JAX backend
            dtype       = final_dtype,  # Pass the COMPUTATION dtype to FlaxInterface
            seed        = seed
        )

        #! For the analytic gradient, we need to compile the function
        self._compiled_grad_fn  = jax.jit(partial(RBM.analytic_grad_jax, input_activation=self._in_activation))
        # self._has_analytic_grad = True

    # ------------------------------------------------------------
    #! Analytic Gradient
    # ------------------------------------------------------------
    
    @staticmethod
    @partial(jax.jit, static_argnames=("input_activation",))
    def analytic_grad_jax(params: Any, x: jax.Array, input_activation: Optional[Callable] = None) -> Any:
        r"""
        Computes the analytical gradient of log(psi(s)) for the RBM.

        Calculates the derivatives d log(psi)/dp averaged over the batch,
        where p are the parameters (visible_bias, hidden_bias, weights).

        Gradient Formulas:
            d log(psi) / d a_i  = s_i
            d log(psi) / d b_j  = tanh(theta_j)
            d log(psi) / d W_ij = s_i * tanh(theta_j)
        where theta_j           = sum_i W_ij * s_i + b_j

        Args:
            params (Any):   
                PyTree of network parameters (matching _FlaxRBM structure).
                Expected keys: 'visible_bias', 'VisibleToHidden' {'kernel', 'bias'}.
            x (jax.Array):
                Input batch of configurations, shape (batch, n_visible) or (batch, *input_shape).
            input_activation (Optional[Callable]):
                The same activation function used in the forward pass.

        Returns:
            Any:
                A PyTree with the same structure as `params`, containing the
                batch-averaged gradients for each parameter.
        """
        
        #! Parameter Extraction
        has_visible_bias    = 'visible_bias' in params
        has_hidden_bias     = 'bias' in params.get('VisibleToHidden', {})

        W = params['VisibleToHidden']['kernel']                             # Shape (n_visible, n_hidden)
        b = params['VisibleToHidden']['bias'] if has_hidden_bias else None  # Shape (n_hidden,)
        a = params['visible_bias'] if has_visible_bias else None            # Shape (n_visible,)

        #! Input Preprocessing 
        # Ensure input has batch dimension and is flattened
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        # Apply the same input activation as the forward pass
        # Determine compute dtype from weights/biases if possible, fallback needed
        compute_dtype = W.dtype
        visible_state = x.astype(compute_dtype)
        if input_activation is not None:
            visible_state = input_activation(visible_state) # Shape (batch, n_visible)

        #! Intermediate Calculations
        # theta = s * W + b (using einsum for clarity, handles batching)
        # W: (n_vis, n_hid), visible_state: (batch, n_vis) -> theta: (batch, n_hid)
        theta = jnp.einsum('bi,ih->bh', visible_state, W)
        if b is not None:
            theta = theta + b

        tanh_theta = jnp.tanh(theta)                                        # Shape (batch, n_hid)

        #! Gradient Calculations (per batch element)
        # grad_a_batch = visible_state                                      # Shape (batch, n_visible)
        # grad_b_batch = tanh_theta                                         # Shape (batch, n_hidden)
        # grad_W_batch: Need outer product s_i * tanh(theta_j) for each batch item
        # visible_state: (b, i), tanh_theta: (b, j) -> grad_W: (b, i, j)
        grad_W_batch = jnp.einsum('bi,bj->bij', visible_state, tanh_theta)

        # --- Averaging over Batch ---
        # Use tree_map to average only the leaves that correspond to gradients
        batch_grads = {}
        if a is not None:
            batch_grads['visible_bias'] = visible_state # grad_a_batch
        if W is not None: # Should always exist
            # Need to nest W and b correctly
            hidden_grads = {'kernel': grad_W_batch}
            if b is not None:
                hidden_grads['bias'] = tanh_theta # grad_b_batch
            batch_grads['VisibleToHidden'] = hidden_grads

        #! Return Gradient Tree
        # Ensure the structure matches the original params tree.
        # If a bias was missing in params, it shouldn't be in the grad tree.
        # The construction above handles this.
        return batch_grads

    # ------------------------------------------------------------
    #! Public Methods
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        init_status = "initialized" if self.initialized else "uninitialized"
        rbm_type    = "Complex" if self._is_cpx else "Real"
        n_hidden    = self._flax_module.n_hidden if self.initialized else self._net_kwargs.get('n_hidden', '?')
        bias        = "on" if (self._flax_module.bias if self.initialized else self._net_kwargs.get('bias', '?')) else "off"
        vis_bias    = "on" if (self._flax_module.visible_bias if self.initialized else self._net_kwargs.get('visible_bias', '?')) else "off"
        n_params = self.nparams if self.initialized else '?'
        return (f"{rbm_type}RBM(shape={self.input_shape}, hidden={n_hidden}, "
            f"bias={bias}, visible_bias={vis_bias}, dtype={self.dtype}, params={n_params}, analytic_grad={self._has_analytic_grad}, {init_status})")