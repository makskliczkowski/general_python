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
    # Define dummy types/functions if JAX/Flax not available
    class nn: 
        class Module: 
            pass
    jnp                     = np
    random                  = np.random
    DEFAULT_JP_FLOAT_TYPE   = np.float32
    DEFAULT_JP_CPX_TYPE     = np.complex64
    # Dummy decorators/functions
    partial                 = lambda fn, **kwargs: fn
    jit                     = lambda fn: fn
    # Dummy initializer functions
    def cplx_variance_scaling(*args, **kwargs):
        return lambda k, s, d: jnp.zeros(s, dtype=d)
    def lecun_normal(*args, **kwargs):
        return lambda k, s, d: jnp.zeros(s, dtype=d)
    log_cosh                = jnp.log


##########################################################
#! INNER FLAX RBM MODULE DEFINITION
##########################################################

@jax.jit
def stable_logcosh(x):
    sgn_x   = -2 * jnp.signbit(x.real) + 1
    x       = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)

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
        theta           = dense_layer(visible_state.ravel())# Shape (batch, n_hidden)

        # Apply log(cosh) activation to hidden activations
        log_cosh_theta  = stable_logcosh(theta)             # Shape (batch, n_hidden)
        
        # Sum over hidden units to get log(psi) for each batch element
        log_psi         = jnp.sum(log_cosh_theta, axis=-1)  # Shape (batch,)
        if self.visible_bias:
            log_psi     += jnp.sum(visible_state * visible_bias, axis=-1)
            
        # Add visible bias if needed (not included in this implementation)
        return log_psi.reshape(-1)
        # return log_psi.reshape(-1, 1)                     # Ensure output is 1D (batch, 1)
        # return log_psi.reshape(-1)                        # Ensure output is 1D (batch,)

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
                map_to_pm1     : bool           = False,                   # Flag to map {0,1} -> {-1,1}
                dtype          : Any            = DEFAULT_JP_FLOAT_TYPE,   # Default float for real RBM
                param_dtype    : Optional[Any]  = None,
                seed           : int            = 0,
                **kwargs):

        if not JAX_AVAILABLE:
            raise ImportError("RBM requires JAX.")

        # Determine dtypes
        final_dtype             = jnp.dtype(dtype)
        final_param_dtype       = jnp.dtype(param_dtype) if param_dtype is not None else final_dtype

        # if the types are not the same, raise an error
        if final_dtype == jnp.complexfloating and final_param_dtype == jnp.float_ or \
            final_dtype == jnp.float_ and final_param_dtype == jnp.complexfloating:
            raise ValueError("RBM: dtype and param_dtype must be the same floating (complex or real).")
        
        # Define input activation based on map_to_pm1 flag
        # input_activation        = (lambda x: 2 * x - 1) if map_to_pm1 else None

        # Prepare kwargs for _FlaxRBM
        net_kwargs = {
            'n_hidden'        : n_hidden,
            'bias'            : bias,
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

        self._is_cpx = final_param_dtype == jnp.complexfloating

    def __repr__(self) -> str:
        init_status = "initialized" if self.initialized else "uninitialized"
        rbm_type    = "Complex" if self._is_cpx else "Real"
        n_hidden    = self._net_kwargs_in.get('n_hidden', '?')
        bias        = "on" if self._net_kwargs_in.get('bias', False) else "off"
        return (f"{rbm_type}RBM(shape={self.input_shape}, hidden={n_hidden}, "
            f"bias={bias}, dtype={self.dtype}, params={self.num_parameters}, {init_status})")
