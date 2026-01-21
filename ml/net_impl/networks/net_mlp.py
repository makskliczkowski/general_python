"""
general_python.ml.net_impl.networks.net_mlp
=============================================

Multi-Layer Perceptron (MLP) for Quantum States.

A flexible, high-performance MLP implementation with support for:
- Complex-valued parameters.
- Split-complex optimization (Real backbone -> Complex output).
- Arbitrary depth and width.

Usage
-----
    from general_python.ml.networks import choose_network
    
    # 1. Standard MLP
    mlp     = choose_network('mlp', input_shape=(64,), hidden_dims=(128, 64), activations='relu')
    
    # 2. Split-Complex MLP
    mlp_opt = choose_network('mlp', input_shape=(64,), hidden_dims=(128, 64), split_complex=True)

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-15
License         : MIT
----------------------------------------------------------
"""

import  math
import  numpy   as np
from    typing  import Sequence, Callable, Optional, Any, Union, List, Tuple

try:
    import jax
    import jax.numpy    as jnp
    import flax.linen   as nn
    from ....ml.net_impl.interface_net_flax     import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax     import cplx_variance_scaling, lecun_normal
    from ....ml.net_impl.activation_functions   import get_activation_jnp
    JAX_AVAILABLE       = True
except ImportError:
    raise ImportError("MLP requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxMLP(nn.Module):
    hidden_dims     : Sequence[int]
    activations     : Sequence[Callable]
    output_dim      : int
    use_bias        : bool                  = True
    dtype           : Any                   = jnp.complex128
    param_dtype     : Any                   = jnp.complex128
    split_complex   : bool                  = False     # Optimize for Real inputs -> Complex output
    input_trans     : Optional[Callable]    = None      # e.g. x -> 2*x - 1

    def setup(self):
        # Determine internal dtypes
        if self.split_complex:
            # Force float types for backbone
            if jnp.issubdtype(self.dtype, jnp.complexfloating):
                c_dtype = jnp.float32 if self.dtype == jnp.complex64 else jnp.float64
            else:
                c_dtype = self.dtype
                
            if jnp.issubdtype(self.param_dtype, jnp.complexfloating):
                p_dtype = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
            else:
                p_dtype = self.param_dtype
        else:
            c_dtype = self.dtype
            p_dtype = self.param_dtype

        # Initializer
        if jnp.issubdtype(p_dtype, jnp.complexfloating):
            kernel_init = cplx_variance_scaling(1.0, 'fan_in', 'normal', p_dtype)
        else:
            kernel_init = nn.initializers.lecun_normal()

        # Build Hidden Layers
        self.layers = [
            nn.Dense(
                features    = h_dim,
                use_bias    = self.use_bias,
                dtype       = c_dtype,
                param_dtype = p_dtype,
                kernel_init = kernel_init,
                name        = f"Dense_{i}"
            ) for i, h_dim in enumerate(self.hidden_dims)
        ]

        # Output Layer
        out_features = self.output_dim * 2 if self.split_complex else self.output_dim
        self.dense_out = nn.Dense(
            features    = out_features,
            use_bias    = self.use_bias,
            dtype       = c_dtype,
            param_dtype = p_dtype,
            kernel_init = kernel_init,
            name        = "Dense_Out"
        )

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, input_dim)
        
        # 1. Preprocessing
        if self.split_complex:
            x = x.real if jnp.iscomplexobj(x) else x
            
        # Type cast to layer dtype
        x = x.astype(self.layers[0].dtype)

        if self.input_trans is not None:
            x = self.input_trans(x)

        # 2. Hidden Layers
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            x = act(x)

        # 3. Output
        x           = self.dense_out(x)

        # 4. Recombination
        if self.split_complex:
            # Split into Real/Imag parts
            re, im  = jnp.split(x, 2, axis=-1)
            x       = re + 1j * im
            
        return x

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class MLP(FlaxInterface):
    """
    Multi-Layer Perceptron (MLP) Interface.

    Parameters:
        input_shape (tuple): Shape of input (flattened internally).
        hidden_dims (Sequence[int]): Dimensions of hidden layers.
        activations (Union[str, Sequence]): Activation functions.
        output_shape (tuple): Shape of output.
        split_complex (bool): Optimization for real inputs -> complex output.
    """
    def __init__(self,
                input_shape    : tuple,
                hidden_dims    : Sequence[int]         = (64, 64),
                activations    : Union[str, Sequence]  = 'relu',
                output_shape   : tuple                 = (1,),
                use_bias       : bool                  = True,
                split_complex  : bool                  = False,
                transform_input: bool                  = False, # 0/1 -> -1/1
                dtype          : Any                   = jnp.complex128,
                param_dtype    : Optional[Any]         = None,
                seed           : int                   = 0,
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("MLP requires JAX.")

        # Resolve Activations
        if isinstance(activations, str):
            act_fn, _ = get_activation_jnp(activations)
            acts = (act_fn,) * len(hidden_dims)
        elif isinstance(activations, (list, tuple)):
            if len(activations) == 1:
                act_fn, _ = get_activation_jnp(activations[0])
                acts = (act_fn,) * len(hidden_dims)
            elif len(activations) == len(hidden_dims):
                acts = tuple(get_activation_jnp(a)[0] for a in activations)
            else:
                raise ValueError("activations list length must match hidden_dims")
        else:
            # Assume single callable
            acts = (activations,) * len(hidden_dims)

        net_kwargs = {
            'hidden_dims'   : hidden_dims,
            'activations'   : acts,
            'output_dim'    : int(np.prod(output_shape)),
            'use_bias'      : use_bias,
            'dtype'         : dtype,
            'param_dtype'   : param_dtype if param_dtype else dtype,
            'split_complex' : split_complex,
            'input_trans'   : (lambda x: 2*x - 1) if transform_input else None
        }

        super().__init__(
            net_module  = _FlaxMLP,
            net_kwargs  = net_kwargs,
            input_shape = input_shape,
            dtype       = dtype,
            seed        = seed,
            **kwargs
        )
        
        self._output_shape  = output_shape
        self._split_complex = split_complex
        self._name          = 'mlp'

    def __call__(self, x):
        flat_out = super().__call__(x)
        if self._output_shape == (1,):
            return flat_out.reshape(-1)
        return flat_out.reshape((-1,) + self._output_shape)

    def __repr__(self) -> str:
        kind = "SplitComplex" if self._split_complex else ("Complex" if self._iscpx else "Real")
        return f"{kind}MLP(hidden={self._flax_module.hidden_dims}, params={self.nparams})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------