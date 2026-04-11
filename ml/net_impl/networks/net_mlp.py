"""
general_python.ml.net_impl.networks.net_mlp
=============================================

Generic multi-layer perceptron wrapper based on Flax.

The implementation supports complex parameters, split-complex execution, and
optional input preprocessing through a generic callable hook.
The MLP is a versatile and widely used architecture that can serve as a building block for more complex models or as a standalone ansatz. 
The current implementation focuses on flexibility and ease of use, allowing for various configurations of hidden layers, 
activations, and output shapes.

WIP - THIS MODULE IS EXPERIMENTAL AND SUBJECT TO SIGNIFICANT CHANGES. DO NOT USE OUTSIDE TESTS OR INTERNAL PROTOTYPING.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-15
License         : MIT
Version         : 0.1 (Experimental)
----------------------------------------------------------
"""

import numpy as np
from typing import Sequence, Callable, Optional, Any, Union

try:
    import jax
    import jax.numpy    as jnp
    import flax.linen   as nn
    from ....ml.net_impl.interface_net_flax         import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax         import cplx_variance_scaling, lecun_normal
    from ....ml.net_impl.utils.net_wrapper_utils    import (
                                                        combine_split_complex_output,
                                                        prepare_split_complex_input,
                                                        resolve_split_complex_dtypes,
                                                        normalize_activation_sequence,
                                                        resolve_input_adapter,
                                                    )
    JAX_AVAILABLE       = True
except ImportError:
    raise ImportError("MLP requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxMLPBase(nn.Module):
    """Shared dense MLP backbone logic for direct and split-complex variants."""
    hidden_dims     : Sequence[int]
    activations     : Sequence[Callable]
    output_dim      : int
    use_bias        : bool                  = True
    dtype           : Any                   = jnp.complex128
    param_dtype     : Any                   = jnp.complex128
    input_adapter   : Optional[Callable]    = None

    def _resolve_dtypes(self):
        raise NotImplementedError

    def _head_features(self):
        raise NotImplementedError

    def _prepare_input(self, x):
        return x

    def _project_output(self, x):
        raise NotImplementedError

    def setup(self):
        # Determine internal dtypes
        c_dtype, p_dtype = self._resolve_dtypes()
        self._comp_dtype = c_dtype
        self._complex_out_dtype = jnp.complex64 if c_dtype == jnp.float32 else jnp.complex128

        # Initializer
        if jnp.issubdtype(p_dtype, jnp.complexfloating):
            kernel_init = cplx_variance_scaling(1.0, 'fan_in', 'normal', p_dtype)
        else:
            kernel_init = lecun_normal(p_dtype)

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
        self.dense_out = nn.Dense(
            features    = self._head_features(),
            use_bias    = self.use_bias,
            dtype       = c_dtype,
            param_dtype = p_dtype,
            kernel_init = kernel_init,
            name        = "Dense_Out"
        )

    @nn.compact
    def __call__(self, x):
        """Evaluate the backbone on a single sample or a batch of samples."""
        needs_batch = x.ndim == 1
        if needs_batch:
            x = x[jnp.newaxis, ...]
        
        # 1. Preprocessing
        x = self._prepare_input(x)
            
        # Type cast to layer dtype
        x = x.astype(self._comp_dtype)

        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # 2. Hidden Layers
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            x = act(x)

        # 3. Output
        x           = self.dense_out(x)

        # 4. Recombination
        x = self._project_output(x)

        if self.output_dim == 1:
            x = x.reshape((x.shape[0],))

        return x[0] if needs_batch else x


class _FlaxMLPDirect(_FlaxMLPBase):
    """Direct-output MLP for real or complex dtypes."""

    split_complex: bool = False

    def _resolve_dtypes(self):
        return self.dtype, self.param_dtype

    def _head_features(self):
        return self.output_dim

    def _project_output(self, x):
        return x


class _FlaxMLPSplitComplex(_FlaxMLPBase):
    """Split-complex MLP with real hidden layers and complex output reconstruction."""

    split_complex: bool = True

    def _resolve_dtypes(self):
        c_dtype, p_dtype, _ = resolve_split_complex_dtypes(self.dtype, self.param_dtype)
        return c_dtype, p_dtype

    def _head_features(self):
        return 2 * self.output_dim

    def _prepare_input(self, x):
        return prepare_split_complex_input(x)

    def _project_output(self, x):
        re, im = jnp.split(x, 2, axis=-1)
        return combine_split_complex_output(re, im, self._comp_dtype)

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class MLP(FlaxInterface):
    """
    Generic multi-layer perceptron wrapper.

    Parameters:
        input_shape (tuple):
            Input shape excluding the batch dimension.
        hidden_dims (Sequence[int]):
            Width of each hidden layer.
        activations (Union[str, Sequence]):
            Activation specification shared across layers or provided per layer.
        output_shape (tuple):
            Output shape returned by the wrapper.
        split_complex (bool):
            Use a real-valued backbone with paired real/imaginary outputs.
        input_adapter (Optional[Callable]):
            Optional preprocessing applied after casting and before the first dense layer.
    """
    def __init__(self,
                input_shape    : tuple,
                hidden_dims    : Sequence[int]         = (64, 64),
                activations    : Union[str, Sequence]  = 'relu',
                output_shape   : tuple                 = (1,),
                use_bias       : bool                  = True,
                split_complex  : bool                  = False,
                input_adapter  : Optional[Callable]    = None,
                dtype          : Any                   = jnp.complex128,
                param_dtype    : Optional[Any]         = None,
                seed           : int                   = 0,
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("MLP requires JAX.")

        acts = normalize_activation_sequence(
            activations,
            len(hidden_dims),
            default='relu',
            container=tuple,
        )
        input_convention, input_adapter = resolve_input_adapter(kwargs, input_adapter)

        net_kwargs = {
            'hidden_dims'       : hidden_dims,
            'activations'       : acts,
            'output_dim'        : int(np.prod(output_shape)),
            'use_bias'          : use_bias,
            'dtype'             : dtype,
            'param_dtype'       : param_dtype if param_dtype else dtype,
            'split_complex'     : split_complex,
            'input_adapter'     : input_adapter,
        }

        mlp_module = _FlaxMLPSplitComplex if split_complex else _FlaxMLPDirect

        super().__init__(
            net_module  = mlp_module,
            net_kwargs  = net_kwargs,
            input_shape = input_shape,
            dtype       = dtype,
            seed        = seed,
            **kwargs
        )
        
        self._output_shape  = output_shape
        self._output_size   = int(np.prod(output_shape))
        self._split_complex = split_complex
        self._input_convention = dict(input_convention)
        self._name          = 'mlp'

    def __call__(self, x):
        flat_out = super().__call__(x)
        if self._output_shape == (1,):
            if flat_out.ndim == 0:
                return flat_out
            return flat_out.reshape(-1)

        if flat_out.ndim == 0:
            return flat_out
        if flat_out.ndim == 1:
            if hasattr(x, "ndim") and x.ndim > 1:
                return flat_out.reshape((x.shape[0],) + self._output_shape)
            return flat_out.reshape(self._output_shape)
        if flat_out.shape[-1] == self._output_size:
            return flat_out.reshape(flat_out.shape[:-1] + self._output_shape)
        return flat_out

    def __repr__(self) -> str:
        kind = "SplitComplex" if self._split_complex else ("Complex" if self._iscpx else "Real")
        return f"{kind}MLP(hidden={self._flax_module.hidden_dims}, params={self.nparams})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
