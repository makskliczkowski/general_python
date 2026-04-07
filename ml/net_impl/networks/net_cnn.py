"""
general_python.ml.net_impl.networks.net_cnn
===========================================

Generic convolutional wrapper based on Flax.

The wrapper accepts flattened inputs, reshapes them into a configured spatial
layout, and applies periodic or standard convolutions. Optional preprocessing
is handled through generic callable hooks rather than domain-specific input
convention flags.

------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-21
License         : MIT
Version         : 1.0
------------------------------------------
"""


import numpy as np
from typing import List, Tuple, Callable, Optional, Any, Sequence, Union, TYPE_CHECKING
import math
import warnings

try:
    import                                          jax
    import                                          jax.numpy as jnp
    import                                          flax.linen as nn
    from ....ml.net_impl.interface_net_flax         import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax         import cplx_variance_scaling
    from ....ml.net_impl.utils.net_wrapper_utils    import (
                                                        as_spatial_tuple,               # Utility to convert int or tuple to spatial tuple
                                                        combine_split_complex_output,   # Utility to combine separate real outputs into complex output
                                                        prepare_split_complex_input,    # Utility to prepare complex input for split-complex processing (e.g., by concatenating real and imaginary parts)
                                                        configure_nqs_metadata,         # Utility to set metadata attributes for NQS compatibility
                                                        extract_input_convention,       # Utility to parse explicit input-state convention flags
                                                        infer_native_representation,    # Utility to infer preferred external state encoding
                                                        make_state_input_adapter,       # Utility to build a shared signed-spin adapter
                                                        normalize_activation_sequence,  # Utility to normalize activation specifications into a sequence of callables    
                                                        normalize_layerwise_spec,       # Utility to normalize layerwise specifications (like kernel sizes, use_bias) into sequences of correct length
                                                        resolve_activation_spec,        # Utility to resolve activation specifications (string or callable) into a callable activation function
                                                        resolve_split_complex_dtypes,   # Utility to resolve the appropriate real and complex dtypes for split-complex processing based on the provided dtype and parameter dtype
                                                    )
    if TYPE_CHECKING:
        from ....algebra.utils                      import Array
    JAX_AVAILABLE                                   = True
except ImportError as e:
    raise ImportError("Could not import general_python modules. Ensure general_python is properly installed.") from e

##########################################################
#! INNER FLAX CNN MODULE DEFINITION
##########################################################

def circular_pad(x, kernel_sizes):
    """Apply wraparound padding for one convolution kernel."""
    # Create padding configuration
    # (0,0) for batch, (0,0) for channel
    # For spatial dims: pad k//2 left, (k-1)//2 right
    pads = [(0, 0)] 
    for k in kernel_sizes:
        p_left  = k // 2
        p_right = (k - 1) // 2
        pads.append((p_left, p_right))
    pads.append((0, 0))
    
    return jnp.pad(x, pads, mode='wrap')


class _FlaxCNNBase(nn.Module):
    """Shared convolutional backbone logic for direct and split-complex CNNs."""
    reshape_dims   : Tuple[int, ...]            # e.g. (8, 8)
    features       : Sequence[int]              # conv channels, e.g. (16, 32)
    kernel_sizes   : Sequence[Tuple[int, ...]]  # e.g. ((3, 3), (3, 3))
    strides        : Sequence[Tuple[int, ...]]  # e.g. ((1, 1), (1, 1))
    activations    : Sequence[Callable]         # already resolved callables
    use_bias       : Sequence[bool]
    output_feats   : int                        # prod(output_shape)
    
    # Configs
    param_dtype    : jnp.dtype                  = jnp.complex64
    dtype          : jnp.dtype                  = jnp.complex64
    input_channels : int                        = 1
    periodic       : bool                       = True
    use_sum_pool   : bool                       = True
    input_adapter  : Optional[Callable]         = None
    islog          : bool                       = True
    
    in_act         : Optional[Callable]         = None
    out_act        : Optional[Callable]         = None

    def _resolve_comp_dtype(self):
        raise NotImplementedError

    def _head_features(self):
        raise NotImplementedError

    def _prepare_input(self, s: jax.Array) -> jax.Array:
        return s

    def _project_output(self, x: jax.Array, batch_size: int) -> jax.Array:
        raise NotImplementedError

    def setup(self):
        self._comp_dtype = self._resolve_comp_dtype()
        if jnp.issubdtype(self._comp_dtype, jnp.complexfloating):
            self._kernel_init = cplx_variance_scaling(1.0, 'fan_in', 'normal', self._comp_dtype)
        else:
            self._kernel_init = nn.initializers.lecun_normal(dtype=self._comp_dtype)

        # Build convolutional layers by iterating over features, kernel sizes, strides, and bias flags
        self.conv_layers = [
            nn.Conv(
                features        = feat,
                kernel_size     = k_size,
                strides         = stride,
                padding         = 'VALID' if self.periodic else 'SAME',
                use_bias        = bias,
                dtype           = self._comp_dtype,
                param_dtype     = self._comp_dtype,
                kernel_init     = self._kernel_init,
                name            = f'conv_{i}',
            )
            
            # Iterate over layers, zipping together the corresponding features, kernel sizes, strides, and bias flags
            for i, (feat, k_size, stride, bias) in enumerate(
                zip(self.features, self.kernel_sizes, self.strides, self.use_bias)
            )
        ]
        
        # Output projection layer (fully connected) after convolutional backbone
        self.head_projection = nn.Dense(
            features    = self._head_features(),
            dtype       = self._comp_dtype,
            param_dtype = self._comp_dtype,
            kernel_init = self._kernel_init,
            name        = 'head_projection',
        )

    def __call__(self, s: jax.Array) -> jax.Array:
        """Evaluate the convolutional backbone on one sample or a batch."""
        # Ensure batch dimension: (B, N)
        needs_batch = s.ndim == 1
        if s.ndim == 1:
            s = s[jnp.newaxis, ...]

        batch_size      = s.shape[0]
        target_shape    = (batch_size,) + tuple(int(d) for d in self.reshape_dims) + (self.input_channels,)
        s_proc          = self._prepare_input(s)

        # 1. Reshape & Transform
        x               = s_proc.reshape(target_shape)
        x               = x.astype(self._comp_dtype)
            
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # Input Activation
        if self.in_act is not None:
            x = self.in_act(x)

        # 2. Convolutions
        for conv_layer, k_size, act_fn in zip(self.conv_layers, self.kernel_sizes, self.activations):
            # Periodic Padding
            if self.periodic:
                x = circular_pad(x, k_size) # Apply circular padding to the input before convolution
            x = conv_layer(x)
            
            # Activation
            if act_fn is not None:
                x = act_fn(x)

        # Pooling
        if self.use_sum_pool:
            spatial_axes    = tuple(range(1, x.ndim - 1))
            x               = jnp.sum(x, axis=spatial_axes)
        else:
            x               = x.reshape((batch_size, -1))

        # 3. Output Projection
        out_val = self._project_output(x, batch_size)

        # Squeeze last dim if output_feats is 1
        if self.output_feats == 1:
            out_val         = out_val.reshape((batch_size,))

        # 4. Output Activation
        if self.out_act is not None:
            out_val = self.out_act(out_val)

        # Final Scale
        if not self.islog:
            out_val = jnp.exp(out_val)
        
        # Return scalar for single sample, array for batch
        return out_val[0] if needs_batch else out_val


class _FlaxCNNDirect(_FlaxCNNBase):
    """Direct-output CNN backbone for real or complex dtypes."""

    split_complex: bool = False

    def _resolve_comp_dtype(self):
        return self.dtype

    def _head_features(self):
        return self.output_feats

    def _project_output(self, x: jax.Array, batch_size: int) -> jax.Array:
        del batch_size # not needed for direct output, but required for split-complex path
        return self.head_projection(x)

class _FlaxCNNSplitComplex(_FlaxCNNBase):
    """Split-complex CNN backbone with real intermediate arithmetic."""

    split_complex: bool = True

    def _resolve_comp_dtype(self):
        _, param_real_dtype, _ = resolve_split_complex_dtypes(self.dtype, self.param_dtype)
        return param_real_dtype

    def _head_features(self):
        return 2 * self.output_feats

    def _prepare_input(self, s: jax.Array) -> jax.Array:
        return prepare_split_complex_input(s)

    def _project_output(self, x: jax.Array, batch_size: int) -> jax.Array:
        out = self.head_projection(x).reshape((batch_size, self.output_feats, 2))
        return combine_split_complex_output(out[..., 0], out[..., 1], self._comp_dtype)

##########################################################
#! CNN WRAPPER CLASS USING FlaxInterface
##########################################################

class CNN(FlaxInterface):
    r"""
    Convolutional Neural Network (CNN) Interface.

    Parameters
    ----------
    input_shape (tuple): 
        Shape of the 1D input vector.
    reshape_dims (Tuple[int, ...]): 
        Spatial dimensions (Lx, Ly).
    features (Sequence[int]): 
        Channels per layer.
    kernel_sizes (Sequence[int/Tuple]):
        Kernel sizes.
    split_complex (bool):
        If True, uses a real-valued backbone with two real outputs (Re, Im) combined at the end.
    periodic (bool): 
        Periodic boundary conditions. Default: True.
    sum_pooling (bool): 
        Sum over spatial dimensions. Default: True.
    input_adapter (Optional[Callable]):
        Optional preprocessing applied after reshaping and before the first convolution.
    """
    def __init__(self, 
                input_shape         : tuple,
                reshape_dims        : Tuple[int, ...]                                       = None,
                features            : Sequence[int]                                         = (8,),
                kernel_sizes        : Sequence[Union[int, Tuple[int,...]]]                  = 3,
                strides             : Optional[Sequence[Union[int, Tuple[int,...]]]]        = None,
                activations         : Union[str, Callable, Sequence[Union[str, Callable]]]  = 'log_cosh',
                use_bias            : Union[bool, Sequence[bool]]                           = True,
                output_shape        : Tuple[int, ...]                                       = (1,),
                input_activation    : Optional[Any]                                         = None,
                final_activation    : Union[str, Callable, None]                            = None,
                input_adapter       : Optional[Callable]                                    = None,
                *,
                split_complex       : bool                                                  = False,
                islog               : bool                                                  = True,
                dtype               : Any                                                   = jnp.float32,
                param_dtype         : Optional[Any]                                         = None,
                seed                : int                                                   = 0,
                **kwargs):

        if not JAX_AVAILABLE:
            raise ImportError("CNN requires JAX.")

        # Shape Logic 
        if len(input_shape) == 1 and reshape_dims is None:
            reshape_dims = (input_shape[0], 1)
        elif len(input_shape) > 1 and reshape_dims is None:
            reshape_dims = input_shape

        n_visible   = input_shape[0]
        n_dim       = len(reshape_dims)
        n_layers    = len(features)
        if n_layers == 0:
            raise ValueError("features must contain at least one layer")
        
        if math.prod(reshape_dims) != n_visible:
            raise ValueError(f"reshape_dims {reshape_dims} product != input length {n_visible}")

        # Args normalization
        kernels     = tuple(as_spatial_tuple(item, n_dim, name="kernel_sizes") for item in normalize_layerwise_spec(kernel_sizes, n_layers, name="kernel_sizes"))
        stride_spec = 1 if strides is None else strides
        strides_t   = tuple(as_spatial_tuple(item, n_dim, name="strides") for item in normalize_layerwise_spec(stride_spec, n_layers, name="strides"))

        # Activations
        acts        = normalize_activation_sequence(
                        activations,
                        n_layers,
                        default='log_cosh',
                        container=tuple,
                    )
        input_convention = extract_input_convention(kwargs)
        if input_adapter is None:
            input_adapter = make_state_input_adapter(input_convention)

        # Bias
        bias_flags  = tuple(bool(b) for b in normalize_layerwise_spec(use_bias, n_layers, name="use_bias"))

        p_dtype         = param_dtype if param_dtype is not None else dtype
        output_feats    = int(np.prod(output_shape))
        # Build Module Config
        net_kwargs = dict(
            reshape_dims    =   reshape_dims,
            features        =   tuple(features),
            kernel_sizes    =   kernels,
            strides         =   strides_t,
            activations     =   acts,
            use_bias        =   bias_flags,
            output_feats    =   output_feats,
            in_act          =   resolve_activation_spec(input_activation),
            out_act         =   resolve_activation_spec(final_activation),
            dtype           =   dtype,
            param_dtype     =   p_dtype,
            input_channels  =   1,
            periodic        =   kwargs.get('periodic',      True),
            use_sum_pool    =   kwargs.get('sum_pooling',   True),
            split_complex   =   split_complex,
            islog           =   islog,
            input_adapter   =   input_adapter,
        )
        if (jnp.issubdtype(dtype, jnp.complexfloating) != jnp.issubdtype(p_dtype, jnp.complexfloating)):
            warnings.warn(f"Input dtype {dtype} and parameter dtype {p_dtype} have different complex types.")

        # Select Module Class and finalize kwargs based on split_complex
        self._out_shape     = output_shape
        self._output_size   = output_feats
        self._split_complex = split_complex
        cnn_module          = _FlaxCNNSplitComplex if split_complex else _FlaxCNNDirect

        super().__init__(
            net_module  =   cnn_module,
            net_args    =   (),
            net_kwargs  =   net_kwargs,
            input_shape =   input_shape,
            backend     =   "jax",
            dtype       =   dtype,
            seed        =   seed,
        )

        self._has_analytic_grad             = False
        self._name                          = 'cnn'
        configure_nqs_metadata(
            self,
            family="cnn",
            native_representation=infer_native_representation(input_convention),
        )

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._out_shape

    # -------------------------------------------------------------

    def __call__(self, s: 'Array'):
        flat_output = super().__call__(s)

        if self._out_shape == (1,):
            if flat_output.ndim == 0:
                return flat_output
            return flat_output.reshape(-1)

        # Vector/tensor output path
        if flat_output.ndim == 0:
            return flat_output
        
        # If output is already the correct shape, return as is
        if flat_output.ndim == 1:
            if hasattr(s, "ndim") and s.ndim > 1:
                batch = s.shape[0]
                return flat_output.reshape((batch,) + self._out_shape)
            return flat_output.reshape(self._out_shape)
        
        if flat_output.shape[-1] == self._output_size:
            return flat_output.reshape(flat_output.shape[:-1] + self._out_shape)
        
        return flat_output

    # -------------------------------------------------------------

    def __repr__(self) -> str:
        kind    = "SplitComplex" if self._split_complex else ("Complex" if self._iscpx else "Real")
        return (
            f"{kind}CNN(reshape={self._flax_module.reshape_dims}, "
            f"features={self._flax_module.features}, "
            f"kernels={self._flax_module.kernel_sizes}, "
            f"dtype={self.dtype}, periodic={self._flax_module.periodic}, "
            f"params={self.nparams})"
        )

    def __str__(self) -> str:
        return self.__repr__()

###########################################################
#! EOF
###########################################################
