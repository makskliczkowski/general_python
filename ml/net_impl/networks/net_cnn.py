"""
general_python.ml.net_impl.networks.net_cnn
===============================================

Convolutional Neural Network (CNN) for Quantum States.

This module provides a Flax-based implementation of a Convolutional Neural
Network (CNN), designed for representing quantum wavefunctions on a lattice.
It processes 1D input vectors by reshaping them into spatial structures (e.g., a 2D grid)
and applying convolutional layers.

Optimization Features:
----------------------
- **Split-Complex Mode (`split_complex=True`)**:
  Uses real-valued weights and arithmetic for the backbone, outputting two real
  components (Real, Imag) at the final layer.

- **Periodic Boundary Conditions**:
  Efficiently handles torus geometries via circular padding.

Usage
-----
    from general_python.ml.networks import choose_network
    
    # 1. Complex-Valued CNN (Standard)
    # Weights are complex, arithmetic is complex.
    cnn = choose_network('cnn', input_shape=(64,), reshape_dims=(8,8), dtype='complex64')

    # 2. Split-Complex CNN (Optimized)
    # Weights are real, output is complex. Faster.
    cnn_opt = choose_network('cnn', input_shape=(64,), reshape_dims=(8,8), split_complex=True)

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 07.05.2025
Description     : Flax implementation of a Convolutional Neural Network (CNN).
----------------------------------------------------------
"""


import  numpy as np
from    typing import List, Tuple, Callable, Optional, Any, Sequence, Union, TYPE_CHECKING
from    functools import partial
import  math

try:
    import                                      jax
    import                                      jax.numpy as jnp
    import                                      flax.linen as nn
    from ....ml.net_impl.interface_net_flax     import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax     import cplx_variance_scaling
    from ....ml.net_impl.activation_functions   import get_activation_jnp
    if TYPE_CHECKING:
        from ....algebra.utils                  import Array
    JAX_AVAILABLE                               = True
except ImportError as e:
    raise ImportError("Could not import general_python modules. Ensure general_python is properly installed.") from e

##########################################################
#! INNER FLAX CNN MODULE DEFINITION
##########################################################

def circular_pad(x, kernel_sizes):
    """
    Circular padding for periodic boundary conditions.
    Assumes x shape is (Batch, Dim1, Dim2, ..., Channels).
    kernel_sizes corresponds to spatial dimensions only.
    """
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


def _resolve_activation(act: Optional[Any]) -> Optional[Callable]:
    """
    Resolve activation specs produced by ``get_activation_jnp``.
    """
    if act is None:
        return None
    if callable(act):
        return act
    if isinstance(act, (list, tuple)) and len(act) > 0 and callable(act[0]):
        return act[0]
    raise ValueError(f"Invalid activation specification: {act!r}")

##########################################################

class _FlaxCNN(nn.Module):
    """
    Inner Flax module for a Convolutional Neural Network (CNN).
    """
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
    transform_input: bool                       = False
    split_complex  : bool                       = False
    islog          : bool                       = True
    
    in_act         : Optional[Callable]         = None
    out_act        : Optional[Callable]         = None

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        """ 
        Forward pass.
        """
        # Ensure batch dimension: (B, N)
        needs_batch     = s.ndim == 1
        if s.ndim == 1:
            s = s[jnp.newaxis, ...]

        batch_size      = s.shape[0]
        target_shape    = (batch_size,) + tuple(int(d) for d in self.reshape_dims) + (self.input_channels,)

        # Resolve Computation Dtypes
        if self.split_complex:
            # Force Real Arithmetic
            if jnp.issubdtype(self.param_dtype, jnp.complexfloating):
                comp_dtype = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
            else:
                comp_dtype = self.param_dtype
            # Ensure input is real
            s_proc      = s.real if jnp.iscomplexobj(s) else s
        else:
            comp_dtype  = self.dtype
            s_proc      = s

        # 1. Reshape & Transform
        x               = s_proc.reshape(target_shape)
        x               = x.astype(comp_dtype)  # Cast to computation dtype (first)        
            
        # Transform 0/1 -> -1/1 if needed
        if self.transform_input:
            x           = x * 2 - 1

        # Input Activation
        in_act = _resolve_activation(self.in_act)
        if in_act is not None:
            x = in_act(x)

        # ------------
        # Convolutions
        
        if jnp.issubdtype(comp_dtype, jnp.complexfloating):
            k_init = cplx_variance_scaling(1.0, 'fan_in', 'normal', comp_dtype)
        else:
            k_init = nn.initializers.lecun_normal(dtype=comp_dtype)
        
        for i, (feat, k_size, stride, bias, act_fn) in enumerate(zip(
                    self.features, self.kernel_sizes, self.strides, self.use_bias, self.activations
                )):
            # Periodic Padding
            if self.periodic:
                x           = circular_pad(x, k_size)
                pad_type    = 'VALID'
            else:
                pad_type    = 'SAME'

            # Convolution
            x = nn.Conv(
                features        =   feat,
                kernel_size     =   k_size,
                strides         =   stride,
                padding         =   pad_type,
                use_bias        =   bias,
                dtype           =   comp_dtype,
                param_dtype     =   comp_dtype,
                kernel_init     =   k_init,
                name            =   f'conv_{i}'
            )(x)
            
            # Activation
            act = _resolve_activation(act_fn)
            if act is not None:
                x = act(x)

        # Pooling
        if self.use_sum_pool:
            spatial_axes    = tuple(range(1, x.ndim - 1))
            x               = jnp.sum(x, axis=spatial_axes)
        else:
            x               = x.reshape((batch_size, -1))

        if self.split_complex:
            # Output 2 real values: [Real, Imag]
            out             = nn.Dense(
                                features    = 2 * self.output_feats,
                                dtype       = comp_dtype,
                                param_dtype = comp_dtype,
                                kernel_init = k_init,
                                name        = 'head_projection'
                            )(x)
            
            # Reconstruct Complex Number: Re + i*Im
            # Reshape to (batch, output_feats, 2)
            out             = out.reshape((batch_size, self.output_feats, 2))
            out_val         = out[..., 0] + 1j * out[..., 1]
            
        else:
            # Output 1 complex value
            out             = nn.Dense(
                                features    = self.output_feats,
                                dtype       = comp_dtype,
                                param_dtype = comp_dtype,
                                kernel_init = k_init,
                                name        = 'head_projection'
                            )(x)
            out_val         = out # Shape: (batch, output_feats)

        # Squeeze last dim if output_feats is 1
        if self.output_feats == 1:
            out_val         = out_val.reshape((batch_size,))

        # Output
        out_act = _resolve_activation(self.out_act)
        if out_act is not None:
            out_val = out_act(out_val)

        # Final Scale
        if not self.islog:
            out_val = jnp.exp(out_val)
        
        # Return scalar for single sample, array for batch
        return out_val[0] if needs_batch else out_val

##########################################################
#! CNN WRAPPER CLASS USING FlaxInterface
##########################################################

class CNN(FlaxInterface):
    r"""
    Convolutional Neural Network (CNN) Interface.

    Parameters:
        input_shape (tuple): Shape of the 1D input vector.
        reshape_dims (Tuple[int, ...]): Spatial dimensions (Lx, Ly).
        features (Sequence[int]): Channels per layer.
        kernel_sizes (Sequence[int/Tuple]): Kernel sizes.
        split_complex (bool): 
            If True, uses a real-valued backbone with two real outputs (Re, Im) combined at the end.
            Drastically faster for real inputs (spins). Default: False.
        periodic (bool): 
            Periodic boundary conditions. Default: True.
        sum_pooling (bool): 
            Sum over spatial dimensions. Default: True.
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
                in_activation       : Optional[Callable]                                    = None,
                final_activation    : Union[str, Callable, None]                            = None,
                transform_input     : bool                                                  = False,
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
        def _normalize_per_layer(spec, name):
            if isinstance(spec, int):
                return [spec] * n_layers
            if isinstance(spec, tuple) and len(spec) == n_dim and all(isinstance(k, int) for k in spec):
                return [spec] * n_layers
            if isinstance(spec, (list, tuple)):
                items = list(spec)
                if len(items) == 0:
                    raise ValueError(f"{name} cannot be empty")
                if len(items) == 1 and n_layers > 1:
                    items = items * n_layers
                elif len(items) != n_layers:
                    raise ValueError(f"{name} must have length 1 or match len(features)={n_layers}")
                return items
            raise ValueError(f"{name} must be int, tuple[int,...], or sequence")

        def _as_spatial_tuple(item, name):
            if isinstance(item, int):
                return (item,) * n_dim
            if isinstance(item, (list, tuple)) and len(item) == n_dim and all(isinstance(k, int) for k in item):
                return tuple(int(k) for k in item)
            raise ValueError(f"{name} entry {item} must be int or tuple/list of length {n_dim}")

        kernels = tuple(_as_spatial_tuple(item, "kernel_sizes") for item in _normalize_per_layer(kernel_sizes, "kernel_sizes"))
        stride_spec = 1 if strides is None else strides
        strides_t = tuple(_as_spatial_tuple(item, "strides") for item in _normalize_per_layer(stride_spec, "strides"))

        # Activations
        if isinstance(activations, (str, Callable)):
            acts = (get_activation_jnp(activations),) * n_layers
        elif isinstance(activations, (Sequence, List)):
            if len(activations) == 1:
                acts = (get_activation_jnp(activations[0]),) * n_layers
            elif len(activations) == n_layers:
                acts = tuple(get_activation_jnp(a) for a in activations)
            else:
                raise ValueError("activations list length must be 1 or match len(features)")
        else:
            raise ValueError("Invalid activation spec")

        # Bias
        if isinstance(use_bias, bool):
            bias_flags  = (use_bias,) * n_layers
        else:
            bias_vals = list(use_bias)
            if len(bias_vals) == 1 and n_layers > 1:
                bias_vals = bias_vals * n_layers
            elif len(bias_vals) != n_layers:
                raise ValueError("use_bias list length must be 1 or match len(features)")
            bias_flags = tuple(bool(b) for b in bias_vals)

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
            in_act          =   get_activation_jnp(in_activation) if in_activation else None,
            out_act         =   get_activation_jnp(final_activation) if final_activation else None,
            dtype           =   dtype,
            param_dtype     =   p_dtype,
            input_channels  =   1,
            periodic        =   kwargs.get('periodic',      True),
            use_sum_pool    =   kwargs.get('sum_pooling',   True),
            transform_input =   transform_input,
            split_complex   =   split_complex,
            islog           =   islog
        )
        self._out_shape     = output_shape
        self._split_complex = split_complex

        super().__init__(
            net_module  =   _FlaxCNN,
            net_args    =   (),
            net_kwargs  =   net_kwargs,
            input_shape =   input_shape,
            backend     =   "jax",
            dtype       =   dtype,
            seed        =   seed,
        )

        self._has_analytic_grad = False
        self._name              = 'cnn'

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._out_shape

    def __call__(self, s: 'Array'):
        flat_output = super().__call__(s)
        out_size = int(np.prod(self._out_shape))

        # Scalar output (common NQS path)
        if self._out_shape == (1,):
            if flat_output.ndim == 0:
                return flat_output
            return flat_output.reshape(-1)

        # Vector/tensor output path
        if flat_output.ndim == 0:
            return flat_output
        if flat_output.ndim == 1:
            if hasattr(s, "ndim") and s.ndim > 1:
                batch = s.shape[0]
                return flat_output.reshape((batch,) + self._out_shape)
            return flat_output.reshape(self._out_shape)
        if flat_output.shape[-1] == out_size:
            return flat_output.reshape(flat_output.shape[:-1] + self._out_shape)
        return flat_output

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

##########################################################
#! End of CNN File
##########################################################

if __name__ == "__main__":
    cnn = CNN(
        input_shape     =   (64,),
        reshape_dims    =   (8, 8),
        features        =   [16],
        split_complex   =   True,
        dtype           =   'complex64'
    )
    print(cnn)
    x   = np.random.randint(0, 2, (2, 64))
    out = cnn(x)
    print("Output:", out.shape, out.dtype)

###########################################################
#! EOF
###########################################################
