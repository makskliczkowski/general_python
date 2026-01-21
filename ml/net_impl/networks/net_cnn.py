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
    in_act         : Optional[Callable] = None
    out_act        : Optional[Callable] = None
    param_dtype    : jnp.dtype          = jnp.complex64
    dtype          : jnp.dtype          = jnp.complex64
    input_channels : int                = 1
    periodic       : bool               = True
    use_sum_pool   : bool               = True
    transform_input: bool               = False
    split_complex  : bool               = False # New optimization flag

    def setup(self):
        """
        Setup layers. Handles split-complex logic by converting complex dtypes
        to their float equivalents for the backbone if enabled.
        """
        iter_specs = zip(self.features, self.kernel_sizes, self.strides, self.use_bias)
        
        #! Dtype Resolution
        # If split_complex is True, we force the backbone to be REAL.
        # We assume input is real (or projected to real).
        if self.split_complex:
            # Map complex64 -> float32, complex128 -> float64
            if jnp.issubdtype(self.param_dtype, jnp.complexfloating):
                p_dtype = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
            else:
                p_dtype = self.param_dtype
                
            if jnp.issubdtype(self.dtype, jnp.complexfloating):
                c_dtype = jnp.float32 if self.dtype == jnp.complex64 else jnp.float64
            else:
                c_dtype = self.dtype
        else:
            p_dtype = self.param_dtype
            c_dtype = self.dtype

        # Initialization Scaling
        n_spatial   = math.prod(self.reshape_dims)
        init_scale  = 1.0 / jnp.sqrt(n_spatial)
        
        # Choose initializer
        if jnp.issubdtype(p_dtype, jnp.complexfloating):
            kernel_init = cplx_variance_scaling(init_scale, 'fan_in', 'normal', p_dtype)
        else:
            kernel_init = nn.initializers.variance_scaling(init_scale, 'fan_in', 'normal', dtype=p_dtype)

        # --- Convolution Layers ---
        self.conv_layers = [
            nn.Conv(
                features    = feat,
                kernel_size = k_size,
                strides     = stride,
                padding     = 'VALID' if self.periodic else 'SAME',
                use_bias    = bias,
                param_dtype = p_dtype,
                kernel_init = kernel_init,
                dtype       = c_dtype,
                name        = f"conv_{i}",
            )
            for i, (feat, k_size, stride, bias) in enumerate(iter_specs)
        ]

        #! Output Layer
        # If split_complex, we output 2x features (Real, Imag) and combine later.
        out_features    = self.output_feats * 2 if self.split_complex else self.output_feats
        
        self.dense_out  = nn.Dense(
                            features    = out_features,
                            use_bias    = True,
                            param_dtype = p_dtype,
                            dtype       = c_dtype,
                            kernel_init = kernel_init,
                            name        = "dense_out",
                        )

    def __call__(self, s: jax.Array) -> jax.Array:
        """ 
        Forward pass.
        """
        # Ensure batch dimension: (B, N)
        if s.ndim == 1:
            s = s[jnp.newaxis, ...]

        batch_size      = s.shape[0]
        target_shape    = (batch_size,) + tuple(int(d) for d in self.reshape_dims) + (self.input_channels,)
        
        # 1. Reshape & Transform
        x = s.reshape(target_shape)
        
        if self.split_complex:
            # If backbone is real, ensure input is real.
            # For spins (0,1) or (-1,1), this is trivial.
            # Usually NQS inputs are real.
            x = x.real if jnp.iscomplexobj(x) else x
            
        # Transform 0/1 -> -1/1 if needed
        if self.transform_input:
            x = x * 2 - 1
            
        # Cast to computation dtype (float if split_complex)
        comp_dtype  = self.conv_layers[0].dtype     # Get dtype from first layer
        x           = x.astype(comp_dtype)          # Cast input

        # Input Activation
        if self.in_act is not None:
            act_fn  = self.in_act[0] if isinstance(self.in_act, (list, tuple)) else self.in_act
            x       = act_fn(x)

        # 2. Convolutions
        for i, conv in enumerate(self.conv_layers):
            if self.periodic:
                x = circular_pad(x, self.kernel_sizes[i])
            
            # Convolution
            x   = conv(x)
            
            # Activation
            act = self.activations[i]
            x   = act[0](x) if isinstance(act, (list, tuple)) else act(x)

        # 3. Pooling
        if self.use_sum_pool:
            # Sum over spatial dimensions (1 ... N-2)
            spatial_axes    = tuple(range(1, x.ndim - 1))
            x               = jnp.sum(x, axis=spatial_axes)
        else:
            x               = x.reshape((batch_size, -1))

        # 4. Output
        x = self.dense_out(x)

        if self.out_act is not None:
            act_fn          = self.out_act[0] if isinstance(self.out_act, (list, tuple)) else self.out_act
            x               = act_fn(x)
            
        # 5. Complex Recombination 
        if self.split_complex:
            # x shape: (Batch, 2 * output_feats)
            x_real, x_imag  = jnp.split(x, 2, axis=-1)
            x               = x_real + 1j * x_imag
        
        return x.reshape(-1) if batch_size == 1 else x.reshape(batch_size, -1)

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
                reshape_dims        : Tuple[int, ...],
                features            : Sequence[int]                                         = (8,),
                kernel_sizes        : Sequence[Union[int, Tuple[int,...]]]                  = (2,),
                strides             : Optional[Sequence[Union[int, Tuple[int,...]]]]        = None,
                activations         : Union[str, Callable, Sequence[Union[str, Callable]]]  = 'log_cosh',
                use_bias            : Union[bool, Sequence[bool]]                           = True,
                output_shape        : Tuple[int, ...]                                       = (1,),
                in_activation       : Optional[Callable]                                    = None,
                final_activation    : Union[str, Callable, None]                            = None,
                transform_input     : bool                                                  = True,
                *,
                split_complex       : bool                                                  = False,
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
        
        if math.prod(reshape_dims) != n_visible:
            raise ValueError(f"reshape_dims {reshape_dims} product != input length {n_visible}")

        # Args Normalization
        def _as_tuple(v, name):
            out = []
            for item in v:
                if isinstance(item, int):
                    out.append((item,) * n_dim)
                elif isinstance(item, tuple) and len(item) == n_dim:
                    out.append(item)
                else:
                    raise ValueError(f"{name} {item} must be int or tuple of length {n_dim}")
            return tuple(out)

        kernels     = _as_tuple(kernel_sizes, "kernel_size")
        
        if strides is None: strides = (1,) * len(features)
        strides_t   = _as_tuple(strides, "stride")

        # Activations
        if isinstance(activations, (str, Callable)):
            acts = (get_activation_jnp(activations),) * len(features)
        elif isinstance(activations, (Sequence, List)):
            acts = tuple(get_activation_jnp(a) for a in activations[:len(features)])
        else:
            raise ValueError("Invalid activation spec")

        # Bias
        if isinstance(use_bias, bool):
            bias_flags = (use_bias,) * len(features)
        else:
            bias_flags = tuple(bool(b) for b in use_bias)

        p_dtype = param_dtype if param_dtype is not None else dtype

        # Build Module Config
        net_kwargs = dict(
            reshape_dims    =   reshape_dims,
            features        =   tuple(features),
            kernel_sizes    =   kernels,
            strides         =   strides_t,
            activations     =   acts,
            use_bias        =   bias_flags,
            output_feats    =   math.prod(output_shape),
            in_act          =   get_activation_jnp(in_activation) if in_activation else None,
            out_act         =   get_activation_jnp(final_activation) if final_activation else None,
            dtype           =   dtype,
            param_dtype     =   p_dtype,
            input_channels  =   1,
            periodic        =   kwargs.get('periodic', True),
            use_sum_pool    =   kwargs.get('sum_pooling', True),
            transform_input =   transform_input,
            split_complex   =   split_complex
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
        if self._out_shape == (1,):
            return flat_output.reshape(-1)
        target_output_shape = (-1,) + self._out_shape
        return flat_output.reshape(target_output_shape)

    def __repr__(self) -> str:
        kind    = "SplitComplex" if self._split_complex else ("Complex" if self._iscpx else "Real")
        return  (
                    f"{kind}CNN(reshape={self._flax_module.reshape_dims}, "
                    f"features={self._flax_module.features}, "
                    f"kernels={self._flax_module.kernel_sizes}, "
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

