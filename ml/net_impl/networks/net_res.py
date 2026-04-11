"""
general_python.ml.net_impl.networks.net_resnet
==================================================

Deep residual convolutional wrapper based on Flax.

The module provides a generic residual backbone for flattened inputs that can
be reshaped onto regular grids. Optional input preprocessing is exposed through
a generic callable hook.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 04.12.2025
Description     : Flax implementation of a residual convolutional network.
----------------------------------------------------------
"""

import numpy as np
import math
from typing import Tuple, Callable, Optional, Any, Union

try:
    from ....ml.net_impl.interface_net_flax         import FlaxInterface
    from ....ml.net_impl.activation_functions       import log_cosh_jnp
    from ....ml.net_impl.utils.net_init_jax         import cplx_variance_scaling
    from ....ml.net_impl.utils.net_wrapper_utils    import resolve_input_adapter
    from ....algebra.utils                          import JAX_AVAILABLE, DEFAULT_JP_CPX_TYPE
except ImportError as e:
    raise ImportError("Required modules for ResNet not found. Ensure general_python.ml and general_python.algebra are accessible.") from e

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
else:
    raise ImportError("JAX is not available. Please install JAX to use this module.")

##########################################################
#! UTILITIES
##########################################################

def circular_pad(x, kernel_size):
    """
    Manually pads input x with circular (periodic) boundary conditions.
    Assumes x shape is (Batch, Dim1, Dim2, ..., Channels).
    kernel_size corresponds to spatial dimensions.
    """
    pads = [(0, 0)] # Batch dim
    
    # If kernel is int, broadcast it; if tuple, iterate
    if isinstance(kernel_size, int):
        k_dims = [kernel_size] * (x.ndim - 2)
    else:
        k_dims = kernel_size

    for k in k_dims:
        # For a kernel of size k, pad (k//2) on left, (k-1)//2 on right
        p_left  = k // 2
        p_right = (k - 1) // 2
        pads.append((p_left, p_right))
        
    pads.append((0, 0)) # Channel dim
    return jnp.pad(x, pads, mode='wrap')

##########################################################
#! INNER FLAX MODULES
##########################################################

class ResNetBlock(nn.Module):
    """Two-layer residual convolution block."""
    features    : int
    kernel_size : Tuple[int, ...]
    dtype       : Any
    param_dtype : Any
    act_fn      : Callable = log_cosh_jnp
    periodic    : bool = True
    init_fn     : Callable = nn.initializers.lecun_normal()
    use_norm    : bool = True

    def setup(self):
        padding = 'VALID' if self.periodic else 'SAME'
        self.conv1 = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            padding=padding,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.init_fn,
            name="conv1",
        )
        self.conv2 = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            padding=padding,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.init_fn,
            name="conv2",
        )
        if self.use_norm:
            self.norm1 = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm1")
            self.norm2 = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm2")
    
    @nn.compact
    def __call__(self, x):
        residual    = x
        
        # Layer 1
        h = x
        if self.periodic:
            h = circular_pad(h, self.kernel_size)
            padding = 'VALID'
        else:
            padding = 'SAME'
            
        h = self.conv1(h)
        if self.use_norm:
            h = self.norm1(h)
        h = self.act_fn(h)
        
        # Layer 2
        if self.periodic:
            h = circular_pad(h, self.kernel_size)
            padding = 'VALID'
        else:
            padding = 'SAME'
            
        h = self.conv2(h)
        if self.use_norm:
            h = self.norm2(h)
        h = self.act_fn(h)

        # Scale down initialization to start close to Identity
        h = h * 0.1
        
        return residual + h

class _FlaxResNet(nn.Module):
    """
    Residual convolutional network for flattened inputs.
    
    Architecture:
    1. Reshape Input -> Grid
    2. Initial Convolution Projection
    3. Stack of Residual Blocks
    4. Sum Pooling (Spatial)
    5. Final Dense Layer -> Log Amplitude
    """
    reshape_dims        : Tuple[int, ...]
    features            : int
    depth               : int
    kernel_size         : Tuple[int, ...]
    param_dtype         : jnp.dtype
    dtype               : jnp.dtype
    input_channels      : int = 1
    periodic_boundary   : bool = True
    use_pooling         : bool = True
    input_scale         : float = 1.0
    input_shift         : float = 0.0
    input_adapter       : Optional[Callable] = None
    init_scale          : float = 1.0
    init_mode           : str = 'fan_in'
    init_dist           : str = 'normal'
    
    def setup(self):
        is_cplx = jnp.issubdtype(self.param_dtype, jnp.complexfloating)
        if is_cplx:
            self._kernel_init = cplx_variance_scaling(self.init_scale, self.init_mode, self.init_dist, self.param_dtype)
        else:
            self._kernel_init = nn.initializers.variance_scaling(
                self.init_scale, self.init_mode, self.init_dist, dtype=self.param_dtype
            )
        padding = 'VALID' if self.periodic_boundary else 'SAME'
        self.conv_init = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            padding=padding,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self._kernel_init,
            name="conv_init",
        )
        self.res_blocks = [
            ResNetBlock(
                features=self.features,
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                periodic=self.periodic_boundary,
                init_fn=self._kernel_init,
                name=f"res_block_{i}",
            )
            for i in range(self.depth)
        ]
        self.dense_out = nn.Dense(
            features=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self._kernel_init,
            name="dense_out",
        )

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        
        # --- 1. Input Reshaping ---
        needs_batch = s.ndim == 1
        if needs_batch:
            s = s[jnp.newaxis, ...]

        if self.input_adapter is not None:
            s = self.input_adapter(s)
            
        batch_size   = s.shape[0]
        # (Batch, L1, L2, ..., 1)
        target_shape = (batch_size,) + tuple(int(d) for d in self.reshape_dims) + (self.input_channels,)
        
        # Apply input scaling/shifting
        s = s * self.input_scale + self.input_shift
            
        x = s.reshape(target_shape).astype(self.dtype)

        # Initial Projection
        if self.periodic_boundary:
            x = circular_pad(x, self.kernel_size)
            padding = 'VALID'
        else:
            padding = 'SAME'
            
        x = self.conv_init(x)
        x = log_cosh_jnp(x)

        # Residual Blocks
        for block in self.res_blocks:
            x = block(x)

        # Pooling or Flatten
        if self.use_pooling:
            # Sum Pooling: Sum over all spatial dimensions (axes 1 to N-1)
            spatial_axes = tuple(range(1, x.ndim - 1))
            x = jnp.sum(x, axis=spatial_axes)
        else:
            x = x.reshape((x.shape[0], -1))

        # Final Dense Layer
        x = self.dense_out(x)

        out = x.reshape(-1)
        return out[0] if needs_batch else out

##########################################################
#! RESNET WRAPPER CLASS
##########################################################

class ResNet(FlaxInterface):
    """
    Deep Residual Network (ResNet) Interface.

    Generic residual convolutional wrapper for regularly shaped inputs.

    Parameters:
        input_shape (tuple):
            Shape of the 1D input vector (e.g., (N_sites,)).
        reshape_dims (Tuple[int, ...]):
            Lattice dimensions for reshaping, e.g., (Lx, Ly).
        features (int):
            Number of feature channels (width of the network). Default 32.
        depth (int):
            Number of residual blocks. Default 4.
        kernel_size (Union[int, Tuple[int,...]]):
            Spatial kernel size. Default (3, 3).
        periodic_boundary (bool):
            Whether to use periodic boundary conditions. Default True.
        use_pooling (bool):
            Whether to use global sum pooling (TI). Default True.
        input_scale (float):
            Scaling factor for input. Default 1.0.
        input_shift (float):
            Shift factor for input. Default 0.0.
        input_adapter (Optional[Callable]):
            Optional preprocessing applied before reshaping and scaling.
        init_scale (float):
            Scale for variance scaling initialization. Default 1.0.
        init_mode (str):
            Mode for initialization ('fan_in', 'fan_avg', 'fan_out'). Default 'fan_in'.
        init_dist (str):
            Distribution for initialization ('normal', 'uniform'). Default 'normal'.
        dtype (Any):
            Computation data type.
        param_dtype (Optional[Any]):
            Parameter data type. Defaults to dtype.
        seed (int):
            Initialization seed.
    """
    def __init__(self,
                input_shape     : tuple,
                reshape_dims    : Tuple[int, ...],
                features        : int                                   = 32,
                depth           : int                                   = 4,
                kernel_size     : Union[int, Tuple[int,...]]            = 3,
                *,
                periodic_boundary: bool                                 = True,
                use_pooling     : bool                                  = True,
                input_scale     : float                                 = 1.0,
                input_shift     : float                                 = 0.0,
                input_adapter   : Optional[Callable]                    = None,
                init_scale      : float                                 = 1.0,
                init_mode       : str                                   = 'fan_in',
                init_dist       : str                                   = 'normal',
                dtype           : Any                                   = DEFAULT_JP_CPX_TYPE,
                param_dtype     : Optional[Any]                         = None,
                seed            : int                                   = 0,
                backend         : str                                   = "jax",
                **kwargs):

        if not JAX_AVAILABLE:
            raise ImportError("ResNet requires JAX.")

        n_visible = input_shape[0]
        n_dim     = len(reshape_dims)
        
        # Validation
        if math.prod(reshape_dims) != n_visible:
            raise ValueError(f"reshape_dims {reshape_dims} product != input length {n_visible}")

        # Normalize kernel size
        if isinstance(kernel_size, int):
            kernel_tuple = (kernel_size,) * n_dim
        else:
            kernel_tuple = kernel_size
        if len(kernel_tuple) != n_dim:
            raise ValueError(f"kernel_size {kernel_tuple} must have length {n_dim}")

        p_dtype = param_dtype if param_dtype is not None else dtype
        input_convention, input_adapter = resolve_input_adapter(
            kwargs,
            input_adapter,
            map_input_to_spin=kwargs.get("map_input_to_spin", None),
        )

        # Build kwargs for the inner Flax module
        net_kwargs = dict(
            reshape_dims    = reshape_dims,
            features        = features,
            depth           = depth,
            kernel_size     = kernel_tuple,
            dtype           = dtype,
            param_dtype     = p_dtype,
            input_channels  = 1,
            periodic_boundary=periodic_boundary,
            use_pooling     = use_pooling,
            input_scale     = input_scale,
            input_shift     = input_shift,
            input_adapter   = input_adapter,
            init_scale      = init_scale,
            init_mode       = init_mode,
            init_dist       = init_dist,
        )

        super().__init__(
            net_module  = _FlaxResNet,
            net_args    = (),
            net_kwargs  = net_kwargs,
            input_shape = input_shape,
            backend     = backend,
            dtype       = dtype,
            seed        = seed,
            **kwargs
        )

        self._name              = 'resnet'
        self._has_analytic_grad = False # Use AD
        self._input_convention  = dict(input_convention)

    # ----------------------------------------------------------

    def __repr__(self) -> str:
        kind    = "Complex" if self._iscpx else "Real"
        mod     = self._flax_module
        return (
            f"{kind}ResNet(reshape={mod.reshape_dims}, "
            f"features={mod.features}, depth={mod.depth}, "
            f"kernel={mod.kernel_size}, dtype={self.dtype}, "
            f"params={self.nparams})"
        )
        
    def __str__(self) -> str:
        kind    = "Complex" if self._iscpx else "Real"
        mod     = self._flax_module
        return (f"{kind}ResNet(reshape={mod.reshape_dims},features={mod.features},depth={mod.depth},dtype={self.dtype})")

# ----------------------------------------------------------
#! END OF FILE
# ----------------------------------------------------------
