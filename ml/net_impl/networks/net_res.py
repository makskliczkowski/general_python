"""
QES.general_python.ml.net_impl.networks.net_resnet
==================================================

Deep Complex-Valued Residual Network (ResNet) for Quantum States.

This architecture is State-of-the-Art (SOTA) for topological phases like the
Kitaev Spin Liquid. It uses residual connections to allow deep signal propagation
and complex-valued weights to capture non-trivial sign structures.

Usage
-----
    from QES.general_python.ml.networks import choose_network
    
    # Define parameters for a ResNet on a 64-site lattice
    resnet_params = {
        'input_shape'   : (64,),
        'reshape_dims'  : (8, 8),   # Lattice dimensions
        'features'      : 32,       # Hidden channels width
        'depth'         : 4,        # Number of Residual Blocks
        'kernel_size'   : (3, 3),   # Kernel size
    }
    
    # Create a complex-valued ResNet
    net = choose_network('resnet', dtype='complex128', **resnet_params)

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 04.12.2025
Description     : Flax implementation of Complex ResNet.
----------------------------------------------------------
"""

import numpy as np
import math
from typing import Tuple, Callable, Optional, Any, Sequence, Union

try:
    from ....ml.net_impl.interface_net_flax import FlaxInterface
    from ....ml.net_impl.activation_functions import log_cosh_jnp
    from ....algebra.utils import JAX_AVAILABLE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE, Array
except ImportError as e:
    raise ImportError("Required modules for ResNet not found. Ensure QES.general_python.ml and QES.general_python.algebra are accessible.") from e

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
    """
    A single Residual Block: x -> x + Conv(Activation(Conv(x)))
    """
    features    : int
    kernel_size : Tuple[int, ...]
    dtype       : Any
    param_dtype : Any
    act_fn      : Callable = log_cosh_jnp
    
    @nn.compact
    def __call__(self, x):
        residual    = x
        
        # Layer 1
        # 1. Manual Circular Pad
        h           = circular_pad(x, self.kernel_size)
        # 2. Convolution (VALID padding because we manually padded)
        h           = nn.Conv(features=self.features, kernel_size=self.kernel_size, 
                        padding='VALID', dtype=self.dtype, param_dtype=self.param_dtype)(h)
        # 3. Activation
        h           = self.act_fn(h)
        
        # Layer 2
        # 1. Manual Circular Pad
        h           = circular_pad(h, self.kernel_size)
        # 2. Convolution
        h           = nn.Conv(features=self.features, kernel_size=self.kernel_size, 
                        padding='VALID', dtype=self.dtype, param_dtype=self.param_dtype)(h)
        
        # 3. Activation
        h           = self.act_fn(h)

        # Scale down initialization to start close to Identity
        h           = h * 0.1
        
        return residual + h

class _FlaxResNet(nn.Module):
    """
    Deep Residual Network for Quantum States.
    
    Architecture:
    1. Reshape Input -> Grid
    2. Initial Convolution Projection
    3. Stack of Residual Blocks
    4. Sum Pooling (Spatial)
    5. Final Dense Layer -> Log Amplitude
    """
    reshape_dims    : Tuple[int, ...]
    features        : int
    depth           : int
    kernel_size     : Tuple[int, ...]
    param_dtype     : jnp.dtype
    dtype           : jnp.dtype
    input_channels  : int = 1
    
    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        
        # --- 1. Input Reshaping ---
        if s.ndim == 1:
            s = s[jnp.newaxis, ...]
            
        batch_size   = s.shape[0]
        # (Batch, L1, L2, ..., 1)
        target_shape = (batch_size,) + tuple(int(d) for d in self.reshape_dims) + (self.input_channels,)
        x            = s.reshape(target_shape).astype(self.dtype)

        # Initial Projection
        # Map input spins (usually 1 channel) to feature space
        x           = circular_pad(x, self.kernel_size)
        x           = nn.Conv(features=self.features, kernel_size=self.kernel_size, 
                        padding='VALID', dtype=self.dtype, param_dtype=self.param_dtype, name="conv_init")(x)
        x           = log_cosh_jnp(x)

        # Residual Blocks
        for i in range(self.depth):
            x       = ResNetBlock(features=self.features, 
                            kernel_size=self.kernel_size,
                            dtype=self.dtype, 
                            param_dtype=self.param_dtype,
                            name=f"res_block_{i}")(x)

        # Sum Pooling
        # Sum over all spatial dimensions (axes 1 to N-1)
        # Result shape: (Batch, Features)
        spatial_axes = tuple(range(1, x.ndim - 1))
        x = jnp.sum(x, axis=spatial_axes)

        # Final Dense Layer
        # Map features to single scalar log-amplitude
        # (Batch, Features) -> (Batch, 1)
        x = nn.Dense(features=1, dtype=self.dtype, param_dtype=self.param_dtype, name="dense_out")(x)
        
        return x.reshape(-1)

##########################################################
#! RESNET WRAPPER CLASS
##########################################################

class ResNet(FlaxInterface):
    """
    Deep Residual Network (ResNet) Interface.

    State-of-the-Art architecture for 2D topological systems. Uses periodic
    convolutions and residual connections to learn deep representations.

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
        dtype (Any):
            Computation data type. Default complex128 for Physics.
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
            # Try to handle Honeycomb (2 sites per unit cell) or similar logic if needed
            # For now, strict check
            pass # Or warning

        # Normalize kernel size
        if isinstance(kernel_size, int):
            kernel_tuple = (kernel_size,) * n_dim
        else:
            kernel_tuple = kernel_size

        p_dtype = param_dtype if param_dtype is not None else dtype

        # Build kwargs for the inner Flax module
        net_kwargs = dict(
            reshape_dims    = reshape_dims,
            features        = features,
            depth           = depth,
            kernel_size     = kernel_tuple,
            dtype           = dtype,
            param_dtype     = p_dtype,
            input_channels  = 1
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

# Example usage check
if __name__ == "__main__":
    print("--- Testing Complex ResNet ---")
    # 2D Lattice 4x4
    net = ResNet(
        input_shape=(16,),
        reshape_dims=(4, 4),
        features=8,
        depth=2,
        dtype='complex64'
    )
    print(net)
    
    x = np.random.randint(0, 2, (2, 16))
    out = net(x)
    print("Output shape:", out.shape)
    print("Output type:", out.dtype)
    
# ----------------------------------------------------------
#! END OF FILE
# ----------------------------------------------------------