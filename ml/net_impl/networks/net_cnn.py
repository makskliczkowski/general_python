"""
QES.general_python.ml.net_impl.networks.net_cnn
===============================================

Convolutional Neural Network (CNN) for Quantum States.

This module provides a Flax-based implementation of a Convolutional Neural
Network (CNN), designed for representing quantum wavefunctions on a lattice.
It processes 1D input vectors by reshaping them into spatial structures (e.g., a 2D grid)
and applying convolutional layers.

Usage
-----
Import and use the CNN network from the central factory for simplicity.
The factory will correctly instantiate the `CNN` class from this module.

    from QES.general_python.ml.networks import choose_network
    
    # Define parameters for a CNN on a 64-site lattice, reshaped to 8x8
    cnn_params = {
        'input_shape'   : (64,),
        'reshape_dims'  : (8, 8),
        'features'      : (16, 32),
        'kernel_sizes'  : ((3, 3), (3, 3))
    }
    
    # Create a real-valued CNN
    real_cnn_net = choose_network('cnn', **cnn_params)
    
    # Create a complex-valued CNN by specifying the dtype
    complex_cnn_net = choose_network('cnn', dtype='complex64', **cnn_params)

The implementation is wrapped in the `FlaxInterface` for seamless integration
with the QES framework.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 07.05.2025
Description     : Flax implementation of a Convolutional Neural Network (CNN).
----------------------------------------------------------
"""


import  numpy as np
from    typing import List, Tuple, Callable, Optional, Any, Sequence, Union, TYPE_CHECKING
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
    raise ImportError("Could not import QES modules. Ensure QES is properly installed.") from e

##########################################################
#! INNER FLAX CNN MODULE DEFINITION
##########################################################

def circular_pad(x, kernel_sizes):
    """
    Manually pads input x with circular (periodic) boundary conditions.
    Assumes x shape is (Batch, Dim1, Dim2, ..., Channels).
    kernel_sizes corresponds to spatial dimensions only.
    """
    pads = [(0, 0)] # Batch dim
    for k in kernel_sizes:
        # For a kernel of size 3, we need 1 pad on each side.
        # For size k, pad is k//2 and (k-1)//2
        p_left  = k // 2
        p_right = (k - 1) // 2
        pads.append((p_left, p_right))
    pads.append((0, 0)) # Channel dim
    return jnp.pad(x, pads, mode='wrap')

class _FlaxCNN(nn.Module):
    """
    Inner Flax module for a Convolutional Neural Network (CNN).

    Designed to be wrapped by `FlaxInterface`. Processes input reshaped
    into a spatial lattice/feature map and outputs features of a specified shape.

    Attributes:
        reshape_dims (Tuple[int, ...]):
            Dimensions for reshaping the input (e.g., spatial lattice).
        features (Sequence[int]):
            Number of output channels for each *convolutional* layer.
        kernel_sizes (Sequence[Tuple[int,...]]):
            Spatial dimensions of the convolutional kernel for each layer.
        strides (Sequence[Tuple[int,...]]):
            Stride for each spatial dimension for each layer.
        activation_fns (Sequence[Callable]):
            Activation function applied after each convolutional layer.
        use_bias (Sequence[bool]):
            Whether to use bias in each convolutional layer.
        param_dtype (jnp.dtype):
            Data type for CNN parameters (kernels, biases).
        dtype (jnp.dtype):
            Data type for intermediate computations.
        input_channels (int):
            Number of channels after reshaping input (typically 1).
        output_features (int):
            Total number of output features after flattening and final Dense layer.
        final_activation (Optional[Callable]):
            Activation function applied after the final Dense layer. Default is None (linear).
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

    def setup(self):
        iter_specs              = zip(self.features, self.kernel_sizes, self.strides, self.use_bias)

        self.conv_layers : list = [
                        nn.Conv(
                            features    = feat,
                            kernel_size = k_size,
                            strides     = stride,
                            padding     = 'VALID' if self.periodic else 'SAME',
                            use_bias    = bias,
                            param_dtype = self.param_dtype,
                            kernel_init = cplx_variance_scaling(
                                            1.0, 'fan_avg', 'truncated_normal', self.param_dtype
                                        ),
                            dtype       = self.dtype,
                            name        = f"conv_{i}",
                        )
                        for i, (feat, k_size, stride, bias) in enumerate(iter_specs)
                    ]

        self.dense_out  = nn.Dense(
                        features    = self.output_feats,
                        use_bias    = True,
                        param_dtype = self.param_dtype,
                        dtype       = self.dtype,
                        kernel_init = cplx_variance_scaling(
                                        1.0, 'fan_avg', 'truncated_normal', self.param_dtype
                                    ),
                        name        = "dense_out",
                    )

    def __call__(self, s: jax.Array) -> jax.Array:
        """ Forward pass of the CNN. """
        
        # Ensure batch dimension: (B, N)
        if s.ndim == 1:
            s = s[jnp.newaxis, ...]

        batch_size      = s.shape[0]
        target_shape    = (batch_size,) + tuple(int(d) for d in self.reshape_dims) + (self.input_channels,)
        x               = (s.reshape(target_shape) * 2.0).astype(self.dtype)

        # Optional input activation (rarely used)
        if self.in_act is not None:
            x = self.in_act[0](x) if isinstance(self.in_act, (list, tuple)) else self.in_act(x)

        # Convolution stack (jVMC style)
        def _forward_convs(h):
            for i, conv in enumerate(self.conv_layers):
                if self.periodic:
                    h = circular_pad(h, self.kernel_sizes[i])
                h   = conv(h)
                act = self.activations[i][0]
                h   = act(h)
            return h

        if not self.is_initializing():
            _forward_convs = jax.checkpoint(_forward_convs)

        x = _forward_convs(x)   # (B, L1, L2, ..., C_last)

        # Pooling + Dense + normalization
        # We'll compute the normalization based on how many units we sum over.
        if self.use_sum_pool:
            # sum over all spatial dims (keep batch and channels)
            spatial_axes = tuple(range(1, x.ndim - 1))
            n_sum        = jnp.prod(jnp.array([x.shape[a] for a in spatial_axes]))
            x            = jnp.sum(x, axis=spatial_axes)            # (B, C_last)
        else:
            # flatten everything except batch
            n_sum       = jnp.prod(jnp.array(x.shape[1:]))
            x           = x.reshape((batch_size, -1))               # (B, L_flat)

        # Dense to desired output dimension 
        x = self.dense_out(x)                                       # (B, output_feats)

        # Optional final activation
        if self.out_act is not None:
            act = self.out_act[0] if isinstance(self.out_act, (list, tuple)) else self.out_act
            x   = act(x)
        
        # Normalization
        n_spatial   = math.prod(self.reshape_dims)
        n_channels  = x.shape[-1]  # after dense maybe 1, but before dense check conv output
        scale       = jnp.sqrt(n_spatial * n_channels)
        # divide by sqrt(#summed units)
        x           = x / scale
        return x.reshape(-1) # (B, output_feats)

##########################################################
#! CNN WRAPPER CLASS USING FlaxInterface
##########################################################

class CNN(FlaxInterface):
    r"""
    Convolutional Neural Network (CNN) Interface based on FlaxInterface.

    Interprets 1D input vectors as configurations on a lattice, processes them
    through convolutional layers, and outputs features of a specified shape.

    Example:
    --------
        >>> import numpy as np
        >>> from QES.general_python.ml.net_impl.networks.net_cnn import CNN
        >>>
        >>> # --- Real-valued CNN for a 1D system ---
        >>> cnn_1d = CNN(
        ...     input_shape=(16,),      # 16 sites in a 1D chain
        ...     reshape_dims=(16,),     # Reshape to (16,) for 1D convolution
        ...     features=[8, 16],       # Two conv layers with 8 and 16 channels
        ...     kernel_sizes=[3, 3],    # 3x1 kernel for both layers
        ...     output_shape=(1,)       # Scalar output for log-amplitude
        ... )
        >>> print(f"1D CNN (Real): {cnn_1d}")
        >>>
        >>> # --- Complex-valued CNN for a 2D system ---
        >>> cnn_2d_cplx = CNN(
        ...     input_shape=(36,),      # 36 sites on a 6x6 lattice
        ...     reshape_dims=(6, 6),    # Reshape to a 6x6 grid
        ...     features=[16],          # One conv layer with 16 channels
        ...     kernel_sizes=[(3, 3)],  # Single 3x3 kernel
        ...     output_shape=(1,),
        ...     dtype='complex64'       # Use complex numbers for weights
        ... )
        >>> print(f"2D CNN (Complex): {cnn_2d_cplx}")
        >>>
        >>> # Pass a random batch of 2 configurations
        >>> random_configs = np.random.randint(0, 2, size=(2, 36))
        >>> log_amplitudes = cnn_2d_cplx(random_configs)
        >>> print(f"Output shape for batch of 2: {log_amplitudes.shape}")


    Parameters:
        input_shape (tuple):
            Shape of the *1D* input vector (e.g., (n_visible,)).
        reshape_dims (Tuple[int, ...]):
            Spatial/feature dimensions for reshaping the input, e.g., (Lx, Ly).
            Must satisfy `prod(reshape_dims) == input_shape[0]`.
        features (Sequence[int]):
            Number of output channels for each convolutional layer. E.g., (16, 32).
        kernel_sizes (Sequence[Union[int, Tuple[int,...]]]):
            Size of the kernel for each conv layer. E.g., (3, (3,3)).
        strides (Optional[Sequence[Union[int, Tuple[int,...]]]]):
            Stride for each conv layer. Defaults to 1. E.g., (1, (1,1)).
        activations (Union[str, Callable, Sequence[Union[str, Callable]]]):
            Activation function(s) for *convolutional* layers. Defaults to 'relu'.
        use_bias (Union[bool, Sequence[bool]]):
            Whether to use bias in convolutional layers. Default True.
        output_shape (Tuple[int, ...]):
            Desired shape of the network output *excluding the batch dimension*.
            E.g., (1,) for scalar output, (N,) for vector, (N, M) for matrix.
            Default is (1,).
        final_activation (Union[str, Callable, None]):
            Activation function for the *final output layer*. Default None (linear).
        dtype (Any):
            Data type for computations. Default DEFAULT_JP_FLOAT_TYPE.
        param_dtype (Optional[Any]):
            Data type for parameters. Defaults to `dtype`.
        seed (int):
            Seed for parameter initialization. Default 0.
    """
    def __init__(self,
                input_shape         : tuple,
                reshape_dims        : Tuple[int, ...],
                features            : Sequence[int]                                         = (8, 16),
                kernel_sizes        : Sequence[Union[int, Tuple[int,...]]]                  = (3, 3),
                strides             : Optional[Sequence[Union[int, Tuple[int,...]]]]        = None,
                activations         : Union[str, Callable, Sequence[Union[str, Callable]]]  = 'relu',
                use_bias            : Union[bool, Sequence[bool]]                           = True,
                output_shape        : Tuple[int, ...]                                       = (1,),
                in_activation       : Optional[Callable]                                    = None,
                final_activation    : Union[str, Callable, None]                            = None,
                *,
                dtype               : Any                                                   = jnp.float32,
                param_dtype         : Optional[Any]                                         = None,
                seed                : int                                                   = 0,
                **kwargs):

        if not JAX_AVAILABLE:
            raise ImportError("CNN requires JAX.")

        if len(input_shape) != 1:
            raise ValueError("input_shape must be 1-D, e.g. (n_visible,)")

        n_visible   = input_shape[0]
        n_dim       = len(reshape_dims)
        if n_dim not in (1, 2, 3):
            raise ValueError("Only 1-, 2- or 3-D convolutions are supported.")

        if math.prod(reshape_dims) != n_visible:
            raise ValueError(f"reshape_dims {reshape_dims} product != input length {n_visible}")

        #! convert kernel/stride specs
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

        if len(kernel_sizes) != len(features):
            raise ValueError("kernel_sizes and features lengths must match")
        kernels = _as_tuple(kernel_sizes, "kernel_size")

        if strides is None:
            strides = (1,) * len(features)
        if len(strides) != len(features):
            raise ValueError("strides and features lengths must match")
        strides_t = _as_tuple(strides, "stride")

        # ---------------- activation spec ▸ callable tuple -------------------
        if isinstance(activations, (str, Callable)):
            acts = (get_activation_jnp(activations),) * len(features)
        elif isinstance(activations, (Sequence, List)) and len(activations) >= len(features):
            acts = tuple(get_activation_jnp(a) for a in activations[:len(features)])
        else:
            raise ValueError("activations spec must be single item or sequence = features")

        #! bias flags
        if isinstance(use_bias, bool):
            bias_flags = (use_bias,) * len(features)
        elif isinstance(use_bias, Sequence) and len(use_bias) == len(features):
            bias_flags = tuple(bool(b) for b in use_bias)
        else:
            raise ValueError("use_bias must be bool or sequence = features")

        #! dtype handling
        p_dtype = param_dtype if param_dtype is not None else dtype

        #! build kwargs for CNN
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
        )

        self._out_shape = output_shape  # store for __call__

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
        self._compiled_grad_fn  = None
        self._name              = 'cnn'

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Returns the configured output shape (excluding batch dimension)."""
        return self._output_shape

    #! Callable interface
    def __call__(self, s: 'Array'):
        """
        Call the network on input x.
        Output shape will be (batch_size, *output_shape).
        """
        
        # FlaxInterface.__call__ calls the compiled apply function
        # which executes _FlaxCNN.__call__
        flat_output = super().__call__(s) # Shape (batch, output_features)

        if self._out_shape == (1,):
            return flat_output.reshape(-1) # (batch,)
        target_output_shape = (-1,) + self._out_shape
        return flat_output.reshape(target_output_shape)

    # ####################################################

    def __repr__(self) -> str:
        """
        Provides a concise string representation of the CNN.
        Displays the CNN type (Complex/Real), reshape dimensions, features,
        kernel sizes, output shape, data types, number of parameters,
        and initialization status.
        """
        kind = "Complex" if self._iscpx else "Real"
        return (
            f"{kind}CNN(reshape={self._flax_module.reshape_dims}, "
            f"features={self._flax_module.features}, "
            f"kernels={self._flax_module.kernel_sizes}, "
            f"output_shape={self._out_shape}, dtype={self.dtype}, "
            f"params={self.nparams}, "
            f'{"initialized" if self.initialized else "uninitialized"})'
        )

    def __str__(self) -> str:
        """
        Provides a brief string summary of the CNN.
        Displays the CNN type (Complex/Real), input shape, reshape dimensions,
        number of features, and data type.
        """
        kind        = "Complex" if self._iscpx else "Real"
        n_features  = sum(self._flax_module.features)
        return (
            f"{kind}CNN(input_shape={self.input_shape},"
            f"reshape={self._flax_module.reshape_dims},"
            f"total_features={n_features},dtype={self.dtype})"
        )

##########################################################
#! End of CNN File
##########################################################

# Example usage:
if __name__ == "__main__":
    
    print("--- Defining a Real-valued CNN for a 2D lattice (8x8) ---")
    cnn_real = CNN(
        input_shape     =   (64,),
        reshape_dims    =   (8, 8),
        features        =   [16, 32],
        kernel_sizes    =   [(3, 3), (3, 3)],
        activations     =   ['relu', 'relu'],
        output_shape    =   (1,)
    )
    print(cnn_real)
    
    # Create a random batch of 2 configurations {0, 1}
    x_real      = np.random.randint(0, 2, size=(2, 64))
    output_real = cnn_real(x_real)
    print(f"Output shape for real CNN: {output_real.shape}")
    print(f"Example output: {output_real[0]}")
    print("-" * 20)

    print("\n--- Defining a Complex-valued CNN for a 1D chain (16 sites) ---")
    cnn_cplx    = CNN(
        input_shape     =   (16,),
        reshape_dims    =   (16,),
        features        =   [8],
        kernel_sizes    =   [3], # Kernel is automatically expanded to (3,)
        output_shape    =   (2,), # Example of a 2-component output
        dtype           =   'complex64'
    )
    print(cnn_cplx)

    x_cplx      = np.random.randint(0, 2, size=(2, 16))
    output_cplx = cnn_cplx(x_cplx)
    print(f"Output shape for complex CNN: {output_cplx.shape}")
    print(f"Example output: {output_cplx[0]}")
    print("-" * 20)
    
    print(f"\nTotal parameters in real CNN: {cnn_real.nparams}")
    print(f"Total parameters in complex CNN: {cnn_cplx.nparams}")

# --------------------------------------------------------------
#! End of File
# --------------------------------------------------------------