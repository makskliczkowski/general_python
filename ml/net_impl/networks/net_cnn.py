'''
file    :   general_python/ml/net_impl/networks/net_cnn.py
author  :   Maksymilian Kliczkowski
date    :   2025-05-07
brief   :   Convolutional Neural Network (CNN) implementation using Flax.
            This module defines a CNN class that can be used for various tasks,
            such as image classification, feature extraction, etc.
            The CNN is designed to work with JAX and Flax, leveraging their
            capabilities for efficient computation and automatic differentiation. 
        
'''


import numpy as np
from typing import Tuple, Callable, Optional, Any, Sequence, Union
import math

try:
    from ....ml.net_impl.interface_net_flax import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax import cplx_variance_scaling, lecun_normal
    from ....ml.net_impl.activation_functions import get_activation_jnp
    from ....algebra.utils import JAX_AVAILABLE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE, Array
except ImportError as e:
    print(f"Error importing QES base modules: {e}")
    class FlaxInterface:
        pass
    JAX_AVAILABLE = False

#! JAX / Flax Imports

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
else:
    # This module requires JAX
    raise ImportError("JAX is not available. Please install JAX to use this module.")

##########################################################
#! INNER FLAX CNN MODULE DEFINITION
##########################################################

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
    param_dtype    : jnp.dtype          = DEFAULT_JP_CPX_TYPE
    dtype          : jnp.dtype          = DEFAULT_JP_CPX_TYPE
    input_channels : int                = 1
    
    def setup(self):
        is_cplx     = jnp.issubdtype(self.param_dtype, jnp.complexfloating)
        k_init      = (lambda: cplx_variance_scaling(1., 'fan_in', 'normal', self.param_dtype) if is_cplx else lambda: nn.initializers.lecun_normal(self.param_dtype))

        self.convs = [
            nn.Conv(
                features    = f,
                kernel_size = ks,
                strides     = st,
                padding     = 'SAME',
                use_bias    = ub,
                param_dtype = self.param_dtype,
                dtype       = self.dtype,
                kernel_init = k_init(),
                bias_init   = nn.initializers.zeros,
                name        = f"conv_{i}"
            )
            for i, (f, ks, st, ub) in enumerate(zip(self.features, self.kernel_sizes, self.strides, self.use_bias))
        ]

        #! Last layer is a Dense layer
        self.dense = nn.Dense(
            self.output_feats,
            use_bias    = True,
            param_dtype = self.param_dtype,
            dtype       = self.dtype,
            kernel_init = k_init(),
            bias_init   = nn.initializers.zeros,
            name        = "dense_out"
        )

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        r"""
        Calculates a Convolutional neural network (CNN) output.
        The input `s` is reshaped to match the specified reshape dimension and
        passed through a series of convolutional layers.

        Steps:
            0.  Apply an optional activation function to the input.
            1.  Reshape input `s` from (batch, n_visible) to (batch, Lx, Ly, ..., input_channels).
            2.  Apply a sequence of convolutional layers with activations.
            3.  Flatten the output of the last convolutional layer.
            4.  Apply a final Dense layer to produce the desired number of output features.
            5.  Apply an optional final activation function.

        Math (Conceptual for one layer):
            Let :math:`x^{(0)}` be the reshaped input tensor.
            For layer :math:`l`:
            
            .. math::
                z^{(l)} = \text{Conv}^{(l)}(x^{(l-1)}) + b^{(l)}
            
            .. math::
                x^{(l)} = \sigma^{(l)}(z^{(l)})

            where :math:`\text{Conv}^{(l)}` is the convolution operation with kernel :math:`W^{(l)}`,
            :math:`b^{(l)}` is the optional bias, and :math:`\sigma^{(l)}` is the activation function.

            The final output is obtained by summing the elements of the last layer's output :math:`x^{(L)}`:

            .. math::
                \log(\psi(s)) = \sum_{spatial\_dims, channels} x^{(L)}_{spatial\_dims, channels}


        Args:
            s (jax.Array):
                Input configuration(s) with shape (batch, n_visible).

        Returns:
            jax.Array: Log-amplitude(s) log(psi(s)) with shape (batch,).
        """
        
        if s.ndim == 1:                     # shape (n_visible,)
            s = s[jnp.newaxis, ...]         # → (1, n_visible)
        
        # reshape to (B, *spatial, C)
        s = s.reshape((s.shape[0],) + self.reshape_dims + (self.input_channels,))
        if self.in_act is not None:
            activation = self.in_act[0]
            s = activation(s)

        x = s.astype(self.dtype)
        for conv, act in zip(self.convs, self.activations):
            activation = act[0]
            x = activation(conv(x))

        x = x.reshape((x.shape[0], -1))      # flatten
        x = self.dense(x)

        if self.out_act is not None:
            activation = self.out_act[0]
            x = activation(x)

        return x.reshape(-1)

##########################################################
#! CNN WRAPPER CLASS USING FlaxInterface
##########################################################

class CNN(FlaxInterface):
    r"""
    Convolutional Neural Network (CNN) Interface based on FlaxInterface.

    Interprets 1D input vectors as configurations on a lattice, processes them
    through convolutional layers, and outputs features of a specified shape.

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
                dtype               : Any                                                   = DEFAULT_JP_FLOAT_TYPE,
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
        elif isinstance(activations, Sequence) and len(activations) == len(features):
            acts = tuple(get_activation_jnp(a) for a in activations)
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

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Returns the configured output shape (excluding batch dimension)."""
        return self._output_shape

    #! Callable interface
    def __call__(self, s: Array):
        """
        Call the network on input x.
        Output shape will be (batch_size, *output_shape).
        """
        
        # FlaxInterface.__call__ calls the compiled apply function
        # which executes _FlaxCNN.__call__
        flat_output = super().__call__(s) # Shape (batch, output_features)

        # Reshape the flat output to the desired multi-dimensional output shape
        # The batch dimension is handled automatically by Flax/JAX
        # prepend -1 to the output shape to allow for any batch size
        target_output_shape = (-1,) + self._output_shape
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
        
##########################################################
#! End of CNN File
##########################################################

# Example usage:
if __name__ == "__main__":
    # Example configuration
    input_shape     = (64,)             # 1D input vector of length 64
    reshape_dims    = (8, 8)            # Reshape to 8x8 spatial dimensions
    features        = [16, 32]          # Number of output channels for each conv layer
    kernel_sizes    = [(3, 3), (3, 3)]  # Kernel sizes for each conv layer
    activations     = ['relu', 'relu']  # Activation functions for each conv layer
    output_shape    = (1,)              # Desired output shape (e.g., scalar output)
    
    cnn = CNN(input_shape   =   input_shape,
            reshape_dims    =   reshape_dims,
            features        =   features,
            kernel_sizes    =   kernel_sizes,
            activations     =   activations,
            output_shape    =   output_shape)

    print(cnn)
    # Example input
    x = np.random.rand(1, 64)  # Batch of 1 with input shape (64,)
    # Forward pass
    output = cnn(x)
    print("Output shape:", output.shape)  # Should be (1, 1) after reshaping
    print("Output:", output)  # Output values
    # Note: The output will depend on the random initialization of the network parameters.
    # You can also check the number of parameters
    print("Number of parameters:", cnn.nparams)  # Number of trainable parameters in the model
