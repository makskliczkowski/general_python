import numpy as np
from typing import Tuple, Callable, Optional, Any, Sequence, Union
import math

try:
    from general_python.ml.net_impl.interface_net_flax import FlaxInterface
    from general_python.ml.net_impl.utils.net_init_jax import cplx_variance_scaling, lecun_normal
    from general_python.algebra.utils import JAX_AVAILABLE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE
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
    reshape_dims        : Tuple[int, ...]           # e.g., (Lx, Ly)
    features            : Sequence[int]             # e.g., (16, 32)
    kernel_sizes        : Sequence[Tuple[int,...]]  # e.g., ((3,3), (3,3))
    strides             : Sequence[Tuple[int,...]]  # e.g., ((1,1), (1,1))
    activation_fns      : Sequence[Callable]        # e.g., (nn.relu, nn.relu)
    use_bias            : Sequence[bool]            # e.g., (True, True)
    param_dtype         : jnp.dtype                 = DEFAULT_JP_CPX_TYPE
    dtype               : jnp.dtype                 = DEFAULT_JP_CPX_TYPE
    input_channels      : int                       = 1
    output_features     : int                       # Total number of output features (e.g., prod(output_shape))
    final_activation    : Optional[Callable]        = None # Activation for the *output* layer
    in_activation       : Optional[Callable]        = None # Activation for the *input* layer
    
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
        
        #! 1. Input Reshaping and Validation
        n_visible_expected = math.prod(self.reshape_dims)
        if s.shape[-1] != n_visible_expected:
            raise ValueError(f"Input's last dimension ({s.shape[-1]}) does not match "
                            f"the expected number of visible units ({n_visible_expected}) "
                            f"from reshape_dims {self.reshape_dims}.")

        # Reshape: (batch, n_visible) -> (batch, Lx, Ly, ..., input_channels)
        target_shape    = (-1,) + self.reshape_dims + (self.input_channels,)
        x               = s.reshape(target_shape).astype(self.dtype)

        #! Optional Input Activation
        if self.in_activation is not None:
            x = self.in_activation(x)

        #! 2. Convolutional Layers
        num_layers      = len(self.features)
        complex_dtype   = jnp.issubdtype(self.param_dtype, jnp.complexfloating)

        for i in range(num_layers):
            # Determine Initializer based on parameter dtype
            if complex_dtype:
                kernel_init_fn = cplx_variance_scaling(1.0, 'fan_in', 'normal', dtype=self.param_dtype)
            else:
                kernel_init_fn = lecun_normal(dtype=self.param_dtype) # Good default for ReLU-like
            bias_init_fn = jax.nn.initializers.zeros

            # Define Convolution Layer for this step
            conv_layer = nn.Conv(
                features    =   self.features[i],
                kernel_size =   self.kernel_sizes[i],
                strides     =   self.strides[i],
                use_bias    =   self.use_bias[i],
                padding     =   'SAME',             # Keeps spatial dimensions the same (for stride 1)
                dtype       =   self.dtype,         # Computation dtype
                param_dtype =   self.param_dtype,   # Parameter dtype
                kernel_init =   kernel_init_fn,
                bias_init   =   bias_init_fn,
                name        =   f"conv_layer_{i}"
            )

            # Apply Convolution and Activation
            x = conv_layer(x)
            x = self.activation_fns[i](x)

        #! 3. Flatten the output of the convolutional part
        
        # Shape becomes (batch, flattened_features)
        x = x.reshape((x.shape[0], -1))

        #! 4. Final Dense Layer to achieve desired output features
        if complex_dtype:
            dense_kernel_init = cplx_variance_scaling(1.0, 'fan_in', 'normal', dtype=self.param_dtype)
        else:
            dense_kernel_init = lecun_normal(dtype=self.param_dtype)
        dense_bias_init = jax.nn.initializers.zeros

        output_layer = nn.Dense(
            features        =   self.output_features,
            name            =   "output_dense",
            use_bias        =   True, # Usually use bias in the final dense layer
            kernel_init     =   dense_kernel_init,
            bias_init       =   dense_bias_init,
            dtype           =   self.dtype,
            param_dtype     =   self.param_dtype
        )
        output = output_layer(x)                    # Shape: (batch, output_features)

        #! 5. Apply final activation if specified
        if self.final_activation is not None:
            output = self.final_activation(output)

        # Return the final output, shape (batch, output_features)
        return output.reshape(-1) # Flatten to (batch, output_features)

##########################################################
#! CNN WRAPPER CLASS USING FlaxInterface
##########################################################

class CNN(FlaxInterface):
 """
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

        #! Validate Input Shape and Reshape Dimensions
        if len(input_shape) != 1:
            raise ValueError(f"input_shape must be 1D (e.g., (n_visible,)), got {input_shape}")
        
        n_visible   = input_shape[0]
        # Dimensionality of the reshaped input/convolution
        n_dim       = len(reshape_dims) if kwargs.get('n_dim', None) is None else kwargs['n_dim']
        # Allow 1D, 2D, 3D convolutions primarily
        if n_dim not in [1, 2, 3]:
            raise ValueError(f"reshape_dims implies {n_dim}D structure. Common CNNs are 1D, 2D, 3D.")
        reshape_dims= reshape_dims[:n_dim] # Ensure we only use the first n_dim dimensions
        
        if math.prod(reshape_dims) != n_visible:
            raise ValueError(f"Product of reshape_dims {reshape_dims} ({math.prod(reshape_dims)}) "
                            f"must equal input_shape[0] ({n_visible}).")

        #! Resolve Dtypes
        final_dtype             = self._resolve_dtype(dtype)
        final_param_dtype       = self._resolve_dtype(param_dtype) if param_dtype is not None else final_dtype
        self._is_cpx            = jnp.issubdtype(final_param_dtype, jnp.complexfloating)

        #! Process Layer Arguments (Convolutional Layers)
        num_layers              = len(features)

        processed_kernels       = []
        if len(kernel_sizes) != num_layers:
            raise ValueError(f"Length of kernel_sizes ({len(kernel_sizes)}) != features ({num_layers}).")
        
        # Process kernel sizes
        for k in kernel_sizes:
            if isinstance(k, int):
                # if is int, convert to tuple of length n_dim
                processed_kernels.append(tuple([k] * n_dim))
            elif isinstance(k, tuple) and len(k) == n_dim:
                processed_kernels.append(k)
            else:
                raise ValueError(f"Kernel size {k} must be int or tuple of length {n_dim}.")

        processed_strides = []
        if strides is None:
            # Default to 1 for all layers
            strides = [1] * num_layers
            
        if len(strides) != num_layers:
            raise ValueError(f"Length of strides ({len(strides)}) != features ({num_layers}).")
        for s in strides:
            if isinstance(s, int):
                processed_strides.append(tuple([s] * n_dim))
            elif isinstance(s, tuple) and len(s) == n_dim:
                processed_strides.append(s)
            else:
                raise ValueError(f"Stride {s} must be int or tuple of length {n_dim}.")

        if isinstance(activations, (str, Callable.__class__)):
            # If a single activation function is provided, use it for all layers
            processed_activations = [activations] * num_layers
        elif isinstance(activations, Sequence) and len(activations) == num_layers:
            processed_activations = activations
        else:
            raise ValueError(f"activations must be single spec or sequence of length {num_layers}.")
        #! Activation functions themselves are initialized by FlaxInterface

        if isinstance(use_bias, bool):
            processed_bias = [use_bias] * num_layers
        elif isinstance(use_bias, Sequence) and len(use_bias) == num_layers:
            processed_bias = list(use_bias)
        else:
            raise ValueError(f"use_bias must be bool or sequence of length {num_layers}.")

        #! Process Output Layer Arguments
        # Total number of features in the desired output shape, reshaping may happen somewhere else
        output_features         = math.prod(output_shape) 
        # Final activation spec is passed directly; FlaxInterface handles it via net_kwargs
        final_activation_spec   = final_activation

        #! Prepare kwargs for _FlaxCNN
        net_kwargs = {
            'reshape_dims'      : tuple(reshape_dims),
            'features'          : tuple(features),
            'kernel_sizes'      : tuple(processed_kernels),
            'strides'           : tuple(processed_strides),
            'activation_fns'    : tuple(processed_activations), # Pass specs
            'use_bias'          : tuple(processed_bias),
            'param_dtype'       : final_param_dtype,
            'dtype'             : final_dtype,
            'input_channels'    : 1,
            'output_features'   : output_features,              # Pass total output features
            'final_activation'  : final_activation_spec,        # Pass final activation spec
            'in_activation'     : in_activation,                # Pass input activation spec
            **kwargs
        }

        # Store the desired final output shape for potential use outside
        self._output_shape = output_shape

        #! Initialize using FlaxInterface parent class
        super().__init__(
            net_module      = _FlaxCNN,
            net_args        = (),
            net_kwargs      = net_kwargs,
            input_shape     = input_shape,
            in_activation   = in_activation,
            backend         = 'jax',
            dtype           = final_dtype,
            seed            = seed
        )

        self._has_analytic_grad = False
        self._compiled_grad_fn  = None

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Returns the configured output shape (excluding batch dimension)."""
        return self._output_shape

    #! Ca

    def __call__(self, s: 'array-like'):
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
        init_status = "initialized" if self.initialized else "uninitialized"
        cnn_type    = "Complex" if self._is_cpx else "Real"
        if self.initialized:
            features    = self._flax_module.features
            kernels     = self._flax_module.kernel_sizes
            reshape     = self._flax_module.reshape_dims
            n_params    = self.nparams
            dtype       = self.dtype
            out_shape   = self.output_shape
        else: # Fallback to kwargs before initialization
            features    = self._net_kwargs.get('features', '?')
            kernels     = self._net_kwargs.get('kernel_sizes', '?')
            reshape     = self._net_kwargs.get('reshape_dims', '?')
            n_params    = '?'
            dtype       = self._net_kwargs.get('dtype', '?')
            out_shape   = self._output_shape

        return (f"{cnn_type}CNN(reshape={reshape}, features={features}, kernels={kernels}, "
                f"output_shape={out_shape}, dtype={dtype}, params={n_params}, {init_status})")
        
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
