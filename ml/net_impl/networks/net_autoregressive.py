"""
QES.general_python.ml.net_impl.networks.net_autoregressive
==========================================================

Autoregressive Neural Network for Quantum States.

This module provides an implementation of Autoregressive (AR) neural networks
for representing quantum wavefunctions. The wavefunction is factorized as:

    ψ(s1, s2, ..., sn) = p(s1) * p(s2|s1) * ... * p(sn|s1,...,sn-1)

Each conditional probability is modeled by a separate neural network.

Usage
-----
Import and use the Autoregressive network:

    from QES.general_python.ml.networks import Autoregressive
    ar_net = Autoregressive(input_shape=(10,), hidden_layers=(32,))

See the documentation and examples for more details.
----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 01.11.2025
Description     : Autoregressive Neural Network for quantum states. We implement
                  MADE-like architecture using Flax. MADE (Masked Autoencoder for
                  Distribution Estimation) is a well-known autoregressive model.
----------------------------------------------------------
"""

from typing import Sequence, Callable, Any, Tuple
import numpy as np

try:
    from QES.general_python.ml.net_impl.interface_net_flax import FlaxInterface, JAX_AVAILABLE
    
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required to use Autoregressive networks.")
    
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
except ImportError:
    raise ImportError("JAX and Flax are required to use Autoregressive network.")

# ---------------------------------------------------------
# Masked Dense Layer
# ---------------------------------------------------------

class MaskedDense(nn.Module):
    """
    Masked Dense Layer for MADE.
    Applies a dense layer with a fixed binary mask on the weights to enforce
    the autoregressive property.
    
    Attributes:
        features (int): 
            Number of output features.
        mask (jnp.ndarray): 
            Binary mask to apply to the weights.
        dtype (Any):
            Data type of the layer (default: jnp.float32).
    Methods:
        __call__(inputs):
            Forward pass through the masked dense layer.
            
    Example:
        >>> mask = jnp.array([[1, 0], [1, 1]])
        >>> layer = MaskedDense(features=2, mask=mask)
        >>> x = jnp.array([[0.5, 1.0]])
        >>> y = layer(x)
        >>> print(y)
        ... # Output will be the result of the masked dense layer.
    """
    features        : int
    mask            : jnp.ndarray
    dtype           : Any = jnp.float32
    use_bias        : bool = True
    kernel_init     : Any = nn.initializers.lecun_normal()
    bias_init       : Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        # inputs shape: (batch, in_features)
        in_features     = inputs.shape[-1]
        dtype           = self.param_dtype if hasattr(self, 'param_dtype') else self.dtype
        
        # Create Weights
        kernel          = self.param('kernel', self.kernel_init, (in_features, self.features), dtype)
        
        # Apply Mask (Soft masking ensures gradients flow correctly)
        # We cast mask to the kernel's dtype to avoid type promotion errors
        masked_kernel   = kernel * self.mask.astype(kernel.dtype)
        
        # Contraction
        y               = jax.lax.dot_general(inputs, masked_kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        
        # Bias
        if self.use_bias:
            bias    = self.param('bias', self.bias_init, (self.features,), dtype)
            y       = y + bias
        return y
    
# ---------------------------------------------------------
# Topology Helper (Mask Generation)
# ---------------------------------------------------------

def create_masks(n_in, hidden_sizes, n_out_per_site=2, dtype=jnp.float32):
    """
    Creates masks for MADE. The masks enforce the autoregressive property:
    - Each output unit k can only depend on input units with indices less than k.
    - Each hidden unit has a degree that enforces the autoregressive property.
    Parameters:
    -----------
    n_in : int
        Number of input units (e.g., number of sites).
    hidden_sizes : Sequence[int]
        Sizes of hidden layers.
    n_out_per_site : int
        Number of output units per input site (e.g., 2 for spin up/down logits).
    Returns:
    --------
    List of masks (jnp.ndarray) for each layer. Each mask is a binary matrix where
    a 1 indicates a connection is allowed, and 0 indicates no connection.
    The masks are ordered from input to first hidden, between hidden layers,
    and from last hidden to output.
    
    Ensures output k depends only on inputs < k. This is done by assigning
    degrees to hidden units and constructing masks accordingly.
    """
    L           = len(hidden_sizes)
    masks       = []
    
    # degrees[0]    -> input degrees (0 to N-1)
    # degrees[1..L] -> hidden layer degrees
    degrees     = [np.arange(n_in)]
    
    # Generate random degrees for hidden layers
    for i, h in enumerate(hidden_sizes):
        # connectivity constraint: m^l_k >= m^{l-1}_k
        prev_m  = degrees[-1]
        m       = np.random.randint(low=np.min(prev_m), high=n_in - 1, size=h)
        degrees.append(m)
    
    # Construct masks
    # Input -> Hidden 1
    masks.append(degrees[1][:, None] >= degrees[0][None, :])
    
    # Hidden -> Hidden
    for i in range(1, L):
        masks.append(degrees[i+1][:, None] >= degrees[i][None, :])
        
    # Hidden -> Output
    # For site k, we want outputs that predict state k.
    # To predict state k, we need info from < k. 
    # So output k connects to hidden units with degree < k.
    
    # We repeat degrees for the output layer (e.g. 2 outputs per site for spin up/down log-prob)
    out_degrees = np.arange(n_in) 
    out_degrees = np.repeat(out_degrees, n_out_per_site)
    
    masks.append(out_degrees[:, None] > degrees[-1][None, :])
    
    return [jnp.array(m.T, dtype=dtype) for m in masks]

# ---------------------------------------------------------
# The Flax Module
# ---------------------------------------------------------

class FlaxMADE(nn.Module):
    """
    Flax implementation of the Masked Autoencoder for Distribution Estimation (MADE).
    This module constructs an autoregressive neural network using masked dense layers,
    ensuring that each output only depends on the appropriate subset of inputs as defined
    by the autoregressive property. The masks are statically generated at initialization.
    
    Attributes:
        n_sites (int): 
            Number of input/output sites (features).
        hidden_dims (Sequence[int]): 
            List of hidden layer sizes.
        dtype (Any): 
            Data type for the layers (default: jnp.complex128).
    Methods:
        setup():
            Initializes the masked dense layers and generates the masks for each layer.
        __call__(x):
            Forward pass through the masked network.
            Args:
                x (jnp.ndarray): Input tensor of shape (batch_size, n_sites).
            Returns:
                jnp.ndarray: Output tensor of shape (batch_size, n_sites), representing logits.
    """
    
    n_sites         : int
    hidden_dims     : Sequence[int]
    dtype           : Any = jnp.complex128      # Data type for complex outputs
    activation      : Any = nn.gelu             # GELU is often preferred in SOTA Transformers/NQS

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, N_sites)
        
        # Generate Masks on the fly (or cache them)
        # Since topology is static, this overhead is negligible during JIT compilation.
        masks_np    = create_masks(self.n_sites, self.hidden_dims, n_out_per_site=1)
        masks       = [jnp.array(m) for m in masks_np]

        # Forward Pass
        # We iterate through config, creating layers lazily.
        # This completely Bypasses the 'tuple has no append' error in Flax.
        
        # Input -> Hidden layers
        for i, (h_dim, mask) in enumerate(zip(self.hidden_dims, masks[:-1])):
            # Create layer with a unique name based on index
            layer   = MaskedDense(features=h_dim, mask=mask, dtype=self.dtype, name=f'masked_dense_{i}')
            x       = layer(x)
            x       = self.activation(x)
            
        # Hidden -> Output (No activation, these are logits)
        output_layer    = MaskedDense(features=self.n_sites, mask=masks[-1], dtype=self.dtype, name='masked_out')
        x               = output_layer(x)
        
        return x

# ---------------------------------------------------------
# Combined Complex Autoregressive Network
# ---------------------------------------------------------

class FlaxComplexAutoregressive(nn.Module):
    """
    SOTA Combined Model: 
    1. Autoregressive Amplitude (MADE)  - Models the amplitude |ψ(s)|.
    2. Autoregressive Phase     (MADE)  - Crucial for non-trivial sign structures.
    """
    n_sites         : int
    ar_hidden       : Sequence[int]
    phase_hidden    : Sequence[int]
    dtype           : Any       = jnp.complex128
    mu              : float     = 2.0 
    
    def setup(self):
        self.amplitude_net = FlaxMADE(
            n_sites=self.n_sites, 
            hidden_dims=self.ar_hidden, 
            dtype=self.dtype, 
            name='amplitude_made'
        )
        self.phase_net = FlaxMADE(
            n_sites=self.n_sites, 
            hidden_dims=self.phase_hidden, 
            dtype=self.dtype, 
            name='phase_made'
        )    
    
    def _flatten_input(self, x):
        """ 
        Robustly flatten input to (..., N_sites).
        Handles: (100, 1) -> (100,), (B, 100, 1) -> (B, 100)
        """
        # If last dim matches n_sites, we assume it is [..., n_sites]
        if x.shape[-1] == self.n_sites:
            return x
            
        # If total size matches n_sites, it is a single sample (vmapped)
        if x.size == self.n_sites:
            return x.reshape((self.n_sites,))
            
        # Otherwise, assume batch and flatten
        return x.reshape((-1, self.n_sites))
    
    def __call__(self, x):
        """ Returns full complex log_psi(s). """
        # 1. Robust Flatten
        x_flat = self._flatten_input(x)
        # Ensure input to dense layers is the correct dtype (Complex)
        x_input = x_flat.astype(self.dtype)

        # --- AMPLITUDE ---
        # Get logits and FORCE REAL part. 
        # This aligns the training objective with the Sampler (which samples using Real logits).
        logits_amp = jnp.real(self.amplitude_net(x_input))
        
        log_p_1 = -nn.softplus(-logits_amp)
        log_p_0 = -nn.softplus(logits_amp)
        
        # Select based on x (real mask)
        # Note: x_flat might be integer/float, >0.5 works for binary
        log_p_per_site = jnp.where(jnp.real(x_flat) > 0.5, log_p_1, log_p_0)
        log_prob = jnp.sum(log_p_per_site, axis=-1)
        
        # --- PHASE ---
        # Force phase to be Real scalar (theta), so exp(i*theta) is pure phase.
        logits_phase = jnp.real(self.phase_net(x_input))
        theta = jnp.sum(logits_phase * jnp.real(x_flat), axis=-1)

        # Combine
        return log_prob / self.mu + 1j * theta

    def get_logits(self, x):
        """ Used by ARSampler for Amplitude Sampling """
        x_flat = self._flatten_input(x)
        # Return REAL logits to prevent Sampler TypeErrors
        return jnp.real(self.amplitude_net(x_flat.astype(self.dtype)))

    def get_phase(self, x):
        """ Used by ARSampler to compute phase """
        x_flat = self._flatten_input(x)
        logits_phase = jnp.real(self.phase_net(x_flat.astype(self.dtype)))
        return jnp.sum(logits_phase * jnp.real(x_flat), axis=-1)
    
# ---------------------------------------------------------

class ComplexAR(FlaxInterface):
    """
    Autoregressive Neural Quantum State with Phase.
    
    Wraps FlaxComplexAR to provide standard interface.
    """
    def __init__(self,
                input_shape     : tuple,
                ar_hidden       : Tuple[int, ...]   = (32, 32),
                phase_hidden    : Tuple[int, ...]   = (32, 32),
                dtype           : Any               = jnp.complex128,
                seed            : int               = 0,
                backend         : str               = 'jax',
                **kwargs):
        
        if not JAX_AVAILABLE or backend != 'jax':
            raise RuntimeError("JAX is required for Autoregressive networks")
        
        # 1. Prepare Arguments for the Flax Dataclass
        n_sites = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        
        # Extract 'mu' if provided in kwargs, default to 2.0 (standard for log-prob -> log-psi)
        mu_val  = kwargs.get('mu', 2.0)

        net_kwargs = {
            'n_sites'       : n_sites,
            'ar_hidden'     : ar_hidden,
            'phase_hidden'  : phase_hidden,
            'dtype'         : dtype,
            'mu'            : mu_val
        }
        
        # 2. Store config for get_info (Safer than accessing _flax_module attributes later)
        self._ar_hidden     = ar_hidden
        self._phase_hidden  = phase_hidden
        self._n_sites       = n_sites

        # 3. Initialize parent FlaxInterface
        # This will instantiate FlaxComplexAutoregressive(**net_kwargs)
        # and then call .init() using nn.compact logic.
        super().__init__(
            net_module      =   FlaxComplexAutoregressive,
            net_kwargs      =   net_kwargs,
            input_shape     =   input_shape,
            backend         =   'jax',
            dtype           =   dtype,
            seed            =   seed,
            **kwargs
        )
        self._has_analytic_grad = False 

    def get_info(self) -> dict:
        """ Return metadata about the network architecture. """
        return {
            'name'          : 'ComplexAR',
            'type'          : 'autoregressive',
            'sota'          : True,
            'n_sites'       : self._n_sites,
            'ar_layers'     : self._ar_hidden,
            'phase_layers'  : self._phase_hidden,
            'dtype'         : str(self._dtype)
        }
    
# =============================================================================
#! EOF
# =============================================================================
