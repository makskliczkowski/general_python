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

class MaskedDense(nn.Dense):
    """
    A Dense layer where connections are masked to enforce autoregressive ordering.
    """
    mask: jnp.ndarray = None # The binary mask (in_features, out_features)

    @nn.compact
    def __call__(self, inputs):
        # We wrap the standard dense call but inject the mask into the kernel
        kernel          = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features), self.param_dtype)
        masked_kernel   = kernel * self.mask
        y               = jax.lax.dot_general(inputs, masked_kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        
        if self.use_bias:
            bias    = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
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
    dtype           : Any = jnp.complex128
    
    def setup(self):
        # Generate static masks (frozen at init)
        # Note: We output 1 value per site (logit for sigmoid) or 2 for softmax
        # Let's use 1 value per site (log-odds) for binary spins
        masks_np    = create_masks(self.n_sites, self.hidden_dims, n_out_per_site=1)
        self.masks  = [m for m in masks_np]
        self.layers = []
        
        # Hidden Layers
        for i, h_dim in enumerate(self.hidden_dims):
            self.layers.append(MaskedDense(features=h_dim, mask=self.masks[i], dtype=self.dtype))
            
        # Output Layer
        self.layers.append(MaskedDense(features=self.n_sites, mask=self.masks[-1], dtype=self.dtype))

    def __call__(self, x):
        # x shape: (Batch, N_sites)
        
        # Standard MLP pass with masked weights
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))   # or tanh/elu
            
        x = self.layers[-1](x)      # No activation on output (logits)
        return x
    
# ---------------------------------------------------------
# Combined Complex Autoregressive Network
# ---------------------------------------------------------

class PhaseDense(nn.Module):
    """
    A simple Feed-Forward network to estimate the Phase angle theta(s).
    """
    
    hidden_dims : Sequence[int]
    activation  : Any               = nn.tanh
    dtype       : Any               = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, N_sites)
        
        # Flatten input if needed (though Dense handles last dim)
        x = x.astype(self.dtype)
        
        for h in self.hidden_dims:
            x = nn.Dense(features=h, dtype=self.dtype)(x)
            x = self.activation(x)
            
        # Final layer: Output single scalar theta per sample
        # We initialize with small weights to start with ~0 phase
        x = nn.Dense(features=1, kernel_init=nn.initializers.normal(0.01), dtype=self.dtype)(x)
        
        # Output shape: (Batch, 1) -> Squeeze to (Batch,)
        return x.squeeze(axis=-1)

class FlaxComplexAutoregressive(nn.Module):
    """
    Combined Model: 
    1. Autoregressive Amplitude (MADE)
    2. Dense Phase
    """
    
    n_sites         : int
    ar_hidden       : Sequence[int]
    phase_hidden    : Sequence[int]
    dtype           : Any               = jnp.complex128    # Data type for complex outputs
    mu              : float             = 2.0               # Scaling factor for phase network output -> distribution from which phase is drawn
    
    def setup(self):
        ''' Initialize sub-networks '''
        from .net_autoregressive import FlaxMADE 
        self.amplitude_net  = FlaxMADE(n_sites=self.n_sites, hidden_dims=self.ar_hidden, dtype=self.dtype)
        self.phase_net      = PhaseDense(hidden_dims=self.phase_hidden, dtype=self.dtype)

    def __call__(self, x):
        """
        Returns full complex log_psi(s).
        Used by the Trainer for gradients.
        """
        
        # 1. Get Log Probabilities (Real)
        # MADE outputs logits. We need to compute log_prob of the specific configuration x.
        # We delegate this calculation to a helper to keep __call__ clean
        log_prob    = self._compute_log_prob_from_logits(x)
        
        # 2. Get Phase (Real scalar)
        theta       = self.phase_net(x)
        
        # 3. Combine: log_psi = 0.5 * log_P + i * theta
        return log_prob / self.mu + 1j * theta

    def _compute_log_prob_from_logits(self, x):
        """
        Helper to extract log P(s) from the MADE logits.
        """
        
        # logits shape: (Batch, N_sites) - representing logit for spin +1
        logits = self.amplitude_net(x)
        
        # Calculate log_prob for the specific input configuration x (assuming 0/1 inputs)
        # x must be 0 or 1 here.
        # log_p(1) = -softplus(-logit)
        # log_p(0) = -softplus(logit)

        log_p_1         = -nn.softplus(-logits)
        log_p_0         = -nn.softplus(logits)
        
        # Select based on x
        log_p_per_site  = jnp.where(x > 0.5, log_p_1, log_p_0)
        
        # Sum over sites to get total log probability of the configuration
        return jnp.sum(log_p_per_site, axis=-1)

    # Methods for the Sampler
    
    def get_logits(self, x):
        """
        Used by ARSampler to generate samples. 
        Only runs the amplitude network.
        """
        return self.amplitude_net(x)

    def get_phase(self, x):
        """
        Used by ARSampler to compute the phase after sampling.
        """
        return self.phase_net(x)
    
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
        
        n_sites     = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        net_kwargs  = {
            'n_sites'       : n_sites,
            'ar_hidden'     : ar_hidden,
            'phase_hidden'  : phase_hidden,
            'dtype'         : dtype,
        }
        
        # Initialize parent FlaxInterface
        super().__init__(
            net_module      =   FlaxComplexAutoregressive,
            net_kwargs      =   net_kwargs,
            input_shape     =   input_shape,
            backend         =   'jax',
            dtype           =   dtype,
            seed            =   seed,
            **kwargs
        )
        
        # AR networks typically don't have analytic gradients implemented manually
        self._has_analytic_grad = False 

    def get_info(self) -> dict:
        return {
            'name'          :   'ComplexAR',
            'type'          :   'autoregressive',
            'n_sites'       :   self.input_dim,
            'ar_layers'     :   self._flax_module.ar_hidden,
            'phase_layers'  :   self._flax_module.phase_hidden
        }    
    
# =============================================================================
#! EOF
# =============================================================================
