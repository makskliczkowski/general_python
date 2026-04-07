"""
general_python.ml.net_impl.networks.net_autoregressive
======================================================

Autoregressive wrapper built on masked feedforward layers.

The model factorizes the output over an ordered sequence of input sites and
uses static binary masks to enforce the autoregressive dependency structure.


WIP - THIS MODULE IS EXPERIMENTAL AND SUBJECT TO SIGNIFICANT CHANGES. DO NOT USE OUTSIDE TESTS OR INTERNAL PROTOTYPING.


----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-21
License         : MIT
Version         : 0.1 (Experimental)
----------------------------------------------------------
"""

from typing import Sequence, Callable, Any, Tuple
import numpy as np

try:
    from ....ml.net_impl.interface_net_flax         import FlaxInterface, JAX_AVAILABLE
    from ....ml.net_impl.utils.net_wrapper_utils    import configure_nqs_metadata
    
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
    """Dense layer with a fixed binary mask on the weights."""
    features        : int
    mask            : jnp.ndarray
    dtype           : Any = jnp.float32
    param_dtype     : Any = None
    use_bias        : bool = True
    kernel_init     : Any = nn.initializers.lecun_normal()
    bias_init       : Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        in_features     = inputs.shape[-1]
        dtype           = self.dtype if self.param_dtype is None else self.param_dtype
        kernel          = self.param('kernel', self.kernel_init, (in_features, self.features), dtype)
        masked_kernel   = kernel * self.mask.astype(kernel.dtype)
        y               = jax.lax.dot_general(inputs, masked_kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        if self.use_bias:
            bias    = self.param('bias', self.bias_init, (self.features,), dtype)
            y       = y + bias
        return y
    
# ---------------------------------------------------------
# Topology Helper (Mask Generation)
# ---------------------------------------------------------

_MASK_CACHE = {}

def create_masks(n_in, hidden_sizes, n_out_per_site=2, dtype=jnp.float32, seed=None):
    """Create static MADE masks for the requested topology."""
    cache_key = (n_in, tuple(hidden_sizes), n_out_per_site, jnp.dtype(dtype).name, seed)
    if cache_key in _MASK_CACHE:
        return _MASK_CACHE[cache_key]
    
    L = len(hidden_sizes)
    masks = []
    degrees = [np.arange(n_in)]
    
    for i, h in enumerate(hidden_sizes):
        prev_m  = degrees[-1]
        min_deg = int(np.min(prev_m))
        max_deg = n_in - 1
        
        if seed is not None:
            rng = np.random.RandomState(seed + i)
            m   = rng.randint(low=min_deg, high=max_deg + 1, size=h)
        else:
            m   = np.linspace(min_deg, max_deg, h).astype(np.int32)
        degrees.append(m)
    
    masks.append(degrees[1][:, None] >= degrees[0][None, :])
    for i in range(1, L):
        masks.append(degrees[i+1][:, None] >= degrees[i][None, :])

    out_degrees = np.arange(n_in) 
    out_degrees = np.repeat(out_degrees, n_out_per_site)
    
    masks.append(out_degrees[:, None] > degrees[-1][None, :])
    
    result = [jnp.array(m.T, dtype=dtype) for m in masks]
    _MASK_CACHE[cache_key] = result
    
    return result

# ---------------------------------------------------------
# The Flax Module
# ---------------------------------------------------------

class FlaxMADE(nn.Module):
    """MADE backbone used for amplitude and phase autoregressive heads."""
    
    n_sites         : int
    hidden_dims     : Sequence[int]
    dtype           : Any = jnp.complex128
    activation      : Any = nn.gelu
    kernel_init     : Any = nn.initializers.lecun_normal()
    bias_init       : Any = nn.initializers.zeros
    
    def setup(self):
        self._masks = create_masks(
            self.n_sites,
            self.hidden_dims,
            n_out_per_site=1,
            dtype=jnp.bool_,
        )

    @nn.compact
    def __call__(self, x):
        for i, (h_dim, mask) in enumerate(zip(self.hidden_dims, self._masks[:-1])):
            layer = MaskedDense(
                features=h_dim,
                mask=mask,
                dtype=self.dtype,
                name=f'masked_dense_{i}',
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            x       = layer(x)
            x       = self.activation(x)
            
        output_layer = MaskedDense(
            features=self.n_sites,
            mask=self._masks[-1],
            dtype=self.dtype,
            name='masked_out',
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        x               = output_layer(x)
        
        return x

# ---------------------------------------------------------
# Combined Complex Autoregressive Network
# ---------------------------------------------------------

class FlaxComplexAutoregressive(nn.Module):
    """Complex autoregressive ansatz with separate amplitude and phase heads."""
    n_sites         : int
    ar_hidden       : Sequence[int]
    phase_hidden    : Sequence[int]
    dtype           : Any       = jnp.complex128
    mu              : float     = 2.0 
    
    def setup(self):
        self.amplitude_net      = FlaxMADE(
                                    n_sites         =   self.n_sites, 
                                    hidden_dims     =   self.ar_hidden, 
                                    dtype           =   self.dtype, 
                                    name            =   'amplitude_made'
                                )
            
        self.phase_net          = FlaxMADE(
                                    n_sites         =   self.n_sites, 
                                    hidden_dims     =   self.phase_hidden, 
                                    dtype           =   self.dtype, 
                                    name            =   'phase_made',
                                    # Overwrite init to be small noise instead of zero
                                    kernel_init     =   nn.initializers.normal(stddev=0.01), 
                                    bias_init       =   nn.initializers.normal(stddev=0.01)
                                )    

    def _flatten_input(self, x):
        ''' Flatten input to shape (batch_size, n_sites) '''
        if x.shape[-1] == self.n_sites:
            return x
        if x.size == self.n_sites:
            return x.reshape((self.n_sites,))
        return x.reshape((-1, self.n_sites))

    def _to_binary_from_physical(self, x):
        ''' Convert signed or binary inputs to native binary occupancy. '''
        x_real = jnp.real(x)
        return jnp.where(x_real > 0, 1.0, 0.0)

    def _prepare_binary_input(self, x_binary):
        ''' Cast binary inputs once before MADE evaluation. '''
        return x_binary.astype(self.dtype)

    def _phase_from_binary(self, x_binary, x_input):
        ''' Compute accumulated phase from binary input. '''
        logits_phase    = jnp.real(self.phase_net(x_input))
        theta_per_site  = jnp.pi * nn.tanh(logits_phase)
        return jnp.sum(theta_per_site * x_binary, axis=-1)

    def _log_psi_from_binary(self, x_binary):
        ''' Compute log(psi) from native binary inputs. '''
        x_input         = self._prepare_binary_input(x_binary)
        logits_amp      = jnp.real(self.amplitude_net(x_input))
        log_p_1         = -nn.softplus(-logits_amp)
        log_p_0         = -nn.softplus(logits_amp)
        log_p_per_site  = jnp.where(x_binary > 0.5, log_p_1, log_p_0)
        log_prob        = jnp.sum(log_p_per_site, axis=-1)
        theta           = self._phase_from_binary(x_binary, x_input)
        return log_prob / self.mu + 1j * theta

    def __call__(self, x):
        ''' Forward pass to compute log(psi). '''
        x_binary = self._to_binary_from_physical(self._flatten_input(x))
        return self._log_psi_from_binary(x_binary)

    def get_logits_binary(self, x_binary):
        ''' Get amplitude logits for native binary input. '''
        x_binary = self._flatten_input(x_binary)
        return jnp.real(self.amplitude_net(self._prepare_binary_input(x_binary)))

    def get_phase_binary(self, x_binary):
        ''' Get phase for native binary input. '''
        x_binary = self._flatten_input(x_binary)
        x_input = self._prepare_binary_input(x_binary)
        return self._phase_from_binary(x_binary, x_input)

    def get_logits(self, x, is_binary: bool = False):
        ''' Get amplitude logits for a configuration. '''
        x_flat = self._flatten_input(x)
        x_binary = x_flat if is_binary else self._to_binary_from_physical(x_flat)
        return self.get_logits_binary(x_binary)

    def get_phase(self, x, is_binary: bool = False):
        ''' Get phase for a configuration. '''
        x_flat = self._flatten_input(x)
        x_binary = x_flat if is_binary else self._to_binary_from_physical(x_flat)
        return self.get_phase_binary(x_binary)
    
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
        self._name = 'autoregressive'
        configure_nqs_metadata(
            self,
            family="autoregressive",
            native_representation="binary_01",
            supports_exact_sampling=True,
            preferred_sampler="ARSampler",
        )
        self._has_analytic_grad = False 

    def get_info(self) -> dict:
        """ Return metadata about the network architecture. """
        return {
            'name'          : 'ComplexAR',
            'type'          : 'autoregressive',
            'n_sites'       : self._n_sites,
            'ar_layers'     : self._ar_hidden,
            'phase_layers'  : self._phase_hidden,
            'dtype'         : str(self._dtype)
        }
    
    def __str__(self) -> str:
        typek = "Complex" if self._dtype in [jnp.complex64, jnp.complex128] else "Real"
        return f"{typek}AR(n_sites={self._n_sites},ar_layers={self._ar_hidden},phase_layers={self._phase_hidden},dtype={self._dtype})"
    
# =============================================================================
#! EOF
# =============================================================================
