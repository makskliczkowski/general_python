"""
general_python.ml.net_impl.networks.net_jastrow
=================================================

Jastrow Ansatz for Quantum States.

The Jastrow ansatz explicitly models pairwise correlations between particles.
It is defined as:
    log_psi(s) = sum_{i, j} s_i * W_{ij} * s_j + sum_i b_i * s_i

This implementation supports:
- Full Jastrow: Unique W_{ij} for all pairs.
- Translation-Invariant (Convolutional) Jastrow (via kernel_init masking or shared weights - simplified here to full matrix for generality).
- Complex/Real parameters.

Usage
-----
    from general_python.ml.networks import choose_network
    
    # Create a complex Jastrow
    jastrow = choose_network('jastrow', input_shape=(64,), dtype='complex128')

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-21
License         : MIT
----------------------------------------------------------
"""

import jax
import jax.numpy    as jnp
import flax.linen   as nn
from typing         import Any, Optional, Sequence

try:
    from ....ml.net_impl.interface_net_flax import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax import cplx_variance_scaling, lecun_normal
    JAX_AVAILABLE = True
except ImportError:
    raise ImportError("Jastrow requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxJastrow(nn.Module):
    n_sites             : int
    use_bias            : bool  = True
    dtype               : Any   = jnp.complex128
    param_dtype         : Any   = jnp.complex128
    init_scale          : float = 0.01

    def setup(self):
        # Determine initialization function
        if jnp.issubdtype(self.param_dtype, jnp.complexfloating):
            k_init = cplx_variance_scaling(self.init_scale, 'fan_in', 'normal', self.param_dtype)
        else:
            k_init = nn.initializers.normal(stddev=self.init_scale)

        # W matrix (Correlation kernel)
        self.W = self.param('kernel', k_init, (self.n_sites, self.n_sites), self.param_dtype)

        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.n_sites,), self.param_dtype)

    @nn.compact
    def __call__(self, s):
        # s shape: (batch, n_sites)
        if s.ndim == 1:
            s = s[jnp.newaxis, :]
            
        s = s.astype(self.dtype)
        W = self.W.astype(self.dtype)
        
        # Enforce symmetry: 
        # W_sym = 0.5 * (W + W.T)
        Ws          = jnp.matmul(s, W)
        correlation = jnp.sum(s * Ws, axis=-1)

        # Bias term
        if self.use_bias:
            b           = self.bias.astype(self.dtype)
            linear      = jnp.sum(s * b, axis=-1)
            correlation = correlation + linear

        return correlation

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class Jastrow(FlaxInterface):
    """
    Jastrow Ansatz Interface.

    Parameters:
        input_shape (tuple): Shape of input (n_sites,).
        use_bias (bool): Whether to include a one-body bias term.
        init_scale (float): Scale for random initialization.
    """
    def __init__(self,
                input_shape     : tuple,
                use_bias        : bool = True,
                init_scale      : float = 0.01,
                dtype           : Any = jnp.complex128,
                param_dtype     : Optional[Any] = None,
                seed            : int = 0,
                **kwargs):

        if not JAX_AVAILABLE: 
            raise ImportError("Jastrow requires JAX.")

        n_sites     = input_shape[0] if len(input_shape) == 1 else input_shape[0] * input_shape[1] # Approximate handling
        net_kwargs  = {
            'n_sites'       : n_sites,
            'use_bias'      : use_bias,
            'dtype'         : dtype,
            'param_dtype'   : param_dtype if param_dtype else dtype,
            'init_scale'    : init_scale
        }

        super().__init__(
            net_module  =   _FlaxJastrow,
            net_kwargs  =   net_kwargs,
            input_shape =   input_shape,
            backend     =   'jax',
            dtype       =   dtype,
            seed        =   seed,
            **kwargs
        )
        self._name = 'jastrow'

    def __repr__(self) -> str:
        return f"Jastrow(n_sites={self._flax_module.n_sites}, dtype={self.dtype})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------