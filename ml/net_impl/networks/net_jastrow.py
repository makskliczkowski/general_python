"""
general_python.ml.net_impl.networks.net_jastrow
=================================================

Quadratic correlation wrapper with optional linear bias term.

The module evaluates a dense quadratic form over the input together with an
optional one-body correction. It is a compact structured ansatz that remains
usable through the same generic ``FlaxInterface`` path as the other wrappers.

The Jastrow ansatz is a simple and interpretable way to capture pairwise correlations, and can be used as a standalone ansatz or as a building block 
in more complex architectures. The current implementation is a proof-of-concept and may be extended with additional features such as non-symmetric kernels, 
higher-order interactions, or alternative parameterizations. It has a form:
``log_psi(s) = s^T W s + b^T s``

where ``W`` is a learnable correlation kernel and ``b`` is an optional bias vector.

WIP - THIS MODULE IS EXPERIMENTAL AND SUBJECT TO SIGNIFICANT CHANGES. DO NOT USE OUTSIDE TESTS OR INTERNAL PROTOTYPING.


----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-21
License         : MIT
Version         : 0.1 (Experimental)
----------------------------------------------------------
"""

import jax
import jax.numpy    as jnp
import flax.linen   as nn
import numpy        as np
from typing         import Any, Optional

try:
    from ....ml.net_impl.interface_net_flax         import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax         import normal_by_dtype
    from ....ml.net_impl.utils.net_wrapper_utils    import configure_nqs_metadata
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
        k_init = normal_by_dtype(self.init_scale, self.param_dtype)

        # W matrix (Correlation kernel)
        self.W = self.param('kernel', k_init, (self.n_sites, self.n_sites), self.param_dtype)

        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.n_sites,), self.param_dtype)

    @nn.compact
    def __call__(self, s):
        # s shape: (batch, n_sites)
        if s.ndim == 1:
            s = s[jnp.newaxis, :]
            
        s           = s.astype(self.dtype)          # Ensure input is in the correct dtype
        W           = self.W.astype(self.dtype)     # Ensure parameters are in the correct dtype
        W_sym       = 0.5 * (W + jnp.swapaxes(W, -1, -2)) # Symmetrize W to ensure real-valued output for real inputs
        Ws          = jnp.matmul(s, W_sym)
        correlation = jnp.sum(s * Ws, axis=-1) # Quadratic form s^T W s

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
    Jastrow-style quadratic wrapper.

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

        n_sites     = int(np.prod(input_shape))
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
        self._name                          = 'jastrow'
        configure_nqs_metadata(self, family="jastrow")

    def __repr__(self) -> str:
        return f"Jastrow(n_sites={self._flax_module.n_sites}, dtype={self.dtype})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
