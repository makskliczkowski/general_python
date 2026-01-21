"""
general_python.ml.net_impl.networks.net_mps
=============================================

Matrix Product State (MPS) Ansatz.

Implements an MPS using a Recurrent Neural Network (RNN) structure in JAX.
The wavefunction is given by the trace of a product of matrices:
    psi(s) = Trace( A[s_1] * A[s_2] * ... * A[s_N] )

This implementation supports:
- Periodic Boundary Conditions (Trace).
- Complex parameters.
- Efficient evaluation using `jax.lax.scan`.

Usage
-----
    from general_python.ml.networks import choose_network
    
    # Create an MPS with bond dimension 10
    mps = choose_network('mps', input_shape=(64,), bond_dim=10)

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
    raise ImportError("MPS requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxMPS(nn.Module):
    n_sites         : int
    bond_dim        : int
    phys_dim        : int   = 2 # Spin-1/2 default
    dtype           : Any   = jnp.complex128
    param_dtype     : Any   = jnp.complex128
    init_scale      : float = 0.01

    def setup(self):
        # MPS Tensors: A[i] of shape (phys_dim, bond_dim, bond_dim)
        # We share weights across sites (Translation Invariant MPS)
        # Shape: (phys_dim, bond_dim, bond_dim)
        
        if jnp.issubdtype(self.param_dtype, jnp.complexfloating):
            k_init = cplx_variance_scaling(self.init_scale, 'fan_in', 'normal', self.param_dtype)
        else:
            k_init = nn.initializers.normal(stddev=self.init_scale)

        # Tensor A: (d, D, D)
        self.A = self.param('tensors', k_init, (self.phys_dim, self.bond_dim, self.bond_dim), self.param_dtype)

    @nn.compact
    def __call__(self, s):
        
        # s shape: (batch, n_sites)
        if s.ndim == 1: 
            s = s[jnp.newaxis, :]
        
        # 1. Map inputs to indices (assuming 0/1 or similar integer-like inputs)
        # If inputs are +/- 1 or continuous, we might need a mapping.
        # Here we assume inputs are {0, 1} approx.
        s_idx   = jnp.round(s).astype(jnp.int32)
        # Ensure indices are within [0, phys_dim-1]
        s_idx   = jnp.clip(s_idx, 0, self.phys_dim - 1)

        # A: (d, D, D)
        A_eff   = self.A.astype(self.dtype)

        # 2. Contraction using Scan
        # We compute the product of matrices for each sample
        
        def contract_site(carry, site_idx):
            # carry is current matrix product (Batch, D, D)
            # site_idx is (Batch,)
            
            # Gather matrices for this site across the batch
            # matrices: (Batch, D, D)
            matrices    = A_eff[site_idx] 
            
            # Batch Matmul: (B, D, D) @ (B, D, D) -> (B, D, D)
            new_carry   = jnp.matmul(carry, matrices)
            return new_carry, None

        # Initial Identity matrices (Batch, D, D)
        batch_size      = s.shape[0]
        init_carry      = jnp.eye(self.bond_dim, dtype=self.dtype)[None, ...].repeat(batch_size, axis=0)

        # Scan over sites (axis 1 of s_idx)
        # s_idx.T shape is (n_sites, batch)
        final_prod, _   = jax.lax.scan(contract_site, init_carry, s_idx.T)

        # 3. Trace
        # Trace of product (Periodic BC)
        val             = jnp.trace(final_prod, axis1=1, axis2=2)
        
        # Return log(psi)
        # Note: Standard MPS can vanish/explode. Log-MPS is numerically safer but harder to implement directly as trace.
        return jnp.log(val)

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class MPS(FlaxInterface):
    """
    Matrix Product State (MPS) Interface.

    Parameters:
        input_shape (tuple): Shape of input (n_sites,).
        bond_dim (int): Virtual bond dimension (D).
        phys_dim (int): Physical dimension (d). Default 2.
    """
    def __init__(self,
                input_shape     : tuple,
                bond_dim        : int   = 10,
                phys_dim        : int   = 2,
                dtype           : Any   = jnp.complex128,
                param_dtype     : Optional[Any] = None,
                init_scale      : float = 0.01,
                seed            : int   = 0,
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("MPS requires JAX.")

        net_kwargs = {
            'n_sites'       : input_shape[0],
            'bond_dim'      : bond_dim,
            'phys_dim'      : phys_dim,
            'dtype'         : dtype,
            'param_dtype'   : param_dtype if param_dtype else dtype,
            'init_scale'    : init_scale
        }

        super().__init__(
            net_module  =   _FlaxMPS,
            net_kwargs  =   net_kwargs,
            input_shape =   input_shape,
            backend     =   'jax',
            dtype       =   dtype,
            seed        =   seed,
            **kwargs
        )
        self._name = 'mps'

    def __repr__(self) -> str:
        mod = self._flax_module
        return f"MPS(n={mod.n_sites}, bond={mod.bond_dim}, d={mod.phys_dim})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------