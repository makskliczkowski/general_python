"""
general_python.ml.net_impl.networks.net_mps
=============================================

Matrix-product-state style wrapper with explicit local-state indexing.

The module contracts a shared set of local tensors across the input chain and
returns the logarithm of the resulting trace amplitude. The implementation uses
``jax.lax.scan`` for efficient sequential contraction.

The MPS ansatz is a powerful and structured way to capture correlations in 1D systems, 
and can be used as a standalone ansatz or as a building block in more complex architectures. 
The current implementation is a proof-of-concept and may be extended with additional features such as open boundary conditions, 
non-periodic indexing, or alternative contraction schemes. It has a form:
``log_psi(s) = log(Tr[A[s_1] A[s_2] ... A[s_n]])``

where ``A`` is a learnable tensor of shape (phys_dim, bond_dim, bond_dim) and ``s_i`` are the input physical indices.


WIP - THIS MODULE IS EXPERIMENTAL AND SUBJECT TO SIGNIFICANT CHANGES. DO NOT USE OUTSIDE TESTS OR INTERNAL PROTOTYPING.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-21
License         : MIT
Version         : 0.1 (Experimental)
----------------------------------------------------------
"""

from typing import Any, Optional

try:
    import jax
    import jax.numpy    as jnp
    import flax.linen   as nn

    from ....ml.net_impl.interface_net_flax         import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax         import normal_by_dtype
    from ....ml.net_impl.utils.net_state_repr_jax   import state_to_binary_index
    from ....ml.net_impl.utils.net_wrapper_utils    import (
                                                        configure_nqs_metadata,
                                                        extract_input_convention,
                                                        infer_native_representation,
                                                    )
    from ....algebra.utils                          import BACKEND_DEF_SPIN, BACKEND_REPR
    JAX_AVAILABLE = True
except ImportError:
    raise ImportError("MPS requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxMPS(nn.Module):
    n_sites         : int
    bond_dim        : int
    phys_dim        : int   = 2
    dtype           : Any   = jnp.complex128
    param_dtype     : Any   = jnp.complex128
    init_scale      : float = 0.01
    input_is_spin   : bool  = BACKEND_DEF_SPIN
    input_value     : float = BACKEND_REPR

    def setup(self):
        k_init = normal_by_dtype(self.init_scale, self.param_dtype)
        self.A = self.param('tensors', k_init, (self.phys_dim, self.bond_dim, self.bond_dim), self.param_dtype)

    @nn.compact
    def __call__(self, s):
        
        # s shape: (batch, n_sites)
        needs_batch = s.ndim == 1
        if needs_batch:
            s = s[jnp.newaxis, :]
        
        # 1. Map inputs to physical indices
        s_real = jnp.real(s)
        if self.phys_dim == 2:
            s_idx = state_to_binary_index(s_real, self.input_is_spin, self.input_value)
        else:
            s_idx = jnp.round(s_real).astype(jnp.int32)
            s_idx = jnp.clip(s_idx, 0, self.phys_dim - 1)

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
        eye_d           = jnp.eye(self.bond_dim, dtype=self.dtype)
        init_carry      = jnp.broadcast_to(eye_d, (batch_size, self.bond_dim, self.bond_dim))

        # Scan over sites (axis 1 of s_idx)
        # s_idx.T shape is (n_sites, batch)
        final_prod, _   = jax.lax.scan(contract_site, init_carry, s_idx.T)

        # 3. Trace
        # Trace of product (Periodic BC)
        val             = jnp.trace(final_prod, axis1=1, axis2=2)
        
        out = jnp.log(val)
        return out[0] if needs_batch else out

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class MPS(FlaxInterface):
    """
    Matrix-product-state wrapper.

    Parameters:
        input_shape (tuple): Shape of the 1D input chain.
        bond_dim (int): Virtual bond dimension.
        phys_dim (int): Local physical dimension. Default 2.
    """
    def __init__(self,
                input_shape     : tuple,
                bond_dim        : int   = 10,
                phys_dim        : int   = 2,
                dtype           : Any   = jnp.complex128,
                param_dtype     : Optional[Any] = None,
                init_scale      : float = 0.01,
                seed            : int   = 0,
                backend         : str   = 'jax',
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("MPS requires JAX.")

        input_convention = extract_input_convention(kwargs)
        net_kwargs = {
            'n_sites'       : input_shape[0],
            'bond_dim'      : bond_dim,
            'phys_dim'      : phys_dim,
            'dtype'         : dtype,
            'param_dtype'   : param_dtype if param_dtype else dtype,
            'init_scale'    : init_scale,
            **input_convention,
        }

        super().__init__(
            net_module  =   _FlaxMPS,
            net_kwargs  =   net_kwargs,
            input_shape =   input_shape,
            backend     =   backend,
            dtype       =   dtype,
            seed        =   seed,
            **kwargs
        )
        self._name = 'mps'
        configure_nqs_metadata(
            self,
            family="mps",
            native_representation=infer_native_representation(input_convention),
        )

    def __repr__(self) -> str:
        mod = self._flax_module
        return f"MPS(n={mod.n_sites}, bond={mod.bond_dim}, d={mod.phys_dim})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
