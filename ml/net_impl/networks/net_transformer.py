"""
general_python.ml.net_impl.networks.net_transformer
===================================================

Generic transformer-style wrapper for flattened inputs.

The wrapper tokenizes a regular 1D layout into patches and applies standard
self-attention blocks over patch embeddings.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-21
License         : MIT
----------------------------------------------------------
"""

import jax
import jax.numpy    as jnp
import flax.linen   as nn
import numpy as np
from typing         import Sequence, Any, Optional, Tuple, Callable

try:
    from ....ml.net_impl.interface_net_flax import FlaxInterface
    from ....ml.net_impl.utils.net_wrapper_utils import configure_nqs_metadata
    JAX_AVAILABLE = True
except ImportError:
    raise ImportError("Transformer requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Transformer Components
# ----------------------------------------------------------------------

class MLPBlock(nn.Module):
    """Transformer feed-forward block."""
    hidden_dim  : int
    out_dim     : int
    dtype       : Any
    dropout     : float = 0.0 # Not used in VMC typically, but kept for structure

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x

class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block."""
    num_heads   : int
    hidden_dim  : int
    mlp_dim     : int
    dtype       : Any

    @nn.compact
    def __call__(self, x):
        # Attention Block
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.MultiHeadAttention(
            num_heads       =   self.num_heads,
            dtype           =   self.dtype,
            kernel_init     =   nn.initializers.xavier_uniform(),
            deterministic   =   True # VMC is deterministic for a given sample batch
        )(y, y)
        x = x + y

        # MLP Block
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MLPBlock(hidden_dim=self.mlp_dim, out_dim=self.hidden_dim, dtype=self.dtype)(y)
        x = x + y
        
        return x

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxTransformer(nn.Module):
    """Patch-based Transformer backbone for flat inputs."""
    n_sites         : int
    patch_size      : int   # Or tuple for 2D, handled simplified here as int divisor
    embed_dim       : int
    depth           : int
    num_heads       : int
    mlp_ratio       : float = 2.0
    dtype           : Any   = jnp.complex128
    
    def setup(self):
        self.num_patches = self.n_sites // self.patch_size
        
        # Ensure n_sites is divisible by patch_size
        if self.n_sites % self.patch_size != 0:
            raise ValueError(f"n_sites ({self.n_sites}) must be divisible by patch_size ({self.patch_size})")
            
        self.pos_embedding  = self.param('pos_embedding', 
                                        nn.initializers.normal(stddev=0.02), 
                                        (1, self.num_patches + 1, self.embed_dim),
                                        self.dtype)
        
        self.cls_token      = self.param('cls_token', 
                                        nn.initializers.zeros, 
                                        (1, 1, self.embed_dim),
                                        self.dtype)
        
        self.blocks         = [
                                EncoderBlock(
                                    num_heads=self.num_heads,
                                    hidden_dim=self.embed_dim,
                                    mlp_dim=int(self.embed_dim * self.mlp_ratio),
                                    dtype=self.dtype
                                ) for _ in range(self.depth)
                            ]
        
        self.norm           = nn.LayerNorm(dtype=self.dtype)
        self.patch_proj     = nn.Dense(self.embed_dim, dtype=self.dtype)
        self.head           = nn.Dense(1, dtype=self.dtype)

    @nn.compact
    def __call__(self, x):
        """Evaluate the Transformer on a single sample or a batch."""
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
        batch_size          = x.shape[0]
        
        # 1. Patch Embedding
        # Reshape to (batch, num_patches, patch_size)
        x_patches           = x.reshape((batch_size, self.num_patches, self.patch_size))
        
        # Linear projection of flattened patches
        x                   = self.patch_proj(x_patches)
        
        # 2. Add CLS token
        cls_token           = jnp.tile(self.cls_token, (batch_size, 1, 1))
        x                   = jnp.concatenate([cls_token, x], axis=1) # (batch, num_patches + 1, embed_dim)
        
        # 3. Add Position Embedding
        x                   = x + self.pos_embedding
        
        # 4. Transformer Encoder
        for block in self.blocks:
            x               = block(x)
            
        x                   = self.norm(x)
        
        # 5. Output Head (from CLS token)
        cls_out             = x[:, 0]
        out                 = self.head(cls_out)
        
        return out.squeeze(-1) # (batch,)

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class Transformer(FlaxInterface):
    """
    Transformer wrapper for flattened regular inputs.

    Parameters:
        input_shape (tuple):
            Shape of the flattened input.
        patch_size (int):
            Size of patches to divide the system into.
        embed_dim (int):
            Embedding dimension.
        depth (int):
            Number of Transformer blocks.
        num_heads (int):
            Number of attention heads.
        mlp_ratio (float):
            Ratio of MLP hidden dim to embedding dim.
    """
    def __init__(self,
                input_shape     : tuple,
                patch_size      : int   = 4,
                embed_dim       : int   = 32,
                depth           : int   = 2,
                num_heads       : int   = 4,
                mlp_ratio       : float = 2.0,
                dtype           : Any   = jnp.complex128,
                seed            : int   = 0,
                backend         : str   = 'jax',
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("Transformer requires JAX.")

        n_sites = int(np.prod(input_shape))

        net_kwargs = {
            'n_sites'       : n_sites,
            'patch_size'    : patch_size,
            'embed_dim'     : embed_dim,
            'depth'         : depth,
            'num_heads'     : num_heads,
            'mlp_ratio'     : mlp_ratio,
            'dtype'         : dtype
        }

        super().__init__(
            net_module  =   _FlaxTransformer,
            net_kwargs  =   net_kwargs,
            input_shape =   input_shape,
            backend     =   backend,
            dtype       =   dtype,
            seed        =   seed,
            **kwargs
        )
        self._name = 'transformer'
        configure_nqs_metadata(
            self,
            family="transformer",
        )

    def __repr__(self) -> str:
        mod = self._flax_module
        return f"Transformer(n={mod.n_sites}, patch={mod.patch_size}, dim={mod.embed_dim}, depth={mod.depth})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
