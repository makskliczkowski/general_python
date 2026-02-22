"""
Stacked Symmetry-Improved Ansatz Implementation

This module implements a composable stacked ansatz:
Ansatz = (Weakly-Breaking Preblock) o (Exact-Symmetry Block) o (Readout)

It provides a configuration-driven way to define these stacks, allowing for
pluggable symmetry groups and architectures.

----------------------------------------------------------------
file        : general_python/ml/net_impl/networks/net_stacked.py
author      : Maksymilian Kliczkowski
date        : 2025-04-02
----------------------------------------------------------------
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Optional, Sequence, List, Dict, Union

try:
    from ...net_impl.interface_net_flax import FlaxInterface
    from ...net_impl.activation_functions import get_activation
    from ...net_impl.utils.net_init_jax import complex_he_init, real_he_init
except ImportError:
    raise ImportError("Required modules from general_python package are missing.")

# ----------------------------------------------------------------------
# Symmetry helpers (local to stacked API)
# ----------------------------------------------------------------------

class _GroupAveragingOp:
    def __init__(self, mode: str = "mean"):
        self.mode = mode

    def __call__(self, x):
        if self.mode == "mean":
            return jnp.mean(x, axis=-1, keepdims=True)
        if self.mode == "sum":
            return jnp.sum(x, axis=-1, keepdims=True)
        return x

def _make_permutation_symmetry_op(indices_list: Sequence[Sequence[int]]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    indices_array = jnp.asarray(indices_list, dtype=jnp.int32)

    def op(x):
        def apply_single_perm(perm):
            return x[:, perm]

        all_perms = jax.vmap(apply_single_perm)(indices_array)
        return jnp.mean(all_perms, axis=0)

    return op

# ----------------------------------------------------------------------
# Block Definitions
# ----------------------------------------------------------------------

class DenseBlock(nn.Module):
    features: int
    act: Callable = nn.relu
    dtype: Any = jnp.complex128

    @nn.compact
    def __call__(self, x):
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            kernel_init = complex_he_init
        else:
            kernel_init = real_he_init

        x = nn.Dense(self.features, dtype=self.dtype, kernel_init=kernel_init)(x)
        return self.act(x)

class IdentityBlock(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class ReadoutBlock(nn.Module):
    act: Callable = jnp.log
    output_dim: int = 1
    dtype: Any = jnp.complex128

    @nn.compact
    def __call__(self, x):
        # Assuming x is (batch, features)
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            kernel_init = complex_he_init
        else:
            kernel_init = real_he_init

        x = nn.Dense(self.output_dim, dtype=self.dtype, kernel_init=kernel_init)(x)
        x = self.act(x)
        return x.squeeze(-1)

# ----------------------------------------------------------------------
# The Stack Module
# ----------------------------------------------------------------------

class StackedNet(nn.Module):
    """
    A Flax module that executes a sequence of configured blocks.

    Supports registering custom block factories via `block_registry` in kwargs,
    though passing callables through Flax module configuration requires care.
    Here we rely on predefined blocks and dynamic instantiation.
    """
    blocks_config   : List[Dict[str, Any]]
    dtype           : Any = jnp.complex128

    @nn.compact
    def __call__(self, x):
        h = x.astype(self.dtype)

        for i, config in enumerate(self.blocks_config):
            block_type = config.get('type')
            block_args = config.get('args', {})

            # Instantiate Block based on type or callable
            if callable(block_type):
                # If block_type is a class or factory passed directly
                h           = block_type(**block_args)(h)

            elif block_type == 'Dense': # simple dense block with activation
                act_str     = block_args.get('act', 'relu')
                act_fn, _   = get_activation(act_str)
                features    = block_args.get('features', 64)
                h           = DenseBlock(features=features, act=act_fn, dtype=self.dtype)(h)

            elif block_type == 'Identity': # simple pass-through block
                h           = IdentityBlock()(h)

            elif block_type == 'Readout': # reads out to scalar with optional activation
                act_str     = block_args.get('act', 'log_cosh')
                act_fn, _   = get_activation(act_str)
                h           = ReadoutBlock(act=act_fn, dtype=self.dtype)(h)

            elif block_type == 'SymmetryGroup': # applies symmetric information to the ansatz
                # Use provided symmetry operation or factory
                op_callable = block_args.get('op')

                if op_callable is not None and callable(op_callable):
                    h       = op_callable(h)
                else:
                    # Fallback to simple modes if no callable provided
                    mode    = block_args.get('mode', 'identity')
                    indices = block_args.get('indices')

                    if indices is not None:
                        # Use permutation symmetry
                        sym_op  = _make_permutation_symmetry_op(indices)
                        h       = sym_op(h)
                    elif mode in ['mean', 'sum']:
                        # Use simple averaging
                        sym_op  = _GroupAveragingOp(mode=mode)
                        h       = sym_op(h)
                    else:
                        pass # Identity

            else:
                raise ValueError(f"Unknown block type: {block_type}")

        return h

# ----------------------------------------------------------------------
# The Interface
# ----------------------------------------------------------------------

class AnsatzStacked(FlaxInterface):
    """
    Interface for the Stacked Symmetry-Improved Ansatz.

    Allows for "pluggable" blocks by accepting callables in the config `type`
    or `SymmetryGroup` blocks with custom operations.

    Usage:
        def my_custom_sym_op(x): return jnp.sum(x, axis=-1, keepdims=True)

        config = [
            {'type': 'Dense', 'args': {'features': 128, 'act': 'relu'}},
            {'type': 'SymmetryGroup', 'args': {'op': my_custom_sym_op}},
            {'type': 'Readout', 'args': {'act': 'log_cosh'}}
        ]
        net = AnsatzStacked(stack_config=config, input_shape=(16,))
    """

    def __init__(self,
                 stack_config: List[Dict[str, Any]],
                 input_shape: tuple = (10,),
                 backend: str = 'jax',
                 dtype: Any = jnp.complex128,
                 seed: int = 42,
                 **kwargs):

        # Validate config basic structure
        if not isinstance(stack_config, list):
            raise ValueError("stack_config must be a list of dictionaries.")

        net_kwargs = {
            'blocks_config': stack_config,
            'dtype': dtype
        }

        super().__init__(
            net_module=StackedNet,
            net_kwargs=net_kwargs,
            input_shape=input_shape,
            backend=backend,
            dtype=dtype,
            seed=seed,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"AnsatzStacked(blocks={len(self._net_kwargs_in['blocks_config'])}, dtype={self.dtype})"
