r"""
Approximately Symmetric Ansatz Implementation

This module implements the "nonsymmetric χ -> symmetric Ω -> invariant \sigma" architecture pattern.
The idea is to have a flexible nonsymmetric block (chi) that learns a representation,
followed by a symmetrization block (Omega) that enforces physical invariances,
and finally an invariant nonlinearity (sigma) for the readout.

This architecture allows learning "unfattening" maps in the nonsymmetric block while
guaranteeing symmetry in the final wavefunction, as described in Kufel et al. (2025).

----------------------------------------------------------------
file        : general_python/ml/net_impl/networks/net_approx_symmetric.py
author      : Maksymilian Kliczkowski
date        : 2026-01-21
----------------------------------------------------------------
"""

import      jax
import      jax.numpy   as jnp
from        flax        import linen as nn
from        typing      import Any, Callable, Optional, Tuple, Sequence, List

try:
    from ...net_impl.interface_net_flax     import FlaxInterface
    from ...net_impl.activation_functions   import get_activation
except ImportError:
    raise ImportError("Required modules from general_python package are missing.")

# ----------------------------------------------------------------------

class ApproxSymmetricNet(nn.Module):
    """
    Flax module implementing the Approximately Symmetric Ansatz.

    Structure:
        x -> Chi(x) -> Omega(Chi(x)) -> Sigma(Omega(Chi(x)))

    Attributes:
        chi_block: 
            The nonsymmetric block (e.g., a Dense layer or MLP).
        symmetry_op:
            A callable that applies the symmetry operation (e.g., group averaging).
            Signature: (x: Array) -> Array
        readout_act: 
            The invariant nonlinearity (sigma).
        dtype: 
            The dtype for computation.
    """
    chi_features    : Sequence[int]
    symmetry_op     : Callable[[jnp.ndarray], jnp.ndarray]
    readout_act     : Callable[[jnp.ndarray], jnp.ndarray]  = jnp.log           # for real outputs
    chi_act         : Callable[[jnp.ndarray], jnp.ndarray]  = nn.relu           # for chi block
    dtype           : Any                                   = jnp.float32

    @nn.compact
    def __call__(self, x):
        
        # 1. Nonsymmetric Block (Chi)
        h = x.astype(self.dtype)

        # Simple MLP for Chi block
        for i, feat in enumerate(self.chi_features):
            h   = nn.Dense(feat, dtype=self.dtype, name=f'Chi_Dense_{i}')(h)
            h   = self.chi_act(h)

        # 2. Symmetric Block (Omega)
        # Apply the symmetry operation, e.g., group averaging
        # This assumes h has shape (batch, features) and symmetry_op acts on it.
        # The symmetry_op enforces invariance (e.g., pooling, group averaging).
        sym_h   = self.symmetry_op(h)

        # 3. Readout (Sigma)
        # Final projection to scalar if needed and invariant nonlinearity
        out     = nn.Dense(1, dtype=self.dtype, name='Readout')(sym_h)

        # Apply invariant nonlinearity
        out     = self.readout_act(out)

        # Squeeze to scalar per sample if needed (batch, 1) -> (batch,)
        return out.squeeze(-1)

class AnsatzApproxSymmetric(FlaxInterface):
    """
    Interface for the Approximately Symmetric Ansatz.

    Parameters:
        chi_features : Sequence[int]
            Hidden layer sizes for the nonsymmetric block.
        symmetry_op : Callable
            The symmetry enforcing operation.
            Should be a JAX-compatible function taking (batch, features) -> (batch, features).
            Default is identity if not provided.
        readout_act : str or Callable
            Activation function for the readout (invariant nonlinearity).
        chi_act : str or Callable
            Activation function for the chi block.
    """

    def __init__(self,
                chi_features    : Sequence[int] = (64, 64),
                symmetry_op     : Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
                readout_act     : Any           = 'log_cosh', # Common in NQS
                chi_act         : Any           = 'relu',
                input_shape     : tuple         = (10,),
                backend         : str           = 'jax',
                dtype           : Any           = jnp.complex128,
                seed            : int           = 42,
                **kwargs):

        # Resolve activations
        if isinstance(readout_act, str):
            readout_act, _      = get_activation(readout_act)
            
        if isinstance(chi_act, str):
            chi_act, _          = get_activation(chi_act)

        # Default symmetry op: Identity (if not provided)
        if symmetry_op is None:
            def identity_sym(x) : return x
            symmetry_op         = identity_sym

        net_kwargs = {
            'chi_features'      : chi_features,
            'symmetry_op'       : symmetry_op,
            'readout_act'       : readout_act,
            'chi_act'           : chi_act,
            'dtype'             : dtype
        }

        super().__init__(
            net_module          = ApproxSymmetricNet,
            net_kwargs          = net_kwargs,
            input_shape         = input_shape,
            backend             = backend,
            dtype               = dtype,
            seed                = seed,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"AnsatzApproxSymmetric(chi_features={self._net_kwargs_in['chi_features']}, dtype={self.dtype})"

# ----------------------------------------------------------------------
# Symmetry Operations Factories
# ----------------------------------------------------------------------

class GroupAveragingOp:
    """
    Applies group averaging in feature space.
    Assumes that the input x has shape (batch, features) and that
    the group action permutes or transforms these features.

    For a simpler NQS case where features represent spatial locations:
    If x is (batch, L) and G is translations, we compute 1/|G| sum_g Chi(g.x).

    However, in the 'Chi -> Omega' architecture, Omega acts on the OUTPUT of Chi.
    If Chi is an equivariant map (e.g. CNN), then the output features correspond to spatial positions.
    Then Omega can average over these positions.
    """
    def __init__(self, mode='mean'):
        self.mode = mode

    def __call__(self, x):
        # Assumes x is (batch, ..., sites)
        # We average over the last dimension (spatial/sites) to enforce translation invariance
        # if the preceding block (Chi) preserved spatial structure.
        if self.mode == 'mean':
            return jnp.mean(x, axis=-1, keepdims=True)
        elif self.mode == 'sum':
            return jnp.sum(x, axis=-1, keepdims=True)
        return x

def make_translation_symmetry_op(mode='mean'):
    """
    Returns a callable that averages/sums over the last dimension.
    Useful when Chi produces an equivariant feature map (batch, features).
    """
    op = GroupAveragingOp(mode=mode)
    return op

def make_permutation_symmetry_op(indices_list: List[List[int]]):
    """
    Creates a symmetry op that averages over explicit permutations of features.

    Args:
        indices_list: A list of permutations, where each permutation is a list of indices.
                      e.g. [[0, 1, 2], [1, 2, 0], [2, 0, 1]] for C3 on 3 sites.
    """
    indices_array = jnp.array(indices_list) # (n_sym, n_features)

    def op(x):
        # x: (batch, n_features)
        # We want to apply each permutation to x
        # x_permuted: (batch, n_sym, n_features)

        # Use vmap over symmetries? Or simple indexing if n_sym is small.
        # x[:, perm] gives the permuted batch.

        def apply_single_perm(perm):
            return x[:, perm]

        # all_perms: (n_sym, batch, n_features)
        all_perms = jax.vmap(apply_single_perm)(indices_array)

        # Average over n_sym
        # (n_sym, batch, n_features) -> (batch, n_features)
        return jnp.mean(all_perms, axis=0)

    return op

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------