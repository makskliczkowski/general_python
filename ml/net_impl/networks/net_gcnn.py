r"""
general_python.ml.net_impl.networks.net_gcnn
==============================================

Graph & Group-Equivariant Convolutional Neural Networks for NQS.

This module provides two architectures:

1. **GCNN** (Graph CNN):
   Adjacency-based message passing on arbitrary graphs.
   Optimized for non-Bravais lattices (Kagome, Honeycomb, random graphs).
   Uses Deep Sets pooling + Dense Head.

2. **EquivariantGCNN** (Group-Equivariant CNN):
    Convolutions over the symmetry group G (translations ⋊ point group).
    Enforces exact G-invariance of psi(sigma).  Based on:
    - Sharma, Manna, Rao, Sreejith, 
        - arXiv:2505.23728v1 (2025)
    - Roth & MacDonald, 
        - arXiv:2104.05085 (2021)
    - Cohen & Welling, 
        - arXiv:1602.07576 (2016)

Usage
-----
    # Graph-based GCNN
    gcnn = GCNN(input_shape=(N,), graph_edges=edges, split_complex=True)

    # Group-Equivariant GCNN (from lattice)
    net = EquivariantGCNN.from_lattice(lattice, channels=(8, 8))
"""

import  numpy   as np
from    typing  import Sequence, Callable, Optional, Any, Union, List, Tuple
import  functools

try:
    import jax
    import jax.numpy    as jnp
    import flax.linen   as nn
    from ....ml.net_impl.interface_net_flax     import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax     import cplx_variance_scaling
    from ....ml.net_impl.activation_functions   import get_activation_jnp
    JAX_AVAILABLE = True
except ImportError:
    # Minimal fallback for standalone testing
    JAX_AVAILABLE = False

# ----------------------------------------------------------------------
# 1. Optimized Graph Layer
# ----------------------------------------------------------------------

class GraphConv(nn.Module):
    r"""
    Decoupled Graph Convolution.
    Separates self-interaction from neighbor-interaction for better physical expressivity.
    
    Update rule:
        h_i' = Act( W_self * h_i  +  W_neigh * \sum_{j \in N(i)} h_j + b )
    """
    features    : int
    dtype       : Any
    param_dtype : Any
    kernel_init : Callable  
    use_bias    : bool      = False

    @nn.compact
    def __call__(self, x, adj_mat):
        # 1. Self Projection
        w_self      = nn.Dense(self.features, use_bias=self.use_bias, kernel_init=self.kernel_init, name='W_self')
        self_feat   = w_self(x)
        
        # 2. Neighbor Aggregation (Normalized is safer)
        # We normalize by degree inside the call to be sure
        deg         = jnp.sum(jnp.abs(adj_mat), axis=1, keepdims=True)
        adj_norm    = adj_mat / jnp.maximum(deg, 1.0)
        
        neigh_msg   = jnp.einsum('nm,bmc->bnc', adj_norm, x)
        w_neigh     = nn.Dense(self.features, use_bias=self.use_bias, kernel_init=self.kernel_init, name='W_neigh')
        neigh_feat  = w_neigh(neigh_msg)
        
        # 3. Combine + Bias
        out         = self_feat + neigh_feat
        bias        = self.param('bias', nn.initializers.zeros, (self.features,), self.param_dtype)
        return out + bias

# ----------------------------------------------------------------------
# 2. Backbone Module
# ----------------------------------------------------------------------

class _FlaxGCNN(nn.Module):
    """
    GCNN Backbone with Deep Sets Aggregation.
    """
    adj_matrix      : jnp.ndarray           # Frozen Adjacency
    features        : Sequence[int]
    activations     : Sequence[Callable]
    use_bias        : bool
    dtype           : Any
    param_dtype     : Any
    split_complex   : bool
    input_trans     : bool                  # Transform 0/1 -> -1/1
    islog           : bool

    @nn.compact
    def __call__(self, x):
        # x shape: (Batch, N_sites) or (Batch, N_sites, 1)
        
        # Preprocessing Input
        if x.ndim == 2:
            x = x[:, :, jnp.newaxis]
        elif x.ndim == 1:
            # Handle single-sample inputs if necessary
            x = x[jnp.newaxis, :, jnp.newaxis]
        batch_size = x.shape[0]

        # Resolve Dtypes for Split-Complex Mode
        if self.split_complex:
            comp_dtype = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
            if jnp.iscomplexobj(x): x = x.real
        else:
            comp_dtype = self.dtype
        
        # Cast Input to computation dtype
        x = x.astype(comp_dtype)
        
        if self.input_trans:
            x = 2.0 * x - 1.0

        # Init Strategy
        if jnp.issubdtype(comp_dtype, jnp.complexfloating):
            k_init = cplx_variance_scaling(1.0, 'fan_in', 'normal', comp_dtype)
        else:
            k_init = nn.initializers.lecun_normal(dtype=comp_dtype)

        # 2. GCN Layers (Feature Extraction)
        for i, (feat, act) in enumerate(zip(self.features, self.activations)):
            residual    = x
            x           = GraphConv(
                            features        =feat, 
                            dtype           =comp_dtype, 
                            param_dtype     =comp_dtype, 
                            kernel_init     =k_init,
                            name            =f'gconv_{i}'
                        )(x, self.adj_matrix)
            
            x           = act[0](x)
            
            # If dimensions match, add the skip connection
            if residual.shape[-1] == x.shape[-1]:
                x = (x + residual) / jnp.sqrt(2.0)

        # 3. Deep Sets Pooling
        # Mean over NODES (axis 1), preserving FEATURES (axis 2)
        # x: (Batch, N_nodes, Features) -> (Batch, Features)
        # This creates a global feature vector describing the whole system state.
        x_pool      = jnp.mean(x, axis=1)
        head_init   = nn.initializers.normal(stddev=0.1, dtype=comp_dtype)
        
        # Output Head (Mixing)
        # Projects the global features to the final wavefunction amplitude
        if self.split_complex:
            # Output 2 Real numbers [Re, Im]
            out = nn.Dense(
                features    = 2,
                dtype       = comp_dtype,
                param_dtype = comp_dtype,
                kernel_init = head_init,
                bias_init   = nn.initializers.zeros,
                name        = 'head_proj'
            )(x_pool)
            val = out[..., 0] + 1j * out[..., 1]
        else:
            # Output 1 Complex number
            out = nn.Dense(
                features    = 1,
                dtype       = comp_dtype,
                param_dtype = comp_dtype,
                kernel_init = head_init,
                bias_init   = nn.initializers.zeros,
                name        = 'head_proj'
            )(x_pool)
            val = out[..., 0]

        # Final Activation
        return val if self.islog else jnp.exp(val)

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class GCNN(FlaxInterface):
    """
    GCNN Interface for NQS on arbitrary graphs.
    """
    def __init__(self,
                input_shape    : tuple,
                *,
                graph_edges    : Optional[List[Tuple[int, int]]]    = None,
                adj_matrix     : Optional[np.ndarray]               = None,
                features       : Sequence[int]                      = (16, 32),
                activations    : Union[str, Sequence]               = 'log_cosh',
                use_bias       : bool                               = True,
                split_complex  : bool                               = False,
                transform_input: bool                               = False,
                normalize_adj  : bool                               = True,
                dtype          : Any                                = 'complex128',
                seed           : int                                = 0,
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("JAX/Flax missing.")

        # Graph Construction
        n_sites     = input_shape[0]
        
        if adj_matrix is not None:
            adj     = np.array(adj_matrix)
            
        elif graph_edges is not None:
            adj     = np.zeros((n_sites, n_sites), dtype=np.float32)
            # Add self-loops implicitly or explicitly? 
            # GraphConv handles self-loops via W_self, so A usually 0 diagonal.
            for u, v in graph_edges:
                adj[u, v]   = 1.0
                adj[v, u]   = 1.0
        else:
            raise ValueError("Must provide graph_edges or adj_matrix")

        # Optional: Symmetric Normalization (D^-0.5 A D^-0.5)
        # Helps training stability for deep GCNs
        if normalize_adj:
            deg                     = np.sum(adj, axis=1)
            deg_inv_sqrt            = np.power(np.maximum(deg, 1e-12), -0.5)
            deg_inv_sqrt[deg == 0]  = 0
            # D^-0.5 * A * D^-0.5
            adj                     = adj * deg_inv_sqrt[:, None]
            adj                     = adj * deg_inv_sqrt[None, :]

        # Configuration
        if isinstance(activations, str):
            acts            = [get_activation_jnp(activations)] * len(features)
        else:
            acts            = [get_activation_jnp(a) for a in activations]
            
        jax_dtype           = getattr(jnp, dtype) if isinstance(dtype, str) else dtype

        net_kwargs          = {
                                'adj_matrix'    : jnp.array(adj), # Freeze graph structure
                                'features'      : features,
                                'activations'   : acts,
                                'use_bias'      : use_bias,
                                'dtype'         : jax_dtype,
                                'param_dtype'   : jax_dtype,
                                'split_complex' : split_complex,
                                'input_trans'   : transform_input,
                                'islog'         : True
                            }

        self._split_complex = split_complex

        super().__init__(
                net_module  = _FlaxGCNN,
                net_args    = (),
                net_kwargs  = net_kwargs,
                input_shape = input_shape,
                backend     = kwargs.pop('backend', 'jax'),
                dtype       = jax_dtype,
                seed        = seed,
                **kwargs
            )
        
        self._has_analytic_grad     = False
        self._name                  = 'gcnn'

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._out_shape

    def __repr__(self):
        return f"GCNN(sites={self.input_dim}, features={self._flax_module.features}, split_complex={self._flax_module.split_complex})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __call__(self, x):
        
        flat_output = super().__call__(x)
        
        if flat_output.ndim == 0:       # Single scalar
            return flat_output
        elif flat_output.ndim == 1:     # Batch of scalars
            return flat_output
        else:
            # Flatten to (batch_size,) or scalar
            return flat_output.reshape(-1)[:x.shape[0] if x.ndim > 1 else 1]
        
# ----------------------------------------------------------------------
# 4. Group-Equivariant GCNN (arXiv:2505.23728)
# ----------------------------------------------------------------------

if JAX_AVAILABLE:

    class _FlaxEquivariantGCNN(nn.Module):
        r"""
        Group-Equivariant CNN backbone for NQS
        (Sharma et al. 2025, arXiv:2505.23728).

        1. **Embedding**:  $\sigma \to f^1(g)$
           $$f^1_{g,\alpha} = \Gamma\!\bigl(\sum_r (g^{-1}\sigma)_r K^\alpha_r + b^{1,\alpha}\bigr)$$
        2. **Group convolution** ($\ell = 1 \ldots \mathcal{L}-1$):
           $$f^{\ell+1}_{g,\beta} = \Gamma\!\bigl(\sum_{h}\sum_\alpha W^{\ell+1}_{h^{-1}g,\alpha\beta} f^\ell_{h,\alpha} + b^{\ell+1,\beta}\bigr)$$
        3. **Irrep projection** (trivial irrep, $\chi_g=1$):
           $$\psi(\sigma) = \exp\!\bigl(\sum_g\sum_\alpha f^{\mathcal{L}}_{g,\alpha}\bigr)$$

        Activation: complex SELU.
        """
        perm_table      : jnp.ndarray           # (|G|, Ns)   — frozen
        cayley_table    : jnp.ndarray           # (|G|, |G|)  — frozen
        channels        : Sequence[int]
        dtype           : Any
        param_dtype     : Any
        split_complex   : bool
        islog           : bool

        @nn.compact
        def __call__(self, x):
            if x.ndim == 1:
                x = x[jnp.newaxis, :]

            n_group = self.perm_table.shape[0]
            Ns      = x.shape[-1]

            # Resolve computation dtype
            if self.split_complex:
                cd = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
                if jnp.iscomplexobj(x):
                    x = x.real
            else:
                cd = self.dtype
                
            x       = x.astype(cd)
            k_init  = (cplx_variance_scaling(1.0, 'fan_in', 'normal', cd)
                        if jnp.issubdtype(cd, jnp.complexfloating)
                        else nn.initializers.lecun_normal(dtype=cd))
            is_cplx = jnp.issubdtype(cd, jnp.complexfloating)

            def _act(h):
                return (jax.nn.selu(h.real) + 1j * jax.nn.selu(h.imag)) if is_cplx else jax.nn.selu(h)

            # Embedding layer
            x_perm  = x[:, self.perm_table]                         # (B, |G|, Ns)
            c0      = self.channels[0]
            K_emb   = self.param('K_embed', k_init, (Ns, c0), cd)
            b_emb   = self.param('b_embed', nn.initializers.zeros, (c0,), cd)
            f       = _act(jnp.einsum('bgn,nc->bgc', x_perm, K_emb) + b_emb)

            # Group convolution layers
            for l in range(len(self.channels) - 1):
                c_in, c_out = self.channels[l], self.channels[l + 1]
                W           = self.param(f'W_gc_{l}', k_init, (n_group, c_in, c_out), cd)
                b           = self.param(f'b_gc_{l}', nn.initializers.zeros, (c_out,), cd)

                # W_idx[h, g] = W[cayley[h, g]]  →  (|G|, |G|, c_in, c_out)
                W_idx       = W[self.cayley_table.reshape(-1)].reshape(n_group, n_group, c_in, c_out,)
                # f_new[b,g,o] = Σ_h Σ_i f[b,h,i] · W_idx[h,g,i,o]
                f           = _act(jnp.einsum('bhi,hgio->bgo', f, W_idx) + b)

            # Output: trivial-irrep projection
            if self.split_complex:
                f_pool      = jnp.sum(f, axis=1) # (B, c_L)
                head_init   = nn.initializers.normal(stddev=0.1, dtype=cd)
                out         = nn.Dense(2, dtype=cd, param_dtype=cd,
                               kernel_init=head_init, name='head_proj')(f_pool)
                val         = out[..., 0] + 1j * out[..., 1]
            else:
                val         = jnp.sum(f, axis=(1, 2)) # (B,)

            return val if self.islog else jnp.exp(val)

    class EquivariantGCNN(FlaxInterface):
        r"""
        Group-Equivariant CNN for NQS on lattice systems.

        Implements the architecture from Sharma et al. (2025, arXiv:2505.23728)
        and Roth & MacDonald (2021).  Convolutions run over the symmetry group
        $G$ (translations $\rtimes$ point group), guaranteeing *exact*
        $G$-invariance of $\psi(\sigma)$.

        **Key results from arXiv:2505.23728**:

        * $\mathcal{L}=2$ layers achieve fidelity $> 0.99999$ with ED (L=8).
        * Excellent agreement with QMC for $12 \le L \le 32$.
        * QGT rank saturates → more layers don't improve (sampling-limited).

        Usage
        -----
        >>> net = EquivariantGCNN.from_lattice(lattice, channels=(8, 8))
        >>> from QES.general_python.lattices.tools.lattice_symmetry import generate_space_group_perms
        >>> perms = generate_space_group_perms(Lx, Ly, point_group='full')
        >>> net = EquivariantGCNN(input_shape=(Ns,), symmetry_perms=perms)
        """

        def __init__(
            self,
            input_shape     : tuple,
            *,
            symmetry_perms  : "np.ndarray",
            cayley_table    : Optional["np.ndarray"] = None,
            channels        : Sequence[int]         = (8, 8),
            split_complex   : bool                  = False,
            dtype           : Any                   = "complex128",
            seed            : int                   = 0,
            **kwargs,
        ):
            from ....lattices.tools.lattice_symmetry import compute_cayley_table as _cayley

            perm_table = np.asarray(symmetry_perms, dtype=np.int32)
            if cayley_table is None:
                cayley_table = _cayley(perm_table)
            cayley_table = np.asarray(cayley_table, dtype=np.int32)

            jax_dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype

            net_kwargs = dict(
                perm_table      = jnp.array(perm_table),
                cayley_table    = jnp.array(cayley_table),
                channels        = tuple(channels),
                dtype           = jax_dtype,
                param_dtype     = jax_dtype,
                split_complex   = split_complex,
                islog           = True,
            )

            super().__init__(
                net_module  = _FlaxEquivariantGCNN,
                net_args    = (),
                net_kwargs  = net_kwargs,
                input_shape = input_shape,
                backend     = kwargs.pop('backend', 'jax'),
                dtype       = jax_dtype,
                seed        = seed,
                **kwargs,
            )
            self._has_analytic_grad = False
            self._name      = 'eqgcnn'
            self._n_group   = perm_table.shape[0]

        @classmethod
        def from_lattice(
            cls,
            lattice,
            *,
            channels        : Sequence[int] = (8, 8),
            point_group     : str           = "full",
            dtype           : Any           = "complex128",
            **kwargs,
        ):
            """
            Create from a lattice object (recommended entry point).

            Automatically generates symmetry permutations from geometry.

            Parameters
            ----------
            lattice : Lattice
                Must expose ``.Lx``, ``.Ly``, ``.ns``.
            channels : tuple of int
                Channel widths per layer (length = number of layers).
            point_group : str
                ``'full'`` (maximal) or ``'translations'`` only.
            """
            from ....lattices.tools.lattice_symmetry import generate_space_group_perms

            Lx, Ly  = lattice.Lx, lattice.Ly
            Ns      = lattice.ns
            spc     = Ns // (Lx * Ly)              # sites per unit cell
            perms   = generate_space_group_perms(Lx, Ly, spc, point_group)
            return cls(
                input_shape     = (Ns,),
                symmetry_perms  = perms,
                channels        = channels,
                dtype           = dtype,
                **kwargs,
            )

        @property
        def output_shape(self) -> Tuple[int, ...]:
            return self._out_shape

        @property
        def group_size(self) -> int:
            """Order of the symmetry group |G|."""
            return self._n_group

        def __repr__(self):
            ch = self._flax_module.channels
            return (f"EquivariantGCNN(sites={self.input_dim},"
                    f"|G|={self._n_group},channels={ch})")

        def __str__(self):
            return self.__repr__()

        def __call__(self, x):
            out = super().__call__(x)
            if out.ndim <= 1:
                return out
            return out.reshape(-1)[:x.shape[0] if x.ndim > 1 else 1]

# ----------------------------------------------------------------------
# End of File
# ----------------------------------------------------------------------