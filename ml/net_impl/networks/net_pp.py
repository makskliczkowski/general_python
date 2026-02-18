r"""
Pair Product (PP) and RBM+PP Ansatz for Quantum States.

This module provides Flax-based implementations of:
1.  **Pair Product (PP) Ansatz**:
    Captures pairwise correlations using a Pfaffian structure:
    $$ \psi_{PP}(s) = \text{Pf}(X(s)) $$ 
    where $X(s)$ is a skew-symmetric matrix dependent on configuration $s$.

2.  **RBM + PP Ansatz**:
    Combines a Restricted Boltzmann Machine (RBM) with the Pair Product state:
    $$ \psi(s) = \psi_{RBM}(s) \times \psi_{PP}(s) $$ 
    This ansatz is state-of-the-art for many fermionic and frustrated spin systems.

Performance & Precision:
------------------------
- **Mixed Precision**: 
    Supports storing parameters in lower precision (e.g., float32/complex64)
    while performing critical Pfaffian computations in high precision (complex128) for stability.
- **Efficient Gathering**: 
    Uses vectorized indexing to construct the $X$ matrix without explicit loops.
- **Algebraic Utilities**:
    Uses shared `Pfaffian` implementation from `general_python.algebra.utilities`.

Usage
-----
    from general_python.ml.networks import choose_network
    
    # Pure Pair Product
    net = choose_network('pp', input_shape=(64,), use_rbm=False)
    
    # RBM + Pair Product (Recommended)
    net = choose_network('pp', input_shape=(64,), use_rbm=True, alpha=2.0)

----------------------------------------------------------
File            : general_python.ml.net_impl.networks.net_pp
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 2025-11-01
----------------------------------------------------------
"""

from typing import Any, Optional
try:
    import  jax
    import  jax.numpy           as jnp
    import  flax.linen          as nn
except ImportError as e:
    raise ImportError("PairProduct network requires JAX and Flax.") from e

try:
    from ....ml.net_impl.interface_net_flax     import FlaxInterface
    from ....ml.net_impl.utils.net_init_jax     import cplx_variance_scaling
    from ....ml.net_impl.activation_functions   import log_cosh_jnp
    from ....algebra.utils                      import JAX_AVAILABLE
    
    # Import shared Pfaffian utility
    try:
        from ....algebra.utilities.pfaffian_jax import Pfaffian
    except ImportError:
        Pfaffian = None
        
except ImportError as e:
    raise ImportError("general_python core modules required.") from e

# ----------------------------------------------------------------------
# Logic for Log-Pfaffian
# ----------------------------------------------------------------------

def log_pfaffian_proxy(A):
    """
    Computes log(Pf(A)) using the shared Pfaffian utility if available.
    Falls back to a stable slogdet approximation if necessary.
    Supports batched inputs with shape (..., N, N).
    """
    if Pfaffian is not None and hasattr(Pfaffian, 'log_pfaffian'):
        return Pfaffian.log_pfaffian(A)
    
    # Fallback: 0.5 * log(det(A))
    # Approximation: log Pf(A) = 0.5 * (log |det(A)| + i * arg(det(A)))
    sign, logdet = jnp.linalg.slogdet(A)
    complex_dtype = jnp.result_type(A, jnp.complex64)
    return 0.5 * (logdet + jnp.log(sign.astype(complex_dtype)))

# ----------------------------------------------------------------------
# Inner Flax Modules
# ----------------------------------------------------------------------

class _FlaxPP(nn.Module):
    """
    Pair Product Ansatz (Pfaffian only).
    """
    n_sites     : int
    param_dtype : Any       = jnp.complex128
    dtype       : Any       = jnp.complex128  # Computation dtype
    init_scale  : float     = 0.01

    def setup(self):
        # Variational parameters F_{ij}^{\sigma_i \sigma_j}
        # Shape: (N, N, 2, 2)
        
        is_complex  = jnp.issubdtype(self.param_dtype, jnp.complexfloating)
        if is_complex:
            init_fn = cplx_variance_scaling(self.init_scale, "fan_in", "normal", self.param_dtype)
        else:
            init_fn = nn.initializers.normal(stddev=self.init_scale)
            
        self.F      = self.param('F', init_fn, (self.n_sites, self.n_sites, 2, 2), self.param_dtype)
        self._i_idx = jnp.arange(self.n_sites, dtype=jnp.int32)

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        # Ensure batch dimension
        needs_batch = s.ndim == 1
        if needs_batch:
            s = s[jnp.newaxis, :]
        
        # 1. Cast Input to Indices (0, 1)
        s_idx           = (s > 0).astype(jnp.int32)
        
        # 2. Cast Parameters to High Precision for Pfaffian
        F_high          = self.F if self.F.dtype == self.dtype else self.F.astype(self.dtype)
        F_sym           = F_high - F_high.transpose(1, 0, 3, 2)
        
        # 3. Construct X Matrices (Vectorized)
        i_idx = self._i_idx[None, :, None]  # (1, N, 1)
        j_idx = self._i_idx[None, None, :]  # (1, 1, N)
        X_batch = F_sym[i_idx, j_idx, s_idx[:, :, None], s_idx[:, None, :]]
        
        # 4. Compute Log Pfaffian
        out = log_pfaffian_proxy(X_batch)
        return out[0] if needs_batch else out

# ----------------------------------------------------------------------
# RBM + PP Module
# ----------------------------------------------------------------------

class _FlaxRBMPP(nn.Module):
    """
    Combined RBM + Pair Product Ansatz.
    log_psi(s) = log_psi_RBM(s) + log_psi_PP(s)
    """
    n_sites     : int
    n_hidden    : int       # For RBM
    param_dtype : Any       = jnp.complex128
    dtype       : Any       = jnp.complex128
    init_scale  : float     = 0.01
    
    def setup(self):
        # --- PP Part ---
        is_complex  = jnp.issubdtype(self.param_dtype, jnp.complexfloating)
        if is_complex:
            init_fn = cplx_variance_scaling(self.init_scale, "fan_in", "normal", self.param_dtype)
        else:
            init_fn = nn.initializers.normal(stddev=self.init_scale)
            
        self.F      = self.param('F', init_fn, (self.n_sites, self.n_sites, 2, 2), self.param_dtype)
        self._i_idx = jnp.arange(self.n_sites, dtype=jnp.int32)
        
        # --- RBM Part ---
        self.rbm_dense  = nn.Dense(
                            features    = self.n_hidden,
                            use_bias    = True,
                            dtype       = self.dtype,        # Compute dtype
                            param_dtype = self.param_dtype,  # Storage dtype
                            kernel_init = init_fn,
                            bias_init   = nn.initializers.zeros,
                            name        = "rbm_dense"
                        )
        self.vis_bias   = self.param('visible_bias', nn.initializers.zeros, (self.n_sites,), self.param_dtype)

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        needs_batch = s.ndim == 1
        if needs_batch:
            s = s[jnp.newaxis, :]
        
        # --- RBM Part ---
        v_rbm           = s.astype(self.dtype)
        theta           = self.rbm_dense(v_rbm)
        log_rbm         = jnp.sum(log_cosh_jnp(theta), axis=-1)
        log_rbm         = log_rbm + jnp.sum(v_rbm * self.vis_bias.astype(self.dtype), axis=-1)
        
        # --- PP Part ---
        s_idx           = (s > 0).astype(jnp.int32)
        F_high          = self.F if self.F.dtype == self.dtype else self.F.astype(self.dtype)
        F_sym           = F_high - F_high.transpose(1, 0, 3, 2)
        i_idx = self._i_idx[None, :, None]  # (1, N, 1)
        j_idx = self._i_idx[None, None, :]  # (1, 1, N)
        X_batch = F_sym[i_idx, j_idx, s_idx[:, :, None], s_idx[:, None, :]]
        log_pp          = log_pfaffian_proxy(X_batch)

        out = log_rbm + log_pp
        return out[0] if needs_batch else out

# ----------------------------------------------------------------------
# Wrapper Class
# ----------------------------------------------------------------------

class PairProduct(FlaxInterface):
    """
    Pair Product (PP) and RBM+PP Ansatz Interface.
    
    Parameters:
        input_shape (tuple): 
            Shape of input (n_sites,).
        use_rbm (bool):
            If True, adds an RBM layer (RBM+PP). Default True.
        alpha (float):
            RBM hidden unit density (n_hidden = alpha * n_sites). Used if use_rbm=True.
        dtype (Any): 
            Computation data type (e.g. complex128).
        param_dtype (Any): 
            Parameter storage type (e.g. complex64 for memory efficiency).
        init_scale (float):
            Scale for random initialization.
    """
    
    def __init__(self,
                input_shape : tuple,
                use_rbm     : bool          = True,
                alpha       : float         = 2.0,
                dtype       : Any           = jnp.complex128,
                param_dtype : Optional[Any] = None,
                init_scale  : float         = 0.01,
                seed        : int           = 0,
                **kwargs):
        
        if not JAX_AVAILABLE:
            raise ImportError("PairProduct requires JAX.")
            
        n_sites = input_shape[0]
        p_dtype = param_dtype if param_dtype is not None else dtype
        
        # Decide which Module to use
        if use_rbm:
            if alpha <= 0:
                raise ValueError("alpha must be > 0 when use_rbm=True")
            n_hidden = max(1, int(n_sites * alpha))
            net_module = _FlaxRBMPP
            net_kwargs = dict(
                                n_sites     = n_sites,
                                n_hidden    = n_hidden,
                                dtype       = dtype,
                                param_dtype = p_dtype,
                                init_scale  = init_scale
                            )
            name            = "rbm_pp"
        else:
            net_module      = _FlaxPP
            net_kwargs      = dict(
                                n_sites     = n_sites,
                                dtype       = dtype,
                                param_dtype = p_dtype,
                                init_scale  = init_scale
                            )
            name            = "pair_product"
        
        super().__init__(
            net_module  = net_module,
            net_args    = (),
            net_kwargs  = net_kwargs,
            input_shape = input_shape,
            dtype       = dtype,
            seed        = seed,
            **kwargs
        )
        
        self._name              = name
        self._has_analytic_grad = False

    def __repr__(self) -> str:
        mod = self._flax_module
        if isinstance(mod, _FlaxRBMPP):
            return f"RBMPP(n_sites={mod.n_sites}, n_hidden={mod.n_hidden}, dtype={self.dtype})"
        else:
            return f"PP(n_sites={mod.n_sites}, dtype={self.dtype})"
        
    def __str__(self) -> str:
        return self.__repr__()

# ----------------------------------------------------------------------
# End of File
# ----------------------------------------------------------------------
