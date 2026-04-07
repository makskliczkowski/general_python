"""
general_python.ml.net_impl.networks.net_rbm
===============================================

Restricted Boltzmann Machine (RBM) wrapper based on Flax.

This module provides a generic RBM implementation with explicit support for
input preprocessing, real or complex parameters, and optional fast-update
utilities for local proposal rules.

Usage
-----
Import and use the RBM network:

    from general_python.ml.networks import RBM
    rbm_net = RBM(input_shape=(10,), n_hidden=20)

The implementation is wrapped in `FlaxInterface` so it can be used directly or
through higher-level factory code.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Email           : maxgrom97@gmail.com
Date            : 01.04.2025
Description     : Flax implementation of Restricted Boltzmann Machines (RBMs).
----------------------------------------------------------
"""

import  numpy           as np
from    typing          import Tuple, Callable, Optional, Any
from    functools       import partial

try:
    # Base Interface (essential)
    from ....ml.net_impl.interface_net_flax         import FlaxInterface
    from ....ml.net_impl.activation_functions       import log_cosh_jnp
    from ....ml.net_impl.utils.net_init_jax         import cplx_variance_scaling, lecun_normal
    from ....ml.net_impl.utils.net_wrapper_utils    import configure_nqs_metadata
    JAX_AVAILABLE = True
except ImportError as e:
    raise ImportError("RBM requires JAX/Flax and general_python modules.") from e

# JAX / Flax Imports
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
else:
    raise ImportError("JAX is not available. Please install JAX to use this module.")

##########################################################
#! INNER FLAX RBM MODULES
##########################################################

class _FlaxRBMBase(nn.Module):
    r"""
    Base Flax module for a Restricted Boltzmann Machine.

    Designed to be wrapped by `FlaxInterface`. Calculates the log-amplitude
    log(psi(s)) based on RBM formula.
    """
    n_visible           : int
    n_hidden            : int
    bias                : bool                  = True                  # Bias for hidden units is common in RBMs
    input_activation    : Optional[Callable]    = None                  # e.g., lambda x: 2*x-1
    visible_bias        : Optional[bool]        = True
    param_dtype         : jnp.dtype             = jnp.complex64         # Default to complex
    dtype               : jnp.dtype             = jnp.complex64         # Default to complex
    islog               : bool                  = True                  # Logarithmic form of the wavefunction

    def _kernel_init(self):
        raise NotImplementedError

    def setup(self):
        """Construct the visible-to-hidden map and optional visible bias."""
        self.dense = nn.Dense(
            features    =   self.n_hidden,
            use_bias    =   self.bias,
            kernel_init =   self._kernel_init(),
            bias_init   =   nn.initializers.zeros,
            dtype       =   self.dtype,
            param_dtype =   self.param_dtype,
            name        =   "VisibleToHidden"
        )

        if self.visible_bias:
            self.visible_bias_param = self.param("visible_bias", nn.initializers.zeros, (self.n_visible,), self.param_dtype)
    
    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        r"""
        Formula: log(psi(s)) = sum_j log(cosh( W_j * v + b_j )) + a * v
        
        Args:
            s (jax.Array): Input configuration(s) with shape (batch, n_visible) or (n_visible,).

        Returns:
            jax.Array: Log-amplitude(s) log(psi(s)) with shape (batch,) or scalar.
        """
        
        # Fast path: avoid branching for common case
        needs_batch = s.ndim == 1
        v           = s[jnp.newaxis, :] if needs_batch else s
        
        # Fused transformation: cast + activation in one step
        if self.input_activation is not None:
            v = self.input_activation(jnp.asarray(v, dtype=self.dtype))
        else:
            v = jnp.asarray(v, dtype=self.dtype)

        # Dense layer (already JIT-compiled by Flax)
        theta   = self.dense(v)
        
        # Fast log_cosh + reduction
        log_psi = jnp.sum(log_cosh_jnp(theta), axis=-1)

        # Visible bias term (if enabled)
        if self.visible_bias:
            log_psi += jnp.sum(v * self.visible_bias_param, axis=-1)

        # Return scalar for single sample, array for batch
        result = log_psi if self.islog else jnp.exp(log_psi)
        return result[0] if needs_batch else result


class _FlaxRBMReal(_FlaxRBMBase):
    """Real-parameter RBM backbone."""

    def _kernel_init(self):
        return lecun_normal(self.param_dtype)


class _FlaxRBMComplex(_FlaxRBMBase):
    """Complex-parameter RBM backbone."""

    def _kernel_init(self):
        return cplx_variance_scaling(0.1, "fan_in", "normal", self.param_dtype)

##########################################################
#! RBM WRAPPER CLASSES USING FlaxInterface
##########################################################

class RBM(FlaxInterface):
    """
    Restricted Boltzmann Machine (RBM) based on FlaxInterface.

    Supports both real and complex RBMs by configuring the data types.

    Parameters:
        input_shape (tuple): 
            Input shape (excluding batch dimension), e.g. ``(n_sites,)``.
        n_hidden (int):
            Number of hidden units.
        bias (bool):
            Whether to include hidden bias terms.
        input_activation (Optional[Callable]):
            Optional preprocessing applied to visible states before the dense layer.
        dtype (jnp.dtype):
            Data type for computations.
        param_dtype (Optional[jnp.dtype]):
            Data type for parameters. Defaults to ``dtype``.
        seed (int):
            Initialization seed.
    """
    def __init__(self,
                input_shape         : tuple,
                n_hidden            : int                   = 2,
                alpha               : Optional[float]       = None,
                bias                : bool                  = True,
                visible_bias        : bool                  = True,
                input_activation    : Optional[Callable]    = None,
                proposal_update     : Optional[Callable]    = None,
                dtype               : Any                   = jnp.float32, # Default float for real RBM
                param_dtype         : Optional[Any]         = None,
                seed                : int                   = 0,
                **kwargs):
        '''
        Parameters
        ----------
        input_shape (tuple): 
            Shape of the input (excluding batch dimension).
        n_hidden (int):
            Number of hidden units in the RBM.
        bias (bool):
            Whether to include bias terms for hidden units.
        visible_bias (bool):
            Whether to include bias terms for visible units.
        input_activation (Optional[Callable]):
            Optional activation function applied to the input before the dense layer.
        proposal_update (Optional[Callable]):
            Optional callable used by ``log_psi_delta`` to map selected state values
            to their proposed updated values.
        '''
        

        if not JAX_AVAILABLE:
            raise ImportError("RBM requires JAX.")

        # Handle alpha if provided
        if alpha is not None:
            n_visible = np.prod(input_shape)
            n_hidden  = max(1, int(alpha * n_visible))

        # Determine dtypes
        final_dtype             = jnp.dtype(dtype)
        final_param_dtype       = jnp.dtype(param_dtype) if param_dtype is not None else final_dtype
        
        # Basic type compatibility check
        is_final_cpx            = jnp.issubdtype(final_dtype, jnp.complexfloating)
        self._is_cpx            = jnp.issubdtype(final_param_dtype, jnp.complexfloating)
        if is_final_cpx != self._is_cpx:
            self.log(f"Warning: RBM dtype ({final_dtype}) and param_dtype ({final_param_dtype}) differ in complexity. "
                    "Ensure this is intended.", log='warning', lvl=1, color='yellow')
    
        self._in_activation     = input_activation
        self._proposal_update   = proposal_update
        rbm_module              = _FlaxRBMComplex if self._is_cpx else _FlaxRBMReal

        # Prepare kwargs for the inner Flax module
        net_kwargs = {
            'n_visible'         : np.prod(input_shape), # Flatten input shape
            'n_hidden'          : n_hidden,
            'bias'              : bias,
            'visible_bias'      : visible_bias,
            'input_activation'  : input_activation,
            'param_dtype'       : final_param_dtype,
            'dtype'             : final_dtype,
            'islog'             : kwargs.get('islog', True)            
        }

        # Initialize using FlaxInterface parent
        super().__init__(
            net_module          = rbm_module,
            net_args            = (),
            net_kwargs          = net_kwargs,
            input_shape         = input_shape,
            backend             = 'jax',        # Force JAX backend
            dtype               = final_dtype,  # Pass the COMPUTATION dtype to FlaxInterface
            seed                = seed,
            input_activation    = input_activation,
        )
        self._in_activation = input_activation

        #! For the analytic gradient, we need to compile the function
        self._compiled_grad_fn                      = jax.jit(partial(RBM.analytic_grad_jax, input_activation=self._in_activation))
        self._compiled_log_psi_delta_cache_init_fn  = jax.jit(
            partial(RBM._init_log_psi_delta_cache_impl, input_activation=self._in_activation)
        )
        self._compiled_log_psi_delta_fns            = {}
        self._compiled_log_psi_delta_fn             = self._get_log_psi_delta_compiled(self._proposal_update)
        self._has_analytic_grad                     = False
        configure_nqs_metadata(self, family="rbm", supports_fast_updates=True)
        
        # Fast-update support is valid for flip-based rules only.
        self._fast_update_supported_rules           = frozenset({"LOCAL", "MULTI_FLIP", "BOND_FLIP", "WORM"})
        self._name                                  = 'rbm'

    # ------------------------------------------------------------
    #! Analytic Gradient
    # ------------------------------------------------------------
    
    @staticmethod
    @partial(jax.jit, static_argnames=("input_activation",))
    def analytic_grad_jax(params: Any, x: jax.Array, input_activation: Optional[Callable] = None) -> Any:
        r"""
        Computes the analytical gradient of log(psi(s)) for the RBM.

        Calculates the derivatives d log(psi)/dp for each sample in the batch (per-sample gradients),
        where p are the parameters (visible_bias, hidden_bias, weights).

        Gradient Formulas:
            d log(psi) / d a_i  = s_i
            d log(psi) / d b_j  = tanh(theta_j)
            d log(psi) / d W_ij = s_i * tanh(theta_j)
        where theta_j           = sum_i W_ij * s_i + b_j

        Args:
            params (Any):   
                PyTree of network parameters (matching _FlaxRBM structure).
                Expected keys: 'visible_bias', 'VisibleToHidden' {'kernel', 'bias'}.
            x (jax.Array):
                Input batch of configurations, shape (batch, n_visible) or (batch, *input_shape).
            input_activation (Optional[Callable]):
                The same activation function used in the forward pass.

        Returns:
            Any:
                A PyTree with the same structure as `params`, containing the
                gradients for each batch element. The leaves have shape (batch, ...).
        """
        
        #! Parameter Extraction
        has_visible_bias    = 'visible_bias' in params
        has_hidden_bias     = 'bias' in params.get('VisibleToHidden', {})

        W = params['VisibleToHidden']['kernel']                             # Shape (n_visible, n_hidden)
        b = params['VisibleToHidden']['bias'] if has_hidden_bias else None  # Shape (n_hidden,)
        a = params['visible_bias'] if has_visible_bias else None            # Shape (n_visible,)

        #! Input Preprocessing 
        # Ensure input has batch dimension and is flattened
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        # Apply the same input activation as the forward pass
        # Determine compute dtype from weights/biases if possible, fallback needed
        compute_dtype = W.dtype
        visible_state = x.astype(compute_dtype)
        if input_activation is not None:
            visible_state = input_activation(visible_state) # Shape (batch, n_visible)

        #! Intermediate Calculations
        # theta = s * W + b (using einsum for clarity, handles batching)
        # W: (n_vis, n_hid), visible_state: (batch, n_vis) -> theta: (batch, n_hid)
        theta = jnp.einsum('bi,ih->bh', visible_state, W)
        if b is not None:
            theta = theta + b

        tanh_theta = jnp.tanh(theta)                                        # Shape (batch, n_hid)

        #! Gradient Calculations (per batch element)
        # grad_a_batch = visible_state                                      # Shape (batch, n_visible)
        # grad_b_batch = tanh_theta                                         # Shape (batch, n_hidden)
        # grad_W_batch: Need outer product s_i * tanh(theta_j) for each batch item
        # visible_state: (b, i), tanh_theta: (b, j) -> grad_W: (b, i, j)
        grad_W_batch = jnp.einsum('bi,bj->bij', visible_state, tanh_theta)

        # --- Gradients per Batch ---
        # Construct the gradient tree matching params structure
        batch_grads = {}
        if a is not None:
            batch_grads['visible_bias'] = visible_state # grad_a_batch
        if W is not None: # Should always exist
            # Need to nest W and b correctly
            hidden_grads = {'kernel': grad_W_batch}
            if b is not None:
                hidden_grads['bias'] = tanh_theta # grad_b_batch
            batch_grads['VisibleToHidden'] = hidden_grads

        #! Return Gradient Tree
        # Ensure the structure matches the original params tree.
        # If a bias was missing in params, it shouldn't be in the grad tree.
        # The construction above handles this.
        return batch_grads

    # ------------------------------------------------------------
    #! Public Methods
    # ------------------------------------------------------------

    @staticmethod
    def _split_update_info(update_info: Any):
        """Split sampler update metadata into indices and an optional validity mask."""
        if isinstance(update_info, (tuple, list)) and len(update_info) == 2:
            return update_info[0], update_info[1]
        return update_info, None

    @staticmethod
    def _prepare_visible(state: jax.Array, dtype: Any, input_activation: Optional[Callable] = None):
        """Cast a visible configuration to the compute dtype and apply optional preprocessing."""
        visible = jnp.asarray(state, dtype=dtype)
        if input_activation is not None:
            visible = input_activation(visible)
        return visible

    def _get_log_psi_delta_compiled(self, proposal_update: Optional[Callable]):
        """Return a cached JIT-compiled fast-update kernel for the given proposal rule."""
        if proposal_update not in self._compiled_log_psi_delta_fns:
            self._compiled_log_psi_delta_fns[proposal_update] = jax.jit(
                partial(
                    RBM._log_psi_delta_impl,
                    input_activation=self._in_activation,
                    proposal_update=proposal_update,
                )
            )
        return self._compiled_log_psi_delta_fns[proposal_update]

    @staticmethod
    @partial(jax.jit, static_argnames=("input_activation",))
    def _init_log_psi_delta_cache_impl(
        params              : Any,
        states              : jax.Array,
        input_activation    : Optional[Callable] = None,
    ) -> jax.Array:
        """
        Precompute RBM hidden pre-activations theta = v @ W + b for fast updates.
        Supports both single-state and batched states.
        """
        W               = params["VisibleToHidden"]["kernel"]
        b               = params["VisibleToHidden"].get("bias", None)
        compute_dtype   = W.dtype

        single          = states.ndim == 1
        states_b        = states[jnp.newaxis, :] if single else states.reshape((-1, states.shape[-1]))
        v               = RBM._prepare_visible(states_b, compute_dtype, input_activation=input_activation)
        theta           = jnp.matmul(v, W)
        if b is not None:
            theta = theta + b
        if single:
            return theta[0]
        return theta.reshape(states.shape[:-1] + (theta.shape[-1],))

    @staticmethod
    def _log_psi_delta_single(
        params              : Any,
        state               : jax.Array,
        update_info         : Any,
        theta_old           : Optional[jax.Array] = None,
        input_activation    : Optional[Callable] = None,
        proposal_update     : Optional[Callable] = None,
    ):
        """
        Compute delta log(psi) from flip indices and optionally update theta cache.
        This function handles a single state and its corresponding update_info.
        """
        W               = params["VisibleToHidden"]["kernel"]
        b               = params["VisibleToHidden"].get("bias", None)
        a               = params.get("visible_bias", None)
        compute_dtype   = W.dtype

        idx_raw, valid_mask_raw = RBM._split_update_info(update_info)
        idx                     = jnp.asarray(idx_raw, dtype=jnp.int32).reshape(-1)
        if valid_mask_raw is None:
            valid = jnp.ones_like(idx, dtype=jnp.bool_)
        else:
            valid = jnp.asarray(valid_mask_raw, dtype=jnp.bool_).reshape(-1)
            valid = jnp.broadcast_to(valid, idx.shape)

        n_visible       = state.shape[0]
        valid           = valid & (idx >= 0) & (idx < n_visible)
        safe_idx        = jnp.where(valid, idx, 0)

        # Keep exactly one representative for indices flipped an odd number of times.
        m               = safe_idx.shape[0]
        row             = jnp.arange(m)[:, None]
        col             = jnp.arange(m)[None, :]
        same            = safe_idx[:, None] == safe_idx[None, :]
        valid_same      = same & valid[:, None] & valid[None, :]
        odd             = valid & ((jnp.sum(valid_same, axis=1) % 2) == 1)
        seen_before     = jnp.sum(valid_same & (col < row), axis=1) > 0
        take            = odd & (~seen_before)

        s_old_idx       = state[safe_idx]
        if proposal_update is None:
            raise ValueError("RBM fast updates require `proposal_update` to map selected values to proposed values.")
        s_new_idx       = proposal_update(s_old_idx)

        v_old_idx       = RBM._prepare_visible(s_old_idx, compute_dtype, input_activation=input_activation)
        v_new_idx       = RBM._prepare_visible(s_new_idx, compute_dtype, input_activation=input_activation)
        delta_v_idx     = (v_new_idx - v_old_idx) * take.astype(compute_dtype)

        if theta_old is None:
            v_full      = RBM._prepare_visible(state, compute_dtype, input_activation=input_activation)
            theta_old   = jnp.matmul(v_full, W)
            if b is not None:
                theta_old = theta_old + b

        w_rows          = W[safe_idx]  # (m, n_hidden)
        delta_theta     = jnp.sum(w_rows * delta_v_idx[:, None], axis=0)
        theta_new       = theta_old + delta_theta

        hidden_delta = jnp.sum(log_cosh_jnp(theta_new) - log_cosh_jnp(theta_old))
        if a is not None:
            visible_delta = jnp.sum(a[safe_idx] * delta_v_idx)
        else:
            visible_delta = jnp.asarray(0.0, dtype=hidden_delta.dtype)

        return hidden_delta + visible_delta, theta_new

    @staticmethod
    @partial(jax.jit, static_argnames=("input_activation", "proposal_update"))
    def _log_psi_delta_impl(
        params: Any,
        current_log_psi: jax.Array,
        state: jax.Array,
        update_info: Any,
        cache: Optional[jax.Array] = None,
        input_activation: Optional[Callable] = None,
        proposal_update: Optional[Callable] = None,
    ):
        """
        Delta log(psi) for proposed flip updates.
        If cache is provided, returns (delta, updated_cache).
        
        This function handles both single-state and batched states. For single states, it returns a scalar delta (and optionally the updated cache). 
        For batches, it returns an array of deltas (and optionally an array of updated caches).
        """
        del current_log_psi  # Kept for sampler API compatibility.

        if state.ndim == 1:
            delta, theta_new = RBM._log_psi_delta_single(
                params=params,
                state=state,
                update_info=update_info,
                theta_old=cache,
                input_activation=input_activation,
                proposal_update=proposal_update,
            )
            if cache is None:
                return delta
            return delta, theta_new

        if cache is None:
            deltas, _ = jax.vmap(
                lambda s, ui: RBM._log_psi_delta_single(
                    params=params,
                    state=s,
                    update_info=ui,
                    theta_old=None,
                    input_activation=input_activation,
                    proposal_update=proposal_update,
                ),
                in_axes=(0, 0),
            )(state, update_info)
            return deltas

        deltas, cache_new = jax.vmap(
            lambda s, ui, th: RBM._log_psi_delta_single(
                params=params,
                state=s,
                update_info=ui,
                theta_old=th,
                input_activation=input_activation,
                proposal_update=proposal_update,
            ),
            in_axes=(0, 0, 0),
        )(state, update_info, cache)
        return deltas, cache_new

    # ------------------------------------------------------------
    #! Public API for Fast Updates
    # ------------------------------------------------------------

    def init_log_psi_delta_cache(self, params: Any, states: jax.Array):
        """Precompute hidden pre-activations used by ``log_psi_delta``."""
        return self._compiled_log_psi_delta_cache_init_fn(params, states)

    def log_psi_delta(
        self,
        params: Any,
        current_log_psi: jax.Array,
        state: jax.Array,
        update_info: Any,
        cache: Optional[jax.Array] = None,
        proposal_update: Optional[Callable] = None,
    ) -> Any:
        """Evaluate the log-amplitude delta for a proposed local update."""
        eff_proposal_update = self._proposal_update if proposal_update is None else proposal_update
        if eff_proposal_update is None:
            raise ValueError("RBM.log_psi_delta requires `proposal_update` or an instance configured with one.")
        compiled_fn = self._compiled_log_psi_delta_fn
        if eff_proposal_update is not self._proposal_update:
            compiled_fn = self._get_log_psi_delta_compiled(eff_proposal_update)
        return compiled_fn(
            params,
            current_log_psi,
            state,
            update_info,
            cache,
            proposal_update=eff_proposal_update,
        )

    def get_log_psi_delta(self):
        """Return the fast-update callable expected by sampler code."""
        return self.log_psi_delta

    @property
    def fast_update_supported_rules(self):
        return self._fast_update_supported_rules

    # ------------------------------------------------------------
    #! Utility Methods
    # ------------------------------------------------------------ 

    def __repr__(self) -> str:
        init_status = "initialized"                             if self.initialized else "uninitialized"
        rbm_type    = "Complex"                                 if self._is_cpx else "Real"
        n_hidden    = self._flax_module.n_hidden                if self.initialized else self._net_kwargs.get('n_hidden', '?')
        bias        = "on" if (self._flax_module.bias           if self.initialized else self._net_kwargs.get('bias', '?')) else "off"
        vis_bias    = "on" if (self._flax_module.visible_bias   if self.initialized else self._net_kwargs.get('visible_bias', '?')) else "off"
        n_params    = self.nparams                              if self.initialized else '?'
        return (f"{rbm_type}RBM(shape={self.input_shape}, hidden={n_hidden}, "
            f"bias={bias}, visible_bias={vis_bias}, dtype={self.dtype}, params={n_params}, analytic_grad={self._has_analytic_grad}, {init_status})")
    
    def __str__(self) -> str:
        type_str    = "Complex" if self._is_cpx else "Real"
        return f"{type_str}RBM(shape={self.input_shape},hidden={self._flax_module.n_hidden if self.initialized else self._net_kwargs.get('n_hidden','?')},dtype={self.dtype})"
    
    # ------------------------------------------------------------

# ----------------------------------------------------------------
#! End of RBM Class
# ----------------------------------------------------------------
