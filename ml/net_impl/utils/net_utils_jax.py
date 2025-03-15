'''
file    : general_python/ml/net_impl/utils/net_utils_jax.py
author  : Maksymilian Kliczkowski
date    : 2025-03-01

'''

# from general python utils
from typing import Any, Callable, Dict, List, Tuple
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
from functools import partial

if _JAX_AVAILABLE:
    import jax
    from jax import grad
    from jax import numpy as jnp
    from jax.tree_util import tree_flatten, tree_unflatten, tree_map
    from jax.flatten_util import ravel_pytree

    # use flax
    import flax
    import flax.linen as nn
    from flax.core.frozen_dict import freeze, unfreeze

#########################################################################
#! BATCHES
#########################################################################

if _JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(1,))
    def create_batches_jax( data        : jnp.ndarray,
                            batch_size  : int):
        """ JAX version of create_batches """
        
        # For example, if data.shape[0] is 5 and batch_size is 3, then:
        #   ((5 + 3 - 1) // 3) =  (7 // 3) = 2 batches needed, so 2*3 = 6 samples in total.
        #   Then append_data = 6 - 5 = 1 extra sample needed.
        append_data = batch_size * ((data.shape[0] + batch_size - 1) // batch_size) - data.shape[0]
        
        # Create a list of padding widths.
        # First dimension: pad (0, append_data) so that we add 'append_data' rows at the end.
        # For the rest of the dimensions, pad with (0, 0) meaning no padding.
        pads        = [(0, append_data)] + [(0, 0)] * (len(data.shape) - 1)
        # Pad the array along the first dimension using 'edge' mode (repeats the last element),
        # then reshape the array into batches.
        # The reshape uses -1 to infer the number of batches, followed by the batch_size,
        # and then the remaining dimensions.
        return jnp.pad(data, pads, mode='edge').reshape(-1, batch_size, *data.shape[1:])

##########################################################################
#! EVALUATE BATCHED
##########################################################################

if _JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(0,1))
    def eval_batched_jax(batch_size : int,
                        func        : Any,
                        params      : Any,
                        data        : jnp.ndarray):
        """ JAX version of eval_batched """
        
        # Create batches of data
        batches = create_batches_jax(data, batch_size)
        
        def scan_fun(c, x):
            return c, jax.vmap(lambda y: func(params, y), in_axes=(0,))(x)

        # Evaluate the function on each batch using vmap
        return jax.lax.scan(scan_fun, None, jnp.array(batches))[1].reshape((-1,))
    
    @jax.jit
    def eval_batched_jax_simple(batch_size  : int,
                                func        : Any,
                                data        : jnp.ndarray):
        """
        JAX version of eval_batched without params.
        """
        # Create batches of data
        batches = create_batches_jax(data, batch_size)
        # Evaluate the function on each batch using vmap
        return jax.vmap(func)(batches).reshape((-1,))

###########################################################################
#! GRADIENTS
###########################################################################

_ERR_JAX_GRADIENTS_CALLABLE = "The function must be callable."

if _JAX_AVAILABLE:

    def flat_gradient_analytical_jax(func_analitical: Any, params, arg) -> jnp.ndarray:
        """
        Compute a flattened complex gradient using an analytical method (JAX version).
        
        This function assumes that 'fun' provides an attribute 'analytical_gradient' that 
        returns a pytree of gradients.
        
        Parameters
        ----------
        fun : object
            The network/function object. Must provide an analytical_gradient method.
        params : Any
            The network parameters (pytree).
        arg : Any
            The input state.
        
        Returns
        -------
        jnp.ndarray
            A single flattened complex gradient.
        
        Example
        -------
        >>> # Assume fun.analytical_gradient exists.
        >>> flat_grad = flat_gradient_analytical_jax(fun, params, state)
        """
                
        # Call the analytical gradient function.
        grad_val    = func_analitical(params, arg)
        # Flatten the gradient pytree: each leaf is reshaped to 1D.
        flat_grad   = tree_flatten(tree_map(lambda x: x.ravel(), grad_val))[0]
        # Concatenate all flattened arrays into one vector.
        return jnp.concatenate(flat_grad)

    def flat_gradient_numerical_jax(func: Any, params, arg) -> jnp.ndarray:
        """
        Compute a flattened complex gradient using numerical differentiation (JAX version).
        
        Uses jax.grad on both the real and imaginary parts of fun.apply.
        
        Parameters
        ----------
        fun : object
            The network/function object with an apply method.
        params : Any
            The network parameters (pytree).
        arg : Any
            The input state.
        
        Returns
        -------
        jnp.ndarray
            A flattened complex gradient.
        
        Example
        -------
        >>> flat_grad = flat_gradient_numerical_jax(fun, params, state)
        """
        # Compute gradient of the real part.
        gr  = grad(lambda p, y: jnp.sum(jnp.real(func(p, y))))(params, arg)["params"]
        gr  = tree_flatten(tree_map(lambda x: x.ravel(), gr))[0]
        # Compute gradient of the imaginary part.
        gi  = grad(lambda p, y: jnp.sum(jnp.imag(func(p, y))))(params, arg)["params"]
        gi  = tree_flatten(tree_map(lambda x: x.ravel(), gi))[0]
        # Concatenate and combine into a single complex vector.
        return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

    def flat_gradient_jax(func: Any, params: Any, arg: Any, analytical: bool = False) -> jnp.ndarray:
        """
        Wrapper for computing a flattened complex gradient using JAX.
        
        If analytical=True and an analytical gradient method exists, it is used;
        otherwise numerical differentiation is applied.
        
        Parameters
        ----------
        fun : object
            The network/function object.
        params : Any
            The network parameters.
        arg : Any
            The input state.
        analytical : bool, optional
            Whether to use the analytical gradient if available (default: False).
        
        Returns
        -------
        jnp.ndarray
            The flattened complex gradient.
        
        Example
        -------
        >>> grad_vec = flat_gradient_jax(fun, params, state, analytical=True)
        """
        if not callable(func):
            raise ValueError(_ERR_JAX_GRADIENTS_CALLABLE)
        if analytical:
            return flat_gradient_analytical_jax(func, params, arg)
        return flat_gradient_numerical_jax(func, params, arg)

    # -----------------------------------------------------------------------------
    #! Non-holomorphic Gradients: JAX
    # -----------------------------------------------------------------------------

    def flat_gradient_cpx_nonholo_analytical_jax(func_analitical: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute an analytical flattened complex gradient for non-holomorphic networks (JAX).

        Assumes fun returns the analytical gradient as a pytree.
        """
        if not callable(func_analitical):
            raise ValueError(_ERR_JAX_GRADIENTS_CALLABLE)
        grad_val    = func_analitical(params, arg)
        flat_grad   = tree_flatten(tree_map(lambda x: x.ravel(), grad_val))[0]
        return jnp.concatenate(flat_grad)

    def flat_gradient_cpx_nonholo_numerical_jax(func: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute a flattened complex gradient for non-holomorphic networks using numerical differentiation (JAX).

        Adjusts the sign of the imaginary part.
        """
        gr  = grad(lambda p, y: jnp.sum(jnp.real(func(p, y))))(params, arg)["params"]
        gr  = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gr))[0]
        gi  = grad(lambda p, y: jnp.sum(jnp.imag(func(p, y))))(params, arg)["params"]
        gi  = tree_flatten(tree_map(lambda x: [jnp.real(x.ravel()), -jnp.imag(x.ravel())], gi))[0]
        return jnp.concatenate(gr) + 1.j * jnp.concatenate(gi)

    def flat_gradient_cpx_nonholo_jax(func: Any, params: Any, arg: Any, analytical: bool = False) -> jnp.ndarray:
        """
        Wrapper for computing a flattened complex gradient for non-holomorphic networks using JAX.
        Parameters
        ----------
        fun : object
            The network/function object.
        params : Any
            The network parameters.
        arg : Any
            The input state.
        analytical : bool, optional
            If True, use the analytical gradient if available.
        Returns
        -------
        jnp.ndarray
            The flattened complex gradient.
        Example
        -------        
        """
        if analytical:
            return flat_gradient_cpx_nonholo_analytical_jax(func, params, arg)
        return flat_gradient_cpx_nonholo_numerical_jax(func, params, arg)

    # -----------------------------------------------------------------------------
    #! Real Gradients: JAX
    # -----------------------------------------------------------------------------

    def flat_gradient_real_analytical_jax(func: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute an analytical flattened real gradient (JAX).

        Assumes func provides an 'analytical_gradient_real' method.
        """
        if not callable(func):
            raise ValueError(_ERR_JAX_GRADIENTS_CALLABLE)
        grad_val    = func(params, arg).astype(jnp.float32)
        flat_grad   = tree_flatten(tree_map(lambda x: x.ravel(), grad_val))[0]
        return jnp.concatenate(flat_grad).astype(jnp.float32)

    def flat_gradient_real_numerical_jax(func: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute a flattened real gradient using numerical differentiation (JAX).
        """
        g = grad(lambda p, y: jnp.sum(jnp.real(func(p, y))))(params, arg)["params"]
        g = tree_flatten(tree_map(lambda x: x.ravel(), g))[0]
        return jnp.concatenate(g).astype(jnp.float32)

    def flat_gradient_real_jax(func: Any, params: Any, arg: Any, analytical: bool = False) -> jnp.ndarray:
        """
        Wrapper for computing a flattened real gradient using JAX.
        Parameters
        ----------
        fun : object
            The network/function object.        
        params : Any
            The network parameters.
        arg : Any
            The input state.
        analytical : bool, optional
            If True, use the analytical gradient if available.
        Returns
        ------- 
        jnp.ndarray
            The flattened real gradient.
        Example
        -------
        >>> grad_vec = flat_gradient_real_jax(fun, params, state, analytical=True)        
        """
        if analytical:
            return flat_gradient_real_analytical_jax(func, params, arg)
        return flat_gradient_real_numerical_jax(func, params, arg)

    # -----------------------------------------------------------------------------
    #! Holomorphic Gradients: JAX
    # -----------------------------------------------------------------------------

    def flat_gradient_holo_analytical_jax(func: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute an analytical flattened gradient for holomorphic networks (JAX).

        Assumes func provides an 'analytical_gradient_holo' method.
        """
        if not callable(func):
            raise ValueError("func must be callable.")
        grad_val    = func(params, arg)
        flat_grad   = tree_flatten(tree_map(lambda x: x.ravel(), grad_val))[0]
        return jnp.concatenate(flat_grad)

    def flat_gradient_holo_numerical_jax(func: Any, params: Any, arg: Any) -> jnp.ndarray:
        """
        Compute a flattened gradient for holomorphic networks using numerical differentiation (JAX).

        Each parameter's raveled value is repeated once with a multiplier of 1.j.
        """
        g = grad(lambda p, y: jnp.sum(jnp.real(func(p, y))))(params, arg)["params"]
        # Create a list with each flattened leaf repeated with an imaginary component.
        g = tree_flatten(tree_map(lambda x: [x.ravel(), 1.j * x.ravel()], g))[0]
        return jnp.concatenate(g)

    def flat_gradient_holo_jax(func: Any, params: Any, arg: Any, analytical: bool = False) -> jnp.ndarray:
        """
        Wrapper for computing a flattened gradient for holomorphic networks using JAX.
        """
        if analytical:
            return flat_gradient_holo_analytical_jax(func, params, arg)
        return flat_gradient_holo_numerical_jax(func, params, arg)

    # -----------------------------------------------------------------------------
    #! Dictionary of Gradients: JAX
    # -----------------------------------------------------------------------------

    def dict_gradient_analytical_jax(func: Any, params: Any, arg: Any) -> Any:
        """
        Wrapper for computing a dictionary of complex gradients using JAX.
        """
        if not callable(func):
            raise ValueError(_ERR_JAX_GRADIENTS_CALLABLE)
        return func(params, arg)

    def dict_gradient_numerical_jax(func: Any, params: Any, arg: Any) -> Any:
        """
        Compute a dictionary of complex gradients using numerical differentiation (JAX).
        """
        gr  = grad(lambda p, y: jnp.sum(jnp.real(func(p, y))))(params, arg)["params"]
        gr  = tree_map(lambda x: x.ravel(), gr)
        gi  = grad(lambda p, y: jnp.sum(jnp.imag(func(p, y))))(params, arg)["params"]
        gi  = tree_map(lambda x: x.ravel(), gi)
        return tree_map(lambda x, y: x + 1.j * y, gr, gi)

    def dict_gradient_jax(func: Any, params: Any, arg: Any, analytical: bool = False) -> Any:
        """
        Wrapper for computing a dictionary of complex gradients using JAX.
        """
        if analytical:
            return dict_gradient_analytical_jax(func, params, arg)
        return dict_gradient_numerical_jax(func, params, arg)

    def dict_gradient_real_analytical_jax(func: Any, params: Any, arg: Any) -> Any:
        """
        Compute an analytical dictionary of real gradients using JAX.

        Assumes fun provides 'analytical_dict_gradient_real'.
        """
        if not callable(func):
            raise ValueError("fun must be callable.")
        return func(params, arg)
    
    def dict_gradient_real_numerical_jax(func: Any, params: Any, arg: Any) -> Any:
        """
        Compute a dictionary of real gradients using numerical differentiation (JAX).
        """
        g = grad(lambda p, y: jnp.sum(jnp.real(func(p, y))))(params, arg)["params"]
        return tree_map(lambda x: x.ravel(), g)

    def dict_gradient_real_jax(func: Any, params: Any, arg: Any, analytical: bool = False) -> Any:
        """
        Wrapper for computing a dictionary of real gradients using JAX.
        """
        if analytical:
            return dict_gradient_real_analytical_jax(func, params, arg)
        return dict_gradient_real_numerical_jax(func, params, arg)

# -------------------------------------------------------------------------
#! APPLY CALLABLE
# -------------------------------------------------------------------------

if _JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(0,4))
    def apply_callable_jax(func,
                        states,
                        sample_probas,
                        logprobas_in,
                        logproba_fun,
                        parameters):
        """
        Applies a transformation function to each state and computes a locally
        weighted estimate as described in [2108.08631].
        
        Parameters:
            - func: Callable that accepts a state (vector) and returns a tuple (S, V)
                    where S is an array of modified states (M x state_size) and V is an
                    array of corresponding values (M,).
            - states: Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
            - sample_probas: Array of sampling probabilities (matching the leading dimensions of states).
            - logprobas_in: Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
            - logproba_fun: Callable that computes the log-probabilities for given states S.
            
        Returns:
            - estimates: The per-state estimates.
            - mean: Mean of the estimates.
            - std: Standard deviation of the estimates.
        """
        
        # the input of states might be (num_samples, num_chains, num_visible)
        # logprobas_in might be (num_samples, num_chains, 1)
        # transform to (num_samples * num_chains, num_visible) and (num_samples * num_chains, 1)
        # states          = jnp.reshape(states, (-1, states.shape[-1]))
        # logprobas_in    = jnp.reshape(logprobas_in, (-1, 1))
        # sample_probas   = jnp.reshape(sample_probas, (-1, 1))
    
        # function to compute the estimate for a single state
        def compute_estimate(state, logp, sample_p):
            # Squeeze the scalar values.
            logp        = logp[0]
            sample_p    = sample_p[0]
            
            # Obtain modified states and corresponding values.
            new_states, new_vals    = func(state)
            new_logprobas           = logproba_fun(parameters, new_states)
            weights                 = jnp.exp(new_logprobas - logp)
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            
            # Return the weighted sum.
            return jnp.sum(weighted_sum * sample_p, axis=0)
        
        applied = jax.vmap(compute_estimate, in_axes=(0, 0, 0))(states, logprobas_in, sample_probas)
        return applied, jnp.mean(applied, axis = 0), jnp.std(applied, axis = 0)
    
    @partial(jax.jit, static_argnums=(0,3))
    def apply_callable_jax_uniform(func, states, logprobas_in, logproba_fun, parameters):
        """
        Applies a transformation function to each state and computes a locally
        weighted estimate as described in [2108.08631].
        
        Parameters:
            - func: Callable that accepts a state (vector) and returns a tuple (S, V)
                    where S is an array of modified states (M x state_size) and V is an
                    array of corresponding values (M,).
            - states: Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
            - logprobas_in: Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
            - logproba_fun: Callable that computes the log-probabilities for given states S.
            
        Returns:
            - estimates: The per-state estimates.
            - mean: Mean of the estimates.
            - std: Standard deviation of the estimates.
            
        """
        # the input of states might be (num_samples, num_chains, num_visible)
        # logprobas_in might be (num_samples, num_chains, 1)
        # transform to (num_samples * num_chains, num_visible) and (num_samples * num_chains, 1)
        
        # states          = jnp.reshape(states, (-1, states.shape[-1]))
        # logprobas_in    = jnp.reshape(logprobas_in, (-1, 1))
        # function to compute the estimate for a single state
        def compute_estimate(state, logp):
            # Squeeze the scalar values.
            logp        = logp[0]
            
            # Obtain modified states and corresponding values.
            new_states, new_vals    = func(state)
            new_logprobas           = logproba_fun(parameters, new_states)
            weights                 = jnp.exp(new_logprobas - logp)
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            return jnp.sum(weighted_sum, axis=0)
        
        applied = jax.vmap(compute_estimate, in_axes=(0, 0))(states, logprobas_in)
        return applied, jnp.mean(applied, axis = 0), jnp.std(applied, axis = 0)
    
    @partial(jax.jit, static_argnums=(0,4,6))
    def apply_callable_batched_jax(func,
                                    states,
                                    sample_probas,
                                    logprobas_in,
                                    logproba_fun,
                                    parameters,
                                    batch_size: int):
        """
        Applies a transformation function to each state (in batches) and computes a locally weighted estimate
        as described in [2108.08631]. This version incorporates sample probabilities.
        
        Parameters:
            - func          : Callable that accepts a state (vector) and returns a tuple (S, V), where S is an array of modified 
                            states (M x state_size) and V is an array of corresponding values (M,).
            - states        : Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
            - sample_probas : Array of sampling probabilities (matching the leading dimensions of states).
            - logprobas_in  : Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
            - logproba_fun  : Callable that computes the log-probabilities for given states S; should accept (parameters, S).
            - parameters    : Additional parameters for logproba_fun.
            - batch_size    : Batch size to use for evaluation.
        
        Returns:
            - estimates     : Per-state estimates (flattened) -> shape (n_samples * n_chains,).
            - mean          : Mean of the estimates (over sample dimension).
            - std           : Standard deviation of the estimates (over sample dimension).
        """
        # Reshape inputs.
        # states          = jnp.reshape(states, (-1, states.shape[-1]))
        # logprobas_in    = jnp.reshape(logprobas_in, (-1, 1))#[:, 0]
        # sample_probas   = jnp.reshape(sample_probas, (-1, 1))#[:, 0]
        
        # Function to compute the estimate for a single state.
        def compute_estimate(state, logp, sample_p):
            logp                    = logp[0]           # Squeeze scalar.
            sample_p                = sample_p[0]       # Squeeze scalar.
            new_states, new_vals    = func(state)
            new_logprobas           = logproba_fun(parameters, new_states)
            weights                 = jnp.exp(new_logprobas - logp)
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            output                  = jnp.sum(weighted_sum, axis=0)
            return output
        
        # Create batches using a helper (assumed to be defined elsewhere).
        batches         = create_batches_jax(states, batch_size)
        log_batches     = create_batches_jax(logprobas_in, batch_size)
        sample_batches  = create_batches_jax(sample_probas, batch_size)
        
        # jax.debug.print("batches.shape {}", batches.shape)
        # jax.debug.print("log_batches.shape {}", log_batches.shape)
        # jax.debug.print("sample_batches.shape {}", sample_batches.shape)
        nstates         = states.shape[0]

        # Evaluate each batch using vmap.
        def compute_batch(batch_states, batch_logps, batch_sampleps):
            return jax.vmap(compute_estimate, in_axes=(0, 0, 0))(batch_states, batch_logps, batch_sampleps)
        
        # Map over batches.
        batch_estimates = jax.vmap(compute_batch, in_axes=(0, 0, 0))(batches, log_batches, sample_batches)
        estimates       = batch_estimates.reshape(-1)[:nstates]
        return estimates, jnp.mean(estimates, axis=0), jnp.std(estimates, axis=0)
    
    @partial(jax.jit, static_argnums=(0,3,5))
    def apply_callable_batched_jax_uniform(func,
                                        states,
                                        logprobas_in,
                                        logproba_fun,
                                        parameters,
                                        batch_size: int):
        """
        Applies a transformation function to each state (in batches) and computes a locally weighted estimate
        as described in [2108.08631]. This version does not incorporate sample probabilities.
        
        Parameters:
            - func          : Callable that accepts a state (vector) and returns a tuple (S, V), where S is an array of modified 
                            states (M x state_size) and V is an array of corresponding values (M,).
            - states        : Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
            - logprobas_in  : Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
            - logproba_fun  : Callable that computes the log-probabilities for given states S; should accept (parameters, S).
            - parameters    : Additional parameters for logproba_fun.
            - batch_size    : Batch size to use for evaluation.
        
        Returns:
            - estimates     : Per-state estimates (flattened) -> shape (n_samples * n_chains,).
            - mean          : Mean of the estimates (over sample dimension).
            - std           : Standard deviation of the estimates (over sample dimension).
        """
        # Reshape inputs.
        # states          = jnp.reshape(states, (-1, states.shape[-1]))
        # logprobas_in    = jnp.reshape(logprobas_in, (-1, 1))
        
        # Function to compute the estimate for a single state.
        def compute_estimate(state, logp):
            logp                    = logp[0]           # Squeeze scalar.
            new_states, new_vals    = func(state)
            new_logprobas           = logproba_fun(parameters, new_states)
            weights                 = jnp.exp(new_logprobas - logp)
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            # Return the weighted sum.
            return jnp.sum(weighted_sum, axis=0)
        
        # Create batches using a helper (assumed to be defined elsewhere).
        batches         = create_batches_jax(states, batch_size)
        log_batches     = create_batches_jax(logprobas_in, batch_size)
        
        # Evaluate each batch using vmap.
        def compute_batch(batch_states, batch_logps):
            return jax.vmap(compute_estimate, in_axes=(0, 0))(batch_states, batch_logps)
        
        #
        batch_estimates = jax.vmap(compute_batch, in_axes=(0, 0))(batches, log_batches)
        estimates       = batch_estimates.reshape(-1)[:states.shape[0]]
        return estimates, jnp.mean(estimates, axis=0), jnp.std(estimates, axis=0)
    
# -------------------------------------------------------------------------