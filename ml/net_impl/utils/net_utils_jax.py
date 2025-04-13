'''
file    : general_python/ml/net_impl/utils/net_utils_jax.py
author  : Maksymilian Kliczkowski
date    : 2025-03-01

'''

# from general python utils
from typing import Any, Callable, Dict, List, Tuple, NamedTuple
from general_python.algebra.utils import JAX_AVAILABLE, get_backend, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE
from functools import partial

if JAX_AVAILABLE:
    import jax
    from jax import grad
    from jax import numpy as jnp
    from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_leaves
    from jax.flatten_util import ravel_pytree

    # use flax
    import flax
    import flax.linen as nn
    from flax.core.frozen_dict import freeze, unfreeze
else:
    jax             = None
    jnp             = None
    grad            = None
    tree_flatten    = None
    tree_unflatten  = None
    tree_map        = None
    tree_leaves     = None
    ravel_pytree    = None
    freeze          = None
    unfreeze        = None
    nn              = None
    flax            = None
    
#########################################################################
#! BATCHES
#########################################################################

if JAX_AVAILABLE:
    
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

if JAX_AVAILABLE:
    
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

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnames=('apply_fun',))
    def _compute_numerical_grad_parts(apply_fun: Callable[[Any, Any], jnp.ndarray],
                                        params: Any, arg: Any):
        """
        Helper to compute \nabla Re[f] and \nabla Im[f] PyTrees.
        
        Parameters
        ----------
        apply_fun : Callable
            The function to differentiate.
        params : Any
            The parameters of the function.
        arg : Any
            The input to the function.
            
        Returns
        ----------
        grad_real_tree, grad_imag_tree
        """
        grad_real_tree = grad(lambda p, y: jnp.sum(jnp.real(apply_fun(p, y))))(params, arg)
        grad_imag_tree = grad(lambda p, y: jnp.sum(jnp.imag(apply_fun(p, y))))(params, arg)
        return grad_real_tree, grad_imag_tree

    @partial(jax.jit, static_argnames=('apply_fun',))
    def _compute_numerical_grad_real_part(apply_fun: Callable[[Any, Any], jnp.ndarray],
                                        params: Any, arg: Any):
        """
        Helper to compute \nabla Re[f] PyTree.
        
        Parameters
        ----------
        apply_fun : Callable
            The function to differentiate.
        params : Any
            The parameters of the function.
        arg : Any
            The input to the function.
        
        Returns
        ----------
        grad_real_tree
        """
        # Assumes params are real or complex, computes gradient w.r.t. all.
        grad_real_tree = grad(lambda p, y: jnp.sum(jnp.real(apply_fun(p, y))))(params, arg)
        return grad_real_tree

    # -----------------------------------------------------------------------------
    #! Flatten the gradients
    # -----------------------------------------------------------------------------
    
    @jax.jit
    def flatten_gradient_pytree(grad_pytree: Any) -> jnp.ndarray:
        """
        Flattens a PyTree of gradients into a single 1D JAX array.
        """
        leaves = tree_leaves(grad_pytree)
        if not leaves:
            return jnp.array([], dtype=jnp.float32)

        is_complex_output       = any(jnp.iscomplexobj(leaf) for leaf in leaves)
        if is_complex_output:
            target_dtype        = DEFAULT_JP_CPX_TYPE
            # Ravel and cast all leaves to complex
            processed_pytree    = tree_map(lambda x: x.ravel().astype(target_dtype), grad_pytree)
        else:
            target_dtype        = DEFAULT_JP_FLOAT_TYPE
            # Ravel and cast all leaves to float
            processed_pytree = tree_map(
                lambda x: x.ravel().astype(target_dtype) if not jnp.issubdtype(x.dtype, jnp.floating) else x.ravel().astype(target_dtype),
                grad_pytree
            )

        # Flatten the processed PyTree (leaves are now 1D arrays of correct type)
        flat_leaves, _ = tree_flatten(processed_pytree)

        if not flat_leaves:
            return jnp.array([], dtype=target_dtype)
        return jnp.concatenate(flat_leaves)
        
    @jax.jit
    def flatten_gradient_pytree_real(grad_pytree: Any) -> jnp.ndarray:
        """
        Flattens a PyTree of real gradients into a single 1D array.

        Parameters
        ----------
        grad_pytree : Any
            A PyTree (e.g., nested dicts/lists/tuples) containing real gradient arrays as leaves.

        Returns
        -------
        jnp.ndarray
            A single 1D array containing all gradient elements concatenated.
            Dtype will be float.

        Applicability
        -------------------
        Use for flattening real gradient structures.
        """
        flat_grad_list, _ = tree_flatten(tree_map(lambda x: x.ravel(), grad_pytree))

        if not flat_grad_list:
            return jnp.array([], dtype=DEFAULT_JP_FLOAT_TYPE)

        # Ensure all leaves are real and concatenate
        processed_leaves = [leaf.astype(DEFAULT_JP_FLOAT_TYPE) if not jnp.issubdtype(leaf.dtype, jnp.floating) else leaf for leaf in flat_grad_list]
        return jnp.concatenate(processed_leaves)

    @jax.jit
    def flatten_gradient_pytree_complex(grad_pytree: Any) -> jnp.ndarray:
        """
        Flattens a PyTree of complex gradients into a single 1D array.

        Parameters
        ----------
        grad_pytree : Any
            A PyTree (e.g., nested dicts/lists/tuples) containing complex gradient arrays as leaves.

        Returns
        -------
        jnp.ndarray
            A single 1D array containing all gradient elements concatenated.
            Dtype will be complex.

        Applicability
        -------------------
        Use for flattening non-holomorphic complex gradient structures.
        """
        flat_grad_list, _ = tree_flatten(tree_map(lambda x: x.ravel(), grad_pytree))

        if not flat_grad_list:
            return jnp.array([], dtype=DEFAULT_JP_CPX_TYPE)

        # Ensure all leaves are complex and concatenate
        processed_leaves = [leaf.astype(DEFAULT_JP_CPX_TYPE) if not jnp.iscomplexobj(leaf) else leaf for leaf in flat_grad_list]
        return jnp.concatenate(processed_leaves)

    @jax.jit
    def flatten_gradient_pytree_holo_real(complex_grad_pytree: Any) -> jnp.ndarray:
        """
        Flattens a PyTree of complex gradients (like :math:`\\nabla_{p^*} f`) into the
        real vector format `[Re(g)..., Im(g)...]`.

        Assumes the input `complex_grad_pytree` contains the relevant complex gradients
        (i.e., computed analytically or numerically).

        Parameters
        ----------
        complex_grad_pytree : Any
            A PyTree containing complex gradient arrays as leaves (e.g., :math:`\\nabla_{p^*} f`).

        Returns
        -------
        jnp.ndarray
            A single 1D **real** array (float dtype) formatted as `[real_parts..., imag_parts...]`.

        Applicability
        -------------------
        Use when the holomorphic gradient
        
        :math:`\\nabla_{p^*} f`
        
        has been computed (as a complex PyTree)
        and needs to be formatted as a real vector for optimizers.
        
        The holomorphic gradient means that the imaginary part is not
        independent of the real part, and thus the gradient is
        represented as a single complex vector :math:`\\nabla_{p^*} f`.
        This happens when the function is holomorphic in the complex
        parameters :math:`p^*`.
        The output is a real vector of the form:
        :math:`[Re(\\nabla_{p^*} f), Im(\\nabla_{p^*} f)]`.
        
        """
        # Extract real and imaginary parts and flatten them
        real_parts_flat_list = tree_leaves(tree_map(lambda x: jnp.real(x).ravel(), complex_grad_pytree))
        imag_parts_flat_list = tree_leaves(tree_map(lambda x: jnp.imag(x).ravel(), complex_grad_pytree))

        if not real_parts_flat_list and not imag_parts_flat_list:
            return jnp.array([], dtype=DEFAULT_JP_FLOAT_TYPE)

        # Concatenate real and imaginary parts
        flat_real_grad = jnp.concatenate(real_parts_flat_list + imag_parts_flat_list)
        return flat_real_grad.astype(DEFAULT_JP_FLOAT_TYPE)

    # -----------------------------------------------------------------------------
    #! PyTree gradients
    # -----------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def pytree_gradient_numerical_jax(apply_fun : Callable[[Any, Any], jnp.ndarray],
                                    params      : Any,
                                    arg         : Any) -> Any:
        r"""
        Compute PyTree of standard complex gradient :math:`\\nabla_p f = \\nabla_p Re[f] + i \\nabla_p Im[f]` numerically.

        Math
        ----
        Computes :math:`g_R = \\nabla_p \\sum \\text{Re}[f(p, x)]` and :math:`g_I = \\nabla_p \\sum \\text{Im}[f(p, x)]`.
        Returns the PyTree :math:`g = g_R + i g_I`.

        Parameters
        ----------
        apply_fun : Callable[[Any, Any], jnp.ndarray]
            Network apply function `f(p, x)`. Must be JAX traceable.
        params : Any
            Network parameters (pytree).
        arg : Any
            Input state `x`.

        Returns
        -------
        Any
            Complex gradient PyTree :math:`\\nabla_p f`, matching `params` structure.

        Applicability
        -------------------
        Computes standard :math:`\\nabla_p \log \psi` numerically.
        **Can be JIT-compiled** if `apply_fun` is static.
        """
        gr_tree, gi_tree    = _compute_numerical_grad_parts(apply_fun, params, arg)

        # Combine into complex PyTree
        is_complex_params   = any(jnp.iscomplexobj(leaf) for leaf in tree_leaves(params))
        def combine(gr, gi):
            # Always return complex if params are complex or imag part exists structurally
            if is_complex_params or gi is not None:
                # Check gi presence via tree structure match
                return gr.astype(DEFAULT_JP_CPX_TYPE) + 1.j * gi.astype(DEFAULT_JP_CPX_TYPE)
            else:
                return gr.astype(DEFAULT_JP_FLOAT_TYPE)
        return tree_map(combine, gr_tree, gi_tree)

    @partial(jax.jit, static_argnums=(0,))
    def pytree_gradient_cpx_nonholo_numerical_jax(apply_fun : Callable[[Any, Any], jnp.ndarray],
                                                    params  : Any,
                                                    arg     : Any) -> Any:
        r"""
        Compute PyTree of non-holo conjugate gradient :math:`(\\nabla_p f)^* = \\nabla_p Re[f] - i \\nabla_p Im[f]` numerically.

        Math
        ----
        Computes 
            :math:`g_R = \\nabla_p \\sum \\text{Re}[f(p, x)]` 
        and 
            :math:`g_I = \\nabla_p \\sum \\text{Im}[f(p, x)]`.
            
        Returns the PyTree :math:`g^* = g_R - i g_I`. 
        This is often :math:`(\\nabla_{\eta} \log \psi)^*`.

        Parameters
        ----------
        apply_fun : Callable[[Any, Any], jnp.ndarray]
            Network apply function `f(p, x)`. Must be JAX traceable.
        params : Any
            Network parameters (pytree).
        arg : Any
            Input state `x`.

        Returns
        -------
        Any
            Complex conjugate gradient PyTree :math:`(\\nabla_p f)^*`, matching `params` structure.

        Applicability & JIT
        -------------------
        Computes the conjugate gradient :math:`(\\nabla_{\eta} \log \psi)^*` numerically. Crucial for VMC updates.
        **Can be JIT-compiled** if `apply_fun` is static.
        """
        gr_tree, gi_tree    = _compute_numerical_grad_parts(apply_fun, params, arg)
        jax.debug.print("gr_tree: {}", gr_tree)
        # Combine into complex conjugate PyTree
        is_complex_params   = any(jnp.iscomplexobj(leaf) for leaf in tree_leaves(params))
        def combine_conj(gr, gi):
            if is_complex_params or gi is not None:
                return gr.astype(DEFAULT_JP_CPX_TYPE) - 1.j * gi.astype(DEFAULT_JP_CPX_TYPE)
            else:
                return gr.astype(DEFAULT_JP_FLOAT_TYPE)
        combined = tree_map(combine_conj, gr_tree, gi_tree)
        jax.debug.print("combined: {}", combined)
        return combined

    @partial(jax.jit, static_argnums=(0,))
    def pytree_gradient_real_jax(apply_fun  : Callable[[Any, Any], jnp.ndarray],
                                params      : Any,
                                arg         : Any) -> Any:
        """
        Compute PyTree of real gradient :math:`\\nabla_p Re[f]` numerically.

        Math
        ----
        Computes :math:`g_R = \\nabla_p \\sum \\text{Re}[f(p, x)]`. Assumes parameters `p` are real.

        Parameters
        ----------
        apply_fun : Callable[[Any, Any], jnp.ndarray]
            Network apply function `f(p, x)`. Must be JAX traceable.
        params : Any
            Network parameters (pytree), assumed real.
        arg : Any
            Input state `x`.

        Returns
        -------
        Any
            Real gradient PyTree :math:`\\nabla_p Re[f]`, matching `params` structure.

        Applicability & JIT
        -------------------
        Computes gradient for real parameters w.r.t real part of output.
        **Can be JIT-compiled** if `apply_fun` is static.
        """
        g_tree = _compute_numerical_grad_real_part(apply_fun, params, arg)
        return tree_map(lambda x: x.astype(DEFAULT_JP_FLOAT_TYPE), g_tree)

    @partial(jax.jit, static_argnums=(0,))
    def pytree_gradient_cpx_holo_jax(apply_fun  : Callable[[Any, Any], jnp.ndarray],
                                    params      : Any,
                                    arg         : Any) -> Any:
        r"""
        Compute PyTree of holomorphic gradient :math:`\\nabla_{p^*} f` numerically.
        
        """
        
        complex_grads_tree_h = grad(lambda p, y: jnp.sum(apply_fun(p, y)), holomorphic=True)(params, arg)
        return tree_map(lambda x: x.astype(DEFAULT_JP_CPX_TYPE), complex_grads_tree_h)

    # -----------------------------------------------------------------------------
    #! Flattened gradients
    # -----------------------------------------------------------------------------
    
    @partial(jax.jit, static_argnames=('apply_fun',))
    def flat_gradient_cpx_nonholo_numerical_jax(apply_fun   : Callable[[Any, Any], jnp.ndarray],
                                                params      : Any,
                                                arg         : Any) -> jnp.ndarray:
        """
        Numerically compute flattened conjugate gradient :math:`(\\nabla_p f)^*`.
        This computes ∇Re[f] - i∇Im[f] and flattens it.
        Output is complex if params are complex.
        """
        # This function computes the necessary PyTree: ∇Re[f] - i∇Im[f]
        grad_pytree = pytree_gradient_cpx_nonholo_numerical_jax(apply_fun, params, arg)

        # This robust utility flattens the PyTree, returning a complex array
        # if any leaf is complex (which it should be if params are complex),
        # or a real array otherwise.
        return flatten_gradient_pytree(grad_pytree)

    @partial(jax.jit, static_argnames=('apply_fun',))
    def flat_gradient_numerical_jax(apply_fun   : Callable[[Any, Any], jnp.ndarray],
                                    params      : Any,
                                    arg         : Any) -> jnp.ndarray:
        """
        Numerically compute flattened standard complex gradient :math:`\\nabla_p f = \\nabla Re[f] + i \\nabla Im[f]`.
        """
        grad_pytree = pytree_gradient_numerical_jax(apply_fun, params, arg)
        return flatten_gradient_pytree(grad_pytree)

    @partial(jax.jit, static_argnames=('apply_fun',))
    def flat_gradient_real_jax(apply_fun  : Callable[[Any, Any], jnp.ndarray],
                                        params      : Any,
                                        arg         : Any) -> jnp.ndarray:
        """
        Compute the flattened real gradient.
        """
        grad_pytree = pytree_gradient_real_jax(apply_fun, params, arg)
        flat_grad   = flatten_gradient_pytree(grad_pytree)
        return flat_grad.astype(DEFAULT_JP_FLOAT_TYPE)

    @partial(jax.jit, static_argnames=('apply_fun',))
    def flat_gradient_cpx_holo_jax(apply_fun    : Callable[[Any, Any], jnp.ndarray],
                                params          : Any,
                                arg             : Any) -> jnp.ndarray:
        """
        Compute the flattened holomorphic gradient over the complex PyTree.
        """
        grad_pytree = pytree_gradient_cpx_holo_jax(apply_fun, params, arg)
        # This specific flattening function handles the [Re..., Im...] format
        return flatten_gradient_pytree_holo_real(grad_pytree)

    # ----------------------------------------------------------------------------
    #! Compute the gradient
    # ----------------------------------------------------------------------------
    
    @partial(jax.jit, static_argnums=(0, 3, 4))
    def compute_gradients_batched(
                                net_apply                   : Callable[[Any, Any], jnp.ndarray],
                                params                      : Any,
                                states                      : jnp.ndarray,
                                single_sample_flat_grad_fun : Callable[[Any, Any], jnp.ndarray], # Takes (params, single_state) -> flat_grad_vector
                                batch_size                  : int) -> jnp.ndarray:
        '''
        Compute flattened gradients per sample over a batch of states for general networks using JAX.

        Applies the provided `single_sample_flat_grad_fun` to each vector in `states`,
        processing in batches using `jax.lax.scan` and `jax.vmap`. Designed for JIT compilation.
        This function assumes `single_sample_flat_grad_fun` correctly computes the
        desired type of flattened gradient for general networks (e.g., real, complex, or holomorphic).

        Parameters
        ----------
        net_apply : Callable[[Any, Any], jnp.ndarray]
            Network apply function `f(p, x)`. Must be JAX traceable.
        params : Any
            Network parameters. Passed to `single_sample_flat_grad_fun`.
        states : jnp.ndarray
            Input states, shape `(num_samples, ...)`. These are the input vectors to the network.
        single_sample_flat_grad_fun : Callable[[Any, Any], jnp.ndarray]
            A JAX-traceable function computing the **flattened** gradient for a
            **single** input state. Signature: `fun(params, single_state) -> flat_gradient_vector`.
            The output format (real/complex, standard/holo) must match the requirements of the network.
            *Must be passed statically* for JIT compilation.
        batch_size : int
            Batch size. *Must be passed statically* for JIT compilation.

        Returns
        -------
        jnp.ndarray
            Array of flattened gradients, shape `(num_samples, num_flat_params)`.
            Dtype matches the output of `single_sample_flat_grad_fun`.
        '''

        num_total_states    = states.shape[0]
        # Create batches, handles padding
        # Expected shape: (num_batches, batch_size, ...state_dims)
        batched_states      = create_batches_jax(states, batch_size)

        # Define the function to compute gradients for a single batch using vmap
        # Maps over the second dimension (states) of the input batch. Params are fixed.
        def vmap_target(p, s):
            # Calls the user-provided function that expects (net_apply, params, state)
            return single_sample_flat_grad_fun(net_apply, p, s)

        # Map the wrapped function over the states axis (0)
        compute_batch_grads = jax.vmap(vmap_target, in_axes=(None, 0), out_axes=0)
        
        # Define the scan function to iterate over batches
        def scan_fun(carry, batch):
            # Compute gradients for the current batch, passing fixed params and the batch
            batch_grads = compute_batch_grads(params, batch)
            return carry, batch_grads

        _, gradients_batched = jax.lax.scan(scan_fun, None, batched_states)
        # gradients_batched shape: (num_batches, batch_size, num_flat_params)

        # Reshape the stacked gradients to (num_total_padded_samples, num_flat_params)
        gradients_padded = gradients_batched.reshape(-1, gradients_batched.shape[-1])

        # Remove the gradients corresponding to padded states
        gradients = gradients_padded[:num_total_states]
        # Final shape: (num_samples, num_flat_params)

        return gradients
    
    # ----------------------------------------------------------------------------
    #! Transform the vector from the original representation to the real representation [Re..., Im...]
    # ----------------------------------------------------------------------------
    
    @jax.jit
    def _ensure_real_repr_vector_single(vector: jnp.ndarray) -> jnp.ndarray:
        """
        Internal JIT function:
        
        Ensures a single vector is the real representation.
        See ensure_real_repr_vector docstring.
        
        Parameters
        ----------
        vector : jnp.ndarray
            A 1D JAX array, real or complex.
        """
        if jnp.iscomplexobj(vector):
            real_parts = jnp.real(vector)
            imag_parts = jnp.imag(vector)
            return jnp.concatenate([real_parts, imag_parts]).astype(DEFAULT_JP_FLOAT_TYPE)
        else:
            return vector
    
    # map the single-vector function over the first axis (vectors), keep template fixed (None)
    _ensure_real_repr_vector_batch = jax.vmap(_ensure_real_repr_vector_single, in_axes=0, out_axes=0)

    @jax.jit
    def ensure_real_repr_vector(vectors: jnp.ndarray) -> jnp.ndarray:
        """
        Ensures the output vector(s) are the real representation [Re..., Im...] if complex,
        or returns the original vector(s) if already real. Handles single vectors (1D)
        or batches of vectors (2D, where each row is a vector).

        - If input is 1D complex: returns 1D real `[Re(v), Im(v)]`.
        - If input is 1D real: returns 1D real `v`.
        - If input is 2D complex: returns 2D real, each row `[Re(v_i), Im(v_i)]`.
        Output shape: `(batch_size, 2 * original_dim)`.
        - If input is 2D real: returns 2D real (original shape).
        Output shape: `(batch_size, original_dim)`.

        Parameters
        ----------
        vectors : jnp.ndarray
            A 1D or 2D JAX array, real or complex. If 2D, rows are treated as vectors.

        Returns
        -------
        jnp.ndarray
            A 1D or 2D JAX array with a real (float) dtype, representing the
            input vector(s) in the appropriate real format.
            Note the potential change in the last dimension's size for complex inputs.
        """
        if vectors.ndim == 1:
            return _ensure_real_repr_vector_single(vectors)
        elif vectors.ndim == 2:
            # Important check: vmap requires consistent output sizes for the mapped dimension.
            # This function _changes_ the size of the last dimension if the input is complex.
            # Therefore, vmap only works reliably if the *entire input batch* is real
            # OR the *entire input batch* is complex. Mixing is problematic for vmap here.
            if jnp.iscomplexobj(vectors):
                # If the whole batch is complex, apply vmap safely
                return _ensure_real_repr_vector_batch(vectors).astype(DEFAULT_JP_FLOAT_TYPE)
            else:
                # If the whole batch is real, just ensure float type
                return vectors.astype(DEFAULT_JP_FLOAT_TYPE)
        else:
            raise ValueError(f"Input must be 1D or 2D, got ndim={vectors.ndim}")

    # ----------------------------------------------------------------------------
    #! Transform the vector from the real representation [Re..., Im...] to the original representation
    # ----------------------------------------------------------------------------

    @partial(jax.jit, static_argnames=('params_template_is_complex',))
    def _vector_from_real_repr_single_impl(
        real_repr_vector            : jnp.ndarray,
        params_template_is_complex  : bool,
        expected_size_if_real       : int) -> jnp.ndarray:
        """
        Internal JIT function
        
        Converts a single real vector based on template complexity.
        See vector_from_real_repr docstring.
        
        Note:
        ----------
        The real representation is assumed to be in the form [Re..., Im...]
        
        Parameters
        ----------
        real_repr_vector : jnp.ndarray
            A 1D JAX array, real representation of the vector.
        params_template_is_complex : bool
            Indicates if the parameter template is complex.
        expected_size_if_real : int
            Expected size of the vector if it is real.
        Returns
        -------
        jnp.ndarray
            A 1D JAX array, converted to the appropriate type (complex or real).
            
            If `params_template_is_complex` is True, the output will be complex.
            -> The output will be of size `(N,)` where `N = 2 * original_dim`.
            If `params_template_is_complex` is False, the output will be real.
            -> The output will be of size `(M,)` where `M = original_dim`.
        
        Example
        -------
        
        >>> real_repr_vector = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> params_template_is_complex = True
        >>> expected_size_if_real = 4
        >>> result = _vector_from_real_repr_single_impl(real_repr_vector, params_template_is_complex, expected_size_if_real)
        >>> print(result)
        [1.0 + 2.0j, 3.0 + 4.0j]
        """
        if jnp.iscomplexobj(real_repr_vector):
            raise ValueError("Input vector must be real for _vector_from_real_repr_single_impl.")

        # Check if the input vector is complex
        if params_template_is_complex:
            
            # safe check: ensure even size for complex params as [Re..., Im...]
            if real_repr_vector.size % 2 != 0:
                raise ValueError("Input vector size must be even for complex params template.")
            
            n                   = real_repr_vector.size // 2
            real_repr_vector    = real_repr_vector.astype(DEFAULT_JP_FLOAT_TYPE)
            real_parts          = real_repr_vector[:n]
            imag_parts          = real_repr_vector[n:]
            return (real_parts + 1.j * imag_parts).astype(DEFAULT_JP_CPX_TYPE)
        else:
            # Optional: Validate size against template
            if real_repr_vector.size != expected_size_if_real:
                raise ValueError(f"Input vector size ({real_repr_vector.size}) does not match "
                                f"expected size ({expected_size_if_real}) for real params template.")
            return real_repr_vector.astype(DEFAULT_JP_FLOAT_TYPE)

    def _vector_from_real_repr_single(
        real_repr_vector    : jnp.ndarray,
        params_template     : Any) -> jnp.ndarray:
        """
        Internal non-JIT function:
        
        Determines complexity from template and calls impl.
        See vector_from_real_repr docstring.
        
        Parameters
        ----------
        real_repr_vector : jnp.ndarray
            A 1D JAX array, real representation of the vector.
        params_template : Any
            A PyTree defining the target structure and complexity.
        Returns
        -------
        jnp.ndarray
            A 1D JAX array, converted to the appropriate type (complex or real).
        """
        
        # Determine if the parameter template is complex or real
        # tree_leaves returns a list of leaves (arrays) in the PyTree
        # If the PyTree is empty, we assume it's a real template
        template_leaves = tree_leaves(params_template)
        if not template_leaves:
            params_template_is_complex  = False
            expected_size_if_real       = 0
            if real_repr_vector.size == 0:
                return jnp.array([], dtype=DEFAULT_JP_FLOAT_TYPE)
            else: 
                raise ValueError("Non-empty input vector provided with an empty parameter template.")
        else:
            params_template_is_complex  = any(jnp.iscomplexobj(leaf) for leaf in template_leaves)
            if not params_template_is_complex:
                expected_size_if_real   = sum(leaf.size for leaf in template_leaves)
            else:
                # Size check happens inside impl based on evenness
                expected_size_if_real   = -1
        return _vector_from_real_repr_single_impl(
            real_repr_vector,
            params_template_is_complex,
            expected_size_if_real
        )

    # map the single-vector function over the first axis (vectors), keep template fixed (None)
    _vector_from_real_repr_batch = jax.vmap(_vector_from_real_repr_single, in_axes=(0, None), out_axes=0)

    @jax.jit
    def vector_from_real_repr(real_repr_vectors : jnp.ndarray,
                            params_template     : Any) -> jnp.ndarray:
        """
        Converts real representation vector(s) back to original form (complex/real)
        based on a parameter template. Handles single vectors (1D) or batches (2D).

        - If `params_template` is complex: assumes input row(s) are `[Re..., Im...]`
        and converts back to complex. Output shape (if 2D): `(batch, N)`.
        - If `params_template` is real: assumes input row(s) are direct real gradients
        and returns them as float. Output shape (if 2D): `(batch, M)`.

        Parameters
        ----------
        real_repr_vectors : jnp.ndarray
            A 1D or 2D JAX array with a real (float) dtype. If 2D, rows are vectors.
        params_template : Any
            A PyTree defining the target structure and complexity.

        Returns
        -------
        jnp.ndarray
            A 1D or 2D JAX array, matching the input ndim and the target complexity.
        """
        if real_repr_vectors.ndim == 1:
            return _vector_from_real_repr_single(real_repr_vectors, params_template)
        elif real_repr_vectors.ndim == 2:
            # vmap works fine here because the params_template is fixed for all rows
            return _vector_from_real_repr_batch(real_repr_vectors, params_template)
        else:
            raise ValueError(f"Input must be 1D or 2D, got ndim={real_repr_vectors.ndim}")
        
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
        gr  = grad(lambda p, y: jnp.sum(jnp.real(func(p, y))))(params, arg)
        gr  = tree_map(lambda x: x.ravel(), gr)
        gi  = grad(lambda p, y: jnp.sum(jnp.imag(func(p, y))))(params, arg)
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
        g = grad(lambda p, y: jnp.sum(jnp.real(func(p, y))))(params, arg)
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

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(0,4))
    def apply_callable_jax(func,
                        states,
                        sample_probas,
                        logprobas_in,
                        logproba_fun,
                        parameters,
                        mu = 2.0):
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
            weights                 = jnp.exp((new_logprobas - logp))
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            
            # Return the weighted sum.
            return weighted_sum * sample_p
        
        applied = jax.vmap(compute_estimate, in_axes=(0, 0, 0))(states, logprobas_in, sample_probas)
        return applied, jnp.mean(applied, axis = 0), jnp.std(applied, axis = 0)
    
    @partial(jax.jit, static_argnums=(0,3))
    def apply_callable_jax_uniform(func, states, logprobas_in, logproba_fun, parameters, mu = 2.0):
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
            logp                    = logp[0]
            
            # Obtain modified states and corresponding values.
            new_states, new_vals    = func(state)
            new_logprobas           = logproba_fun(parameters, new_states)
            weights                 = jnp.exp((new_logprobas - logp))
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            return weighted_sum
        
        applied = jax.vmap(compute_estimate, in_axes=(0, 0))(states, logprobas_in)
        return applied, jnp.mean(applied, axis = 0), jnp.std(applied, axis = 0)
    
    @partial(jax.jit, static_argnums=(0,4,6))
    def apply_callable_batched_jax(func,
                                    states,
                                    sample_probas,
                                    logprobas_in,
                                    logproba_fun,
                                    parameters,
                                    batch_size: int,
                                    mu = 2.0):
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
            weights                 = jnp.exp((new_logprobas - logp))
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            output                  = (sample_p * weighted_sum)
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
        def compute_batch(batch_states, batch_logps, batch_samples):
            return jax.vmap(compute_estimate, in_axes=(0, 0, 0))(batch_states, batch_logps, batch_samples)
        
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
                                        batch_size: int,
                                        mu = 2.0):
        """
        Applies a transformation function to each state (in batches) and computes a locally weighted estimate
        as described in [2108.08631]. This version does not incorporate sample probabilities.
        
        Parameters:
            - func:
                Callable that accepts a state (vector) and returns a tuple (S, V), where S is an array of modified 
                states (M x state_size) and V is an array of corresponding values (M,).
            - states:
                Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
            - logprobas_in:
                Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
            - logproba_fun:
                Callable that computes the log-probabilities for given states S; should accept (parameters, S).
            - parameters:
                Additional parameters for logproba_fun.
            - batch_size:
                Batch size to use for evaluation.
        
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
            weights                 = jnp.exp((new_logprobas - logp))
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
    #! ADD
    # -------------------------------------------------------------------------

    class SliceInfo(NamedTuple):
        start       : int
        size        : int
        shape       : Tuple[int, ...]
        is_complex  : bool
    
    def prepare_unflatten_metadata(
        shapes  : List[Tuple[int, Tuple[int, ...]]],
        iscpx   : bool,
        params  : Any) -> Tuple[SliceInfo, ...]:
        """
        Extracts slicing metadata for unflattening flat vectors into parameter trees.

        Parameters
        ----------
        shapes : List[Tuple[int, Tuple[int, ...]]]
            A list of (num_real_components, shape) for each parameter leaf.
        iscpx : bool
            Whether the network stores complex-valued parameters.
        params : Any
            Current parameter PyTree (used to infer if leaves are complex).
        
        Returns
        -------
        Tuple[SliceInfo, ...]
            Metadata for use in fast_unflatten.
        """
        flat_template, _ = tree_flatten(params)

        slice_indices   = []
        start           = 0

        for i, (num_real, shape) in enumerate(shapes):
            is_leaf_complex = bool(jnp.iscomplexobj(flat_template[i]) and iscpx)
            shape_py        = tuple(int(s) for s in shape)
            num_elements    = int(jnp.prod(jnp.array(shape_py)))

            # Sanity check: does num_real match shape and type?
            expected_real   = 2 * num_elements if is_leaf_complex else num_elements
            if int(num_real) != expected_real:
                raise ValueError(
                    f"Mismatch in leaf {i}: shape {shape_py}, complex={is_leaf_complex} -> "
                    f"expected {expected_real} real values, got {num_real}"
                )

            slice_indices.append(SliceInfo(start=int(start), size=int(num_real), shape=shape_py, is_complex=is_leaf_complex))
            start += int(num_real)

        return tuple(slice_indices)
    
    @partial(jax.jit, static_argnums=(1,))
    def fast_unflatten(d_par: jnp.ndarray, slice_info: List[Tuple[int, int, Tuple[int], bool]]):
        """
        JIT-compiled unflattening of a real flat vector into parameter leaves.
        """
        leaves = []
        for start, size, shape, is_cpx in slice_info:
            segment = d_par[start:start + size]
            if is_cpx:
                re, im = jnp.split(segment, 2)
                leaf = (re + 1j * im).reshape(shape)
            else:
                leaf = segment.reshape(shape).astype(DEFAULT_JP_FLOAT_TYPE)
            leaves.append(leaf)
        return leaves
    
    @jax.jit
    def add_tree(p1, p2):
        """JIT-compiled tree addition."""
        return tree_map(jax.lax.add, p1, p2)
    
# -------------------------------------------------------------------------