'''
file    : general_python/ml/net_impl/utils/net_utils_jax.py
author  : Maksymilian Kliczkowski
date    : 2025-03-01

'''

# from general python utils
from typing import Any, Callable, Dict, List, Tuple, NamedTuple, Optional
from ....algebra.utils import JAX_AVAILABLE, get_backend, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE
from functools import partial

if JAX_AVAILABLE:
    import jax
    from jax import grad
    from jax import numpy as jnp
    from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_leaves
    from jax.flatten_util import ravel_pytree
    from jax import lax

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
    lax             = None
    
#########################################################################
#! BATCHES
#########################################################################

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(1,))
    def create_batches_jax( data        : jnp.ndarray,
                            batch_size  : int):
        """
        JAX version of create_batches.
        
        The function takes a JAX array and a batch size as input.
        It calculates the number of samples needed to fill the last batch
        and pads the array with the last sample to ensure all batches are of equal size.
        
        Parameters
        ----------
        data : jnp.ndarray
            The input JAX array to be batched.
        batch_size : int
            The size of each batch.
        Returns
        -------
        jnp.ndarray
            The input array reshaped into batches of the specified size.
            The last batch may contain repeated samples if padding is needed.
        
        """
        
        # if batch_size < 2:
        #     return data
        
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
    
    # @partial(jax.jit, static_argnums=(0,1))
    def eval_batched_jax(batch_size : int,
                        func        : Any,
                        params      : Any,
                        data        : jnp.ndarray):
        """
        JAX version of eval_batched.
        
        Evaluates a function on batches of data using JAX.
        The function takes a batch size, a callable function, parameters for the function,
        and a JAX array as input. It creates batches of the data and evaluates the function
        on each batch using JAX's vmap and scan functions.
        
        Parameters
        ----------
        batch_size : int
            The size of each batch.
        func : Callable
            The function to evaluate on the data. It should accept parameters and a single data point.
        params : Any
            The parameters for the function.
        data : jnp.ndarray
            The input JAX array to be evaluated in batches.
        Returns
        -------
        jnp.ndarray
            The results of evaluating the function on the input data in batches.
        """
        # n           = data.shape[0]
        # num_batches = (n + batch_size - 1) // batch_size

        # def eval_single_batch(i):
        #     start   = i * batch_size
        #     end     = jnp.minimum((i + 1) * batch_size, n)
        #     batch   = jax.lax.dynamic_slice_in_dim(data, start, end - start, axis=0)
        #     return jax.vmap(partial(func, params))(batch)

        # # Evaluate each batch using lax.map (no carry needed)
        # results = jax.lax.map(eval_single_batch, jnp.arange(num_batches))
        # return jnp.concatenate(results, axis=0)
    
        # #! Create batches of data
        batches = create_batches_jax(data, batch_size)
        
        def scan_fun(c, x):
            return c, jax.vmap(lambda y: func(params, y), in_axes=(0,))(x)

        # Evaluate the function on each batch using vmap
        st = jax.lax.scan(scan_fun, None, jnp.array(batches))[1].reshape((-1,))
        return st[:data.shape[0]]
    
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

if JAX_AVAILABLE:

    # -----------------------------------------------------------------------------
    #! Compute the numerical gradient
    # -----------------------------------------------------------------------------

    @partial(jax.jit, static_argnames=('apply_fun',))
    def _compute_numerical_grad_parts(apply_fun: Callable[[Any, Any], jnp.ndarray],
                                        params: Any,
                                        state: Any):
        """
        Computes PyTrees for ∇_p Re[f] and ∇_p Im[f] using jax.grad.

        This is a fundamental building block for numerical complex gradients.
        It calculates the gradients of the real and imaginary parts of the
        scalarized function output (sum over output elements) with respect
        to the parameters `p`.

        Parameters
        ----------
        apply_fun : Callable[[Any, Any], jnp.ndarray]
            The function f(p, s) to differentiate. Must be JAX traceable.
        params : Any
            The parameters `p` (PyTree) of the function.
        state : Any
            The single input state `s` to the function.

        Returns
        -------
        Tuple[Any, Any]
            Tuple containing (grad_real_tree, grad_imag_tree).
            `grad_real_tree`:
                PyTree matching `params`, containing ∇_p(sum(Re[f(p, s)])).
            `grad_imag_tree`: 
                PyTree matching `params`, containing ∇_p(sum(Im[f(p, s)])).
        """
        # Use jax.grad to compute gradients w.r.t. the *first* argument (params)
        grad_real_tree = jax.grad(lambda p, s: jnp.sum(jnp.real(apply_fun(p, s))))(params, state)
        grad_imag_tree = jax.grad(lambda p, s: jnp.sum(jnp.imag(apply_fun(p, s))))(params, state)
        return grad_real_tree, grad_imag_tree

    @partial(jax.jit, static_argnames=('apply_fun',))
    def _compute_numerical_grad_real_part(apply_fun: Callable[[Any, Any], jnp.ndarray],
                                            params: Any,
                                            state: Any):
        """
        Computes PyTree for ∇_p Re[f] using jax.grad.

        Used when only the gradient of the real part is needed, typically
        for real-valued functions or when optimizing only the real part.

        Parameters
        ----------
        apply_fun : Callable[[Any, Any], jnp.ndarray]
            The function f(p, s) to differentiate. Must be JAX traceable.
        params : Any
            The parameters `p` (PyTree) of the function.
        state : Any
            The single input state `s` to the function.

        Returns
        -------
        Any
            grad_real_tree: PyTree matching `params`, containing ∇_p(sum(Re[f(p, s)])).
        """
        # Gradient of the real part w.r.t. the first argument (params)
        grad_real_tree = jax.grad(lambda p, s: jnp.sum(jnp.real(apply_fun(p, s))))(params, state)
        return grad_real_tree

    # -----------------------------------------------------------------------------
    #! Flatten the gradients
    # -----------------------------------------------------------------------------
    
    # @jax.jit
    def flatten_gradient_pytree(grad_pytree: Any) -> jnp.ndarray:
        """
        Flattens a PyTree of gradients into a single 1D JAX array.
        
        Each leaf is flattened in C order (row-major), meaning that if a leaf is a vector it
        remains as is, and if it is a matrix it is flattened as [row0, row1, ...].
        
        Parameters
        ----------
        grad_pytree : Any
            A PyTree containing gradient arrays (leaves).
        
        Returns
        -------
        jnp.ndarray
            A 1D array containing all gradient elements concatenated.
        """
        # Flatten the tree; tree_flatten returns leaves in a deterministic (depth-first) order.
        leaves, tree = jax.tree_util.tree_flatten(grad_pytree)
        if not leaves:
            return jnp.array([], dtype=jnp.float32)  # or your DEFAULT_JP_FLOAT_TYPE

        # For each leaf, reshape to 1D (C order is the default in JAX)
        shapes      = [leaf.shape for leaf in leaves]
        sizes       = [leaf.size for leaf in leaves]
        flat_leaves = [leaf.ravel() for leaf in leaves]
        complexity  = [jnp.iscomplexobj(leaf) for leaf in leaves]
        
        # jax.debug.print("jax flatten gradient pytree: {}", flat_leaves)
        # jax.debug.print("jax flatten gradient pytree shapes: {}", shapes)
        # jax.debug.print("jax flatten gradient pytree sizes: {}", sizes)
        # jax.debug.print("jax flatten gradient pytree complexity: {}", complexity)
        
        # Concatenate all 1D arrays together.
        return jnp.concatenate(flat_leaves), shapes, sizes, complexity

    # ---
    
    # @jax.jit
    def flatten_gradient_pytree_real(grad_pytree: Any) -> jnp.ndarray:
        """
        Flattens a PyTree of real gradients into a single 1D JAX array (float).

        Parameters
        ----------
        grad_pytree : Any
            A PyTree containing real gradient arrays (leaves).

        Returns
        -------
        jnp.ndarray
            A 1D float array containing all gradient elements concatenated.
        """
        leaves, _ = tree_flatten(grad_pytree)
        if not leaves:
            return jnp.array([], dtype=DEFAULT_JP_FLOAT_TYPE)

        # Ravel and ensure float type for all leaves before concatenating
        flat_leaves = [leaf.ravel().astype(DEFAULT_JP_FLOAT_TYPE) for leaf in leaves]

        return jnp.concatenate(flat_leaves)
    
    # ---
    
    # @jax.jit
    def flatten_gradient_pytree_complex(grad_pytree: Any) -> jnp.ndarray:
        """
        Flattens a PyTree of complex gradients into a single 1D JAX array (complex).

        Parameters
        ----------
        grad_pytree : Any
            A PyTree containing complex gradient arrays (leaves).

        Returns
        -------
        jnp.ndarray
            A 1D complex array containing all gradient elements concatenated.
        """
        # leaves, _ = tree_flatten(grad_pytree)
        # if not leaves:
        #     return jnp.array([], dtype=DEFAULT_JP_CPX_TYPE)

        # # Ravel and ensure complex type for all leaves before concatenating
        # flat_leaves = [leaf.ravel().astype(DEFAULT_JP_CPX_TYPE) for leaf in leaves]

        # return jnp.concatenate(flat_leaves)

        real_leaves = tree_leaves(tree_map(lambda x: jnp.real(x).ravel(), grad_pytree))
        imag_leaves = tree_leaves(tree_map(lambda x: jnp.imag(x).ravel(), grad_pytree))

        if not real_leaves and not imag_leaves: # Should have same structure
            return jnp.array([], dtype=DEFAULT_JP_FLOAT_TYPE)

        # Concatenate all real parts, then all imaginary parts
        flat_real_grad = jnp.concatenate(real_leaves + imag_leaves)

        return flat_real_grad.astype(DEFAULT_JP_FLOAT_TYPE)

    # ---

    # @jax.jit
    def flatten_gradient_pytree_holo(complex_grad_pytree: Any) -> jnp.ndarray:
        """
        Flattens a complex PyTree (like ∇_{p*} f) into `[Re(g)..., Im(g)...]`.

        This converts the result of a holomorphic gradient calculation (which is
        a complex PyTree) into the real vector format often required by optimizers
        that operate on real parameter vectors.

        Parameters
        ----------
        complex_grad_pytree : Any
            PyTree containing complex gradient arrays (e.g., from holomorphic grad).

        Returns
        -------
        jnp.ndarray
            A 1D *real* array (float dtype) formatted as `[real_parts..., imag_parts...]`.
        """
        # Extract real and imaginary parts, flatten them individually
        real_leaves = tree_leaves(tree_map(lambda x: jnp.real(x).ravel(), complex_grad_pytree))
        imag_leaves = tree_leaves(tree_map(lambda x: jnp.imag(x).ravel(), complex_grad_pytree))

        if not real_leaves and not imag_leaves: # Should have same structure
            return jnp.array([], dtype=DEFAULT_JP_FLOAT_TYPE)

        # Concatenate all real parts, then all imaginary parts
        flat_real_grad = jnp.concatenate(real_leaves + imag_leaves)

        return flat_real_grad.astype(DEFAULT_JP_FLOAT_TYPE)

    # -----------------------------------------------------------------------------
    #! PyTree gradients
    # -----------------------------------------------------------------------------
    
    # @partial(jax.jit, static_argnums=(0,))
    def pytree_gradient_real_jax(apply_fun  : Callable[[Any, Any], jnp.ndarray],
                                params      : Any,
                                state       : Any) -> Any:
        """
        Compute PyTree of ∇_p Re[f] numerically. Assumes real parameters or
        interest only in gradient w.r.t real part of output.

        Math: Computes g = ∇_p sum(Re[f(p, s)]).

        Parameters
        ----------
        apply_fun : Callable[[Any, Any], jnp.ndarray]
            Network apply function f(p, s).
        params : Any
            Network parameters `p` (PyTree).
        state : Any
            Input state `s`.

        Returns
        -------
        Any
            Real gradient PyTree `g`, matching `params` structure.
        """
        # Directly computes gradient of the real part
        g_tree = _compute_numerical_grad_real_part(apply_fun, params, state)
        # Returns the PyTree structure, flattening happens later if needed
        return g_tree
    
    # ---

    # @partial(jax.jit, static_argnums=(0,))
    def pytree_gradient_cpx_nonholo_jax(apply_fun   : Callable[[Any, Any], jnp.ndarray],
                                        params      : Any,
                                        state       : Any) -> Any:
        r"""
        Compute non-holomorphic conjugate gradient (PyTree): (∇_p f)* = ∇_p Re[f] - i ∇_p Im[f].

        Why this gradient?
        -------------------
        When dealing with a real-valued loss function L(p) that depends on complex
        parameters p (e.g., L = <E_L> in VMC), the gradient descent update rule
        should ideally move `p` in the direction that maximally decreases `L`.
        Using Wirtinger calculus, the direction of steepest descent for `L` with
        respect to `p` is given by -∇_{p*} L, where ∇_{p*} = ∂/∂p* = (∂/∂Re[p] + i ∂/∂Im[p])/2.
        If L depends on a complex function f(p) (like log Psi(p)), such that L = Re[g(f(p))] or |f(p)|^2,
        the chain rule involves terms like ∂f/∂p* and ∂f/∂p.
        
        For many physical systems/wavefunctions (non-holomorphic cases), the relevant gradient
        component driving the optimization of the real loss function turns out to be proportional
        to ∇_p Re[f] - i ∇_p Im[f], which is precisely (∇_p f)^*. This is the gradient
        conjugate to the standard complex gradient ∇_p f = ∇_p Re[f] + i ∇_p Im[f].
        In VMC, this often corresponds to (∇_p log Psi)*.

        Math: Computes g_R = ∇_p sum(Re[f(p, s)]) and g_I = ∇_p sum(Im[f(p, s)]).
            Returns the complex PyTree g = g_R - i g_I.

        Parameters
        ----------
        apply_fun : Callable[[Any, Any], jnp.ndarray]
            Network apply function f(p, s).
        params : Any
            Network parameters `p` (PyTree).
        state : Any
            Input state `s`.

        Returns
        -------
        Any
            Complex conjugate gradient PyTree `g`, matching `params` structure.
        """
        gr_tree, gi_tree = _compute_numerical_grad_parts(apply_fun, params, state)

        # Combine: Real part is gr_tree, Imaginary part is -gi_tree
        def combine_conjugate(gr, gi):
            # Ensure complex output type even if inputs are real
            return gr.astype(DEFAULT_JP_CPX_TYPE) - 1j * gi.astype(DEFAULT_JP_CPX_TYPE)

        return tree_map(combine_conjugate, gr_tree, gi_tree)
    
    # ---
    
    # @partial(jax.jit, static_argnums=(0,))
    def pytree_gradient_cpx_holo_jax(apply_fun  : Callable[[Any, Any], jnp.ndarray],
                                    params      : Any,
                                    state       : Any) -> Any:
        r"""
        Compute holomorphic gradient (PyTree): ∇_{p*} f using `holomorphic=True`.

        Why this gradient?
        -------------------
        A function f(p) of a complex variable p is holomorphic if it is complex
        differentiable in a neighborhood of every point in its domain. This implies
        it satisfies the Cauchy-Riemann equations (∂Re[f]/∂Re[p] = ∂Im[f]/∂Im[p] and
        ∂Re[f]/∂Im[p] = -∂Im[f]/∂Re[p]).
        If f is holomorphic w.r.t. p, its local behavior is simple:
            like multiplication
            by a complex number df/dp. The variation of f is fully captured by a single
            complex derivative.
        
        Using Wirtinger derivatives (∂/∂p = (∂/∂Re[p] - i ∂/∂Im[p])/2 and
        ∂/∂p* = (∂/∂Re[p] + i ∂/∂Im[p])/2), holomorphicity means ∂f/∂p = 0.
        
        The entire change in f comes from ∂f/∂p*.
        `jax.grad(..., holomorphic=True)` directly computes ∂f/∂p* (often denoted ∇_{p*} f).
        This is the relevant gradient if your function f is known to be holomorphic
        in the parameters `p` you are differentiating with respect to.

        Math: Computes g = ∇_{p*} sum(f(p, s)) using `jax.grad(..., holomorphic=True)`.

        Parameters
        ----------
        apply_fun : Callable[[Any, Any], jnp.ndarray]
            Network apply function f(p, s), assumed holomorphic in `p`.
        params : Any
            Network parameters `p` (PyTree).
        state : Any
            Input state `s`.

        Returns
        -------
        Any
            Complex holomorphic gradient PyTree `g`, matching `params` structure.
        """
        # Compute the gradient w.r.t. params (1st arg), assuming holomorphicity
        # The result is df/dp*
        holo_grad_tree = jax.grad(lambda p, s: jnp.sum(apply_fun(p, s)),
                                argnums     =   0, # Differentiate w.r.t. params
                                holomorphic =   True)(params, state)
        # Returns the complex PyTree structure
        return holo_grad_tree

    # ---
    
    # @partial(jax.jit, static_argnames=('a_grad_fun',))
    def pytree_gradient_analytic_jax(a_grad_fun: Callable[[Any, Any], jnp.ndarray],
                                    params      : Any,
                                    state       : Any) -> Any:
        """
        Compute PyTree of the analytic gradient using the provided function.

        Parameters
        ----------
        a_grad_fun : Callable[[Any, Any], jnp.ndarray]
            The function to compute the gradient. Should return a PyTree.
        params : Any
            Network parameters `p` (PyTree).
        state : Any
            Input state `s`.

        Returns
        -------
        Any
            Gradient PyTree `g`, matching `params` structure.
        """
        # Compute the gradient using the provided function
        grad_pytree = a_grad_fun(params, state)
        # Returns the PyTree structure
        return grad_pytree
    
    # -----------------------------------------------------------------------------
    #! Flattened gradients
    # -----------------------------------------------------------------------------
    
    # @partial(jax.jit, static_argnames=('apply_fun',))
    def flat_gradient_real_jax(apply_fun    : Callable[[Any, Any], jnp.ndarray],
                                params      : Any,
                                state       : Any) -> jnp.ndarray:
        """
        Compute flattened real gradient ∇_p Re[f]. Output is 1D float array.
        """
        grad_pytree = pytree_gradient_real_jax(apply_fun, params, state)
        flat, shapes, size, is_cpx = flatten_gradient_pytree(grad_pytree)
        return flat, shapes, size, is_cpx
    
    # return flatten_gradient_pytree_real(grad_pytree)

    # @partial(jax.jit, static_argnames=('apply_fun',))
    def flat_gradient_cpx_nonholo_jax(apply_fun : Callable[[Any, Any], jnp.ndarray],
                                        params  : Any,
                                        state   : Any) -> jnp.ndarray:
        """
        Compute flattened non-holomorphic conjugate gradient (∇_p f)*.
        Output is 1D complex array.
        """
        grad_pytree = pytree_gradient_cpx_nonholo_jax(apply_fun, params, state)
        flat, shapes, sizes, is_cpx = flatten_gradient_pytree(grad_pytree)
        return flat, shapes, sizes, is_cpx
        # return flatten_gradient_pytree_complex(grad_pytree)

    # @partial(jax.jit, static_argnames=('apply_fun',))
    def flat_gradient_cpx_holo_jax(apply_fun    : Callable[[Any, Any], jnp.ndarray],
                                    params      : Any,
                                    state       : Any) -> jnp.ndarray:
        """
        Compute flattened holomorphic gradient ∇_{p*} f, formatted as
        a 1D *real* array `[Re(g)..., Im(g)...]`.
        """
        grad_pytree = pytree_gradient_cpx_holo_jax(apply_fun, params, state)
        # This specific flattener handles the complex -> [Re, Im] real format
        flat, shapes, sizes, is_cpx = flatten_gradient_pytree(grad_pytree)
        return flat, shapes, sizes, is_cpx 
        # return flatten_gradient_pytree_holo(grad_pytree)

    # @partial(jax.jit, static_argnames=('a_grad_fun',))
    def flat_gradient_analytic_jax(a_grad_fun   : Callable[[Any, Any], jnp.ndarray],
                                    params      : Any,
                                    state       : Any) -> jnp.ndarray:
        """
        Compute flattened analytic gradient using provided function.
        Output is 1D float array.
        """
        grad_pytree = a_grad_fun(params, state)
        # # jax.debug.print("jax pytree: {}", grad_pytree)
        flat, shapes, sizes, is_cpx = flatten_gradient_pytree(grad_pytree)
        return flat, shapes, sizes, is_cpx
        # if any(jnp.iscomplexobj(leaf) for leaf in tree_leaves(grad_pytree)):
            # out = flatten_gradient_pytree_complex(grad_pytree)
            # # jax.debug.print("jax flattened complex: {}", out)
            # return out
        # return flatten_gradient_pytree_real(grad_pytree)
    
    # ----------------------------------------------------------------------------
    #! Compute the gradient
    # ----------------------------------------------------------------------------
    
    # @partial(jax.jit, static_argnums=(0, 3, 4))                                         # apply_fun and single_sample_flat_grad_fun are static
    def compute_gradients_batched(
            net_apply                   : Callable[[Any, Any], jnp.ndarray],            # Network apply function f(p, s).
            params                      : Any,                                          # Network parameters `p` (PyTree).
            states                      : jnp.ndarray,                                  # Expects shape (n_samples, ...)
            single_sample_flat_grad_fun : Callable[[Callable, Any, Any], jnp.ndarray],
            batch_size                  : int = 1,                                      # Batch size for gradient computation.
        ) -> jnp.ndarray:
        """
        Compute flattened gradients per sample over a batch of states using jax.vmap.

        Applies the provided `single_sample_flat_grad_fun` (which itself calls
        `net_apply`) to each state vector in `states`. Designed for JIT compilation.

        Parameters
        ----------
        net_apply : Callable[[Any, Any], jnp.ndarray]
            Network apply function f(p, s). Passed to `single_sample_flat_grad_fun`.
            *Must be static* for JIT.
        params : Any
            Network parameters `p` (PyTree). Passed to `single_sample_flat_grad_fun`.
        states : jnp.ndarray
            Input states, shape `(num_samples, ...state_dims)`.
        single_sample_flat_grad_fun : Callable[[Callable, Any, Any], jnp.ndarray]
            A JIT-compatible function computing the **flattened** gradient for a
            **single** input state. Signature should be compatible with one of:
            `flat_gradient_real_jax`, `flat_gradient_cpx_nonholo_jax`, `flat_gradient_cpx_holo_jax`.
            It takes `(net_apply, params, single_state)` and returns a 1D gradient vector.
            *Must be static* for JIT.
        batch_size : int, optional
            Batch size for gradient computation. Default is 1 (no batching).

        Returns
        -------
        jnp.ndarray
            Array of flattened gradients, shape `(num_samples, num_flat_params)`.
            Dtype matches the output of `single_sample_flat_grad_fun`.
        """
        if states.ndim == 0 or states.shape[0] == 0:
            raise ValueError("Input states must have at least one sample.")

        # get full info from the first state
        gradients, shapes, sizes, is_cpx = single_sample_flat_grad_fun(net_apply, params, states[0])

        # Define the function to be vmapped. It takes params and a single state.
        # net_apply and single_sample_flat_grad_fun are closed over or passed statically.
        def compute_grad_for_one_state(p, s):
            return single_sample_flat_grad_fun(net_apply, p, s)[0]

        # Create batches of states
        batches = create_batches_jax(states, batch_size)

        # vmap this function.
        # - `in_axes=(None, 0)` means:
        #   - Don't map over the first argument (`params`). Pass the whole thing.
        #   - Map over the first dimension (axis 0) of the second argument (`states`).
        # - `out_axes=0` means the output gradients should be stacked along axis 0.
        batch_grads_fun = jax.vmap(compute_grad_for_one_state, in_axes=(None, 0), out_axes=0)

        # Apply the vmapped function to the parameters and the batch of states
        gradients = jax.vmap(batch_grads_fun, in_axes=(None, 0), out_axes=0)(params, batches)
        # # jax.debug.print("jax compute gradients: {}", gradients[0])
        # Reshape gradients to match the original number of samples
        gradients = gradients.reshape(-1, gradients.shape[-1])      # flatten first
        gradients = gradients[: states.shape[0]]                    # drop padding
        return gradients, shapes, sizes, is_cpx

    
    # ----------------------------------------------------------------------------
    #! Transform the vector from the original representation to the real representation [Re..., Im...]
    # ----------------------------------------------------------------------------
    
    @partial(jax.jit, static_argnames=('force_complex',))
    def _to_real_repr_single(vector: jnp.ndarray, force_complex: bool) -> jnp.ndarray:
        if jnp.iscomplexobj(vector):
            real_parts      = jnp.real(vector)
            imag_parts      = jnp.imag(vector)
            # # jax.debug.print("jax to real repr complex: {}", vector)
            # # jax.debug.print("jax to real repr real: {}", real_parts)
            # # jax.debug.print("jax to real repr imag: {}", imag_parts)
            out             = jnp.concatenate([real_parts, imag_parts])
            # # jax.debug.print("jax to real repr out: {}", out)
            return out
        elif force_complex:
            complex_vector  = vector.astype(DEFAULT_JP_CPX_TYPE)
            real_parts      = jnp.real(complex_vector)
            imag_parts      = jnp.imag(complex_vector)
            return jnp.concatenate([real_parts, imag_parts]).astype(DEFAULT_JP_FLOAT_TYPE)
        else:
            return vector.astype(DEFAULT_JP_FLOAT_TYPE)
    
    # map the single-vector function over the first axis (vectors), keep template fixed (None)
    _to_real_repr_batch = jax.vmap(_to_real_repr_single, in_axes=(0, None), out_axes=0)
    
    @partial(jax.jit, static_argnames=('force_complex',))
    def to_real_representation(vectors: jnp.ndarray, force_complex: bool = False) -> jnp.ndarray:
        # Keep ndim check for safety
        if vectors.ndim == 1:
            return _to_real_repr_single(vectors, force_complex)
        if vectors.ndim == 2:
            return _to_real_repr_batch(vectors, force_complex)
        raise ValueError(f"Input must be 1D or 2D, got ndim={vectors.ndim}")

    # ----------------------------------------------------------------------------
    #! Transform the vector from the real representation [Re..., Im...] to the original representation
    # ----------------------------------------------------------------------------

    @partial(jax.jit, static_argnames=('target_is_complex',))
    def _from_real_repr_single_impl(real_repr_vector: jnp.ndarray, target_is_complex: bool) -> jnp.ndarray:
        '''
        Converts a single real vector based on target complexity.
        
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
        '''
        
        # Keep even size check for complex target - crucial for correctness
        if target_is_complex:
            vector_size = real_repr_vector.size
            if vector_size == 0:
                return jnp.array([], dtype=DEFAULT_JP_CPX_TYPE)
            if vector_size % 2 != 0:
                raise ValueError("Input vector size must be even if target is complex.")
            n           = vector_size // 2
            real_vector = real_repr_vector.astype(DEFAULT_JP_FLOAT_TYPE)
            out         = (real_vector[:n] + 1.j * real_vector[n:]).astype(DEFAULT_JP_CPX_TYPE)
            # # jax.debug.print("jax from real repr complex: {}", out)
            return out
        else:
            if real_repr_vector.size == 0:
                return jnp.array([], dtype=DEFAULT_JP_FLOAT_TYPE)
            return real_repr_vector.astype(DEFAULT_JP_FLOAT_TYPE)

    def _get_template_info(params_template: Any) -> Tuple[bool, int]:
        '''
        Determines if the parameter template is complex or real.
        Returns a tuple (is_complex, total_real_elements).
        is_complex:
            True if any leaf in the template is complex.
        total_real_elements:
            Total number of real elements in the template.
        '''
        template_leaves     = tree_leaves(params_template)
        if not template_leaves:
            return False, 0
        is_complex          = any(jnp.iscomplexobj(leaf) for leaf in template_leaves)
        total_real_elements = sum(leaf.size * (2 if jnp.iscomplexobj(leaf) else 1) for leaf in template_leaves)
        return is_complex, total_real_elements

    def _from_real_repr_single(real_repr_vector: jnp.ndarray, params_template: Any) -> jnp.ndarray:
        
        # check the target
        target_is_complex, expected_real_repr_size = _get_template_info(params_template)

        if real_repr_vector.size != expected_real_repr_size and not (real_repr_vector.size == 0 and expected_real_repr_size == 0):
            if real_repr_vector.size == expected_real_repr_size // 2 and target_is_complex:
                # If the real_repr_vector is half the expected size and the target is complex,
                # it might be already in the correct format.
                return _from_real_repr_single_impl(real_repr_vector, target_is_complex)
            raise ValueError(f"Input vector size ({real_repr_vector.size}) != expected ({expected_real_repr_size}) from template.")
        return _from_real_repr_single_impl(real_repr_vector, target_is_complex)

    _from_real_repr_batch = jax.vmap(_from_real_repr_single, in_axes=(0, None), out_axes=0)

    def from_real_representation(real_repr_vectors: jnp.ndarray, params_template: Any) -> jnp.ndarray:
        if real_repr_vectors.ndim == 1:
            return _from_real_repr_single(real_repr_vectors, params_template)
        if real_repr_vectors.ndim == 2:
            return _from_real_repr_batch(real_repr_vectors, params_template)
        raise ValueError(f"Input must be 1D or 2D, got ndim={real_repr_vectors.ndim}")

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
                        batch_size: int = 1,
                        *op_args):
        """
        Applies a transformation function to each state and computes a locally
        weighted estimate as described in [2108.08631].
        
        Parameters:
            - func: Callable that accepts a state (vector) and optional *op_args, returning a tuple (S, V)
                    where S is an array of modified states (M x state_size) and V is an
                    array of corresponding values (M,).
            - states: Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
            - sample_probas: Array of sampling probabilities (matching the leading dimensions of states).
            - logprobas_in: Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
            - logproba_fun: Callable that computes the log-probabilities for given states S.
            - batch_size: Size of batches to process states in - dummy for now.
            - *op_args: Additional arguments to pass to func. These are broadcast across states.
            
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
        def compute_estimate(state, logp, sample_p, *args):
            # Squeeze the scalar values.
            logp                    = logp[0]
            sample_p                = sample_p[0]
            
            # Obtain modified states and corresponding values.
            new_states, new_vals    = func(state, *args) if args else func(state)
            new_logprobas           = logproba_fun(parameters, new_states)
            # weights                 = jnp.exp(mu * (new_logprobas - logp))
            weights                 = jnp.exp(new_logprobas - logp)
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            
            # Return the weighted sum.
            return weighted_sum * sample_p
        
        # Build in_axes: (0, 0, 0) for state, logp, sample_p, then None for each op_arg
        in_axes_config  = (0, 0, 0) + (None,) * len(op_args)
        applied         = jax.vmap(compute_estimate, in_axes=in_axes_config)(states, logprobas_in, sample_probas, *op_args)
        return applied, jnp.mean(applied, axis = 0), jnp.std(jnp.real(applied), axis = 0)
    
    # @partial(jax.jit, static_argnums=(0,3))
    def apply_callable_jax_uniform(func, states, logprobas_in, logproba_fun, parameters, mu = 2.0):
        """
        Applies a transformation function to each state and computes a locally
        weighted estimate as described in [2108.08631].
        
        Parameters:
            - func: 
                Callable that accepts a state (vector) and returns a tuple (S, V)
                where S is an array of modified states (M x state_size) and V is an
                array of corresponding values (M,).
            - states:
                Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
            - logprobas_in:
                Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
            - logproba_fun:
                Callable that computes the log-probabilities for given states S.
            
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
            weights                 = jnp.exp(mu * (new_logprobas - logp))
            weighted_sum            = jnp.sum(new_vals * weights, axis=0)
            return weighted_sum
        
        applied = jax.vmap(compute_estimate, in_axes=(0, 0))(states, logprobas_in)
        return applied, jnp.mean(applied, axis = 0), jnp.std(applied, axis = 0)

    def apply_callable_batched_jax(func,
                                states,
                                sample_probas,
                                logprobas_in,
                                logproba_fun,
                                parameters,
                                batch_size : int,
                                *op_args):
        r"""
        Batched application of a local estimator.

        For each input state :math:`s` this computes

        .. math::

            \hat O(s) = p_{\mathrm{samp}}(s)
            \sum_{k} v_k(s) \exp\left[\log\psi(s_k) - \log\psi(s)\right],

        where :math:`s_k` and :math:`v_k` come from ``func`` and
        :math:`\log\psi` comes from ``logproba_fun``.
        """

        # ----------------------------------------------------------------------
        # Flatten over (samples, chains, ...) -> (N, D)
        # ----------------------------------------------------------------------
        flat_states   = states.reshape(-1, states.shape[-1])   # (N, D)
        flat_logp_in  = logprobas_in.reshape(-1)               # (N,)
        flat_samp_p   = sample_probas.reshape(-1)              # (N,)

        # N, pad, N_pad are ordinary Python ints -> OK for jnp.pad
        N             = flat_states.shape[0]
        pad           = (-N) % batch_size          # 0 … batch_size-1
        N_pad         = N + pad
        n_batches     = N_pad // batch_size

        if pad:
            pad_spec_states  = ((0, pad), (0, 0))
            pad_spec_1d      = ((0, pad),)
            p_states         = jnp.pad(flat_states,  pad_spec_states)
            p_logp_in        = jnp.pad(flat_logp_in, pad_spec_1d)
            p_samp_p         = jnp.pad(flat_samp_p,  pad_spec_1d)
        else:
            p_states         = flat_states
            p_logp_in        = flat_logp_in
            p_samp_p         = flat_samp_p

        # ----------------------------------------------------------------------
        # Per–state estimator
        # ----------------------------------------------------------------------
        def _estimate_one(state, logp0, p_sample, *args):
            new_states, new_vals = func(state, *args) if args else func(state)
            new_states           = jnp.asarray(new_states)
            new_vals             = jnp.asarray(new_vals)

            logp_new             = logproba_fun(parameters, new_states)
            w                    = jnp.exp(logp_new - logp0)      # same shape as new_vals

            # scalar for this original state
            return p_sample * jnp.sum(new_vals * w)

        in_axes      = (0, 0, 0) + (None,) * len(op_args)
        batch_kernel = jax.vmap(_estimate_one, in_axes=in_axes, out_axes=0)

        # ----------------------------------------------------------------------
        # Scan over batches, writing directly into a length-N_pad buffer
        # ----------------------------------------------------------------------
        # infer dtype once from a tiny dummy call
        dummy_out    = batch_kernel(p_states[:1], p_logp_in[:1], p_samp_p[:1], *op_args)
        init_out     = jnp.empty((N_pad,), dtype=dummy_out.dtype)

        def body(carry, idx):
            start   = idx * batch_size

            bs_s    = lax.dynamic_slice_in_dim(p_states,  start, batch_size, axis=0)
            bs_lp   = lax.dynamic_slice_in_dim(p_logp_in, start, batch_size, axis=0)
            bs_sp   = lax.dynamic_slice_in_dim(p_samp_p,  start, batch_size, axis=0)

            chunk   = batch_kernel(bs_s, bs_lp, bs_sp, *op_args)   # (batch_size,)
            carry   = lax.dynamic_update_slice(carry, chunk, (start,))
            return carry, None

        estimates_pad, _ = lax.scan(body, init_out, xs=jnp.arange(n_batches))
        estimates        = estimates_pad[:N]  # drop padded tail

        return estimates, jnp.mean(estimates), jnp.std(estimates)
        
    # @partial(jax.jit, static_argnums=(0,3,5))
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
#! ADD AND UPDATE
# -------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    class LeafInfo(NamedTuple):
        """Holds static information about a parameter leaf."""
        shape       : Tuple[int, ...]
        is_complex  : bool
        num_elements: int # Pre-calculated product of shape

    class SliceInfo(NamedTuple):
        """Holds slicing information derived from LeafInfo."""
        start       : int
        size        : int # Number of real components in the flat vector
        shape       : Tuple[int, ...]
        is_complex  : bool

    def prepare_leaf_info(params: Any) -> Tuple[LeafInfo, ...]:
        """
        Extracts static LeafInfo (shape, complexity, size) from a parameter PyTree.
        This should be called once when the network structure is known/stable.
        
        Parameters
        ----------
        params : Any
            Network parameters `p` (PyTree).
        
        Returns
        -------
        Tuple[LeafInfo, ...]
            A tuple containing LeafInfo for each parameter leaf.
        """
        flat_leaves, l_tree = tree_flatten(params)
        
        # Check if any leaves are complex
        leaf_info_list      = []
        for i, leaf in enumerate(flat_leaves):
            shape           = tuple(leaf.shape)
            is_complex      = jnp.iscomplexobj(leaf)
            num_elements    = int(jnp.prod(jnp.array(shape)))
            leaf_info_list.append(LeafInfo(shape=shape, is_complex=is_complex, num_elements=num_elements))
        return tuple(leaf_info_list)

    def prepare_unflatten_metadata_from_leaf_info(leaf_info: Tuple[LeafInfo, ...]) -> Tuple[SliceInfo, ...]:
        """
        Computes SliceInfo metadata efficiently from pre-computed LeafInfo.
        This is very fast as it only involves arithmetic on static info.
        Not typically JITted as it produces static metadata.

        Parameters
        ----------
        leaf_info : Tuple[LeafInfo, ...]
            Static information about each parameter leaf.

        Returns
        -------
        Tuple[SliceInfo, ...]
            Metadata tuple for use in fast_unflatten.
        """
        slice_metadata          = []
        current_start_index     = 0
        for info in leaf_info:
            # Calculate size in the real representation vector
            num_real_components = info.num_elements * (2 if info.is_complex else 1)

            slice_metadata.append(SliceInfo(
                start       =   current_start_index,
                size        =   num_real_components,
                shape       =   info.shape,
                is_complex  =   info.is_complex                
            ))
            current_start_index += num_real_components
        # Check total size consistency (optional sanity check, assumes flat vec exists)
        # total_expected_size = current_start_index
        # if flat_vector is not None and flat_vector.size != total_expected_size:
        #    raise ValueError(...)

        return tuple(slice_metadata)
    
    def prepare_slice_info(shapes, sizes, is_cpx):
        """
        Converts the shapes, sizes, and is_cpx lists into a list of SliceInfo objects.
        This is a simple transformation for easier handling of the metadata.

        Parameters
        ----------
        shapes : Tuple[Tuple[int, ...], ...]
            Shapes of the parameter leaves.
        sizes : Tuple[int, ...]
            Sizes of the parameter leaves.
        is_cpx : Tuple[bool, ...]
            Complexity (real/complex) of the parameter leaves.

        Returns
        -------
        List[SliceInfo]
            A list of SliceInfo objects.
        """
        start               = 0
        slice_metadata      = []
        for i in range(len(shapes)):
            # Force each element of the shape tuple to be a Python int.
            py_shape        = tuple(int(x) for x in shapes[i])
            py_size         = int(sizes[i])
            py_is_cpx       = bool(is_cpx[i])
            # If the leaf is complex, size is multiplied by 2.
            effective_size  = py_size * (2 if py_is_cpx else 1)
            slice_metadata.append(SliceInfo(start=start, size=effective_size, shape=py_shape, is_complex=py_is_cpx))
            start           += effective_size
        return tuple(slice_metadata)
        
    # --------------------------------------------------------------------------
    
    # @partial(jax.jit, static_argnums=(1,))
    def fast_unflatten(
        flat_real_vector    : jnp.ndarray,
        slice_metadata      : Tuple[SliceInfo, ...]) -> List[jnp.ndarray]:
        
        """
        Optimized JIT-compiled unflattening of a real flat vector into parameter leaves.

        Parameters
        ----------
        flat_real_vector : jnp.ndarray
            The 1D vector containing parameters in real representation format.
            Assumed to have the correct size matching `slice_metadata`.
        slice_metadata : Tuple[SliceInfo, ...]
            Static metadata generated by `prepare_unflatten_metadata_from_leaf_info`.

        Returns
        -------
        List[jnp.ndarray]
            A list of parameter leaves (JAX arrays) in the correct order.
        """
        leaves      = []
        full_size   = sum(slice_info.size for slice_info in slice_metadata)
        half_size   = full_size // 2
        
        # # jax.debug.print("jax fast unflatten full size: {}", full_size)
        # # jax.debug.print("jax fast unflatten half size: {}", half_size)
        # jax.debug.print("jax fast unflatten flat vector: {}", flat_real_vector)
        
        start_idx   = 0
        for slice_info in slice_metadata:
            leaf        = None
            size_in     = slice_info.size
            shape_in    = slice_info.shape
            is_cpx_in   = slice_info.is_complex
            try:
                if is_cpx_in:
                    idx_re      = start_idx
                    idx_im      = start_idx + half_size
                    size_in     = size_in // 2
                    reals       = jax.lax.dynamic_slice_in_dim(flat_real_vector, idx_re, size_in, axis=0)
                    imags       = jax.lax.dynamic_slice_in_dim(flat_real_vector, idx_im, size_in, axis=0)

                    # re, im = jnp.split(segment, 2) # Alternative
                    leaf        = (reals + 1j * imags)
                    leaf        = leaf.reshape(shape_in)
                    # jax.debug.print(
                    #     "jax fast unflatten complex leaf: {}, shape: {}, size: {}, idx_re: {}, idx_im: {}, after reshape: {}",
                    #     leaf, leaf.shape, leaf.size, idx_re, idx_im, leaf
                    # )
                else:
                    idx_start   = start_idx
                    leaf        = jax.lax.dynamic_slice_in_dim(flat_real_vector, idx_start, shape_in, axis=0).reshape(shape_in)
            except Exception as e:
                return leaves
            start_idx += size_in
            leaves.append(leaf)
        return leaves

    # --------------------------------------------------------------------------

    # @partial(jax.jit, static_argnames=('params_tree_def', 'params_slice_metadata', 'params_total_size'))
    # def transform_flat_params_jit(d_par, params_tree_def, params_slice_metadata, params_total_size):
    # @partial(jax.jit, static_argnames=('params_tree_def', 'params_slice_metadata', 'params_total_size'))
    def transform_flat_params_jit(d_par, params_tree_def, params_slice_metadata, params_total_size):
        """
        JIT-compiled helper to transform the flat parameter update vector
        into the model's parameter structure.
        
        Args:
            d_par (jnp.ndarray):
                A 1D flat parameter update vector.
            params_tree_def:
                The tree definition that the update must be unflattened into.
            params_slice_metadata:
                Slicing information to rebuild the parameter tree.
            params_total_size (int):
                Expected total size of the flat parameters.
            
        Returns:
            The parameter update tree reconstructed from d_par.
            
        Raises:
            ValueError:
                If the input vector does not have the correct shape/size.
        """
        if d_par.ndim != 1:
            # Note: In a pure jitted function, raising exceptions is possible
            # but may cause compilation if the condition is ever True.
            raise ValueError(f"Flat parameter update `d_par` must be 1D, got shape {d_par.shape}.")

        # Convert to the real representation vector.
        # The call to to_real_representation should be pure.
        flat_real_update_vector = to_real_representation(d_par)

        if flat_real_update_vector.size != params_total_size:
            raise ValueError(f"Flat update vector size ({flat_real_update_vector.size}) "
                            f"does not match expected size ({params_total_size}) "
                            f"based on model parameters.")

        leaves      = fast_unflatten(flat_real_update_vector, params_slice_metadata)
        update_tree = tree_unflatten(params_tree_def, leaves)
        return update_tree
        # return leaves
    
    # --------------------------------------------------------------------------
    
    @jax.jit
    def add_tree(p1: Any, p2: Any) -> Any:
        """JIT-compiled tree addition."""
        return tree_map(jax.lax.add, p1, p2)
    
    @jax.jit
    def sub_tree(p1: Any, p2: Any) -> Any:
        """JIT-compiled tree subtraction."""
        return tree_map(jax.lax.sub, p1, p2)
    
    @jax.jit
    def mul_tree(p1: Any, multipler: Any) -> Any:
        """JIT-compiled tree multiplication."""
        return tree_map(jax.lax.mul, p1, multipler)
    
    @jax.jit
    def div_tree(p1: Any, divisor: Any) -> Any:
        """JIT-compiled tree division."""
        return tree_map(jax.lax.div, p1, divisor)

# -------------------------------------------------------------------------