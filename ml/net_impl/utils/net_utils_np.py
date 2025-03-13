import numpy as np
import numba
from typing import Union, Callable, Optional, Any

# try to import autograd for numpy
try:
    import autograd.numpy as anp
    from autograd import grad as np_grad
    from autograd.misc.flatten import flatten_func
    AUTOGRAD_AVAILABLE = True
except ImportError:
    AUTOGRAD_AVAILABLE = False
    
#########################################################################
#! BATCHES
#########################################################################

@numba.njit
def create_batches_np(  data        : np.ndarray,
                        batch_size  : int):
    """
    NumPy version of create_batches with manual padding (mode 'edge')
    to be compatible with Numba's nopython mode.
    
    Parameters
    ----------
    data : np.ndarray
        Input array with shape (n_samples, ...).
    batch_size : int
        Desired batch size.
    
    Returns
    -------
    np.ndarray
        Reshaped array of shape (n_batches, batch_size, ...).
        If the first dimension is not a multiple of batch_size, the
        last row is repeated as necessary.
    """
    n           = data.shape[0]
    remainder   = n % batch_size
    pad         = 0
    if remainder != 0:
        pad = batch_size - remainder
    new_n = n + pad

    # Allocate a new array with the padded size.
    # This works for any data.shape[1:].
    out_shape   = (new_n,) + data.shape[1:]
    out         = np.empty(out_shape, dtype=data.dtype)
    
    # Copy the original data.
    for i in range(n):
        out[i] = data[i]
    # For the padded part, copy the last element (edge padding).
    for i in range(n, new_n):
        out[i] = data[n - 1]
    
    # Reshape to batches.
    new_shape = (new_n // batch_size, batch_size) + data.shape[1:]
    return out.reshape(new_shape)

##########################################################################
#! EVALUATE BATCHED
##########################################################################

# @numba.njit
def eval_batched_np(batch_size      : int,
                    func            : Any,
                    params          : Any,
                    data            : np.ndarray):
    """
    NumPy version of eval_batched.
    
    Parameters
    ----------
    batch_size : int
        The size of each batch.
    func : Callable
        The function to evaluate on each individual sample.
    data : np.ndarray
        The input data array.
    
    Returns
    -------
    np.ndarray
        A 1D array containing the results of applying func to each sample.
    
    Example
    -------
    >>> data = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
    >>> # With batch_size=3, data is padded to shape (6,2) and reshaped to (2,3,2)
    >>> result = eval_batched_np(3, lambda x: np.sum(x), data)
    >>> # result is concatenated to a 1D array with 6 elements.
    """
    # Create batches from the data using our NumPy version of create_batches.
    batches = create_batches_np(data, batch_size)
    # For each batch, apply func to each sample using a list comprehension.
    # Then, concatenate the results into one array.
    return np.concatenate([np.array([func(params, x) for x in batch]) for batch in batches])

def eval_batched_np_simple(batch_size      : int,
                            func           : Any,
                            data           : np.ndarray):
    """
    NumPy version of eval_batched without params.
    
    Parameters
    ----------
    batch_size : int
        The size of each batch.
    func : Callable
        The function to evaluate on each individual sample.
    data : np.ndarray
        The input data array.
    
    Returns
    -------
    np.ndarray
        A 1D array containing the results of applying func to each sample.
    
    Example
    -------
    >>> data = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
    >>> result = eval_batched_simple(3, lambda x: np.sum(x), data)
    >>> # result is concatenated to a 1D array with 6 elements.
    """
    batches = create_batches_np(data, batch_size)
    return np.concatenate([np.array([func(x) for x in batch]) for batch in batches])

##########################################################################
#! GRADIENTS
##########################################################################

# ==============================================================================
#! NumPy (Autograd) Implementations: Analytical and Numerical Gradient Functions
# ==============================================================================

if AUTOGRAD_AVAILABLE:
    def flat_gradient_analytical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened complex gradient using an analytical method (NumPy version).

        Assumes fun is callable and has an 'analytical_gradient' attribute.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        grad_val    = fun.gradient(params, arg)
        flat_grad   = np.concatenate([v.ravel() for v in grad_val.values()])
        return flat_grad

    def flat_gradient_numerical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened complex gradient using numerical differentiation (NumPy/Autograd).

        Uses autograd's np_grad for both real and imaginary parts.
        """
        gr = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        gi = np_grad(lambda p, y: anp.imag(fun.apply(p, y)))(params, arg)["params"]
        return np.concatenate([gr.ravel(), 1.j * gi.ravel()])

    def flat_gradient_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> np.ndarray:
        """
        Wrapper for computing a flattened complex gradient using NumPy.
        """
        if analytical:
            return flat_gradient_analytical_np(fun, params, arg)
        else:
            return flat_gradient_numerical_np(fun, params, arg)

    def flat_gradient_cpx_nonholo_analytical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute an analytical flattened complex gradient for non-holomorphic networks (NumPy).
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if not hasattr(fun, "analytical_gradient"):
            raise NotImplementedError("Analytical gradient not implemented for this function.")
        grad_val = fun.gradient(params, arg)
        return np.concatenate([v.ravel() for v in grad_val.values()])

    def flat_gradient_cpx_nonholo_numerical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened complex gradient for non-holomorphic networks using numerical differentiation (NumPy).
        """
        gr = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        gi = np_grad(lambda p, y: anp.imag(fun.apply(p, y)))(params, arg)["params"]
        return np.concatenate([gr.ravel(), 1.j * gi.ravel()])

    def flat_gradient_cpx_nonholo_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> np.ndarray:
        """
        Wrapper for computing a flattened complex gradient for non-holomorphic networks using NumPy.
        """
        if analytical:
            return flat_gradient_cpx_nonholo_analytical_np(fun, params, arg)
        return flat_gradient_cpx_nonholo_numerical_np(fun, params, arg)

    def flat_gradient_real_analytical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute an analytical flattened real gradient using NumPy.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if not hasattr(fun, "analytical_gradient_real"):
            raise NotImplementedError("Analytical real gradient not implemented for this function.")
        grad_val = fun.analytical_gradient_real(params, arg)
        return np.concatenate([v.ravel() for v in grad_val.values()])

    def flat_gradient_real_numerical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened real gradient using numerical differentiation (NumPy).
        """
        g = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        return g.ravel()

    def flat_gradient_real_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> np.ndarray:
        """
        Wrapper for computing a flattened real gradient using NumPy.
        """
        if analytical:
            return flat_gradient_real_analytical_np(fun, params, arg)
        else:
            return flat_gradient_real_numerical_np(fun, params, arg)

    def flat_gradient_holo_analytical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute an analytical flattened gradient for holomorphic networks using NumPy.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        grad_val = fun.gradient(params, arg)
        return np.concatenate([v.ravel() for v in grad_val.values()])

    def flat_gradient_holo_numerical_np(fun: Any, params: Any, arg: Any) -> np.ndarray:
        """
        Compute a flattened gradient for holomorphic networks using numerical differentiation (NumPy).
        """
        g = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        return np.concatenate([g.ravel(), 1.j * g.ravel()])

    def flat_gradient_holo_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> np.ndarray:
        """
        Wrapper for computing a flattened gradient for holomorphic networks using NumPy.
        """
        if analytical:
            return flat_gradient_holo_analytical_np(fun, params, arg)
        else:
            return flat_gradient_holo_numerical_np(fun, params, arg)

    def dict_gradient_analytical_np(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute an analytical dictionary of complex gradients using NumPy.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if not hasattr(fun, "analytical_dict_gradient"):
            raise NotImplementedError("Analytical dict gradient not implemented for this function.")
        return fun.analytical_dict_gradient(params, arg)

    def dict_gradient_numerical_np(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute a dictionary of complex gradients using numerical differentiation (NumPy).
        """
        gr = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        gi = np_grad(lambda p, y: anp.imag(fun.apply(p, y)))(params, arg)["params"]
        return {key: gr[key].ravel() + 1.j * gi[key].ravel() for key in gr}

    def dict_gradient_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> Any:
        """
        Wrapper for computing a dictionary of complex gradients using NumPy.
        """
        if analytical:
            return dict_gradient_analytical_np(fun, params, arg)
        else:
            return dict_gradient_numerical_np(fun, params, arg)

    def dict_gradient_real_analytical_np(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute an analytical dictionary of real gradients using NumPy.
        """
        if not callable(fun):
            raise ValueError("fun must be callable.")
        if not hasattr(fun, "analytical_dict_gradient_real"):
            raise NotImplementedError("Analytical dict real gradient not implemented for this function.")
        return fun.analytical_dict_gradient_real(params, arg)

    def dict_gradient_real_numerical_np(fun: Any, params: Any, arg: Any) -> Any:
        """
        Compute a dictionary of real gradients using numerical differentiation (NumPy).
        """
        g = np_grad(lambda p, y: anp.real(fun.apply(p, y)))(params, arg)["params"]
        return g  # Preserving structure

    def dict_gradient_real_np(fun: Any, params: Any, arg: Any, analytical: bool = False) -> Any:
        """
        Wrapper for computing a dictionary of real gradients using NumPy.
        """
        if analytical:
            return dict_gradient_real_analytical_np(fun, params, arg)
        else:
            return dict_gradient_real_numerical_np(fun, params, arg)
        
# ==============================================================================
#! APPLY CALLABLE
# ==============================================================================

@numba.njit
def apply_callable_np(func, states, sample_probas, logprobas_in, logproba_fun):
    """
    Applies a transformation function to each state and computes a locally
    weighted estimate as motivated in the paper [2108.08631].
    
    Parameters:
        - func: Callable that accepts a state (vector) and returns a tuple (S, V)
                where S is an array of modified states (M x state_size) and V is an
                array of corresponding values (M,).
        - states: Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
        - sample_probas: Array of sampling probabilities (same leading dimensions as states).
        - logprobas_in: Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
        - logproba_fun: Callable that computes the log-probabilities for given states S.

    Returns:
    - values    : The per-state estimates.
    - mean      : Mean of the estimates.
    - std       : Standard deviation of the estimates.
    """
    
    # the input of states might be (num_samples, num_chains, num_visible)
    # logprobas_in might be (num_samples, num_chains, 1)
    # transform to (num_samples * num_chains, num_visible) and (num_samples * num_chains, 1)
    if len(states.shape) == 3:
        states          = states.reshape(-1, states.shape[-1])
        logprobas_in    = logprobas_in.reshape(-1, 1)
        sample_probas   = sample_probas.reshape(-1, 1)
    
    size    = states.shape[0]
    values  = np.zeros(size, dtype=logprobas_in.dtype)
    for i in numba.prange(size):
        # apply the function to the state
        logp                    = logprobas_in[i, 0]
        sample_p                = sample_probas[i, 0]
        new_states, new_vals    = func(states[i])
        
        # compute the new logprobas
        new_logprobas           = logproba_fun(new_states)
        
        # compute the weighted sum of the new logprobas
        weights                 = sample_p * np.exp(new_logprobas - logp)
        values[i]               = np.sum(new_vals * weights, axis=0)
    return values, np.mean(values, axis = 0), np.std(values, axis = 0)

@numba.njit
def apply_callable_np_uniform(func, states, logprobas_in, logproba_fun):
    """
    Applies a transformation function to each state and computes a locally
    weighted estimate as motivated in the paper [2108.08631].
    
    Parameters:
        - func: Callable that accepts a state (vector) and returns a tuple (S, V)
                where S is an array of modified states (M x state_size) and V is an
                array of corresponding values (M,).
        - states: Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
        - logprobas_in: Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
        - logproba_fun: Callable that computes the log-probabilities for given states S.

    Returns:
    - values    : The per-state estimates.
    - mean      : Mean of the estimates.
    - std       : Standard deviation of the estimates.
    """
    
    # the input of states might be (num_samples, num_chains, num_visible)
    # logprobas_in might be (num_samples, num_chains, 1)
    # transform to (num_samples * num_chains, num_visible) and (num_samples * num_chains, 1)
    
    if len(states.shape) == 3:
        states          = states.reshape(-1, states.shape[-1])
        logprobas_in    = logprobas_in.reshape(-1, 1)
    
    size = states.shape[0]
    values = np.zeros(size, dtype=logprobas_in.dtype)
    
    for i in numba.prange(size):
        # apply the function to the state
        logp                    = logprobas_in[i, 0]
        new_states, new_vals    = func(states[i])
        
        # compute the new logprobas
        new_logprobas           = logproba_fun(new_states)
        
        # compute the weighted sum of the new logprobas
        weights                 = np.exp(new_logprobas - logp)
        values[i]               = np.sum(new_vals * weights, axis=0)
        
    return values, np.mean(values, axis=0), np.std(values, axis=0)

@numba.njit(parallel=True)
def apply_callable_batched_np(func, states, sample_probas, logprobas_in, logproba_fun, batch_size: int):
    """
    Batched version of apply_callable_np with sample probabilities.
    
    Applies a transformation function to each state and computes a locally
    weighted estimate as motivated in [2108.08631]. For each state x, a modified
    set of states S and corresponding values V is produced by `func(x)`. Then,
    the local estimate is computed as:
    
        estimate(x) = sum_i { V[i] * (sample_proba * exp(new_logproba[i] - logprobas_in(x)) ) }
    
    Parameters:
        func          : Callable that accepts a state (vector) and returns a tuple (S, V)
                        where S is an array of modified states (M x state_size) and V is an
                        array of corresponding values (M,).
        states        : Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
        sample_probas : Array of sampling probabilities (with the same leading dimensions as states).
        logprobas_in  : Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
        logproba_fun  : Callable that computes the log-probabilities for given states S.
        batch_size    : Integer size for batching the evaluation.
    
    Returns:
        values : Array of per-state estimates.
        mean   : Mean of the estimates.
        std    : Standard deviation of the estimates.
    """
    # Reshape inputs if provided in 3D.
    if states.ndim == 3:
        states = states.reshape(-1, states.shape[-1])
        logprobas_in = logprobas_in.reshape(-1, 1)
        sample_probas = sample_probas.reshape(-1, 1)
    
    # Create batches.
    states_batches      = create_batches_np(states, batch_size)
    logprobas_batches   = create_batches_np(logprobas_in, batch_size)
    sample_batches      = create_batches_np(sample_probas, batch_size)
    
    n_batches           = states_batches.shape[0]
    total_samples       = states.shape[0]
    estimates           = np.empty(total_samples, dtype=logprobas_in.dtype)
    
    index = 0  # Global index in the flattened estimates array.
    # Loop over batches in parallel.
    for b in numba.prange(n_batches):
        batch_states    = states_batches[b]     # (batch_size, state_size)
        batch_logps     = logprobas_batches[b]  # (batch_size, 1)
        batch_samples   = sample_batches[b]     # (batch_size, 1)
        for j in range(batch_states.shape[0]):
            # Extract scalar values.
            logp          = batch_logps[j, 0]
            sample_p      = batch_samples[j, 0]
            # Apply the callable to the single state.
            new_states, new_vals = func(batch_states[j])
            new_logprobas = logproba_fun(new_states)
            # Compute weighted sum: sample_p * exp(new_logprobas - logp) scales new_vals.
            weights = sample_p * np.exp(new_logprobas - logp)
            estimates[index] = np.sum(new_vals * weights, axis=0)
            index += 1
    
    mean_val    = np.mean(estimates, axis=0)
    std_val     = np.std(estimates, axis=0)
    return estimates, mean_val, std_val

@numba.njit(parallel=True)
def apply_callable_batched_np_uniform(func, states, logprobas_in, logproba_fun, batch_size: int):
    """
    Batched version of apply_callable_np for the uniform case (without sample probabilities).
    
    For each state x, this function applies a transformation `func` to obtain modified
    states S and corresponding values V. The local estimate is computed as:
    
        estimate(x) = sum_i { V[i] * exp(new_logproba[i] - logprobas_in(x)) }
    
    which is equivalent to assuming that sample probabilities are uniform.
    
    Parameters:
        func         : Callable that accepts a state (vector) and returns a tuple (S, V)
                        where S is an array of modified states (M x state_size) and V is an
                        array of corresponding values (M,).
        states       : Array of states with shape (n_samples, n_chains, state_size) or (n_states, state_size).
        logprobas_in : Array of original log-probabilities (shape (n_samples, n_chains, 1) or (n_states, 1)).
        logproba_fun : Callable that computes the log-probabilities for given states S.
        batch_size   : Integer size for batching the evaluation.
    
    Returns:
        values : Array of per-state estimates.
        mean   : Mean of the estimates.
        std    : Standard deviation of the estimates.
    """
    # Reshape inputs if provided in 3D.
    if states.ndim == 3:
        states          = states.reshape(-1, states.shape[-1])
        logprobas_in    = logprobas_in.reshape(-1, 1)
    
    # Create batches.
    states_batches      = create_batches_np(states, batch_size)
    logprobas_batches   = create_batches_np(logprobas_in, batch_size)
    
    n_batches           = states_batches.shape[0]
    total_samples       = states.shape[0]
    estimates           = np.empty(total_samples, dtype=logprobas_in.dtype)
    
    index = 0  # Global index in the flattened estimates array.
    
    # Loop over batches in parallel.
    for b in numba.prange(n_batches):
        batch_states    = states_batches[b]     # (batch_size, state_size)
        batch_logps     = logprobas_batches[b]  # (batch_size, 1)
        for j in range(batch_states.shape[0]):
            logp                    = batch_logps[j, 0]
            new_states, new_vals    = func(batch_states[j])
            new_logprobas           = logproba_fun(new_states)
            weights                 = np.exp(new_logprobas - logp)
            estimates[index]        = np.sum(new_vals * weights)
            index                   += 1
            
    mean_val    = np.mean(estimates, axis=0)
    std_val     = np.std(estimates, axis=0)
    return estimates, mean_val, std_val

# ==============================================================================