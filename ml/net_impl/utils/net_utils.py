'''
file    : general_python/ml/net_impl/utils/net_utils.py
author  : Maksymilian Kliczkowski
date    : 2025-03-01


-----------------


# Some info:

This file contains functions for creating batches of data, evaluating functions on batches, 
and handling various network types and their corresponding operations.
It includes implementations for both JAX and NumPy backends,
as well as functions for computing gradients
using both analytical and numerical methods.
It also provides a wrapper for selecting
the appropriate gradient function based on the input parameters.
The code is designed to be flexible and can handle complex and real-valued functions,
as well as holomorphic and non-holomorphic networks.

## Holomorphic networks:
If the function (for example, a variational ansatz for a quantum state)
is holomorphic with respect to its complex parameters, 
then the derivative with respect to the complex variable 
is well defined in the usual sense. 
The gradient can be computed using standard complex differentiation rules,
and the real and imaginary parts of the gradient are not
independent—they satisfy the Cauchy-Riemann conditions.

## Non-holomorphic networks:
When we say the gradient is not holomorphic,
it means that the function is not complex differentiable in the standard sense.
In this case, the function does not satisfy
the Cauchy-Riemann equations and the differentiation with respect
to the complex parameters must be done by
treating the real and imaginary parts as independent variables. This results
in a gradient that generally has extra
degrees of freedom compared to the holomorphic case and requires more care in its computation.

For example, if you have a wave function ansatz ψ(s;θ),
where θ is complex, a holomorphic ansatz would allow
you to compute derivatives with respect to θ directly.
However, if the ansatz is non-holomorphic,
you need to compute the derivatives with respect to Re(θ) and Im(θ)
separately and then combine them appropriately.
'''

import numpy as np
import numba
from typing import Union, Callable, Optional, Any, Tuple

# from general python utils
from general_python.algebra.utils import JAX_AVAILABLE, get_backend
import general_python.ml.net_impl.utils.net_utils_np as numpy
if JAX_AVAILABLE:
    import general_python.ml.net_impl.utils.net_utils_jax as jaxpy

#########################################################################
#! BATCHES
#########################################################################

def create_batches( data,
                    batch_size  : int,
                    backend     : str = 'default'):
    """
    Create batches of data with a specified batch size. If the data cannot be evenly divided into batches,
    the data is padded with the last element of the data.
    The function supports both JAX and NumPy backends.
    
    Parameters
    ----------
    data : jnp.ndarray
        The input array (for example, an array of samples).
    batch_size : int
        The desired size of each batch.
    
    Returns
    -------
    jnp.ndarray
        The padded and reshaped array where the first dimension is split into batches.
    
    Example
    -------
    >>> # Suppose data has shape (5, 2) and batch_size is 3.
    >>> data = jnp.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
    >>> # This function pads data to shape (6,2) and then reshapes it to (2, 3, 2)
    >>> create_batches_jax(data, 3)
    DeviceArray([[[ 1,  2],
                    [ 3,  4],
                    [ 5,  6]],
                    [[ 7,  8],
                    [ 9, 10],
                    [ 9, 10]]], dtype=int32)
    >>> This means that batch is appended in front of the data.
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        data = np.asarray(data)
        return numpy.create_batches_np(data, batch_size)
    return jaxpy.create_batches_jax(data, batch_size)

##########################################################################
#! EVALUATE BATCHED
##########################################################################

def eval_batched(batch_size : int,
                func        : Callable,
                params      : Any,
                data        : np.ndarray,
                backend     : str = 'default'):
    """ Evaluate a function on batches of data using either JAX or NumPy.
    Parameters
    ----------
    batch_size : int
        The size of each batch.
    func : Callable
        The function to evaluate on each batch.
    data : np.ndarray
        The input data to be processed.
    backend : str, optional
        The backend to use for evaluation ('default' uses NumPy if available).

    Returns
    -------
    np.ndarray
        The concatenated results of the function applied to each batch.
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return numpy.eval_batched_np(batch_size, func, params, data)
    return jaxpy.eval_batched_jax(batch_size, func, params, data)

##########################################################################
#! GRADIENTS
##########################################################################

# ==============================================================================
#! Global Wrapper Functions (for both backends)
# ==============================================================================

def flat_gradient(fun: Any, params: Any, arg: Any,
            backend: str = "jax", analytical: Optional[bool] = False) -> Any:
    """
    Compute a flattened complex gradient using either JAX or NumPy.

    Parameters
    ----------
    fun : Callable
        The function or analytical gradient function.
    params : Any
        The network parameters.
    arg : Any
        The input state.
    backend : str, optional
        Backend to use ("jax" or "numpy").
    analytical : bool, optional
        If True, use the analytical gradient if available.

    Returns
    -------
    A flattened complex gradient (jnp.ndarray or np.ndarray).
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return numpy.flat_gradient_np(fun, params, arg, analytical)
    return jaxpy.flat_gradient_jax(fun, params, arg, analytical)

def flat_gradient_cpx_nonholo(fun: Any, params: Any, arg: Any,
                            backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a flattened complex gradient for non-holomorphic networks using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return numpy.flat_gradient_cpx_nonholo_np(fun, params, arg, analytical)
    return jaxpy.flat_gradient_cpx_nonholo_jax(fun, params, arg, analytical)

def flat_gradient_real(fun: Any, params: Any, arg: Any,
                    backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a flattened real gradient using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return numpy.flat_gradient_real_np(fun, params, arg, analytical)
    return jaxpy.flat_gradient_real_jax(fun, params, arg, analytical)

def flat_gradient_holo(fun: Any, params: Any, arg: Any,
                    backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a flattened gradient for holomorphic networks using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return numpy.flat_gradient_holo_np(fun, params, arg, analytical)
    return jaxpy.flat_gradient_holo_jax(fun, params, arg, analytical)

def dict_gradient(fun: Any, params: Any, arg: Any,
                backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a dictionary of complex gradients using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return numpy.dict_gradient_np(fun, params, arg, analytical)
    return jaxpy.dict_gradient_jax(fun, params, arg, analytical)

def dict_gradient_real(fun: Any, params: Any, arg: Any,
                    backend: str = "jax", analytical: bool = False) -> Any:
    """
    Compute a dictionary of real gradients using either JAX or NumPy.
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return numpy.dict_gradient_real_np(fun, params, arg, analytical)
    return jaxpy.dict_gradient_real_jax(fun, params, arg, analytical)

# ==============================================================================

def decide_grads(
    iscpx         : bool,
    isjax         : bool,
    isanalytic    : bool,
    isholomorphic : bool
    ) -> Tuple[Callable, Callable]:
    """
    Decide which gradient functions (flattened and PyTree) to use based on flags.

    Parameters
    ----------
    iscpx         : bool
        True if the function/parameters are complex.
    isjax         : bool
        True if JAX backend should be used.
    isanalytic    : bool
        True if analytical gradients should be used.
    isholomorphic : bool
        True if the function is holomorphic (if complex).

    Returns
    -------
    Tuple[Callable, Callable]
        (flat_gradient_func, pytree_gradient_func)
        The appropriate functions for computing flattened gradients per sample
        and PyTree gradients, respectively.
    """
    if not isjax:   # NumPy backend
        raise NotImplementedError("NumPy backend gradient functions not provided.")
    else:           # JAX backend
        if iscpx:   # Complex functions
            if isholomorphic:  # Holomorphic
                if isanalytic:
                    # return jaxpy.flat_gradient_cpx_holo_analytical_jax, jaxpy.pytree_gradient_cpx_holo_analytical_jax
                    return jaxpy.flat_gradient_analytic_jax, jaxpy.pytree_gradient_analytic_jax
                else:
                    # Numerical: ∇_{p*}f (flat outputs [Re,Im] float, pytree outputs complex)
                    return jaxpy.flat_gradient_cpx_holo_jax, jaxpy.pytree_gradient_cpx_holo_jax
            else:   # Non-holomorphic
                if isanalytic:
                    # return jaxpy.flat_gradient_cpx_nonholo_analytical_jax, jaxpy.pytree_gradient_cpx_nonholo_analytical_jax
                    raise NotImplementedError("Analytical non-holomorphic JAX gradients not implemented.")
                else:
                    # Numerical: (∇_p f)* (flat outputs complex, pytree outputs complex)
                    return jaxpy.flat_gradient_cpx_nonholo_jax, jaxpy.pytree_gradient_cpx_nonholo_jax
        else:       # Real functions
            if isanalytic:
                # return jaxpy.flat_gradient_real_analytical_jax, jaxpy.pytree_gradient_real_analytical_jax
                raise NotImplementedError("Analytical real JAX gradients not implemented.")
            else:
                # Numerical: ∇_p Re[f] (flat outputs float, pytree outputs float)
                return jaxpy.flat_gradient_real_jax, jaxpy.pytree_gradient_real_jax

    # Should not be reached if logic is correct
    raise ValueError("Invalid combination of gradient flags for JAX backend.")

# ==============================================================================