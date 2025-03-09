'''
file: general_python/ml/activation_functions.py

author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl

This module provides various activation functions for neural networks with both JAX and NumPy implementations.
The module supports common activation functions like sigmoid, tanh, ReLU variants, and more specialized functions.
Each function has both NumPy and JAX (when available) implementations to support different computation backends.

Functions are accessible through the get_activation factory function that returns the appropriate implementation
based on the specified backend.
'''

import numpy as np
from typing import Optional, Tuple, Callable

from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
if _JAX_AVAILABLE:
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
    
###################################################################
#! SET OF ACTIVATION FUNCTIONS
###################################################################

def square(x):
    ''' 
    Square activation function.
    
    Parameters:
        x: Input tensor/array
        
    Returns:
        Square of the input
    '''
    return x ** 2

def poly6(x):
    ''' 
    Polynomial activation function of degree 6.
    Efficient implementation of the polynomial: 0.022222222*x^6 - 0.083333333*x^4 + 0.5*x^2
    
    Parameters:
        x: Input tensor/array
        
    Returns:
        Polynomial evaluation at x
    '''
    x2 = x ** 2
    return ((0.022222222 * x2 - 0.083333333) * x2 + 0.5) * x2

def poly5(x):
    ''' 
    Polynomial activation function of degree 5.
    Efficient implementation of the polynomial: 0.133333333*x^5 - 0.333333333*x^3 + x
    
    Parameters:
        x: Input tensor/array
        
    Returns:
        Polynomial evaluation at x
    '''
    xsq = x ** 2
    return ((0.133333333 * xsq - 0.333333333) * xsq + 1.) * x

if _JAX_AVAILABLE:
    def log_cosh_jnp(x):
        ''' 
        Logarithm of the hyperbolic cosine activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            log(cosh(x)) computed in a numerically stable way
        '''
        sgn_x   = -2 * jnp.signbit(x.real) + 1
        x       = x * sgn_x
        return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)
    
    def tanh_jnp(x):
        ''' 
        Hyperbolic tangent activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            tanh(x)
        '''
        sgn_x   = -2 * jnp.signbit(x.real) + 1
        x       = x * sgn_x
        return jnp.tanh(x)
    
    def sigmoid_jnp(x):
        ''' 
        Sigmoid activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            1/(1+exp(-x))
        '''
        return nn.sigmoid(x)
    
    def sigmoid_inv_jnp(x):
        ''' 
        Inverse of the sigmoid activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array (values should be in the range (0,1))
            
        Returns:
            -log(1/x - 1)
        '''
        sgn_x   = -2 * jnp.signbit(x.real) + 1
        x       = x * sgn_x
        return -jnp.log(1 / x - 1)
    
    def relu_jnp(x):
        ''' 
        Rectified linear unit activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            max(0, x)
        '''
        return nn.relu(x)
    
    def leaky_relu_jnp(x, alpha=0.01):
        ''' 
        Leaky rectified linear unit activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            alpha: Slope for negative values
            
        Returns:
            x if x > 0 else alpha*x
        '''
        return nn.leaky_relu(x, negative_slope=alpha)

    def elu_jnp(x, alpha=1.0):
        ''' 
        Exponential linear unit activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            alpha: Scale for negative values
            
        Returns:
            x if x > 0 else alpha*(exp(x)-1)
        '''
        return nn.elu(x, alpha=alpha)
    
    def softplus_jnp(x):
        ''' 
        Softplus activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            log(1 + exp(x))
        '''
        return nn.softplus(x)
    
    def identity_jnp(x):
        ''' 
        Identity activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            x
        '''
        return x
    
    activations_jnp = {
        'identity'      : identity_jnp,
        'log_cosh'      : log_cosh_jnp,
        'tanh'          : tanh_jnp,
        'sigmoid'       : sigmoid_jnp,
        'sigmoid_inv'   : sigmoid_inv_jnp,
        'relu'          : relu_jnp,
        'leaky_relu'    : leaky_relu_jnp,
        'elu'           : elu_jnp,
        'softplus'      : softplus_jnp
    }
    
    activations_jnp_parameters = {
        'identity'      : None,
        'log_cosh'      : None,
        'tanh'          : None,
        'sigmoid'       : None,
        'sigmoid_inv'   : None,
        'relu'          : None,
        'leaky_relu'    : {'alpha': 0.01},
        'elu'           : {'alpha': 1.0},
        'softplus'      : None
    }
    
    def get_activation_jnp(name: str, params: Optional[dict] = None) -> Callable:
        """
        Get the JAX activation function by name and parameters.
        
        Parameters:
            name : str
                Name of the activation function.
            params : Optional[dict]
                Parameters for the activation function.
        
        Returns:
            Callable: Activation function.
        
        Raises:
            ValueError: If the activation function name is not found.
        """
        if name not in activations_jnp:
            raise ValueError(f"Activation function '{name}' not found.")
        
        if params is None:
            params = activations_jnp_parameters[name]
        
        return activations_jnp[name], params

# NumPy implementations
def log_cosh(x):
    ''' 
    Logarithm of the hyperbolic cosine activation function (NumPy implementation).
    
    Parameters:
        x: Input array
        
    Returns:
        log(cosh(x)) computed in a numerically stable way
    '''
    sgn_x   = -2 * np.signbit(x.real) + 1
    x       = x * sgn_x
    return x + np.log1p(np.exp(-2.0 * x)) - np.log(2.0)

def tanh(x):
    ''' 
    Hyperbolic tangent activation function (NumPy implementation).
    
    Parameters:
        x: Input array
        
    Returns:
        tanh(x)
    '''
    sgn_x   = -2 * np.signbit(x.real) + 1
    x       = x * sgn_x
    return np.tanh(x)

def sigmoid(x):
    ''' 
    Sigmoid activation function (NumPy implementation).
    
    Parameters:
        x: Input array
        
    Returns:
        1/(1+exp(-x))
    '''
    return 1 / (1 + np.exp(-x))

def sigmoid_inv(x):
    ''' 
    Inverse of the sigmoid activation function (NumPy implementation).
    
    Parameters:
        x: Input array (values should be in the range (0,1))
        
    Returns:
        -log(1/x - 1)
    '''
    sgn_x   = -2 * np.signbit(x.real) + 1
    x       = x * sgn_x
    return -np.log(1 / x - 1)

def relu(x):
    ''' 
    Rectified linear unit activation function (NumPy implementation).
    
    Parameters:
        x: Input array
        
    Returns:
        max(0, x)
    '''
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    ''' 
    Leaky rectified linear unit activation function (NumPy implementation).
    
    Parameters:
        x: Input array
        alpha: Slope for negative values
        
    Returns:
        x if x > 0 else alpha*x
    '''
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    ''' 
    Exponential linear unit activation function (NumPy implementation).
    
    Parameters:
        x: Input array
        alpha: Scale for negative values
        
    Returns:
        x if x > 0 else alpha*(exp(x)-1)
    '''
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softplus(x):
    ''' 
    Softplus activation function (NumPy implementation).
    
    Parameters:
        x: Input array
        
    Returns:
        log(1 + exp(x))
    '''
    return np.log1p(np.exp(x))

activations_np = {
    'identity'      : lambda x: x,
    'log_cosh'      : log_cosh,
    'tanh'          : tanh,
    'sigmoid'       : sigmoid,
    'sigmoid_inv'   : sigmoid_inv,
    'relu'          : relu,
    'leaky_relu'    : leaky_relu,
    'elu'           : elu,
    'softplus'      : softplus
}

activations_np_parameters = {
    'identity'      : None,
    'log_cosh'      : None,
    'tanh'          : None,
    'sigmoid'       : None,
    'sigmoid_inv'   : None,
    'relu'          : None,
    'leaky_relu'    : {'alpha': 0.01},
    'elu'           : {'alpha': 1.0},
    'softplus'      : None
}

def get_activation_np(name: str, params: Optional[dict] = None) -> Callable:
    """
    Get the NumPy activation function by name and parameters.
    
    Parameters:
        name : str
            Name of the activation function.
        params : Optional[dict]
            Parameters for the activation function.
    
    Returns:
        Callable: Activation function.
    
    Raises:
        ValueError: If the activation function name is not found.
    """
    if name not in activations_np:
        raise ValueError(f"Activation function '{name}' not found.")
    
    if params is None:
        params = activations_np_parameters[name]
    
    return activations_np[name], params

############################################################################

def get_activation(name: str,
        params: Optional[dict] = None, backend: str = 'default') -> Tuple[Callable, dict]:
    """
    Factory function to get an activation function by name and parameters.
    
    This function routes to the appropriate backend implementation (NumPy or JAX)
    based on the specified backend parameter.
    
    Parameters:
        name : str
            Name of the activation function.
        params : Optional[dict]
            Parameters for the activation function.
        backend : str
            Backend to use ('default', 'numpy', or 'jax').
    
    Returns:
        Tuple[Callable, dict]: Activation function and its parameters.
    
    Examples:
        >>> func, params = get_activation('relu', backend='numpy')
        >>> func(np.array([-1.0, 0.0, 1.0]))
        array([0., 0., 1.])
    """
    if backend in ['np', 'numpy'] or backend == np:
        return get_activation_np(name, params)
    return get_activation_jnp(name, params)

#############################################################################