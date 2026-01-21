'''

This module provides various activation functions for neural networks with both JAX and NumPy implementations.
The module supports common activation functions like sigmoid, tanh, ReLU variants, and more specialized functions.
Each function has both NumPy and JAX (when available) implementations to support different computation backends.

Functions are accessible through the get_activation factory function that returns the appropriate implementation
based on the specified backend.

--------------------------------------------------------------
file    : general_python/ml/activation_functions.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
--------------------------------------------------------------
'''

import numpy    as np
from typing     import Optional, Tuple, Callable

try:
    import  jax
    import  jax.numpy   as jnp
    import  jax.nn      as jnn
    from    jax         import random
    import flax.linen   as nn
    JAX_AVAILABLE       = True
except ImportError:
    JAX_AVAILABLE       = False
    
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

if JAX_AVAILABLE:
    import jax
    
    # @partial(jax.jit, inline = True)
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
        return x + jax.lax.log1p(jax.lax.exp(-2.0 * x)) - jax.lax.log(2.0)

    # @partial(jax.jit, inline = True)
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
    
    # # @jax.jit
    def sigmoid_jnp(x):
        ''' 
        Sigmoid activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            1/(1+exp(-x))
        '''
        return nn.sigmoid(x)
    
    # # @jax.jit
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

    # # @jax.jit
    def relu_jnp(x):
        ''' 
        Rectified linear unit activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            max(0, x)
        '''
        return nn.relu(x)
    
    # # @jax.jit
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
    
    # @jax.jit
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
    
    # @jax.jit
    def softplus_jnp(x):
        ''' 
        Softplus activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            log(1 + exp(x))
        '''
        return nn.softplus(x)
    
    # @jax.jit
    def identity_jnp(x):
        ''' 
        Identity activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            x
        '''
        return x
    
    # @jax.jit
    def poly6_jnp(x):
        ''' 
        Polynomial activation function of degree 6 (JAX implementation).
        Efficient implementation of the polynomial: 0.022222222*x^6 - 0.083333333*x^4 + 0.5*x^2
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            Polynomial evaluation at x
        '''
        x2 = x ** 2
        return ((0.022222222 * x2 - 0.083333333) * x2 + 0.5) * x2
    
    # @jax.jit
    def poly5_jnp(x):
        ''' 
        Polynomial activation function of degree 5 (JAX implementation).
        Efficient implementation of the polynomial: 0.133333333*x^5 - 0.333333333*x^3 + x
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            Polynomial evaluation at x
        '''
        xsq = x ** 2
        return ((0.133333333 * xsq - 0.333333333) * xsq + 1.) * x
    
    def elu_p1_jnp(x, alpha=1.0):
        ''' 
        Exponential linear unit activation function shifted by +1 (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            alpha: Scale for negative values
            
        Returns:
            x if x > 0 else alpha*(exp(x)-1) + 1
        '''
        return nn.elu(x, alpha=alpha) + 1
    
    def swish_jnp(x):
        ''' 
        Swish activation function (JAX implementation).
        
        Parameters:
            x: Input tensor/array
            
        Returns:
            Swish activation
        '''
        return x * jnn.sigmoid(x)
        
    def mish_jnp(x):
        """Mish activation: x * tanh(softplus(x))"""
        return x * jnp.tanh(jnn.softplus(x))

    def gelu_jnp(x, approximate=False):
        """Gaussian Error Linear Unit. Excellent for Transformers/Deep CNNs."""
        return jnn.gelu(x, approximate=approximate)

    def celu_jnp(x, alpha=1.0):
        """Continuously Differentiable Exponential Linear Unit."""
        return jnn.celu(x, alpha=alpha)
    def sin_jnp(x): 
        """Sine. Holomorphic and periodic. Good for NQS."""
        return jnp.sin(x)

    def cos_jnp(x): 
        """Cosine. Holomorphic and periodic."""
        return jnp.cos(x)
    
    activations_jnp = {
        'identity'      : identity_jnp,
        'log_cosh'      : log_cosh_jnp,
        'tanh'          : tanh_jnp,
        'sigmoid'       : sigmoid_jnp,
        'sigmoid_inv'   : sigmoid_inv_jnp,
        'relu'          : relu_jnp,
        'leaky_relu'    : leaky_relu_jnp,
        'elu'           : elu_jnp,
        'elu1'          : elu_p1_jnp,
        'softplus'      : softplus_jnp,
        'poly6'         : poly6_jnp,
        'poly5'         : poly5_jnp,
        # Additional activations
        'gelu'          : gelu_jnp,
        'swish'         : swish_jnp,
        'mish'          : mish_jnp,
        'celu'          : celu_jnp,
        'sin'           : sin_jnp,
        'cos'           : cos_jnp,
        
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
        'elu1'          : {'alpha': 1.0},
        'softplus'      : None,
        'poly6'         : None,
        'poly5'         : None,
        # Additional activations
        'gelu'          : {'approximate': False},
        'swish'         : None,
        'mish'          : None,
        'celu'          : {'alpha': 1.0},
        'sin'           : None,
        'cos'           : None,
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
        if not isinstance(name, str) and callable(name):
            return name, params
        
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
    if not isinstance(name, str) and callable(name):
        return name, params
    
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

def list_activations(backend: str = 'default') -> list:
    """
    List available activation functions for the specified backend.
    
    Parameters:
        backend : str
            Backend to use ('default', 'numpy', or 'jax').
    Returns:
        list: List of available activation function names.
    """
    if backend in ['np', 'numpy'] or backend == np:
        return list(activations_np.keys())
    return list(activations_jnp.keys())

#############################################################################