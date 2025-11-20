"""
file        : general_python/ml/net_impl/utils/net_init.py
author      : Maksymilian Kliczkowski
email       :
date        : 2025-03-10
"""

# import us
import numpy as np
from . import net_init_np as numpy
from ....algebra.utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from . import net_init_jax as jaxpy
    
from typing import Optional, Tuple, Callable
from enum import Enum, unique, auto

@unique
class Initializers(Enum):
    '''
    Enum for different weight initializers.
    '''
    HE       = auto()
    UNIFORM  = auto()
    XAVIER   = auto()
    NORMAL   = auto()
    CONSTANT = auto()

#########################################################################
#! INITIALIZER FUNCTION
#########################################################################

_ERR_BACKEND_NOT_SUPPORTED = "Backend not supported. Please choose 'jax' or 'numpy'."
_ERR_INITIALIZER_NOT_FOUND = "Initializer not found. Please check the name."

def get_initializer(name: str, backend: Optional[str] = None, dtype: Optional = None) -> Callable:
    """
    Get the initializer function based on the name, backend, and dtype.
    
    Args:
        name (str)              : Name of the initializer (e.g., "he", "uniform", "xavier", "normal", "constant").
        backend (Optional[str]) : Backend to use ('jax' or 'numpy'). If None, defaults to 'jax'.
        dtype (Optional)        : Data type to decide between real or complex initializer.
                    If None, defaults to jnp.float32 for JAX and np.float32 for NumPy.
    
    Returns:
        Callable: Initializer function.
    """
    # Default to JAX if backend is not provided.
    if isinstance(backend, str):
        backend = backend.lower()
        if backend in ["jax", "jnp"] and JAX_AVAILABLE:
            backend = "jax"
        else:
            backend = "numpy"
    else:
        if backend == numpy:
            backend = "numpy"
        else:
            backend = "jax"
    
    # Default dtype if not provided.
    if dtype is None:
        dtype = jnp.float32 if backend == "jax" else np.float32

    # Determine if dtype is complex based on backend.
    if backend == "jax" and JAX_AVAILABLE:
        is_complex = jnp.issubdtype(dtype, jnp.complexfloating)
        mapping = {
            "he":       { "real": jaxpy.real_he_init,           "complex": jaxpy.complex_he_init        },
            "uniform":  { "real": jaxpy.real_uniform_init,      "complex": jaxpy.complex_uniform_init   },
            "xavier":   { "real": jaxpy.real_xavier_init,       "complex": jaxpy.complex_xavier_init    },
            "normal":   { "real": jaxpy.real_normal_init,       "complex": jaxpy.complex_normal_init    },
            "constant": { "real": jaxpy.real_constant_init,     "complex": jaxpy.complex_constant_init  }
        }
    elif backend in ("numpy", "np"):
        is_complex = np.issubdtype(dtype, np.complexfloating)
        mapping = {
            "he":       { "real": numpy.real_he_init,           "complex": np.complex_he_init           },
            "uniform":  { "real": numpy.real_uniform_init,      "complex": np.complex_uniform_init      },
            "xavier":   { "real": numpy.real_xavier_init,       "complex": np.complex_xavier_init       },
            "normal":   { "real": numpy.real_normal_init,       "complex": np.complex_normal_init       },
            "constant": { "real": numpy.real_constant_init,     "complex": np.complex_constant_init     }
        }
    else:
        raise ValueError(_ERR_BACKEND_NOT_SUPPORTED)
    
    key = name.lower()
    if key not in mapping:
        raise ValueError(_ERR_INITIALIZER_NOT_FOUND + f" Available: {', '.join(mapping.keys())}")
    
    return mapping[key]["complex" if is_complex else "real"]
