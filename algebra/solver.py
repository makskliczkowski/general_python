import numpy as np
import scipy
from abc import ABC, abstractmethod
from typing import Union

# Optionally, if JAX is available, you can set Backend accordingly.
try:
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax import jit
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
# Use JAX's numpy if available, else fall back to standard NumPy.
Backend = jnp if _JAX_AVAILABLE else np

def maybe_jit(func):
    """
    Decorator that applies the JAX JIT compilation if JAX is available.
    
    Parameters
    ----------
    func : function
        The function to be compiled with JIT.
    
    Returns
    -------
    function
        The JIT-compiled function (if JAX is available).
    """
    return jit(func) if _JAX_AVAILABLE else func

from enum import Enum, auto, unique                 # for enumerations

# ---------------------------------------------------------------------

@unique
class SolverType(Enum):
    """
    Enumeration class for the different types of solvers.
    """
    DIRECT          = auto()
    SCIPY_DIRECT    = auto()
    SCIPY_CJ        = auto()
    SCIPY_MINRES    = auto()
    


# ---------------------------------------------------------------------

class Solver(ABC):
    
    def __init__(self):
        self.solver_type = SolverType.DIRECT

