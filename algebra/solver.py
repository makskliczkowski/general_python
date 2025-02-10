import numpy as np
import scipy as sp
from abc import ABC, abstractmethod
from typing import Union

# ---------------------------------------------------------------------

from .__utils__ import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend as __backend, _KEY

# ---------------------------------------------------------------------

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

