'''
Solver module for various linear algebra solvers.


Initialization file for the solvers module. Exports solver classes,
the SolverType enum, and the choose_solver factory function.
----------------------------------------------------------------
File        : general_python/algebra/solvers/__init__.py
Author      : Maksymilian Kliczkowski
License     : MIT
Description : This module provides a factory function to choose and instantiate
              different solver types based on user input. It includes various
              solver implementations (direct, iterative, etc.) and allows for
              customization through keyword arguments. The module also includes
              a utility function to generate test matrix-vector pairs for
              testing purposes. The solvers are designed to work with different
              numerical backends (e.g., NumPy, SciPy, JAX) and support various
              data types. The module is part of a larger algebra library and
              aims to provide a flexible and extensible framework for solving
              linear algebra problems.
----------------------------------------------------------------
'''

import inspect
from typing import Union, Optional, Any, Type
from enum import Enum, auto

# Import base classes and types (Lightweight)
# Adjust relative path if needed based on your file structure
try:
    from ..solver           import Solver, SolverResult, SolverError, SolverErrorMsg, SolverType, Array, MatVecFunc, StaticSolverFunc
    from ..preconditioners  import Preconditioner, choose_precond
except ImportError:
    raise ImportError("Could not import base solver classes. Check the module structure.")

# -----------------------------------------------------------------------------
# Lazy Loading Configuration
# -----------------------------------------------------------------------------

_LAZY_MODULES = {
    'CgSolver'                  : '.cg',
    'CgSolverScipy'             : '.cg',
    'DirectSolver'              : '.direct',
    'DirectScipy'               : '.direct',
    'DirectJaxScipy'            : '.direct',
    'DirectInvSolver'           : '.direct',
    'BackendSolver'             : '.backend',
    'BackendDirectSolver'       : '.backend',
    'PseudoInverseSolver'       : '.pseudoinverse',
    'MinresQLPSolver'           : '.minres_qlp',
    'MinresSolverScipy'         : '.minres',
    'MinresSolver'              : '.minres',
}

# -----------------------------------------------------------------------------

class SolverForm(Enum):
    """
    Enum for solver forms.
    """
    MATRIX      = auto()    # Matrix form -> expects a matrix formation explicitly
    MATVEC      = auto()    # Matrix-vector form -> expects a matrix-vector multiplication function (f(x))
    GRAM        = auto()    # Gram form -> expects a Gram matrix decomposition S and S^+ as input

    def __eq__(self, other):
        if isinstance(other, SolverForm):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

# -----------------------------------------------------------------------------

def choose_solver(solver_id     : Union[str, int, SolverType, Type[Solver]],
                backend         : str                       = "default",
                *,
                sigma           : Optional[float]           = None,
                is_gram         : bool                      = False,
                default_precond : Optional[Preconditioner]  = None,
                **kwargs) -> Solver:
    """
    Factory function to select and instantiate a solver based on identifier.
    Uses lazy loading to import specific solver classes only when requested.
    
    Parameters
    ----------
    solver_id : Union[str, int, SolverType, Type[Solver]]
        Identifier for the solver. Can be a string name, integer code,
        SolverType enum, or a Solver subclass.
    backend : str, optional
        Numerical backend to use (e.g., "numpy", "scipy", "jax"). Default is "default".
    sigma : Optional[float], optional
        Shift parameter for the solver, if applicable. Default is None.
    is_gram : bool, optional
        Whether to treat the problem as a Gram matrix problem. Default is False.
    default_precond : Optional[Preconditioner], optional
        Default preconditioner to use if none is specified. Default is None.
    **kwargs
        Additional keyword arguments to pass to the solver constructor.
        
    Returns
    -------
    Solver
        An instance of the selected solver class.
        
    Examples
    --------
    >>> solver = choose_solver("CG", backend="numpy", sigma=0.1)
    >>> solver = choose_solver(SolverType.MINRES, is_gram=True)
    >>> solver = choose_solver(MyCustomSolverClass, custom_param=42)
    """
    
    # 1. Handle Instance Passthrough
    if isinstance(solver_id, Solver):
        if kwargs.get('logger'):
            kwargs['logger'].warning(f"Solver instance provided; ignoring kwargs: {kwargs}")
        return solver_id

    # 2. Resolve SolverType Enum
    solver_type = None
    if isinstance(solver_id, str):
        if solver_id.upper() in SolverType.__members__:
            solver_type = SolverType[solver_id.upper()]
        elif solver_id.lower() == "default":
            solver_type = SolverType.BACKEND
        else:
            solver_type = None
    elif isinstance(solver_id, int):
        solver_type = SolverType(solver_id)
    elif isinstance(solver_id, SolverType):
        solver_type = solver_id

    # 3. Import Class based on Type (Lazy Import Logic)
    target_class: Type[Solver] = None

    if isinstance(solver_id, type) and issubclass(solver_id, Solver):
        target_class = solver_id
    elif solver_type:
        # Mapping Logic inside the function to delay imports
        if solver_type == SolverType.CG:
            from .cg import CgSolver as target_class
        elif solver_type == SolverType.SCIPY_CG:
            from .cg import CgSolverScipy as target_class
        #
        elif solver_type == SolverType.MINRES:
            from .minres import MinresSolver as target_class
        elif solver_type == SolverType.SCIPY_MINRES:
            from .minres import MinresSolverScipy as target_class
        # 
        elif solver_type == SolverType.MINRES_QLP:
            from .minres_qlp import MinresQLPSolver as target_class
        elif solver_type == SolverType.PSEUDO_INVERSE:
            from .pseudoinverse import PseudoInverseSolver as target_class
        #
        elif solver_type == SolverType.DIRECT:
            from .direct import DirectSolver as target_class
        elif solver_type == SolverType.SCIPY_DIRECT:
            from .direct import DirectScipy as target_class
        #
        elif solver_type == SolverType.BACKEND:
            from .backend import BackendSolver as target_class
        else:
            raise NotImplementedError(f"Solver type {solver_type} is defined but not mapped to a class.")
    else:
        # Fallback for string names not in Enum but potentially valid classes
        if isinstance(solver_id, str) and solver_id in _LAZY_MODULES:
            import importlib
            module          = importlib.import_module(_LAZY_MODULES[solver_id], package = __name__)
            target_class    = getattr(module, solver_id)
        else:
            raise ValueError(f"Unknown solver identifier: {solver_id}")

    # 4. Instantiate
    # Prepare kwargs
    init_kwargs = kwargs.copy()
    init_kwargs.update({
        'sigma'             : sigma,
        'is_gram'           : is_gram,
        'default_precond'   : default_precond,
        'backend'           : backend
    })

    try:
        # Introspect constructor to pass only valid arguments
        sig             = inspect.signature(target_class.__init__)
        valid_params    = sig.parameters
        base_params     = Solver.__init__.__code__.co_varnames
        
        filtered_kwargs = {}
        has_varkw       = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values())

        for k, v in init_kwargs.items():
            if k in valid_params or k in base_params or has_varkw:
                filtered_kwargs[k] = v
                
        return target_class(**filtered_kwargs)
        
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate {target_class.__name__}: {e}")

# -----------------------------------------------------------------------------
# Module-level __getattr__ for Lazy Imports
# -----------------------------------------------------------------------------

def __getattr__(name):
    """
    Lazy import of solver classes when accessed directly (e.g. solvers.CgSolver).
    """
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'Solver', 'SolverResult', 'SolverError', 'SolverErrorMsg', 'SolverType', 
    'SolverForm', 'choose_solver',
    # List all lazy classes so IDEs/tools know they exist
    'CgSolver', 'CgSolverScipy',
    'DirectSolver', 'DirectScipy', 'DirectJaxScipy', 'DirectInvSolver',
    'PseudoInverseSolver', 'MinresQLPSolver', 'MinresSolverScipy', 'MinresSolver'
]

__author__      = "Maksymilian Kliczkowski"
__version__     = "1.0"
__license__     = "MIT"
__status__      = "Development"
__maintainer__  = "Maksymilian Kliczkowski"
__email__       = "maksymilian.kliczkowski@pwr.edu.pl"
__description__ = """
                This module provides a factory function to choose and instantiate
                different solver types based on user input. It includes various
                solver implementations (direct, iterative, etc.) and allows for
                customization through keyword arguments. The module also includes
                a utility function to generate test matrix-vector pairs for
                testing purposes. The solvers are designed to work with different
                numerical backends (e.g., NumPy, SciPy, JAX) and support various
                data types. The module is part of a larger algebra library and
                aims to provide a flexible and extensible framework for solving
                linear algebra problems.
                """

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------