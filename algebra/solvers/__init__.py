'''
file:       general_python/algebra/solvers/__init__.py
author:     Maksymilian Kliczkowski

Initialization file for the solvers module. Exports solver classes,
the SolverType enum, and the choose_solver factory function.
'''

import inspect
from typing import Union, Optional, Any, Type
from enum import Enum, auto, unique

# Import base classes and types from solver.py (assuming it's in the parent directory or path)
from ...algebra.solver import Solver, SolverResult, SolverError, SolverErrorMsg, SolverType, Array, MatVecFunc, StaticSolverFunc

# Import concrete solver implementations
from ...algebra.solvers.cg import CgSolver, CgSolverScipy
from ...algebra.solvers.direct import DirectSolver, DirectScipy, DirectJaxScipy, DirectInvSolver
from ...algebra.solvers.pseudoinverse import PseudoInverseSolver
from ...algebra.solvers.minres_qlp import MinresQLPSolver

# Import utility and preconditioner chooser
from ...algebra.utils import get_backend, JAX_AVAILABLE
from ...algebra.preconditioners import Preconditioner, choose_precond

# -----------------------------------------------------------------------------
#! Helper function: choose_solver
# -----------------------------------------------------------------------------

# Map SolverType Enums to their corresponding classes
_SOLVER_TYPE_TO_CLASS_MAP: dict[SolverType, Type[Solver]] = {
    # Direct Solvers
    SolverType.DIRECT           : DirectSolver,         # Uses backend linalg.solve
    SolverType.PSEUDO_INVERSE   : PseudoInverseSolver,  # Uses backend linalg.pinv
    SolverType.SCIPY_DIRECT     : DirectScipy,          # Uses scipy.linalg.solve
    # SolverType.DIRECT_INV     : DirectInvSolver,
    # SolverType.JAX_DIRECT     : DirectJaxScipy,

    # Iterative Solvers
    #! symmetric
    SolverType.CG               : CgSolver,
    SolverType.MINRES_QLP       : MinresQLPSolver,
    # SolverType.MINRES: MinresSolver, # Add when implemented
    #! general
    # SolverType.GMRES: GmresSolver, # Add when implemented

    # Iterative Solvers (SciPy Wrappers)
    SolverType.SCIPY_CG         : CgSolverScipy,
    # SolverType.SCIPY_MINRES: MinresScipy, # Add when implemented
    # SolverType.SCIPY_GMRES: GmresScipy, # Add when implemented

    # Others
    SolverType.BACKEND_SOLVER   : DirectSolver,
}

# -----------------------------------------------------------------------------

def choose_solver(solver_id     : Union[str, int, SolverType, Type[Solver]],
                sigma           : Optional[float] = None,
                *args,
                **kwargs) -> Solver:
    """
    Factory function to select and instantiate a solver based on identifier.

    Accepts various identifiers for the solver type and passes additional
    keyword arguments (like 'backend', 'eps', 'maxiter', 'a', 's', 'sp', etc.)
    to the specific solver's constructor.

    Args:
        solver_id (Union[str, int, SolverType, Type[Solver]]):
            Identifier for the solver. Can be:
                - A Solver class (e.g., `CgSolver`): Instantiates this class.
                - A Solver instance: Returns the instance directly (kwargs ignored).
                - A SolverType Enum member.
                - A string matching an Enum member name (case-insensitive).
                - An integer matching an Enum member value.
        **kwargs:
            Keyword arguments passed directly to the selected solver's constructor.
            Common args: `backend`, `dtype`, `eps`, `maxiter`, `a`, `s`, `sp`,
                        `matvec_func`, `sigma`, `is_gram`, `default_precond`.

    Returns:
        Solver: An instance of the selected solver class.

    Raises:
        TypeError: If solver_id has an unsupported type.
        ValueError: If a string/int identifier doesn't match a known SolverType
                    or if the mapped type isn't found.
        NotImplementedError: If the identified solver type is not yet mapped/implemented.
    """
    #! Handle Instance Passthrough
    if isinstance(solver_id, Solver):
        if kwargs:
            print(f"Warning: Solver instance provided; ignoring kwargs: {kwargs}")
        return solver_id

    #! Handle Class Passthrough
    if isinstance(solver_id, type) and issubclass(solver_id, Solver):
        target_class = solver_id
        print(f"Instantiating provided solver class: {target_class.__name__}")
    else:
        #! Resolve ID (str, int, Enum) to Enum Type
        solver_type: Optional[SolverType] = None
        if isinstance(solver_id, str):
            try:
                solver_type = SolverType[solver_id.upper()]
            except KeyError as e:
                raise ValueError(f"Unknown solver name: '{solver_id}'. Valid names: {[e.name for e in SolverType]}") from e
        elif isinstance(solver_id, int):
            try:
                solver_type = SolverType(solver_id)
            except ValueError as e:
                raise ValueError(f"Unknown solver value: {solver_id}. Valid values: {[e.value for e in SolverType]}") from e
        elif isinstance(solver_id, SolverType):
            solver_type = solver_id
        else:
            raise TypeError(f"Unsupported type for solver_id: {type(solver_id)}. Expected Solver class/instance, Enum, str, or int.")

        #! Map Enum to Class
        target_class = _SOLVER_TYPE_TO_CLASS_MAP.get(solver_type)
        if target_class is None:
            raise NotImplementedError(f"Solver type '{solver_type.name}' is not currently mapped to an implemented class.")

    #! Filter Kwargs for Constructor and Instantiate
    try:
        valid_args      = inspect.signature(target_class.__init__).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args or k in Solver.__init__.__code__.co_varnames}
        ignored_kwargs  = {k: v for k, v in kwargs.items() if k not in filtered_kwargs and k != 'self'}

        if ignored_kwargs:
            print(f"Warning: Ignoring potentially invalid kwargs for {target_class.__name__}: {ignored_kwargs}")
        return target_class(**filtered_kwargs)
    except Exception as e:
        print(f"Error instantiating {target_class.__name__} with kwargs {filtered_kwargs}: {e}")
        raise

# -----------------------------------------------------------------------------
#! Random test method (Keep as provided in solver.py - maybe move to testing utils?)
# -----------------------------------------------------------------------------

def generate_test_mat_vec(make_random: bool,
        symmetric: bool, size: int = 4, dtype=None, backend="default"):
    """
    Generate a test matrix-vector pair (A, b) for solver testing.

    Args:
        make_random (bool): 
            If True, generate random values for A and b. 
            If False, use fixed predefined values.
        symmetric (bool): 
            If True, ensure A is symmetric. 
            If False, A is not symmetric.
        size (int, optional): 
            Size of the square matrix A (default: 4).
        dtype (optional): 
            Data type of the matrix and vector. 
            If None, the default dtype of the backend is used.
        backend (str, optional): 
            Backend to use for matrix and vector generation. 
            If "default", the default backend is used.

    Returns:
        tuple: 
            A tuple (A, b) where:
            - A is a square complex matrix of shape (size, size).
            - b is a complex vector of shape (size,).

    Notes:
        - If make_random is False and size is 4, predefined values are used.
        - If symmetric is True, A is symmetrized using (A + A.H) / 2.
    """
    backend_mod, (rdg, _)   = get_backend(backend, random=True)
    dtype                   = dtype if dtype is not None else backend_mod.float32
    if not make_random and size == 4:
        A = backend_mod.array([
            [1.+0.j, 2.-1.j, 3.+2.j, -1.+1.j],
            [2.+1.j, 2.+0.j, 1.-1.j,  4.+1.j],
            [3.-2.j, 1.+1.j, 3.+0.j,  1.+3.j],
            [-1.-1.j,4.-1.j, 1.-3.j,  1.+0.j]
            ], dtype=dtype)
        if symmetric:
            A = (A + backend_mod.conjugate(A).T) / 2.0
        b = backend_mod.array([1.+2.j, 0.-1.j, 3.-1.j, 2.+3.j], dtype=dtype)
    else:
        # Random complex matrix and vector
        A_real = rdg.random((size, size), dtype=dtype)
        A_imag = rdg.random((size, size), dtype=dtype)
        A = A_real + 1j * A_imag
        if symmetric:
            A = (A + backend_mod.conjugate(A).T) / 2.0
        b_real = rdg.random((size,), dtype=dtype)
        b_imag = rdg.random((size,), dtype=dtype)
        b = b_real + 1j * b_imag
    return A, b

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

__all__ = [
    # Base classes/types
    'Solver', 'SolverResult', 'SolverError', 'SolverErrorMsg', 'SolverType',
    'Array', 'MatVecFunc', 'StaticSolverFunc',
    # Concrete Solver Classes
    'CgSolver', 'CgSolverScipy',
    'DirectSolver', 'DirectScipy', 'DirectJaxScipy', 'DirectInvSolver',
    'PseudoInverseSolver',
    # Factory function
    'choose_solver',
    # Testing utility
    'generate_test_mat_vec'
]

__author__      = "Maksymilian Kliczkowski"
__version__     = "0.1"
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