# solvers/__init__.py
from typing import Union, Optional, Callable
from enum import Enum, auto, unique

# Import solvers from submodules.
from .cg import CgSolver, CgSolverScipy
from .direct import DirectSolver, DirectScipy, DirectBackend
from .pseudoinverse import PseudoInverseSolver

# -----------------------------------------------------------------------------

from general_python.algebra.utils import get_backend as __backend
from general_python.algebra.solver import SolverType, _SOL_TYPE_ERROR

# -----------------------------------------------------------------------------
# Helper function: choose_solver
# -----------------------------------------------------------------------------

def choose_solver(solver_type, backend: str, size: int, 
        eps: float = 1e-10, maxiter: int = 1000, reg: Optional[float] = None):
    """
    Choose and instantiate a solver based on the provided type.
    
    Parameters:
        solver_type: Either a string or a SolverType enum value.
        backend (str): Backend string (e.g. "default", "jax", etc.).
        size (int): The size of the system (number of columns of A).
        eps (float): Convergence tolerance.
        maxiter (int): Maximum iterations.
        reg: Regularization parameter.
    
    Returns:
        An instance of the chosen solver.
    """
    # Convert string input to enum if needed.
    if isinstance(solver_type, str):
        try:
            solver_type = SolverType[solver_type.upper()]
        except KeyError as exc:
            raise ValueError(_SOL_TYPE_ERROR) from exc
    
    if solver_type == SolverType.DIRECT:
        return DirectSolver(backend=backend, size=size, eps=eps, maxiter=maxiter, reg=reg)
    elif solver_type in (SolverType.BACKEND_SOLVER, SolverType.CJ):
        return CgSolver(backend=backend, size=size, eps=eps, maxiter=maxiter, reg=reg)
    elif solver_type == SolverType.SCIPY_DIRECT:
        return DirectScipy(backend=backend, size=size, eps=eps, maxiter=maxiter, reg=reg)
    elif solver_type == SolverType.SCIPY_CJ:
        return CgSolverScipy(backend=backend, size=size, eps=eps, maxiter=maxiter, reg=reg)
    elif solver_type == SolverType.CJ:
        return CgSolver(backend=backend, size=size, eps=eps, maxiter=maxiter, reg=reg)
    else:
        raise ValueError(f"Unsupported solver type: {solver_type}")
    return None

# -----------------------------------------------------------------------------
# Random test method for solvers
# -----------------------------------------------------------------------------

def generate_test_mat_vec(make_random: bool, symmetric: bool, size: int = 4, dtype=None, backend = "default"):
    """
    Generate a pair (A, b) for testing solvers.
    
    Parameters:
        make_random (bool): If True, generate random A and b.
        symmetric (bool): If True and not random, produce a symmetric A.
        dtype: The desired dtype for the output. If a real type (np.float64 or np.float32)
            then the real parts are returned.
    Returns:
        A tuple (A, b) where A is a 4x4 matrix and b is a 4x1 vector.
    """
    # Create a complex matrix and vector of fixed size.
    backend_str         = backend
    backend, (rdg, _)   = __backend(backend, random = True)
    dtype               = dtype if dtype is not None else backend.float32
    
    if not make_random and size == 4:
        A = backend.zeros((size, size), dtype=dtype)
        b = backend.empty((size,), dtype=dtype)
        # Fill upper triangle and diagonal
        A[0, 0] = 1 + 0j
        A[0, 1] = 2 - 1j
        A[0, 2] = 3 + 2j
        A[0, 3] = -1 + 1j

        A[1, 1] = 2 + 0j
        A[1, 2] = 1 - 1j
        A[1, 3] = 4 + 1j

        A[2, 2] = 3 + 0j
        A[2, 3] = 1 + 3j

        A[3, 3] = 1 + 0j

        if symmetric:
            # Fill lower triangle with conjugates.
            A[1, 0] = backend.conj(A[0, 1])
            A[2, 0] = backend.conj(A[0, 2])
            A[2, 1] = backend.conj(A[1, 2])
            A[3, 0] = backend.conj(A[0, 3])
            A[3, 1] = backend.conj(A[1, 3])
            A[3, 2] = backend.conj(A[2, 3])
        else:
            # Fill lower triangle with different (non-conjugate) values.
            A[1, 0] = 2 + 1j
            A[2, 0] = 3 - 2j
            A[2, 1] = 1 + 1j
            A[3, 0] = -1 - 1j
            A[3, 1] = 4 - 1j
            A[3, 2] = 1 - 3j

        # Fixed vector b.
        b[0] = 1 + 2j
        b[1] = 0 - 1j
        b[2] = 3 - 1j
        b[3] = 2 + 3j
    else:
        # Random matrix and vector.
        A = rdg.random(size=(size, size), dtype=dtype)
        b = rdg.random(size=(size,), dtype=dtype)
        if dtype not in (backend.float32, backend.float64):
            if symmetric:
                A = (A + A.T.conj()) / 2.0
            return A, b
        A = A + 1j * rdg.random(size=(size, size), dtype=dtype)
        b = b + 1j * rdg.random(size=(size,), dtype=dtype)
        if symmetric:
            A = (A + A.T.conj()) / 2.0
        return A, b

# -----------------------------------------------------------------------------

_all_ = [
    'CgSolver', 'CgSolverScipy', 'DirectSolver', 'DirectScipy', 'DirectBackend', 'PseudoInverseSolver',
    'choose_solver', 'generate_test_mat_vec'
]

# -----------------------------------------------------------------------------