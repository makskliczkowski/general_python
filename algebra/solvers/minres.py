'''
file        : general_python/algebra/solvers/minres.py
author      : Maksymilian Kliczkowski
'''

from typing import Optional, Callable
import numpy as np

from ..solver import SolverType, Solver, SolverError, SolverErrorMsg
from ..utils import JAX_AVAILABLE, get_backend
from ..preconditioners import Preconditioner

# -----------------------------------------------------------------------------

try:
    import scipy.sparse.linalg as sp_la
    # Check if JAX has minres (it might in newer versions or separate libraries)
    if JAX_AVAILABLE:
        import jax.scipy.sparse.linalg as jsp_la
    else:
        jsp_la  = None
except ImportError:
    # Handle cases where scipy is not installed, though it's common
    sp_la       = None
    jsp_la      = None

# -----------------------------------------------------------------------------
#! MINRES Solver Class using SciPy
# -----------------------------------------------------------------------------

class MinresSolverScipy(Solver):
    '''
    Minimum Residual (MINRES) Solver for symmetric (possibly indefinite)
    matrices using SciPy or compatible backends.

    Solves Ax = b or min ||Ax - b||.
    '''

    def __init__(self,
                backend     ='default',
                size        = 1,
                dtype       = None,
                eps         = 1e-10,
                maxiter     = 1000,
                reg         = None,     # MINRES usually doesn't use reg directly, shifts can be used
                precond     = None,     # Preconditioner to be used (as M in SciPy)
                restart     = False,    # MINRES doesn't typically restart in the same way as GMRES
                maxrestarts = 1):
        super().__init__(backend, size, dtype, eps, maxiter, reg, precond, restart, maxrestarts)
        # MINRES is designed for symmetric matrices
        self._symmetric   = True
        self._solver_type = SolverType.SCIPY_MINRES

    # -------------------------------------------------------------------------

    def solve(self, b, x0: Optional[np.ndarray] = None, precond: Optional[Preconditioner] = None) -> np.ndarray:
        '''
        Solve the linear system Ax = b using MINRES.

        Parameters:
            b : array-like
                The right-hand side vector.
            x0 : array-like, optional
                Initial guess for the solution.
            precond : Preconditioner, optional
                Preconditioner to be used (as M in SciPy).

        Returns:
            array-like
                The solution x.

        Raises:
            SolverError: If convergence fails or SciPy is unavailable.
            NotImplementedError: If the selected backend doesn't support MINRES.
        '''
        super().init(b, x0) # Basic initialization like setting _n, _solution
        self.check_mat_or_matvec() # Ensure _mat_vec_mult is set up

        if precond is not None:
            self._preconditioner = precond
            # Note: Regularization 'reg' might need to be applied *within* _mat_vec_mult
            # if it's a shift, e.g., A' = A + reg*I. MINRES itself doesn't have a 'reg' param.
            # Preconditioner sigma might be relevant if the precond depends on reg.
            if self._reg is not None and self._preconditioner is not None:
                 # Assuming preconditioner knows how to use sigma if needed
                self.set_preconditioner_sigma(self._reg)

        # --- Choose Backend Implementation ---
        # For now, primarily relies on SciPy
        if sp_la is None:
             raise ImportError("scipy.sparse.linalg is required for MinresSolverScipy.")

        current_backend, current_backend_sp = get_backend(self._backend_str, scipy=True)

        # Prepare linear operator for A
        A_op = current_backend_sp.sparse.linalg.LinearOperator(
            (self._n, self._n),
            matvec=lambda v: self._mat_vec_mult(v, self._reg if self._reg is not None else 0.0), # Pass reg to matvec if used as shift
            dtype=self._dtype
        )

        # Prepare linear operator for Preconditioner M
        M_op = None
        if self._preconditioner is not None:
            M_op = current_backend_sp.sparse.linalg.LinearOperator(
                (self._n, self._n),
                matvec=self._preconditioner, # Preconditioner applies M^-1
                dtype=self._dtype
            )

        try:
            # Using SciPy's MINRES
            # Note: SciPy returns x, info. info=0 means success. >0 means no convergence. <0 means illegal input.
            sol, info = current_backend_sp.sparse.linalg.minres(
                A_op,
                b,
                x0=self._solution, # Use initial guess
                tol=self._eps,
                maxiter=self._maxiter,
                M=M_op,
                # Other MINRES options like 'show', 'check' could be added
            )

            self._solution = sol
            if info == 0:
                self._converged = True
            elif info > 0:
                self._converged = False
                # Optionally raise error or just return unconverged solution
                print(f"Warning: MINRES did not converge within {self._maxiter} iterations. Info code: {info}")
                # raise SolverError(SolverErrorMsg.CONV_FAILED)
            else: # info < 0
                self._converged = False
                raise ValueError(f"MINRES failed due to illegal input or breakdown. Info code: {info}")

            # MINRES doesn't track iterations internally in the Solver class in this setup
            # self._iter = ? # SciPy doesn't easily return iter count here

            return self._solution

        except Exception as e:
            self._converged = False
            # More specific error handling could be added based on SciPy exceptions
            raise SolverError(SolverErrorMsg.CONV_FAILED) from e

# -----------------------------------------------------------------------------
# Native MINRES Solver Class (Placeholder)
# -----------------------------------------------------------------------------

class MinresSolver(Solver):
    '''
    Native Minimum Residual (MINRES) Solver for symmetric matrices.
    (Implementation requires the C++ algorithm details)
    '''

    def __init__(self,
                 backend     ='default',
                 size        = 1,
                 dtype       = None,
                 eps         = 1e-10,
                 maxiter     = 1000,
                 reg         = None,
                 precond     = None,
                 restart     = False,
                 maxrestarts = 1):
        super().__init__(backend, size, dtype, eps, maxiter, reg, precond, restart, maxrestarts)
        self._symmetric   = True
        self._solver_type = SolverType.MINRES
        # Add MINRES specific internal variables if needed (like in C++)
        # self.r, self.pkm1, self.pk, ...

    def init(self, b: np.ndarray, x0: Optional[np.ndarray] = None) -> None:
        """
        Initialize solver-specific structures for MINRES.
        """
        super().init(b, x0)
        # Initialize MINRES vectors (r, p_k, Ap_k etc.) here based on b and x0
        # Example:
        # self.r = b - self._mat_vec_mult(self._solution, self._reg if self._reg is not None else 0.0)
        # self.beta0 = self._backend.linalg.norm(self.r)
        # ... other initializations based on MINRES algorithm
        print("Native MinresSolver init called - needs implementation details.")


    def solve(self, b, x0: Optional[np.ndarray] = None, precond: Optional[Preconditioner] = None) -> np.ndarray:
        '''
        Solve the linear system Ax = b using a native MINRES implementation.

        Parameters:
            b : array-like
                The right-hand side vector.
            x0 : array-like, optional
                Initial guess for the solution.
            precond : Preconditioner, optional
                Preconditioner to apply M^-1.

        Returns:
            array-like
                The solution x.

        Raises:
            NotImplementedError: This is currently a placeholder.
        '''
        self.init(b, x0) # Initialize vectors etc.
        self.check_mat_or_matvec()

        if precond is not None:
            self._preconditioner = precond
            if self._reg is not None:
                 self.set_preconditioner_sigma(self._reg)

        # --------------------------------------------
        # --- NATIVE MINRES IMPLEMENTATION NEEDED ----
        # --------------------------------------------
        # This section requires translating the MINRES algorithm, similar to
        # how CgSolver was implemented (potentially with JAX loop).
        # It would involve Lanczos process, Givens rotations, updating solution.
        # Please provide the C++ implementation details for `MINRES_s::solve`.
        # --------------------------------------------

        raise NotImplementedError("Native MinresSolver requires the algorithm implementation based on the C++ version.")

        # Example structure (needs actual algorithm):
        # try:
        #     # Loop up to self._maxiter
        #     #   Perform one MINRES step (Lanczos, Givens, update x, check convergence)
        #     #   Update self._solution, self._iter, self._converged
        #     pass # Replace with actual MINRES loop
        # except Exception as e:
        #     self._converged = False
        #     raise SolverError(SolverErrorMsg.CONV_FAILED) from e
        #
        # return self._solution

    