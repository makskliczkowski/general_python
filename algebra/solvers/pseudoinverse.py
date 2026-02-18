r'''
Implements a solver for linear systems $ Ax = b $ using the Moore-Penrose
pseudoinverse $ A^+ $. The solution obtained is the minimum norm solution
among all vectors $ x $ that minimize the least-squares residual $ ||Ax - b||_2 $.

Mathematical Formulation:
------------------------
The Moore-Penrose pseudoinverse $ A^+ $ of a matrix $ A $ is unique and satisfies:
1. $ A A^+ A = A $
2. $ A^+ A A^+ = A^+ $
3. $ (A A^+)^* = A A^+ $ (conjugate transpose)
4. $ (A^+ A)^* = A^+ A $

The minimum norm least-squares solution to $ Ax = b $ is given by:
$ x = A^+ b $

If a regularization (or shift) $ \sigma $ is applied, the solver computes:
$ x = (A + \sigma I)^+ b $
where $ I $ is the identity matrix. This can help with ill-conditioned matrices.

Computationally, $ A^+ $ is typically found via the Singular Value Decomposition (SVD).
If $ A = U \Sigma V^H $ is the SVD of A, then $ A^+ = V \Sigma^+ U^H $, where
$ \Sigma^+ $ is obtained by taking the reciprocal of the non-zero singular values
in $ \Sigma $ and transposing.

References:
-----------
    - Penrose, R. (1955). A generalized inverse for matrices. Proc. Cambridge Philos. Soc.
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). JHU Press. Chapter 5.
    
------------------------------------------------------------------------------
File            : general_python/algebra/solvers/pseudoinverse.py
Author          : Maksymilian Kliczkowski
Date            : 2025-10-15
Description     : Pseudo-inverse solver for linear systems using NumPy/JAX.
License         : MIT
------------------------------------------------------------------------------
'''

from typing import Optional, Callable, Union, Any, Type, Tuple
import numpy as np

# Assuming these are correctly imported from the solver module
try:
    from ..solver import Solver, SolverResult, SolverError, SolverErrorMsg, SolverType, Array, MatVecFunc, StaticSolverFunc
    from ..preconditioners import Preconditioner
except ImportError:
    raise ImportError("Could not import base solver classes. Check the module structure.")

# Check for JAX availability
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE   = True
except ImportError:
    jax             = None
    jnp             = np
    JAX_AVAILABLE   = False

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    _NUMBA_AVAILABLE = False

# -----------------------------------------------------------------------------
#! Core Logic for Pseudo-Inverse Solve
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    def _pinv_jax_core(matrix_a     : jnp.ndarray, 
                       b            : jnp.ndarray, 
                       tol          : float,
                       sigma        : float = 0.0,
                       hermitian    : bool = False) -> Tuple[jnp.ndarray, float]:
        """
        JAX implementation of (A + sigma*I)^+ b.
        """
        sigma_val = 0.0 if sigma is None else sigma
        matrix_a = matrix_a + sigma_val * jnp.eye(matrix_a.shape[0], dtype=matrix_a.dtype)
        
        # Compute Pseudo-Inverse
        # rcond     =   tol ensures we filter small singular values
        # hermitian =   False is safer generally, but could be True for Fisher matrices.
        # We leave it False to be general-purpose.
        a_pinv      = jnp.linalg.pinv(matrix_a, rcond=tol, hermitian=hermitian)
        x           = jnp.dot(a_pinv, b)
        
        # Compute Residual (for reporting)
        # r = b - A_orig * x (Use original or reg? Usually reg for 'solve' check)
        # Let's compute residual against the regularized system solved.
        residual    = b - jnp.dot(matrix_a, x)
        res_norm    = jnp.linalg.norm(residual)
        
        return x, res_norm

def _pinv_numpy_core(matrix_a: np.ndarray, b: np.ndarray, tol: float, sigma: float):
    """NumPy implementation."""
    if sigma:
        matrix_a = matrix_a + sigma * np.eye(matrix_a.shape[0], dtype=matrix_a.dtype)
    
    # NumPy uses rcond
    a_pinv      = np.linalg.pinv(matrix_a, rcond=tol)
    x           = a_pinv @ b
    res_norm    = np.linalg.norm(b - matrix_a @ x)
    return x, res_norm

# -----------------------------------------------------------------------------
#! Pseudo-inverse Solver Class
# -----------------------------------------------------------------------------

class PseudoInverseSolver(Solver):
    r'''
    Solves $ Ax = b $ using the Moore-Penrose pseudoinverse $ x = A^+ b $.
    Provides the minimum norm least-squares solution.
    Supports regularization $ x = (A + \sigma I)^+ b $.
    '''
    _solver_type    = SolverType.PSEUDO_INVERSE
    _symmetric      = False # Pseudo-inverse works for non-symmetric matrices as well

    # --------------------------------------------------
    #! Static Methods Implementation
    # --------------------------------------------------

    @staticmethod
    def get_solver_func(backend_module  : Any,
                        use_matvec      : bool              = True,
                        use_fisher      : bool              = False,
                        use_matrix      : bool              = False,
                        sigma           : Optional[float]   = None, 
                        a               : Optional[Any]     = None, 
                        s               : Optional[Any]     = None, 
                        s_p             : Optional[Any]     = None, **kwargs) -> StaticSolverFunc:
        """
        Returns a backend-adapted function compatible with the common solver wrapper.

        The returned function conforms to the StaticSolverFunc signature used by
        Solver._solver_wrap_compiled and internally calls PseudoInverseSolver.solve,
        forwarding the matrix as kwarg 'A' (accepting either 'a' or 'A').
        """
        
        if backend_module is jnp:
            if not JAX_AVAILABLE: raise ImportError("JAX not installed.")
            sigma_default = 0.0 if sigma is None else float(sigma)

            # Static arguments for JIT: tol, maxiter, precond_apply (ignored but present in signature)
            static_argnames = ('tol', 'maxiter', 'precond_apply')

            # Fisher Mode: (S, Sp) -> Form Matrix -> Pinv
            if use_fisher:
                def wrapper_fisher(s, s_p, b, x0, tol, maxiter, precond_apply=None, sigma=None, **ignored_kwargs):
                    
                    # Form Gram Matrix: A = S^H @ S / N
                    # Optimize contraction: (params, samples) @ (samples, params)
                    n_samples   = s.shape[0]
                    matrix_a    = jnp.dot(s_p, s) / n_samples
                    sigma_val   = sigma_default if sigma is None else sigma
                    # Fisher/Gram matrix is Hermitian by construction.
                    x, res      = _pinv_jax_core(matrix_a, b, tol, sigma=sigma_val, hermitian=True)
                    return SolverResult(x, True, 1, res)
                
                return jax.jit(wrapper_fisher, static_argnames=static_argnames)

            # Matrix Mode: A -> Pinv
            elif use_matrix:
                def wrapper_matrix(a, b, x0, tol, maxiter, precond_apply=None, sigma=None, **ignored_kwargs):
                    sigma_val = sigma_default if sigma is None else sigma
                    x, res  = _pinv_jax_core(a, b, tol, sigma=sigma_val, hermitian=False)
                    return SolverResult(x, True, 1, res)

                return jax.jit(wrapper_matrix, static_argnames=static_argnames)

            else:
                raise SolverError(SolverErrorMsg.INVALID_INPUT, "PseudoInverse requires use_matrix=True or use_fisher=True")

        #! NumPy Backend
        elif backend_module is np:
            sigma_default = 0.0 if sigma is None else float(sigma)
            
            if use_fisher:
                def wrapper_np_fisher(s, s_p, b, x0, tol, maxiter, precond_apply=None, sigma=None, **kwargs):
                    reg         = sigma_default if sigma is None else sigma
                    n           = s.shape[0]
                    matrix_a    = (s_p @ s) / n
                    x, res      = _pinv_numpy_core(matrix_a, b, tol, reg)
                    return SolverResult(x, True, 1, res)
                return wrapper_np_fisher
            
            else:
                def wrapper_np_matrix(a, b, x0, tol, maxiter, precond_apply=None, sigma=None, **kwargs):
                    reg         = sigma_default if sigma is None else sigma
                    x, res      = _pinv_numpy_core(a, b, tol, reg)
                    return SolverResult(x, True, 1, res)
                return wrapper_np_matrix

        else:
            raise ValueError(f"Unsupported backend: {backend_module}")

    # --------------------------------------------------

    @staticmethod
    def solve(
            matvec          : MatVecFunc,                   # Ignored
            b               : Array,
            x0              : Array,                        # Ignored
            *,
            tol             : float,                        # rcond
            maxiter         : int,                          # Ignored
            precond_apply   : Optional[Callable]    = None, # Ignored
            backend_module  : Any                   = jnp,
            a               : Optional[Array]       = None,
            s               : Optional[Array]       = None,
            s_p             : Optional[Array]       = None,
            **kwargs        : Any) -> SolverResult:
        r"""
        Static solve implementation using the backend's `linalg.pinv`.

        Args:
            matvec, x0, maxiter, precond_apply: Ignored.
            b (Array): RHS vector.
            tol (float): Tolerance (rcond/rtol) for `linalg.pinv`.
            backend_module (Any): Backend (numpy or jax.numpy).
            A (Array): **Required** kwarg. Matrix $ A $.
            sigma (float, optional): **Optional** kwarg. Regularization $ \\sigma $.
            **kwargs: Other ignored arguments.

        Returns:
            SolverResult: Contains the minimum norm least-squares solution.
        """
        sigma       = kwargs.get('sigma')
        
        # Determine mode
        use_fisher  = (s is not None)
        use_matrix  = (a is not None and not use_fisher)

        if not (use_fisher or use_matrix):
             raise SolverError(SolverErrorMsg.MAT_NOT_SET, "PseudoInverse requires matrix A or Fisher factors S, Sp.")

        # Get solver function (creating simple wrapper here without caching)
        solver_func = PseudoInverseSolver.get_solver_func(
                            backend_module, 
                            use_matvec      =   False,
                            use_fisher      =   use_fisher,
                            use_matrix      =   use_matrix,
                            sigma           =   sigma
                        )
        
        # Run it using the Solver base helper
        return Solver.run_solver_func(backend_module, solver_func, matvec=None, 
            a=a, s=s, s_p=s_p, b=b, x0=x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply)
    
# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------
