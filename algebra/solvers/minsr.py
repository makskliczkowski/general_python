'''
This module implements an exact solver for the MinSR linear system
using spectral regularization via eigendecomposition.
It is based on the method described in arXiv:2302.01941.

-----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2025-12-01
Description     : Spectral Exact Solver for MinSR
-----------------------------------------------------------
'''

import jax
import jax.numpy as jnp

try:
    from ..solver import Solver, SolverResult, SolverType
except ImportError as e:
    raise ImportError("Solver base class not found. Ensure the algebra.solvers module is accessible.") from e

# -----------------------------------------------------------
# The JIT Kernel (The math from arXiv:2302.01941)
# -----------------------------------------------------------

def _spectral_solve_kernel(matrix, b, rcond, sigma):
    """
    JIT-friendly core solver.
    Solves (matrix + sigma*I) x = b using Eigendecomposition + SNR Cutoff.
    """
    
    # 1. Apply Diagonal Shift (Tikhonov Regularization)
    # This is crucial even with Spectral Cutoff to handle null spaces gracefully.
    if sigma is not None and sigma > 0.0:
        # matrix is (N, N)
        matrix = matrix + sigma * jnp.eye(matrix.shape[0])

    # 2. Hermitian Eigendecomposition
    # faster and more stable than SVD for square symmetric/hermitian matrices
    w, v    = jnp.linalg.eigh(matrix)
    
    # 3. Spectral Filtering (SNR Cutoff)
    # Filter eigenvalues that are indistinguishable from noise (relative to max eigenval)
    max_w   = jnp.max(jnp.abs(w))
    cutoff  = rcond * max_w
    
    # Invert: 1/w if w > cutoff else 0
    inv_w   = jnp.where(w > cutoff, 1.0 / w, 0.0)
    
    # 4. Reconstruct Solution: x = V @ diag(1/w) @ V^H @ b
    vt_b    = jnp.matmul(v.conj().T, b)
    scaled  = inv_w * vt_b
    x       = jnp.matmul(v, scaled)
    
    return x

# -----------------------------------------------------------
# The Solver Wrapper Class
# -----------------------------------------------------------

class SpectralExactSolver(Solver):
    """
    Exact solver for MinSR using Eigendecomposition and Spectral Filtering.
    """
    
    _solver_type = SolverType.DIRECT

    @staticmethod
    def get_solver_func(backend_module, **kwargs):
        # We ignore 'use_matvec' because we need the raw matrices.
        
        def _solve_wrapper(matvec, b, x0, tol, maxiter, precond_apply, a=None, s=None, s_p=None, sigma=0.0):
            """
            Args:
                matvec: Ignored (we use direct matrix)
                tol:    Used as SNR Cutoff (rcond)
                sigma:  Diagonal shift
                a:      Dense matrix (if use_matrix=True)
                s, s_p: Fisher components (if use_fisher=True)
            """
            
            # Case 1: Pre-formed Matrix A (or T) provided
            if a is not None:
                matrix      = a
                
            # Important: s, s_p are the MinSR matrices
            elif s is not None and s_p is not None:
                n_samples   = s_p.shape[0]
                matrix      = jnp.matmul(s_p, s) / n_samples
                
            else:
                matrix      = jnp.eye(b.shape[0]) # Dummy fallback

            # Call the kernel
            x = _spectral_solve_kernel(matrix, b, rcond=tol, sigma=sigma)
            return SolverResult(x, True, 1, 0.0)

    @staticmethod
    def solve(matvec, b, x0, tol, maxiter, precond_apply, backend_module, **kwargs):
        ''' Static method to solve the linear system using SpectralExactSolver. '''
        solver_func = SpectralExactSolver.get_solver_func(backend_module, **kwargs)
        return solver_func(matvec, b, x0, tol, maxiter, precond_apply, **kwargs)
    
# -----------------------------------------------------------
# End of spectral_minsr.py
# -----------------------------------------------------------