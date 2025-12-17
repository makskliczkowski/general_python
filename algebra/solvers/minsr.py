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

def _spectral_solve_kernel(matrix, b, rcond, sigma, snr_tol=0.0):
    """
    JIT-friendly core solver.
    Solves (matrix + sigma*I) x = b using Eigendecomposition + SNR Cutoff.
    
    Steps:
    1. Apply diagonal shift (Tikhonov regularization)
    2. Compute eigendecomposition of the shifted matrix
    3. Apply spectral filtering (truncate small eigenvalues)
    4. Reconstruct solution via pseudoinverse
    """
    
    # 1. Apply Diagonal Shift (Tikhonov Regularization)
    # Always apply - when sigma=0 this is a no-op
    # Avoids conditional branching which breaks JIT tracing
    matrix = matrix + sigma * jnp.eye(matrix.shape[0], dtype=matrix.dtype)

    # 2. Hermitian Eigendecomposition
    # Faster and more stable than SVD for square symmetric/hermitian matrices
    w, v = jnp.linalg.eigh(matrix)
    
    # 3. Spectral Filtering (SNR Cutoff)
    # Filter eigenvalues that are indistinguishable from noise (relative to max eigenval)
    max_w   = jnp.max(jnp.abs(w))
    cutoff  = rcond * max_w
    
    # Invert: 1/w if w > cutoff else 0 (pseudoinverse)
    inv_w   = jnp.where(w > cutoff, 1.0 / w, 0.0)
    
    # 4. Reconstruct Solution: x = V @ diag(1/w) @ V^H @ b
    vt_b    = jnp.matmul(v.conj().T, b)
    scaled  = inv_w * vt_b
    x       = jnp.matmul(v, scaled)
    
    return x

# Compile the kernel
_spectral_solve_kernel_jit = jax.jit(_spectral_solve_kernel, static_argnums=(2, 4))

# -----------------------------------------------------------
# The Solver Wrapper Class
# -----------------------------------------------------------

class SpectralExactSolver(Solver):
    """
    Exact solver for MinSR using Eigendecomposition and Spectral Filtering.
    
    This is a direct solver (not iterative) that:
    1. Applies diagonal shift for regularization
    2. Computes eigendecomposition
    3. Filters small eigenvalues (spectral cutoff)
    4. Computes pseudoinverse solution
    
    Best for small to medium sized problems where direct solve is feasible.
    """
    
    _solver_type = SolverType.DIRECT

    @staticmethod
    def get_solver_func(backend_module, use_matvec=True, use_fisher=False, use_matrix=False, sigma=None, **kwargs):
        """
        Returns a solver function compatible with the Solver wrapper interface.
        
        Args:
            backend_module: jax.numpy or numpy
            use_matvec: Ignored (we need direct matrix access)
            use_fisher: If True, expects s, s_p (Fisher/Gram components)
            use_matrix: If True, expects a (dense matrix)
            sigma: Default diagonal shift
            
        Returns:
            Callable solver function
        """
        default_sigma = 0.0 if sigma is None else float(sigma)
        
        def _solve_wrapper(a=None, s=None, s_p=None, b=None, x0=None, tol=1e-10, maxiter=100, 
                          precond_apply=None, sigma=None, snr_tol=0.0, **extra_kwargs):
            """
            Solve the linear system using spectral decomposition.
            
            Parameters:
            ------------
                a: 
                    Dense matrix (if use_matrix=True)
                s, s_p: 
                    Fisher components O and O^H (if use_fisher=True)
                b: 
                    Right-hand side vector
                x0: 
                    Ignored (direct solver)
                tol:
                    Used as spectral cutoff (rcond)
                maxiter: 
                    Ignored (direct solver)
                precond_apply: 
                    Ignored (not needed for direct solve)
                sigma: 
                    Diagonal shift (overrides default)
                snr_tol: 
                    Signal-to-noise ratio tolerance (passed to kernel)
            
            Returns:
                SolverResult with solution
            """
            # Use runtime sigma if provided, else default
            effective_sigma = default_sigma if sigma is None else sigma
            
            # Case 1: Pre-formed Matrix A provided
            if a is not None:
                matrix = a
                
            # Case 2: Fisher/Gram components (s, s_p) provided
            # Compute: matrix = (1/n) * s_p @ s = (1/n) * O^H @ O
            elif s is not None and s_p is not None:
                n_samples   = s.shape[0] if s.shape[0] > s.shape[1] else s.shape[1]
                matrix      = jnp.matmul(s_p, s) / n_samples
                
            else:
                # Fallback: identity (should not happen in practice)
                matrix      = jnp.eye(b.shape[0], dtype=b.dtype)

            # Call the JIT-compiled kernel
            x = _spectral_solve_kernel_jit(matrix, b, tol, effective_sigma, snr_tol)
            return SolverResult(x=x, converged=True, iterations=1, residual_norm=0.0)
        
        # Return the wrapper (no additional JIT needed - kernel is already JIT'd)
        return _solve_wrapper

    @staticmethod
    def solve(matvec, b, x0, tol, maxiter, precond_apply, backend_module, **kwargs):
        """
        Static method to solve the linear system using SpectralExactSolver.
        """
        solver_func = SpectralExactSolver.get_solver_func(backend_module, **kwargs)
        return solver_func(b=b, x0=x0, tol=tol, maxiter=maxiter, precond_apply=precond_apply, **kwargs)
    
# -----------------------------------------------------------
# End of spectral_minsr.py
# -----------------------------------------------------------