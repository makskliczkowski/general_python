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
    JIT-friendly core solver for MinSR.
    Solves (matrix + sigma*I) x = b using Eigendecomposition.
    
    Equivalent to: x = pinv(matrix + sigma*I, rcond=rcond) @ b
    
    Args:
        matrix: The Gram matrix T = O @ O^H
        b: The vector b (local energies)
        rcond: Cutoff for small eigenvalues (relative to max eigenvalue)
        sigma: Diagonal shift (regularization)
    """
    
    # 1. Apply Diagonal Shift (Tikhonov Regularization)
    # Always apply - when sigma=0 this is a no-op
    # Avoids conditional branching which breaks JIT tracing
    matrix = matrix + sigma * jnp.eye(matrix.shape[0], dtype=matrix.dtype)

    # 2. Hermitian Eigendecomposition
    # Faster and more stable than SVD for square symmetric/hermitian matrices
    w, v = jnp.linalg.eigh(matrix)
    
    # 3. Spectral Filtering
    # Filter eigenvalues that are indistinguishable from noise (relative to max eigenval)
    max_w   = jnp.max(jnp.abs(w))
    cutoff  = rcond * max_w
    
    # Invert: 1/w if w > cutoff else 0 (pseudoinverse behavior)
    inv_w   = jnp.where(w > cutoff, 1.0 / w, 0.0)
    
    # 4. Reconstruct Solution: x = V @ diag(1/w) @ V^H @ b
    vt_b    = jnp.matmul(v.conj().T, b)
    scaled  = inv_w * vt_b
    x       = jnp.matmul(v, scaled)
    
    return x

# Compile the kernel
_spectral_solve_kernel_jit = jax.jit(_spectral_solve_kernel)

# -----------------------------------------------------------
# The Solver Wrapper Class
# -----------------------------------------------------------

class SpectralExactSolver(Solver):
    """
    Exact solver for MinSR using Eigendecomposition and Spectral Filtering.
    
    Matches logic from jVMC's MinSR and TDVP implementations:
    - Constructs T = O @ O^H (Gram matrix)
    - Applies diagonal shift (sigma/diagonalShift)
    - Inverts using spectral decomposition with rcond cutoff (tol/pinvTol)
    """
    
    _solver_type = SolverType.DIRECT

    @staticmethod
    def get_solver_func(backend_module, use_matvec=True, use_fisher=False, use_matrix=False, sigma=None, **kwargs):
        """
        Returns a solver function compatible with the Solver wrapper interface.
        """
        default_sigma = 0.0 if sigma is None else float(sigma)
        
        def _solve_wrapper(a=None, s=None, s_p=None, b=None, x0=None, tol=1e-10, maxiter=100, 
                          precond_apply=None, sigma=None, snr_tol=0.0, **extra_kwargs):
            """
            Solve the linear system using spectral decomposition.
            
            Args:
                a: Dense matrix (if use_matrix=True)
                s, s_p: Fisher components O^H and O (if use_fisher=True). 
                        For MinSR: s=O^H (Np, Ns), s_p=O (Ns, Np).
                b: Right-hand side vector
                tol: Used as spectral cutoff (rcond)
                sigma: Diagonal shift (overrides default)
                snr_tol: (Unused in exact kernel currently, kept for interface compatibility)
                **extra_kwargs: Supports 'pinvTol' and 'diagonalShift' aliases from jVMC.
            """
            # Use runtime sigma if provided, else default
            effective_sigma = default_sigma if sigma is None else sigma
            
            # Support aliases from jVMC/other contexts
            # jVMC uses 'pinvTol' for cutoff and 'diagonalShift' for regularization
            rcond       = extra_kwargs.get('rcond', extra_kwargs.get('pinvTol', tol))
            diag_shift  = extra_kwargs.get('diagonalShift', effective_sigma)
            
            # Case 1: Pre-formed Matrix A provided
            if a is not None:
                matrix = a
                
            # Case 2: Fisher/Gram components (s, s_p) provided
            # For MinSR: T = (1/n) * s_p @ s = (1/n) * O @ O^H
            # Note: s_p is O (Ns, Np), s is O^H (Np, Ns)
            elif s is not None and s_p is not None:
                # We normalize by n_samples to be consistent with the 1/N scaling of the loss vector b
                # and to keep sigma (diagonal shift) as an intensive parameter.
                n_samples   = s_p.shape[0] # Assumes O is (Ns, Np)
                matrix      = jnp.matmul(s_p, s) / n_samples
                
            else:
                # Fallback: identity (should not happen in practice)
                matrix      = jnp.eye(b.shape[0], dtype=b.dtype)

            # Call the JIT-compiled kernel
            x = _spectral_solve_kernel_jit(matrix, b, rcond, diag_shift)
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