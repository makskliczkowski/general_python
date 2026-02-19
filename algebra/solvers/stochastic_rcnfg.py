r'''
file    : general_python/algebra/solvers/stochastic_rcnfg.py
author  : Maksymilian Kliczkowski
date    : 2025-04-01
version : 2.0.0

Stochastic Reconfiguration (SR) and Minimum-Step SR (MinSR) algorithms.

This module implements the algebraic backbone of NQS optimization.
It handles the construction of the Quantum Geometric Tensor (S) or the 
Neural Tangent Kernel (T) and solves the linear system for the parameter update.

Mathematical Formulation:
-------------------------
Let O be the matrix of centered log-derivatives (shape N_samples x N_params).
Let E be the vector of centered local energies (shape N_samples).

1. Standard SR (Iterative or Exact):
   Solve: S * update = F
   Where: S = (1/N) * O^dag @ O    (Fisher Matrix, N_p x N_p)
          F = (1/N) * O^dag @ E    (Force Vector)

2. MinSR (Efficient for Deep Networks where N_p >> N_s):
   Solve: T * x = E
   Where: T = O @ O^dag            (Neural Tangent Kernel, N_s x N_s)
   Update: update = O^dag @ x

   Note on MinSR Scaling:
   The implementation ensures scale invariance. We typically normalize T by 1/N
   and E by 1/N to keep numerical values stable, yielding the same physical update.


Standard Stochastic Reconfiguration (SR) and Minimum-Step SR (MinSR)

=====================================================================================================

# Overview

--------
In variational Monte Carlo (VMC), the goal is to optimize the variational 
parameters θ such that the variational wave function |\psi(θ)>
approaches the ground state of a given Hamiltonian.
Both standard stochastic reconfiguration (SR)
[and its efficient variant—minimum-step SR (MinSR) - see below] — aim to update 
the parameters by approximately following an imaginary-time evolution:

    |\psi'> = exp(-H \cdot  delta τ) |\psi(θ)>                              (Eq. 1)

The update is performed by minimizing the Fubini-Study (FS)
distance between the evolved state |\psi'> and the 
variational state |\psi(θ+delta θ)>.

## Fubini-Study Distance
----------------------
For small changes delta θ and a small time step delta τ, the FS distance is expanded as:

    d^2(\psi(θ+delta θ), \psi) = ∑₍\sigma₎ | ∑₍k₎ O₍\sigma,k₎ \cdot  delta θ_k  - \varepsilon₍\sigma₎ |^2    (Eq. 2)

with:
    
    O₍\sigma,k₎ = (1/\psi₍\sigma₎) ∂\psi₍\sigma₎/∂θ_k  - <(1/\psi₍\sigma₎) ∂\psi₍\sigma₎/∂θ_k >,
    
computed over Ns Monte Carlo samples:
    
    \varepsilon₍\sigma₎ = -delta τ \cdot  (E^loc₍\sigma₎ - <E^loc>)/\sqrt(Ns),
    
where the local energy is given by:

    E^loc₍\sigma₎ = ∑₍\sigma'₎ (\psi₍\sigma'₎/\psi₍\sigma₎) \cdot  H₍\sigma,\sigma'₎.

The minimization of this distance is equivalent to solving the linear equation:

    O \cdot  delta θ = \varepsilon                                              (Eq. 3)

## Standard Stochastic Reconfiguration (SR)
------------------------------------------
In conventional SR, one defines the **quantum metric** (or Fisher information matrix) as:

    S = O\dag \cdot  O                                              (Eq. 4)

This metric measures the change in the quantum state induced by a parameter update,
and the FS distance can be 
written as:

    d^2(\psi(θ), \psi(θ+delta θ)) = delta θ\dag \cdot  S \cdot  delta θ                        (Eq. 5)

The SR method then updates the variational parameters using the solution of the linear equation (Eq. 3). The 
standard solution is obtained as:

    delta θ = S^{-1}  \cdot  O\dag \cdot  \varepsilon                                       (Eq. 6)

This approach requires computing and inverting the matrix S,
which is of size NₚxNₚ (Nₚ is the number of 
variational parameters). When Nₚ is large,
the inversion becomes computationally expensive with a typical 
cost scaling of O(Nₚ^3 ), or O(Nₚ^2\cdot Nₛ + Nₚ^3 ) when iterative solvers are employed.
Moreover, for deep networks 
with Nₚ ≫ Nₛ (the number of Monte Carlo samples),
the matrix S is rank-deficient (its rank is at most Nₛ), 
posing additional numerical challenges.

## Minimum-Step Stochastic Reconfiguration (MinSR)
------------------------------------------------
To overcome the computational bottleneck in standard SR, MinSR reformulates the optimization by introducing the 
**neural tangent kernel**:

    T = O \cdot  O\dag

T is an NₛxNₛ matrix and shares the same nonzero eigenvalues as S.
By imposing a minimum-norm (or minimum-step) 
condition—i.e. selecting, among all solutions of Eq. (3),
the one with the smallest ||delta θ||—the MinSR update is 
given by:

    delta θ = O\dag \cdot  T^{-1}  \cdot  \varepsilon                                  (Eq. 5)

This formulation avoids the costly inversion of the full S matrix.
The inversion is now only on the smaller T matrix, 
reducing the computational complexity to approximately O(Nₚ\cdot Nₛ^2 + Nₛ^3 ).
Two derivations support this result:

1. **Lagrangian Multiplier Approach:**  
    The method minimizes ||delta θ|| subject to O \cdot  delta θ = \varepsilon by forming the Lagrangian:

    L({delta θ_k }, {\alpha₍\sigma₎}) = ∑₍k₎ |delta θ_k |^2 - [∑₍\sigma₎ \alpha₍\sigma₎* (∑₍k₎ O₍\sigma,k₎ \cdot  delta θ_k  - \varepsilon₍\sigma₎) + c.c.]

Solving the resulting equations leads to delta θ = O\dag \cdot  (O \cdot  O\dag)^{-1}  \cdot  \varepsilon, which is equivalent to Eq. (5).

2. **Pseudo-Inverse Method:**  
By showing that the least-squares minimum-norm solution of O \cdot  delta θ = \varepsilon is given by:

    delta θ = O\dag \cdot  (O \cdot  O\dag)^{-1}  \cdot  \varepsilon,

and using properties of the pseudo-inverse,
one establishes the equivalence (O \cdot  O\dag)^{-1}  = T^{-1} , thereby recovering 
Eq. (5).

Regularization is typically applied to T^{-1} 
(using a cutoff with, for example, relative tolerance rtol = 1e-12) 
to stabilize the inversion in the presence of small eigenvalues.

Usage Example
-------------
Assume matrices `O` and vector `epsilon` have been obtained from Monte Carlo sampling. The update step in the 
optimization loop can be implemented as follows:

    import numpy as np

    # Compute the neural tangent kernel T
    T = O @ O.conj().T

    # Compute the pseudo-inverse of T with regularization
    T_inv = np.linalg.pinv(T, rtol=1e-12)

    # Compute the parameter update
    delta_theta = O.conj().T @ T_inv @ epsilon

References
----------
For detailed derivations and benchmarks, refer to:
- "Efficient optimization of deep neural quantum states toward machine precision",
'''

import numba
import numpy as np
from enum import Enum, auto
from typing import TYPE_CHECKING
from functools import partial
from abc import ABC

try:
    from ..utils import JAX_AVAILABLE, get_backend, Array
    if TYPE_CHECKING:
        from ..solver import Solver
except ImportError:
    raise ImportError("Please install the 'general_python' package to use stochastic_rcnfg module.")

#####################################
# Configuration
#####################################

class SRMode(Enum):
    STANDARD = auto() # Use N_p x N_p Fisher Matrix S
    MINSR    = auto() # Use N_s x N_s Neural Tangent Kernel T, see 2023 Chen, Heyl

#####################################
# JAX Implementations
#####################################

# jax specific
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    
    # @jax.jit
    def loss_centered_jax(loss: jnp.ndarray, loss_m: jnp.ndarray) -> jnp.ndarray:
        """L_c = L - <L>"""
        return loss - loss_m

    # @jax.jit
    def loss_centered_jax_modified_ratios(loss, loss_m, betas, r_el, r_le):
        """
        Adjusts local energy vector to include excited state penalties.
        This allows solving (S * dt = F_total) using the standard solver infrastructure.
        """
        
        # 1. Standard centering
        loss_c      = loss - loss_m 

        # 2. Penalty Correction (Effective Energy modification)
        # Force F_penalty = sum_j beta_j * (<psi|psi_j><psi_j|O|psi> - ...)
        # We map this to E_eff so that <O E_eff> = F_total
        
        # r_el: ratios excited/lower on configs
        # r_le: ratios lower/excited on configs_j (lower samples)
        
        # Mean overlaps
        r_el_mean   = jnp.mean(r_el, axis=1) # <psi_exc | psi_low>
        r_le_mean   = jnp.mean(r_le, axis=1) # <psi_low | psi_exc>

        # Centered term on lower samples
        delta_r_le  = r_le - r_le_mean[:, None]

        # Correction vector (broadcasted)
        # This implementation assumes we approximate the mixed expectation values 
        # by reweighting the current samples.
        # Simplified proxy for the full orthogonality force:
        corr        = jnp.sum(betas[:, None] * delta_r_le * r_el_mean[:, None], axis=0)
        
        return loss_c + corr

    def derivatives_centered_jax(derivatives: jnp.ndarray, derivatives_m: jnp.ndarray) -> jnp.ndarray:
        """O_c = O - <O>"""
        return derivatives - derivatives_m

    # @jax.jit
    def gradient_jax(derivatives_c: jnp.ndarray, loss_c: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        """F = (1/N) * O^dag @ E_c"""
        return jnp.matmul(derivatives_c.T.conj(), loss_c) / n_samples
    
    # @jax.jit
    def gradient_jax_batched(batched_jacobian, loss_c: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        """
        F = (1/N) * O^dag @ E_c using BatchedJacobian.
        O_c = J - 1 m^T (conceptually).
        F = <O_c^* E_c> = 1/N * sum(O^* * E_c).
        """
        return batched_jacobian.compute_weighted_sum(loss_c.conj()).conj() / n_samples

    # @jax.jit
    def covariance_jax(derivatives_c: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        """S = (1/N) * O^dag @ O (Standard SR)"""
        return (derivatives_c.T.conj() @ derivatives_c) / n_samples

    @jax.jit
    def covariance_jax_minsr(derivatives_c: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        """T = (1/N) * O @ O^dag (MinSR)"""
        # Note: derivatives_c is    (N_samples, N_params)
        # We want                   (N_s, N_s).
        # Correct math:             T = O . O^dag
        return (derivatives_c @ derivatives_c.T.conj()) / n_samples
    
    # -------------------------------------------------------
    
    def get_matvec_batched(mat_O, mode, n_samples):
        """
        Constructs a matrix-vector product function for BatchedJacobian.
        
        Handles centering corrections implicitly:
        Standard: S v = (1/N) (O - m)^dag (O - m) v
        MinSR:    T v = (1/N) (O - m) (O - m)^dag v
        """
        
        # Ensure mean is available
        mean_val = getattr(mat_O, "mean_val", None)
        if mean_val is None:
            mean_val = mat_O.mean(axis=0)
            
        if mode == 'minsr':
            # T = (1/N) (J - m) (J - m)^dag
            # v in Sample space (N_samples)
            def _matvec(v, sigma):
                # 1. Apply (J - m)^dag v = J^dag v - m^* (1^T v)
                sum_v           = jnp.sum(v)
                J_dag_v         = mat_O.rmv(v)
                inter           = J_dag_v - jnp.conj(mean_val) * sum_v
                
                # 2. Apply (J - m) inter = J inter - m (1 . inter) (Wait, 1.inter is dot?)
                J_inter         = mat_O.mv(inter)
                m_dot_inter     = jnp.sum(mean_val * inter)
                res             = J_inter - m_dot_inter
                
                return res / n_samples + sigma * v
        else:
            # S = (1/N) (J - m)^dag (J - m)
            # v in Parameter space
            def _matvec(v, sigma):
                # 1. Apply O_c v = J v - 1 (m . v)
                m_dot_v         = jnp.sum(mean_val * v)
                J_v             = mat_O.mv(v)
                inter           = J_v - m_dot_v # Broadcast subtraction
                
                # 2. Apply O_c^dag inter = J^dag inter - m^* sum(inter)
                sum_inter       = jnp.sum(inter)
                J_dag_inter     = mat_O.rmv(inter)
                res             = J_dag_inter - jnp.conj(mean_val) * sum_inter
                
                return res / n_samples + sigma * v
                
        return _matvec

    def get_matvec_dense(mat_O, mode, n_samples):
        """
        Constructs a matrix-vector product function for dense matrices.
        """
        if mode == 'minsr':
             def _matvec(v, sigma):
                 inter = jnp.matmul(mat_O.T.conj(), v)
                 res   = jnp.matmul(mat_O, inter)
                 return res / n_samples + sigma * v
        else:
             def _matvec(v, sigma):
                 inter = jnp.matmul(mat_O, v)
                 res   = jnp.matmul(mat_O.T.conj(), inter)
                 return res / n_samples + sigma * v
        return _matvec

    @jax.jit
    def solve_jax_prepare(loss: jnp.ndarray, var_deriv: jnp.ndarray):
        """Standard preprocessing."""
        n_samples       = loss.shape[0]
        loss_m          = jnp.mean(loss, axis=0)
        loss_c          = loss_centered_jax(loss, loss_m)
        
        if hasattr(var_deriv, "compute_weighted_sum"):
            # BatchedJacobian
            full_size           = var_deriv._n_params
            n_samples           = var_deriv._n_samples
            var_deriv_m         = var_deriv.mean(axis=0)
            var_deriv_c         = var_deriv # Treat the object as the centered derivative carrier
            return loss_c, var_deriv_c, var_deriv_m, n_samples, full_size

        full_size       = var_deriv.shape[1]
        var_deriv_m     = jnp.mean(var_deriv, axis=0)
        var_deriv_c     = derivatives_centered_jax(var_deriv, var_deriv_m)
        # var_deriv_c_h   = jnp.conj(var_deriv_c.T) # Optimized out
        
        return loss_c, var_deriv_c, var_deriv_m, n_samples, full_size

    @jax.jit
    def solve_jax_prepare_modified_ratios(loss: jnp.ndarray, var_deriv: jnp.ndarray, betas: jnp.ndarray, r_el: jnp.ndarray, r_le: jnp.ndarray):
        """Excited state preprocessing."""
        n_samples       = loss.shape[0]
        loss_m          = jnp.mean(loss, axis=0)
        
        # Use the modified loss calculator
        loss_c          = loss_centered_jax_modified_ratios(loss, loss_m, betas, r_el, r_le)
        
        if hasattr(var_deriv, "compute_weighted_sum"):
            # BatchedJacobian
            full_size           = var_deriv._n_params
            n_samples           = var_deriv._n_samples
            var_deriv_m         = var_deriv.mean(axis=0)
            var_deriv_c         = var_deriv
            return loss_c, var_deriv_c, var_deriv_m, n_samples, full_size

        full_size       = var_deriv.shape[1]
        var_deriv_m     = jnp.mean(var_deriv, axis=0)
        var_deriv_c     = derivatives_centered_jax(var_deriv, var_deriv_m)
        # var_deriv_c_h   = jnp.conj(var_deriv_c.T) # Optimized out
        
        return loss_c, var_deriv_c, var_deriv_m, n_samples, full_size

    # Unified Solver Logic

    @partial(jax.jit, static_argnames=('solve_func', 'min_sr', 'precond_apply', 'maxiter', 'tol'))
    def solve_jax(solve_func, loss: jnp.ndarray, var_deriv: jnp.ndarray, min_sr: bool, x0=None, precond_apply=None, maxiter=500, tol=1e-8):
        """
        Unified solver entry point.
        """
        
        # Prepare
        loss_c, O_c, _, N, _ = solve_jax_prepare(loss, var_deriv)
        
        if min_sr:
            # MinSR: T = (1/N) O O^dag. Solve T x = E/N. Update = O^dag x.
            # Calculate T
            T       = covariance_jax_minsr(O_c, N)
            rhs     = loss_c / N
            
            # Solve T x = rhs
            # We pass T as 's' to the generic solver
            x_minsr = solve_func(s=T, s_p=None, b=rhs, x0=None, precond_apply=precond_apply, maxiter=maxiter, tol=tol)
            
            # Map back to params
            return jnp.matmul(O_c.T.conj(), x_minsr)
        else:
            # Standard: S = (1/N) O^dag O. Solve S x = F.
            F       = gradient_jax(O_c, loss_c, N)
            
            # Solve S x = F
            # Pass (O^dag, O) so solver can form S lazily or explicitly
            return solve_func(s=O_c.T.conj(), s_p=O_c, b=F, x0=x0, precond_apply=precond_apply, maxiter=maxiter, tol=tol)    

#####################################
# NumPy Implementation (Legacy/CPU)
#####################################

if True:
    @numba.njit
    def loss_centered(loss, loss_m):
        return loss - loss_m
    
    @numba.njit
    def derivatives_centered(derivatives, derivatives_m):
        return derivatives - derivatives_m
    
    @numba.njit
    def gradient_np(derivatives_c, loss_c, num_samples):
        # Implicit transpose: (N_s, N_p).conj().T -> (N_p, N_s)
        # Numba supports matmul with array.T
        return np.matmul(derivatives_c.conj().T, loss_c) / num_samples

    @numba.njit
    def covariance_np(derivatives_c, num_samples):
        # S = O^dag O = (N_p, N_s) @ (N_s, N_p)
        return np.matmul(derivatives_c.conj().T, derivatives_c) / num_samples

    @numba.njit
    def covariance_np_minsr(derivatives_c, num_samples):
        # T = O O^dag = (N_s, N_p) @ (N_p, N_s)
        return np.matmul(derivatives_c, derivatives_c.conj().T) / num_samples

    def get_matvec_dense_np(mat_O, mode, n_samples):
        """
        Constructs a matrix-vector product function for dense matrices (NumPy).
        """
        if mode == 'minsr':
             def _matvec(v, sigma):
                 inter = np.matmul(mat_O.conj().T, v)
                 res   = np.matmul(mat_O, inter)
                 return res / n_samples + sigma * v
        else:
             def _matvec(v, sigma):
                 inter = np.matmul(mat_O, v)
                 res   = np.matmul(mat_O.conj().T, inter)
                 return res / n_samples + sigma * v
        return _matvec

    @numba.njit
    def solve_numpy_prepare(loss, var_deriv):
        n_samples = loss.shape[0]
        loss_m = np.mean(loss, axis=0)
        loss_c = loss_centered(loss, loss_m)
        var_deriv_m = np.mean(var_deriv, axis=0)
        var_deriv_c = derivatives_centered(var_deriv, var_deriv_m)
        # var_deriv_c_h = np.conj(var_deriv_c).T # Optimized out
        return loss_c, var_deriv_c, var_deriv_m, n_samples, var_deriv.shape[1]

    # NumPy solver wrapper (simplified)
    def solve_numpy(solver, loss, var_deriv, min_sr, **kwargs):
        loss_c, O_c, _, N, _ = solve_numpy_prepare(loss, var_deriv)
        
        # Compute explicit O_dag locally if needed for legacy solver interfaces
        # Most solvers expect (S, S_p) or A.
        # Standard: S = O^dag O. Pass (O^dag, O)
        # MinSR: T = O O^dag. Pass T.
        
        if min_sr:
            T = covariance_np_minsr(O_c, N)
            rhs = loss_c / N
            if hasattr(solver, 'init_from_matrix'): solver.init_from_matrix(T, rhs)
            x_sol = solver.solve(rhs, **kwargs)
            
            # Update = O^dag x
            return np.matmul(O_c.conj().T, x_sol)
        else:
            F = gradient_np(O_c, loss_c, N)
            
            # Construct O_dag for solver (some solvers might need explicit matrix)
            # If solver supports implicit operations, we could avoid this, but 
            # Solver interface usually takes (s, s_p) which are matrices/arrays.
            O_dag = O_c.conj().T
            
            if hasattr(solver, 'init_from_fisher'): solver.init_from_fisher(O_dag, O_c, F)
            return solver.solve(F, **kwargs)
        
#####################################

class StochasticReconfiguration(ABC):
    '''
    High-level handler for SR updates.
    '''
    
    def __init__(self, solver: 'Solver', backend: str = 'default'):
        self._backend_type      = get_backend(backend)
        self._isjax             = (self._backend_type != np)
        self._solver: Solver    = solver
        self._solution          = None

        # Expose JIT-compiled functions to call depending on backend
        if self._isjax:
            self._gradient_fun                      = gradient_jax
            self._covariance_fun                    = covariance_jax
            self._covariance_minres_fun             = covariance_jax_minsr
            self._prepare_fun                       = solve_jax_prepare
            # Add back missing hooks
            self.solve_jax_prepare                  = solve_jax_prepare
            self.solve_jax_prepare_modified_ratios  = solve_jax_prepare_modified_ratios
        else:
            raise NotImplementedError("Only JAX backend is supported currently.")
            # self._gradient_fun                      = gradient_np
            # self._covariance_fun                    = covariance_np
            # self._prepare_fun                       = solve_numpy_prepare

    def solve(self, loss: Array, var_deriv: Array, use_minsr=False, **kwargs):
        solve_fn = self._solver.solve if hasattr(self._solver, 'solve') else self._solver
        if self._isjax:
            return solve_jax(solve_fn, loss, var_deriv, use_minsr, **kwargs)
        else:
            return solve_numpy(self._solver, loss, var_deriv, use_minsr, **kwargs)

######################################
