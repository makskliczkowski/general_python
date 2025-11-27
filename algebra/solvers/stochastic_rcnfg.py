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

import numpy as np
from enum import Enum, auto
from typing import Union, Tuple, Callable, Optional, NamedTuple
from functools import partial
from abc import ABC, abstractmethod

try:
    from ..utils import JAX_AVAILABLE, get_backend, Array
    from .. import solver as solver_utils
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
    
    @jax.jit
    def center_data(data: jnp.ndarray, mean_val: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Calculates A_centered = A - <A>."""
        if mean_val is None:
            mean_val = jnp.mean(data, axis=0)
        return data - mean_val

    @jax.jit
    def compute_forces(O_dag: jnp.ndarray, E_c: jnp.ndarray, n_samples: float) -> jnp.ndarray:
        """
        Computes Force vector F = (1/N) * O^dag @ E_c
        O_dag: (N_params, N_samples)
        E_c:   (N_samples,)
        """
        return jnp.dot(O_dag, E_c) / n_samples

    @jax.jit
    def compute_fisher_matrix(O_c: jnp.ndarray, O_dag: jnp.ndarray, n_samples: float) -> jnp.ndarray:
        """
        Computes Fisher Matrix S = (1/N) * O^dag @ O
        Returns shape (N_params, N_params) -- WARNING: Large for deep nets.
        """
        return jnp.matmul(O_dag, O_c) / n_samples

    @jax.jit
    def compute_ntk_matrix(O_c: jnp.ndarray, O_dag: jnp.ndarray, n_samples: float) -> jnp.ndarray:
        """
        Computes Neural Tangent Kernel 
            T = (1/N) * O @ O^dag
        Returns shape (N_samples, N_samples) -- Efficient for MinSR.
        """
        # O_c is (N_s, N_p), O_dag is (N_p, N_s) -> result (N_s, N_s)
        return jnp.matmul(O_c, O_dag) / n_samples

    # -------------------------------------------------------
    # Preparation Routine
    # -------------------------------------------------------

    @jax.jit
    def solve_jax_prepare(loss: jnp.ndarray, var_deriv: jnp.ndarray):
        """
        Preprocessing step. 
        Returns (loss_centered, O_centered, O_dag_centered, n_samples)
        """
        n_samples   = loss.shape[0]
        
        # Center Loss (Energy)
        loss_m      = jnp.mean(loss, axis=0)
        loss_c      = loss - loss_m

        # Center Derivatives (O)
        var_deriv_m = jnp.mean(var_deriv, axis=0)
        O_c         = var_deriv - var_deriv_m       # Shape (N_s, N_p)
        O_dag       = jnp.conj(O_c.T)               # Shape (N_p, N_s)
        
        return loss_c, O_c, O_dag, n_samples, var_deriv.shape[1]

    # -------------------------------------------------------
    # Unified Solver Logic
    # -------------------------------------------------------

    @partial(jax.jit, static_argnames=('solve_func', 'mode', 'maxiter', 'tol'))
    def solve_jax(
        solve_func      : Callable,
        loss            : jnp.ndarray,
        var_deriv       : jnp.ndarray,
        mode            : SRMode,
        x0              : Optional[jnp.ndarray] = None,
        precond_apply   : Optional[Callable] = None,
        maxiter         : int = 500,
        tol             : float = 1e-8
    ):
        """
        Main JAX entry point for SR/MinSR optimization.
        
        Parameters
        ----------
        solve_func : Callable
            The linear solver function (e.g. from QES.algebra.solvers).
            Signature: (A, b, x0, ...) -> solution
        loss : Array
            Raw local energies/loss (N_samples,)
        var_deriv : Array
            Raw logarithmic derivatives (N_samples, N_params)
        mode : SRMode
            STANDARD or MINSR.
        """
        # 1. Prepare Data
        E_c, O_c, O_dag, N, _ = solve_jax_prepare(loss, var_deriv)
        
        # 2. Branch based on Mode
        if mode == SRMode.MINSR:
            # MinSR Path (N_s x N_s)
            # Solve: T * x = E_c
            # Note: We normalize both sides by 1/N for numerical stability
            
            # Construct T = (1/N) O O^dag
            # NOTE: O_c is (N_s, N_p), O_dag is (N_p, N_s). Result is (N_s, N_s).
            # Correct order for MinSR: O @ O^dag
            T       = compute_ntk_matrix(O_c, O_dag, N)
            
            # RHS must also be normalized by 1/N to match T
            rhs     = E_c / N
            
            # Solve T * x = rhs
            # x0 for MinSR is size N_samples. If x0 provided is N_params, ignore or project it.
            # (Usually x0 is None for dense solves)
            x_minsr = solve_func(s=T, s_p=None, b=rhs, # Pass T as the matrix 's'
                x0=None, precond_apply=precond_apply, maxiter=maxiter, tol=tol)
            
            # Transform back to parameter space: update = O^dag @ x
            # update shape: (N_p, N_s) @ (N_s, 1) -> (N_p, 1)
            return jnp.matmul(O_dag, x_minsr)

        else:
            # Standard SR Path (N_p x N_p)
            # Solve: S * update = F
            
            # 1. Compute Forces F = (1/N) O^dag E_c
            F = compute_forces(O_dag, E_c, N)
            
            # 2. Solver interaction
            # If solver supports 'gram' (lazy evaluation), we pass O and O_dag.
            # If solver needs 'matrix', we assume it calculates O_dag @ O itself or we do it.
            # Here, we stick to the interface expected by `solvers.py`:
            # It likely expects (s, s_p) to form s @ s_p.
            
            # For Standard SR, S = O^dag @ O. So s=O_dag, s_p=O_c.
            return solve_func(s=O_dag, s_p=O_c, b=F,  x0=x0, precond_apply=precond_apply, maxiter=maxiter, tol=tol)

#####################################
# NumPy Implementation (Legacy/CPU)
#####################################

if True:
    # Numba-optimized helper for numpy backend
    # (Keeping it simple here, assuming parallel logic to JAX)
    
    def solve_numpy(solver, loss, var_deriv, min_sr: bool, **kwargs):
        # 1. Prepare
        loss_mean   = np.mean(loss, axis=0)
        E_c         = loss - loss_mean
        
        grad_mean   = np.mean(var_deriv, axis=0)
        O_c         = var_deriv - grad_mean
        O_dag       = O_c.conj().T
        N           = loss.shape[0]
        
        if min_sr:
            # MinSR: T = (1/N) O @ O^dag
            T       = np.matmul(O_c, O_dag) / N
            rhs     = E_c / N
            
            # Use solver directly on the dense matrix T
            if hasattr(solver, 'init_from_matrix'):
                solver.init_from_matrix(T, rhs)
                
            x_minsr = solver.solve(rhs, **kwargs)
            
            # Map back: update = O^dag @ x
            return np.matmul(O_dag, x_minsr)
            
        else:
            # Standard SR: S = (1/N) O^dag @ O
            F = np.matmul(O_dag, E_c) / N
            
            # Initialize solver with Gram vectors if supported
            if hasattr(solver, 'init_from_fisher'):
                # Pass O^dag and O. S = O^dag @ O
                solver.init_from_fisher(O_dag, O_c, F, **kwargs)
                
            return solver.solve(F, **kwargs)
        
#####################################

class StochasticReconfiguration(ABC):
    '''
    High-level handler for SR updates.
    '''
    
    def __init__(self, solver, backend: str = 'default'):
        self._backend_type  = get_backend(backend)
        self._isjax         = (self._backend_type != np)
        self._solver        = solver
        self._solution      = None

    def solve(self, loss, var_deriv, use_minsr: bool = False, **kwargs):
        """
        Solves the SR equation.
        
        Parameters
        ----------
        loss : array (N_samples,)
            Local energies.
        var_deriv : array (N_samples, N_params)
            Log-derivatives of the wavefunction.
        use_minsr : bool
            If True, uses the efficient N_s x N_s Neural Tangent Kernel formulation.
            Recommended when N_params >> N_samples (Deep Networks).
        """
        
        mode = SRMode.MINSR if use_minsr else SRMode.STANDARD
        
        if self._isjax:
            # Extract the pure function handle from the solver object
            # The solver object in QES usually has a 'solve' or 'get_solver_func' method
            # We assume 'self._solver' is already the callable function or has a wrapper.
            
            # If self._solver is a class instance, get its callable
            solve_fn = self._solver.solve if hasattr(self._solver, 'solve') else self._solver
            
            self._solution = solve_jax(
                                solve_func  =   solve_fn,
                                loss        =   loss,
                                var_deriv   =   var_deriv,
                                mode        =   mode,
                                **kwargs
                            )
        else:
            self._solution = solve_numpy(
                                solver      =   self._solver,
                                loss        =   loss,
                                var_deriv   =   var_deriv,
                                min_sr      =   use_minsr,
                                **kwargs
                            )
            
        return self._solution

######################################