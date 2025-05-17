'''
file    : general_python/algebra/solvers/stochastic_rcnfg.py
author  : Maksymilian Kliczkowski
date    : 2025-04-01
version : 1.0.0

Standard Stochastic Reconfiguration (SR) and Minimum-Step SR (MinSR)

=====================================================================================================

# Overview

--------
In variational Monte Carlo (VMC), the goal is to optimize the variational 
parameters θ such that the variational wave function |Ψ(θ)⟩
approaches the ground state of a given Hamiltonian.
Both standard stochastic reconfiguration (SR)
[and its efficient variant—minimum-step SR (MinSR) - see below] — aim to update 
the parameters by approximately following an imaginary-time evolution:

    |Ψ'⟩ = exp(-H · δτ) |Ψ(θ)⟩                              (Eq. 1)

The update is performed by minimizing the Fubini-Study (FS)
distance between the evolved state |Ψ'⟩ and the 
variational state |Ψ(θ+δθ)⟩.

## Fubini-Study Distance
----------------------
For small changes δθ and a small time step δτ, the FS distance is expanded as:

    d²(Ψ(θ+δθ), Ψ′) = ∑₍σ₎ | ∑₍k₎ O₍σ,k₎ · δθₖ - ε₍σ₎ |²    (Eq. 2)

with:
    
    O₍σ,k₎ = (1/ψ₍σ₎) ∂ψ₍σ₎/∂θₖ – ⟨(1/ψ₍σ₎) ∂ψ₍σ₎/∂θₖ⟩,
    
computed over Ns Monte Carlo samples:
    
    ε₍σ₎ = –δτ · (E^loc₍σ₎ – ⟨E^loc⟩)/√(Ns),
    
where the local energy is given by:

    E^loc₍σ₎ = ∑₍σ'₎ (ψ₍σ'₎/ψ₍σ₎) · H₍σ,σ'₎.

The minimization of this distance is equivalent to solving the linear equation:

    O · δθ = ε                                              (Eq. 3)

## Standard Stochastic Reconfiguration (SR)
------------------------------------------
In conventional SR, one defines the **quantum metric** (or Fisher information matrix) as:

    S = O† · O                                              (Eq. 4)

This metric measures the change in the quantum state induced by a parameter update,
and the FS distance can be 
written as:

    d²(Ψ(θ), Ψ(θ+δθ)) = δθ† · S · δθ                        (Eq. 5)

The SR method then updates the variational parameters using the solution of the linear equation (Eq. 3). The 
standard solution is obtained as:

    δθ = S⁻¹ · O† · ε                                       (Eq. 6)

This approach requires computing and inverting the matrix S,
which is of size NₚxNₚ (Nₚ is the number of 
variational parameters). When Nₚ is large,
the inversion becomes computationally expensive with a typical 
cost scaling of O(Nₚ³), or O(Nₚ²·Nₛ + Nₚ³) when iterative solvers are employed.
Moreover, for deep networks 
with Nₚ ≫ Nₛ (the number of Monte Carlo samples),
the matrix S is rank-deficient (its rank is at most Nₛ), 
posing additional numerical challenges.

## Minimum-Step Stochastic Reconfiguration (MinSR)
------------------------------------------------
To overcome the computational bottleneck in standard SR, MinSR reformulates the optimization by introducing the 
**neural tangent kernel**:

    T = O · O†

T is an NₛxNₛ matrix and shares the same nonzero eigenvalues as S.
By imposing a minimum-norm (or minimum-step) 
condition—i.e. selecting, among all solutions of Eq. (3),
the one with the smallest ||δθ||—the MinSR update is 
given by:

    δθ = O† · T⁻¹ · ε                                  (Eq. 5)

This formulation avoids the costly inversion of the full S matrix.
The inversion is now only on the smaller T matrix, 
reducing the computational complexity to approximately O(Nₚ·Nₛ² + Nₛ³).
Two derivations support this result:

1. **Lagrangian Multiplier Approach:**  
    The method minimizes ||δθ|| subject to O · δθ = ε by forming the Lagrangian:

    L({δθₖ}, {α₍σ₎}) = ∑₍k₎ |δθₖ|² – [∑₍σ₎ α₍σ₎* (∑₍k₎ O₍σ,k₎ · δθₖ – ε₍σ₎) + c.c.]

Solving the resulting equations leads to δθ = O† · (O · O†)⁻¹ · ε, which is equivalent to Eq. (5).

2. **Pseudo-Inverse Method:**  
By showing that the least-squares minimum-norm solution of O · δθ = ε is given by:

    δθ = O† · (O · O†)⁻¹ · ε,

and using properties of the pseudo-inverse,
one establishes the equivalence (O · O†)⁻¹ = T⁻¹, thereby recovering 
Eq. (5).

Regularization is typically applied to T⁻¹
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
import numba 
from typing import Union, Tuple, Callable, Optional, NamedTuple
from functools import partial

from abc import ABC, abstractmethod

from general_python.algebra.utils import JAX_AVAILABLE, get_backend, Array
import general_python.algebra.solver as solver_utils

#####################################

# jax specific
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    
    @jax.jit
    def loss_centered_jax(loss: jnp.ndarray, loss_m: jnp.ndarray) -> jnp.ndarray:
        '''
        Calculates the centered loss:
        
        centered loss L - <L>_{samples}
        where L is the loss function and <L>_{samples} is the mean of the loss
        function over the samples.
        
        Parameters:
            loss:    
                loss function L
            loss_m:
                mean of the loss function <L>_{samples}
        
        Returns:
            centered loss L - <L>_{samples}
        '''
        return loss - loss_m
    
    @jax.jit
    def loss_centered_jax_modified_ratios(  loss        : jnp.ndarray,
                                            loss_m      : jnp.ndarray,
                                            betas       : jnp.ndarray,
                                            ratios_exc  : jnp.ndarray,
                                            ratios_low  : jnp.ndarray
                                            ) -> jnp.ndarray:
        # loss_c  = loss - loss_m
        # samples = loss.shape[0]
        
        # for i, beta in enumerate(betas):
        #     r_excited   =   ratios_exc[i]
        #     r_low       =   ratios_low[i]
        #     m_lower     =   jnp.mean(r_low, axis=0)
        #     m_excited   =   jnp.mean(r_excited, axis=0)
        #     loss_c     +=   (r_excited - m_excited) * ((beta / samples) * m_lower)
        # return loss_c
        
        # center the loss
        loss_c      = loss - loss_m
        n           = loss.shape[0]

        # per-beta means over the sample axis
        m_low       = jnp.mean(ratios_low, axis=1)  # shape [n_betas]
        m_exc       = jnp.mean(ratios_exc, axis=1)  # shape [n_betas]

        # deviations of excited ratios - centering
        delta_exc   = ratios_exc - m_exc[:, None]   # shape [n_betas, n]

        # scale factor for each beta
        scales      = betas * (m_low / n)           # shape [n_betas]

        # contract over the beta‐dimension
        # result has shape [n], same as loss
        corr        = jnp.einsum('i,ij->j', scales, delta_exc)
        return loss_c + corr

    @jax.jit
    def derivatives_centered_jax(derivatives: jnp.ndarray, derivatives_m: jnp.ndarray) -> jnp.ndarray:
        '''
        Calculates the centered derivatives:
        O_k - <O_k> = O_k - <O_k>_{samples}
        where O_k is the variational derivative and <O_k>_{samples} is the mean
        of the variational derivative over the samples.
        
        --- 
        The centered derivatives are used to calculate the covariance matrix
        S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
        where <O_k^*> is the mean of the variational derivative over the samples.
        
        Parameters:
            derivatives:
                variational derivatives O_k
            derivatives_m:
                mean of the variational derivative <O_k>_{samples}    
        Returns:
            centered derivatives O_k - <O_k>_{samples}
        '''
        return derivatives - derivatives_m
    
    @jax.jit
    def covariance_jax_minsr(derivatives_c: jnp.ndarray, derivatives_c_h: jnp.ndarray) -> jnp.ndarray:
        '''
        Calculates the covariance matrix for stochastic reconfiguration from
        the variational derivatives.
        
        Parameters:
            derivatives_c:
                centered variational derivatives O_k - <O_k>_{samples}
            derivatives_c_h:
                centered variational derivatives hermitian conjugate
                O_k^* - <O_k^*>_{samples}
        Returns:
            covariance matrix S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
        '''
        return jnp.matmul(derivatives_c, derivatives_c_h) / derivatives_c.shape[1]
    
    @jax.jit
    def covariance_jax(derivatives_c: jnp.ndarray, derivatives_c_h: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        '''
        Calculates the covariance matrix for stochastic reconfiguration from
        the variational derivatives.
        Parameters:
            derivatives_c:
                centered variational derivatives O_k - <O_k>_{samples}
            derivatives_c_h:
                centered variational derivatives hermitian conjugate
                O_k^* - <O_k^*>_{samples}
        Returns:
            covariance matrix S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
        '''
        return jnp.matmul(derivatives_c_h, derivatives_c) / n_samples
    
    @jax.jit
    def gradient_jax(derivatives_c_h: jnp.ndarray,  loss_c: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        '''
        Calculates the gradient of the loss function with respect to the
        variational parameters.
        Parameters:
            derivatives_c_h:
                centered variational derivatives hermitian conjugate
                O_k^* - <O_k^*>_{samples}
            loss_c:
                centered loss function L - <L>_{samples}
            num_samples:
                number of samples used to calculate the gradient
        Returns:
            gradient of the loss function with respect to the variational parameters
        '''
        return jnp.matmul(derivatives_c_h, loss_c) / n_samples

    @jax.jit
    def solve_jax_prepare(loss, var_deriv):
        """
        Prepares the loss and variational derivatives for the stochastic reconfiguration solver.
        Namely, it calculates the centered loss and centered variational derivatives.
        This is a helper function to be used before calling the solver.
        
        $$
        L_c = L - <L>_{samples}
        $$

        $$        
        O_k^* - <O_k^*>_{samples}
        $$
        
        Parameters:
            loss:
                Array of loss function values.
            var_deriv:
                Array of variational derivatives.
        
        Returns:
            A tuple (loss_c, var_deriv_c, var_deriv_c_h, n_samples, full_size) where:
                - loss_c is the centered loss,
                - var_deriv_c is the centered variational derivatives,
                - var_deriv_c_h is the Hermitian conjugate (transpose of the complex conjugate)
                of var_deriv_c,
                - n_samples is the number of samples (loss.shape[0]),
                - full_size is var_deriv.shape[1].
        """
        n_samples       = loss.shape[0]
        loss_m          = jnp.mean(loss, axis=0)
        # Use the jitted versions explicitly if needed for clarity
        loss_c          = loss_centered_jax(loss, loss_m)

        full_size       = var_deriv.shape[1]
        var_deriv_m     = jnp.mean(var_deriv, axis=0)
        var_deriv_c     = derivatives_centered_jax(var_deriv, var_deriv_m)
        var_deriv_c_h   = jnp.conj(var_deriv_c.T)
        return loss_c, var_deriv_c, var_deriv_c_h, n_samples, full_size

    @jax.jit
    def solve_jax_prepare_modified_ratios(loss, var_deriv, betas, ratios_exc, ratios_low):
        """
        Prepares centered loss and derivative tensors for stochastic optimization using JAX.
        Args:
            loss (jnp.ndarray):
                Array of loss values with shape (n_samples, ...).
            var_deriv (jnp.ndarray):
                Array of variational derivatives with shape (n_samples, full_size).
            betas (Any):
                Parameters used for centering the loss (passed to loss_centered_jax).
            ratios_exc (Any):
                Excitation ratios used in loss centering.
            ratios_low (Any):
                Lower ratios used in loss centering.
        Returns:
            tuple:
                loss_c (jnp.ndarray): Centered loss tensor.
                var_deriv_c (jnp.ndarray): Centered variational derivatives.
                var_deriv_c_h (jnp.ndarray): Hermitian conjugate (complex conjugate transpose) of centered derivatives.
                n_samples (int): Number of samples in the batch.
                full_size (int): Size of the variational parameter space.
        """
    
        n_samples       = loss.shape[0]
        loss_m          = jnp.mean(loss, axis=0)
        loss_c          = loss_centered_jax(loss, loss_m, betas, ratios_exc, ratios_low)

        full_size       = var_deriv.shape[1]
        var_deriv_m     = jnp.mean(var_deriv, axis=0)
        var_deriv_c     = derivatives_centered_jax(var_deriv, var_deriv_m)
        var_deriv_c_h   = jnp.conj(var_deriv_c.T)
        return loss_c, var_deriv_c, var_deriv_c_h, n_samples, full_size

    @partial(jax.jit, static_argnames=('solve_func', 'min_sr', 'precond_apply', 'maxiter', 'tol'))
    def solve_jax_cov_in(solve_func,
                        loss_c          :   jnp.ndarray,
                        var_deriv_c     :   jnp.ndarray,
                        var_deriv_c_h   :   jnp.ndarray,
                        n_samples       :   int,
                        min_sr          :   bool,
                        x0              =   None,
                        precond_apply         =   None,
                        s               :   Optional[jnp.ndarray] = None,
                        maxiter         :   int   = 500,
                        tol             :   float = 1e-8):
        """
        Solves the covariance problem using the specified solver with precomputed parameters.
        
        Parameters:
            solver:
                Solver object (with methods init_from_matrix and solve).
            loss_c:
                Centered loss function values.
            var_deriv_c:
                Centered variational derivatives.
            var_deriv_c_h:
                Hermitian conjugate of the centered variational derivatives.
            n_samples:
                Number of samples used in the covariance calculation.
            min_sr:
                Boolean flag indicating whether to use the MinSR method.
            x0:
                Optional initial guess for the solution.
            precond_apply:
                Optional precond_applyitioner for the solver.
            s:
                Optional covariance matrix; if None, it will be computed.
        
        Returns:
            A tuple (solution, s) where:
            - For the MinSR branch, solution = var_deriv_c_h @ (solver.solve(...))
            - Otherwise, solution is the direct output from the solver.
        """
        
        # In the MinSR case, compute S using the MinSR covariance function if not provided.
        if min_sr:
            if s is None:
                s = covariance_jax_minsr(var_deriv_c, var_deriv_c_h, n_samples)
            
            # Solve the linear equation using the solver.
            solution = solve_func(s, loss_c, x0=x0, precond_apply=precond_apply, maxiter=maxiter, tol=tol)
            return jnp.matmul(var_deriv_c_h, solution), s
        
        # Compute forces using a gradient function; assume gradient_jax is defined.
        f = gradient_jax(var_deriv_c_h, loss_c)
        
        if s is None:
            # Compute the covariance matrix if not provided.
            s = covariance_jax(var_deriv_c, var_deriv_c_h, n_samples)
            
        # Solve the linear equation using the solver.
        solution = solve_func(s, f, x0=x0, precond_apply=precond_apply, maxiter=maxiter, tol=tol)
        return solution, s

    @partial(jax.jit, static_argnames=('solve_func', 'min_sr', 'precond_apply', 'maxiter', 'tol'))
    def solve_jax_cov(solve_func,
                    loss,
                    var_deriv,
                    min_sr,
                    x0          = None,
                    precond_apply     = None,
                    s           : Optional[jnp.ndarray] = None,
                    maxiter     : int = 500,
                    tol         : float = 1e-8):
        """
        Solves the covariance problem from scratch using the specified solver.
        
        This function first prepares the loss and variational derivatives, then solves the covariance problem.
        
        Parameters:
            solver:
                Solver object (with methods for initialization and solving).
            loss:
                Array of loss function values.
            var_deriv:
                Array of variational derivatives.
            min_sr:
                Boolean flag indicating whether to use the MinSR method.
            x0:
                Optional initial guess for the solution.
            precond_apply:
                Optional precond_applyitioner for the solver.
            s:
                Optional covariance matrix.
            maxiter:
                Maximum number of iterations for the solver.
            tol:
                Tolerance for convergence of the solver.
        Returns:
            A tuple (solution, s) with the computed solution and the covariance matrix.
        """
        loss_c, var_deriv_c, var_deriv_c_h, n_samples, _ = solve_jax_prepare(loss, var_deriv)
        # Pass solve_func correctly
        return solve_jax_cov_in(solve_func, loss_c, var_deriv_c, var_deriv_c_h, n_samples,
                                min_sr, x0=x0, precond_apply=precond_apply, s=s, maxiter=maxiter, tol=tol)

    @partial(jax.jit, static_argnames=('solve_func', 'min_sr', 'precond_apply', 'maxiter', 'tol'))
    def solve_jax_in(solve_func,
                    loss_c,
                    var_deriv_c,
                    var_deriv_c_h,
                    min_sr,
                    n_samples,
                    x0=None, precond_apply=None, maxiter=500, tol=1e-8):
        """
        Solves for the parameter update using the Fisher formulation without explicitly creating a covariance matrix.
        
        Depending on the min_sr flag:
        - If True, the solver is initialized with (var_deriv_c_h, var_deriv_c, loss_c) and the update is computed
            as var_deriv_c_h @ solution.
        - Otherwise, forces are computed via a gradient approximation (using gradient_jax) and the solver is initialized
            with (var_deriv_c, var_deriv_c_h, f).
        
        Parameters:
            solver:
                Solver object (with methods init_from_fisher and solve).
            loss_c:
                Centered loss function values.
            var_deriv_c:
                Centered variational derivatives.
            var_deriv_c_h:
                Hermitian conjugate of the centered variational derivatives.
            n_samples:
                Number of samples.
            min_sr:
                Boolean flag indicating whether to use the MinSR approach.
            x0:
                Optional initial guess for the solution.
            precond_apply:
                Optional precond_applyitioner.
        
        Returns:
            The computed update vector.
        """
        if min_sr:
            solution = solve_func(s=var_deriv_c_h, s_p=var_deriv_c, b=loss_c, x0=x0, precond_apply=precond_apply, maxiter=maxiter, tol=tol)
            return jnp.matmul(var_deriv_c_h, solution)

        f = gradient_jax(var_deriv_c_h, loss_c, n_samples)
        solution = solve_func(s=var_deriv_c, s_p=var_deriv_c_h, b=f, x0=x0, precond_apply=precond_apply, maxiter=maxiter, tol=tol)
        return solution

    @partial(jax.jit, static_argnames=('solve_func', 'min_sr', 'precond_apply', 'maxiter', 'tol'))
    def solve_jax(solve_func, loss, var_deriv, min_sr, x0=None, precond_apply=None, maxiter=500, tol=1e-8):
        """
        Solves the stochastic reconfiguration problem using the specified solver without creating the
        covariance matrix explicitly.
        
        Parameters:
            solver:
                Solver object (with methods for initialization from Fisher information and solving).
            loss:
                Array of loss function values.
            var_deriv:
                Array of variational derivatives.
            min_sr:
                Boolean flag indicating whether to use the MinSR method.
            x0:
                Optional initial guess for the solution.
            precond_apply:
                Optional precond_applyitioner.
        
        Returns:
            The computed update vector.
        """
        loss_c, var_deriv_c, var_deriv_c_h, n_samples, _ = solve_jax_prepare(loss, var_deriv)
        return solve_jax_in(solve_func, loss_c, var_deriv_c, var_deriv_c_h, min_sr, n_samples,
                            x0=x0, precond_apply=precond_apply, maxiter=maxiter, tol=tol)

# numpy specific
if True:
    
    @numba.njit
    def loss_centered(loss, loss_m):
        '''
        Calculates the centered loss:
        
        centered loss L - <L>_{samples}
        where L is the loss function and <L>_{samples} is the mean of the loss
        function over the samples.
        
        Parameters:
            loss:    
                loss function L
            loss_m:
                mean of the loss function <L>_{samples}
        
        Returns:
            centered loss L - <L>_{samples}
        '''
        return loss - loss_m
    
    @numba.njit
    def derivatives_centered(derivatives, derivatives_m):
        '''
        Calculates the centered derivatives:
        O_k - <O_k> = O_k - <O_k>_{samples}
        where O_k is the variational derivative and <O_k>_{samples} is the mean
        of the variational derivative over the samples.
        
        --- 
        The centered derivatives are used to calculate the covariance matrix
        S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
        where <O_k^*> is the mean of the variational derivative over the samples.
        
        Parameters:
            derivatives:
                variational derivatives O_k
            derivatives_m:
                mean of the variational derivative <O_k>_{samples}    
        Returns:
            centered derivatives O_k - <O_k>_{samples}
        '''
        return derivatives - derivatives_m
    
    @numba.njit
    def covariance_np_minsr(derivatives_c, derivatives_c_h, num_samples):
        '''
        Calculates the covariance matrix for stochastic reconfiguration from
        the variational derivatives.
        Parameters:
            derivatives_c:
                centered variational derivatives O_k - <O_k>_{samples}
            derivatives_c_h:
                centered variational derivatives hermitian conjugate
                O_k^* - <O_k^*>_{samples}
            num_samples:
                number of samples used to calculate the covariance matrix
        Returns:
            
        Note:
            The MinSR method is used to 
            
        '''
        return np.matmul(derivatives_c, derivatives_c_h) / num_samples
    
    @numba.njit
    def covariance_np(derivatives_c, derivatives_c_h, num_samples):
        '''
        Calculates the covariance matrix for stochastic reconfiguration from
        the variational derivatives.
        
        Parameters:
            derivatives_c:
                centered variational derivatives O_k - <O_k>_{samples}
            derivatives_c_h:
                centered variational derivatives hermitian conjugate
                O_k^* - <O_k^*>_{samples}
            num_samples:
                number of samples used to calculate the covariance matrix
        Returns:
            covariance matrix S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
        '''
        return np.matmul(derivatives_c_h, derivatives_c) / num_samples
    
    @numba.njit
    def gradient_np(derivatives_c_h, loss_c, num_samples):
        '''
        Calculates the gradient of the loss function with respect to the
        variational parameters.
        Parameters:
            derivatives_c_h:
                centered variational derivatives hermitian conjugate
                O_k^* - <O_k^*>_{samples}
            loss_c:
                centered loss function L - <L>_{samples}
            num_samples:
                number of samples used to calculate the gradient
        Returns:
            gradient of the loss function with respect to the variational parameters
        '''
        return np.matmul(derivatives_c_h, loss_c) / num_samples

    @numba.njit
    def solve_numpy_prepare(loss, var_deriv):
        '''
        Prepares the loss and variational derivatives for the stochastic reconfiguration solver.
        
        Parameters:
            loss:
                Loss function values.
            var_deriv:
                Variational derivatives.
        
        Returns:
            Tuple containing the centered loss and centered variational derivatives.
        '''
        # calculate the centered loss
        n_samples       = loss.shape[0]
        loss_m          = np.mean(loss, axis=0)
        loss_c          = loss_centered(loss, loss_m)
        
        # calculate the centered derivatives
        full_size       = var_deriv.shape[1]
        var_deriv_m     = np.mean(var_deriv, axis=0)
        var_deriv_c     = derivatives_centered(var_deriv, var_deriv_m)
        var_deriv_c_h   = np.conj(var_deriv_c).T
        
        return loss_c, var_deriv_c, var_deriv_c_h, n_samples, full_size
    
    @numba.njit
    def solve_numpy_cov_in(
        solver          : solver_utils.Solver,
        loss_c          : np.ndarray,
        var_deriv_c     : np.ndarray,
        var_deriv_c_h   : np.ndarray,
        n_samples       : int,
        min_sr          : bool,
        x0              : Optional[np.ndarray] = None,
        precond_apply         = None,
        s               : Optional[np.ndarray] = None):
        '''
        Solves the covariance problem using the specified solver.
        Uses precomputed parameters.
        Parameters:
            solver:
                Solver object for solving the covariance problem.
            loss_c:
                Centered loss function values.
            var_deriv_c:
                Centered variational derivatives.
            var_deriv_c_h:
                Conjugate transpose of the centered variational derivatives.
            n_samples:
                Number of samples used in the covariance calculation.
            min_sr:
                Boolean flag indicating whether to use MinSR.
            x0:
                Initial guess for the solution (optional).
            precond_apply:
                precond_applyitioner for the solver (optional).
            s:
                Covariance matrix (optional).
        Returns:
            Solution to the covariance problem.
        '''
        
        # calculate the covariance matrix
        if min_sr:
            if s is None:
                s = covariance_np_minsr(var_deriv_c, var_deriv_c_h, n_samples)
                solver.init_from_matrix(s, loss_c, x0 = x0)
                
            solution = solver.solve(loss_c, x0, precond_apply = precond_apply)
            return np.matmul(var_deriv_c_h, solution), s
        
        # calculate forces
        f           = gradient_np(var_deriv_c_h, loss_c, n_samples)
        if s is None:
            s = covariance_np(var_deriv_c, var_deriv_c_h, n_samples)
            solver.init_from_matrix(s, f, x0 = x0)
        solution = solver.solve(f, x0 = x0, precond_apply = precond_apply)
        return solution, s
    
    @numba.njit
    def solve_numpy_cov(
        solver      : solver_utils.Solver,
        loss        : np.ndarray,
        var_deriv   : np.ndarray,
        min_sr      : bool,
        x0          : Optional[np.ndarray] = None,
        precond_apply     : Optional[np.ndarray] = None,
        s           : Optional[np.ndarray] = None):
        '''
        Solves the covariance problem using the specified solver.
        Does it from scratch.
        
        Parameters:
            solver:
                Solver object for solving the covariance problem.
            loss:
                Loss function values.
            var_deriv:
                Variational derivatives.
            min_sr:
                Boolean flag indicating whether to use MinSR.
            x0:
                Initial guess for the solution (optional).
            precond_apply:
                precond_applyitioner for the solver (optional).
        Returns:
            Solution to the covariance problem.
            
        Notes:
            - This function uses Numba for JIT compilation to improve performance.
            - The function calculates the centered loss and centered derivatives,
                and then computes the covariance matrix.
            - The covariance matrix is used to solve the linear system.
            - The function handles both the standard covariance and MinSR cases.
        '''
        #! TODO: solver replacement
        loss_c, var_deriv_c, var_deriv_c_h, n_samples, _ = solve_numpy_prepare(loss, var_deriv)
        
        return solve_numpy_cov_in(solver,
                                loss_c,
                                var_deriv_c,
                                var_deriv_c_h,
                                n_samples,
                                min_sr,
                                x0      = x0,
                                precond_apply = precond_apply,
                                s       = s)
    
    @numba.njit
    def solve_numpy_in(solver       : solver_utils.Solver,
                        loss_c          : np.ndarray,
                        var_deriv_c     : np.ndarray,
                        var_deriv_c_h   : np.ndarray,
                        n_samples       : int,
                        min_sr          : bool,
                        x0              : Optional[np.ndarray] = None,
                        precond_apply         = None,
                        maxiter         : int   = 500,
                        tol             : float = 1e-8):
        """
        Solves for the parameter update by applying a linear solver configured using input covariance
        and derivative matrices, optionally following a minimal sampling residual (min_sr) procedure.
        This function initializes the solver based on the provided Fisher information encoded in the
        derivative matrices and either directly solves the system involving the loss coefficients (when
        min_sr is True) or computes the forces via a gradient approximation and solves for the update.
        
        ---
        Parameters:
            solver (solver_utils.Solver):
                An instance of a solver class that provides methods for
                initializing from Fisher information and solving linear systems.
            loss_c (np.ndarray):
                The loss coefficients or residual vector used in the solver.
            var_deriv_c (np.ndarray):
                The covariance or derivative matrix used as part of the Fisher
                information to initialize the solver.
            var_deriv_c_h (np.ndarray):
                The Hermitian (or corresponding transpose) form of the derivative
                matrix used in the initialization and subsequent multiplication of the solution.
            n_samples (int):
                The number of samples used to compute the gradient approximation for the forces.
            min_sr (bool):
                A flag indicating whether to use the minimal sampling residual approach.
                If True, the solver is initialized with (var_deriv_c_h, var_deriv_c, loss_c) and the solution
                is post-multiplied by var_deriv_c_h. Otherwise, the forces are computed from a gradient of loss.
            x0 (Optional[np.ndarray], optional):
                An optional initial guess for the solver. Defaults to None.
            precond_apply (optional):
                A precond_applyitioner to be used by the solver. The type and required format
                depend on the specific solver implementation. Defaults to None.
            maxiter (int, optional):
            
            tol (float, optional)
        
        Returns:
            np.ndarray:
                The computed update vector. If min_sr is True, the solution is transformed by
                multiplying with var_deriv_c_h; otherwise, it is returned directly from the solver.
        """
        if min_sr:
            solver.init_from_fisher(var_deriv_c_h, var_deriv_c, loss_c, x0 = x0)
            solution = solver.solve(loss_c, x0, precond_apply = precond_apply)
            return np.matmul(var_deriv_c_h, solution)
        
        # calculate forces
        f           = gradient_np(var_deriv_c_h, loss_c, n_samples)
        solver.init_from_fisher(var_deriv_c, var_deriv_c_h, f, x0 = x0)
        solution = solver.solve(f, x0 = x0, precond_apply = precond_apply, maxiter = maxiter, tol = tol)
        return solution
    
    @numba.njit
    def solve_numpy(solver      : solver_utils.Solver,
                    loss        : np.ndarray,
                    var_deriv   : np.ndarray,
                    min_sr      : bool,
                    x0          : Optional[np.ndarray] = None,
                    precond_apply     : Optional[np.ndarray] = None,
                    maxiter     : int   = 500,
                    tol         : float = 1e-8):        
        '''
        Solves the stochastic reconfiguration problem using the specified solver.
        Parameters:
            solver:
                Solver object for solving the stochastic reconfiguration problem.
            loss:
                Loss function values.
            var_deriv:
                Variational derivatives.
            min_sr:
                Boolean flag indicating whether to use MinSR.
            x0:
                Initial guess for the solution (optional).
            precond_apply:
                precond_applyitioner for the solver (optional).
        Returns:
            Solution to the stochastic reconfiguration problem.
        Notes:
            - This function uses Numba for JIT compilation to improve performance.
            - This function does not create the covariance matrix explicitly.
        '''
        loss_c, var_deriv_c, var_deriv_c_h, n_samples, _ = solve_numpy_prepare(loss, var_deriv)
        return solve_numpy_in(solver,
                            loss_c,
                            var_deriv_c,
                            var_deriv_c_h,
                            n_samples,
                            min_sr,
                            x0      = x0,
                            precond_apply = precond_apply,
                            maxiter = maxiter,
                            tol     = tol)

#####################################

class StochasticReconfiguration(ABC):
    '''
    This is a class that handles the stochastic reconfiguration process
    '''
    
    def __init__(self,
                solver      : solver_utils.Solver,
                backend     : str = 'default'):
        '''
        Initializes the StochasticReconfiguration class.
        The class is used to handle the stochastic reconfiguration process
        and the minimum-step stochastic reconfiguration process.
        
        Parameters:
            solver:
                solver to use for the linear system of equations:
                    At one point of stochastic reconfiguration, the solver will be used to
                    solve the system of equations S_{kk'} x_k = F_k
                    where S_{kk'} is the covariance matrix, x_k is the solution
                    and F_k is the variational gradient of the loss function.
            backend:
                backend to use for the stochastic reconfiguration
                'jax' or 'numpy'
                'default' will use the default backend for the system
        '''
        super().__init__()
        
        self._backend           = get_backend(backend)
        self._isjax             = self._backend != np
        self._backendstr        = "jax" if self._isjax else "numpy"
        
        # info size and methods
        self._full_size         = 1
        self._nsamples          = 1
        self._minsr             = False
        
        # arrays
        self._loss_m            = None      # mean loss
        self._loss              = None      # loss function (can be energies) 
        self._loss_c            = None      # centered loss function L_c = L - <L>_{samples} [(E_k - <E_k>)]
        self._derivatives_m     = None      # mean derivatives <O_k>                    - (full_size)
        self._derivatives       = None      # variational derivatives O_k               - (n_samples x full_size)
        self._derivatives_c     = None      # centered derivatives (O_k - <O_k>)        - (n_samples x full_size)
        self._derivatives_c_h   = None      # centered derivatives hermitian conjugate 
        self._s                 = None      # the covariance matrix S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
        self._f                 = None      # the variational gradient of loss F_k = Cov[O_k, L_k]
        self._solution          = None
        self._solver            = solver
        
        # functions
        if self._isjax:
            self._covariance_minres_fun = covariance_jax_minsr
            self._covariance_fun        = covariance_jax
            self._gradient_fun          = gradient_jax
            self._der_c_fun             = derivatives_centered_jax
            self._loss_c_fun            = loss_centered_jax
        else:
            self._covariance_minres_fun = covariance_np_minsr
            self._covariance_fun        = covariance_np
            self._gradient_fun          = gradient_np
            self._der_c_fun             = derivatives_centered
            self._loss_c_fun            = loss_centered
        
    ##################################
    #! CALCULATORS
    ##################################
    
    def _calculate_loss(self, mean_loss = None):
        """
        Calculate and update the loss metrics for the stochastic solver.

        This method computes two primary attributes:
            _loss_m:
                The mean of the loss values. If the optional parameter `mean_loss` is not provided,
                it is calculated using the backend's mean function over the specified axis.
            _loss_c:
                A custom computed loss metric generated by applying the `_loss_c_fun` function
                to the raw loss values and the computed or provided mean loss.

        Parameters:
            mean_loss (optional):
                A precomputed mean loss value. If provided, it overrides the automatically
                computed mean loss using the backend. Default is None.

        Side Effects:
            Updates the instance attributes `_loss_m` and `_loss_c` with the newly calculated values.
        """

        self._loss_m    = self._backend.mean(self._loss, axis = 0) if mean_loss is None else mean_loss
        self._loss_c    = self._loss_c_fun(self._loss, self._loss_m)
        
    def _calculate_derivatives(self, mean_deriv = None):
        """
        Calculate and update the derivative attributes for the instance.

        This method computes several derivative-related quantities:
        - self._derivatives_m: The mean of self._derivatives along axis 0 is calculated using the backend's mean function.
            If a pre-computed mean (mean_deriv) is provided as an argument, that value is used instead.
        - self._derivatives_c: The centered derivatives are computed by the function self._der_c_fun, which processes the
            original derivatives (self._derivatives) using self._derivatives_m.
        - self._derivatives_c_h: The conjugate transpose of self._derivatives_c is computed using the backend's conj function.

        Parameters:
            mean_deriv (optional):
                Pre-computed mean derivatives. If provided, it overrides the computed mean from
                self._derivatives.

        Side Effects:
            Updates the following instance attributes:
                - self._derivatives_m
                - self._derivatives_c
                - self._derivatives_c_h
        """

        self._derivatives_m     = self._backend.mean(self._derivatives, axis = 0) if mean_deriv is None else mean_deriv
        self._derivatives_c     = self._der_c_fun(self._derivatives, self._derivatives_m)
        self._derivatives_c_h   = self._backend.conj(self._derivatives_c).T
    
    def _calculate_s(self):
        """
        Calculate the covariance matrix or its minimum residual variant.
        This method computes the covariance based on the internal derivatives and the number
        of samples. If the '_minsr' flag is set to True, it calls the covariance_minres_fun; 
        otherwise, it returns the result from covariance_fun.
        
        Returns:
            The covariance result computed using the corresponding internal function.
        """
        if self._minsr:
            return self._covariance_minres_fun(self._derivatives_c, self._derivatives_c_h, self._nsamples)
        return self._covariance_fun(self._derivatives_c, self._derivatives_c_h, self._nsamples)
    
    ##################################
    #! SETTERS
    ##################################
    
    def set_values(self,
                loss,
                derivatives,
                mean_loss    = None,
                mean_deriv   = None,
                calculate_s  = False,
                use_minsr    : Optional[bool] = False):
        '''
        Sets the values for the Stochastic Reconfiguration (Natural Gradient)
        
        Parameters:
            loss:
                loss function L
            derivatives:
                variational derivatives O_k
            mean_loss:
                mean of the loss function <L>_{samples}
            mean_deriv:
                mean of the variational derivative <O_k>_{samples}
            calculate_s:
                whether to calculate the covariance matrix S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
            use_minsr:
                whether to use the minres solver for the covariance matrix.
        '''
        self._nsamples      = loss.shape[0]
        
        # get the loss
        self._loss          = loss
        self._calculate_loss(mean_loss)
        
        # handle derivatives
        self._derivatives   = derivatives
        self._full_size     = derivatives.shape[1]
        self._calculate_derivatives(mean_deriv)
        
        # use min_sr?
        if use_minsr is not None:
            self._minsr     = use_minsr
            
        # calculate covariance
        if calculate_s:
            self._s = self._calculate_s()
        else:
            self._s = None
            
        # calculate F
        self._f     = self._gradient_fun(self._derivatives_c_h, self._loss_c, self._nsamples)
    
    def set_solver(self, solver):
        '''
        Sets the solver for the Stochastic Reconfiguration (Natural Gradient)
        The solver shall be able to solve the system of equations
        
        $$
        S_{kk'} x_k = F_k
        $$
        
        where S_{kk'} is the covariance matrix, x_k is the solution and F_k is the
        variational gradient of the loss function.
        
        Parameters:
            solver:
                solver to use for the stochastic reconfiguration
        
        ---
        Notes:
            The solver must be able to handle the case where S_{kk'} is not
            a square matrix. This is the case when the number of samples is
            less than the number of variational parameters.
        
        '''
        self._solver = solver
    
    ##################################
    #! SOLVER
    ##################################
    
    def solve(self, use_s = False, use_minsr = False):
        '''
        Solves the stochastic reconfiguration problem.
        Parameters:
            use_s:
                whether to use the covariance matrix S. This
                step involves the creation of the covariance matrix
                S = <O_k^*O_k'> - <O_k^*><O_k> / n_samples
                This is a slow step and should be avoided if possible.
            use_minsr:
                whether to use the minres solver for the covariance matrix.
        '''
        
        self._minsr = use_minsr
        
        if use_s:
            self._s = self._calculate_s()
        
        if self._isjax:
            if use_s:
                self._solution = solve_jax_cov_in(self._solver, self._loss_c, self._derivatives_c, self._derivatives_c_h, self._nsamples, self._minsr, self._s)
            else:
                self._solution = solve_jax_in(self._solver, self._loss_c, self._derivatives_c, self._derivatives_c_h, self._nsamples, self._minsr)
        else:
            if use_s:
                self._solution = solve_numpy_cov_in(self._solver, self._loss_c, self._derivatives_c, self._derivatives_c_h, self._nsamples, self._minsr, self._s)
            else:
                self._solution = solve_numpy_in(self._solver, self._loss_c, self._derivatives_c, self._derivatives_c_h, self._nsamples, self._minsr)
        return self._solution
    
    ##################################
    #! PROPERTIES
    ##################################

    @property
    def forces(self):
        '''
        Returns the forces (solution) of the stochastic reconfiguration
        '''
        return self._f
    
    @property
    def derivatives(self):
        '''
        Returns the logarithmic derivatives of the ansatz
        '''
        return self._derivatives
    
    @property
    def derivatives_c(self):
        '''
        Returns the centered derivatives of the ansatz
        '''
        return self._derivatives_c
    
    @property
    def derivatives_c_h(self):
        ''' 
        Returns the centered derivatives of the ansatz -> conjugate transpose
        '''
        return self._derivatives_c_h
    
    @property
    def solution(self):
        '''
        Returns the solution of the stochastic reconfiguration
        '''
        return self._solution
    
    @property
    def covariance_matrix(self):
        '''
        Returns the covariance matrix S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
        '''
        if self._s is None:
            self._s = self._calculate_s()
        return self._s
    
    ##################################

######################################