#!/usr/bin/env python3
"""
GRAM Form Solver - TDVP/NQS Example

This example demonstrates the GRAM form solver, which is the most common
pattern in Time-Dependent Variational Principle (TDVP) and Neural Quantum
States (NQS) applications.

Mathematical Problem:
    Given matrices S and vector Sp, solve:
        (S\dag S) p = S\dag Sp
    
    This is equivalent to the least-squares problem:
        min_p ||Sp - S p||^2 

Common Use Cases:
    - TDVP quantum dynamics             : S = overlap derivatives, Sp = energy derivatives
    - NQS optimization                  : S = log-derivative (Jacobian), Sp = energy gradient
    - Stochastic reconfiguration (SR)   : Natural gradient descent
    
File        : QES/general_python/algebra/examples/example_gram_form.py
Version     : 0.1.0
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
"""

import numpy as np
from QES.general_python.algebra import solvers

# ------------------------------------
#! GRAM Form Example
# ------------------------------------

def create_gram_problem(n_samples, n_params, rank_deficient=False):
    """
    Create a typical GRAM problem as seen in TDVP/NQS.
    
    Parameters:
    -----------
        n_samples:
            Number of samples (typically Monte Carlo samples)
        n_params:
            Number of variational parameters
        rank_deficient:
            Whether to create rank-deficient S
        
    Returns:
    ---------
        S: 
            Sample matrix (n_samples \times n_params)
        Sp: 
            Target vector (n_samples,)
    """
    # S represents log-derivative or overlap derivatives
    # In NQS: S[i,j] = ∂log ψ/∂θ_j evaluated at sample i
    S = np.random.randn(n_samples, n_params) + 1j * np.random.randn(n_samples, n_params)
    
    if rank_deficient:
        # Make some columns linearly dependent (common in practice)
        S[:, -1] = 2 * S[:, 0] + 0.5 * S[:, 1]
    
    # Sp represents energy derivatives or target values
    # In TDVP: Sp[i] = -∂E/∂θ_i
    Sp = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    
    return S, Sp

# ------------------------------------

def example_basic_gram():
    """Basic GRAM form solve for TDVP/NQS."""
    print("=" * 70)
    print("Example 1: Basic GRAM Form (TDVP/NQS Pattern)")
    print("=" * 70)
    
    # Typical NQS problem sizes
    n_samples = 1000   # Monte Carlo samples
    n_params = 50      # Variational parameters
    
    S, Sp = create_gram_problem(n_samples, n_params)
    
    print(f"Problem setup (NQS-like):")
    print(f"  Number of samples (MC): {n_samples}")
    print(f"  Number of parameters: {n_params}")
    print(f"  S shape: {S.shape}")
    print(f"  Sp shape: {Sp.shape}")
    
    # Solve using GRAM form (Fisher form)
    # This solves (S\dag S) p = S\dag Sp
    # Step 1: Choose solver
    solver      = solvers.choose_solver(solver_id='cg', sigma=0.0)
    
    # Step 2: Get solver function with Fisher/GRAM form
    solve_func  = solver.get_solver_func(
                    backend_module  =   np,
                    use_matvec      =   False,
                    use_fisher      =   True,   # GRAM/Fisher form
                    use_matrix      =   True,
                    sigma           =   0.0
                )
    
    # Step 3: Solve with s=S, s_p=S\dag, b=forces
    # The wrapper creates matvec from Fisher form: (S\dag S) x = b
    # Pass s=S (Jacobian), s_p=S\dag (Hermitian transpose), b=S\dag Sp (RHS)
    S_dag       = S.conj().T  # Hermitian transpose
    forces      = S_dag @ Sp  # RHS = S\dag Sp
    result      = solve_func(s=S, s_p=S_dag, b=forces, x0=None, tol=1e-6, maxiter=None, precond_apply=None)
    
    print(f"\nSolver results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual norm: {result.residual_norm:.2e}")
    
    # Verify solution by checking least-squares residual
    residual_ls = np.linalg.norm(S @ result.x - Sp)
    print(f"\nLeast-squares residual ||Sp - S p||: {residual_ls:.2e}")
    
    # Compare with direct solution
    p_direct = np.linalg.lstsq(S, Sp, rcond=None)[0]
    error = np.linalg.norm(result.x - p_direct)
    print(f"Error vs direct lstsq: {error:.2e}")
    
    return result


def example_tdvp_time_step():
    """Simulate a TDVP time step."""
    print("\n" + "=" * 70)
    print("Example 2: TDVP Time Step")
    print("=" * 70)
    
    # In TDVP, we solve for parameter updates:
    # S dt/dθ = -∂E/∂θ
    # where S_ij = <∂ψ/∂θ_i | ∂ψ/∂θ_j> (quantum geometric tensor)
    
    n_params = 30
    
    # Create overlap matrix S (typically from <O_i O_j*> - <O_i><O_j*>)
    # where O_i = ∂log ψ/∂θ_i
    n_samples = 500
    O = np.random.randn(n_samples, n_params) + 1j * np.random.randn(n_samples, n_params)
    
    # Quantum geometric tensor (covariance of log-derivatives)
    O_mean = np.mean(O, axis=0)
    O_centered = O - O_mean[np.newaxis, :]
    
    # S matrix for GRAM form
    S = O_centered
    
    # Energy gradient
    E_grad = np.random.randn(n_params) + 1j * np.random.randn(n_params)
    
    # Forces (from Monte Carlo)
    forces = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    forces_centered = forces - np.mean(forces)
    Sp = forces_centered
    
    print(f"TDVP setup:")
    print(f"  Variational parameters: {n_params}")
    print(f"  Monte Carlo samples: {n_samples}")
    
    # Solve for parameter updates using Fisher form
    solver      = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func  = solver.get_solver_func(
                    backend_module  =   np,
                    use_matvec      =   False,
                    use_fisher      =   True,
                    use_matrix      =   True,
                    sigma           =   0.0)
    S_dag   = S.conj().T
    forces  = S_dag @ Sp
    result  = solve_func(s=S, s_p=S_dag, b=forces, x0=None, tol=1e-8, maxiter=None, precond_apply=None)

    print(f"\nParameter update solution:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Update norm: {np.linalg.norm(result.x):.4f}")
    
    # In practice, you would update: θ_new = θ_old + dt * result.x
    dt = 0.01
    print(f"\nWith time step dt = {dt}:")
    print(f"  Parameter change: {dt * np.linalg.norm(result.x):.6f}")
    
    return result

# ----------------------------------------------------------------------------------------
#! Stochastic Reconfiguration
# ----------------------------------------------------------------------------------------

def example_stochastic_reconfiguration():
    """Stochastic Reconfiguration (SR) for NQS optimization."""
    
    print("\n" + "=" * 70)
    print("Example 3: Stochastic Reconfiguration (Natural Gradient)")
    print("=" * 70)
    
    # SR solves: S^-1 F where F is the force vector
    # This is the natural gradient direction
    
    n_params        = 40
    n_samples       = 800

    # Log-derivatives of wavefunction
    log_deriv       = np.random.randn(n_samples, n_params) + 1j * np.random.randn(n_samples, n_params)
    
    # Center the log-derivatives (important for numerical stability)
    log_deriv_mean  = np.mean(log_deriv, axis=0)
    S               = log_deriv - log_deriv_mean[np.newaxis, :]
    
    # Energy local times log-derivative (force)
    E_local         = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    E_mean          = np.mean(E_local)
    forces          = (E_local[:, np.newaxis] * np.conj(log_deriv)).mean(axis=0)
    forces         -= E_mean * np.conj(log_deriv_mean)

    # Forces as vector (for GRAM form, needs to be applied to samples)
    # In practice: Sp = E_local * log_deriv^*
    Sp              = (E_local - E_mean)[:, np.newaxis] * np.conj(log_deriv)
    Sp              = Sp.sum(axis=1) / n_samples  # Average over parameters? 
    # Actually, let's use proper formulation
    Sp              = E_local - E_mean
    
    print(f"Stochastic Reconfiguration setup:")
    print(f"  Parameters: {n_params}")
    print(f"  Samples: {n_samples}")
    print(f"  Energy mean: {E_mean.real:.4f}")
    
    # Solve for natural gradient direction
    # This gives S^-1 @ forces
    solver          = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func      = solver.get_solver_func(
                        backend_module  = np,
                        use_matvec      = False,
                        use_fisher      = True,
                        use_matrix      = True,
                        sigma           = 0.0
                    )
    S_dag           = S.conj().T
    forces_vec      = S_dag @ Sp
    result          = solve_func(s=S, s_p=S_dag, b=forces_vec, x0=None, tol=1e-6, maxiter=None, precond_apply=None)

    print(f"\nNatural gradient solution:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    
    # Gradient descent update
    learning_rate = 0.01
    update = -learning_rate * result.x
    print(f"\nWith learning rate = {learning_rate}:")
    print(f"  Update norm: {np.linalg.norm(update):.6f}")
    
    return result

# ----------------------------------------------------------------------------------------
#! Regularization
# ----------------------------------------------------------------------------------------

def example_regularization():
    """GRAM solve with regularization (important for stability)."""
    print("\n" + "=" * 70)
    print("Example 4: Regularization for Numerical Stability")
    print("=" * 70)
    
    # Small regularization is crucial for NQS/TDVP
    # Solves: (S\dag S + ε I) p = S\dag Sp
    
    n_samples       = 200
    n_params        = 50
    S, Sp           = create_gram_problem(n_samples, n_params, rank_deficient=True)

    # Compute condition number of S\dag S
    gram_matrix     = S.conj().T @ S
    cond            = np.linalg.cond(gram_matrix)
    
    print(f"Problem setup:")
    print(f"  Samples: {n_samples} < {n_params} (underdetermined)")
    print(f"  Gram matrix condition number: {cond:.2e}")
    
    # Without regularization - may be unstable
    print(f"\nWithout regularization:")
    solver          = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func      = solver.get_solver_func(
                        backend_module  = np,
                        use_matvec      = False,
                        use_fisher      = True,
                        use_matrix      = True,
                        sigma           = 0.0
                    )
    S_dag           = S.conj().T
    forces          = S_dag @ Sp
    result_no_reg   = solve_func(s=S, s_p=S_dag, b=forces, x0=None, tol=1e-6, maxiter=200, precond_apply=None)
    print(f"  Converged: {result_no_reg.converged}")
    print(f"  Iterations: {result_no_reg.iterations}")
    print(f"  Residual: {result_no_reg.residual_norm:.2e}")
    
    # With regularization (add diagonal shift)
    epsilon         = 1e-4
    print(f"\nWith regularization (ε = {epsilon}):")
    # We need to manually add regularization to S
    # (S\dag S + ε I) = (S\dag S) + ε I
    # For GRAM form, we can use MATRIX form with regularized Gram matrix
    gram_regularized = gram_matrix + epsilon * np.eye(n_params)
    Sp_transformed   = S.conj().T @ Sp

    solver_reg       = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func_reg   = solver_reg.get_solver_func(
                        backend_module  = np,
                        use_matvec      = False,
                        use_fisher      = False,
                        use_matrix      = True,
                        sigma           = 0.0
                        )
    result_reg       = solve_func_reg(a=gram_regularized, b=Sp_transformed, x0=None, tol=1e-6, maxiter=None, precond_apply=None)
    print(f"  Converged: {result_reg.converged}")
    print(f"  Iterations: {result_reg.iterations}")
    print(f"  Residual: {result_reg.residual_norm:.2e}")
    
    print(f"\nTip: In practice, use small ε \in  [1e-4, 1e-3] for stability")
    
    return result_no_reg, result_reg


def example_complex_gram():
    """GRAM solve with complex matrices (common in quantum mechanics)."""
    print("\n" + "=" * 70)
    print("Example 5: Complex GRAM Form")
    print("=" * 70)

    n_samples       = 500
    n_params        = 35

    # Complex S and Sp (typical for quantum wavefunctions)
    S               = np.random.randn(n_samples, n_params) + 1j * np.random.randn(n_samples, n_params)
    Sp              = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)

    print(f"Problem setup:")
    print(f"  S dtype: {S.dtype} (complex)")
    print(f"  Sp dtype: {Sp.dtype} (complex)")
    
    # Solve complex GRAM problem
    solver          = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func      = solver.get_solver_func(
        backend_module  = np,
        use_matvec      = False,
        use_fisher      = True,
        use_matrix      = True,
        sigma           = 0.0
    )
    S_dag           = S.conj().T
    forces          = S_dag @ Sp
    result          = solve_func(s=S, s_p=S_dag, b=forces, x0=None, tol=1e-7, maxiter=None, precond_apply=None)
    
    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Solution dtype: {result.x.dtype}")
    print(f"  Solution norm: {np.linalg.norm(result.x):.4f}")
    
    # Verify
    gram = S.conj().T @ S
    rhs = S.conj().T @ Sp
    residual = np.linalg.norm(gram @ result.x - rhs)
    print(f"  Gram residual: {residual:.2e}")
    
    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRAM FORM SOLVER - TDVP/NQS EXAMPLES")
    print("=" * 70)
    print("\nGRAM form solves: (S\dag S) p = S\dag Sp")
    print("Common in: TDVP, NQS, Stochastic Reconfiguration")
    print("=" * 70)
    
    # Run all examples
    example_basic_gram()
    example_tdvp_time_step()
    example_stochastic_reconfiguration()
    example_regularization()
    example_complex_gram()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. GRAM form is THE standard pattern for TDVP/NQS applications")
    print("2. Equivalent to least-squares: min ||Sp - S p||^2 ")
    print("3. Always use regularization (ε ∼ 1e-4) for numerical stability")
    print("4. Works naturally with complex matrices (quantum wavefunctions)")
    print("5. S is typically log-derivatives (∂log ψ/∂θ) from MC samples")
    print("\nNext: See example_with_preconditioners.py for acceleration")
