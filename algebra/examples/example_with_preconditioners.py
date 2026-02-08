#!/usr/bin/env python3
"""
Preconditioned Solvers - Example

This example demonstrates how to use preconditioners to accelerate iterative
solver convergence, especially for ill-conditioned systems.

Mathematical Background:
    Instead of solving Ax = b directly, solve:
        M^(-1) A x = M^(-1) b
    where M ≈ A is easy to invert.

A good preconditioner M satisfies:
    1. M ≈ A (close approximation)
    2. M^(-1) is cheap to apply
    3. k(M^(-1) A) << k(A) (better condition number)
"""

import numpy as np
from general_python.algebra import solvers
from general_python.algebra.preconditioners import (
    choose_precond, JacobiPreconditioner, CholeskyPreconditioner,
    SSORPreconditioner
)

# ----------------------------------------------------------------------------------------
#! Helper function to create ill-conditioned SPD matrix
# ----------------------------------------------------------------------------------------

def create_ill_conditioned_spd(n, condition_number=1000.0):
    """Create ill-conditioned symmetric positive-definite matrix."""
    eigenvalues     = np.linspace(1.0, condition_number, n)
    Q, _            = np.linalg.qr(np.random.randn(n, n))
    A               = Q @ np.diag(eigenvalues) @ Q.T
    return A

# ----------------------------------------------------------------------------------------
#! Example functions for different preconditioners
# ----------------------------------------------------------------------------------------

def example_no_preconditioner():
    """Baseline: Ill-conditioned system without preconditioner."""
    print("=" * 70)
    print("Example 1: Ill-Conditioned System (Baseline - No Preconditioner)")
    print("=" * 70)
    
    n                   = 100
    condition_number    = 1000.0
    A                   = create_ill_conditioned_spd(n, condition_number)
    b                   = np.random.randn(n)
    
    print(f"Problem size: {n}")
    print(f"Condition number k(A): {condition_number:.2e}")
    print(rf"Expected CG iterations: O(\sqrt k) ≈ {int(np.sqrt(condition_number))}")
    
    # Solve without preconditioner
    solver      = solvers.choose_solver('cg', sigma=0.0)
    solve_func  = solver.get_solver_func(
                    backend_module  = np,
                    use_matvec      = False,
                    use_fisher      = False,
                    use_matrix      = True,
                    sigma           = 0.0
                )
    result      = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=500, precond_apply=None)
    
    print(f"\nResults (no preconditioner):")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual: {result.residual_norm:.2e}")
    
    if not result.converged:
        print(f"  (warning) Failed to converge in {result.iterations} iterations")
    
    return result

# ----------------------------------------------------------------------------------------
#! Other example functions for different preconditioners
# ----------------------------------------------------------------------------------------

def example_jacobi_preconditioner():
    """Jacobi (diagonal) preconditioner."""
    print("\n" + "=" * 70)
    print("Example 2: Jacobi (Diagonal) Preconditioner")
    print("=" * 70)
    
    n                   = 100
    condition_number    = 1000.0

    # Create diagonally dominant matrix (Jacobi works well here)
    A                   = create_ill_conditioned_spd(n, condition_number)
    # Make it more diagonally dominant
    A                   = A + 10.0 * np.diag(np.abs(np.random.randn(n)))
    b                   = np.random.randn(n)
            
    print(f"Problem size: {n}")
    print(f"Matrix: Diagonally dominant SPD")
    
    # Solve without preconditioner (baseline)
    solver              = solvers.choose_solver('cg', sigma=0.0)
    solve_func          = solver.get_solver_func(
                                    backend_module  = np,
                                    use_matvec      = False,
                                    use_fisher      = False,
                                    use_matrix      = True,
                                    sigma           = 0.0
                                )
    result_no_precond   = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=500, precond_apply=None)
    
    # Create Jacobi preconditioner
    precond             = JacobiPreconditioner()
    precond.set(A)
    
    # Solve with Jacobi preconditioner
    precond_apply       = precond.get_apply()  # Get the instance apply function
    result_precond      = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=500, precond_apply=precond_apply)
    
    print(f"\nResults comparison:")
    print(f"{'Method':<25} {'Converged':<12} {'Iterations':<12} {'Residual'}")
    print("-" * 70)
    print(f"{'No preconditioner':<25} {str(result_no_precond.converged):<12} "
          f"{result_no_precond.iterations:<12} {result_no_precond.residual_norm:.2e}")
    print(f"{'Jacobi preconditioner':<25} {str(result_precond.converged):<12} "
          f"{result_precond.iterations:<12} {result_precond.residual_norm:.2e}")
    
    speedup = result_no_precond.iterations / max(result_precond.iterations, 1)
    print(f"\nSpeedup: {speedup:.1f}\times")
    
    return result_precond

# ----------------------------------------------------------------------------------------
#! Example functions for different preconditioners
# ----------------------------------------------------------------------------------------

def example_cholesky_preconditioner():
    """Cholesky (incomplete/approximate) preconditioner."""
    print("\n" + "=" * 70)
    print("Example 3: Cholesky Preconditioner")
    print("=" * 70)
    
    n                   = 80
    condition_number    = 500.0
    A                   = create_ill_conditioned_spd(n, condition_number)
    b                   = np.random.randn(n)

    print(f"Problem size: {n}")
    print(f"Condition number: {condition_number:.2e}")
    
    # Baseline
    solver      = solvers.choose_solver('cg', sigma=0.0)
    solve_func  = solver.get_solver_func(
                            backend_module  =np,
                            use_matvec      =False,
                            use_fisher      =False,
                            use_matrix      =True,
                            sigma           =0.0
                        )
    result_no_precond = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=300, precond_apply=None)
    
    # Cholesky preconditioner (uses full Cholesky factorization)
    precond = CholeskyPreconditioner()
    precond.set(A)
    
    precond_apply   = precond.get_apply()
    result_precond  = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=300, precond_apply=precond_apply)
    
    print(f"\nResults comparison:")
    print(f"{'Method':<25} {'Converged':<12} {'Iterations':<12} {'Residual'}")
    print("-" * 70)
    print(f"{'No preconditioner':<25} {str(result_no_precond.converged):<12} "
          f"{result_no_precond.iterations:<12} {result_no_precond.residual_norm:.2e}")
    print(f"{'Cholesky preconditioner':<25} {str(result_precond.converged):<12} "
          f"{result_precond.iterations:<12} {result_precond.residual_norm:.2e}")
    
    print(f"\nNote: Cholesky uses M = LL^T factorization of A")
    print(f"      Very effective but O(n^3) setup cost")
    
    return result_precond

# ----------------------------------------------------------------------------------------
#! Compare 
# ----------------------------------------------------------------------------------------

def example_preconditioner_comparison():
    """Compare all available preconditioners."""
    print("\n" + "=" * 70)
    print("Example 5: Preconditioner Comparison")
    print("=" * 70)
    
    n                   = 60
    condition_number    = 1000.0
    A                   = create_ill_conditioned_spd(n, condition_number)
    b                   = np.random.randn(n)

    print(f"Problem size: {n}")
    print(f"Condition number: {condition_number:.2e}")
    
    # Setup solver
    solver               = solvers.choose_solver('cg', sigma=0.0)
    solve_func           = solver.get_solver_func(
                            backend_module  =   np,
                            use_matvec      =   False,
                            use_fisher      =   False,
                            use_matrix      =   True,
                            sigma           =   0.0
                        )
    
    # Test different preconditioners
    preconditioners         = [
        (None, "None (baseline)"),
        (JacobiPreconditioner(), "Jacobi"),
        (CholeskyPreconditioner(), "Cholesky"),
        (SSORPreconditioner(omega=1.0), r"SSOR (\omega =1.0)"),
        (SSORPreconditioner(omega=1.5), r"SSOR (\omega =1.5)"),
    ]
    
    print(f"\nComparing preconditioners:")
    print(f"{'Preconditioner':<20} {'Setup(ms)':<12} {'Converged':<12} {'Iterations':<12} {'Residual'}")
    print("-" * 80)
    
    results = []
    for precond, name in preconditioners:
        import time
        
        # Setup preconditioner
        if precond is not None:
            t0              = time.time()
            precond.set(A)
            setup_time      = (time.time() - t0) * 1000
            precond_apply   = precond.get_apply()
        else:
            setup_time      = 0.0
            precond_apply   = None

        # Solve
        result = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=500, precond_apply=precond_apply)
        
        results.append((name, setup_time, result))
        
        print(f"{name:<20} {setup_time:<12.2f} {str(result.converged):<12} "
              f"{result.iterations:<12} {result.residual_norm:.2e}")
    
    # Find best
    converged_results = [(n, r) for n, s, r in results if r.converged]
    if converged_results:
        best_name, best_result = min(converged_results, key=lambda x: x[1].iterations)
        print(f"\nv Best performance: {best_name} ({best_result.iterations} iterations)")
    
    return results

# ----------------------------------------------------------------------------------------
#! Example functions for different preconditioners
# ----------------------------------------------------------------------------------------

def example_gram_form_with_preconditioner():
    """Preconditioner with GRAM form (TDVP/NQS use case)."""
    print("\n" + "=" * 70)
    print("Example 6: GRAM Form with Preconditioner (TDVP/NQS)")
    print("=" * 70)
    
    # Typical NQS/TDVP problem
    n_samples   = 500
    n_params    = 50
    
    S           = np.random.randn(n_samples, n_params) + 1j * np.random.randn(n_samples, n_params)
    Sp          = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)

    print(f"GRAM problem:")
    print(f"  Samples: {n_samples}")
    print(f"  Parameters: {n_params}")
    print(rf"  Solving: (S\dag S) p = S\dag Sp")
    
    # Setup solver for Fisher/GRAM form
    solver      = solvers.choose_solver('cg', sigma=0.0)
    solve_func  = solver.get_solver_func(
                            backend_module  = np,
                            use_matvec      = False,
                            use_fisher      = True,
                            use_matrix      = False,
                            sigma           = 0.0
                        )
    
    # Prepare Fisher form inputs: s=S, s_p=S\dag, b=S\dagSp
    S_dag               = S.conj().T  # [n_params, n_samples]
    forces              = S_dag @ Sp  # [n_params]

    # Without preconditioner
    result_no_precond   = solve_func(s=S, s_p=S_dag, b=forces, x0=None, tol=1e-6, maxiter=200, precond_apply=None)
    
    # With Jacobi preconditioner for GRAM form
    # Preconditioner acts on Gram matrix S\dag S
    precond             = JacobiPreconditioner(is_gram=True)
    precond.set(S, ap=S_dag)  # Pass both S and Sdagger  for Gram form

    precond_apply       = precond.get_apply()
    result_precond      = solve_func(s=S, s_p=S_dag, b=forces, x0=None, tol=1e-6, maxiter=200, precond_apply=precond_apply)

    print(f"\nResults comparison:")
    print(f"{'Method':<25} {'Converged':<12} {'Iterations':<12} {'Residual'}")
    print("-" * 70)
    print(f"{'No preconditioner':<25} {str(result_no_precond.converged):<12} "
          f"{result_no_precond.iterations:<12} {result_no_precond.residual_norm:.2e}")
    print(f"{'Jacobi (GRAM)':<25} {str(result_precond.converged):<12} "
          f"{result_precond.iterations:<12} {result_precond.residual_norm:.2e}")
    
    print(rf"\nNote: Preconditioner is applied to Gram matrix S\dag S")
    print(f"      Common in TDVP for quantum geometric tensor preconditioning")
    
    return result_precond

# ----------------------------------------------------------------------------------------
#! Factory
# ----------------------------------------------------------------------------------------

def example_factory_usage():
    """Using preconditioner factory."""
    
    print("\n" + "=" * 70)
    print("Example 7: Preconditioner Factory")
    print("=" * 70)
    
    n = 80
    A = create_ill_conditioned_spd(n, condition_number=500.0)
    b = np.random.randn(n)
    
    print(f"Problem size: {n}")
    
    # Use factory to create preconditioner by name
    precond_names = ["identity", "jacobi"]
    
    print(f"\nCreating preconditioners via factory:")
    for name in precond_names:
        precond     = choose_precond(name)
        print(f"  choose_precond('{name}'): {type(precond).__name__}")
    
    # Use case-insensitive, flexible naming (examples that work)
    print(f"\nFactory supports enum-based naming:")
    flexible_names  = ["IDENTITY", "JACOBI"]
    for name in flexible_names:
        try:
            precond     = choose_precond(name)
            print(f"  choose_precond('{name}'): {type(precond).__name__}")
        except (ValueError, TypeError) as e:
            print(f"  choose_precond('{name}'): Not implemented")
    
    # Solve with factory-created preconditioner
    solver          = solvers.choose_solver('cg', sigma=0.0)
    solve_func      = solver.get_solver_func(
                                    backend_module  = np,
                                    use_matvec      = False,
                                    use_fisher      = False,
                                    use_matrix      = True,
                                    sigma           = 0.0
                                )
    
    precond         = choose_precond("jacobi")
    precond.set(A)
    precond_apply   = precond.get_apply()
    result          = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=None, precond_apply=precond_apply)
    
    print(f"\nSolve with factory preconditioner:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    
    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PRECONDITIONED SOLVERS - EXAMPLES")
    print("=" * 70)
    print("\nPreconditioners accelerate convergence for ill-conditioned systems")
    print("=" * 70)
    
    # Run all examples
    example_no_preconditioner()
    example_jacobi_preconditioner()
    example_cholesky_preconditioner()
    example_preconditioner_comparison()
    example_gram_form_with_preconditioner()
    example_factory_usage()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Preconditioners drastically reduce iterations for ill-conditioned systems")
    print("2. Jacobi: Simple, cheap, good for diagonally dominant matrices")
    print("3. Cholesky: Very effective but expensive O(n^3) setup")
    print(r"4. SSOR: Good balance, omega \in [1.0, 2.0] for tuning")
    print(r"5. GRAM form: Preconditioner applies to S\dag S (common in TDVP/NQS)")
    print("6. Factory: choose_precond() for flexible creation")
    print("\nRecommendation: Start with Jacobi, upgrade to SSOR/Cholesky if needed")

# ------------------------------------------------------
#! EOF
# ------------------------------------------------------