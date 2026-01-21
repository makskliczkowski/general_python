#!/usr/bin/env python3
"""
Conjugate Gradient (CG) Solver - Basic Example

This example demonstrates how to use the Conjugate Gradient solver to solve
symmetric positive-definite linear systems.

Mathematical Problem:
    Solve Ax = b where A is symmetric positive-definite (SPD)

CG is optimal for SPD systems - it finds the exact solution in at most n iterations
(in exact arithmetic) and often converges much faster in practice.

API Usage Pattern:
    1. Create solver instance: solver = choose_solver('cg', sigma=0.0)
    2. Get solve function: solve_func = solver.get_solver_func(backend_module=np, ...)
    3. Call solve function: result = solve_func(a=A, b=b, ...)
"""

import numpy as np
from general_python.algebra import solvers

def create_spd_matrix(n, condition_number=10.0):
    """
    Create a symmetric positive-definite matrix with specified condition number.
    
    The condition number k(A) = \lambda_max / \lambda_min affects convergence rate:
    - Small k (well-conditioned): Fast convergence
    - Large k (ill-conditioned): Slow convergence
    
    Args:
        n: Matrix dimension
        condition_number: Desired condition number
        
    Returns:
        n\timesn SPD matrix
    """
    # Create eigenvalues from 1 to condition_number
    eigenvalues = np.linspace(1.0, condition_number, n)
    
    # Create random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # A = Q @ diag(eigenvalues) @ Q^T
    A = Q @ np.diag(eigenvalues) @ Q.T
    
    return A


def example_basic_solve():
    """Basic CG solve with default parameters."""
    print("=" * 70)
    print("Example 1: Basic CG Solve")
    print("=" * 70)
    
    # Problem setup
    n = 100
    A = create_spd_matrix(n, condition_number=10.0)
    x_true = np.random.randn(n)
    b = A @ x_true  # Create b from true solution
    
    print(f"Problem size: {n}")
    print(f"Matrix condition number: {np.linalg.cond(A):.2e}")
    
    # Solve using CG with correct API:
    # Step 1: Create solver instance
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    
    # Step 2: Get solve function for MATRIX form
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0
    )
    
    # Step 3: Call solve function
    result = solve_func(
        a=A,
        b=b,
        x0=None,
        tol=1e-10,
        maxiter=None,
        precond_apply=None
    )
    
    # Check solution
    error = np.linalg.norm(result.x - x_true)
    residual = np.linalg.norm(A @ result.x - b)
    
    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual norm: {result.residual_norm:.2e}")
    print(f"  Solution error: {error:.2e}")
    print(f"  True residual: {residual:.2e}")
    
    return result


def example_with_tolerance():
    """CG with custom tolerance settings."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Tolerance")
    print("=" * 70)
    
    n = 50
    A = create_spd_matrix(n, condition_number=100.0)
    b = np.random.randn(n)
    
    print(f"Problem size: {n}")
    print(f"Matrix condition number: {np.linalg.cond(A):.2e}")
    
    # Solve with tight tolerance
    tol = 1e-10
    
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0
    )
    
    result = solve_func(
        a=A,
        b=b,
        x0=None,
        tol=tol,
        maxiter=None,
        precond_apply=None
    )
    
    print(f"\nTolerance: {tol:.2e}")
    print(f"Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual norm: {result.residual_norm:.2e}")
    
    return result


def example_ill_conditioned():
    """CG on ill-conditioned system showing slow convergence."""
    print("\n" + "=" * 70)
    print("Example 3: Ill-Conditioned System")
    print("=" * 70)
    
    n = 50
    condition_number = 1000.0
    A = create_spd_matrix(n, condition_number=condition_number)
    b = np.random.randn(n)
    
    print(f"Problem size: {n}")
    print(f"Matrix condition number: {condition_number:.2e}")
    
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0
    )
    
    # Solve with limited max iterations
    result = solve_func(
        a=A,
        b=b,
        x0=None,
        tol=1e-10,
        maxiter=100,
        precond_apply=None
    )
    
    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual norm: {result.residual_norm:.2e}")
    
    if not result.converged:
        print(f"\n  Note: CG did not converge in {result.iterations} iterations")
        print(f"  This is expected for ill-conditioned systems")
        print(f"  Consider using a preconditioner (see example_with_preconditioners.py)")
    
    return result


def example_solver_function():
    """Using the solver function for multiple solves with the same matrix."""
    print("\n" + "=" * 70)
    print("Example 4: Reusable Solver Function")
    print("=" * 70)
    
    n = 30
    A = create_spd_matrix(n, condition_number=5.0)
    b1 = np.random.randn(n)
    b2 = np.random.randn(n)
    
    print(f"Problem size: {n}")
    
    # Get solver function once
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0
    )
    
    # Can now call this function multiple times efficiently
    result1 = solve_func(a=A, b=b1, x0=None, tol=1e-6, maxiter=None, precond_apply=None)
    result2 = solve_func(a=A, b=b2, x0=None, tol=1e-6, maxiter=None, precond_apply=None)
    
    print(f"\nFirst solve (RHS #1):")
    print(f"  Iterations: {result1.iterations}")
    print(f"  Residual: {result1.residual_norm:.2e}")
    
    print(f"\nSecond solve (RHS #2):")
    print(f"  Iterations: {result2.iterations}")
    print(f"  Residual: {result2.residual_norm:.2e}")
    
    return result1, result2


def example_initial_guess():
    """Using an initial guess to warm-start CG."""
    print("\n" + "=" * 70)
    print("Example 5: Initial Guess (Warm Start)")
    print("=" * 70)
    
    n = 50
    A = create_spd_matrix(n, condition_number=20.0)
    x_true = np.random.randn(n)
    b = A @ x_true
    
    print(f"Problem size: {n}")
    
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0
    )
    
    # Solve from zero initial guess (cold start)
    result_cold = solve_func(
        a=A,
        b=b,
        x0=None,  # Will default to zeros
        tol=1e-10,
        maxiter=None,
        precond_apply=None
    )
    
    # Solve with initial guess close to solution (warm start)
    x0 = x_true + 0.1 * np.random.randn(n)  # Perturbed true solution
    result_warm = solve_func(
        a=A,
        b=b,
        x0=x0,
        tol=1e-10,
        maxiter=None,
        precond_apply=None
    )
    
    print(f"\nCold start (x0 = 0):")
    print(f"  Iterations: {result_cold.iterations}")
    print(f"  Residual: {result_cold.residual_norm:.2e}")
    
    print(f"\nWarm start (x0 ≈ x_true):")
    print(f"  Iterations: {result_warm.iterations}")
    print(f"  Residual: {result_warm.residual_norm:.2e}")
    print(f"\n  Warm start reduced iterations by: {result_cold.iterations - result_warm.iterations}")
    
    return result_cold, result_warm


if __name__ == "__main__":
    # Run all examples
    example_basic_solve()
    example_with_tolerance()
    example_ill_conditioned()
    example_solver_function()
    example_initial_guess()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
