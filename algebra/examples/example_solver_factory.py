#!/usr/bin/env python3
"""
Solver Factory - Example

This example demonstrates the flexible solver factory interface for
creating and switching between solvers programmatically.

The factory supports multiple input formats:
    - Enum: SolverType.CG
    - String: "cg", "minres", "direct"
    - Integer: SolverType.CG.value
    - Instance: CgSolver()
    - Class: CgSolver
"""

import numpy as np
from enum import Enum
try:
    from QES.general_python.algebra.solvers import (
        choose_solver, SolverType,
        CgSolver, MinresSolverScipy, DirectSolver
    )
except ImportError:
    raise ImportError("QES package is required to run this example.")

# ----------------------------------------------------------------------------------------
#! SolverForm Enum
# ----------------------------------------------------------------------------------------

class SolverForm(Enum):
    MATRIX      = "matrix"
    VECTOR      = "vector"
    LINEAR      = "linear"
    
class ProblemType(Enum):
    SPD         = "symmetric_positive_definite"
    INDEFINITE  = "symmetric_indefinite"
    GENERAL     = "general"

# ----------------------------------------------------------------------------------------

def create_test_problem(n=50, problem_type: str = ProblemType.SPD.value):
    """Create test problems of different types."""

    if problem_type == ProblemType.SPD.value:
        # Symmetric positive-definite
        A = np.random.randn(n, n)
        A = A.T @ A + np.eye(n)
        b = np.random.randn(n)
    elif problem_type == ProblemType.INDEFINITE.value:
        # Symmetric indefinite
        eigenvalues = np.concatenate([
                        -np.linspace(0.1, 2.0, n//3),
                        np.linspace(0.1, 2.0, n - n//3)
                        ])
        # Ensure the matrix is indefinite
        Q, _    = np.linalg.qr(np.random.randn(n, n))
        A       = Q @ np.diag(eigenvalues) @ Q.T
        A       = 0.5 * (A + A.T)
        b       = np.random.randn(n)
        
    elif problem_type == ProblemType.GENERAL.value:
        # General (non-symmetric)
        A       = np.random.randn(n, n)
        b       = np.random.randn(n)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    return A, b

# ----------------------------------------------------------------------------------------

def example_enum_factory():
    """Create solver using enum."""
    print("=" * 70)
    print("Example 1: Solver Factory with Enum")
    print("=" * 70)
    
    A, b = create_test_problem(n=50, problem_type=ProblemType.SPD.value)
    
    # Create solver using SolverType enum
    solver_types = [
        SolverType.CG,
        SolverType.SCIPY_MINRES,
        SolverType.DIRECT,
    ]
    
    print(f"Creating solvers via SolverType enum:")
    for solver_type in solver_types:
        solver = choose_solver(solver_type)
        print(f"  {solver_type.name:<20} -> {type(solver).__name__}")
    
    # Use one of them
    solver      = choose_solver(SolverType.CG)
    solve_func  = solver.get_solver_func(
                            backend_module  = np,
                            use_matvec      = False,
                            use_fisher      = False,
                            use_matrix      = True,
                            sigma           = 0.0,
                        )
    result      = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=None, precond_apply=None)
    
    print(f"\nSolved with {type(solver).__name__}:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    
    return result

# ----------------------------------------------------------------------------------------

def example_string_factory():
    """Create solver using string names."""
    print("\n" + "=" * 70)
    print("Example 2: Solver Factory with Strings")
    print("=" * 70)
    
    A, b = create_test_problem(n=50, problem_type=ProblemType.SPD.value)
    
    # Factory supports flexible string naming
    solver_names = [
        "cg",
        "CG",
        "conjugate-gradient",
        "minres",
        "MINRES",
        "scipy-minres",
        "direct",
        "DIRECT",
    ]
    
    print(f"Creating solvers via string names (case-insensitive):")
    for name in solver_names:
        try:
            solver = choose_solver(name)
            print(f"  '{name}':<25 -> {type(solver).__name__}")
        except (ValueError, KeyError) as e:
            print(f"  '{name}' -> Error: {e}")
    
    # Solve with string-created solver
    solver      = choose_solver("cg")
    solve_func  = solver.get_solver_func(
        backend_module  = np,
        use_matvec      = False,
        use_fisher      = False,
        use_matrix      = True,
        sigma           = 0.0,
    )
    result      = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=None, precond_apply=None)
    
    print(f"\nSolved with choose_solver('cg'):")
    print(f"  Converged: {result.converged}")
    
    return result

# ----------------------------------------------------------------------------------------

def example_int_factory():
    """Create solver using integer IDs."""
    print("\n" + "=" * 70)
    print("Example 3: Solver Factory with Integer IDs")
    print("=" * 70)
    
    A, b = create_test_problem(n=50, problem_type=ProblemType.SPD.value)
    
    # Can use enum values as integers
    print(f"SolverType enum values:")
    for solver_type in SolverType:
        print(f"  {solver_type.name:<20} = {solver_type.value}")
    
    # Create using integer
    solver_id = SolverType.CG.value
    solver = choose_solver(solver_id)
    
    print(f"\nCreated solver from int {solver_id}: {type(solver).__name__}")
    
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0,
    )
    result = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=None, precond_apply=None)
    print(f"  Converged: {result.converged}")
    
    return result

# ----------------------------------------------------------------------------------------

def example_instance_factory():
    """Pass solver instance directly."""
    print("\n" + "=" * 70)
    print("Example 4: Using Solver Instance Directly")
    print("=" * 70)
    
    A, b = create_test_problem(n=50, problem_type=ProblemType.SPD.value)
    
    # Can pass instance directly (factory returns it as-is)
    solver_instance = CgSolver()
    solver = choose_solver(solver_instance)
    
    print(f"Passing instance: {type(solver_instance).__name__}")
    print(f"Factory returns: {type(solver).__name__}")
    print(f"Same object: {solver is solver_instance}")
    
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0,
    )
    result = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=None, precond_apply=None)
    print(f"\nConverged: {result.converged}")
    
    return result


def example_adaptive_solver_selection():
    """Automatically select solver based on problem type."""
    print("\n" + "=" * 70)
    print("Example 5: Adaptive Solver Selection")
    print("=" * 70)
    
    def select_solver_for_problem(A):
        """Heuristic to choose appropriate solver."""
        n = A.shape[0]
        
        # Check symmetry
        is_symmetric = np.allclose(A, A.T)
        
        if not is_symmetric:
            return "direct", "Non-symmetric -> Direct"
        
        # Check definiteness (for symmetric)
        eigenvalues = np.linalg.eigvalsh(A)
        is_positive_definite = np.all(eigenvalues > 1e-10)
        
        if is_positive_definite:
            if n < 100:
                return "direct", "SPD, small -> Direct"
            else:
                return "cg", "SPD, large -> CG"
        else:
            if n < 100:
                return "direct", "Indefinite, small -> Direct"
            else:
                return "minres", "Indefinite, large -> MINRES"
    
    # Test different problem types
    problems = [
        ("SPD small", 50,           ProblemType.SPD.value),
        ("SPD large", 200,          ProblemType.SPD.value),
        ("Indefinite small", 50,    ProblemType.INDEFINITE.value),
        ("Indefinite large", 200,   ProblemType.INDEFINITE.value),
        ("General", 80,             ProblemType.GENERAL.value),
    ]
    
    print(f"Adaptive solver selection:")
    print(f"{'Problem':<20} {'Size':<8} {'Selected Solver':<15} {'Reason'}")
    print("-" * 75)
    
    for name, size, ptype in problems:
        A, b = create_test_problem(n=size, problem_type=ptype)
        solver_name, reason = select_solver_for_problem(A)
        solver = choose_solver(solver_name)
        
        print(f"{name:<20} {size:<8} {type(solver).__name__:<15} {reason}")
    
    print(f"\nAdaptive selection balances efficiency and applicability")

# ------------------------------------------------------------------------------------

def example_solver_config():
    """Configure solver parameters through factory."""
    print("\n" + "=" * 70)
    print("Example 6: Solver Configuration")
    print("=" * 70)
    
    A, b = create_test_problem(n=80, problem_type=ProblemType.SPD.value)
    
    # Create solver
    solver = choose_solver("cg")

    # Prepare solve function once
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0,
    )
    
    print(f"Solver: {type(solver).__name__}")
    print(f"\nSolving with different tolerances:")
    
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    print(f"{'Tolerance':<12} {'Iterations':<12} {'Residual'}")
    print("-" * 45)
    
    for tol in tolerances:
        result = solve_func(a=A, b=b, x0=None, tol=tol, maxiter=None, precond_apply=None)
        print(f"{tol:<12.2e} {result.iterations:<12} {result.residual_norm:.2e}")
    
    print(f"\nConfiguration controls convergence behavior")


def example_solver_switching():
    """Switch between solvers for the same problem."""
    print("\n" + "=" * 70)
    print("Example 7: Solver Switching")
    print("=" * 70)
    
    A, b = create_test_problem(n=60, problem_type=ProblemType.SPD.value)
    
    # Try multiple solvers on same problem
    solver_choices = [
        ("cg", "Conjugate Gradient"),
        ("minres", "MINRES (SciPy)"),
        ("direct", "Direct (LU)"),
    ]
    
    print(f"Solving same SPD system with different solvers:")
    print(f"{'Solver':<25} {'Converged':<12} {'Iterations':<12} {'Residual'}")
    print("-" * 70)
    
    for solver_name, description in solver_choices:
        solver = choose_solver(solver_name)
        solve_func = solver.get_solver_func(
            backend_module=np,
            use_matvec=False,
            use_fisher=False,
            use_matrix=True,
            sigma=0.0,
        )
        result = solve_func(a=A, b=b, x0=None, tol=1e-8, maxiter=None, precond_apply=None)
        
        iters = result.iterations if result.iterations is not None else "N/A"
        print(f"{description:<25} {str(result.converged):<12} {str(iters):<12} "
              f"{result.residual_norm:.2e}")
    
    print(f"\nAll solvers reach the same solution (different paths)")


def example_programmatic_workflow():
    """Realistic workflow: user-configurable solver."""
    print("\n" + "=" * 70)
    print("Example 8: Programmatic Workflow")
    print("=" * 70)
    
    # Simulate user configuration (could come from config file, CLI args, etc.)
    user_config = {
        'solver': 'cg',  # or 'minres', 'direct', etc.
        'tolerance': 1e-6,
        'max_iterations': 200,
    }
    
    print(f"User configuration:")
    for key, value in user_config.items():
        print(f"  {key}: {value}")
    
    # Create problem
    A, b = create_test_problem(n=100, problem_type=ProblemType.SPD.value)
    
    # Create solver from config
    solver = choose_solver(user_config['solver'])
    
    print(f"\nCreated solver: {type(solver).__name__}")
    
    # Solve with configured parameters
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.0,
    )
    result = solve_func(
        a=A,
        b=b,
        x0=None,
        tol=user_config['tolerance'],
        maxiter=user_config['max_iterations'],
        precond_apply=None,
    )
    
    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual: {result.residual_norm:.2e}")
    
    # Easy to switch solver by changing config
    print(f"\nTo switch solver, just change config['solver']")
    print(f"  e.g., config['solver'] = 'minres'")
    
    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SOLVER FACTORY - EXAMPLES")
    print("=" * 70)
    print("\nFlexible solver creation: Enum, String, Int, Instance, Class")
    print("=" * 70)
    
    # Run all examples
    example_enum_factory()
    example_string_factory()
    example_int_factory()
    example_instance_factory()
    example_adaptive_solver_selection()
    example_solver_config()
    example_solver_switching()
    example_programmatic_workflow()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. choose_solver() accepts: Enum, string, int, instance, class")
    print("2. String names are case-insensitive and flexible")
    print("3. Factory enables programmatic solver selection")
    print("4. Easy to switch solvers via configuration")
    print("5. Adaptive selection based on problem properties")
    print("6. Same interface for all solvers -> clean code")
    print("\nRecommendation: Use factory for flexibility and maintainability")

# ----------------------------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------------------------
