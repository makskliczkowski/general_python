#!/usr/bin/env python3
"""
Direct Solvers - Example

This example demonstrates direct solvers (LU decomposition, pseudo-inverse)
for linear systems. Direct methods compute the exact solution (in exact arithmetic)
in a finite number of operations, unlike iterative methods.

When to use direct solvers:
    - Small to medium systems (n < 10,000)
    - When exact solution is required
    - Non-symmetric or general matrices
    - Rank-deficient systems (use pseudo-inverse)

File        : general_python/algebra/examples/example_direct.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
"""

import numpy as np

try:
    from QES.general_python.algebra import solvers
except ImportError as e:
    print(f"Failed to import solvers module: {e}")
    raise

# --------------------------------------------------------------------
#! Example 1: Basic Direct Solve
# --------------------------------------------------------------------

def example_basic_direct():
    """Basic direct solver for general linear system."""
    print("=" * 70)
    print("Example 1: Basic Direct Solve")
    print("=" * 70)
    
    # General (non-symmetric) system
    n       = 100
    A       = np.random.randn(n, n)
    x_true  = np.random.randn(n)
    b       = A @ x_true

    print(f"Problem size: {n}")
    print(f"Matrix type: General (non-symmetric)")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    # Solve using direct method (LU decomposition) via unified 3-step API
    solver      = solvers.choose_solver('direct')
    solve_func  = solver.get_solver_func(
                            backend_module  = np,
                            use_matvec      = False,
                            use_fisher      = False,
                            use_matrix      = True,
                            sigma           = 0.0,
                        )
    result      = solve_func(a=A, b=b, x0=None, tol=0.0, maxiter=None, precond_apply=None)
    
    # Check solution
    error       = np.linalg.norm(result.x - x_true)
    residual    = np.linalg.norm(A @ result.x - b)

    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Solution error: {error:.2e}")
    print(f"  Residual ||Ax - b||: {residual:.2e}")
    print(f"\nNote: Direct solver gives exact solution (within machine precision)")
    
    return result

# --------------------------------------------------------------------
#! Example 2: Complex System
# --------------------------------------------------------------------

def example_complex_system():
    """Direct solver for complex-valued system."""
    
    print("\n" + "=" * 70)
    print("Example 2: Complex Linear System")
    print("=" * 70)
    
    n = 50
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    b = np.random.randn(n) + 1j * np.random.randn(n)

    print(f"Problem size: {n}")
    print(f"Matrix dtype: {A.dtype} (complex)")
    
    # Direct solve
    solver      = solvers.choose_solver('direct')
    solve_func  = solver.get_solver_func(
                        backend_module  = np,
                        use_matvec      = False,
                        use_fisher      = False,
                        use_matrix      = True,
                        sigma           = 0.0,
                    )
    result      = solve_func(a=A, b=b, x0=None, tol=0.0, maxiter=None, precond_apply=None)
    residual    = np.linalg.norm(A @ result.x - b)
    
    print(f"\nResults:")
    print(f"  Solution dtype: {result.x.dtype}")
    print(f"  Residual: {residual:.2e}")
    print(f"\nDirect solver handles complex systems naturally")
    
    return result

# --------------------------------------------------------------------
#! Example 3: Rank-Deficient System
# --------------------------------------------------------------------

def example_rank_deficient():
    """Pseudo-inverse for rank-deficient system."""
    
    print("\n" + "=" * 70)
    print("Example 3: Rank-Deficient System (Pseudo-Inverse)")
    print("=" * 70)
    
    # Create rank-deficient matrix
    n               = 50
    rank            = 30  # < n
    
    # A = U @ Σ @ V^T with only `rank` non-zero singular values
    U, _            = np.linalg.qr(np.random.randn(n, n))
    V, _            = np.linalg.qr(np.random.randn(n, n))
    sigma           = np.zeros(n)
    sigma[:rank]    = np.linspace(1.0, 10.0, rank)
    A               = U @ np.diag(sigma) @ V.T
    b               = np.random.randn(n)

    print(f"Problem size: {n}")
    print(f"Matrix rank: {rank} (rank-deficient)")
    print(f"Numerical rank: {np.linalg.matrix_rank(A)}")
    
    # Direct solver will fail or give unstable solution
    print(f"\nAttempting standard direct solver:")
    try:
        solver          = solvers.choose_solver('direct')
        solve_func      = solver.get_solver_func(
                            backend_module  = np,
                            use_matvec      = False,
                            use_fisher      = False,
                            use_matrix      = True,
                            sigma           = 0.0,
                        )
        result_direct   = solve_func(a=A, b=b, x0=None, tol=0.0, maxiter=None, precond_apply=None)
        residual_direct = np.linalg.norm(A @ result_direct.x - b)
        print(f"  Residual: {residual_direct:.2e}")
        print(f"  Solution norm: {np.linalg.norm(result_direct.x):.2e}")
    except Exception as e:
        print(f"  (x) Failed: {e}")
    
    # Pseudo-inverse finds minimum norm solution
    print(f"\nUsing pseudo-inverse solver:")
    solver      = solvers.choose_solver('pseudo_inverse')
    solve_func  = solver.get_solver_func(
                    backend_module  = np,
                    use_matvec      = False,
                    use_fisher      = False,
                    use_matrix      = True,
                    sigma           = 0.0,
                )
    result_pinv     = solve_func(a=A, b=b, x0=None, tol=1e-12, maxiter=None, precond_apply=None)
    residual_pinv   = np.linalg.norm(A @ result_pinv.x - b)
    
    print(f"  Converged: {result_pinv.converged}")
    print(f"  Residual: {residual_pinv:.2e}")
    print(f"  Solution norm: {np.linalg.norm(result_pinv.x):.2e}")
    print(f"\nv Pseudo-inverse finds minimum-norm least-squares solution")
    
    return result_pinv

# --------------------------------------------------------------------
#! Example 4: Underdetermined System
# --------------------------------------------------------------------

def example_underdetermined():
    """Pseudo-inverse for underdetermined system (more variables than equations)."""
    print("\n" + "=" * 70)
    print("Example 4: Underdetermined System (m < n)")
    print("=" * 70)
    
    m = 30  # equations
    n = 50  # variables (n > m)
    
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    
    print(f"System size: {m} equations, {n} variables")
    print(f"Underdetermined: infinitely many solutions exist")
    
    # Pseudo-inverse finds minimum norm solution
    solver      = solvers.choose_solver('pseudo_inverse')
    solve_func  = solver.get_solver_func(
        backend_module  = np,
        use_matvec      = False,
        use_fisher      = False,
        use_matrix      = True,
        sigma           = 0.0,
    )
    result      = solve_func(a=A, b=b, x0=None, tol=1e-12, maxiter=None, precond_apply=None)
    residual    = np.linalg.norm(A @ result.x - b)
    sol_norm    = np.linalg.norm(result.x)
    
    print(f"\nPseudo-inverse solution:")
    print(f"  Residual: {residual:.2e}")
    print(f"  Solution norm: {sol_norm:.4f}")
    
    # Verify it's minimum norm by comparing with another solution
    # Any x = x_pinv + z where Az = 0 is also a solution
    # but has larger norm
    null_vector     = np.random.randn(n)
    null_vector     = null_vector - (A @ null_vector).T @ np.linalg.lstsq(A.T, null_vector, rcond=None)[0]
    x_alt           = result.x + 0.1 * null_vector / np.linalg.norm(null_vector)

    residual_alt    = np.linalg.norm(A @ x_alt - b)
    norm_alt        = np.linalg.norm(x_alt)

    print(f"\nAlternate solution (in null space):")
    print(f"  Residual: {residual_alt:.2e} (same)")
    print(f"  Norm: {norm_alt:.4f} (larger)")
    print(f"\nv Pseudo-inverse gives minimum-norm solution among all solutions")
    
    return result

# --------------------------------------------------------------------
#! Example 5: Overdetermined System
# --------------------------------------------------------------------

def example_overdetermined():
    """Pseudo-inverse for overdetermined system (least-squares)."""
    print("\n" + "=" * 70)
    print("Example 5: Overdetermined System (Least-Squares)")
    print("=" * 70)
    
    m           = 100  # equations
    n           = 30   # variables (m > n)
    
    A           = np.random.randn(m, n)
    x_true      = np.random.randn(n)
    b_exact     = A @ x_true
    # Add noise
    noise       = 0.1 * np.random.randn(m)
    b           = b_exact + noise
    
    print(f"System size: {m} equations, {n} variables")
    print(f"Overdetermined: No exact solution (due to noise)")
    print(f"Noise level: {np.linalg.norm(noise):.4f}")
    
    # Pseudo-inverse solves least-squares problem: min ||Ax - b||^2 
    solver      = solvers.choose_solver('pseudo_inverse')
    solve_func  = solver.get_solver_func(
        backend_module  = np,
        use_matvec      = False,
        use_fisher      = False,
        use_matrix      = True,
        sigma           = 0.0,
    )
    result      = solve_func(a=A, b=b, x0=None, tol=1e-12, maxiter=None, precond_apply=None)

    residual    = np.linalg.norm(A @ result.x - b)
    error       = np.linalg.norm(result.x - x_true)

    print(f"\nLeast-squares solution:")
    print(f"  Residual ||Ax - b||: {residual:.4f}")
    print(f"  Error ||x - x_true||: {error:.4f}")
    
    # Compare with numpy lstsq
    x_lstsq     = np.linalg.lstsq(A, b, rcond=None)[0]
    error_lstsq = np.linalg.norm(x_lstsq - x_true)
    
    print(f"\nNumPy lstsq error: {error_lstsq:.4f} (should match)")
    return result

# --------------------------------------------------------------------
#! Example 6: Performance Comparison
# --------------------------------------------------------------------

def example_performance_comparison():
    """Compare direct vs iterative solvers."""
    
    print("\n" + "=" * 70)
    print("Example 6: Direct vs Iterative Solvers")
    print("=" * 70)

    import time    
    sizes = [50, 100, 200, 500]
    
    print(f"Comparing direct vs CG for SPD systems")
    print(f"\n{'Size':<8} {'Direct(ms)':<15} {'CG(ms)':<15} {'CG iters':<12}")
    print("-" * 60)
    
    for n in sizes:
        # Create SPD system
        A = np.random.randn(n, n)
        A = A.T @ A + np.eye(n)  # Make SPD
        b = np.random.randn(n)
        
        # Direct solver
        t0 = time.time()
        solver_d        = solvers.choose_solver('direct')
        solve_d         = solver_d.get_solver_func(backend_module=np, use_matvec=False, use_fisher=False, use_matrix=True, sigma=0.0)
        result_direct   = solve_d(a=A, b=b, x0=None, tol=0.0, maxiter=None, precond_apply=None)
        time_direct     = (time.time() - t0) * 1000

        # CG solver
        t0 = time.time()
        solver_cg       = solvers.choose_solver('cg')
        solve_cg        = solver_cg.get_solver_func(backend_module=np, use_matvec=False, use_fisher=False, use_matrix=True, sigma=0.0)
        result_cg       = solve_cg(a=A, b=b, x0=None, tol=1e-8, maxiter=None, precond_apply=None)
        time_cg         = (time.time() - t0) * 1000
        
        print(f"{n:<8} {time_direct:<15.2f} {time_cg:<15.2f} {result_cg.iterations:<12}")
    
    print(f"\nObservation:")
    print(f"  - Direct: O(n^3) complexity, predictable cost")
    print(f"  - CG: O(kn²) complexity, k depends on condition number")
    print(f"  - CG faster for large, well-conditioned systems")
    print(f"  - Direct better for small systems or when exact solution needed")

# --------------------------------------------------------------------
#! Example 7: Solver Selection Guide
# --------------------------------------------------------------------

def example_solver_selection():
    """Guide for choosing between direct and iterative solvers."""
    print("\n" + "=" * 70)
    print("Example 7: Solver Selection Guide")
    print("=" * 70)
    
    scenarios = [
        ("Small SPD system (n=100)", 100, "spd", "Either (Direct faster)"),
        ("Large SPD system (n=10000)", 10000, "spd", "CG (Direct too expensive)"),
        ("Non-symmetric (n=200)", 200, "general", "Direct (CG doesn't apply)"),
        ("Rank-deficient", None, "rank-def", "Pseudo-inverse"),
        ("Underdetermined (m<n)", None, "under", "Pseudo-inverse (min norm)"),
        ("Overdetermined (m>n)", None, "over", "Pseudo-inverse (least-sq)"),
    ]
    
    print(f"\n{'Scenario':<30} {'System Size':<15} {'Recommended Solver'}")
    print("-" * 70)
    
    for scenario, size, sys_type, recommendation in scenarios:
        size_str = f"n={size}" if size else "varies"
        print(f"{scenario:<30} {size_str:<15} {recommendation}")
    
    print(f"\nGeneral Rules:")
    print(f"  1. Direct: n < 1000, exact solution needed, or general matrices")
    print(f"  2. Iterative (CG/MINRES): Large SPD/symmetric systems")
    print(f"  3. Pseudo-inverse: Rank-deficient, under/over-determined systems")
    print(f"  4. Consider memory: Direct stores LU factors (O(n^2) memory)")

# --------------------------------------------------------------------
#! Main Execution
# --------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DIRECT SOLVERS - EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_basic_direct()
    example_complex_system()
    example_rank_deficient()
    example_underdetermined()
    example_overdetermined()
    example_performance_comparison()
    example_solver_selection()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Direct solver: Exact solution via LU decomposition")
    print("2. Works for general (non-symmetric) matrices")
    print("3. Pseudo-inverse: For rank-deficient or rectangular systems")
    print("4. O(n^3) complexity - use for small to medium systems")
    print("5. No iteration - predictable performance")
    print("6. Higher memory usage than iterative methods")
    print("\nRecommendation: Use direct for n < 1000 or when exactness is critical")

# --------------------------------------------------------------------
#! End of file
# --------------------------------------------------------------------