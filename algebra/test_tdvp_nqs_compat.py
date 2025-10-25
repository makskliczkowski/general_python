"""
TDVP/NQS Solver Compatibility Tests

Validates that the refactored solver infrastructure maintains compatibility
with TDVP and NQS usage patterns. Tests the solver factory, SolverResult,
and different solver forms (GRAM, MATRIX, MATVEC).
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'QES'))

from QES.general_python.algebra import solvers
from QES.general_python.algebra.preconditioners import choose_precond, PreconditionersTypeSym

print("=" * 70)
print("TDVP/NQS Solver Compatibility Tests")
print("=" * 70)

def test_solver_factory():
    """Test that choose_solver works with different identifiers."""
    print("\n[1/7] Testing solver factory (choose_solver)...")
    
    # Test with SolverType enum
    solver_cg = solvers.choose_solver(solver_id=solvers.SolverType.CG, sigma=0.01)
    assert solver_cg is not None, "CG solver creation failed"
    assert hasattr(solver_cg, 'solve'), "CG solver missing solve method"
    
    # Test with string
    solver_minres = solvers.choose_solver(solver_id='minres', sigma=0.01)
    assert solver_minres is not None, "MINRES solver creation failed"
    
    # Test with int (SolverType.DIRECT.value = 1)
    solver_direct = solvers.choose_solver(solver_id=solvers.SolverType.DIRECT.value, sigma=0.0)
    assert solver_direct is not None, "Direct solver creation failed"
    
    print("  v choose_solver works with SolverType, string, and int")

def test_solver_result():
    """Test SolverResult structure used by TDVP."""
    print("\n[2/7] Testing SolverResult structure...")
    
    x_solution = np.array([1.0, 2.0, 3.0])
    result = solvers.SolverResult(
        x=x_solution,
        converged=True,
        iterations=5,
        residual_norm=1e-10
    )
    
    assert hasattr(result, 'x'), "SolverResult missing 'x' attribute"
    assert hasattr(result, 'converged'), "SolverResult missing 'converged' attribute"
    assert hasattr(result, 'iterations'), "SolverResult missing 'iterations' attribute"
    assert hasattr(result, 'residual_norm'), "SolverResult missing 'residual_norm' attribute"
    assert np.allclose(result.x, x_solution), "SolverResult.x not preserved"
    
    print("  v SolverResult has all required attributes")

def test_solver_form_enum():
    """Test SolverForm enum used by TDVP."""
    print("\n[3/7] Testing SolverForm enum...")
    
    assert hasattr(solvers, 'SolverForm'), "SolverForm enum not found"
    assert hasattr(solvers.SolverForm, 'GRAM'), "SolverForm.GRAM not found"
    assert hasattr(solvers.SolverForm, 'MATRIX'), "SolverForm.MATRIX not found"
    assert hasattr(solvers.SolverForm, 'MATVEC'), "SolverForm.MATVEC not found"
    
    # TDVP accesses .value
    assert hasattr(solvers.SolverForm.GRAM, 'value'), "SolverForm.GRAM missing value"
    
    print("  v SolverForm enum has GRAM, MATRIX, MATVEC")

def test_gram_form_solve():
    """Test solver with GRAM form (S, Sp) - most common in TDVP."""
    print("\n[4/7] Testing GRAM form solver (S, Sp)...")
    
    # Create test problem: A = Sp @ S / N
    n_samples = 100
    n_params = 20
    
    S = np.random.randn(n_samples, n_params) + 1j * np.random.randn(n_samples, n_params)
    Sp = S.conj()
    b = np.random.randn(n_params) + 1j * np.random.randn(n_params)
    
    # Use CG solver
    solver_cg = solvers.choose_solver(solver_id='cg', sigma=0.01)
    
    # Test get_solver_func (static interface)
    solve_func = solver_cg.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=True,
        use_matrix=False,
        sigma=0.01
    )
    
    result = solve_func(
        s=S,
        s_p=Sp,
        b=b,
        x0=np.zeros(n_params, dtype=complex),
        tol=1e-8,
        maxiter=100,
        precond_apply=None
    )
    
    assert isinstance(result, solvers.SolverResult), "Result not a SolverResult"
    assert result.x is not None, "Solution is None"
    assert len(result.x) == n_params, "Solution dimension mismatch"
    
    # Verify solution quality
    A = Sp @ S / n_samples + 0.01 * np.eye(n_params)
    residual = np.linalg.norm(A @ result.x - b)
    assert residual < 1e-5, f"Residual too large: {residual:.2e}"
    
    print(f"  v GRAM form solve: converged={result.converged}, iterations={result.iterations}, residual={residual:.2e}")

def test_matrix_form_solve():
    """Test solver with MATRIX form - used when matrix is explicitly formed."""
    print("\n[5/7] Testing MATRIX form solver...")
    
    n = 50
    A = np.random.randn(n, n)
    A = A + A.T + 10 * np.eye(n)  # Make SPD
    b = np.random.randn(n)
    
    solver_cg = solvers.choose_solver(solver_id='cg', sigma=0.1)
    
    solve_func = solver_cg.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.1
    )
    
    result = solve_func(
        a=A,
        b=b,
        x0=np.zeros(n),
        tol=1e-10,
        maxiter=100,
        precond_apply=None
    )
    
    A_reg = A + 0.1 * np.eye(n)
    residual = np.linalg.norm(A_reg @ result.x - b)
    assert residual < 1e-7, f"Residual too large: {residual:.2e}"
    
    print(f"  v MATRIX form solve: converged={result.converged}, residual={residual:.2e}")

def test_matvec_form_solve():
    """Test solver with MATVEC form - most efficient for large systems."""
    print("\n[6/7] Testing MATVEC form solver...")
    
    n = 30
    A = np.random.randn(n, n)
    A = A + A.T + 5 * np.eye(n)
    b = np.random.randn(n)
    
    def matvec(v):
        return A @ v
    
    solver_minres = solvers.choose_solver(solver_id='scipy_minres', sigma=0.05)
    
    solve_func = solver_minres.get_solver_func(
        backend_module=np,
        use_matvec=True,
        use_fisher=False,
        use_matrix=False,
        sigma=0.05
    )
    
    result = solve_func(
        matvec=matvec,
        b=b,
        x0=np.zeros(n),
        tol=1e-10,
        maxiter=100,
        precond_apply=None
    )
    
    A_reg = A + 0.05 * np.eye(n)
    residual = np.linalg.norm(A_reg @ result.x - b)
    assert residual < 1e-7, f"Residual too large: {residual:.2e}"
    
    print(f"  v MATVEC form solve: converged={result.converged}, residual={residual:.2e}")

def test_preconditioner_integration():
    """Test that preconditioners work with solvers (used in TDVP)."""
    print("\n[7/7] Testing preconditioner integration...")
    
    n = 40
    A = np.random.randn(n, n)
    A = A + A.T + 20 * np.eye(n)
    b = np.random.randn(n)
    
    # Create Jacobi preconditioner
    precond = choose_precond('jacobi', backend='numpy')
    precond.set(A, sigma=0.1)
    precond_apply = precond.get_apply()
    
    # Solve with preconditioner
    solver_cg = solvers.choose_solver(solver_id='cg', sigma=0.1)
    
    solve_func = solver_cg.get_solver_func(
        backend_module=np,
        use_matvec=False,
        use_fisher=False,
        use_matrix=True,
        sigma=0.1
    )
    
    result = solve_func(
        a=A,
        b=b,
        x0=np.zeros(n),
        tol=1e-10,
        maxiter=100,
        precond_apply=precond_apply
    )
    
    A_reg = A + 0.1 * np.eye(n)
    residual = np.linalg.norm(A_reg @ result.x - b)
    assert residual < 1e-7, f"Residual too large: {residual:.2e}"
    
    print(f"  v Preconditioned solve: converged={result.converged}, iterations={result.iterations}, residual={residual:.2e}")

def main():
    """Run all compatibility tests."""
    try:
        test_solver_factory()
        test_solver_result()
        test_solver_form_enum()
        test_gram_form_solve()
        test_matrix_form_solve()
        test_matvec_form_solve()
        test_preconditioner_integration()
        
        print("\n" + "=" * 70)
        print("✅ ALL TDVP/NQS COMPATIBILITY TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  • Solver factory (choose_solver) works correctly")
        print("  • SolverResult structure preserved")
        print("  • SolverForm enum (GRAM/MATRIX/MATVEC) available")
        print("  • All solver forms produce correct solutions")
        print("  • Preconditioner integration functional")
        print("\nv TDVP and NQS usage patterns remain unchanged\n")
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
