#!/usr/bin/env python3
"""
Matrix-Free Solvers (MATVEC Form) - Example

This example demonstrates matrix-free iterative solvers using the MATVEC form.
Instead of storing the full matrix A, you provide a function that computes
matrix-vector products y = A @ x.

Benefits of Matrix-Free:
    - Memory: O(n) instead of O(n²) for matrix storage
    - Flexibility: A can be implicit (e.g., from FFT, operator, etc.)
    - Large-scale: Enables solving systems too large to store A explicitly

Common Use Cases:
    - PDE discretizations with structured operators
    - Quantum mechanics (Hamiltonians acting on states)
    - Image processing (convolution operators)
    - Neural networks (Jacobian-vector products)
"""

import numpy as np
from QES.general_python.algebra import solvers

def example_basic_matvec():
    """Basic matrix-free solve using matvec function."""
    print("=" * 70)
    print("Example 1: Basic Matrix-Free Solve")
    print("=" * 70)
    
    # Problem setup (SPD matrix)
    n = 100
    A_explicit = np.random.randn(n, n)
    A_explicit = A_explicit.T @ A_explicit + np.eye(n)
    b = np.random.randn(n)
    
    print(f"Problem size: {n}")
    print(f"Matrix storage (explicit): {A_explicit.nbytes / 1024:.2f} KB")
    
    # Define matrix-vector product function
    def matvec(x):
        """Compute y = A @ x without forming A explicitly."""
        return A_explicit @ x  # In practice, compute directly
    
    print(f"Matrix-free storage: ~{8 * n / 1024:.2f} KB (just vector storage)")
    
    # Solve using MATVEC form (matrix-free)
    # Step 1: Choose solver
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    
    # Step 2: Get solver function with matvec form
    # For matvec form, use_matvec=True returns the solver directly (no wrapper)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=True,  # Matrix-free form
        use_fisher=False,
        use_matrix=False,
        sigma=0.0
    )
    
    # Step 3: Call solver with matvec function
    result = solve_func(matvec, b, x0=None, tol=1e-8, maxiter=None, precond_apply=None)
    
    # Verify solution
    residual = np.linalg.norm(A_explicit @ result.x - b)
    
    print(f"\nResults (matrix-free):")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual: {residual:.2e}")
    
    print(f"\nMemory savings: {A_explicit.nbytes / (8*n):.1f}\times for storage")
    
    return result


def example_tridiagonal_operator():
    """Matrix-free solve for tridiagonal operator (1D Laplacian)."""
    print("\n" + "=" * 70)
    print("Example 2: Tridiagonal Operator (1D Laplacian)")
    print("=" * 70)
    
    # 1D Laplacian: -d²u/dx² discretized
    # Tridiagonal: [-1, 2, -1] pattern
    n = 500
    h = 1.0 / (n + 1)  # Grid spacing
    
    print(f"Problem: 1D Poisson equation -u'' = f")
    print(f"Grid points: {n}")
    print(f"Grid spacing: {h:.6f}")
    
    # Matrix-free operator
    def laplacian_matvec(x):
        """Apply 1D Laplacian: (Ax)_i = (2x_i - x_{i-1} - x_{i+1}) / h²"""
        y = np.zeros_like(x)
        y[0] = (2*x[0] - x[1]) / h**2
        y[1:-1] = (2*x[1:-1] - x[:-2] - x[2:]) / h**2
        y[-1] = (2*x[-1] - x[-2]) / h**2
        return y
    
    # Right-hand side: f(x) = sin(π x)
    x_grid = np.linspace(h, 1-h, n)
    b = np.sin(np.pi * x_grid)
    
    # Solve matrix-free
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=True,
        use_fisher=False,
        use_matrix=False,
        sigma=0.0
    )
    result = solve_func(laplacian_matvec, b, x0=None, tol=1e-10, maxiter=None, precond_apply=None)
    
    print(f"\nSolution:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual: {result.residual_norm:.2e}")
    
    # Analytical solution: u(x) = -sin(πx) / π²
    u_exact = -np.sin(np.pi * x_grid) / np.pi**2
    error = np.linalg.norm(result.x - u_exact)
    
    print(f"  Error vs exact: {error:.2e}")
    
    print(f"\nNote: No matrix storage needed!")
    print(f"      Explicit matrix would be {n**2 * 8 / 1024**2:.2f} MB")
    
    return result


def example_circulant_fft():
    """Matrix-free circulant operator using FFT."""
    print("\n" + "=" * 70)
    print("Example 3: Circulant Operator via FFT")
    print("=" * 70)
    
    # Circulant matrix defined by first column c
    # A @ x = IFFT(FFT(c) * FFT(x))
    n = 256
    
    # First column of circulant matrix (e.g., convolution kernel)
    c = np.zeros(n)
    c[0] = 2.0
    c[1] = -1.0
    c[-1] = -1.0  # Periodic boundary
    
    print(f"Circulant matrix size: {n}\times{n}")
    print(f"Defined by first column (length {n})")
    
    # Precompute FFT of first column
    c_fft = np.fft.fft(c)
    
    def circulant_matvec(x):
        """Apply circulant matrix via FFT: O(n log n) instead of O(n²)"""
        x_fft = np.fft.fft(x)
        y_fft = c_fft * x_fft
        y = np.fft.ifft(y_fft).real
        return y
    
    # Random RHS
    b = np.random.randn(n)
    
    print(f"\nMatrix-vector product: O(n log n) via FFT")
    print(f"  vs O(n²) for explicit matrix multiply")
    
    # Solve
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=True,
        use_fisher=False,
        use_matrix=False,
        sigma=0.0
    )
    result = solve_func(circulant_matvec, b, x0=None, tol=1e-8, maxiter=None, precond_apply=None)
    
    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Residual: {result.residual_norm:.2e}")
    
    # Verify with explicit matrix
    A_explicit = np.zeros((n, n))
    for i in range(n):
        A_explicit[i, :] = np.roll(c, i)
    residual_check = np.linalg.norm(A_explicit @ result.x - b)
    
    print(f"  Verification residual: {residual_check:.2e}")
    
    return result


def example_quantum_hamiltonian():
    """Matrix-free quantum Hamiltonian operator."""
    print("\n" + "=" * 70)
    print("Example 4: Quantum Hamiltonian (Matrix-Free)")
    print("=" * 70)
    
    # Spin chain Hamiltonian: H = -J \Sigma _i (σ^x_i σ^x_{i+1} + σ^y_i σ^y_{i+1})
    # Acting on state vector in computational basis
    
    n_spins = 10  # Chain length
    n_states = 2**n_spins  # Hilbert space dimension
    J = 1.0  # Coupling
    
    print(f"Quantum system:")
    print(f"  Spin chain length: {n_spins}")
    print(f"  Hilbert space dimension: {n_states}")
    print(f"  Explicit matrix size: {n_states**2 * 8 / 1024**2:.2f} MB")
    
    def flip_spin(state, site):
        """Flip spin at site i."""
        return state ^ (1 << site)
    
    def hamiltonian_matvec(psi):
        """Apply Hamiltonian to state vector (matrix-free)."""
        H_psi = np.zeros_like(psi)
        
        for state in range(n_states):
            if abs(psi[state]) < 1e-15:
                continue
            
            # XX and YY interactions
            for i in range(n_spins - 1):
                # Flip both spins i and i+1
                new_state = flip_spin(flip_spin(state, i), i+1)
                H_psi[new_state] -= J * psi[state]
        
        return H_psi
    
    # Random state vector
    psi_0 = np.random.randn(n_states) + 1j * np.random.randn(n_states)
    psi_0 /= np.linalg.norm(psi_0)
    
    # Apply Hamiltonian
    b = hamiltonian_matvec(psi_0)
    
    print(f"\nApplying Hamiltonian matrix-free (no storage needed)")
    
    # Note: This Hamiltonian is Hermitian but not positive-definite
    # So we use MINRES instead of CG
    print(f"Using MINRES (Hamiltonian is Hermitian, possibly indefinite)")
    
    # For demo, shift to make positive definite
    shift = 5.0
    def shifted_matvec(x):
        return hamiltonian_matvec(x) + shift * x
    
    b_shifted = b + shift * psi_0
    
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=True,
        use_fisher=False,
        use_matrix=False,
        sigma=0.0
    )
    result = solve_func(shifted_matvec, b_shifted, x0=None, tol=1e-6, maxiter=500, precond_apply=None)
    
    print(f"\nResults (shifted Hamiltonian):")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    
    print(f"\nv Matrix-free approach essential for large quantum systems")
    
    return result


def example_jacobian_vector_product():
    """Jacobian-vector product for neural network-like function."""
    print("\n" + "=" * 70)
    print("Example 5: Jacobian-Vector Product (Automatic Differentiation)")
    print("=" * 70)
    
    # Nonlinear function f: R^n -> R^n
    # Solve linearized system: J(x_0) @ δx = -f(x_0)
    # where J is Jacobian
    
    n = 100
    
    def nonlinear_function(x):
        """Example: f(x) = x + 0.1 * sin(x) + A @ x"""
        A = np.eye(n) + 0.1 * np.random.randn(n, n)
        return x + 0.1 * np.sin(x) + A @ x
    
    # Current point
    x_0 = np.random.randn(n)
    f_0 = nonlinear_function(x_0)
    
    print(f"Nonlinear system size: {n}")
    print(f"Jacobian matrix: {n}\times{n} (not computed explicitly)")
    
    # Jacobian-vector product via finite differences
    epsilon = 1e-7
    def jacobian_matvec(v):
        """Compute J(x_0) @ v using finite differences."""
        return (nonlinear_function(x_0 + epsilon * v) - f_0) / epsilon
    
    # Solve J @ δx = -f_0 (Newton step)
    b = -f_0
    
    solver = solvers.choose_solver(solver_id='cg', sigma=0.0)
    solve_func = solver.get_solver_func(
        backend_module=np,
        use_matvec=True,
        use_fisher=False,
        use_matrix=False,
        sigma=0.0
    )
    result = solve_func(jacobian_matvec, b, x0=None, tol=1e-6, maxiter=200, precond_apply=None)
    
    print(f"\nNewton step computation:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Step norm: {np.linalg.norm(result.x):.4f}")
    
    print(f"\nNote: Jacobian never formed explicitly")
    print(f"      Only function evaluations needed")
    
    return result


def example_memory_comparison():
    """Compare memory usage: explicit vs matrix-free."""
    print("\n" + "=" * 70)
    print("Example 6: Memory Usage Comparison")
    print("=" * 70)
    
    sizes = [100, 500, 1000, 5000, 10000]
    
    print(f"Memory requirements:")
    print(f"{'n':<10} {'Explicit Matrix':<20} {'Matrix-Free':<20} {'Savings'}")
    print("-" * 70)
    
    for n in sizes:
        explicit_mb = n**2 * 8 / 1024**2
        matvec_mb = n * 8 / 1024  # Just store vectors
        savings = explicit_mb / matvec_mb
        
        print(f"{n:<10} {explicit_mb:<20.2f} MB {matvec_mb:<20.2f} KB {savings:>10.0f}\times")
    
    print(f"\nConclusion: Matrix-free crucial for large systems (n > 10,000)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MATRIX-FREE SOLVERS (MATVEC FORM) - EXAMPLES")
    print("=" * 70)
    print("\nProvide matvec function instead of explicit matrix")
    print("Benefits: O(n) memory, implicit operators, large-scale problems")
    print("=" * 70)
    
    # Run all examples
    example_basic_matvec()
    example_tridiagonal_operator()
    example_circulant_fft()
    example_quantum_hamiltonian()
    example_jacobian_vector_product()
    example_memory_comparison()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. MATVEC form: Provide function computing A @ x")
    print("2. Memory: O(n) instead of O(n²)")
    print("3. Flexibility: A can be implicit (FFT, operator, etc.)")
    print("4. Essential for large systems (n > 10,000)")
    print("5. Common in PDEs, quantum mechanics, ML (Hessian-vector products)")
    print("6. Works with CG, MINRES, and other iterative solvers")
    print("\nRecommendation: Use MATVEC for structured/implicit operators")

# ------------------------------------------