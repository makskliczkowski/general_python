import time
import numpy as np
import scipy.sparse as sp
from general_python.algebra.eigen.lanczos import LanczosEigensolver

def create_1d_laplacian(n):
    """Create a 1D Laplacian matrix using scipy.sparse."""
    # 2 on diagonal, -1 on off-diagonals
    diags = [np.ones(n)*2, -np.ones(n-1), -np.ones(n-1)]
    return sp.diags(diags, [0, 1, -1], format='csr')

def benchmark_lanczos_laplacian(n, k):
    """
    Benchmark Lanczos eigensolver on a 1D Laplacian.
    """
    A = create_1d_laplacian(n)

    start_time = time.perf_counter()

    solver = LanczosEigensolver(k=k, which='smallest', tol=1e-10)

    # matvec wrapper for sparse matrix
    def matvec(x):
        return A @ x

    result = solver.solve(matvec=matvec, n=n)

    end_time = time.perf_counter()
    duration = end_time - start_time

    if result.residual_norms is not None:
        min_residual = np.min(result.residual_norms)
    else:
        # Calculate residuals manually: ||Av - lambda v||
        residuals = []
        for i in range(k):
            val = result.eigenvalues[i]
            vec = result.eigenvectors[:, i]
            res = np.linalg.norm(A @ vec - val * vec)
            residuals.append(res)
        min_residual = np.min(residuals)

    return {
        "name": f"Lanczos Laplacian 1D (n={n}, k={k})",
        "duration": duration,
        "iterations": result.iterations,
        "min_residual": min_residual,
        "converged": result.converged
    }

def run_benchmarks(heavy=False):
    results = []

    # Standard benchmarks
    configs = [
        (1000, 6),
        (2000, 10),
    ]

    if heavy:
        # Larger scale
        configs.append((5000, 10))
        configs.append((10000, 20))

    for n, k in configs:
        res = benchmark_lanczos_laplacian(n, k)
        results.append(res)

    return results
