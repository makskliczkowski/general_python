import time
import numpy as np
import scipy.sparse as sp
from general_python.algebra.eigen.lanczos import LanczosEigensolver
from general_python.algebra.eigen.arnoldi import ArnoldiEigensolverScipy
from .utils import create_convection_diffusion_matrix, create_1d_laplacian, create_2d_laplacian

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
        min_residual = np.min(residuals) if residuals else float('inf')

    return {
        "name": f"Lanczos Laplacian 1D (n={n}, k={k})",
        "duration": duration,
        "iterations": result.iterations,
        "min_residual": min_residual,
        "converged": result.converged
    }

def benchmark_lanczos_laplacian_2d(nx, ny, k):
    """
    Benchmark Lanczos eigensolver on a 2D Laplacian.
    """
    A = create_2d_laplacian(nx, ny)
    n = nx * ny

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
        min_residual = np.min(residuals) if residuals else float('inf')

    return {
        "name": f"Lanczos Laplacian 2D ({nx}x{ny}, k={k})",
        "duration": duration,
        "iterations": result.iterations,
        "min_residual": min_residual,
        "converged": result.converged
    }

def benchmark_arnoldi_non_symmetric(n, k):
    """
    Benchmark Arnoldi eigensolver on a non-symmetric Convection-Diffusion matrix.
    Uses SciPy wrapper (ARPACK) for production-grade performance.
    """
    A = create_convection_diffusion_matrix(n)

    start_time = time.perf_counter()

    # Arnoldi for Largest Magnitude ('LM') eigenvalues
    try:
        solver = ArnoldiEigensolverScipy(k=k, which='LM', tol=1e-10)
        # Pass A directly as it handles LinearOperator or matrix better if possible,
        # but here we use matvec to be consistent or just A.
        # ArnoldiEigensolverScipy.solve supports A or matvec.
        # Let's pass A directly to allow optimization if available, or matvec wrapper.
        # The benchmark wrapper creates a sparse matrix.
        result = solver.solve(A=A)
    except Exception as e:
        # Fallback or report error
        return {
            "name": f"Arnoldi Conv-Diff (n={n}, k={k})",
            "duration": 0.0,
            "error": str(e)
        }

    end_time = time.perf_counter()
    duration = end_time - start_time

    if result.residual_norms is not None:
        min_residual = np.min(result.residual_norms)
    else:
        # Calculate residuals manually
        residuals = []
        for i in range(k):
            val = result.eigenvalues[i]
            vec = result.eigenvectors[:, i]
            res = np.linalg.norm(A @ vec - val * vec)
            residuals.append(res)
        min_residual = np.min(residuals) if residuals else float('inf')

    return {
        "name": f"Arnoldi Conv-Diff (n={n}, k={k})",
        "duration": duration,
        "iterations": result.iterations,
        "min_residual": min_residual,
        "converged": result.converged
    }

def run_benchmarks(heavy=False):
    results = []

    # Standard benchmarks
    configs_lanczos = [
        (1000, 6),
        (2000, 10),
    ]

    configs_lanczos_2d = [
        (30, 30, 6), # 900 sites
        (50, 50, 10), # 2500 sites
    ]

    configs_arnoldi = [
        (500, 6),
        (1000, 10),
    ]

    if heavy:
        # Larger scale
        configs_lanczos.append((5000, 10))
        configs_lanczos.append((10000, 20))

        configs_lanczos_2d.append((100, 100, 10)) # 10000 sites
        configs_lanczos_2d.append((200, 200, 20)) # 40000 sites

        configs_arnoldi.append((2000, 20))
        configs_arnoldi.append((5000, 20))

    for n, k in configs_lanczos:
        res = benchmark_lanczos_laplacian(n, k)
        results.append(res)

    for nx, ny, k in configs_lanczos_2d:
        res = benchmark_lanczos_laplacian_2d(nx, ny, k)
        results.append(res)

    for n, k in configs_arnoldi:
        res = benchmark_arnoldi_non_symmetric(n, k)
        results.append(res)

    return results
