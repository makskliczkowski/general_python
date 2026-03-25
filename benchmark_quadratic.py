import numpy as np
import time
from general_python.physics.spectral.quadratic.spectral_backend_quadratic import greens_function_quadratic, greens_function_quadratic_finite_T


def run_benchmark():
    np.random.seed(42)
    N = 500
    omega = 0.5
    eta = 0.01
    beta = 1.0

    eigenvalues = np.sort(np.random.randn(N))
    operator_a = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    operator_b = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    occ = np.random.randint(0, 2, size=N)

    start = time.time()
    res1 = greens_function_quadratic(
        omega,
        eigenvalues,
        operator_a=operator_a,
        operator_b=operator_b,
        occupations=occ,
        eta=eta,
        backend="numpy",
        basis_transform=False,
    )
    end = time.time()
    print(f"greens_function_quadratic: {end - start:.4f}s")

    start = time.time()
    res2 = greens_function_quadratic_finite_T(
        omega,
        eigenvalues,
        operator_a=operator_a,
        operator_b=operator_b,
        beta=beta,
        eta=eta,
        backend="numpy",
        basis_transform=False,
    )
    end = time.time()
    print(f"greens_function_quadratic_finite_T: {end - start:.4f}s")


if __name__ == '__main__':
    run_benchmark()
