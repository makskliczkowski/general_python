import numpy as np
import time
from physics.response.susceptibility import susceptibility_multi_omega

def benchmark_susceptibility():
    # Setup a reasonably sized system
    N = 100
    np.random.seed(42)

    # Random Hamiltonian eigenvalues and eigenvectors
    hamiltonian_eigvals = np.sort(np.random.randn(N))
    H = np.random.randn(N, N)
    H = (H + H.T) / 2
    _, hamiltonian_eigvecs = np.linalg.eigh(H)

    # Random operator
    operator_q = np.random.randn(N, N) + 1j * np.random.randn(N, N)

    # Frequency grid
    omega_grid = np.linspace(-2, 2, 200)

    eta = 0.05
    temperature = 0.1

    print(f"Benchmarking susceptibility_multi_omega with N={N}, n_omega={len(omega_grid)}...")

    start_time = time.time()
    chi = susceptibility_multi_omega(
        hamiltonian_eigvals,
        hamiltonian_eigvecs,
        operator_q,
        omega_grid,
        eta=eta,
        temperature=temperature
    )
    end_time = time.time()

    duration = end_time - start_time
    print(f"Original implementation took: {duration:.4f} seconds")

    return duration, chi

if __name__ == "__main__":
    benchmark_susceptibility()
