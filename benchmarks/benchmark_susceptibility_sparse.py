import numpy as np
import time
from general_python.physics.response.susceptibility import susceptibility_multi_omega, susceptibility_lehmann

def benchmark_susceptibility_sparse():
    # Setup a reasonably sized system
    N = 500
    np.random.seed(42)

    # Random Hamiltonian eigenvalues and eigenvectors
    hamiltonian_eigvals = np.sort(np.random.randn(N))
    H = np.random.randn(N, N)
    H = (H + H.T) / 2
    _, hamiltonian_eigvecs = np.linalg.eigh(H)

    # Sparse operator (realistic for physical observables)
    operator_q = np.zeros((N, N), dtype=complex)
    # Fill only a few off-diagonals and diagonals
    for i in range(N):
        operator_q[i, i] = np.random.randn()
        if i < N - 1:
            operator_q[i, i+1] = np.random.randn()
            operator_q[i+1, i] = operator_q[i, i+1]

    # Frequency grid
    omega_grid = np.linspace(-2, 2, 200)

    eta = 0.05
    temperature = 0.1

    print(f"Benchmarking susceptibility_multi_omega with N={N}, n_omega={len(omega_grid)}...")

    start_time = time.time()
    try:
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
        print(f"Current implementation took: {duration:.4f} seconds")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        duration = None

    return duration

if __name__ == "__main__":
    benchmark_susceptibility_sparse()
