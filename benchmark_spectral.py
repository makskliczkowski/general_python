import numpy as np
import time
from general_python.physics.spectral.spectral_backend import operator_spectral_function_lehmann, susceptibility_bubble

def main():
    np.random.seed(42)
    N = 500
    omega = 0.5
    eta = 0.01
    temperature = 1.0

    eigenvalues = np.sort(np.random.randn(N))
    eigenvectors = np.linalg.qr(np.random.randn(N, N) + 1j * np.random.randn(N, N))[0]
    operator = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    operator = operator + operator.conj().T

    # Benchmark Lehmann
    start = time.perf_counter()
    res1 = operator_spectral_function_lehmann(omega, eigenvalues, eigenvectors, operator, eta, temperature, backend="numpy")
    end = time.perf_counter()
    print(f"operator_spectral_function_lehmann: {end - start:.4f}s, result: {res1}")

    # Benchmark Bubble
    vertex = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    occupation = np.random.rand(N)
    start = time.perf_counter()
    res2 = susceptibility_bubble(omega, eigenvalues, vertex, occupation, eta, backend="numpy")
    end = time.perf_counter()
    print(f"susceptibility_bubble: {end - start:.4f}s, result: {res2}")

if __name__ == "__main__":
    main()
