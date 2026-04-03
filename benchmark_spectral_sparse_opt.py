import numpy as np
import time
from general_python.physics.spectral.spectral_backend import thermal_weights

def operator_spectral_function_lehmann_opt(
        omega, eigenvalues, eigenvectors, operator,
        eta=0.01, temperature=0.0, backend="numpy"
):
    be = np
    # Use standard numpy
    be = np

    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    eigenvectors = be.asarray(eigenvectors, dtype=be.complex128)
    operator = be.asarray(operator, dtype=be.complex128)

    if operator.ndim == 0:
        operator = be.eye(len(eigenvalues), dtype=be.complex128) * operator
    elif operator.ndim == 1:
        operator = be.diag(operator)

    N = len(eigenvalues)
    O_eigen = eigenvectors.conj().T @ operator @ eigenvectors
    rho = thermal_weights(eigenvalues, temperature, backend)

    def lorentzian(delta_E):
        return (eta / np.pi) / (delta_E**2 + eta**2)

    A = 0.0
    for m in range(N):
        rho_diff = rho[m] - rho

        matrix_element_sq = be.abs(O_eigen[m, :])**2

        # ⚡ Bolt: Optimization - Filter out negligible matrix elements
        mask = (be.abs(rho_diff) >= 1e-14) & (matrix_element_sq > 1e-15)

        if not be.any(mask):
            continue

        delta_E = omega - (eigenvalues - eigenvalues[m])

        A_vec = rho_diff[mask] * matrix_element_sq[mask] * lorentzian(delta_E[mask])
        A += be.sum(A_vec)

    return float(be.real(A))

def susceptibility_bubble_opt(
        omega, eigenvalues, vertex, occupation=None,
        eta=0.01, backend="numpy"
):
    be = np

    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    omega_complex = be.asarray(omega, dtype=be.complex128)
    eta_complex = be.asarray(eta, dtype=be.complex128)

    N = len(eigenvalues)

    if vertex is None:
        vertex = be.eye(N, dtype=be.complex128)
    else:
        vertex = be.asarray(vertex, dtype=be.complex128)

    if occupation is None:
        occupation = be.where(eigenvalues < 0, 1.0, 0.0)
    else:
        occupation = be.asarray(occupation, dtype=be.float64)

    chi = 0.0 + 0.0j
    for m in range(N):
        occ_diff = occupation[m] - occupation
        V_mn_sq = be.abs(vertex[m, :])**2

        # ⚡ Bolt: Optimization - Filter out negligible matrix elements
        mask = (be.abs(occ_diff) >= 1e-14) & (V_mn_sq > 1e-15)

        if not be.any(mask):
            continue

        denom = omega_complex + 1j * eta_complex - (eigenvalues - eigenvalues[m])

        chi_vec = occ_diff[mask] * V_mn_sq[mask] / denom[mask]
        chi += be.sum(chi_vec)

    return complex(chi)

def main():
    np.random.seed(42)
    N = 1000
    omega = 0.5
    eta = 0.01
    temperature = 1.0

    eigenvalues = np.sort(np.random.randn(N))
    eigenvectors = np.linalg.qr(np.random.randn(N, N) + 1j * np.random.randn(N, N))[0]

    # Sparse operator
    operator = np.zeros((N, N), dtype=complex)
    for i in range(N):
        if i + 1 < N: operator[i, i+1] = 1.0
        if i - 1 >= 0: operator[i, i-1] = 1.0
        operator[i, i] = 0.5

    operator = operator + operator.conj().T

    # Benchmark Lehmann Opt
    start = time.perf_counter()
    res1 = operator_spectral_function_lehmann_opt(omega, eigenvalues, eigenvectors, operator, eta, temperature, backend="numpy")
    end = time.perf_counter()
    print(f"operator_spectral_function_lehmann_opt: {end - start:.4f}s, result: {res1}")

    # Sparse vertex Opt
    vertex = operator.copy()
    occupation = np.random.rand(N)
    start = time.perf_counter()
    res2 = susceptibility_bubble_opt(omega, eigenvalues, vertex, occupation, eta, backend="numpy")
    end = time.perf_counter()
    print(f"susceptibility_bubble_opt: {end - start:.4f}s, result: {res2}")

if __name__ == "__main__":
    main()
