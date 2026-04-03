import numpy as np
import time

def main():
    np.random.seed(42)
    N = 2000
    omega = 0.5
    eta = 0.01

    eigenvalues = np.sort(np.random.randn(N))
    eigenvectors = np.linalg.qr(np.random.randn(N, N) + 1j * np.random.randn(N, N))[0]

    # Very sparse operator
    operator = np.zeros((N, N), dtype=complex)
    for i in range(N):
        if i + 5 < N: operator[i, i+5] = 1.0
        if i - 5 >= 0: operator[i, i-5] = 1.0
    operator = operator + operator.conj().T

    # Dense operator eigenbasis
    O_eigen = eigenvectors.conj().T @ operator @ eigenvectors
    matrix_element_sq = np.abs(O_eigen)**2

    rho = np.random.rand(N)

    start1 = time.perf_counter()
    A1 = 0.0
    for m in range(N):
        rho_diff = rho[m] - rho
        mask = np.abs(rho_diff) >= 1e-14
        if not np.any(mask): continue
        delta_E = omega - (eigenvalues - eigenvalues[m])
        lorentzian = (eta / np.pi) / (delta_E**2 + eta**2)
        A_vec = rho_diff[mask] * matrix_element_sq[m, mask] * lorentzian[mask]
        A1 += np.sum(A_vec)
    end1 = time.perf_counter()

    start2 = time.perf_counter()
    A2 = 0.0
    for m in range(N):
        rho_diff = rho[m] - rho
        # Pre-filter matrix_element_sq > 1e-15
        mask = (np.abs(rho_diff) >= 1e-14) & (matrix_element_sq[m, :] > 1e-15)
        if not np.any(mask): continue
        delta_E = omega - (eigenvalues - eigenvalues[m])
        lorentzian = (eta / np.pi) / (delta_E[mask]**2 + eta**2)
        A_vec = rho_diff[mask] * matrix_element_sq[m, mask] * lorentzian
        A2 += np.sum(A_vec)
    end2 = time.perf_counter()

    # What if it's actually dense?
    # Dense operator eigenbasis
    O_eigen_dense = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    matrix_element_sq_dense = np.abs(O_eigen_dense)**2

    start3 = time.perf_counter()
    A3 = 0.0
    for m in range(N):
        rho_diff = rho[m] - rho
        mask = np.abs(rho_diff) >= 1e-14
        if not np.any(mask): continue
        delta_E = omega - (eigenvalues - eigenvalues[m])
        lorentzian = (eta / np.pi) / (delta_E**2 + eta**2)
        A_vec = rho_diff[mask] * matrix_element_sq_dense[m, mask] * lorentzian[mask]
        A3 += np.sum(A_vec)
    end3 = time.perf_counter()

    start4 = time.perf_counter()
    A4 = 0.0
    for m in range(N):
        rho_diff = rho[m] - rho
        mask = (np.abs(rho_diff) >= 1e-14) & (matrix_element_sq_dense[m, :] > 1e-15)
        if not np.any(mask): continue
        delta_E = omega - (eigenvalues - eigenvalues[m])
        lorentzian = (eta / np.pi) / (delta_E[mask]**2 + eta**2)
        A_vec = rho_diff[mask] * matrix_element_sq_dense[m, mask] * lorentzian
        A4 += np.sum(A_vec)
    end4 = time.perf_counter()

    print(f"Sparse Operator, Original: {end1 - start1:.4f}s")
    print(f"Sparse Operator, Mask Opt: {end2 - start2:.4f}s")
    print(f"Dense Operator, Original: {end3 - start3:.4f}s")
    print(f"Dense Operator, Mask Opt: {end4 - start4:.4f}s")

if __name__ == "__main__":
    main()
