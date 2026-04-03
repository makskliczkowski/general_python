import numpy as np
import time

def main():
    np.random.seed(42)
    N = 2000
    omega = 0.5
    eta = 0.01

    eigenvalues = np.sort(np.random.randn(N))

    # Dense operator eigenbasis
    O_eigen = np.random.randn(N, N) + 1j * np.random.randn(N, N)
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

    # Fully vectorized version using broadcasting
    # shape: (N, N)
    rho_diff = rho[:, np.newaxis] - rho[np.newaxis, :]
    delta_E = omega - (eigenvalues[np.newaxis, :] - eigenvalues[:, np.newaxis])
    lorentzian = (eta / np.pi) / (delta_E**2 + eta**2)

    # Only keep terms where np.abs(rho_diff) >= 1e-14
    mask = np.abs(rho_diff) >= 1e-14

    # Since matrix_element_sq is dense, we might want to filter small ones too?
    # mask = mask & (matrix_element_sq > 1e-15)

    A2 = np.sum(rho_diff[mask] * matrix_element_sq[mask] * lorentzian[mask])
    end2 = time.perf_counter()

    print(f"Original Loop: {end1 - start1:.4f}s, Res: {A1}")
    print(f"Vectorized Broadcast: {end2 - start2:.4f}s, Res: {A2}")

if __name__ == "__main__":
    main()
