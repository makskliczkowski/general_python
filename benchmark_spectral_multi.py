import numpy as np
import time

def main():
    np.random.seed(42)
    N = 500
    n_omega = 100
    omegas = np.linspace(-5, 5, n_omega)
    eta = 0.01

    eigenvalues = np.sort(np.random.randn(N))
    O_eigen = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    matrix_element_sq = np.abs(O_eigen)**2
    rho = np.random.rand(N)

    # Original multi_omega (calls lehmann for each omega)
    start1 = time.perf_counter()
    A1 = np.zeros(n_omega)
    for i, omega in enumerate(omegas):
        A_val = 0.0
        for m in range(N):
            rho_diff = rho[m] - rho
            mask = np.abs(rho_diff) >= 1e-14
            if not np.any(mask): continue
            delta_E = omega - (eigenvalues - eigenvalues[m])
            lorentzian = (eta / np.pi) / (delta_E**2 + eta**2)
            A_vec = rho_diff[mask] * matrix_element_sq[m, mask] * lorentzian[mask]
            A_val += np.sum(A_vec)
        A1[i] = A_val
    end1 = time.perf_counter()

    # Vectorized over omega (Loop over N, but vectorize across omegas)
    start2 = time.perf_counter()
    A2 = np.zeros(n_omega)
    # rho_diff: (N,)
    # matrix_element_sq[m]: (N,)
    for m in range(N):
        rho_diff = rho[m] - rho
        mask = np.abs(rho_diff) >= 1e-14
        if not np.any(mask): continue

        # We need to broadcast over omegas
        # omega: (n_omega, 1)
        # eigenvalues: (1, N)
        # delta_E: (n_omega, N)
        delta_E = omegas[:, np.newaxis] - (eigenvalues - eigenvalues[m])[np.newaxis, :]
        lorentzian = (eta / np.pi) / (delta_E**2 + eta**2)

        # A_vec: (n_omega, N)
        # Sum over N (axis=1)
        A_vec = rho_diff[mask] * matrix_element_sq[m, mask] * lorentzian[:, mask]
        A2 += np.sum(A_vec, axis=1)
    end2 = time.perf_counter()

    print(f"Original Loop: {end1 - start1:.4f}s")
    print(f"Vectorized Omega Loop: {end2 - start2:.4f}s")
    print(f"Max Diff: {np.max(np.abs(A1 - A2))}")

if __name__ == "__main__":
    main()
