import numpy as np
import time

def main():
    np.random.seed(42)
    N = 500
    n_omega = 100
    omegas = np.linspace(-5, 5, n_omega)
    eta = 0.01

    eigenvalues = np.sort(np.random.randn(N))
    vertex = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    V_mn_sq_full = np.abs(vertex)**2
    occupation = np.random.rand(N)

    start1 = time.perf_counter()
    chi1 = np.zeros(n_omega, dtype=complex)
    for i, omega in enumerate(omegas):
        chi_val = 0.0 + 0.0j
        for m in range(N):
            occ_diff = occupation[m] - occupation
            V_mn_sq = V_mn_sq_full[m, :]
            mask = np.abs(occ_diff) >= 1e-14
            if not np.any(mask): continue

            denom = omega + 1j * eta - (eigenvalues - eigenvalues[m])
            chi_vec = occ_diff[mask] * V_mn_sq[mask] / denom[mask]
            chi_val += np.sum(chi_vec)
        chi1[i] = chi_val
    end1 = time.perf_counter()

    start2 = time.perf_counter()
    chi2 = np.zeros(n_omega, dtype=complex)
    for m in range(N):
        occ_diff = occupation[m] - occupation
        V_mn_sq = V_mn_sq_full[m, :]
        mask = np.abs(occ_diff) >= 1e-14
        if not np.any(mask): continue

        # omegas: (n_omega, 1)
        # eigenvalues: (1, N)
        denom = omegas[:, np.newaxis] + 1j * eta - (eigenvalues - eigenvalues[m])[np.newaxis, :]
        chi_vec = occ_diff[mask] * V_mn_sq[mask] / denom[:, mask]
        chi2 += np.sum(chi_vec, axis=1)
    end2 = time.perf_counter()

    print(f"Original Loop: {end1 - start1:.4f}s")
    print(f"Vectorized Omega Loop: {end2 - start2:.4f}s")
    print(f"Max Diff: {np.max(np.abs(chi1 - chi2))}")

if __name__ == "__main__":
    main()
