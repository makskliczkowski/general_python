import time
import numpy as np
from physics.thermal import thermal_scan

def thermal_scan_opt(energies, temperatures, observables=None):
    temperatures = np.asarray(temperatures)
    energies = np.asarray(energies)

    n_temps = len(temperatures)
    results = {
        'T'     : temperatures,
        'beta'  : 1.0 / temperatures,
        'F'     : np.zeros(n_temps),
        'U'     : np.zeros(n_temps),
        'S'     : np.zeros(n_temps),
        'C_V'   : np.zeros(n_temps),
    }

    # Precompute squared arrays
    energies_sq = energies ** 2
    E_min = np.min(energies)
    energies_shifted = energies - E_min

    obs_arrays = {}
    if observables is not None:
        for name, obs_diag in observables.items():
            obs_diag = np.asarray(obs_diag)
            obs_arrays[name] = (obs_diag, obs_diag ** 2)
            results[f'{name}_avg'] = np.zeros(n_temps)
            results[f'{name}_chi'] = np.zeros(n_temps)

    # Compute for each temperature
    for i, T in enumerate(temperatures):
        beta = 1.0 / T

        exp_factors = np.exp(-beta * energies_shifted)
        Z = np.sum(exp_factors)

        if Z > 0:
            probs = exp_factors / Z
            F = -np.log(Z) / beta + E_min
            U = np.sum(energies * probs)
            S = beta * (U - F)
            U2 = np.sum(energies_sq * probs)
            C_V = (beta ** 2) * (U2 - U ** 2)
        else:
            probs = np.zeros_like(exp_factors)
            F = np.inf
            U = 0.0
            S = 0.0
            C_V = 0.0

        results['F'][i] = F
        results['U'][i] = U
        results['S'][i] = S
        results['C_V'][i] = C_V

        if observables is not None:
            for name, (obs, obs_sq) in obs_arrays.items():
                if Z > 0:
                    avg = np.sum(obs * probs)
                    avg2 = np.sum(obs_sq * probs)
                    chi = beta * (avg2 - avg ** 2)
                else:
                    avg = 0.0
                    chi = 0.0

                results[f'{name}_avg'][i] = avg
                results[f'{name}_chi'][i] = chi

    return results

def run_benchmark():
    energies = np.linspace(-10, 10, 10000)
    temperatures = np.linspace(0.1, 10.0, 500)
    observables = {
        'M': np.random.randn(10000),
        'N': np.random.rand(10000)
    }

    start = time.perf_counter()
    results = thermal_scan(energies, temperatures, observables)
    end = time.perf_counter()
    print(f"Original Time taken: {end - start:.4f} seconds")

    start = time.perf_counter()
    results_opt = thermal_scan_opt(energies, temperatures, observables)
    end = time.perf_counter()
    print(f"Optimized Time taken: {end - start:.4f} seconds")

    # Verify correctness
    for key in results:
        np.testing.assert_allclose(results[key], results_opt[key], rtol=1e-5, err_msg=f"Mismatch in {key}")

if __name__ == '__main__':
    run_benchmark()
