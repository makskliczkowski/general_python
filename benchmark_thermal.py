import time
import numpy as np
from general_python.physics.thermal import thermal_scan

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

    print(f"Time taken: {end - start:.4f} seconds")
    return results

if __name__ == '__main__':
    run_benchmark()
