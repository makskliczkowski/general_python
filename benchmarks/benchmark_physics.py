import time
import numpy as np
from .utils import create_spin_chain_hamiltonian

def benchmark_hamiltonian(n_spins):
    """
    Benchmark building a Hamiltonian and applying it.
    """
    # 1. Construction
    start_time = time.perf_counter()
    H = create_spin_chain_hamiltonian(n_spins)
    build_time = time.perf_counter() - start_time

    n_states = 2**n_spins

    # 2. Application (Matrix-Vector product)
    # Create random vector
    v = np.random.rand(n_states)
    v /= np.linalg.norm(v)

    start_time = time.perf_counter()
    Hv = H @ v
    op_time = time.perf_counter() - start_time

    return [
        {
            "name": f"Hamiltonian Build (Spin Chain n={n_spins}, dim={n_states})",
            "duration": build_time,
            "n_spins": n_spins,
            "dim": n_states
        },
        {
            "name": f"Hamiltonian Apply (Spin Chain n={n_spins}, dim={n_states})",
            "duration": op_time,
            "n_spins": n_spins,
            "dim": n_states
        }
    ]

def run_benchmarks(heavy=False):
    results = []

    # Standard benchmarks
    configs = [8, 10, 12] # 12 -> 4096 states

    if heavy:
        configs.append(13) # 8192 states

    for n in configs:
        res = benchmark_hamiltonian(n)
        results.extend(res)

    return results
