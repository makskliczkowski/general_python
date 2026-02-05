import time
import numpy as np
from .utils import create_spin_chain_hamiltonian, create_sparse_spin_chain_hamiltonian

def benchmark_hamiltonian(n_spins):
    """
    Benchmark building a Hamiltonian and applying it (Dense).
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

def benchmark_hamiltonian_sparse(n_spins):
    """
    Benchmark building a Hamiltonian and applying it (Sparse).
    """
    # 1. Construction
    start_time = time.perf_counter()
    H = create_sparse_spin_chain_hamiltonian(n_spins)
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
            "name": f"Hamiltonian Build Sparse (Spin Chain n={n_spins}, dim={n_states})",
            "duration": build_time,
            "n_spins": n_spins,
            "dim": n_states
        },
        {
            "name": f"Hamiltonian Apply Sparse (Spin Chain n={n_spins}, dim={n_states})",
            "duration": op_time,
            "n_spins": n_spins,
            "dim": n_states
        }
    ]

def run_benchmarks(heavy=False):
    results = []

    # Standard benchmarks
    # Small dense
    for n in [8, 10]:
        results.extend(benchmark_hamiltonian(n))

    # Larger sparse
    for n in [12, 14]: # 14 is 16k states
        results.extend(benchmark_hamiltonian_sparse(n))

    if heavy:
        # 16 -> 65k states
        # 18 -> 262k states
        configs_heavy = [16, 18]
        for n in configs_heavy:
            results.extend(benchmark_hamiltonian_sparse(n))

    return results
