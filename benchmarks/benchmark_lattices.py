import time
from general_python.lattices.square import SquareLattice
from general_python.lattices import LatticeBC

def benchmark_lattice_init(lx, ly, bc_name):
    """
    Benchmark initialization of SquareLattice (which computes neighbors).
    """
    bc = LatticeBC.PBC if bc_name == 'PBC' else LatticeBC.OBC

    start_time = time.perf_counter()

    # SquareLattice init calls calculate_nn and calculate_nnn
    # Note: SquareLattice allocates a full DFT matrix (Ns x Ns) in init,
    # limiting the maximum scalable size to ~100x100 (10000 sites -> 1.6GB) on standard machines.
    lat = SquareLattice(lx=lx, ly=ly, dim=2, bc=bc)

    end_time = time.perf_counter()
    duration = end_time - start_time

    return {
        "name": f"Lattice Init 2D ({bc_name}, {lx}x{ly}, N={lx*ly})",
        "duration": duration,
        "N": lx*ly,
        "BC": bc_name
    }

def run_benchmarks(heavy=False):
    results = []

    # Standard benchmarks
    # Kept small due to dense DFT allocation in SquareLattice
    configs = [
        (30, 30, 'PBC'), # 900 sites
        (50, 50, 'PBC'), # 2500 sites
        (50, 50, 'OBC'),
    ]

    if heavy:
        configs.append((80, 80, 'PBC')) # 6400 sites -> ~600MB DFT matrix
        configs.append((100, 100, 'PBC')) # 10000 sites -> ~1.6GB DFT matrix

    for lx, ly, bc in configs:
        res = benchmark_lattice_init(lx, ly, bc)
        results.append(res)

    return results
