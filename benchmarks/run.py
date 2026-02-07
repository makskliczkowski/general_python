"""
Minimal benchmark runner for general_python.
Run with: python3 -m benchmarks.run [--heavy]
"""
import argparse
import sys
import os
import time

# Ensure we can import from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks import benchmark_algebra, benchmark_lattices, benchmark_physics

def main():
    parser = argparse.ArgumentParser(description="Run general_python benchmarks.")
    parser.add_argument("--heavy", action="store_true", help="Run heavy benchmarks (longer duration).")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Running general_python Benchmarks (Heavy={args.heavy})")
    print("=" * 80)

    all_results = []

    modules = [benchmark_algebra, benchmark_lattices, benchmark_physics]

    start_total = time.perf_counter()

    for mod in modules:
        try:
            print(f"Running {mod.__name__}...")
            results = mod.run_benchmarks(heavy=args.heavy)
            all_results.extend(results)
        except Exception as e:
            print(f"Error running benchmarks in {mod.__name__}: {e}")
            import traceback
            traceback.print_exc()

    end_total = time.perf_counter()

    print("\n" + "=" * 80)
    print(f"{'Benchmark Name':<50} | {'Time (s)':<10} | {'Details'}")
    print("-" * 80)

    for res in all_results:
        name = res['name']
        duration = res['duration']
        # Filter out details that are too long or redundant
        details = ", ".join([f"{k}={v}" for k, v in res.items() if k not in ['name', 'duration'] and not isinstance(v, (list, dict))])
        print(f"{name:<50} | {duration:<10.4f} | {details}")

    print("-" * 80)
    print(f"Total time: {end_total - start_total:.2f} s")
    print("=" * 80)

if __name__ == "__main__":
    main()
