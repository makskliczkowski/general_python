"""
Debug Block Lanczos implementation.
"""

import numpy as np
import sys
import os

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from QES.general_python.algebra.eigen import (
    BlockLanczosEigensolver,
    LanczosEigensolver,
    full_diagonalization
)

print("Debugging Block Lanczos")
print("="*60)

# Create small test matrix with known eigenvalues
n = 20
k = 4
np.random.seed(42)
A = np.random.randn(n, n)
A = 0.5 * (A + A.T)  # Make symmetric

print(f"Test matrix: {n}x{n} symmetric")
print(f"Finding {k} smallest eigenvalues")

# Get exact eigenvalues
result_exact = full_diagonalization(A, hermitian=True)
print(f"\nExact smallest {k} eigenvalues:")
print(result_exact.eigenvalues[:k])

# Test regular Lanczos (should work)
print(f"\nRegular Lanczos:")
solver_lanczos = LanczosEigensolver(k=k, which='smallest', max_iter=50)
result_lanczos = solver_lanczos.solve(A=A)
print(f"Eigenvalues: {result_lanczos.eigenvalues}")
error_lanczos = np.linalg.norm(result_lanczos.eigenvalues - result_exact.eigenvalues[:k])
print(f"Error: {error_lanczos:.2e}")
print(f"Converged: {result_lanczos.converged}")

# Test Block Lanczos with small block
print(f"\nBlock Lanczos (block_size=2):")
try:
    solver_block = BlockLanczosEigensolver(k=k, block_size=2, which='smallest', max_iter=20)
    result_block = solver_block.solve(A=A)
    print(f"Eigenvalues: {result_block.eigenvalues}")
    error_block = np.linalg.norm(result_block.eigenvalues - result_exact.eigenvalues[:k])
    print(f"Error: {error_block:.2e}")
    print(f"Converged: {result_block.converged}")
    print(f"Residual norms: {result_block.residual_norms}")
    
    # Check if eigenvalues are reasonable
    if error_block > 1:
        print("\n⚠️ ERROR: Block Lanczos eigenvalues diverged!")
        print(f"Expected: {result_exact.eigenvalues[:k]}")
        print(f"Got:      {result_block.eigenvalues}")
    else:
        print("\nv Block Lanczos works correctly!")
except Exception as e:
    print(f"⚠️ Block Lanczos failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
