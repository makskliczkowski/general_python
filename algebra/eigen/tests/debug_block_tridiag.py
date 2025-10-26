"""
Debug Block Lanczos with detailed output.
"""

import numpy as np
import sys
import os

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

print("Testing Block Tridiagonal Construction")
print("="*60)

# Simple test: 4x4 matrix, block_size=2
p = 2  # block size
n_blocks = 2  # 2 blocks

# Create simple blocks
alpha_blocks = [
    np.array([[1.0, 0.1], [0.1, 2.0]]),
    np.array([[3.0, 0.2], [0.2, 4.0]])
]

beta_blocks = [
    np.array([[0.5, 0.0], [0.0, 0.6]])
]

print(f"\nAlpha blocks (diagonal):")
for i, A in enumerate(alpha_blocks):
    print(f"  A_{i} = \n{A}")

print(f"\nBeta blocks (off-diagonal):")
for i, B in enumerate(beta_blocks):
    print(f"  B_{i} = \n{B}")

# Construct block tridiagonal
m_blocks = len(alpha_blocks)
m = m_blocks * p

T = np.zeros((m, m))

for i in range(m_blocks):
    # Diagonal block
    T[i*p:(i+1)*p, i*p:(i+1)*p] = alpha_blocks[i]
    
    # Off-diagonal blocks
    if i < m_blocks - 1:
        T[i*p:(i+1)*p, (i+1)*p:(i+2)*p] = beta_blocks[i]
        T[(i+1)*p:(i+2)*p, i*p:(i+1)*p] = beta_blocks[i].T.conj()

print(f"\nBlock Tridiagonal Matrix T:")
print(T)

print(f"\nIs T symmetric? {np.allclose(T, T.T)}")

evals = np.linalg.eigvalsh(T)
print(f"\nEigenvalues of T: {evals}")

print("\n" + "="*60)
print("\nNow testing with actual Block Lanczos iteration:")
print("="*60)

# Create simple 4x4 matrix
A = np.array([
    [4, -1, 0, 0],
    [-1, 4, -1, 0],
    [0, -1, 4, -1],
    [0, 0, -1, 4]
], dtype=float)

print(f"\nTest matrix A (4x4 tridiagonal):")
print(A)

exact_evals = np.linalg.eigvalsh(A)
print(f"\nExact eigenvalues: {exact_evals}")

# Manual Block Lanczos - 1 iteration
p = 2
V0 = np.random.randn(4, 2)
V0, _ = np.linalg.qr(V0)

print(f"\nInitial block V0 (orthonormal):")
print(V0)
print(f"V0^T V0 (should be I):\n{V0.T @ V0}")

# Apply A to V0
W = A @ V0
print(f"\nW = A @ V0:")
print(W)

# Compute A_0 = V0^T @ W
A_0 = V0.T @ W
A_0 = 0.5 * (A_0 + A_0.T)
print(f"\nA_0 = V0^T @ W (symmetrized):")
print(A_0)

# Compute W = W - V0 @ A_0
W = W - V0 @ A_0
print(f"\nW after orthogonalization:")
print(W)
print(f"V0^T @ W (should be ~0):\n{V0.T @ W}")

# QR of W
V1, B_0 = np.linalg.qr(W)
print(f"\nV1 from QR(W):")
print(V1)
print(f"\nB_0 from QR(W):")
print(B_0)

print(f"\nV0^T @ V1 (should be ~0):\n{V0.T @ V1}")
print(f"V1^T @ V1 (should be I):\n{V1.T @ V1}")

# Construct 2-block tridiagonal
T_manual = np.zeros((4, 4))
T_manual[0:2, 0:2] = A_0
T_manual[0:2, 2:4] = B_0
T_manual[2:4, 0:2] = B_0.T

# Compute A_1 for second block
W = A @ V1
A_1 = V1.T @ W
A_1 = 0.5 * (A_1 + A_1.T)
T_manual[2:4, 2:4] = A_1

print(f"\nManual block tridiagonal T:")
print(T_manual)
print(f"Is symmetric? {np.allclose(T_manual, T_manual.T)}")

evals_T = np.linalg.eigvalsh(T_manual)
print(f"\nEigenvalues of T: {evals_T}")
print(f"Expected (exact): {exact_evals}")
print(f"Error: {np.linalg.norm(evals_T - exact_evals):.2e}")
