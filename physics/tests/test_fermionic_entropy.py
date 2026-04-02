"""
Test fermionic RDM with sign corrections for non-contiguous subsystems.

For fermionic systems, computing the reduced density matrix of a non-contiguous
subsystem requires sign corrections due to fermionic anticommutation relations.

This test validates that:
1. Fermionic RDM matches correlation matrix entropy for Gaussian states
2. Works for ALL subsystem geometries (contiguous and non-contiguous)
3. Produces valid density matrices (Hermitian, positive, trace=1)
4. Satisfies S(A) = S(complement) for pure states
"""

import  os
import  sys
import  numpy as np
from    pathlib import Path

pyqes_path = Path(os.getenv("QES_PYPATH", "/home/klimak/Codes/QuantumEigenSolver/pyqusolver/Python")).expanduser().resolve()
if not pyqes_path.exists():
    raise FileNotFoundError(f"QES_PYPATH directory not found: {pyqes_path}")
sys.path.insert(0, str(pyqes_path))

from    QES.general_python.physics.density_matrix       import rho
from    QES.general_python.physics.entropy              import sp_correlation_entropy, vn_entropy

# =============================================================================
#! Helper Functions
# =============================================================================

def random_quadratic_hamiltonian(ns: int, seed: int = 42) -> np.ndarray:
    """Create random Hermitian single-particle Hamiltonian."""
    np.random.seed(seed)
    H = np.random.randn(ns, ns) + 1j * np.random.randn(ns, ns)
    return (H + H.conj().T) / 2


def slater_determinant(eigvecs: np.ndarray, occupied: list) -> np.ndarray:
    """
    Build many-body state from occupied single-particle orbitals.
    
    For N fermions in orbitals {k1, k2, ...}, the coefficient of basis state
    |n_0, n_1, ...> is the determinant of the submatrix eigvecs[sites, occupied].
    """
    ns          = eigvecs.shape[0]
    n_particles = len(occupied)
    psi         = np.zeros(2**ns, dtype=np.complex128)
    
    for idx in range(2**ns):
        if bin(idx).count('1') != n_particles:
            continue
        sites       = [i for i in range(ns) if (idx >> i) & 1]
        psi[idx]    = np.linalg.det(eigvecs[np.ix_(sites, occupied)])
    
    return psi / np.linalg.norm(psi)


def correlation_entropy(eigvecs: np.ndarray, occupied: list, subsystem: list) -> float:
    """
    Compute entropy from single-particle correlation matrix.
    
    The correlation matrix C has eigenvalues nu in [0,1] (occupation numbers).
    sp_correlation_entropy expects eigenvalues lambda in [-1,1] and maps via p = 0.5*(1+lambda).
    So we convert: lambda = 2*nu - 1.
    
    Formula: S = -sum_k [nu*log(nu) + (1-nu)*log(1-nu)]
    """
    ns  = eigvecs.shape[0]
    
    # Build full correlation matrix C = W @ diag(occ) @ W^dag
    occ             = np.zeros(ns)
    occ[occupied]   = 1.0
    C               = eigvecs @ np.diag(occ) @ eigvecs.conj().T
    
    # Extract subsystem correlation matrix
    C_A             = C[np.ix_(subsystem, subsystem)]
    
    # Eigenvalues nu in [0,1], convert to lambda in [-1,1] for sp_correlation_entropy
    nu              = np.linalg.eigvalsh(C_A).real
    lam             = 2.0 * nu - 1.0
    return sp_correlation_entropy(lam, q=1.0, base=None)


def rdm_entropy(psi: np.ndarray, subsystem: list, ns: int) -> float:
    """
    Compute entropy from reduced density matrix with fermionic corrections.
    Uses rho() -> eigvalsh -> vn_entropy pipeline.
    """
    rho_A       = rho(psi, va=subsystem, ns=ns, fermionic=True)
    eigvals     = np.linalg.eigvalsh(rho_A).real
    return vn_entropy(eigvals, base=None)


def is_contiguous(subsystem: list) -> bool:
    """Check if subsystem sites are contiguous."""
    s = sorted(subsystem)
    return s == list(range(min(s), max(s) + 1))


# =============================================================================
#! Tests
# =============================================================================

def test_fermionic_vs_correlation_matrix():
    """
    Test: Fermionic RDM matches correlation matrix for all subsystem types.
    This is the main validation that sign corrections work correctly.
    """
    print("Test 1: Fermionic RDM vs Correlation Matrix")
    print("-" * 50)
    
    ns, n_particles = 6, 3
    H               = random_quadratic_hamiltonian(ns, seed=42)
    eigvals, eigvecs = np.linalg.eigh(H)
    occupied        = list(range(n_particles))
    psi             = slater_determinant(eigvecs, occupied)
    
    subsystems = [
        [0, 1],        # contiguous
        [0, 1, 2],     # contiguous
        [0, 2],        # non-contiguous
        [0, 2, 4],     # non-contiguous (alternating)
        [1, 3, 5],     # non-contiguous (odd sites)
    ]
    
    all_pass = True
    for sub in subsystems:
        S_corr  = correlation_entropy(eigvecs, occupied, sub)
        S_fermi = rdm_entropy(psi, sub, ns)
        
        match   = np.isclose(S_fermi, S_corr, atol=1e-10)
        all_pass &= match
        
        status  = "PASS" if match else "FAIL"
        contig  = "contig" if is_contiguous(sub) else "non-contig"
        print(f"  {str(sub):12} ({contig:10}): {status}")
    
    return all_pass

def test_rdm_properties():
    """
    Test: RDM is Hermitian, positive semi-definite, trace = 1.
    """
    print("\nTest 2: RDM Properties")
    print("-" * 50)
    
    ns, n_particles = 5, 2
    H               = random_quadratic_hamiltonian(ns, seed=123)
    _, eigvecs      = np.linalg.eigh(H)
    psi             = slater_determinant(eigvecs, list(range(n_particles)))
    
    subsystem       = [0, 2, 4]  # non-contiguous
    rho_A           = rho(psi, va=subsystem, ns=ns, fermionic=True)
    
    is_hermitian    = np.allclose(rho_A, rho_A.conj().T)
    eigvals         = np.linalg.eigvalsh(rho_A)
    is_positive     = np.all(eigvals >= -1e-12)
    trace_one       = np.isclose(np.trace(rho_A).real, 1.0)
    
    print(f"  Hermitian:    {is_hermitian}")
    print(f"  Positive:     {is_positive}")
    print(f"  Trace = 1:    {trace_one}")
    
    return is_hermitian and is_positive and trace_one

def test_entropy_symmetry():
    """
    Test: S(A) = S(complement of A) for pure states.
    """
    print("\nTest 3: Entropy Symmetry S(A) = S(B)")
    print("-" * 50)
    
    ns, n_particles = 6, 3
    H               = random_quadratic_hamiltonian(ns, seed=456)
    _, eigvecs      = np.linalg.eigh(H)
    psi             = slater_determinant(eigvecs, list(range(n_particles)))
    
    test_cases      = [[0, 2], [0, 2, 4], [1, 3, 5]]
    
    all_pass = True
    for sub in test_cases:
        complement  = [i for i in range(ns) if i not in sub]
        
        S_A         = rdm_entropy(psi, sub, ns)
        S_B         = rdm_entropy(psi, complement, ns)
        
        match       = np.isclose(S_A, S_B, atol=1e-10)
        all_pass    &= match
        
        status      = "PASS" if match else "FAIL"
        print(f"  A={sub}, B={complement}: {status}")
    
    return all_pass

def test_contiguous_no_correction_needed():
    """
    Test: Subsystems starting from site 0 need no permutation, so no correction.
    Only [0], [0,1], [0,1,2], etc. have trivial permutation order.
    """
    print("\nTest 4: Contiguous from Site 0 (no permutation needed)")
    print("-" * 50)
    
    ns = 5
    np.random.seed(789)
    psi = np.random.randn(2**ns) + 1j * np.random.randn(2**ns)
    psi /= np.linalg.norm(psi)
    
    # Only subsystems starting from 0 have trivial order
    subsystems = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    
    all_pass = True
    for sub in subsystems:
        rho_std     = rho(psi, va=sub, ns=ns, fermionic=False)
        rho_fermi   = rho(psi, va=sub, ns=ns, fermionic=True)
        
        match       = np.allclose(rho_std, rho_fermi)
        all_pass    &= match
        
        status      = "PASS" if match else "FAIL"
        print(f"  {str(sub):12}: std == fermi? {status}")
    
    return all_pass

# =============================================================================
#! Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Fermionic RDM Sign Correction Tests")
    print("=" * 50)
    
    results = [
        test_fermionic_vs_correlation_matrix(),
        test_rdm_properties(),
        test_entropy_symmetry(),
        test_contiguous_no_correction_needed(),
    ]
    
    print("\n" + "=" * 50)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 50)

# =============================================================================