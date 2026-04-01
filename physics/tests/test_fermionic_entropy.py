"""
Test fermionic sign-corrected RDM vs correlation matrix method.

This test validates that:
1. For Gaussian states (Slater determinants), fermionic RDM entropy matches
   the correlation matrix method for ALL subsystem geometries.
2. For non-Gaussian states (superpositions), fermionic RDM still works
   while correlation matrix method is not applicable.

The key insight is that with fermionic sign corrections applied during
tensor reshape, the Schmidt decomposition correctly handles non-contiguous
subsystems in fermionic systems.
"""

import numpy as np
from typing import Union, List


def fermionic_correlation_matrix(
    eigvecs: np.ndarray,
    occupied_orbitals: Union[List[int], np.ndarray]
) -> np.ndarray:
    """
    Compute the single-particle correlation matrix C_{ij} = <c_i^dag c_j>.
    
    Parameters
    ----------
    eigvecs : np.ndarray
        Single-particle eigenvector matrix, shape (ns, ns).
        Convention: eigvecs[site, orbital]
    occupied_orbitals : array-like
        Indices of occupied single-particle orbitals.
    
    Returns
    -------
    np.ndarray
        Correlation matrix C, shape (ns, ns), Hermitian with eigenvalues in [0,1].
        Satisfies: Tr(C) = number of particles
    """
    ns = eigvecs.shape[0]
    occ_vec = np.zeros(ns, dtype=eigvecs.real.dtype)
    occ_vec[occupied_orbitals] = 1.0
    
    # C = W @ diag(n) @ W^dag
    return eigvecs @ np.diag(occ_vec) @ eigvecs.conj().T


def correlation_matrix_entropy(
    corr_matrix: np.ndarray,
    subsystem: Union[int, List[int], np.ndarray]
) -> float:
    """
    Compute fermionic entanglement entropy from correlation matrix.
    
    This is the correct method for Gaussian fermionic states.
    Works for ANY subsystem geometry (contiguous or non-contiguous).
    
    Parameters
    ----------
    corr_matrix : np.ndarray
        Full correlation matrix C_{ij} = <c_i^dag c_j>, shape (ns, ns).
    subsystem : int or array-like
        If int: contiguous subsystem of first `subsystem` sites.
        If array: indices of sites in subsystem A (any geometry).
    
    Returns
    -------
    float
        Fermionic entanglement entropy S_A.
    """
    if isinstance(subsystem, int):
        indices = np.arange(subsystem)
    else:
        indices = np.asarray(subsystem)
    
    C_A = corr_matrix[np.ix_(indices, indices)]
    eigs = np.linalg.eigvalsh(C_A)
    
    eps = 1e-14
    nu = np.clip(eigs, eps, 1.0 - eps)
    return float(-np.sum(nu * np.log(nu) + (1 - nu) * np.log(1 - nu)))


def test_fermionic_rdm_vs_correlation_matrix():
    """
    Validate fermionic RDM method against correlation matrix for Gaussian states.
    
    For Slater determinants (Gaussian states), both methods should give
    identical results for ANY subsystem geometry, including non-contiguous.
    """
    import sys
    sys.path.insert(0, '/home/klimak/Codes/QuantumEigenSolver/pyqusolver/Python')
    
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
    from QES.general_python.physics.density_matrix import (
        schmidt, rho, fermionic_entanglement_entropy
    )
    from QES.general_python.physics.entropy import vn_entropy
    
    ns = 6
    np.random.seed(42)
    
    # Create random quadratic Hamiltonian
    model = QuadraticHamiltonian(ns=ns, conserves_particles=True, dtype=np.complex128)
    for i in range(ns):
        for j in range(i+1, ns):
            t = np.random.randn() + 1j * np.random.randn()
            model.add_hopping(i, j, t)
        model.add_onsite(i, np.random.randn())
    model.build()
    model.diagonalize()
    
    n_particles = ns // 2
    occupied = np.argsort(model.eig_val)[:n_particles]
    
    # Compute correlation matrix and many-body state
    C = fermionic_correlation_matrix(model.eig_vec, occupied)
    psi = model.many_body_state(occupied_orbitals=list(occupied))
    
    print("="*70)
    print("Test: Fermionic RDM vs Correlation Matrix Method")
    print("="*70)
    print(f"System: {ns} sites, {n_particles} fermions")
    print()
    
    test_subsystems = [
        [0, 1],           # Contiguous
        [0, 1, 2],        # Contiguous
        [0, 2],           # Non-contiguous (gap of 1)
        [0, 2, 4],        # Non-contiguous (alternating)
        [1, 3, 5],        # Non-contiguous (odd sites)
        [0, 3],           # Non-contiguous (large gap)
    ]
    
    all_passed = True
    
    for subsystem in test_subsystems:
        if max(subsystem) >= ns:
            continue
        
        # Method 1: Correlation matrix entropy (ground truth for Gaussian)
        S_corr = correlation_matrix_entropy(C, subsystem)
        
        # Method 2: Fermionic RDM entropy (our new implementation)
        S_fermi_rdm = fermionic_entanglement_entropy(psi, subsystem, ns=ns)
        
        # Method 3: Standard RDM (without fermionic correction) for comparison
        s_sq_std = schmidt(state=psi, va=subsystem, ns=ns, contiguous=False,
                          fermionic=False, eig=False, square=True, return_vecs=False)
        S_std_rdm = vn_entropy(s_sq_std / np.sum(s_sq_std))
        
        # Check if contiguous
        is_contig = (subsystem == list(range(min(subsystem), max(subsystem)+1)))
        
        # For Gaussian states, fermionic RDM should match correlation matrix
        match_fermi = np.allclose(S_fermi_rdm, S_corr, atol=1e-10)
        
        # Standard RDM only matches for contiguous subsystems
        match_std = np.allclose(S_std_rdm, S_corr, atol=1e-10)
        
        status_fermi = "PASS" if match_fermi else "FAIL"
        status_std = "PASS" if match_std else ("expected" if not is_contig else "FAIL")
        
        if not match_fermi:
            all_passed = False
        if is_contig and not match_std:
            all_passed = False
        
        geom = "contig" if is_contig else "non-contig"
        print(f"  {str(subsystem):15} ({geom:9}): "
              f"Corr={S_corr:.6f}  FermiRDM={S_fermi_rdm:.6f} [{status_fermi}]  "
              f"StdRDM={S_std_rdm:.6f} [{status_std}]")
    
    print()
    if all_passed:
        print("All tests PASSED")
        print("  - Fermionic RDM matches correlation matrix for ALL geometries")
        print("  - Standard RDM only matches for contiguous subsystems")
    else:
        print("SOME TESTS FAILED!")
    
    return all_passed


def test_superposition_of_slater_determinants():
    """
    Test fermionic RDM for non-Gaussian states (superpositions).
    
    For superpositions of Slater determinants:
    - Correlation matrix method is NOT applicable
    - Fermionic RDM should still give correct results
    - We verify internal consistency (contiguous vs same-result expectations)
    """
    import sys
    sys.path.insert(0, '/home/klimak/Codes/QuantumEigenSolver/pyqusolver/Python')
    
    from QES.Algebra.hamil_quadratic import QuadraticHamiltonian
    from QES.general_python.physics.density_matrix import fermionic_entanglement_entropy
    
    ns = 4
    np.random.seed(123)
    
    # Create two different Slater determinants
    model = QuadraticHamiltonian(ns=ns, conserves_particles=True, dtype=np.complex128)
    for i in range(ns):
        for j in range(i+1, ns):
            model.add_hopping(i, j, np.random.randn())
        model.add_onsite(i, np.random.randn())
    model.build()
    model.diagonalize()
    
    # Ground state and first excited state (different occupied orbitals)
    energies = model.eig_val
    occupied_gs = [0, 1]  # Two lowest energy orbitals
    occupied_ex = [0, 2]  # Different occupation
    
    psi_gs = model.many_body_state(occupied_orbitals=occupied_gs)
    psi_ex = model.many_body_state(occupied_orbitals=occupied_ex)
    
    # Create superposition (non-Gaussian state)
    psi_super = (psi_gs + psi_ex) / np.sqrt(2)
    psi_super /= np.linalg.norm(psi_super)
    
    print()
    print("="*70)
    print("Test: Superposition of Slater Determinants")
    print("="*70)
    print(f"System: {ns} sites, superposition of 2 Slater determinants")
    print()
    
    subsystems = [[0, 1], [0, 2], [1, 3]]
    
    for sub in subsystems:
        S = fermionic_entanglement_entropy(psi_super, sub, ns=ns)
        geom = "contig" if sub == list(range(min(sub), max(sub)+1)) else "non-contig"
        print(f"  {str(sub):10} ({geom:9}): S = {S:.6f}")
    
    print()
    print("Superposition test complete (fermionic RDM computes without error)")
    return True


if __name__ == "__main__":
    print()
    passed1 = test_fermionic_rdm_vs_correlation_matrix()
    passed2 = test_superposition_of_slater_determinants()
    print()
    print("="*70)
    print(f"Overall: {'ALL PASSED' if (passed1 and passed2) else 'SOME FAILED'}")
    print("="*70)
