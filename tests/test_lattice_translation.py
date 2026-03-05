
import pytest
import numpy as np
from general_python.lattices.square import SquareLattice
from general_python.lattices.tools.lattice_kspace import build_translation_operators

def test_translation_unitary_2d():
    lattice = SquareLattice(lx=10, ly=10, dim=2)
    T1, T2, T3 = build_translation_operators(lattice)

    # Check unitarity
    assert np.allclose(T1 @ T1.conj().T, np.eye(lattice.Ns))
    assert np.allclose(T2 @ T2.conj().T, np.eye(lattice.Ns))
    assert np.allclose(T3, np.zeros((lattice.Ns, lattice.Ns))) # T3 should be zero for 2D?

def test_translation_unitary_1d():
    lattice = SquareLattice(lx=20, dim=1)
    T1, T2, T3 = build_translation_operators(lattice)

    assert np.allclose(T1 @ T1.conj().T, np.eye(lattice.Ns))
    # T2, T3 zero
    assert np.allclose(T2, np.zeros((lattice.Ns, lattice.Ns)))
    assert np.allclose(T3, np.zeros((lattice.Ns, lattice.Ns)))

def test_translation_unitary_3d():
    lattice = SquareLattice(lx=4, ly=4, lz=4, dim=3)
    T1, T2, T3 = build_translation_operators(lattice)

    assert np.allclose(T1 @ T1.conj().T, np.eye(lattice.Ns))
    assert np.allclose(T2 @ T2.conj().T, np.eye(lattice.Ns))
    assert np.allclose(T3 @ T3.conj().T, np.eye(lattice.Ns))

def test_translation_commutation():
    # T1 T2 should equal T2 T1
    lattice = SquareLattice(lx=5, ly=5, dim=2)
    T1, T2, T3 = build_translation_operators(lattice)

    assert np.allclose(T1 @ T2, T2 @ T1)

def test_translation_periodicity():
    # T1^Lx should be identity (if no flux)
    Lx = 5
    lattice = SquareLattice(lx=Lx, ly=3, dim=2)
    T1, T2, T3 = build_translation_operators(lattice)

    T1_pow = np.eye(lattice.Ns, dtype=complex)
    for _ in range(Lx):
        T1_pow = T1_pow @ T1

    assert np.allclose(T1_pow, np.eye(lattice.Ns))
