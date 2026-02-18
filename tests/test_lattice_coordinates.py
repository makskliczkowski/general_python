import pytest
import numpy as np
import sys
import os

# Ensure the package can be imported
try:
    from general_python.lattices.square import SquareLattice
except ImportError:
    sys.path.append(os.getcwd())
    from lattices.square import SquareLattice


def test_lattice_coordinates_square_2d():
    Lx, Ly, Lz = 10, 10, 1
    lat = SquareLattice(lx=Lx, ly=Ly, lz=Lz, dim=2)
    lat.calculate_coordinates()

    # Check shape
    assert lat.coordinates.shape == (Lx * Ly, 3)
    assert lat.cells.shape == (Lx * Ly, 3)
    assert lat.fracs.shape == (Lx * Ly, 3)

    # Check a few specific points
    # Site 0: (0,0)
    assert np.allclose(lat.coordinates[0], [0, 0, 0])

    # Site 1: (1,0)
    assert np.allclose(lat.coordinates[1], [1, 0, 0])

    # Site Lx: (0,1)
    assert np.allclose(lat.coordinates[Lx], [0, 1, 0])

    # Site Lx*Ly-1: (Lx-1, Ly-1)
    # The last site should be (Lx-1, Ly-1, 0)
    assert np.allclose(lat.coordinates[-1], [Lx - 1, Ly - 1, 0])


def test_lattice_coordinates_square_3d():
    Lx, Ly, Lz = 4, 4, 4
    lat = SquareLattice(lx=Lx, ly=Ly, lz=Lz, dim=3)
    lat.calculate_coordinates()

    # Check shape
    assert lat.coordinates.shape == (Lx * Ly * Lz, 3)

    # Site Lx*Ly: (0,0,1)
    assert np.allclose(lat.coordinates[Lx * Ly], [0, 0, 1])


def test_lattice_coordinates_square_1d():
    Lx, Ly, Lz = 10, 1, 1
    lat = SquareLattice(lx=Lx, ly=Ly, lz=Lz, dim=1)
    lat.calculate_coordinates()

    assert lat.coordinates.shape == (Lx, 3)
    assert np.allclose(lat.coordinates[1], [1, 0, 0])

    # Check fracs
    # For 1D, ny and nz should be 0
    assert np.all(lat.fracs[:, 1] == 0)
    assert np.all(lat.fracs[:, 2] == 0)


def test_subs_and_indices():
    Lx, Ly = 5, 5
    lat = SquareLattice(lx=Lx, ly=Ly, dim=2)
    lat.calculate_coordinates()

    # Since SquareLattice has 1 basis vector, all subs should be 0
    assert np.all(lat.subs == 0)

    # Check if indices match expectation
    for i in range(lat.Ns):
        nx = i % Lx
        ny = i // Lx
        expected_coord = np.array([nx, ny, 0])
        assert np.allclose(lat.coordinates[i], expected_coord)
