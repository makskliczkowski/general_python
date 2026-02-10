import pytest
import numpy as np
from general_python.lattices import SquareLattice, LatticeBC

class TestLatticeNew:

    def test_square_lattice_1d(self):
        """Test 1D Square Lattice (Chain)."""
        L = 10
        lat = SquareLattice(dim=1, lx=L, bc=LatticeBC.PBC)
        lat.init()

        assert lat.Ns == L

        # Site 0
        nn_0 = lat.neighbors(0, order=1)
        # Left (9), Right (1)
        assert set(nn_0) == {1, 9}

        # Site 5
        nn_5 = lat.neighbors(5, order=1)
        assert set(nn_5) == {4, 6}

    def test_square_lattice_3d(self):
        """Test 3D Square Lattice (Cubic)."""
        L = 3
        lat = SquareLattice(dim=3, lx=L, ly=L, lz=L, bc=LatticeBC.OBC)
        lat.init()

        assert lat.Ns == 27

        # Center site (1,1,1) -> 1*1 + 1*3 + 1*9 = 13 if indexing is x + y*Lx + z*Lx*Ly
        # Let's verify indexing:
        # site_index = z * Lx * Ly + y * Lx + x
        # 1 * 9 + 1 * 3 + 1 = 9 + 3 + 1 = 13. Correct.

        # Coordinates: 0,1,2 for x,y,z
        # Site 13 is at (1,1,1)
        # Neighbors: +/- x, +/- y, +/- z
        # All valid in 3x3x3 OBC (bounds are [0,2])
        nn = lat.neighbors(13, order=1)
        nn = [n for n in nn if not np.isnan(n)]
        assert len(nn) == 6

        # Corner (0,0,0) -> 0
        nn_0 = lat.neighbors(0, order=1)
        nn_0 = [n for n in nn_0 if not np.isnan(n)]
        # Neighbors: +x, +y, +z (3 valid). -x,-y,-z are out.
        assert len(nn_0) == 3

    def test_get_coordinates_shapes(self):
        """Verify coordinate shapes."""
        lat = SquareLattice(dim=2, lx=4, ly=3)
        lat.init()

        coords = lat.coordinates
        assert coords.shape == (12, 3) # Always 3D coords even for 2D

        # Check specific coordinate
        # Site 5 -> (1, 1) if indexing x + y*Lx
        # 1 + 1*4 = 5.
        # z should be 0
        assert np.isclose(coords[5, 2], 0.0)

    def test_mbc_boundary(self):
        """Test Mixed Boundary Conditions (Cylinder: PBC x, OBC y)."""
        # 3x3
        # y=2: 6 7 8
        # y=1: 3 4 5
        # y=0: 0 1 2
        #      x=0 x=1 x=2

        lat = SquareLattice(dim=2, lx=3, ly=3, bc=LatticeBC.MBC)
        lat.init()

        # Site 0 (0,0)
        # Neighbors:
        # +x: 1
        # -x: 2 (PBC wrapping)
        # +y: 3
        # -y: nan (OBC)
        nn_0 = lat.neighbors(0, order=1)
        # Filter nan and invalid/None
        valid_nn = {n for n in nn_0 if n is not None and not np.isnan(n)}

        assert 2 in valid_nn # Check x-wrapping
        assert 1 in valid_nn
        assert 3 in valid_nn
        assert len(valid_nn) == 3

        # Site 6 (0,2) (Top-left)
        # +y: nan (OBC)
        nn_6 = lat.neighbors(6, order=1)
        valid_nn6 = {n for n in nn_6 if n is not None and not np.isnan(n)}
        # +x: 7
        # -x: 8 (PBC)
        # -y: 3
        assert valid_nn6 == {7, 8, 3}
