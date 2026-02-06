
import pytest
import numpy as np
from general_python.lattices import SquareLattice, LatticeBC

class TestLatticeDimensions:

    def test_square_lattice_1d_pbc(self):
        """Test 1D Square Lattice with PBC."""
        lx = 5
        lat = SquareLattice(dim=1, lx=lx, bc=LatticeBC.PBC)
        lat.init()

        assert lat.Ns == 5
        # Neighbors of 0: 1 (right), 4 (left due to PBC)
        nn_0 = lat.neighbors(0, order=1)
        assert set(nn_0) == {1, 4}

        # Neighbors of 4: 0 (right due to PBC), 3 (left)
        nn_4 = lat.neighbors(4, order=1)
        assert set(nn_4) == {0, 3}

        # Coords
        # 0 -> (0,0,0)
        # 4 -> (4,0,0)
        c0 = lat.get_coordinates(0)
        c4 = lat.get_coordinates(4)
        assert np.allclose(c0, [0, 0, 0])
        assert np.allclose(c4, [4, 0, 0])

    def test_square_lattice_1d_obc(self):
        """Test 1D Square Lattice with OBC."""
        lx = 5
        lat = SquareLattice(dim=1, lx=lx, bc=LatticeBC.OBC)
        lat.init()

        # Neighbors of 0: 1 (right). Left is boundary.
        nn_0 = lat.neighbors(0, order=1)
        nn_0 = {n for n in nn_0 if not np.isnan(n) and n is not None and n >= 0} # Filter invalid
        assert nn_0 == {1}

        # Neighbors of 4: 3 (left). Right is boundary.
        nn_4 = lat.neighbors(4, order=1)
        nn_4 = {n for n in nn_4 if not np.isnan(n) and n is not None and n >= 0}
        assert nn_4 == {3}

    def test_square_lattice_3d_pbc(self):
        """Test 3D Square Lattice with PBC."""
        lx, ly, lz = 2, 2, 2
        lat = SquareLattice(dim=3, lx=lx, ly=ly, lz=lz, bc=LatticeBC.PBC)
        lat.init()

        assert lat.Ns == 8

        # Site 0 at (0,0,0)
        # Neighbors:
        # +x: (1,0,0) -> 1
        # -x: (1,0,0) -> 1 (PBC 2-1=1)
        # +y: (0,1,0) -> 2 (since stride x is 2)
        # -y: (0,1,0) -> 2
        # +z: (0,0,1) -> 4 (since stride xy is 4)
        # -z: (0,0,1) -> 4
        # Set should be {1, 2, 4}
        nn_0 = lat.neighbors(0, order=1)
        assert set(nn_0) == {1, 2, 4}

        # Site 7 at (1,1,1)
        # +x: (0,1,1) -> 6
        # +y: (1,0,1) -> 5
        # +z: (1,1,0) -> 3
        nn_7 = lat.neighbors(7, order=1)
        assert set(nn_7) == {6, 5, 3}

    def test_square_lattice_3d_obc(self):
        """Test 3D Square Lattice with OBC."""
        lx, ly, lz = 3, 3, 3
        lat = SquareLattice(dim=3, lx=lx, ly=ly, lz=lz, bc=LatticeBC.OBC)
        lat.init()

        # Center site 13 (1,1,1)
        # 1 + 1*3 + 1*9 = 13. Correct.
        # Neighbors:
        # +x: 14 (2,1,1)
        # -x: 12 (0,1,1)
        # +y: 16 (1,2,1)
        # -y: 10 (1,0,1)
        # +z: 22 (1,1,2)
        # -z: 4  (1,1,0)
        nn_13 = lat.neighbors(13, order=1)
        nn_13 = {n for n in nn_13 if not np.isnan(n) and n is not None and n >= 0}
        assert nn_13 == {14, 12, 16, 10, 22, 4}

        # Corner 0 (0,0,0)
        # +x: 1, +y: 3, +z: 9. Others out.
        nn_0 = lat.neighbors(0, order=1)
        nn_0 = {n for n in nn_0 if not np.isnan(n) and n is not None and n >= 0}
        assert nn_0 == {1, 3, 9}
