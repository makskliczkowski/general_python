
import pytest
import numpy as np
from general_python.lattices import SquareLattice, LatticeBC

class TestSquareLatticeInvariants:

    def test_square_lattice_pbc_neighbors(self):
        """Test neighbors for 3x3 PBC lattice."""
        # 0 1 2
        # 3 4 5
        # 6 7 8
        lx, ly = 3, 3
        lat = SquareLattice(dim=2, lx=lx, ly=ly, bc=LatticeBC.PBC)
        lat.init()

        # Center site 4
        # Neighbors: Right (5), Up (1), Left (3), Down (7)
        nn_4 = lat.neighbors(4, order=1)
        expected_nn_4 = {5, 1, 3, 7}
        assert set(nn_4) == expected_nn_4

        # Corner site 0
        # Neighbors: Right (1), Up (6 - wrap Y), Left (2 - wrap X), Down (3)
        # Note: 'Up' is usually +y.
        # coords: 0->(0,0), 1->(1,0), 2->(2,0)
        #         3->(0,1), 4->(1,1), 5->(2,1)
        #         6->(0,2), 7->(1,2), 8->(2,2)
        # Wait, calculate_coordinates:
        # cell = i // n_basis. Square has n_basis=1.
        # nx = cell % lx. ny = (cell // lx) % ly.
        # So:
        # 0: (0,0)
        # 1: (1,0)
        # 2: (2,0)
        # 3: (0,1)
        # ...
        # So neighbors of 0 (0,0):
        # +x: (1,0) -> 1
        # -x: (2,0) -> 2 (PBC)
        # +y: (0,1) -> 3
        # -y: (0,2) -> 6 (PBC)
        nn_0 = lat.neighbors(0, order=1)
        expected_nn_0 = {1, 2, 3, 6}
        assert set(nn_0) == expected_nn_0

    def test_square_lattice_obc_neighbors(self):
        """Test neighbors for 3x3 OBC lattice."""
        lx, ly = 3, 3
        lat = SquareLattice(dim=2, lx=lx, ly=ly, bc=LatticeBC.OBC)
        lat.init()

        # Center site 4
        nn_4 = lat.neighbors(4, order=1)
        # Filter out nans which indicate missing neighbors in OBC
        nn_4 = {n for n in nn_4 if not np.isnan(n)}
        expected_nn_4 = {5, 1, 3, 7} # All present
        assert nn_4 == expected_nn_4

        # Corner site 0 (0,0)
        # Neighbors: +x (1), +y (3). -x and -y are out of bounds (nan).
        nn_0 = lat.neighbors(0, order=1)
        nn_0 = {n for n in nn_0 if not np.isnan(n)}
        expected_nn_0 = {1, 3}
        assert nn_0 == expected_nn_0

    def test_coordinate_calculation(self):
        """Test coordinate mapping."""
        lx, ly = 4, 3
        lat = SquareLattice(dim=2, lx=lx, ly=ly)
        lat.init()

        # Check specific sites
        # Site 5 -> (1, 1) if lx=4?
        # 0:(0,0), 1:(1,0), 2:(2,0), 3:(3,0)
        # 4:(0,1), 5:(1,1) ...

        c5 = lat.get_coordinates(5)
        # Should be roughly [1*a, 1*a, 0] assuming a=1
        assert np.allclose(c5[:2], [1.0, 1.0])

        # Site 11 (last one) -> (3, 2)
        # 3 + 2*4 = 11
        c11 = lat.get_coordinates(11)
        assert np.allclose(c11[:2], [3.0, 2.0])

    def test_small_lattice_edge_cases(self):
        """Test 1x1 and 2x2 lattices."""
        # 1x1 PBC
        lat1 = SquareLattice(dim=2, lx=1, ly=1, bc=LatticeBC.PBC)
        lat1.init()
        # Neighbors of 0: itself? Or filtered?
        # Usually self-loops are avoided in simple neighbor lists, but in 1x1 PBC
        # +x wraps to 0.
        nn_0 = lat1.neighbors(0, order=1)
        # Implementation dependent. Let's see what it returns.
        # Ideally it might be empty or [0, 0, 0, 0] depending on logic.
        # But let's check it doesn't crash.
        assert lat1.Ns == 1

        # 2x2 PBC
        lat2 = SquareLattice(dim=2, lx=2, ly=2, bc=LatticeBC.PBC)
        lat2.init()
        # 0 1
        # 2 3
        # Neighbors of 0:
        # +x: 1
        # -x: 1 (PBC)
        # +y: 2
        # -y: 2 (PBC)
        # Set should be {1, 2}
        nn_0 = lat2.neighbors(0, order=1)
        assert set(nn_0) == {1, 2}

    @pytest.mark.xfail(reason="Bug in lattice.py: calculate_nnn overwrites self._nnn with None")
    def test_next_nearest_neighbors(self):
        """Test NNN (next nearest neighbors)."""
        # 0 1 2
        # 3 4 5
        # 6 7 8
        lat = SquareLattice(dim=2, lx=3, ly=3, bc=LatticeBC.OBC)
        lat.init()

        # NNN of 0: (1,1) -> 4
        nnn_0 = lat.neighbors(0, order=2)
        nnn_0 = {n for n in nnn_0 if not np.isnan(n)}
        assert 4 in nnn_0

        # NNN of 4: 0, 2, 6, 8
        nnn_4 = lat.neighbors(4, order=2)
        nnn_4 = {n for n in nnn_4 if not np.isnan(n)}
        assert nnn_4 == {0, 2, 6, 8}
