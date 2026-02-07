
import pytest
import numpy as np
from general_python.lattices import SquareLattice, LatticeBC

class TestSquareLatticeComprehensive:

    def test_1d_chain_pbc(self):
        """Test 1D chain with PBC."""
        L = 5
        lat = SquareLattice(dim=1, lx=L, bc=LatticeBC.PBC)
        lat.init()

        assert lat.Ns == L

        # Check neighbors
        # 0 -> 1 (right), 4 (left)
        nn_0 = lat.neighbors(0, order=1)
        assert set(nn_0) == {1, 4}

        # 4 -> 0 (right), 3 (left)
        nn_4 = lat.neighbors(4, order=1)
        assert set(nn_4) == {0, 3}

    def test_1d_chain_obc(self):
        """Test 1D chain with OBC."""
        L = 5
        lat = SquareLattice(dim=1, lx=L, bc=LatticeBC.OBC)
        lat.init()

        # 0 -> 1 (right). Left is boundary.
        nn_0 = lat.neighbors(0, order=1)
        # Filter nans/invalid
        nn_0 = {n for n in nn_0 if n is not None and not np.isnan(n) and n >= 0}
        assert nn_0 == {1}

        # 4 -> 3 (left). Right is boundary.
        nn_4 = lat.neighbors(4, order=1)
        nn_4 = {n for n in nn_4 if n is not None and not np.isnan(n) and n >= 0}
        assert nn_4 == {3}

    def test_3d_cubic_pbc(self):
        """Test 3D cubic lattice with PBC."""
        L = 3
        lat = SquareLattice(dim=3, lx=L, ly=L, lz=L, bc=LatticeBC.PBC)
        lat.init()

        assert lat.Ns == L*L*L # 27

        # Site 13 (center) -> (1, 1, 1)
        # Neighbors: +/- x, +/- y, +/- z
        # (2,1,1), (0,1,1), (1,2,1), (1,0,1), (1,1,2), (1,1,0)
        c13 = lat.get_coordinates(13)
        assert np.allclose(c13, [1, 1, 1])

        nn_13 = lat.neighbors(13, order=1)
        assert len(nn_13) == 6

        # Check specific neighbor indices
        # (2,1,1) -> 1*9 + 1*3 + 2 = 14
        # (0,1,1) -> 1*9 + 1*3 + 0 = 12
        # (1,2,1) -> 1*9 + 2*3 + 1 = 16
        # (1,0,1) -> 1*9 + 0*3 + 1 = 10
        # (1,1,2) -> 2*9 + 1*3 + 1 = 22
        # (1,1,0) -> 0*9 + 1*3 + 1 = 4
        expected = {14, 12, 16, 10, 22, 4}
        assert set(nn_13) == expected

    def test_broken_lattices_are_stubbed(self):
        """
        Document that Hexagonal/Triangular lattices are currently stubs or broken.
        This test serves as a marker for future implementation.
        """
        from general_python.lattices import HexagonalLattice, TriangularLattice

        # TriangularLattice is abstract (missing methods)
        with pytest.raises(TypeError):
            lat_tri = TriangularLattice(dim=2, lx=3, ly=3)

        # HexagonalLattice fails instantiation due to missing _cells calculation
        with pytest.raises(Exception): # ValueError or AttributeError
            lat_hex = HexagonalLattice(dim=2, lx=3, ly=3, lz=1, bc=LatticeBC.PBC)
