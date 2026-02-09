import pytest
import numpy as np
from general_python.lattices import SquareLattice, LatticeBC

class TestLatticeBoundaries:

    def test_1x1_lattice(self):
        """Test degenerate 1x1 lattice."""
        # 1x1 OBC
        lat = SquareLattice(dim=2, lx=1, ly=1, bc=LatticeBC.OBC)
        lat.init()
        assert lat.Ns == 1

        # Site 0 neighbors in OBC should be empty (or none, depending on impl)
        # Neighbors: +x (invalid), -x (invalid), +y (invalid), -y (invalid)
        nn = lat.neighbors(0, order=1)
        valid_nn = [n for n in nn if not np.isnan(n)]
        assert len(valid_nn) == 0

        # 1x1 PBC
        # In PBC, neighbor of 0 is 0?
        # Typically x+1 mod 1 = 0.
        # But usually we don't include self as neighbor unless specified.
        # Let's check implementation behavior.
        lat_pbc = SquareLattice(dim=2, lx=1, ly=1, bc=LatticeBC.PBC)
        lat_pbc.init()

        nn_pbc = lat_pbc.neighbors(0, order=1)
        # Should contain 0 four times? Or handle self-interaction?
        # If it returns [0, 0, 0, 0], that's technically correct for PBC formula.
        assert len(nn_pbc) == 4
        assert all(n == 0 for n in nn_pbc)

    def test_2x1_lattice(self):
        """Test 2x1 lattice (dimer)."""
        # 0 -- 1
        lat = SquareLattice(dim=2, lx=2, ly=1, bc=LatticeBC.OBC)
        lat.init()

        assert lat.Ns == 2

        # Site 0
        # +x -> 1. -x -> None. +y -> None. -y -> None.
        nn0 = lat.neighbors(0, order=1)
        valid_nn0 = {n for n in nn0 if not np.isnan(n)}
        assert valid_nn0 == {1}

        # Site 1
        # +x -> None. -x -> 0.
        nn1 = lat.neighbors(1, order=1)
        valid_nn1 = {n for n in nn1 if not np.isnan(n)}
        assert valid_nn1 == {0}

    def test_2x2_pbc(self):
        """Test 2x2 lattice with PBC."""
        # 2 3
        # 0 1
        lat = SquareLattice(dim=2, lx=2, ly=2, bc=LatticeBC.PBC)
        lat.init()

        # Site 0 (0,0)
        # +x (1,0) -> 1
        # -x (-1,0) -> (1,0) -> 1
        # +y (0,1) -> 2
        # -y (0,-1) -> (0,1) -> 2

        nn0 = lat.neighbors(0, order=1)
        # Convert to list of ints for comparison, handling numpy types
        nn0_list = sorted([int(x) for x in nn0])
        assert nn0_list == [1, 1, 2, 2]

    def test_coordinate_consistency(self):
        """Verify coordinates match neighbor relations."""
        lat = SquareLattice(dim=2, lx=3, ly=3, bc=LatticeBC.OBC)
        lat.init()

        coords = lat.coordinates
        # Check distance between neighbors is 1
        for i in range(lat.Ns):
            nn = lat.neighbors(i, order=1)
            valid_nn = [n for n in nn if not np.isnan(n)]

            p1 = coords[i]
            for n_idx in valid_nn:
                p2 = coords[int(n_idx)]
                dist = np.linalg.norm(p1 - p2)
                assert np.isclose(dist, 1.0)
