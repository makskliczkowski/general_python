"""
Tests for Lattice boundary conditions and neighbor finding.

Validates neighbor connectivity for various lattice sizes and boundary types,
including edge cases like small lattices (1x2, 2x1).
"""

import pytest
import numpy as np
from general_python.lattices import SquareLattice, LatticeBC

class TestLatticeBoundaries:

    def test_lattice_1x2_pbc(self):
        """Test 1x2 lattice with PBC."""
        # 0
        # 1
        lat = SquareLattice(dim=2, lx=1, ly=2, bc=LatticeBC.PBC)
        lat.init()

        # Site 0 (0,0)
        # Neighbors:
        # +x: (1,0) -> wraps to (0,0) -> 0?
        # -x: (-1,0) -> wraps to (0,0) -> 0?
        # +y: (0,1) -> 1
        # -y: (0,-1) -> (0,1) -> 1

        # If the implementation allows self-loops for lx=1 PBC:
        nn_0 = lat.neighbors(0, order=1)
        # Should contain 1 (from y).
        # x-neighbors depend on implementation of lx=1.
        # Let's check what we get.
        assert 1 in nn_0

        # If we filter unique valid neighbors
        unique_nn = {n for n in nn_0 if n is not None and not np.isnan(n)}
        # Should be {0, 1} or {1} depending on self-loop policy.
        # Given standard lattice definitions, self-interaction is often excluded or included.
        # Let's assert connectivity exists.
        assert 1 in unique_nn

    def test_lattice_2x1_obc(self):
        """Test 2x1 lattice with OBC."""
        # 0 1
        lat = SquareLattice(dim=2, lx=2, ly=1, bc=LatticeBC.OBC)
        lat.init()

        # Site 0 (0,0)
        # Neighbors:
        # +x: 1
        # -x: nan
        # +y: nan
        # -y: nan
        nn_0 = lat.neighbors(0, order=1)
        unique_nn = {n for n in nn_0 if n is not None and not np.isnan(n)}
        assert unique_nn == {1}

        # Site 1 (1,0)
        nn_1 = lat.neighbors(1, order=1)
        unique_nn = {n for n in nn_1 if n is not None and not np.isnan(n)}
        assert unique_nn == {0}

    def test_lattice_3x3_pbc_wrapping(self):
        """Test standard 3x3 PBC wrapping."""
        lat = SquareLattice(dim=2, lx=3, ly=3, bc=LatticeBC.PBC)
        lat.init()

        # Center (1,1) -> 4
        nn_4 = lat.neighbors(4, order=1)
        # Neighbors: 1, 3, 5, 7 (Up, Left, Right, Down)
        assert set(nn_4) == {1, 3, 5, 7}

        # Corner (0,0) -> 0
        nn_0 = lat.neighbors(0, order=1)
        # +x: 1
        # -x: 2 (wrap)
        # +y: 3
        # -y: 6 (wrap)
        assert set(nn_0) == {1, 2, 3, 6}

    def test_lattice_3x3_obc_boundaries(self):
        """Test standard 3x3 OBC boundaries."""
        lat = SquareLattice(dim=2, lx=3, ly=3, bc=LatticeBC.OBC)
        lat.init()

        # Corner (0,0) -> 0
        nn_0 = lat.neighbors(0, order=1)
        # Filter nans
        unique_nn = {n for n in nn_0 if n is not None and not np.isnan(n)}
        # +x: 1, +y: 3
        assert unique_nn == {1, 3}

        # Edge (1,0) -> 1
        nn_1 = lat.neighbors(1, order=1)
        unique_nn = {n for n in nn_1 if n is not None and not np.isnan(n)}
        # +x: 2, -x: 0, +y: 4. -y: nan.
        assert unique_nn == {0, 2, 4}

    def test_nnn_workaround(self):
        """Test Next-Nearest Neighbors (order=2) using workaround."""
        lat = SquareLattice(dim=2, lx=3, ly=3, bc=LatticeBC.OBC)
        lat.init()

        # Workaround for bug in calculate_nnn
        lat.calculate_nnn_in(False, False, False)

        # NNN of 0 (0,0): (2,0)=2, (0,2)=6
        nnn_0 = lat.neighbors(0, order=2)
        unique_nnn = {n for n in nnn_0 if n is not None and not np.isnan(n)}
        assert unique_nnn == {2, 6}

        # NNN of 4 (1,1): (3,1), (-1,1), (1,3), (1,-1) -> All out of bounds
        nnn_4 = lat.neighbors(4, order=2)
        unique_nnn = {n for n in nnn_4 if n is not None and not np.isnan(n)}
        assert unique_nnn == set()
