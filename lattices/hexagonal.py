r"""
Armchair Hexagonal (Honeycomb) Lattice implementation.

This module provides the ``HexagonalLattice`` class which implements an
*armchair-oriented* honeycomb lattice.  Unlike the zig-zag
:class:`HoneycombLattice`, the primitive vectors here are chosen so that
the lattice grows **vertically and horizontally** (aligned with *x* and *y*
coordinate axes) with armchair edges.  There are no dangling bonds at the
boundary when periodic boundary conditions are used.

Geometry
--------
The unit cell contains **2 sites** (A and B sublattices).
Primitive lattice vectors (``a = 1`` by default):

.. math::
    \\mathbf{a}_1 = (\\sqrt{3}\\,a,\\; 0,\\; 0), \\qquad
    \\mathbf{a}_2 = (\\tfrac{\\sqrt{3}}{2}\\,a,\\; \\tfrac{3}{2}\\,a,\\; 0).

Basis positions inside each unit cell:

.. math::
    \\mathbf{d}_A = (0,\\; 0,\\; 0), \\qquad
    \\mathbf{d}_B = (0,\\; a,\\; 0).

Nearest neighbours (coordination z = 3 per site):

*   **A** at cell (n_x, n_y):
    → B(n_x, n_y),  B(n_x, n_y−1),  B(n_x+1, n_y−1)
*   **B** at cell (n_x, n_y):
    → A(n_x, n_y),  A(n_x, n_y+1),  A(n_x−1, n_y+1)

High-symmetry points in the Brillouin zone:
    - Γ (Gamma): Zone center (0, 0)
    - K: Corner of hexagonal BZ (2/3, 1/3)
    - K': Inequivalent corner (1/3, 2/3)
    - M: Edge midpoint (1/2, 0)

    Default path: Γ -> K -> M -> Γ

------------------------------------------------------------------------
File    : general_python/lattices/hexagonal.py
Author  : Maksymilian Kliczkowski
Date    : 2025-02-13
------------------------------------------------------------------------
"""

import numpy as np
from typing import Optional
from . import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType
from .tools.lattice_kspace import HighSymmetryPoints

# Bond-type indices (consistent with 3-coordination of the honeycomb)
X_BOND = 0  # intra-cell bond  (A <-> B within same unit cell)
Y_BOND = 1  # bond along -a2 direction
Z_BOND = 2  # bond along (a1-a2) direction

# ---------------------------------------------------------------------------

class HexagonalLattice(Lattice):
    """
    Armchair-oriented hexagonal (honeycomb) lattice up to 3 dimensions.

    The lattice is constructed so that armchair edges lie along the
    horizontal (*x*) axis, giving a rectangular bounding box aligned
    with the coordinate system.  Two sites per unit cell (A / B
    sublattices).

    Parameters
    ----------
    dim : int
        Lattice dimensionality (1, 2, or 3).
    lx, ly, lz : int
        Number of unit cells along each lattice-vector direction.
    bc : str or LatticeBC
        Boundary conditions (``'pbc'``, ``'obc'``, etc.).
    **kwargs
        Forwarded to :class:`Lattice` (e.g. ``flux``).
    """

    def __init__(self, *, dim=2, lx=3, ly=3, lz=1, bc='pbc', **kwargs):
        super().__init__(dim, lx, ly, lz, bc, **kwargs)

        self._type = LatticeType.HEXAGONAL

        # Primitive lattice vectors  (armchair orientation, a = 1)
        _a          = self.a
        _s3         = np.sqrt(3.0)

        self._a1 = np.array([_s3 * _a,          0.0,        0.0])
        self._a2 = np.array([_s3 / 2.0 * _a,    1.5 * _a,   0.0])
        self._a3 = np.array([0.0, 0.0, self.c])

        # Basis: A at origin, B shifted vertically by *a*
        self._basis = np.array([
            [0.0,  0.0, 0.0],   # A sublattice
            [0.0,  _a,  0.0],   # B sublattice
        ])

        # NN displacement vectors (from A to its three B neighbours)
        self._delta_intra = self._basis[1] - self._basis[0]                          # (0, a, 0)
        self._delta_a2    = self._basis[1] - self._basis[0] - self._a2               # (-√3/2 a, -a/2, 0)  via -a2
        self._delta_z     = self._basis[1] - self._basis[0] + self._a1 - self._a2    # (+√3/2 a, -a/2, 0)  via a1-a2

        # Adjust dimension
        match self._dim:
            case 1:
                self._ly = 1
                self._lz = 1
            case 2:
                self._lz = 1

        # Two atoms per elementary cell
        self._ns = 2 * self._lx * self._ly * self._lz

        # Use the standard init() pipeline (coordinates, kvectors, nn, nnn, ...)
        self.init(**kwargs)

    # ------------------------------------------------------------------
    #! String representation
    # ------------------------------------------------------------------

    def __str__(self):
        return (f"HEX(arm),{self.bc},d={self.dim},"
                f"Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"
                f"{self._flux_suffix}")

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------------------------------
    #! High-symmetry points
    # ------------------------------------------------------------------

    def high_symmetry_points(self) -> Optional[HighSymmetryPoints]:
        """
        Return high-symmetry points for the hexagonal BZ.
        """
        return HighSymmetryPoints.hexagonal_2d()

    def contains_special_point(self, point, *, tol: float = 1e-12) -> bool:
        """Check if a hexagonal special point is present in the current k-grid."""
        return super().contains_special_point(point, tol=tol)

    @staticmethod
    def dispersion(k, a=1.0):
        """
        Hexagonal/honeycomb (armchair) nearest-neighbour dispersion magnitude.
        Uses the three NN vectors defined in the hexagonal geometry.
        """
        k = np.asarray(k)
        s3 = np.sqrt(3.0)
        d1 = np.array([0.0, a])
        d2 = np.array([-s3 * a / 2.0, -a / 2.0])
        d3 = np.array([ s3 * a / 2.0, -a / 2.0])
        def _f(kx, ky):
            z1 = np.exp(-1j * (kx * d1[0] + ky * d1[1]))
            z2 = np.exp(-1j * (kx * d2[0] + ky * d2[1]))
            z3 = np.exp(-1j * (kx * d3[0] + ky * d3[1]))
            return np.abs(z1 + z2 + z3)
        if k.ndim == 1:
            kx, ky = k[0], k[1]
            return _f(kx, ky)
        else:
            kx = k[..., 0]
            ky = k[..., 1]
            return _f(kx, ky)

    # ------------------------------------------------------------------
    #! Geometry helpers
    # ------------------------------------------------------------------

    def get_real_vec(self, x: int, y: int, z: int):
        """
        Real-space position for stored coordinate tuple ``(x, y, z)``.

        The base class ``calculate_coordinates`` already stores proper
        vectors via ``_a1, _a2, _a3, _basis``.  This helper is kept for
        backwards compatibility and any custom coordinate look-ups.
        """
        cell_x = x
        cell_y = y // 2
        sub    = y % 2
        return cell_x * self._a1 + cell_y * self._a2 + self._basis[sub] + z * self._a3

    def get_norm(self, x: int, y: int, z: int):
        """Euclidean norm of the real-space vector."""
        v = self.get_real_vec(x, y, z)
        return np.linalg.norm(v)

    def get_nn_direction(self, site: int, direction: LatticeDirection):
        """
        Return the nearest-neighbour in the specified bond direction.

        Mapping:
            X -> intra-cell bond (A<->B within same cell)
            Y -> bond along a2
            Z -> bond along a1
        """
        mapping = {
            LatticeDirection.X: X_BOND,
            LatticeDirection.Y: Y_BOND,
            LatticeDirection.Z: Z_BOND,
        }
        idx = mapping.get(direction, -1)
        if idx < 0 or idx >= len(self._nn[site]):
            return -1
        return self._nn[site][idx]

    def bond_type(self, s1: int, s2: int) -> int:
        """Return directional bond type (X_BOND, Y_BOND, Z_BOND) or -1."""
        for bt in (X_BOND, Y_BOND, Z_BOND):
            if bt < len(self._nn[s1]) and self._nn[s1][bt] == s2:
                return bt
        return -1

    # ------------------------------------------------------------------
    #! NN calculation
    # ------------------------------------------------------------------

    def calculate_nn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculate nearest neighbours for the armchair hexagonal lattice.

        Each site has exactly 3 nearest neighbours (honeycomb coordination).

        Bond convention (for an A-site at cell (cx, cy)):
            [X_BOND] intra-cell    -> B(cx,   cy  )
            [Y_BOND] along -a2     -> B(cx,   cy-1)
            [Z_BOND] along a1 - a2 -> B(cx+1, cy-1)

        For a B-site at cell (cx, cy):
            [X_BOND] intra-cell    -> A(cx,   cy  )
            [Y_BOND] along +a2     -> A(cx,   cy+1)
            [Z_BOND] along -a1+a2  -> A(cx-1, cy+1)
        """
        Lx, Ly, Lz = self._lx, self._ly, self._lz
        Ns = self._ns

        self._nn         = [[-1, -1, -1] for _ in range(Ns)]
        self._nn_forward = [[-1, -1, -1] for _ in range(Ns)]

        def _bc(val, L, pbc):
            if pbc:
                return val % L
            return val if 0 <= val < L else -1

        def _idx(cx, cy, cz, sub):
            """Convert cell coordinates + sublattice to linear site index."""
            return ((cz * Ly + cy) * Lx + cx) * 2 + sub

        for i in range(Ns):
            sub  = i % 2        # 0 = A, 1 = B
            cell = i // 2
            cx   = cell % Lx
            cy   = (cell // Lx) % Ly
            cz   = cell // (Lx * Ly)

            if sub == 0:
                # --- A site -------------------------------------------------
                # X_BOND: intra-cell -> B in same cell (always valid)
                self._nn[i][X_BOND] = _idx(cx, cy, cz, 1)
                self._nn_forward[i][X_BOND] = self._nn[i][X_BOND]

                # Y_BOND: -> B in cell (cx, cy-1)  [along -a2]
                cy_m = _bc(cy - 1, Ly, pbcy)
                self._nn[i][Y_BOND] = _idx(cx, cy_m, cz, 1) if cy_m != -1 else -1
                self._nn_forward[i][Y_BOND] = -1  # backward for A

                # Z_BOND: -> B in cell (cx+1, cy-1)  [along a1-a2]
                cx_p = _bc(cx + 1, Lx, pbcx)
                cy_m = _bc(cy - 1, Ly, pbcy)
                self._nn[i][Z_BOND] = _idx(cx_p, cy_m, cz, 1) if (cx_p != -1 and cy_m != -1) else -1
                self._nn_forward[i][Z_BOND] = -1  # backward for A

            else:
                # --- B site -------------------------------------------------
                # X_BOND: intra-cell -> A in same cell (always valid)
                self._nn[i][X_BOND] = _idx(cx, cy, cz, 0)
                self._nn_forward[i][X_BOND] = -1  # backward for B

                # Y_BOND: -> A in cell (cx, cy+1)  [along +a2]
                cy_p = _bc(cy + 1, Ly, pbcy)
                self._nn[i][Y_BOND] = _idx(cx, cy_p, cz, 0) if cy_p != -1 else -1
                self._nn_forward[i][Y_BOND] = self._nn[i][Y_BOND]

                # Z_BOND: -> A in cell (cx-1, cy+1)  [along -a1+a2]
                cx_m = _bc(cx - 1, Lx, pbcx)
                cy_p = _bc(cy + 1, Ly, pbcy)
                self._nn[i][Z_BOND] = _idx(cx_m, cy_p, cz, 0) if (cx_m != -1 and cy_p != -1) else -1
                self._nn_forward[i][Z_BOND] = self._nn[i][Z_BOND]

        # 3D: add interlayer bonds
        if self._dim == 3 and Lz > 1:
            for i in range(Ns):
                sub  = i % 2
                cell = i // 2
                cx   = cell % Lx
                cy   = (cell // Lx) % Ly
                cz   = cell // (Lx * Ly)

                cz_p = _bc(cz + 1, Lz, pbcz)
                cz_m = _bc(cz - 1, Lz, pbcz)

                self._nn[i].append(_idx(cx, cy, cz_p, sub) if cz_p != -1 else -1)
                self._nn[i].append(_idx(cx, cy, cz_m, sub) if cz_m != -1 else -1)

        return self._nn

    # ------------------------------------------------------------------
    #! NNN calculation
    # ------------------------------------------------------------------

    def calculate_nnn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculate next-nearest neighbours for the armchair hexagonal lattice.

        NNN connect sites within the **same sublattice**.  Each site has 6
        NNN in the full 2D honeycomb; finite clusters may have fewer
        depending on boundary conditions.

        NNN displacements (same sublattice) in cell coordinates::

            +a1, -a1, +a2, -a2, +(a1-a2), -(a1-a2)
            i.e.  (+-1, 0), (0, +-1), (+-1, -+1)
        """
        Lx, Ly, Lz = self._lx, self._ly, self._lz
        Ns = self._ns

        self._nnn         = [[] for _ in range(Ns)]
        self._nnn_forward = [[] for _ in range(Ns)]

        def _bc(val, L, pbc):
            if pbc:
                return val % L
            return val if 0 <= val < L else -1

        def _idx(cx, cy, cz, sub):
            return ((cz * Ly + cy) * Lx + cx) * 2 + sub

        # NNN cell-offset vectors (same sublattice)
        nnn_offsets = [
            (+1,  0),
            (-1,  0),
            ( 0, +1),
            ( 0, -1),
            (+1, -1),
            (-1, +1),
        ]

        for i in range(Ns):
            sub  = i % 2
            cell = i // 2
            cx   = cell % Lx
            cy   = (cell // Lx) % Ly
            cz   = cell // (Lx * Ly)

            for dx, dy in nnn_offsets:
                nx = _bc(cx + dx, Lx, pbcx)
                ny = _bc(cy + dy, Ly, pbcy)
                if nx == -1 or ny == -1:
                    self._nnn[i].append(-1)
                else:
                    self._nnn[i].append(_idx(nx, ny, cz, sub))

            # Forward: only keep neighbours with strictly higher index
            self._nnn_forward[i] = [j for j in self._nnn[i] if j > i]

        # 3D: add interlayer NNN
        if self._dim == 3 and Lz > 1:
            for i in range(Ns):
                sub  = i % 2
                cell = i // 2
                cx   = cell % Lx
                cy   = (cell // Lx) % Ly
                cz   = cell // (Lx * Ly)

                cz_p = _bc(cz + 1, Lz, pbcz)
                cz_m = _bc(cz - 1, Lz, pbcz)
                self._nnn[i].append(_idx(cx, cy, cz_p, sub) if cz_p != -1 else -1)
                self._nnn[i].append(_idx(cx, cy, cz_m, sub) if cz_m != -1 else -1)

        return self._nnn

    # ------------------------------------------------------------------
    #! Symmetry helpers
    # ------------------------------------------------------------------

    def get_sym_pos(self, x, y, z):
        """
        Map coordinates to a position in the symmetry norm array.

        For the armchair lattice with 2 sublattices, ``y`` ranges over
        ``0 .. 2*Ly - 1`` (cell index ×2 + sublattice).
        """
        return (x + self.Lx - 1, y + 2 * self.Ly - 1, z + self.Lz - 1)

    def get_sym_pos_inv(self, x, y, z):
        """Inverse of :meth:`get_sym_pos`."""
        return (x - (self.Lx - 1), y - (2 * self.Ly - 1), z - (self.Lz - 1))

    def symmetry_checker(self, x, y, z):
        """Always returns True (placeholder for future symmetry calculations)."""
        return True

    # ------------------------------------------------------------------
    #! Plaquettes
    # ------------------------------------------------------------------

    def calculate_plaquettes(self):
        """
        Calculate hexagonal plaquettes (6-site loops) of the armchair lattice.

        Each plaquette is a list of 6 site indices forming a closed loop
        around one hexagonal face.  Only unique plaquettes are returned.
        """
        plaquettes = []
        seen       = set()

        for i in range(self._ns):
            if i % 2 != 0:
                continue  # anchor on A-sites only

            # Walk around hexagon via alternating A->B, B->A bonds
            # Path: A --(X)--> B --(Y)--> A --(Z)--> B --(X)--> A --(Y)--> B --(Z)--> A(start)
            bond_cycle = [X_BOND, Y_BOND, Z_BOND, X_BOND, Y_BOND, Z_BOND]
            loop  = [i]
            cur   = i
            valid = True

            for b in bond_cycle:
                if b >= len(self._nn[cur]) or self._nn[cur][b] == -1:
                    valid = False
                    break
                nxt = self._nn[cur][b]
                if nxt < 0:
                    valid = False
                    break
                loop.append(nxt)
                cur = nxt

            if not valid or loop[-1] != i:
                continue

            hex_sites = tuple(loop[:-1])
            key       = tuple(sorted(hex_sites))
            if key not in seen:
                seen.add(key)
                plaquettes.append(list(hex_sites))

        self._plaquettes = plaquettes
        return plaquettes

    ######################################### PLOTTING #########################################
