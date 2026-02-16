"""
Region handler for lattice geometries.

This module provides the :class:`LatticeRegionHandler` class, which
encapsulates methods for defining and extracting spatial regions on a
lattice.  It supports various region types including:

- **Half-system cuts** — bipartitions along x / y / z axes.
- **Disk / circular regions** — sites within a Euclidean radius (PBC-aware).
- **Graph-distance balls** — BFS expansion up to depth *d*.
- **Sublattice selections** — all sites of a given sublattice label.
- **Plaquette-based regions** — union of plaquette site sets.
- **Kitaev-Preskill (KP) sectors** — pie-slice angular sectors meeting at a
  point, used for extracting the topological entanglement entropy (TEE).
- **Levin-Wen (LW) annular regions** — concentric annuli, another standard
  partition for the TEE.

Kitaev-Preskill Background
--------------------------
The Kitaev-Preskill construction [PRB 71, 045110 (2006)] extracts the TEE
``gamma`` via the linear combination::

    S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

where A, B, C are three sectors meeting at a single point, each subtending
an angle of 2π/3.  The key requirement is that each pair of sectors shares
only a boundary (no interior overlap) and that the union AuBuC forms a disk.

- ``n_sectors`` controls how many sectors the disk is divided into (default 3).
- ``rotation`` rotates the sector boundaries (in radians, default 0).
- ``radius`` sets the disk size.

Levin-Wen Background
---------------------
The Levin-Wen construction [PRL 96, 110405 (2006)] uses concentric annuli::

    S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

where A is an inner disk, B a surrounding annulus, and C the region outside
the outer radius (up to the system boundary).

- ``inner_radius`` / ``outer_radius`` control the annulus geometry.
- Origin may be a site index or an arbitrary coordinate vector.

-------------------------------------------------------------------------------
File    : general_python/lattices/tools/region_handler.py
Author  : Maksymilian Kliczkowski
Date    : 2025-12-30
-------------------------------------------------------------------------------
"""

import  numpy   as np
from    typing  import List, Optional, Union, Dict, Any, Tuple, TYPE_CHECKING
from    enum    import Enum, auto
from    itertools import combinations


class RegionType(Enum):
    """Supported region types for :meth:`LatticeRegionHandler.get_region`."""
    HALF            = "half"
    HALF_X          = "half_x"
    HALF_Y          = "half_y"
    HALF_Z          = "half_z"
    HALF_XY         = "half_xy"
    HALF_YX         = "half_yx"
    DISK            = "disk"
    SUBLATTICE      = "sublattice"
    GRAPH           = "graph"
    PLAQUETTE       = "plaquette"
    KITAEV_PRESKILL = "kitaev_preskill"
    LEVIN_WEN       = "levin_wen"
    CUSTOM          = "custom"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RegionType.{self.name}"

if TYPE_CHECKING:
    from general_python.lattices.lattice import Lattice

class LatticeRegionHandler:
    r"""
    Handles region definitions and extractions for a :class:`Lattice`.

    Usage::

        handler = LatticeRegionHandler(lattice)
        # or equivalently via the lattice shortcut:
        sites = lattice.regions.get_region('disk', origin=10, radius=3.0)
        kp    = lattice.regions.get_region('kitaev_preskill', radius=5.0)
        lw    = lattice.regions.get_region('levin_wen', inner_radius=2, outer_radius=5)
    """

    def __init__(self, lattice: "Lattice"):
        self.lattice: "Lattice" = lattice

    # ------------------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------------------

    def _resolve_origin(self, origin) -> np.ndarray:
        """Convert *origin* (site-index or coordinate vector) to a 3-vector."""
        if origin is None:
            return np.mean(self.lattice.rvectors, axis=0)
        if isinstance(origin, (int, np.integer)):
            return self.lattice.rvectors[int(origin)].copy()
        return np.asarray(origin, dtype=float)

    def _raw_displacements(self, r0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute **raw** (no PBC wrapping) displacement vectors and distances
        from *r0* to every lattice site.

        Returns
        -------
        displacements : ndarray, shape (Ns, 3)
        distances     : ndarray, shape (Ns,)
        """
        displacements = self.lattice.rvectors - r0
        distances     = np.linalg.norm(displacements, axis=1)
        return displacements, distances

    def _pbc_displacements(self, r0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PBC-aware displacement vectors and distances from *r0* to
        every lattice site.

        Returns
        -------
        displacements : ndarray, shape (Ns, 3)
        distances     : ndarray, shape (Ns,)
        """
        coords      = self.lattice.rvectors
        pbc_flags   = self.lattice.periodic_flags()
        a1, a2, a3  = self.lattice.a1, self.lattice.a2, self.lattice.a3

        M           = np.array([a1, a2, a3], dtype=float)    # rows = lattice vectors
        Minv        = np.linalg.pinv(M)                       # pseudo-inverse

        dr_cart     = coords - r0                           # (Ns, 3)
        dr_frac     = dr_cart @ Minv.T                      # fractional coords

        dims        = [self.lattice.Lx, max(self.lattice.Ly, 1), max(self.lattice.Lz, 1)]
        for d in range(3):
            if pbc_flags[d]:
                L = dims[d]
                dr_frac[:, d] -= L * np.round(dr_frac[:, d] / L)

        displacements = dr_frac @ M                     # back to Cartesian
        distances     = np.linalg.norm(displacements, axis=1)
        return displacements, distances

    def _displacements(self, r0: np.ndarray, use_pbc: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to PBC-aware or raw displacements."""
        if use_pbc:
            return self._pbc_displacements(r0)
        return self._raw_displacements(r0)

    def get_shortest_displacement(self, i: int, j: int) -> np.ndarray:
        """
        Compute the shortest displacement vector r_j - r_i respecting PBC.
        """
        n_i         = self.lattice.fracs[i]
        n_j         = self.lattice.fracs[j]
        dn          = np.array(n_j - n_i, dtype=float)

        dims        = [self.lattice.Lx, max(self.lattice.Ly, 1), max(self.lattice.Lz, 1)]
        pbc_flags   = self.lattice.periodic_flags()
        for d in range(3):
            if pbc_flags[d]:
                L       = dims[d]
                dn[d]   = dn[d] - L * np.round(dn[d] / L)

        disp  = dn[0] * self.lattice.a1 + dn[1] * self.lattice.a2 + dn[2] * self.lattice.a3
        disp += self.lattice.basis[self.lattice.subs[j]] - self.lattice.basis[self.lattice.subs[i]]
        return disp

    # ------------------------------------------------------------------------------
    # Main region extraction method
    # ------------------------------------------------------------------------------

    def get_region(
        self,
        kind        : Union[str, RegionType] = RegionType.HALF,
        *,
        origin      : Optional[Union[int, List[float]]] = None,
        radius      : Optional[float]                   = None,
        direction   : Optional[str]                     = None,
        sublattice  : Optional[int]                     = None,
        sites       : Optional[List[int]]               = None,
        depth       : Optional[int]                     = None,
        plaquettes  : Optional[List[int]]               = None,
        **kwargs
    ) -> Union[List[int], Dict[str, List[int]]]:
        r"""
        Return site indices defining a spatial region.

        Parameters
        ----------
        kind : str or RegionType
            ``'half'``, ``'half_x'``, ``'half_y'``, ``'half_z'``, ``'disk'``,
            ``'sublattice'``, ``'graph'``, ``'plaquette'``,
            ``'kitaev_preskill'``, ``'levin_wen'``, ``'custom'``.
        origin : int or list[float], optional
            Center of the region (site index **or** coordinate).
        radius : float, optional
            Radius for ``'disk'`` and ``'kitaev_preskill'`` regions.
        direction : str, optional
            Direction for ``'half'`` cuts (``'x'``, ``'y'``, ``'z'``).
        sublattice : int, optional
            Sublattice index for ``'sublattice'`` regions.
        sites : list[int], optional
            Explicit site list for ``'custom'`` regions.
        depth : int, optional
            Hop-distance for ``'graph'`` balls.
        plaquettes : list[int], optional
            Plaquette indices for ``'plaquette'`` regions.

        Keyword-only (forwarded)
        ------------------------
        inner_radius, outer_radius : float
            For ``'levin_wen'``.
        n_sectors : int
            Number of angular sectors for ``'kitaev_preskill'`` (default 3).
        rotation : float
            Rotation of sector boundaries (radians) for KP (default 0).
        use_pbc : bool
            Whether to use PBC-wrapped distances for KP/LW.  **Default
            False** — regions must not wrap around the torus boundary.
        region : str
            Which single region key to return for KP/LW  (e.g. ``'A'``,
            ``'AB'``).  If given, a *flat list* is returned instead of a dict.

        Returns
        -------
        list[int]  or  dict[str, list[int]]
            Sorted site list for simple regions, or a dict of labelled site
            lists for topological partitions.

        Examples
        --------
        >>> lat.regions.get_region('half_x')
        [0, 1, 2, ...]
        >>> lat.regions.get_region('disk', origin=10, radius=2.5)
        [5, 6, 10, 11, 15]
        >>> lat.regions.get_region('kitaev_preskill', radius=5.0)
        {'A': [...], 'B': [...], 'C': [...], 'AB': [...], ...}
        >>> lat.regions.get_region('kitaev_preskill', radius=5.0, region='A')
        [5, 6, 10, ...]
        >>> lat.regions.get_region('levin_wen', inner_radius=2.0, outer_radius=5.0)
        {'A': [...], 'B': [...], 'C': [...], 'AB': [...], ...}
        """
        # Normalise kind to lowercase string
        if isinstance(kind, str):
            kind = kind.strip().lower()
        elif isinstance(kind, RegionType):
            kind = kind.value
        else:
            raise ValueError("kind must be a string or RegionType enum")

        # ------- half-system cuts -------
        if kind in ("half", "half_x", "half-x"):
            return self.region_half(direction or "x")
        if kind in ("half_y", "half-y"):
            return self.region_half("y")
        if kind in ("half_z", "half-z"):
            return self.region_half("z")
        if kind in ("half_xy", "half-xy"):
            return self.region_half("xy")
        if kind in ("half_yx", "half-yx"):
            return self.region_half("yx")

        # ------- disk -------
        if kind == "disk":
            if origin is None or radius is None:
                raise ValueError("'disk' requires 'origin' and 'radius'.")
            return self.region_disk(origin, radius)

        # ------- graph-distance ball -------
        if kind == "graph":
            if origin is None or depth is None:
                raise ValueError("'graph' requires 'origin' (int) and 'depth'.")
            if not isinstance(origin, (int, np.integer)):
                raise ValueError("'graph' origin must be a site index.")
            return self.region_graph_ball(int(origin), depth)

        # ------- sublattice -------
        if kind == "sublattice":
            if sublattice is None:
                raise ValueError("'sublattice' requires 'sublattice' index.")
            return self.region_sublattice(sublattice)

        # ------- plaquette union -------
        if kind == "plaquette":
            if plaquettes is None:
                raise ValueError("'plaquette' requires 'plaquettes' list.")
            return self.region_plaquettes(plaquettes)

        # ------- Kitaev-Preskill -------
        if kind.startswith("kitaev") or kind.startswith("kp"):
            regions = self.region_kitaev_preskill(
                origin      = origin,
                radius      = radius,
                n_sectors   = kwargs.get("n_sectors",   3),
                rotation    = kwargs.get("rotation",    0.0),
                use_pbc     = kwargs.get("use_pbc",     False),
            )
            region_id = kwargs.get("region")
            if region_id is not None:
                return regions.get(region_id.upper(), [])
            return regions

        # ------- Levin-Wen -------
        if kind.startswith("levin") or kind.startswith("lw"):
            regions = self.region_levin_wen(
                origin          = origin,
                inner_radius    = kwargs.get("inner_radius", radius),
                outer_radius    = kwargs.get("outer_radius"),
                use_pbc         = kwargs.get("use_pbc", False),
            )
            region_id = kwargs.get("region")
            if region_id is not None:
                return regions.get(region_id.upper(), [])
            return regions

        # ------- custom -------
        if kind == "custom":
            return sorted(list(set(sites))) if sites else []

        raise ValueError(f"Unknown region type: {kind!r}")

    # ------------------------------------------------------------------------------
    # Specific region definitions
    # ------------------------------------------------------------------------------

    def region_half(self, direction: str = "x") -> List[int]:
        r"""
        Half-system cut along a cardinal or tilted direction.
        
        Useful for area law scaling. Handles PBC by cutting based on coordinates relative to median.
        
        Parameters
        ----------
        direction : str
            Direction to cut ('x', 'y', 'z', 'xy', or 'yx').
            'xy' is a diagonal cut along x+y, 'yx' is along x-y.
        Returns
        -------
        list[int]
            Sorted list of site indices in the half-region.
            
        Example
        -------
        >>> half_x_sites = lattice.regions.get_region(kind='half_x')
        ... [10, 11, 12, 13, 14, 15, ...]  # sites in the left half of the lattice
        """
        coords      = self.lattice.rvectors
        direction   = direction.lower()
        
        if direction == "x":
            vals = coords[:, 0]
        elif direction == "y":
            vals = coords[:, 1]
        elif direction == "z":
            vals = coords[:, 2]
        elif direction in ("xy", "half-xy", "half_xy"):
            vals = coords[:, 0] + coords[:, 1]
        elif direction in ("yx", "half-yx", "half_yx"):
            vals = coords[:, 0] - coords[:, 1]
        else:
            raise ValueError("direction must be 'x', 'y', 'z', 'xy', or 'yx'")
        
        # Use median coordinate to define half cut, which is more robust for PBC than mean
        cut_val = np.median(vals)

        # Return sites with coordinate less than cut_val along the specified axis
        return sorted(np.where(vals < cut_val)[0].tolist())

    # ------------------------------------------------------------------------------

    def region_disk(self, center: Union[int, List[float]], radius: float, pbc: bool = False) -> List[int]:
        """
        Spherical / circular region (PBC-aware).

        Parameters
        ----------
        center : int or array-like
            Site index **or** coordinate vector.
        radius : float
            Inclusion radius.
        """
        r0          = self._resolve_origin(center)
        if pbc:
            disps, dists = self._pbc_displacements(r0)
        else:
            disps, dists = self._displacements(r0)
        return sorted(np.where(dists <= radius)[0].tolist())

    # ------------------------------------------------------------------------------

    def region_sublattice(self, sub: int) -> List[int]:
        """
        Return all sites belonging to a specific sublattice.
        """
        return [i for i in range(self.lattice.Ns) if self.lattice.sublattice(i) == sub]

    # ------------------------------------------------------------------------------

    def region_graph_ball(self, center: int, depth: int) -> List[int]:
        """
        Graph-distance ball (breadth-first search).
        
        Returns all sites within `depth` bonds from `center` (inclusive).
        """
        visited     = {center}
        frontier    = {center}

        for _ in range(depth):
            new_nodes = set()
            for s in frontier:
                for n in self.lattice.neighbors(s, order=1):
                    if n is not None and n >= 0 and n not in visited and n != self.lattice.bad_lattice_site:
                        new_nodes.add(n)
            visited |= new_nodes
            frontier = new_nodes

        return sorted(list(visited))

    # ------------------------------------------------------------------------------

    def region_plaquettes(self, plaquette_ids: List[int]) -> List[int]:
        """
        Region defined by a union of plaquettes.
        
        Requires the lattice to implement `calculate_plaquettes()` and store `_plaquettes`.
        """
        if not hasattr(self.lattice, "_plaquettes") or self.lattice._plaquettes is None:
            try:
                self.lattice.calculate_plaquettes()
            except NotImplementedError:
                raise ValueError("Plaquettes are not defined for this lattice.")

        sites = set()
        for pid in plaquette_ids:
            if 0 <= pid < len(self.lattice._plaquettes):
                sites.update(self.lattice._plaquettes[pid])
        return sorted(list(sites))

    # ------------------------------------------------------------------------------

    def region_kitaev_preskill(
        self,
        origin      : Optional[Union[int, List[float]]] = None,
        radius      : Optional[float]                   = None,
        n_sectors   : int                               = 3,
        rotation    : float                              = 0.0,
        use_pbc     : bool                               = False,
    ) -> Dict[str, List[int]]:
        r"""
        Divide the lattice into angular sectors meeting at *origin*.

        The disk of the given *radius* is split into *n_sectors* equal
        pie-slices.  By default (``n_sectors=3, rotation=0``)::

            A : angles in [-π, -π/3)
            B : angles in [-π/3, +π/3)
            C : angles in [+π/3, +π]

        The *rotation* parameter (radians) rotates **all** sector boundaries
        counter-clockwise.

        .. warning::

            ``use_pbc`` defaults to **False**.  For the TEE construction the
            regions must **not** wrap around the boundary — otherwise a
            single angular sector may pick up sites from the opposite side of
            the torus and the linear combination becomes ill-defined.

        Parameters
        ----------
        origin : int or array-like, optional
            Center of the disk.  Defaults to the centroid of all sites.
        radius : float, optional
            Disk radius.  Defaults to ``min(Lx, Ly)/2 - 0.5`` (or 1.0).
        n_sectors : int
            Number of angular sectors (default 3).
        rotation : float
            Global rotation of sector boundaries in radians (default 0).
        use_pbc : bool
            If *True*, use minimum-image (PBC-wrapped) displacements when
            computing distances and angles.  **Default False** — raw
            Euclidean distances are used so the regions cannot wrap around
            the boundary.

        Returns
        -------
        dict[str, list[int]]
            Keys: single-sector labels ``'A'``, ``'B'``, …  and all pairwise
            / triple unions (``'AB'``, ``'BC'``, ``'ABC'``, …).

        Notes
        -----
        For the TEE one computes::

            S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

        The sectors must *tile* a disk without overlapping interiors so that
        the linear combination isolates the topological contribution ``-γ``.
        The ``rotation`` parameter lets you sweep the partition to check
        independence.
        """
        r0 = self._resolve_origin(origin)

        # Default radius
        if radius is None:
            Ls     = [self.lattice.Lx]
            if self.lattice.dim >= 2:
                Ls.append(self.lattice.Ly)
            radius = max((min(Ls) / 2.0) - 0.5, 1.0)

        disps, dists = self._displacements(r0, use_pbc=use_pbc)
        mask_r       = dists <= radius

        # Azimuthal angles shifted by rotation
        angles = np.arctan2(disps[:, 1], disps[:, 0]) - rotation
        # Wrap to [-π, π)
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        # Sector boundaries
        sector_width = 2 * np.pi / n_sectors
        labels       = [chr(ord('A') + k) for k in range(n_sectors)]

        sectors: Dict[str, List[int]] = {}
        for k, lbl in enumerate(labels):
            lo = -np.pi + k * sector_width
            hi = lo + sector_width
            if k < n_sectors - 1:
                mask = (angles >= lo) & (angles < hi) & mask_r
            else:
                # Last sector includes the upper boundary
                mask = (angles >= lo) & (angles <= hi) & mask_r
            sectors[lbl] = sorted(np.where(mask)[0].tolist())

        # Build all non-empty unions (pairs, triples, …)
        for size in range(2, n_sectors + 1):
            for combo in combinations(labels, size):
                key   = "".join(combo)
                union = set()
                for lbl in combo:
                    union |= set(sectors[lbl])
                sectors[key] = sorted(union)

        return sectors
    
    # ---------------------------------------------------------------------------
    
    def region_levin_wen(
        self,
        origin          : Optional[Union[int, List[float]]] = None,
        inner_radius    : Optional[float]                   = None,
        outer_radius    : Optional[float]                   = None,
        use_pbc         : bool                               = False,
    ) -> Dict[str, List[int]]:
        r"""
        Define three concentric regions around *origin*.

        - **A** (inner disk) : distance < ``inner_radius``
        - **B** (annulus)    : ``inner_radius`` ≤ distance < ``outer_radius``
        - **C** (exterior)   : distance ≥ ``outer_radius``

        .. warning::

            ``use_pbc`` defaults to **False**.  Wrapping would make sites
            from the far side of the torus appear close to the origin,
            contaminating the annular regions and invalidating the TEE
            linear combination.

        Parameters
        ----------
        origin : int or array-like, optional
            Centre of the annuli.  Defaults to the centroid of all sites.
        inner_radius : float, optional
            Boundary between A and B.  Default 1.0.
        outer_radius : float, optional
            Boundary between B and C.  Default ``min(Lx, Ly)/2 - 0.5``.
        use_pbc : bool
            If *True*, use minimum-image (PBC-wrapped) displacements.
            **Default False** — raw Euclidean distances prevent wrap-around.

        Returns
        -------
        dict[str, list[int]]
            Keys: ``'A'``, ``'B'``, ``'C'`` and unions ``'AB'``, ``'BC'``,
            ``'AC'``, ``'ABC'``.

        Notes
        -----
        For the TEE::

            S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

        In the original Levin-Wen paper the region outside the outer annulus
        is called *D*; here we label it **C** for consistency with the KP
        convention.
        """
        r0 = self._resolve_origin(origin)

        if inner_radius is None:
            inner_radius = 1.0
        if outer_radius is None:
            Ls = [self.lattice.Lx]
            if self.lattice.dim >= 2:
                Ls.append(self.lattice.Ly)
            outer_radius = max((min(Ls) / 2.0) - 0.5, inner_radius + 1.0)

        _, dists = self._displacements(r0, use_pbc=use_pbc)

        A_mask = dists <  inner_radius
        B_mask = (dists >= inner_radius) & (dists < outer_radius)
        C_mask = dists >= outer_radius

        regions: Dict[str, List[int]] = {
            'A':   sorted(np.where(A_mask)[0].tolist()),
            'B':   sorted(np.where(B_mask)[0].tolist()),
            'C':   sorted(np.where(C_mask)[0].tolist()),
        }
        regions['AB']  = sorted(set(regions['A']) | set(regions['B']))
        regions['BC']  = sorted(set(regions['B']) | set(regions['C']))
        regions['AC']  = sorted(set(regions['A']) | set(regions['C']))
        regions['ABC'] = sorted(set(regions['A']) | set(regions['B']) | set(regions['C']))
        return regions

# -------------------------------------------------------------------------------
#! End of region_handler.py
# -------------------------------------------------------------------------------