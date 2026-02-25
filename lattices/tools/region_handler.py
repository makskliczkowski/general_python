r"""
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

import  numpy       as np
from    typing      import List, Optional, Union, Dict, Any, Tuple, TYPE_CHECKING
from    enum        import Enum
from    itertools   import combinations


class RegionType(Enum):
    """Supported region types for :meth:`LatticeRegionHandler.get_region`."""
    HALF            = "half"
    HALF_X          = "half_x"
    HALF_Y          = "half_y"
    HALF_Z          = "half_z"
    HALF_XY         = "half_xy"
    HALF_YX         = "half_yx"
    QUARTER         = "quarter"
    SWEEP           = "sweep"
    DISK            = "disk"
    SUBLATTICE      = "sublattice"
    GRAPH           = "graph"
    PLAQUETTE       = "plaquette"
    KITAEV_PRESKILL = "kitaev_preskill"
    LEVIN_WEN       = "levin_wen"
    CUSTOM          = "custom"
    FRACTION        = "fraction"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RegionType.{self.name}"

if TYPE_CHECKING:
    from general_python.lattices.lattice import Lattice
    from .regions.region import Region

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

    def _sorted_unique_sites(self, sites: List[int]) -> List[int]:
        """Normalize a site list to sorted unique valid indices."""
        ns = int(self.lattice.Ns)
        return sorted({int(s) for s in sites if 0 <= int(s) < ns})

    def _complement_sites(self, sites_a: List[int]) -> List[int]:
        """Return sorted complement of A in [0, Ns)."""
        set_a = set(self._sorted_unique_sites(sites_a))
        return sorted(i for i in range(int(self.lattice.Ns)) if i not in set_a)

    def _make_bipartite_region(self, sites_a: List[int], region_cls):
        """Construct a bipartite Region-like object with A and its complement B."""
        a   = self._sorted_unique_sites(sites_a)
        b   = self._complement_sites(a)
        return region_cls(A=a, B=b, C=[])

    def _make_topological_region(self, regions: Dict[str, List[int]], *, kind: str):
        """Construct KP/LW region objects from dictionary output."""
        from .regions import KitaevPreskillRegion, LevinWenRegion

        a   = self._sorted_unique_sites(regions.get("A", []))
        b   = self._sorted_unique_sites(regions.get("B", []))
        c   = self._sorted_unique_sites(regions.get("C", []))
        ab  = self._sorted_unique_sites(regions.get("AB", []))
        ac  = self._sorted_unique_sites(regions.get("AC", []))
        bc  = self._sorted_unique_sites(regions.get("BC", []))
        abc = self._sorted_unique_sites(regions.get("ABC", []))

        if kind == "kitaev_preskill":
            return KitaevPreskillRegion(A=a, B=b, C=c, AB=ab, AC=ac, BC=bc, ABC=abc)
        if kind == "levin_wen":
            return LevinWenRegion(A=a, B=b, C=c, AB=ab, AC=ac, BC=bc, ABC=abc)
        raise ValueError(f"Unsupported topological region kind: {kind}")

    def _normalize_predefined_kind(self, kind: Optional[str]) -> Optional[str]:
        """Normalize user-friendly kind aliases to registry kind names."""
        if kind is None:
            return None
        k = str(kind).strip().lower()
        if "kitaev" in k or k.startswith("kp"):
            return "kitaev_preskill"
        if "levin" in k or k.startswith("lw"):
            return "levin_wen"
        if k.startswith("half"):
            return "half"
        if "disk" in k:
            return "disk"
        if "plaquette" in k:
            return "plaquette"
        if "custom" in k:
            return "custom"
        if "sublattice" in k:
            return "sublattice"
        if "graph" in k:
            return "graph"
        return k

    def _normalize_lattice_type_filter(self, lattice_type: Optional[Union[str, Any]]):
        """Normalize optional lattice-type filter to LatticeType enum."""
        if lattice_type is None:
            return None
        from .lattice_tools import LatticeType
        if isinstance(lattice_type, LatticeType):
            return lattice_type
        if isinstance(lattice_type, str):
            key = lattice_type.strip().lower()
            mapping = {
                "square": LatticeType.SQUARE,
                "triangular": LatticeType.TRIANGULAR,
                "honeycomb": LatticeType.HONEYCOMB,
                "hexagonal": LatticeType.HEXAGONAL,
                "graph": LatticeType.GRAPH,
            }
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unknown lattice type filter: {lattice_type!r}")

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
        kind            : Union[str, RegionType] = RegionType.HALF,
        *,
        origin          : Optional[Union[int, List[float]]] = None,
        radius          : Optional[float]                   = None,
        direction       : Optional[str]                     = None,
        sublattice      : Optional[int]                     = None,
        sites           : Optional[List[int]]               = None,
        depth           : Optional[int]                     = None,
        plaquettes      : Optional[List[int]]               = None,
        configuration   : Optional[int]                     = None,
        predefined      : Optional[Union[bool, int, str]]   = None,
        as_region       : bool                              = True,
        **kwargs
    ) -> Union[List[int], Dict[str, List[int]], Dict[str, Any], List[Dict[str, Any]], "Region"]:
        r"""
        Return site indices defining a spatial region.

        Parameters
        ----------
        kind : str or RegionType
            ``'half'``, ``'half_x'``, ``'half_y'``, ``'half_z'``, ``'disk'``,
            ``'quarter'``, ``'sweep'``, ``'sublattice'``, ``'graph'``, ``'plaquette'``,
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
        configuration : int, optional
            Legacy predefined configuration index (1-based) for the given
            lattice and kind.
        predefined : bool | int | str, optional
            Convenience selector for predefined regions.
            - ``True``  : list available predefined entries for given lattice/kind.
            - ``int``   : select predefined entry by 0-based index.
            - ``str``   : select predefined entry by label.
        as_region : bool, optional
            If True (default), return Region-class objects for supported kinds.
            If False, keep legacy list/dict return shapes.

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
        Region  or  list[int]  or  dict[str, list[int]]  or  list[dict]
            Sorted site list for simple regions, a dict of labelled site
            lists for topological partitions, metadata entries when ``predefined=True``,
            or Region objects when ``as_region=True``.

        Examples
        --------
        >>> lat.regions.get_region('half_x')
        HalfRegion(A=[...], B=[...])
        >>> lat.regions.get_region('disk', origin=10, radius=2.5)
        DiskRegion(A=[...], B=[...], C=[])
        >>> lat.regions.get_region('kitaev_preskill', configuration=1)
        KitaevPreskillRegion(A=[...], B=[...], C=[...])
        >>> lat.regions.get_region('kitaev_preskill', predefined=0)
        KitaevPreskillRegion(A=[...], B=[...], C=[...])
        >>> lat.regions.get_region('half_x', as_region=False)
        [0, 1, 2, ...]
        """
        # Normalise kind to lowercase string
        if isinstance(kind, str):
            kind_str = kind.strip().lower()
        elif isinstance(kind, RegionType):
            kind_str = kind.value
        else:
            raise ValueError("kind must be a string or RegionType enum")

        # handle predefined configurations/selection
        if configuration is not None and predefined is not None:
            raise ValueError("Use only one of 'configuration' or 'predefined'.")

        if configuration is not None or predefined is not None:
            try:
                from .regions import get_predefined_region, list_predefined_regions
            except ImportError as exc:
                raise ImportError(
                    "Predefined regions are unavailable. Ensure "
                    "'lattices/tools/regions' package is importable."
                ) from exc

            if predefined is True:
                return list_predefined_regions(
                    self.lattice._type,
                    self.lattice.Lx,
                    self.lattice.Ly,
                    self.lattice.Lz,
                    kind=kind_str,
                )

            predefined_region = None
            if configuration is not None:
                predefined_region = get_predefined_region(
                    self.lattice._type,
                    self.lattice.Lx,
                    self.lattice.Ly,
                    self.lattice.Lz,
                    kind_str,
                    configuration,
                )
            elif isinstance(predefined, (int, np.integer)):
                predefined_region = get_predefined_region(
                    self.lattice._type,
                    self.lattice.Lx,
                    self.lattice.Ly,
                    self.lattice.Lz,
                    kind_str,
                    index=int(predefined),
                )
            elif isinstance(predefined, str):
                predefined_region = get_predefined_region(
                    self.lattice._type,
                    self.lattice.Lx,
                    self.lattice.Ly,
                    self.lattice.Lz,
                    kind_str,
                    label=predefined,
                )
            else:
                raise ValueError(
                    "'predefined' must be one of: True, int (0-based index), str (label)."
                )

            if predefined_region is None:
                raise ValueError(
                    f"Predefined region not found for {self.lattice._type} "
                    f"{self.lattice.Lx}x{self.lattice.Ly}x{self.lattice.Lz} "
                    f"{kind_str!r} (configuration={configuration!r}, predefined={predefined!r})."
                )

            region_id = kwargs.get("region")
            if region_id is not None:
                return predefined_region.get(region_id.upper(), [])

            if as_region:
                return predefined_region

            kind_key = kind_str.replace("-", "_")
            if kind_key.startswith(("kitaev", "kp", "levin", "lw")):
                return predefined_region.to_dict()
            if kind_key.startswith("half") or kind_key in (
                "disk",
                "sublattice",
                "graph",
                "plaquette",
                "custom",
            ):
                return predefined_region.A
            return predefined_region.to_dict()

        # 0) fraction
        if "frac" in kind_str:
            fraction    = kwargs.get("fraction", 0.5)
            sites_frac  = self.region_fraction(fraction)
            if as_region:
                from .regions import RegionFraction
                return self._make_bipartite_region(sites_frac, RegionFraction)
            return sites_frac
            
            
        # a) half-system cuts
        if kind_str in ("half", "half_x", "half-x"):
            sites_half = self.region_half(direction or "x")
            if as_region:
                from .regions import HalfRegions
                return self._make_bipartite_region(sites_half, HalfRegions)
            return sites_half
        if kind_str in ("half_y", "half-y"):
            sites_half = self.region_half("y")
            if as_region:
                from .regions import HalfRegions
                return self._make_bipartite_region(sites_half, HalfRegions)
            return sites_half
        if kind_str in ("half_z", "half-z"):
            sites_half = self.region_half("z")
            if as_region:
                from .regions import HalfRegions
                return self._make_bipartite_region(sites_half, HalfRegions)
            return sites_half
        if kind_str in ("half_xy", "half-xy"):
            sites_half = self.region_half("xy")
            if as_region:
                from .regions import HalfRegions
                return self._make_bipartite_region(sites_half, HalfRegions)
            return sites_half
        if kind_str in ("half_yx", "half-yx"):
            sites_half = self.region_half("yx")
            if as_region:
                from .regions import HalfRegions
                return self._make_bipartite_region(sites_half, HalfRegions)
            return sites_half
        if kind_str == "quarter":
            sites_quarter = self.region_quarter()
            if as_region:
                from .regions import HalfRegions
                return self._make_bipartite_region(sites_quarter, HalfRegions)
            return sites_quarter
        if kind_str == "sweep":
            sweep_regions = self.region_sweep(by_unit_cell=kwargs.get("by_unit_cell", None))
            if as_region:
                from .regions import HalfRegions
                return {name: self._make_bipartite_region(cut_sites, HalfRegions) for name, cut_sites in sweep_regions.items()}
            return sweep_regions

        # b) disk
        if kind_str == "disk":
            if origin is None or radius is None:
                raise ValueError("'disk' requires 'origin' and 'radius'.")
            sites_disk = self.region_disk(origin, radius)
            if as_region:
                from .regions import DiskRegion
                return self._make_bipartite_region(sites_disk, DiskRegion)
            return sites_disk

        # c) graph-distance ball
        if kind_str == "graph":
            if origin is None or depth is None:
                raise ValueError("'graph' requires 'origin' (int) and 'depth'.")
            if not isinstance(origin, (int, np.integer)):
                raise ValueError("'graph' origin must be a site index.")
            sites_graph = self.region_graph_ball(int(origin), depth)
            if as_region:
                from .regions import CustomRegion
                return self._make_bipartite_region(sites_graph, CustomRegion)
            return sites_graph

        # d) sublattice
        if kind_str == "sublattice":
            if sublattice is None:
                raise ValueError("'sublattice' requires 'sublattice' index.")
            sites_sub = self.region_sublattice(sublattice)
            if as_region:
                from .regions import CustomRegion
                return self._make_bipartite_region(sites_sub, CustomRegion)
            return sites_sub

        # e) plaquette union 
        if kind_str == "plaquette":
            if plaquettes is None:
                raise ValueError("'plaquette' requires 'plaquettes' list.")
            sites_plaq = self.region_plaquettes(plaquettes)
            if as_region:
                from .regions import PlaquetteRegion
                return self._make_bipartite_region(sites_plaq, PlaquetteRegion)
            return sites_plaq

        # f) Kitaev-Preskill
        if kind_str.startswith("kitaev") or kind_str.startswith("kp"):
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
            if as_region:
                return self._make_topological_region(regions, kind="kitaev_preskill")
            return regions

        # g) Levin-Wen
        if kind_str.startswith("levin") or kind_str.startswith("lw"):
            regions = self.region_levin_wen(
                origin          = origin,
                inner_radius    = kwargs.get("inner_radius", radius),
                outer_radius    = kwargs.get("outer_radius"),
                use_pbc         = kwargs.get("use_pbc", False),
            )
            region_id = kwargs.get("region")
            if region_id is not None:
                return regions.get(region_id.upper(), [])
            if as_region:
                return self._make_topological_region(regions, kind="levin_wen")
            return regions

        # h) custom 
        if kind_str == "custom":
            custom_sites = sorted(list(set(sites))) if sites else []
            if as_region:
                from .regions import CustomRegion
                return self._make_bipartite_region(custom_sites, CustomRegion)
            return custom_sites

        raise ValueError(f"Unknown region type: {kind_str!r}")

    # -------------------------------------------------------------------------------
    # Convenience methods for querying predefined regions
    # -------------------------------------------------------------------------------
    
    def list_predefined(
        self,
        kind: Optional[str] = None,
        *,
        lattice_type: Optional[Union[str, Any]] = None,
        lx: Optional[int] = None,
        ly: Optional[int] = None,
        lz: Optional[int] = None,
        include_region: bool = False,
        labels_only: bool = False,
    ) -> List[Any]:
        """
        List predefined regions with optional filtering by type/size/kind.

        Defaults to the current lattice type and size if no filters are provided.
        """
        from .regions import PREDEFINED_META, PREDEFINED_REGIONS

        kind_norm = self._normalize_predefined_kind(kind)
        lt_filter = self._normalize_lattice_type_filter(lattice_type)
        use_default_scope = (
            lattice_type is None and lx is None and ly is None and lz is None
        )

        if use_default_scope:
            lt_filter = self.lattice._type
            lx = int(self.lattice.Lx)
            ly = int(self.lattice.Ly)
            lz = int(self.lattice.Lz)

        selected: List[Dict[str, Any]] = []
        keys = sorted(
            PREDEFINED_REGIONS.keys(),
            key=lambda k: (str(k[0]), int(k[1]), int(k[2]), int(k[3]), str(k[4]), int(k[5])),
        )
        for key in keys:
            lt, kx, ky, kz, kkind, cfg = key
            if lt_filter is not None and lt != lt_filter:
                continue
            if lx is not None and int(kx) != int(lx):
                continue
            if ly is not None and int(ky) != int(ly):
                continue
            if lz is not None and int(kz) != int(lz):
                continue
            if kind_norm is not None and kkind != kind_norm:
                continue

            meta = dict(PREDEFINED_META.get(key, {}))
            entry = {
                "lattice_type": lt,
                "lx": int(kx),
                "ly": int(ky),
                "lz": int(kz),
                "kind": str(kkind),
                "configuration": int(cfg),
                "label": meta.get("label", f"{kkind}_{cfg}"),
                "tags": tuple(meta.get("tags", ())),
            }
            if include_region:
                entry["region"] = PREDEFINED_REGIONS[key]
            selected.append(entry)

        for idx, entry in enumerate(selected):
            entry["index"] = idx

        if labels_only:
            return [entry["label"] for entry in selected]
        return selected
    
    def list_predefined_kinds(
        self,
        *,
        lattice_type: Optional[Union[str, Any]] = None,
        lx: Optional[int] = None,
        ly: Optional[int] = None,
        lz: Optional[int] = None,
    ) -> List[str]:
        """Return sorted predefined-kind names for the selected type/size filters."""
        entries = self.list_predefined(
            kind=None,
            lattice_type=lattice_type,
            lx=lx,
            ly=ly,
            lz=lz,
            include_region=False,
            labels_only=False,
        )
        return sorted({str(entry["kind"]) for entry in entries})

    def list_predefined_sizes(
        self,
        *,
        kind: Optional[str] = None,
        lattice_type: Optional[Union[str, Any]] = None,
    ) -> List[Tuple[int, int, int]]:
        """Return available (Lx, Ly, Lz) sizes for selected lattice type and optional kind."""
        entries = self.list_predefined(
            kind=kind,
            lattice_type=lattice_type,
            lx=None,
            ly=None,
            lz=None,
            include_region=False,
            labels_only=False,
        )
        return sorted({(int(entry["lx"]), int(entry["ly"]), int(entry["lz"])) for entry in entries})

    def get_predefined(
        self,
        kind: str,
        *,
        configuration: Optional[Union[int, str]] = None,
        index: Optional[int] = None,
        label: Optional[str] = None,
        lattice_type: Optional[Union[str, Any]] = None,
        lx: Optional[int] = None,
        ly: Optional[int] = None,
        lz: Optional[int] = None,
        region: Optional[str] = None,
        return_meta: bool = False,
    ):
        """
        Fetch one predefined region by configuration/index/label with optional type/size filters.
        """
        from .regions import get_predefined_region

        if sum(v is not None for v in (configuration, index, label)) > 1:
            raise ValueError("Provide only one selector: configuration, index, or label.")

        lt = self._normalize_lattice_type_filter(lattice_type)
        if lt is None:
            lt = self.lattice._type

        kx = int(self.lattice.Lx) if lx is None else int(lx)
        ky = int(self.lattice.Ly) if ly is None else int(ly)
        kz = int(self.lattice.Lz) if lz is None else int(lz)

        result = get_predefined_region(
            lt,
            kx,
            ky,
            kz,
            kind,
            config=configuration,
            index=index,
            label=label,
            return_meta=return_meta,
        )
        if result is None:
            raise ValueError(
                f"No predefined region found for kind={kind!r}, "
                f"selector=(configuration={configuration!r}, index={index!r}, label={label!r}), "
                f"lattice={lt}, size=({kx}, {ky}, {kz})."
            )

        if return_meta:
            region_obj, meta = result
            if region is not None:
                return region_obj.get(region.upper(), []), meta
            return region_obj, meta

        region_obj = result
        if region is not None:
            return region_obj.get(region.upper(), [])
        return region_obj

    def show_predefined(
        self,
        kind            : Optional[str] = None,
        *,
        lattice_type    : Optional[Union[str, Any]] = None,
        lx              : Optional[int] = None,
        ly              : Optional[int] = None,
        lz              : Optional[int] = None,
        limit           : Optional[int] = 40,
    ) -> List[Dict[str, Any]]:
        """
        Pretty-print predefined entries and return them.
        
        Filters default to the current lattice type and size if not provided.
        
        Parameters
        ----------
        kind : str, optional
            Filter by region kind (e.g. 'half_x', 'kitaev_preskill', etc.).
        lattice_type : str or LatticeType, optional
            Filter by lattice type (e.g. 'square', 'honeycomb').
        lx, ly, lz : int, optional
            Filter by lattice size. Defaults to current lattice dimensions.
        limit : int, optional
            Maximum number of entries to show (default 40). Use None for no limit.        
        """
        entries = self.list_predefined(
            kind=kind,
            lattice_type=lattice_type,
            lx=lx,
            ly=ly,
            lz=lz,
            include_region=False,
            labels_only=False,
        )
        n_total = len(entries)
        if n_total == 0:
            print("No predefined regions found for the requested filters.")
            return entries

        n_show  = n_total if limit is None else min(max(int(limit), 0), n_total)
        rows    = entries[:n_show]

        headers = ["index", "type", "size", "kind", "cfg", "label", "tags"]
        body    = []
        for e in rows:
            body.append(
                [
                    str(e["index"]),
                    str(e["lattice_type"]),
                    f"{e['lx']}x{e['ly']}x{e['lz']}",
                    str(e["kind"]),
                    str(e["configuration"]),
                    str(e["label"]),
                    ",".join(map(str, e.get("tags", ()))),
                ]
            )

        widths = [len(h) for h in headers]
        for row in body:
            for i, cell in enumerate(row):
                widths[i] = min(max(widths[i], len(cell)), 40)

        def _clip(txt: str, w: int) -> str:
            if len(txt) <= w:
                return txt
            if w <= 1:
                return txt[:w]
            return txt[: w - 1] + "…"

        def _fmt(cols: List[str]) -> str:
            return " | ".join(_clip(c, widths[i]).ljust(widths[i]) for i, c in enumerate(cols))

        print(f"predefined regions: showing {n_show}/{n_total}")
        print(_fmt(headers))
        print("-+-".join("-" * w for w in widths))
        for row in body:
            print(_fmt(row))
        return entries

    # ------------------------------------------------------------------------------
    # Entropy-oriented cut helpers
    # ------------------------------------------------------------------------------

    def get_entropy_cuts(
        self,
        cut_type: str = "all",
        *,
        include_sublattice: bool = True,
        sweep_by_unit_cell: Optional[bool] = None,
    ) -> Dict[str, List[int]]:
        """
        Return canonical bipartition cuts for entanglement-entropy studies.

        Supported cut types:
        - ``half_x``, ``half_y``, ``quarter``, ``sublattice_A``, ``sweep``, ``all``.

        Notes
        -----
        - ``quarter`` 
            - is defined as the intersection of ``half_x`` and ``half_y``.
        - ``sweep`` 
            - returns nested prefixes for scaling analyses.
        -   For non-Bravais lattices (e.g. honeycomb), ``sweep`` defaults to unit-cell
            increments; for Bravais lattices, it defaults to site increments.
        """
        cut_type_norm   = cut_type.strip().lower()
        valid           = {"half_x", "half_y", "quarter", "sublattice_a", "sweep", "all"}
        
        if cut_type_norm not in valid:
            raise ValueError(f"Unknown cut_type '{cut_type}'. Use one of {sorted(valid)}.")

        cuts            : Dict[str, List[int]] = {}

        if cut_type_norm in ("half_x", "all"):
            cuts["half_x"] = self.region_half("x")

        if cut_type_norm in ("half_y", "all"):
            cuts["half_y"] = self.region_half("y")

        if cut_type_norm in ("quarter", "all"):
            half_x          = set(cuts.get("half_x", self.region_half("x")))
            half_y          = set(cuts.get("half_y", self.region_half("y")))
            cuts["quarter"] = sorted(half_x & half_y)

        if include_sublattice and cut_type_norm in ("sublattice_a", "all"):
            try:
                cuts["sublattice_A"] = self.region_sublattice(0)
            except Exception:
                # Some custom lattices may not expose meaningful sublattice labels.
                pass

        if cut_type_norm in ("sweep", "all"):
            cuts.update(self.region_sweep(by_unit_cell=sweep_by_unit_cell))

        return cuts

    def _coords_to_index(self, x: int, y: int, z: int = 0, sub: int = 0) -> int:
        """Helper to map coordinates to a linear site index."""
        from .lattice_tools import LatticeType
        if self.lattice._type == LatticeType.SQUARE:
            # Row-major: z varies slowest, x fastest
            return int(z * self.lattice.Lx * self.lattice.Ly + y * self.lattice.Lx + x)
        elif self.lattice._type == LatticeType.HONEYCOMB:
            # Each unit cell (x,y) has 2 sites: index = (y * Lx + x) * 2 + sub
            return int((z * self.lattice.Ly * self.lattice.Lx + y * self.lattice.Lx + x) * 2 + sub)
        elif self.lattice._type == LatticeType.TRIANGULAR:
            return int(z * self.lattice.Lx * self.lattice.Ly + y * self.lattice.Lx + x)
        else:
            # Fallback for other lattices if they implement site_index or similar
            if hasattr(self.lattice, "site_index"):
                try:
                    return self.lattice.site_index(x, y, z)
                except TypeError:
                    pass
            raise NotImplementedError(f"Coordinate mapping not implemented for lattice type {self.lattice._type}")

    # ------------------------------------------------------------------------------
    # Specific region definitions
    # ------------------------------------------------------------------------------

    def region_fraction(self, fraction: Union[float, int]) -> List[int]:
        ''' Return a fraction of the system as a contiguous block of sites in index order. '''
        
        if not (0 < fraction < 1) and fraction <= self.lattice.Ns:
            fraction = fraction // self.lattice.Ns 
            
        n_sites = int(fraction * self.lattice.Ns)
        return [i for i in range(n_sites)]

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
        Region
            The half-region as a CustomRegion object.
            
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
        sites   = sorted(np.where(vals < cut_val)[0].tolist())
        return sites

    # ------------------------------------------------------------------------------

    def region_quarter(self) -> List[int]:
        """
        Return a quarter-system region as ``half_x ∩ half_y``.
        """
        from .regions import CustomRegion
        
        half_x  = self.region_half("x")
        half_y  = self.region_half("y")
        A       = sorted(set(half_x.A) & set(half_y.A))
        B       = sorted(set(half_x.B) & set(half_y.B))
        return A
    # ------------------------------------------------------------------------------

    def region_sweep(self, *, by_unit_cell: Optional[bool] = None) -> Dict[str, List[int]]:
        """
        Return nested sweep cuts used for entropy scaling analyses.

        Parameters
        ----------
        by_unit_cell : bool or None
            - ``True``: grow by unit cells.
            - ``False``: grow by individual sites.
            - ``None``: auto-select (unit cells for multi-sublattice lattices).
        """
        Ns = int(self.lattice.Ns)
        if Ns <= 1:
            return {}

        subs = np.asarray(getattr(self.lattice, "subs", np.zeros(Ns, dtype=int)))
        if by_unit_cell is None:
            by_unit_cell = len(np.unique(subs)) > 1

        cuts: Dict[str, List[int]] = {}

        if by_unit_cell and hasattr(self.lattice, "fracs"):
            fracs = np.asarray(self.lattice.fracs)
            if fracs.shape[0] != Ns:
                by_unit_cell = False
            else:
                cells_to_sites: Dict[Tuple[int, int, int], List[int]] = {}
                for i in range(Ns):
                    cx, cy, cz = map(int, fracs[i])
                    cells_to_sites.setdefault((cx, cy, cz), []).append(i)

                ordered_cells = sorted(cells_to_sites.keys(), key=lambda c: (c[2], c[1], c[0]))
                total_cells = len(ordered_cells)
                if total_cells <= 1:
                    return {}

                for n_cell in range(1, total_cells):
                    region = []
                    for cell in ordered_cells[:n_cell]:
                        region.extend(sorted(cells_to_sites[cell], key=lambda s: int(subs[s])))
                    cuts[f"sweep_{n_cell}_of_{total_cells}"] = sorted(region)

                return cuts

        # Fallback: site-ordered sweep in real-space lexicographic order.
        coords = np.asarray(self.lattice.rvectors)
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        for n_site in range(1, Ns):
            cuts[f"sweep_{n_site}_of_{Ns}"] = sorted(order[:n_site].tolist())
        return cuts

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
