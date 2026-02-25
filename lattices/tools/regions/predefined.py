"""
Predefined region registry for common lattice geometries and system sizes.

This module stores curated region partitions (Kitaev-Preskill, Levin-Wen,
half cuts, disks, etc.) and supports lookup by:
- configuration id (legacy, 1-based),
- zero-based index within available predefined entries,
- human-readable label.
"""

from    __future__ import annotations

from    abc import ABC, abstractmethod
from    typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import  math

import  numpy as np

try:
    from .region            import KitaevPreskillRegion, LevinWenRegion, HalfRegions, DiskRegion, Region
    from ..lattice_tools    import LatticeType
except ImportError as exc:
    raise ImportError("This module relies on region definitions from the same package. Ensure package structure is intact.") from exc

# ----------------------------------------
#! Internal data structures and helper functions for managing the predefined region registry.
# ----------------------------------------

PredefinedKey                                           = Tuple[LatticeType, int, int, int, str, int]

# (lattice_type, Lx, Ly, Lz, kind, config) -> Region
PREDEFINED_REGIONS: Dict[PredefinedKey, Region]         = {}
# Same key -> metadata dictionary
PREDEFINED_META: Dict[PredefinedKey, Dict[str, Any]]    = {}

def _normalize_lattice_type(lattice_type: Union[LatticeType, str]) -> LatticeType:
    if isinstance(lattice_type, LatticeType):
        return lattice_type
    if isinstance(lattice_type, str):
        key     = lattice_type.strip().lower()
        mapping = {
            "square"        : LatticeType.SQUARE,
            "triangular"    : LatticeType.TRIANGULAR,
            "honeycomb"     : LatticeType.HONEYCOMB,
            "hexagonal"     : LatticeType.HEXAGONAL,
            "graph"         : LatticeType.GRAPH,
        }
        if key in mapping:
            return mapping[key]
    raise ValueError(f"Unknown lattice type: {lattice_type!r}")

def _normalize_kind(kind: Optional[str]) -> Optional[str]:
    if kind is None:
        return None
    k = str(kind).strip().lower()
    if "kitaev" in k or k.startswith("kp"):
        return "kitaev_preskill"
    if "levin" in k or k.startswith("lw"):
        return "levin_wen"
    if k in {"half", "half_x", "half-y", "half_y", "half-z", "half_z", "half-xy", "half_xy", "half-yx", "half_yx"}:
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

def _sorted_unique(values: Sequence[int]) -> List[int]:
    return sorted({int(v) for v in values})

def _site_index_multipartite(x: int, y: int, z: int, sub: int, lx: int, ly: int, lz: int, num_sub: int) -> int:
    return ((z * ly + y) * lx + x) * num_sub + sub

def _site_index_chain(x: int, lx: int) -> int:
    return x % lx

def _site_index_square(x: int, y: int, lx: int) -> int:
    return _site_index_multipartite(x, y, 0, 0, lx, 1, 1, 1)

def _site_index_honeycomb(x: int, y: int, sub: int, lx: int) -> int:
    return _site_index_multipartite(x, y, 0, sub, lx, 1, 1, 2)

def _as_meta_entry(key: PredefinedKey, *, index: int, include_region: bool = False) -> Dict[str, Any]:
    lattice_type, lx, ly, lz, kind, config = key
    meta = dict(PREDEFINED_META.get(key, {}))
    meta.update(
        {
            "index"         : index,
            "lattice_type"  : lattice_type,
            "lx"            : lx,
            "ly"            : ly,
            "lz"            : lz,
            "kind"          : kind,
            "configuration" : config,
        }
    )
    if include_region:
        meta["region"] = PREDEFINED_REGIONS[key]
    return meta

# ----------------------------------------
#! Public API for registering and looking up predefined regions.
# ----------------------------------------

def register_region(
    lattice_type    : Union[LatticeType, str],
    lx              : int,
    ly              : int,
    lz              : int,
    kind            : str,
    config          : int,
    region          : Region,
    *,
    label           : Optional[str] = None,
    tags            : Optional[Sequence[str]] = None,
    overwrite       : bool = True,
):
    """Register one predefined region entry."""
    lt          = _normalize_lattice_type(lattice_type)
    kind_norm   = _normalize_kind(kind)

    if kind_norm is None:
        raise ValueError("kind cannot be None")
    if int(config) < 1:
        raise ValueError("configuration id must be >= 1")

    key: PredefinedKey = (lt, int(lx), int(ly), int(lz), kind_norm, int(config))
    if not overwrite and key in PREDEFINED_REGIONS:
        raise KeyError(f"Predefined region already exists for key={key}")

    if getattr(region, "configuration", None) is None:
        region.configuration = int(config)

    PREDEFINED_REGIONS[key]     = region
    PREDEFINED_META[key]        = {
                                    "label": label or f"{kind_norm}_{int(config)}",
                                    "tags": tuple(tags or ()),
                                }

def list_predefined_regions(
    lattice_type    : Union[LatticeType, str],
    lx              : int,
    ly              : int,
    lz              : int,
    kind            : Optional[str] = None,
    *,
    include_region  : bool = False,
) -> List[Dict[str, Any]]:
    """List all predefined entries for a lattice size and optional kind."""
    lt          = _normalize_lattice_type(lattice_type)
    kind_norm   = _normalize_kind(kind)
    keys        = [
        key
        for key in PREDEFINED_REGIONS.keys()
        if key[0] == lt and key[1] == int(lx) and key[2] == int(ly) and key[3] == int(lz)
    ]
    if kind_norm is not None:
        keys = [key for key in keys if key[4] == kind_norm]

    keys.sort(key=lambda k: (k[4], k[5]))
    return [_as_meta_entry(k, index=i, include_region=include_region) for i, k in enumerate(keys)]

def get_predefined_region(
    lattice_type: Union[LatticeType, str],
    lx: int,
    ly: int,
    lz: int,
    kind: str,
    config: Optional[Union[int, str]] = None,
    *,
    index: Optional[int] = None,
    label: Optional[str] = None,
    return_meta: bool = False,
) -> Optional[Union[Region, Tuple[Region, Dict[str, Any]]]]:
    """
    Look up a predefined region.

    Priority:
    1) `label`
    2) `index` (0-based among matching entries)
    3) exact `config` (legacy)
    """
    lt = _normalize_lattice_type(lattice_type)
    kind_norm = _normalize_kind(kind)

    if isinstance(config, str) and label is None and index is None:
        cfg = config.strip()
        if cfg.isdigit():
            config = int(cfg)
        else:
            label = cfg

    if label is not None:
        entries = list_predefined_regions(lt, lx, ly, lz, kind=kind_norm, include_region=True)
        for entry in entries:
            if str(entry.get("label", "")).strip().lower() == label.strip().lower():
                region = entry["region"]
                if return_meta:
                    return region, entry
                return region
        return None

    if index is not None:
        entries = list_predefined_regions(lt, lx, ly, lz, kind=kind_norm, include_region=True)
        if 0 <= int(index) < len(entries):
            entry = entries[int(index)]
            region = entry["region"]
            if return_meta:
                return region, entry
            return region
        return None

    if config is None:
        config = 1

    key: PredefinedKey = (lt, int(lx), int(ly), int(lz), kind_norm, int(config))
    region = PREDEFINED_REGIONS.get(key)
    if region is None:
        return None

    if return_meta:
        meta = _as_meta_entry(key, index=-1, include_region=False)
        return region, meta
    return region


class RegionConfigurator(ABC):
    """Base class for lattice-specific predefined-region configurators."""

    @abstractmethod
    def register_all(self):
        pass


class _RectangularConfigurator(RegionConfigurator):
    """Shared generator for 1-site-per-cell rectangular lattices."""

    lattice_type    : LatticeType = LatticeType.SQUARE
    sizes           : Tuple[Tuple[int, int], ...] = ((3, 3), (4, 4), (5, 5), (6, 6))

    def _coords(self, lx: int, ly: int):
        xs, ys = np.meshgrid(np.arange(lx, dtype=float), np.arange(ly, dtype=float), indexing="xy")
        return xs, ys

    def _all_sites(self, lx: int, ly: int) -> List[int]:
        return list(range(lx * ly))

    def _disk_sites(self, lx: int, ly: int, cx: float, cy: float, radius: float) -> List[int]:
        xs, ys  = self._coords(lx, ly)
        d       = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        mask    = d <= float(radius)
        idx     = np.where(mask.ravel())[0].tolist()
        return _sorted_unique(idx)

    def _kp_sectors(self, lx: int, ly: int, cx: float, cy: float, radius: float, rotation: float = 0.0):
        xs, ys      = self._coords(lx, ly)
        dx          = xs - cx
        dy          = ys - cy
        d           = np.sqrt(dx ** 2 + dy ** 2)
        angles      = np.arctan2(dy, dx) - rotation
        angles      = (angles + np.pi) % (2 * np.pi) - np.pi

        inside      = d <= radius
        sector_w    = 2 * np.pi / 3.0
        A           = np.where(((angles >= -np.pi) & (angles < -np.pi + sector_w) & inside).ravel())[0].tolist()
        B           = np.where(((angles >= -np.pi + sector_w) & (angles < -np.pi + 2 * sector_w) & inside).ravel())[0].tolist()
        C           = np.where((inside.ravel() & ~(np.isin(np.arange(lx * ly), A + B))).ravel())[0].tolist()

        inside_sites = np.where(inside.ravel())[0].tolist()
        if inside_sites:
            for bucket in (A, B, C):
                if not bucket:
                    bucket.append(int(inside_sites[0]))

        return _sorted_unique(A), _sorted_unique(B), _sorted_unique(C)

    def _lw_partition(self, lx: int, ly: int, cx: float, cy: float, r_inner: float, r_outer: float):
        xs, ys  = self._coords(lx, ly)
        d       = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).ravel()

        A       = np.where(d <= r_inner)[0].tolist()
        B       = np.where((d > r_inner) & (d <= r_outer))[0].tolist()
        C       = np.where(d > r_outer)[0].tolist()

        all_sites = self._all_sites(lx, ly)
        if not A and all_sites:
            A = [all_sites[0]]

        # Fallback injections can create overlaps with precomputed shells;
        # enforce a strict partition A/B/C before filling missing pieces.
        aset    = set(A)
        B       = [s for s in B if s not in aset]
        if not B:
            remaining = [s for s in all_sites if s not in A]
            if remaining:
                B = [remaining[0]]

        bset = set(B)
        C = [s for s in C if s not in aset and s not in bset]
        if not C:
            remaining = [s for s in all_sites if s not in set(A) | set(B)]
            if remaining:
                C = [remaining[0]]
            elif B:
                C = [B.pop()]
            elif A:
                C = [A.pop()]

        return _sorted_unique(A), _sorted_unique(B), _sorted_unique(C)

    def _register_for_size(self, lx: int, ly: int):
        all_sites = self._all_sites(lx, ly)

        # HALF (two configurations)
        cfg = 1
        A   = [s for s in all_sites if (s % lx) < (lx / 2.0)]
        B   = [s for s in all_sites if s not in set(A)]
        register_region(
            self.lattice_type,
            lx,
            ly,
            1,
            "half",
            cfg,
            HalfRegions(A=_sorted_unique(A), B=_sorted_unique(B), C=[]),
            label="half_x",
            tags=("generated", "half", "x"),
        )
        cfg += 1
        A   = [s for s in all_sites if (s // lx) < (ly / 2.0)]
        B   = [s for s in all_sites if s not in set(A)]
        register_region(
            self.lattice_type,
            lx,
            ly,
            1,
            "half",
            cfg,
            HalfRegions(A=_sorted_unique(A), B=_sorted_unique(B), C=[]),
            label="half_y",
            tags=("generated", "half", "y"),
        )
        
        ################################
        # DISK (two configurations)
        ################################
        center  = ((lx - 1) / 2.0, (ly - 1) / 2.0)
        r_small = max(0.75, min(lx, ly) / 4.0)
        r_large = max(r_small + 0.75, min(lx, ly) / 3.0)

        cfg     = 1
        A       = self._disk_sites(lx, ly, center[0], center[1], r_small)
        if not A:
            A = [all_sites[len(all_sites) // 2]]
        B = [s for s in all_sites if s not in set(A)]
        register_region(
            self.lattice_type,
            lx,
            ly,
            1,
            "disk",
            cfg,
            DiskRegion(A=_sorted_unique(A), B=_sorted_unique(B), C=[]),
            label="disk_center_small",
            tags=("generated", "disk", "center"),
        )
        cfg += 1
        A   = self._disk_sites(lx, ly, center[0] + 0.35, center[1] - 0.35, r_large)
        B   = [s for s in all_sites if s not in set(A)]
        register_region(
            self.lattice_type,
            lx,
            ly,
            1,
            "disk",
            cfg,
            DiskRegion(A=_sorted_unique(A), B=_sorted_unique(B), C=[]),
            label="disk_shifted_large",
            tags=("generated", "disk", "shifted"),
        )
        
        ################################
        # LEVIN-WEN (two configurations)
        ################################
        cfg     = 1
        A, B, C = self._lw_partition(lx, ly, center[0], center[1], r_small * 0.8, r_large)
        register_region(
            self.lattice_type,
            lx,
            ly,
            1,
            "levin_wen",
            cfg,
            LevinWenRegion(A=A, B=B, C=C),
            label="lw_center",
            tags=("generated", "levin_wen", "center"),
        )
        cfg    += 1
        A, B, C = self._lw_partition(lx, ly, center[0] - 0.5, center[1] + 0.25, r_small * 0.7, r_large * 0.9)
        register_region(
            self.lattice_type,
            lx,
            ly,
            1,
            "levin_wen",
            cfg,
            LevinWenRegion(A=A, B=B, C=C),
            label="lw_shifted",
            tags=("generated", "levin_wen", "shifted"),
        )

        ################################
        # KITAEV-PRESKILL (two configurations)
        ################################
        
        cfg     = 1
        A, B, C = self._kp_sectors(lx, ly, center[0], center[1], max(r_large, 1.25), rotation=0.0)
        register_region(
            self.lattice_type,
            lx,
            ly,
            1,
            "kitaev_preskill",
            cfg,
            KitaevPreskillRegion(A=A, B=B, C=C),
            label="kp_center",
            tags=("generated", "kitaev_preskill", "center"),
        )
        cfg    += 1
        A, B, C = self._kp_sectors(lx, ly, center[0], center[1], max(r_large, 1.25), rotation=math.pi / 6.0)
        register_region(
            self.lattice_type,
            lx,
            ly,
            1,
            "kitaev_preskill",
            cfg,
            KitaevPreskillRegion(A=A, B=B, C=C),
            label="kp_rotated",
            tags=("generated", "kitaev_preskill", "rotated"),
        )

    def register_all(self):
        for lx, ly in self.sizes:
            self._register_for_size(lx, ly)


class SquareRegionConfigurator(_RectangularConfigurator):
    lattice_type    = LatticeType.SQUARE
    sizes           = ((3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5))

class TriangularRegionConfigurator(_RectangularConfigurator):
    lattice_type    = LatticeType.TRIANGULAR
    sizes           = ((3, 3), (4, 4), (5, 5))


class HoneycombRegionConfigurator(RegionConfigurator):
    """Generated predefined regions for honeycomb lattices (2 sites per cell)."""

    sizes: Tuple[Tuple[int, int], ...] = ((3, 3), (3, 4), (4, 3), (4, 4), (4,5), (5,4), (5, 5))

    def _cell_centers(self, lx: int, ly: int):
        xs, ys = np.meshgrid(np.arange(lx, dtype=float), np.arange(ly, dtype=float), indexing="xy")
        return xs, ys

    def _all_sites(self, lx: int, ly: int) -> List[int]:
        return list(range(2 * lx * ly))

    def _sites_from_cell_mask(self, mask: np.ndarray, lx: int) -> List[int]:
        sites: List[int]    = []
        ys, xs              = np.where(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            sites.append(_site_index_honeycomb(int(x), int(y), 0, lx))
            sites.append(_site_index_honeycomb(int(x), int(y), 1, lx))
        return _sorted_unique(sites)

    def _kp_cells(self, lx: int, ly: int, cx: float, cy: float, radius: float, rotation: float = 0.0):
        ''' Partition cells into three sectors for Kitaev-Preskill construction, then assign sites accordingly.'''
        xs, ys          = self._cell_centers(lx, ly)
        dx              = xs - cx
        dy              = ys - cy
        d               = np.sqrt(dx ** 2 + dy ** 2)
        angles          = np.arctan2(dy, dx) - rotation
        angles          = (angles + np.pi) % (2 * np.pi) - np.pi
        inside          = d <= radius
        sector_w        = 2 * np.pi / 3.0

        A               = self._sites_from_cell_mask((angles >= -np.pi) & (angles < -np.pi + sector_w) & inside, lx)
        B               = self._sites_from_cell_mask((angles >= -np.pi + sector_w) & (angles < -np.pi + 2 * sector_w) & inside, lx)
        used            = set(A) | set(B)
        C               = [s for s in self._sites_from_cell_mask(inside, lx) if s not in used]

        inside_sites = self._sites_from_cell_mask(inside, lx)
        if inside_sites:
            for bucket in (A, B, C):
                if not bucket:
                    bucket.append(int(inside_sites[0]))

        return _sorted_unique(A), _sorted_unique(B), _sorted_unique(C)

    def _lw_partition(self, lx: int, ly: int, cx: float, cy: float, r_inner: float, r_outer: float):
        xs, ys  = self._cell_centers(lx, ly)
        d       = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        A       = self._sites_from_cell_mask(d <= r_inner, lx)
        B       = self._sites_from_cell_mask((d > r_inner) & (d <= r_outer), lx)
        C       = self._sites_from_cell_mask(d > r_outer, lx)

        all_sites = self._all_sites(lx, ly)
        if not A and all_sites:
            A = [all_sites[0]]

        # Keep A/B/C disjoint after fallback adjustments.
        aset = set(A)
        B = [s for s in B if s not in aset]
        if not B:
            remaining = [s for s in all_sites if s not in A]
            if remaining:
                B = [remaining[0]]

        bset = set(B)
        C = [s for s in C if s not in aset and s not in bset]
        if not C:
            remaining = [s for s in all_sites if s not in set(A) | set(B)]
            if remaining:
                C = [remaining[0]]
            elif B:
                C = [B.pop()]
            elif A:
                C = [A.pop()]

        return _sorted_unique(A), _sorted_unique(B), _sorted_unique(C)

    def _register_for_size(self, lx: int, ly: int):
        all_sites = self._all_sites(lx, ly)
        xs, ys = self._cell_centers(lx, ly)

        # HALF
        cfg = 1
        A = self._sites_from_cell_mask(xs < (lx / 2.0), lx)
        B = [s for s in all_sites if s not in set(A)]
        register_region(
            LatticeType.HONEYCOMB,
            lx,
            ly,
            1,
            "half",
            cfg,
            HalfRegions(A=_sorted_unique(A), B=_sorted_unique(B), C=[]),
            label="half_x",
            tags=("generated", "half", "x"),
        )
        cfg += 1
        A = self._sites_from_cell_mask(ys < (ly / 2.0), lx)
        B = [s for s in all_sites if s not in set(A)]
        register_region(
            LatticeType.HONEYCOMB,
            lx,
            ly,
            1,
            "half",
            cfg,
            HalfRegions(A=_sorted_unique(A), B=_sorted_unique(B), C=[]),
            label="half_y",
            tags=("generated", "half", "y"),
        )

        center = ((lx - 1) / 2.0, (ly - 1) / 2.0)
        r_small = max(0.75, min(lx, ly) / 4.0)
        r_large = max(r_small + 0.75, min(lx, ly) / 3.0)

        # DISK
        cfg = 1
        A = self._sites_from_cell_mask(np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2) <= r_small, lx)
        if not A:
            A = [all_sites[len(all_sites) // 2]]
        B = [s for s in all_sites if s not in set(A)]
        register_region(
            LatticeType.HONEYCOMB,
            lx,
            ly,
            1,
            "disk",
            cfg,
            DiskRegion(A=_sorted_unique(A), B=_sorted_unique(B), C=[]),
            label="disk_center_small",
            tags=("generated", "disk", "center"),
        )
        cfg += 1
        A = self._sites_from_cell_mask(np.sqrt((xs - (center[0] + 0.25)) ** 2 + (ys - (center[1] - 0.25)) ** 2) <= r_large, lx)
        B = [s for s in all_sites if s not in set(A)]
        register_region(
            LatticeType.HONEYCOMB,
            lx,
            ly,
            1,
            "disk",
            cfg,
            DiskRegion(A=_sorted_unique(A), B=_sorted_unique(B), C=[]),
            label="disk_shifted_large",
            tags=("generated", "disk", "shifted"),
        )

        # LEVIN-WEN
        cfg = 1
        A, B, C = self._lw_partition(lx, ly, center[0], center[1], r_small * 0.75, r_large)
        register_region(
            LatticeType.HONEYCOMB,
            lx,
            ly,
            1,
            "levin_wen",
            cfg,
            LevinWenRegion(A=A, B=B, C=C),
            label="lw_center",
            tags=("generated", "levin_wen", "center"),
        )

        # KITAEV-PRESKILL
        cfg = 1
        A, B, C = self._kp_cells(lx, ly, center[0], center[1], max(r_large, 1.25), rotation=0.0)
        register_region(
            LatticeType.HONEYCOMB,
            lx,
            ly,
            1,
            "kitaev_preskill",
            cfg,
            KitaevPreskillRegion(A=A, B=B, C=C),
            label="kp_center",
            tags=("generated", "kitaev_preskill", "center"),
        )
        cfg += 1
        A, B, C = self._kp_cells(lx, ly, center[0], center[1], max(r_large, 1.25), rotation=math.pi / 6.0)
        register_region(
            LatticeType.HONEYCOMB,
            lx,
            ly,
            1,
            "kitaev_preskill",
            cfg,
            KitaevPreskillRegion(A=A, B=B, C=C),
            label="kp_rotated",
            tags=("generated", "kitaev_preskill", "rotated"),
        )

    def register_all(self):
        for lx, ly in self.sizes:
            self._register_for_size(lx, ly)

# ----------------------------------------
#! Initialization of the predefined region registry with curated entries for supported lattice types and sizes.
# ----------------------------------------

def initialize_predefined_regions():
    """Populate registry using all configurators (idempotent re-init)."""
    PREDEFINED_REGIONS.clear()
    PREDEFINED_META.clear()

    configurators: List[RegionConfigurator] = [
        SquareRegionConfigurator(),
        TriangularRegionConfigurator(),
        HoneycombRegionConfigurator(),
    ]
    for configurator in configurators:
        configurator.register_all()


# Initialize on module load.
initialize_predefined_regions()

# ------------------------
#! EOF
# ------------------------