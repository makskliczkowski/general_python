"""
This module defines a registry of predefined regions for various lattice types and configurations. It
stores specific site index sets for regions like Kitaev-Preskill, Levin-Wen, half-regions, and disk regions
on common lattice geometries (e.g., square, honeycomb). The registry allows users to easily retrieve
these standard regions by specifying the lattice type, dimensions, region kind, and configuration number.

It uses configurator classes to manage regions in a controllable manner.

------------------------------
Author          : Maksymilian Kliczkowski
Created         : 2026-02-15
-------------------------------
"""

from __future__ import annotations
from abc        import ABC, abstractmethod
from typing     import Dict, List, Tuple, Optional
try:
    from . import KitaevPreskillRegion, LevinWenRegion, HalfRegions, DiskRegion, Region
    from ..lattice_tools import LatticeType
        
except ImportError:
    raise ImportError("This module relies on region definitions from the same package. Ensure that the package is properly structured and all dependencies are available.")

# Registry for predefined regions: (lattice_type, Lx, Ly, Lz, kind, configuration) -> Region
# Site indices are integers
PREDEFINED_REGIONS: Dict[Tuple[LatticeType, int, int, int, str, int], Region] = {}

def register_region(lattice_type: LatticeType, lx: int, ly: int, lz: int, kind: str, config: int, region: Region):
    PREDEFINED_REGIONS[(lattice_type, lx, ly, lz, kind, config)] = region

# ═══════════════════════════════════════════════════════════════════════════
#! Configurator Classes
# ═══════════════════════════════════════════════════════════════════════════

class RegionConfigurator(ABC):
    """
    Base class for lattice-specific region configurators.
    """
    @abstractmethod
    def register_all(self):
        """Register all predefined regions for this lattice type."""
        pass

# ----------------------------------------------------------------------------

class SquareRegionConfigurator(RegionConfigurator):
    """
    Predefined regions for Square lattices.
    """
    def register_all(self):
        self._register_3x3()
        self._register_4x4()

    def _register_3x3(self):
        # 1) Kitaev-Preskill 
        
        #? 1)
        register_region(
            LatticeType.SQUARE, 3, 3, 1, "kitaev_preskill", 1,
            KitaevPreskillRegion(
                A=[0, 1, 2, 3, 6],
                B=[
                C=[3, 4, 6, 7],
                AB=[0, 1, 2, 3, 4, 5],
                AC=[0, 1, 3, 4, 6, 7],
                BC=[1, 2, 3, 4, 5, 6, 7],
                ABC=[0, 1, 2, 3, 4, 5, 6, 7],
                configuration=1
            )
        )
        
        # 3x3 Disk Config 1
        register_region(
            LatticeType.SQUARE, 3, 3, 1, "disk", 1,
            DiskRegion(
                A=[4],
                B=[0, 1, 2, 3, 5, 6, 7, 8],
                C=[],
                AB=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                AC=[4],
                BC=[0, 1, 2, 3, 5, 6, 7, 8],
                ABC=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                configuration=1
            )
        )

        # 3x3 Levin-Wen Config 1
        # A: site 4 (inner disk)
        # B: sites 1, 3, 5, 7 (annulus)
        # C: sites 0, 2, 6, 8 (exterior)
        register_region(
            LatticeType.SQUARE, 3, 3, 1, "levin_wen", 1,
            LevinWenRegion(
                A=[4],
                B=[1, 3, 5, 7],
                C=[0, 2, 6, 8],
                AB=[1, 3, 4, 5, 7],
                AC=[0, 2, 4, 6, 8],
                BC=[0, 1, 2, 3, 5, 6, 7, 8],
                ABC=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                configuration=1
            )
        )

    def _register_4x4(self):
        # 4x4 KP Config 1
        A = [0, 1, 4, 5, 8, 9]
        B = [1, 2, 5, 6, 9, 10]
        C = [4, 5, 6, 8, 9, 10, 12, 13, 14]
        
        union_ab  = sorted(list(set(A) | set(B)))
        union_ac  = sorted(list(set(A) | set(C)))
        union_bc  = sorted(list(set(B) | set(C)))
        union_abc = sorted(list(set(A) | set(B) | set(C)))
        
        register_region(
            LatticeType.SQUARE, 4, 4, 1, "kitaev_preskill", 1,
            KitaevPreskillRegion(
                A=A, B=B, C=C, AB=union_ab, AC=union_ac, BC=union_bc, ABC=union_abc,
                configuration=1
            )
        )

# ----------------------------------------------------------------------------

class HoneycombRegionConfigurator(RegionConfigurator):
    """
    Predefined regions for Honeycomb lattices.
    """
    def register_all(self):
        self._register_3x3()

    def _register_3x3(self):
        register_region(
            LatticeType.HONEYCOMB, 3, 3, 1, "kitaev_preskill", 1,
            KitaevPreskillRegion(
                A=[0, 1, 2, 3],
                B=[2, 3, 4, 5],
                C=[1, 3, 6, 8],
                AB=[0, 1, 2, 3, 4, 5],
                AC=[0, 1, 2, 3, 6, 8],
                BC=[1, 2, 3, 4, 5, 6, 8],
                ABC=[0, 1, 2, 3, 4, 5, 6, 8],
                configuration=1
            )
        )

# ═══════════════════════════════════════════════════════════════════════════
#! Initialization
# ═══════════════════════════════════════════════════════════════════════════

def initialize_predefined_regions():
    """Populate the registry using configurators."""
    from ..lattice_tools import LatticeType
    configurators = [
        SquareRegionConfigurator(),
        HoneycombRegionConfigurator(),
    ]
    for conf in configurators:
        conf.register_all()

# Initialize on module load
initialize_predefined_regions()

# ----------------------------------------------------------------------------
#! Helper functions
# ----------------------------------------------------------------------------

def get_predefined_region(lattice_type: LatticeType, lx: int, ly: int, lz: int, kind: str, config: int) -> Optional[Region]:
    """
    Look up a predefined region in the registry.
    """
    # Normalize kind
    kind = kind.lower()
    if "kitaev" in kind or "kp" in kind:
        kind = "kitaev_preskill"
    elif "levin" in kind or "lw" in kind:
        kind = "levin_wen"
    elif "half" in kind:
        kind = "half"
    elif "disk" in kind:
        kind = "disk"
    
    return PREDEFINED_REGIONS.get((lattice_type, lx, ly, lz, kind, config))
