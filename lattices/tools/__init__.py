"""Public lattice-tool exports for convenient one-line imports.

Examples
--------
from QES.general_python.lattices.tools import (
    LatticeType, LatticeBC, BoundaryFlux,
    LatticeRegionHandler, RegionType,
    Region, get_predefined_region,
)
"""

import importlib

from .lattice_tools import (
    LatticeDirection,
    LatticeBC,
    LatticeType,
    handle_twist_flux,
    handle_boundary_conditions,
    handle_boundary_conditions_detailed,
    handle_dim,
)
from .lattice_flux import BoundaryFlux
from .lattice_symmetry import (
    generate_translation_perms,
    generate_point_group_perms_square,
    generate_space_group_perms,
    compute_cayley_table,
)
from .region_handler import RegionType, LatticeRegionHandler

_LAZY_REGION_EXPORTS = {
    "Region"                        : (".regions", "Region"),
    "KitaevPreskillRegion"          : (".regions", "KitaevPreskillRegion"),
    "LevinWenRegion"                : (".regions", "LevinWenRegion"),
    "HalfRegions"                   : (".regions", "HalfRegions"),
    "DiskRegion"                    : (".regions", "DiskRegion"),
    "PlaquetteRegion"               : (".regions", "PlaquetteRegion"),
    "CustomRegion"                  : (".regions", "CustomRegion"),
    "KPRegion"                      : (".regions", "KPRegion"),
    "LWRegion"                      : (".regions", "LWRegion"),
    "register_region"               : (".regions", "register_region"),
    "initialize_predefined_regions" : (".regions", "initialize_predefined_regions"),
    "get_predefined_region"         : (".regions", "get_predefined_region"),
    "list_predefined_regions"       : (".regions", "list_predefined_regions"),
}

__all__ = [
    # Basic lattice enums and normalization helpers
    "LatticeDirection",
    "LatticeBC",
    "LatticeType",
    "handle_twist_flux",
    "handle_boundary_conditions",
    "handle_boundary_conditions_detailed",
    "handle_dim",
    # Flux container
    "BoundaryFlux",
    # Symmetry helpers
    "generate_translation_perms",
    "generate_point_group_perms_square",
    "generate_space_group_perms",
    "compute_cayley_table",
    # Region handler
    "RegionType",
    "LatticeRegionHandler",
    # Region containers and registry
    "Region",
    "KitaevPreskillRegion",
    "LevinWenRegion",
    "HalfRegions",
    "DiskRegion",
    "PlaquetteRegion",
    "CustomRegion",
    "KPRegion",
    "LWRegion",
    "register_region",
    "initialize_predefined_regions",
    "get_predefined_region",
    "list_predefined_regions",
]

def __getattr__(name: str):
    if name in _LAZY_REGION_EXPORTS:
        module_name, attr_name = _LAZY_REGION_EXPORTS[name]
        module = importlib.import_module(module_name, package=__name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_REGION_EXPORTS.keys()))

# ----------------------------------------------
#! EOF
# ----------------------------------------------
