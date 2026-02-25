"""Public exports for region containers and predefined-region registry."""

from __future__ import annotations

import importlib

from .region import (
    Region,
    KitaevPreskillRegion,
    LevinWenRegion,
    HalfRegions,
    DiskRegion,
    PlaquetteRegion,
    CustomRegion,
    RegionFraction,
)


# Short aliases for frequently-used region classes.
KPRegion = KitaevPreskillRegion
LWRegion = LevinWenRegion


def _predefined_module():
    return importlib.import_module(f"{__name__}.predefined")


def register_region(*args, **kwargs):
    return _predefined_module().register_region(*args, **kwargs)


def initialize_predefined_regions(*args, **kwargs):
    return _predefined_module().initialize_predefined_regions(*args, **kwargs)


def get_predefined_region(*args, **kwargs):
    return _predefined_module().get_predefined_region(*args, **kwargs)


def list_predefined_regions(*args, **kwargs):
    return _predefined_module().list_predefined_regions(*args, **kwargs)


def __getattr__(name: str):
    if name in {"PREDEFINED_REGIONS", "PREDEFINED_META"}:
        return getattr(_predefined_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["PREDEFINED_REGIONS", "PREDEFINED_META"])


__all__ = [
    "Region",
    "KitaevPreskillRegion",
    "LevinWenRegion",
    "HalfRegions",
    "DiskRegion",
    "PlaquetteRegion",
    "CustomRegion",
    "RegionFraction",
    "KPRegion",
    "LWRegion",
    "register_region",
    "initialize_predefined_regions",
    "get_predefined_region",
    "list_predefined_regions",
    "PREDEFINED_REGIONS",
    "PREDEFINED_META",
]
