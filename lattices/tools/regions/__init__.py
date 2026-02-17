'''
This folder contains predefined regions for the use in the region_handler. Those are most common 
regions for different lattices, depending on the system size and lattice type (e.g. square lattice, honeycomb lattice, etc.).

----------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Created         : 2026-02-15
----------------------------------------------------------------
'''

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# --------------------------------------------------------------
#! Different region types
# --------------------------------------------------------------

@dataclass
class Region:
    """Base class for defining a region on a lattice."""

    A             : List[int]       # List of site indices for region A
    B             : List[int]       # List of site indices for region B
    C             : List[int]       # List of site indices for region C - for bipartite lattices, this can be empty
    AB            : List[int]       = field(default_factory=list)   # List of site indices for region AB (A union B)
    AC            : List[int]       = field(default_factory=list)   # List of site indices for region AC (A union C)
    BC            : List[int]       = field(default_factory=list)   # List of site indices for region BC (B union C)
    ABC           : List[int]       = field(default_factory=list)   # List of site indices for region ABC (A union B union C)
    configuration : Optional[int]   = None                          # Configuration index for the region
    
    def to_dict(self) -> Dict[str, List[int]]:
        """Convert the region to a dictionary format."""
        return {field.name.upper(): getattr(self, field.name) for field in self.__dataclass_fields__.values() 
                if field.name not in ["configuration"]}

    def bipartite(self) -> bool:
        """Check if the region is bipartite (i.e., C is empty)."""
        return len(self.C) == 0
    
    def tripartite(self) -> bool:
        """Check if the region is tripartite (i.e., C is non-empty)."""
        return len(self.C) > 0
    
    # --------------------------------------------------------------
    #! Dictionary-like behavior
    # --------------------------------------------------------------

    def __getitem__(self, key: str) -> List[int]:
        """Allow dictionary-style access to region components."""
        key = key.upper()
        if key in ["A", "B", "C", "AB", "AC", "BC", "ABC"]:
            return getattr(self, key)
        raise KeyError(f"Invalid region key: {key}")
    
    def __iter__(self):
        """Allow iteration over the region's keys."""
        return (field.name.upper() for field in self.__dataclass_fields__.values() 
                if field.name not in ["configuration"])
        
    def items(self):
        """Return an iterator over the region's items (key-value pairs)."""
        return ((field.name.upper(), getattr(self, field.name)) for field in self.__dataclass_fields__.values() 
                if field.name not in ["configuration"])
        
    def values(self):
        """Return a list of the region's values (lists of coordinates)."""
        return [getattr(self, field.name) for field in self.__dataclass_fields__.values() 
                if field.name not in ["configuration"]]
        
    def get(self, key: str, default: Any = None) -> Any:
        """Allow dictionary-style get access."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Return the keys of the region (A, B, C, AB, AC, BC, ABC)."""
        return ["A", "B", "C", "AB", "AC", "BC", "ABC"]

    # --------------------------------------------------------------
    #! Inner methods
    # --------------------------------------------------------------
    
    def __post_init__(self):
        """Ensure that the union of A, B, and C matches the union of AB, AC, BC, and ABC."""
        all_sites = set(self.A) | set(self.B) | set(self.C)
        
        # Get the unions if they are not provided
        if not self.AB:
            self.AB = sorted(list(set(self.A) | set(self.B)))
        if not self.AC:
            self.AC = sorted(list(set(self.A) | set(self.C)))
        if not self.BC:
            self.BC = sorted(list(set(self.B) | set(self.C)))
        if not self.ABC:
            self.ABC = sorted(list(set(self.A) | set(self.B) | set(self.C)))
        
        combined_sites = set(self.AB) | set(self.AC) | set(self.BC) | set(self.ABC)
        if all_sites != combined_sites:
            raise ValueError("The union of A, B, and C must match the union of AB, AC, BC, and ABC.")
        
    def __str__(self) -> str:
        """String representation of the region."""
        return f"Region(A={len(self.A)} sites, B={len(self.B)} sites, C={len(self.C)} sites)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the region."""
        return f"Region(A={self.A}, B={self.B}, C={self.C}, AB={self.AB}, AC={self.AC}, BC={self.BC}, ABC={self.ABC})"
    
    def __del__(self):
        """Clean up resources if necessary."""
        pass
    
    def __getattr__(self, name: str) -> Optional[List[int]]:
        """Allow access to region components as attributes."""
        
        # Convert the attribute name to uppercase to match the field names
        name = name.upper()
        
        if name in self.__dataclass_fields__:
            return getattr(self, name)
        raise AttributeError(f"'Region' object has no attribute '{name}'")
    
    def summary(self) -> str:
        """ Provide a summary of the region's properties. """
        return f"Region Summary: A={len(self.A)} sites, B={len(self.B)} sites, C={len(self.C)} sites, AB={len(self.AB)} sites, AC={len(self.AC)} sites, BC={len(self.BC)} sites, ABC={len(self.ABC)} sites"
    
# --------------------------------------------------------------

@dataclass
class KitaevPreskillRegion(Region):
    '''
    Kitaev-Preskill region is a specific tripartite region used in the calculation of topological entanglement entropy.
    It consists of three regions A, B, and C that are arranged such that their union forms a disk, 
    and each pair of regions intersects in a way that allows for the extraction of the
    topological entanglement entropy from the combination of their entanglement entropies.
    
    See: region_handler.py for more details on how to use this region in the context of entanglement entropy calculations.
    '''
    
    def __post_init__(self):
        """Ensure that the region is tripartite (i.e., C is non-empty)."""
        if len(self.C) == 0:
            raise ValueError("Kitaev-Preskill region must be tripartite (C cannot be empty).")
        
        # make sure all regions are disjoint
        if (set(self.A) & set(self.B)) or (set(self.A) & set(self.C)) or (set(self.B) & set(self.C)):
            raise ValueError("Kitaev-Preskill region must have disjoint A, B, and C regions.")
        
    def __str__(self) -> str:
        """String representation of the Kitaev-Preskill region."""
        return f"KitaevPreskillRegion(A={len(self.A)} sites, B={len(self.B)} sites, C={len(self.C)} sites)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the Kitaev-Preskill region."""
        return f"KitaevPreskillRegion(A={self.A}, B={self.B}, C={self.C})"
    
@dataclass
class LevinWenRegion(Region):
    '''
    Levin-Wen region is a specific bipartite region used in the calculation of topological entanglement entropy.
    It consists of two regions A and B that are arranged such that their union forms a disk, 
    and their intersection allows for the extraction of the topological entanglement entropy from the combination of their entanglement entropies.
    
    See: region_handler.py for more details on how to use this region in the context of entanglement entropy calculations.
    '''
    
    @property
    def annulus(self) -> List[int]:
        """Return the site indices of the annulus region (A union B)."""
        return self.B
    
    @property
    def inner(self) -> List[int]:
        """Return the site indices of the inner region (A)."""
        return self.A
    
    @property
    def exterior(self) -> List[int]:
        """Return the site indices of the exterior region (not A or B)."""
        return self.C
    
    def __post_init__(self):
        """ C cannot be empty for a Levin-Wen region, as it represents the exterior of the annulus formed by A and B. """
        if len(self.C) == 0:
            raise ValueError("Levin-Wen region must have a non-empty exterior (C).")
        
        # make sure all regions are disjoint
        if (set(self.A) & set(self.B)) or (set(self.A) & set(self.C)) or (set(self.B) & set(self.C)):
            raise ValueError("Levin-Wen region must have disjoint A, B, and C regions.")
        
        super().__post_init__()  # Call the base class post-init to ensure consistency of unions
        
    def __str__(self) -> str:
        """String representation of the Levin-Wen region."""
        return f"LevinWenRegion(inner={len(self.A)} sites, annulus={len(self.B)} sites, exterior={len(self.C)} sites)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the Levin-Wen region."""
        return f"LevinWenRegion(inner={self.A}, annulus={self.B}, exterior={self.C})"
    
@dataclass
class HalfRegions(Region):
    '''
    Half regions are a specific type of bipartite region where the lattice is divided into two equal halves, A and B.
    This can be useful for studying entanglement properties across a simple bipartition of the system.
    
    See: region_handler.py for more details on how to use this region in the context of entanglement entropy calculations.
    '''
    
    def __post_init__(self):
        """Ensure that the region is bipartite (i.e., C is empty)."""
        if len(self.C) > 0:
            raise ValueError("Half region must be bipartite (C must be empty).")
        
        super().__post_init__()  # Call the base class post-init to ensure consistency of unions

    def __str__(self) -> str:
        """String representation of the half region."""
        return f"HalfRegion(A={len(self.A)} sites, B={len(self.B)} sites)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the half region."""
        return f"HalfRegion(A={self.A}, B={self.B})"
    
@dataclass
class DiskRegion(Region):
    '''
    Disk region is a specific type of region where the sites are arranged in a disk-like shape. 
    This can be useful for studying entanglement properties in a more localized manner, as the disk region can capture local correlations more effectively than larger, more complex regions.
    
    See: region_handler.py for more details on how to use this region in the context of entanglement entropy calculations.
    '''
    
    def __post_init__(self):
        """ Bipartite regions only """
        if len(self.C) > 0:
            raise ValueError("Disk region must be bipartite (C must be empty).")
        
        super().__post_init__()  # Call the base class post-init to ensure consistency of unions
    
    def __str__(self) -> str:
        """String representation of the disk region."""
        return f"DiskRegion(A={len(self.A)} sites, B={len(self.B)} sites, C={len(self.C)} sites)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the disk region."""
        return f"DiskRegion(A={self.A}, B={self.B}, C={self.C})"
    
@dataclass
class PlaquetteRegion(Region):
    '''
    Plaquette region is a specific type of region where the sites are arranged in a plaquette-like shape. 
    This can be useful for studying entanglement properties in a more localized manner, as the plaquette region can capture local correlations more effectively than larger, more complex regions.
    
    See: region_handler.py for more details on how to use this region in the context of entanglement entropy calculations.
    '''
    
    def __post_init__(self):
        """ Bipartite regions only """
        if len(self.C) > 0:
            raise ValueError("Plaquette region must be bipartite (C must be empty).")
    
        super().__post_init__()  # Call the base class post-init to ensure consistency of unions
    
    def __str__(self) -> str:
        """String representation of the plaquette region."""
        return f"PlaquetteRegion(A={len(self.A)} sites, B={len(self.B)} sites, C={len(self.C)} sites)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the plaquette region."""
        return f"PlaquetteRegion(A={self.A}, B={self.B}, C={self.C})"

# --------------------------------------------------------------
#! Different lattices
# --------------------------------------------------------------