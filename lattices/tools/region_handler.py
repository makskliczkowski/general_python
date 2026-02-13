"""
Region handler for lattice geometries.

This module provides the LatticeRegionHandler class, which encapsulates
methods for defining and extracting spatial regions on a lattice.
It supports various region types including:
- Half-system cuts
- Disk/circular regions
- Graph-distance balls
- Sublattice selections
- Plaquette-based regions
- Topological sectors (Kitaev-Preskill, Levin-Wen)

-------------------------------------------------------------------------------
File    : general_python/lattices/tools/region_handler.py
Author  : Maksymilian Kliczkowski
Date    : 2025-12-30
-------------------------------------------------------------------------------
"""

import  numpy   as np
from    typing  import List, Optional, Union, Dict, Any, TYPE_CHECKING
from    enum    import Enum, auto

class RegionType(Enum):
    HALF            = auto()    # Half-system cut
    HALF_X          = auto()    # Half-system cut along X direction
    HALF_Y          = auto()    # Half-system cut along Y direction
    HALF_Z          = auto()    # Half-system cut along Z direction
    # Other geometric regions
    DISK            = auto()    # Disk/circular region
    SUBLATTICE      = auto()    # Sublattice selection
    GRAPH           = auto()    # Graph-distance ball
    PLAQUETTE       = auto()    # Plaquette-based region
    # Topological sectors
    KITAEV_PRESKIL  = auto()    # Kitaev-Preskill sectors
    LEVIN_WEN       = auto()    # Levin-Wen annular sectors
    # Custom region
    CUSTOM          = auto()    # Custom list of sites
    
    def __str__(self):
        return self.name.lower()
    
    def __repr__(self):
        return f"RegionType.{self.name}"

if TYPE_CHECKING:
    from general_python.lattices.lattice import Lattice

class LatticeRegionHandler:
    r"""
    Handles region definitions and extractions for a Lattice instance.
    
    This class isolates the logic for selecting subsets of sites based on
    geometric or topological criteria. We want to be able to extract the following types of regions:
    - Half-system cuts (e.g. left half of the lattice)
    - Disk/circular regions defined by a center and radius
    - Graph-distance balls (sites within a certain graph distance from a center)
    - Sublattice selections (e.g. all sites belonging to a specific sublattice)
    - Plaquette-based regions (sites belonging to specified plaquettes)
    - Topological sectors:
        - Kitaev-Preskill regions A, B, C defined by angular sectors around a point
        - Levin-Wen regions defined by concentric annuli around a point
        
    The main method is `get_region()`, which takes a `kind` argument to specify the type of region and additional parameters as needed.
    The output is a list of site indices that belong to the defined region.
    """
    
    def __init__(self, lattice: "Lattice"):
        """
        Initialize the region handler.
        
        Parameters
        ----------
        lattice : Lattice
            The lattice instance to operate on.
        """
        self.lattice : "Lattice" = lattice

    # ------------------------------------------------------------------------------
    # Helper methods for region calculations
    # ------------------------------------------------------------------------------

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
                L = dims[d]
                dn[d] = dn[d] - L * np.round(dn[d] / L)
                
        disp = dn[0] * self.lattice.a1 + dn[1] * self.lattice.a2 + dn[2] * self.lattice.a3
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
    ) -> List[int]:
        r"""
        Return a list of site indices defining a spatial region.
        
        Parameters
        ----------
        kind : str or RegionType
            Type of region: 'half', 'disk', 'sublattice', 'graph', 'plaquette', 
            'kitaev_preskill', 'custom'. We also support specific half cuts like 'half_x', 'half_y', 'half_z'.
        origin : int or list[float], optional
            Center of the region. Can be a site index or coordinate vector.
        radius : float, optional
            Radius for 'disk' regions.
        direction : str, optional
            Direction for 'half' cuts ('x', 'y', 'z').
        sublattice : int, optional
            Sublattice index for 'sublattice' regions.
        sites : list[int], optional
            Explicit list of sites for 'custom' regions.
        depth : int, optional
            Depth/distance for 'graph' regions.
        plaquettes : list[int], optional
            List of plaquette indices for 'plaquette' regions.
            
        Returns
        -------
        list[int]
            Sorted list of site indices belonging to the region.
            
        Examples
        --------
        >>> half_x_sites = lattice.regions.get_region(kind='half_x')
        ... [10, 11, 12, 13, 14, 15, ...]  # sites in the left half of the lattice
        >>> disk_sites = lattice.regions.get_region(
        ...                 kind="disk",
        ...                 origin=10,          # center at site index 10
        ...                 radius=2.5          # radius of 2.5 units
        ...             )
        ... [5, 6, 10, 11, 15, ...]  # sites within the disk region
        >>> graph_sites = lattice.regions.get_region(
        ...                 kind="graph",
        ...                 origin=10,          # center at site index 10
        ...                 depth=2            # sites within graph distance 2
        ...             )
        ... [10, 5, 6, 11, 15, 20, ...]  # sites within graph distance 2 from site 10
        >>> sublattice_sites = lattice.regions.get_region(
        ...                     kind="sublattice",
        ...                     sublattice=0       # all sites in sublattice 0
        ...                 )
        ... [0, 2, 4, 6, 8, ...]  # sites belonging to sublattice 0
        >>> plaquette_sites = lattice.regions.get_region(
        ...                     kind="plaquette",
        ...                     plaquettes=[0, 1]  # sites belonging to plaquettes 0 and 1
        ...                 )
        ... [0, 1, 2, 3, 4, 5, ...]  # sites belonging to the specified plaquettes
        >>> kp_regions = lattice.regions.get_region(
        ...                 kind="kitaev_preskill",
        ...                 origin=10,          # center point for defining A, B, C
        ...                 radius=5.0          # radius of the disk containing A, B, C
        ...             )
        ... {'A': [5, 6, 10, ...], 'B': [11, 15, ...], 'C': [20, 25, ...], 'AB': [...], 'BC': [...], 'AC': [...], 'ABC': [...]} # sites in Kitaev-Preskill regions
        >>> lw_regions = lattice.regions.get_region(
        ...                 kind="levin_wen",
        ...                 origin=10,          # center point for defining A, B, C
        ...                 inner_radius=2.0,   # inner radius for region A
        ...                 outer_radius=5.0    # outer radius for region C
        ...             )
        ... {'A': [5, 6, 10, ...], 'B': [11, 15, ...], 'C': [20, 25, ...], 'AB': [...], 'BC': [...], 'AC': [...], 'ABC': [...]} # sites in Levin-Wen regions
        """
        if isinstance(kind, str):
            kind = kind.lower()
        elif isinstance(kind, RegionType):
            kind = kind.value
        else:
            raise ValueError("kind must be a string or RegionType enum")
        
        if kind == "half":
            return self.region_half(direction or "x")
        
        elif kind == "half_x":
            return self.region_half("x")
        
        elif kind == "half_y":
            return self.region_half("y")
        
        elif kind == "half_z":
            return self.region_half("z")
        
        elif kind == "disk":
            if origin is None or radius is None:
                raise ValueError("region_disk requires 'origin' and 'radius'.")
            return self.region_disk(origin, radius)
        
        elif kind == "graph":
            if origin is None or depth is None:
                raise ValueError("region_graph requires 'origin' (center site) and 'depth'.")
            if not isinstance(origin, int):
                 raise ValueError("region_graph 'origin' must be a site index.")
            return self.region_graph_ball(origin, depth)
        
        elif kind == "sublattice":
            if sublattice is None:
                raise ValueError("region_sublattice requires 'sublattice' index.")
            return self.region_sublattice(sublattice)
        
        elif kind == "plaquette":
            if plaquettes is None:
                raise ValueError("region_plaquette requires 'plaquettes' list.")
            return self.region_plaquettes(plaquettes)
        
        elif kind == "kitaev_preskill":
            regions     = self.region_kitaev_preskill(origin=origin)
            region_id   = kwargs.get('region', 'A').upper()
            return regions.get(region_id, [])

        elif kind == "custom":
            return sorted(list(set(sites))) if sites else []
        
        else:
            raise ValueError(f"Unknown region type: {kind}")

    # ------------------------------------------------------------------------------
    # Specific region definitions
    # ------------------------------------------------------------------------------

    def region_half(self, direction: str = "x") -> List[int]:
        r"""
        Half-system cut along a cardinal direction.
        
        Useful for area law scaling. Handles PBC by cutting based on coordinates relative to median.
        
        Parameters
        ----------
        direction : str
            Direction to cut ('x', 'y', or 'z').
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
            axis = 0
        elif direction == "y":
            axis = 1
        elif direction == "z":
            axis = 2
        else:
            raise ValueError("direction must be 'x', 'y', or 'z'")
        
        # Use median coordinate to define half cut, which is more robust for PBC than mean
        cut_val = np.median(coords[:, axis])

        # Return sites with coordinate less than cut_val along the specified axis
        return sorted(np.where(coords[:, axis] < cut_val)[0].tolist())

    # ------------------------------------------------------------------------------

    def region_disk(self, center: Union[int, List[float]], radius: float) -> List[int]:
        r"""
        Spherical/Circular region based on Euclidean distance (PBC-aware). We take
        the center point on the lattice and include all sites within a specified radius, accounting for PBC.
        To do this, we compute the shortest displacement vectors from the center to all sites, and include those within the radius.
        
        Parameters
        ----------
        center : int or list[float]
            Site index or coordinate vector of the center.
        radius : float
            Radius of the disk.
            
        Example
        --------
        >>> disk_sites = lattice.regions.get_region(
        ...                 kind="disk",
        ...                 origin=10,          # center at site index 10
        ...                 radius=2.5          # radius of 2.5 units
        ...             )
        
        Returns
        -------
        list[int]
            Sorted list of site indices within the disk region.
        """
        if not isinstance(center, int):
            # Fallback for arbitrary coordinates (not fully PBC aware without cell info)
            r0      = np.array(center)
            dist    = np.linalg.norm(self.lattice.rvectors - r0, axis=1)
            return sorted(np.where(dist <= radius)[0].tolist())

        # Vectorized PBC distance for site center
        n_center    = self.lattice.fracs[center]
        dn_all      = np.array(self.lattice.fracs - n_center, dtype=float)
        
        # Apply PBC adjustments
        dims        = [self.lattice.Lx, max(self.lattice.Ly, 1), max(self.lattice.Lz, 1)]
        pbc_flags   = self.lattice.periodic_flags()
        
        for d in range(3):
            if pbc_flags[d]:
                L               = dims[d]
                dn_all[:, d]    = dn_all[:, d] - L * np.round(dn_all[:, d] / L) # PBC adjustment for each coordinate direction
        
        # Calculate total displacements, including basis offsets
        R_disp          = np.outer(dn_all[:, 0], self.lattice.a1) + np.outer(dn_all[:, 1], self.lattice.a2) + np.outer(dn_all[:, 2], self.lattice.a3)
        r_basis_diff    = self.lattice.basis[self.lattice.subs] - self.lattice.basis[self.lattice.subs[center]]
        
        # Compute distances
        total_disp  = R_disp + r_basis_diff
        dists       = np.linalg.norm(total_disp, axis=1)
        
        # Find sites within radius, filter and return
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

    def region_kitaev_preskill(self, origin: Optional[Union[int, List[float]]] = None, radius: Optional[float] = None) -> Dict[str, List[int]]:
        r"""
        Divide the lattice into three regions A, B, C meeting at a point (origin).
        Uses azimuthal angles from the origin to define sectors.
        
        The three regions are defined as:
        - A: angles in [-Pi, -Pi/3)
        - B: angles in [-Pi/3, Pi/3)
        - C: angles in [Pi/3, Pi]
        
        Parameters
        ----------
        origin : int or list[float], optional
            Center point for defining regions. Can be a site index or coordinate vector.
        radius : float, optional
            Radius of the disk containing A, B, C. If None, defaults to slightly less than half the system size
            to ensure A+B+C is a proper subsystem (essential for TEE of pure states).
        
        Returns
        -------
        dict[str, list[int]]
            Dictionary with keys 'A', 'B', 'C' and their site indices.
            Also includes combinations 'AB', 'BC', 'AC', 'ABC'.
        """
        coords = self.lattice.rvectors
        if origin is None:
            r0 = np.mean(coords, axis=0)
        elif isinstance(origin, int):
            r0 = coords[origin]
        else:
            r0 = np.array(origin)
        
        # Determine radius if not provided
        if radius is None:
            # Default to slightly smaller than half the smallest dimension to fit in the torus/system
            # without wrapping issues, and ensuring we leave a region D.
            min_L   = min(self.lattice.Lx, self.lattice.Ly if self.lattice.dim >= 2 else self.lattice.Lx)
            radius  = (min_L / 2.0) - 0.5
            if radius < 1.0: 
                radius = 1.0 # Minimal fallback

        # Calculate distances to filter by radius
        dr      = coords - r0
        dists   = np.linalg.norm(dr, axis=1)
        
        # Angular sectors
        angles  = np.arctan2(dr[:, 1], dr[:, 0])    # -pi to pi -> extract azimuthal angles
        
        # sectors: divide 2pi into 3 equal parts
        # A: [-pi, -pi/3), B: [-pi/3, pi/3), C: [pi/3, pi]
        # AND restrict to radius
        mask_r  = dists <= radius
        
        A_mask  = (angles >= -np.pi) & (angles < -np.pi/3) & mask_r
        B_mask  = (angles >= -np.pi/3) & (angles < np.pi/3) & mask_r
        C_mask  = (angles >= np.pi/3) & (angles <= np.pi) & mask_r
        
        regions = {
            'A': np.where(A_mask)[0].tolist(),
            'B': np.where(B_mask)[0].tolist(),
            'C': np.where(C_mask)[0].tolist()
        }
        # combinations
        regions['AB']   = sorted(list(set(regions['A']) | set(regions['B'])))
        regions['BC']   = sorted(list(set(regions['B']) | set(regions['C'])))
        regions['AC']   = sorted(list(set(regions['A']) | set(regions['C'])))
        regions['ABC']  = sorted(list(set(regions['A']) | set(regions['B']) | set(regions['C'])))
        
        return regions
    
    # ---------------------------------------------------------------------------
    
    def region_levin_wen(self, origin: Optional[Union[int, List[float]]] = None, inner_radius: Optional[float] = None, outer_radius: Optional[float] = None) -> Dict[str, List[int]]:
        r"""
        Define Levin-Wen annular regions around a point (origin).
        
        The regions are defined as:
        - A: sites with distance < inner_radius
        - B: sites with inner_radius <= distance < outer_radius
        - C: sites with distance >= outer_radius
        
        Parameters
        ----------
        origin : int or list[float], optional
            Center point for defining regions. Can be a site index or coordinate vector.
        inner_radius : float, optional
            Inner radius of the annulus. Defaults to 1.0 if not provided.
        outer_radius : float, optional
            Outer radius of the annulus. Defaults to slightly less than half the system size if not provided.
        
        Returns
        -------
        dict[str, list[int]]
            Dictionary with keys 'A', 'B', 'C' and their site indices.
            Also includes combinations 'AB', 'BC', 'AC', 'ABC'.
        """
        coords = self.lattice.rvectors
        if origin is None:
            r0 = np.mean(coords, axis=0)
        elif isinstance(origin, int):
            r0 = coords[origin]
        else:
            r0 = np.array(origin)
        
        # Default radii if not provided
        if inner_radius is None:
            inner_radius = 1.0  # Minimal inner radius
        if outer_radius is None:
            min_L       = min(self.lattice.Lx, self.lattice.Ly if self.lattice.dim >= 2 else self.lattice.Lx)
            outer_radius = (min_L / 2.0) - 0.5
        
        dr      = coords - r0
        dists   = np.linalg.norm(dr, axis=1)
        
        A_mask  = dists < inner_radius
        B_mask  = (dists >= inner_radius) & (dists < outer_radius)
        C_mask  = dists >= outer_radius
        
        regions = {
            'A': np.where(A_mask)[0].tolist(),
            'B': np.where(B_mask)[0].tolist(),
            'C': np.where(C_mask)[0].tolist()
        }
        
        # combinations
        regions['AB']   = sorted(list(set(regions['A']) | set(regions['B'])))
        regions['BC']   = sorted(list(set(regions['B']) | set(regions['C'])))
        regions['AC']   = sorted(list(set(regions['A']) | set(regions['C'])))
        regions['ABC']  = sorted(list(set(regions['A']) | set(regions['B']) | set(regions['C'])))
        
# -------------------------------------------------------------------------------
#! End of region_handler.py
# -------------------------------------------------------------------------------