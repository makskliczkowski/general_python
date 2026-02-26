'''
This folder contains predefined regions for the use in the region_handler. Those are most common 
regions for different lattices, depending on the system size and lattice type (e.g. square lattice, honeycomb lattice, etc.).

----------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Created         : 2026-02-15
----------------------------------------------------------------
'''

from dataclasses    import dataclass, field
from itertools      import combinations
from typing         import Dict, List, Optional, Any, Callable, Set, Iterable, Tuple, FrozenSet, Type
import              numpy as np

# --------------------------------------------------------------

Adjacency           = Dict[int, Set[int]]
RegionSubset        = FrozenSet[int]

# --------------------------------------------------------------
#! Different region types
# --------------------------------------------------------------

@dataclass
class Region:
    """Base class for defining a region on a lattice."""

    A             : List[int]       # List of site indices for region A
    B             : List[int]       # List of site indices for region B
    C             : List[int]       = field(default_factory=list)   # List of site indices for region C - for bipartite lattices, this can be empty
    AB            : List[int]       = field(default_factory=list)   # List of site indices for region AB (A union B)
    AC            : List[int]       = field(default_factory=list)   # List of site indices for region AC (A union C)
    BC            : List[int]       = field(default_factory=list)   # List of site indices for region BC (B union C)
    ABC           : List[int]       = field(default_factory=list)   # List of site indices for region ABC (A union B union C)
    configuration : Optional[int]   = None                          # Configuration index for the region
    ns            : Optional[int]   = None                          # Total number of sites in the lattice (optional, can be inferred from the union of A, B, and C)
    
    def to_dict(self) -> Dict[str, List[int]]:
        """Convert the region to a dictionary format."""    
        return {
            'A': self.A,
            'B': self.B,
            'C': self.C
        }
        
    def bipartite(self) -> bool:
        """Check if the region is bipartite (i.e., C is empty)."""
        return len(self.C) == 0
    
    def tripartite(self) -> bool:
        """Check if the region is tripartite (i.e., C is non-empty)."""
        return len(self.C) > 0
    
    @staticmethod
    def is_valid_region(region: 'Region', adj: Optional[Adjacency] = None, extra: Optional[Callable[['Region'], bool]] = None) -> bool:
        """Check if the given region is valid (i.e., the union of A, B, and C matches the union of AB, AC, BC, and ABC)."""
        if adj is not None:
            for part in (region.A, region.B, region.C):
                if len(part) == 0:
                    continue
                if not Region.is_connected(part, adj):
                    return False
        
        if extra is not None and not extra(region):
            return False
        
        return True
    
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
        return (field.name.upper() for field in self.__dataclass_fields__.values() if field.name not in ["configuration", "ns"])
        
    def items(self):
        """Return an iterator over the region's items (key-value pairs)."""
        return ((field.name.upper(), getattr(self, field.name)) for field in self.__dataclass_fields__.values() if field.name not in ["configuration", "ns"])
        
    def values(self):
        """Return a list of the region's values (lists of coordinates)."""
        return [getattr(self, field.name) for field in self.__dataclass_fields__.values() if field.name not in ["configuration", "ns"]]
        
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
        
        ns_inferred = len(all_sites)
        if self.ns is not None and self.ns != ns_inferred:
            raise ValueError(f"The total number of sites (ns) must match the number of unique sites in A, B, and C. Expected {ns_inferred}, got {self.ns}.")
        
        self.ns     = ns_inferred  # Set ns to the inferred value if it was not
        
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

        # Support canonical field names and case-insensitive access.
        if name in self.__dataclass_fields__:
            return getattr(self, name)
        name_u = name.upper()
        if name_u in self.__dataclass_fields__:
            return getattr(self, name_u)
        name_l = name.lower()
        if name_l in self.__dataclass_fields__:
            return getattr(self, name_l)
        raise AttributeError(f"'Region' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        """Allow setting region components as attributes."""
        if name in self.__dataclass_fields__:
            super().__setattr__(name, value)
            return

        name_u = name.upper()
        if name_u in self.__dataclass_fields__:
            super().__setattr__(name_u, value)
            return

        name_l = name.lower()
        if name_l in self.__dataclass_fields__:
            super().__setattr__(name_l, value)
            return

        raise AttributeError(f"'Region' object has no attribute '{name}'")
        
    # --------------------------------------------------------------
    #! List utility methods
    # --------------------------------------------------------------
    
    @staticmethod
    def normalize_adjacency(
        adj                 : Any,
        ns                  : Optional[int] = None,
        *,
        weight_threshold    : float = 0.0,
        use_abs_weights     : bool = True,
    ) -> Adjacency:
        """
        Normalize adjacency input to ``dict[int, set[int]]``.
        
        If ``adj`` is a dict, it should be in the form {site: [neighbor1, neighbor2, ...]}.
        If ``adj`` is a 2D array, it will be treated as an adjacency matrix, where nonzero entries indicate edges.
        If ``adj`` is a 1D array of lists, it will be treated as an adjacency list, where each entry is a list of neighbors for the corresponding site index.
        """
        thr = float(weight_threshold)

        if isinstance(adj, dict):
            out: Adjacency = {}
            for i, neigh in adj.items():
                ii      = int(i)
                out[ii] = {int(j) for j in neigh if int(j) != ii}
        else:
            arr = np.asarray(adj)
            if arr.ndim == 2:
                if arr.shape[0] != arr.shape[1]:
                    raise ValueError(f"Adjacency matrix must be square, got shape={arr.shape}.")
                n   = int(arr.shape[0])
                out = {}
                for i in range(n):
                    row     = np.asarray(arr[i], dtype=float)
                    mask    = np.abs(row) > thr if use_abs_weights else row > thr
                    neigh   = set(np.where(mask)[0].tolist())
                    neigh.discard(i)
                    out[i]  = {int(j) for j in neigh}
            elif arr.ndim == 1:
                out = {}
                for i, neigh in enumerate(arr.tolist()):
                    ii      = int(i)
                    out[ii] = {int(j) for j in neigh if int(j) != ii}
            else:
                raise TypeError(f"Unsupported adjacency shape: {arr.shape}.")

        nodes = set(out.keys())
        for neigh in out.values():
            nodes.update(int(j) for j in neigh)
        if ns is not None:
            nodes.update(range(int(ns)))

        for node in nodes:
            out.setdefault(int(node), set())

        for i in list(out.keys()):
            for j in list(out[i]):
                out.setdefault(j, set()).add(i)

        return out

    @staticmethod
    def adjacency_from_lattice(lattice: Any, *, include_nnn: bool = False) -> Adjacency:
        """Build adjacency map from a lattice-like object."""
        ns      = int(getattr(lattice, "Ns", getattr(lattice, "ns")))
        adj_mat = getattr(lattice, "_adj_mat", None)
        if adj_mat is not None:
            return Region.normalize_adjacency(adj_mat, ns=ns, weight_threshold=0.0, use_abs_weights=True)

        out: Adjacency = {}
        for i in range(ns):
            neigh = set()
            for j in lattice.neighbors(i, order=1):
                if j is None:
                    continue
                jj = int(j)
                if 0 <= jj < ns and jj != i and jj != lattice.bad_lattice_site:
                    neigh.add(jj)
            if include_nnn:
                for j in lattice.neighbors(i, order=2):
                    if j is None:
                        continue
                    jj = int(j)
                    if 0 <= jj < ns and jj != i and jj != lattice.bad_lattice_site:
                        neigh.add(jj)
            out[i] = neigh
        return Region.normalize_adjacency(out, ns=ns)

    @staticmethod
    def is_connected(region: Iterable[int], adj: Any) -> bool:
        """Check if the provided region is connected in the graph."""
        
        nodes = {int(i) for i in region}
        if not nodes:
            return False

        adj_map     = Region.normalize_adjacency(adj)
        start       = next(iter(nodes))
        visited     = {start}
        stack       = [start]
        while stack:
            i = stack.pop()
            for j in adj_map.get(i, ()):
                if j in nodes and j not in visited:
                    visited.add(j)
                    stack.append(j)
        return len(visited) == len(nodes)

    @staticmethod
    def n_components(region: Iterable[int], adj: Any) -> int:
        """Return the number of connected components in the region."""
        
        adj_map     = Region.normalize_adjacency(adj)
        remaining   = {int(i) for i in region}
        n_comp      = 0
        while remaining:
            n_comp += 1
            start   = remaining.pop()
            stack   = [start]
            while stack:
                i = stack.pop()
                for j in adj_map.get(i, ()):
                    if j in remaining:
                        remaining.remove(j)
                        stack.append(j)
        return n_comp

    @staticmethod
    def connected_subsets(
        adj         : Any,
        max_size    : int,
        nodes       : Optional[Iterable[int]] = None,
        *,
        min_size    : int = 1,
        max_regions : Optional[int] = None,
    ) -> List[RegionSubset]:
        """
        Enumerate connected subsets with ``min_size <= |R| <= max_size``.
        """
        max_size = int(max_size)
        min_size = int(min_size)
        if max_size < 1 or min_size > max_size:
            return []

        adj_map = Region.normalize_adjacency(adj)
        allowed = set(adj_map.keys()) if nodes is None else {int(i) for i in nodes}
        allowed = {i for i in allowed if i in adj_map}
        if not allowed:
            return []

        regions: Set[RegionSubset] = set()
        for start in sorted(allowed):
            stack = [(frozenset({start}), set(adj_map[start]) & allowed)]
            # While the stack (which is a list of (current_cluster, frontier) pairs) is not empty:
            while stack:
                
                # Pop the last element from the stack, which gives us the current cluster and its frontier.
                cluster, frontier = stack.pop()
                if min_size <= len(cluster) <= max_size:
                    regions.add(cluster)
                    if max_regions is not None and len(regions) >= int(max_regions):
                        return sorted(regions, key=lambda s: (len(s), tuple(sorted(s))))
                
                # If the current cluster has already reached the maximum size, we skip expanding it further.
                if len(cluster) >= max_size:
                    continue
                
                # Iterate over the vertices in the frontier. 
                # For each vertex, if it is not already in the cluster, 
                # we create a new cluster by adding this vertex to the current cluster.
                for v in sorted(frontier):
                    if v in cluster:
                        continue
                    new_cluster = frozenset(set(cluster) | {v})
                    if len(new_cluster) > max_size:
                        continue
                    new_frontier = ((frontier | adj_map[v]) - set(new_cluster)) & allowed
                    stack.append((new_cluster, new_frontier))

        return sorted(regions, key=lambda s: (len(s), tuple(sorted(s))))

    @staticmethod
    def regions_touch(left: Iterable[int], right: Iterable[int], adj: Any) -> bool:
        """Return True if there is at least one adjacency edge between regions."""
        left_set    = {int(i) for i in left}
        right_set   = {int(i) for i in right}
        if not left_set or not right_set:
            return False
        adj_map     = Region.normalize_adjacency(adj)
        return any(any(j in right_set for j in adj_map.get(i, ())) for i in left_set)

    @staticmethod
    def triple_junction_count(A: Iterable[int], B: Iterable[int], C: Iterable[int], adj: Any) -> int:
        """
        Count sites that touch all three labels {A,B,C} through local adjacency.
        """
        a_set = {int(i) for i in A}
        b_set = {int(i) for i in B}
        c_set = {int(i) for i in C}

        labels: Dict[int, str] = {}
        labels.update({i: "A" for i in a_set})
        labels.update({i: "B" for i in b_set})
        labels.update({i: "C" for i in c_set})

        adj_map     = Region.normalize_adjacency(adj)
        n_junction  = 0
        for site, lbl in labels.items():
            touched = {lbl}
            for nb in adj_map.get(site, ()):
                nb_lbl = labels.get(nb)
                if nb_lbl is not None:
                    touched.add(nb_lbl)
            if len(touched) == 3:
                n_junction += 1
        return n_junction
    
    # --------------------------------------------------------------
    #! Region generation methods
    # --------------------------------------------------------------

    @classmethod
    def generate_bipartite_regions(
        cls,
        *,
        region_cls                      : Optional[Type['Region']] = None,
        adj                             : Any,
        ns                              : Optional[int] = None,
        size_a                          : Tuple[int, int] = (1, 3),
        nodes                           : Optional[Iterable[int]] = None,
        require_connected_parts         : bool = True,
        require_connected_complement    : bool = False,
        forbid_full_system              : bool = True,
        max_regions                     : Optional[int] = None,
        extra                           : Optional[Callable[['Region'], bool]] = None,
    ) -> List['Region']:
        """
        Generate many bipartite regions ``(A, B=complement(A))``.
        """
        region_type     = region_cls or cls
        adj_map         = cls.normalize_adjacency(adj, ns=ns)
        ns_total        = int(ns) if ns is not None else len(adj_map)
        allowed         = set(range(ns_total)) if nodes is None else {int(i) for i in nodes}
        allowed         = {i for i in allowed if i in adj_map}

        min_a, max_a    = int(size_a[0]), int(size_a[1])
        if min_a < 1 or max_a < min_a:
            return []

        if require_connected_parts:
            # Keep full subset coverage here; cap only final output regions.
            candidates  = cls.connected_subsets(adj_map, max_size=max_a, nodes=allowed, min_size=min_a, max_regions=None)
        else:
            candidates  = []
            ordered     = sorted(allowed)
            for k in range(min_a, max_a + 1):
                for combo in combinations(ordered, k):
                    candidates.append(frozenset(combo))

        out         : List[Region] = []
        seen        : Set[Tuple[int, ...]] = set()
        all_nodes   = set(range(ns_total))
        
        # Iterate over the candidate subsets for region A. For each candidate subset, 
        # we will check if it forms a valid region with its complement (which will be region B).
        for a_set in candidates:
            key = tuple(sorted(a_set))
            if key in seen:
                continue
            seen.add(key)

            if forbid_full_system and len(a_set) >= ns_total:
                continue

            b_set = sorted(all_nodes - set(a_set))
            if not b_set:
                continue
            if require_connected_complement and not cls.is_connected(b_set, adj_map):
                continue

            try:
                region = region_type(A=sorted(a_set), B=b_set, C=[])
            except ValueError:
                continue
            if not region_type.is_valid_region(region, adj_map, extra):
                continue

            out.append(region)
            if max_regions is not None and len(out) >= int(max_regions):
                break
        return out

    @classmethod
    def generate_tripartite_regions(
        cls,
        *,
        region_cls                      : Optional[Type['Region']] = None,
        adj                             : Any,
        ns                              : Optional[int] = None,
        size_a                          : Tuple[int, int] = (1, 3),
        size_b                          : Optional[Tuple[int, int]] = None,
        size_c                          : Optional[Tuple[int, int]] = None,
        nodes                           : Optional[Iterable[int]] = None,
        # Region constraints:
        require_connected_parts         : bool = True,
        require_connected_union         : bool = True,
        require_pairwise_touch          : bool = True,
        require_single_triple_junction  : bool = False,
        forbid_full_system              : bool = True,
        # Output constraints:
        max_regions                     : Optional[int] = None,
        extra                           : Optional[Callable[['Region'], bool]] = None,
    ) -> List['Region']:
        """
        Generate many disjoint tripartite regions ``(A, B, C)``.
        Parameters
        ----------
        region_cls : Optional[Type['Region']]
            The class to use for creating region instances. If None, the base Region class will be used.
        adj : Any            
            The adjacency information for the lattice. Can be a dict, adjacency matrix, or adjacency list.
        ns : Optional[int]
            Total number of sites in the lattice. If None, it will be inferred from the adjacency information.
        size_a : Tuple[int, int]
            Minimum and maximum size for region A.
        size_b : Optional[Tuple[int, int]]
            Minimum and maximum size for region B. If None, it will be set to the same as size_a.
        size_c : Optional[Tuple[int, int]]
            Minimum and maximum size for region C. If None, it will be set to the same as size_a.
        nodes : Optional[Iterable[int]]
            A subset of nodes to consider for forming regions. If None, all nodes from the adjacency will be considered.
        require_connected_parts : bool
            If True, each of the regions A, B, and C must be connected.
        require_connected_union : bool
            If True, the union of A, B, and C must be connected.
        require_pairwise_touch : bool
            If True, each pair of regions (A-B, A-C, B-C) must have at least one edge between them.
        require_single_triple_junction : bool
            If True, there must be exactly one site that is adjacent to all three regions A, B, and C.
        forbid_full_system : bool
            If True, the union of A, B, and C cannot include all sites in the system.
        max_regions : Optional[int]
            Maximum number of regions to generate. If None, there is no limit.
        extra : Optional[Callable[['Region'], bool]]
            An optional function that takes a Region instance and returns True if it should be included in the
            output list, and False otherwise. This allows for additional custom filtering of the generated regions.
            
        Returns
        -------
        List[Region]
            A list of generated Region instances that satisfy the specified constraints.
        """
        region_type = region_cls or cls
        adj_map     = cls.normalize_adjacency(adj, ns=ns)
        ns_total    = int(ns) if ns is not None else len(adj_map)
        allowed     = set(range(ns_total)) if nodes is None else {int(i) for i in nodes}
        allowed     = {i for i in allowed if i in adj_map}

        sa          = (int(size_a[0]), int(size_a[1]))
        sb          = (int(size_b[0]), int(size_b[1])) if size_b is not None else sa
        sc          = (int(size_c[0]), int(size_c[1])) if size_c is not None else sa

        if any(lo < 1 or hi < lo for lo, hi in (sa, sb, sc)):
            return []

        max_part    = max(sa[1], sb[1], sc[1])
        min_part    = min(sa[0], sb[0], sc[0])
        if require_connected_parts:
            # Do not truncate subset candidates at this stage. Early truncation
            # can bias toward low-index anchors and eliminate all valid
            # disjoint A/B/C combinations for larger min_size.
            subsets = cls.connected_subsets(adj_map, max_size=max_part, nodes=allowed, min_size=min_part, max_regions=None)
        else:
            subsets = []
            ordered = sorted(allowed)
            for k in range(min_part, max_part + 1):
                for combo in combinations(ordered, k):
                    subsets.append(frozenset(combo))

        cand_a = [s for s in subsets if sa[0] <= len(s) <= sa[1]]
        cand_b = [s for s in subsets if sb[0] <= len(s) <= sb[1]]
        cand_c = [s for s in subsets if sc[0] <= len(s) <= sc[1]]

        out: List[Region] = []
        for a_set in cand_a:
            a_nodes = set(a_set)
            
            # For each candidate subset for region A, we will iterate over the candidate subsets for region B.
            for b_set in cand_b:
                b_nodes = set(b_set)
                if a_nodes & b_nodes:
                    continue

                if require_pairwise_touch and not cls.regions_touch(a_nodes, b_nodes, adj_map):
                    continue

                ab_nodes = a_nodes | b_nodes
                for c_set in cand_c:
                    c_nodes = set(c_set)
                    if ab_nodes & c_nodes:
                        continue

                    abc_nodes = ab_nodes | c_nodes
                    if forbid_full_system and len(abc_nodes) >= ns_total:
                        continue

                    if require_connected_union and not cls.is_connected(abc_nodes, adj_map):
                        continue

                    if require_pairwise_touch:
                        if not cls.regions_touch(a_nodes, c_nodes, adj_map):
                            continue
                        if not cls.regions_touch(b_nodes, c_nodes, adj_map):
                            continue

                    if require_single_triple_junction:
                        if cls.triple_junction_count(a_nodes, b_nodes, c_nodes, adj_map) != 1:
                            continue

                    try:
                        region = region_type(A=sorted(a_nodes), B=sorted(b_nodes), C=sorted(c_nodes))
                    except ValueError:
                        continue
                    if not region_type.is_valid_region(region, adj_map, extra):
                        continue

                    out.append(region)
                    if max_regions is not None and len(out) >= int(max_regions):
                        return out
        return out

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
        
        super().__post_init__()  # Call the base class post-init to ensure consistency of unions
        
    def __str__(self) -> str:
        """String representation of the Kitaev-Preskill region."""
        return f"KitaevPreskillRegion(A={len(self.A)} sites, B={len(self.B)} sites, C={len(self.C)} sites)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the Kitaev-Preskill region."""
        return f"KitaevPreskillRegion(A={self.A}, B={self.B}, C={self.C})"
    
    # ----------------------------------------------------------
    
    @staticmethod
    def is_valid_region(region: 'KitaevPreskillRegion', adj: Optional[Adjacency] = None, extra: Optional[Callable[[Region], bool]] = None) -> bool:
        """Check if the given region is a valid Kitaev-Preskill region."""
        if not isinstance(region, KitaevPreskillRegion):
            return False
        
        if any(len(R) == 0 for R in [region.A, region.B, region.C]):
            return False

        return super().is_valid_region(region, adj, extra)

    
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
    
    # ----------------------------------------------------------
    
    @staticmethod
    def is_valid_region(region: 'LevinWenRegion', adj: Optional[Adjacency] = None, extra: Optional[Callable[[Region], bool]] = None) -> bool:
        """Check if the given region is a valid Levin-Wen region."""
        if not isinstance(region, LevinWenRegion):
            return False
        
        if any(len(R) == 0 for R in [region.A, region.B, region.C]):
            return False

        return super().is_valid_region(region, adj, extra)
    
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

@dataclass
class RegionFraction(Region):
    '''
    Region fraction is a specific type of region where the sites are arranged in a fractional manner, such as a fraction of the total lattice. 
    This can be useful for studying entanglement properties in a more flexible manner, as the region fraction can capture specific configurations of interest that may not fit into predefined shapes like disks or plaquettes.
    
    See: region_handler.py for more details on how to use this region in the context of entanglement entropy calculations.
    '''
    
    def fraction(self) -> float:
        """Calculate the fraction of the total lattice that is covered by region A."""
        total_sites = len(self.A) + len(self.B) + len(self.C)
        if total_sites == 0:
            return 0.0
        return len(self.A) / total_sites
    
    def __post_init__(self):
        """ Bipartite regions only """
        
        if len(self.A) == 0:
            raise ValueError("Region fraction must have a non-empty A region.")
        
        super().__post_init__()  # Call the base class post-init to ensure consistency of unions 
        
    def __str__(self) -> str:
        """String representation of the region fraction."""
        return f"RegionFraction(A={len(self.A)} sites, B={len(self.B)} sites, fraction={self.fraction():.2f})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the region fraction."""
        return f"RegionFraction(A={self.A}, B={self.B}, C={self.C}, fraction={self.fraction():.2f})"

@dataclass
class CustomRegion(Region):
    '''
    Custom region allows for arbitrary definitions of A, B, and C regions, as well as their unions. 
    This can be useful for studying entanglement properties in a more flexible manner, as the custom region can capture specific configurations of interest that may not fit into predefined shapes like disks or plaquettes.
    
    See: region_handler.py for more details on how to use this region in the context of entanglement entropy calculations.
    '''
    
    def __post_init__(self):
        """ No specific constraints on the custom region, but we can still ensure consistency of unions. """
        super().__post_init__()  # Call the base class post-init to ensure consistency of unions
    
    def __str__(self) -> str:
        """String representation of the custom region."""
        return f"CustomRegion(A={len(self.A)} sites, B={len(self.B)} sites, C={len(self.C)} sites)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the custom region."""
        return f"CustomRegion(A={self.A}, B={self.B}, C={self.C}, AB={self.AB}, AC={self.AC}, BC={self.BC}, ABC={self.ABC})"
    
# --------------------------------------------------------------
#! EOF
# --------------------------------------------------------------
