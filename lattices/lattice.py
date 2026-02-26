"""
Contains the general lattice class hierarchy and helpers.

This module defines the base :class:`Lattice` API used across general_python, together with
utility routines for boundary handling and symmetry metadata.

Currently, up to 3-spatial dimensions are supported...

------------------------------------------------------------------------------
File    : general_python/lattices/lattice.py
Author  : Maksymilian Kliczkowski
Date    : 2025-02-01
Version : 2.0
------------------------------------------------------------------------------
"""

from    __future__  import annotations
from    abc         import ABC, abstractmethod
from token import OP
from    typing      import Dict, List, Mapping, Optional, Tuple, Union
import  numpy       as np

from .tools.lattice_tools   import (
                                LatticeDirection, LatticeBC, LatticeType, 
                                handle_boundary_conditions, handle_boundary_conditions_detailed, handle_dim
                            )
from .tools.lattice_flux    import BoundaryFlux, _normalize_flux_dict
from .tools.lattice_kspace  import ( extract_bz_path_data, StandardBZPath, PathTypes, brillouin_zone_path,
                                reciprocal_from_real, extract_momentum, reconstruct_k_grid_from_blocks,
                                build_translation_operators, HighSymmetryPoints, HighSymmetryPoint,
                                KPathResult, find_nearest_kpoints
                            )
from .tools.region_handler  import LatticeRegionHandler, RegionType
from ..common               import hdf5man as HDF5Mod

################################################################################

Backend = np
class Lattice(ABC):
    r"""
    Abstract Base Class for defining lattice structures.

    This class serves as the foundation for all lattice implementations in the `lattices` module.
    It handles geometry, connectivity, boundary conditions, and k-space properties.

    Indexing Convention
    -------------------
    Lattice sites are indexed linearly from ``0`` to ``Ns - 1``.
    The mapping from spatial coordinates to linear index depends on the concrete implementation,
    but typically follows a row-major (lexicographic) order:

    *   **1D**: Left to right.
    *   **2D**: Bottom-left to top-right (x varies fastest).
    *   **3D**: Front-bottom-left to back-top-right.

    Features
    --------
    *   **Geometry**: Calculation of real-space coordinates, unit vectors, and basis vectors.
    *   **Connectivity**: Automatic identification of Nearest Neighbors (NN) and Next-Nearest Neighbors (NNN).
    *   **Boundaries**: Support for various boundary conditions:
        *   ``PBC``: Periodic Boundary Conditions (torus topology).
            * X-direction periodic, Y-direction periodic, Z-direction periodic
        *   ``OBC``: Open Boundary Conditions (hard edges).
            * X-direction open, Y-direction open, Z-direction open
        *   ``MBC``: Mixed Boundary Conditions (e.g., cylinder topology).
            * X-direction periodic, Y-direction open, Z-direction open
        *   ``SBC``: Switched Boundary Conditions (e.g. twisted cylinder).
            * X-direction open, Y-direction periodic, Z-direction open
        *   **TWISTED**: Twisted Boundary Conditions with specified fluxes.
    *   **Reciprocal Space**: Automatic calculation of reciprocal lattice vectors and Brillouin Zone paths.
    *   **Visualization**: Integration with plotting utilities via ``.plot``.

    Attributes
    ----------
    Ns : int
        Total number of sites in the lattice.
    dim : int
        Spatial dimension of the lattice (1, 2, or 3).
    Lx, Ly, Lz : int
        Linear dimensions of the lattice.
    bc : LatticeBC
        Active boundary condition.
    coordinates : np.ndarray
        Array of shape ``(Ns, 3)`` containing real-space coordinates of all sites.
    nn : List[List[int]]
        Adjacency list for nearest neighbors. ``nn[i]`` is a list of neighbors for site ``i``.
    """
    
    _BAD_LATTICE_SITE   = None
    _DFT_LIMIT_SITES    = 100 
    
    # ---------------------------------------------
    #! INITIALIZATION
    # ---------------------------------------------
    
    @property
    def bad_lattice_site(self):
        ''' Bad lattice site '''
        return self._BAD_LATTICE_SITE
    
    # Lattice constants - physical units where applicable
    a           = 1
    b           = 1
    c           = 1
    unit_length = 1     # unit length in Angstroms - helper for physical calculations

    def __init__(self, 
                dim     : int           = None,
                lx      : int           = 1,
                ly      : int           = 1,
                lz      : int           = 1,
                bc      : str           = None,             # boundary conditions
                adj_mat : np.ndarray    = None,             # can be controlled by the user for generic graphs
                flux    : np.ndarray    = None,             # flux piercing the boundaries - for each direction - for topological models
                *args,
                **kwargs):
        r'''
        General Lattice class. This class contains the general lattice model.
        
        Parameters
        ----------
        dim : int, optional
            Dimension of the lattice (1, 2, or 3). If None, inferred from lx, ly, lz.
        lx : int, optional
            Length of the lattice in the x-direction.
        ly : int, optional
            Length of the lattice in the y-direction.
        lz : int, optional
            Length of the lattice in the z-direction.
        bc : str, optional
            Boundary conditions (e.g., 'PBC', 'OBC').
        adj_mat : np.ndarray, optional
            Adjacency matrix for the lattice.
        flux : np.ndarray, optional
            Flux piercing the boundaries. This can be a dictionary specifying the
            flux in each direction, or a single value applied to all directions. Importantly, 
            this automatically implies **TWISTED** boundary conditions, so the `bc` parameter can be left as None or set to 'TWISTED' for clarity.
        '''
        
        self._dim           = handle_dim(lx, ly, lz)[0] if dim is None else dim
        self._bc            = handle_boundary_conditions(bc, flux=flux)  # Normalize boundary conditions and handle flux
        
        # flux piercing the boundaries - for topological models
        _raw_flux           = None
        if isinstance(self._bc, tuple):
            # If we have a tuple, it means we have TWISTED BCs with flux information
            self._bc, _raw_flux = self._bc
        self._flux          = _normalize_flux_dict(_raw_flux if _raw_flux is not None else flux)
        self._raw_flux      = _raw_flux if _raw_flux is not None else flux

        self._lx            = lx
        self._ly            = ly
        self._lz            = lz
        self._lxly          = lx * ly
        self._lxlz          = lx * lz
        self._lylz          = ly * lz
        self._lxlylz        = lx * ly * lz
        self._ns            = lx * ly * lz                  # Number of sites - set only initially as it is implemented in the children
        self._type          = LatticeType.SQUARE
        self._adj_mat       = adj_mat
        
        # Region handler
        self.regions        = LatticeRegionHandler(self)
        
        super().__init__(*args, **kwargs)
        
        # neighbors
        self._nn            = [[]]
        self._nn_forward    = [[]]
        self._nn_max_num    = 0
        self._nnn           = [[]]
        self._nnn_forward   = [[]]
        
        # helping lists
        self._cells         = []                                    # real space coordinates
        self._fracs         = []                                    # fractional coordinates
        self._subs          = []                                    # sub-lattice indices
        self._spatial_norm  = [[[]]]                                # three dimensional array for the spatial norm
        
        # matrices for real space and inverse space vectors
        vec_size            = max(3, self._dim)
        self._vectors       = Backend.zeros((vec_size, vec_size))   # real space vectors - base vectors of the lattice
        self._a1            = Backend.zeros(vec_size)               # real space vectors - base vectors of the lattice
        self._a2            = Backend.zeros(vec_size)
        self._a3            = Backend.zeros(vec_size)
        self._basis         = Backend.zeros((0, vec_size))          # basis vectors within the unit cell
        # inverse space vectors - ALWAYS 3D vectors for consistency
        self._k1            = Backend.zeros(3)                      # inverse space vectors - reciprocal lattice vectors (3D)
        self._k2            = Backend.zeros(3)
        self._k3            = Backend.zeros(3)
        # normal vectors (along the bonds - if required)
        self._n1            = Backend.zeros((self._dim, self._dim ))# normal vectors along the bonds
        self._n2            = Backend.zeros((self._dim, self._dim ))
        self._n3            = Backend.zeros((self._dim, self._dim ))
        # nearest neighbors vectors of the cells
        self._delta_z       = np.array([0.0, 0.0, self.a])          # UP
        self._delta_x       = np.array([self.a, 0.0, 0.0])          # RIGHT
        self._delta_y       = np.array([0.0, self.a, 0.0])          # FRONT
        
        # bonds
        self._bonds         = []                                    # empty if not calculated...
        
        self._rvectors      = Backend.zeros((self._ns, 3))          # allowed values of the real space vectors
        self._kvectors      = Backend.zeros((self._ns, 3))          # allowed values of the inverse space vectors
        # initialize dft matrix
        self._dft           = Backend.zeros((self._ns, self._ns), dtype = complex) # Discrete Fourier Transform matrix for the lattice model
        
        # symmetries for momenta (if one uses the symmetry,
        # returning to original one may result in using normalization)
        self.sym_norm       = {}
        self.sym_map        = {}

    # -----------------------------------------------------------------------------
            
    def __str__(self):
        ''' String representation of the lattice '''
        return "General Lattice"
        
    def __repr__(self):
        ''' Representation of the lattice '''
        if self._bc is LatticeBC.TWISTED:
            return f"{self._type.name},{self._bc.name},flux={self._flux},d={self._dim},Ns={self._ns},Lx={self._lx},Ly={self._ly},Lz={self._lz}"
        return f"{self._type.name},{self._bc.name},d={self._dim},Ns={self._ns},Lx={self._lx},Ly={self._ly},Lz={self._lz}"

    @property
    def _flux_suffix(self) -> str:
        """Return a suffix string for ``__str__`` / ``__repr__`` that includes flux info."""
        if self._flux is not None and self._flux.is_nontrivial:
            return f",flux={self._flux}"
        return ""
    
    def __len__(self):
        ''' Length of the lattice (number of sites) '''
        return self._ns
    
    def __getitem__(self, index: int):
        ''' Get the site at the given index '''
        if index < 0 or index >= self._ns:
            raise IndexError("Lattice index out of range")
        return index
    
    def __iter__(self):
        ''' Iterate over the lattice sites '''
        for i in range(self._ns):
            yield i
            
    def __contains__(self, item: int):
        ''' Check if the lattice contains the given site '''
        return 0 <= item < self._ns

    # -----------------------------------------------------------------------------
    
    def init(self, verbose: bool = False, *, force_dft: bool = False, **kwargs):
        """
        Initializes the lattice object by calculating coordinates, reciprocal vectors, and neighbor lists.
        
        This method performs the following steps:
        1. Calculates the real-space coordinates, r-vectors, and k-vectors of the lattice.
        2. If the number of sites (`self.Ns`) is less than 100, computes the discrete Fourier transform (DFT) matrix.
        3. If an adjacency matrix (`self._adj_mat`) is provided:
            - Determines the number of sites (`Ns`) from the adjacency matrix.
            - For each site, identifies nearest neighbors (nn) as those connected by the highest weight in the adjacency matrix, and next-nearest neighbors (nnn) as those connected by the next highest distinct weight.
            - Stores forward neighbors (indices greater than the current site) for both nn and nnn.
        4. If no adjacency matrix is provided, calculates nearest and next-nearest neighbors using default methods.
        5. Calculates normalization or symmetry properties of the lattice.
        This method sets up all necessary neighbor lists and lattice properties required for further computations.
        """

        self.calculate_reciprocal_vectors()
        self.calculate_coordinates()
        if verbose: print(" Lattice: Calculated coordinates.")
        
        self.calculate_r_vectors()
        if verbose: print(" Lattice: Calculated r-vectors.")

        self.calculate_k_vectors()
        if verbose: print(" Lattice: Calculated k-vectors.")

        if self.Ns < Lattice._DFT_LIMIT_SITES or force_dft:
            self.calculate_dft_matrix()
            if verbose: print(" Lattice: Calculated DFT matrix.")

        if self._adj_mat is not None:
            Ns                  = self._adj_mat.shape[0]
            self._ns            = Ns
            W                   = self._adj_mat
            nn_list             = []
            nnn_list            = []
            for i in range(Ns):
                #! sort by |weight| so signed couplings keep topology semantics
                js              = [j for j in range(Ns) if j != i and W[i, j] != 0]
                sorted_js       = sorted(js, key=lambda j: abs(W[i, j]), reverse=True)
                if not sorted_js:
                    nn_list.append([])
                    nnn_list.append([])
                    continue
                
                #! highest |weight| defines nn
                max_w_abs       = abs(W[i, sorted_js[0]])
                nn_js           = [j for j in sorted_js if abs(W[i, j]) == max_w_abs]
                nn_list.append(nn_js)
                
                if len(sorted_js) > len(nn_js):
                    #! find next distinct |weight|
                    remaining   = [abs(W[i, j]) for j in sorted_js if abs(W[i, j]) != max_w_abs]
                    if remaining:
                        second_w    = max(remaining)
                        nnn_js      = [j for j in sorted_js if abs(W[i, j]) == second_w]
                    else:
                        nnn_js      = []
                else:
                    nnn_js = []
                nnn_list.append(nnn_js)
            self._nn            = nn_list
            self._nn_forward    = [[j for j in nn_list[i] if j>i] for i in range(Ns)]
            self._nnn           = nnn_list
            self._nnn_forward   = [[j for j in nnn_list[i] if j>i] for i in range(Ns)]
            if verbose: print(" Lattice: Calculated neighbors from adjacency matrix.")
            if verbose: print(" Lattice: Calculated forward neighbors from adjacency matrix.")
            if verbose: print(" Lattice: Calculated next-nearest neighbors from adjacency matrix.")
            if verbose: print(" Lattice: Calculated forward next-nearest neighbors from adjacency matrix.")
        else:
            self.calculate_nn()
            if verbose: print(" Lattice: Calculated nearest neighbors.")
            self.calculate_nnn()
            if verbose: print(" Lattice: Calculated next-nearest neighbors.")
        
        self.calculate_norm_sym()
        if verbose: print(" Lattice: Calculated normalization/symmetry.")
    
        # Initialize the normal vectors along the bonds
        self._n1    = self._delta_x / np.linalg.norm(self._delta_x)
        self._n2    = self._delta_y / np.linalg.norm(self._delta_y)
        self._n3    = self._delta_z / np.linalg.norm(self._delta_z)
    
    # -----------------------------------------------------------------------------
    #! Region generators
    # -----------------------------------------------------------------------------

    def get_region(
        self,
        kind                : Union[str, RegionType]            = RegionType.HALF,
        *,
        origin              : Optional[Union[int, List[float]]] = None,
        radius              : Optional[float]                   = None,
        direction           : Optional[str]                     = None,
        sublattice          : Optional[int]                     = None,
        sites               : Optional[List[int]]               = None,
        depth               : Optional[int]                     = None,
        plaquettes          : Optional[List[int]]               = None,
        **kwargs
    ) -> List[int]:
        r"""
        Return a list of site indices defining a spatial region.
        
        Parameters
        ----------
        kind : str or RegionType
            Type of region: 'half', 'disk', 'sublattice', 'graph', 'plaquette', 'custom'.
            We also support specific half cuts like 'half_x', 'half_y', 'half_z' for convenience.
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
        """
        return self.regions.get_region(
            kind=kind,
            origin=origin,
            radius=radius,
            direction=direction,
            sublattice=sublattice,
            sites=sites,
            depth=depth,
            plaquettes=plaquettes,
            **kwargs
        )

    def get_entropy_cuts(
        self,
        cut_type: str = "all",
        *,
        include_sublattice: bool = True,
        sweep_by_unit_cell: Optional[bool] = None,
    ) -> Dict[str, List[int]]:
        """
        Return canonical bipartition cuts for entanglement-entropy workflows.

        This is a convenience wrapper around :meth:`self.regions.get_entropy_cuts`.
        """
        return self.regions.get_entropy_cuts(
            cut_type=cut_type,
            include_sublattice=include_sublattice,
            sweep_by_unit_cell=sweep_by_unit_cell,
        )

    def generate_regions(self, kind: Union[str, RegionType] = RegionType.KITAEV_PRESKILL, **kwargs,):
        """
        Generate many region candidates for a selected region type.

        This is a thin wrapper around :meth:`self.regions.generate_regions`.
        """
        return self.regions.generate_regions(kind=kind, **kwargs)

    ################################### GETTERS ###################################
    
    @property
    def lx(self):           return self._lx
    @property
    def Lx(self):           return self._lx
    @lx.setter
    def lx(self, value):    self._lx = value; self._lxly = self._lx * self._ly; self._lxlylz = self._lxly * self._lz; self._lxlz = self._lx * self._lz
    @Lx.setter
    def Lx(self, value):    self._lx = value; self._lxly = self._lx * self._ly; self._lxlylz = self._lxly * self._lz; self._lxlz = self._lx * self._lz
    
    @property
    def ly(self):           return self._ly
    @property
    def Ly(self):           return self._ly
    @ly.setter
    def ly(self, value):    self._ly = value; self._lxly = self._lx * self._ly; self._lxlylz = self._lxly * self._lz; self._lylz = self._ly * self._lz
    @Ly.setter
    def Ly(self, value):    self._ly = value; self._lxly = self._lx * self._ly; self._lxlylz = self._lxly * self._lz; self._lylz = self._ly * self._lz

    @property
    def lz(self):           return self._lz
    @property
    def Lz(self):           return self._lz
    @lz.setter
    def lz(self, value):    self._lz = value; self._lxlylz = self._lxly * self._lz; self._lylz = self._ly * self._lz; self._lxlz = self._lx * self._lz
    @Lz.setter
    def Lz(self, value):    self._lz = value; self._lxlylz = self._lxly * self._lz; self._lylz = self._ly * self._lz; self._lxlz = self._lx * self._lz

    @property
    def area(self):         return self._lxly
    @property
    def volume(self):       return self._lxlylz
    @property
    def lxly(self):         return self._lxly
    @property
    def lxlz(self):         return self._lxlz
    @property
    def lylz(self):         return self._lylz
    @property
    def lxlylz(self):       return self._lxlylz
    
    @property
    def dim(self):          return self._dim
    @dim.setter
    def dim(self, value):   self._dim = value
    
    @property
    def ns(self):           return self._ns
    @property
    def Ns(self):           return self._ns
    @property
    def sites(self):        return self._ns
    @property
    def size(self):         return self._ns
    @property
    def nsites(self):       return self._ns

    @ns.setter
    def ns(self, value):    self._ns = value
    @Ns.setter
    def Ns(self, value):    self._ns = value

    # -----------------------------------------------------------------------------
    #! Physical 
    # -----------------------------------------------------------------------------

    @property
    def sites_per_cell(self) -> int:
        """Sites per unit cell (1 for Bravais, 2 for honeycomb, etc.)."""
        n_cells = max(1, self._lx * self._ly * self._lz)
        return max(1, self._ns // n_cells)

    def symmetry_perms(self, point_group: str = "full") -> np.ndarray:
        """
        Generate space-group permutation table for this lattice.

        Delegates to :func:`~.tools.lattice_symmetry.generate_space_group_perms`.

        When TWISTED boundary conditions are active, the point-group part is
        disabled (only translations are returned) because a generic flux
        breaks point-group symmetry unless the flux respects it.

        Parameters
        ----------
        point_group : str
            ``'full'`` for maximal point group, ``'translations'`` for translations only.

        Returns
        -------
        ndarray, shape (|G|, Ns)
        """
        from .tools.lattice_symmetry import generate_space_group_perms
        # Flux generically breaks point-group symmetry → translations only
        if self.is_twisted and point_group == "full":
            point_group = "translations"
        return generate_space_group_perms(self.Lx, self.Ly, self.sites_per_cell, point_group)
    
    # ------------------------------------------------------------------
    #! Lattice symmetry information
    # ------------------------------------------------------------------

    def lattice_symmetries(self) -> Dict[str, object]:
        """
        Return a dictionary describing the spatial symmetries of this lattice.

        The information is consistent for both single-particle and many-body
        representations.  When TWISTED boundary conditions are present the
        point-group part is absent (flux generically breaks it).

        Returns
        -------
        dict
            Keys:
            - ``'lattice_type'``        : :class:`LatticeType` enum
            - ``'sites_per_cell'``      : int
            - ``'n_cells'``             : number of unit cells
            - ``'dim'``                 : spatial dimension
            - ``'bc'``                  : boundary condition enum
            - ``'is_periodic'``         : (bool, bool, bool) per direction
            - ``'is_twisted'``          : bool
            - ``'translation_group'``   : ZL_x x ZL_y (as tuple ``(Lx, Ly)``)
            - ``'point_group'``         : str or None (``'D4'`` for square Lx==Ly, etc.)
            - ``'space_group_order'``   : total number of space-group elements
            - ``'flux'``                : :class:`BoundaryFlux` or None
        """
        # Get periodicity flags for each direction
        pbc_flags   = self.periodic_flags()
        
        # Determine point group
        pg          = None
        if not self.is_twisted:
            if hasattr(self, '_type') and self._type == LatticeType.SQUARE:
                if self._lx == self._ly and pbc_flags[0] and pbc_flags[1]:
                    pg = 'D4'
                elif pbc_flags[0] and pbc_flags[1]:
                    pg = 'D2'
            elif hasattr(self, '_type') and self._type in (LatticeType.HONEYCOMB, LatticeType.HEXAGONAL):
                if self._lx == self._ly and pbc_flags[0] and pbc_flags[1]:
                    pg = 'C6v'          # full hexagonal point group for the lattice

        n_cells     = max(1, self._lx * self._ly * self._lz)
        n_trans     = self._lx * (self._ly if pbc_flags[1] else 1) * (self._lz if pbc_flags[2] else 1) if pbc_flags[0] else 1
        pg_order    = {'D4': 8, 'D2': 4, 'C6v': 12}.get(pg, 1) if pg else 1
        
        return {
            'lattice_type':         self._type if hasattr(self, '_type') else None,
            'sites_per_cell':       self.sites_per_cell,
            'n_cells':              n_cells,
            'dim':                  self._dim,
            'bc':                   self._bc,
            'is_periodic':          pbc_flags,
            'is_twisted':           self.is_twisted,
            'translation_group':    (self._lx, self._ly if self._dim >= 2 else 1),
            'point_group':          pg,
            'space_group_order':    n_trans * pg_order,
            'flux':                 self._flux,
        }

    def symmetry_info(self) -> str:
        """
        Return a human-readable summary of the lattice symmetries.

        Consistent for both single-particle (band-structure / Bloch) and
        many-body (Hilbert-space symmetry sectors) viewpoints.

        Returns
        -------
        str
        """
        d = self.lattice_symmetries()
        lines = [
            f"Lattice symmetry info ({d['lattice_type']})",
            f"  dim               = {d['dim']}",
            f"  sites / cell      = {d['sites_per_cell']}",
            f"  unit cells        = {d['n_cells']}",
            f"  boundary cond.    = {d['bc']}",
            f"  periodic (x,y,z)  = {d['is_periodic']}",
        ]
        if d['is_twisted']:
            lines.append(f"  twisted = True  (flux breaks point-group!)")
            lines.append(f"  flux    = {d['flux']}")
        lines.append(f"  translation group = Z_{d['translation_group'][0]} x Z_{d['translation_group'][1]}")
        lines.append(f"  point group       = {d['point_group'] or 'trivial'}")
        lines.append(f"  |space group|     = {d['space_group_order']}")
        return "\n".join(lines)
    
    @property
    def a1(self):           return self._a1
    @a1.setter
    def a1(self, value):    self._a1 = value
    
    @property
    def a2(self):           return self._a2
    @a2.setter
    def a2(self, value):    self._a2 = value

    @property
    def a3(self):           return self._a3
    @a3.setter
    def a3(self, value):    self._a3 = value

    # -----------------------------------------------------------------------------
    # Inverse space vectors
    # -----------------------------------------------------------------------------

    @property
    def k1(self):                   return self._k1
    @k1.setter
    def k1(self, value):            self._k1 = value
    @property
    def b1(self):                   return self._k1
    @b1.setter
    def b1(self, value):            self._k1 = value
    
    @property
    def k2(self):                   return self._k2
    @k2.setter
    def k2(self, value):            self._k2 = value
    @property
    def b2(self):                   return self._k2
    @b2.setter
    def b2(self, value):            self._k2 = value
    
    @property
    def k3(self):                   return self._k3
    @k3.setter
    def k3(self, value):            self._k3 = value
    @property
    def b3(self):                   return self._k3
    @b3.setter
    def b3(self, value):            self._k3 = value
    
    # ------------------------------------------------------------------
    
    @property
    def n1(self):                   return self._n1
    @n1.setter
    def n1(self, value):            self._n1 = value

    @property
    def n2(self):                   return self._n2
    @n2.setter
    def n2(self, value):            self._n2 = value
    
    @property
    def n3(self):                   return self._n3
    @n3.setter
    def n3(self, value):            self._n3 = value
    
    @property
    def basis(self):                return self._basis
    @basis.setter
    def basis(self, value):         self._basis = value
    @property
    def multipartity(self):         return self._basis.shape[0]
    
    @property
    def vectors(self):              return self._vectors
    @vectors.setter
    def vectors(self, value):       self._vectors = value
    
    @property
    def avec(self):                 return np.stack((self._a1, self._a2, self._a3), axis=0)
    @avec.setter
    def avec(self, value):          self._a1 = value[0]; self._a2 = value[1]; self._a3 = value[2]
    
    @property
    def bvec(self):                 return np.stack((self._k1, self._k2, self._k3), axis=0)
    @bvec.setter
    def bvec(self, value):          self._k1 = value[0]; self._k2 = value[1]; self._k3 = value[2]

    # ------------------------------------------------------------------
    #! DFT Matrix
    # ------------------------------------------------------------------
    
    @property
    def dft(self):
        ''' Return the discrete Fourier transform (DFT) matrix for the lattice. '''
        return self._dft
    @dft.setter
    def dft(self, value):           self._dft = value
    
    @property
    def nn(self):
        ''' Return the nearest-neighbor connectivity matrix for the lattice. '''
        return self._nn
    @nn.setter
    def nn(self, value):            self._nn = value

    @property
    def bonds(self):
        ''' Return the bond connectivity matrix for the lattice. '''
        return self._bonds
    @bonds.setter
    def bonds(self, value):         self._bonds = value

    @property
    def nn_forward(self):           
        ''' Return the forward nearest-neighbor connectivity matrix for the lattice. '''
        return self._nn_forward
    @nn_forward.setter
    def nn_forward(self, value):    self._nn_forward = value

    @property
    def nnn(self):                  
        ''' Return the next-nearest-neighbor connectivity matrix for the lattice. '''
        return self._nnn
    @nnn.setter
    def nnn(self, value):           self._nnn = value
    
    @property
    def nnn_forward(self):
        ''' Return the forward next-nearest-neighbor connectivity matrix for the lattice. '''
        return self._nnn_forward
    @nnn_forward.setter
    def nnn_forward(self, value):   self._nnn_forward = value
    
    @property
    def coordinates(self):
        ''' Return the real-space coordinates of the lattice sites. '''
        return self._coordinates
    @coordinates.setter
    def coordinates(self, value):   self._coordinates = value
    
    @property
    def subs(self):
        ''' Return the sublattice indices of the lattice sites. 
        For a Bravais lattice, this would simply be an array of zeros.
        For a non-Bravais lattice, this would indicate which sublattice each site belongs to. '''
        return self._subs
    @subs.setter
    def subs(self, value):          self._subs = value
    
    @property
    def cells(self):
        ''' Return the unit cell coordinates of the lattice sites. For a Bravais lattice, 
        this would simply be the integer coordinates of the unit cells. 
        For a non-Bravais lattice, this would include the basis vectors as well. '''
        return self._cells
    @cells.setter
    def cells(self, value):         self._cells = value
    
    @property
    def fracs(self):                
        ''' Return fractional coordinates of the lattice sites. Example: for a square lattice, these would be (x/Lx, y/Ly, z/Lz) for each site. '''
        return self._fracs
    @fracs.setter
    def fracs(self, value):         self._fracs = value

    @property
    def kvectors(self):
        ''' Return the allowed k-vectors in reciprocal space for the lattice. '''
        return self._kvectors   
    @kvectors.setter
    def kvectors(self, value):      self._kvectors = value
    
    @property
    def rvectors(self):
        ''' Return the allowed r-vectors in real space for the lattice. '''
        return self._rvectors
    @rvectors.setter
    def rvectors(self, value):      self._rvectors = value
    
    @property
    def bc(self):                   return self._bc
    @bc.setter
    def bc(self, value):            self._bc = value
    
    @property
    def bc_x(self):                 return handle_boundary_conditions_detailed(self._bc, self._raw_flux).get('x', False)
    @property
    def bc_y(self):                 return handle_boundary_conditions_detailed(self._bc, self._raw_flux).get('y', False)
    @property
    def bc_z(self):                 return handle_boundary_conditions_detailed(self._bc, self._raw_flux).get('z', False)
    
    @property
    def cardinality(self):          return self.get_nn_forward_num_max()
    @cardinality.setter
    def cardinality(self, value):   self._nn_max_num = value
    
    @property
    def flux(self):                 return self._flux
    @flux.setter
    def flux(self, flux: Union[BoundaryFlux, Dict[str, float], None]):
        ''' 
        Set the flux piercing the boundaries for twisted boundary conditions.
        '''        
        
        self._bc            = handle_boundary_conditions(self._bc, flux=flux)  # Normalize boundary conditions and handle flux
        
        # flux piercing the boundaries - for topological models
        _raw_flux           = None
        if isinstance(self._bc, tuple):
            # If we have a tuple, it means we have TWISTED BCs with flux information
            self._bc, _raw_flux = self._bc
        self._flux          = _normalize_flux_dict(_raw_flux if _raw_flux is not None else flux)
        self._raw_flux      = _raw_flux if _raw_flux is not None else flux

    @property
    def name(self):                 return self.__str__()
    @property
    def type(self):                 return self._type if hasattr(self, '_type') else None
    
    # ------------------------------------------------------------------
    #! Sublattice
    # ------------------------------------------------------------------
    
    def sublattice(self, site: int) -> int:
        """
        Return the sublattice index for a given site.
        By default, returns 0 for all sites (single sublattice).
        Override in subclasses for multi-sublattice lattices.
        """
        return site % self.multipartity
    
    # ------------------------------------------------------------------
    #! K-space
    # ------------------------------------------------------------------
    
    def k_vector(self, qx, qy=0.0, qz=0.0) -> np.ndarray:
        """
        Return the k-vector in Cartesian coordinates for given (qx, qy, qz)
        in reciprocal lattice units.
        """
        if self.k1 == None or self.k2 == None or self.k3 == None:
            self.k1, self.k2, self.k3 = reciprocal_from_real(self.a1, self.a2, self.a3)
        
        kvec = qx * self.k1[0,:]
        if self.dim > 1:
            kvec += qy * self.k2[0,:]
        if self.dim > 2:
            kvec += qz * self.k3[0,:]
        return kvec
    
    def extract_momentum(self, 
                        eigvecs    : np.ndarray, 
                        *,
                        eigvals     : np.ndarray = None,
                        tol         : float      = 1e-10,
                        ) -> np.ndarray:
        """
        Extract crystal momentum vectors k from real-space eigenvectors.
        """
        return extract_momentum(eigvecs, self, eigvals=eigvals, tol=tol)

    # ------------------------------------------------------------------
    #! High-symmetry points and BZ paths
    # ------------------------------------------------------------------
    
    def high_symmetry_points(self) -> Optional[HighSymmetryPoints]:
        """
        Return high-symmetry points for this lattice type.
        
        Override in subclasses to provide lattice-specific high-symmetry points.
        Returns None if not defined for this lattice type.
        
        Returns
        -------
        HighSymmetryPoints or None
            High-symmetry points with default path, or None if not defined.
        
        Example
        -------
        >>> lattice = SquareLattice(dim=2, lx=4, ly=4)
        >>> pts = lattice.high_symmetry_points()
        >>> print(pts.Gamma.frac_coords)  # (0.0, 0.0, 0.0)
        >>> print(pts.default_path())     # ['Gamma', 'X', 'M', 'Gamma']
        """
        # Base implementation tries to guess from lattice type
        if hasattr(self, '_type'):
            if self._type == LatticeType.SQUARE:
                if self.dim == 1:
                    return HighSymmetryPoints.chain_1d()
                elif self.dim == 2:
                    return HighSymmetryPoints.square_2d()
                elif self.dim == 3:
                    return HighSymmetryPoints.cubic_3d()
            elif self._type == LatticeType.HONEYCOMB:
                return HighSymmetryPoints.honeycomb_2d()
            elif self._type == LatticeType.HEXAGONAL:
                return HighSymmetryPoints.hexagonal_2d()
        return None
    
    def default_bz_path(self) -> Optional[List[Tuple[str, List[float]]]]:
        """
        Return the default Brillouin zone path for this lattice.
        
        Returns
        -------
        List[Tuple[str, List[float]]] or None
            Default path as list of (label, [f1, f2, f3]) tuples, or None if not defined.
        """
        hs_pts = self.high_symmetry_points()
        if hs_pts is not None:
            return hs_pts.get_default_path_points()
        return None

    def contains_special_point(
        self,
        point: Union[str, HighSymmetryPoint, Tuple[float, ...], np.ndarray],
        *,
        tol: float = 1e-12,
    ) -> bool:
        r"""
        Return ``True`` if the lattice momentum grid contains a special point.

        Parameters
        ----------
        point
            Special point identifier. Accepted forms:
            - label string (e.g. ``"Gamma"``, ``"K"``, ``"K'"``),
            - :class:`HighSymmetryPoint`,
            - explicit fractional coordinate tuple/array.
        tol
            Absolute tolerance used in the coordinate match.

        Notes
        -----
        The check is done in *fractional* reciprocal coordinates and naturally
        includes flux-induced shifts from twisted boundary conditions because it
        uses ``self.kvectors_frac``.
        """
        
        # Get fractional k-vectors, calculating if not already available
        kfrac = getattr(self, "kvectors_frac", None)
        if kfrac is None:
            self.calculate_k_vectors()
            kfrac = getattr(self, "kvectors_frac", None)
        if kfrac is None:
            return False

        # Resolve target point to fractional coordinates
        target_frac = None
        if isinstance(point, HighSymmetryPoint):
            target_frac = np.asarray(point.frac_coords, dtype=float)
            
        elif isinstance(point, str): # Label string - look up in high_symmetry_points
            hs_pts = self.high_symmetry_points()
            if hs_pts is None:
                return False
            p_obj = hs_pts.resolve(point) if hasattr(hs_pts, "resolve") else hs_pts.get(point)
            if p_obj is None:
                return False
            target_frac = np.asarray(p_obj.frac_coords, dtype=float)
            
        else: # Fractional coordinate tuple/array
            try:
                target_frac = np.asarray(point, dtype=float).reshape(-1)
            except Exception:
                return False

        # Check if any k-vector matches the target fractional coordinates within tolerance
        if target_frac is None or target_frac.size == 0:
            return False

        if target_frac.size < 3:
            target_frac = np.pad(target_frac, (0, 3 - target_frac.size), mode="constant")

        # Check dimensions and wrap to [0, 1) - we work in fractional coordinates so this naturally includes any flux-induced shifts
        dim     = 1 if self.dim == 1 else (2 if self.dim == 2 else 3)
        grid    = np.asarray(kfrac, dtype=float)
        if grid.ndim != 2 or grid.shape[1] < dim:
            return False
        
        # Grid and target are wrapped to [0, 1) in fractional coordinates, so this check naturally includes any flux-induced shifts from twisted boundary conditions
        grid    = np.mod(grid[:, :dim], 1.0)
        tgt     = np.mod(target_frac[:dim], 1.0)
        hits    = np.all(np.isclose(grid, tgt[None, :], atol=tol, rtol=0.0), axis=1)
        return bool(np.any(hits))
    
    def generate_bz_path(
        self,
        path            : Optional[Union[List[str], str, StandardBZPath]] = None,
        *,
        points_per_seg  : int = 40,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], np.ndarray]:
        """
        Generate k-points along a Brillouin zone path.
        
        Parameters
        ----------
        path : list of str, str, StandardBZPath, or None
            Path specification. Can be:
            - List of high-symmetry point names: ['Gamma', 'X', 'M', 'Gamma']
            - StandardBZPath enum or string: 'SQUARE_2D'
            - None: use default path for this lattice
        points_per_seg : int
            Number of interpolated points per path segment.
        
        Returns
        -------
        k_path : np.ndarray, shape (Npath, 3)
            Cartesian k-points along the path.
        k_dist : np.ndarray, shape (Npath,)
            Cumulative distance for plotting x-axis.
        labels : List[Tuple[int, str]]
            Indices and labels for high-symmetry points.
        k_path_frac : np.ndarray, shape (Npath, 3)
            Fractional k-coordinates along the path.
        
        Example
        -------
        >>> lattice = SquareLattice(dim=2, lx=4, ly=4)
        >>> k_path, k_dist, labels, k_frac = lattice.generate_bz_path()
        >>> # Or with custom path:
        >>> k_path, k_dist, labels, k_frac = lattice.generate_bz_path(['Gamma', 'M', 'Gamma'])
        """
        # Resolve path
        if path is None:
            resolved_path = self.default_bz_path()
            if resolved_path is None:
                raise ValueError(f"No default BZ path for {type(self).__name__}. "
                               "Specify path explicitly.")
        elif isinstance(path, list) and all(isinstance(p, str) for p in path):
            # List of point names - look up in high_symmetry_points
            hs_pts = self.high_symmetry_points()
            if hs_pts is None:
                raise ValueError(f"Cannot resolve point names for {type(self).__name__}. "
                               "Use explicit fractional coordinates instead.")
            resolved_path = hs_pts.get_path_points(path)
        elif isinstance(path, (str, StandardBZPath)):
            resolved_path = path  # Will be resolved by brillouin_zone_path
        else:
            resolved_path = path
        
        return brillouin_zone_path(self, resolved_path, points_per_seg=points_per_seg)

    def extract_bz_path_data(self, 
                            k_vectors           : np.ndarray,
                            k_vectors_frac      : np.ndarray,
                            values              : np.ndarray,
                            path                : Optional[Union[List[str], PathTypes, str, StandardBZPath]] = None,
                            *,
                            points_per_seg      : int = 40,
                            return_result       : bool = True,
                            ) -> Union[KPathResult, Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], np.ndarray]]:
        """
        Extract k-point path data in the Brillouin zone for band structure calculations.
        
        This method finds the nearest k-points on the actual grid to an ideal path
        through high-symmetry points. It returns both the matched data and indices
        for mapping back to the original grid.
        
        Parameters
        ----------
        k_vectors : np.ndarray
            Array of k-vectors in Cartesian coordinates, shape (..., 3).
        k_vectors_frac : np.ndarray
            Fractional coordinates of k-vectors, shape (..., 3).
        values : np.ndarray
            Corresponding values (e.g., energies) at each k-vector, shape (..., n_bands).
        path : list of str, PathTypes, str, StandardBZPath, or None
            Path specification. Can be:
            - List of high-symmetry point names: ['Gamma', 'X', 'M', 'Gamma']
            - StandardBZPath enum or string: 'SQUARE_2D'
            - None: use default path for this lattice
        points_per_seg : int
            Number of points per segment for path interpolation.
        return_result : bool
            If True (default), return KPathResult dataclass.
            If False, return tuple for backwards compatibility.
        
        Returns
        -------
        KPathResult or tuple
            If return_result=True: KPathResult with k_cart, k_frac, k_dist, labels, 
                values, indices, and matched_distances.
            If return_result=False: (k_cart, k_frac, k_dist, labels, values) tuple.
        
        Example
        -------
        >>> lattice = SquareLattice(dim=2, lx=8, ly=8, bc='pbc')
        >>> # Get k-grid from Hamiltonian
        >>> k_grid, k_frac, energies = ham.to_kspace()
        >>> 
        >>> # Extract band structure along path
        >>> result = lattice.extract_bz_path_data(k_grid, k_frac, energies)
        >>> 
        >>> # Plot
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result.k_dist, result.values)
        >>> for pos, label in zip(result.label_positions, result.label_texts):
        ...     plt.axvline(pos, color='k', linestyle='--')
        >>> plt.xticks(result.label_positions, result.label_texts)
        """
        # Resolve path if given as list of names
        if isinstance(path, list) and all(isinstance(p, str) for p in path):
            hs_pts = self.high_symmetry_points()
            if hs_pts is None:
                raise ValueError(f"Cannot resolve point names for {type(self).__name__}. "
                               "Use explicit fractional coordinates instead.")
            resolved_path = hs_pts.get_path_points(path)
        elif path is None:
            # Use default path or high_symmetry_points
            hs_pts = self.high_symmetry_points()
            if hs_pts is not None:
                resolved_path = hs_pts
            else:
                raise ValueError(f"No default path for {type(self).__name__}. Specify path explicitly.")
        else:
            resolved_path = path
        
        return extract_bz_path_data(
            self, k_vectors, k_vectors_frac, values, resolved_path,
            points_per_seg=points_per_seg, return_result=return_result
        )
    
    # ------------------------------------------------------------------
    #! Boundary fluxes
    # ------------------------------------------------------------------
    
    @property
    def flux(self) -> BoundaryFlux: 
        return self._flux
    
    @flux.setter
    def flux(self, value: Optional[Union[float, Mapping[Union[str, LatticeDirection], float]]]):
        self._flux = _normalize_flux_dict(value)
        # When flux changes the BC becomes TWISTED (if non-trivial)
        if self._flux is not None and self._flux.is_nontrivial:
            self._bc = LatticeBC.TWISTED

    def set_flux(self, value: Optional[Union[float, Mapping[Union[str, LatticeDirection], float]]], *, reinit: bool = True) -> None:
        """
        Set boundary flux and optionally recalculate k-vectors, DFT, and neighbors.

        Parameters
        ----------
        value : float, Mapping, or None
            New flux specification (see :func:`_normalize_flux_dict`).
        reinit : bool
            If ``True`` (default), recalculate reciprocal vectors, k-vectors,
            DFT matrix, and neighbor lists to be consistent with the new flux.
        """
        self.flux = value   # use the property setter
        if reinit:
            self.calculate_k_vectors()
            if self.Ns < Lattice._DFT_LIMIT_SITES:
                self.calculate_dft_matrix()
            self.calculate_nn()
            self.calculate_nnn()

    @property
    def has_flux(self) -> bool:
        """``True`` when a non-trivial boundary flux is attached."""
        return self._flux is not None and bool(self._flux)

    @property
    def is_twisted(self) -> bool:
        """``True`` when the boundary conditions are TWISTED."""
        return self._bc is LatticeBC.TWISTED

    @property
    def is_topological(self) -> bool:
        r"""
        ``True`` when the lattice carries a non-trivial boundary flux.

        A non-trivial flux (mod :math:`2\pi`) introduces a measurable Aharonov-Bohm
        phase and may change the topological sector of the ground state.
        """
        return self.has_flux

    def flux_summary(self) -> str:
        """Return a human-readable summary of the boundary-flux configuration."""
        if self._flux is None:
            return "No boundary flux (standard BC)"
        parts = []
        for d in LatticeDirection:
            phi     = self._flux.get(d)
            phase   = self._flux.phase(d)
            parts.append(f"  {d.name}: phi={phi:.4f} rad  ->  exp(i*phi)={phase:.4f}")
        trivial = "TRIVIAL" if self._flux.is_trivial else "NON-TRIVIAL"
        return f"Boundary fluxes ({trivial}):\n" + "\n".join(parts)

    # ------------------------------------------------------------------

    def boundary_phase(self, direction: LatticeDirection, winding: int = 1) -> complex:
        """
        Return the complex phase accumulated after crossing the boundary along ``direction``.
        
        Parameters:
        -----------
        direction : LatticeDirection
            The lattice direction (X, Y, or Z).
        winding : int
            The winding number (number of times crossing the boundary).
        Returns:
        --------
        complex
            The complex phase factor e^{i * flux * winding}.
        """
        if self._flux is None:
            return 1.0
        return self._flux.phase(direction, winding=winding)
    
    def boundary_phases(self) -> np.ndarray:
        """
        Return a lookup table of complex boundary phases.

        Returns
        -------
        table : np.ndarray, shape ``(3, Ns+1)``
            ``table[d, w]`` is ``exp(i * w * phi_d)`` for direction *d* and
            winding number *w*.
        """
        ndirs   = 3
        ns      = self.ns
        table   = np.ones((ndirs, ns + 1), dtype=np.complex128)
        if self._flux is not None:
            for d in LatticeDirection:
                for w in range(ns + 1):
                    table[d.value, w] = self.boundary_phase(d, winding=w)
        return table

    def boundary_phase_from_winding(self, wx: int, wy: int, wz: int) -> complex:
        """
        Return total complex boundary phase accumulated from winding numbers.
        If no winding (all zero), returns real 1.0.
        """
        if wx == 0 and wy == 0 and wz == 0:
            return 1.0
        phase = 1.0
        if wx != 0:
            phase *= self.boundary_phase(LatticeDirection.X, winding=wx)
        if wy != 0:
            phase *= self.boundary_phase(LatticeDirection.Y, winding=wy)
        if wz != 0:
            phase *= self.boundary_phase(LatticeDirection.Z, winding=wz)
        return phase if np.iscomplexobj(phase) and not np.isreal(phase) else float(np.real(phase))

    def bond_winding(self, i: int, j: int) -> tuple[int, int, int]:
        """
        Compute how many times a bond (i -> j) crosses the periodic boundary
        in each lattice direction.
        
        Returns (wx, wy, wz), where each entry is 0 if no crossing,
        +1 if wrapped positively, -1 if wrapped negatively.
        
        Parameters:
        -----------
        i : int
            Index of the starting lattice site.
        j : int
            Index of the ending lattice site.
        Returns:
        --------
        tuple[int, int, int]
            A tuple indicating the winding numbers (wx, wy, wz) for the bond from site i to site j.
        """
        i, j        = int(i), int(j)
        x1, y1, z1  = self.get_coordinates(i)
        x2, y2, z2  = self.get_coordinates(j)
        wx          = 0
        wy          = 0
        wz          = 0

        # detect wrapping based on system size
        if abs(x2 - x1) > self.Lx // 2:                     # assume even sizes, we wrap when crossing half the system
            wx = -1 if x2 > x1 else +1
        if self.dim > 1 and abs(y2 - y1) > self.Ly // 2:    # assume even sizes, we wrap when crossing half the system
            wy = -1 if y2 > y1 else +1
        if self.dim > 2 and abs(z2 - z1) > self.Lz // 2:    # assume even sizes, we wrap when crossing half the system
            wz = -1 if z2 > z1 else +1

        return (wx, wy, wz)

    def bond_phase(self, i: int, j: int) -> complex:
        r"""
        Return the complex hopping phase factor for the bond :math:`i \to j`.

        For bonds that do **not** cross a periodic boundary, this is 1.
        For boundary-crossing bonds under TWISTED BC, the phase is
        :math:`\exp(i\,\phi_\mu)` for each direction :math:`\mu` in which
        the bond wraps.

        This is the factor that should multiply the bare hopping amplitude
        in real-space Hamiltonian construction.

        Parameters
        ----------
        i, j : int
            Source and target site indices.

        Returns
        -------
        complex
            Phase factor (unit modulus).
        """
        if self._flux is None:
            return 1.0
        wx, wy, wz = self.bond_winding(i, j)
        return self.boundary_phase_from_winding(wx, wy, wz)

    def hopping_matrix_with_flux(self, *, include_nnn: bool = False) -> np.ndarray:
        r"""
        Build an :math:`N_s \times N_s` matrix of complex hopping amplitudes
        that includes the Peierls phases from boundary fluxes.

        Diagonal is zero.  Off-diagonal ``H[i,j] = t_{ij} * phase(i->j)``
        where ``t_{ij} = 1`` for all connected pairs and ``phase`` is the
        product of boundary phases along directions that the bond wraps.

        Parameters
        ----------
        include_nnn : bool
            If ``True``, include next-nearest-neighbor hoppings as well.

        Returns
        -------
        H : np.ndarray, shape ``(Ns, Ns)``
            Complex hopping matrix.
        """
        Ns  = self.Ns
        H   = np.zeros((Ns, Ns), dtype=complex)
        for i in range(Ns):
            for j in self._nn[i]:
                if self.wrong_nei(j):
                    continue
                j = int(j)
                if 0 <= j < Ns:
                    H[i, j] = self.bond_phase(i, j)
                    
        if include_nnn and self._nnn is not None:
            for i in range(Ns):
                for j in self._nnn[i]:
                    if self.wrong_nei(j):
                        continue
                    j = int(j)
                    if 0 <= j < Ns:
                        H[i, j] += self.bond_phase(i, j)
        return H

    # ------------------------------------------------------------------
    #! Chirality helpers
    # ------------------------------------------------------------------
    
    def get_nnn_middle_sites(self, i: int, j: int, orientation: Optional[str] = None) -> list[int]:
        """
        Return the list of 'middle' sites l that are nearest neighbors
        of both i and j - i.e., sites forming two-step NNN paths i-l-j.

        Works for any lattice that implements get_nn(site, idx)
        and get_nn_num(site).

        Parameters
        ----------
        i, j : int
            Site indices.
        orientation : {'anticlockwise', 'clockwise', None}, optional
            If provided, will sort/choose based on geometric angle.
            Default: None (return all middle sites).

        Returns
        -------
        list[int]
            List of middle-site indices (can be 0, 1, or 2 elements).
        """
        nn_i = [self.get_nn(i, k) for k in range(self.get_nn_num(i))]
        nn_j = [self.get_nn(j, k) for k in range(self.get_nn_num(j))]
        mids = list(set(nn_i).intersection(nn_j))
        if not mids or orientation is None:
            return mids

        # Optional: choose one by local geometry
        if len(mids) > 1:
            ri      = np.array(self.get_coordinates(i))
            rj      = np.array(self.get_coordinates(j))
            centers = []
            for l in mids:
                rl      = np.array(self.get_coordinates(l))
                cross_z = np.cross(rl - ri, rj - rl)[-1]
                centers.append((l, cross_z))
            if orientation.lower().startswith("anti"):
                mids    = [l for (l, cz) in centers if cz > 0]
            elif orientation.lower().startswith("clock"):
                mids    = [l for (l, cz) in centers if cz < 0]
        return mids

    def get_chirality_sign(self, i: int, j: int, normal: Optional[np.ndarray] = None, orientation: Optional[str] = None) -> int:
        r"""
        Compute the local orientation (chirality) sign \nu_{ij} = \pm 1 for a NNN pair (i,j),
        defined by the cross product of the two bond vectors i-l and l-j.

        Works for any 2D or quasi-2D lattice with known site coordinates.

        Parameters
        ----------
        i, j : int
            Site indices (next-nearest neighbors).
        normal : np.ndarray, optional
            Orientation of the lattice plane (default: +z for 2D).

        Returns
        -------
        int
            +1 for anticlockwise, -1 for clockwise, 0 if not a valid NNN pair.
        """
        if self.dim < 2:
            raise ValueError("Chirality sign is only defined for 2D or higher-dimensional lattices.")
        
        # Default normal: +z
        if normal is None:
            normal = np.array([0, 0, 1.0])

        # find common neighbor(s)
        mids = self.get_nnn_middle_sites(i, j, orientation=orientation)
        if not mids:
            return 0

        # Choose one middle site (if multiple, pick the first or average)
        l       = mids[0]
        ri      = np.array(self.get_coordinates(i), dtype=float)
        rj      = np.array(self.get_coordinates(j), dtype=float)
        rl      = np.array(self.get_coordinates(l), dtype=float)

        d1      = rl - ri
        d2      = rj - rl
        cross   = np.cross(d1, d2)
        sign    = np.sign(np.dot(cross, normal))

        return int(sign)

    # ------------------------------------------------------------------
    #! Bond type helper
    # ------------------------------------------------------------------

    def bond_type(self, i: int, j: int) -> str:
        """
        Determine the bond type between sites i and j.

        Parameters
        ----------
        i, j : int
            Site indices.

        Returns
        -------
        str
            'nn' for nearest neighbor, 'nnn' for next-nearest neighbor, 'none' otherwise.
        """
        i, j = int(i), int(j)
        if int(j) in self.nn[i]:
            return 'nn'
        elif int(j) in self.nnn[i]:
            return 'nnn'
        else:
            return 'none'

    # ------------------------------------------------------------------
    #! Boundary helpers
    # ------------------------------------------------------------------

    def periodic_flags(self) -> Tuple[bool, bool, bool]:
        """
        Return booleans indicating whether (x, y, z) directions are periodic.

        TWISTED boundary conditions are topologically equivalent to PBC
        (the lattice is still a torus), so all three directions are periodic.
        """
        match self._bc:
            case LatticeBC.PBC:
                return True, True, True
            case LatticeBC.OBC:
                return False, False, False
            case LatticeBC.MBC:
                return True, False, False
            case LatticeBC.SBC:
                return False, True, False
            case LatticeBC.TWISTED:
                # Twisted BCs are periodic with extra phases on boundary hops
                return True, True, True
            case _:
                raise ValueError(f"Unsupported boundary condition {self._bc!r}")

    def is_periodic(self, direction: Optional[LatticeDirection] = None, allow_twisted: bool = True) -> bool:
        """
        Check if a given direction has periodic boundary conditions.
        """
        if direction is None:
            return self.bc == LatticeBC.PBC or (allow_twisted and self.bc == LatticeBC.TWISTED)
        
        flags = self.periodic_flags()
        index = {LatticeDirection.X: 0, LatticeDirection.Y: 1, LatticeDirection.Z: 2}[direction]
        return bool(flags[index])
    
    @property
    def typek(self):                return self._type
    @typek.setter
    def typek(self, value):         self._type = value
    
    @property
    def spatial_norm(self):         return self._spatial_norm
    @spatial_norm.setter
    def spatial_norm(self, value):  self._spatial_norm = value

    # -----------------------------------------------------------------------------
    
    def site_index(self, x : int, y : int, z : int):
        """Convert (x, y, z) coordinates to a unique site index (row-major).
        
        Default implementation uses standard lexicographic ordering.
        Override in subclasses if a different indexing convention is needed.
        """
        return z * (self._lx * self._ly) + y * self._lx + x

    # -----------------------------------------------------------------------------
    #! SITE HELPERS
    # -----------------------------------------------------------------------------

    def site_diff(
        self,
        i: Union[int, tuple],
        j: Union[int, tuple],
        *,
        minimum_image: bool = False,
        real_space: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Return the displacement ``i -> j`` with optional PBC minimum-image wrapping.

        Parameters
        ----------
        i, j : int or tuple
            Site indices or explicit coordinates.
        minimum_image : bool, default=False
            If True, wrap each periodic direction to the shortest displacement.
        real_space : bool, default=False
            If True and ``i, j`` are site indices, return displacement in real-space
            vectors (uses :meth:`displacement`). Otherwise use lattice coordinates.
        """
        if real_space and isinstance(i, int) and isinstance(j, int):
            dr = np.asarray(self.displacement(i, j, minimum_image=minimum_image), dtype=float).reshape(-1)
            if dr.size < 3:
                dr = np.pad(dr, (0, 3 - dr.size), mode="constant")
            return float(dr[0]), float(dr[1]), float(dr[2])

        c1 = np.asarray(self.get_coordinates(i) if isinstance(i, int) else i, dtype=float).reshape(-1)
        c2 = np.asarray(self.get_coordinates(j) if isinstance(j, int) else j, dtype=float).reshape(-1)
        if c1.size < 3:
            c1 = np.pad(c1, (0, 3 - c1.size), mode="constant")
        if c2.size < 3:
            c2 = np.pad(c2, (0, 3 - c2.size), mode="constant")

        delta = c2[:3] - c1[:3]
        if minimum_image:
            flags   = self.periodic_flags()
            dims    = (float(self.Lx), float(max(self.Ly, 1)), float(max(self.Lz, 1)))
            for d, is_periodic in enumerate(flags):
                if is_periodic and dims[d] > 0.0:
                    delta[d] -= dims[d] * np.round(delta[d] / dims[d])

        return float(delta[0]), float(delta[1]), float(delta[2])
        
    def site_distance(
        self,
        i: Union[int, tuple],
        j: Union[int, tuple],
        *,
        minimum_image: bool = False,
        real_space: bool = False,
    ) -> float:
        """
        Return Euclidean distance between two sites/coordinates.

        Parameters
        ----------
        minimum_image : bool, default=False
            If True, periodic directions use minimum-image convention.
        real_space : bool, default=False
            If True and inputs are indices, measure in real-space lattice vectors.
        """
        if real_space and isinstance(i, int) and isinstance(j, int):
            return float(self.distance(i, j, minimum_image=minimum_image))
        x, y, z = self.site_diff(i, j, minimum_image=minimum_image, real_space=real_space)
        return float(np.sqrt(x**2 + y**2 + z**2))

    # -----------------------------------------------------------------------------
    #! DFT MATRIX
    # -----------------------------------------------------------------------------

    def calculate_reciprocal_vectors(self):
        '''
        Calculates the reciprocal lattice vectors based on the primitive vectors.
        Always returns 3D vectors (padding with zeros for lower dimensions).
        
        Returns:
        - k1, k2, k3 : Reciprocal lattice vectors (always 3D)
        '''
        self._k1, self._k2, self._k3 = reciprocal_from_real(self.a1, self.a2, self.a3)
        
        # Ensure 3D form (pad zeros if 1D or 2D)
        self._k1    = np.pad(self._k1, (0, 3 - len(self._k1)))
        self._k2    = np.pad(self._k2, (0, 3 - len(self._k2)))
        self._k3    = np.pad(self._k3, (0, 3 - len(self._k3)))
        return self._k1, self._k2, self._k3
    
    def calculate_dft_matrix(self, phase = False, use_fft: bool = False) -> np.ndarray:
        r'''
        Bloch-type DFT matrix on the site basis.
        
        Indices:
            i  = (R, beta)   real-space cell R and sublattice beta
            n  = (k, alpha)  k-point k and sublattice alpha

        Elements:
        $$
            F_{(k,alpha),(R,beta)} =
                1/sqrt(Nc) * delta_{alpha,beta} * exp(-i k . R).
        $$
        This is unitary:
        $$
            F^\dagger F = I_{Ns},   F F^\dagger = I_{Ns},
        $$
        where Ns = Nc * Nb is the total number of sites, Nc is the number of
        unit cells, and Nb is the number of sublattices.
        
        IMPORTANT:
            When boundary fluxes are present (TWISTED BC), the k-grid used to
            build the DFT matrix is shifted by ``phi_mu / (2 pi L_mu)`` in each
            direction, exactly as in :meth:`calculate_k_vectors`.
        
        Note that this DFT matrix does not include basis-dependent phases
        (i.e., exp(-i k . r_basis)).
        
        Calculates the Discrete Fourier Transform (DFT) matrix for the lattice.
        This method can be optimized using FFT (Fast Fourier Transform) in the future.
        Reference: https://en.wikipedia.org/wiki/DFT_matrix
        
        Args:
        - phase (bool): If True, adds a complex phase to the k-vectors.
        
        Returns:
        - DFT matrix (ndarray): The calculated DFT matrix.
        '''
        Ns              = self.Ns
        Lx, Ly, Lz      = self._lx, max(self._ly, 1), max(self._lz, 1)
        Nc              = Lx * Ly * Lz
        Nb              = len(self._basis) if (self._basis is not None and len(self._basis) > 0) else 1  # Avoid division by zero
        
        # Get site coordinates
        cells           = np.asarray(self._cells, dtype=float)  # (Ns, 3)
        if not cells.shape[0] == Ns:
            raise ValueError("Mismatch in number of sites and coordinates.")

        sub_idx         = self.subs # (Ns,)

        # Generate k-vectors (with flux-induced shift when applicable)
        frac_x          = np.linspace(0, 1, Lx, endpoint=False)
        frac_y          = np.linspace(0, 1, Ly, endpoint=False)
        frac_z          = np.linspace(0, 1, Lz, endpoint=False)

        # Apply flux-induced shift to k-grid fractions
        dfx, dfy, dfz   = self._flux_frac_shift()
        frac_x          = frac_x + dfx
        frac_y          = frac_y + dfy
        frac_z          = frac_z + dfz
        
        kx_frac, ky_frac, kz_frac = np.meshgrid(frac_x, frac_y, frac_z, indexing='ij')
        
        b1              = np.asarray(self._k1, float).reshape(3)
        b2              = np.asarray(self._k2, float).reshape(3)
        b3              = np.asarray(self._k3, float).reshape(3)
        
        kgrid           = (kx_frac[..., None] * b1 + 
                           ky_frac[..., None] * b2 + 
                           kz_frac[..., None] * b3)
        k_vectors       = kgrid.reshape(-1, 3) # (Nc, 3)
        
        # Build block DFT matrix
        # Row index: ik*Nb + \alpha (k-point ik, sublattice \alpha)
        # Col index: i (site i in real space)
        F_block         = np.zeros((Nc * Nb, Ns), dtype=complex)    # DFT matrix
        norm            = np.sqrt(Nc)                               # Bloch normalization factor

        # numpy path version (not used in loop)
        phase_matrix    = np.exp(-1j * (k_vectors @ cells.T)) / norm        # (Nc, Ns)
        selector        = (sub_idx[None, :] == np.arange(Nb)[:, None])      # (Nb, Ns)
        F               = phase_matrix[:, None, :] * selector[None, :, :]   # (Nc,Nb,Ns)
        F_block         = F.reshape(Nc * Nb, Ns)
        
        self._dft       = F_block
        return F_block

        # leave loop path
        # for ik in range(Nc):
        #     # Phases for all sites at this k
        #     k       = k_vectors[ik]
        #     phases  = np.exp(-1j * (k @ r_vectors.T)) / norm  # (Ns,)
            
        #     # Fill rows for this k-point (one row per sublattice)
        #     for alpha in range(Nb):
        #         row_idx = ik * Nb + alpha
                
        #         # Only connect to sites of sublattice \alpha
        #         for i in range(Ns):
        #             if sub_idx[i] == alpha:
        #                 F_block[row_idx, i] = phases[i]
        
        # return F_block
    
    # -----------------------------------------------------------------------------
    #! NEAREST NEIGHBORS
    # -----------------------------------------------------------------------------
    
    def get_nei(self, site: int, **kwargs):
        '''
        Returns the nearest neighbors of a given site. 
        
        Args:
            - direction : direction of the lattice (can be X, Y, Z - default is X)        
        '''
        if 'corr_len' in kwargs and 'direction' in kwargs:
            direction   = kwargs.get('direction', LatticeDirection.X)
            corr_len    = kwargs.get('corr_len', 1)
            direction   = self._adjust_direction(direction)
            return self._get_neighbor_with_corr_len(site, direction, corr_len)
        elif 'direction' in kwargs:
            direction   = kwargs.get('direction', LatticeDirection.X)
            direction   = self._adjust_direction(direction)
            return self.get_nn_direction(site, direction)
        else:
            return self._nn[site]

    def get_nei_forward(self, site: int, num : int = -1):
        '''
        Returns the forward nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of nearest neighbors
        Returns:
            - list of nearest neighbors        
        '''
        if num < 0:
            return self._nn_forward[site]
        return self._nn_forward[site][num]
    
    def _adjust_direction(self, direction : Union[LatticeDirection, int]):
        '''
        Adjust the direction to the lattice dimension
        Args:
            - direction : direction of the lattice (can be X, Y, Z - default is X)
        '''
        if self.dim == 1:
            return LatticeDirection.X
        elif self.dim == 2 and LatticeDirection(direction) > LatticeDirection.Y:
            return LatticeDirection.Y
        elif self.dim == 3 and LatticeDirection(direction) > LatticeDirection.Z:
            return LatticeDirection.Z
        return direction

    def _get_neighbor_with_corr_len(self, site : int, direction : Union[LatticeDirection, int], corr_len : int = 1):
        '''
        Returns the neighbor with a correlation length
        - site      : lattice site
        - direction : direction of the lattice
        - corr_len  : correlation length
        Returns:
        - neighbor site (int)
        '''
        
        if self.bc == LatticeBC.PBC:
            return self._pbc_neighbor(site, direction, corr_len)
        elif self.bc == LatticeBC.OBC:
            return self._obc_neighbor(site, direction, corr_len)
        elif self.bc == LatticeBC.MBC:
            return self._mbc_neighbor(site, direction, corr_len)
        elif self.bc == LatticeBC.SBC:
            return self._sbc_neighbor(site, direction, corr_len)
        return (site + corr_len) % self._lxlylz

    # -----------------------------------------------------------------------------
    #! BOUNDARY CONDITIONS HELPERS
    # -----------------------------------------------------------------------------

    def _pbc_neighbor(self, site, direction, corr_len):
        if direction == LatticeDirection.X:
            return (site + corr_len) % self._lx
        elif direction == LatticeDirection.Y:
            return (site + corr_len * self._lx) % self._lxly
        elif direction == LatticeDirection.Z:
            return (site + corr_len * self._lxly) % self._lxlylz

    def _obc_neighbor(self, site, direction, corr_len):
        if direction == LatticeDirection.X:
            return (site + corr_len) if (site + corr_len) < self._lx else self.bad_lattice_site
        elif direction == LatticeDirection.Y:
            return (site + corr_len * self._lx) if (site + corr_len * self._lx) < self._lxly else self.bad_lattice_site
        elif direction == LatticeDirection.Z:
            return (site + corr_len * self._lxly) if (site + corr_len * self._lxly) < self._lxlylz else self.bad_lattice_site

    def _mbc_neighbor(self, site, direction, corr_len):
        if direction == LatticeDirection.X:
            return (site + corr_len) % self._lx
        elif direction == LatticeDirection.Y:
            return (site + corr_len * self._lx) if (site + corr_len * self._lx) < self._lxly else self.bad_lattice_site
        elif direction == LatticeDirection.Z:
            return (site + corr_len * self._lxly) if (site + corr_len * self._lxly) < self._lxlylz else self.bad_lattice_site

    def _sbc_neighbor(self, site, direction, corr_len):
        if direction == LatticeDirection.Y:
            return (site + corr_len * self._lx) % self._lxly
        elif direction == LatticeDirection.X:
            return (site + corr_len) if (site + corr_len) < self._lx else self.bad_lattice_site
        elif direction == LatticeDirection.Z:
            return (site + corr_len * self._lxly) if (site + corr_len * self._lxly) < self._lxlylz else self.bad_lattice_site
        
    # -----------------------------------------------------------------------------
    #! Virtual methods
    # -----------------------------------------------------------------------------
    
    @abstractmethod
    def get_real_vec(self, x : int, y : int, z : int):
        '''
        Returns the real vector given the coordinates. Uses the lattice constants.
        '''
        pass
    
    @abstractmethod
    def get_norm(self, x : int, y : int, z : int):
        '''
        Returns the norm of the vector given the coordinates.
        '''
        pass
    
    @abstractmethod
    def get_nn_direction(self, site : int, direction : LatticeDirection):
        '''
        Returns the nearest neighbors in a given direction.
        '''
        pass
    
    def get_nnn_direction(self, site : int, direction : LatticeDirection):
        '''
        Returns the next nearest neighbors in a given direction.
        '''
        pass
    
    # -----------------------------------------------------------------------------
    #! NEAREST NEIGHBORS HELPERS
    # -----------------------------------------------------------------------------
    
    def wrong_nei(self, nei):
        """
        Check if a given neighbor index is invalid.

        A neighbor is considered invalid if it is:
            - None
            - Equal to self.bad_lattice_site
            - NaN (not a number)
            - Less than 0

        Parameters
        ----------
        nei : Any
            The neighbor index to check.

        Returns
        -------
        bool
            True if the neighbor index is invalid, False otherwise.
        """
        return  nei is None or                  \
                nei == self.bad_lattice_site or \
                np.isnan(nei) or                \
                nei < 0
    
    def get_nn_num(self, site : int):
        '''
        Returns the number of nearest neighbors of a given site.
        
        Args:
        - site : lattice site
        Returns:
        - number of nearest neighbors
        '''
        if self._nn is None:
            return 0
        return len(self.nn[site])
    
    def get_nn(self, site, num : int = -1):
        '''
        Returns the nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of nearest neighbors
        Returns:
            - list of nearest neighbors        
        '''
        if num < 0:
            return self._nn[site]
        if self._nn is None:
            return []
        return self._nn[site][num]
    
    def get_nnn_num(self, site : int):
        '''
        Returns the number of next nearest neighbors of a given site.
        
        Args:
        - site : lattice site
        Returns:
        - number of next nearest neighbors
        '''
        if self._nnn is None:
            return 0
        return len(self._nnn[site])
    
    def get_nnn(self, site, num : int = -1):
        '''
        Returns the next nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of next nearest neighbors
        Returns:
            - list of next nearest neighbors        
        '''
        if num < 0:
            return self._nnn[site]
        if self._nnn is None:
            return []
        return self._nnn[site][num]
    
    # -----------------------------------------------------------------------------
    #! FORWARD NEAREST NEIGHBORS HELPERS
    # -----------------------------------------------------------------------------
    
    def get_nn_forward_num_max(self):
        '''
        Returns the maximum number of forward nearest neighbors in the lattice.
        
        Returns:
        - maximum number of nearest neighbors
        '''
        if (self._nn_max_num is None or self._nn_max_num == 0) and self.nn_forward is not None:
            max_nn = 0
            for site in range(self.ns):
                nn_num = len(self.nn_forward[site])
                if nn_num > max_nn:
                    max_nn = nn_num
            self._nn_max_num = max_nn
        return self._nn_max_num
    
    def get_nn_forward_num(self, site : int):
        '''
        Returns the number of forward nearest neighbors of a given site.
        
        Args:
        - site : lattice site
        Returns:
        - number of nearest neighbors
        '''
        return len(self.nn_forward[site])
    
    def get_nn_forward(self, site : int, num : int = -1):
        '''
        Returns the forward nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of nearest neighbors
        Returns:
            - list of nearest neighbors        
        '''
        if not hasattr(self, '_nn_forward') or self._nn_forward is None:
            return [] if num < 0 else -1
        if num < 0:
            return self._nn_forward[site]
        return self._nn_forward[site][num] if num < len(self._nn_forward[site]) else -1
    
    # -----------------------------------------------------------------------------
    #! FORWARD NEXT NEAREST NEIGHBORS HELPERS
    # -----------------------------------------------------------------------------
    
    def get_nnn_forward_num(self, site : int):
        '''
        Returns the number of forward next nearest neighbors of a given site.
        
        Args:
        - site : lattice site
        Returns:
        - number of next nearest neighbors
        '''
        return len(self.nnn_forward[site])
    
    def get_nnn_forward(self, site : int, num : int = -1):
        '''
        Returns the forward next nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of next nearest neighbors
        Returns:
            - list of next nearest neighbors        
        '''
        if not hasattr(self, '_nnn_forward') or self._nnn_forward is None:
            return [] if num < 0 else -1
        if num < 0:
            return self._nnn_forward[site]
        return self._nnn_forward[site][num] if num < len(self._nnn_forward[site]) else -1
    
    # -----------------------------------------------------------------------------
    #! GENERAL NEIGHBORS HELPERS
    # -----------------------------------------------------------------------------
    
    def neighbors(self, site: int, order=1):
        '''Return neighbors of a site: 1 for nn (all with highest weight), 2 for nnn (all with second-highest), 'all' for both.'''
        if order == 1:
            return self._nn[site]
        elif order == 2:
            return self._nnn[site]
        elif order == 'all':
            if self._adj_mat is not None:
                # return all neighbors from adjacency matrix
                non_zero_indices = np.nonzero(self._adj_mat[site])[0]
                return [i for i in non_zero_indices if i != site]
            else:
                return list(set(self._nn[site]) | set(self._nnn[site]))
        else:
            raise ValueError(f"Invalid neighbor order: {order}")
        
    def neighbors_forward(self, site: int, order=1):
        '''Return forward neighbors of a site: 1 for nn (all with highest weight), 2 for nnn (all with second-highest), 'all' for both.'''
        if order == 1:
            return self._nn_forward[site]
        elif order == 2:
            return self._nnn_forward[site]
        elif order == 'all':
            if self._adj_mat is not None:
                # return all neighbors from adjacency matrix
                non_zero_indices = np.nonzero(self._adj_mat[site])[0]
                return [i for i in non_zero_indices if i != site and i > site]
            else:
                return list(set(self._nn_forward[site]) | set(self._nnn_forward[site]))
        else:
            raise ValueError(f"Invalid neighbor order: {order}")
    
    def any_neighbor(self, site: int, order=1):
        '''Return any neighbor (first) of given order or None.'''
        neigh = self.neighbors(site, order)
        return neigh[0] if neigh else Lattice._BAD_LATTICE_SITE
    
    def any_neighbor_forward(self, site: int, order=1):
        '''Return any forward neighbor (first) of given order or None.'''
        neigh = self.neighbors_forward(site, order)
        return neigh[0] if neigh else Lattice._BAD_LATTICE_SITE

    # =========================================================================
    #! NetKet-inspired convenience API
    # =========================================================================

    @property
    def n_nodes(self) -> int:
        """Number of nodes (sites) in the lattice — alias for ``Ns``."""
        return self._ns

    @property
    def n_edges(self) -> int:
        """Number of unique undirected nearest-neighbour edges."""
        return len(self.edges())

    @property
    def positions(self) -> np.ndarray:
        """Real-space position vectors (same as ``rvectors``)."""
        return self.rvectors

    @property
    def site_offsets(self) -> np.ndarray:
        """Position offsets of sites inside the unit cell (same as ``basis``)."""
        return self._basis

    @property
    def basis_coords(self) -> np.ndarray:
        """
        Integer basis coordinates ``[nx, ny, nz, sub]`` for every site.

        Shape ``(Ns, 4)`` — the first three columns are the cell-index
        triplet and the last column is the sublattice label.
        """
        if self._fracs is None or self._subs is None:
            return None
        return np.column_stack([self._fracs, self._subs])

    @property
    def ndim(self) -> int:
        """Spatial dimensionality of the lattice."""
        return self._dim

    @property
    def extent(self) -> Tuple[int, ...]:
        """Number of unit cells in each direction ``(Lx, Ly, Lz)``."""
        return (self._lx, self._ly, self._lz)

    @property
    def pbc(self) -> Tuple[bool, bool, bool]:
        """Per-axis periodicity flags (alias for ``periodic_flags()``)."""
        return self.periodic_flags()

    # Edge / bond queries

    def edges(self, *, filter_color: Optional[int] = None,
              return_color: bool = False) -> List:
        """
        Return list of nearest-neighbour edges.

        Parameters
        ----------
        filter_color : int, optional
            If given, return only edges whose ``bond_type`` equals this colour.
        return_color : bool
            If *True* each element is ``(i, j, color)``; otherwise ``(i, j)``.

        Returns
        -------
        list[tuple]
            Unique undirected edges ``(i, j)`` with ``i < j``.
        """
        if not hasattr(self, '_bonds') or not self._bonds:
            self.calculate_bonds()

        result = []
        for i, j in self._bonds:
            a, b    = (i, j) if i < j else (j, i)
            c       = self.bond_type(a, b)
            if filter_color is not None and c != filter_color:
                continue
            if return_color:
                result.append((a, b, c))
            else:
                result.append((a, b))
        # Deduplicate (forward list may still have symmetric pairs)
        if not return_color:
            result = sorted(set(result))
        return result

    @property
    def edge_colors(self) -> List[int]:
        """
        Sequence of bond-type colours for every edge in ``edges()``,
        matching the order returned by ``edges()``.
        """
        return [c for (_, _, c) in self.edges(return_color=True)]

    # -- Displacement helpers -----------------------------------------------

    def displacement(self, i: int, j: int, *, minimum_image: bool = True) -> np.ndarray:
        """
        Real-space displacement vector from site *i* to site *j*.

        Parameters
        ----------
        i, j : int
            Site indices.
        minimum_image : bool
            If *True* (default) and the lattice is periodic, return the
            shortest displacement under periodic boundary conditions.

        Returns
        -------
        np.ndarray  shape (3,)
        """
        i, j    = int(i), int(j)
        dr      = self.rvectors[j] - self.rvectors[i]
        if not minimum_image:
            return dr

        # Minimum-image convention using fractional coordinates
        flags   = self.periodic_flags()
        dims    = [self._lx, max(self._ly, 1), max(self._lz, 1)]
        dn      = np.array(self._fracs[j], dtype=float) - np.array(self._fracs[i], dtype=float)
        for d in range(3):
            if flags[d]:
                L       = dims[d]
                dn[d]   -= L * np.round(dn[d] / L)
        dr      = dn[0] * self._a1 + dn[1] * self._a2 + dn[2] * self._a3
        dr      += self._basis[self._subs[j]] - self._basis[self._subs[i]]
        return dr

    def distance(self, i: int, j: int, *, minimum_image: bool = True) -> float:
        """Euclidean distance between sites *i* and *j* (PBC-aware by default)."""
        return float(np.linalg.norm(self.displacement(i, j, minimum_image=minimum_image)))
    
    # -----------------------------------------------------------------------------
    #! Standard getters
    # -----------------------------------------------------------------------------

    def get_coordinates(self, *args):           return self._coordinates    if len(args) == 0 else self._coordinates[args[0]]
    def get_r_vectors(self,*args):              return self._rvectors       if len(args) == 0 else self._rvectors[args[0]]
    def get_k_vectors(self, *args):             return self._kvectors       if len(args) == 0 else self._kvectors[args[0]]
    def get_site_diff(self, i: int, j: int):    return self.get_coordinates(j) - self.get_coordinates(i)
    def get_k_vec_idx(self, sym = False):       pass

    def get_dft(self, *args):
        '''
        Returns the DFT matrix
        '''
        if len(args) == 0:
            return self.dft
        elif len(args) == 1:
            # return row
            return self.dft[args[0]]
        else:
            # return element
            return self.dft[args[0], args[1]]
    
    # -----------------------------------------------------------------------------
    #! Spatial information
    # -----------------------------------------------------------------------------
    
    def get_spatial_norm(self, *args):
        '''
        Returns the spatial norm at lattice site i or all of them
        '''
        if len(args) == 0:
            return self.spatial_norm
        elif len(args) == 1:
            return self.spatial_norm[args[0]]
        elif len(args) == 2:
            return self.spatial_norm[args[0]][args[1]]
        else:
            return self.spatial_norm[args[0]][args[1]][args[2]]

    def get_difference_idx_matrix(self, cut = True) -> list:
        '''
        Returns the matrix with indcies corresponding to a slice from the QMC. 
        A usefull function for reading the position Green's function saved from:
        @url https://github.com/makskliczkowski/DQMC
        The Green's functions are saved in the following manner. If cut is True, data 
        has (2L_i - 1) possible position differences, otherwise we skip the negative ones and use L_i.
        For 1D simulation: 1 column and (2 * Lx - 1) rows for possition differences (-Lx, -Lx + 1, ..., 0, ..., Lx)
        For 2D simulation: (2 * Lx - 1) rows for possition differences (-Lx, -Lx + 1, ..., 0, ..., Lx) and (2 * Ly - 1) columns for possition differences (-Ly, -Ly + 1, ..., 0, ..., Ly)
        For 3D simulation: Same as in 2D but after (2 * Lx - 1) x (2 * Ly - 1) matrix has finished, a new slice for Lz appears for next columns Lz * (2*Ly - 1)
        - cut : if true (2L_i - 1) possible position differences, otherwise we skip the negative ones and use L_i.
        '''
        Lx, Ly, Lz  = self.Lx, self.Ly, self.Lz
        xnum        = 2 * Lx - 1 if cut else Lx
        ynum        = 2 * Ly - 1 if cut else Ly
        znum        = 2 * Lz - 1 if cut else Lz
        
        _slice  = [[[0, 0, 0] for _ in range(ynum * znum)] for _ in range(xnum)]
        for k in range(znum):
            z   = k - (Lz if cut else 0)
            for i in range(xnum):
                x = i - (Lx if cut else 0)
                for j in range(ynum):
                    y = j - (Ly if cut else 0)
                    # x's are the rows and y's (*z's) are the columns 
                    _slice[i][j + k * ynum][0] = x
                    _slice[i][j + k * ynum][1] = y
                    _slice[i][j + k * ynum][2] = z            
        return [[tuple(_slice[i][j]) for j in range(ynum * znum)] for i in range(xnum)]
    
    ############################ ABSTRACT CALCULATORS #############################
    
    def calculate_bonds(self):
        '''
        Calculates the bonds for the lattice using forward nn.
        '''
        self._bonds = []
        for i in range(self.Ns):
            for idx in range(self.get_nn_forward_num(i)):
                j = self.get_nn_forward(i, idx)
                if self.wrong_nei(j):
                    continue
                self._bonds.append((i, j))
        return self._bonds
    
    def calculate_coordinates(self):
        """
        Calculates the coordinates for each lattice site in up to 3D.

        Each site index i corresponds to:
            cell = i // n_basis
            sub  = i % n_basis
        where n_basis = len(self._basis) (e.g., 2 for honeycomb).
        
        Works for any lattice with defined self._a1, _a2, _a3 and self._basis list.
        """
        n_basis             = len(self._basis)
        indices = np.arange(self.Ns)

        cell    = indices // n_basis          # integer division
        sub     = indices % n_basis           # remainder

        nx      =  cell              % self.Lx
        ny      = (cell // self.Lx)  % self.Ly if self._dim >= 2 else np.zeros_like(cell)
        nz      = (cell // (self.Lx  * self.Ly)) % self.Lz if self._dim >= 3 else np.zeros_like(cell)

        R       = nx[:, None] * self._a1 + ny[:, None] * self._a2 + nz[:, None] * self._a3     # lattice vector
        r       = R + self._basis[sub]                              # add basis vector
            
        self._coordinates   = r
        self._cells         = R
        self._fracs         = np.stack((nx, ny, nz), axis=1)
        self._subs          = sub
        return self._coordinates
        
    def calculate_r_vectors(self):
        """
        Calculates the real-space vectors (r) for each site.
        Must match the ordering in calculate_coordinates().
        """
        n_basis = len(self._basis)
        rv = np.zeros((self.Ns, 3))
        
        for i in range(self.Ns):
            cell    = i // n_basis
            sub     = i % n_basis
            
            nx      = cell % self.Lx
            ny      = (cell // self.Lx) % self.Ly if self._dim >= 2 else 0
            nz      = (cell // (self.Lx * self.Ly)) % self.Lz if self._dim >= 3 else 0
            
            rv[i]   = nx * self._a1 + ny * self._a2 + nz * self._a3 + self._basis[sub]
        
        self.rvectors = rv
        return self.rvectors

    def _flux_frac_shift(self) -> Tuple[float, float, float]:
        r"""
        Return the fractional k-grid shift induced by boundary fluxes.

        If the flux in direction :math:`\mu` is :math:`\phi_\mu`, the standard
        fractional coordinate :math:`f_\mu = n_\mu / L_\mu` is shifted to
        :math:`f_\mu + \phi_\mu / (2\pi\,L_\mu)`.

        Returns
        -------
        (dfx, dfy, dfz) : tuple[float, float, float]
        """
        if self._flux is None:
            return (0.0, 0.0, 0.0)
        Ly = self._ly if self._dim >= 2 else 1
        Lz = self._lz if self._dim >= 3 else 1
        return self._flux.k_shift_fractions(self._lx, Ly, Lz)

    def calculate_k_vectors(self):
        """
        Calculates the allowed reciprocal-space k-vectors (momentum grid)
        consistent with the lattice size and primitive reciprocal vectors.

        When boundary fluxes are present (TWISTED BC), the fractional
        coordinates are shifted by :math:`\\phi_\\mu / (2\\pi L_\\mu)` in
        each direction, so that the Bloch condition matches the twisted
        boundary.

        The sampling follows the same fftfreq ordering used by the Bloch
        transform (Γ at index [0,0,0], followed by positive frequencies and
        finally the negative branch).  This keeps the analytic grids aligned
        with the numerically constructed H(k) blocks.
        """
        Lx              = self.Lx
        Ly              = self.Ly if self._dim >= 2 else 1
        Lz              = self.Lz if self._dim >= 3 else 1

        frac_x          = np.fft.fftfreq(Lx)
        frac_y          = np.fft.fftfreq(Ly)
        frac_z          = np.fft.fftfreq(Lz)

        # Apply flux-induced shift to k-grid fractions
        dfx, dfy, dfz   = self._flux_frac_shift()
        frac_x          = frac_x + dfx
        frac_y          = frac_y + dfy
        frac_z          = frac_z + dfz

        kx_frac, ky_frac, kz_frac = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")

        k_grid = (
              kx_frac[..., None] * self._k1
            + ky_frac[..., None] * self._k2
            + kz_frac[..., None] * self._k3
        )

        self.kvectors       = k_grid.reshape(-1, 3)
        self.kvectors_frac  = np.stack([kx_frac, ky_frac, kz_frac], axis=-1).reshape(-1, 3)
        return self.kvectors
    
    def filter_k_vectors(self, qx: Optional[int] = None, qy: Optional[int] = None, qz: Optional[int] = None) -> np.ndarray:
        """
        Filters the k-vectors to find those matching the specified fractional components.

        Args:
            qx (int): Fractional component in the x-direction.
            qy (int, optional): Fractional component in the y-direction. Defaults to None.
            qz (int, optional): Fractional component in the z-direction. Defaults to None.

        Returns:
            np.ndarray: Array of indices of k-vectors matching the specified components.
        """
        if self.kvectors_frac is None:
            raise ValueError("k-vectors have not been calculated yet.")
        
        mask = np.ones(len(self.kvectors_frac), dtype=bool)
        
        if qx is not None:
            mask &= (self.kvectors_frac[:, 0] == qx / self.Lx)
        if qy is not None and self._dim >= 2:
            mask &= (self.kvectors_frac[:, 1] == qy / self.Ly)
        if qz is not None and self._dim >= 3:
            mask &= (self.kvectors_frac[:, 2] == qz / self.Lz)
        
        return np.where(mask)[0]

    def translation_operators(self):
        """Return translation matrices T1, T2, T3 on the one-hot basis."""
        self._T1, self._T2, self._T3 = build_translation_operators(self)
        return self._T1, self._T2, self._T3

    # -----------------------------------------------------------------------------
    #! Spatial norm calculators
    # -----------------------------------------------------------------------------

    def calculate_norm_sym(self):
        """
        Calculate a symmetry-normalization measure for each site.

        Default: Euclidean norm of the coordinate vector.
        Override in subclasses for lattice-specific behaviour.
        """
        self._spatial_norm = { i: np.linalg.norm(self._coordinates[i]) for i in range(self._ns) }
    
    # -----------------------------------------------------------------------------
    #! Nearest neighbors
    # -----------------------------------------------------------------------------
    
    def _calculate_nn_pbc(self):        return self.calculate_nn_in(True, True, True)
    def _calculate_nn_obc(self):        return self.calculate_nn_in(False, False, False)
    def _calculate_nn_mbc(self):        return self.calculate_nn_in(True, False, False)
    def _calculate_nn_sbc(self):        return self.calculate_nn_in(False, True, False)
    
    @abstractmethod
    def calculate_nn_in(self, pbcx : bool, pbcy : bool, pbcz : bool): pass

    def calculate_nn(self):
        '''
        Calculates the nearest neighbors.

        For TWISTED boundary conditions the neighbor *connectivity* is
        identical to PBC — the flux phases are applied separately when
        building the Hamiltonian or the DFT matrix.
        '''
        
        match (self._bc):
            case LatticeBC.PBC:
                self._calculate_nn_pbc()
            case LatticeBC.OBC:
                self._calculate_nn_obc()
            case LatticeBC.MBC:
                self._calculate_nn_mbc()
            case LatticeBC.SBC:
                self._calculate_nn_sbc()
            case LatticeBC.TWISTED:
                # Twisted BC: same neighbor connectivity as PBC
                self._calculate_nn_pbc()
            case _:
                raise ValueError("The boundary conditions are not implemented.")

    def calculate_plaquettes(self, use_obc: bool = True): raise NotImplementedError("Plaquette calculation not implemented for this lattice.")

    def calculate_wilson_loops(self):
        """
        Calculates the Wilson loops (non-contractible loops) for the lattice based on its boundary conditions.
        Returns a list of lists, where each inner list contains the site indices of a Wilson loop.
        
        Assumes standard lexicographic site indexing (x + y*Lx + z*Lx*Ly).
        """
        loops                           = []
        is_pbc_x, is_pbc_y, is_pbc_z    = self.periodic_flags()
        
        # Wilson loop along X (at y=0, z=0)
        if is_pbc_x and self.lx > 0:
            loops.append([x for x in range(self.lx)])
            
        # Wilson loop along Y (at x=0, z=0)
        if is_pbc_y and self.dim >= 2 and self.ly > 0:
            loops.append([y * self.lx for y in range(self.ly)])
            
        # Wilson loop along Z (at x=0, y=0)
        if is_pbc_z and self.dim >= 3 and self.lz > 0:
            loops.append([z * self.lx * self.ly for z in range(self.lz)])
            
        return loops

    # -----------------------------------------------------------------------------
    #! Next nearest neighbors
    # -----------------------------------------------------------------------------
    
    def _calculate_nnn_pbc(self):       return self.calculate_nnn_in(True, True, True)
    def _calculate_nnn_obc(self):       return self.calculate_nnn_in(False, False, False)
    def _calculate_nnn_mbc(self):       return self.calculate_nnn_in(True, False, False)
    def _calculate_nnn_sbc(self):       return self.calculate_nnn_in(False, True, False)
    
    @abstractmethod
    def calculate_nnn_in(self, pbcx : bool, pbcy : bool, pbcz : bool): pass
    
    def calculate_nnn(self):
        '''
        Calculates the next nearest neighbors.

        Like :meth:`calculate_nn`, each ``calculate_nnn_in`` implementation
        is expected to set ``self._nnn`` (and optionally ``self._nnn_forward``)
        directly.  The return value—if any—is stored as a fallback.
        '''
        
        match (self._bc):
            case LatticeBC.PBC:
                self._calculate_nnn_pbc()
            case LatticeBC.OBC:
                self._calculate_nnn_obc()
            case LatticeBC.MBC:
                self._calculate_nnn_mbc()
            case LatticeBC.SBC:
                self._calculate_nnn_sbc()
            case LatticeBC.TWISTED:
                # Twisted BC: same neighbor connectivity as PBC
                self._calculate_nnn_pbc()
            case _:
                raise ValueError("The boundary conditions are not implemented.")

    # -----------------------------------------------------------------------------
    #! Saving the lattice
    # -----------------------------------------------------------------------------
    
    def adjacency_matrix(self, 
                        sparse              : bool  = False, 
                        save                : bool  = True, 
                        *,
                        mode                : str   = 'binary',
                        include_self        : bool  = False,
                        include_nnn         : bool  = False,
                        typed_self_separate : bool  = True, 
                        n_types             : int   = 3
                        ) -> np.ndarray:
        r"""
        Construct adjacency matrix A_ij = 1 if i and j are neighbors.

        Parameters:
            save        (bool):
                save the adjacency matrix in the lattice object for future use.
            mode        (str):
                'binary' : 
                    A_ij = 1 if i and j are neighbors, 0 otherwise.
                'typed' : 
                    A_ij = weight of the bond between i and  
                    j (1 for nn, 2 for nnn, etc.), 0 otherwise.
            include_self   (bool):
                include self-connections (diagonal elements) if True.
            include_nnn    (bool):
                include next-nearest neighbors if True.
            typed_self_separate (bool):
                if True, self-connections are given a unique weight (n_types) 
                to distinguish them from other types of connections.
            n_types     (int):
                number of different neighbor types (nn, nnn, etc.) to consider.
            sparse      (bool):
                return a scipy.sparse CSR matrix if True.

        Returns:
            A (ndarray or sparse CSR): adjacency matrix of size (Ns, Ns).
        """
        mode    = str(mode).lower()
        
        if mode not in ("binary", "typed"):
            raise ValueError("mode must be 'binary' or 'typed'")

        Ns      = int(self.ns)

        # caching: keep separate caches per mode (so you don't collide)
        if not hasattr(self, "_adj_cache"):
            self._adj_cache = {}
        
        cache_key = (mode, include_self, include_nnn, typed_self_separate, n_types, sparse)
        if save and cache_key in self._adj_cache:
            return self._adj_cache[cache_key]

        # =====================================================================
        # Binary adjacency
        # =====================================================================
        if mode == "binary":
            A = np.zeros((Ns, Ns), dtype=np.float32)

            # Nearest neighbors from _nn lists
            if getattr(self, "_nn", None):
                for i in range(Ns):
                    nbrs    = self._nn[i] if self._nn[i] else []
                    for j in nbrs:
                        if self.wrong_nei(j):
                            continue
                        
                        j   = int(j)
                        if 0 <= j < Ns and j != i:
                            A[i, j] = 1.0
                            A[j, i] = 1.0

            # Optional NNN
            if include_nnn and getattr(self, "_nnn", None):
                for i in range(Ns):
                    nbrs = self._nnn[i] if self._nnn[i] else []
                    for j in nbrs:
                        if self.wrong_nei(j):
                            continue
                        j = int(j)
                        if 0 <= j < Ns and j != i:
                            A[i, j] = 1.0
                            A[j, i] = 1.0

            if include_self:
                np.fill_diagonal(A, 1.0)

            if sparse:
                import scipy.sparse as sp
                A = sp.csr_matrix(A)

            if save:
                self._adj_cache[cache_key] = A
            return A

        # =====================================================================
        # Typed adjacency (e.g., Kitaev x/y/z bonds)
        # =====================================================================
        # We need a way to get bond types. We support multiple lattice APIs:
        #   1) self.bonds_by_type()             -> list/tuple length n_types, each a list of (i,j)
        #   2) self.edge_types dict {(i,j): t}  or {(min(i,j),max(i,j)): t}
        #   3) self._bond_types list of (i,j,t)
        # Otherwise: we raise with a clear message.
        A_types = np.zeros((n_types, Ns, Ns), dtype=np.float32)

        def add_typed_edge(i, j, t):
            i = int(i); j = int(j); t = int(t)
            
            if not (0 <= i < Ns and 0 <= j < Ns):
                return
            if i == j:
                return
            if not (0 <= t < n_types):
                raise ValueError(f"Bond type t={t} outside [0,{n_types})")
            
            A_types[t, i, j] = 1.0
            A_types[t, j, i] = 1.0

        used = False

        # (1) bonds_by_type()
        if hasattr(self, "bonds_by_type") and callable(self.bonds_by_type):
            by_t = self.bonds_by_type()
            if by_t is not None and len(by_t) >= n_types:
                for t in range(n_types):
                    for (i, j) in by_t[t]:
                        add_typed_edge(i, j, t)
                used = True

        # (2) edge_types mapping
        if not used and hasattr(self, "edge_types"):
            et = getattr(self, "edge_types")
            if isinstance(et, dict) and len(et) > 0:
                for (i, j), t in et.items():
                    add_typed_edge(i, j, t)
                used = True

        # (3) internal list of typed bonds
        if not used and hasattr(self, "_bond_types"):
            bt = getattr(self, "_bond_types")
            if bt is not None and len(bt) > 0:
                for (i, j, t) in bt:
                    add_typed_edge(i, j, t)
                used = True

        # Fallback: if lattice has only _nn but no type info, you *cannot* make Kitaev-typed adjacency
        if not used:
            raise ValueError(
                "typed adjacency requested, but no bond-type information found on this Lattice.\n"
                "Add one of:\n"
                "  - Lattice.bonds_by_type() -> list length 3 of (i,j) bonds for x,y,z\n"
                "  - Lattice.edge_types dict {(i,j): t}\n"
                "  - Lattice._bond_types list of (i,j,t)\n"
                "Otherwise use mode='binary'."
            )

        A_self = None
        if include_self:
            if typed_self_separate:
                A_self = np.eye(Ns, dtype=np.float32)
            else:
                # add self-loops into every type channel (rarely what you want)
                for t in range(n_types):
                    np.fill_diagonal(A_types[t], 1.0)

        if sparse:
            import scipy.sparse as sp
            A_types = np.array([sp.csr_matrix(A_types[t]) for t in range(n_types)], dtype=object)
            if A_self is not None:
                A_self = sp.csr_matrix(A_self)

        out = (A_types, A_self)
        if save:
            self._adj_cache[cache_key] = out
        return out
    
    def print_neighbors(self, logger : 'Logger'):
        """
        Logs the neighbors of each site in the lattice using the provided logger.

        For each site in the lattice, this method retrieves its nearest neighbors and logs their indices.
        Additionally, for each neighbor, it logs detailed information using a higher verbosity level.

        Args:
            logger: An object with an `info` method for logging messages. The `info` method should accept
                    parameters `lvl` (int) for verbosity level and `color` (str) for message color.

        """
        def print_nei(msg, lvl = 1, color = 'green'):
            if logger is not None:
                logger.info(msg, lvl = lvl, color = color)
            else:
                print(msg)
        
        for i in range(self.ns):
            neighbors = self.get_nn(i)
            print_nei(f"Neighbors of site {i}: {neighbors}", lvl = 1, color = 'green')
            for j in range(len(neighbors)):
                nei_in = self.get_nei(i, j)
                print_nei(f"Neighbor {j} of site {i}: {nei_in}", lvl = 2, color = 'blue')

    def print_forward(self, logger : 'Logger'):
        """
        Logs the forward nearest neighbors for each site in the lattice.

        For each site in the lattice, this method retrieves the number of forward nearest neighbors
        and logs their indices using the provided logger. The method outputs two levels of information:
        - Level 1 (green): Lists the neighbors of each site.
        - Level 2 (blue): Details each neighbor's index for the site.

        Args:
            logger: A logging object with an `info` method that accepts a message, 
                    a logging level (`lvl`), and a color (`color`).
        """
        
        def print_nei(msg, lvl = 1, color = 'green'):
            if logger is not None:
                logger.info(msg, lvl = lvl, color = color)
            else:
                print(msg)
        
        for i in range(self.ns):
            neighbors = self.get_nn_forward_num(i)
            print_nei(f"Neighbors of site {i}: {neighbors}", lvl = 1, color = 'green')
            for j in range(neighbors):
                nei_in = self.get_nn_forward(i, j)
                print_nei(f"Neighbor {j} of site {i}: {nei_in}", lvl = 2, color = 'blue')

    # -------------------------------------------------------------------------
    #! Bloch Transform & Basis Operations
    # -------------------------------------------------------------------------

    def get_geometric_encoding(self, *, tol=1e-6):
        """
        Map each site i to (cell_idx, sub_idx) purely from geometry.

        Returns
        -------
        cell_idx : (Ns,) int array in [0, Nc-1]
        sub_idx  : (Ns,) int array in [0, Nb-1]
        """

        coords      = np.asarray(self._coordinates, float)          # (Ns,3)
        a1          = np.asarray(self._a1, float).reshape(3)
        a2          = np.asarray(self._a2, float).reshape(3)
        a3          = np.asarray(self._a3, float).reshape(3)
        A           = np.column_stack([a1, a2, a3])                 # (3,3)
        Ainv        = np.linalg.inv(A)

        Nb          = len(self._basis)
        basis       = np.zeros((Nb,3), float)
        basis[:, :self._basis.shape[1]] = np.asarray(self._basis, float)

        Lx, Ly, Lz  = self._lx, max(self._ly,1), max(self._lz,1)
        Nc          = Lx*Ly*Lz
        Ns          = coords.shape[0]

        # fractional cell coords (may be non-integers due to numeric noise)
        frac        = (Ainv @ coords.T).T                           # (Ns,3)
        # wrap to [0,L) and round to nearest cell
        cx          = np.mod(np.rint(frac[:,0]), Lx).astype(int)
        cy          = np.mod(np.rint(frac[:,1]), max(Ly,1)).astype(int) if self._dim >= 2 else np.zeros(Ns, int)
        cz          = np.mod(np.rint(frac[:,2]), max(Lz,1)).astype(int) if self._dim >= 3 else np.zeros(Ns, int)

        # residual within unit cell
        Rrec        = (cx[:,None]*a1[None,:] + cy[:,None]*a2[None,:] + cz[:,None]*a3[None,:])   # (Ns,3)
        r_in        = coords - Rrec

        # assign sublattice by nearest basis vector
        d2          = ((r_in[:,None,:] - basis[None,:,:])**2).sum(axis=2)   # (Ns,Nb)
        sub         = np.argmin(d2, axis=1)
        if not np.all(np.take_along_axis(d2, sub[:,None], axis=1)[:,0] < tol):
            # If this trips, increase tol or check a1,a2,a3/basis consistency
            raise ValueError("Some sites could not be matched to a basis position; adjust tolerance or geometry.")

        cell        = ((cz*Ly + cy)*Lx + cx).astype(int)            # (Ns,)
        return cell, sub

    # ============================================================
    #  INVERSE BLOCH TRANSFORM & K-SPACE OPERATIONS
    # ============================================================

    def realspace_from_kspace(
        self,
        H_k                     : np.ndarray,
        *,
        block_diag              : bool = True,
        kgrid                   : Optional[np.ndarray] = None):
        """
        Inverse Bloch transform: H(k) blocks -> H_real (Ns x Ns).
        
        See lattice_kspace.realspace_from_kspace for full documentation.
        """
        from .tools.lattice_kspace import realspace_from_kspace, full_k_space_transform
        if block_diag is False:
            return full_k_space_transform(lattice=self, mat_k=H_k, inverse=True)
        return realspace_from_kspace(
            lattice = self,
            H_k     = H_k,
            kgrid   = kgrid)

    def kspace_from_realspace(self, mat: np.ndarray, block_diag: bool = False):
        """
        Transform real-space Hamiltonian to k-space.
        
        Parameters
        ----------
        mat : np.ndarray
            Real-space matrix (Ns x Ns)
        block_diag : bool
            If True, return k-space blocks (Lx, Ly, Lz, Nb, Nb)
            If False, return full transformed matrix (Ns x Ns)
            
        Returns
        -------
        If block_diag=True:
            H_k, k_grid, k_frac : k-space blocks and grid
        If block_diag=False:
            H_k_full : full transformed matrix (Ns x Ns)
        """
        from .tools.lattice_kspace import full_k_space_transform, kspace_from_realspace
        if block_diag:
            return kspace_from_realspace(lattice=self, H_real=mat)
        return full_k_space_transform(lattice=self, mat=mat)

    # -------------------------------------------------------------------------
    #! Presentation helpers (text / plots)
    # -----------------------------------------------------------------------------

    def summary_string(self, *, precision: int = 3) -> str:
        """
        Return a textual summary of lattice metadata.
        """
        from .visualization import format_lattice_summary

        return format_lattice_summary(self, precision=precision)

    def real_space_table(self, *, max_rows: int = 10, precision: int = 3) -> str:
        """
        Return a formatted table of real-space vectors.
        """
        from .visualization import format_real_space_vectors

        return format_real_space_vectors(self, max_rows=max_rows, precision=precision)

    def reciprocal_space_table(self, *, max_rows: int = 10, precision: int = 3) -> str:
        """
        Return a formatted table of reciprocal-space vectors.
        """
        from .visualization import format_reciprocal_space_vectors

        return format_reciprocal_space_vectors(self, max_rows=max_rows, precision=precision)

    def brillouin_zone_overview(self, *, precision: int = 3) -> str:
        """
        Return a textual overview of the sampled Brillouin zone.
        """
        from .visualization import format_brillouin_zone_overview

        return format_brillouin_zone_overview(self, precision=precision)

    def describe(self, *, 
                precision               : int = 3,
                max_rows                : int = 10,
                include_vectors         : bool = True,
                include_reciprocal      : bool = True,
                include_brillouin_zone  : bool = True) -> str:
        """
        Combine multiple presentation helpers into a single multi-section string.
        """
        sections: list[str] = [self.summary_string(precision=precision)]

        if include_vectors:
            sections.append("Real-space vectors:\n" + self.real_space_table(max_rows=max_rows, precision=precision))

        if include_reciprocal:
            sections.append(
                "Reciprocal-space vectors:\n" + self.reciprocal_space_table(max_rows=max_rows, precision=precision)
            )

        if include_brillouin_zone:
            sections.append("Brillouin zone:\n" + self.brillouin_zone_overview(precision=precision))

        return "\n\n".join(sections)

    # -----------------------------------------------------------------------------
    #! PLOTTING HELPERS
    # -----------------------------------------------------------------------------
    
    def plot_real_space(self, **kwargs):
        """
        Convenience wrapper returning the matplotlib figure and axes for a real-space scatter plot.
        """
        from .visualization import plot_real_space
        return plot_real_space(self, **kwargs)

    def plot_reciprocal_space(self, **kwargs):
        """
        Convenience wrapper returning the matplotlib figure and axes for a reciprocal-space scatter plot.
        """
        from .visualization import plot_reciprocal_space
        return plot_reciprocal_space(self, **kwargs)

    def plot_brillouin_zone(self, **kwargs):
        """
        Convenience wrapper returning the matplotlib figure and axes for a Brillouin zone plot.
        """
        from .visualization import plot_brillouin_zone
        return plot_brillouin_zone(self, **kwargs)
    
    def plot_structure(self, **kwargs):
        """
        Convenience wrapper returning the matplotlib figure and axes for a detailed lattice structure plot.

        Parameters
        ----------
        show_indices : bool
            If True, annotates nodes with their site indices.
        highlight_boundary : bool
            If True, draws boundary nodes with a distinct color/edge.
        show_axes : bool
            If False, hides the coordinate axes for a cleaner diagram.
        partition_colors : tuple of str, optional
            Colors to use for bipartite/sublattice coloring. If provided, nodes are
            colored based on sublattice parity.
        show_periodic_connections : bool
            If True, indicates wrap-around connections textually or graphically.
        show_primitive_cell : bool
            If True, overlays the primitive unit cell vectors/box.
        ... other kwargs passed to the underlying plotting function (e.g., node size, color map, etc.), see plot_lattice_structure() for details.
        """
        from .visualization import plot_lattice_structure
        return plot_lattice_structure(self, **kwargs)

    @property
    def plot(self):
        """
        Access plotting utilities for this lattice.
        
        Returns a LatticePlotter instance providing methods:
        - real_space(**kwargs)          : Scatter plot of sites.
        - reciprocal_space(**kwargs)    : Scatter plot of reciprocal lattice vectors.
        - brillouin_zone(**kwargs)      : Visualization of the Brillouin Zone.
        - structure(**kwargs)           : Detailed connectivity plot with boundaries.
        
        Example:
            >>> lat.plot.structure(show_indices=True, highlight_boundary=True)
            >>> lat.plot.brillouin_zone()
        """
        from .visualization.plotting import LatticePlotter
        return LatticePlotter(self)

#############################################################################################################
#! SAVE LATTICE HELPERS
#############################################################################################################

def save_bonds(lattice : Lattice, directory : Union[str], filename : str):
    '''
    Saves the bonds of the lattice to a file
    Args:
    - lattice   : lattice model
    - directory : directory to save the file
    - filename  : filename
    
    Returns:
    - True if the file has been saved, False otherwise
    '''
    if lattice.type == LatticeType.HONEYCOMB:
        
        # get the bonds
        bonds   =   -Backend.ones((lattice.ns, 3))
        
        # go through all
        for i in range(lattice.ns):
            
            num_of_nn = len(lattice.get_nn_forward_num(i))
            
            for nn in range(num_of_nn):
                nei = lattice.get_nei_forward(i, nn)
                if nei >= 0:
                    bonds[i, nn] = nei
        # save the bonds
        try:
            HDF5Mod.save_hdf5(directory, filename, bonds)
        except Exception as e:
            print(f"An error has occured while saving the bonds: {e}")
            return False
        return True
    return False
    
#############################################################################################################
#! END OF FILE
#############################################################################################################
