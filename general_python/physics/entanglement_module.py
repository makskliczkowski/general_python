r"""
This file contains the EntanglementModule class.

--------------------------------------------
file    : general_python/physics/entanglement_module.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
--------------------------------------------

Unified entanglement calculation module for both quadratic and many-body Hamiltonians.

Features
--------
- Single-particle correlation matrix methods (fast, for quadratic/non-interacting Hamiltonians)
- Many-body reduced density matrix methods (exact, for any state)
- Arbitrary bipartitions (contiguous and non-contiguous subsystems)
- Multipartite entropy calculations (topological entanglement entropy)
- Wick's theorem verification for quadratic systems
- JAX backend for GPU acceleration
- Mask generation utilities for subsystem selection

Theoretical Background
----------------------
For quadratic (non-interacting) Hamiltonians, the entanglement entropy can be computed
efficiently from the single-particle correlation matrix C_ij = <c_i^dag c_j>:

    S = -Tr[C log C + (1-C) log(1-C)]

This scales as O(L^3) compared to O(2^L) for exact diagonalization.

For many-body states, we use Schmidt decomposition of the wavefunction:
    |psi> = sum_i sqrt(lambda_i) |i_A> |i_B>
    S = -sum_i lambda_i log(lambda_i)

Topological Entanglement Entropy (TEE)
--------------------------------------
For topological phases, the entanglement entropy follows:
    S(A) = alpha * L - gamma + O(1/L)
    
where gamma is the topological entanglement entropy. Using Kitaev-Preskill
or Levin-Wen constructions:
    gamma = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

Examples
--------
Basic usage with quadratic Hamiltonians:
    >>> hamil   = QuadraticHamiltonian(ns=12, ...)
    >>> hamil.diagonalize()
    >>> ent     = hamil.entanglement
    >>> 
    >>> # Define bipartition and calculate entropy
    >>> bipart      = ent.bipartition([0, 1, 2, 3])
    >>> orbitals    = [0, 1, 2, 3, 4, 5]  # occupied states
    >>> S = ent.entropy_correlation(bipart, orbitals)
    
Access correlation matrices:
    >>> C_full      = ent.correlation_matrix(orbitals)
    >>> C_A         = ent.correlation_matrix(orbitals, bipartition=bipart)
    
Batch calculations:
    >>> results     = ent.entropy_multipartition(
    ...     bipartitions=[[0,1], [0,1,2], [0,1,2,3]],
    ...     occupied_orbitals=orbitals
    ... )
    >>> entropies               = results['entropies']
    >>> correlation_matrices    = results['correlation_matrices']
    
JAX backend for GPU acceleration:
    >>> S_jax = ent.entropy_correlation(bipart, orbitals, backend='jax')
    >>> C_jax = ent.correlation_matrix(orbitals, backend='jax')
    
Mask generation utilities:
    >>> masks = MaskGenerator.contiguous(ns=12, size_a=4)  # First 4 sites
    >>> masks = MaskGenerator.alternating(ns=12)           # Even/odd sites
    >>> masks = MaskGenerator.random(ns=12, size_a=6)      # Random 6 sites
    >>> masks = MaskGenerator.kitaev_preskill(ns=12)       # ABC regions for TEE
    
Topological entanglement entropy:
    >>> gamma = ent.topological_entropy(orbitals, construction='kitaev_preskill')
    
Wick's theorem verification:
    >>> is_valid, error = ent.verify_wicks_theorem(orbitals, state)
    
Manual many-body entropy calculations:
    >>> bipart      = ent.bipartition([0, 1, 2, 3])
    >>> S_manual    = ent.entropy_correlation(bipart, orbitals)
"""

import  numpy as np
from    enum                    import Enum
from    typing                  import Union, List, Tuple, Optional, Callable, Dict, Literal, TYPE_CHECKING
from    dataclasses             import dataclass

try:
    import jax                  # type: ignore
    JAX_AVAILABLE               = True
except ImportError:
    JAX_AVAILABLE               = False
    
try:
    from .sp                    import correlation_matrix as Corr
    
    if JAX_AVAILABLE:
        import jax.numpy as jnp
except ImportError as e:
    raise ImportError("Required QES modules not found") from e

if TYPE_CHECKING:
    from ..algebra.utils        import Array
    
###############################################################################
#! Mask Generation Utilities
###############################################################################

class MaskGenerator:
    r"""
    Utility class for generating subsystem masks for entanglement calculations.
    
    Provides convenient methods to create site masks for various bipartition
    geometries, including contiguous, alternating, random, and topological
    (Kitaev-Preskill) constructions.
    
    Examples
    --------
    Basic contiguous mask:
        >>> mask_a = MaskGenerator.contiguous(ns=12, size_a=4)
        >>> print(mask_a)       # array([0, 1, 2, 3])
    
    Alternating (even/odd) sites:
        >>> mask_even, mask_odd = MaskGenerator.alternating(ns=12)
        >>> print(mask_even)    # array([0, 2, 4, 6, 8, 10])
        
    Random subsystem:
        >>> mask = MaskGenerator.random(ns=12, size_a=6, seed=42)
        
    For topological entanglement entropy (Kitaev-Preskill construction):
        >>> regions = MaskGenerator.kitaev_preskill(ns=12)
        >>> A, B, C = regions['A'], regions['B'], regions['C']
    """
    
    @staticmethod
    def contiguous(ns: int, size_a: int, start: int = 0) -> np.ndarray:
        """
        Create a contiguous subsystem mask [start, start+1, ..., start+size_a-1].
        
        Parameters
        ----------
        ns : int
            Total number of sites
        size_a : int
            Size of subsystem A
        start : int
            Starting site index (default: 0)
            
        Returns
        -------
        np.ndarray
            Array of site indices in subsystem A
        """
        if start + size_a > ns:
            # Wrap around for periodic systems
            indices = np.arange(start, start + size_a) % ns
            return np.sort(np.unique(indices))
        return np.arange(start, start + size_a, dtype=np.int64)
    
    @staticmethod
    def alternating(ns: int, offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Create alternating (even/odd) site masks.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        offset : int
            Offset for even sites (0 = sites 0,2,4,...; 1 = sites 1,3,5,...)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (mask_even, mask_odd) site index arrays
        """
        # Optimized: create arrays directly using steps instead of filtering
        start_even  = offset % 2
        mask_even   = np.arange(start_even, ns, 2, dtype=np.int64)
        
        start_odd   = (offset + 1) % 2
        mask_odd    = np.arange(start_odd, ns, 2, dtype=np.int64)
        return mask_even, mask_odd
    
    @staticmethod
    def every_n(ns: int, n: int, start: int = 0) -> np.ndarray:
        """
        Create a mask selecting every n-th site.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        n : int
            Step size (select every n-th site)
        start : int
            Starting site index (default: 0)
            
        Returns
        -------
        np.ndarray
            Array of site indices selected every n-th site
        """
        all_sites   = np.arange(ns, dtype=np.int64)
        indices     = all_sites[(all_sites - start) % n == 0]
        return indices
    
    @staticmethod
    def random(ns: int, size_a: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Create a random subsystem mask.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        size_a : int
            Size of subsystem A
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Sorted array of randomly selected site indices
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(ns, size=size_a, replace=False)
        return np.sort(indices).astype(np.int64)
    
    @staticmethod
    def periodic_interval(ns: int, start: int, size_a: int) -> np.ndarray:
        """
        Create a contiguous mask with periodic boundary conditions.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        start : int
            Starting site index
        size_a : int
            Size of subsystem A
            
        Returns
        -------
        np.ndarray
            Sorted array of site indices (wrapped around if necessary)
        """
        indices = np.arange(start, start + size_a) % ns
        return np.sort(np.unique(indices)).astype(np.int64)
    
    @staticmethod
    def sublattice(ns: int, sublattice_id: int = 0, n_sublattices: int = 2) -> np.ndarray:
        """
        Create a sublattice mask (e.g., A/B sublattices in bipartite lattices).
        
        Parameters
        ----------
        ns : int
            Total number of sites
        sublattice_id : int
            Which sublattice (0, 1, ..., n_sublattices-1)
        n_sublattices : int
            Total number of sublattices (default: 2 for bipartite)
            
        Returns
        -------
        np.ndarray
            Array of site indices in the specified sublattice
        """
        all_sites = np.arange(ns, dtype=np.int64)
        return all_sites[all_sites % n_sublattices == sublattice_id]
    
    @staticmethod
    def kitaev_preskill(ns: int, center: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate regions A, B, C for Kitaev-Preskill topological entanglement entropy.
        
        The Kitaev-Preskill construction divides the system into three regions
        meeting at a point. The topological entanglement entropy is:
            gamma = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
        
        Parameters
        ----------
        ns : int
            Total number of sites (should be divisible by 3 for equal regions)
        center : int, optional
            Central site index (default: ns // 2)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with keys 'A', 'B', 'C', 'AB', 'BC', 'AC', 'ABC'
            containing site index arrays for each region
            
        Notes
        -----
        For 1D chains, regions are consecutive thirds of the chain.
        For 2D systems, you should define regions based on geometry.
        
        References
        ----------
        - Kitaev & Preskill, PRL 96, 110404 (2006)
        - Levin & Wen, PRL 96, 110405 (2006)
        """
        if center is None:
            center = ns // 2
            
        # Divide into 3 approximately equal regions
        size_per_region = ns // 3
        remainder       = ns % 3
        
        # Region sizes
        size_a  = size_per_region + (1 if remainder > 0 else 0)
        size_b  = size_per_region + (1 if remainder > 1 else 0)
        size_c  = ns - size_a - size_b
        
        # Region boundaries
        A       = np.arange(0, size_a, dtype=np.int64)
        B       = np.arange(size_a, size_a + size_b, dtype=np.int64)
        C       = np.arange(size_a + size_b, ns, dtype=np.int64)
        
        return {
            'A'     : A,
            'B'     : B,
            'C'     : C,
            'AB'    : np.concatenate([A, B]),
            'BC'    : np.concatenate([B, C]),
            'AC'    : np.concatenate([A, C]),
            'ABC'   : np.arange(ns, dtype=np.int64)
        }
    
    @staticmethod
    def levin_wen_disk(ns: int, n_annuli: int = 3) -> Dict[str, np.ndarray]:
        """
        Generate annular regions for Levin-Wen construction.
        
        For a disk geometry, creates concentric annuli to extract
        topological entanglement entropy with area law subtraction.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        n_annuli : int
            Number of concentric annuli (default: 3).
            
            What are annuli?
            - 1 annulus: inner region only
            - 2 annuli: inner + middle regions
            - 3 annuli: inner + middle + outer regions
            
            Therefore, the annuli represent nested regions of increasing size.
            
            Example:
            - n_annuli=1: 'inner' region
                - S_inner           = alpha * L_inner - gamma
            - n_annuli=2: 'inner', 'middle', 'inner_middle' regions
                - S_inner           = alpha * L_inner - gamma
                - S_middle          = alpha * L_middle - gamma
                - S_inner_middle    = alpha * (L_inner + L_middle) - gamma
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'inner', 'middle', 'outer', and combined regions
        """
        
        # Determine sizes
        size_per_annulus    = ns // n_annuli
        
        # Create regions
        regions             = {}
        names               = ['inner', 'middle', 'outer'][:n_annuli]
        
        start = 0
        for i, name in enumerate(names):
            if i == n_annuli - 1:
                end = ns
            else:
                end = start + size_per_annulus
            regions[name] = np.arange(start, end, dtype=np.int64)
            start = end
        
        # Combinations
        if n_annuli >= 2:
            regions['inner_middle'] = np.concatenate([regions['inner'], regions['middle']])
        if n_annuli >= 3:
            regions['middle_outer'] = np.concatenate([regions['middle'], regions['outer']])
            regions['inner_outer']  = np.concatenate([regions['inner'], regions['outer']])
        
        return regions
    
    @staticmethod
    def from_bitmask(mask_int: int, ns: int) -> np.ndarray:
        """
        Convert an integer bitmask to an array of site indices.
        
        Parameters
        ----------
        mask_int : int
            Integer whose bits indicate included sites
        ns : int
            Total number of sites
            
        Returns
        -------
        np.ndarray
            Array of site indices where bits are set
            
        Example
        -------
        >>> MaskGenerator.from_bitmask(0b1010, ns=4)
        array([1, 3])
        """
        indices = []
        for i in range(ns):
            if (mask_int >> i) & 1:
                indices.append(i)
        return np.array(indices, dtype=np.int64)
    
    @staticmethod
    def to_bitmask(indices: np.ndarray) -> int:
        """
        Convert an array of site indices to an integer bitmask.
        
        Parameters
        ----------
        indices : np.ndarray
            Array of site indices
            
        Returns
        -------
        int
            Integer bitmask with bits set at specified positions
            
        Example
        -------
        >>> MaskGenerator.to_bitmask(np.array([1, 3]))
        10  # = 0b1010
        """
        mask = 0
        for i in indices:
            mask |= (1 << int(i))
        return mask

###############################################################################

@dataclass
class BipartitionInfo:
    """Information about a bipartition of the system."""
    mask_a          : np.ndarray          # Indices in subsystem A
    mask_b          : np.ndarray          # Indices in subsystem B
    size_a          : int                 # Number of sites in A
    size_b          : int                 # Number of sites in B
    order           : tuple               # Reordering: (mask_a..., mask_b...)
    extractor_a     : Callable            # Function to extract A indices from state
    extractor_b     : Callable            # Function to extract B indices from state

###############################################################################
#! Entanglement Module
###############################################################################

class EntanglementModule:
    r"""
    Entanglement calculation module for Hamiltonians.
    
    Provides unified interface for calculating entanglement entropy using:
    - Single-particle correlation matrices (quadratic Hamiltonians, fast)
        - To be optimized when system sizes are large, we probably don't want to compute np.arange(ns) as it can be large!
    - Many-body reduced density matrices (any state, exact)
        - Features:
            - Wick's theorem verification
            - Topological entanglement entropy (Kitaev-Preskill, Levin-Wen)
            - Manual bipartition handling
            - Multipartite entropy calculations
            - Symmetry sector support
    - JAX backend for GPU acceleration
    - Batch calculations for multiple bipartitions
    
    Automatically handles arbitrary bipartitions including non-contiguous subsystems (those are more problematic but handled here).
    
    Examples
    --------
    1. Quadratic Hamiltonian (non-interacting):
        - Basic entropy calculation:
        
    >>> hamil       = QuadraticHamiltonian(ns=12, ...)
    >>> hamil.diagonalize()
    >>> ent         = hamil.entanglement
    >>> bipart      = ent.bipartition([0, 1, 2, 3])                             # subsystem A
    >>> orbitals    = [0, 1, 2, 3, 4, 5]                                        # occupied quasi-particle states 
    >>> S           = ent.entropy_correlation(bipart, orbitals)                 # entropy from correlation matrix
    
        - Access correlation matrices themselves:
        
    >>> C_full      = ent.correlation_matrix(orbitals)                          # (ns, ns)
    >>> C_A         = ent.correlation_matrix(orbitals, bipartition=bipart)      # (4, 4)

    - Batch calculations:
    
    >>> results     = ent.entropy_multipartition(
    ...             [[0,1], [0,1,2], [0,1,2,3]],
    ...             orbitals
    ...             )                                                           # computes entropies for 3 bipartitions
    >>> entropies   = results['entropies']                                      # array of 3 entropies for each bipartition
    >>> C_matrices  = results['correlation_matrices']                           # list of 3 matrices

    - JAX backend:
    
    >>> S_jax       = ent.entropy_correlation(bipart, orbitals, backend='jax')
    >>> results_jax = ent.entropy_multipartition(
    ...             [[0,1], [0,1,2]], orbitals, backend='jax'
    ...             )
    
    - Mutual information:
    
    >>> I_AB        = ent.mutual_information([0,1,2], [3,4,5], orbitals)        # I(A:B) = S_A + S_B - S_AB, A=[0,1,2], B=[3,4,5], occupied orbitals
    
    - Entropy scaling:
    
    >>> results     = ent.entropy_scan(orbitals, sizes=[1,2,3,4,5])             # entropies for subsystems of sizes 1 to 5, consqutive sites starting from 0
    
    2. Many-body Hamiltonian (interacting):
        - Manual bipartition entropy:
        
    >>> hamil       = ManyBodyHamiltonian(ns=8, ...)
    >>> hamil.diagonalize()
    >>> ent         = hamil.entanglement
    >>> bipart      = ent.bipartition([0, 1, 2, 3])                             # subsystem A
    >>> state       = hamil.eig_vec[:, 0]                                       # ground state wavefunction
    >>> S_manual    = ent.entropy_manybody(bipart, state)                       # entropy from reduced density matrix, it happens internally
    
    - Density matrix access:
    >>> rho_A       = ent.reduced_density_matrix(bipart, state)                 # reduced density matrix for subsystem A
    >>> rho_B       = ent.reduced_density_matrix(bipart, state, subsystem='B')  # for subsystem B
    
    """
    
    def __init__(self, operator):
        """
        Initialize entanglement module for a Hamiltonian.
        
        Parameters
        ----------
        operator : object
            The Operator object (quadratic or many-body)
        """
        self._operator              = operator
        self._cached_bipartitions   = {}
    
    # -----------------------------
    #! MASKING AND BIPARTITIONING
    # -----------------------------
    
    def bipartition(self, mask_a: Union[List[int], np.ndarray, int], *, cache: bool = True) -> BipartitionInfo:
        """
        Create bipartition information for subsystem A.
        
        Parameters
        ----------
        mask_a : array-like or int
            Indices of sites in subsystem A, or number of sites in A (takes first N sites).
        cache : bool
            Whether to cache the bipartition for reuse
            
        Returns
        -------
        BipartitionInfo
            Information about the bipartition
            
        Examples
        --------
        >>> # Contiguous partition
        >>> bipart = ent.bipartition(5)                 # First 5 sites
        >>> 
        >>> # Non-contiguous partition
        >>> bipart = ent.bipartition([0, 2, 4, 6, 8])   # Even sites
        """
        # Convert to array
        if isinstance(mask_a, int):
            mask_a  = np.arange(mask_a)
        else:
            mask_a  = np.asarray(mask_a, dtype=np.int64)
            
        # Check cache
        cache_key   = tuple(sorted(mask_a))
        if cache and cache_key in self._cached_bipartitions:
            return self._cached_bipartitions[cache_key]
        
        try:
            from ..common.binary import extract as Extractor
        except ImportError as e:
            raise ImportError("Extractor module not found") from e
        
        # Create complement
        ns          = self._operator.ns
        mask_a      = np.sort(mask_a)
        mask_b      = np.setdiff1d(np.arange(ns), mask_a)
        
        size_a      = len(mask_a)
        size_b      = len(mask_b)
        
        # Create ordering tuple
        order       = tuple(mask_a) + tuple(mask_b) # ordering of sites: A first, then B
        
        # Create extractors
        extractor_a = Extractor.make_extractor(mask_a, size=ns, backend='numba_vnb')
        extractor_b = Extractor.make_extractor(mask_b, size=ns, backend='numba_vnb')
        
        bipart = BipartitionInfo(
            mask_a          =   mask_a,
            mask_b          =   mask_b,
            size_a          =   size_a,
            size_b          =   size_b,
            # 
            order           =   order,
            # -----------------------------
            extractor_a     =   extractor_a,
            extractor_b     =   extractor_b
        )
        
        if cache:
            self._cached_bipartitions[cache_key] = bipart
            
        return bipart
    
    # -----------------------------
    #! SINGLE-PARTICLE CORRELATION MATRIX METHODS
    # -----------------------------
    
    def correlation_matrix(self,
                          occupied_orbitals     : Union[List[int], np.ndarray],
                          *,
                          bipartition           : Optional[BipartitionInfo]     = None,
                          subtract_identity     : bool                          = False,
                          raw                   : bool                          = True,
                          mode                  : Literal['slater', 'BdG']      = 'slater',
                          backend               : str                           = 'numpy',
                          **kwargs
                          ) -> np.ndarray:
        r"""
        Get single-particle correlation matrix C_ij = <c_i^\\dag c_j>.
        
        Computes the correlation matrix for a free fermion state defined by
        occupied orbitals. Uses spin-unpolarized convention (factor of 2).
        
        Parameters
        ----------
        occupied_orbitals : array-like
            Indices of occupied orbitals (in eigenstate basis).
            For ground state, use [0, 1, ..., N-1] for N particles.
        bipartition : BipartitionInfo, optional
            If provided, returns correlation matrix restricted to subsystem A.
            If None, returns full correlation matrix for all sites.
        subtract_identity : bool
            Whether to subtract identity from the correlation matrix.
        backend : str
            'numpy' or 'jax' for GPU acceleration.
            
        Returns
        -------
        np.ndarray or jax.numpy.ndarray
            Correlation matrix C_ij = <c_i^\\dag c_j>.
            Shape: (size_a, size_a) if bipartition given, else (ns, ns).
            Diagonal elements are site occupations (in [0,2] range with factor 2).
        
        Examples
        --------
        - Full correlation matrix for ground state:
        >>> hamil   = QuadraticHamiltonian(ns=8, dtype=np.complex128)
        >>> # ... add hopping terms ...
        >>> hamil.diagonalize()
        >>> ent     = hamil.entanglement                        # Get entanglement module
        >>> 
        >>> # Half-filling: occupy lowest 4 orbitals
        >>> orbitals    = [0, 1, 2, 3]
        >>> C_full      = ent.correlation_matrix(orbitals)
        >>> print(C_full.shape)                                 # (8, 8)
        >>> print(np.trace(C_full))                             # Should be 2*4 = 8 (spin-unpolarized)
    
        - Subsystem correlation matrix:
        >>> bipart      = ent.bipartition([0, 1, 2])            # First 3 sites
        >>> C_A = ent.correlation_matrix(orbitals, bipartition=bipart)
        >>> print(C_A.shape)                                    # (3, 3)
        >>> # Use for entropy: eigenvalues -> occupations -> entropy
        
        - JAX backend for GPU, same result as NumPy:
        >>> C_jax       = ent.correlation_matrix(orbitals, backend='jax')
        >>> # Same result as NumPy, but runs on GPU
        
        - Verify correlation matrix properties:
        >>> C           = ent.correlation_matrix(orbitals)
        >>> # Hermitian
        >>> assert np.allclose(C, C.conj().T)
        >>> # Occupations in [0, 2]
        >>> assert np.all(np.diag(C) >= 0) and np.all(np.diag(C) <= 2)
        """
        if not hasattr(self._operator, 'eig_vec') or self._operator.eig_vec is None:
            raise RuntimeError("The Operator doesn't seem to contain 'eig_vec'. Make sure the Operator is diagonalized before calling correlation_matrix()")
        
        W                   = self._operator.eig_vec            # W is (sites, orbitals), need to transpose
        orbitals            = np.asarray(occupied_orbitals, dtype=np.int64)
        occ_mask            = np.zeros(self._operator.ns, dtype=bool)
        occ_mask[orbitals]  = True
        
        # Corr.corr_full expects W with shape (orbitals, sites), so transpose
        C_full              = Corr.corr_full(W.T, occ_mask, subtract_identity=subtract_identity, raw=raw, mode=mode, **kwargs)
        
        # Extract subblock if needed
        if bipartition is not None:
            C_result        = C_full[np.ix_(bipartition.mask_a, bipartition.mask_a)]    # Extract C_A -> (size_a, ns)
        else:
            C_result        = C_full                                                    # Full matrix (ns, ns)
        
        # Convert to JAX if requested
        if backend == 'jax' and JAX_AVAILABLE:
            return jnp.array(C_result)
        elif backend == 'numpy':
            return C_result
        else:
            raise ValueError(f"Backend '{backend}' not available or not supported")
    
    def entropy_correlation(self,
                          bipartition           : BipartitionInfo,
                          occupied_orbitals     : Union[List[int], np.ndarray],
                          *,
                          q                     : float                                 = 1.0,
                          C_A                   : Optional[Union[np.ndarray, 'Array']]  = None,
                          subtract_identity     : bool                                  = False,
                          backend               : str                                   = 'numpy',
                          **kwargs
                          ) -> float:
        """
        Calculate entanglement entropy from single-particle correlation matrix.
        
        **SINGLE-PARTICLE METHOD** - Fast O(L_A³) method for non-interacting (quadratic)
        Hamiltonians. Computes entropy from correlation matrix eigenvalues.
        
        Works for ANY bipartition (contiguous or non-contiguous) of free fermion states.
        
        Parameters
        ----------
        bipartition : BipartitionInfo
            Bipartition of the system (use ent.bipartition() to create).
            Works for both contiguous and non-contiguous subsystems.
        occupied_orbitals : array-like
            Indices of occupied orbitals (in eigenstate basis).
            For ground state with N particles, use [0, 1, ..., N-1].
        C_A : np.ndarray or jax.numpy.ndarray, optional
            Precomputed correlation matrix for subsystem A.
            If provided, uses this instead of computing from occupied_orbitals.
        subtract_identity : bool
            Whether to subtract identity from correlation matrix (advanced)
        backend : str
            'numpy' or 'jax' for computation backend
            
        Returns
        -------
        float
            Entanglement entropy (in natural log units, always positive)
            
        Notes
        -----
        Algorithm:
        1. Compute full correlation matrix C_ij = <c_i^\dag c_j> from occupied orbitals. For BdG, use all <c_i c_j> etc.
        2. Extract subblock C_A for sites in subsystem A (handles non-contiguous)
        3. Diagonalize C_A to get eigenvalues (occupations in [0,1])
        4. Apply single-particle entropy formula:
           S = - sum_k [ n_k log(n_k) + (1-n_k) log(1-n_k) ]
        
        This gives the EXACT entanglement entropy for ANY bipartition of
        non-interacting (quadratic) Hamiltonians and matches entropy_many_body().
        
        Limitations:
        - Requires diagonalized Hamiltonian
        - Only works for quadratic (non-interacting) Hamiltonians
        - For interacting systems, use entropy_many_body()
        """
        
        # Get correlation matrix for subsystem A using Corr methods
        C_A                 = self.correlation_matrix(occupied_orbitals, bipartition=bipartition, subtract_identity=False, backend=backend, **kwargs) if C_A is None else C_A
        
        # Diagonalize to get eigenvalues (in [0,2] range for occupation with spin-unpolarized convention)
        if backend == 'numpy':
            corr_eigs, _    = np.linalg.eigh(C_A)
        elif backend == 'jax' and JAX_AVAILABLE:
            corr_eigs       = jnp.linalg.eigh(C_A)[0]
        else:
            raise ValueError(f"Backend '{backend}' not available")
        
        # Divide by 2 to convert from spin-unpolarized to spin-polarized (occupation in [0,1])
        # corr_eigs_polarized     = corr_eigs / 2.0
        
        # Transform to [-1,1] range for SINGLE entropy formula
        corr_eigs_transformed   = 2.0 * corr_eigs - 1.0
        
        try:
            from .entropy import entropy, Entanglement
        except ImportError as e:
            raise ImportError("Entropy module not found") from e
        
        # Use unified entropy function which handles both numpy and jax
        return entropy(corr_eigs_transformed, q=q, typek=Entanglement.SINGLE, backend=backend)
    
    def entropy_many_body(self,
                        bipartition         : BipartitionInfo,
                        *,
                        # a) Precomputed reduced density matrix
                        rho_a               : Optional[np.ndarray]  = None,
                        # b) Many-body state vector
                        state               : Optional[np.ndarray]  = None,
                        q                   : float                 = 1.0,
                        method              : str                   = 'auto',
                        use_eig             : bool                  = False,
                        hilbert             = None,
                        occupied_orbitals   : Optional[Union[List[int], np.ndarray]] = None) -> float:
        r"""
        Calculate entanglement entropy from many-body state.
        
        **MANY-BODY METHOD** - Exact method that works for ANY quantum state,
        including interacting systems. Performs Schmidt decomposition of the
        many-body wavefunction.
        
        Parameters
        ----------
        bipartition : BipartitionInfo
            Bipartition of the system (use ent.bipartition() to create)
        q : float
            Renyi index (default: 1.0 for von Neumann entropy)
        rho_a : np.ndarray, optional
            Precomputed reduced density matrix for subsystem A.
            If provided, uses this instead of computing from state.
        state : np.ndarray, optional
            Many-body state vector (length 2^ns).
            If None, `occupied_orbitals` must be provided to construct the state (for free fermions).
        method : str
            'auto'    : Choose best method based on bipartition geometry
            'schmidt' : Use Schmidt decomposition with mask (for non-contiguous)
            'numpy'   : Use direct numpy Schmidt (for contiguous, faster)
        use_eig : bool
            Whether to use eigenvalue decomposition (True) or SVD (False)
        hilbert : HilbertSpace, optional
            Hilbert space with symmetries. If provided and has symmetries,
            symmetry-based reduced density matrix computation is used.
            This is only available when QES is installed correctly, as it is not
            a part of General Python.
        occupied_orbitals : array-like, optional
            Indices of occupied orbitals. Required only if `state` is None.
            
        Returns
        -------
        float
            Von Neumann entanglement entropy (always positive)
        """
        
        try:
            from .entropy import entropy, Entanglement
        except ImportError as e:
            raise ImportError("Entropy module not found") from e
        
        if rho_a is not None:
            # Directly use provided reduced density matrix
            schmidt_vals = np.linalg.eigvalsh(rho_a)
            return entropy(schmidt_vals, q=q, typek=Entanglement.VN)
        
        if state is None and rho_a is None:
            if occupied_orbitals is None:
                raise ValueError("Either `state` or `occupied_orbitals` must be provided.")
            state = self._operator.many_body_state(occupied_orbitals)
            
        hilbert = hilbert or getattr(self._operator, 'hilbert', None)
        
        # Symmetry-based path
        if hilbert is not None and hasattr(hilbert, 'has_sym') and hilbert.has_sym:
            
            # This path requires QES installation. Check for imports.
            try:
                from QES.Algebra.Symmetries.jit.density_jit     import rho_symmetries
            except ImportError:
                raise ImportError("QES.Algebra.Symmetries.jit.density_jit not found. Check installation.")
            
            try:
                from general_python.physics.density_matrix  import rho_spectrum
            except ImportError:
                raise ImportError("general_python.physics.density_matrix not found. Check installation.")
            
            # Use symmetry-aware reduced density matrix (supports mask)
            rho             = rho_symmetries(state, va=bipartition.mask_a, hilbert=hilbert)
            schmidt_vals    = rho_spectrum(rho)
            return entropy(schmidt_vals, typek=Entanglement.VN)

        try:
            from general_python.physics.density_matrix      import schmidt_numpy, schmidt_numba_mask, rho_numba_mask
        except ImportError:
            raise ImportError("general_python.physics.density_matrix not found. Check installation.")
        
        # Standard state-vector path
        dimA            = 1 << bipartition.size_a
        dimB            = 1 << bipartition.size_b
        
        # Check if bipartition is contiguous
        is_contiguous   = np.all(np.diff(bipartition.mask_a) == 1) and bipartition.mask_a[0] == 0
        
        if method == 'auto':
            method = 'numpy' if is_contiguous else 'schmidt'
        
        if method == 'numpy' and is_contiguous:
            # Fast path for contiguous bipartitions
            schmidt_vals, _, _  = schmidt_numpy(state, dimA, dimB, eig=use_eig)
            return entropy(schmidt_vals, q=q, typek=Entanglement.VN)
            
        elif method == 'schmidt':
            # Reshape state according to bipartition order
            psi_reshaped        = rho_numba_mask(state, bipartition.order, bipartition.size_a)
            schmidt_vals, _     = schmidt_numba_mask(
                                    psi     =   psi_reshaped,
                                    order   =   bipartition.order,
                                    size_a  =   bipartition.size_a,
                                    eig     =   use_eig
                                )
            
            return entropy(schmidt_vals, q=q, typek=Entanglement.VN)
        else:
            raise ValueError(f"Unknown method: {method}")

    
    # -----------------------------
    #! HIGH-LEVEL ENTROPY CALCULATIONS
    # -----------------------------
    
    def entropy_scan(self,
                    *,
                    state               : Optional[np.ndarray]                      = None,
                    occupied_orbitals   : Optional[Union[List[int], np.ndarray]]    = None,
                    subsystem_sizes     : Optional[List[int]]                       = None,
                    q                   : float                                     = 1.0,
                    method              : str                                       = 'auto',
                    contiguous          : bool                                      = True,
                    ) -> dict:
        r"""
        Calculate entanglement entropy for multiple subsystem sizes.
        
        Parameters
        ----------
        occupied_orbitals : array-like, optional
            Occupied orbitals for the state (for free fermions).
        subsystem_sizes : list of int, optional
            Sizes of subsystem A to scan. If None, scans all sizes from 1 to ns-1
        q : float
            Renyi index (default: 1.0 for von Neumann entropy)
        method : str
            'auto'        : Use correlation matrix for quadratic, many-body otherwise
            'correlation' : Force correlation matrix method
            'many_body'   : Force many-body method
        contiguous : bool
            If True, use contiguous partitions [0:size_a]
            If False, use random partitions
        state : np.ndarray, optional
            Many-body state vector. Required if occupied_orbitals is None and method is 'many_body' or 'auto' (for interacting systems).
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'sizes'     : Subsystem sizes
            - 'entropies' : Entanglement entropies
            - 'method'    : Method used
            
        Examples
        --------
        >>> results = ent.entropy_scan(orbitals=[0,1,2,3,4])
        >>> plt.plot(results['sizes'], results['entropies'])
        
        - Use many-body state for interacting system:
        >>> results = ent.entropy_scan(state=ground_state_vector)
        >>> plt.plot(results['sizes'], results['entropies'])
        """
        
        if subsystem_sizes is None:
            subsystem_sizes = list(range(1, self._operator.ns))
            
        # Determine method
        is_quadratic = hasattr(self._operator, '_particle_conserving')
        if method == 'auto':
            if state is not None:
                method = 'many_body'
            else:
                method = 'correlation' if is_quadratic else 'many_body'
            
        entropies = []
        
        if method == 'correlation':
            if occupied_orbitals is None:
                 raise ValueError("occupied_orbitals required for correlation method")
            
            # Fast correlation matrix method
            for size_a in subsystem_sizes:
                if contiguous:
                    bipart = self.bipartition(size_a)
                else:
                    mask_a = np.sort(np.random.choice(self._operator.ns, size_a, replace=False))
                    bipart = self.bipartition(mask_a)
                    
                S = self.entropy_correlation(bipart, occupied_orbitals)
                entropies.append(S)
                
        else:
            # Many-body method (requires state construction)
            if state is None:
                if occupied_orbitals is None:
                    raise ValueError("Either state or occupied_orbitals required for many_body method")
                state = self._operator.many_body_state(occupied_orbitals)
            
            for size_a in subsystem_sizes:
                if contiguous:
                    bipart = self.bipartition(size_a)
                else:
                    mask_a = np.sort(np.random.choice(self._operator.ns, size_a, replace=False))
                    bipart = self.bipartition(mask_a)
                    
                S = self.entropy_many_body(bipart, state=state, q=q)
                entropies.append(S)
                
        return {
            'sizes'     : subsystem_sizes,
            'entropies' : np.array(entropies),
            'method'    : method
        }
    
    def mutual_information(self,
                        mask_a                      : Union[List[int], np.ndarray],
                        mask_b                      : Union[List[int], np.ndarray],
                        *,
                        q                           : float                                     = 1.0,
                        occupied_orbitals           : Optional[Union[List[int], np.ndarray]]    = None,
                        method                      : str                                       = 'auto',
                        state                       : Optional[np.ndarray]                      = None,
                        **kwargs
                        ) -> float:
        """
        Calculate mutual information I(A:B) = S(A) + S(B) - S(AB). Importantly, regions A and B can be
        smaller than total system size, for instance, a common use case is to compute mutual information
        for single site subsystems, for instance I(i:j) between sites i and j.
        In such case A = {i}, B = {j}, AB = {i,j}.
        
        Parameters
        ----------
        mask_a, mask_b : array-like
            Indices for subsystems A and B
        occupied_orbitals : array-like, optional
            Occupied orbitals (for free fermions)
        method : str
            'auto', 'correlation', or 'many_body'
        state : np.ndarray, optional
            Many-body state vector.
    
        Examples
        --------
        >>> I_AB = ent.mutual_information(
        ...             mask_a              = [0,1,2],
        ...             mask_b              = [3,4,5],
        ...             occupied_orbitals   = [0,1,2,3,4,5]
        ...         )
        
        - Using many-body state:
        >>> I_AB = ent.mutual_information(
        ...             mask_a  = [0,1,2],
        ...             mask_b  = [3,4,5],
        ...             state   = ground_state_vector,
        ...             method  = 'many_body'
        ...         )
    
        Returns
        -------
        float
            Mutual information
        """
        mask_a          = np.asarray(mask_a)
        mask_b          = np.asarray(mask_b)
        mask_ab         = np.union1d(mask_a, mask_b)
        
        # Create bipartitions
        bipart_a        = self.bipartition(mask_a)
        bipart_b        = self.bipartition(mask_b)
        bipart_ab       = self.bipartition(mask_ab)
        
        # Determine method
        is_quadratic    = hasattr(self._operator, '_particle_conserving')
        if method == 'auto':
            if state is not None:
                method = 'many_body'
            else:
                method = 'correlation' if is_quadratic else 'many_body'
        
        if method == 'correlation':
            if occupied_orbitals is None:
                raise ValueError("occupied_orbitals required for correlation method")
            S_a         = self.entropy_correlation(bipart_a, occupied_orbitals, q=q)
            S_b         = self.entropy_correlation(bipart_b, occupied_orbitals, q=q)
            S_ab        = self.entropy_correlation(bipart_ab, occupied_orbitals, q=q)
        else:
            if state is None:
                if occupied_orbitals is None:
                    raise ValueError("Either state or occupied_orbitals required for many_body method")
                state = self._operator.many_body_state(occupied_orbitals)
                
            S_a         = self.entropy_many_body(bipart_a,  state=state, q=q)
            S_b         = self.entropy_many_body(bipart_b,  state=state, q=q)
            S_ab        = self.entropy_many_body(bipart_ab, state=state, q=q)
            
        return S_a + S_b - S_ab
    
    # -----------------------------
    #! MULTIPARTITION ENTROPY CALCULATIONS
    # -----------------------------
    
    def entropy_multipartition(self,
                               bipartitions: List[Union[BipartitionInfo, List[int], np.ndarray]],
                               occupied_orbitals: Optional[Union[List[int], np.ndarray]] = None,
                               *,
                               method: str = 'auto',
                               backend: str = 'numpy',
                               state: Optional[np.ndarray] = None) -> dict:
        """
        Calculate entanglement entropy for multiple bipartitions simultaneously.
        
        Efficient batch calculation that computes correlation matrix once and
        reuses it for all bipartitions, or computes many-body state once for
        all Schmidt decompositions.
        
        Parameters
        ----------
        bipartitions : list
            List of BipartitionInfo objects or site masks (will create BipartitionInfo).
        occupied_orbitals : array-like, optional
            Occupied orbitals for correlation method.
        method : str
            'auto', 'correlation', or 'many_body'.
        backend : str
            'numpy' or 'jax' (for correlation method).
        state : np.ndarray, optional
            Pre-computed many-body state (for many_body method).
            If None and method is many_body, will be computed from occupied_orbitals.
            
        Returns
        -------
        dict
            Results dictionary containing:
            - 'entropies': array of entropies for each bipartition
            - 'bipartitions': list of BipartitionInfo objects
            - 'method': method used ('correlation' or 'many_body')
            - 'correlation_matrices': list of C_A matrices (if method='correlation')
            
        Examples
        --------
        Basic usage:
            >>> masks = [[0,1], [0,1,2], [0,1,2,3]]
            >>> results = ent.entropy_multipartition(masks, orbitals=[0,1,2,3,4])
            >>> print(results['entropies'])  # array([S_1, S_2, S_3])
        
        Access correlation matrices:
            >>> C_matrices = results['correlation_matrices']
            >>> for i, C_A in enumerate(C_matrices):
            ...     print(f"Bipartition {i}: C_A shape = {C_A.shape}")
        
        JAX backend:
            >>> results_jax = ent.entropy_multipartition(
            ...     masks, orbitals, backend='jax'
            ... )
        
        Many-body method:
            >>> state = hamil.many_body_state(orbitals)
            >>> results_mb = ent.entropy_multipartition(
            ...     masks, orbitals, method='many_body', state=state
            ... )
        """
        # Prepare bipartitions
        bipart_list = []
        for bp in bipartitions:
            if isinstance(bp, BipartitionInfo):
                bipart_list.append(bp)
            else:
                bipart_list.append(self.bipartition(bp))
        
        # Determine method
        is_quadratic = hasattr(self._operator, '_particle_conserving')
        if method == 'auto':
            if state is not None:
                method = 'many_body'
            else:
                method = 'correlation' if is_quadratic else 'many_body'
        
        entropies = []
        corr_matrices = [] if method == 'correlation' else None
        
        if method == 'correlation':
            if occupied_orbitals is None:
                raise ValueError("occupied_orbitals required for correlation method")
                
            # Compute full correlation matrix once
            C_full = self.correlation_matrix(occupied_orbitals, bipartition=None,
                                            subtract_identity=False, backend=backend)
            
            # Extract subblocks and compute entropies for each bipartition
            for bipart in bipart_list:
                if backend == 'numpy':
                    C_A = C_full[np.ix_(bipart.mask_a, bipart.mask_a)]
                    corr_matrices.append(C_A.copy())
                    
                    # Diagonalize and convert from spin-unpolarized to spin-polarized
                    corr_eigs, _ = np.linalg.eigh(C_A)
                    corr_eigs_polarized = corr_eigs / 2.0
                    corr_eigs_transformed = 2.0 * corr_eigs_polarized - 1.0
                    S = entropy(corr_eigs_transformed, q=1.0, typek=Entanglement.SINGLE, backend='numpy')
                    
                elif backend == 'jax' and JAX_AVAILABLE:
                    mask_a_jax = jnp.array(bipart.mask_a)
                    C_A = C_full[jnp.ix_(mask_a_jax, mask_a_jax)]
                    corr_matrices.append(np.array(C_A))
                    
                    # Diagonalize and convert from spin-unpolarized to spin-polarized
                    corr_eigs = jnp.linalg.eigh(C_A)[0]
                    corr_eigs_polarized = corr_eigs / 2.0
                    corr_eigs_transformed = 2.0 * corr_eigs_polarized - 1.0
                    S = float(entropy(corr_eigs_transformed, q=1.0, typek=Entanglement.SINGLE, backend='jax'))
                else:
                    raise ValueError(f"Backend '{backend}' not available")
                    
                entropies.append(S)
                
        else:  # many_body method
            # Compute state once if not provided
            if state is None:
                if occupied_orbitals is None:
                    raise ValueError("Either state or occupied_orbitals required for many_body method")
                state = self._operator.many_body_state(occupied_orbitals)
            
            # Compute entropy for each bipartition
            for bipart in bipart_list:
                S = self.entropy_many_body(bipart, state=state, method='auto')
                entropies.append(S)
        
        result = {
            'entropies': np.array(entropies),
            'bipartitions': bipart_list,
            'method': method
        }
        
        if corr_matrices is not None:
            result['correlation_matrices'] = corr_matrices
            
        return result
    
    # =========================================================================
    #! Topological Entanglement Entropy
    # =========================================================================
    
    def topological_entropy(self,
                           *,
                           q                        : float                                     = 1.0,
                           state                    : Optional[np.ndarray]                      = None,
                           occupied_orbitals        : Union[List[int], np.ndarray]              = None,
                           construction             : Literal['kitaev_preskill', 'levin_wen']   =  'kitaev_preskill',
                           method                   : str                                       = 'auto',
                           regions                  : Optional[Dict[str, np.ndarray]]           = None,
                           hilbert                  = None) -> Dict[str, float]:
        r"""
        Calculate topological entanglement entropy (TEE).
        
        The topological entanglement entropy γ characterizes topological order.
        For topologically ordered states, S(A) = αL - γ + O(1/L), where γ > 0.
        
        Parameters
        ----------
        occupied_orbitals : array-like
            Occupied orbitals defining the state
        construction : str
            'kitaev_preskill' : γ = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
            'levin_wen' : Alternative construction with disk geometry
        method : str
            Entropy calculation method ('auto', 'correlation', 'many_body')
        regions : dict, optional
            Custom region definitions. If None, uses MaskGenerator.
        hilbert : HilbertSpace, optional
            Hilbert space object for symmetry-aware calculations.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'gamma' : Topological entanglement entropy
            - 'entropies' : Individual region entropies
            - 'regions' : Region masks used
            
        Notes
        -----
        For the Kitaev-Preskill construction:
            γ = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
            
        This combination cancels the area law contribution and extracts
        the universal topological term. For topological phases like the
        toric code, γ = log(D) where D is the total quantum dimension.
        
        References
        ----------
        - Kitaev & Preskill, PRL 96, 110404 (2006)
        - Levin & Wen, PRL 96, 110405 (2006)
        
        Examples
        --------
        >>> result = ent.topological_entropy(orbitals, construction='kitaev_preskill')
        >>> print(f"Topological entropy: γ = {result['gamma']:.4f}")
        """
        ns = self._operator.ns
        
        # Generate regions if not provided
        if regions is None:
            if construction == 'kitaev_preskill':
                regions = MaskGenerator.kitaev_preskill(ns)
            elif construction == 'levin_wen':
                regions = MaskGenerator.levin_wen_disk(ns, n_annuli=3)
            else:
                raise ValueError(f"Unknown construction: {construction}")
        
        # Calculate entropies for each region
        entropies = {}
        
        # Determine method
        is_quadratic    = hasattr(self._operator, '_particle_conserving')
        if method == 'auto':
            method      = 'correlation' if is_quadratic else 'many_body'
        
        # Compute state if needed for many-body method
        if method == 'many_body' and state is None:
            state       = self._operator.many_body_state(occupied_orbitals)

        # Try to get hilbert from operator if not provided
        if hilbert is None:
            hilbert     = getattr(self._operator, 'hilbert', None)
        
        for region_name, mask in regions.items():
            if len(mask) == 0 or len(mask) == ns:
                entropies[region_name] = 0.0
                continue
                
            bipart = self.bipartition(mask)
            
            if method == 'correlation':
                S = self.entropy_correlation(bipart, occupied_orbitals, check_contiguous=False, backend='numpy', q=q)
            else:
                S = self.entropy_many_body(bipart, state=state, hilbert=hilbert, q=q)
            
            # Store entropy
            entropies[region_name] = S
        
        # Calculate topological entropy
        if construction == 'kitaev_preskill':
            # Kitaev-Preskill construction
            # γ = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC -> get rid of area law terms
            gamma = (entropies.get('A', 0) + entropies.get('B', 0) + entropies.get('C', 0) - entropies.get('AB', 0) - entropies.get('BC', 0) - entropies.get('AC', 0) + entropies.get('ABC', 0))
        elif construction == 'levin_wen':
            # Levin-Wen uses annular geometry with inner and outer regions
            # This corresponds to: γ = S_inner + S_outer - S_inner_outer
            gamma = (entropies.get('inner', 0) + entropies.get('outer', 0)  - entropies.get('inner_outer', 0))
        else:
            gamma = 0.0
        
        return {
            'gamma'         : gamma,
            'entropies'     : entropies,
            'regions'       : regions,
            'construction'  : construction
        }
    
    # =========================================================================
    #! Wick's Theorem Verification
    # =========================================================================

    def _compute_local_expectation(self, rho: np.ndarray, sites_subset: np.ndarray, ops: List[Tuple[int, int]]) -> complex:
        """
        Compute <Op1 Op2 ...> from RDM.
        
        Parameters
        ----------
        rho : np.ndarray
            Reduced density matrix for the subsystem defined by sites_subset.
        sites_subset : np.ndarray
            Array of site indices corresponding to rho.
        ops : list of (site_idx, op_type)
            List of operators to apply. site_idx is global index.
            op_type: 1 for creation (c†), 0 for annihilation (c).
            Operators are ordered as they appear in the product Op1 Op2 ...
            
        Returns
        -------
        complex
            Expectation value Tr(rho * Op1 * Op2 ...)
        """
        dim = rho.shape[0]
        # Map global site to local index
        # sites_subset is assumed sorted or at least consistent with rho's basis
        global_to_local = {site: i for i, site in enumerate(sites_subset)}
        
        value = 0.0j
        
        # Iterate over basis states |m> of the density matrix
        for m in range(dim):
            if abs(rho[m, m]) < 1e-15 and np.all(np.abs(rho[m, :]) < 1e-15):
                continue
                
            # Apply operators to ket |m>: Op1 Op2 ... |m>
            # This produces a superposition: sum_k coeff_k |k>
            
            # current_state represents Op |n> (where n is loop variable 'm' here for convenience)
            current_state = {m: 1.0 + 0.0j} 
            
            # Apply ops in reverse order (Op1 Op2... acting on ket means Op_last ... Op_1 |ket>)
            for site_global, op_type in reversed(ops):
                if site_global not in global_to_local:
                     raise ValueError(f"Site {site_global} not in subsystem")
                local_site = global_to_local[site_global]
                
                next_state = {}
                for basis, amp in current_state.items():
                    # Check occupation
                    occ = (basis >> local_site) & 1
                    
                    if op_type == 0: # Annihilation c
                        if occ == 0:
                            continue # kills it
                        else:
                            # phase: count bits < local_site
                            parity = bin(basis & ((1 << local_site) - 1)).count('1')
                            phase = 1.0 if parity % 2 == 0 else -1.0
                            new_basis = basis ^ (1 << local_site)
                            next_state[new_basis] = next_state.get(new_basis, 0.0) + amp * phase
                            
                    else: # Creation c†
                        if occ == 1:
                            continue # kills it
                        else:
                            parity = bin(basis & ((1 << local_site) - 1)).count('1')
                            phase = 1.0 if parity % 2 == 0 else -1.0
                            new_basis = basis | (1 << local_site)
                            next_state[new_basis] = next_state.get(new_basis, 0.0) + amp * phase
                
                current_state = next_state
                if not current_state:
                    break
            
            # Tr(rho Op) = sum_{n, k} rho_{nk} Op_{kn} = sum_{n, k} rho_{nk} <k|Op|n>
            # Here 'm' is 'n' (input basis state).
            # current_state gives Op|m> = sum_k coeff_k |k>.
            # So <k|Op|m> = coeff_k.
            # We sum over k: rho[m, k] * coeff_k
            
            for k, coeff in current_state.items():
                value += rho[m, k] * coeff
                
        return value
    
    def verify_wicks_theorem(self,
                            occupied_orbitals: Union[List[int], np.ndarray],
                            state: Optional[np.ndarray] = None,
                            *,
                            test_sites: Optional[Tuple[int, int, int, int]] = None,
                            tolerance: float = 1e-10,
                            hilbert = None) -> Dict[str, Union[bool, float, np.ndarray]]:
        """
        Verify Wick's theorem for a state: check if it's a valid Slater determinant.
        
        For free fermion (quadratic) Hamiltonians, all correlation functions
        factorize according to Wick's theorem. This method verifies this property.
        
        Wick's theorem states that for a Slater determinant:
            <c_i† c_j† c_l c_k> = <c_i† c_k><c_j† c_l> - <c_i† c_l><c_j† c_k>
        
        Parameters
        ----------
        occupied_orbitals : array-like
            Occupied orbitals defining the expected Slater determinant
        state : np.ndarray, optional
            Many-body state to verify. If None, constructs from occupied_orbitals.
        test_sites : tuple, optional
            Specific (i, j, k, l) sites to test. If None, tests random sites.
        tolerance : float
            Tolerance for numerical comparison
        hilbert : HilbertSpace, optional
             Hilbert space object for symmetry-aware calculations.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'is_valid' : bool - True if Wick's theorem is satisfied
            - 'max_error' : float - Maximum deviation from Wick's theorem
            - 'errors' : np.ndarray - Error matrix for all tested site combinations
            - 'correlation_matrix' : np.ndarray - Single-particle correlation matrix
            
        Examples
        --------
        >>> result = ent.verify_wicks_theorem(orbitals)
        >>> if result['is_valid']:
        ...     print("State satisfies Wick's theorem (is a Slater determinant)")
        >>> else:
        ...     print(f"Max error: {result['max_error']:.2e}")
        
        Notes
        -----
        A state satisfies Wick's theorem if and only if it is a Slater determinant
        (or a mixture thereof for finite temperature). This is equivalent to
        the state being a Gaussian state for fermions.
        """
        ns = self._operator.ns
        
        # Construct state if not provided
        if state is None:
            state = self._operator.many_body_state(occupied_orbitals)
        
        # Try to get hilbert
        if hilbert is None:
            hilbert = getattr(self._operator, 'hilbert', None)
        
        # Get single-particle correlation matrix from orbitals
        C_sp = self.correlation_matrix(occupied_orbitals, subtract_identity=False)
        # Convert to [0,1] occupations (divide by 2 for spin-unpolarized convention)
        C_sp = C_sp / 2.0
        
        # Get exact 2-particle correlation from many-body state
        # <c_i† c_j† c_l c_k> for all i,j,k,l
        
        errors = np.zeros((ns, ns), dtype=np.float64)
        max_error = 0.0
        
        # Test specific sites or scan through pairs
        if test_sites is not None:
            i, j, k, l = test_sites
            test_pairs = [(i, j, k, l)]
        else:
            # Sample a few representative pairs
            test_pairs = []
            for j in range(min(ns, 4)):
                for l in range(min(ns, 4)):
                    if j != l:
                        test_pairs.append((0, j, l, 0))
        
        has_sym = hilbert is not None and hasattr(hilbert, 'has_sym') and hilbert.has_sym
        
        # Pre-import if needed
        rho_symmetries = None
        if has_sym:
            try:
                 from QES.Algebra.Symmetries.jit.density_jit import rho_symmetries
            except ImportError:
                 has_sym = False
        
        if not has_sym:
             from .sp.correlation_matrix import corr_from_statevector
        
        for (i, j, k, l) in test_pairs:
            if i == j or k == l:
                continue
            
            # Wick's theorem prediction
            wick_pred = C_sp[i, k] * C_sp[j, l] - C_sp[i, l] * C_sp[j, k]
            
            # Get exact value from many-body state
            try:
                if has_sym:
                    # Use RDM for subsystem {i, j, k, l}
                    sites_subset = np.unique(np.array([i, j, k, l], dtype=np.int64))
                    sites_subset.sort()
                    
                    rho_sub = rho_symmetries(state, va=sites_subset, hilbert=hilbert)
                    
                    # Compute <c_i† c_j† c_l c_k>
                    # Ops list: [(i, 1), (j, 1), (l, 0), (k, 0)]
                    ops = [(i, 1), (j, 1), (l, 0), (k, 0)]
                    exact_val = self._compute_local_expectation(rho_sub, sites_subset, ops)
                    exact_val = exact_val.real
                else:
                    N_exact = corr_from_statevector(state, ns, order=4, j=j, l=l)
                    exact_val = N_exact[i, k]
            except Exception:
                exact_val = wick_pred  # Fallback if not implemented
            
            error = np.abs(exact_val - wick_pred)
            errors[i, k] = max(errors[i, k], error)
            max_error = max(max_error, error)
        
        is_valid = max_error < tolerance
        
        return {
            'is_valid': is_valid,
            'max_error': max_error,
            'errors': errors,
            'correlation_matrix': C_sp,
            'tolerance': tolerance
        }
    
    def help(self):
        """Print usage help for the entanglement module."""
        help_text = """
        EntanglementModule - Unified entanglement calculations
        ======================================================
        
        Quick Start:
        -----------
        >>> hamil = FreeFermions(ns=12, t=1.0)
        >>> ent = hamil.entanglement  # Access entanglement module
        >>> 
        >>> # Define bipartition
        >>> bipart = ent.bipartition([0, 1, 2, 3, 4])  # First 5 sites
        >>> 
        >>> # Calculate entropy (correlation matrix method - fast)
        >>> S = ent.entropy_correlation(bipart, orbitals=[0,1,2,3,4])
        >>> 
        >>> # Calculate entropy (many-body method - exact)
        >>> state = hamil.many_body_state([0,1,2,3,4])
        >>> S = ent.entropy_many_body(bipart, state)
        
        Main Methods:
        ------------
        bipartition(mask_a)
            Create bipartition info for subsystem A
            
        correlation_matrix(orbitals, bipartition=None, backend='numpy')
            Get single-particle correlation matrix C_ij = <c_i^\\dag c_j>
            Returns full matrix if bipartition=None, or C_A if bipartition given
            Supports 'numpy' and 'jax' backends
            
        entropy_correlation(bipart, orbitals, backend='numpy')
            Fast method using correlation matrix (quadratic Hamiltonians)
            Supports 'numpy' and 'jax' backends
            WARNING: Only exact for contiguous bipartitions!
            
        entropy_many_body(bipart, state)
            Exact method using reduced density matrix (any state)
            Works for ANY bipartition (contiguous or non-contiguous)
            
        entropy_multipartition(bipartitions, orbitals, method='auto', backend='numpy')
            Calculate entropy for multiple bipartitions efficiently
            Returns dict with 'entropies', 'bipartitions', 'correlation_matrices'
            
        entropy_scan(orbitals, sizes=[...])
            Calculate entropy for multiple subsystem sizes
            
        mutual_information(mask_a, mask_b, orbitals)
            Calculate I(A:B) = S(A) + S(B) - S(AB)
        
        Topological Entropy (NEW):
        -------------------------
        topological_entropy(orbitals, construction='kitaev_preskill')
            Calculate topological entanglement entropy γ
            Uses Kitaev-Preskill or Levin-Wen constructions
            Returns dict with 'gamma', 'entropies', 'regions'
        
        Wick's Theorem (NEW):
        --------------------
        verify_wicks_theorem(orbitals, state=None)
            Verify if a state satisfies Wick's theorem (is a Slater determinant)
            Returns dict with 'is_valid', 'max_error', 'correlation_matrix'
        
        compare_methods(bipart, orbitals)
            Compare correlation vs many-body methods
            Useful for checking when fast method is accurate
        
        Mask Generation (use MaskGenerator class):
        -----------------------------------------
        >>> from general_python.physics.entanglement_module import MaskGenerator
        >>> 
        >>> # Contiguous mask
        >>> mask = MaskGenerator.contiguous(ns=12, size_a=4)  # [0,1,2,3]
        >>> 
        >>> # Alternating (even/odd) sites
        >>> even, odd = MaskGenerator.alternating(ns=12)
        >>> 
        >>> # Random subsystem
        >>> mask = MaskGenerator.random(ns=12, size_a=6, seed=42)
        >>> 
        >>> # Sublattice (bipartite lattice A/B)
        >>> mask_A = MaskGenerator.sublattice(ns=12, sublattice_id=0)
        >>> 
        >>> # Kitaev-Preskill regions for topological entropy
        >>> regions = MaskGenerator.kitaev_preskill(ns=12)
        >>> A, B, C = regions['A'], regions['B'], regions['C']
        >>> 
        >>> # Convert to/from bitmasks
        >>> bitmask = MaskGenerator.to_bitmask(np.array([0, 2, 4]))  # -> 0b10101
        >>> indices = MaskGenerator.from_bitmask(0b10101, ns=6)      # -> [0, 2, 4]
        
        Examples:
        --------
        # Access correlation matrix
        >>> C_full = ent.correlation_matrix([0,1,2,3,4])  # Full (ns x ns)
        >>> C_A = ent.correlation_matrix([0,1,2,3,4], bipartition=bipart)  # Subblock
        
        # JAX backend for GPU acceleration
        >>> S = ent.entropy_correlation(bipart, [0,1,2,3,4], backend='jax')
        
        # Multipartition - compute multiple cuts efficiently
        >>> masks = [[0,1,2], [0,1,2,3], [0,1,2,3,4]]
        >>> results = ent.entropy_multipartition(masks, [0,1,2,3,4,5])
        >>> print(results['entropies'])  # Array of entropies
        >>> C_matrices = results['correlation_matrices']  # Access all C_A matrices
        
        # Non-contiguous bipartition (use many-body method!)
        >>> bipart = ent.bipartition([0, 2, 4, 6, 8])  # Even sites
        >>> S_exact = ent.entropy_many_body(bipart, state)  # EXACT
        >>> # DON'T use entropy_correlation for non-contiguous!
        
        # Compare methods to check accuracy
        >>> result = ent.compare_methods(bipart, orbitals)
        >>> print(f"Correlation: {result['correlation']:.4f}")
        >>> print(f"Many-body:   {result['many_body']:.4f}")
        >>> print(f"Difference:  {result['difference']:.4e}")
        
        # Topological entanglement entropy
        >>> result = ent.topological_entropy(orbitals, construction='kitaev_preskill')
        >>> print(f"TEE γ = {result['gamma']:.4f}")
        
        # Verify Wick's theorem
        >>> result = ent.verify_wicks_theorem(orbitals)
        >>> print(f"Is Slater determinant: {result['is_valid']}")
        
        # Entropy scaling
        >>> results = ent.entropy_scan([0,1,2,3,4])
        >>> plt.plot(results['sizes'], results['entropies'])
        
        # Mutual information
        >>> I_AB = ent.mutual_information([0,1,2], [3,4,5], [0,1,2,3,4,5])
        
        """
        print(help_text)
    
    def __repr__(self):
        return f"EntanglementModule(ns={self._operator.ns}, quadratic={hasattr(self._operator, '_particle_conserving')})"


def get_entanglement_module(hamiltonian) -> EntanglementModule:
    """
    Factory function to create entanglement module for a Hamiltonian.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian object
        
    Returns
    -------
    EntanglementModule
        Entanglement module instance
    """
    return EntanglementModule(hamiltonian)
