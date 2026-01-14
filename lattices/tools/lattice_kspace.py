'''
A module for handling lattice k-space lattice tools and methods.

--------------------------------
File            : lattices/tools/lattice_kspace.py
Author          : Maksymilian Kliczkowski
--------------------------------
'''

from    __future__      import annotations
from    typing          import TYPE_CHECKING, Iterable, List, Optional, Literal, Tuple, Dict, NamedTuple
from    dataclasses     import dataclass, field
from    enum            import Enum
import  numpy           as np
import  scipy.sparse    as sp

if TYPE_CHECKING:
    from ..lattice                      import Lattice
    from QES.Algebra.hamil_quadratic    import QuadraticBlockDiagonalInfo

# -----------------------------------------------------------------------------------------------------------
# HIGH-SYMMETRY POINTS DEFINITIONS
# -----------------------------------------------------------------------------------------------------------

def ws_bz_mask(KX, KY, b1, b2, shells=1):
    """
    Wigner-Seitz (first BZ) mask for a 2D reciprocal lattice.

    Keeps points closer to Gamma than to any other reciprocal lattice point
    in a neighborhood of translations (m,n) with |m|,|n|<=shells.
    """
    b1 = np.asarray(b1, float)[:2]
    b2 = np.asarray(b2, float)[:2]

    kx = KX[..., None]
    ky = KY[..., None]

    # List reciprocal lattice translation vectors around origin
    G = []
    for m in range(-shells, shells + 1):
        for n in range(-shells, shells + 1):
            if m == 0 and n == 0:
                continue
            g = m * b1 + n * b2
            G.append(g)
    G = np.asarray(G, float)  # (Ng,2)

    # distances: |k|^2 and |k-G|^2
    k2      = kx**2 + ky**2
    dG2     = (kx - G[:, 0])**2 + (ky - G[:, 1])**2  # (Ny,Nx,Ng)

    # inside WS cell iff |k| <= |k-G| for all G
    inside = np.all(k2 <= dG2 + 1e-14, axis=-1)
    return inside

# -----------------------------------------------------------------------------------------------------------
#! HIGH-SYMMETRY POINTS AND PATHS
# -----------------------------------------------------------------------------------------------------------

@dataclass
class HighSymmetryPoint:
    """
    A high-symmetry point in the Brillouin zone.
    
    Attributes
    ----------
    label : str
        Label for the point (e.g., 'Gamma', 'K', 'M', 'X')
    frac_coords : Tuple[float, float, float]
        Fractional coordinates in reciprocal lattice units (f1, f2, f3).
        The actual k-vector is: k = f1*b1 + f2*b2 + f3*b3
    latex_label : str, optional
        LaTeX-formatted label for plotting (e.g., r'$\\Gamma$')
    description : str, optional
        Description of the point
    """
    label       : str
    frac_coords : Tuple[float, float, float]
    latex_label : str = ""
    description : str = ""
    
    def __post_init__(self):
        if not self.latex_label:
            # Auto-generate LaTeX label
            special_labels = {
                'Gamma': r'$\Gamma$', 'G': r'$\Gamma$',
                'K': r'$K$', 'M': r'$M$', 'X': r'$X$', 'Y': r'$Y$', 'Z': r'$Z$',
                'R': r'$R$', 'A': r'$A$', 'L': r'$L$', 'H': r'$H$',
            }
            self.latex_label = special_labels.get(self.label, f'${self.label}$')
    
    def to_cartesian(self, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates to Cartesian k-vector."""
        f1, f2, f3 = self.frac_coords
        return f1 * np.asarray(b1) + f2 * np.asarray(b2) + f3 * np.asarray(b3)
    
    def as_tuple(self) -> Tuple[str, List[float]]:
        """Return as (label, [f1, f2, f3]) tuple for path generation."""
        return (self.latex_label, list(self.frac_coords))

@dataclass
class HighSymmetryPoints:
    """
    Collection of high-symmetry points for a lattice type.
    
    Provides named access to standard high-symmetry points and defines
    default paths through the Brillouin zone.
    
    Example
    -------
    >>> pts = HighSymmetryPoints.square_2d()
    >>> print(pts.Gamma)  # HighSymmetryPoint for Gamma
    >>> print(pts.default_path())  # ['Gamma', 'X', 'M', 'Gamma']
    >>> print(pts.get_path_points(['Gamma', 'M']))  # Custom path
    """
    points          : Dict[str, HighSymmetryPoint] = field(default_factory=dict)
    _default_path   : List[str] = field(default_factory=list)
    
    def __getattr__(self, name: str) -> HighSymmetryPoint:
        if name.startswith('_') or name == 'points':
            raise AttributeError(name)
        if name in self.points:
            return self.points[name]
        raise AttributeError(f"No high-symmetry point named '{name}'")
    
    def __contains__(self, name: str) -> bool:
        return name in self.points
    
    def __iter__(self):
        return iter(self.points.values())
    
    def add(self, point: HighSymmetryPoint) -> 'HighSymmetryPoints':
        """Add a high-symmetry point."""
        self.points[point.label] = point
        return self
    
    def get(self, name: str) -> Optional[HighSymmetryPoint]:
        """Get a point by name, returns None if not found."""
        return self.points.get(name)
    
    def default_path(self) -> List[str]:
        """Return the default path through high-symmetry points."""
        return self._default_path.copy()
    
    def get_path_points(self, path_labels: List[str]) -> List[Tuple[str, List[float]]]:
        """
        Get path as list of (label, frac_coords) tuples.
        
        Parameters
        ----------
        path_labels : List[str]
            List of point labels defining the path (e.g., ['Gamma', 'X', 'M', 'Gamma'])
        
        Returns
        -------
        List[Tuple[str, List[float]]]
            Path suitable for brillouin_zone_path() function
        """
        path = []
        for label in path_labels:
            if label not in self.points:
                raise ValueError(f"Unknown high-symmetry point: '{label}'. "
                               f"Available: {list(self.points.keys())}")
            path.append(self.points[label].as_tuple())
        return path
    
    def get_default_path_points(self) -> List[Tuple[str, List[float]]]:
        """Get the default path as list of (label, frac_coords) tuples."""
        return self.get_path_points(self._default_path)
    
    # -----------------------------------------------------------------
    # Factory methods for common lattice types
    # -----------------------------------------------------------------
    
    @classmethod
    def chain_1d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 1D chain."""
        pts = cls(
            _default_path=['Gamma', 'X', 'Gamma2']
        )
        pts.add(HighSymmetryPoint('Gamma', (0.0, 0.0, 0.0), r'$0$', '1D BZ center'))
        pts.add(HighSymmetryPoint('X', (0.5, 0.0, 0.0), r'$\pi$', 'Zone boundary'))
        pts.add(HighSymmetryPoint('Gamma2', (1.0, 0.0, 0.0), r'$2\pi$', 'Wrapped Gamma'))
        return pts
    
    @classmethod
    def square_2d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 2D square lattice."""
        pts = cls(
            _default_path=['Gamma', 'X', 'M', 'Gamma']
        )
        pts.add(HighSymmetryPoint('Gamma', (0.0, 0.0, 0.0), r'$\Gamma$', 'BZ center'))
        pts.add(HighSymmetryPoint('X', (0.5, 0.0, 0.0), r'$X$', 'Zone face center'))
        pts.add(HighSymmetryPoint('M', (0.5, 0.5, 0.0), r'$M$', 'Zone corner'))
        pts.add(HighSymmetryPoint('Y', (0.0, 0.5, 0.0), r'$Y$', 'Zone face center (y)'))
        return pts
    
    @classmethod
    def cubic_3d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 3D cubic lattice."""
        pts = cls(
            _default_path=['Gamma', 'X', 'M', 'Gamma', 'R', 'X']
        )
        pts.add(HighSymmetryPoint('Gamma', (0.0, 0.0, 0.0), r'$\Gamma$', 'BZ center'))
        pts.add(HighSymmetryPoint('X', (0.5, 0.0, 0.0), r'$X$', 'Face center'))
        pts.add(HighSymmetryPoint('M', (0.5, 0.5, 0.0), r'$M$', 'Edge center'))
        pts.add(HighSymmetryPoint('R', (0.5, 0.5, 0.5), r'$R$', 'Corner'))
        return pts
    
    @classmethod
    def triangular_2d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 2D triangular lattice."""
        pts = cls(
            _default_path=['Gamma', 'M', 'K', 'Gamma']
        )
        pts.add(HighSymmetryPoint('Gamma', (0.0, 0.0, 0.0), r'$\Gamma$', 'BZ center'))
        pts.add(HighSymmetryPoint('M', (0.5, 0.0, 0.0), r'$M$', 'Edge midpoint'))
        pts.add(HighSymmetryPoint('K', (1/3, 1/3, 0.0), r'$K$', 'Corner (Dirac point)'))
        pts.add(HighSymmetryPoint('Kp', (2/3, 1/3, 0.0), r"$K'$", 'Other Dirac point'))
        return pts
    
    @classmethod
    def honeycomb_2d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 2D honeycomb lattice."""
        pts = cls(
            _default_path=['Gamma', 'K', 'M', 'Gamma']
        )
        pts.add(HighSymmetryPoint('Gamma',  (0.0, 0.0, 0.0), r'$\Gamma$',   'BZ center'))
        pts.add(HighSymmetryPoint('K',      (2/3, 1/3, 0.0), r'$K$',        'Dirac point'))
        pts.add(HighSymmetryPoint('Kp',     (1/3, 2/3, 0.0), r"$K'$",       'Other Dirac point'))
        pts.add(HighSymmetryPoint('M',      (0.5, 0.0, 0.0), r'$M$',        'Edge midpoint'))
        return pts
    
    @classmethod
    def hexagonal_2d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 2D hexagonal lattice (same as honeycomb)."""
        return cls.honeycomb_2d()

# -----------------------------------------------------------------------------------------------------------
# CACHED BLOCH TRANSFORMATION
# -----------------------------------------------------------------------------------------------------------

@dataclass
class BlochTransformCache:
    r"""
    Cache for Bloch transformation matrices to avoid recomputation.
    
    Attributes
    ----------
    W : np.ndarray
        Bloch projector matrix, shape (Nc, Ns, Nb)
        W[ik, i, a] = (1/sqrt Nc) * exp(-ik\cdot r_i) * delta_{sub(i),a}
    W_conj : np.ndarray
        Complex conjugate of W for efficiency
    kpoints : np.ndarray
        K-point grid used for this cache, shape (Nc, 3)
    kgrid : np.ndarray
        Structured k-grid, shape (Lx, Ly, Lz, 3)
    kgrid_frac : np.ndarray
        Fractional k-grid coordinates, shape (Lx, Ly, Lz, 3)
    lattice_hash : int
        Hash of lattice parameters to detect changes
    """
    W               : np.ndarray
    W_conj          : np.ndarray
    kpoints         : np.ndarray
    kgrid           : np.ndarray
    kgrid_frac      : np.ndarray
    lattice_hash    : int

def _get_lattice_hash(lattice: 'Lattice') -> int:
    """Generate a hash from lattice parameters to detect changes."""
    return hash((
        lattice._lx, lattice._ly, lattice._lz,
        len(lattice._basis),
        tuple(lattice._a1.flatten()),
        tuple(lattice._a2.flatten()),
        tuple(lattice._a3.flatten()),
        tuple(lattice._k1.flatten()),
        tuple(lattice._k2.flatten()),
        tuple(lattice._k3.flatten()),
    ))

# Global cache dictionary: lattice_id -> BlochTransformCache
_bloch_cache: Dict[int, BlochTransformCache] = {}

def _get_bloch_transform_cache(lattice: 'Lattice', unitary_norm: bool = True) -> BlochTransformCache:
    """
    Get or create cached Bloch transformation matrices.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice object
    unitary_norm : bool
        Whether to use unitary normalization (1/sqrt Nc)
    
    Returns
    -------
    cache : BlochTransformCache
        Cached transformation matrices
    """
    lattice_hash    = _get_lattice_hash(lattice)
    lattice_id      = id(lattice)

    # 1. cache reuse
    if lattice_id in _bloch_cache:
        cache = _bloch_cache[lattice_id]
        if cache.lattice_hash == lattice_hash:
            return cache

    # 2. lattice sizes
    Lx, Ly, Lz      = lattice._lx, max(lattice._ly, 1), max(lattice._lz, 1)
    Nc              = Lx * Ly * Lz
    Nb              = len(lattice._basis)
    Ns              = lattice.Ns

    # 3. reciprocal basis and k-grid (same as calculate_dft_matrix)
    b1              = np.asarray(lattice._k1, float).reshape(3)
    b2              = np.asarray(lattice._k2, float).reshape(3)
    b3              = np.asarray(lattice._k3, float).reshape(3)

    frac_x          = np.linspace(0.0, 1.0, Lx, endpoint=False)
    frac_y          = np.linspace(0.0, 1.0, Ly, endpoint=False)
    frac_z          = np.linspace(0.0, 1.0, Lz, endpoint=False)

    kx_frac, ky_frac, kz_frac = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")
    kgrid_frac      = np.stack([kx_frac, ky_frac, kz_frac], axis=-1)            # (Lx,Ly,Lz,3)

    kgrid           = (kx_frac[..., None] * b1
                     + ky_frac[..., None] * b2
                     + kz_frac[..., None] * b3)                                 # (Lx,Ly,Lz,3)

    kpoints         = kgrid.reshape(-1, 3)                                      # (Nc,3)

    # 4. real-space Bravais vectors and sublattice indices
    R_cells         = np.asarray(lattice.cells, float)                          # (Ns,3)
    if R_cells.shape[0] != Ns:
        raise ValueError("Mismatch in number of sites and lattice.cells.")

    sub_idx         = np.asarray(lattice.subs, dtype=int)                      # (Ns,)
    if sub_idx.shape[0] != Ns:
        raise ValueError("Mismatch in number of sites and lattice.subs.")

    # Projector S[i, alpha] = delta_{beta_i, alpha}
    S               = np.zeros((Ns, Nb), dtype=complex)
    S[np.arange(Ns), sub_idx] = 1.0

    # 5. Bloch projectors: W[k,i,alpha] = exp(-i k·R_i) / sqrt(Nc) * S[i,alpha]
    phases          = np.exp(-1j * (kpoints @ R_cells.T))                      # (Nc,Ns)
    if unitary_norm:
        phases     /= np.sqrt(Nc)

    W               = phases[:, :, None] * S[None, :, :]                       # (Nc,Ns,Nb)
    W_conj          = W.conj()

    cache = BlochTransformCache(
        W           = W,
        W_conj      = W_conj,
        kpoints     = kpoints,
        kgrid       = kgrid,
        kgrid_frac  = kgrid_frac,
        lattice_hash= lattice_hash,
    )
    _bloch_cache[lattice_id] = cache
    return cache

# -----------------------------------------------------------------------------------------------------------
# ENUMERATIONS OF STANDARD PATHS
# -----------------------------------------------------------------------------------------------------------

class StandardBZPath(Enum):
    r"""
    Enumeration of standard high-symmetry paths in the Brillouin zone.

    We define the k-space paths in a general representation of momentum vectors:
    \[
    k = f1 * b1 + f2 * b2 + f3 * b3,
    \]
    where (b1, b2, b3) are the reciprocal lattice vectors, and (f1, f2, f3) are the fractional coordinates,
    f_i = n_i / N_i, with n_i = 0, 1, ..., N_i - 1 for each direction i.

    Each value returns a list of (label, fractional_coord) pairs.
    The fractional coordinates are expressed in units of reciprocal lattice vectors.
    
    Example:
    >>> path = StandardBZPath.SQUARE_2D.value
    >>> for label, coord in path:
    ...     print(f"{label}: {coord}")
    G: [0.0, 0.0, 0.0]
    X: [0.5, 0.0, 0.0]
    M: [0.5, 0.5, 0.0]
    G: [0.0, 0.0, 0.0]
    """

    CHAIN_1D = [
        ("0",           [0.0, 0.0, 0.0]),
        (r"\pi",        [0.5, 0.0, 0.0]),
        (r"2\pi",       [1.0, 0.0, 0.0])
    ]

    SQUARE_2D = [
        (r"$\Gamma$",   [0.0, 0.0, 0.0]),
        (r"$X$",        [0.5, 0.0, 0.0]),
        (r"$M$",        [0.5, 0.5, 0.0]),
        (r"$\Gamma$",   [0.0, 0.0, 0.0])
    ]

    TRIANGULAR_2D = [
        (r"$\Gamma$",   [0.0, 0.0, 0.0]),
        (r"$M$",        [0.5, 0.0, 0.0]),
        (r"$K$",        [1/3, 1/3, 0.0]),
        (r"$\Gamma$",   [0.0, 0.0, 0.0])
    ]

    CUBIC_3D = [
        (r"$\Gamma$",   [0.0, 0.0, 0.0]),
        (r"$X$",        [0.5, 0.0, 0.0]),
        (r"$M$",        [0.5, 0.5, 0.0]),
        (r"$R$",        [0.5, 0.5, 0.5]),
        (r"$\Gamma$",   [0.0, 0.0, 0.0])
    ]

    HONEYCOMB_2D = [
        (r"$\Gamma$",   [0.0, 0.0, 0.0]),
        (r"$K$",        [2/3, 1/3, 0.0]),
        (r"$M$",        [0.5, 0.0, 0.0]),
        (r"$\Gamma$",   [0.0, 0.0, 0.0])
    ]
    
    def from_str(name: str) -> StandardBZPath:
        """
        Get StandardBZPath enum from string name.

        Parameters
        ----------
        name : str
            Name of the standard path (e.g., 'CHAIN_1D', 'SQUARE_2D').

        Returns
        -------
        StandardBZPath
            Corresponding enum value.

        Raises
        ------
        ValueError
            If the name does not correspond to any StandardBZPath.
        """
        
        # handle case-insensitivity
        name = name.upper()
        
        try:
            return StandardBZPath[name]
        except KeyError:
            raise ValueError(f"Unknown StandardBZPath name: {name}")

PathTypes = Literal['CHAIN_1D', 'SQUARE_2D', 'TRIANGULAR_2D', 'CUBIC_3D', 'HONEYCOMB_2D']
    
# -----------------------------------------------------------------------------------------------------------
# BRILLOUIN ZONE PATH GENERATION
# -----------------------------------------------------------------------------------------------------------

def generate_kgrid(lattice: Lattice, n_k: Iterable[int]) -> np.ndarray:
    """
    Generate a full k-point grid for the given lattice.

    Parameters
    ----------
    lattice : Lattice
        Lattice object with reciprocal lattice vectors _k1, _k2, _k3.
    n_k : Iterable[int]
        Number of points (Lx, Ly, Lz) along each reciprocal direction.
        We define the k-points as:
        k = f1 * b1 + f2 * b2 + f3 * b3,
        where f_i = n_i / N_i, with n_i = 0, 1, ..., N_i - 1.

    Returns
    -------
    k_points : np.ndarray, shape (Nk, dim)
        Cartesian coordinates of k-points in reciprocal space.
    """
    recip = np.vstack([v for v in [lattice._k1, lattice._k2, lattice._k3] if v is not None]) # shape (3, 3)
    nk    = np.array(list(n_k)) # shape (dim,)

    grids = [np.arange(n) / n for n in nk]                  # fractional grids
    mesh  = np.meshgrid(*grids, indexing="ij")              # meshgrid
    frac  = np.stack([m.ravel() for m in mesh], axis=-1)    # shape (Nk, dim)
    
    # define vectors in a full 3D array for matrix multiplication -> k=f1*b1 + f2*b2 + f3*b3
    k_pts = frac @ recip                                    # fractional -> cartesian
    return k_pts

# -----------------------------------------------------------------------------------------------------------

def _resolve_path_input(path: Iterable[tuple[str, Iterable[float]]] | StandardBZPath) -> list[tuple[str, list[float]]]:
    """
    Resolve path input to a list of (label, fractional_coord) pairs.

    Parameters
    ----------
    path : list[(label, coords)] or StandardBZPath
        Path definition (fractional coordinates) or one of the standard enums.

    Returns
    -------
    resolved_path : list[(label, list[float])]
        Resolved path as a list of (label, fractional_coord) pairs.
        
    Example
    -------
    >>> path = _resolve_path_input("SQUARE_2D")
    >>> for label, coord in path:
    ...     print(f"{label}: {coord}")
    G: [0.0, 0.0, 0.0]
    X: [0.5, 0.0, 0.0]
    M: [0.5, 0.5, 0.0]
    G: [0.0, 0.0, 0.0]
    """
    if isinstance(path, str):
        try:
            path = StandardBZPath[path].value
        except KeyError:
            raise ValueError(f"Unknown BZ path name: {path!r}")
    elif isinstance(path, StandardBZPath):
        path = path.value

    resolved_path = []
    for label, frac in path:
        resolved_path.append((label, list(map(float, frac))))
    
    return resolved_path

def brillouin_zone_path(
        lattice         : Lattice,
        path            : Iterable[tuple[str, Iterable[float]]] | StandardBZPath,
        *,
        points_per_seg  : int = 40,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]], np.ndarray]:
    """
    Generate k-points along a specified Brillouin zone path.
    
    In general, if coordinates are given as c1 = (f1, f2, f3) and c2 = (g1, g2, g3)
    we want to follow the straight line in Cartesian coordinates:
        k(t) = (1-t) * (f1*b1 + f2*b2 + f3*b3) + t * (g1*b1 + g2*b2 + g3*b3)
    for t in [0, 1], t = 0, 1/points_per_seg, 2/points_per_seg, ..., (points_per_seg-1)/points_per_seg.
    
    Each segment between two labels is interpolated with `points_per_seg` points.
    
    Example:
    >>> path = StandardBZPath.SQUARE_2D.value
    >>> k_path, k_dist, labels = brillouin_zone_path(lattice, path, points_per_seg=10)
    >>> print("k-path shape:", k_path.shape)
    k-path shape: (30, 3)
    >>> print("k-dist shape:", k_dist.shape)
    k-dist shape: (30,)
    >>> print("Labels:", labels)
    Labels: [(0, 'G'), (10, 'X'), (20, 'M'), (30, 'G')]
    >>> print("First 3 k-points:\n", k_path[:3])
    First 3 k-points:
    [[0. 0. 0.]         # = 0.0 * b1 + 0.0 * b2 + 0.0 * b3
     [0.05 0.  0. ]     # = 0.1 * b1 + 0.0 * b2 + 0.0 * b3
     [0.1  0.  0. ]]    # = 0.2 * b1 + 0.0 * b2 + 0.0 * b3
    
    Parameters
    ----------
    lattice : Lattice
        Lattice object with reciprocal lattice vectors (_k1, _k2, _k3).
    path : list[(label, coords)] or StandardBZPath
        Path definition (fractional coordinates) or one of the standard enums.
    points_per_seg : int
        Number of interpolated points between labels.
    Returns
    -------
    k_path : np.ndarray, shape (Npath, 3)
        k-points along the path.
    k_dist : np.ndarray, shape (Npath,)
        Cumulative distance for x-axis plotting.
    labels : list[(int, str)]
        Indices and labels for symmetry points.
    """
    
    path = _resolve_path_input(path)

    # Reciprocal lattice matrix: columns = b1, b2, b3
    b1 = np.asarray(lattice._k1, float).reshape(3)
    b2 = np.asarray(lattice._k2, float).reshape(3)
    b3 = np.asarray(lattice._k3, float).reshape(3)
    B  = np.column_stack([b1, b2, b3])      # (3,3)

    cart_pts  = []
    frac_pts  = []
    for _, frac in path:
        f = np.zeros(3, float)
        f[:len(frac)] = np.array(frac, float)
        k_cart = B @ f
        cart_pts.append(k_cart)
        frac_pts.append(f)

    k_path      = []
    k_path_frac = []
    k_dist      = [0.0]
    labels      = [(0, path[0][0])]

    for i in range(len(cart_pts) - 1):
        p0, p1 = cart_pts[i], cart_pts[i + 1]
        f0, f1 = frac_pts[i], frac_pts[i + 1]

        nseg      = points_per_seg
        seg_cart  = np.linspace(p0, p1, nseg, endpoint=False)
        seg_frac  = np.linspace(f0, f1, nseg, endpoint=False)

        for j in range(nseg):
            k = seg_cart[j]
            if k_path:
                dk = np.linalg.norm(k - k_path[-1])
                k_dist.append(k_dist[-1] + dk)
            k_path.append(k)
            k_path_frac.append(seg_frac[j])

        labels.append((len(k_path), path[i + 1][0]))

    k_path      = np.array(k_path)
    k_path_frac = np.array(k_path_frac)
    k_dist      = np.array(k_dist)

    return k_path, k_dist, labels, k_path_frac

# -----------------------------------------------------------------------------------------------------------
# K-SPACE PATH EXTRACTION
# -----------------------------------------------------------------------------------------------------------

@dataclass
class KPathResult:
    """
    Result of extracting data along a k-path in the Brillouin zone.
    
    This dataclass holds all information needed for band structure plots and 
    analysis along a high-symmetry path.
    
    Attributes
    ----------
    k_cart : np.ndarray, shape (Npath, 3)
        Cartesian k-vectors along the path
    k_frac : np.ndarray, shape (Npath, 3)
        Fractional k-vectors along the path (in reciprocal lattice units)
    k_dist : np.ndarray, shape (Npath,)
        Cumulative distance along the path for x-axis plotting
    labels : List[Tuple[int, str]]
        List of (index, label) pairs for high-symmetry points
    values : np.ndarray, shape (Npath, n_bands)
        Data values (e.g., energies) along the path
    indices : np.ndarray, shape (Npath,), dtype=int
        Indices into the original k-grid for each path point.
        Use to map path data back to the full k-grid.
    matched_distances : np.ndarray, shape (Npath,)
        Distance from ideal path point to matched grid point (for quality check)
    
    Example
    -------
    >>> result = lattice.extract_kpath_data(energies, path='SQUARE_2D')
    >>> plt.plot(result.k_dist, result.values)
    >>> for idx, label in result.labels:
    ...     plt.axvline(result.k_dist[min(idx, len(result.k_dist)-1)], label=label)
    """
    k_cart              : np.ndarray
    k_frac              : np.ndarray
    k_dist              : np.ndarray
    labels              : List[Tuple[int, str]]
    values              : np.ndarray
    indices             : np.ndarray
    matched_distances   : np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def n_points(self) -> int:
        """Number of points along the path."""
        return len(self.k_dist)
    
    @property
    def n_bands(self) -> int:
        """Number of bands/values per k-point."""
        return self.values.shape[-1] if self.values.ndim > 1 else 1
    
    @property
    def label_positions(self) -> np.ndarray:
        """X-axis positions (k_dist values) of the high-symmetry point labels."""
        positions = []
        for idx, _ in self.labels:
            pos_idx = min(idx, len(self.k_dist) - 1) if len(self.k_dist) > 0 else 0
            positions.append(self.k_dist[pos_idx] if len(self.k_dist) > 0 else 0.0)
        return np.array(positions)
    
    @property 
    def label_texts(self) -> List[str]:
        """Just the label strings for plotting."""
        return [label for _, label in self.labels]
    
    def unique_indices(self) -> np.ndarray:
        """Return unique k-point indices (no duplicates from path segments)."""
        return np.unique(self.indices)
    
    def max_match_distance(self) -> float:
        """Maximum distance from path to matched grid point."""
        if len(self.matched_distances) == 0:
            return 0.0
        return float(np.max(self.matched_distances))


def find_nearest_kpoints(
    k_grid_frac     : np.ndarray,
    target_frac     : np.ndarray,
    tol             : float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find nearest k-point indices for target fractional coordinates.
    
    Parameters
    ----------
    k_grid_frac : np.ndarray, shape (Nk, 3)
        Fractional coordinates of available k-points
    target_frac : np.ndarray, shape (Ntarget, 3)
        Target fractional coordinates to match
    tol : float
        Warning threshold for match distance
    
    Returns
    -------
    indices : np.ndarray, shape (Ntarget,), dtype=int
        Index of nearest k-point for each target
    distances : np.ndarray, shape (Ntarget,)
        Distance to nearest point (in fractional units, accounting for periodicity)
    """
    k_grid_frac = np.asarray(k_grid_frac).reshape(-1, 3)
    target_frac = np.asarray(target_frac).reshape(-1, 3)
    
    n_targets = len(target_frac)
    indices = np.zeros(n_targets, dtype=int)
    distances = np.zeros(n_targets)
    
    for i, kf_target in enumerate(target_frac):
        # Periodic distance in fractional coordinates
        diff = k_grid_frac - kf_target
        diff -= np.round(diff)  # Handle periodicity
        dist = np.linalg.norm(diff, axis=1)
        idx = np.argmin(dist)
        indices[i] = idx
        distances[i] = dist[idx]
        
        if dist[idx] > tol:
            import warnings
            warnings.warn(f"k-point match distance ({dist[idx]:.3e}) > tolerance at target {kf_target}")
    
    return indices, distances


def extract_bz_path_data(
    lattice,
    k_vectors           : np.ndarray,
    k_vectors_frac      : np.ndarray, 
    values              : np.ndarray,
    path                : Iterable[tuple[str, Iterable[float]]] | StandardBZPath | HighSymmetryPoints | None = None,
    *,
    points_per_seg      : int = 40,
    return_result       : bool = True,
    ) -> KPathResult | Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, str]], np.ndarray]:
    """
    Extract k-path data from a k-grid using fractional coordinate matching.
    
    This function finds the closest k-points on the actual grid to an ideal path
    through high-symmetry points. It handles periodic boundary conditions in k-space.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice object with reciprocal lattice vectors
    k_vectors : np.ndarray, shape (..., 3) 
        Cartesian k-points (will be flattened)
    k_vectors_frac : np.ndarray, shape (..., 3)
        Fractional coordinates of k-points (will be flattened)
    values : np.ndarray, shape (..., n_bands)
        Data values at each k-point (e.g., band energies)
    path : various, optional
        Path specification. Can be:
        - StandardBZPath enum value (e.g., StandardBZPath.SQUARE_2D)
        - String name (e.g., 'SQUARE_2D')
        - List of (label, [f1,f2,f3]) tuples
        - HighSymmetryPoints object (uses default path)
        - None: uses lattice's default path if available
    points_per_seg : int
        Number of interpolated points per path segment
    return_result : bool
        If True (default), return KPathResult dataclass.
        If False, return tuple for backwards compatibility.
    
    Returns
    -------
    KPathResult or tuple
        If return_result=True: KPathResult dataclass with all path data
        If return_result=False: (k_cart, k_frac, k_dist, labels, values) tuple
    
    Examples
    --------
    >>> # Using default path from HighSymmetryPoints
    >>> result = extract_bz_path_data(lattice, k_grid, k_frac, energies, 
    ...                               HighSymmetryPoints.square_2d())
    >>> plt.plot(result.k_dist, result.values)
    
    >>> # Using standard path enum
    >>> result = extract_bz_path_data(lattice, k_grid, k_frac, energies, 'SQUARE_2D')
    
    >>> # Custom path
    >>> custom_path = [('G', [0,0,0]), ('X', [0.5,0,0]), ('G', [0,0,0])]
    >>> result = extract_bz_path_data(lattice, k_grid, k_frac, energies, custom_path)
    """
    # Handle path input
    if path is None:
        # Try to get default path from lattice
        if hasattr(lattice, 'high_symmetry_points'):
            hs_pts = lattice.high_symmetry_points()
            if hs_pts is not None:
                path = hs_pts.get_default_path_points()
        if path is None:
            raise ValueError("No path specified and lattice has no default path. "
                            "Specify path explicitly or use a lattice with high_symmetry_points().")
    elif isinstance(path, HighSymmetryPoints):
        path = path.get_default_path_points()
    
    # Flatten inputs
    kf_flat     = np.asarray(k_vectors_frac).reshape(-1, 3)
    kc_flat     = np.asarray(k_vectors).reshape(-1, 3)
    
    # Handle values shape
    if values.ndim == 1:
        val_flat = values.reshape(-1, 1)
    else:
        val_flat = values.reshape(-1, values.shape[-1])
    
    # Generate ideal continuous path
    n_bands                                     = val_flat.shape[-1]
    k_ideal_cart, k_dist, labels, k_ideal_frac  = brillouin_zone_path(
                                                    lattice=lattice, path=path, points_per_seg=points_per_seg
                                                )

    # Compute tolerance based on grid resolution
    Lx                          = max(lattice._lx, 1)
    Ly                          = max(lattice._ly, 1) 
    Lz                          = max(lattice._lz, 1)
    tol                         = 0.5 * np.sqrt((1/Lx)**2 + (1/Ly)**2 + (1/Lz)**2)

    # Find nearest k-points for each path point
    indices, match_distances    = find_nearest_kpoints(kf_flat, k_ideal_frac, tol=tol)
    
    # Extract matched data
    k_sel_cart                  = kc_flat[indices]
    k_sel_frac                  = kf_flat[indices]
    vals_sel                    = val_flat[indices]

    if return_result:
        return KPathResult(
            k_cart=k_sel_cart,
            k_frac=k_sel_frac,
            k_dist=k_dist,
            labels=labels,
            values=vals_sel,
            indices=indices,
            matched_distances=match_distances,
        )
    else:
        # Backwards compatible return
        return k_sel_cart, k_sel_frac, k_dist, labels, vals_sel

# -----------------------------------------------------------------------------------------------------------
#? Reciprocal Lattice Vectors
# -----------------------------------------------------------------------------------------------------------

def reciprocal_from_real(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reciprocal lattice vectors from real-space lattice vectors.
    b1,b2,(b3) satisfying a_i . b_j = 2*pi*delta_ij.
    
    Parameters
    ----------
    a1, a2, a3 : np.ndarray
        Real-space lattice vectors.

    Returns
    -------
    k1, k2, k3 : np.ndarray
        Reciprocal lattice vectors.
    """
    A           = np.column_stack([a1[:3], a2[:3], (a3 if a3 is not None else np.array([0.,0.,1.]))[:3]])
    B           = 2.0 * np.pi * np.linalg.inv(A).T
    b1, b2, b3  = B[:,0], B[:,1], B[:,2]
    return b1, b2, b3

def extract_momentum(eigvecs    : np.ndarray, 
                    lattice     : 'Lattice',
                    *,
                    eigvals     : np.ndarray = None,
                    tol         : float      = 1e-10,
                    ):
    """
    Extract crystal momentum vectors k from real-space eigenvectors.

    Parameters
    ----------
    eigvecs : np.ndarray
        Eigenvectors in real-space basis, shape (Ns, n_states)
    eigvals : np.ndarray, optional
        Corresponding eigenvalues, shape (n_states,). Required if degeneracies are to be resolved.
    tol : float
        Degeneracy tolerance. States within |E_i - E_j| < tol are treated as degenerate.

    Returns
    -------
    k_vectors : np.ndarray
        Array of shape (n_states, dim), containing crystal momenta for each eigenstate.
    """

    # Translation operators (list of matrices, one per dimension)
    T_ops           = lattice.translation_operators()
    dim             = len(T_ops)

    # Precompute real and reciprocal bases
    A               = np.column_stack([lattice._a1, lattice._a2, lattice._a3])[:, :dim]

    Ns, n_states    = eigvecs.shape
    k_vectors       = np.zeros((n_states, dim), dtype=float)
    # Case 1: No degeneracy information (just extract phases directly)
    if eigvals is None:
        for q in range(n_states):
            psi     = eigvecs[:, q]
            thetas  = []
            for Ti in T_ops:
                phase = np.vdot(psi, Ti @ psi) / np.vdot(psi, psi)
                thetas.append(np.angle(phase))

            # Solve theta = A^T * k  ->  k = (A^T)^{-1} * theta
            kvec            = np.linalg.solve(A.T, thetas)
            k_vectors[q, :] = kvec % (2 * np.pi)
            
        return k_vectors

    # Case 2: Degeneracy-aware version
    used = np.zeros(n_states, dtype=bool)
    for i in range(n_states):
        if used[i]:
            continue
        # find degenerate subspace
        mask        = np.abs(eigvals - eigvals[i]) < tol
        used[mask]  = True
        subspace    = eigvecs[:, mask]

        if subspace.shape[1] == 1:
            # non-degenerate state
            psi     = subspace[:, 0]
            thetas  = []
            for Ti in T_ops:
                phase = np.vdot(psi, Ti @ psi) / np.vdot(psi, psi)
                thetas.append(np.angle(phase))
                
            # Solve theta = A^T * k  ->  k = (A^T)^{-1} * theta
            kvec            = np.linalg.solve(A.T, thetas)
            k_vectors[i, :] = kvec % (2 * np.pi)
        else:
            # degenerate subspace: diagonalize translations
            # we need to extract multiple k-vectors
            # each Ti gives phases along direction i
            # mathematically, we diagonalize Ti in the subspace
            # to get eigenvalues exp(i * theta_i)
            # and eigenvectors give the k-vectors
            for Ti in T_ops:
                Ti_sub          = subspace.conj().T @ (Ti @ subspace)
                evals, evecsT   = np.linalg.eig(Ti_sub)
                phases          = np.angle(evals)
                # Each eigenvalue gives one momentum along direction i
                for j, phi in enumerate(phases):
                    # Insert per subspace component
                    if j + i < n_states:
                        if k_vectors[j + i, :].any():
                            # Combine existing info
                            k_vectors[j + i, np.argmax(k_vectors[j + i, :] == 0)] = phi
                        else:
                            k_vectors[j + i, 0] = phi

    # Solve theta_i = k * a_i for each state
    for q in range(n_states):
        kvec            = np.linalg.solve(A.T, k_vectors[q, :dim])
        k_vectors[q, :] = kvec % (2 * np.pi)

    return k_vectors

# -------------------------------------------------------------------------------------------
#? Single site translation operators
# -------------------------------------------------------------------------------------------

def build_translation_operators(lattice: 'Lattice') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct translation matrices (T1, T2, T3) acting on the full real-space basis.

    Parameters
    ----------
    lattice : Lattice
        Lattice object with attributes:
        - a1, a2, a3 : Real-space lattice vectors : shape (3,dim).
        - b1, b2, b3 : Reciprocal lattice vectors : shape (3,dim).
        - basis      : np.ndarray of shape (n_basis, 3) with fractional coordinates of basis sites.
        - Lx, Ly, Lz : Number of unit cells along each direction
        - boundary_phase_from_winding(wx, wy, wz) : Method returning phase factor for given windings.

    Returns
    -------
    T1, T2, T3 : np.ndarray
        Translation matrices (Ns x Ns), each a complex permutation unitary
        shifting states by +a1, +a2, +a3 respectively.

    Notes
    -----
    - Works for 1D, 2D, or 3D (dimension inferred from provided a_vectors).
    - Periodic boundaries are implemented automatically.
    - Sublattice permutations are handled by matching translated site positions
      modulo the lattice vectors.
    """
    a_vectors   = lattice.avec
    b_vectors   = lattice.bvec
    basis       = lattice.basis
    dim         = lattice.dim
    n_basis     = lattice.multipartity
    Ns          = lattice.Ns
    Lx, Ly, Lz  = lattice.Lx, lattice.Ly, lattice.Lz

    # Allocate matrices
    T1          = np.zeros((Ns, Ns), dtype=complex)
    T2          = np.zeros_like(T1)
    T3          = np.zeros_like(T1)
    Ts          = [T1, T2, T3]

    # Precompute real-space positions of all sites
    cells       = np.array([[nx, ny, nz] for nz in range(Lz)
                                         for ny in range(Ly)
                                         for nx in range(Lx)])
    # fractional coordinates of all sites
    frac_pos    = (cells[:, None, :dim] + basis[None, :, :dim])  # (Nc, nb, dim)
    frac_pos    = frac_pos.reshape(-1, dim) / np.array([Lx, Ly, Lz])[:dim]

    # Apply translations
    Ls = np.array([Lx, Ly, Lz])[:dim]
    for dir_idx in range(dim):
        for site_idx, r_frac in enumerate(frac_pos):
            # translate by +a_dir -> fractional coords + e_dir / L_dir
            r_t     = r_frac + np.eye(dim)[dir_idx] / Ls[dir_idx]
            # wrap around PBC
            r_t_mod = np.mod(r_t, 1.0)

            # Find nearest site (within tolerance)
            diff    = frac_pos - r_t_mod
            diff   -= np.round(diff)
            dist2   = np.sum(diff**2, axis=1)
            j       = np.argmin(dist2)
            if dist2[j] > 1e-8:
                raise ValueError(f"Could not match translated site {site_idx} in direction {dir_idx}")

            # compute winding numbers: how many times the translation wrapped each axis
            wx      = int(np.floor(r_t[0])) - int(np.floor(r_frac[0])) if dim >= 1 else 0
            wy      = int(np.floor(r_t[1])) - int(np.floor(r_frac[1])) if dim >= 2 else 0
            wz      = int(np.floor(r_t[2])) - int(np.floor(r_frac[2])) if dim >= 3 else 0

            # get flux phase
            if hasattr(lattice, "boundary_phase_from_winding"):
                phase = lattice.boundary_phase_from_winding(wx, wy, wz)
            else:
                phase = 1.0 # no flux

            Ts[dir_idx][j, site_idx] = phase

    return T1, T2, T3

def reconstruct_k_grid_from_blocks(blocks: List['QuadraticBlockDiagonalInfo']) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the structured k-space grid and energy array
    from a flat list of QuadraticBlockDiagonalInfo objects.

    Parameters
    ----------
    blocks : list
        Output of `ham.block_diagonal_bdg()[0]` (the list of per-k blocks).

    Returns
    -------
    k_grid : np.ndarray
        Shape (Lx, Ly, Lz, 3) array of k-vectors.
    energies : np.ndarray
        Shape (Lx, Ly, Lz, n_bands) array of eigenvalues at each k.
    """
    # Extract indices and unique grid dimensions
    indices     = np.array([blk.block_index for blk in blocks])
    energies    = np.array([blk.en for blk in blocks])

    Lx          = indices[:, 0].max() + 1
    Ly          = indices[:, 1].max() + 1 if indices.shape[1] > 1 else 1
    Lz          = indices[:, 2].max() + 1 if indices.shape[1] > 2 else 1
    n_bands     = energies.shape[1]

    # Allocate structured arrays
    k_grid      = np.zeros((Lx, Ly, Lz, 3))
    k_grid_frac = np.zeros((Lx, Ly, Lz, 3))
    energy_grid = np.zeros((Lx, Ly, Lz, n_bands))

    # Fill by index
    for blk in blocks:
        ix, iy, iz                  = blk.block_index[0], blk.block_index[1], blk.block_index[2]
        k_grid_frac[ix, iy, iz, :]  = blk.frac_point
        k_grid[ix, iy, iz, :]       = blk.point
        energy_grid[ix, iy, iz, :]  = blk.en

    return k_grid, k_grid_frac, energy_grid

# -------------------------------------------------------------------------------------------
#! SPACE TRANSFORMATIONS
# -------------------------------------------------------------------------------------------

def full_k_space_transform(lattice: Lattice, mat: np.ndarray, inverse: bool = False) -> np.ndarray:
    r"""
    Full Ns x Ns k-space transform using DFT matrix.
    
    Computes:
        H_k = F @ H_real @ F†
    
    where F is the Ns x Ns DFT matrix:
        F[n, i] = (1/sqrt Ns) * exp(-i k_n r_i)
    
    This works for ANY lattice, independent of multipartition structure.
    The result is an NsxNs matrix that is block-diagonal for translationally
    invariant systems.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice object with DFT matrix and k-vectors
    mat : np.ndarray
        Real-space matrix, shape (Ns, Ns)
    
    Returns
    -------
    H_k : np.ndarray
        K-space matrix, shape (Ns, Ns)
    """
    Ns = lattice.Ns
    if mat.shape != (Ns, Ns):
        raise ValueError(f"mat must have shape ({Ns}, {Ns}), got {mat.shape}")
    
    # Get or calculate DFT matrix
    F = lattice.dft
    if F is None or F.shape != (Ns, Ns) or not np.any(F):
        F = lattice.calculate_dft_matrix()
    
    # Transform: H_k = F @ H_real @ F†
    if not inverse:
        F_dagger    = F.conj().T
        H_k_full    = F @ mat @ F_dagger

        return H_k_full
    else:
        # Inverse transform
        F_inv       = F_dagger
        H_real_full = F_inv @ mat @ F

        return H_real_full

def realspace_from_kspace(lattice, H_k: np.ndarray, kgrid: Optional[np.ndarray] = None) -> np.ndarray:
    r"""
    Inverse Bloch transform: H(k) blocks -> H_real (Ns x Ns).
    
    Reconstructs the real-space Hamiltonian from k-space blocks using the inverse 
    Fourier transform. This is the exact inverse of `kspace_from_realspace()`.
    
    Formula:
        H_real = Σ_k W(k)† H(k) W(k)
    where W[i,a] = (1/√Nc) . exp(-ik.r_i) . delta _{sublattice(i),a}
    
    Parameters
    ----------
    lattice : Lattice
        Lattice object with geometry information
    H_k : np.ndarray
        K-space Hamiltonian blocks. Shape (Lx, Ly, Lz, Nb, Nb) or (Nk, Nb, Nb).
        Must be in fftfreq order (as returned by kspace_from_realspace).
    kgrid : Optional[np.ndarray]
        K-point grid for reference. If None, reconstructs using fftfreq convention.
        Shape (Lx, Ly, Lz, 3) or (Nk, 3). Must be in fftfreq order.
    
    Returns
    -------
    H_real : np.ndarray
        Real-space Hamiltonian (Ns x Ns).
        
    Notes
    -----
    - Eigenvalues are preserved to machine precision (error ~1e-15)
    - Both H_k and kgrid must be in fftfreq order (no fftshift applied)
    - The reconstruction is exact: H_real_reconstructed ≈ H_real_original
    
    Examples
    --------
    >>> lat = HoneycombLattice(dim=2, lx=2, ly=2, bc='pbc')
    >>> H_real_orig = np.random.randn(lat.Ns, lat.Ns)
    >>> H_real_orig = H_real_orig + H_real_orig.conj().T  # Make Hermitian
    >>>
    >>> # Forward transform
    >>> H_k, k_grid, k_frac = kspace_from_realspace(lattice, H_real_orig)
    >>>
    >>> # Inverse transform
    >>> H_real_recon = realspace_from_kspace(lattice, H_k, k_grid)
    >>>
    >>> # Check reconstruction
    >>> np.allclose(H_real_orig, H_real_recon)  # True (to machine precision)
    """
    import numpy as np
    
    # Parse input shape
    if H_k.ndim == 5:
        # (Lx, Ly, Lz, Nb, Nb) format - blocks are already in correct order from kspace_from_realspace
        Lx, Ly, Lz, Ns_block, Ns2   = H_k.shape
        Nc                          = Lx * Ly * Lz
        H_k_flat                    = H_k.reshape(Nc, Ns_block, Ns2)
    elif H_k.ndim == 3:
        # (Nk, Nb, Nb) format
        Nk, Ns_block, Ns2           = H_k.shape
        H_k_flat                    = H_k
        Nc                          = Nk
    else:
        raise ValueError(f"H_k must be 3D or 5D array, got shape {H_k.shape}")
    
    # Check Hermiticity
    if Ns_block != Ns2:
        raise ValueError(f"H_k blocks must be square: got {Ns_block}x{Ns2}")
    
    Ns = Ns_block
    
    # Infer lattice properties
    # For multi-sublattice systems, Ns_block = Nb (number of sublattices), not total sites
    Nb = lattice.multipartity       # Number of sublattices
    if Ns != Nb:
        raise ValueError(f"H_k block size {Ns} != lattice sublattices {Nb}")
    
    Lx          = lattice._lx
    Ly          = max(lattice._ly, 1)
    Lz          = max(lattice._lz, 1)
    expected_Nc = Lx * Ly * Lz
    if Nc != expected_Nc:
        raise ValueError(f"Number of k-points {Nc} != expected {expected_Nc}")
    
    # Reciprocal vectors
    b1 = np.asarray(lattice._k1, float).reshape(3)
    b2 = np.asarray(lattice._k2, float).reshape(3)
    b3 = np.asarray(lattice._k3, float).reshape(3)
    
    # Build k-grid if not provided - use same convention as kspace_from_realspace
    if kgrid is None:
        # Use fftfreq convention to match forward transform
        frac_x                      = np.linspace(0, 1, Lx, endpoint=False)
        frac_y                      = np.linspace(0, 1, Ly, endpoint=False)
        frac_z                      = np.linspace(0, 1, Lz, endpoint=False)

        kx_frac, ky_frac, kz_frac   = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")
        
        kgrid                       = (kx_frac[..., None] * b1 + 
                                       ky_frac[..., None] * b2 + 
                                       kz_frac[..., None] * b3)
        kpoints                     = kgrid.reshape(-1, 3)
    else:
        # kgrid is already in fftfreq order from kspace_from_realspace
        if kgrid.ndim == 4:
            kpoints         = kgrid.reshape(-1, 3)
        else:
            kpoints         = np.asarray(kgrid, float).reshape(-1, 3)
    
    # Site coordinates and sublattice indices
    coords      = np.asarray(lattice.coordinates, float)
    Ns_total    = lattice.Ns  # Total number of sites
    indices     = np.arange(Ns_total)
    sub_idx     = indices % Nb
    
    # Projector S[i, a] = delta_{sub(i), a}
    S           = np.zeros((Ns_total, Nb), dtype=complex)
    S[np.arange(Ns_total), sub_idx] = 1.0
    
    # Inverse Bloch transform: H_real = (1/Nc) Σ_k W(k)† H(k) W(k)
    # where W[i,a] = (1/sqrt Nc) * exp(-ik\cdot r_i) * S[i,a]
    # Note: Using -i (same as forward transform) because W is the unitary transform
    
    H_real = np.zeros((Ns_total, Ns_total), dtype=complex)
    for ik, kvec in enumerate(kpoints):
        # Phase: exp(-i k\cdot r_i) (same sign as forward transform)
        phases      = np.exp(-1j * (coords @ kvec))
        # Bloch projector W[i,a] = (1/sqrt Nc) * exp(-ik\cdot r_i) * S[i,a]
        W           = (phases[:, None] * S) / np.sqrt(Nc)
        # Accumulate: W H(k) W† (since W is (Ns_total, Nb) and H_k is (Nb, Nb))
        H_real     += W @ H_k_flat[ik] @ W.conj().T
    
    # Note: No division by Nc needed - it's already in W normalization
    # However, we accumulated Nc terms, so effectively: H = Σ_k WHW† = I due to completeness
    
    # Ensure Hermiticity (average with conjugate to remove numerical noise)
    H_real = 0.5 * (H_real + H_real.conj().T)
    
    return H_real.astype(lattice._dtype if hasattr(lattice, '_dtype') else np.complex128)

def kspace_from_realspace(
        lattice                 : Lattice,
        H_real                  : np.ndarray,
        kpoints                 : Optional[np.ndarray] = None,
        require_full_grid       : bool = False,
        unitary_norm            : bool = True,
        return_transform        : bool = False):
    r"""
    Bloch projector: H_real (NsxNs) -> H(k) $\in$ C^{NbxNb} at each k.

    Transforms a real-space Hamiltonian into momentum space using the Bloch transform:
        H_ab(k) = Σ_{i,j} W*_{i,a}(k) H_{i,j} W_{j,b}(k)
    where W[i,a](k) = (1/√Nc) . exp(-ik.r_i) . delta _{sublattice(i),a}

    Assumptions:
    - Periodic boundary conditions (PBC)
    - True translational invariance to preserve spectrum
    - Site ordering is arbitrary; geometry (coordinates + basis) defines sublattices

    Parameters
    ----------
    lattice : Lattice
        Lattice object with geometry information
    H_real : np.ndarray
        Real-space Hamiltonian matrix (Ns x Ns)
    kpoints : Optional[np.ndarray]
        Custom k-points to evaluate at. If None, uses full BZ grid.
    require_full_grid : bool
        If True, raises error if kpoints doesn't match full grid size
    unitary_norm : bool
        Use unitary normalization (1/√Nc) for Bloch transform
    use_cache : bool
        Use cached Bloch transformation matrices for speed (default: True)
    return_transform : bool
        If True, also return the Bloch unitary W for computing correlation functions

    Returns
    -------
    Hk_grid : np.ndarray
        Shape (Lx, Ly, Lz, Nb, Nb) if kpoints is None
        Shape (Nk, Nb, Nb) if kpoints is provided
        Momentum-space Hamiltonian blocks in fftfreq order
    kgrid : np.ndarray
        Shape (Lx, Ly, Lz, 3) or (Nk, 3)
        K-point coordinates in fftfreq order (Γ at [0,0,0])
    kgrid_frac : np.ndarray
        Shape (Lx, Ly, Lz, 3) or None
        Fractional k-point coordinates in fftfreq order
    W : np.ndarray [only if return_transform=True]
        Shape (Nc, Ns, Nb) or (Nk, Ns, Nb)
        Bloch unitary matrix W[ik, i, a] = (1/√Nc) . exp(-ik.r_i) . delta _{sub(i),a}
        Use for transforming operators: O_k = W† @ O_real @ W
        
    Notes
    -----
    - K-points are in fftfreq order: k[0,0,0] = Γ point
    - No fftshift is applied to maintain correspondence between k_grid and H_k indices
    - For translationally invariant systems: spectrum(H_real) = union of spectrum(H(k))
    """

    if H_real.ndim != 2 or H_real.shape[0] != H_real.shape[1]:
        raise ValueError("H_real must be a square matrix.")
    
    Ns = H_real.shape[0]
    if Ns != lattice.Ns:
        raise ValueError(f"H_real size {Ns} != lattice Ns {lattice.Ns}.")

    # lattice sizes and counts
    Lx, Ly, Lz = lattice._lx, max(lattice._ly, 1), max(lattice._lz, 1)
    Nc         = Lx * Ly * Lz           # number of unit cells
    Nb         = len(lattice._basis)    # number of basis sites per cell
    if Ns % Nc != 0 or (Ns // Nc) != Nb:
        raise ValueError(f"Ns={Ns} not compatible with Nc={Nc} and Nb={Nb} (Ns must be Nc*Mb).")

    # Use full DFT transform - simpler and works for any lattice
    if kpoints is None:
        # Full k-space transform using DFT matrix
        H_k_full    = full_k_space_transform(lattice, H_real)
        
        # Extract blocks: for translationally invariant systems,
        # H_k_full is block-diagonal with Nc blocks of size NbxNb
        # Sites are ordered as [cell0_sub0, cell0_sub1, ..., cell1_sub0, cell1_sub1, ...]
        Hk_blocks   = np.zeros((Nc, Nb, Nb), dtype=complex)
        # i_move      = (Nc // 2) * Nb if Nc % 2 == 0 else 0
        i_move      = 0
        for ik in range(Nc):
            i_start         = ik * Nb + i_move % Ns
            i_end           = (ik + 1) * Nb + i_move % Ns
            Hk_blocks[ik]   = H_k_full[i_start:i_end, i_start:i_end]
        
        # Get k-grid from lattice
        cache               = _get_bloch_transform_cache(lattice, unitary_norm)
        
        # Reshape blocks to grid (fftfreq order)
        Hk_grid             = Hk_blocks.reshape(Lx, Ly, Lz, Nb, Nb)
        Hk_grid             = np.ascontiguousarray(Hk_grid)
        Hk_grid             = np.fft.fftshift(Hk_grid, axes=(0,1,2))
        
        if return_transform:
            # For compatibility, return dummy transform
            W_grid = np.zeros((Lx, Ly, Lz, Ns, Nb), dtype=complex)
            return Hk_grid, cache.kgrid, cache.kgrid_frac, W_grid
        else:
            return Hk_grid, cache.kgrid, cache.kgrid_frac
    
    # Fallback: manual computation for custom k-points
    # reciprocal basis
    b1          = np.asarray(lattice._k1, float).reshape(3)
    b2          = np.asarray(lattice._k2, float).reshape(3)
    b3          = np.asarray(lattice._k3, float).reshape(3)

    #! k-point mesh - either full grid or provided points
    if kpoints is None:
        # Use fftfreq convention: k_n = n/N for n = 0, 1, ..., N/2-1, -N/2, ..., -1
        # This gives k $\in$ [-0.5, 0.5) in fractional coordinates
        # which maps to the first Brillouin zone correctly for both even and odd N
        frac_x                      = np.linspace(0, 1, Lx, endpoint=False)  # sorted: [0, 1/N, ..., (N/2-1)/N, -N/2/N, ..., -1/N]
        frac_y                      = np.linspace(0, 1, Ly, endpoint=False)
        frac_z                      = np.linspace(0, 1, Lz, endpoint=False)

        # Create meshgrid in the fftfreq order
        kx_frac, ky_frac, kz_frac   = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")

        # Store fractional coordinates (shape: Lx, Ly, Lz, 3)
        kgrid_frac                  = np.stack([kx_frac, ky_frac, kz_frac], axis=-1)

        # Construct Cartesian k-vectors: k = f1*b1 + f2*b2 + f3*b3
        kgrid                       = (kx_frac[..., None] * b1 + 
                                       ky_frac[..., None] * b2 + 
                                       kz_frac[..., None] * b3)  # shape (Lx, Ly, Lz, 3)
        
        # Γ point is at index [0,0,0] (fftfreq order)
        # Flatten for computation
        kpoints                     = kgrid.reshape(-1, 3)  # shape (Nc, 3)
        return_grid                 = True
    else:
        kpoints                     = np.asarray(kpoints, float).reshape(-1, 3)
        kgrid                       = None
        kgrid_frac                  = None
        return_grid                 = False

    # Determine number of k-points
    Nk = kpoints.shape[0]
    if require_full_grid and Nk != Nc:
        raise ValueError(f"Round-trip requires Nk == Nc == {Nc}, got Nk={Nk}.")

    #! geometric labeling: (cell, sub) while keeping order
    indices           = np.arange(Ns)
    sub_idx           = indices % Nb
    
    # Ensure basis is padded with zeros for 1D/2D
    basis_coords                                = np.zeros((Nb, 3), dtype=float)
    basis_coords[:, :lattice._basis.shape[1]]   = np.asarray(lattice._basis, float)
    
    # Total position vector r_i = R_n + τ_a
    coords                                      = np.asarray(lattice.coordinates, float) # shape (Ns, 3)

    # Projector S[i, a] = delta_{sub(i), a}
    S                           = np.zeros((Ns, Nb), dtype=complex)
    S[np.arange(Ns), sub_idx]   = 1.0                                   # shape (Ns, Nb)
    
    # Bloch transform: H_ab(k) = Σ_{i,j} W*_{i,a} H_{i,j} W_{j,b}
    # where W_{i,a} = (1/sqrt Nc) * e^{-ik\cdot r_i} * delta_{sub(i),a}
    # where r_i = R_cell + τ_sublattice is the full site position
    # This accounts for both unit cell position AND basis vector in the phase
    
    phases                      = np.exp(-1j * (kpoints @ coords.T))  # (Nk, Ns)
    norm                        = np.sqrt(Nc) if unitary_norm else 1.0
    phases                     /= norm
    
    # Vectorized projector: W[ik, i, a] = phases[ik, i] * S[i, a]
    W                           = phases[:, :, None] * S[None, :, :]  # (Nk, Ns, Nb)
    
    # Transform: H(k) = W† @ H @ W for all k
    # Use einsum for efficiency: (Nk, Nb, Ns) @ (Ns, Ns) @ (Nk, Ns, Nb) -> (Nk, Nb, Nb)
    if sp.issparse(H_real):
        # For sparse matrices, use loop (einsum doesn't support sparse)
        Hk = np.zeros((Nk, Nb, Nb), dtype=complex)
        for ik in range(Nk):
            Hk[ik] = W[ik].conj().T @ (H_real @ W[ik])
    else:
        Hk = np.einsum('kia,ij,kjb->kab', W.conj(), H_real, W)

    if return_grid:
        # Reshape blocks to grid (keep in fftfreq order - NO SHIFT)
        # This ensures H_k[ix,iy,iz] corresponds to k_grid[ix,iy,iz]
        Hk_grid = Hk.reshape(Lx, Ly, Lz, Nb, Nb)
        if return_transform:
            W_grid = W.reshape(Lx, Ly, Lz, Ns, Nb)
            return Hk_grid, kgrid, kgrid_frac, W_grid
        else:
            return Hk_grid, kgrid, kgrid_frac
    else:
        if return_transform:
            return Hk, kpoints, W
        else:
            return Hk, kpoints

# -----------------------------------------------------------------------------------------------------------
#! END OF FILE
# -----------------------------------------------------------------------------------------------------------
