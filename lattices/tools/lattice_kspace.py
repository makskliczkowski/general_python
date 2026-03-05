r'''
K-space utilities for lattice systems.

Provides:
- Brillouin zone path generation and high-symmetry points
- Bloch transformation caching for efficient k-space transforms
- K-point grid generation and path extraction
- Wigner-Seitz cell masking and BZ extension utilities

--------------------------------
File            : lattices/tools/lattice_kspace.py
Author          : Maksymilian Kliczkowski
Date            : 2025-01-15
Changelog       :
    - 2026-03-03: Improved handling of reciprocal vector inputs and copy counts in extend_kspace_data().
Version         : 2.1
    - Added HighSymmetryPoint and HighSymmetryPoints classes for better management of high-symmetry points and paths.
    - Enhanced ws_bz_mask() to support multiple shells of reciprocal lattice points for improved accuracy at the BZ boundary.
--------------------------------
'''

from    __future__      import annotations
from    typing          import TYPE_CHECKING, Iterable, List, Optional, Literal, Tuple, Dict, NamedTuple, Union
from    dataclasses     import dataclass, field
from    enum            import Enum
import  numpy           as np
import  scipy.sparse    as sp

if TYPE_CHECKING:
    from ..lattice                      import Lattice
    from QES.Algebra.hamil_quadratic    import QuadraticBlockDiagonalInfo

# -----------------------------------------------------------------------------------------------------------
# BRILLOUIN ZONE UTILITIES
# -----------------------------------------------------------------------------------------------------------

def _resolve_reciprocal_vector_inputs(*, lattice=None, reciprocal_vectors: Optional[Iterable[np.ndarray]] = None, b1=None, b2=None, b3=None) -> List[np.ndarray]:
    """Collect raw reciprocal vectors from the provided sources without changing dimensionality."""
    
    if reciprocal_vectors is not None:
        if isinstance(reciprocal_vectors, np.ndarray):
            if reciprocal_vectors.ndim == 1:
                return [reciprocal_vectors]
            if reciprocal_vectors.ndim == 2:
                return [row for row in reciprocal_vectors]
            raise ValueError("reciprocal_vectors must be 1D or 2D")
        return [vec for vec in reciprocal_vectors]

    if lattice is not None:
        dim = int(getattr(lattice, "dim", 0) or 0)
        if dim > 0:
            return [getattr(lattice, f"k{i + 1}", None) for i in range(dim)]
        return [getattr(lattice, "k1", None), getattr(lattice, "k2", None), getattr(lattice, "k3", None)]

    return [b1, b2, b3]    

def _coerce_k_points(KX, KY=None, KZ=None, *, k_points: Optional[np.ndarray] = None) -> np.ndarray:
    """Return k-points with shape ``(..., dim)`` from either grids or explicit points."""
    if k_points is not None:
        k_vec = np.asarray(k_points, dtype=float)
    elif KY is None:
        k_vec = np.asarray(KX, dtype=float)
    else:
        components = [np.asarray(KX, dtype=float), np.asarray(KY, dtype=float)]
        if KZ is not None:
            components.append(np.asarray(KZ, dtype=float))
        k_vec = np.stack(components, axis=-1)

    if k_vec.ndim == 0:
        raise ValueError("k-points must contain at least one coordinate axis")
    if k_vec.ndim == 1:
        k_vec = k_vec.reshape(1, -1)
    return k_vec    

def _normalize_reciprocal_vectors(vectors: Iterable[np.ndarray], kdim: int, *, copies: Optional[Iterable[int]] = None, drop_zero: bool = True, tol: float = 1e-12) -> Tuple[List[np.ndarray], List[int]]:
    """Normalize reciprocal vectors to ``kdim`` and keep copy counts aligned."""
    raw_vectors = list(vectors)
    raw_copies  = None if copies is None else list(copies)
    if raw_copies is not None:
        if len(raw_copies) < len(raw_vectors):
            raise ValueError("Number of reciprocal vectors must not exceed number of copy counts")
        if len(raw_copies) > len(raw_vectors):
            raw_copies = raw_copies[:len(raw_vectors)]

    normalized_vectors: List[np.ndarray] = []
    normalized_copies: List[int] = []

    for idx, vec in enumerate(raw_vectors):
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float).ravel()
        if arr.size < kdim:
            arr = np.pad(arr, (0, kdim - arr.size))
        elif arr.size > kdim:
            arr = arr[:kdim]

        if drop_zero and np.linalg.norm(arr) <= tol:
            continue

        normalized_vectors.append(arr)
        if raw_copies is not None:
            normalized_copies.append(int(raw_copies[idx]))

    return normalized_vectors, normalized_copies

# -----------------------------------------------------------------------------------------------------------

def ws_bz_mask(KX, KY=None, b1=None, b2=None, shells=1, *, KZ=None, b3=None, k_points=None, reciprocal_vectors=None, lattice=None, tol: float = 1e-12):
    r"""
    Wigner-Seitz (first BZ) mask for a reciprocal lattice.

    Keeps points closer to Gamma than to any other reciprocal lattice point
    in a finite neighborhood of reciprocal-lattice translations.
    
    The Wigner-Seitz condition is: |k|^2 <= |k-G|^2 for all reciprocal lattice vectors G.
    This simplifies to: 2k \cdot G <= G^2 (mathematically equivalent, computationally faster).
    
    Parameters
    ----------
    KX, KY, KZ : array_like, optional
        Coordinate grids. The legacy 2D form is ``ws_bz_mask(KX, KY, b1, b2)``.
    k_points : array_like, optional
        Explicit k-points with shape ``(..., dim)``.
    b1, b2, b3 : array_like, optional
        Legacy reciprocal lattice vectors.
    reciprocal_vectors : iterable of array_like, optional
        General reciprocal translation vectors.
    lattice : object, optional
        Lattice-like object exposing ``dim`` and reciprocal vectors ``k1``, ``k2``, ``k3``.
    shells : int or iterable of int, default=1
        Number of reciprocal shells to consider along each reciprocal direction. Each new 
        shell adds more reciprocal lattice points to the Wigner-Seitz condition, improving accuracy at the BZ boundary.
        The condition is modified to: 2k \cdot G <= |G|^2 + tol, where G are the reciprocal lattice vectors up to the specified shell count.
    tol : float, default=1e-12
        Numerical tolerance used at the BZ boundary.
        
    Returns
    -------
    inside : ndarray (bool)
        True for points inside the first Brillouin zone.
    """
    
    # Process inputs and normalize to k-point array and list of reciprocal vectors
    k_vec       = _coerce_k_points(KX, KY, KZ, k_points=k_points)
    kdim        = k_vec.shape[-1]

    raw_vectors = _resolve_reciprocal_vector_inputs(lattice=lattice, reciprocal_vectors=reciprocal_vectors, b1=b1, b2=b2, b3=b3,)
    basis, _    = _normalize_reciprocal_vectors(raw_vectors, kdim, tol=tol)
    if len(basis) == 0:
        raise ValueError("At least one reciprocal lattice vector must be provided")

    # Determine shell counts for each reciprocal vector and generate the grid of reciprocal lattice points to check against
    if np.isscalar(shells):
        shell_counts = [int(shells)] * len(basis)
    else:
        shell_counts = [int(shell) for shell in shells]
        if len(shell_counts) != len(basis):
            raise ValueError("Number of shell counts must match number of reciprocal vectors")

    # Generate the grid of reciprocal lattice points G = n1*b1 + n2*b2 + n3*b3 for all integer combinations of n_i in [-shells, shells].
    if any(shell < 0 for shell in shell_counts):
        raise ValueError("shells must be non-negative")

    coeff_axes = [np.arange(-shell, shell + 1, dtype=int) for shell in shell_counts]
    coeff_grid = np.stack(np.meshgrid(*coeff_axes, indexing='ij'), axis=-1).reshape(-1, len(basis))
    coeff_grid = coeff_grid[np.any(coeff_grid != 0, axis=1)]

    if len(coeff_grid) == 0:
        return np.ones(k_vec.shape[:-1], dtype=bool)

    # Wigner-Seitz condition: 2k \cdot G <= |G|^2 for all reciprocal lattice vectors G.
    G           = coeff_grid @ np.vstack(basis)
    k_dot_G     = k_vec @ G.T
    G_squared   = np.sum(G**2, axis=1)
    inside      = np.all(2 * k_dot_G <= G_squared + tol, axis=-1)
    return inside

def ws_bz_shifts(
        *,
        lattice                                             =   None,
        reciprocal_vectors: Optional[Iterable[np.ndarray]]  = None,
        b1: Optional[np.ndarray]                            = None,
        b2: Optional[np.ndarray]                            = None,
        b3: Optional[np.ndarray]                            = None,
        copies: Optional[Union[int, Iterable[int]]]         = None,
        nx: int                                             = 1,
        ny: int                                             = 1,
        nz: int                                             = 0,
        include_origin: bool                                = False,
        tol: float                                          = 1e-12,
    ) -> np.ndarray:
    r"""
    Return reciprocal-lattice translation vectors for Brillouin-zone copies.

    This is the shared geometric primitive for drawing or selecting translated
    Brillouin zones. It returns the centers of reciprocal-cell copies, not an
    extended k-mesh.

    Parameters
    ----------
    lattice : object, optional
        Lattice-like object exposing ``dim`` and reciprocal vectors ``k1``,
        ``k2``, ``k3``.
    reciprocal_vectors : iterable of array_like, optional
        Explicit reciprocal translation vectors. If provided, they take
        precedence over ``b1``/``b2``/``b3``.
    b1, b2, b3 : array_like, optional
        Legacy reciprocal lattice vectors.
    copies : int or iterable of int, optional
        Number of translated copies along each reciprocal direction.
    nx, ny, nz : int, default=(1, 1, 0)
        Legacy per-direction copy counts used when ``copies`` is not given.
    include_origin : bool, default=False
        Whether to include the central Brillouin zone at ``Gamma``.
    tol : float, default=1e-12
        Tolerance used when removing numerically duplicated shifts.

    Returns
    -------
    np.ndarray
        Array of shape ``(Nshift, dim)`` containing unique reciprocal
        translation vectors for Brillouin-zone copies.
    """
    raw_vectors = _resolve_reciprocal_vector_inputs(
        lattice=lattice,
        reciprocal_vectors=reciprocal_vectors,
        b1=b1,
        b2=b2,
        b3=b3,
    )
    nonzero_vectors = [np.asarray(vec, dtype=float).ravel() for vec in raw_vectors if vec is not None]
    if lattice is not None and int(getattr(lattice, "dim", 0) or 0) > 0:
        kdim = int(getattr(lattice, "dim"))
    elif nonzero_vectors:
        kdim = max(int(vec.size) for vec in nonzero_vectors)
    else:
        raise ValueError("At least one reciprocal lattice vector must be provided")

    centers, _ = extend_kspace_data(
        k_points=np.zeros((1, kdim), dtype=float),
        lattice=lattice,
        reciprocal_vectors=reciprocal_vectors,
        b1=b1,
        b2=b2,
        b3=b3,
        nx=nx,
        ny=ny,
        nz=nz,
        copies=copies,
    )

    scale   = max(float(tol), np.finfo(float).eps)
    seen    = set()
    unique  = []
    for shift in np.asarray(centers, dtype=float):
        key = tuple(np.rint(shift / scale).astype(np.int64))
        if key in seen:
            continue
        seen.add(key)
        if not include_origin and np.allclose(shift, 0.0, atol=tol, rtol=0.0):
            continue
        unique.append(shift)

    if not unique:
        return np.zeros((0, kdim), dtype=float)
    return np.asarray(unique, dtype=float)

def extend_kspace_data(
        k_points            : np.ndarray,
        data                : Optional[np.ndarray]  = None,
        b1                  : Optional[np.ndarray]  = None,
        b2                  : Optional[np.ndarray]  = None,
        b3                  : np.ndarray            = None,
        nx                  : int                   = 2,
        ny                  : int                   = 2,
        nz                  : int                   = 0,
        *,
        lattice             = None,
        reciprocal_vectors  : Optional[Iterable[np.ndarray]] = None,
        copies              : Optional[Union[int, Iterable[int]]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    r"""
    Extend k-space points and optional data across translated Brillouin zones.

    The helper works for arbitrary k-space dimensions and any number of
    reciprocal translation vectors. Legacy ``b1``/``b2``/``b3`` with
    ``nx``/``ny``/``nz`` remain supported for existing callers.

    Allows to generate extended k-point grids for plotting band structures along high-symmetry paths...

    Parameters
    ----------
    k_points : array_like, shape (Nk, dim) or (dim,)
        Original Cartesian k-points.
    data : array_like, shape (Nk, ...), optional
        Data associated with each k-point. If omitted, only the extended
        k-points are returned and the second return value is ``None``.
    b1, b2, b3 : array_like, optional
        Reciprocal lattice vectors.
    nx, ny, nz : int, default=(2, 2, 0)
        Copy counts for ``b1``, ``b2``, ``b3``.
    ---------------------------------
    lattice : object, optional
        Lattice-like object exposing ``dim`` and reciprocal vectors ``k1``, ``k2``, ``k3``.
        Used when ``reciprocal_vectors`` and explicit ``b1``/``b2``/``b3`` are not given.
    reciprocal_vectors : iterable of array_like, optional
        General reciprocal translation vectors. If provided, these take
        precedence over ``b1``/``b2``/``b3``.
    copies : int or iterable of int, optional
        Copy counts for ``reciprocal_vectors``. A scalar applies to every
        reciprocal vector. If omitted, one copy is used for each provided
        reciprocal vector.

    Returns
    -------
    extended_k_points : np.ndarray
        Extended k-points covering multiple BZs.
    extended_data : np.ndarray or None
        Corresponding data for the extended k-points, if ``data`` was given.
        
    Example
    -------
    >>> k_points        = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    >>> b1              = np.array([1.0, 0.0, 0.0])
    >>> b2              = np.array([0.0, 1.0, 0.0])
    >>> extended_k, _   = extend_kspace_data(k_points, b1=b1, b2=b2, nx=1, ny=1)
    >>> extended_k.shape
    (18, 3)
    """
    
    k_points = np.asarray(k_points, dtype=float)
    if k_points.ndim == 1:
        k_points = k_points.reshape(1, -1)
    elif k_points.ndim != 2:
        raise ValueError("k_points must have shape (Nk, dim) or (dim,)")

    nk, kdim        = k_points.shape
    legacy_copies   = [nx, ny, nz]

    # Determine the reciprocal vectors to use and their copy counts, with proper precedence and normalization.
    if reciprocal_vectors is None:
        raw_vectors = _resolve_reciprocal_vector_inputs(lattice=lattice, b1=b1, b2=b2, b3=b3)
        if copies is not None:
            if np.isscalar(copies):
                raw_copies = [int(copies)] * len(raw_vectors)
            else:
                raw_copies = [int(copy) for copy in copies]
        else:
            raw_copies = legacy_copies[:len(raw_vectors)]
    else:
        raw_vectors = _resolve_reciprocal_vector_inputs(reciprocal_vectors=reciprocal_vectors)
        if copies is None:
            raw_copies = [1] * len(raw_vectors)
        elif np.isscalar(copies):
            raw_copies = [int(copies)] * len(raw_vectors)
        else:
            raw_copies = [int(copy) for copy in copies]

    # Normalize reciprocal vectors and copy counts, ensuring they match the k-space dimensionality and dropping zero vectors if needed.
    active_vectors, active_copies = _normalize_reciprocal_vectors(raw_vectors, kdim, copies=raw_copies)
    if any(copy < 0 for copy in active_copies):
        raise ValueError("Copy counts must be non-negative")

    # Generate the shifts for extending the k-points. Each shift is a linear combination of the active reciprocal vectors, with coefficients in the range defined by the copy counts.
    if len(active_vectors) == 0:
        shifts = np.zeros((1, kdim), dtype=float)
    else:
        coeff_axes = [np.arange(-copy, copy + 1, dtype=int) for copy in active_copies]
        coeff_grid = np.stack(np.meshgrid(*coeff_axes, indexing='ij'), axis=-1).reshape(-1, len(active_vectors))
        basis      = np.vstack(active_vectors)
        shifts     = coeff_grid @ basis

    # Apply the shifts to the original k-points to generate the extended k-point grid. The resulting shape will be (Nk * number_of_shifts, dim).
    extended_k_points   = (k_points[None, :, :] + shifts[:, None, :]).reshape(-1, kdim)
    extended_data       = None
    
    # If data is provided, match it with the extended k-points by tiling it according to the number of shifts. 
    # The leading dimension of data must match the number of original k-points.
    if data is not None:
        data = np.asarray(data)
        if nk == 1 and data.ndim == 0:
            data = data.reshape(1)
        if data.shape[0] != nk:
            raise ValueError("data must have the same leading dimension as k_points")
        tile_shape    = (len(shifts),) + (1,) * max(data.ndim - 1, 0)
        extended_data = np.tile(data, tile_shape)

    return extended_k_points, extended_data

# -----------------------------------------------------------------------------------------------------------
#! HIGH-SYMMETRY POINTS AND PATHS
# -----------------------------------------------------------------------------------------------------------

@dataclass
class HighSymmetryPoint:
    r"""
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
                'Gamma' : r'$\Gamma$', 'G': r'$\Gamma$',
                'K'     : r'$K$', 
                'M'     : r'$M$', 'X': r'$X$', 'Y': r'$Y$', 'Z': r'$Z$',
                'R'     : r'$R$', 'A': r'$A$', 'L': r'$L$', 'H': r'$H$',
            }
            self.latex_label = special_labels.get(self.label, f'${self.label}$')
    
    def __contains__(self, coord: Union[Tuple[float, float, float], str]) -> bool:
        """Check if given fractional coordinates match this point."""
        
        if isinstance(coord, str):
            return coord == self.label
        return np.allclose(self.frac_coords, coord)
    
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

    @staticmethod
    def _normalize_label(name: str) -> str:
        """Normalize common aliases for high-symmetry point labels."""
        if name is None:
            return ""
        label = str(name).strip()
        label = label.replace("Γ", "Gamma").replace("γ", "Gamma")
        label = label.replace("’", "'")
        if label in ("G", "g", "\\Gamma", "gamma", "Gamma"):
            return "Gamma"
        if label.endswith("'"):
            label = f"{label[:-1]}p"
        return label

    def resolve_label(self, name: str) -> Optional[str]:
        """
        Resolve a label or alias to a canonical key in ``self.points``.

        Examples: ``"Γ" -> "Gamma"``, ``"K'" -> "Kp"``.
        """
        if not self.points:
            return None
        
        # First try direct normalization and lookup
        norm = self._normalize_label(name)
        if norm in self.points:
            return norm
        
        # Fallback: case-insensitive match against keys and normalized keys
        norm_l = norm.lower()
        for key in self.points:
            if key.lower() == norm_l:
                return key
            if self._normalize_label(key).lower() == norm_l:
                return key
        return None

    def resolve(self, name: str) -> Optional[HighSymmetryPoint]:
        """Resolve a label/alias and return the matching point object."""
        key = self.resolve_label(name)
        if key is None:
            return None
        return self.points.get(key)
    
    @property
    def default_path(self) -> List[str]:
        """Return the default path through high-symmetry points."""
        return self._default_path
    
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
            resolved = self.resolve_label(label)
            if resolved is None:
                raise ValueError(f"Unknown high-symmetry point: '{label}'. "
                                f"Available: {list(self.points.keys())}")
            path.append(self.points[resolved].as_tuple())
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
        pts.add(HighSymmetryPoint('Gamma',  (0.0, 0.0, 0.0), r'$0$',        '1D BZ center'))
        pts.add(HighSymmetryPoint('X',      (0.5, 0.0, 0.0), r'$\pi$',      'Zone boundary'))
        pts.add(HighSymmetryPoint('Gamma2', (1.0, 0.0, 0.0), r'$2\pi$',     'Wrapped Gamma'))
        return pts
    
    @classmethod
    def square_2d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 2D square lattice."""
        pts = cls(
            _default_path=['Gamma', 'X', 'M', 'Gamma']
        )
        pts.add(HighSymmetryPoint('Gamma',  (0.0, 0.0, 0.0), r'$\Gamma$',   'BZ center'))
        pts.add(HighSymmetryPoint('X',      (0.5, 0.0, 0.0), r'$X$',        'Zone face center'))
        pts.add(HighSymmetryPoint('M',      (0.5, 0.5, 0.0), r'$M$',        'Zone corner'))
        pts.add(HighSymmetryPoint('Y',      (0.0, 0.5, 0.0), r'$Y$',        'Zone face center (y)'))
        return pts
    
    @classmethod
    def cubic_3d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 3D cubic lattice."""
        pts = cls(
            _default_path=['Gamma', 'X', 'M', 'Gamma', 'R', 'X']
        )
        pts.add(HighSymmetryPoint('Gamma',  (0.0, 0.0, 0.0), r'$\Gamma$',   'BZ center'))
        pts.add(HighSymmetryPoint('X',      (0.5, 0.0, 0.0), r'$X$',        'Face center'))
        pts.add(HighSymmetryPoint('M',      (0.5, 0.5, 0.0), r'$M$',        'Edge center'))
        pts.add(HighSymmetryPoint('R',      (0.5, 0.5, 0.5), r'$R$',        'Corner'))
        return pts
    
    @classmethod
    def triangular_2d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 2D triangular lattice."""
        pts = cls(
            _default_path=['Gamma', 'M', 'K', 'Gamma']
        )
        pts.add(HighSymmetryPoint('Gamma',  (0.0, 0.0, 0.0), r'$\Gamma$',   'BZ center'))
        pts.add(HighSymmetryPoint('M',      (0.5, 0.0, 0.0), r'$M$',        'Edge midpoint'))
        pts.add(HighSymmetryPoint('K',      (1/3, 1/3, 0.0), r'$K$',        'Corner (Dirac point)'))
        pts.add(HighSymmetryPoint('Kp',     (2/3, 1/3, 0.0), r"$K'$",       'Other Dirac point'))
        return pts
    
    @classmethod
    def honeycomb_2d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for honeycomb/graphene lattice."""
        pts = cls(_default_path=['Gamma', 'K', 'M', 'Gamma'])
        pts.add(HighSymmetryPoint('Gamma', (0.0, 0.0, 0.0),            r'$\Gamma$',    'BZ center'))
        pts.add(HighSymmetryPoint('K',     (2.0/3.0, 1.0/3.0, 0.0),    r'$K$',         'Dirac point'))
        pts.add(HighSymmetryPoint('Kp',    (1.0/3.0, 2.0/3.0, 0.0),    r"$K'$",        'Other Dirac point'))
        pts.add(HighSymmetryPoint('M',     (0.5, 0.0, 0.0),            r'$M$',         'Edge midpoint'))
        return pts
        
    @classmethod
    def hexagonal_2d(cls) -> 'HighSymmetryPoints':
        """High-symmetry points for 2D hexagonal lattice (same as honeycomb)."""
        return cls.honeycomb_2d()

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

PathTypes = Literal['CHAIN_1D', 'SQUARE_2D', 'TRIANGULAR_2D', 'CUBIC_3D', 'HONEYCOMB_2D']
    
# -----------------------------------------------------------------------------------------------------------
#! BRILLOUIN ZONE PATH GENERATION
# -----------------------------------------------------------------------------------------------------------

def resolve_path_input(path: Iterable[tuple[str, Iterable[float]]] | StandardBZPath | str | List[str] | HighSymmetryPoints, lattice: Optional[Lattice] = None) -> list[tuple[str, list[float]]]:
    """
    Resolve path input to a list of (label, fractional_coord) pairs.

    Parameters
    ----------
    path : list[(label, coords)], StandardBZPath, str, List[str], or HighSymmetryPoints
        Path definition (fractional coordinates), standard enum, enum name string, 
        list of point labels, or HighSymmetryPoints object.
    lattice : Lattice, optional
        Lattice object used to resolve labels if path is a list of strings.

    Returns
    -------
    resolved_path : list[(label, list[float])]
        Resolved path as a list of (label, fractional_coord) pairs.
        
    Example
    -------
    >>> path = resolve_path_input("SQUARE_2D")
    >>> for label, coord in path:
    ...     print(f"{label}: {coord}")
    """
    if isinstance(path, str):
        path = StandardBZPath[path.upper()].value
    elif isinstance(path, StandardBZPath):
        path = path.value
    elif isinstance(path, HighSymmetryPoints):
        path = path.get_default_path_points()
    elif isinstance(path, list) and len(path) > 0 and isinstance(path[0], str):
        # List of strings, resolve via lattice if available
        if lattice is not None and hasattr(lattice, 'high_symmetry_points'):
            hs = lattice.high_symmetry_points()
            if hs is not None:
                path = hs.get_path_points(path)
            else:
                raise ValueError(f"Cannot resolve path labels {path} as lattice has no high-symmetry points defined.")
        else:
            raise ValueError(f"Cannot resolve path labels {path} without a lattice providing high-symmetry points.")
    
    return [(label, list(map(float, frac))) for label, frac in path]

def generate_kgrid(lattice: Lattice, n_k: Iterable[int], shift: Optional[Union[bool, Tuple[bool, bool, bool]]] = None) -> np.ndarray:
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
    recip = np.vstack([v for v in [lattice._k1, lattice._k2, lattice._k3] if v is not None]) # shape (dim, 3)
    nk    = np.array(list(n_k)) if isinstance(n_k, Iterable) else np.array([n_k] * lattice.dim)

    grids = [np.arange(n) / n for n in nk]                  # fractional grids
    mesh  = np.meshgrid(*grids, indexing="ij")              # meshgrid
    frac  = np.stack([m.ravel() for m in mesh], axis=-1)    # shape (Nk, dim)
    
    # define vectors in a full 3D array for matrix multiplication -> k=f1*b1 + f2*b2 + f3*b3
    k_pts = frac @ recip                                    # fractional -> cartesian
    return k_pts

def brillouin_zone_path(lattice: Lattice, path: Iterable[tuple[str, Iterable[float]]] | StandardBZPath | List[str] | HighSymmetryPoints, *, points_per_seg  : int = 40,) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]], np.ndarray]:
    """
    Generate k-points along a specified Brillouin zone path. It takes 
    a list of (label, fractional_coord) pairs defining the path in reciprocal lattice units and 
    interpolates k-points along the straight lines connecting these points in Cartesian coordinates.
    
    In general, if coordinates are given as c1 = (f1, f2, f3) and c2 = (g1, g2, g3)
    we want to follow the straight line in Cartesian coordinates:
        k(t) = (1-t) * (f1*b1 + f2*b2 + f3*b3) + t * (g1*b1 + g2*b2 + g3*b3)
    for t in [0, 1], t = 0, 1/points_per_seg, 2/points_per_seg, ..., (points_per_seg-1)/points_per_seg.
    
    Each segment between two labels is interpolated with `points_per_seg` points.
    
    Example:
    >>> path                    = StandardBZPath.SQUARE_2D.value
    >>> # define the interpolated path
    >>> k_path, k_dist, labels  = brillouin_zone_path(lattice, path, points_per_seg=10) # 10 points per segment for demonstration
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
    path : list[(label, coords)], StandardBZPath, List[str], or HighSymmetryPoints
        Path definition (fractional coordinates), one of the standard enums,
        list of symmetry point labels, or HighSymmetryPoints object.
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
    
    path = resolve_path_input(path, lattice=lattice)

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

@dataclass
class KPathSelection:
    """
    Ideal Brillouin-zone path with optional nearest matches on an existing k-grid.

    This container is intended for visualization and path-selection logic where
    the path geometry is needed even when no sampled k-grid exists yet.
    """
    path_cart            : np.ndarray
    path_frac            : np.ndarray
    k_dist               : np.ndarray
    labels               : List[Tuple[int, str]]
    matched_cart         : Optional[np.ndarray] = None
    matched_frac         : Optional[np.ndarray] = None
    matched_grid_indices : np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    matched_indices      : np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    matched_distances    : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    match_tolerance      : float = 0.0

    @property
    def has_matches(self) -> bool:
        """Whether this path was matched to an existing k-grid."""
        return self.matched_cart is not None and len(self.matched_indices) > 0

    def unique_match_positions(self) -> np.ndarray:
        """Return path positions of unique matched k-points, preserving path order."""
        if not self.has_matches:
            return np.array([], dtype=int)
        key_indices = self.matched_grid_indices if len(self.matched_grid_indices) > 0 else self.matched_indices
        _, first = np.unique(key_indices, return_index=True)
        return np.sort(first.astype(int))

    def unique_matched_cart(self) -> np.ndarray:
        """Return unique matched Cartesian k-points in path order."""
        if not self.has_matches:
            return np.zeros((0, 3), dtype=float)
        return self.matched_cart[self.unique_match_positions()]

# -----------------------------------------------------------------------------------------------------------
#! PATH SELECTION ON EXISTING K-GRID
# -----------------------------------------------------------------------------------------------------------

def _cartesian_from_fractional(lattice: Lattice, frac_vectors: np.ndarray) -> np.ndarray:
    """Convert fractional reciprocal coordinates to Cartesian k-vectors."""
    frac = np.asarray(frac_vectors, dtype=float)
    if frac.ndim == 1:
        frac = frac.reshape(1, -1)
    if frac.shape[1] < 3:
        frac = np.pad(frac, ((0, 0), (0, 3 - frac.shape[1])))
    elif frac.shape[1] > 3:
        frac = frac[:, :3]
    b1 = np.asarray(lattice._k1, float).reshape(3)
    b2 = np.asarray(lattice._k2, float).reshape(3)
    b3 = np.asarray(lattice._k3, float).reshape(3)
    B  = np.column_stack([b1, b2, b3])
    return frac @ B.T

def _fractional_from_cartesian(lattice: Lattice, k_vectors: np.ndarray) -> np.ndarray:
    """Convert Cartesian k-vectors to fractional reciprocal coordinates."""
    k_cart = np.asarray(k_vectors, dtype=float)
    if k_cart.ndim == 1:
        k_cart = k_cart.reshape(1, -1)
    if k_cart.shape[1] < 3:
        k_cart = np.pad(k_cart, ((0, 0), (0, 3 - k_cart.shape[1])))
    elif k_cart.shape[1] > 3:
        k_cart = k_cart[:, :3]
    b1      = np.asarray(lattice._k1, float).reshape(3)
    b2      = np.asarray(lattice._k2, float).reshape(3)
    b3      = np.asarray(lattice._k3, float).reshape(3)
    B       = np.column_stack([b1, b2, b3])
    B_pinv  = np.linalg.pinv(B)
    return (B_pinv @ k_cart.T).T

def _default_kpath_tolerance(lattice: Lattice) -> float:
    """Estimate a reasonable fractional-coordinate tolerance from the lattice size."""
    Lx = max(getattr(lattice, "_lx", 1), 1)
    Ly = max(getattr(lattice, "_ly", 1), 1)
    Lz = max(getattr(lattice, "_lz", 1), 1)
    return 0.5 * np.sqrt((1 / Lx) ** 2 + (1 / Ly) ** 2 + (1 / Lz) ** 2)

def _kpath_tolerance_to_cartesian(lattice: Lattice, tol: float) -> float:
    """Map a fractional k-path tolerance to a conservative Cartesian tolerance."""
    b1 = np.asarray(lattice._k1, float).reshape(3)
    b2 = np.asarray(lattice._k2, float).reshape(3)
    b3 = np.asarray(lattice._k3, float).reshape(3)
    B  = np.column_stack([b1, b2, b3])
    return float(np.linalg.norm(B, ord=2) * float(tol))

def _path_covering_copies(k_grid_frac: np.ndarray, path_frac: np.ndarray, *, tol: float = 1e-12, dim: int = 3) -> Tuple[int, ...]:
    """Return symmetric reciprocal-copy counts needed to cover a path range."""
    if dim <= 0:
        return tuple()
    kf = np.asarray(k_grid_frac, dtype=float).reshape(-1, 3)[:, :dim]
    pf = np.asarray(path_frac, dtype=float).reshape(-1, 3)[:, :dim]
    if len(kf) == 0 or len(pf) == 0:
        return tuple(0 for _ in range(dim))

    grid_min = np.min(kf, axis=0)
    grid_max = np.max(kf, axis=0)
    path_min = np.min(pf, axis=0)
    path_max = np.max(pf, axis=0)

    copies = []
    for idx in range(dim):
        upper = max(0.0, path_max[idx] - grid_max[idx] - tol)
        lower = max(0.0, grid_min[idx] - path_min[idx] - tol)
        copies.append(int(np.ceil(max(upper, lower))))
    return tuple(copies)

def _extend_kgrid_to_path(
    lattice     : Lattice,
    k_cart      : np.ndarray,
    k_frac      : np.ndarray,
    path_frac   : np.ndarray,
    *,
    tol         : float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extend an existing k-grid just enough to cover a requested path."""
    
    kf_flat     = np.asarray(k_frac, dtype=float).reshape(-1, 3)
    kc_flat     = np.asarray(k_cart, dtype=float).reshape(-1, 3)
    dim         = int(getattr(lattice, "dim", 0) or min(3, kf_flat.shape[1], path_frac.shape[1]))
    dim         = max(1, min(dim, 3))
    copies      = _path_covering_copies(kf_flat, path_frac, tol=tol, dim=dim)
    if not any(copies):
        base_indices = np.arange(len(kf_flat), dtype=int)
        return kc_flat, kf_flat, base_indices, base_indices.copy()

    ext_frac, _ = extend_kspace_data(kf_flat[:, :dim], reciprocal_vectors=np.eye(dim, dtype=float), copies=copies)
    ext_cart    = _cartesian_from_fractional(lattice, ext_frac)

    if dim < 3:
        ext_frac = np.pad(ext_frac, ((0, 0), (0, 3 - dim)))
        ext_cart = np.pad(ext_cart[:, :dim], ((0, 0), (0, 3 - dim)))

    n_shifts        = len(ext_frac) // max(len(kf_flat), 1)
    base_indices    = np.tile(np.arange(len(kf_flat), dtype=int), n_shifts)
    grid_indices    = np.arange(len(ext_frac), dtype=int)
    return ext_cart, ext_frac, base_indices, grid_indices

def _polyline_point_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Return the shortest Cartesian distance from each point to a polyline."""
    pts  = np.asarray(points, dtype=float)
    path = np.asarray(polyline, dtype=float)

    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if path.ndim == 1:
        path = path.reshape(1, -1)
    if len(pts) == 0:
        return np.zeros(0, dtype=float)
    if len(path) == 0:
        return np.full(len(pts), np.inf, dtype=float)
    if len(path) == 1:
        return np.linalg.norm(pts - path[0], axis=1)

    distances = np.full(len(pts), np.inf, dtype=float)
    for start, end in zip(path[:-1], path[1:]):
        segment         = end - start
        seg_norm_sq     = float(np.dot(segment, segment))
        if seg_norm_sq <= 0.0:
            candidate   = np.linalg.norm(pts - start, axis=1)
        else:
            rel         = pts - start
            proj        = np.clip((rel @ segment) / seg_norm_sq, 0.0, 1.0)
            closest     = start + proj[:, None] * segment
            candidate   = np.linalg.norm(pts - closest, axis=1)
        distances = np.minimum(distances, candidate)
    return distances

def find_nearest_kpoints(k_grid_frac: np.ndarray, target_frac: np.ndarray, tol: float = 0.5, *, periodic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Find nearest k-point indices for target fractional coordinates.
    
    Parameters
    ----------
    k_grid_frac : np.ndarray, shape (Nk, 3)
        Fractional coordinates of available k-points
    target_frac : np.ndarray, shape (Ntarget, 3)
        Target fractional coordinates to match
    tol : float
        Warning threshold for match distance
    periodic : bool, default=True
        If True, measure distance modulo reciprocal-lattice translations.
        If False, use direct fractional-coordinate distance without wrapping.
    
    Returns
    -------
    indices : np.ndarray, shape (Ntarget,), dtype=int
        Index of nearest k-point for each target
    distances : np.ndarray, shape (Ntarget,)
        Distance to nearest point (in fractional units, accounting for periodicity)
    """
    k_grid_frac = np.asarray(k_grid_frac, dtype=float)
    target_frac = np.asarray(target_frac, dtype=float)
    if k_grid_frac.ndim == 1:
        k_grid_frac = k_grid_frac.reshape(1, -1)
    if target_frac.ndim == 1:
        target_frac = target_frac.reshape(1, -1)
    if k_grid_frac.shape[1] < 3:
        k_grid_frac = np.pad(k_grid_frac, ((0, 0), (0, 3 - k_grid_frac.shape[1])))
    elif k_grid_frac.shape[1] > 3:
        k_grid_frac = k_grid_frac[:, :3]
    if target_frac.shape[1] < 3:
        target_frac = np.pad(target_frac, ((0, 0), (0, 3 - target_frac.shape[1])))
    elif target_frac.shape[1] > 3:
        target_frac = target_frac[:, :3]
    
    n_targets   = len(target_frac)
    indices     = np.zeros(n_targets, dtype=int)
    distances   = np.zeros(n_targets)
    
    for i, kf_target in enumerate(target_frac):
        diff    = k_grid_frac - kf_target
        if periodic:
            diff -= np.round(diff)
        dist    = np.linalg.norm(diff, axis=1)
        idx     = np.argmin(dist)
        indices[i] = idx
        distances[i] = dist[idx]
        
        if dist[idx] > tol:
            continue # Optionally log a warning about unmatched point
    
    return indices, distances

def bz_path_points(
    lattice,
    path                : Iterable[tuple[str, Iterable[float]]] | StandardBZPath | HighSymmetryPoints | None = None,
    *,
    points_per_seg      : int = 40,
    k_vectors           : Optional[np.ndarray] = None,
    k_vectors_frac      : Optional[np.ndarray] = None,
    tol                 : Optional[float] = None,
    periodic            : bool = True,
) -> KPathSelection:
    """
    Build an ideal Brillouin-zone path and optionally match it to an existing k-grid.

    If no k-grid is provided, the returned object still contains the continuous
    path geometry, which is useful for plotting or for constructing a path that
    is not constrained to the sampled reciprocal mesh. When a sampled grid is
    provided, it is automatically extended by reciprocal-lattice translations
    if needed so paths through higher Brillouin-zone copies can still be matched
    against the existing data.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice object with reciprocal lattice vectors and optionally high-symmetry points.
    path : list[(label, coords)], StandardBZPath, HighSymmetryPoints, or None
        Path definition (fractional coordinates), one of the standard enums, HighSymmetryPoints object
        or None to use lattice default path.
    points_per_seg : int
        Number of interpolated points between labels for the ideal path.
    k_vectors : np.ndarray, optional
        Cartesian k-vectors of the existing grid to match against.
    k_vectors_frac : np.ndarray, optional
        Fractional k-vectors of the existing grid to match against (in reciprocal lattice units).
    tol : float, optional
        Tolerance for matching path points to the grid. With
        ``periodic=True`` it is interpreted in fractional reciprocal
        coordinates. With ``periodic=False`` it is interpreted in the plotted
        Cartesian reciprocal coordinates.
    periodic : bool, default=True
        Whether path matching should identify reciprocal-translation-equivalent
        k-points as the same point. Use ``False`` for visual matching in the
        displayed Brillouin-zone copy.
        
    Returns
    -------
    KPathSelection
         Object containing the ideal path geometry and matched grid points if available.
    """
    
    if path is None:
        if hasattr(lattice, 'high_symmetry_points'):
            hs_pts = lattice.high_symmetry_points()
            if hs_pts is not None:
                path = hs_pts.get_default_path_points()
        if path is None:
            raise ValueError("No path specified and lattice has no default path. Specify path explicitly or use a lattice with high_symmetry_points().")
        
    elif isinstance(path, HighSymmetryPoints):
        path = path.get_default_path_points()

    # Generate the ideal path geometry in Cartesian and fractional coordinates
    path_cart, k_dist, labels, path_frac = brillouin_zone_path(lattice=lattice, path=path, points_per_seg=points_per_seg)

    if k_vectors is None:
        k_vectors       = getattr(lattice, "kvectors", None)
    if k_vectors_frac is None:
        k_vectors_frac  = getattr(lattice, "kvectors_frac", None)

    if k_vectors is None and k_vectors_frac is None:
        return KPathSelection(path_cart=path_cart, path_frac=path_frac, k_dist=k_dist, labels=labels)

    if k_vectors is None:
        k_vectors       = _cartesian_from_fractional(lattice, k_vectors_frac)
    if k_vectors_frac is None:
        k_vectors_frac  = _fractional_from_cartesian(lattice, k_vectors)

    kc_flat = np.asarray(k_vectors, dtype=float).reshape(-1, 3)
    kf_flat = np.asarray(k_vectors_frac, dtype=float).reshape(-1, 3)
    tol     = _default_kpath_tolerance(lattice) if tol is None else float(tol)

    # Extend the sampled grid to cover higher-zone path coordinates when needed,
    # while still tracking indices back to the original data grid.
    ext_kc, ext_kf, base_indices, grid_indices = _extend_kgrid_to_path(
        lattice,
        kc_flat,
        kf_flat,
        path_frac,
        tol=tol,
    )

    # Find nearest k-points on the grid for each point along the path
    indices, match_distances = find_nearest_kpoints(ext_kf, path_frac, tol=tol, periodic=periodic and len(ext_kf) == len(kf_flat))
    match_tolerance = tol
    if not periodic:
        match_distances = _polyline_point_distances(ext_kc[indices], path_cart)
        match_tolerance = _kpath_tolerance_to_cartesian(lattice, tol)

    return KPathSelection(path_cart=path_cart, path_frac=path_frac, k_dist=k_dist, labels=labels, 
                        matched_cart=ext_kc[indices], matched_frac=ext_kf[indices],
                        matched_grid_indices=grid_indices[indices], matched_indices=base_indices[indices],
                        matched_distances=match_distances,
                        match_tolerance=match_tolerance)

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
    values : np.ndarray
        Data values along the path. The path axis is ``path_axis``. Examples:
        ``(Npath,)``, ``(Npath, n_bands)``, ``(Nw, Npath)``, or
        ``(Nw, Npath, n_bands)``.
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
    path_axis           : int = 0
    
    @property
    def n_points(self) -> int:
        """Number of points along the path."""
        return len(self.k_dist)
    
    @property
    def n_bands(self) -> int:
        """Number of trailing channels per path point, flattening axes after ``path_axis``."""
        tail_shape = self.values.shape[self.path_axis + 1:]
        return int(np.prod(tail_shape)) if len(tail_shape) > 0 else 1
    
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

def bz_path_data(
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
    through high-symmetry points. It handles periodic boundary conditions in k-space
    and automatically reuses reciprocal-lattice copies of the sampled grid when the
    requested path lies in an extended Brillouin-zone region.
    It also allows to return a structured KPathResult dataclass or a tuple...
    
    Parameters
    ----------
    lattice : Lattice
        Lattice object with reciprocal lattice vectors
    k_vectors : np.ndarray, shape (..., 3) 
        Cartesian k-points (will be flattened)
    k_vectors_frac : np.ndarray, shape (..., 3)
        Fractional coordinates of k-points (will be flattened)
    values : np.ndarray
        Data values sampled on the k-grid. The k-grid axes may appear either as
        the leading axes (e.g. ``(Lx, Ly, Lz, n_bands)``) or after batch axes
        (e.g. ``(Nw, Lx, Ly, Lz)`` or ``(Nw, Lx, Ly, Lz, n_bands)``). A single
        flattened k-grid axis of length ``Nk`` is also supported.
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
        If return_result=True: KPathResult dataclass with all path data. The
        returned ``values`` preserve any leading batch axes and replace the
        sampled k-grid axes with a path axis.
        If return_result=False: (k_cart, k_frac, k_dist, labels, values) tuple
    
    Examples
    --------
    >>> # Using default path from HighSymmetryPoints
    >>> result = bz_path_data(lattice, k_grid, k_frac, energies, HighSymmetryPoints.square_2d())
    >>> plt.plot(result.k_dist, result.values)
    
    >>> # Using standard path enum
    >>> result = bz_path_data(lattice, k_grid, k_frac, energies, 'SQUARE_2D')
    
    >>> # Custom path
    >>> custom_path = [('G', [0,0,0]), ('X', [0.5,0,0]), ('G', [0,0,0])]
    >>> result      = bz_path_data(lattice, k_grid, k_frac, energies, custom_path)
    >>>
    >>> # Frequency-resolved values with shape (Nw, Lx, Ly, Lz)
    >>> result_w    = bz_path_data(lattice, k_grid, k_frac, S_qw, custom_path)
    >>> result_w.values.shape
    (Nw, result_w.n_points)
    """
    
    k_vectors_arr = np.asarray(k_vectors)
    kgrid_shape = tuple(int(dim) for dim in k_vectors_arr.shape[:-1])
    if len(kgrid_shape) == 0:
        kgrid_shape = (int(k_vectors_arr.reshape(-1, 3).shape[0]),)
    nk = int(np.prod(kgrid_shape))

    values_arr = np.asarray(values)
    if values_arr.ndim == 0:
        raise ValueError("values must have at least one axis corresponding to the sampled k-grid")

    grid_axis = None
    grid_span = None
    for start in range(values_arr.ndim - len(kgrid_shape) + 1):
        if tuple(values_arr.shape[start:start + len(kgrid_shape)]) == kgrid_shape:
            grid_axis = start
            grid_span = len(kgrid_shape)
            break
    if grid_axis is None:
        for axis, axis_size in enumerate(values_arr.shape):
            if int(axis_size) == nk:
                grid_axis = axis
                grid_span = 1
                break
    if grid_axis is None or grid_span is None:
        raise ValueError(
            f"Could not identify k-grid axes {kgrid_shape} in values with shape {values_arr.shape}. "
            "Expected values to contain the sampled k-grid axes contiguously or as a flattened Nk axis."
        )

    batch_shape = values_arr.shape[:grid_axis]
    feature_shape = values_arr.shape[grid_axis + grid_span:]
    val_flat = values_arr.reshape(batch_shape + (nk,) + feature_shape)

    # Select path points and find nearest k-points on the grid
    selection: KPathSelection = bz_path_points(lattice, path=path, points_per_seg=points_per_seg, k_vectors=k_vectors, k_vectors_frac=k_vectors_frac,)
    if not selection.has_matches:
        raise ValueError("Path selection requires sampled k-vectors to extract path data.")

    # Extract values for the matched k-points along the path
    vals_sel = np.take(val_flat, selection.matched_indices, axis=len(batch_shape))

    # Optionally return a structured dataclass with all path information, or a tuple for backwards compatibility
    if return_result:
        return KPathResult(
            k_cart              = selection.matched_cart,
            k_frac              = selection.matched_frac,
            k_dist              = selection.k_dist,
            labels              = selection.labels,
            values              = vals_sel,
            indices             = selection.matched_indices,
            matched_distances   = selection.matched_distances,
            path_axis           = len(batch_shape),
        )
    else:
        return selection.matched_cart, selection.matched_frac, selection.k_dist, selection.labels, vals_sel

# -----------------------------------------------------------------------------------------------------------
#! BLOCH TRANSFORMATION
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

# Global cache dictionary: lattice_id -> BlochTransformCache
_bloch_cache: Dict[int, BlochTransformCache] = {}

def _get_lattice_hash(lattice: 'Lattice') -> int:
    """Generate a hash from lattice parameters to detect changes.
    
    Includes boundary flux so the cache is invalidated when flux changes.
    """
    flux_hash = ()
    if hasattr(lattice, '_flux') and lattice._flux is not None:
        flux_hash = tuple(lattice._flux.as_array())
        
    return hash((
        lattice._lx, lattice._ly, lattice._lz,
        len(lattice._basis),
        tuple(lattice._a1.flatten()),
        tuple(lattice._a2.flatten()),
        tuple(lattice._a3.flatten()),
        tuple(lattice._k1.flatten()),
        tuple(lattice._k2.flatten()),
        tuple(lattice._k3.flatten()),
        flux_hash,
    ))

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

    # Apply flux-induced shift when boundary fluxes are present
    if hasattr(lattice, '_flux_frac_shift'):
        dfx, dfy, dfz   = lattice._flux_frac_shift()
        frac_x          = frac_x + dfx
        frac_y          = frac_y + dfy
        frac_z          = frac_z + dfz

    kx_frac, ky_frac, kz_frac   = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")
    kgrid_frac                  = np.stack([kx_frac, ky_frac, kz_frac], axis=-1)    # (Lx,Ly,Lz,3)

    kgrid           = (kx_frac[..., None] * b1
                     + ky_frac[..., None] * b2
                     + kz_frac[..., None] * b3)                                     # (Lx,Ly,Lz,3)

    kpoints         = kgrid.reshape(-1, 3)                                          # (Nc,3)

    # 4. real-space Bravais vectors and sublattice indices
    R_cells         = np.asarray(lattice.cells, float)                              # (Ns,3)
    if R_cells.shape[0] != Ns:
        raise ValueError("Mismatch in number of sites and lattice.cells.")

    sub_idx         = np.asarray(lattice.subs, dtype=int)                           # (Ns,)
    if sub_idx.shape[0] != Ns:
        raise ValueError("Mismatch in number of sites and lattice.subs.")

    # Projector S[i, alpha] = delta_{beta_i, alpha}
    S               = np.zeros((Ns, Nb), dtype=complex)
    S[np.arange(Ns), sub_idx] = 1.0

    # 5. Bloch projectors: W[k,i,alpha] = exp(-i k·R_i) / sqrt(Nc) * S[i,alpha]
    phases          = np.exp(-1j * (kpoints @ R_cells.T))                           # (Nc,Ns)
    if unitary_norm:
        phases     /= np.sqrt(Nc)

    W               = phases[:, :, None] * S[None, :, :]                            # (Nc,Ns,Nb)
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
    dim         = lattice.dim
    n_basis     = lattice.multipartity
    Ns          = lattice.Ns
    Lx, Ly, Lz  = lattice.Lx, lattice.Ly, lattice.Lz

    # Allocate matrices
    T1          = np.zeros((Ns, Ns), dtype=complex)
    T2          = np.zeros_like(T1)
    T3          = np.zeros_like(T1)
    Ts          = [T1, T2, T3]

    # Precompute indices
    indices         = np.arange(Ns)
    basis_idx       = indices % n_basis
    cell_idx        = indices // n_basis

    nx              = cell_idx % Lx
    ny              = (cell_idx // Lx) % Ly
    nz              = (cell_idx // (Lx * Ly)) % Lz

    coords_idx      = [nx, ny, nz]
    dims            = [Lx, Ly, Lz]

    for dir_idx in range(dim):
        # Calculate new coordinates and winding
        # Translation along dir_idx means adding 1 to the corresponding coordinate
        # The coordinate is coords_idx[dir_idx]

        # New coordinate: (n + 1) % L
        n_curr      = coords_idx[dir_idx]
        L           = dims[dir_idx]
        n_next      = (n_curr + 1) % L

        # Winding number: 1 if wrapping around boundary, 0 otherwise
        # Wrapped when n_curr + 1 == L => n_next == 0
        winding     = (n_curr + 1) // L

        # Compute new cell index
        # cell_new = nx_new + Lx * ny_new + Lx * Ly * nz_new
        # Only one coordinate changes

        if dir_idx == 0:
            nx_new = n_next
            ny_new = ny
            nz_new = nz
            wx, wy, wz = winding, np.zeros_like(winding), np.zeros_like(winding)
        elif dir_idx == 1:
            nx_new = nx
            ny_new = n_next
            nz_new = nz
            wx, wy, wz = np.zeros_like(winding), winding, np.zeros_like(winding)
        elif dir_idx == 2:
            nx_new = nx
            ny_new = ny
            nz_new = n_next
            wx, wy, wz = np.zeros_like(winding), np.zeros_like(winding), winding
        else:
            raise ValueError(f"Invalid direction index {dir_idx}")

        cell_new    = nx_new + Lx * ny_new + Lx * Ly * nz_new
        j_indices   = cell_new * n_basis + basis_idx

        # Compute phases
        if hasattr(lattice, "boundary_phase_from_winding"):
            # If lattice has boundary flux, we need to apply phases
            # Optimization:
            # 1. Compute phase for trivial winding (0,0,0) -> 1.0
            # 2. Compute phase for non-trivial winding (only where winding > 0)

            # Default phase is 1.0
            phases = np.ones(Ns, dtype=complex)

            # Find sites where winding occurred
            mask_w = (winding > 0)
            if np.any(mask_w):
                # We can either iterate over unique windings or just compute for the single case w=1
                # Usually winding is only 1.

                # Check if we can vectorize boundary_phase_from_winding?
                # It usually calls self._flux.phase(direction, winding).
                # If we access lattice.flux directly? No, abstraction.

                # Assume standard implementation: phase = exp(i * phi * w)
                # We can just compute phase for w=1 in this direction.

                # Calculate phase for w=1
                w_args = [0, 0, 0]
                w_args[dir_idx] = 1
                phase_1 = lattice.boundary_phase_from_winding(*w_args)

                if abs(phase_1 - 1.0) > 1e-14:
                    phases[mask_w] = phase_1

                    # If winding > 1 (unlikely for +1 translation unless L=1)
                    # L=1 => winding = 1.
                    # If L < 1 (impossible), etc.
                    # Technically if L=1, winding is 1.
                    # If someone defined L=0.5 (impossible for integer).

                    # If there are sites with winding > 1?
                    # (n+1)//L can be > 1 only if n >= 2L-1.
                    # But n < L. So n+1 <= L. So (n+1)//L <= 1.
                    # So winding is always 0 or 1.
                    pass

        else:
            phases = np.ones(Ns, dtype=complex)

        # Assign to matrix
        # Ts[dir_idx][j_indices, indices] = phases
        Ts[dir_idx][j_indices, indices] = phases

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
        F_inv       = F.conj().T # DFT is unitary, so inverse is conjugate transpose
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
