'''
A module for handling lattice k-space lattice tools and methods.

--------------------------------
File            : lattices/tools/lattice_kspace.py
Author          : Maksymilian Kliczkowski
--------------------------------
'''

from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Optional, Literal
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from ..lattice import Lattice

# -----------------------------------------------------------------------------------------------------------
# ENUMERATIONS OF STANDARD PATHS
# -----------------------------------------------------------------------------------------------------------

class StandardBZPath(Enum):
    """
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
        ("0",       [0.0, 0.0, 0.0]),
        (r"\pi",    [0.5, 0.0, 0.0]),
        (r"2\pi",   [1.0, 0.0, 0.0])
    ]

    SQUARE_2D = [
        ("G",       [0.0, 0.0, 0.0]),
        ("X",       [0.5, 0.0, 0.0]),
        ("M",       [0.5, 0.5, 0.0]),
        ("G",       [0.0, 0.0, 0.0])
    ]

    TRIANGULAR_2D = [
        ("G",       [0.0, 0.0, 0.0]),
        ("M",       [0.5, 0.0, 0.0]),
        ("K",       [1/3, 1/3, 0.0]),
        ("G",       [0.0, 0.0, 0.0])
    ]

    CUBIC_3D = [
        ("G",       [0.0, 0.0, 0.0]),
        ("X",       [0.5, 0.0, 0.0]),
        ("M",       [0.5, 0.5, 0.0]),
        ("R",       [0.5, 0.5, 0.5]),
        ("G",       [0.0, 0.0, 0.0])
    ]

    HONEYCOMB_2D = [
        ("G",       [0.0, 0.0, 0.0]),
        ("K",       [1/3, 1/3, 0.0]),
        ("M",       [0.5, 0.0, 0.0]),
        ("G",       [0.0, 0.0, 0.0])
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
    
    # resolve path
    path        = _resolve_path_input(path)
    recip       = np.vstack([v for v in [lattice._k1, lattice._k2, lattice._k3] if v is not None]) # shape (3, 3) - our lattice always has 3D vectors
    dim         = lattice.dim

    # fractional -> cartesian
    cart_pts    = []
    for label, frac in path:
        f       = np.array(frac, dtype=float)[:dim]
        cart_pts.append(f @ recip)  # k = f1*b1 + f2*b2 + f3*b3

    k_path      = []                    # will hold all k-points along path
    k_path_frac = []                    # fractional coords along path
    k_dist      = [0.0]                 # cumulative distance for plotting
    labels      = [(0, path[0][0])]     # list of (index, label) for symmetry points

    # interpolate segments given in the path
    for i in range(len(cart_pts) - 1):
        p0          = cart_pts[i]               # start point
        p1          = cart_pts[i + 1]           # end point
        
        if points_per_seg is None or points_per_seg < 1:
            # determine from the number of k-points in segment
            # for instance, on a square lattice, the maximal number of points along each direction is Lx, Ly, Lz
            # therefore, when going from p0 to p1, we can estimate the number of points needed to sample the segment
            # based on the length of the segment in reciprocal space and the lattice size
            # we scale by 10 to ensure sufficient sampling -> later we can downsample using the closest k-point method
            # Example: if segment length is 0.5 * |b1| and Lx=10, we want ~5 points along that segment
            points_per_seg_in = int(np.linalg.norm(p1 - p0) / (2.0 * np.pi) * max(lattice._lx, lattice._ly, lattice._lz) * 10)
        
        # interpolate segment       
        seg         = np.linspace(p0, p1, points_per_seg, endpoint=False)                       # cartesian coords - true k-points
        seg_frac    = np.linspace(path[i][1], path[i + 1][1], points_per_seg, endpoint=False)   # fractional coords, e.g., [0.0, 0.0, 0.0] -> [0.5, 0.0, 0.0]
        
        # append to path
        for k in seg:
            if k_path:
                dk = np.linalg.norm(k - k_path[-1])
                k_dist.append(k_dist[-1] + dk)
            k_path.append(k)
            
        for kf in seg_frac:
            k_path_frac.append(kf)

        # add label at end of segment
        labels.append((len(k_path) - 1, path[i + 1][0]))

    return np.array(k_path), np.array(k_dist), labels, np.array(k_path_frac) # fractional coords along path

# -----------------------------------------------------------------------------------------------------------
# K-SPACE PATH EXTRACTION
# -----------------------------------------------------------------------------------------------------------

def extract_bz_path_data(
    lattice         : Lattice,                                                  # Lattice object
    k_vectors       : np.ndarray,                                               # shape (Nk, 3)
    values          : np.ndarray,                                               # shape (Nk, ...)
    path            : Iterable[tuple[str, Iterable[float]]] | StandardBZPath,   # path definition
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]], np.ndarray]:
    """
    Select or interpolate k-points and corresponding values along a Brillouin zone path.

    Parameters
    ----------
    lattice : Lattice
        Lattice object with reciprocal lattice vectors (_k1, _k2, _k3).
    k_vectors : np.ndarray, shape (Nk, 3)
        All computed k-points in reciprocal-space Cartesian coordinates.
        This is not the fractional representation!
    values : np.ndarray, shape (Nk, ...)
        Values (e.g., eigenvalues, observables) associated with each k-point.
    path : list[(label, coords)] or StandardBZPath
        Path definition (fractional coordinates) or one of the standard enums.

    Returns
    -------
    k_path : np.ndarray, shape (Npath, 3)
        k-points along the path.
    k_dist : np.ndarray, shape (Npath,)
        Cumulative distance for x-axis plotting.
    labels : list[(int, str)]
        Indices and labels for symmetry points.
    values_path : np.ndarray, shape (Npath, ...)
        Values corresponding to points along the path (interpolated or discrete).
    """

    k_vectors = np.asarray(k_vectors, dtype=float)
    values    = np.asarray(values)

    # Ensure our k_vectors are a 2D array of shape (Nk, 3), each row a k-point
    if k_vectors.ndim > 2:
        k_vectors = k_vectors.reshape(-1, k_vectors.shape[-1])
        
    # Ensure values is 2D: (Nk, nvals = 1 or more [e.g., bands])
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])

    # Check size consistency
    if k_vectors.shape[0] != values.shape[0]:
        raise ValueError(f"k_vectors and values size mismatch: {k_vectors.shape[0]} vs {values.shape[0]}")

    # ideal geometric path in k-space (cartesian)
    k_ideal, k_dist, labels, k_ideal_frac = brillouin_zone_path(
        lattice        = lattice,
        path           = path,
        points_per_seg = None, # we want only true k-points along path
    )

    # allocate output arrays
    npath               = k_ideal.shape[0]
    nvals               = values.shape[1]
    k_path_sel          = np.zeros_like(k_ideal)
    values_path         = np.zeros((npath, nvals), dtype=values.dtype)

    # convert ideal path to array for distance computations
    # first, get fractional coordinates of ideal path
    # nearest-neighbor sampling along path
    # go through each ideal k-point, find closest in k_vectors
    # compute effective fractional coordinates of existing k-vectors
    
    # go through each ideal k-point, find closest in k_vectors
    for i, kf_frac in enumerate(k_ideal_frac):
        
        # get cartesian coords of ideal k-point -> if we want to match in cartesian space
        k_cart          = k_ideal[i]  # shape (3,)
        # find closest in k_vectors
        diff            = np.linalg.norm(k_vectors - k_cart, axis=1) # (\vec{k} - \vec{k_ideal}) norm
        idx             = np.argmin(diff)
        k_path_sel[i]   = k_vectors[idx]
        values_path[i]  = values[idx]

    return k_path_sel, k_dist, labels, values_path

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

def reconstruct_k_grid_from_blocks(blocks):
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
    energy_grid = np.zeros((Lx, Ly, Lz, n_bands))

    # Fill by index
    for blk in blocks:
        ix, iy, iz                  = blk.block_index
        k_grid[ix, iy, iz, :]       = blk.point
        energy_grid[ix, iy, iz, :]  = blk.en

    return k_grid, energy_grid

# -------------------------------------------------------------------------------------------
#! SPACE TRANSFORMATIONS
# -------------------------------------------------------------------------------------------

def realspace_from_kspace(lattice, H_k: np.ndarray, kgrid: Optional[np.ndarray] = None) -> np.ndarray:
    r"""
    Inverse Bloch transform: H(k) blocks -> H_real (Ns times Ns).
    
    Reconstructs the real-space Hamiltonian from k-space blocks using the inverse 
    Fourier transform. This is the exact inverse of `kspace_from_realspace()` with 
    `extract_bands=False`.
    
    Formula:
        H_real = (1/Nc) \sum_k U^\dag(k) H(k) U(k)
    where U(k)_ij = exp(+i k·r_i) delta_ij (note: +i for inverse transform).
    
    Parameters
    ----------
    H_k : np.ndarray
        K-space Hamiltonian blocks. Shape (Lx, Ly, Lz, Ns, Ns) or (Nk, Ns, Ns).
    kgrid : Optional[np.ndarray]
        K-point grid for reference. If None, reconstructs from shape assuming full grid.
        Shape (Lx, Ly, Lz, 3) or (Nk, 3).
    
    Returns
    -------
    H_real : np.ndarray
        Real-space Hamiltonian (Ns times Ns).
        
    Notes
    -----
    - Eigenvalues are preserved to machine precision (error ~1e-15)
    - Matrix coefficients differ due to eigenvector basis rotation (expected)
    - Only works with full NstimesNs matrices (raises error for NbtimesNb band blocks)
    
    Examples
    --------
    >>> lat = HoneycombLattice(dim=2, lx=2, ly=2, bc='pbc')
    >>> H_real_orig = np.random.randn(lat.Ns, lat.Ns)
    >>> H_real_orig = H_real_orig + H_real_orig.conj().T  # Make Hermitian
    >>>
    >>> # Forward transform
    >>> H_k, k_grid = lat.kspace_from_realspace(H_real_orig, extract_bands=False)
    >>>
    >>> # Inverse transform
    >>> H_real_recon = lat.realspace_from_kspace_exact(H_k, k_grid)
    >>>
    >>> # Check eigenvalues match
    >>> evals_orig = np.linalg.eigvalsh(H_real_orig)
    >>> evals_recon = np.linalg.eigvalsh(H_real_recon)
    >>> np.allclose(np.sort(evals_orig), np.sort(evals_recon))  # True
    """
    import numpy as np
    
    # Parse input shape
    if H_k.ndim == 5:
        # (Lx, Ly, Lz, Ns, Ns) format
        Lx, Ly, Lz, Ns_block, Ns2 = H_k.shape
        Nc = Lx * Ly * Lz
        H_k_flat = H_k.reshape(Nc, Ns_block, Ns2)
    elif H_k.ndim == 3:
        # (Nk, Ns, Ns) format
        Nk, Ns_block, Ns2 = H_k.shape
        H_k_flat = H_k
        Nc = Nk
    else:
        raise ValueError(f"H_k must be 3D or 5D array, got shape {H_k.shape}")
    
    # Check Hermiticity
    if Ns_block != Ns2:
        raise ValueError(f"H_k blocks must be square: got {Ns_block}times{Ns2}")
    
    Ns = Ns_block
    
    # Infer lattice properties
    if Ns != lattice.Ns:
        raise ValueError(f"H_k block size {Ns} != lattice Ns {lattice.Ns}")
    
    Lx = lattice._lx
    Ly = max(lattice._ly, 1)
    Lz = max(lattice._lz, 1)
    expected_Nc = Lx * Ly * Lz
    if Nc != expected_Nc:
        raise ValueError(f"Number of k-points {Nc} != expected {expected_Nc}")
    
    # Reciprocal vectors
    b1 = np.asarray(lattice._k1, float).reshape(3)
    b2 = np.asarray(lattice._k2, float).reshape(3)
    b3 = np.asarray(lattice._k3, float).reshape(3)
    
    # Build k-grid if not provided
    if kgrid is None:
        n_x = np.arange(Lx, dtype=float) / Lx
        n_y = np.arange(Ly, dtype=float) / Ly
        n_z = np.arange(Lz, dtype=float) / Lz
        kgrid = (n_x[:, None, None, None] * b1[None, None, None, :]
            + n_y[None, :, None, None] * b2[None, None, None, :]
            + n_z[None, None, :, None] * b3[None, None, None, :])
        kpoints = kgrid.reshape(-1, 3)
    else:
        if kgrid.ndim == 4:
            kpoints = kgrid.reshape(-1, 3)
        else:
            kpoints = np.asarray(kgrid, float).reshape(-1, 3)
    
    # Site coordinates
    coords = np.asarray(lattice.coordinates, float)
    
    # Inverse Bloch transform: H_real = (1/Nc) Σ_k U†(k) H(k) U(k)
    # U(k) = diag(exp(+i k·r_i)) / sqrt(Nc) for consistency with forward transform
    # But we divide by Nc at the end, so normalization is (1/Nc) Σ ...
    
    H_real = np.zeros((Ns, Ns), dtype=complex)
    for ik, kvec in enumerate(kpoints):
        # Phase: exp(+i k·r_i) for inverse transform
        phi = np.exp(1j * (coords @ kvec))
        U = np.diag(phi)
        # Accumulate: U†(k) H(k) U(k) / Nc
        H_real += U.conj().T @ H_k_flat[ik] @ U
    
    H_real = H_real / Nc
    
    # Ensure Hermiticity (average with conjugate to remove numerical noise)
    H_real = 0.5 * (H_real + H_real.conj().T)
    
    return H_real.astype(lattice._dtype if hasattr(lattice, '_dtype') else np.complex128)

def kspace_from_realspace(
        lattice                 : Lattice,
        H_real                  : np.ndarray,
        kpoints                 : Optional[np.ndarray] = None,
        require_full_grid       : bool = False,
        unitary_norm            : bool = True):
    """
    Bloch projector: H_real (Nstimes Ns) -> H(k) ∈ C^{Nbtimes Nb} at each k, **order-respecting**.

    Assumptions:
    - PBC and true translational invariance if you expect the union of H(k) spectra
        to equal the full real-space spectrum.
    - Site ordering is arbitrary; geometry (coordinates + basis) defines sublattices.

    Returns
    -------
    Hk_grid : (Lx, Ly, Lz, Nb, Nb)   if kpoints is None
        or (Nk, Nb, Nb)          if kpoints is provided
    kgrid   : (Lx, Ly, Lz, 3)       or (Nk, 3)
    """
    import numpy as np
    import scipy.sparse as sp

    #! VALIDATE INPUTS
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
        raise ValueError(f"Ns={Ns} not compatible with Nc={Nc} and Nb={Nb} (Ns must be Nc*Nb).")

    # reciprocal basis
    b1          = np.asarray(lattice._k1, float).reshape(3)
    b2          = np.asarray(lattice._k2, float).reshape(3)
    b3          = np.asarray(lattice._k3, float).reshape(3)

    #! k-point mesh - either full grid or provided points
    if kpoints is None:
        # fractional coordinates in reciprocal lattice basis
        frac_x  = np.linspace(0, 1, Lx, endpoint=False)
        frac_y  = np.linspace(0, 1, Ly, endpoint=False)
        frac_z  = np.linspace(0, 1, Lz, endpoint=False)

        # handle dimensionality automatically (1D/2D/3D)
        # Calculate k-grid for all dimensions, ensuring a (Lx, Ly, Lz, 3) shape
        # even if some dimensions are effectively 1.
        kx, ky, kz  = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")
        kgrid       = kx[..., None] * b1 + ky[..., None] * b2 + kz[..., None] * b3  # shape (Lx, Ly, Lz, 3)
        kpoints     = kgrid.reshape(-1, 3)                                          # shape (Nc, 3)
        return_grid = True
    else:
        kpoints     = np.asarray(kpoints, float).reshape(-1, 3)
        return_grid = False

    # Determine number of k-points
    Nk = kpoints.shape[0]
    if require_full_grid and Nk != Nc:
        raise ValueError(f"Round-trip requires Nk == Nc == {Nc}, got Nk={Nk}.")

    #! geometric labeling: (cell, sub) while keeping order
    # (uses coords + a1,a2,a3 + basis; no reindexing of H_real)
    cell, sub       = lattice.get_geometric_encoding()  # both (Ns,)

    # one-hot projector onto sublattice columns (Nstimes Nb)
    S = np.zeros((Ns, Nb), dtype=complex)
    S[np.arange(Ns), sub] = 1.0

    # site coordinates for Bloch phases (Nstimes 3)
    coords = np.asarray(lattice.coordinates, float)

    # normalization: unitary Bloch projector uses 1/sqrt(Nc)
    norm = (Nc ** 0.5) if unitary_norm else 1.0

    # allow sparse H
    Hdot = (H_real.dot if sp.issparse(H_real) else (lambda X: H_real @ X))

    # --- main loop: w = diag(phi) @ S, then Hk = w^† H w ---
    Hk = np.zeros((Nk, Nb, Nb), dtype=complex)
    for ik, kvec in enumerate(kpoints):
        phi = np.exp(-1j * (coords @ kvec)) / norm      # (Ns,)
        w   = phi[:, None] * S                          # (Ns,Nb)
        Hw  = Hdot(w)                                   # (Ns,Nb)
        Hk[ik] = w.conj().T @ Hw                        # (Nb,Nb)

    if return_grid:
        return Hk.reshape(Lx, Ly, Lz, Nb, Nb), kgrid
    else:
        return Hk, kpoints

# -----------------------------------------------------------------------------------------------------------
#! END OF FILE
# -----------------------------------------------------------------------------------------------------------