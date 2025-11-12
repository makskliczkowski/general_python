'''
A module for handling lattice k-space lattice tools and methods.

--------------------------------
File            : lattices/tools/lattice_kspace.py
Author          : Maksymilian Kliczkowski
--------------------------------
'''

from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, List, Optional, Literal, Tuple

from enum import Enum
import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from ..lattice import Lattice

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

def extract_bz_path_data(
    lattice,
    k_vectors           : np.ndarray,
    k_vectors_frac      : np.ndarray, 
    values              : np.ndarray,
    path                : Iterable[tuple[str, Iterable[float]]] | StandardBZPath,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], np.ndarray]:
    """
    Extract k-path data using FRACTIONAL coordinate matching with periodicity.
    
    This correctly handles the periodic Brillouin zone by comparing
    fractional coordinates modulo 1.
    
    Parameters
    ----------
    k_vectors : np.ndarray, shape (Nk, 3)
        Cartesian k-points
    k_vectors_frac : np.ndarray, shape (Nk, 3)
        Fractional coordinates of k-points (NEW!)
    tolerance : float
        Tolerance for fractional coordinate matching (accounts for numerical error)
    """
    
    # flatten everything for easy search
    kf_flat     = k_vectors_frac.reshape(-1, 3)
    kc_flat     = k_vectors.reshape(-1, 3)
    val_flat    = values.reshape(-1, values.shape[-1])

    # Generate continuous ideal path
    k_ideal_cart, k_dist, labels, k_ideal_frac = brillouin_zone_path(
        lattice=lattice, path=path, points_per_seg=80
    )

    # grid resolution for tolerance
    Lx, Ly, Lz  = lattice._lx, lattice._ly, lattice._lz
    tol         = 0.5 * np.sqrt((1/Lx)**2 + (1/Ly)**2 + (1/Lz)**2)

    # prepare outputs
    Np          = len(k_ideal_frac)
    nb          = values.shape[-1]
    k_sel_cart  = np.zeros((Np, 3))
    k_sel_frac  = np.zeros((Np, 3))
    vals_sel    = np.zeros((Np, nb))

    # loop through path points
    for i, kf_target in enumerate(k_ideal_frac):
        diff    = kf_flat - kf_target
        diff   -= np.round(diff)
        dist    = np.linalg.norm(diff, axis=1)
        idx     = np.argmin(dist)
        if dist[idx] > tol:
            print(f"Warning: far match ({dist[idx]:.3e}) at k={kf_target}")
        k_sel_cart[i]   = kc_flat[idx]
        k_sel_frac[i]   = kf_flat[idx]
        vals_sel[i]     = val_flat[idx]

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
        frac_x                      = np.linspace(0, 1, Lx, endpoint=False)
        frac_y                      = np.linspace(0, 1, Ly, endpoint=False)
        frac_z                      = np.linspace(0, 1, Lz, endpoint=False)
        kx_frac, ky_frac, kz_frac   = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")

        # Store fractional coordinates for return
        # kgrid_frac has shape (Lx, Ly, Lz, 3)
        kgrid_frac                  = np.stack([kx_frac, ky_frac, kz_frac], axis=-1)

        # Construct the Cartesian k-vectors
        # (Lx,Ly,Lz,1) * (3,) -> (Lx,Ly,Lz,3)
        kgrid                       = (kx_frac[..., None] * b1 + 
                                       ky_frac[..., None] * b2 + 
                                       kz_frac[..., None] * b3)  # shape (Lx, Ly, Lz, 3)
        
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
    # cell_idx, sub_idx = lattice.get_geometric_encoding()      # shape (Ns,)
    indices           = np.arange(Ns)
    sub_idx           = indices % Nb
    
    # Ensure basis is padded with zeros for 1D/2D
    basis_coords                                = np.zeros((Nb, 3), dtype=float)
    basis_coords[:, :lattice._basis.shape[1]]   = np.asarray(lattice._basis, float)
    
    # Total position vector r_i = R_n + r_a
    # This should be *identical* to lattice.coordinates if they were
    # generated in the same order as H_real.
    # Using lattice.coordinates is simpler and safer if we trust it.
    coords                                      = np.asarray(lattice.coordinates, float) # shape (Ns, 3)

    # We need the projector S[i, a] = delta_{sub(i), a}
    S                           = np.zeros((Ns, Nb), dtype=complex)     # Bloch projector
    S[np.arange(Ns), sub_idx]   = 1.0                                   # shape (Ns, Nb)
    
    # The FT formula is: H_ab(k) = sum_{n,m} e^{-ik(R_n-R_m)} H_{n,a; m,b}
    # Or, using the projector W: H_ab(k) = sum_{i,j} W*_{i,a} H_{i,j} W_{j,b}
    # Where W_{i,a} = (1/sqrt(Nc)) * e^{-ik . r_i} * S_{i,a}
    #               = (1/sqrt(Nc)) * e^{-ik . (R_n + r_a)} * delta_{sub(i), a}
    
    phases                      = np.exp(-1j * (kpoints @ coords.T))  # (Nc, Ns)
    norm                        = np.sqrt(Nc) if unitary_norm else 1.0
    phases                     /= norm
    
    # Vectorized projector: W[ik, i, a] = phases[ik, i] * S[i, a]
    W                           = phases[:, :, None] * S[None, :, :]  # (Nc, Ns, Nb)
    
    # Transform: H(k) = W† @ H @ W for all k
    # (Nc, Nb, Ns) @ (Ns, Ns) @ (Nc, Ns, Nb) -> (Nc, Nb, Nb)
    Hk                          = np.einsum('kia,ij,kjb->kab', W.conj(), H_real, W)
    
    # Reshape and shift
    Hk                          = Hk.reshape(Lx, Ly, Lz, Nb, Nb)
    kgrid                       = np.fft.fftshift(kgrid, axes=(0, 1, 2))
    kgrid_frac                  = np.fft.fftshift(kgrid_frac, axes=(0, 1, 2))
    return Hk, kgrid, kgrid_frac
    
    norm                    = (Nc ** 0.5) if unitary_norm else 1.0
    Hk                      = np.zeros((Nk, Nb, Nb), dtype=complex)
    for ik in range(Nk):
        # Phase factors for this k-point
        # phi[i] = e^{-i k * r_i} / norm
        # r_i is the full coordinate vector r_i = R_n + r_a
        phi     = np.exp(-1j * (kpoints[ik] @ coords.T)) / norm  # shape (Ns,)
        
        # Bloch projector: W[i,a] = phi[i] * S[i,a]
        # S[i,a] is 1 only if site 'i' belongs to sublattice 'a', 0 otherwise.
        # This correctly applies the phase e^{-ik.r_i} only to the
        # basis state 'a' that site 'i' corresponds to.
        W       = phi[:, None] * S  # shape (Ns, Nb)
        
        # Transform: H(k) = W† H W
        if sp.issparse(H_real):
            # For sparse: (Nb, Ns) @ sparse(Ns, Ns) @ (Ns, Nb)
            Hk[ik] = W.conj().T @ (H_real @ W)
        else:
            Hk[ik] = W.conj().T @ H_real @ W

    if return_grid:
        # Also return fractional coordinates
        Hk          = Hk.reshape(Lx, Ly, Lz, Nb, Nb)
        k_grid      = np.fft.fftshift(k_grid,       axes=(0, 1, 2))
        k_grid_frac = np.fft.fftshift(k_grid_frac,  axes=(0, 1, 2))
        result      = (Hk, k_grid, k_grid_frac) # This is already (Lx, Ly, Lz, 3)
        return result
    else:
        return Hk, kpoints

def kspace_from_realspace_fft(
        lattice,
        H_real: np.ndarray,
        unitary_norm: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FFT-based Bloch transform: H_real (Ns×Ns) -> H(k) (Lx,Ly,Lz,Nb,Nb)
    
    This uses numpy's FFT for computational efficiency.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice object with geometry info
    H_real : np.ndarray
        Real-space Hamiltonian (Ns × Ns)
    unitary_norm : bool
        If True, use 1/√Nc normalization (unitary transform)
        
    Returns
    -------
    Hk : np.ndarray, shape (Lx, Ly, Lz, Nb, Nb)
        k-space Hamiltonian blocks
    kgrid : np.ndarray, shape (Lx, Ly, Lz, 3)
        Cartesian k-vectors (fftshifted to center at Γ)
    kgrid_frac : np.ndarray, shape (Lx, Ly, Lz, 3)
        Fractional k-coordinates (fftshifted)
    """
    # Validate inputs
    if H_real.ndim != 2 or H_real.shape[0] != H_real.shape[1]:
        raise ValueError("H_real must be square")
    
    Ns = H_real.shape[0]
    if Ns != lattice.Ns:
        raise ValueError(f"H_real size {Ns} != lattice.Ns {lattice.Ns}")
    
    # Convert sparse to dense if needed
    if sp.issparse(H_real):
        H_real = H_real.toarray()
    
    # Lattice parameters
    Lx, Ly, Lz = lattice._lx, max(lattice._ly, 1), max(lattice._lz, 1)
    Nc = Lx * Ly * Lz  # number of unit cells
    Nb = len(lattice._basis)  # sublattices per cell
    
    if Ns != Nc * Nb:
        raise ValueError(f"Ns={Ns} must equal Nc*Nb = {Nc}*{Nb} = {Nc*Nb}")
    
    # Reciprocal lattice vectors
    b1 = np.asarray(lattice._k1, float).reshape(3)
    b2 = np.asarray(lattice._k2, float).reshape(3)
    b3 = np.asarray(lattice._k3, float).reshape(3)
    
    # Reshape Hamiltonian: (Ns, Ns) -> (Lx, Ly, Lz, Nb, Lx, Ly, Lz, Nb)
    # Index as: [ix, iy, iz, a, jx, jy, jz, b]
    # where (ix,iy,iz) is cell index and a,b are sublattice indices
    H_reshaped = np.zeros((Lx, Ly, Lz, Nb, Lx, Ly, Lz, Nb), dtype=complex)
    
    # Fill reshaped array
    for i in range(Ns):
        for j in range(Ns):
            # Decompose site indices
            cell_i, sub_i = i // Nb, i % Nb
            cell_j, sub_j = j // Nb, j % Nb
            
            # Cell coordinates
            ix = cell_i % Lx
            iy = (cell_i // Lx) % Ly
            iz = (cell_i // (Lx * Ly)) % Lz
            
            jx = cell_j % Lx
            jy = (cell_j // Lx) % Ly
            jz = (cell_j // (Lx * Ly)) % Lz
            
            H_reshaped[ix, iy, iz, sub_i, jx, jy, jz, sub_j] = H_real[i, j]
    
    # Apply FFT over spatial dimensions (axes 4, 5, 6 = jx, jy, jz)
    # This computes: H_ab(R, k) = Σ_R' H_ab(R, R') e^(-ik·R')
    norm_factor = np.sqrt(Nc) if unitary_norm else 1.0
    
    Hk_temp = np.fft.fftn(H_reshaped, axes=(4, 5, 6), norm='ortho' if unitary_norm else None)
    
    # For translationally invariant systems, H(k) should be independent of R
    # Average over R (or just take R=0)
    # H_ab(k) = H_ab(R=0, k)
    Hk = Hk_temp[0, 0, 0, :, :, :, :, :]  # Take R=0 reference cell
    
    # Rearrange to (Lx, Ly, Lz, Nb, Nb)
    Hk = np.transpose(Hk, (1, 2, 3, 0, 4))  # (kx, ky, kz, sub_i, sub_j)
    
    # Generate k-grid
    frac_x = np.fft.fftfreq(Lx)  # FFT frequencies: [0, 1/L, ..., (L-1)/L] -> [-0.5, ..., 0.5)
    frac_y = np.fft.fftfreq(Ly)
    frac_z = np.fft.fftfreq(Lz)
    
    kx_frac, ky_frac, kz_frac = np.meshgrid(frac_x, frac_y, frac_z, indexing='ij')
    kgrid_frac = np.stack([kx_frac, ky_frac, kz_frac], axis=-1)
    
    # Convert to Cartesian k-vectors
    kgrid = (kx_frac[..., None] * b1 + 
             ky_frac[..., None] * b2 + 
             kz_frac[..., None] * b3)
    
    # Apply fftshift to center at Γ
    Hk = np.fft.fftshift(Hk, axes=(0, 1, 2))
    kgrid = np.fft.fftshift(kgrid, axes=(0, 1, 2))
    kgrid_frac = np.fft.fftshift(kgrid_frac, axes=(0, 1, 2))
    
    return Hk, kgrid, kgrid_frac

# -----------------------------------------------------------------------------------------------------------
#! END OF FILE
# -----------------------------------------------------------------------------------------------------------