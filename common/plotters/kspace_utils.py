'''
K-space utilities for quantum physics plotting.

General-purpose utilities for k-space operations:
- High-symmetry path extraction from discrete k-grids
- Structure factor computation from real-space correlations
- Distance calculations and path matching
- Visualization helpers for Brillouin zones

These functions are not specific to ED and can be used with
any quantum physics calculation that involves k-space.

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-01-15
License             : MIT
----------------------------------------------------------------
'''

from    typing import List, TYPE_CHECKING, Optional
import  numpy as np

if TYPE_CHECKING:
    from general_python.lattices.lattice import Lattice

# ==============================================================================
# K-SPACE PATH UTILITIES
# ==============================================================================

def point_to_segment_distance_2d(
        points      : np.ndarray,
        p1          : np.ndarray,
        p2          : np.ndarray
    ) -> np.ndarray:
    """
    Compute perpendicular distance from points to line segment p1-p2 in 2D.
    
    Uses vector projection to find closest point on segment, then computes distance.
    Handles endpoints correctly by clamping projection parameter to [0,1].
    
    Parameters
    ----------
    points : (N, 2) array
        Points to check
    p1, p2 : (2,) arrays
        Segment endpoints
        
    Returns
    -------
    distances : (N,) array
        Perpendicular distance from each point to segment
        
    Notes
    -----
    Algorithm:
    1. Project point onto infinite line: t = (p-p1)·(p2-p1) / |p2-p1|²
    2. Clamp t to [0,1] to stay on segment
    3. Find closest point: c = p1 + t*(p2-p1)
    4. Distance: |p - c|
    """
    points      = np.asarray(points, dtype=float)
    p1          = np.asarray(p1, dtype=float).ravel()
    p2          = np.asarray(p2, dtype=float).ravel()
    
    if points.ndim == 1:
        points = points[None, :]
    
    # Vector from p1 to p2
    v           = p2 - p1
    v_len_sq    = np.dot(v, v)
    
    if v_len_sq < 1e-14:
        # Degenerate segment (p1 ≈ p2)
        return np.linalg.norm(points - p1[None, :], axis=1)
    
    # Project each point onto line
    # t = (points - p1) · v / |v|²
    dp          = points - p1[None, :]
    t           = np.dot(dp, v) / v_len_sq
    
    # Clamp to [0,1] to stay on segment
    t           = np.clip(t, 0.0, 1.0)
    
    # Closest point on segment
    closest     = p1[None, :] + t[:, None] * v[None, :]
    
    # Distance
    dist        = np.linalg.norm(points - closest, axis=1)
    
    return dist

def select_kpoints_along_path(
        lattice         : "Lattice",
        k_vectors       : np.ndarray,
        path_labels     : Optional[List[str]]   = None,
        tolerance       : Optional[float]       = None,
        use_extend      : bool                  = False,
        extend_copies   : int                   = 2
    ) -> dict:
    """
    Select k-points along high-symmetry path.
    
    Parameters
    ----------
    lattice : Lattice
        Lattice with high_symmetry_points() and reciprocal vectors
    k_vectors : (Nk, D) array
        K-point coordinates (Cartesian)
    path_labels : list of str, optional
        Path specification (e.g., ['Gamma', 'K', 'M', 'Gamma'])
        If None, uses lattice default path
    tolerance : float, optional
        Distance tolerance for k-point selection
        If None, auto-determined from k-grid spacing
    use_extend : bool
        If True, extend k-space to show multiple BZ copies
    extend_copies : int
        Number of BZ copies in each direction
        
    Returns
    -------
    result : dict
        {
            'indices'           : (Npath,) array of selected k-indices,
            'distances'         : (Npath,) array of cumulative path distances,
            'label_positions'   : (Nlabels,) array of label x-positions,
            'label_texts'       : list of label strings,
            'k_cart'            : (Npath, D) array of selected k-vectors
        }
    """
    selection = lattice.bz_path_points(
        path=path_labels,
        points_per_seg=40,
        k_vectors=k_vectors,
        tol=tolerance,
        periodic=not use_extend,
    )

    return {
        'indices'           : np.asarray(selection.matched_indices, dtype=int),
        'distances'         : np.asarray(selection.k_dist, dtype=float),
        'label_positions'   : np.asarray(selection.label_positions, dtype=float),
        'label_texts'       : list(selection.label_texts),
        'k_cart'            : np.asarray(selection.matched_cart if selection.matched_cart is not None else selection.path_cart, dtype=float),
    }

# ==============================================================================
# STRUCTURE FACTOR COMPUTATION
# ==============================================================================

def compute_structure_factor_from_corr(
        C           : np.ndarray,
        r_vectors   : np.ndarray,
        k_vectors   : np.ndarray,
        normalize   : bool = True) -> np.ndarray:
    r"""
    Compute structure factor S(k) from correlation matrix C(i,j).
    
    S(k)    = (1/Ns) sum _{i,j} C_{ij} exp[-ik . (r_i - r_j)]
            = (1/Ns) Re[Tr(P C P†)]
    
    where P_{k,i} = exp(-ik . r_i).
    
    Parameters
    ----------
    C : (Ns, Ns) array
        Correlation matrix in site basis
    r_vectors : (Ns, D) array
        Real-space position vectors
    k_vectors : (Nk, D) array
        Momentum vectors
    normalize : bool
        If True, divide by Ns
        
    Returns
    -------
    Sk : (Nk,) array
        Structure factor at each k-point
    """
    C       = np.asarray(C)
    r_arr   = np.asarray(r_vectors, float)
    k_arr   = np.asarray(k_vectors, float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square correlation matrix.")
    if r_arr.shape[0] != C.shape[0]:
        raise ValueError("r_vectors must have one position vector per site.")

    dim     = min(r_arr.shape[1], k_arr.shape[1])
    phases  = np.exp(-1j * (k_arr[:, :dim] @ r_arr[:, :dim].T))
    Sk      = np.einsum('ki,ij,kj->k', phases, C, np.conjugate(phases)).real

    if normalize:
        Sk /= float(C.shape[0])
        
    return Sk

# ==============================================================================
# VISUALIZATION HELPERS
# ==============================================================================

def label_high_sym_points(ax, lattice: "Lattice", bz_copies: int = 2, show_labels: bool = True, **kwargs):
    """
    Add markers and labels for high-symmetry points in all BZ copies.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    lattice : Lattice
        Lattice object with high_symmetry_points() method
    bz_copies : int
        Number of BZ copies in each direction (0 = first BZ only)
    show_labels : bool
        If True, show text labels for high-symmetry points
    **kwargs : dict
        Styling options (markersize, markerfacecolor, label_fontsize, etc.)
    """
    try:
        hs_points = lattice.high_symmetry_points()
        if hs_points is None:
            return
        
        # Get reciprocal basis
        k1_vec  = lattice.k1
        k2_vec  = lattice.k2
        k3_vec  = lattice.k3 if lattice.dim == 3 else np.array([0.0, 0.0, 0.0])
        k1_2d   = k1_vec[:2]
        k2_2d   = k2_vec[:2]
        
        # Plot high-symmetry points - ONLY in center BZ to avoid clutter
        # unless specifically requested by setting bz_copies > 0
        m_range = range(-bz_copies, bz_copies + 1)
        n_range = range(-bz_copies, bz_copies + 1)
        
        for m in m_range:
            for n in n_range:
                G   = m * k1_2d + n * k2_2d
                
                for point in hs_points:
                    # Convert fractional to Cartesian
                    k_cart  = point.to_cartesian(k1_vec, k2_vec, k3_vec)
                    kx, ky  = k_cart[0] + G[0], k_cart[1] + G[1]
                    
                    # Draw marker
                    ax.plot(
                        kx, ky,
                        marker          =   'o',
                        markersize      =   kwargs.get('markersize', 5),
                        markerfacecolor =   kwargs.get('markerfacecolor', 'white'),
                        markeredgecolor =   kwargs.get('markeredgecolor', 'black'),
                        markeredgewidth =   kwargs.get('markeredgewidth', 1.0),
                        zorder          =   25
                    )
                    
                    # Add label - show for center BZ, and others only if explicitly requested
                    # To avoid clutter, we usually only label the first BZ
                    if show_labels and (m == 0 and n == 0):
                        ax.text(
                            kx - kwargs.get('label_offset_x', 0.5), 
                            ky - kwargs.get('label_offset_y', 0.5),
                            point.latex_label,
                            color       =   kwargs.get('label_color', 'black'),
                            fontsize    =   kwargs.get('label_fontsize', 11),
                            fontweight  =   'bold',
                            ha          =   'center',
                            va          =   'bottom',
                            bbox        =   kwargs.get('label_bbox', None),
                            zorder      =   26
                        )
    except Exception:
        pass

def format_pi_ticks(ax, axis='both'):
    """
    Format axis ticks as multiples of Pi.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    axis : str
        Which axis to format: 'x', 'y', or 'both'
    """
    import matplotlib.ticker as ticker
    
    def pi_formatter(x, pos):
        """Format tick labels as multiples of Pi."""
        if abs(x) < 1e-10:
            return '0'
        
        # Convert to multiples of Pi
        multiple = x / np.pi
        
        # Handle simple cases
        if abs(multiple - round(multiple)) < 0.01:
            m = int(round(multiple))
            if m == 1:
                return r'$\pi$'
            elif m == -1:
                return r'$-\pi$'
            elif m == 0:
                return '0'
            else:
                return fr'${m}\pi$'
        
        # Handle fractional cases
        if abs(multiple - 0.5) < 0.01:
            return r'$\pi/2$'
        elif abs(multiple + 0.5) < 0.01:
            return r'$-\pi/2$'
        elif abs(multiple - 1.5) < 0.01:
            return r'$3\pi/2$'
        elif abs(multiple + 1.5) < 0.01:
            return r'$-3\pi/2$'
        elif abs(multiple) < 3:
            # Try to express as simple fraction
            from fractions import Fraction
            frac = Fraction(multiple).limit_denominator(6)
            if abs(float(frac) - multiple) < 0.01:
                if frac.numerator == 1:
                    return fr'$\pi/{frac.denominator}$'
                elif frac.numerator == -1:
                    return fr'$-\pi/{frac.denominator}$'
                else:
                    return fr'${frac.numerator}\pi/{frac.denominator}$'
        
        # Fallback to decimal
        return f'{x:.2f}'
    
    if axis in ['x', 'both']:
        xticks  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        xlabels = [pi_formatter(x, None) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
        yticks  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        ylabels = [pi_formatter(y, None) for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

# =============================================================================
#! EOF
# =============================================================================
