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
    from general_python.lattices.tools.lattice_kspace import extend_kspace_data
    
    # Get high-symmetry points
    hs_points_obj       = lattice.high_symmetry_points()
    if hs_points_obj is None:
        raise ValueError("Lattice does not define high_symmetry_points")
    
    # Determine path
    if path_labels is None:
        path_labels     = hs_points_obj.default_path
    if path_labels is None:
        raise ValueError("No path specified and lattice has no default path")
    
    # Extend k-space if requested
    if use_extend:
        k1_vec          = np.asarray(lattice.k1, float).ravel()[:2]
        k2_vec          = np.asarray(lattice.k2, float).ravel()[:2]
        k2_extended, _  = extend_kspace_data(
                            k_vectors[:, :2],
                            np.arange(len(k_vectors)),
                            k1_vec, k2_vec,
                            nx=extend_copies,
                            ny=extend_copies
                        )
        k2 = k2_extended
    else:
        k2 = k_vectors[:, :2]
    
    # Convert path labels to Cartesian coordinates
    path_points_cart    = []
    k1_vec              = np.asarray(lattice.k1, float).reshape(3)
    k2_vec              = np.asarray(lattice.k2, float).reshape(3)
    
    for label in path_labels:
        if label not in hs_points_obj.points:
            raise ValueError(f"High-symmetry point '{label}' not defined for this lattice")
        pt_frac = np.array(hs_points_obj.points[label], dtype=float)
        pt_cart = pt_frac[0] * k1_vec[:2] + pt_frac[1] * k2_vec[:2]
        path_points_cart.append(pt_cart)
    
    # Auto-determine tolerance
    if tolerance is None:
        k_spacing = np.median(np.diff(np.sort(k2[:, 0])))
        tolerance = k_spacing * 0.5
    
    # Select k-points along path segments
    selected_k_indices  = []
    cumulative_dist     = 0.0
    k_distances         = []
    label_positions     = [0.0]
    label_texts         = [path_labels[0]]
    
    for i in range(len(path_points_cart) - 1):
        p1                  = path_points_cart[i]
        p2                  = path_points_cart[i + 1]
        
        # Find k-points close to this segment
        distances           = point_to_segment_distance_2d(k2, p1, p2)
        close_mask          = distances < tolerance
        segment_k_indices   = np.where(close_mask)[0]
        
        if len(segment_k_indices) > 0:
            # Project onto path direction
            segment_k_points    = k2[segment_k_indices]
            path_vec            = p2 - p1
            path_length         = np.linalg.norm(path_vec)
            
            if path_length > 1e-14:
                path_dir            = path_vec / path_length
                proj                = np.dot(segment_k_points - p1[None, :], path_dir)
                
                # Sort by projection
                sort_idx            = np.argsort(proj)
                segment_k_indices   = segment_k_indices[sort_idx]
                proj                = proj[sort_idx]
                
                # Add to total distance
                segment_distances   = cumulative_dist + proj
                k_distances.extend(segment_distances.tolist())
                selected_k_indices.extend(segment_k_indices.tolist())
        
        # Update cumulative distance
        cumulative_dist += np.linalg.norm(p2 - p1)
        label_positions.append(cumulative_dist)
        label_texts.append(path_labels[i + 1])
    
    # Remove duplicates
    unique_indices      = []
    seen                = set()
    unique_distances    = []
    for idx, dist in zip(selected_k_indices, k_distances):
        if idx not in seen:
            unique_indices.append(idx)
            unique_distances.append(dist)
            seen.add(idx)
    
    selected_k_indices = np.array(unique_indices, dtype=int)
    k_distances = np.array(unique_distances, dtype=float)
    
    return {
        'indices'           : selected_k_indices,
        'distances'         : k_distances,
        'label_positions'   : np.array(label_positions),
        'label_texts'       : label_texts,
        'k_cart'            : k_vectors[selected_k_indices] if not use_extend else k2[selected_k_indices]
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
    C   = np.asarray(C, float)
    r2  = np.asarray(r_vectors, float)[:, :2]
    k2  = np.asarray(k_vectors, float)[:, :2]
    
    Ns  = C.shape[0]
    P   = np.exp(-1j * (k2 @ r2.T))     # (Nk, Ns)
    PC  = P @ C                         # (Nk, Ns)
    Sk  = np.real((PC * np.conjugate(P)).sum(axis=1))
    
    if normalize:
        Sk /= Ns
        
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
        # When showing extended zones, only mark center (m=0, n=0)
        m_range = range(0, 1) if bz_copies > 0 else range(0, 1)
        n_range = range(0, 1) if bz_copies > 0 else range(0, 1)
        
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
                    
                    # Add label - always show for center BZ
                    if show_labels:
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