'''
This module contains common plotting utilities for Quantum EigenSolver.

Modular architecture for ED plotting with support for:
- Real-space and k-space correlation functions
- Dynamical spectral functions A(k,w) and S(k,w)
- Discrete k-point handling
- High-symmetry path extraction
- Flexible visualization options

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-12-01
License             : MIT
----------------------------------------------------------------
'''

from    typing              import List, Callable, TYPE_CHECKING, Optional, Tuple, Union, Literal, Any
from    dataclasses         import dataclass, field
from    scipy.interpolate   import griddata
import  numpy               as np
import  matplotlib.pyplot   as plt

try:
    from general_python.lattices.lattice    import Lattice
    from .data_loader                       import load_results, PlotDataHelpers
    from ..plot                             import Plotter
except ImportError:
    raise ImportError("Failed to import required modules from the current package.")

if TYPE_CHECKING:
    from general_python.common.flog         import Logger

# ==============================================================================
# CONFIGURATION DATACLASSES
# ==============================================================================

@dataclass
class PlotStyle:
    """Styling configuration for plots."""
    cmap                : str               = 'viridis'
    vmin                : Optional[float]   = None
    vmax                : Optional[float]   = None
    vmin_strategy       : Literal['auto', 'percentile', 'absolute'] = 'auto'
    vmax_strategy       : Literal['auto', 'percentile', 'absolute'] = 'auto'
    percentile_low      : float             = 2.0
    percentile_high     : float             = 98.0
    # fontsizes
    fontsize_label      : int               = 10
    fontsize_tick       : int               = 8
    fontsize_title      : int               = 12
    fontsize_annotation : int               = 8
    # line styles
    marker              : str               = 'o'
    markersize          : float             = 5.0
    linewidth           : float             = 1.5
    alpha               : float             = 0.8

@dataclass
class KSpaceConfig:
    """Configuration for k-space plotting."""
    grid_n              : int   = 220
    interp_method       : Literal['linear', 'cubic', 'nearest'] = 'linear'
    mask_outside_bz     : bool  = True
    show_discrete_points: bool  = True
    point_size          : float = 10.0
    point_alpha         : float = 0.35
    draw_bz_outline     : bool  = True
    label_high_symmetry : bool  = True
    ws_shells           : int   = 1
    blob_radius_factor  : float = 2.5
    imshow_interp       : str   = 'bilinear'

@dataclass
class KPathConfig:
    """Configuration for k-path extraction and plotting."""
    path                : Optional[Union[str, List[str]]] = None  # None = use lattice default
    points_per_seg      : Optional[int]     = None  # None = auto-detect from k-grid
    auto_pps_factor     : float             = 0.5  # sqrt(Nk) * factor
    auto_pps_min        : int               = 20
    tick_format         : Literal['labels', 'fractional', 'distance'] = 'labels'
    show_separators     : bool              = True
    separator_style     : dict              = field(default_factory=lambda: {"ls": "--", "lw": 1.0, "alpha": 0.35})
    tolerance           : Optional[float]   = None  # Tolerance for k-point selection (None = auto from k-spacing)

@dataclass
class SpectralConfig:
    """Configuration for spectral function plotting."""
    omega_grid          : Optional[np.ndarray] = None  # Custom omega grid
    broadening_type     : Literal['none', 'lorentzian', 'gaussian'] = 'lorentzian'
    eta                 : float             = 0.1  # Broadening parameter
    normalization       : Literal['none', 'per_k', 'global', 'sum_rule'] = 'none'
    sum_rule_target     : Optional[float]   = None  # For sum rule enforcement
    energy_shift        : float             = 0.0  # Subtract E0 or mu
    log_scale           : bool              = False
    omega_label         : str               = r'$\omega$'
    omega_units         : str               = ''
    intensity_label     : str               = r'$A(\mathbf{k},\omega)$'
    vmin_omega          : Optional[float]   = None
    vmax_omega          : Optional[float]   = None

# ==============================================================================
# HELPER FUNCTIONS: PHYSICS TRANSFORMS
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
    points  = np.asarray(points, dtype=float)
    p1      = np.asarray(p1, dtype=float).ravel()
    p2      = np.asarray(p2, dtype=float).ravel()
    
    if points.ndim == 1:
        points = points[None, :]
    
    # Vector from p1 to p2
    v       = p2 - p1
    v_len_sq= np.dot(v, v)
    
    if v_len_sq < 1e-14:
        # Degenerate segment (p1 ≈ p2)
        return np.linalg.norm(points - p1[None, :], axis=1)
    
    # Project each point onto line
    # t = (points - p1) · v / |v|²
    dp      = points - p1[None, :]
    t       = np.dot(dp, v) / v_len_sq
    
    # Clamp to [0,1] to stay on segment
    t       = np.clip(t, 0.0, 1.0)
    
    # Closest point on segment
    closest = p1[None, :] + t[:, None] * v[None, :]
    
    # Distance
    dist    = np.linalg.norm(points - closest, axis=1)
    
    return dist

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

def compute_spectral_broadening(
        energies        : np.ndarray,
        weights         : np.ndarray,
        omega_grid      : np.ndarray,
        eta             : float = 0.1,
        kind            : Literal['lorentzian', 'gaussian', 'none'] = 'lorentzian'
    ) -> np.ndarray:
    """
    Apply spectral broadening to discrete delta peaks (ED spectra).
    
    Converts discrete excitation spectrum into smooth spectral function:
    A(ω) = Σ_n w_n * f(ω - E_n)
    
    where f is Lorentzian or Gaussian kernel.
    
    Parameters
    ----------
    energies : (N,) array
        Excitation energies (delta peak positions)
    weights : (N,) array
        Spectral weights (delta peak heights)
    omega_grid : (Nω,) array
        Frequency/energy grid for output
    eta : float
        Broadening parameter (FWHM for Lorentzian, std for Gaussian)
    kind : str
        Broadening kernel: 'lorentzian', 'gaussian', or 'none'
        
    Returns
    -------
    spectrum : (Nω,) array
        Broadened spectral function
        
    Notes
    -----
    - Lorentzian: L(ω) = (1/Pi) * (η / ((ω-E)² + η²))
    - Gaussian: G(ω) = (1/√(2Piη²)) * exp(-(ω-E)²/(2η²))
    """
    energies    = np.asarray(energies, dtype=float).ravel()
    weights     = np.asarray(weights, dtype=float).ravel()
    omega_grid  = np.asarray(omega_grid, dtype=float).ravel()
    
    if len(energies) == 0 or len(weights) == 0:
        return np.zeros_like(omega_grid)
    
    if len(energies) != len(weights):
        raise ValueError(f"energies and weights must have same length: {len(energies)} vs {len(weights)}")
    
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    
    # Filter out invalid values
    valid_mask  = np.isfinite(energies) & np.isfinite(weights)
    energies    = energies[valid_mask]
    weights     = weights[valid_mask]
    
    if len(energies) == 0:
        return np.zeros_like(omega_grid)
    
    # Vectorized broadening: omega[Nω, None] - energies[None, N] -> (Nω, N)
    omega_diff  = omega_grid[:, None] - energies[None, :]
    
    if kind == 'lorentzian':
        # Lorentzian: L(ω-E) = (1/Pi) * η / ((ω-E)² + η²)
        kernel      = (eta / np.pi) / (omega_diff**2 + eta**2)
    elif kind == 'gaussian':
        # Gaussian: G(ω-E) = (1/√(2Piη²)) * exp(-(ω-E)²/(2η²))
        norm        = 1.0 / (np.sqrt(2 * np.pi) * eta)
        kernel      = norm * np.exp(-omega_diff**2 / (2 * eta**2))
    elif kind == 'none':
        # No broadening - just return histogram
        spectrum    = np.zeros_like(omega_grid)
        for E, w in zip(energies, weights):
            idx     = np.argmin(np.abs(omega_grid - E))
            spectrum[idx] += w
        return spectrum
    else:
        raise ValueError(f"Unknown broadening kind: {kind}")
    
    # Sum weighted kernels: A(ω) = Σ_n w_n * kernel(ω - E_n)
    spectrum        = np.sum(weights[None, :] * kernel, axis=1)
    
    return spectrum

def extract_spectral_data(
        result,
        key                 : str,
        state_idx           : Optional[int]         = None,
        component           : Optional[str]         = None,
        reshape_to_komega   : bool                  = True,
        omega_key           : Optional[str]         = '/omega', 
        kvectors_key        : Optional[str]         = '/kvectors',
        kvectors            : Optional[np.ndarray]  = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract spectral function data from result object.
    
    Handles various common storage layouts and reshapes to canonical form:
    - omega: (Nw,)
    - k_vectors: (Nk, D)
    - data: (Nk, Nw)
    
    Parameters
    ----------
    result : Result object
        ED result containing spectral data
    key : str
        Data key (e.g., 'akw', 'spectral/skw')
    state_idx : int, optional
        Which state to extract if multiple
    component : str, optional
        Which component (e.g., 'xx', 'zz') for multi-component data
    reshape_to_komega : bool
        If True, ensure output is (Nk, Nω); otherwise keep original shape
        
    Returns
    -------
    omega : (Nω,) array
    k_vectors : (Nk, D) array
    data : (Nk, Nω) array
    """
    # Try to extract data
    data_raw    = result.get(key, None)
    if data_raw is None:
        raise ValueError(f"Key '{key}' not found in result")
    
    # Extract omega grid
    omega       = result.get(key + omega_key, None)
    if omega is None:
        omega   = result.get('omega', None)
    if omega is None:
        # Fallback: create uniform grid
        omega   = np.arange(data_raw.shape[-1])
    
    # Extract k-vectors (if available)
    k_vectors   = kvectors if kvectors is not None else result.get(kvectors_key, None)
    if k_vectors is None:
        k_vectors   = np.zeros((data_raw.shape[0], 3))
    
    # Convert data to array
    data        = np.asarray(data_raw, dtype=complex)
    
    # Handle state selection
    if state_idx is not None and data.ndim >= 3:
        data    = data[..., state_idx]
    
    # Reshape to (Nk, Nw) if requested
    if reshape_to_komega:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            # Flatten extra dimensions
            data = data.reshape(-1, data.shape[-1])
    
    # Take absolute value if complex
    if np.iscomplexobj(data):
        data = np.abs(data)
    
    omega       = np.asarray(omega, float)
    k_vectors   = np.asarray(k_vectors, float)
    
    return omega, k_vectors, data

def select_k_indices(
        lattice         : "Lattice",
        k_vectors       : np.ndarray,
        selector_spec   : Union[str, int, np.ndarray, List[int], dict]
    ) -> dict:
    """
    Unified k-point selection interface.
    
    Parameters
    ----------
    lattice : Lattice
    k_vectors : (Nk, D) array
    selector_spec :
        - "all" or "sum": 
            all k-points
        - int: 
            single k-point by index
        - array (D,): 
            nearest k-point to vector
        - list of int: 
            multiple indices
        - dict with 'path', 'indices', or 'vectors'
        
    Returns
    -------
    selection : dict
        {
            'indices'   : array of selected indices,
            'labels'    : list of labels,
            'distances' : array of path distances (if applicable),
            'type'      : 'all'|'single'|'indices'|'path'
        }
    """
    if isinstance(selector_spec, str):
        if selector_spec.lower() in ('all', 'sum'):
            return {
                'indices'       : np.arange(len(k_vectors)),
                'labels'        : [f'k{i}' for i in range(len(k_vectors))],
                'distances'     : None,
                'type'          : 'all'
            }
    
    elif isinstance(selector_spec, int):
        return {
            'indices'           : np.array([selector_spec]),
            'labels'            : [f'k{selector_spec}'],
            'distances'         : None,
            'type'              : 'single'
        }
    
    elif isinstance(selector_spec, (list, tuple)):
        return {
            'indices'           : np.array(selector_spec),
            'labels'            : [f'k{i}' for i in selector_spec],
            'distances'         : None,
            'type'              : 'indices'
        }
    
    elif isinstance(selector_spec, np.ndarray):
        # Find nearest k-point to vector
        k2      = k_vectors[:, :2]
        vec2    = np.asarray(selector_spec)[:2]
        dists   = np.sum((k2 - vec2)**2, axis=1)
        idx     = np.argmin(dists)
        return {
            'indices'           : np.array([idx]),
            'labels'            : [f'k≈({vec2[0]:.2f},{vec2[1]:.2f})'],
            'distances'         : None,
            'type'              : 'single'
        }
    
    elif isinstance(selector_spec, dict):
        if 'path' in selector_spec:
            # Use lattice path extraction
            # This would call lattice.extract_bz_path_data or similar
            # For now, return placeholder
            return lattice.extract_bz_path_data(
                k_vectors,
                **selector_spec['path']
            )
            
        elif 'indices' in selector_spec:
            return {
                'indices'       : np.array(selector_spec['indices']),
                'labels'        : selector_spec.get('labels', []),
                'distances'     : selector_spec.get('distances', None),
                'type'          : 'indices'
            }
    
    raise ValueError(f"Invalid selector_spec: {selector_spec}")

# ==============================================================================
# PLOTTING PRIMITIVES
# ==============================================================================

def plot_komega_heatmap(
        ax,
        k_dist          : np.ndarray,
        omega           : np.ndarray,
        intensity       : np.ndarray,
        *,
        style           : Optional[PlotStyle] = None,
        spectral_config : Optional[SpectralConfig] = None,
        aspect          : str = 'auto',
        origin          : str = 'lower'
    ):
    """
    Plot intensity(k_path_index, ω) as heatmap.
    
    Parameters
    ----------
    ax : Axes
    k_dist : (Nk_path,)
        Distance along k-path
    omega : (Nω,)
        Energy/frequency grid
    intensity : (Nk_path, Nω)
        Spectral intensity
    """
    if style is None:
        style = PlotStyle()
    if spectral_config is None:
        spectral_config = SpectralConfig()
    
    # Determine vmin/vmax
    vmin = style.vmin if style.vmin is not None else np.nanmin(intensity)
    vmax = style.vmax if style.vmax is not None else np.nanmax(intensity)
    
    if spectral_config.log_scale and vmin <= 0:
        vmin = np.nanmin(intensity[intensity > 0]) if np.any(intensity > 0) else 1e-10
    
    K, Omega = np.meshgrid(k_dist, omega, indexing='ij')
    
    if spectral_config.log_scale:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    
    im = ax.pcolormesh(
        K, Omega, intensity,
        cmap=style.cmap,
        vmin=vmin if not spectral_config.log_scale else None,
        vmax=vmax if not spectral_config.log_scale else None,
        norm=norm,
        shading='auto'
    )
    
    ax.set_ylabel(spectral_config.omega_label)
    ax.set_xlim(k_dist.min(), k_dist.max())
    
    if spectral_config.vmin_omega is not None:
        ax.set_ylim(bottom=spectral_config.vmin_omega)
    if spectral_config.vmax_omega is not None:
        ax.set_ylim(top=spectral_config.vmax_omega)
    
    return im

def plot_kspace_intensity(
        ax,
        k2: np.ndarray,
        intensity: np.ndarray,
        *,
        style: Optional[PlotStyle] = None,
        ks_config: Optional[KSpaceConfig] = None,
        lattice: Optional["Lattice"] = None,
        show_extended_bz: bool = True,
        bz_copies: int = 2
    ):
    """
    Plot intensity(k) in 2D k-space.
    
    Can show:
    - Discrete scatter plot of k-points
    - Interpolated map with optional BZ masking
    - BZ boundary outline
    - High-symmetry point markers in all BZ copies
    - Extended k-space showing multiple Brillouin zones
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    k2 : (Nk, 2) array
        k-point coordinates (first BZ)
    intensity : (Nk,) array
        Values at each k-point
    style : PlotStyle, optional
        Styling configuration
    ks_config : KSpaceConfig, optional
        K-space plotting configuration
    lattice : Lattice, optional
        Lattice object for BZ info and high-symmetry points
    show_extended_bz : bool
        If True, replicate k-points to show multiple BZ copies from -2Pi to 2Pi
    bz_copies : int
        Number of BZ copies in each direction (default 2 for -2Pi to 2Pi)
    """
    if style is None:
        style = PlotStyle()
    if ks_config is None:
        ks_config = KSpaceConfig()
    
    k2_orig     = np.asarray(k2, dtype=float)
    intensity   = np.asarray(intensity, dtype=float)
    
    # ALWAYS interpolate on original k-points only
    k2_for_interp = k2_orig
    intensity_for_interp = intensity
    
    # Determine color scale
    vmin = style.vmin if style.vmin is not None else np.nanmin(intensity)
    vmax = style.vmax if style.vmax is not None else np.nanmax(intensity)
    
    # Interpolated background (optional)
    if ks_config.grid_n > 0:
        # Interpolate on ORIGINAL k-points only
        kx_min, kx_max = k2_for_interp[:, 0].min(), k2_for_interp[:, 0].max()
        ky_min, ky_max = k2_for_interp[:, 1].min(), k2_for_interp[:, 1].max()
        
        pad = 0.05
        kx = np.linspace(kx_min - pad * (kx_max - kx_min), 
                        kx_max + pad * (kx_max - kx_min), 
                        ks_config.grid_n)
        ky = np.linspace(ky_min - pad * (ky_max - ky_min),
                        ky_max + pad * (ky_max - ky_min),
                        ks_config.grid_n)
        KX, KY = np.meshgrid(kx, ky)
        
        try:
            # Interpolate on original k-points
            Z = griddata(k2_for_interp, intensity_for_interp, (KX, KY), method=ks_config.interp_method)
            
            # Don't mask when showing extended BZ
            if ks_config.mask_outside_bz and lattice is not None and not show_extended_bz:
                try:
                    from general_python.lattices.tools.lattice_kspace import ws_bz_mask
                    k1_vec = None
                    k2_vec = None
                    
                    for attr1 in ['k1', 'b1']:
                        if hasattr(lattice, attr1):
                            k1_vec = np.asarray(getattr(lattice, attr1), float).ravel()[:2]
                            break
                    
                    for attr2 in ['k2', 'b2']:
                        if hasattr(lattice, attr2):
                            k2_vec = np.asarray(getattr(lattice, attr2), float).ravel()[:2]
                            break
                    
                    if k1_vec is not None and k2_vec is not None:
                        inside = ws_bz_mask(KX, KY, k1_vec, k2_vec, shells=ks_config.ws_shells)
                        Z = np.where(inside, Z, np.nan)
                except Exception:
                    pass
            
            # Display: tile if extended BZ, otherwise show single image
            if show_extended_bz and lattice is not None:
                try:
                    if hasattr(lattice, 'calculate_reciprocal_vectors'):
                        lattice.calculate_reciprocal_vectors()
                    
                    k1_vec = None
                    k2_vec = None
                    
                    for attr1 in ['k1', 'b1']:
                        if hasattr(lattice, attr1):
                            vec = getattr(lattice, attr1)
                            if vec is not None:
                                k1_vec = np.asarray(vec, float).ravel()[:2]
                                break
                    
                    for attr2 in ['k2', 'b2']:
                        if hasattr(lattice, attr2):
                            vec = getattr(lattice, attr2)
                            if vec is not None:
                                k2_vec = np.asarray(vec, float).ravel()[:2]
                                break
                    
                    if k1_vec is not None and k2_vec is not None:
                        # Tile: show same interpolated image at each G shift
                        for m in range(-bz_copies, bz_copies + 1):
                            for n in range(-bz_copies, bz_copies + 1):
                                G               = m * k1_vec + n * k2_vec
                                extent_shifted  = (
                                    kx[0] + G[0], kx[-1] + G[0],
                                    ky[0] + G[1], ky[-1] + G[1]
                                )
                                ax.imshow(
                                    Z,
                                    extent=extent_shifted,
                                    origin='lower',
                                    cmap=style.cmap,
                                    vmin=vmin,
                                    vmax=vmax,
                                    interpolation=ks_config.imshow_interp,
                                    alpha=0.9,
                                    zorder=1
                                )
                    else:
                        ax.imshow(Z, extent=(kx[0], kx[-1], ky[0], ky[-1]),
                                origin='lower', cmap=style.cmap, vmin=vmin, vmax=vmax,
                                interpolation=ks_config.imshow_interp, alpha=0.9)
                except Exception:
                    ax.imshow(Z, extent=(kx[0], kx[-1], ky[0], ky[-1]),
                            origin='lower', cmap=style.cmap, vmin=vmin, vmax=vmax,
                            interpolation=ks_config.imshow_interp, alpha=0.9)
            else:
                # Not extended: show single image
                ax.imshow(
                    Z,
                    extent=(kx[0], kx[-1], ky[0], ky[-1]),
                    origin='lower',
                    cmap=style.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation=ks_config.imshow_interp,
                    alpha=0.9
                )
        except Exception:
            pass
    
    # Prepare k-points for scatter plot (extend if needed)
    k2_scatter = k2_orig
    intensity_scatter = intensity
    
    if show_extended_bz and lattice is not None and ks_config.show_discrete_points:
        try:
            # Get reciprocal vectors
            if hasattr(lattice, 'calculate_reciprocal_vectors'):
                lattice.calculate_reciprocal_vectors()
            
            k1_vec = None
            k2_vec = None
            
            for attr1 in ['k1', 'b1']:
                if hasattr(lattice, attr1):
                    vec = getattr(lattice, attr1)
                    if vec is not None:
                        k1_vec = np.asarray(vec, float).ravel()[:2]
                        break
            
            for attr2 in ['k2', 'b2']:
                if hasattr(lattice, attr2):
                    vec = getattr(lattice, attr2)
                    if vec is not None:
                        k2_vec = np.asarray(vec, float).ravel()[:2]
                        break
            
            if k1_vec is not None and k2_vec is not None:
                # Replicate scatter points for each BZ copy
                k2_list = []
                intensity_list = []
                
                for m in range(-bz_copies, bz_copies + 1):
                    for n in range(-bz_copies, bz_copies + 1):
                        G = m * k1_vec + n * k2_vec
                        k2_shifted = k2_orig + G[np.newaxis, :]
                        k2_list.append(k2_shifted)
                        intensity_list.append(intensity)
                
                k2_scatter = np.vstack(k2_list)
                intensity_scatter = np.concatenate(intensity_list)
        except Exception:
            pass
    
    # Discrete points overlay
    if ks_config.show_discrete_points:
        ax.scatter(
            k2_scatter[:, 0], k2_scatter[:, 1],
            c=intensity_scatter,
            cmap=style.cmap,
            vmin=vmin,
            vmax=vmax,
            s=ks_config.point_size,
            alpha=ks_config.point_alpha,
            edgecolors='none',
            zorder=10
        )
    
    # Draw BZ outline - skip when showing extended BZ
    if ks_config.draw_bz_outline and lattice is not None and not show_extended_bz:
        try:
            _draw_bz_boundary(ax, lattice)
        except Exception:
            pass
    
    # Label high-symmetry points in all BZ copies
    if ks_config.label_high_symmetry and lattice is not None:
        try:
            _label_high_symmetry_points_extended(ax, lattice, bz_copies if show_extended_bz else 0, show_labels=True)
        except Exception:
            pass
    
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$', fontsize=style.fontsize_label)
    ax.set_ylabel(r'$k_y$', fontsize=style.fontsize_label)
    ax.tick_params(labelsize=style.fontsize_tick)
    
    # Set strict limits to -2Pi to 2Pi when showing extended BZ
    if show_extended_bz and bz_copies >= 1:
        ax.set_xlim(-bz_copies * np.pi / 2, bz_copies * np.pi / 2)
        ax.set_ylim(-bz_copies * np.pi / 2, bz_copies * np.pi / 2)
    
    # Format tick labels as multiples of Pi
    _format_pi_ticks(ax, axis='both')

def _draw_bz_boundary(ax, lattice: "Lattice"):
    """
    Draw the first Brillouin zone boundary.
    
    For 2D lattices, draws the hexagon/rectangle boundary using
    reciprocal lattice vectors.
    """
    # Get reciprocal vectors - try multiple attribute names
    k1_vec = None
    k2_vec = None
    
    for attr1 in ['k1', 'b1', 'avec']:
        if hasattr(lattice, attr1):
            k1_vec = np.asarray(getattr(lattice, attr1), float).ravel()[:2]
            break
    
    for attr2 in ['k2', 'b2', 'bvec']:
        if hasattr(lattice, attr2):
            k2_vec = np.asarray(getattr(lattice, attr2), float).ravel()[:2]
            break
    
    if k1_vec is None or k2_vec is None:
        return
    
    # Generate BZ boundary points using perpendicular bisectors
    # For a 2D reciprocal lattice, the BZ is bounded by planes
    # perpendicular to G at |G|/2 for reciprocal lattice vectors G
    
    # Simple approach: find BZ vertices by intersecting bisector planes
    # For common lattices (square, hexagonal), we can use the corners
    
    # Generate corner candidates from ±k1/2 ± k2/2 and their combinations
    corners = []
    for m in [-1, 0, 1]:
        for n in [-1, 0, 1]:
            if m == 0 and n == 0:
                continue
            G = m * k1_vec + n * k2_vec
            # Perpendicular bisector passes through G/2
            corners.append(G / 2)
    
    corners = np.array(corners)
    
    # Find convex hull to get BZ boundary
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(corners)
        boundary_points = corners[hull.vertices]
        
        # Close the polygon
        boundary_points = np.vstack([boundary_points, boundary_points[0]])
        
        ax.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            color='white',
            linewidth=2.0,
            linestyle='-',
            alpha=0.8,
            zorder=20
        )
    except Exception:
        # Fallback: just draw a hexagon or rectangle based on k1, k2
        # For hexagonal BZ (like honeycomb):
        # vertices are at ±k1/2, ±k2/2, ±(k1-k2)/2, etc.
        pass

def _label_high_symmetry_points(ax, lattice: "Lattice"):
    """
    Add markers and labels for high-symmetry points (Gamma, K, M, etc.).
    """
    try:
        hs_points = lattice.high_symmetry_points()
        if hs_points is None:
            return
        
        # Get reciprocal basis to convert fractional to Cartesian
        k1_vec = None
        k2_vec = None
        
        for attr1 in ['k1', 'b1', 'avec']:
            if hasattr(lattice, attr1):
                k1_vec = np.asarray(getattr(lattice, attr1), float).ravel()
                if len(k1_vec) < 3:
                    k1_vec = np.pad(k1_vec, (0, 3 - len(k1_vec)))
                break
        
        for attr2 in ['k2', 'b2', 'bvec']:
            if hasattr(lattice, attr2):
                k2_vec = np.asarray(getattr(lattice, attr2), float).ravel()
                if len(k2_vec) < 3:
                    k2_vec = np.pad(k2_vec, (0, 3 - len(k2_vec)))
                break
        
        if k1_vec is None or k2_vec is None:
            return
        
        k3_vec = np.array([0.0, 0.0, 1.0])  # Dummy for 2D
        
        for point in hs_points:
            # Convert fractional to Cartesian
            k_cart = point.to_cartesian(k1_vec, k2_vec, k3_vec)
            kx, ky = k_cart[0], k_cart[1]
            
            # Draw marker
            ax.plot(
                kx, ky,
                marker='o',
                markersize=8,
                markerfacecolor='none',
                markeredgecolor='white',
                markeredgewidth=2.0,
                zorder=25
            )
            
            # Add label with slight offset
            ax.text(
                kx, ky,
                point.latex_label,
                color='white',
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='bottom',
                zorder=26
            )
    except Exception:
        pass


def _format_pi_ticks(ax, axis='both'):
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
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))

def _label_high_symmetry_points_extended(ax, lattice: "Lattice", bz_copies: int = 2, show_labels: bool = True):
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
    """
    try:
        hs_points = lattice.high_symmetry_points()
        if hs_points is None:
            return
        
        # Get reciprocal basis
        k1_vec = None
        k2_vec = None
        
        for attr1 in ['k1', 'b1', 'avec']:
            if hasattr(lattice, attr1):
                k1_vec = np.asarray(getattr(lattice, attr1), float).ravel()
                if len(k1_vec) < 3:
                    k1_vec = np.pad(k1_vec, (0, 3 - len(k1_vec)))
                break
        
        for attr2 in ['k2', 'b2', 'bvec']:
            if hasattr(lattice, attr2):
                k2_vec = np.asarray(getattr(lattice, attr2), float).ravel()
                if len(k2_vec) < 3:
                    k2_vec = np.pad(k2_vec, (0, 3 - len(k2_vec)))
                break
        
        if k1_vec is None or k2_vec is None:
            return
        
        k3_vec = np.array([0.0, 0.0, 1.0])  # Dummy for 2D
        k1_2d = k1_vec[:2]
        k2_2d = k2_vec[:2]
        
        # Plot high-symmetry points - ONLY in center BZ to avoid clutter
        # When showing extended zones, only mark center (m=0, n=0)
        m_range = range(0, 1) if bz_copies > 0 else range(0, 1)
        n_range = range(0, 1) if bz_copies > 0 else range(0, 1)
        
        for m in m_range:
            for n in n_range:
                G = m * k1_2d + n * k2_2d
                
                for point in hs_points:
                    # Convert fractional to Cartesian
                    k_cart = point.to_cartesian(k1_vec, k2_vec, k3_vec)
                    kx, ky = k_cart[0] + G[0], k_cart[1] + G[1]
                    
                    # Draw marker
                    ax.plot(
                        kx, ky,
                        marker='o',
                        markersize=8,
                        markerfacecolor='yellow',
                        markeredgecolor='black',
                        markeredgewidth=1.5,
                        zorder=25
                    )
                    
                    # Add label - always show for center BZ
                    if show_labels:
                        ax.text(
                            kx, ky + 0.15,  # Small offset above point
                            point.latex_label,
                            color='black',
                            fontsize=11,
                            fontweight='bold',
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                            zorder=26
                        )
    except Exception:
        pass

# ==============================================================================
# HIGH-LEVEL SPECTRAL PLOTTERS
# ==============================================================================

def plot_spectral_function(
        directory: str,
        param_name: str,
        lattice: "Lattice",
        *,
        x_parameters: list,
        y_parameters: list,
        x_param: str = 'J',
        y_param: str = 'hx',
        filters=None,
        state_idx: int = 0,
        k_selector: Union[str, int, dict] = "sum",
        spectral_config: Optional[SpectralConfig] = None,
        style: Optional[PlotStyle] = None,
        figsize_per_panel: tuple = (4, 3.5),
        title: str = '',
        **kwargs
    ):
    """
    Plot spectral function A(k,ω) or S(k,ω) from ED results.
    
    Parameters
    ----------
    k_selector :
        - "sum": sum _k A(k,ω)
        - "path": A(k_path, ω) heatmap along high-symmetry path
        - int: A(k_i, ω) line plot for single k
        - list[int]: multiple k-point line cuts
        
    Examples
    --------
    # Summed spectral function
    fig, axes = plot_spectral_function(
        directory='./ed_data',
        param_name='/spectral/akw',
        lattice=lattice,
        k_selector="sum",
        x_parameters=[1.0],
        y_parameters=[0.0, 0.5, 1.0]
    )
    
    # Spectral function along k-path
    fig, axes = plot_spectral_function(
        directory='./ed_data',
        param_name='/spectral/akw',
        lattice=lattice,
        k_selector={'path': ['Gamma', 'K', 'M', 'Gamma']},
        x_parameters=[1.0],
        y_parameters=[0.0]
    )
    """
    if spectral_config is None:
        spectral_config = SpectralConfig()
    if style is None:
        style = PlotStyle()
    
    # Load results
    lx, ly = lattice.lx, lattice.ly
    results = load_results(
        data_dir=directory,
        filters=filters,
        lx=lx, ly=ly,
        logger=kwargs.get('logger', None)
    )
    
    if not results:
        return None, None
    
    # Setup grid
    unique_x = sorted(set(x_parameters))
    unique_y = sorted(set(y_parameters))
    n_rows, n_cols = len(unique_y), len(unique_x)
    
    fig, axes, _, _ = PlotDataHelpers.create_subplot_grid(
        n_panels=n_rows * n_cols,
        max_cols=n_cols,
        figsize_per_panel=figsize_per_panel,
        sharex=True,
        sharey=True
    )
    axes_grid = np.array(axes).reshape((n_rows, n_cols))
    
    for ii, y_val in enumerate(unique_y):
        for jj, x_val in enumerate(unique_x):
            ax = axes_grid[ii, jj]
            
            # Find matching result
            subset = [r for r in results 
                     if abs(r.params.get(x_param, np.nan) - x_val) < 1e-5
                     and abs(r.params.get(y_param, np.nan) - y_val) < 1e-5]
            
            if not subset:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.axis('off')
                continue
            
            result = subset[0]
            
            try:
                omega, k_vectors, akw = extract_spectral_data(
                    result, param_name, state_idx=state_idx
                )
            except:
                ax.text(0.5, 0.5, 'Data Error', ha='center', va='center',
                       transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Handle k-selection
            if k_selector == "sum":
                # Sum over all k
                intensity_omega = np.sum(akw, axis=0)
                ax.plot(omega, intensity_omega, lw=style.linewidth, 
                       color='C0', alpha=style.alpha)
                ax.set_ylabel(spectral_config.intensity_label)
                ax.set_xlabel(spectral_config.omega_label)
                ax.grid(alpha=0.3)
                
            elif isinstance(k_selector, int):
                # Single k-point
                if k_selector < len(k_vectors):
                    intensity_omega = akw[k_selector, :]
                    ax.plot(omega, intensity_omega, lw=style.linewidth,
                           color='C0', alpha=style.alpha, marker=style.marker,
                           ms=style.markersize)
                    ax.set_ylabel(spectral_config.intensity_label)
                    ax.set_xlabel(spectral_config.omega_label)
                    ax.grid(alpha=0.3)
            
            elif isinstance(k_selector, dict) and 'path' in k_selector:
                # k-path heatmap
                try:
                    kpath_cfg = KPathConfig(**k_selector) if isinstance(k_selector, dict) else KPathConfig()
                    
                    # Extract path data
                    k_frac = lattice.kvectors_frac if hasattr(lattice, 'kvectors_frac') else None
                    if k_frac is None:
                        k_frac = np.zeros_like(k_vectors)
                    
                    # Get path result
                    path_result = lattice.extract_bz_path_data(
                        k_vectors=k_vectors,
                        k_vectors_frac=k_frac,
                        values=akw,
                        path=kpath_cfg.path,
                        points_per_seg=kpath_cfg.points_per_seg or 50,
                        return_result=True
                    )
                    
                    k_dist = np.asarray(path_result.k_dist)
                    intensity_kw = np.asarray(path_result.values)  # (Npath, Nω)
                    
                    plot_komega_heatmap(
                        ax, k_dist, omega, intensity_kw,
                        style=style,
                        spectral_config=spectral_config
                    )
                    
                    # Add high-symmetry ticks
                    try:
                        xs = np.asarray(path_result.label_positions)
                        ls = list(path_result.label_texts)
                        ax.set_xticks(xs)
                        ax.set_xticklabels(ls)
                        for xv in xs:
                            ax.axvline(xv, **kpath_cfg.separator_style)
                    except:
                        pass
                        
                except Exception as e:
                    ax.text(0.5, 0.5, f'Path Error', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.axis('off')
            
            # Annotation
            Plotter.set_annotate_letter(
                ax, iter=0, x=0.05, y=0.9, boxaround=False,
                addit=f'{x_param}={x_val:.2g}, {y_param}={y_val:.2g}'
            )
    
    if title:
        fig.suptitle(title, fontsize=style.fontsize_title)
    
    return fig, axes_grid

def plot_spectral_slice_at_omega(
        ax,
        lattice: "Lattice",
        k_vectors: np.ndarray,
        omega_grid: np.ndarray,
        akw: np.ndarray,
        omega0: float,
        omega_window: float = 0.0,
        *,
        style: Optional[PlotStyle] = None,
        ks_config: Optional[KSpaceConfig] = None
    ):
    """
    Plot spectral intensity in k-space at fixed energy.
    
    I(k) = A(k, ω0) or ∫_{ω0-Δ}^{ω0+Δ} A(k,ω) dω
    
    Parameters
    ----------
    omega0 : float
        Central energy
    omega_window : float
        Integration window (if > 0)
    """
    if style is None:
        style = PlotStyle()
    if ks_config is None:
        ks_config = KSpaceConfig()
    
    # Find omega indices
    if omega_window > 0:
        mask = np.abs(omega_grid - omega0) <= omega_window
        if not np.any(mask):
            # Fallback to nearest
            idx = np.argmin(np.abs(omega_grid - omega0))
            intensity_k = akw[:, idx]
        else:
            intensity_k = np.trapz(akw[:, mask], omega_grid[mask], axis=1)
    else:
        idx = np.argmin(np.abs(omega_grid - omega0))
        intensity_k = akw[:, idx]
    
    k2 = k_vectors[:, :2]
    plot_kspace_intensity(
        ax, k2, intensity_k,
        style=style,
        ks_config=ks_config,
        lattice=lattice
    )
    
    title = f'ω = {omega0:.3f}'
    if omega_window > 0:
        title += f' ± {omega_window:.3f}'
    ax.set_title(title, fontsize=style.fontsize_title)

# ==============================================================================
# Plotting functions
# ==============================================================================

# a) Phase diagram plotter

def plot_phase_diagram_states(
        directory           : str, 
        param_name          : str,
        x_param             : str,
        y_param             : str, 
        filters             = None, 
        *,
        lx                  = None, 
        ly                  = None,
        lz                  = None,
        Ns                  = None,
        post_process_func   = None,
        # other
        figsize_per_panel   : tuple = (6, 5), 
        nstates             = 4,
        # parameter labels
        param_labels        : dict  = {},
        param_fun           : callable          = lambda r, param_name: r.get(param_name, []),
        param_x_fun         : callable          = None,
        param_lbl           : Optional[str]     = None,
        # plot limits
        xlim                : Optional[tuple]   = None,
        ylim                : Optional[tuple]   = None,
        # colormap
        vmin                = None, 
        vmax                = None, 
        # plot settings
        ylabel              : Optional[str]     = None,
        xlabel              : Optional[str]     = None,
        cmap                = 'viridis', 
        logger              : Optional['Logger'] = None,
        save                : bool = False,
        **kwargs):
    
    """Plot phase diagram from ED results in specified directory."""

    results = load_results(data_dir=directory, filters=filters, lx=lx, ly=ly, lz=lz, Ns=Ns, post_process_func=post_process_func, logger=logger)
    if len(results) == 0:
        if logger: logger.warning("No results found for phase diagram.")
        return None, None
    
    # extract unique parameters
    x_vals, y_vals, unique_x, unique_y  = PlotDataHelpers.extract_parameter_arrays(results, x_param=x_param, y_param=y_param)
    plot_map                            = len(unique_x) > 1 and len(unique_y) > 1
    if logger:                          logger.info(f"Plot type: {'Colormap' if plot_map else 'Line plots'}", color='cyan')
    
    if nstates <= 1:
        ncols, nrows, npanels = 1, 1, 1
    elif nstates <= 3:
        ncols   = 1 if not plot_map else nstates
        nrows   = 1
        npanels = nstates if plot_map else 1
    else:
        ncols   = 1 if not plot_map else nstates // 2
        nrows   = 1 if not plot_map else ((nstates + ncols - 1) // ncols)
        npanels = nstates if plot_map else 1
        
    fig, axes, _, _                     = PlotDataHelpers.create_subplot_grid(n_panels=npanels , max_cols=ncols, figsize_per_panel=figsize_per_panel, sharex=True, sharey=True)
    axes                                = axes.flatten()
    
    if xlabel is None:
        xlabel                          = param_labels.get(x_param, x_param) if len(unique_x) > 1 else param_labels.get(y_param, y_param)
    if ylabel is None:
        ylabel                          = param_labels.get(y_param, y_param) if len(unique_y) > 1 else param_labels.get(x_param, x_param)
    
    if plot_map:
        labelcond = lambda state_idx: {'x': state_idx // ncols == nrows - 1, 'y': state_idx % ncols == 0}
    else:
        labelcond = lambda state_idx: {'x': True, 'y': True}
    
    if vmin is None or vmax is None:
        vmin_n, vmax_n                  = PlotDataHelpers.determine_vmax_vmin(results, param_name, param_fun, nstates) if vmin is None or vmax is None else (vmin, vmax)
        vmin                            = vmin_n if vmin is None else vmin
        vmax                            = vmax_n if vmax is None else vmax
        
    letter_x, letter_y                  = kwargs.pop('letter_x', 0.05), kwargs.pop('letter_y', 0.85) # annotation position

    # a) Only X Variation (Line Plot)
    if len(unique_x) > 1 and len(unique_y) == 1 or len(unique_x) == 1 and len(unique_y) > 1:
        param_plot  = x_param if len(unique_x) > 1 else y_param
        ax          = axes[0]
        label       = xlabel if len(unique_x) > 1 else ylabel
        getcolor    = Plotter.get_colormap(values=np.linspace(0, nstates, nstates), cmap=cmap, elsecolor='black', get_mappable=False)[0]
        
        for ii in range(nstates):
            x_plot, y_plot = [], []
            for r in results:
                y   = param_fun(r, param_name)
                x   = param_x_fun(r, param_plot) if param_x_fun is not None else r.params.get(param_plot, 0.0)
                if len(y) > ii:
                    x_plot.append(x)
                    y_plot.append(y[ii])
            
            sort_idx    = np.argsort(x_plot)
            x_plot      = np.array(x_plot)[sort_idx]
            y_plot      = np.array(y_plot)[sort_idx]
            Plotter.plot(ax, x=x_plot, y=y_plot, ls='-', marker='o', ms=4, color=getcolor(ii), label=rf'$|\Psi_{{{ii}}}\rangle$', zorder=100-ii)
        
        Plotter.set_ax_params(ax, xlabel=label, ylabel=param_lbl if param_lbl is not None else param_labels.get(param_name, param_name), xlim=xlim, ylim=ylim, yscale=kwargs.get('yscale', 'linear'), xscale=kwargs.get('xscale', 'linear'),)
        Plotter.set_annotate_letter(ax, iter=0, x=letter_x, y=letter_y, boxaround=False)
        Plotter.set_legend(ax, loc=kwargs.get('legend_loc', 'lower right'), fontsize=kwargs.get('legend_fontsize', 8))
        Plotter.grid(ax, alpha=0.3)

    # b) Colormap - Both X and Y Variation
    elif len(unique_x) > 1 and len(unique_y) > 1:
        xi                          = np.linspace(min(unique_x), max(unique_x), 200)
        yi                          = np.linspace(min(unique_y), max(unique_y), 200)
        Xi, Yi                      = np.meshgrid(xi, yi)
        scale_type                  = kwargs.get('cbar_scale', 'linear')
        getcolor, _, _, mappable    = Plotter.get_colormap(vmin=vmin, vmax=vmax, cmap=cmap, elsecolor='black', get_mappable=True, scale=scale_type)
        
        for ii in range(nstates):
            ax                              = axes[ii]
            x_scatter, y_scatter, z_scatter = [], [], []
            
            for r in results:
                val     = param_fun(r, param_name)
                # X param
                x_val   = param_x_fun(r, x_param) if param_x_fun is not None else r.params.get(x_param, 0.0)
                
                # Y param
                y_val   = r.params.get(y_param, 0.0)

                if isinstance(val, (list, np.ndarray)) and len(val) > ii:
                    x_scatter.append(x_val)
                    y_scatter.append(y_val)
                    z_scatter.append(val[ii])
                    
                elif isinstance(val, (int, float)) and ii == 0:
                    x_scatter.append(x_val)
                    y_scatter.append(y_val)
                    z_scatter.append(val)
            
            if len(x_scatter) < 3: # Need at least 3 points for griddata
                continue

            try:
                if kwargs.get('gridmethod', 'cubic') is None:
                    # do not use griddata, just scatter points where they are, color by value
                    for x, y, z in zip(x_scatter, y_scatter, z_scatter):
                        ax.scatter(x, y, color=getcolor(z), s=100, edgecolor=None, linewidth=0.5, marker='o')
                else:
                    Zi = griddata((x_scatter, y_scatter), z_scatter, (Xi, Yi), method=kwargs.get('gridmethod', 'cubic'), rescale=True)
                    cf = ax.contourf(Xi, Yi, Zi, cmap=cmap, extend='both', vmin=vmin, vmax=vmax)
            
                    if not np.all(np.isnan(Zi)):
                        cs = ax.contour(Xi, Yi, Zi, levels=kwargs.get('levels', 10), colors=kwargs.get('c_levels', 'white'), linewidths=0.5, alpha=0.3, vmin=vmin, vmax=vmax)
                        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2e', colors=kwargs.get('c_labels', 'white'))
            except Exception as e:
                if logger: logger.error(f"Error interpolating data for state {ii}: {e}")

            text_color      = kwargs.get('text_color', 'white')
            text_fontsize   = kwargs.get('text_fontsize', 8)
            Plotter.set_ax_params(ax, xlabel=xlabel, ylabel=ylabel, labelCond=labelcond(ii))
            Plotter.set_annotate_letter(ax, iter=ii, x=letter_x, y=letter_y, boxaround=False, color=text_color, addit=rf'$|\Psi_{{{ii}}}\rangle$', fontsize=text_fontsize)
        
        # Colorbar for all panels
        cbar_pos    = kwargs.get('cbar_pos', [0.92, 0.15, 0.02, 0.7])
        cbar_scale  = kwargs.pop('cbar_scale', 'linear')
        vmin        = vmin if cbar_scale == 'linear' else None
        Plotter.add_colorbar(fig=fig, pos=cbar_pos, mappable=mappable, cmap=cmap, scale=cbar_scale,
                    vmin=vmin, vmax=vmax, label=param_labels.get(param_name, param_name) if param_lbl is None else param_lbl,
                    extend=None, format='%.1e' if cbar_scale == 'log' else '%.2f', remove_pdf_lines=True
                )
    else:
        if logger: logger.warning("Insufficient parameter variation for phase diagram plot.")
        return None, None
        
    for ax in axes:
        Plotter.set_tickparams(ax)
        
    title = kwargs.get('title', '')
    if title:
        fig.suptitle(title, fontsize=kwargs.get('title_fontsize', 12))

    Plotter.hide_unused_panels(axes, npanels)
    
    if save:
        PlotDataHelpers.savefig(fig, directory, param_name, x_param, y_param if plot_map else None, **kwargs)
        
    return fig, axes

# b) Correlation grid plotter

def plot_bz_path_from_corr(
        ax,
        lattice         : "Lattice",
        corr_matrix     : np.ndarray,
        *,
        path                                    = None,
        points_per_seg  : int                   = None,
        value_label     : str                   = r"$S(\mathbf{k})$",
        line_kw         : dict                  = None,
        hsline_kw       : dict                  = None,
        print_vectors   : bool                  = False,
        kpath_config    : Optional[KPathConfig] = None,
        style           : Optional[PlotStyle]   = None,
    ):
    r"""
    Plot correlation-derived structure factor along high-symmetry path.
    
    **TOLERANCE-BASED VERSION** using actual k-points only:
    - compute_structure_factor_from_corr() for S(k) calculation
    - NO interpolation: uses only discrete k-points from lattice.kvectors
    - Selects k-points within tolerance of path segments
    - Configurable via KPathConfig and PlotStyle dataclasses
    
    Computes (site-based) structure factor:
        S(k) = (1/Ns) sum_{i,j} C_{ij} exp[-i k . (r_i - r_j)]
    
    Algorithm:
    1. Compute S(k) at ALL discrete k-points
    2. Build path segments from high-symmetry points
    3. For each segment, find k-points within tolerance
    4. Sort by distance along path
    5. Plot S(k) at exact k-point values (no interpolation)
    
    Parameters
    ----------
    ax : matplotlib Axes
    lattice : Lattice
        Must have .rvectors, .kvectors, .high_symmetry_points()
    corr_matrix : (Ns,Ns) array
        Correlation matrix in site basis
    path : optional
        Path specification (None = lattice default, or list like ['Gamma','K','M'])
    kpath_config : KPathConfig, optional
        Configuration object (overrides individual parameters)
    style : PlotStyle, optional
        Styling configuration
    
    Returns
    -------
    result : KPathResult
        Path data with k_dist, values, label_positions, label_texts
        
    Notes
    -----
    Uses ONLY discrete k-points (no interpolation). Tolerance-based selection:
    1. Computes S(k) at ALL discrete k-points
    2. Builds path segments from high-symmetry points
    3. For each segment, finds k-points within tolerance of segment
    4. Sorts selected k-points by distance along path
    5. Plots S(k) at exact k-point values
    
    Tolerance is auto-determined from k-grid spacing if not specified in KPathConfig.
    """
    
    # Setup configuration
    if kpath_config is None:
        kpath_config = KPathConfig(
                        path            =   path,
                        points_per_seg  =   points_per_seg
                    )
    if style is None:
        style       = PlotStyle()
    
    # Override config with explicit parameters if provided
    if path is not None:
        kpath_config.path           = path
    if points_per_seg is not None:
        kpath_config.points_per_seg = points_per_seg
    
    # Set up default styling - emphasize discrete points
    if line_kw is None:
        line_kw = {
                    "lw"            : style.linewidth,
                    "ls"            : "-",
                    "color"         : "C0",
                    "marker"        : style.marker,
                    "ms"            : style.markersize,
                    "mfc"           : "C0",
                    "mec"           : "white",
                    "mew"           : 0.5,
                    "alpha"         : style.alpha
                }
    
    if hsline_kw is None:
        hsline_kw   = kpath_config.separator_style
    
    C = np.asarray(corr_matrix, float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("corr_matrix must be square (Ns,Ns).")

    r_cart          = np.asarray(lattice.rvectors, float)
    k_cart          = np.asarray(lattice.kvectors, float)
    
    # Auto-determine points_per_seg based on k-grid density
    if kpath_config.points_per_seg is None:
        Nk                          = k_cart.shape[0]
        kpath_config.points_per_seg = max(int(np.sqrt(Nk) * kpath_config.auto_pps_factor), kpath_config.auto_pps_min)
    
    if print_vectors:
        print("\n=== DEBUG: BZ path correlation plotting ===")
        try:
            k1_vec = np.asarray(lattice.k1, float).reshape(3)
            k2_vec = np.asarray(lattice.k2, float).reshape(3)
            print(f"Reciprocal vectors:")
            print(f"  k1 = {k1_vec}")
            print(f"  k2 = {k2_vec}")
        except:
            print("  (Could not retrieve k1/k2)")
        print(f"Lattice dimensions: lx={getattr(lattice, 'lx', '?')}, "
              f"ly={getattr(lattice, 'ly', '?')}")
        print(f"Number of sites: {r_cart.shape[0]}")
        print(f"Number of k-points: {k_cart.shape[0]}")
        print(f"Path: {kpath_config.path}")
        print(f"Tolerance: {kpath_config.tolerance if kpath_config.tolerance else 'auto'}")
        print("NOTE: Using TOLERANCE-BASED selection - actual k-points only, NO interpolation")
        print("=" * 40 + "\n")

    Ns  = C.shape[0]
    if r_cart.shape[0] != Ns:
        raise ValueError(f"Ns mismatch: corr {Ns}, lattice {r_cart.shape[0]}")

    # MODULAR HELPER: Compute S(k) at all k-points
    Sk  = compute_structure_factor_from_corr(C, r_cart, k_cart, normalize=True)

    # Fractional k coords for robust path extraction
    if hasattr(lattice, "kvectors_frac") and lattice.kvectors_frac is not None:
        k_frac              = np.asarray(lattice.kvectors_frac, float)
    else:
        # Fallback: approximate fractional coordinates
        try:
            k1_vec          = np.asarray(lattice.k1, float).reshape(3)
            k2_vec          = np.asarray(lattice.k2, float).reshape(3)
            B               = np.vstack([k1_vec[:2], k2_vec[:2]])
            k2              = k_cart[:, :2]
            k_frac          = np.zeros((k_cart.shape[0], 3), float)
            k_frac[:, :2]   = (k2 @ np.linalg.inv(B).T)
        except Exception as e:
            raise ValueError(f"Cannot determine fractional k-coords: {e}")

    # Get high-symmetry points for path construction
    hs_points_obj = lattice.high_symmetry_points()
    if hs_points_obj is None:
        raise ValueError("Lattice does not define high_symmetry_points")
    
    # Build path from labels
    path_labels = kpath_config.path if kpath_config.path is not None else hs_points_obj.default_path
    if path_labels is None:
        raise ValueError("No path specified and lattice has no default path")
    
    # Get Cartesian coordinates of high-symmetry points (2D)
    k2 = k_cart[:, :2]  # Use only x,y components
    
    path_points_cart = []
    for label in path_labels:
        if label not in hs_points_obj.points:
            raise ValueError(f"High-symmetry point '{label}' not defined for this lattice")
        pt_frac = np.array(hs_points_obj.points[label], dtype=float)
        # Convert fractional to Cartesian
        try:
            k1_vec = np.asarray(lattice.k1, float).reshape(3)
            k2_vec = np.asarray(lattice.k2, float).reshape(3)
            pt_cart = pt_frac[0] * k1_vec[:2] + pt_frac[1] * k2_vec[:2]
        except:
            raise ValueError(f"Cannot convert fractional to Cartesian for point '{label}'")
        path_points_cart.append(pt_cart)
    
    # Tolerance for k-point selection (in k-space units)
    # Auto-determine from k-grid spacing
    k_spacing = np.median(np.diff(np.sort(k2[:, 0])))  # Approximate k-spacing
    tolerance = kpath_config.tolerance if hasattr(kpath_config, 'tolerance') else k_spacing * 0.5
    
    # Select k-points close to path segments
    selected_k_indices = []
    cumulative_dist = 0.0
    k_distances = []
    label_positions = [0.0]
    label_texts = [path_labels[0]]
    
    for i in range(len(path_points_cart) - 1):
        p1 = path_points_cart[i]
        p2 = path_points_cart[i + 1]
        
        # Find k-points close to this segment
        distances = point_to_segment_distance_2d(k2, p1, p2)
        close_mask = distances < tolerance
        segment_k_indices = np.where(close_mask)[0]
        
        if len(segment_k_indices) > 0:
            # Compute distance along path for each k-point
            segment_k_points = k2[segment_k_indices]
            
            # Project onto path direction
            path_vec = p2 - p1
            path_length = np.linalg.norm(path_vec)
            if path_length > 1e-14:
                path_dir = path_vec / path_length
                proj = np.dot(segment_k_points - p1[None, :], path_dir)
                
                # Sort by projection (distance along segment)
                sort_idx = np.argsort(proj)
                segment_k_indices = segment_k_indices[sort_idx]
                proj = proj[sort_idx]
                
                # Add to total distance
                segment_distances = cumulative_dist + proj
                k_distances.extend(segment_distances.tolist())
                selected_k_indices.extend(segment_k_indices.tolist())
            
        # Update cumulative distance for next segment
        cumulative_dist += np.linalg.norm(p2 - p1)
        label_positions.append(cumulative_dist)
        label_texts.append(path_labels[i + 1])
    
    # Remove duplicates (keep first occurrence)
    unique_indices = []
    seen = set()
    unique_distances = []
    for idx, dist in zip(selected_k_indices, k_distances):
        if idx not in seen:
            unique_indices.append(idx)
            unique_distances.append(dist)
            seen.add(idx)
    
    selected_k_indices = np.array(unique_indices, dtype=int)
    k_distances = np.array(unique_distances, dtype=float)
    
    # Extract S(k) values at selected k-points
    if len(selected_k_indices) == 0:
        raise ValueError("No k-points found along path. Try increasing tolerance or checking path definition.")
    
    y = Sk[selected_k_indices]
    x = k_distances

    # Plot discrete k-points with markers
    ax.plot(x, y, **line_kw)
    ax.set_ylabel(value_label)
    ax.set_xlim(x.min(), x.max())

    # Add high-symmetry separators + ticks
    if kpath_config.show_separators:
        for xv in label_positions:
            ax.axvline(xv, **hsline_kw)
        ax.set_xticks(label_positions)
        ax.set_xticklabels(label_texts)

    ax.grid(alpha=0.25)
    
    # Create result object (compatibility)
    from dataclasses import dataclass
    @dataclass
    class KPathResult:
        k_dist: np.ndarray
        values: np.ndarray
        label_positions: np.ndarray
        label_texts: list
    
    result = KPathResult(
        k_dist=x,
        values=y,
        label_positions=np.array(label_positions),
        label_texts=label_texts
    )
    
    return result

def plot_correlation_grid(
                        directory           : str,
                        param_name          : str,
                        lattice             : 'Lattice',
                        *,
                        x_parameters        : list,
                        y_parameters        : list,
                        x_param             : str       = 'J',
                        y_param             : str       = 'hx',
                        mode                : str       = 'lattice',
                        filters             = None,
                        state_idx           : int       = 0,
                        ref_site_idx        : int       = 0,
                        figsize_per_panel   : tuple     = (4, 3.5),
                        cmap                : str       = 'RdBu_r',
                        vmin                = None,
                        vmax                = None,
                        title               : str       = '',
                        # data extraction
                        param_fun           : callable  = lambda r, param_name: r.get(param_name, []),
                        param_labels        : dict      = {},
                        post_process_func   = None,
                        **kwargs):
    r"""
    Plot correlation matrices from ED results in a parameter grid with multiple visualization modes.
    
    This function creates state-of-the-art visualizations of quantum correlations across parameter space,
    supporting real-space, momentum-space, and band-structure representations that work universally
    for any lattice geometry (square, triangular, honeycomb, etc.).

    Parameters
    ----------
    directory : str
        Path to directory containing ED result files
    param_name : str
        Name of correlation data parameter to extract from results (e.g., 'spin_corr', 'density_corr')
    lattice : Lattice
        Lattice object defining the geometry (must have rvectors, kvectors, and k-space methods)
    x_parameters : list
        Values of x_param to include in grid
    y_parameters : list
        Values of y_param to include in grid
    x_param : str
        Parameter name for grid x-axis (default 'J')
    y_param : str
        Parameter name for grid y-axis (default 'hx')
    mode : str
        Visualization mode (see Modes section)
    state_idx : int
        Which eigenstate to plot (default 0 for ground state)
    ref_site_idx : int
        Reference site for real-space correlations (default 0)
    figsize_per_panel : tuple
        (width, height) for each subplot panel
    cmap : str
        Matplotlib colormap name
    vmin, vmax : float or None
        Color scale limits (auto-determined if None)
    title : str
        Figure super-title
    param_fun : callable
        Function to extract parameter data from result object
    param_labels : dict
        Pretty labels for parameters (e.g., {'hx': r'$h_x$'})
    post_process_func : callable or None
        Function to process results before plotting

    Modes
    -----
    'matrix'    : Full correlation matrix C_{ij} as heatmap
    'lattice'   : Real-space correlations from reference site
                    - 1D: line plot of C(x) vs. displacement
                    - 2D: smooth field with scattered points on lattice   
    'kspace'    : Structure factor S(k) in full Brillouin zone
                    - Smooth interpolated map with optional BZ outline
                    - Wigner-Seitz masking for proper BZ shape
                    - High-symmetry point labels (Γ, K, M, etc.)
    'kpath'     : Structure factor along high-symmetry path
                    - Band-structure style plot with vertical separators
                    - Uses lattice's default path or custom specification

    Structure Factor
    ----------------
    The momentum-space structure factor is computed as:
        S(k) = (1/Ns) sum _{i,j} C_{ij} exp[-i k . (r_i - r_j)]
    This properly handles multi-site unit cells by using full position vectors r_i
    (including basis offsets), not just Bravais lattice vectors.

    Key Parameters (kwargs)
    -----------------------
    **K-Space 2D Map (mode='kspace')**
        ks_grid_n           : int, interpolation grid resolution (default 220)
        ks_interp           : 'linear'|'cubic'|'nearest', interpolation method (default 'linear')
        ks_show_points      : bool, overlay discrete k-points (default True)
        ks_point_size       : float, marker size for k-points (default 10)
        ks_alpha_points     : float, transparency of k-point overlay (default 0.35)
        ks_draw_bz          : bool, draw Brillouin zone outline (default True)
        ks_label_hs         : bool, label high-symmetry points (default True)
        ks_mask_outside_bz  : bool, mask regions outside BZ (default True)
        ks_im_interp        : str, imshow interpolation (default 'bilinear')
        ks_blob_radius      : float, masking radius scale (default 2.5)
        ks_xlabel, ks_ylabel: str, axis labels (default r'$k_x$', r'$k_y$')
        auto_vlim_kspace    : bool, auto color limits from S(k) (default True)
        bz_shells           : int, WS cell neighbor shells for masking (default 1)
        hs_color            : str, high-symmetry label color (default 'white')
        hs_fs               : int, high-symmetry label fontsize (default 10)
    
    **K-Path (mode='kpath')**
        kpath               : path specification (None→lattice default, or StandardBZPath, 
                              or list like ['Gamma', 'K', 'M', 'Gamma'])
        kpath_pps           : int or None, points per path segment (None=auto-detect from k-grid density)
        kpath_ylabel        : str, y-axis label (default r"$S(\mathbf{k})$")
        kpath_line_kw       : dict, line plot styling. Default shows discrete points with markers.
                              For markers only (no lines): {"ls": "none", "marker": "o", "ms": 6}
        kpath_hs_kw         : dict, vertical separator styling (default {"color": "k", "alpha": 0.3})
        print_vectors       : bool, print debug info about k-grid (default False, True for first panel)
    
    **Real-Space (mode='lattice')**
        rs_interp           : str, 2D interpolation method (default 'linear')
        rs_point_size       : float, scatter marker size (default 55)
        rs_blob_radius      : float, field masking radius (default 2.5)
        rs_xlabel, rs_ylabel: str, axis labels (default r'$\Delta x$', r'$C(\Delta x)$')
    
    **General Styling**
        logger              : Logger, optional logging object
        param_label         : str, colorbar label override
        suptitle_fontsize   : int, title fontsize (default 14)
        letter_x, letter_y  : float, panel annotation position (default 0.05, 0.9)
        text_color          : str, annotation color (default 'black')
        show_panel_labels   : bool, show parameter values on panels (default True)
        cbar_pos            : list, colorbar [left, bottom, width, height] (default [0.92, 0.15, 0.02, 0.7])
        cbar_scale          : 'linear'|'log', colorbar scale (default 'linear')

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    axes_grid : np.ndarray
        2D array of axes objects, shaped (n_y_params, n_x_params)

    Examples
    --------
    >>> # K-space structure factor for square lattice
    >>> from general_python.lattices.square import SquareLattice
    >>> lattice = SquareLattice(dim=2, lx=8, ly=8, bc='pbc')
    >>> fig, axes = plot_correlation_grid(
    ...     directory='./ed_results',
    ...     param_name='spin_corr',
    ...     lattice=lattice,
    ...     x_parameters=[0.5, 1.0, 1.5],
    ...     y_parameters=[0.0, 0.1, 0.2],
    ...     x_param='J',
    ...     y_param='hx',
    ...     mode='kspace',
    ...     ks_draw_bz=True,
    ...     ks_label_hs=True,
    ...     param_labels={'J': r'$J$', 'hx': r'$h_x$'}
    ... )
    
    >>> # Band structure along high-symmetry path
    >>> fig, axes = plot_correlation_grid(
    ...     directory='./ed_results',
    ...     param_name='density_corr',
    ...     lattice=lattice,
    ...     x_parameters=[1.0],
    ...     y_parameters=[0.0, 0.5, 1.0],
    ...     mode='kpath',
    ...     kpath=['Gamma', 'X', 'M', 'Gamma'],
    ...     kpath_pps=100
    ... )
    
    >>> # Real-space correlation on honeycomb lattice
    >>> from general_python.lattices.honeycomb import HoneycombLattice
    >>> lattice = HoneycombLattice(dim=2, lx=6, ly=6, bc='pbc')
    >>> fig, axes = plot_correlation_grid(
    ...     directory='./ed_results',
    ...     param_name='spin_corr',
    ...     lattice=lattice,
    ...     x_parameters=np.linspace(0, 2, 5),
    ...     y_parameters=[0.0],
    ...     mode='lattice',
    ...     ref_site_idx=0,
    ...     rs_point_size=80,
    ...     cmap='RdBu_r'
    ... )

    Notes
    -----
    - Works with any lattice type implementing the standard Lattice interface
    - Automatically handles multi-site unit cells (e.g., honeycomb has 2 sites per cell)
    - K-space modes require lattice.kvectors and lattice.rvectors attributes
    - For best results with kpath mode, lattice should implement high_symmetry_points()
    - Wigner-Seitz BZ masking uses general_python.lattices.tools.lattice_kspace.ws_bz_mask
    """

    # Needs lattice
    if lattice is not None:
        lx, ly, lz, Ns  = lattice.lx, lattice.ly, lattice.lz, lattice.Ns
    else:
        lx, ly, lz, Ns  = kwargs.get('lx', None), kwargs.get('ly', None), kwargs.get('lz', None), kwargs.get('Ns', None)
    
    # Require lattice or lx, ly, Ns
    if lx is None or ly is None or Ns is None:
        raise ValueError("Lattice or lx, ly, Ns must be provided.")
    
    results             = load_results(
                            data_dir            =   directory,
                            filters             =   filters,
                            lx=lx, ly=ly, lz=lz, Ns=Ns,
                            logger              =   kwargs.get('logger', None),
                            post_process_func   =   post_process_func
                        )
    if not results:
        return None, None

    _, _, unique_x, unique_y = PlotDataHelpers.extract_parameter_arrays(results, x_param, y_param)

    unique_x        = np.array([v for v in unique_x if any(abs(v - p) < 1e-5 for p in x_parameters)])
    unique_y        = np.array([v for v in unique_y if any(abs(v - p) < 1e-5 for p in y_parameters)])
    unique_x        = np.sort(unique_x)
    unique_y        = np.sort(unique_y)
    unique_y_plot   = unique_y[::-1]

    n_rows, n_cols  = len(unique_y_plot), len(unique_x)
    if n_rows == 0 or n_cols == 0:
        return None, None

    fig, axes, _, _ = PlotDataHelpers.create_subplot_grid(
                        n_panels            = n_rows * n_cols,
                        max_cols            = n_cols,
                        figsize_per_panel   = figsize_per_panel,
                        sharex              = True,
                        sharey              = True
                    )
    axes_grid       = np.array(axes).reshape((n_rows, n_cols))

    # --------------------------------------------------------------
    # resolve plot_mode
    # --------------------------------------------------------------
    plot_mode   = 'matrix'
    mode_l      = (mode or '').lower().strip()
    if mode_l in ('matrix',):
        plot_mode = 'matrix'
    elif mode_l in ('lattice', 'real', 'realspace'):
        plot_mode = 'lattice'
    elif mode_l in ('kspace', 'bz', 'brillouin', 'brillouin_zone'):
        plot_mode = 'kspace'
    elif mode_l == 'kpath':
        plot_mode = 'kpath'
    else:
        plot_mode = 'matrix'

    # --------------------------------------------------------------
    # real-space geometry
    # --------------------------------------------------------------
    positions   = None
    d_dim       = None
    if lattice is not None:
        try:
            positions   = np.asarray(lattice.rvectors, float)
            d_dim       = positions.shape[1]
        except Exception:
            positions   = None
            d_dim       = None

    # --------------------------------------------------------------
    # k-space precomputes (shared across panels)
    # --------------------------------------------------------------
    k_cart              = None          # (Nk,3)
    k2                  = None          # (Nk,2)
    r2                  = None          # (Ns,2)
    phase               = None          # (Nk,Ns)
    phase_conj          = None          # (Nk,Ns)
    Ns_lat              = None          # number of sites in lattice, if needed
    Nk                  = None          # number of k-points

    ks_grid_n           = int(kwargs.get('ks_grid_n',           Ns))        # interpolation grid resolution
    ks_interp           = kwargs.get('ks_interp',               'linear')   # interpolation method
    ks_show_points      = bool(kwargs.get('ks_show_points',     True))      # show discrete k-points
    ks_point_size       = float(kwargs.get('ks_point_size',     10))        # k-point marker size
    ks_alpha_points     = float(kwargs.get('ks_alpha_points',   0.35))      # k-point alpha - transparency
    ks_draw_bz          = bool(kwargs.get('ks_draw_bz',         True))      # draw BZ outline
    ks_label_hs         = bool(kwargs.get('ks_label_hs',        True))      # label high-symmetry points
    ks_mask_outside_bz  = bool(kwargs.get('ks_mask_outside_bz', True))      # mask outside BZ
    auto_vlim_kspace    = bool(kwargs.get('auto_vlim_kspace',   True))      # auto vmin/vmax

    # Helpers for BZ    : use the new modular helper functions
    def _try_draw_bz(ax):
        """Try to draw BZ boundary using modular helper."""
        if not ks_draw_bz or lattice is None:
            return
        _draw_bz_boundary(ax, lattice)

    def _try_label_hs(ax):
        """Try to label high-symmetry points using modular helper."""
        if not ks_label_hs or lattice is None:
            return
        _label_high_symmetry_points(ax, lattice)

    # If kspace: build phase once.
    if plot_mode == 'kspace' and lattice is not None:
        try:
            k_cart      = np.asarray(lattice.kvectors, float)   # (Nk,3) usually Nk = Nc
            r_cart      = np.asarray(lattice.rvectors, float)   # (Ns,3) Ns = Nc*Nb

            k2          = k_cart[:, :2]
            r2          = r_cart[:, :2]

            phase       = np.exp(-1j * (k2 @ r2.T))             # (Nk,Ns)
            phase_conj  = np.conjugate(phase)                   # (Nk,Ns) for S(k) calc

            Ns_lat      = r2.shape[0]
            Nk          = k2.shape[0]
        except Exception:
            plot_mode   = 'matrix'

    # --------------------------------------------------------------
    # determine vmin/vmax
    # --------------------------------------------------------------
    
    def _extract_corr_matrix(res):
        ns          = int(res.params.get('Ns', Ns))
        data_root   = param_fun(res, param_name)
        if data_root is None or getattr(data_root, "size", 0) == 0:
            return None

        if data_root.ndim == 3:
            return np.asarray(np.real(data_root[:, :, state_idx]), float)
        if data_root.ndim == 2 and data_root.shape == (ns, ns):
            return np.asarray(np.real(data_root), float)
        return None

    def _selected_results():
        out = []
        for y_val in unique_y_plot:
            for x_val in unique_x:
                for r in results:
                    rx = r.params.get(x_param, np.nan)
                    ry = r.params.get(y_param, np.nan)
                    if abs(rx - x_val) < 1e-5 and abs(ry - y_val) < 1e-5:
                        out.append(r)
                        break
        return out

    if (vmin is None or vmax is None):
        if plot_mode == 'kspace' and phase is not None and auto_vlim_kspace:
            sk_min, sk_max  = +np.inf, -np.inf
            for r in _selected_results():
                C           = _extract_corr_matrix(r)
                if C is None:
                    continue
                if C.shape[0] != Ns_lat:
                    continue
                tmp         = phase @ C
                Sk          = np.real((tmp * phase_conj).sum(axis=1) / C.shape[0])
                if np.all(np.isnan(Sk)):
                    continue
                sk_min      = min(sk_min, float(np.nanmin(Sk)))
                sk_max      = max(sk_max, float(np.nanmax(Sk)))
            if np.isfinite(sk_min) and np.isfinite(sk_max):
                if vmin is None: vmin = sk_min
                if vmax is None: vmax = sk_max
            else:
                # fallback
                if vmin is None: vmin = -1.0
                if vmax is None: vmax = +1.0
        else:
            vmin_n, vmax_n  = PlotDataHelpers.determine_vmax_vmin(results, param_name, param_fun, nstates=1)
            if vmin is None: vmin = vmin_n
            if vmax is None: vmax = vmax_n

    # shared color normalization
    _, _, _, mappable = Plotter.get_colormap(values=[vmin, vmax], cmap=cmap, get_mappable=True)

    # axis label logic
    def _labelcond(i, j):
        return {'x': i == n_rows - 1, 'y': j == 0}

    def _set_axis_labels(ax, i, j):
        if plot_mode == 'matrix':
            Plotter.set_ax_params(
                ax,
                xlabel      = r'Site $j$',
                ylabel      = r'Site $i$',
                labelCond   = _labelcond(i, j),
                labelPos    = {'x': 'bottom', 'y': 'left'},
                tickPos     = {'x': 'bottom', 'y': 'left'},
            )
            return

        if plot_mode == 'lattice':
            # In 1D we show x; in 2D and lattice maps we usually hide axes.
            if positions is None:
                return
            if d_dim == 1:
                Plotter.set_ax_params(
                    ax,
                    xlabel      = kwargs.get('rs_xlabel', r'$\Delta x$'),
                    ylabel      = kwargs.get('rs_ylabel', r'$C(\Delta x)$'),
                    labelCond   = _labelcond(i, j),
                    labelPos    = {'x': 'bottom', 'y': 'left'},
                    tickPos     = {'x': 'bottom', 'y': 'left'},
                )
            else:
                ax.axis('off')
            return

        if plot_mode == 'kspace':
            # BZ panels style: only outer labels
            if _labelcond(i, j).get('x', False):
                ax.set_xlabel(kwargs.get('ks_xlabel', r'$k_x$'))
            if _labelcond(i, j).get('y', False):
                ax.set_ylabel(kwargs.get('ks_ylabel', r'$k_y$'))
            return

        if plot_mode == 'kpath':
            # k-path already sets its own labels via plot_bz_path_from_corr
            return

    # --------------------------------------------------------------
    # k-space interpolation grid (for smooth imshow-like maps)
    # --------------------------------------------------------------
    
    def _kspace_grid_and_mask():
        if k2 is None:
            return None

        kx_min, kx_max = float(np.min(k2[:, 0])), float(np.max(k2[:, 0]))
        ky_min, ky_max = float(np.min(k2[:, 1])), float(np.max(k2[:, 1]))

        pad_x = 0.05 * (kx_max - kx_min + 1e-12)
        pad_y = 0.05 * (ky_max - ky_min + 1e-12)

        kx = np.linspace(kx_min - pad_x, kx_max + pad_x, ks_grid_n)
        ky = np.linspace(ky_min - pad_y, ky_max + pad_y, ks_grid_n)
        KX, KY = np.meshgrid(kx, ky)

        # Wigner-Seitz BZ mask if requested and lattice supports it
        mask = None
        if ks_mask_outside_bz and lattice is not None:
            try:
                from general_python.lattices.tools.lattice_kspace import ws_bz_mask
                
                # Try multiple attribute names for reciprocal vectors
                k1_vec = None
                k2_vec = None
                
                for attr1 in ['k1', 'b1', 'avec']:
                    if hasattr(lattice, attr1):
                        k1_vec = np.asarray(getattr(lattice, attr1), float).ravel()[:2]
                        break
                
                for attr2 in ['k2', 'b2', 'bvec']:
                    if hasattr(lattice, attr2):
                        k2_vec = np.asarray(getattr(lattice, attr2), float).ravel()[:2]
                        break
                
                if k1_vec is not None and k2_vec is not None:
                    inside  = ws_bz_mask(KX, KY, k1_vec, k2_vec, shells=kwargs.get('bz_shells', 1))
                    mask    = ~inside  # invert: True where we want to mask out
            except Exception:
                # Fallback: mask points far from discrete k-cloud
                try:
                    from scipy.spatial import cKDTree
                    tree    = cKDTree(k2)
                    d, _    = tree.query(np.column_stack([KX.ravel(), KY.ravel()]), k=1)
                    d       = d.reshape(KX.shape)
                    d0      = np.median(d[np.isfinite(d)])
                    dmax    = float(kwargs.get('ks_blob_radius', 2.5 * d0))
                    mask    = (d > dmax)
                except Exception:
                    mask = None

        return (kx, ky, KX, KY, mask)

    k_grid_pack = _kspace_grid_and_mask() if plot_mode == 'kspace' else None

    # --------------------------------------------------------------
    # plotting loop
    # --------------------------------------------------------------
    for ii, y_val in enumerate(unique_y_plot):
        for jj, x_val in enumerate(unique_x):

            ax      = axes_grid[ii, jj]
            subset  = []

            for r in results:
                rx = r.params.get(x_param, np.nan)
                ry = r.params.get(y_param, np.nan)
                if abs(rx - x_val) < 1e-5 and abs(ry - y_val) < 1e-5:
                    subset.append(r)

            if not subset:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            result      = subset[0]
            corr_matrix = _extract_corr_matrix(result)
            if corr_matrix is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            # ----------------------------------------------------------
            # plot per mode
            # ----------------------------------------------------------
            if plot_mode == 'matrix':
                ax.imshow(
                    corr_matrix,
                    cmap            = cmap,
                    vmin            = vmin,
                    vmax            = vmax,
                    interpolation   = 'nearest',
                    origin          = 'lower'
                )

            elif plot_mode == 'lattice':
                if positions is None:
                    ax.imshow(
                        corr_matrix,
                        cmap            = cmap,
                        vmin            = vmin,
                        vmax            = vmax,
                        interpolation   = 'nearest',
                        origin          = 'lower'
                    )
                else:
                    ref_site_idx_use = ref_site_idx if (0 <= ref_site_idx < positions.shape[0]) else 0
                    site_vals        = corr_matrix[ref_site_idx_use, :]

                    # 1D: correlation vs displacement coordinate
                    if d_dim == 1 or (positions.shape[1] == 1):
                        x   = positions[:, 0] - positions[ref_site_idx_use, 0]
                        idx = np.argsort(x)
                        Plotter.plot(ax, x[idx], site_vals[idx], marker='o', lw=1.5)
                        Plotter.vline(ax, 0.0, ls='--', lw=1, color='k', alpha=0.4)
                        Plotter.grid(ax, alpha=0.3)

                    # 2D: smooth “blob” field + point overlay
                    else:
                        pos = positions[:, :2] - positions[ref_site_idx_use, :2]

                        x_min, x_max    = pos[:, 0].min(), pos[:, 0].max()
                        y_min, y_max    = pos[:, 1].min(), pos[:, 1].max()
                        pad_x           = 0.15 * (x_max - x_min + 1e-12)
                        pad_y           = 0.15 * (y_max - y_min + 1e-12)

                        xx              = np.linspace(x_min - pad_x, x_max + pad_x, 220)
                        yy              = np.linspace(y_min - pad_y, y_max + pad_y, 220)
                        X, Y            = np.meshgrid(xx, yy)

                        # Interpolate but keep it local (avoid filling big empty regions)
                        Z = None
                        try:
                            from scipy.interpolate import griddata
                            Z = griddata(pos, site_vals, (X, Y), method=kwargs.get('rs_interp', 'linear'))
                        except Exception:
                            Z = None

                        if Z is None:
                            d2  = (X[..., None] - pos[:, 0])**2 + (Y[..., None] - pos[:, 1])**2
                            Z   = site_vals[np.argmin(d2, axis=2)]

                        # Local-support mask so you get blobs, not a filled convex hull
                        try:
                            from scipy.spatial import cKDTree
                            tree = cKDTree(pos)
                            d, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]), k=1)
                            d    = d.reshape(X.shape)
                            d0   = np.median(d[np.isfinite(d)])
                            dmax = float(kwargs.get('rs_blob_radius', 2.5 * d0))
                            Z    = np.where(d <= dmax, Z, np.nan)
                        except Exception:
                            pass

                        ax.imshow(
                            Z,
                            extent  = (xx[0], xx[-1], yy[0], yy[-1]),
                            origin  = 'lower',
                            cmap    = cmap,
                            vmin    = vmin,
                            vmax    = vmax,
                            alpha   = 0.95,
                        )

                        Plotter.scatter(
                            ax,
                            pos[:, 0], pos[:, 1],
                            c           = site_vals,
                            cmap        = cmap,
                            vmin        = vmin,
                            vmax        = vmax,
                            edgecolor   = 'k',
                            linewidths  = 0.5,
                            s           = kwargs.get('rs_point_size', 55),
                            zorder      = 10
                        )
                        Plotter.scatter(ax, [0], [0], marker='*', s=120, c='yellow', edgecolor='k', zorder=11)
                        ax.set_aspect('equal')
                        ax.axis('off')

            elif plot_mode == 'kspace':
                if phase is None or k2 is None:
                    ax.text(0.5, 0.5, 'k-space unavailable', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                else:
                    # Compute S(k)
                    if corr_matrix.shape[0] != Ns_lat:
                        ax.text(0.5, 0.5, 'Ns mismatch', ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                    else:
                        C   = np.asarray(corr_matrix, float)
                        Ns0 = C.shape[0]

                        tmp = phase @ C
                        Sk  = np.real((tmp * phase_conj).sum(axis=1) / Ns0)  # (Nk,)

                        # Use the new modular k-space plotter
                        # Create configuration objects
                        style = PlotStyle(
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            fontsize_label=kwargs.get('fontsize_label', 10),
                            fontsize_tick=kwargs.get('fontsize_tick', 8),
                            linewidth=1.5,
                            markersize=3,
                            marker='o',
                            alpha=ks_alpha_points
                        )
                        
                        ks_config = KSpaceConfig(
                            grid_n=ks_grid_n,
                            interp_method=ks_interp,
                            show_discrete_points=ks_show_points,
                            point_size=ks_point_size,
                            point_alpha=ks_alpha_points,
                            draw_bz_outline=ks_draw_bz,
                            label_high_symmetry=ks_label_hs,
                            mask_outside_bz=ks_mask_outside_bz,
                            imshow_interp=kwargs.get('ks_im_interp', 'bilinear'),
                            ws_shells=kwargs.get('bz_shells', 1)
                        )
                        
                        # Use new plot_kspace_intensity with extended BZ and Pi labels
                        plot_kspace_intensity(
                            ax=ax,
                            k2=k2,
                            intensity=Sk,
                            style=style,
                            ks_config=ks_config,
                            lattice=lattice,
                            show_extended_bz=kwargs.get('show_extended_bz', True),
                            bz_copies=kwargs.get('bz_copies', 2)
                        )

            elif plot_mode == 'kpath':
                # k-path mode: plot S(k) along high-symmetry path using EXACT k-points
                if corr_matrix is None:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                else:
                    plot_bz_path_from_corr(
                        ax              = ax,
                        lattice         = lattice,
                        corr_matrix     = corr_matrix,
                        path            = kwargs.get('kpath',           None),
                        points_per_seg  = kwargs.get('kpath_pps',       None),  # None = auto-detect based on k-grid
                        value_label     = kwargs.get('kpath_ylabel',    r"$S(\mathbf{k})$"),
                        line_kw         = kwargs.get('kpath_line_kw',   None),  # None = use default (discrete markers)
                        hsline_kw       = kwargs.get('kpath_hs_kw',     {"color": "k", "alpha": 0.3}),
                        print_vectors   = kwargs.get('print_vectors',   (ii == 0 and jj == 0)),  # print only once
                    )

            # ----------------------------------------------------------
            # per-panel annotations
            # ----------------------------------------------------------
            if kwargs.get('show_panel_labels', True):
                ann_color = kwargs.get('text_color', 'black')
                if plot_mode == 'kspace' and ks_mask_outside_bz:
                    ann_color = kwargs.get('text_color', 'white')  # better contrast on BZ maps
                Plotter.set_annotate_letter(
                    ax,
                    iter        = 0,
                    x           = kwargs.get('letter_x', 0.05),
                    y           = kwargs.get('letter_y', 0.9),
                    boxaround   = False,
                    color       = ann_color,
                    addit       = rf'{param_labels.get(x_param, x_param)}$={x_val:.2g}$, '
                                  rf'{param_labels.get(y_param, y_param)}$={y_val:.2g}$'
                )

            if plot_mode not in ('lattice', 'kpath'):
                Plotter.grid(ax, alpha=0.25)
            elif plot_mode == 'lattice' and (positions is None or d_dim == 1):
                Plotter.grid(ax, alpha=0.25)

            Plotter.set_tickparams(ax)
            _set_axis_labels(ax, ii, jj)

    # --------------------------------------------------------------
    # colorbar + title
    # --------------------------------------------------------------
    cbar_scale  = kwargs.pop('cbar_scale', 'linear')
    vmin_cb     = vmin if cbar_scale != 'linear' else None
    
    # For kpath mode, use fewer ticks (discrete k-points)
    cbar_format = '%.2f'
    if plot_mode == 'kpath':
        cbar_format = '%.3f'  # more precision for discrete values

    Plotter.add_colorbar(
        fig                 = fig,
        pos                 = kwargs.get('cbar_pos', [0.92, 0.15, 0.02, 0.7]),
        mappable            = mappable,
        cmap                = cmap,
        vmin                = vmin_cb,
        vmax                = vmax,
        label               = kwargs.get('param_label', param_labels.get(param_name, param_name)),
        format              = cbar_format,
        scale               = cbar_scale,
        remove_pdf_lines    = True
    )

    fig.suptitle(title, fontsize=kwargs.get('suptitle_fontsize', 14))
    plt.show()
    return fig, axes_grid

# c) Multi-state vs parameter plotter

def plot_multistate_vs_param(
        directory           : str,
        param_name          : str,
        x_param             : str = 'hx', 
        p_param             : str = 'Gz', 
        p_values            : list = None,
        filters             = None, 
        *,
        lx                  = None, 
        ly                  = None,
        lz                  = None,
        Ns                  = None,
        # plot settings
        nstates             : int   = 10, 
        figsize_per_panel   : tuple = (4, 3),
        xlim                = None, 
        ylim                = None,
        cmap                : str = 'viridis',
        # data extraction
        param_fun           : callable  = lambda r, key: r.get(key, []),
        param_labels        : dict      = {},
        trans_fun           : callable  = lambda raw_data_array, state_index: raw_data_array[state_index],
        post_process_func   : callable  = None,
        # labels
        ylabel              : str = None,
        title               : str = '',
        # advanced
        derivative_order    : int  = 0,
        susceptibility_sign : bool = False,
        **kwargs):

    results = load_results(data_dir=directory, filters=filters, lx=lx, ly=ly, lz=lz, Ns=Ns, post_process_func=post_process_func, logger=kwargs.get('logger', None))
    if not results: return None, None

    if p_values is None:
        _, _, _, unique_p       = PlotDataHelpers.extract_parameter_arrays(results, x_param, p_param)
        p_values                = sorted(unique_p)
        if len(p_values) > 9 and kwargs.get('limit_panels', True):
            p_values = p_values[:9]

    n_panels                    = len(p_values)
    fig, axes, n_rows, n_cols   = PlotDataHelpers.create_subplot_grid(n_panels, max_cols=3, figsize_per_panel=figsize_per_panel, sharex=True, sharey=True)
    if hasattr(fig, 'set_constrained_layout'):
        fig.set_constrained_layout(True)
    
    axes                        = axes.flatten()
    labelcond                   = lambda idx: {'x': idx // n_cols == n_rows - 1, 'y': idx % n_cols == 0}
    xlabel_str                  = param_labels.get(x_param, x_param)
    ylabel_str                  = ylabel if ylabel else param_labels.get(param_name, param_name)
    getcolor, _, _, mappable    = Plotter.get_colormap(values=np.arange(nstates), cmap=cmap, elsecolor='black', get_mappable=True)
    
    for idx, p_val in enumerate(p_values):
        # Filter for panel param
        ax              = axes[idx]
        panel_results   = []
        
        for r in results:
            val = r.params.get(p_param, np.nan)
            if abs(val - p_val) < 1e-6:
                panel_results.append(r)
        
        if not panel_results:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        x_vals, sort_idx    = PlotDataHelpers.sort_results_by_param(panel_results, x_param)
        sorted_res          = [panel_results[i] for i in sort_idx]

        for state_i in range(nstates):
            y_vals, valid_x = [], []
            for k, r in enumerate(sorted_res):
                raw = param_fun(r, param_name)
                try:
                    is_valid = isinstance(raw, (list, np.ndarray)) and len(raw) > state_i
                    if is_valid:
                        val = trans_fun(raw, int(state_i))
                        if not np.isnan(val):
                            y_vals.append(val)
                            valid_x.append(x_vals[k])
                except Exception:
                    pass

            if len(valid_x) > 2:
                x_arr = np.array(valid_x)
                y_arr = np.array(y_vals)
                
                if derivative_order >= 1: y_arr = np.gradient(y_arr, x_arr)
                if derivative_order >= 2: y_arr = np.gradient(y_arr, x_arr) * (-1 if susceptibility_sign else 1)
                ax.plot(x_arr, y_arr, color=getcolor(state_i), marker='o', markersize=3, ls='-', zorder=100 - state_i, label=rf'$|\Psi_{{{state_i}}}\rangle$')

        p_label = param_labels.get(p_param, p_param)
        Plotter.set_annotate_letter(ax, iter=idx, x=kwargs.get('annotate_x', 0.05), y=kwargs.get('annotate_y', 0.9), addit=rf' {p_label}$={p_val:.2g}$', boxaround=False)
        Plotter.set_ax_params(ax, xlabel=xlabel_str, ylabel=ylabel_str, labelCond=labelcond(idx), xlim=xlim, ylim=ylim, yscale=kwargs.get('yscale', 'linear'), xscale=kwargs.get('xscale', 'linear'))
        Plotter.set_tickparams(ax)
        Plotter.grid(ax, alpha=0.3)

    Plotter.hide_unused_panels(axes, n_panels)
    
    cbar_pos    = kwargs.get('cbar_pos', [0.92, 0.2, 0.02, 0.6])
    cbar_scale  = kwargs.get('cbar_scale', 'linear')
    vmin        = 0 if cbar_scale == 'linear' else None
    if nstates > 6:
        Plotter.add_colorbar(fig=fig, pos=cbar_pos, mappable=mappable, cmap=cmap, scale=cbar_scale, vmin=vmin, vmax=nstates - 1, label=r'State Index $n$', extend=None, format='%d', remove_pdf_lines=True)
    else:
        Plotter.set_legend(axes[0], loc=kwargs.get('legend_loc', 'upper right'), fontsize=kwargs.get('legend_fontsize', 8))
        
    if title: 
        fig.suptitle(title, fontsize=kwargs.get('suptitle_fontsize', 12))
        
    return fig, axes

# d) Size scaling plotter

def plot_scaling_analysis(
        directory: str,
        param_name: str,
        scaling_param: str = 'Ns',
        series_param: str = 'hx',
        filters=None,
        state_idx: int = 0,
        *,
        param_fun: callable = lambda r, name: r.get(name, []),
        param_labels: dict = {},
        figsize: tuple = (6, 5),
        logger: Optional['Logger'] = None,
        **kwargs):
    """
    Plots a scaling analysis of a parameter (e.g., energy, gap) vs system size (Ns or 1/L),
    grouping lines by another parameter (e.g., field strength).
    """
    
    results = load_results(data_dir=directory, filters=filters, logger=logger)
    if not results: return None, None

    # Get unique series values
    _, _, _, unique_series = PlotDataHelpers.extract_parameter_arrays(results, scaling_param, series_param)
    unique_series = sorted(unique_series)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = plt.get_cmap(kwargs.get('cmap', 'viridis'))
    norm = plt.Normalize(min(unique_series), max(unique_series))
    
    for s_val in unique_series:
        # Filter for this series
        subset = [r for r in results if abs(r.params.get(series_param, -999) - s_val) < 1e-5]
        if not subset: continue
        
        x_vals, sort_idx = PlotDataHelpers.sort_results_by_param(subset, scaling_param)
        sorted_subset = [subset[i] for i in sort_idx]
        
        y_vals = []
        x_plot = []
        for r, x_v in zip(sorted_subset, x_vals):
            val = param_fun(r, param_name)
            
            # Handle list/array or scalar
            if isinstance(val, (list, np.ndarray)):
                if len(val) > state_idx:
                    y_vals.append(val[state_idx])
                    x_plot.append(x_v)
            elif isinstance(val, (float, int)):
                 y_vals.append(val)
                 x_plot.append(x_v)
        
        if len(x_plot) > 0:
            color = cmap(norm(s_val))
            label = f"{param_labels.get(series_param, series_param)}={s_val:.2g}"
            ax.plot(x_plot, y_vals, marker='o', linestyle='-', color=color, label=label)

    xlabel = param_labels.get(scaling_param, scaling_param)
    ylabel = param_labels.get(param_name, param_name)
    
    Plotter.set_ax_params(ax, xlabel=xlabel, ylabel=ylabel, title=kwargs.get('title', ''))
    Plotter.set_tickparams(ax)
    Plotter.grid(ax)
    
    # Legend
    if len(unique_series) <= 10:
        ax.legend(frameon=False, fontsize=8)
    else:
        # If too many lines, use colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(param_labels.get(series_param, series_param))

    PlotDataHelpers.savefig(fig, directory, param_name, scaling_param, series_param, suffix='_scaling')
    return fig, ax

# ------------------------------------------------------------------
#! End of file
# ------------------------------------------------------------------
