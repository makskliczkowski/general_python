'''
General plotting helpers for spectral and k-space data.

Framework-agnostic plotting functions for 2D spectral data visualization:
- Spectral function plots A(k,w) in path or grid mode
- Static structure factors S(k,n)
- K-space intensity maps with BZ extension

These functions accept preprocessed numpy arrays and can be used
with any calculation method (ED, DMRG, QMC, etc.).

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-01-15
License             : MIT
----------------------------------------------------------------
'''

from typing import TYPE_CHECKING, Optional, Literal
import numpy as np
from scipy.interpolate import griddata

try:
    from .config        import PlotStyle, KSpaceConfig, SpectralConfig
    from .kspace_utils  import label_high_sym_points
except ImportError:
    raise ImportError("Failed to import plotting configurations or k-space utilities.")

if TYPE_CHECKING:
    from general_python.lattices.lattice import Lattice

# ==============================================================================
# 2D SPECTRAL FUNCTION PLOTTING
# ==============================================================================

def plot_spectral_function_2d(
        ax,
        k_values        : np.ndarray,
        omega           : np.ndarray,
        intensity       : np.ndarray,
        *,
        mode            : Literal['kpath', 'grid']  = 'kpath',
        path_info       : Optional[dict]            = None,
        style           : Optional[PlotStyle]       = None,
        spectral_config : Optional[SpectralConfig]  = None,
        lattice         : Optional["Lattice"]       = None,
        use_extend      : bool                      = False,
        extend_copies   : int                       = 2,
        colorbar        : bool                      = True
    ):
    """
    Plot 2D spectral function A(k, w) or S(k, w).
    
    Universal plotter for dynamical correlation functions in momentum-energy space.
    Supports both path mode (band-structure style) and full grid mode.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Target axes
    k_values : (Nk,) or (Nk, D) array
        K-point coordinates (path distance for mode='kpath', Cartesian for mode='grid')
    omega : (Nw,) array
        Energy/frequency grid
    intensity : (Nk, Nw) array
        Spectral intensity matrix
    mode : str
        - 'kpath' : Plot along high-symmetry path with vertical separators
        - 'grid'  : 2D k-space grid with optional BZ extension
    path_info : dict, optional
        For mode='kpath': {label_positions, label_texts} from select_kpoints_along_path
    style : PlotStyle, optional
        Styling configuration
    spectral_config : SpectralConfig, optional
        Spectral function configuration
    lattice : Lattice, optional
        Lattice object (required for mode='grid' with use_extend=True)
    use_extend : bool
        If True, extend k-space to show multiple BZ copies
    extend_copies : int
        Number of BZ copies in each direction
    colorbar : bool
        Add colorbar
        
    Returns
    -------
    im : AxesImage or QuadMesh
        The plotted image for colorbar creation
    """
    if style is None:
        style = PlotStyle()
    if spectral_config is None:
        spectral_config = SpectralConfig()
    
    # Determine color scale
    vmin = style.vmin if style.vmin is not None else np.nanmin(intensity)
    vmax = style.vmax if style.vmax is not None else np.nanmax(intensity)
    
    if spectral_config.log_scale and vmin <= 0:
        vmin = np.nanmin(intensity[intensity > 0]) if np.any(intensity > 0) else 1e-10
    
    # Log normalization if requested
    if spectral_config.log_scale:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    
    if mode == 'kpath':
        # Path mode: k_values are 1D path distances
        if k_values.ndim > 1:
            k_values = k_values[:, 0]  # Extract distance
        
        K, Omega = np.meshgrid(k_values, omega, indexing='ij')
        
        im = ax.pcolormesh(
            K, Omega, intensity,
            cmap=style.cmap,
            vmin=vmin if not spectral_config.log_scale else None,
            vmax=vmax if not spectral_config.log_scale else None,
            norm=norm,
            shading='auto'
        )
        
        ax.set_ylabel(spectral_config.omega_label, fontsize=style.fontsize_label)
        ax.set_xlim(k_values.min(), k_values.max())
        
        # Add high-symmetry separators
        if path_info is not None:
            label_positions = path_info.get('label_positions', [])
            label_texts = path_info.get('label_texts', [])
            
            if len(label_positions) > 0:
                for xv in label_positions:
                    ax.axvline(xv, color='k', ls='--', lw=1.0, alpha=0.35)
                ax.set_xticks(label_positions)
                ax.set_xticklabels(label_texts)
        
    elif mode == 'grid':
        # Grid mode: show full k-space
        if k_values.ndim == 1:
            raise ValueError("For mode='grid', k_values must be 2D array (Nk, 2)")
        
        k2 = k_values[:, :2]
        
        # Extend k-space if requested
        if use_extend and lattice is not None:
            from general_python.lattices.tools.lattice_kspace import extend_kspace_data
            k1_vec = np.asarray(lattice.k1, float).ravel()[:2]
            k2_vec = np.asarray(lattice.k2, float).ravel()[:2]
            
            # Extend each omega slice
            extended_intensity = []
            for iw in range(intensity.shape[1]):
                k2_ext, int_ext = extend_kspace_data(
                    k2, intensity[:, iw],
                    k1_vec, k2_vec,
                    nx=extend_copies, ny=extend_copies
                )
                extended_intensity.append(int_ext)
            
            k2 = k2_ext
            intensity = np.array(extended_intensity).T  # (Nk_extended, Nw)
        
        # Create 2D visualization (average over w or select specific w)
        if spectral_config.omega_grid is not None:
            # Select specific omega
            omega0 = spectral_config.energy_shift
            idx = np.argmin(np.abs(omega - omega0))
            intensity_2d = intensity[:, idx]
        else:
            # Integrate over omega
            intensity_2d = np.trapz(intensity, omega, axis=1)
        
        # Scatter plot in k-space
        im = ax.scatter(
            k2[:, 0], k2[:, 1],
            c=intensity_2d,
            cmap=style.cmap,
            vmin=vmin,
            vmax=vmax,
            s=style.markersize * 2,
            alpha=style.alpha,
            edgecolors='none'
        )
        
        ax.set_xlabel(r'$k_x$', fontsize=style.fontsize_label)
        ax.set_ylabel(r'$k_y$', fontsize=style.fontsize_label)
        ax.set_aspect('equal')
    
    # Energy limits
    if spectral_config.vmin_omega is not None:
        ax.set_ylim(bottom=spectral_config.vmin_omega)
    if spectral_config.vmax_omega is not None:
        ax.set_ylim(top=spectral_config.vmax_omega)
    
    ax.tick_params(labelsize=style.fontsize_tick)
    
    return im

def plot_static_structure_factor(
        ax,
        k_values        : np.ndarray,
        state_indices   : np.ndarray,
        intensity       : np.ndarray,
        *,
        mode            : Literal['kpath', 'grid']  = 'kpath',
        path_info       : Optional[dict]            = None,
        style           : Optional[PlotStyle]       = None,
        lattice         : Optional["Lattice"]       = None,
        use_extend      : bool                      = False,
        extend_copies   : int                       = 2
    ):
    """
    Plot static structure factor S(k, n) where n is eigenstate index.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Target axes
    k_values : (Nk,) or (Nk, D) array
        K-point coordinates
    state_indices : (Nstates,) array
        Eigenstate indices to show
    intensity : (Nk, Nstates) array
        Structure factor for each k-point and state
    mode : str
        'kpath' or 'grid'
    path_info : dict, optional
        Path information for mode='kpath'
    style : PlotStyle, optional
        Styling configuration
    lattice : Lattice, optional
        For grid mode with extension
    use_extend : bool
        Extend k-space
    extend_copies : int
        BZ copies
        
    Returns
    -------
    im : plot object
        The plotted image
    """
    if style is None:
        style = PlotStyle()
    
    vmin = style.vmin if style.vmin is not None else np.nanmin(intensity)
    vmax = style.vmax if style.vmax is not None else np.nanmax(intensity)
    
    if mode == 'kpath':
        # Path mode
        if k_values.ndim > 1:
            k_values = k_values[:, 0]
        
        K, States = np.meshgrid(k_values, state_indices, indexing='ij')
        
        im = ax.pcolormesh(
            K, States, intensity,
            cmap=style.cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        
        ax.set_ylabel('Eigenstate Index', fontsize=style.fontsize_label)
        ax.set_xlim(k_values.min(), k_values.max())
        
        # Add high-symmetry separators
        if path_info is not None:
            label_positions = path_info.get('label_positions', [])
            label_texts = path_info.get('label_texts', [])
            
            if len(label_positions) > 0:
                for xv in label_positions:
                    ax.axvline(xv, color='k', ls='--', lw=1.0, alpha=0.35)
                ax.set_xticks(label_positions)
                ax.set_xticklabels(label_texts)
    
    elif mode == 'grid':
        # Grid mode - show for specific state
        if k_values.ndim == 1:
            raise ValueError("For mode='grid', k_values must be 2D array")
        
        k2 = k_values[:, :2]
        
        # Show first state or average
        intensity_2d = intensity[:, 0]  # Can be made selectable
        
        # Extend if requested
        if use_extend and lattice is not None:
            from general_python.lattices.tools.lattice_kspace import extend_kspace_data
            k1_vec = np.asarray(lattice.k1, float).ravel()[:2]
            k2_vec = np.asarray(lattice.k2, float).ravel()[:2]
            k2, intensity_2d = extend_kspace_data(
                k2, intensity_2d,
                k1_vec, k2_vec,
                nx=extend_copies, ny=extend_copies
            )
        
        im = ax.scatter(
            k2[:, 0], k2[:, 1],
            c=intensity_2d,
            cmap=style.cmap,
            vmin=vmin,
            vmax=vmax,
            s=style.markersize * 2,
            alpha=style.alpha,
            edgecolors='none'
        )
        
        ax.set_xlabel(r'$k_x$', fontsize=style.fontsize_label)
        ax.set_ylabel(r'$k_y$', fontsize=style.fontsize_label)
        ax.set_aspect('equal')
    
    ax.tick_params(labelsize=style.fontsize_tick)
    
    return im

# ==============================================================================
# K-SPACE INTENSITY PLOTS
# ==============================================================================

def plot_kspace_intensity(
        ax,
        k2                  : np.ndarray,
        intensity           : np.ndarray,       # e.g., S(k)
        *,
        style               : Optional[PlotStyle]       = None,
        ks_config           : Optional[KSpaceConfig]    = None,
        lattice             : Optional["Lattice"]       = None,
        show_extended_bz    : bool                      = True,
        bz_copies           : int                       = 2,
        **kwargs
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
    **kwargs : dict
        Additional arguments (k1_vec, k2_vec if lattice not provided)
    """
    
    if style is None:
        style               = PlotStyle()
    if ks_config is None:
        ks_config           = KSpaceConfig()
    
    k2_orig                 = np.asarray(k2, dtype=float)
    intensity               = np.asarray(intensity, dtype=float)
    k1_vec                  = lattice.k1 if lattice is not None else kwargs.get('k1_vec', None)
    k2_vec                  = lattice.k2 if lattice is not None else kwargs.get('k2_vec', None)
    if k1_vec is None or k2_vec is None:
        raise ValueError("Reciprocal lattice vectors k1 and k2 must be provided via lattice or kwargs.")
    k1_vec                  = np.asarray(k1_vec, dtype=float).ravel()[:2]
    k2_vec                  = np.asarray(k2_vec, dtype=float).ravel()[:2]
        
    # Determine color scale
    vmin                    = style.vmin if style.vmin is not None else np.nanmin(intensity)
    vmax                    = style.vmax if style.vmax is not None else np.nanmax(intensity)
    
    if show_extended_bz:
        from general_python.lattices.tools.lattice_kspace import extend_kspace_data
        final_k, final_intensity    = extend_kspace_data(k2_orig, intensity, b1=k1_vec, b2=k2_vec, nx=bz_copies, ny=bz_copies)
    else:
        final_k, final_intensity    = k2_orig, intensity
        
    # Interpolated background (optional)
    if ks_config.grid_n > 0:
        # Interpolate on ORIGINAL k-points only
        KX, KY              = np.meshgrid(np.linspace(-np.pi * 2, np.pi * 2, ks_config.grid_n), np.linspace(-np.pi * 2, np.pi * 2, ks_config.grid_n))
        
        try:
            # Interpolate on original k-points
            Z               = griddata(final_k, final_intensity, (KX, KY), method=ks_config.interp_method)
            ax.imshow(
                Z,
                extent          =   (KX.min(), KX.max(), KY.min(), KY.max()),
                origin          =   'lower',
                cmap            =   style.cmap,
                vmin            =   vmin,
                vmax            =   vmax,
                interpolation   =   ks_config.imshow_interp,
                alpha           =   0.9
            )
        except Exception as e:
            raise RuntimeError(f"{e}")
        
    # Discrete points overlay
    if ks_config.show_discrete_points:
        ax.scatter(
            k2_orig[:, 0], k2_orig[:, 1],
            c           =   intensity,
            cmap        =   style.cmap,
            vmin        =   vmin,
            vmax        =   vmax,
            s           =   ks_config.point_size,
            alpha       =   ks_config.point_alpha,
            edgecolors  =   'none',
            zorder      =   120
        )
    
    # Label high-symmetry points in all BZ copies
    if ks_config.label_high_symmetry and lattice is not None:
        try:
            label_high_sym_points(ax, lattice, bz_copies if show_extended_bz else 0, show_labels=True, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to label high-symmetry points: {e}")
    
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$', fontsize=style.fontsize_label)
    ax.set_ylabel(r'$k_y$', fontsize=style.fontsize_label)
    ax.tick_params(labelsize=style.fontsize_tick)
    
    # Set strict limits to -2Pi to 2Pi when showing extended BZ
    if show_extended_bz and bz_copies >= 1:
        ax.set_xlim(-bz_copies * np.pi / 2, bz_copies * np.pi / 2)
        ax.set_ylim(-bz_copies * np.pi / 2, bz_copies * np.pi / 2)

# ==============================================================================
#! EOF
# ==============================================================================