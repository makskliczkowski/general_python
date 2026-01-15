'''
Configuration dataclasses for plotting utilities.

Provides reusable configuration objects for:
- General plot styling                  (PlotStyle)
- K-space visualization                 (KSpaceConfig)
- K-path extraction and plotting        (KPathConfig)
- Spectral function plotting            (SpectralConfig)
- Figure layout and grid configuration  (FigureConfig)

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-01-15
License             : MIT
----------------------------------------------------------------
'''

from    typing import Optional, Literal, Tuple, Dict, Any
from    dataclasses import dataclass, field
import  numpy as np

# ==============================================================================
# GENERAL PLOT STYLING
# ==============================================================================

@dataclass
class PlotStyle:
    """
    General styling configuration for plots.
    
    Attributes
    ----------
    cmap : str
        Matplotlib colormap name
    vmin, vmax : float, optional
        Color scale limits (auto if None)
    vmin_strategy, vmax_strategy : str
        Strategy for auto-determining limits: 'auto', 'percentile', 'absolute'
    percentile_low, percentile_high : float
        Percentile values for 'percentile' strategy
    fontsize_* : int
        Font sizes for various elements
    marker : str
        Marker style for scatter/line plots
    markersize : float
        Marker size
    linewidth : float
        Line width
    alpha : float
        Transparency (0-1)
    edgecolor : str, optional
        Edge color for markers
    edgewidth : float
        Edge width for markers
    """
    # Color mapping
    cmap                : str               = 'viridis'
    vmin                : Optional[float]   = None
    vmax                : Optional[float]   = None
    vmin_strategy       : Literal['auto', 'percentile', 'absolute'] = 'auto'
    vmax_strategy       : Literal['auto', 'percentile', 'absolute'] = 'auto'
    percentile_low      : float             = 2.0
    percentile_high     : float             = 98.0
    
    # Font sizes
    fontsize_label      : int               = 10
    fontsize_tick       : int               = 8
    fontsize_title      : int               = 12
    fontsize_legend     : int               = 9
    fontsize_annotation : int               = 8
    fontsize_colorbar   : int               = 9
    
    # Line and marker styles
    marker              : str               = 'o'
    markersize          : float             = 5.0
    linewidth           : float             = 1.5
    linestyle           : str               = '-'
    alpha               : float             = 0.8
    edgecolor           : Optional[str]     = None
    edgewidth           : float             = 0.5
    
    # Grid and axes
    grid_alpha          : float             = 0.3
    grid_linestyle      : str               = '--'
    spine_width         : float             = 1.0
    
    def to_scatter_kwargs(self) -> Dict[str, Any]:
        """Return kwargs dict for ax.scatter()."""
        kwargs = {
            'cmap'          : self.cmap,
            's'             : self.markersize,
            'alpha'         : self.alpha,
            'edgecolors'    : self.edgecolor if self.edgecolor else 'none',
            'linewidths'    : self.edgewidth,
        }
        if self.vmin is not None:
            kwargs['vmin']  = self.vmin
        if self.vmax is not None:
            kwargs['vmax']  = self.vmax
        return kwargs
    
    def to_plot_kwargs(self) -> Dict[str, Any]:
        """Return kwargs dict for ax.plot()."""
        return {
            'marker'            : self.marker,
            'markersize'        : self.markersize,
            'linewidth'         : self.linewidth,
            'linestyle'         : self.linestyle,
            'alpha'             : self.alpha,
            'markeredgecolor'   : self.edgecolor,
            'markeredgewidth'   : self.edgewidth,
        }
    
    def to_imshow_kwargs(self) -> Dict[str, Any]:
        """Return kwargs dict for ax.imshow()."""
        kwargs = {'cmap': self.cmap}
        if self.vmin is not None:
            kwargs['vmin'] = self.vmin
        if self.vmax is not None:
            kwargs['vmax'] = self.vmax
        return kwargs

# ==============================================================================
# K-SPACE VISUALIZATION
# ==============================================================================

@dataclass
class KSpaceConfig:
    """
    Configuration for k-space plotting.
    
    Attributes
    ----------
    grid_n : int
        Interpolation grid resolution for smooth backgrounds
    interp_method : str
        Interpolation method: 'linear', 'cubic', 'nearest'
    mask_outside_bz : bool
        Apply Wigner-Seitz masking to show only first BZ
    show_discrete_points : bool
        Overlay discrete k-point scatter on top of interpolation
    point_size : float
        Size of discrete k-point markers
    point_alpha : float
        Transparency of discrete points
    draw_bz_outline : bool
        Draw Brillouin zone boundary
    label_high_symmetry : bool
        Add labels for high-symmetry points
    ws_shells : int
        Number of shells for Wigner-Seitz cell calculation
    blob_radius_factor : float
        Radius scaling for blob masking in real-space plots
    imshow_interp : str
        Interpolation for imshow: 'bilinear', 'nearest', etc.
    extend_bz : bool
        Show extended Brillouin zones (-2π to 2π)
    bz_copies : int
        Number of BZ copies in each direction for extension
    """
    # Interpolation settings
    grid_n              : int               = 220
    interp_method       : Literal['linear', 'cubic', 'nearest'] = 'linear'
    
    # Masking and boundaries
    mask_outside_bz     : bool              = True
    ws_shells           : int               = 1
    draw_bz_outline     : bool              = True
    
    # Discrete points overlay
    show_discrete_points: bool              = True
    point_size          : float             = 10.0
    point_alpha         : float             = 1.0
    point_marker        : str               = 'o'
    
    # High-symmetry points
    label_high_symmetry : bool              = True
    hs_marker_size      : float             = 5.0
    hs_marker_color     : str               = 'white'
    hs_marker_edge      : str               = 'black'
    hs_label_offset_x   : float             = 0.5
    hs_label_offset_y   : float             = 0.5
    hs_label_fontsize   : int               = 11
    hs_label_color      : str               = 'black'
    
    # Display settings
    blob_radius_factor  : float             = 2.5
    imshow_interp       : str               = 'bilinear'
    
    # BZ extension
    extend_bz           : bool              = False
    bz_copies           : int               = 2
    
    def should_extend(self) -> bool:
        """Check if BZ extension is enabled."""
        return self.extend_bz and self.bz_copies > 0

# ==============================================================================
# K-PATH CONFIGURATION
# ==============================================================================

@dataclass
class KPathConfig:
    """
    Configuration for k-path extraction and plotting.
    
    Attributes
    ----------
    path : str, list of str, optional
        Path specification:
        - None: use lattice default
        - str: StandardBZPath enum name (e.g., 'SQUARE_2D')
        - list: custom path labels (e.g., ['Gamma', 'K', 'M', 'Gamma'])
    points_per_seg : int, optional
        Points per path segment (None = auto from k-grid density)
    auto_pps_factor : float
        Factor for auto points_per_seg: sqrt(Nk) * factor
    auto_pps_min : int
        Minimum points per segment for auto mode
    tolerance : float, optional
        Tolerance for k-point selection (None = auto from k-spacing)
    tick_format : str
        Tick format: 'labels' (Γ, K, M), 'fractional' (0, 0.5, 1), 'distance'
    show_separators : bool
        Draw vertical lines at high-symmetry points
    separator_style : dict
        Style kwargs for separator lines
    use_extend : bool
        Use extended k-space for path extraction
    extend_copies : int
        BZ copies for extension
    """
    # Path specification
    path                : Optional[Any]     = None  # Can be str, list, or None
    points_per_seg      : Optional[int]     = None
    
    # Auto-detection parameters
    auto_pps_factor     : float             = 0.5
    auto_pps_min        : int               = 20
    
    # Tolerance and matching
    tolerance           : Optional[float]   = None
    
    # Visualization
    tick_format         : Literal['labels', 'fractional', 'distance'] = 'labels'
    show_separators     : bool              = True
    separator_style     : dict              = field(default_factory=lambda: {
        "color": "k", "ls": "--", "lw": 1.0, "alpha": 0.35
    })
    
    # Extension
    use_extend          : bool              = False
    extend_copies       : int               = 2
    
    def get_separator_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for axvline calls."""
        return self.separator_style.copy()

# ==============================================================================
# SPECTRAL FUNCTION CONFIGURATION
# ==============================================================================

@dataclass
class SpectralConfig:
    """
    Configuration for spectral function plotting.
    
    Attributes
    ----------
    omega_grid : array, optional
        Custom omega/energy grid (None = use data grid)
    broadening_type : str
        Broadening kernel: 'none', 'lorentzian', 'gaussian'
    eta : float
        Broadening parameter (FWHM or std)
    normalization : str
        Normalization: 'none', 'per_k', 'global', 'sum_rule'
    sum_rule_target : float, optional
        Target value for sum rule enforcement
    energy_shift : float
        Shift energies (e.g., subtract ground state)
    log_scale : bool
        Use logarithmic color scale
    omega_label : str
        Y-axis label for energy/frequency
    omega_units : str
        Units for omega (for labels)
    intensity_label : str
        Intensity label (e.g., A(k,ω))
    vmin_omega, vmax_omega : float, optional
        Energy axis limits
    cmap_spectral : str, optional
        Override colormap for spectral plots
    """
    # Energy grid
    omega_grid          : Optional[np.ndarray] = None
    omega_min           : Optional[float]   = None
    omega_max           : Optional[float]   = None
    n_omega             : int               = 200
    
    # Broadening
    broadening_type     : Literal['none', 'lorentzian', 'gaussian'] = 'lorentzian'
    eta                 : float             = 0.1
    
    # Normalization
    normalization       : Literal['none', 'per_k', 'global', 'sum_rule'] = 'none'
    sum_rule_target     : Optional[float]   = None
    
    # Energy shifts
    energy_shift        : float             = 0.0
    
    # Visualization
    log_scale           : bool              = False
    omega_label         : str               = r'$\omega$'
    omega_units         : str               = ''
    intensity_label     : str               = r'$A(\mathbf{k},\omega)$'
    
    # Axis limits
    vmin_omega          : Optional[float]   = None
    vmax_omega          : Optional[float]   = None
    
    # Optional colormap override
    cmap_spectral       : Optional[str]     = None
    
    def get_omega_grid(self) -> Optional[np.ndarray]:
        """Generate omega grid if min/max specified."""
        if self.omega_grid is not None:
            return self.omega_grid
        if self.omega_min is not None and self.omega_max is not None:
            return np.linspace(self.omega_min, self.omega_max, self.n_omega)
        return None
    
    def get_intensity_label_with_units(self) -> str:
        """Get intensity label with units appended."""
        if self.omega_units:
            return f'{self.intensity_label} [{self.omega_units}]'
        return self.intensity_label

# ==============================================================================
# FIGURE LAYOUT CONFIGURATION
# ==============================================================================

@dataclass
class FigureConfig:
    """
    Configuration for figure layout and grid arrangement.
    
    Attributes
    ----------
    figsize_per_panel : tuple
        (width, height) for each subplot panel
    max_cols : int
        Maximum columns in subplot grid
    sharex, sharey : bool
        Share axes across subplots
    constrained_layout : bool
        Use constrained layout engine
    tight_layout : bool
        Apply tight_layout (alternative to constrained)
    wspace, hspace : float
        Width/height spacing between subplots
    colorbar_position : list
        [left, bottom, width, height] for colorbar
    colorbar_orientation : str
        'vertical' or 'horizontal'
    suptitle_fontsize : int
        Super-title font size
    panel_label_position : tuple
        (x, y) for panel letter annotations
    panel_label_box : bool
        Draw box around panel labels
    """
    # Subplot sizing
    figsize_per_panel   : Tuple[float, float] = (4.0, 3.5)
    max_cols            : int               = 3
    
    # Axis sharing
    sharex              : bool              = True
    sharey              : bool              = True
    
    # Layout engines
    constrained_layout  : bool              = True
    tight_layout        : bool              = False
    
    # Spacing
    wspace              : float             = 0.3
    hspace              : float             = 0.3
    
    # Colorbar
    colorbar_position   : list              = field(default_factory=lambda: [0.92, 0.15, 0.02, 0.7])
    colorbar_orientation: str               = 'vertical'
    colorbar_label_pad  : float             = 10.0
    
    # Titles and labels
    suptitle_fontsize   : int               = 14
    panel_label_position: Tuple[float, float] = (0.05, 0.9)
    panel_label_box     : bool              = False
    panel_label_color   : str               = 'black'
    
    def get_colorbar_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for colorbar creation."""
        return {
            'orientation': self.colorbar_orientation,
            'pad': self.colorbar_label_pad,
        }

# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================

class StylePresets:
    """Collection of preset style configurations."""
    
    @staticmethod
    def publication() -> PlotStyle:
        """High-quality publication-ready style."""
        return PlotStyle(
            fontsize_label  =12,
            fontsize_tick   =10,
            fontsize_title  =14,
            fontsize_legend =10,
            linewidth       =2.0,
            markersize      =6.0,
            spine_width     =1.5,
        )
    
    @staticmethod
    def presentation() -> PlotStyle:
        """Large fonts for presentations."""
        return PlotStyle(
            fontsize_label  =16,
            fontsize_tick   =14,
            fontsize_title  =18,
            fontsize_legend =14,
            linewidth       =2.5,
            markersize      =8.0,
            spine_width     =2.0,
        )
    
    @staticmethod
    def notebook() -> PlotStyle:
        """Compact style for notebooks."""
        return PlotStyle(
            fontsize_label  =9,
            fontsize_tick   =8,
            fontsize_title  =10,
            fontsize_legend =8,
            linewidth       =1.2,
            markersize      =4.0,
        )
    
    @staticmethod
    def kspace_extended() -> KSpaceConfig:
        """K-space config with BZ extension enabled."""
        return KSpaceConfig(
            extend_bz           =True,
            bz_copies           =2,
            show_discrete_points=True,
            label_high_symmetry =True,
        )
    
    @staticmethod
    def kpath_fine() -> KPathConfig:
        """High-resolution k-path."""
        return KPathConfig(
            points_per_seg      =100,
            show_separators     =True,
            use_extend          =False,
        )

# ==============================================================================
# END OF FILE
# ==============================================================================
