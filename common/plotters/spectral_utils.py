'''
Spectral function utilities for quantum physics.

General-purpose utilities for computing and manipulating spectral functions:
- Broadening of discrete spectra (Lorentzian, Gaussian)
- Data extraction and reshaping
- Normalization and sum rules

These functions are framework-agnostic and can be used with any
method that produces spectral data (ED, DMRG, QMC, etc.).

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-01-15
License             : MIT
----------------------------------------------------------------
'''

from    typing              import Tuple, Optional, Literal, TYPE_CHECKING
import  numpy               as np
from    scipy.interpolate   import griddata

try:
    
    from .kspace_utils      import label_high_sym_points
except ImportError:
    raise ImportError("Failed to import plotting configurations or k-space utilities.")

if TYPE_CHECKING:
    from general_python.lattices.lattice    import Lattice
    from .config                            import PlotStyle, KSpaceConfig, SpectralConfig
    
# ==============================================================================
# SPECTRAL BROADENING
# ==============================================================================

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
    A(w) = sum_n w_n * f(w - E_n)
    
    where f is Lorentzian or Gaussian kernel.
    
    Parameters
    ----------
    energies : (N,) array
        Excitation energies (delta peak positions)
    weights : (N,) array
        Spectral weights (delta peak heights)
    omega_grid : (Nw,) array
        Frequency/energy grid for output
    eta : float
        Broadening parameter (FWHM for Lorentzian, std for Gaussian)
    kind : str
        Broadening kernel: 'lorentzian', 'gaussian', or 'none'
        
    Returns
    -------
    spectrum : (Nw,) array
        Broadened spectral function
        
    Notes
    -----
    - Lorentzian: L(w) = (1/Pi) * (η / ((w-E)² + η²))
    - Gaussian: G(w) = (1/√(2Piη²)) * exp(-(w-E)²/(2η²))
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
    
    # Vectorized broadening: omega[Nw, None] - energies[None, N] -> (Nw, N)
    omega_diff  = omega_grid[:, None] - energies[None, :]
    
    if kind == 'lorentzian':
        # Lorentzian: L(w-E) = (1/Pi) * η / ((w-E)² + η²)
        kernel      = (eta / np.pi) / (omega_diff**2 + eta**2)
    elif kind == 'gaussian':
        # Gaussian: G(w-E) = (1/√(2Piη²)) * exp(-(w-E)²/(2η²))
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
    
    # Sum weighted kernels: A(w) = sum_n w_n * kernel(w - E_n)
    spectrum        = np.sum(weights[None, :] * kernel, axis=1)
    
    return spectrum

# ==============================================================================
# DATA EXTRACTION
# ==============================================================================

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
    - omega:        (Nw,)
    - k_vectors:    (Nk, D)
    - data:         (Nk, Nw)
    
    Parameters
    ----------
    result : Result object
        Result container with spectral data
    key : str
        Data key (e.g., 'akw', 'spectral/skw')
    state_idx : int, optional
        Which state to extract if multiple
    component : str, optional
        Which component (e.g., 'xx', 'zz') for multi-component data. It will
        append to key as 'key/component'.
    reshape_to_komega : bool
        If True, ensure output is (Nk, Nw); otherwise keep original shape
    omega_key : str, optional
        Suffix for omega data key (default: '/omega')
    kvectors_key : str, optional
        Key for k-vectors (default: '/kvectors')
    kvectors : array, optional
        Override k-vectors from result
        
    Returns
    -------
    omega :
        (Nw,) array
    k_vectors :
        (Nk, D) array
    data :
        (Nk, Nw) array
    """
    
    # Try to extract data
    if component is not None:
        key = f"{key}/{component}"
    has_get = hasattr(result, 'get')
    
    if has_get:
        data_raw    = result.get(key, None)
        if data_raw is None:
            raise ValueError(f"Key '{key}' not found in result")
    elif isinstance(result, np.ndarray):
        data_raw    = result
    else:
        raise TypeError("result must have 'get' method or be a numpy array")
    
    # Extract omega grid
    if has_get:
        omega       = result.get(omega_key, None)
        if omega is None:
            omega   = result.get('omega', None)
        if omega is None:
            omega   = result.get('/omega', None)
    else:
        omega       = None
    
    if omega is None:
        omega       = np.linspace(0, 1, data_raw.shape[-1]) # Fallback omega grid
    
    # Extract k-vectors (if available)
    if has_get:
        k_vectors   = kvectors if kvectors is not None else result.get(kvectors_key, None)
    else:
        k_vectors   = kvectors
        
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
        data    = np.abs(data)
    
    omega       = np.asarray(omega, float)
    k_vectors   = np.asarray(k_vectors, float)
    
    return omega, k_vectors, data

# ==============================================================================
# 2D SPECTRAL FUNCTION PLOTTING
# ==============================================================================

def plot_spectral_function_2d(
        ax,
        k_values        : np.ndarray,
        omega           : np.ndarray,
        intensity       : np.ndarray,
        *,
        mode            : Literal['kpath', 'grid']      = 'kpath',
        path_info       : Optional[dict]                = None,
        style           : Optional['PlotStyle']         = None,
        spectral_config : Optional['SpectralConfig']    = None,
        lattice         : Optional["Lattice"]           = None,
        use_extend      : bool                          = False,
        extend_copies   : int                           = 2,
        colorbar        : bool                          = True,
        **kwargs
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
    mode : Literal['kpath', 'grid']
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
    try:
        from ..plot     import Plotter
        from .config    import PlotStyle, SpectralConfig
    except ImportError as e:
        raise ImportError("Failed to import plotting configurations: " + str(e))
    
    if style is None:               style = PlotStyle()
    if spectral_config is None:     spectral_config = SpectralConfig()
    
    # Determine color scale
    vmin                            = style.vmin if style.vmin is not None else np.nanmin(intensity)
    vmax                            = style.vmax if style.vmax is not None else np.nanmax(intensity)
    
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
        
        K, Omega    = np.meshgrid(k_values, omega, indexing='ij')
        im          = ax.pcolormesh(
                        K, Omega, intensity,
                        cmap    =   style.cmap,
                        vmin    =   vmin if not spectral_config.log_scale else None,
                        vmax    =   vmax if not spectral_config.log_scale else None,
                        norm    =   norm,
                        shading =   'auto'
                    )
        Plotter.set_ax_params(
            ax,
            xlabel          =   r'$k$',
            ylabel          =   spectral_config.omega_label,
            fontsize        =   style.fontsize_label,
            xlim            =   (k_values.min(), k_values.max())
        )
        
        # Add high-symmetry separators
        if path_info is not None:
            label_positions     = path_info.get('label_positions',  [])
            label_texts         = path_info.get('label_texts',      [])
            
            if len(label_positions) > 0:
                for xv in label_positions:
                    ax.axvline(xv, color='k', ls='--', lw=kwargs.get('linewidth', 1.0), alpha=kwargs.get('alpha', 0.7))
                Plotter.set_tickparams(ax, xticks=label_positions, xticklabels=label_texts)
        
    elif mode == 'grid':
        
        # Grid mode: show full k-space
        if k_values.ndim == 1:
            raise ValueError("For mode='grid', k_values must be 2D array (Nk, 2)")
        
        k2 = k_values[:, :2]
        
        # Extend k-space if requested
        if use_extend and lattice is not None:
            try:
                from general_python.lattices.tools.lattice_kspace import extend_kspace_data
            except ImportError as e:
                raise ImportError("Failed to import k-space extension utility: " + str(e))
            
            k1_vec              = np.asarray(lattice.k1, float).ravel()[:2]
            k2_vec              = np.asarray(lattice.k2, float).ravel()[:2]
            extended_intensity  = []
            
            # Extend each omega slice
            for iw in range(intensity.shape[1]):
                k2_ext, int_ext = extend_kspace_data(
                                    k2, intensity[:, iw],
                                    k1_vec, k2_vec,
                                    nx=extend_copies, ny=extend_copies
                                )
                extended_intensity.append(int_ext)
            
            k2          = k2_ext
            intensity   = np.array(extended_intensity).T # (Nk_extended, Nw)
        
        kx_unique = np.unique(k2[:, 0])
        ky_unique = np.unique(k2[:, 1])
        
        # Create 2D visualization (average over w or select specific w)
        if spectral_config.omega_grid is not None:
            # Select specific omega
            
            # Try to reshape into regular grid first
            if len(kx_unique) * len(ky_unique) == len(k2):
                # Regular grid - can use imshow/pcolormesh
                nx, ny  = len(kx_unique), len(ky_unique)
                
                # Reshape intensity for each omega and create heatmap
                # For spectral function, we want to show A(kx, ky, omega) as 2D heatmap
                # Option 1: Integrate over omega
                # Option 2: Show specific omega slice
                if spectral_config.omega_grid is not None:
                    # Select specific omega
                    omega0          = spectral_config.energy_shift
                    idx             = np.argmin(np.abs(omega - omega0))
                    intensity_slice = intensity[:, idx]
                else:
                    intensity_slice = np.trapezoid(intensity, omega, axis=1)
                
                # Reshape to grid
                try:
                    intensity_grid  = intensity_slice.reshape(ny, nx)
                    
                    # Create meshgrid for proper plotting
                    KX, KY          = np.meshgrid(kx_unique, ky_unique, indexing='xy')
                    
                    im = ax.pcolormesh(
                        KX, KY, intensity_grid,
                        cmap        =   style.cmap,
                        vmin        =   vmin if not spectral_config.log_scale else None,
                        vmax        =   vmax if not spectral_config.log_scale else None,
                        norm        =   norm,
                        shading     =   'auto'
                    )
                except:
                    # Fallback to scatter if reshape fails
                    im = ax.scatter(
                        k2[:, 0], k2[:, 1],
                        c           =   intensity_slice,
                        cmap        =   style.cmap,
                        vmin        =   vmin,
                        vmax        =   vmax,
                        s           =   style.markersize * 2,
                        alpha       =   style.alpha,
                        edgecolors  =   'none'
                    )
        else:
            # Integrate over omega
            # Irregular grid - use scatter
            if spectral_config.omega_grid is not None:
                omega0          = spectral_config.energy_shift
                idx             = np.argmin(np.abs(omega - omega0))
                intensity_2d    = intensity[:, idx]
            else:
                intensity_2d    = np.trapezoid(intensity, omega, axis=1)
            
            im                  = ax.scatter(
                                    k2[:, 0], k2[:, 1],
                                    c           =   intensity_2d,
                                    cmap        =   style.cmap,
                                    vmin        =   vmin,
                                    vmax        =   vmax,
                                    s           =   style.markersize * 2,
                                    alpha       =   style.alpha,
                                    edgecolors  =   'none'
                                )
        Plotter.set_ax_params(
            ax,
            xlabel          =   r'$k_x$',
            ylabel          =   r'$k_y$',
            fontsize        =   style.fontsize_label
        )
        ax.set_aspect('equal')
    
    # Energy limits
    if spectral_config.vmin_omega is not None:
        ax.set_ylim(bottom=spectral_config.vmin_omega)
    if spectral_config.vmax_omega is not None:
        ax.set_ylim(top=spectral_config.vmax_omega)
    
    ax.tick_params(labelsize=style.fontsize_tick)
    
    if colorbar and kwargs.get('fig', None) is not None:
        Plotter.add_colorbar(
            fig         = kwargs.get('fig', None),
            mappable    = im,
            label       = spectral_config.colorbar_label,
        )
    
    return im

# ==============================================================================
#! EOF
# ==============================================================================