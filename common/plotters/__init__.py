'''
Plotting utilities for Quantum EigenSolver.

Exports the small public plotting surface:
- configuration dataclasses
- lightweight data loading helpers
- generic single-axis plotters
- thin ED multistate / parameter-grid wrappers
'''

from .config import (
    PlotStyle,
    KSpaceConfig,
    KPathConfig,
    SpectralConfig,
    FigureConfig,
    StylePresets,
)

from .data_loader import (
    load_results,
    filter_results,
    ResultSet,
    PlotData,
)

from .kspace_utils import point_to_segment_distance_2d, label_high_sym_points, format_pi_ticks

from .spectral_utils import (
    compute_spectral_broadening,
    extract_spectral_data,
)

from .plot_helpers import (
    compute_correlation_kspace,
    plot_correlation,
    plot_realspace_correlations,
    plot_kspace_path,
    plot_spectral_function,
    plot_kspace_intensity,
)

__all__ = [
    # Configuration
    'PlotStyle',
    'KSpaceConfig',
    'KPathConfig',
    'SpectralConfig',
    'FigureConfig',
    'StylePresets',
    # Data loading
    'load_results',
    'filter_results',
    'ResultSet',
    'PlotData',
    # K-space utilities
    'point_to_segment_distance_2d',
    'label_high_sym_points',
    'format_pi_ticks',
    # Spectral utilities
    'compute_spectral_broadening',
    'extract_spectral_data',
    # Plot helpers
    'compute_correlation_kspace',
    'plot_correlation',
    'plot_kspace_path',
    'plot_realspace_correlations',
    'plot_spectral_function',
    'plot_kspace_intensity',
]

# ------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------
