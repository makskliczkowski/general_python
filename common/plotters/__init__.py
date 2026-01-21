'''
Plotting utilities for Quantum EigenSolver.

Exports:
- Configuration dataclasses (PlotStyle, KSpaceConfig, KPathConfig, SpectralConfig, FigureConfig)
- Data loaders and helpers
- ED-specific plotters
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
    PlotData,
)

from .kspace_utils import (
    point_to_segment_distance_2d,
    select_kpoints_along_path,
    compute_structure_factor_from_corr,
    label_high_sym_points,
    format_pi_ticks,
)

from .spectral_utils import (
    compute_spectral_broadening,
    extract_spectral_data,
)

from .plot_helpers import (
    plot_static_structure_factor,
    plot_kspace_intensity
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
    'PlotData',
    # K-space utilities
    'point_to_segment_distance_2d',
    'select_kpoints_along_path',
    'compute_structure_factor_from_corr',
    'label_high_sym_points',
    'format_pi_ticks',
    # Spectral utilities
    'compute_spectral_broadening',
    'extract_spectral_data',
    # Plot helpers
    'plot_spectral_function_2d',
    'plot_static_structure_factor',
    'plot_kspace_intensity',
]

# ------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------