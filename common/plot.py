r'''
Plotting Utilities for Scientific Publications
===============================================

This module provides a comprehensive set of tools for creating publication-quality
plots using Matplotlib. It is designed for scientific computing workflows with
support for Nature/Science journal formatting.

Features
--------
- **Publication Styles**: Pre-configured styles for Nature, Science, and PRL journals
- **Color Management**: Colorblind-safe palettes (Tableau, Plastic, Pastel)
- **Scientific Formatters**: LaTeX-style axis labels with scientific notation
- **Flexible Colorbars**: Full control over position, scale (log/linear), and discretization
- **Grid Layouts**: GridSpec-based subplot management with insets
- **Data Filtering**: Parameter-based filtering for multi-experiment datasets

Quick Start
-----------
>>> from general_python.common.plot import Plotter
>>> fig, axes = Plotter.get_subplots(nrows=2, ncols=2, sizex=8, sizey=6)
>>> Plotter.plot(axes[0], x, y, color='C0', label='Data')
>>> Plotter.semilogy(axes[1], x, y_log, color='C1')
>>> Plotter.set_ax_params(axes[0], xlabel='Time (s)', ylabel='Signal', title='Panel A')
>>> Plotter.set_legend(axes[0], style='publication')
>>> Plotter.save_fig('.', 'figure', format='pdf', dpi=300)

Examples
--------
Creating a figure with error bars::

    fig, ax = Plotter.get_subplots(1, 1, sizex=4, sizey=3)
    Plotter.errorbar(ax[0], x, y, yerr=sigma, color='C0', label='Measurement')
    Plotter.set_ax_params(ax[0], xlabel=r'$x$', ylabel=r'$f(x)$', yscale='log')

Adding a colorbar::

    cbar, cax = Plotter.add_colorbar(
        fig, [0.92, 0.15, 0.02, 0.7], data,
        cmap='viridis', scale='log', label=r'$|\psi|^2$'
    )

For more examples, call: Plotter.help()

----------------------------------
Author              : Maksymilian Kliczkowski
Date                : December 2025
Email               : maxgrom97@gmail.com
----------------------------------
'''


import  json
import  string
import  itertools
import  warnings
import  numpy as np
from    math import fsum
from    typing import Tuple, Union, Optional, List, Literal, TYPE_CHECKING, Dict, Any

try:
    import scienceplots
except ImportError:
    print("scienceplots not found, some styles may not be available.")

# Matplotlib
import matplotlib               as mpl
import matplotlib.pyplot        as plt
import matplotlib.colors        as mcolors
import matplotlib.ticker        as mticker
import matplotlib.tri           as mtri

# Grids
from matplotlib.colors          import Normalize, ListedColormap
from matplotlib.gridspec        import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches         import Rectangle, Circle
from matplotlib.ticker          import FixedLocator, NullFormatter, LogLocator, LogFormatterMathtext
from matplotlib.legend          import Legend
from matplotlib.legend_handler  import HandlerBase, HandlerPatch

if TYPE_CHECKING:
    from .plotters.config import FigureConfig, KPathConfig, KSpaceConfig, PlotStyle, SpectralConfig

# -------------------------------------------
# MATPLOTLIB CONFIGURATION
# -------------------------------------------
mpl.rcParams.update(mpl.rcParamsDefault)

def configure_style(style       : str = 'publication', 
                    font_size   : int = 10,
                    use_latex   : bool = False,
                    dpi         : int = 150,
                    **overrides):
    """
    Configure matplotlib rcParams for publication-quality figures.
    
    This function sets up consistent styling across all plots. Call it once
    at the start of your script or notebook.
    
    Parameters
    ----------
    style       : str, default='publication'
        Style preset to apply.
        - 'publication'   : compact Nature/Science-like defaults
        - 'presentation'  : larger text and strokes for talks
        - 'poster'        : very large sizes for posters
        - 'minimal'       : stripped-down axes visuals
        - 'default'       : reset to Matplotlib defaults
    font_size   : int, default=10
        Base typographic scale. Label/tick/title/legend sizes are derived
        from this value in each preset.
    use_latex   : bool, default=False
        If True, tries LaTeX-backed rendering via scienceplots profile.
        If unavailable, falls back gracefully to non-LaTeX settings.
    dpi         : int, default=150
        Screen/display DPI used in interactive rendering.
    **overrides : dict
        Additional rcParams overrides. Underscores are accepted and converted
        to dots, e.g. `axes_linewidth=1.0` -> `axes.linewidth`.
        Typical high-impact keys:
        - 'savefig.dpi'
        - 'axes.prop_cycle'
        - 'figure.constrained_layout.use'
        - 'font.family'
    
    Examples
    --------
    >>> # Standard publication setup
    >>> configure_style('publication', font_size=10)
    
    >>> # Presentation with larger fonts
    >>> configure_style('presentation', font_size=14, dpi=100)
    
    >>> # Custom overrides
    >>> configure_style('publication', **{'axes.linewidth': 1.5, 'lines.linewidth': 2})
    
    Notes
    -----
    This function modifies global matplotlib rcParams. To reset to defaults,
    use `configure_style('default')` or `mpl.rcParams.update(mpl.rcParamsDefault)`.
    
    Recommended figure sizes for journals:
    - Nature: single column = 3.5 in, double column = 7 in
    - Science: single column = 3.5 in, double column = 7.25 in
    - PRL: single column = 3.4 in, double column = 7 in
    """
    # Reset to defaults first
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # Try to load scienceplots styles if available
    if style != 'default':
        try:
            if use_latex:
                plt.style.use(['science', 'nature'])
            else:
                plt.style.use(['science', 'no-latex'])
        except Exception:
            pass  # Fall back to manual configuration
    
    # Base configuration (applies to all styles)
    base_config = {
        # Figure
        'figure.dpi'            : dpi,
        'figure.facecolor'      : 'white',
        'figure.edgecolor'      : 'white',
        'figure.autolayout'     : False,
        'figure.constrained_layout.use': True,
        
        # Saving
        'savefig.dpi'           : 300,
        'savefig.facecolor'     : 'white',
        'savefig.edgecolor'     : 'white',
        'savefig.bbox'          : 'tight',
        'savefig.pad_inches'    : 0.05,
        'savefig.transparent'   : False,
        'pdf.fonttype'          : 42,
        'ps.fonttype'           : 42,
        'svg.fonttype'          : 'none',
        
        # Axes
        'axes.facecolor'        : 'white',
        'axes.edgecolor'        : 'black',
        'axes.labelcolor'       : 'black',
        'axes.unicode_minus'    : True,
        'axes.axisbelow'        : True,
        'axes.grid'             : False,
        'axes.spines.top'       : True,
        'axes.spines.right'     : True,
        'axes.titlelocation'    : 'left',
        'axes.titleweight'      : 'regular',
        'axes.formatter.limits' : (-3, 4),
        'axes.formatter.use_mathtext': True,
        'axes.formatter.useoffset': False,
        'axes.prop_cycle'       : mpl.cycler(color=["#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B", "#17BECF"]),
        
        # Ticks
        'xtick.direction'       : 'in',
        'ytick.direction'       : 'in',
        'xtick.top'             : True,
        'ytick.right'           : True,
        'xtick.color'           : 'black',
        'ytick.color'           : 'black',
        'xtick.minor.visible'   : True,
        'ytick.minor.visible'   : True,
        
        # Grid
        'grid.color'            : '#D9D9D9',
        'grid.linestyle'        : '-',
        'grid.linewidth'        : 0.6,
        'grid.alpha'            : 0.55,
        
        # Lines
        'lines.linewidth'       : 1.25,
        'lines.markersize'      : 4.5,
        'lines.markeredgewidth' : 0.75,
        'lines.solid_capstyle'  : 'round',
        'lines.solid_joinstyle' : 'round',
        
        # Scatter / Error bars / Hist
        'scatter.marker'        : 'o',
        'errorbar.capsize'      : 2.5,
        'hist.bins'             : 40,
        
        # Images
        'image.cmap'            : 'viridis',
        'image.interpolation'   : 'nearest',
        'image.origin'          : 'lower',
        
        # Patches
        'patch.linewidth'       : 0.8,
        
        # Rendering behavior
        'path.simplify'         : True,
        'path.simplify_threshold': 0.0,
        'agg.path.chunksize'    : 20000,
        
        # Fonts: keep serif text + STIX math for consistent publication look.
        'font.family'           : 'serif',
        'font.serif'            : ['STIXGeneral', 'DejaVu Serif', 'Times New Roman', 'Times', 'Computer Modern Roman'],
        'mathtext.fontset'      : 'stix',
        
        # Legend
        'legend.frameon'        : False,
        'legend.framealpha'     : 1.0,
        'legend.edgecolor'      : 'none',
        'legend.fancybox'       : False,
        'legend.borderaxespad'  : 0.4,
        'legend.handlelength'   : 1.6,
        'legend.handletextpad'  : 0.5,
        'legend.columnspacing'  : 1.0,
    }
    
    # Style-specific configurations
    style_configs = {
        'publication': {
            'font.size'             : font_size,
            'axes.titlesize'        : font_size,
            'axes.labelsize'        : font_size,
            'xtick.labelsize'       : font_size - 1,
            'ytick.labelsize'       : font_size - 1,
            'legend.fontsize'       : font_size - 1,
            'axes.linewidth'        : 0.9,
            'axes.spines.top'       : True,
            'axes.spines.right'     : True,
            'xtick.top'             : True,
            'ytick.right'           : True,
            'xtick.major.width'     : 0.9,
            'ytick.major.width'     : 0.9,
            'xtick.minor.width'     : 0.6,
            'ytick.minor.width'     : 0.6,
            'xtick.major.size'      : 4,
            'ytick.major.size'      : 4,
            'xtick.minor.size'      : 2,
            'ytick.minor.size'      : 2,
            'lines.linewidth'       : 1.3,
            'lines.markersize'      : 4.2,
        },
        'presentation': {
            'font.size'             : font_size,
            'axes.titlesize'        : font_size + 2,
            'axes.labelsize'        : font_size,
            'xtick.labelsize'       : font_size - 2,
            'ytick.labelsize'       : font_size - 2,
            'legend.fontsize'       : font_size - 2,
            'axes.linewidth'        : 1.2,
            'xtick.major.width'     : 1.2,
            'ytick.major.width'     : 1.2,
            'xtick.major.size'      : 6,
            'ytick.major.size'      : 6,
            'xtick.minor.size'      : 3,
            'ytick.minor.size'      : 3,
            'lines.linewidth'       : 2.0,
            'lines.markersize'      : 8,
        },
        'poster': {
            'font.size'             : font_size,
            'axes.titlesize'        : font_size + 4,
            'axes.labelsize'        : font_size + 2,
            'xtick.labelsize'       : font_size,
            'ytick.labelsize'       : font_size,
            'legend.fontsize'       : font_size,
            'axes.linewidth'        : 1.5,
            'xtick.major.width'     : 1.5,
            'ytick.major.width'     : 1.5,
            'xtick.major.size'      : 8,
            'ytick.major.size'      : 8,
            'lines.linewidth'       : 2.5,
            'lines.markersize'      : 10,
        },
        'minimal': {
            'font.size'             : font_size,
            'axes.spines.top'       : False,
            'axes.spines.right'     : False,
            'xtick.top'             : False,
            'ytick.right'           : False,
        },
        'default': {},
    }
    
    # Apply configurations
    if style != 'default':
        for key, value in base_config.items():
            try:
                mpl.rcParams[key] = value
            except (KeyError, ValueError):
                pass
        
        if style in style_configs:
            for key, value in style_configs[style].items():
                try:
                    mpl.rcParams[key] = value
                except (KeyError, ValueError):
                    pass
    
    # Apply user overrides
    for key, value in overrides.items():
        key = key.replace('_', '.')  # Allow underscores: axes_linewidth -> axes.linewidth
        try:
            mpl.rcParams[key] = value
        except (KeyError, ValueError):
            print(f"Warning: Invalid rcParam '{key}'")

def get_rcparams_summary() -> dict:
    """
    Get a summary of current rcParams relevant to plotting.
    
    Returns
    -------
    dict
        Dictionary with current settings for fonts, lines, axes, etc.
    
    Examples
    --------
    >>> params = get_rcparams_summary()
    >>> print(params['font.size'])
    """
    keys = [
        'font.size', 'axes.labelsize', 'axes.titlesize', 
        'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize',
        'lines.linewidth', 'lines.markersize',
        'scatter.marker', 'errorbar.capsize',
        'image.cmap', 'image.interpolation',
        'axes.linewidth', 'xtick.major.size', 'ytick.major.size',
        'axes.formatter.limits', 'axes.formatter.use_mathtext', 'axes.formatter.useoffset',
        'figure.dpi', 'savefig.dpi',
        'axes.spines.top', 'axes.spines.right',
        'savefig.transparent', 'pdf.fonttype', 'svg.fonttype',
    ]
    return {k: mpl.rcParams.get(k, 'N/A') for k in keys}

# Apply default publication style on import
try:
    configure_style('publication', font_size=10)
except Exception:
    pass

# labellines
try:
    from labellines import labelLines
except ImportError:
    print("labellines not found, labelLines function will not be available.")

# style
SMALL_SIZE                  =   12
MEDIUM_SIZE                 =   14
BIGGER_SIZE                 =   16
ADDITIONAL_LINESTYLES       =   {
    'loosely dotted'        : (0, (1, 5)),
    'dotted'                : (0, (1, 1)),
    'densely dotted'        : (0, (1, 1)),
    'loosely dashed'        : (0, (2, 5)),
    'dashed'                : (0, (5, 5)),
    'densely dashed'        : (0, (5, 1)),
    'loosely dashdotted'    : (0, (3, 10, 1, 10)),
    'dashdotted'            : (0, (3, 5, 1, 5)),
    'densely dashdotted'    : (0, (3, 1, 1, 1)),
    'dashdotdotted'         : (0, (3, 5, 1, 5, 1, 5)),
    'loosely dashdotdotted' : (0, (3, 10, 1, 10, 1, 10)),
    'densely dashdotdotted' : (0, (3, 1, 1, 1, 1, 1)),
    'solid'                 : (0, ()),
    'loosely long dashed'   : (0, (5, 10)),
    'long dashed'           : (0, (5, 15)),
    'densely long dashed'   : (0, (5, 1)),
    'loosely spaced dots'   : (0, (1, 10)),
    'spaced dots'           : (0, (1, 15)),
    'densely spaced dots'   : (0, (1, 1)),
    'loosely spaced dashes' : (0, (5, 10)),
    'spaced dashes'         : (0, (5, 15)),
    'densely spaced dashes' : (0, (5, 1))
}

# Set the font dictionaries (for plot title and axis titles)
try:
    plt.style.use(['science', 'no-latex', 'colors5-light'])
except Exception:
    try:
        plt.style.use(['science', 'no-latex'])
    except Exception:
        # Fallback to default if science styles are missing
        pass
    
# Safely set additional rcParams (for compatibility with documentation build systems)
try:
    mpl.rcParams['mathtext.fontset']    = 'stix'
    mpl.rcParams['font.family']         = 'serif'
    mpl.rcParams['font.serif']          = ['STIXGeneral', 'DejaVu Serif', 'Times New Roman', 'Times', 'Computer Modern Roman']
except (TypeError, AttributeError, KeyError):
    # Handle cases where rcParams doesn't support item assignment (e.g., during doc builds)
    pass

#####################################
colorsList                          =   (list(mcolors.TABLEAU_COLORS))
colorsCycle                         =   itertools.cycle(colorsList)
colorsCyclePlastic                  =   itertools.cycle(list(["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
colorsCycleBright                   =   itertools.cycle(list(mcolors.CSS4_COLORS))
colorsCycleDark                     =   itertools.cycle(list(mcolors.BASE_COLORS))
colorsCyclePastel                   =   itertools.cycle(list(mcolors.XKCD_COLORS))
#####################################

@staticmethod
def reset_color_cycles(which=None):
    '''
    Reset the color cycles to the default ones.
    - which     :   which color cycle to reset
    Returns:
    - the cycle to take
    '''
    global colorsCycle, colorsCyclePlastic, colorsCycleBright, colorsCycleDark, colorsCyclePastel
    cycle2take = None
    cycle2take = _reset_cycle(which, 'TABLEAU', colorsCycle, list(mcolors.TABLEAU_COLORS), cycle2take)
    cycle2take = _reset_cycle(which, 'Plastic', colorsCyclePlastic, ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"], cycle2take)
    cycle2take = _reset_cycle(which, 'Bright', colorsCycleBright, list(mcolors.CSS4_COLORS), cycle2take)
    cycle2take = _reset_cycle(which, 'Dark', colorsCycleDark, list(mcolors.BASE_COLORS), cycle2take)
    cycle2take = _reset_cycle(which, 'Pastel', colorsCyclePastel, list(mcolors.XKCD_COLORS), cycle2take)
    return cycle2take

@staticmethod
def _reset_cycle(which, cycle_name, cycle_var, colors, cycle2take):
    '''
    Reset the color cycle to the default ones.
    - which     :   which color cycle to reset
    - cycle_name:   name of the cycle to reset
    - cycle_var :   cycle variable to reset
    - colors    :   colors to reset to
    - cycle2take:   cycle to take
    Returns:
    - the cycle to take
    '''
    if which is None or which == cycle_name:
        cycle_var = itertools.cycle(colors)
        cycle2take = cycle_var if cycle2take is None else cycle2take
    return cycle2take
    
@staticmethod
def get_color_cycle(which=None):
    ''' 
    Get the color cycle to use.
    - which     :   which color cycle to use
    Returns:
    - the cycle to take
    '''
    global colorsCycle, colorsCyclePlastic, colorsCycleBright, colorsCycleDark, colorsCyclePastel
    if which is None or which == 'TABLEAU':
        return colorsCycle
    if which is None or which == 'Plastic':
        return colorsCyclePlastic
    if which is None or which == 'Bright':
        return colorsCycleBright
    if which is None or which == 'Dark':
        return colorsCycleDark
    if which is None or which == 'Pastel':
        return colorsCyclePastel        

#####################################
markersList                         =   ['o','s','v', '+', 'o', '*'] + ['D', 'h', 'H', 'p', 'P', 'X', 'd', '|', '_']
markersCycle                        =   itertools.cycle(["4", "2", "3", "1", "+", "x", "."] + markersList)
markerNorm                          =   lambda x: markersList[int(x[1:])] if (x is not None and x.startswith("M") and x[1:].isdigit() and int(x[1:]) >= 0 and int(x[1:]) <= len(markersList) - 1) else x

linestylesList                      =   ['-', '--', '-.', ':']
linestylesCycle                     =   itertools.cycle(['-', '--', '-.', ':'])
linestylesCycleExtended             =   itertools.cycle(['-', '--', '-.', ':'] + list(ADDITIONAL_LINESTYLES.keys()))
linestyleNorm                       =   lambda x: linestylesList[int(x[1:])] if (x is not None and x.startswith("L") and x[1:].isdigit() and int(x[1:]) >= 0 and int(x[1:]) <= len(linestylesList) - 1) else (ADDITIONAL_LINESTYLES.get(x, x) if x in ADDITIONAL_LINESTYLES else x)
@staticmethod
def reset_linestyles(which=None):
    '''
    Reset the line styles to the default ones.
    - which     :   which line style to reset
    Returns:
    - the cycle to take
    '''
    global linestylesCycle, linestylesCycleExtended
    cycle2take = None
    if which is None or which == 'Normal':
        linestylesCycle = itertools.cycle(['-', '--', '-.', ':'])
        cycle2take = linestylesCycle
    if which is None or which == 'Extended':
        linestylesCycleExtended = itertools.cycle(['-', '--', '-.', ':'] + list(ADDITIONAL_LINESTYLES.keys()))
        cycle2take = linestylesCycleExtended if cycle2take is None else cycle2take
    return cycle2take

@staticmethod
def get_linestyle_cycle(which=None):
    ''' 
    Get the line style cycle to use.
    - which     :   which line style cycle to use
    Returns:
    - the cycle to take
    '''
    global linestylesCycle, linestylesCycleExtended
    if which is None or which == 'Normal':
        return linestylesCycle
    if which is None or which == 'Extended':
        return linestylesCycleExtended

############################ latex ############################
colorMean                           =   'PuBu'
colorTypical                        =   'BrBG'
colorExtreme                        =   'RdYlBu'
colorDiverging                      =   'coolwarm'
colorSequential                     =   'viridis'
colorCategorical                    =   'tab20'
colorQualitative                    =   'tab20'

########################## functions ##########################

class CustomFormatter(mticker.Formatter):
    """Matplotlib formatter backed by a Python format string."""

    def __init__(self, fmt="{x:.2f}"):
        """
        Initialize the object with a format string.
        
        Args:
        fmt (str): The format string to be used for formatting.
        """
        self.fmt = fmt
        
    def __call__(self, x, pos=None):
        return self.fmt.format(x=x)

class PercentFormatter(mticker.PercentFormatter):
    """Percent formatter with concise defaults for publication plots."""

    def __init__(self, decimals=2, symbol='%'):
        """
        Initialize the object with a format string.
        
        Args:
        decimals (int): The number of decimal places to use.
        symbol (str): The symbol to use for percentage.
        """
        super().__init__(decimals=decimals, symbol=symbol)

class MathTextSciFormatter(mticker.Formatter):
    """Scientific-notation formatter that renders exponents as math text."""

    def __init__(self, fmt="%1.2e"):
        """
        Initialize the object with a format string.
        
        Args:
        fmt (str): The format string to be used for formatting.
        """
        self.fmt = fmt
        
    def __call__(self, x, pos = None):
        # get formating
        s               = self.fmt % x
        decimal_point   = '.'
        positive_sign   = '+'
        tup             = s.split('e')
        significand     = tup[0].rstrip(decimal_point)
        sign            = tup[1][0].replace(positive_sign, '')
        exponent        = tup[1][1:].lstrip('0')
        if exponent:
            exponent    = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s           =  r'%s{\times}%s' % (significand, exponent)
        else:
            s           =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

def set_formatter(ax, formatter_type="sci", fmt="%1.2e", axis='xy'):
    """
    Sets the formatter for the given axis on the plot.
    
    fmt can take the value:
        - "%1.2e": Scientific notation with 2 decimal places.
        - "%1.2f": Fixed notation with 2 decimal places.
        - "%1.0f": Fixed notation with 0 decimal places.
        - "%1.0e": Scientific notation with 0 decimal places
        - "%1.0%": Percentage notation with 0 decimal places.
        - "%1.2%": Percentage notation with 2 decimal places.
        - "%1.0g": General notation with 0 decimal places.
        - "%1.2g": General notation with 2 decimal places.
        Integers:
        - "%d": Integer notation.
        - "%i": Integer notation.
        - "%u": Unsigned integer notation.
        - "%o": Octal notation.
        - "%x": Hexadecimal notation.
        - "%X": Hexadecimal notation.
        - "%c": Character notation.
        - "%r": Repr notation.    
    
    Args:
        ax (object): The axis object on which to set the formatter.
        formatter_type (str): The type of formatter to use. Options are "sci", "custom", "percent".
        fmt (str, optional): The format string for the axis labels. Defaults to "%1.2e". 
        axis (str, optional): The axis on which to set the formatter. Defaults to 'xy'.
    
    Additional formatter options:
        - "sci": Scientific notation formatter.
        - "custom": Custom formatter using a provided format string.
        - "percent": Percentage formatter.
        
    Returns:
        None
    """
    if formatter_type == "sci":
        formatter = MathTextSciFormatter(fmt)
    elif formatter_type == "custom":
        formatter = CustomFormatter(fmt)
    elif formatter_type == "percent":
        formatter = PercentFormatter()
    else:
        raise ValueError("Unsupported formatter type. Choose from 'sci', 'custom', 'percent'.")

    if 'y' in axis:
        ax.yaxis.set_major_formatter(formatter)
    if 'x' in axis:
        ax.xaxis.set_major_formatter(formatter)

###############################################

class _IgnoredAxisAttr:
    """Chainable no-op attribute proxy used by disabled axes."""

    def __init__(self, owner, path: str):
        self._owner = owner
        self._path  = str(path)

    def __call__(self, *args, **kwargs):
        self._owner._warn(self._path)
        return self

    def __getattr__(self, name):
        return _IgnoredAxisAttr(self._owner, f"{self._path}.{name}")

    def __getitem__(self, key):
        # Prevent Python's legacy sequence iteration fallback from looping forever
        # on proxies (e.g., tuple unpacking of ignored return values).
        if isinstance(key, int):
            raise IndexError(key)
        self._owner._warn(f"{self._path}[{key!r}]")
        return self

    def __setitem__(self, key, value):
        self._owner._warn(f"{self._path}[{key!r}]=...")

class IgnoredAxis:
    """
    No-op stand-in for a disabled axis.

    Any call or chained attribute access is ignored by default. Optionally,
    warnings can be emitted to make ignored operations explicit.
    """

    def __init__(self, *, index: Optional[int] = None, row_col: Optional[Tuple[int, int]] = None, names: Optional[List[str]] = None, warn: bool = False, reason: Optional[str] = None):
        self._qes_axis_disabled = True
        self._qes_axis_index    = index
        self._qes_axis_row_col  = row_col
        self._qes_axis_names    = list(names or [])
        self._qes_axis_warn     = bool(warn)
        self._qes_axis_reason   = str(reason) if reason else None

    @property
    def is_disabled(self) -> bool:
        """Return ``True`` for compatibility with regular axes checks."""
        return True

    def set_warn(self, enabled: bool = True):
        """Enable or disable warnings for ignored axis operations."""
        self._qes_axis_warn = bool(enabled)
        return self

    def description(self) -> str:
        """Return a compact description of the disabled axis target."""
        bits = []
        if self._qes_axis_names:
            bits.append(f"name={self._qes_axis_names}")
        if self._qes_axis_row_col is not None:
            bits.append(f"position={self._qes_axis_row_col}")
        if self._qes_axis_index is not None:
            bits.append(f"index={self._qes_axis_index}")
        if self._qes_axis_reason:
            bits.append(f"reason='{self._qes_axis_reason}'")
        return ", ".join(bits) if bits else "unknown axis"

    def _warn(self, op: str):
        if not self._qes_axis_warn:
            return
        warnings.warn(f"Ignored call '{op}' on disabled axis ({self.description()}).", RuntimeWarning, stacklevel=3,)

    def __call__(self, *args, **kwargs):
        self._warn("__call__")
        return self

    def __getattr__(self, name):
        # Do not emulate NumPy array protocol attributes; returning a proxy here
        # breaks np.asarray(..., dtype=object) used by AxesList grid indexing.
        if str(name).startswith("__array"):
            raise AttributeError(name)
        
        return _IgnoredAxisAttr(self, str(name))

    def __bool__(self):
        return False

    def __repr__(self):
        return f"IgnoredAxis({self.description()})"

class AxesList(list):
    """
    List-like container for subplot axes with optional grid-aware helpers.

    Behaviors:
    - Inherits from ``list`` (all list operations remain available).
    - Supports 2D indexing when grid metadata is available: ``axes[row, col]``.
    - Forwards unknown attribute access to the first axis, enabling
      single-axis-like usage in quick scripts.
    """

    def __init__(
        self,
        axes,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        panel_map: Optional[Dict[str, Any]] = None,
    ):
        '''
        Initialize the AxesList.
        '''
        super().__init__(list(axes))
        self.nrows          = int(nrows) if nrows is not None else None
        self.ncols          = int(ncols) if ncols is not None else None
        self._panel_map     = dict(panel_map or {})

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        """Grid shape ``(nrows, ncols)`` when known, otherwise ``None``."""
        if self.nrows is None or self.ncols is None:
            return None
        return (self.nrows, self.ncols)

    def first(self):
        """Return the first axis or raise if the container is empty."""
        if len(self) == 0:
            raise IndexError("AxesList is empty.")
        return self[0]

    # ---------------------------------
    # Panel management
    # ---------------------------------

    @property
    def panel_names(self) -> List[str]:
        """Names of registered semantic panels."""
        return list(self._panel_map.keys())

    def has_panel(self, name: str) -> bool:
        """Return whether a semantic panel name is registered."""
        return str(name) in self._panel_map

    def panel(self, name: str):
        """Return the axis or nested axes registered for ``name``."""
        key = str(name)
        if key not in self._panel_map:
            raise KeyError(f"Unknown panel '{name}'. Available: {self.panel_names}")
        return self._panel_map[key]

    def panels(self) -> Dict[str, Any]:
        """Return a copy of the semantic panel map."""
        return dict(self._panel_map)

    def rename_panel(self, old: str, new: str):
        """Rename a semantic panel while preserving its mapped axes."""
        old_k, new_k = str(old), str(new)
        if old_k not in self._panel_map:
            raise KeyError(f"Unknown panel '{old}'.")
        if new_k in self._panel_map and new_k != old_k:
            raise KeyError(f"Panel '{new}' already exists.")
        self._panel_map[new_k] = self._panel_map.pop(old_k)
        return self

    # ---------------------------------
    # Selection and indexing
    # ---------------------------------

 
    def select(self, *names: str):
        """Return an :class:`AxesList` containing the named panels."""
        selected = [self.panel(name) for name in names]
        flat = []
        for entry in selected:
            if isinstance(entry, AxesList):
                flat.extend(entry)
            elif isinstance(entry, list):
                flat.extend(entry)
            else:
                flat.append(entry)
        sub_map = {str(name): self.panel(name) for name in names}
        return AxesList(flat, panel_map=sub_map)

    # ---------------------------------
    # Operations
    # ---------------------------------

    def as_grid(self):
        """Return axes as a rectangular object array using stored grid shape."""
        if self.shape is None:
            raise ValueError("Grid shape is unknown for this AxesList.")
        
        expected      = int(self.nrows) * int(self.ncols)
        if len(self) != expected:
            raise ValueError(f"AxesList grid shape mismatch: len={len(self)} but nrows*ncols={expected}. Use flat indexing or ensure the layout has full rectangular occupancy.")
        return np.asarray(self, dtype=object).reshape(self.nrows, self.ncols)

    def at(self, row: int, col: int):
        """Return the axis at grid location ``(row, col)``."""
        if self.shape is None:
            raise ValueError("Grid shape is unknown for this AxesList.")
        return self[row, col]

    def span(self, rows, cols):
        """
        Return axes in a rectangular grid window.

        Examples:
        - ``axes.span(slice(0, 2), slice(1, 3))``
        - ``axes.span(0, slice(None))``
        """
        return self[rows, cols]

    def row(self, row: int):
        """Return one grid row as an :class:`AxesList`."""
        if self.shape is None:
            raise ValueError("Grid shape is unknown for this AxesList.")
        start   = row * self.ncols
        end     = start + self.ncols
        return AxesList(self[start:end], nrows=1, ncols=self.ncols)

    def col(self, col: int):
        """Return one grid column as an :class:`AxesList`."""
        if self.shape is None:
            raise ValueError("Grid shape is unknown for this AxesList.")
        return AxesList([self[r, col] for r in range(self.nrows)], nrows=self.nrows, ncols=1)

    # Convenience methods for applying functions to all axes

    def set_title(self, title, **kwargs):
        """Set the title for all axes."""
        return self.apply(lambda ax: ax.set_title(title, **kwargs))
    
    def apply(self, fn, *args, **kwargs):
        """Apply ``fn`` to each axis and return self."""
        for ax in self:
            fn(ax, *args, **kwargs)
        return self

    def _subset(self, items):
        """Return an AxesList subset while preserving matching panel aliases."""
        items_list = list(items)
        kept_ids = {id(ax) for ax in items_list}
        sub_map = {name: ax for name, ax in self._panel_map.items() if id(ax) in kept_ids}
        return AxesList(items_list, panel_map=sub_map)

    def disable(self, target, *, warn: bool = False, hide: bool = True, reason: Optional[str] = None):
        """
        Disable one or more axes by replacing them with no-op placeholders.

        Parameters
        ----------
        target : int | tuple[int, int] | str | Axes | list-like
            Axis selector. Supported forms:
            - flat index (int)
            - grid index ``(row, col)``
            - panel name (str)
            - an axis instance already contained in this AxesList
            - list/tuple/ndarray of the above (disabled recursively)
        warn : bool, default=False
            If True, any later operation on the disabled axis emits a warning.
        hide : bool, default=True
            If True and target is a real matplotlib axis, call ``set_axis_off``
            before disabling.
        reason : str, optional
            Optional note included in warning messages.

        Returns
        -------
        AxesList
            Returns ``self`` for chaining.
        """

        def _panel_names_for_axis(ax_obj) -> List[str]:
            return [name for name, ref in self._panel_map.items() if ref is ax_obj]

        def _row_col_for_index(index: int) -> Optional[Tuple[int, int]]:
            if self.shape is None or self.ncols in (None, 0):
                return None
            return (int(index // self.ncols), int(index % self.ncols))

        def _disable_index(index: int):
            if index < 0 or index >= len(self):
                raise IndexError(f"Axis index out of range: {index}")

            axis_obj = super(AxesList, self).__getitem__(index)
            if isinstance(axis_obj, IgnoredAxis):
                axis_obj.set_warn(warn)
                return

            if hide:
                try:
                    axis_obj.set_axis_off()
                except Exception:
                    pass

            names = _panel_names_for_axis(axis_obj)
            proxy = IgnoredAxis(index=index, row_col=_row_col_for_index(index), names=names, warn=warn, reason=reason)

            super(AxesList, self).__setitem__(index, proxy)

            for name, ref in list(self._panel_map.items()):
                if ref is axis_obj:
                    self._panel_map[name] = proxy

        if isinstance(target, tuple) and len(target) == 2:
            if self.shape is None:
                raise ValueError("Tuple indexing requires known grid shape.")
            row, col = target
            flat_idx = np.ravel_multi_index((int(row), int(col)), (self.nrows, self.ncols))
            _disable_index(int(flat_idx))
            return self

        if isinstance(target, (list, tuple, np.ndarray)):
            for item in list(np.asarray(target, dtype=object).ravel()):
                self.disable(item, warn=warn, hide=hide, reason=reason)
            return self

        if isinstance(target, str):
            item = self.panel(target)
            if isinstance(item, (AxesList, list, tuple, np.ndarray)):
                self.disable(item, warn=warn, hide=hide, reason=reason)
                return self
            try:
                idx = self.index(item)
            except ValueError as exc:
                raise KeyError(f"Panel '{target}' is not mapped to a direct axis.") from exc
            _disable_index(int(idx))
            return self

        if isinstance(target, (int, np.integer)):
            _disable_index(int(target))
            return self

        try:
            idx = self.index(target)
        except ValueError as exc:
            raise TypeError(
                "Unsupported target for AxesList.disable. Use index, (row,col), panel name, axis, or list-like of these."
            ) from exc
        _disable_index(int(idx))
        return self

    def collapse(self, *, redraw: bool = True):
        """
        Collapse rows that are fully disabled and reflow remaining axes.

        Behavior
        --------
        - A row is removed only if *all* entries in that row are disabled.
        - Partially disabled rows are kept intact, so grid holes remain.
        - Recomputes subplot positions for kept rows to remove vertical whitespace.

        Returns
        -------
        AxesList
            Returns ``self`` for chaining.
        """
        if self.shape is None:
            raise ValueError("Grid shape is unknown for this AxesList.")

        expected = int(self.nrows) * int(self.ncols)
        if len(self) != expected:
            raise ValueError(f"AxesList grid shape mismatch: len={len(self)} but nrows*ncols={expected}. Row compaction requires a full rectangular AxesList.")

        def _is_disabled(ax_obj) -> bool:
            return isinstance(ax_obj, IgnoredAxis) or bool(getattr(ax_obj, "_qes_axis_disabled", False))

        grid            = np.asarray(self, dtype=object).reshape(self.nrows, self.ncols)
        disabled_rows   = [r for r in range(self.nrows) if all(_is_disabled(grid[r, c]) for c in range(self.ncols))]

        if not disabled_rows:
            return self

        if len(disabled_rows) == int(self.nrows):
            warnings.warn("All rows are disabled; collapse() skipped.", RuntimeWarning, stacklevel=2)
            return self

        kept_rows   = [r for r in range(self.nrows) if r not in disabled_rows]
        fig         = None
        for r in kept_rows:
            for c in range(self.ncols):
                candidate = grid[r, c]
                if not _is_disabled(candidate):
                    fig = getattr(candidate, "figure", None)
                    if fig is not None:
                        break
            if fig is not None:
                break

        if fig is None:
            return self

        sp      = fig.subplotpars
        new_gs  = GridSpec(len(kept_rows), self.ncols, figure=fig, left=sp.left, right=sp.right, bottom=sp.bottom, top=sp.top, wspace=sp.wspace, hspace=sp.hspace)

        for new_r, old_r in enumerate(kept_rows):
            for c in range(self.ncols):
                ax = grid[old_r, c]
                if _is_disabled(ax):
                    continue
                slot = new_gs[new_r, c]
                ax.set_position(slot.get_position(fig))
                if hasattr(ax, "set_subplotspec"):
                    try:
                        ax.set_subplotspec(slot)
                    except Exception:
                        pass

        new_items       = [grid[r, c] for r in kept_rows for c in range(self.ncols)]
        self[:]         = new_items
        self.nrows      = len(kept_rows)

        kept_ids        = {id(ax) for ax in new_items}
        self._panel_map = {name: ax for name, ax in self._panel_map.items() if id(ax) in kept_ids}

        if redraw:
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

        return self

    def adjust(
        self,
        same: str = "xy",
        *,
        hide                    : str = "both",
        keep_x                  : str = "bottom",
        keep_y                  : str = "left",
        xlabel                  : Optional[str] = None,
        ylabel                  : Optional[str] = None,
        xlabel_kwargs           : Optional[dict] = None,
        ylabel_kwargs           : Optional[dict] = None,
        x_label_position        : Optional[str] = None,
        y_label_position        : Optional[str] = None,
        x_label_coords          : Optional[Tuple[float, float]] = None,
        y_label_coords          : Optional[Tuple[float, float]] = None,
        x_label_coords_system   : str = "axes",
        y_label_coords_system   : str = "axes",
        x_tick_params           : Optional[dict] = None,
        y_tick_params           : Optional[dict] = None,
        interior_x_tick_params  : Optional[dict] = None,
        interior_y_tick_params  : Optional[dict] = None,
    ):
        """
        Remove duplicated axis labels/ticklabels for multi-panel layouts.

        Parameters
        ----------
        same : {'x', 'y', 'xy'}, default='xy'
            Which directions should be de-duplicated.
        hide : {'both', 'labels', 'ticklabels'}, default='both'
            What to hide on interior panels.
        keep_x : {'bottom', 'top', 'all', 'vbottom', 'vtop'}, default='bottom'
            Which row keeps x-axis labels/ticklabels.
            ``vbottom``/``vtop`` additionally force visible labels/ticks to be
            drawn at the corresponding side (not only by grid-edge selection).
        keep_y : {'left', 'right', 'all', 'vleft', 'vright'}, default='left'
            Which column keeps y-axis labels/ticklabels.
            ``vleft``/``vright`` additionally force visible labels/ticks to be
            drawn at the corresponding side (not only by grid-edge selection).
        xlabel, ylabel : str, optional
            Label text applied to kept outer x/y axes.
        xlabel_kwargs, ylabel_kwargs : dict, optional
            Forwarded to ``ax.set_xlabel`` / ``ax.set_ylabel`` on kept axes.
        x_label_position, y_label_position : str, optional
            Manual label side position (x: ``top|bottom``, y: ``left|right``).
        x_label_coords, y_label_coords : tuple(float, float), optional
            Manual label coordinates for x/y labels.
        x_label_coords_system, y_label_coords_system : {'axes', 'data'}
            Coordinate system used for manual label coordinates.
        x_tick_params, y_tick_params : dict, optional
            Tick styling applied to kept x/y axes via ``ax.tick_params``.
        interior_x_tick_params, interior_y_tick_params : dict, optional
            Tick styling for interior x/y axes after de-duplication. Useful to
            keep ticks but hide labels, adjust lengths, etc.
        """
        key                     = str(same).lower().strip()
        do_x                    = "x" in key
        do_y                    = "y" in key
        hide_key                = str(hide).lower().strip()
        hide_labels             = hide_key in {"both", "label", "labels"}
        hide_ticklabels         = hide_key in {"both", "tick", "ticks", "ticklabel", "ticklabels"}
        xlabel_kwargs           = dict(xlabel_kwargs or {})
        ylabel_kwargs           = dict(ylabel_kwargs or {})
        x_tick_params           = dict(x_tick_params or {})
        y_tick_params           = dict(y_tick_params or {})
        interior_x_tick_params  = dict(interior_x_tick_params or {})
        interior_y_tick_params  = dict(interior_y_tick_params or {})

        keep_x = str(keep_x).lower().strip()
        keep_y = str(keep_y).lower().strip()
        if keep_x not in {"bottom", "top", "all", "vbottom", "vtop"}:
            raise ValueError("keep_x must be one of: 'bottom', 'top', 'all', 'vbottom', 'vtop'.")
        if keep_y not in {"left", "right", "all", "vleft", "vright"}:
            raise ValueError("keep_y must be one of: 'left', 'right', 'all', 'vleft', 'vright'.")

        force_x_side = None
        if keep_x in {"vbottom", "vtop"}:
            force_x_side = "bottom" if keep_x == "vbottom" else "top"
            keep_x = force_x_side
        force_y_side = None
        if keep_y in {"vleft", "vright"}:
            force_y_side = "left" if keep_y == "vleft" else "right"
            keep_y = force_y_side

        def _edge_flags(ax, idx):
            # Prefer subplot-spec edge info for spans/mosaic layouts.
            try:
                ss = ax.get_subplotspec()
                return {
                    "first_row" : bool(ss.is_first_row()),
                    "last_row"  : bool(ss.is_last_row()),
                    "first_col" : bool(ss.is_first_col()),
                    "last_col"  : bool(ss.is_last_col()),
                }
            except Exception:
                pass

            # Fallback for regular flattened grids.
            if self.shape is not None and self.ncols is not None and self.ncols > 0:
                row = idx // self.ncols
                col = idx % self.ncols
                return {
                    "first_row" : row == 0,
                    "last_row"  : row == (self.nrows - 1),
                    "first_col" : col == 0,
                    "last_col"  : col == (self.ncols - 1),
                }
            return {"first_row": True, "last_row": True, "first_col": True, "last_col": True}

        for idx, ax in enumerate(self):
            flags = _edge_flags(ax, idx)

            if do_x:
                keep_this_x = (
                    True
                    if keep_x == "all"
                    else flags["last_row"] if keep_x == "bottom"
                    else flags["first_row"]
                )
                if not keep_this_x:
                    if hide_ticklabels:
                        ax.tick_params(axis="x", which="both", labelbottom=False, labeltop=False)
                    if interior_x_tick_params:
                        ax.tick_params(axis="x", which="both", **interior_x_tick_params)
                    if hide_labels:
                        ax.set_xlabel("")
                else:
                    if xlabel is not None:
                        ax.set_xlabel(xlabel, **xlabel_kwargs)
                    if x_label_position in {"top", "bottom"}:
                        ax.xaxis.set_label_position(x_label_position)
                    elif force_x_side is not None:
                        ax.xaxis.set_label_position(force_x_side)
                    if x_label_coords is not None:
                        x_t = ax.transData if str(x_label_coords_system).lower() == "data" else ax.transAxes
                        ax.xaxis.set_label_coords(float(x_label_coords[0]), float(x_label_coords[1]), transform=x_t)
                    if force_x_side is not None:
                        if force_x_side == "top":
                            ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False)
                        else:
                            ax.tick_params(axis="x", which="both", top=False, bottom=True, labeltop=False, labelbottom=True)
                    if x_tick_params:
                        ax.tick_params(axis="x", which="both", **x_tick_params)

            if do_y:
                keep_this_y = (
                    True
                    if keep_y == "all"
                    else flags["first_col"] if keep_y == "left"
                    else flags["last_col"]
                )
                if not keep_this_y:
                    if hide_ticklabels:
                        ax.tick_params(axis="y", which="both", labelleft=False, labelright=False)
                    if interior_y_tick_params:
                        ax.tick_params(axis="y", which="both", **interior_y_tick_params)
                    if hide_labels:
                        ax.set_ylabel("")
                else:
                    if ylabel is not None:
                        ax.set_ylabel(ylabel, **ylabel_kwargs)
                    if y_label_position in {"left", "right"}:
                        ax.yaxis.set_label_position(y_label_position)
                    elif force_y_side is not None:
                        ax.yaxis.set_label_position(force_y_side)
                    if y_label_coords is not None:
                        y_t = ax.transData if str(y_label_coords_system).lower() == "data" else ax.transAxes
                        ax.yaxis.set_label_coords(float(y_label_coords[0]), float(y_label_coords[1]), transform=y_t)
                    if force_y_side is not None:
                        if force_y_side == "right":
                            ax.tick_params(axis="y", which="both", right=True, left=False, labelright=True, labelleft=False)
                        else:
                            ax.tick_params(axis="y", which="both", right=False, left=True, labelright=False, labelleft=True)
                    if y_tick_params:
                        ax.tick_params(axis="y", which="both", **y_tick_params)

        return self

    # ---------------------------------

    def __getitem__(self, key):
        ''' Support panel name access and 2D grid indexing.'''
        
        if isinstance(key, str):
            return self.panel(key)
        
        if isinstance(key, tuple):
            if self.shape is None:
                raise ValueError("Tuple indexing requires known grid shape.")
        
            if len(key) != 2:
                raise IndexError("Use axes[row, col] for 2D indexing.")
            row, col    = key

            # Fast path for scalar tuple indexing without grid reshape.
            if isinstance(row, (int, np.integer)) and isinstance(col, (int, np.integer)):
                flat_idx = int(np.ravel_multi_index((int(row), int(col)), (self.nrows, self.ncols)))
                return super().__getitem__(flat_idx)

            grid        = self.as_grid()
            result      = grid[row, col]
            if isinstance(result, np.ndarray):
                flat = result.ravel().tolist()
                return self._subset(flat)
            return result

        # Standard slicing / fancy indexing should stay AxesList-aware.
        if isinstance(key, slice):
            return self._subset(super().__getitem__(key))
        if isinstance(key, (list, tuple, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == bool:
                if arr.size != len(self):
                    raise IndexError("Boolean index length must match AxesList length.")
                idx = np.flatnonzero(arr)
                return self._subset([super().__getitem__(int(i)) for i in idx])
            return self._subset([super().__getitem__(int(i)) for i in arr.ravel()])

        return super().__getitem__(key)

    def __getattr__(self, name):
        # Convenience passthrough for quick one-axis-like usage.
        if len(self) == 0:
            raise AttributeError(name)
        return getattr(self[0], name)

# ---------------------------------

class Plotter:
    """
    Publication-quality plotting utilities for scientific computing.
    
    This class provides static methods for creating, customizing, and saving
    Matplotlib figures suitable for scientific journals (Nature, Science, PRL, etc.).
    
    All methods are @staticmethod, so you can use them without instantiation::
    
    >>> Plotter.plot(ax, x, y, color='C0', label='Data')
    >>> Plotter.set_legend(ax, style='publication')
    
    Main Categories
    ---------------
    **Plotting Methods**    : plot, scatter, tripcolor_field, semilogy, semilogx, loglog, errorbar, fill_between, histogram
    **Axis Setup**          : set_ax_params, set_tickparams, setup_log_x, setup_log_y
    **Annotations**         : set_annotate, set_annotate_letter, set_arrow
    **Colorbars**           : add_colorbar, get_colormap, discrete_colormap  
    **Layouts**             : get_subplots, get_grid, get_inset
    **Legends**             : set_legend, set_legend_custom
    **Saving**              : save_fig, savefig
    
    For full documentation, call: Plotter.help()
    """
    
    # Publication-ready default settings
    _PUBLICATION_DEFAULTS = {
        'linewidth'     : 1.3,
        'markersize'    : 4.2,
        'capsize'       : 2.5,
        'fontsize'      : 10,
        'tick_length'   : 4,
        'tick_width'    : 0.9,
    }
    
    def __init__(self, default_cmap='viridis', font_size=12, dpi=200):
        """
        Initialize the Plotter with default parameters.
        
        Parameters
        ----------
        default_cmap : str, default='viridis'
            Default colormap for heatmaps and colorbars.
        font_size : int, default=12
            Default font size for labels and text.
        dpi : int, default=200
            Resolution for rasterized output (PNG, TIFF).
        
        Note
        ----
        Most methods are @staticmethod and don't require instantiation.
        Use the class directly: `Plotter.plot(ax, x, y)`
        """
        self.default_cmap   = default_cmap
        self.font_size      = font_size
        self.dpi            = dpi
        
    def ax_off(self, ax: Union[plt.Axes, List[plt.Axes]]):
        """
        Completely turn off the axis (no ticks, labels, spines, data)."""
        """"""
        if isinstance(ax, (list, tuple)):
            for a in ax:
                self.ax_off(a)
        elif isinstance(ax, plt.Axes):
            ax.set_axis_off()
        elif is_callable(ax):
            ax(self.ax_off)
        else:
            raise TypeError("ax must be a matplotlib Axes, list of Axes, or callable that accepts an ax_off function.")
    
    @staticmethod
    def ax(ax: Union[plt.Axes, List[plt.Axes]], *args, **kwargs):
        """
        Alias for ax method to allow direct calls.
        """
        if isinstance(ax, (list, tuple)):
            return ax[kwargs.get('index', 0)]
        elif isinstance(ax, IgnoredAxis):
            return ax
        elif isinstance(ax, plt.Axes):
            return ax
        elif is_callable(ax):
            return ax(*args, **kwargs)
        else:
            return ax

    @staticmethod
    def disable(axes, target, *,
        warn    : bool = False,
        hide    : bool = True,
        reason  : Optional[str] = None,
    ) -> AxesList:
        """
        Convenience wrapper for ``AxesList.disable``.

        Parameters
        ----------
        axes : AxesList or list-like of axes
            Axes container to operate on.
        target : selector
            Forwarded to :meth:`AxesList.disable`.
        warn : bool, default=False
            Emit warnings when disabled axes are used.
        hide : bool, default=True
            Hide underlying axes before disabling.
        reason : str, optional
            Optional warning context.

        Returns
        -------
        AxesList
            The modified axes list.
        """
        if isinstance(axes, AxesList):
            return axes.disable(target, warn=warn, hide=hide, reason=reason)
        if isinstance(axes, (list, tuple, np.ndarray)):
            wrapped = AxesList(list(axes))
            return wrapped.disable(target, warn=warn, hide=hide, reason=reason)
        raise TypeError("axes must be an AxesList or list-like of axes.")
    
    @staticmethod
    def help(topic: str = None):
        """
        Print help information about available plotting methods.
        
        Parameters
        ----------
        topic : str, optional
            Specific topic to get help on. Options:
            - 'plot': Basic plotting methods
            - 'axis': Axis configuration
            - 'color': Colors and colorbars
            - 'layout': Subplots and grids
            - 'save': Saving figures
            - None: Print overview of all topics
        
        Examples
        --------
        >>> Plotter.help()           # Overview
        >>> Plotter.help('plot')     # Plotting methods
        >>> Plotter.help('axis')     # Axis configuration
        """
        from .plotters.help import PLOTTER_HELP
        
        if topic is None:
            print(PLOTTER_HELP['overview'])
        elif topic.lower() in PLOTTER_HELP:
            print(PLOTTER_HELP[topic.lower()])
        else:
            print(f"Unknown topic: '{topic}'. Available: plot, axis, color, layout, grid, legend, annotate, style, plotters, save")
    
    ###########################################################
    #! Utilities for plotting
    ###########################################################

    @staticmethod
    def plot_style(**kwargs) -> "PlotStyle":
        """Return a PlotStyle config instance."""
        from .plotters.config import PlotStyle
        return PlotStyle(**kwargs)

    @staticmethod
    def kspace_config(**kwargs) -> "KSpaceConfig":
        """Return a KSpaceConfig instance."""
        from .plotters.config import KSpaceConfig
        return KSpaceConfig(**kwargs)

    @staticmethod
    def kpath_config(**kwargs) -> "KPathConfig":
        """Return a KPathConfig instance."""
        from .plotters.config import KPathConfig
        return KPathConfig(**kwargs)

    @staticmethod
    def spectral_config(**kwargs) -> "SpectralConfig":
        """Return a SpectralConfig instance."""
        from .plotters.config import SpectralConfig
        return SpectralConfig(**kwargs)

    @staticmethod
    def figure_config(**kwargs) -> "FigureConfig":
        """Return a FigureConfig instance."""
        from .plotters.config import FigureConfig
        return FigureConfig(**kwargs)

    @staticmethod
    def plotters():
        """Expose the ``general_python.common.plotters`` package."""
        from . import plotters
        return plotters

    @staticmethod
    def statistical_fitter():
        """Backward-compatible alias for :meth:`fitter`."""
        return Plotter.fitter()

    @staticmethod
    def fitter():
        """Expose shared fitting/scaling helpers from ``general_python.maths.math_utils``."""
        from QES.general_python.maths.math_utils import Fitter
        return Fitter

    @staticmethod
    def math(label: str, *, auto_wrap: bool = True, escape_text: bool = True, **values: Any) -> str:
        r"""
        Build a LaTeX-ready math label from a template.

        This method extends Python ``str.format``-style placeholders with simple
        math filters for scientific labels.

        Parameters
        ----------
        label : str
            Template string. Standard format fields are supported, e.g.
            ``{J:.3g}``, and can be combined with filters, e.g.
            ``{point|vec:.2f}``, ``{kpoint|sym}``.
            Greek-name values (for example ``Gamma`` or ``omega``) are
            automatically rendered as LaTeX variables.
        auto_wrap : bool, default=True
            If True, wrap the final string with ``$...$`` when no dollar sign is
            present in the rendered output.
        escape_text : bool, default=True
            If True, plain substituted strings are LaTeX-escaped by default.
            Use ``|raw`` or ``|tex`` to bypass escaping for a specific field.
        **values : Any
            Values used by template fields.

        Supported filters
        -----------------
        ``raw`` / ``tex``
            Insert value as-is (no escaping).
        ``sym``
            Force symbol conversion for common Greek names
            (e.g. ``Gamma`` -> ``\Gamma``).
        ``num``
            Numeric formatting helper. Uses format spec if provided, otherwise
            ``.6g``.
        ``vec``
            Render iterable as ``\left(v_1, v_2, ...\right)``.
        ``set``
            Render iterable as ``\left\{v_1, v_2, ...\right\}``.

        Returns
        -------
        str
            Rendered LaTeX/mathtext-compatible label.

        Examples
        --------
        >>> Plotter.math(r"\\langle S_i^z \\rangle = {value|num:.3e}", value=1.2e-4)
        '$\\langle S_i^z \\rangle = 1.200e-04$'
        >>> Plotter.math(r"{kx|sym}-{ky|sym} path, q={q|vec:.2f}", kx="Gamma", ky="K", q=[0, 1/3])
        '$\\Gamma-K path, q=\\left(0, 0.33\\right)$'
        >>> Plotter.math(r"E={expr|raw}", expr=r"E_0 + \\Delta")
        '$E=E_0 + \\Delta$'
        """

        greek_map = {
            "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma", "delta": r"\delta",
            "epsilon": r"\epsilon", "zeta": r"\zeta", "eta": r"\eta", "theta": r"\theta",
            "iota": r"\iota", "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
            "nu": r"\nu", "xi": r"\xi", "pi": r"\pi", "rho": r"\rho", "sigma": r"\sigma",
            "tau": r"\tau", "upsilon": r"\upsilon", "phi": r"\phi", "chi": r"\chi",
            "psi": r"\psi", "omega": r"\omega",
            # large letters
            "Gamma": r"\Gamma", "Delta": r"\Delta", "Theta": r"\Theta", "Lambda": r"\Lambda",
            "Xi": r"\Xi", "Pi": r"\Pi", "Sigma": r"\Sigma", "Upsilon": r"\Upsilon",
            "Phi": r"\Phi", "Psi": r"\Psi", "Omega": r"\Omega",
            "Gamma'": r"\Gamma^{\prime}", "Kp" : r'K^{\prime}', "M'": r"M^{\prime}",
        }

        tex_escape_map = {
            "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }

        def _escape_tex(text: str) -> str:
            return "".join(tex_escape_map.get(ch, ch) for ch in text)

        def _resolve_field(name: str) -> Any:
            current: Any = values
            for token in name.split("."):
                if isinstance(current, dict):
                    if token not in current:
                        raise KeyError(
                            f"Missing key '{token}' while resolving field '{name}'. "
                            f"Available top-level keys: {sorted(values.keys())}"
                        )
                    current = current[token]
                elif isinstance(current, (list, tuple, np.ndarray)) and token.isdigit():
                    current = current[int(token)]
                else:
                    if not hasattr(current, token):
                        raise KeyError(f"Cannot resolve '{token}' in field '{name}'.")
                    current = getattr(current, token)
            return current

        def _fmt_scalar(v: Any, spec: str = "") -> str:
            if isinstance(v, (int, float, np.integer, np.floating)):
                return format(v, spec if spec else ".6g")
            txt = str(v)
            return greek_map.get(txt, txt)

        def _iterable(v: Any) -> bool:
            return isinstance(v, (list, tuple, np.ndarray))

        formatter       = string.Formatter()
        out: List[str]  = []
        has_fields      = False

        for literal_text, field_name, format_spec, conversion in formatter.parse(label):
            out.append(literal_text)
            if field_name is None:
                continue
            has_fields = True

            field_expr = field_name.strip()
            if not field_expr:
                raise ValueError("Empty field expression '{}' is not supported in Plotter.math().")

            name, *filters  = [part.strip() for part in field_expr.split("|")]
            value           = _resolve_field(name)
            if conversion:
                value       = formatter.convert_field(value, conversion)

            raw_mode = False
            for flt in filters:
                if flt in ("raw", "tex"):
                    raw_mode = True
                elif flt == "sym":
                    vtxt        = str(value)
                    value       = greek_map.get(vtxt, vtxt)
                    raw_mode    = True
                elif flt == "num":
                    value       = _fmt_scalar(value, format_spec)
                    format_spec = ""
                elif flt == "vec":
                    if _iterable(value):
                        items   = [_fmt_scalar(v, format_spec) for v in value]
                        value   = r"\left(" + ", ".join(items) + r"\right)"
                    else:
                        value   = r"\left(" + _fmt_scalar(value, format_spec) + r"\right)"
                    format_spec = ""
                    raw_mode    = True
                elif flt == "set":
                    if _iterable(value):
                        items   = [_fmt_scalar(v, format_spec) for v in value]
                        value   = r"\left\{" + ", ".join(items) + r"\right\}"
                    else:
                        value   = r"\left\{" + _fmt_scalar(value, format_spec) + r"\right\}"
                    format_spec = ""
                    raw_mode    = True
                elif flt:
                    raise ValueError(f"Unknown Plotter.math() filter '{flt}'. Supported filters: raw, tex, sym, num, vec, set.")

            if format_spec and not isinstance(value, str):
                value = format(value, format_spec)
            elif format_spec and isinstance(value, str):
                value = format(value, format_spec)

            text = str(value)
            if not raw_mode and text in greek_map:
                text        = greek_map[text]
                raw_mode    = True
            if escape_text and (not raw_mode):
                text        = _escape_tex(text)
            out.append(text)

        rendered = "".join(out)
        if not has_fields:
            stripped = rendered.strip()
            if stripped in greek_map:
                rendered = rendered.replace(stripped, greek_map[stripped], 1)
        if auto_wrap and "$" not in rendered:
            rendered = f"${rendered}$"
        return rendered
    
    @staticmethod
    def ensure_list(x):
        return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

    @staticmethod
    def unify_limits(axes, which='y'):
        
        # Unify the limits of the given axes for either 'x' or 'y' axis.
        axes    = Plotter.ensure_list(axes)
        if which == 'y':
            limits  = [ax.get_ylim() for ax in axes]
            min_l   = min(l[0] for l in limits)
            max_l   = max(l[1] for l in limits)
            for ax in axes:
                ax.set_ylim(min_l, max_l)
        elif which == 'x':
            limits  = [ax.get_xlim() for ax in axes]
            min_l   = min(l[0] for l in limits)
            max_l   = max(l[1] for l in limits)
            for ax in axes:
                ax.set_xlim(min_l, max_l)

    @staticmethod
    def resolve_planar_limits(
        points,
        *,
        limits          : Optional[Union[tuple, list, np.ndarray]] = None,
        x_limits        : Optional[Union[tuple, list, np.ndarray]] = None,
        y_limits        : Optional[Union[tuple, list, np.ndarray]] = None,
        xmin            : Optional[float] = None,
        xmax            : Optional[float] = None,
        ymin            : Optional[float] = None,
        ymax            : Optional[float] = None,
        limit_to_pi     : bool = False,
        pad_fraction    : float = 0.08,
    ) -> Tuple[tuple, tuple]:
        """
        Resolve visible ``(xlim, ylim)`` for planar data.

        Parameters
        ----------
        points : array-like
            Planar sample points shaped like ``(N, 2)`` or ``(N, D)``.
        limits : sequence, optional
            Explicit bounds. Length 2 means shared ``(min, max)`` for both axes.
            Length 4 means ``(xmin, xmax, ymin, ymax)``.
        x_limits, y_limits : sequence, optional
            Explicit bounds for each axis separately.
        xmin, xmax, ymin, ymax : float, optional
            Scalar axis bound overrides. These take precedence over inferred
            limits and can refine ``limits`` / ``x_limits`` / ``y_limits``.
        limit_to_pi : bool, default=False
            If True and ``limits`` is not provided, use ``[-pi, pi]`` on both axes.
        pad_fraction : float, default=0.08
            Relative padding applied when limits are inferred from ``points``.
        """
        if limits is not None:
            resolved = tuple(float(v) for v in limits)
            if len(resolved) == 2:
                return (resolved[0], resolved[1]), (resolved[0], resolved[1])
            if len(resolved) == 4:
                return (resolved[0], resolved[1]), (resolved[2], resolved[3])
            raise ValueError("limits must have length 2 or 4.")

        if limit_to_pi:
            xlim, ylim = (-np.pi, np.pi), (-np.pi, np.pi)
        else:
            pts = np.asarray(points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] < 2:
                raise ValueError("points must be shaped like (N, 2) or (N, D) with D >= 2.")

            mins = np.min(pts[:, :2], axis=0)
            maxs = np.max(pts[:, :2], axis=0)
            span = np.maximum(maxs - mins, 1e-12)
            pad = float(pad_fraction) * span
            xlim, ylim = (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])

        if x_limits is not None:
            resolved = tuple(float(v) for v in x_limits)
            if len(resolved) != 2:
                raise ValueError("x_limits must have length 2.")
            xlim = (resolved[0], resolved[1])
        if y_limits is not None:
            resolved = tuple(float(v) for v in y_limits)
            if len(resolved) != 2:
                raise ValueError("y_limits must have length 2.")
            ylim = (resolved[0], resolved[1])

        xlim = (
            float(xmin) if xmin is not None else float(xlim[0]),
            float(xmax) if xmax is not None else float(xlim[1]),
        )
        ylim = (
            float(ymin) if ymin is not None else float(ylim[0]),
            float(ymax) if ymax is not None else float(ylim[1]),
        )
        return xlim, ylim
    
    ###########################################################
    
    @staticmethod
    def markers():
        ''' Markers with common options for line and scatter plots.'''
        return markersList
    @staticmethod
    def markersC():
        ''' Markers cycle with common options for line and scatter plots.'''
        return markersCycle
    
    @staticmethod
    def colors():
        ''' Colors with common options for line and scatter plots.'''
        return colorsList

    @staticmethod
    def linestyles():
        ''' Linestyles with solid and dashed options.'''
        return linestylesList
    @staticmethod
    def linestylesC():
        ''' Linestyles cycle with solid and dashed options.'''
        return linestylesCycle
    @staticmethod
    def linestylesCE():
        ''' Extended linestyles cycle with more options for dashed lines.'''
        return linestylesCycleExtended

    ###########################################################
    #! Color Utilities
    ###########################################################

    # ------------------------------------------------------------------
    # Named Palettes registry
    # Carefully curated for scientific publication use. All entries have
    # been verified against colorblindness simulators where noted.
    # ------------------------------------------------------------------
    _PALETTES: Dict[str, List[str]] = {
        # ---- Colorblind-safe (recommended for all publications) ------
        # Wong (2011) Nature Methods – the gold standard for CBF plots
        'wong':         ["#000000","#E69F00","#56B4E9","#009E73",
                         "#F0E442","#0072B2","#D55E00","#CC79A7"],
        # Identical to wong; alias used by Okabe & Ito (2008)
        'okabe':        ["#000000","#E69F00","#56B4E9","#009E73",
                         "#F0E442","#0072B2","#D55E00","#CC79A7"],
        # Paul Tol muted qualitative – great for dark backgrounds too
        'tol':          ["#332288","#88CCEE","#44AA99","#117733",
                         "#999933","#DDCC77","#CC6677","#882255",
                         "#AA4499","#DDDDDD"],
        # Paul Tol bright – higher contrast on white
        'tol_bright':   ["#4477AA","#EE6677","#228833","#CCBB44",
                         "#66CCEE","#AA3377","#BBBBBB"],
        # IBM Carbon accessible palette (5 colors)
        'ibm':          ["#648FFF","#785EF0","#DC267F","#FE6100","#FFB000"],
        # ---- Perceptual / seaborn-style ----------------------------
        # seaborn colorblind-10
        'colorblind':   ["#0173B2","#DE8F05","#029E73","#D55E00",
                         "#CC78BC","#CA9161","#FBAFE4","#949494",
                         "#ECE133","#56B4E9"],
        # seaborn deep-10
        'deep':         ["#4C72B0","#DD8452","#55A868","#C44E52",
                         "#8172B3","#937860","#DA8BC3","#8C8C8C",
                         "#CCB974","#64B5CD"],
        # seaborn muted-10 (toned-down, print-friendly)
        'muted':        ["#4878D0","#EE854A","#6ACC65","#D65F5F",
                         "#956CB4","#8C613C","#DC7EC0","#797979",
                         "#D5BB67","#82C6E2"],
        # ---- Standard Matplotlib cycles ----------------------------
        'tableau':      list(mcolors.TABLEAU_COLORS.values()),
        'classic':      ["#1F77B4","#FF7F0E","#2CA02C","#D62728",
                         "#9467BD","#8C564B","#E377C2","#7F7F7F",
                         "#BCBD22","#17BECF"],
        # ---- Journal-style palettes --------------------------------
        # Nature / BioRxiv warm editorial tones
        'nature':       ["#E64B35","#4DBBD5","#00A087","#3C5488",
                         "#F39B7F","#8491B4","#91D1C2","#DC0000",
                         "#7E6148","#B09C85"],
        # Science journal-inspired muted palette
        'science':      ["#3B4992","#EE0000","#008B45","#631879",
                         "#008280","#BB0021","#5F559B","#A20056",
                         "#808180","#1B1919"],
        # ---- Presentation / poster --------------------------------
        # Soft pastel – good for talk backgrounds
        'pastel':       ["#AEC6CF","#FFD1DC","#B5EAD7","#FFDAC1",
                         "#C7CEEA","#E2F0CB","#F8C8D4","#D4E6F1"],
        # ---- Diverging sequences ----------------------------------
        # 9-stop cool→warm (Blue→Red) diverging run
        'sunset':       ["#364B9A","#4A7BB7","#98CAE1","#EAECCC",
                         "#FEDA8B","#FDB366","#F67E4B","#DD3D2D","#A50026"],
    }

    @staticmethod
    def palette(name: str = 'tableau', n: Optional[int] = None) -> List[str]:
        """
        Return a named color palette as a list of hex strings.

        Parameters
        ----------
        name : str, default='tableau'
            Palette name. Built-in options:

            =============== =====================================================
            Name            Description
            =============== =====================================================
            ``wong``        Wong (2011) 8-color CBF palette (Nature Methods)
            ``okabe``       Okabe & Ito – identical to ``wong``
            ``tol``         Paul Tol's muted 10-color qualitative set
            ``tol_bright``  Paul Tol's bright 7-color high-contrast set
            ``ibm``         IBM Carbon 5-color accessible palette
            ``colorblind``  seaborn *colorblind* 10-color cycle
            ``deep``        seaborn *deep* 10-color perceptual palette
            ``muted``       seaborn *muted* toned-down palette
            ``tableau``     Matplotlib default Tableau-10 cycle
            ``classic``     Matplotlib pre-2.0 default cycle
            ``nature``      Nature/BioRxiv warm editorial palette
            ``science``     Science journal-inspired muted palette
            ``pastel``      Soft pastel tones for presentations
            ``sunset``      9-stop cool→warm diverging gradient
            =============== =====================================================

        n : int, optional
            Return exactly ``n`` colors. When ``n > len(palette)`` colors are
            repeated cyclically.

        Returns
        -------
        list[str]
            Hex color strings.

        Examples
        --------
        >>> Plotter.palette('wong')           # 8 colorblind-safe colors
        >>> Plotter.palette('nature', n=4)    # first 4 of nature palette
        >>> Plotter.palette('deep', n=12)     # 12 colors (cycles)
        """
        p = Plotter._PALETTES.get(name)
        if p is None:
            raise ValueError(
                f"Unknown palette '{name}'. "
                f"Available: {sorted(Plotter._PALETTES.keys())}"
            )
        if n is None:
            return list(p)
        return [p[i % len(p)] for i in range(n)]

    @staticmethod
    def palette_cycle(name: str = 'tableau') -> itertools.cycle:
        """
        Return an infinite ``itertools.cycle`` over a named palette.

        Parameters
        ----------
        name : str
            Same keys as :meth:`palette`.

        Examples
        --------
        >>> cyc   = Plotter.palette_cycle('wong')
        >>> color = next(cyc)
        """
        return itertools.cycle(Plotter.palette(name))

    @staticmethod
    def colorsC(palette: str = 'tableau') -> itertools.cycle:
        """
        Return a color cycle for a named palette.

        Alias for :meth:`palette_cycle`.

        Examples
        --------
        >>> cyc = Plotter.colorsC('wong')
        >>> c1  = next(cyc)
        """
        return Plotter.palette_cycle(palette)

    @staticmethod
    def colorsN(n: int, palette: str = 'tableau') -> List[str]:
        """
        Return exactly ``n`` colors from a named palette (cycling if needed).

        Parameters
        ----------
        n : int
        palette : str
            Named palette (see :meth:`palette`).

        Examples
        --------
        >>> c5 = Plotter.colorsN(5, 'wong')
        """
        return Plotter.palette(palette, n=n)

    @staticmethod
    def set_color_cycle(ax, palette: Union[str, List] = 'tableau') -> None:
        """
        Set the Matplotlib color cycle on one or more axes.

        This makes subsequent ``ax.plot(...)`` calls auto-pick colors from the
        selected palette in order.

        Parameters
        ----------
        ax : axes or list of axes
        palette : str or list of colors, default='tableau'
            Named palette string (see :meth:`palette`) or an explicit list of
            any Matplotlib-compatible color specs.

        Examples
        --------
        >>> Plotter.set_color_cycle(ax, 'wong')
        >>> ax.plot(x1, y1)   # first wong color
        >>> ax.plot(x2, y2)   # second wong color
        """
        colors = Plotter.palette(palette) if isinstance(palette, str) else list(palette)
        cycler = mpl.cycler(color=colors)
        for a in Plotter.ensure_list(ax):
            a.set_prop_cycle(cycler)

    @staticmethod
    def apply_palette(axes, palette: str = 'tableau') -> None:
        """
        Apply a named color palette as the default color cycle for one or
        more axes. Shorthand for :meth:`set_color_cycle`.

        Examples
        --------
        >>> fig, axes = Plotter.get_subplots(1, 3)
        >>> Plotter.apply_palette(axes, 'wong')
        """
        Plotter.set_color_cycle(axes, palette)

    # ------------------------------------------------------------------
    # Color Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def to_rgba(color, alpha: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        Convert any Matplotlib-compatible color spec to an ``(r, g, b, a)`` tuple.

        Parameters
        ----------
        color : color spec
            Named string, hex, RGB/RGBA tuple, ``'C0'``-style, etc.
        alpha : float, optional
            Override the alpha channel (0–1).

        Returns
        -------
        tuple[float, float, float, float]

        Examples
        --------
        >>> Plotter.to_rgba('C0')
        >>> Plotter.to_rgba('#E64B35', alpha=0.5)
        """
        r, g, b, a = mcolors.to_rgba(color)
        if alpha is not None:
            a = float(alpha)
        return (r, g, b, a)

    @staticmethod
    def to_hex(color, keep_alpha: bool = False) -> str:
        """
        Convert any Matplotlib-compatible color spec to a hex string.

        Parameters
        ----------
        color : color spec
        keep_alpha : bool, default=False
            If True, return an 8-character ``#RRGGBBAA`` string.

        Examples
        --------
        >>> Plotter.to_hex('C0')              # '#1f77b4'
        >>> Plotter.to_hex((0.2, 0.4, 0.6, 0.8), keep_alpha=True)
        """
        return mcolors.to_hex(color, keep_alpha=keep_alpha)

    # ------------------------------------------------------------------
    # Color Manipulation (HLS-based, perceptually motivated)
    # ------------------------------------------------------------------

    @staticmethod
    def adjust_color(
        color,
        *,
        lighten     : float = 0.0,
        darken      : float = 0.0,
        saturate    : float = 0.0,
        desaturate  : float = 0.0,
        alpha       : Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Perceptually adjust a color in HLS space.

        Each parameter shifts the corresponding channel by a *fraction of the
        remaining headroom*, so operations compose gracefully and values are
        always clamped to [0, 1].

        Parameters
        ----------
        color : color spec
            Any Matplotlib-compatible color.
        lighten : float, default=0.0
            Push lightness toward 1 (white). 0 = no change, 1 = white.
        darken : float, default=0.0
            Push lightness toward 0 (black). 0 = no change, 1 = black.
        saturate : float, default=0.0
            Push saturation toward 1. 0 = no change, 1 = fully saturated.
        desaturate : float, default=0.0
            Push saturation toward 0 (grey). 0 = no change, 1 = grey.
        alpha : float, optional
            Override alpha channel (0–1).

        Returns
        -------
        tuple[float, float, float, float]
            Adjusted RGBA color.

        Examples
        --------
        >>> Plotter.adjust_color('C0', lighten=0.3)
        >>> Plotter.adjust_color('#E64B35', darken=0.4)
        >>> Plotter.adjust_color('C2', desaturate=0.5, alpha=0.7)
        """
        import colorsys
        r, g, b, a = mcolors.to_rgba(color)
        h, l, s    = colorsys.rgb_to_hls(r, g, b)

        l = l + (1.0 - l) * max(0.0, min(1.0, float(lighten)))
        l = l * (1.0 - max(0.0, min(1.0, float(darken))))
        s = s + (1.0 - s) * max(0.0, min(1.0, float(saturate)))
        s = s * (1.0 - max(0.0, min(1.0, float(desaturate))))

        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        return (r2, g2, b2, float(alpha) if alpha is not None else a)

    @staticmethod
    def lighten(color, amount: float = 0.3) -> Tuple[float, float, float, float]:
        """
        Return a lightened version of *color* (push lightness toward white).

        Parameters
        ----------
        color : color spec
        amount : float, default=0.3
            0 = no change, 1 = white.

        Examples
        --------
        >>> fill = Plotter.lighten('C0', 0.5)
        >>> Plotter.fill_between(ax, x, y1, y2, color=fill)
        """
        return Plotter.adjust_color(color, lighten=amount)

    @staticmethod
    def darken(color, amount: float = 0.3) -> Tuple[float, float, float, float]:
        """
        Return a darkened version of *color* (push lightness toward black).

        Parameters
        ----------
        color : color spec
        amount : float, default=0.3
            0 = no change, 1 = black.

        Examples
        --------
        >>> edge = Plotter.darken('C0', 0.25)
        """
        return Plotter.adjust_color(color, darken=amount)

    @staticmethod
    def desaturate(color, amount: float = 0.5) -> Tuple[float, float, float, float]:
        """
        Return a desaturated (greyed-out) version of *color*.

        Parameters
        ----------
        color : color spec
        amount : float, default=0.5
            0 = original, 1 = fully grey.

        Examples
        --------
        >>> faded = Plotter.desaturate('C1', 0.6)
        """
        return Plotter.adjust_color(color, desaturate=amount)

    @staticmethod
    def with_alpha(color, a: float = 0.5) -> Tuple[float, float, float, float]:
        """
        Return *color* with a modified alpha channel.

        Parameters
        ----------
        color : color spec
        a : float
            New alpha value (0–1).

        Examples
        --------
        >>> Plotter.fill_between(ax, x, y1, y2, color=Plotter.with_alpha('C0', 0.25))
        """
        return Plotter.to_rgba(color, alpha=a)

    @staticmethod
    def blend(
        c1,
        c2,
        t   : float = 0.5,
        *,
        n   : Optional[int] = None,
    ) -> Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]:
        """
        Linearly interpolate between two colors in linear RGB space.

        Parameters
        ----------
        c1, c2 : color spec
            Start and end colors.
        t : float, default=0.5
            Blend position: 0 → ``c1``, 1 → ``c2``. Ignored when ``n`` is set.
        n : int, optional
            If provided, return ``n`` evenly-spaced colors from ``c1`` to ``c2``
            (inclusive of both endpoints).

        Returns
        -------
        tuple or list[tuple]
            Single RGBA tuple when ``n`` is None; list of ``n`` tuples otherwise.

        Examples
        --------
        >>> mid      = Plotter.blend('red', 'blue')
        >>> gradient = Plotter.blend('#E64B35', '#4DBBD5', n=7)
        """
        a = np.array(mcolors.to_rgba(c1))
        b = np.array(mcolors.to_rgba(c2))
        if n is not None:
            ts = np.linspace(0.0, 1.0, int(n))
            return [tuple((1.0 - s) * a + s * b) for s in ts]
        return tuple((1.0 - float(t)) * a + float(t) * b)

    # ------------------------------------------------------------------
    # Sampling colors from a continuous colormap
    # ------------------------------------------------------------------

    @staticmethod
    def n_colors(
        n       : int,
        cmap    : Union[str, mpl.colors.Colormap] = 'viridis',
        vmin    : float = 0.0,
        vmax    : float = 1.0,
        *,
        as_hex  : bool = False,
    ) -> List:
        """
        Sample ``n`` evenly-spaced colors from a colormap.

        Ideal for encoding a continuous parameter (temperature, time, β …) as
        line colors when you want a smooth gradient rather than a categorical
        palette.

        Parameters
        ----------
        n : int
            Number of colors to sample.
        cmap : str or Colormap, default='viridis'
            Source colormap.
        vmin, vmax : float, default 0.0, 1.0
            Fraction range to sample from (allows using only a sub-range of the
            colormap, e.g. ``vmin=0.1, vmax=0.9`` avoids the near-white ends of
            sequential maps).
        as_hex : bool, default=False
            If True, return hex strings instead of RGBA tuples.

        Returns
        -------
        list[tuple] or list[str]

        Examples
        --------
        >>> colors = Plotter.n_colors(5, 'plasma')
        >>> for c, (x, y) in zip(colors, datasets):
        ...     Plotter.plot(ax, x, y, color=c)

        >>> # Avoid extreme ends of the colormap
        >>> colors = Plotter.n_colors(8, 'RdBu_r', vmin=0.1, vmax=0.9)
        """
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        values   = np.linspace(float(vmin), float(vmax), int(n))
        if as_hex:
            return [mcolors.to_hex(cmap_obj(v)) for v in values]
        return [cmap_obj(v) for v in values]

    @staticmethod
    def cmap_colors(
        cmap    : Union[str, mpl.colors.Colormap],
        values  : np.ndarray,
        *,
        vmin    : Optional[float] = None,
        vmax    : Optional[float] = None,
        norm    : Optional[mpl.colors.Normalize] = None,
        scale   : str = 'linear',
    ) -> List[Tuple]:
        """
        Map an array of scalar values to RGBA colors via a colormap.

        Convenience wrapper around :meth:`get_colormap` when you only need
        the list of colors (not the full ``getcolor / norm / mappable`` bundle).

        Parameters
        ----------
        cmap : str or Colormap
        values : array-like
            Scalar values to map.
        vmin, vmax : float, optional
            Color limits. Default to ``min / max(values)``.
        norm : Normalize, optional
            Explicit normalization. Takes precedence over ``scale``.
        scale : {'linear', 'log', 'symlog'}, default='linear'

        Returns
        -------
        list[tuple]
            RGBA tuples, one per value.

        Examples
        --------
        >>> beta_values = np.linspace(0.1, 2.0, 8)
        >>> colors      = Plotter.cmap_colors('plasma', beta_values)
        >>> for val, c in zip(beta_values, colors):
        ...     Plotter.plot(ax, x, data[val], color=c, label=rf'$\\beta={val:.1f}$')
        """
        getcolor, _, _ = Plotter.get_colormap(
            values  = np.asarray(values),
            vmin    = vmin,
            vmax    = vmax,
            cmap    = cmap,
            norm    = norm,
            scale   = scale,
        )
        return [getcolor(v) for v in values]

    ###########################################################
    #! Filter results
    ###########################################################

    @staticmethod
    def filter_results(results, filters=None, get_params_fun: callable = None, *, tol=1e-9):
        """Backward-compatible wrapper around `plotters.data_loader.filter_results`."""
        from .plotters.data_loader import filter_results as _filter_results

        return _filter_results(
            results=results,
            filters=filters,
            get_params_fun=get_params_fun,
            tol=tol,
        )
    
    ###########################################################
    
    @staticmethod
    def get_figsize(columnwidth, wf=0.5, hf=None, aspect_ratio=None):
        r"""
        Parameters:
            - wf [float]:  width fraction in columnwidth units
            - hf [float]:  height fraction in columnwidth units. If None, it will be calculated based on aspect_ratio.
            - aspect_ratio [float]: Aspect ratio (height/width). If None, defaults to golden ratio.
            - columnwidth [float]: width of the column in latex. Get this from LaTeX using \showthe\columnwidth
        Returns:  [fig_width, fig_height]: that should be given to matplotlib
        """
        if aspect_ratio is None:
            aspect_ratio = (5.**0.5 - 1.0) / 2.0  # golden ratio
        if hf is None:
            hf = aspect_ratio

        fig_width_pt    = columnwidth * wf
        inches_per_pt   = 1.0 / 72.27  # Convert pt to inch
        fig_width       = fig_width_pt * inches_per_pt  # width in inches
        fig_height      = fig_width * hf  # height in inches
        return [fig_width, fig_height]
    
    ###########################################################
    
    @staticmethod 
    def get_color(color,
                alpha     = None,
                edgecolor = (0,0,0,1), 
                facecolor = (1,1,1,0)):
        '''
        Get a dictionary with color properties for matplotlib patches.
        Parameters:
            - color [str or tuple]: Color to use, can be a named color or an RGB tuple.
            - alpha [float]: Transparency level (0 to 1).
            - edgecolor [tuple]: Edge color as an RGB tuple.
            - facecolor [tuple]: Face color as an RGB tuple.
        Returns:
            - dictionary [dict]: Dictionary with color properties.
        '''
        dictionary = dict(facecolor = facecolor, edgecolor = edgecolor, color=color)
        if alpha is not None:
            dictionary['alpha'] = alpha
        return dictionary
    
    ####################### C O L O R S #######################

    @staticmethod
    def add_colorbar(fig                : mpl.figure.Figure,
                    pos                 : List[float],
                    mappable            : Union[np.ndarray, list, mpl.cm.ScalarMappable],
                    cmap                : Union[str, mpl.colors.Colormap] = 'viridis',
                    norm                : Optional[mpl.colors.Normalize] = None,
                    vmin                : Optional[float] = None,
                    vmax                : Optional[float] = None,
                    scale               : str = 'linear',
                    orientation         : str = 'vertical',
                    label               : str = '',
                    label_kwargs        : dict = None,
                    title               : str = '',
                    title_kwargs        : dict = None,
                    ticks               : Optional[Union[List, np.ndarray]] = None,
                    ticklabels          : Optional[List[str]] = None,
                    tick_location       : str = 'auto', 
                    tick_params         : dict = None,
                    extend              : str = None,
                    format              : Optional[Union[str, mpl.ticker.Formatter]] = None,
                    discrete            : Union[bool, int] = False,
                    boundaries          : List[float] = None,
                    invert              : bool = False,
                    remove_pdf_lines    : bool = True,
                    **kwargs) -> Tuple[mpl.colorbar.Colorbar, mpl.axes.Axes]:
        """
        Add a fully customizable colorbar to the figure at a specific position.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Parent figure onto which the colorbar axis is added.
        pos : list[float] | tuple[float, float, float, float]
            [left, bottom, width, height] in figure coordinates (0..1).
        mappable : array-like | matplotlib.cm.ScalarMappable
            - If array-like: a new ScalarMappable is built from `cmap`/`norm` (and `scale`, `vmin`, `vmax`).
            - If ScalarMappable: it is used directly. `vmin`/`vmax` update its clim; `norm` is taken from it
            when not provided. Note: in this case `discrete`/`boundaries` resampling is not applied.
        cmap : str | Colormap, default='viridis'
            Colormap name or object. If `mappable` is a ScalarMappable, its cmap is used unless `cmap`
            is explicitly different from the default and a new mappable is constructed (array-like path).
        norm : matplotlib.colors.Normalize, optional
            Normalization to map data to 0-1. Ignored if `mappable` is ScalarMappable and `norm` is None
            (then the mappable's norm is used).
        vmin, vmax : float, optional
            Data limits. When `scale='log'`, non-positive `vmin` is clamped internally.
        scale : {'linear', 'log', 'symlog'}, default='linear'
            Creates a suitable Normalize when `mappable` is array-like and `norm` is None.
            - 'linear'  -> Normalize
            - 'log'     -> LogNorm (vmin<=0 clamped to ~1e-10)
            - 'symlog'  -> SymLogNorm with linthresh=0.1
        orientation : {'vertical', 'horizontal'}, default='vertical'
            Colorbar orientation.
        label : str, default=''
            Axis label along the long side of the colorbar.
        label_kwargs : dict, optional
            Passed to ColorbarBase.set_label (e.g., dict(fontsize=..., labelpad=...)).
        title : str, default=''
            Title text set at the end/top of the colorbar. For horizontal bars, the title is placed to the side.
        title_kwargs : dict, optional
            Text properties for the title (e.g., dict(fontsize=..., pad=...)).
        ticks : list[float] | np.ndarray, optional
            Explicit major tick locations.
        ticklabels : list[str], optional
            Custom labels for the ticks (same length as `ticks`).
        tick_location : {'auto','left','right','top','bottom'}, default='auto'
            Side on which to draw ticks/labels (respects `orientation`).
        tick_params : dict, optional
            Passed to cbar.ax.tick_params (e.g., dict(length=4, width=1, direction='in')).
        extend : {'neither','both','min','max','neutral'}, default='neutral'
            Colorbar extension behavior. Standard Matplotlib values are 'neither', 'both', 'min', 'max'.
            'neutral' is treated as a pass-through here and may behave like 'neither' depending on Matplotlib.
        format : str | matplotlib.ticker.Formatter, optional
            Tick formatting. If str (e.g., '%.2e'), uses FormatStrFormatter.
        discrete : bool | int, default=False
            Discretize colormap when building from array-like:
            - True  -> 10 bins
            - int N -> N bins
            Ignored when `mappable` is a ScalarMappable.
        boundaries : list[float], optional
            Discrete bin edges. Enables BoundaryNorm and passes `boundaries` to fig.colorbar
            (default spacing='proportional', overridable via kwargs['spacing']).
        invert : bool, default=False
            If True, invert the colorbar axis direction.
        remove_pdf_lines : bool, default=True
            Set solids edgecolor to 'face' to avoid white hairlines in vector exports (PDF/SVG).
        **kwargs :
            Additional arguments forwarded to `fig.colorbar`, e.g.:
            - alpha, spacing ('uniform'|'proportional'), fraction, pad, shrink, aspect, drawedges, etc.

        Returns
        -------
        (cbar, cax) : tuple[matplotlib.colorbar.Colorbar, matplotlib.axes.Axes]
            The created colorbar and its axes.

        Notes
        -----
        - When `mappable` is a ScalarMappable, this helper does not modify its colormap discretization.
            To use `discrete`/`boundaries`, pass raw data (array-like) instead.
        - For 'log' scale, ensure your data are strictly positive (this function clamps vmin if needed).

        Examples
        --------
        # Vertical, linear scale from raw data
        cbar, cax = Plotter.add_colorbar(fig, [0.92, 0.15, 0.02, 0.7], data, label='Mz')

        # Horizontal, log scale with sci formatting and extensions
        cbar, cax = Plotter.add_colorbar(
            fig, [0.2, 0.9, 0.6, 0.03], data,
            scale='log', orientation='horizontal',
            format='%.0e', extend='both',
            tick_location='top', label='Conductance'
        )

        # Discrete categorical-like bar with custom tick labels
        cbar, cax = Plotter.add_colorbar(
            fig, [0.85, 0.1, 0.03, 0.8], [0, 1, 2],
            cmap='Set1', discrete=3,
            ticklabels=['Insulator', 'Metal', 'SC']
        )

        # Non-uniform boundaries
        cbar, cax = Plotter.add_colorbar(
            fig, [0.86, 0.15, 0.02, 0.7], data,
            boundaries=[0, 0.5, 2.0, 10.0], spacing='proportional'
        )
        """
        import matplotlib.colors as mcolors
        import matplotlib.ticker as mticker
        
        # 1. Handle Normalization and Mappable
        if isinstance(mappable, mpl.cm.ScalarMappable):
            sm = mappable
            if vmin is not None or vmax is not None:
                sm.set_clim(vmin, vmax)
            # If a mappable is passed, we might need to extract the cmap/norm for modifications
            if norm is None: norm = sm.norm
            if cmap == 'viridis': cmap = sm.cmap # Only override default if specific one not passed
        else:
            data    = np.asarray(mappable)
            _vmin   = vmin if vmin is not None else np.nanmin(data)
            _vmax   = vmax if vmax is not None else np.nanmax(data)

            # Discrete Boundaries (e.g., for Phase Diagrams)
            if boundaries is not None:
                cmap_obj    = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
                norm        = mcolors.BoundaryNorm(boundaries, cmap_obj.N, clip=True)
            
            # Standard Scales
            elif norm is None:
                if scale == 'log':
                    if _vmin <= 0: _vmin = 1e-10
                    norm = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
                elif scale == 'symlog':
                    norm = mcolors.SymLogNorm(linthresh=0.1, vmin=_vmin, vmax=_vmax)
                else:
                    norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

            # Discretize Colormap (e.g. 10 distinct colors)
            if discrete:
                cmap_obj    = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
                n_bins      = discrete if isinstance(discrete, int) and discrete > 1 else 10
                cmap        = cmap_obj.resampled(n_bins)

            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

        # Formatting
        if isinstance(format, str):
            format = mticker.FormatStrFormatter(format)

        # Create Axes and Colorbar
        # Pass boundaries to colorbar if they exist (ensures spacing is correct)
        cax         = fig.add_axes(pos)
        cbar_kwargs = kwargs.copy()
        if boundaries is not None:
            cbar_kwargs['boundaries']   = boundaries
            cbar_kwargs['spacing']      = kwargs.get('spacing', 'proportional')
            
        cbar        = fig.colorbar(sm, cax=cax, orientation=orientation, extend=extend, format=format, **cbar_kwargs)

        # Labels and Titles
        if label:
            l_kwargs = label_kwargs or {}
            cbar.set_label(label, **l_kwargs)
        
        if title:
            # Smart default padding for title
            t_kwargs    = title_kwargs or {}
            pad         = t_kwargs.pop('pad', 10)
            if orientation == 'vertical':
                cbar.ax.set_title(title, pad=pad, **t_kwargs)
            else:
                # For horizontal, title usually makes sense as a side label or top label
                cbar.ax.text(1.05, 0.5, title, transform=cbar.ax.transAxes, va='center', ha='left', **t_kwargs)

        # Tick Customization
        if ticks is not None:
            cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
        
        # Tick Location (Top/Bottom/Left/Right)
        if tick_location != 'auto':
            if orientation == 'horizontal':
                cax.xaxis.set_ticks_position(tick_location)
                cax.xaxis.set_label_position(tick_location)
            else:
                cax.yaxis.set_ticks_position(tick_location)
                cax.yaxis.set_label_position(tick_location)

        if tick_params:
            cbar.ax.tick_params(**tick_params)

        # Utilities
        if invert:
            cbar.ax.invert_axis()
            
        if scale == 'log' and ticks is None and boundaries is None:
            cbar.ax.minorticks_on()

        # PDF Export Fix (Removes white lines between colors)
        if remove_pdf_lines:
            cbar.solids.set_edgecolor("face")

        return cbar, cax
    
    ###########################################################
    
    @staticmethod
    def get_colormap(values: Optional[np.ndarray] = None, vmin=None, vmax=None, *,
            cmap='PuBu', elsecolor='blue', get_mappable: bool = False, return_mappable: Optional[bool] = None,
            norm=None, scale='linear', **kwargs):
        """
        Get a colormap for the given values.
        
        Parameters:
        - values (array-like): The values to map to colors.
        - cmap (str, optional): The colormap to use. Defaults to 'PuBu'.
        - elsecolor (str, optional): The color to use if there is only one value. Defaults to 'blue'.
        - get_mappable (bool, optional): If True, also return a ScalarMappable as
          the 4th item, ready to pass into `Plotter.add_colorbar(..., mappable=...)`.
        - return_mappable (bool, optional): Alias for `get_mappable`.
        
        Returns:
        - getcolor (function): A function that maps a value to a color.
        - colors (Colormap): The colormap object.
        - norm (Normalize): The normalization object.
        - mappable (ScalarMappable, optional): Returned when `get_mappable=True`
          (or `return_mappable=True`).
        
        Example:
        >>> getcolor, colors, norm = Plotter.get_colormap([1, 2, 3], cmap='viridis')
        >>> color = getcolor(2.5)
        >>> getcolor, colors, norm, mappable = Plotter.get_colormap(
        ...     [1, 2, 3], cmap='viridis', return_mappable=True
        ... )
        """
        from matplotlib.colors import Normalize, LogNorm, SymLogNorm

        if return_mappable is not None:
            get_mappable = bool(return_mappable)
        
        # Resolve vmin/vmax
        if values is None and (vmin is None or vmax is None):
            raise ValueError("Either 'values' or both 'vmin' and 'vmax' must be provided.")
        
        if vmin is None: vmin = np.nanmin(values)
        if vmax is None: vmax = np.nanmax(values)

        # Create Norm (if not provided externally)
        if norm is None:
            if scale == 'log':
                if vmin <= 0: 
                    vmin = 1e-10 
                norm = LogNorm(vmin=vmin, vmax=vmax)
            elif scale == 'symlog':
                norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
        
        colors = plt.get_cmap(cmap)
        
        # Create getcolor function
        # Use norm(x) instead of manual math to respect Log vs Linear
        if vmin == vmax:
            getcolor = lambda x: elsecolor
        else:
            getcolor = lambda x: colors(norm(x))

        # Create Mappable for downstream colorbar reuse.
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=colors)
        if values is not None:
            mappable.set_array(np.asarray(values))
        else:
            mappable.set_array(np.asarray([vmin, vmax], dtype=float))
        
        if get_mappable:
            return getcolor, colors, norm, mappable
        return getcolor, colors, norm

    @staticmethod
    def apply_colormap(ax, data, cmap='PuBu', colorbar=True, **kwargs):
        """
        Apply a colormap to the given data and plot it on the provided axis.
        
        Parameters:
        - ax (object): The axis object to plot on.
        - data (array-like): The data to plot.
        - cmap (str, optional): The colormap to use. Defaults to 'PuBu'.
        - colorbar (bool, optional): Whether to add a colorbar. Defaults to True.
        
        Returns:
        - img (AxesImage): The image object.
        """
        norm    = Normalize(np.min(data), np.max(data))
        img     = ax.imshow(data, cmap=cmap, norm=norm, **kwargs)
        if colorbar:
            plt.colorbar(img, ax=ax)
        return img

    @staticmethod
    def discrete_colormap(N, base_cmap=None):
        """
        Create an N-bin discrete colormap from the specified input map.
        
        Parameters:
        - N (int): Number of discrete colors.
        - base_cmap (str or Colormap, optional): The base colormap to use. Defaults to None.
        
        Returns:
        - cmap (Colormap): The discrete colormap.
        """
        base        = plt.cm.get_cmap(base_cmap)
        color_list  = base(np.linspace(0, 1, N))
        cmap_name   = base.name + str(N)
        return ListedColormap(color_list, name=cmap_name)
    
    ##################################################
    
    @staticmethod
    def set_annotate(ax, elem: str, x: float = 0, y: float = 0, fontsize=None, xycoords='axes fraction', cond=True, zorder=50, boxaround=True, **kwargs):
        '''
        Make an annotation on the plot.
        - ax        : axis to annotate on
        - elem      : annotation string
        - x         : x coordinate (ignored if xycoords='best')
        - y         : y coordinate (ignored if xycoords='best')
        - fontsize  : fontsize of the annotation
        - xycoords  : how to interpret the coordinates (from MPL), or 'best' to find the best corner
        - cond      : condition to make the annotation
        '''
        if not cond:
            return

        # Take the default fontsize if not specified
        if fontsize is None:
            fontsize = plt.rcParams['font.size']

        if xycoords == 'best':
            offset_x = 0.05  # Horizontal offset to prevent overlap with axes
            offset_y = 0.05  # Vertical offset to prevent overlap with axes

            corners = {
                "upper left": (0.05 + offset_x, 0.95 - offset_y),
                "upper right": (0.95 - offset_x, 0.95 - offset_y),
                "lower left": (0.05 + offset_x, 0.05 + offset_y),
                "lower right": (0.95 - offset_x, 0.05 + offset_y)
            }

            fig = ax.figure
            renderer = fig.canvas.get_renderer()

            # Get the data bounding box
            data_bbox = ax.transAxes.transform_bbox(ax.get_position())  # Use axis fraction for bounds

            # Track text extents for all corners
            text_extents = {}
            adjusted_corners = {}
            for corner, (cx, cy) in corners.items():
                text = ax.annotate(
                    elem, xy=(cx, cy), fontsize=fontsize,
                    xycoords='axes fraction', alpha=0, zorder=zorder, **kwargs
                )
                bbox = text.get_window_extent(renderer=renderer)
                text_extents[corner] = bbox

                # Adjust corner coordinates to ensure text stays within the plot
                x_adjust = min(max(0, cx - bbox.width / fig.get_size_inches()[0] + offset_x), 0.95)
                y_adjust = min(max(0, cy - bbox.height / fig.get_size_inches()[1] + offset_y), 0.95)
                adjusted_corners[corner] = (x_adjust + offset_x, y_adjust + offset_y)
                text.remove()

            # Find the best corner without overlap
            best_corner = None
            for corner, bbox in text_extents.items():
                if not data_bbox.overlaps(bbox):  # Check for overlap
                    best_corner = corner
                    break

            # If no corner avoids overlap, default to upper left and add a white background
            if best_corner is None:
                best_corner = "upper left"
            kwargs['bbox'] = dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.85)

            # Now we check for inset axes and avoid placement below them
            insets = [inset for inset in ax.figure.get_axes() if inset != ax]
            for inset_ax in insets:
                if inset_ax != ax:
                    inset_bbox = inset_ax.get_position()

                    # Adjust placement based on inset location (avoid placement below the inset)
                    if best_corner == "upper left" and inset_bbox.y0 > 0.5:
                        best_corner = "upper right"  # Change position to upper right
                    elif best_corner == "lower left" and inset_bbox.y1 < 0.5:
                        best_corner = "lower right"  # Change position to lower right


            # Annotate in the selected corner
            cx, cy = adjusted_corners[best_corner]
            ax.annotate(
                elem, xy=(cx, cy), fontsize=fontsize,
                xycoords='axes fraction', zorder=zorder, **kwargs)
            return
        
        if boxaround:
            kwargs['bbox'] = dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.85)
        ax.annotate(elem, xy=(x, y), fontsize=fontsize, xycoords=xycoords, zorder=zorder, **kwargs)
    
    @staticmethod
    def set_annotate_letter(
        ax          : plt.Axes,
        iter        : int,
        x           : float = 0,
        y           : float = 0,
        fontsize    = 12,
        xycoords    = 'axes fraction',
        addit       = '',
        condition   = True,
        zorder      = 50,
        boxaround   = False,
        fontweight  = 'normal',
        color       = 'black',
        **kwargs        
        ):
        """
        Annotate plot with the letter.
        
        Params:
        -----------
        ax: matplotlib.axes.Axes
            axis to annotate on
        iter: 
            iteration number
        x: 
            x coordinate
        y: 
            y coordinate
        fontsize: 
            fontsize 
        xycoords: 
            how to interpret the coordinates (from MPL)
        addit: 
            additional string to add after the letter
        condition: 
            condition to make the annotation
        zorder: 
            zorder of the annotation
        boxaround: 
            whether to put a box around the annotation
        fontweight: 
            weight of the text ('bold', 'normal', etc.)
        kwargs: 
            additional arguments for annotation
            - color : color of the text
            - weight: weight of the text
            
        Example:
        --------
        >>> Plotter.set_annotate_letter(ax, 0, x=0.1, y=0.9, fontsize=14, addit=' Test', color='red')
        """
        ax = Plotter.ax(ax)
        Plotter.set_annotate(ax, elem = f'({chr(97 + iter)})' + addit, x = x, y = y, color = color, fontweight = fontweight,
                fontsize = fontsize, cond = condition, xycoords = xycoords, zorder = zorder, boxaround = boxaround, **kwargs)
    
    @staticmethod
    def set_arrow(  ax,
                    start_T     : str,
                    end_T       : str,
                    xystart     : float,
                    xystart_T   : float,
                    xyend       : float,
                    xyend_T     : float,
                    arrowprops  = dict(arrowstyle="->"), 
                    startcolor  = 'black',
                    endcolor    = 'black',
                    **kwargs):
        '''
        @staticmethod
        
        Make an annotation on the plot.
        - ax        :   axis to annotate on
        - start_T   :   start text
        - end_T     :   end text
        - xystart   :   x coordinate start
        - xystart_T :   x coordinate start text
        - xyend     :   x coordinate end
        - xyend_T   :   x coordinate end text
        - arrowprops:   properties of the arrow
        - startcolor:   color of the arrow at the start
        - endcolor  :   color of the arrow in the end
        - kwargs    :   additional arguments for annotation
        '''
        ax.annotate(start_T,
                xy           =   xystart, 
                xytext       =   xystart_T, 
                arrowprops   =   arrowprops, 
                color        =   startcolor)
        
        ax.annotate(end_T,
                    xy       =   xyend, 
                    xytext   =   xyend_T, 
                    color    =   endcolor)

    @staticmethod
    def callout(ax, text: str, xy, xytext=None, *, xycoords: str = 'data', textcoords: Optional[str] = None, arrowstyle: str = '->',
        color: str = 'black', lw: float = 1.0, boxaround: bool = True, box_alpha: float = 0.85, zorder: int = 20, **kwargs,
    ):
        """Add a compact callout (text + optional arrow) to an axis."""
        ax = Plotter.ax(ax)
        if xytext is None:
            xytext = xy
        if textcoords is None:
            textcoords = xycoords

        bbox = None
        if boxaround:
            bbox = dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.25', alpha=box_alpha)

        arrowprops = None
        if xytext != xy:
            arrowprops = dict(arrowstyle=arrowstyle, color=color, lw=lw)

        return ax.annotate(text, xy=xy, xytext=xytext, xycoords=xycoords, textcoords=textcoords,
            color=color, arrowprops=arrowprops, bbox=bbox, zorder=zorder, **kwargs,
        )

    @staticmethod
    def highlight_box(
        ax, x: float, y: float, width: float, height: float,
        *,
        coords: str = 'data', edgecolor='crimson', facecolor='none', lw: float = 1.3, ls='-', alpha: float = 0.95, zorder: int = 15,
        **kwargs,
    ):
        """Draw a highlighted rectangular region in data or axes coordinates."""
        ax          = Plotter.ax(ax)
        transform   = ax.transAxes if str(coords).lower().startswith('axes') else ax.transData
        rect = Rectangle(
            (x, y),
            width,
            height,
            transform=transform,
            edgecolor=edgecolor,
            facecolor=facecolor,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            **kwargs,
        )
        ax.add_patch(rect)
        return rect

    @staticmethod
    def highlight_circle(
        ax,
        x: float,
        y: float,
        radius: float,
        *,
        coords: str = 'data',
        edgecolor='darkorange',
        facecolor='none',
        lw: float = 1.3,
        ls='-',
        alpha: float = 0.95,
        zorder: int = 15,
        **kwargs,
    ):
        """Draw a highlighted circular region in data or axes coordinates."""
        ax = Plotter.ax(ax)
        transform = ax.transAxes if str(coords).lower().startswith('axes') else ax.transData
        circ = Circle(
            (x, y),
            radius,
            transform=transform,
            edgecolor=edgecolor,
            facecolor=facecolor,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            **kwargs,
        )
        ax.add_patch(circ)
        return circ

    ##################### F I T S #####################
    
    @staticmethod
    def plot_fit(   ax,   
                    funct,
                    x,
                    **kwargs):
        """ 
        @staticmethod
        
        Plots the fitting function provided by the user on a given axis using 
        the **kwargs provider afterwards.
        - ax        :   axis to annotate on
        - funct     :   function to use for the fitting
        - x         :   arguments to the function
        """
        y = funct(x)
        ax.plot(x, y, **kwargs)
    
    #################### L I N E S ####################
    
    @staticmethod
    def hline(  ax      : mpl.axes.Axes,
                val     : float,
                ls      = '--',
                lw      = 2.0,
                color   = 'black',
                label   = None,
                zorder  = 10,
                label_cond = True,
                **kwargs):
        '''
        horizontal line plotting
        '''
        
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]

        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
        
        if label is None or label == '':
            label_cond = False

        ax.axhline(val, ls = ls,  lw = lw,
                label = label if (label is not None and len(label) != 0 and label_cond) else None,
                color = color,
                zorder = zorder,
                **kwargs)
    
    @staticmethod
    def vline(  ax, 
                val         : float,
                ls          = '--',
                lw          = 2.0,
                color       = 'black',
                label       = None,
                zorder      = 10,
                label_cond   = True,
                **kwargs):
        '''
        vertical line plotting
        '''
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]

        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]

        if label is None or label == '':
            label_cond = False

        ax.axvline(val, 
                ls      = ls,  
                lw      = lw, 
                label   = label if (label is not None and len(label) != 0 and label_cond) else None, 
                color   = color,
                zorder  = zorder,
                **kwargs)
    
    ################## S C A T T E R ##################
    
    @staticmethod
    def scatter(ax, x, y, *,
        s           =   10,
        c           =   'blue',
        marker      =   'o',
        alpha       =   1.0,
        label       =   None,
        edgecolor   =   None,
        zorder      =   5,
        label_cond   =   True,
        linewidths  =   1.0,
        cmap        =   None,
        norm        =   None,
        vmin        =   None,
        vmax        =   None,
        plotnonfinite = False,
        clip_on     =   True,
        rasterized  =   False,
        **kwargs):
        """
        Creates a scatter plot on the provided axis, styled for Nature-like plots.

        Parameters:
            ax (matplotlib.axes.Axes): The axis on which to draw the scatter plot.
            x (array-like): The x-coordinates of the points.
            y (array-like): The y-coordinates of the points.
            s (float or array-like, optional): The size of the points (default: 10).
            c (color or array-like, optional): The color of the points (default: 'blue').
            marker (str, optional): The shape of the points (default: 'o').
            alpha (float, optional): The transparency of the points (0.0 to 1.0, default: 1.0).
            label (str, optional): The label for the points (default: None).
            edgecolor (str or array-like, optional): The edge color of the points (default: 'white').
            zorder (int, optional): The drawing order of the points (default: 5).
            **kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.scatter`.

        Example:
            scatter(ax, x_data, y_data, s=20, c='red', alpha=0.5, label='Sample Data')
        """
        if isinstance(x, (float, int)):
            x = [x]
        if isinstance(y, (float, int)):
            y = [y]
        
        # override the color if it is provided in the kwargs
        if 'color' in kwargs:
            c = None
        
        if 'edgecolors' in kwargs:
            edgecolor = None
            
        if isinstance(c, int):
            c = colorsList[c % len(colorsList)]

        if isinstance(marker, int):
            marker = markersList[marker % len(markersList)]

        if label is None or label == '':
            label_cond = False    
        
        ax.scatter(
            x, y, linewidths=linewidths,
            s=s, c=c, marker=marker,
            alpha=alpha, label=label if label_cond else '', edgecolor=edgecolor, zorder=zorder,
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, plotnonfinite=plotnonfinite,
            clip_on=clip_on, rasterized=rasterized, **kwargs
        )

    @staticmethod
    def tripcolor_field(
        ax,
        points,
        values,
        *,
        triangles=None,
        mask=None,
        shading: str = "gouraud",
        **kwargs,
    ):
        """
        Plot a scalar field sampled on irregular planar points using triangulation.

        This helper is meant for 2D scattered data where `imshow` is not
        appropriate because the samples do not lie on a regular rectangular grid.
        Matplotlib first builds a triangulation of the point cloud and then
        interpolates values inside each triangle.

        Typical use cases:
        - real-space lattice-site scalar fields
        - Brillouin-zone data on irregular planar k-point sets
        - any scattered 2D measurement data

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axis.
        points : array-like
            Sample positions shaped like `(N, 2)` or `(N, D)` with `D >= 2`.
            Only the first two Cartesian components are used.
        values : array-like
            Scalar values of length `N`.
        triangles : array-like, optional
            Explicit connectivity passed to `matplotlib.tri.Triangulation`.
        mask : array-like of bool, optional
            Triangle mask. `True` hides the corresponding triangle.
        shading : {'flat', 'gouraud'}, default='gouraud'
            Interpolation mode inside triangles.
        **kwargs
            Forwarded to `Axes.tripcolor`.

        Returns
        -------
        matplotlib.collections.Collection | None
            The created artist, or `None` if fewer than three points are given.
        """
        pts = np.asarray(points, dtype=float)
        vals = np.asarray(values)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("points must be shaped like (N, 2) or (N, D) with D >= 2.")
        if len(pts) != len(vals):
            raise ValueError("points and values must have the same length.")
        if len(pts) < 3:
            return None

        tri = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles=triangles)
        if mask is not None:
            tri.set_mask(np.asarray(mask, dtype=bool))
        return ax.tripcolor(tri, vals, shading=shading, **kwargs)
    
    #################### P L O T S ####################
    
    @staticmethod
    def plot(  ax, *args,
                y               = None,
                x               = None,
                ls              = '-',
                lw              = 2.0,
                color           = 'black',
                # label 
                label           = None,
                label_cond      = True,
                # marker
                marker          = None,
                ms              = None,
                # other
                zorder          = 5,
                drawstyle       = 'default',
                markevery       = None,
                clip_on         = True,
                rasterized      = False,
                antialiased     = True,
                solid_capstyle  = None,
                solid_joinstyle = None,
                **kwargs):
        '''
        plot the data
        '''
        ax  = Plotter.ax(ax)
        if len(args) == 1:
            y = args[0]
        elif len(args) >= 2:
            x = args[0]
            y = args[1]
        
        if 'linestyle' in kwargs:
            ls = linestyleNorm(kwargs['linestyle'])
            kwargs.pop('linestyle')
            
        if 'linewidth' in kwargs:
            lw = kwargs['linewidth']
            kwargs.pop('linewidth')

        if 'markersize' in kwargs:
            ms = kwargs['markersize']
            kwargs.pop('markersize')
            
        if 'c' in kwargs:
            color = kwargs['c']
            kwargs.pop('c')
        
        if label is None or label == '':
            label_cond = False
        
        # use the defaults
        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
            
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]
        
        if isinstance(marker, int):
            marker = markersList[marker % len(markersList)]
        marker = markerNorm(marker)
        
        line_style_kwargs = {}
        if solid_capstyle is not None:
            line_style_kwargs["solid_capstyle"] = solid_capstyle
        if solid_joinstyle is not None:
            line_style_kwargs["solid_joinstyle"] = solid_joinstyle

        if x is None and y is None:
            x = [None]
            y = [None]
        elif x is None:
            x = np.arange(len(y))
        elif y is None:
            raise ValueError("y cannot be None if x is provided.")

        ax.plot(
            x, y,
            ls=ls,
            lw=lw,
            color=color,
            label=label if label_cond else '',
            zorder=zorder,
            marker=marker,
            ms=ms,
            drawstyle=drawstyle,
            markevery=markevery,
            clip_on=clip_on,
            rasterized=rasterized,
            antialiased=antialiased,
            **line_style_kwargs,
            **kwargs
        )

    @staticmethod
    def fill_between(
        ax,
        x,
        y1,
        y2,
        color       =   'blue',
        alpha       =   0.5,
        where       =   None,
        interpolate =   False,
        step        =   None,
        linewidth   =   0.0,
        edgecolor   =   None,
        zorder      =   4,
        clip_on     =   True,
        rasterized  =   False,
        **kwargs):
        """
        Fills the area between two curves on the provided axis.

        Parameters:
            ax (matplotlib.axes.Axes): The axis on which to fill the area.
            x (array-like): The x-coordinates of the points.
            y1 (array-like): The y-coordinates of the first curve.
            y2 (array-like): The y-coordinates of the second curve.
            color (str, optional): The color of the filled area (default: 'blue').
            alpha (float, optional): The transparency of the filled area (0.0 to 1.0, default: 0.5).
            **kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.fill_between`.

        Example:
            fill_between(ax, x_data, y1_data, y2_data, color='red', alpha=0.3)
        """
        ax = Plotter.ax(ax)
        ax.fill_between(
            x, y1, y2, color=color, alpha=alpha,
            where=where, interpolate=interpolate, step=step,
            linewidth=linewidth, edgecolor=edgecolor,
            zorder=zorder, clip_on=clip_on, rasterized=rasterized,
            **kwargs
        )
    
    # ################ LOG SCALE PLOTS ################
    
    @staticmethod
    def semilogy(ax, x, y, ls='-', lw=1.5, color='black', label=None, marker=None, ms=None, label_cond=True, zorder=5, **kwargs):
        """
        Plot with logarithmic y-axis.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        x, y : array-like
            Data to plot.
        ls : str, default='-'
            Line style.
        lw : float, default=1.5
            Line width.
        color : str or int, default='black'
            Line color. If int, uses colorsList[color].
        label : str, optional
            Legend label.
        marker : str, optional
            Marker style.
        ms : float, optional
            Marker size.
        **kwargs
            Additional arguments passed to ax.semilogy.
        
        Examples
        --------
        >>> Plotter.semilogy(ax, x, np.exp(-x), color='C0', label=r'$e^{-x}$')
        """
        ax      = Plotter.ax(ax)
        color   = color     or kwargs.pop('c', 'black')
        ls      = ls        or kwargs.pop('linestyle', '-')
        lw      = lw        or kwargs.pop('linewidth', 1.5)
        ms      = ms        or kwargs.pop('markersize', None)
        marker  = marker    or kwargs.pop('marker', None)
        
        if isinstance(color, int):
            color   = colorsList[color % len(colorsList)]
        if isinstance(ls, int):
            ls      = linestylesList[ls % len(linestylesList)]
        if isinstance(marker, int):
            marker  = markersList[marker % len(markersList)]
        if label is None or label == '':
            label_cond = False
            
        ax.semilogy(x, y, ls=ls, lw=lw, color=color, label=label if label_cond else '', marker=marker, ms=ms, zorder=zorder, **kwargs)
    
    @staticmethod
    def semilogx(ax, x, y, ls='-', lw=1.5, color='black', label=None, marker=None, ms=None, label_cond=True, zorder=5, **kwargs):
        """
        Plot with logarithmic x-axis.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        x, y : array-like
            Data to plot.
        ls : str, default='-'
            Line style.
        lw : float, default=1.5
            Line width.
        color : str or int, default='black'
            Line color. If int, uses colorsList[color].
        label : str, optional
            Legend label.
        marker : str, optional
            Marker style.
        ms : float, optional
            Marker size.
        **kwargs
            Additional arguments passed to ax.semilogx.
        
        Examples
        --------
        >>> Plotter.semilogx(ax, np.logspace(-3, 3, 100), y, color='C1')
        """
        ax = Plotter.ax(ax)
        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]
        if isinstance(marker, int):
            marker = markersList[marker % len(markersList)]
        if label is None or label == '':
            label_cond = False
            
        ax.semilogx(x, y, ls=ls, lw=lw, color=color, label=label if label_cond else '', marker=marker, ms=ms, zorder=zorder, **kwargs)
    
    @staticmethod
    def loglog(ax, x, y, ls='-', lw=1.5, color='black', label=None, marker=None, ms=None, label_cond=True, zorder=5, **kwargs):
        """
        Plot with logarithmic x and y axes.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        x, y : array-like
            Data to plot (must be positive).
        ls : str, default='-'
            Line style.
        lw : float, default=1.5
            Line width.
        color : str or int, default='black'
            Line color. If int, uses colorsList[color].
        label : str, optional
            Legend label.
        marker : str, optional
            Marker style.
        ms : float, optional
            Marker size.
        **kwargs
            Additional arguments passed to ax.loglog.
        
        Examples
        --------
        >>> # Power law: y = x^(-2)
        >>> x = np.logspace(0, 3, 50)
        >>> Plotter.loglog(ax, x, x**(-2), label=r'$x^{-2}$', color='C2')
        """
        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]
        if isinstance(marker, int):
            marker = markersList[marker % len(markersList)]
        if label is None or label == '':
            label_cond = False
            
        ax.loglog(x, y, ls=ls, lw=lw, color=color, label=label if label_cond else '', marker=marker, ms=ms, zorder=zorder, **kwargs)
    
    # -------------------- ERROR BARS --------------------
    
    @staticmethod
    def errorbar(ax, x, y, yerr=None, xerr=None, fmt='o', color='black', capsize=2, capthick=1.0, elinewidth=1.0, markersize=5, label=None, label_cond=True, alpha=1.0, zorder=5,
                 ecolor=None, errorevery=1, barsabove=False, uplims=False, lolims=False, xuplims=False, xlolims=False,
                 clip_on=True, rasterized=False, **kwargs):
        """
        Plot data with error bars.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        x, y : array-like
            Data points.
        yerr : float or array-like, optional
            Vertical error bars. Can be:
            - scalar: symmetric error for all points
            - 1D array: symmetric errors
            - 2D array (2, N): asymmetric [lower, upper] errors
        xerr : float or array-like, optional
            Horizontal error bars (same format as yerr).
        fmt : str, default='o'
            Format string for markers ('' for no markers, just error bars).
        color : str or int, default='black'
            Color for markers and error bars.
        capsize : float, default=2
            Length of error bar caps.
        capthick : float, default=1.0
            Thickness of error bar caps.
        elinewidth : float, default=1.0
            Width of error bar lines.
        markersize : float, default=5
            Size of markers.
        label : str, optional
            Legend label.
        alpha : float, default=1.0
            Transparency.
        **kwargs
            Additional arguments passed to ax.errorbar.
        
        Examples
        --------
        >>> # Symmetric error
        >>> Plotter.errorbar(ax, x, y, yerr=sigma, label='Data')
        
        >>> # Asymmetric error
        >>> Plotter.errorbar(ax, x, y, yerr=[lower_err, upper_err])
        
        >>> # Error band without markers
        >>> Plotter.errorbar(ax, x, y, yerr=sigma, fmt='', elinewidth=2)
        """
        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
        if label is None or label == '':
            label_cond = False
        
        ax.errorbar(x, y, yerr=yerr, xerr=xerr,
                    fmt=fmt, color=color,
                    capsize=capsize, capthick=capthick,
                    elinewidth=elinewidth, markersize=markersize,
                    label=label if label_cond else '',
                    alpha=alpha, zorder=zorder,
                    ecolor=ecolor, errorevery=errorevery, barsabove=barsabove,
                    uplims=uplims, lolims=lolims, xuplims=xuplims, xlolims=xlolims,
                    clip_on=clip_on, rasterized=rasterized, **kwargs)
    
    # -------------------- HISTOGRAM --------------------
    
    @staticmethod
    def histogram(ax, data, bins=50, density=True, histtype='stepfilled', alpha=0.7, color='C0', edgecolor='black',
                linewidth=1.0, label=None, orientation='vertical', cumulative=False, log=False, label_cond=True, zorder=5,
                weights=None, range=None, align='mid', rwidth=None, stacked=False, hatch=None,
                **kwargs):
        """
        Plot a histogram.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        data : array-like
            Input data.
        bins : int or array-like, default=50
            Number of bins or bin edges.
        density : bool, default=True
            If True, normalize to form a probability density.
        histtype : str, default='stepfilled'
            Type of histogram: 'bar', 'barstacked', 'step', 'stepfilled'.
        alpha : float, default=0.7
            Transparency.
        color : str, default='C0'
            Fill color.
        edgecolor : str, default='black'
            Edge color.
        linewidth : float, default=1.0
            Edge line width.
        label : str, optional
            Legend label.
        orientation : str, default='vertical'
            'vertical' or 'horizontal'.
        cumulative : bool, default=False
            If True, plot cumulative histogram.
        log : bool, default=False
            If True, use log scale for counts axis.
        **kwargs
            Additional arguments passed to ax.hist.
        
        Returns
        -------
        n : array
            Histogram values.
        bins : array
            Bin edges.
        patches : list
            Patch objects.
        
        Examples
        --------
        >>> # Basic histogram
        >>> Plotter.histogram(ax, data, bins=30, label='Distribution')
        
        >>> # Step histogram (unfilled)
        >>> Plotter.histogram(ax, data, histtype='step', linewidth=2)
        
        >>> # Cumulative distribution
        >>> Plotter.histogram(ax, data, cumulative=True, density=True)
        """
        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
        if label is None or label == '':
            label_cond = False
        
        return ax.hist(data, bins=bins, density=density,
                    histtype=histtype, alpha=alpha,
                    color=color, edgecolor=edgecolor,
                    linewidth=linewidth, 
                    label=label if label_cond else '',
                    orientation=orientation,
                    cumulative=cumulative, log=log,
                    weights=weights, range=range, align=align, rwidth=rwidth, stacked=stacked, hatch=hatch,
                    zorder=zorder, **kwargs)

    ###################################################
    
    @staticmethod
    def contourf(ax, x, y, z, **kwargs):
        '''
        contourf plotting
        '''
        cs = ax.contourf(x, y, z, **kwargs)
        return cs
    
    @staticmethod
    def grid(ax, **kwargs):
        '''
        grid plotting
        
        Kwargs include:
        - which : {'major', 'minor', 'both'}, optional, default: 'major'
            - Specifies which grid lines to apply the settings to.
        - axis : {'both', 'x', 'y'}, optional, default: 'both
            - Specifies which axis to apply the grid settings to.
        - color : color, optional
            - Color of the grid lines.
        - linestyle : str, optional
            - Style of the grid lines (e.g., '-', '--', '-.', ':').
        - linewidth : float, optional
            - Width of the grid lines.
        - alpha : float, optional
            - Transparency of the grid lines (0.0 to 1.0).
        '''
        return ax.grid(**kwargs)
    
    #################### T I C K S ####################
    
    @staticmethod
    def set_tickparams( ax,
                        labelsize       =   None,
                        left            =   True,
                        right           =   True,
                        top             =   True,
                        bottom          =   True,
                        xticks          =   None,
                        yticks          =   None,
                        xticklabels     =   None,
                        yticklabels     =   None,
                        maj_tick_l      =   4,
                        min_tick_l      =   2,
                        **kwargs
                        ):
        '''
        Sets tickparams to the desired ones.
        - ax        :   axis to use
        - labelsize :   fontsize
        - left      :   whether to show the left side
        - right     :   whether to show the right side
        - top       :   whether to show the top side
        - bottom    :   whether to show the bottom side
        - xticks    :   list of xticks
        - yticks    :   list of yticks
        '''
        ax = Plotter.ax(ax)
        
        ax.tick_params(axis='both', which='major', left=left, right=right, 
                        top=top, bottom=bottom, labelsize=labelsize)
        ax.tick_params(axis="both", which='major', left=left, right=right, 
                        top=top, bottom=bottom, direction="in",length=maj_tick_l, **kwargs)
        ax.tick_params(axis="both", which='minor', left=left, right=right, 
                        top=top, bottom=bottom, direction="in",length=min_tick_l, **kwargs)

        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
            
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)

    @staticmethod
    def set_ax_params(
            ax,
            # Axis specification
            which                   : str                                   = 'both',
            # Labels    
            xlabel                  : Optional[str]                         = None,
            ylabel                  : Optional[str]                         = None,
            title                   : Optional[str]                         = None,
            # Label styling 
            fontsize                : Optional[int]                         = None,
            labelsize_title         : Optional[int]                         = None,
            labelsize_tick          : Optional[int]                         = None,
            labelpad                : Union[float, dict]                    = 0.0,
            title_pad               : float                                 = 10.0,
            # Label positions
            xlabel_position         : Literal['top', 'bottom']              = 'bottom',
            ylabel_position         : Literal['left', 'right']              = 'left',
            # Axis limits and scales
            xlim                    : Optional[tuple]                       = None,
            ylim                    : Optional[tuple]                       = None,
            xscale                  : Literal['linear', 'log', 'symlog']    = 'linear',
            yscale                  : Literal['linear', 'log', 'symlog']    = 'linear',
            # Ticks and tick labels
            xticks                  : Optional[Union[list, np.ndarray]]     = None,
            yticks                  : Optional[Union[list, np.ndarray]]     = None,
            xticklabels             : Optional[list]                        = None,
            yticklabels             : Optional[list]                        = None,
            xtickpos                : Literal['top', 'bottom', 'both']      = None,
            ytickpos                : Literal['left', 'right', 'both']      = None,
            tick_length_major       : float                                 = 4.0,
            tick_length_minor       : float                                 = 2.0,
            tick_width              : float                                 = 0.8,
            tick_direction          : Literal['in', 'out', 'inout']         = 'in',
            # Minor ticks
            show_minor_ticks        : bool                              = True,
            minor_tick_locator      : Optional[str]                     = 'auto',  # 'auto' or 'log' for log scale
            # Grid
            grid                    : bool                              = False,
            grid_axis               : Literal['both', 'x', 'y']         = 'both',
            grid_which              : Literal['major', 'minor', 'both']  = 'major',
            grid_style              : str                               = '--',
            grid_color              : Optional[str]                     = None,
            grid_alpha              : float                             = 0.3,
            grid_linewidth          : float                             = 0.8,
            # Spines
            show_spines             : Union[bool, dict]                 = True,
            spine_width             : float                             = 1.0,
            spine_color             : str                               = 'black',
            # Aspect ratio
            aspect                  : Optional[Union[str, float]]       = None,
            # Tight layout
            tight_layout            : bool                              = False,
            # Legend
            legend                  : bool                              = False,
            legend_kwargs           : Optional[dict]                    = None,
            # Advanced options
            invert_xaxis            : bool                              = False,
            invert_yaxis            : bool                              = False,
            auto_formatter          : bool                              = True,
            # label condition
            label_cond              : bool                              = True,
            label_pos               : dict                              = None,
            tick_pos                : dict                              = None,
            **kwargs
        ):
        r"""
        Comprehensive axis configuration method for publication-quality plots.
        
        This method provides centralized control over all major axis properties
        with sensible defaults and advanced options for fine-tuning. It integrates
        with other Plotter methods for a cohesive styling experience.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis object to modify.
        which : {'both', 'x', 'y'}, default='both'
            Specifies which axes to update. Allows independent configuration
            of x and y axes.
        
        **Labels and Titles**
        
        xlabel, ylabel : str, optional
            Axis labels. Set to '' to hide labels while maintaining formatting.
        title : str, optional
            Axis title.
        fontsize : int, optional
            Default font size for labels (overridable per-element).
        labelsize_title : int, optional
            Font size for title. If None, uses `fontsize` + 2.
        labelsize_tick : int, optional
            Font size for tick labels. If None, uses `fontsize` - 2.
        labelpad : float or dict, default=0.0
            Padding between label and axis. Can be {'x': val, 'y': val}.
        title_pad : float, default=10.0
            Vertical padding between title and plot area.
        
        **Label Positioning**
        
        xlabel_position : {'top', 'bottom'}, default='bottom'
            Position of x-axis label.
        ylabel_position : {'left', 'right'}, default='left'
            Position of y-axis label.
        
        **Axis Limits and Scales**
        
        xlim, ylim : tuple, optional
            Axis limits as (min, max). Use None for auto limits.
        xscale, yscale : {'linear', 'log', 'symlog'}, default='linear'
            Axis scale type. 'symlog' uses symmetric log scaling.
        
        **Tick Configuration**
        
        xticks, yticks : list or np.ndarray, optional
            Explicit tick positions. Leave None for matplotlib auto-ticks.
        xticklabels, yticklabels : list, optional
            Custom tick labels. Must match length of ticks if provided.
        tick_length_major : float, default=4.0
            Length of major ticks in points.
        tick_length_minor : float, default=2.0
            Length of minor ticks in points.
        tick_width : float, default=0.8
            Width of ticks in points.
        tick_direction : {'in', 'out', 'inout'}, default='in'
            Direction ticks point ('in' recommended for publication).
        show_minor_ticks : bool, default=True
            Whether to show minor ticks.
        minor_tick_locator : {'auto', 'log'}, default='auto'
            How to locate minor ticks. 'log' uses LogLocator for log scales.
        
        **Grid Configuration**
        
        grid : bool, default=False
            Enable gridlines.
        grid_axis : {'both', 'x', 'y'}, default='both'
            Which axes to show grid on.
        grid_which : {'major', 'minor', 'both'}, default='major'
            Which ticks to grid on.
        grid_style : str, default='--'
            Line style ('-', '--', '-.', ':').
        grid_color : str, optional
            Grid color. If None, uses current axes color scheme.
        grid_alpha : float, default=0.3
            Transparency of gridlines (0=transparent, 1=opaque).
        grid_linewidth : float, default=0.8
            Width of gridlines in points.
        
        **Spine Configuration**
        
        show_spines : bool or dict, default=True
            Visibility of spines. 
            - True: show all spines
            - False: hide all spines
            - dict: {'top': bool, 'bottom': bool, 'left': bool, 'right': bool}
        spine_width : float, default=1.0
            Width of spines in points.
        spine_color : str, default='black'
            Color of spines.
        
        **Appearance**
        
        aspect : str or float, optional
            Aspect ratio ('equal', 'auto') or numeric value.
        tight_layout : bool, default=False
            Apply tight layout after configuration.
        
        **Legend**
        
        legend : bool, default=False
            Whether to display legend using set_legend().
        legend_kwargs : dict, optional
            Arguments to pass to set_legend() if legend=True.
        
        **Advanced Options**
        
        invert_xaxis, invert_yaxis : bool, default=False
            Invert the direction of the axes.
        auto_formatter : bool, default=True
            Automatically apply scientific notation formatter for large/small numbers.
        **kwargs
            Additional keyword arguments passed to matplotlib functions.
        
        Examples
        --------
        **Example 1: Basic publication-ready plot**
        
        >>> ax = plt.gca()
        >>> Plotter.set_ax_params(
        ...     ax,
        ...     xlabel=r'$x$ (nm)',
        ...     ylabel=r'Energy (eV)',
        ...     title='Band Structure',
        ...     xlim=(0, 10),
        ...     ylim=(-5, 5),
        ...     grid=True
        ... )
        
        **Example 2: Log-scale with custom ticks**
        
        >>> Plotter.set_ax_params(
        ...     ax,
        ...     xlabel='Frequency (Hz)',
        ...     ylabel='Magnitude',
        ...     yscale='log',
        ...     yticks=[1, 10, 100, 1000],
        ...     yticklabels=['1', '10', '100', '1 k'],
        ...     grid=True,
        ...     grid_which='both',
        ...     minor_tick_locator='log'
        ... )
        
        **Example 3: Detailed styling**
        
        >>> Plotter.set_ax_params(
        ...     ax,
        ...     xlabel='Temperature (K)',
        ...     ylabel=r'$\rho$ (Ω·cm)',
        ...     title='Resistivity vs Temperature',
        ...     fontsize=12,
        ...     labelsize_title=14,
        ...     labelsize_tick=10,
        ...     xlim=(0, 300),
        ...     ylim=(0, None),  # auto max
        ...     grid=True,
        ...     grid_style='--',
        ...     grid_alpha=0.4,
        ...     show_spines={'top': False, 'right': False},
        ...     spine_width=1.5,
        ...     tick_length_major=6,
        ...     legend=True,
        ...     tight_layout=True
        ... )
        
        **Example 4: Custom tick labels and positions**
        
        >>> import numpy as np
        >>> Plotter.set_ax_params(
        ...     ax,
        ...     xticks=np.linspace(0, 2*np.pi, 5),
        ...     xticklabels=['0', 'π/2', 'π', '3π/2', '2π'],
        ...     xlabel=r'Phase',
        ...     ylabel=r'$\sin(\phi)$'
        ... )
        
        **Example 5: Asymmetric spines (Nature style)**
        
        >>> Plotter.set_ax_params(
        ...     ax,
        ...     xlabel='Parameter',
        ...     ylabel='Value',
        ...     show_spines={'left': True, 'bottom': True, 'top': False, 'right': False},
        ...     grid=False,
        ...     tick_direction='out'
        ... )
        
        Notes
        -----
        - Set `labelsize_*` to None to auto-scale relative to `fontsize`
        - Grid is best used with light colors and low alpha (0.2-0.4)
        - For log scales, minor_tick_locator='log' is recommended
        - Use `which='x'` or `which='y'` for independent axis control
        - Integrates with Plotter.set_legend() for unified styling
        
        See Also
        --------
        set_legend : Configure legend appearance
        set_tickparams : Alternative tick configuration method
        grid : Add gridlines to axis
        """
        ax = Plotter.ax(ax)
        
        if isinstance(label_cond, (bool, int)):
            label_cond_x            = label_cond
            label_cond_y            = label_cond
        elif isinstance(label_cond, dict):
            label_cond_x            = label_cond.get('x', label_cond.get('X', label_cond.get('both', True)))
            label_cond_y            = label_cond.get('y', label_cond.get('Y', label_cond.get('both', True)))
        else:
            label_cond_x            = True
            label_cond_y            = True
        
        # Label position dictionary
        if label_pos is not None and isinstance(label_pos, dict):
            xlabel_position        = label_pos.get('x', label_pos.get('X', xlabel_position))
            ylabel_position        = label_pos.get('y', label_pos.get('Y', ylabel_position))
        
        # Tick positions
        if tick_pos is not None and isinstance(tick_pos, dict):
            xtickpos               = tick_pos.get('x', tick_pos.get('X', xtickpos))
            ytickpos               = tick_pos.get('y', tick_pos.get('Y', ytickpos))
            if xtickpos is not None:
                ax.xaxis.set_ticks_position(xtickpos)
            if ytickpos is not None:
                ax.yaxis.set_ticks_position(ytickpos)
        
        # Resolve font sizes
        if fontsize is None:
            fontsize        = plt.rcParams.get('font.size', 10)
        if labelsize_title is None:
            labelsize_title = fontsize + 2
        if labelsize_tick is None:
            labelsize_tick  = max(fontsize - 2, 8)
        
        # Resolve labelpad
        if isinstance(labelpad, (int, float)):
            labelpad = {'x': labelpad, 'y': labelpad}
        elif not isinstance(labelpad, dict):
            labelpad = {'x': 0.0, 'y': 0.0}
        
        # ===== X-AXIS CONFIGURATION =====
        if 'x' in which or 'both' in which:
            # Label
            if xlabel is not None and label_cond_x and xlabel != '':
                ax.set_xlabel(
                    xlabel,
                    fontsize=fontsize,
                    labelpad=labelpad.get('x', 0.0)
                )
            
            # Label position
            if xlabel_position in ['top', 'bottom']:
                ax.xaxis.set_label_position(xlabel_position)
            
            # Limits
            if xlim is not None:
                ax.set_xlim(xlim)
            
            # Scale
            ax.set_xscale(xscale)
            
            # Ticks
            if xticks is not None:
                ax.set_xticks(xticks)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels)
            
            # Minor ticks
            if show_minor_ticks and xscale == 'log' and minor_tick_locator == 'auto':
                ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='all', numticks=100))
            
            # Inversion
            if invert_xaxis:
                ax.invert_xaxis()
        
        # ===== Y-AXIS CONFIGURATION =====
        if 'y' in which or 'both' in which:
            # Label
            if ylabel is not None and label_cond_y and ylabel != '':
                ax.set_ylabel(
                    ylabel,
                    fontsize=fontsize,
                    labelpad=labelpad.get('y', 0.0)
                )
            
            # Label position
            if ylabel_position in ['left', 'right']:
                ax.yaxis.set_label_position(ylabel_position)
            
            # Limits
            if ylim is not None:
                ax.set_ylim(ylim)
            
            # Scale
            ax.set_yscale(yscale)
            
            # Ticks
            if yticks is not None:
                ax.set_yticks(yticks)
            if yticklabels is not None:
                ax.set_yticklabels(yticklabels)
            
            # Minor ticks
            if show_minor_ticks and yscale == 'log' and minor_tick_locator == 'auto':
                ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='all', numticks=100))
            
            # Inversion
            if invert_yaxis:
                ax.invert_yaxis()
        
        # ===== TITLE CONFIGURATION =====
        if title is not None and title != '':
            ax.set_title(
                title,
                fontsize=labelsize_title,
                pad=title_pad
            )
        
        # ===== TICK CONFIGURATION =====
        ax.tick_params(
            axis='both',
            which='major',
            length=tick_length_major,
            width=tick_width,
            direction=tick_direction,
            labelsize=labelsize_tick,
            **kwargs
        )
        if show_minor_ticks:
            ax.tick_params(
                axis='both',
                which='minor',
                length=tick_length_minor,
                width=tick_width,
                direction=tick_direction
            )
        
        # ===== GRID CONFIGURATION =====
        if grid:
            ax.grid(
                True,
                axis=grid_axis,
                which=grid_which,
                linestyle=grid_style,
                color=grid_color or ax.grid.__defaults__[0] if hasattr(ax.grid, '__defaults__') else 'gray',
                alpha=grid_alpha,
                linewidth=grid_linewidth
            )
        else:
            ax.grid(False)
        
        # ===== SPINE CONFIGURATION =====
        if isinstance(show_spines, bool):
            # Show or hide all spines
            for spine in ax.spines.values():
                spine.set_visible(show_spines)
                if show_spines:
                    spine.set_linewidth(spine_width)
                    spine.set_color(spine_color)
        else:
            # Selective spine configuration
            spine_config = {
                'left': True, 'bottom': True, 'top': True, 'right': True
            }
            spine_config.update(show_spines)
            
            for spine_name, visible in spine_config.items():
                if spine_name in ax.spines:
                    ax.spines[spine_name].set_visible(visible)
                    if visible:
                        ax.spines[spine_name].set_linewidth(spine_width)
                        ax.spines[spine_name].set_color(spine_color)
        
        # ===== ASPECT RATIO =====
        if aspect is not None:
            ax.set_aspect(aspect)
        
        # ===== AUTO-FORMATTER =====
        if auto_formatter:
            try:
                from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation
                # Apply scientific notation for very large/small numbers
                if xscale == 'log':
                    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
                if yscale == 'log':
                    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
            except Exception:
                pass  # Graceful fallback if formatter not available
        
        # ===== LEGEND =====
        if legend:
            legend_kw = legend_kwargs or {}
            Plotter.set_legend(ax, **legend_kw)
        
        # ===== TIGHT LAYOUT =====
        if tight_layout:
            try:
                ax.figure.tight_layout()
            except Exception:
                pass  # Some backends don't support tight_layout

    @staticmethod
    def set_xlabel(
        ax,
        xlabel,
        fontsize=None,
        labelpad=0,
        loc=None,
        x=None,
        y=None,
        coords: str = "axes",
        transform=None,
        **kwargs,
    ):
        """
        Set x-axis label with optional alignment and explicit coordinates.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axis.
        xlabel : str
            Label text.
        fontsize : int, optional
            Label font size.
        labelpad : float, default=0
            Padding in points.
        loc : {'left', 'center', 'right'}, optional
            Matplotlib label location argument.
        x, y : float, optional
            Explicit label coordinates (if either is provided).
        coords : {'axes', 'data'}, default='axes'
            Coordinate system used for ``x``/``y`` when ``transform`` is not provided.
        transform : matplotlib transform, optional
            Explicit transform for label coordinates.
        **kwargs
            Forwarded to ``ax.set_xlabel``.
        """
        ax = Plotter.ax(ax)
        if loc is not None:
            kwargs.setdefault("loc", loc)
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad if labelpad != 0 else None, **kwargs)

        if x is not None or y is not None:
            x0, y0 = ax.xaxis.get_label().get_position()
            x_set = x0 if x is None else x
            y_set = y0 if y is None else y
            t = transform
            if t is None:
                t = ax.transData if str(coords).lower() == "data" else ax.transAxes
            ax.xaxis.set_label_coords(x_set, y_set, transform=t)
        
    @staticmethod
    def set_ylabel(
        ax,
        ylabel,
        fontsize=None,
        labelpad=0,
        loc=None,
        x=None,
        y=None,
        coords: str = "axes",
        transform=None,
        **kwargs,
    ):
        """
        Set y-axis label with optional alignment and explicit coordinates.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axis.
        ylabel : str
            Label text.
        fontsize : int, optional
            Label font size.
        labelpad : float, default=0
            Padding in points.
        loc : {'bottom', 'center', 'top'}, optional
            Matplotlib label location argument.
        x, y : float, optional
            Explicit label coordinates (if either is provided).
        coords : {'axes', 'data'}, default='axes'
            Coordinate system used for ``x``/``y`` when ``transform`` is not provided.
        transform : matplotlib transform, optional
            Explicit transform for label coordinates.
        **kwargs
            Forwarded to ``ax.set_ylabel``.
        """
        ax = Plotter.ax(ax)
        if loc is not None:
            kwargs.setdefault("loc", loc)
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad if labelpad != 0 else None, **kwargs)

        if x is not None or y is not None:
            x0, y0 = ax.yaxis.get_label().get_position()
            x_set = x0 if x is None else x
            y_set = y0 if y is None else y
            t = transform
            if t is None:
                t = ax.transData if str(coords).lower() == "data" else ax.transAxes
            ax.yaxis.set_label_coords(x_set, y_set, transform=t)

    @staticmethod
    def set_ax_labels(  ax,
                        fontsize    =   None,
                        xlabel      =   "",
                        ylabel      =   "",
                        title       =   "",
                        xPad        =   0,
                        yPad        =   0,
                        xloc        =   None,
                        yloc        =   None,
                        xcoords     : str = "axes",
                        ycoords     : str = "axes",
                        x_pos       =   None,
                        y_pos       =   None):
        '''
        Sets the labels of the x and y axes
        '''
        ax = Plotter.ax(ax)
        if xlabel != "":
            x_kwargs = {}
            if isinstance(x_pos, (tuple, list)) and len(x_pos) == 2:
                x_kwargs.update({"x": x_pos[0], "y": x_pos[1], "coords": xcoords})
            Plotter.set_xlabel(
                ax,
                xlabel,
                fontsize=fontsize,
                labelpad=xPad,
                loc=xloc,
                **x_kwargs,
            )
        if ylabel != "":
            y_kwargs = {}
            if isinstance(y_pos, (tuple, list)) and len(y_pos) == 2:
                y_kwargs.update({"x": y_pos[0], "y": y_pos[1], "coords": ycoords})
            Plotter.set_ylabel(
                ax,
                ylabel,
                fontsize=fontsize,
                labelpad=yPad,
                loc=yloc,
                **y_kwargs,
            )
        # check the title
        if len(title) != 0:
            ax.set_title(title)    
    
    @staticmethod 
    def set_label_cords(ax, 
                        which   : str,
                        inX     = 0.0,
                        inY     = 0.0,
                        **kwargs):
        '''
        Sets the coordinates of the labels
        '''
        if 'x' in which:
            ax.xaxis.set_label_coords(inX, inY, **kwargs)
        if 'y' in which:
            ax.yaxis.set_label_coords(inX, inY, **kwargs)
    
    @staticmethod
    def setup_log_y(ax: plt.Axes, ylims=(1e-12, 1e6), decade_step=4):
        """Configure clean log-scale y ticks at powers of 10 with LaTeX-like labels."""
        ax.set_yscale('log')
        ax.set_ylim(*ylims)

        # major ticks every `decade_step` decades (e.g. 1e-8, 1e-4, 1e0, 1e4)
        lo, hi  = np.log10(ylims[0]), np.log10(ylims[1])
        start   = int(np.ceil(lo / decade_step) * decade_step)
        stop    = int(np.floor(hi / decade_step) * decade_step)
        majors  = 10.0 ** np.arange(start, stop + 1, decade_step, dtype=float)

        ax.yaxis.set_major_locator(FixedLocator(majors))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))  # shows 10^{n}

        # minors at 2..9 within each decade
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())

        # cosmetic tick lengths once (avoid mixing with Plotter if it already does this)
        ax.tick_params(axis='y', which='major', length=4)
        ax.tick_params(axis='y', which='minor', length=2)

    @staticmethod
    def setup_log_x(ax: plt.Axes, xlims=(1e-12, 1e6), decade_step=4):
        """Configure clean log-scale x ticks at powers of 10 with LaTeX-like labels."""
        ax.set_xscale('log')
        ax.set_xlim(*xlims)

        # major ticks every `decade_step` decades (e.g. 1e-8, 1e-4, 1e0, 1e4)
        lo, hi  = np.log10(xlims[0]), np.log10(xlims[1])
        start   = int(np.ceil(lo / decade_step) * decade_step)
        stop    = int(np.floor(hi / decade_step) * decade_step)
        majors  = 10.0 ** np.arange(start, stop + 1, decade_step, dtype=float)

        ax.xaxis.set_major_locator(FixedLocator(majors))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))  # shows 10^{n}

        # minors at 2..9 within each decade
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())

        # cosmetic tick lengths once (avoid mixing with Plotter if it already does this)
        ax.tick_params(axis='x', which='major', length=4)
        ax.tick_params(axis='x', which='minor', length=2)

    @staticmethod
    def set_smart_lim(
        ax,
        *,
        which           : str = "both",             # "x", "y", or "both"
        data            : Union[np.ndarray, None] = None, # 1-D or 2-D; if None infer from artists
        margin_p        : float = 0,                # log_scale moves
        margin_m        : float = 1,                # log_scale moves
        xlim            : Union[tuple, None] = None,      # x limits
        ylim            : Union[tuple, None] = None,      # y limits
        ):
        """
        Auto-compute robust axis limits and apply them to *ax*.
        """
        if which not in ["x", "y", "both"]:
            raise ValueError(f"Invalid axis: {which}. Use 'x', 'y', or 'both'.")
        
        if which == 'y' or which == 'both':
            if data is not None:
                d_max   = np.max(data)
                d_max   = np.ceil(np.log10(d_max))
                d_min   = np.floor(np.log10(np.min(data[data > 0])))
                d_lim   = (10**(d_min-margin_m), 10**(d_max+margin_p))
                d_lim   = ylim if ylim is not None else d_lim
            else:
                d_lim   = ylim
            ax.set_ylim(d_lim)
            
        if which == 'x' or which == 'both':
            if data is not None:
                d_max   = np.max(data)
                d_max   = np.ceil(np.log10(d_max))
                d_min   = np.floor(np.log10(np.min(data[data > 0])))
                d_lim   = (10**(d_min-margin_m), 10**(d_max+margin_p))
                d_lim   = xlim if xlim is not None else d_lim
            else:
                d_lim   = xlim
            ax.set_xlim(d_lim)
        return d_lim
    
    #################### L A B E L ####################
    
    @staticmethod
    def hide_unused_panels(axes: plt.Axes, n_panels: int):
        """Hide unused panels in a subplot grid."""
        if n_panels > 1:
            for idx in range(n_panels, len(axes.flatten())):
                axes.flatten()[idx].set_visible(False)
    
    @staticmethod
    def labellines(ax, 
                    align       = False,
                    xvals       = None,
                    yoffsets    = [],
                    zorder      = 2,
                    **kwargs):
        """
        Add labels to lines with a given slope. Uses labelLines package. 
        Args:
            - ax: Matplotlib axis object.
            - align: Align the label with the slope of the line.
            - xvals: The x values to place the labels at.
            - yoffsets: The y offsets for the labels.
            - zorder: The zorder of the labels.
        """
        labelLines(ax.get_lines(), 
                   align    =   align, 
                   xvals    =   xvals, 
                   yoffsets =   yoffsets, 
                   zorder   =   zorder,
                   **kwargs)
    
    #################### U N S E T ####################

    @staticmethod
    def unset_spines(ax,
                     top: bool = True,
                     right: bool = True,
                     bottom: bool = False,
                     left: bool = False):
        """
        Remove specified spines from the axis for cleaner publication-style plots.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to modify.
        top : bool, default=True
            If True, REMOVE the top spine. If False, KEEP it.
        right : bool, default=True
            If True, REMOVE the right spine. If False, KEEP it.
        bottom : bool, default=False
            If True, REMOVE the bottom spine. If False, KEEP it.
        left : bool, default=False
            If True, REMOVE the left spine. If False, KEEP it.
        
        Examples
        --------
        # Nature-style (remove top and right, keep left and bottom) - DEFAULT
        >>> Plotter.unset_spines(ax)
        
        # Remove all spines (frameless plot)
        >>> Plotter.unset_spines(ax, top=True, right=True, bottom=True, left=True)
        
        # Keep all spines
        >>> Plotter.unset_spines(ax, top=False, right=False, bottom=False, left=False)
        
        # Only keep bottom spine (minimal style)
        >>> Plotter.unset_spines(ax, top=True, right=True, bottom=False, left=True)
        
        Notes
        -----
        The default settings (top=True, right=True) produce the classic 
        "Nature" or "Science" journal style with only left and bottom spines.
        """
        # True means REMOVE, so we set_visible(not param)
        ax.spines['top'].set_visible(not top)
        ax.spines['right'].set_visible(not right)
        ax.spines['bottom'].set_visible(not bottom)
        ax.spines['left'].set_visible(not left)

    @staticmethod
    def unset_ticks(ax,
                    xticks: bool = False,
                    yticks: bool = False,
                    xlabel: bool = False,
                    ylabel: bool = False,
                    remove_labels_only: bool = True):
        """
        Remove tick labels (and optionally tick marks) from the axis.
        
        Useful for creating clean shared-axis plots where inner panels
        don't need redundant tick labels.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to modify.
        xticks : bool, default=False
            If True, REMOVE x-tick labels. If False, KEEP them.
        yticks : bool, default=False
            If True, REMOVE y-tick labels. If False, KEEP them.
        xlabel : bool, default=False
            If True, also REMOVE the x-axis label.
        ylabel : bool, default=False
            If True, also REMOVE the y-axis label.
        remove_labels_only : bool, default=True
            If True, only remove the text labels, keeping tick marks visible.
            If False, remove both the tick marks and labels.
        
        Examples
        --------
        # Remove x-tick labels for stacked plots with shared x-axis
        >>> for ax in axes[:-1]:  # All except bottom
        ...     Plotter.unset_ticks(ax, xticks=True, xlabel=True)
        
        # Remove all tick labels (keep tick marks)
        >>> Plotter.unset_ticks(ax, xticks=True, yticks=True)
        
        # Remove tick marks AND labels (completely clean)
        >>> Plotter.unset_ticks(ax, xticks=True, yticks=True, remove_labels_only=False)
        
        # Remove y-ticks and y-axis label for side-by-side shared y-axis
        >>> for ax in axes[1:]:  # All except leftmost
        ...     Plotter.unset_ticks(ax, yticks=True, ylabel=True)
        
        Notes
        -----
        This function is commonly used in combination with sharex/sharey in 
        multi-panel figures to avoid redundant labels.
        """
        if xticks:
            if remove_labels_only:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        if yticks:
            if remove_labels_only:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        
        if xlabel:
            ax.set_xlabel('')
        
        if ylabel:
            ax.set_ylabel('')

    @staticmethod
    def unset_all(ax, spines: bool = True, ticks: bool = True, labels: bool = True):
        """
        Completely strip an axis of spines, ticks, and labels.
        
        Useful for image plots, heatmaps, or decorative panels where
        axis elements are not needed.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to modify.
        spines : bool, default=True
            If True, remove all spines.
        ticks : bool, default=True
            If True, remove all tick marks and labels.
        labels : bool, default=True
            If True, remove axis labels.
        
        Examples
        --------
        # Completely clean axis (for images/heatmaps)
        >>> Plotter.unset_all(ax)
        
        # Keep only spines (box around plot)
        >>> Plotter.unset_all(ax, spines=False)
        """
        if spines:
            Plotter.unset_spines(ax, top=True, right=True, bottom=True, left=True)
        if ticks:
            Plotter.unset_ticks(ax, xticks=True, yticks=True, remove_labels_only=False)
        if labels:
            ax.set_xlabel('')
            ax.set_ylabel('')

    @staticmethod
    def unset_ticks_and_spines(ax, xticks: bool = True, yticks: bool = True, top: bool = True, right: bool = True, bottom: bool = False, left: bool = False):
        """
        Convenience method to remove both ticks and spines in one call.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to modify.
        xticks : bool, default=True
            If True, REMOVE x-tick labels.
        yticks : bool, default=True
            If True, REMOVE y-tick labels.
        top, right, bottom, left : bool
            If True, REMOVE the corresponding spine.
            Defaults remove top and right (Nature-style).
        
        Examples
        --------
        # Clean Nature-style with no tick labels
        >>> Plotter.unset_ticks_and_spines(ax)
        
        # Only remove top/right spines, keep all ticks
        >>> Plotter.unset_ticks_and_spines(ax, xticks=False, yticks=False)
        """
        Plotter.unset_spines(ax, top=top, right=right, bottom=bottom, left=left)
        Plotter.unset_ticks(ax, xticks=xticks, yticks=yticks)
    
    ################### F O R M A T ###################
    
    @staticmethod
    def set_formater(ax, 
                     formater = "%.1e",
                     axis     = 'xy'):
        """
        Sets the formatter for the given axis on the plot.
        Args:
            ax (object): The axis object on which to set the formatter.
            formater (str, optional): The format string for the axis labels. Defaults to "%.1e".
            axis (str, optional): The axis on which to set the formatter. Defaults to 'xy'.
        Returns:
            None
        """
        if 'y' in axis:
            ax.yaxis.set_major_formatter(MathTextSciFormatter(formater))
        if 'x' in axis:
            ax.xaxis.set_major_formatter(MathTextSciFormatter(formater))
    
    @staticmethod
    def set_standard_formater(ax, axis = 'xy'):
        """
        Sets the formatter for the given axis on the plot.
        Args:
            ax (object): The axis object on which to set the formatter.
            axis (str, optional): The axis on which to set the formatter. Defaults to 'xy'.
        Returns:
            None
        """
        if 'x' in axis:
            ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda x, pos: "%g"%x))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "%g"%x))
        if 'y' in axis:
            ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda x, pos: "%g"%x))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "%g"%x))
    
    #################### G R I D S ####################
    #
    # Grid Layout System for Complex Multi-Panel Figures
    # ===================================================
    #
    # This section provides comprehensive tools for creating complex figure layouts
    # with fine control over panel sizes, spacing, and nesting.
    #
    # KEY CONCEPTS:
    # - GridSpec: Define a grid layout with rows/columns and spacing
    # - Subplots: Individual axes within the grid
    # - Nesting: GridSpecFromSubplotSpec allows grids within grids
    #
    # COMMON WORKFLOWS:
    #
    # 1. Simple grid (equal panels):
    #    >>> fig, axes = Plotter.make_grid(2, 3, figsize=(10, 8))
    #
    # 2. Unequal panel widths:
    #    >>> fig, axes = Plotter.make_grid(1, 3, width_ratios=[2, 1, 1])
    #    # First panel is 2x wider than others
    #
    # 3. Complex nested layouts:
    #    >>> builder = Plotter.GridBuilder(figsize=(12, 8))
    #    >>> builder.add_row(ncols=1, height_ratio=1)
    #    >>> builder.add_row(ncols=3, height_ratio=2)
    #    >>> fig, axes = builder.build()
    #
    ##########################################
    
    # GridBuilder class for complex nested layouts
    class GridBuilder:
        """
        Builder class for creating complex figure layouts with nested grids.
        
        Use this when you need different numbers of columns in different rows,
        or complex nested arrangements that can't be achieved with a simple grid.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Figure size in inches (width, height).
        
        Examples
        --------
        Create a layout with varying column counts per row::
        
        >>> builder = Plotter.GridBuilder(figsize=(12, 8))
        >>> builder.add_row(ncols=1, height_ratio=1)     # Header row (1 panel)
        >>> builder.add_row(ncols=3, height_ratio=2)     # Main row (3 panels)
        >>> builder.add_row(ncols=2, height_ratio=1.5)   # Footer row (2 panels)
        >>> fig, axes = builder.build(wspace=0.2, hspace=0.3)
        >>> # axes = [[ax00], [ax10, ax11, ax12], [ax20, ax21]]
        
        Access axes::
        
        >>> header_ax = axes[0][0]
        >>> main_left, main_center, main_right = axes[1]
        >>> footer_left, footer_right = axes[2]
        """
        
        def __init__(self, figsize=(10, 8)):
            self.figsize    = figsize
            self.rows       = []
            
        def add_row(self, ncols: int, height_ratio: float = 1.0, 
                    width_ratios: List[float] = None):
            """
            Add a row to the layout.
            
            Parameters
            ----------
            ncols : int
                Number of columns in this row.
            height_ratio : float, default=1.0
                Relative height of this row compared to others.
            width_ratios : list of float, optional
                Relative widths of columns within this row.
                If None, columns are equal width.
            
            Returns
            -------
            self : GridBuilder
                For method chaining.
            """
            if width_ratios is not None and len(width_ratios) != ncols:
                raise ValueError(f"width_ratios length ({len(width_ratios)}) must match ncols ({ncols})")
            self.rows.append({
                'ncols'         : ncols,
                'height_ratio'  : height_ratio,
                'width_ratios'  : width_ratios or [1.0] * ncols
            })
            return self
        
        def build(self, wspace: float = 0.2, hspace: float = 0.2,
                  left: float = 0.1, right: float = 0.95,
                  top: float = 0.95, bottom: float = 0.1):
            """
            Build the figure with the specified layout.
            
            Parameters
            ----------
            wspace : float, default=0.2
                Horizontal space between columns within rows.
            hspace : float, default=0.2
                Vertical space between rows.
            left, right, top, bottom : float
                Figure margins (fraction of figure size).
            
            Returns
            -------
            fig : matplotlib.figure.Figure
                The created figure.
            axes : list of lists
                2D list of axes, where axes[row][col] gives the axis
                at that position.
            """
            fig = plt.figure(figsize=self.figsize)
            
            nrows = len(self.rows)
            height_ratios = [r['height_ratio'] for r in self.rows]
            
            # Create outer grid (one column, multiple rows)
            outer_gs = GridSpec(nrows=nrows, ncols=1, figure=fig,
                               height_ratios=height_ratios, hspace=hspace,
                               left=left, right=right, top=top, bottom=bottom)
            
            axes = []
            for i, row_spec in enumerate(self.rows):
                # Create inner grid for this row
                inner_gs = GridSpecFromSubplotSpec(
                    nrows=1, ncols=row_spec['ncols'],
                    subplot_spec=outer_gs[i],
                    width_ratios=row_spec['width_ratios'],
                    wspace=wspace
                )
                row_axes = [fig.add_subplot(inner_gs[0, j]) for j in range(row_spec['ncols'])]
                axes.append(row_axes)
            
            return fig, axes
    
    @staticmethod
    def make_grid(nrows             : int, 
                ncols               : int, 
                figsize             : tuple = (10, 8),
                width_ratios        : List[float] = None,
                height_ratios       : List[float] = None,
                wspace              : float = 0.2,
                hspace              : float = 0.2,
                left                : float = 0.1, right: float = 0.95,
                top                 : float = 0.95, bottom: float = 0.1,
                sharex              : str = False,
                sharey              : str = False,
                panel_labels        : bool = False,
                panel_label_style   : str = 'parenthesis',
                despine             : bool = False):
        """
        Create a figure with a grid of subplots with full control over layout.
        
        This is the recommended method for creating publication-quality
        multi-panel figures with precise control over spacing and sizing.
        
        Parameters
        ----------
        nrows : int
            Number of rows.
        ncols : int
            Number of columns.
        figsize : tuple, default=(10, 8)
            Figure size in inches (width, height).
        width_ratios : list of float, optional
            Relative widths of columns. Length must equal ncols.
            Example: [2, 1, 1] makes first column 2x wider.
        height_ratios : list of float, optional
            Relative heights of rows. Length must equal nrows.
            Example: [1, 3] makes second row 3x taller.
        wspace : float, default=0.2
            Horizontal space between columns (fraction of avg width).
        hspace : float, default=0.2
            Vertical space between rows (fraction of avg height).
        left, right, top, bottom : float
            Figure margins (0 to 1, fraction of figure size).
        sharex : str or bool, default=False
            Share x-axis: 'row', 'col', 'all', or False.
        sharey : str or bool, default=False
            Share y-axis: 'row', 'col', 'all', or False.
        panel_labels : bool, default=False
            Add panel labels (a), (b), (c), etc.
        panel_label_style : str, default='parenthesis'
            Style for panel labels: 'parenthesis', 'plain', 'bold'.
        despine : bool, default=False
            Remove top and right spines (Nature-style).
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axes : list of Axes
            Flat list of axes [ax0, ax1, ax2, ...], row-major order.
        
        Examples
        --------
        Basic 2x3 grid::
        
        >>> fig, axes = Plotter.make_grid(2, 3, figsize=(10, 6))
        >>> ax0, ax1, ax2, ax3, ax4, ax5 = axes
        
        Unequal column widths::
        
        >>> fig, axes = Plotter.make_grid(1, 2, width_ratios=[3, 1])
        
        Stacked panels with shared x-axis::
        
        >>> fig, axes = Plotter.make_grid(3, 1, sharex='col', hspace=0.05)
        >>> for ax in axes[:-1]:
        ...     Plotter.unset_ticks(ax, xticks=True, xlabel=True)
        
        Publication figure::
        
        >>> fig, axes = Plotter.make_grid(2, 2, figsize=(8, 8), 
        ...                               panel_labels=True, despine=True)
        """
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize,
                                gridspec_kw =   {
                                    'width_ratios'  : width_ratios,
                                    'height_ratios' : height_ratios,
                                    'wspace'        : wspace,
                                    'hspace'        : hspace,
                                    'left'          : left,
                                    'right'         : right,
                                    'top'           : top,
                                    'bottom'        : bottom,
                                },
                                sharex              =   sharex, 
                                sharey              =   sharey,
                                squeeze             =   False)
        
        # Flatten to list
        axes = axs.flatten().tolist()
        
        # Apply panel labels
        if panel_labels:
            labels = 'abcdefghijklmnopqrstuvwxyz'
            for i, ax in enumerate(axes):
                if i < len(labels):
                    if panel_label_style == 'parenthesis':
                        label = f'({labels[i]})'
                    elif panel_label_style == 'bold':
                        label = f'\\textbf{{{labels[i]}}}'
                    else:
                        label = labels[i]
                    ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')
        
        # Apply despine
        if despine:
            for ax in axes:
                Plotter.unset_spines(ax, top=True, right=True)
        
        return fig, axes
    
    ##########################################
    
    @staticmethod
    def figure(figsize: tuple = (10, 8), **kwargs) -> plt.Figure:
        """
        Create a Matplotlib figure with specified size and options.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Figure size in inches (width, height).
        **kwargs
            Additional keyword arguments passed to plt.figure().
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object.
        
        Examples
        --------
        Basic figure creation::
        
        >>> fig = Plotter.figure(figsize=(12, 6))
        
        With additional options::
        
        >>> fig = Plotter.figure(figsize=(8, 8), dpi=150, facecolor='white')
        """
        return plt.figure(figsize=figsize, **kwargs)
    
    @staticmethod
    def get_grid(nrows          : int,
                ncols           : int,
                *,
                wspace          : float         = None,
                hspace          : float         = None,
                width_ratios    : List[float]   = None,
                height_ratios   : List[float]   = None,
                ax_sub          = None,
                left            : float         = None, 
                right           : float         = None,
                top             : float         = None, 
                bottom          : float         = None,
                figure          = None,
                **kwargs) -> GridSpec:
        """
        Create a GridSpec for flexible subplot layouts.
        
        This is the foundation for creating complex multi-panel figures with
        control over panel sizes and spacing.
        
        Parameters
        ----------
        nrows : int
            Number of rows in the grid.
        ncols : int
            Number of columns in the grid.
        wspace : float, optional
            Width space between columns (0.0 to 1.0, fraction of average axis width).
            Recommended: 0.2-0.4 for labels, 0.05-0.1 for tight layouts.
        hspace : float, optional
            Height space between rows (0.0 to 1.0, fraction of average axis height).
            Recommended: 0.2-0.4 for titles, 0.05-0.1 for tight layouts.
        width_ratios : list of float, optional
            Relative widths of columns. E.g., [2, 1, 1] makes first column 2x wider.
            Length must equal ncols.
        height_ratios : list of float, optional
            Relative heights of rows. E.g., [1, 2] makes second row 2x taller.
            Length must equal nrows.
        ax_sub : SubplotSpec, optional
            If provided, creates a nested GridSpec within this subplot.
            Use for complex layouts with grids inside grids.
        left, right, top, bottom : float, optional
            Figure margins (0.0 to 1.0). Controls space for labels.
        **kwargs
            Additional arguments passed to GridSpec.
        
        Returns
        -------
        GridSpec or GridSpecFromSubplotSpec
            The grid specification object.
        
        Examples
        --------
        Basic 2x3 grid::
        
        >>> fig = plt.figure(figsize=(12, 8))
        >>> gs  = Plotter.get_grid(2, 3, wspace=0.3, hspace=0.4)
        >>> ax0 = fig.add_subplot(gs[0, 0])  # Row 0, Col 0
        >>> ax1 = fig.add_subplot(gs[0, 1:]) # Row 0, Cols 1-2 (span)
        >>> ax2 = fig.add_subplot(gs[1, :])  # Row 1, all columns (span)
        
        Unequal widths (main panel + sidebar)::
        
        >>> gs = Plotter.get_grid(1, 2, width_ratios=[3, 1])
        >>> # First column is 3x wider than second
        
        Nested grid (inset layout)::
        
        >>> outer       = Plotter.get_grid(1, 2)
        >>> ax_left     = fig.add_subplot(outer[0])
        >>> inner       = Plotter.get_grid(2, 2, ax_sub=outer[1], wspace=0.1, hspace=0.1)
        >>> ax_inner_00 = fig.add_subplot(inner[0, 0])
        
        Control margins::
        
        >>> gs = Plotter.get_grid(2, 2, left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        See Also
        --------
        get_grid_subplot    : Create subplot from GridSpec index
        get_subplots        : High-level function for simple layouts
        """
        # Validate ratios
        if width_ratios is not None and len(width_ratios) != ncols:
            raise ValueError(f"width_ratios length ({len(width_ratios)}) must equal ncols ({ncols})")
        if height_ratios is not None and len(height_ratios) != nrows:
            raise ValueError(f"height_ratios length ({len(height_ratios)}) must equal nrows ({nrows})")
        
        # Build kwargs for GridSpec
        gs_kwargs = {k: v for k, v in {
            'wspace'        : wspace,
            'hspace'        : hspace,
            'width_ratios'  : width_ratios,
            'height_ratios' : height_ratios,
            'left'          : left,
            'right'         : right,
            'top'           : top,
            'bottom'        : bottom,
            'figure'        : figure,
        }.items() if v is not None}
        gs_kwargs.update(kwargs)
        
        if ax_sub is not None:
            return GridSpecFromSubplotSpec(nrows=nrows, ncols=ncols, subplot_spec=ax_sub, **gs_kwargs)
        else:
            return GridSpec(nrows=nrows, ncols=ncols, **gs_kwargs)
    
    @staticmethod
    def get_grid_subplot(gs, fig, index, sharex=None, sharey=None, **kwargs):
        """
        Create a subplot from a GridSpec at the specified index.
        
        Parameters
        ----------
        gs : GridSpec
            The GridSpec object.
        fig : matplotlib.figure.Figure
            The figure to add the subplot to.
        index : int, tuple, or slice
            Position in the grid. Can be:
            - int: Linear index (0, 1, 2, ...)
            - tuple: (row, col) for single cell
            - slice/tuple with slices: For spanning multiple cells
        sharex, sharey : Axes, optional
            Share axis with another subplot. Use for aligned multi-panel figures.
        **kwargs
            Additional arguments passed to fig.add_subplot.
        
        Returns
        -------
        matplotlib.axes.Axes
            The created subplot.
        
        Examples
        --------
        Single cell by linear index::
        
            ax0 = Plotter.get_grid_subplot(gs, fig, 0)  # First cell
            ax1 = Plotter.get_grid_subplot(gs, fig, 1)  # Second cell
        
        Single cell by (row, col)::
        
            ax = fig.add_subplot(gs[1, 2])  # Row 1, Col 2
        
        Span multiple cells::
        
            ax_wide = fig.add_subplot(gs[0, :])   # Entire first row
            ax_tall = fig.add_subplot(gs[:, 0])   # Entire first column
            ax_block = fig.add_subplot(gs[0:2, 1:3])  # 2x2 block
        
        Shared axes (for aligned panels)::
        
            ax0 = Plotter.get_grid_subplot(gs, fig, 0)
            ax1 = Plotter.get_grid_subplot(gs, fig, 1, sharex=ax0)
            ax2 = Plotter.get_grid_subplot(gs, fig, 2, sharex=ax0, sharey=ax0)
            # ax1 and ax2 share x-axis with ax0; ax2 also shares y-axis
        """
        return fig.add_subplot(gs[index], sharex=sharex, sharey=sharey, **kwargs)
    
    @staticmethod
    def get_grid_map(nrows: int, ncols: int) -> dict:
        """
        Generate a mapping from panel labels to grid indices.
        
        Useful for referencing panels by name rather than index.
        
        Parameters
        ----------
        nrows : int
            Number of rows.
        ncols : int
            Number of columns.
        
        Returns
        -------
        dict
            Mapping with keys:
            - 'by_index': {0: (0,0), 1: (0,1), ...}
            - 'by_letter': {'a': 0, 'b': 1, ...}
            - 'by_rowcol': {(0,0): 0, (0,1): 1, ...}
            - 'grid': 2D list of indices
        
        Examples
        --------
        >>> gmap = Plotter.get_grid_map(2, 3)
        >>> gmap['by_letter']['c']  # Get index for panel 'c'
        2
        >>> gmap['by_index'][4]  # Get (row, col) for index 4
        (1, 1)
        >>> gmap['grid']  # 2D layout
        [[0, 1, 2], [3, 4, 5]]
        """
        by_index = {}
        by_letter = {}
        by_rowcol = {}
        grid = []
        
        idx = 0
        for row in range(nrows):
            row_list = []
            for col in range(ncols):
                by_index[idx] = (row, col)
                by_rowcol[(row, col)] = idx
                by_letter[chr(97 + idx)] = idx  # 'a', 'b', 'c', ...
                row_list.append(idx)
                idx += 1
            grid.append(row_list)
        
        return {
            'by_index': by_index,
            'by_letter': by_letter,
            'by_rowcol': by_rowcol,
            'grid': grid,
            'nrows': nrows,
            'ncols': ncols,
        }
    
    @staticmethod
    def configure_axes(ax,
                    # Visibility
                    visible: bool = True,
                    # Spines
                    spines: Union[bool, dict, str] = True,
                    # Ticks
                    ticks: Union[bool, dict, str] = True,
                    tick_labels: Union[bool, dict, str] = True,
                    # Labels
                    xlabel: str = None,
                    ylabel: str = None,
                    title: str = None,
                    # Scale
                    xscale: str = None,
                    yscale: str = None,
                    # Limits
                    xlim: tuple = None,
                    ylim: tuple = None,
                    # Style
                    fontsize: int = None,
                    **kwargs):
        r"""
        Configure axis visibility, spines, ticks, and labels in one call.
        
        This is a convenience function for common axis customizations.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to configure.
        visible : bool, default=True
            If False, hide the entire axis (ax.axis('off')).
        spines : bool, dict, or str, default=True
            Control spine visibility:
            - True: Show all spines
            - False: Hide all spines
            - 'left': Hide all except left
            - 'bottom': Hide all except bottom
            - 'minimal': Hide top and right (Nature-style)
            - dict: {'top': False, 'right': False, ...}
        ticks : bool, dict, or str, default=True
            Control tick visibility:
            - True/False: Show/hide all ticks
            - 'x'/'y': Show only x/y ticks
            - dict: {'x': True, 'y': False}
        tick_labels : bool, dict, or str, default=True
            Control tick label visibility (same format as ticks).
        xlabel, ylabel, title : str, optional
            Axis labels and title.
        xscale, yscale : str, optional
            Axis scale: 'linear', 'log', 'symlog'.
        xlim, ylim : tuple, optional
            Axis limits as (min, max).
        fontsize : int, optional
            Font size for labels.
        **kwargs
            Additional arguments (e.g., labelpad).
        
        Examples
        --------
        Minimal style (no top/right spines)::
        
            Plotter.configure_axes(ax, spines='minimal', xlabel='Time', ylabel='Value')
        
        Hide axis completely (for images/heatmaps)::
        
            Plotter.configure_axes(ax, visible=False)
        
        Keep only left spine and y-ticks::
        
            Plotter.configure_axes(ax, spines='left', ticks='y', tick_labels='y')
        
        Log scale with custom limits::
        
            Plotter.configure_axes(ax, yscale='log', ylim=(1e-6, 1e0))
        
        Full configuration::
        
            Plotter.configure_axes(
                ax,
                spines='minimal',
                xlabel=r'$x$ (nm)', ylabel=r'$\\rho$ (a.u.)',
                xscale='linear', yscale='log',
                xlim=(0, 100), ylim=(1e-3, 1),
                fontsize=12
            )
        """
        if not visible:
            ax.axis('off')
            return
        
        # --- Spines ---
        if isinstance(spines, bool):
            for sp in ['top', 'right', 'bottom', 'left']:
                ax.spines[sp].set_visible(spines)
        elif spines == 'minimal':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        elif spines == 'left':
            for sp in ['top', 'right', 'bottom']:
                ax.spines[sp].set_visible(False)
        elif spines == 'bottom':
            for sp in ['top', 'right', 'left']:
                ax.spines[sp].set_visible(False)
        elif isinstance(spines, dict):
            for sp, vis in spines.items():
                ax.spines[sp].set_visible(vis)
        
        # --- Ticks ---
        def _parse_bool_dict(val, default=True):
            if isinstance(val, bool):
                return {'x': val, 'y': val}
            elif val == 'x':
                return {'x': True, 'y': False}
            elif val == 'y':
                return {'x': False, 'y': True}
            elif isinstance(val, dict):
                return {'x': val.get('x', default), 'y': val.get('y', default)}
            return {'x': default, 'y': default}
        
        ticks_cfg = _parse_bool_dict(ticks)
        labels_cfg = _parse_bool_dict(tick_labels)
        
        ax.tick_params(axis='x', which='both', bottom=ticks_cfg['x'], top=ticks_cfg['x'], labelbottom=labels_cfg['x'])
        ax.tick_params(axis='y', which='both', left=ticks_cfg['y'], right=ticks_cfg['y'], labelleft=labels_cfg['y'])
        
        # --- Labels ---
        fs = fontsize or plt.rcParams.get('axes.labelsize', 10)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=fs, **{k: v for k, v in kwargs.items() if 'x' in k.lower() or k == 'labelpad'})
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fs, **{k: v for k, v in kwargs.items() if 'y' in k.lower() or k == 'labelpad'})
        if title is not None:
            ax.set_title(title, fontsize=fs)
        
        # --- Scale & Limits ---
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    
    @staticmethod
    def disable_axis(ax, which: str = 'both'):
        """
        Disable axis components for clean images/heatmaps.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to modify.
        which : str, default='both'
            What to disable:
            - 'both': Disable x and y (full axis off)
            - 'x': Disable x-axis only
            - 'y': Disable y-axis only
            - 'labels': Keep ticks but hide labels
            - 'ticks': Keep labels but hide ticks
            - 'spines': Hide all spines
        
        Examples
        --------
        >>> Plotter.disable_axis(ax)  # Completely clean
        >>> Plotter.disable_axis(ax, 'x')  # Keep y-axis
        >>> Plotter.disable_axis(ax, 'labels')  # Keep ticks, no labels
        """
        if which == 'both':
            ax.axis('off')
        elif which == 'x':
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
        elif which == 'y':
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
        elif which == 'labels':
            ax.tick_params(labelbottom=False, labelleft=False)
        elif which == 'ticks':
            ax.tick_params(bottom=False, left=False, top=False, right=False)
        elif which == 'spines':
            for sp in ax.spines.values():
                sp.set_visible(False)
    
    @staticmethod
    def get_grid_ax(nrows           : int, 
                    ncols           : int,
                    wspace          : float = None,
                    hspace          : float = None,
                    width_ratios    : List[float] = None,
                    height_ratios   : List[float] = None,
                    ax_sub          = None,
                    **kwargs) -> Tuple[GridSpec, list]:
        """
        Get a GridSpec and an empty list for axes (convenience wrapper).
        
        Parameters
        ----------
        nrows, ncols : int
            Grid dimensions.
        wspace, hspace : float, optional
            Spacing between subplots.
        width_ratios, height_ratios : list, optional
            Relative sizes.
        ax_sub : SubplotSpec, optional
            For nested grids.
        **kwargs
            Additional GridSpec arguments.
        
        Returns
        -------
        tuple
            (GridSpec, empty_axes_list)
        
        Examples
        --------
        >>> gs, axes = Plotter.get_grid_ax(2, 3, wspace=0.3)
        >>> for i in range(6):
        ...     Plotter.app_grid_subplot(axes, gs, fig, i)
        """
        return Plotter.get_grid(nrows, ncols, wspace, hspace, width_ratios, height_ratios, ax_sub, **kwargs), []

    @staticmethod
    def app_grid_subplot(axes: list, gs, fig, index: int, sharex=None, sharey=None, **kwargs):
        """
        Append a subplot to an axes list (convenience method).
        
        Parameters
        ----------
        axes : list
            List to append the new axis to.
        gs : GridSpec
            The GridSpec.
        fig : Figure
            The figure.
        index : int
            Grid index.
        sharex, sharey : Axes, optional
            Share axes with another subplot.
        **kwargs
            Additional arguments.
        
        Examples
        --------
        >>> gs, axes    = Plotter.get_grid_ax(2, 2)
        >>> fig         = plt.figure()
        >>> for i in range(4):
        ...     Plotter.app_grid_subplot(axes, gs, fig, i)
        >>> # axes is now [ax0, ax1, ax2, ax3]
        """
        axes.append(Plotter.get_grid_subplot(gs, fig, index, sharex=sharex, sharey=sharey, **kwargs))
    
    #################### T W I N  A X I S ####################
    
    @staticmethod
    def twin_axis(ax, which='y', label='', color='C1', scale='linear', lim=None, fontsize=None, labelpad=0, **kwargs):
        """
        Create a twin axis with a secondary scale.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Primary axis.
        which : str, default='y'
            Which axis to twin: 'y' creates twinx(), 'x' creates twiny().
        label : str, default=''
            Label for the secondary axis.
        color : str, default='C1'
            Color for the secondary axis (spine, ticks, label).
        scale : str, default='linear'
            Scale for secondary axis: 'linear' or 'log'.
        lim : tuple, optional
            Limits for the secondary axis.
        fontsize : int, optional
            Font size for the label.
        labelpad : float, default=0
            Padding for the label.
        **kwargs
            Additional arguments passed to set_ylabel/set_xlabel.
        
        Returns
        -------
        ax2 : matplotlib.axes.Axes
            The secondary axis.
        
        Examples
        --------
        >>> ax2 = Plotter.twin_axis(ax, which='y', label='Temperature (K)', color='red')
        >>> Plotter.plot(ax2, x, temperature, color='red')
        """
        if which == 'y':
            ax2 = ax.twinx()
            ax2.set_ylabel(label, color=color, fontsize=fontsize, labelpad=labelpad, **kwargs)
            ax2.set_yscale(scale)
            
            if lim is not None:
                ax2.set_ylim(lim)
            
            ax2.tick_params(axis='y', labelcolor=color, colors=color)
            ax2.spines['right'].set_color(color)

        else: # which == 'x'
            ax2 = ax.twiny()
            ax2.set_xlabel(label, color=color, fontsize=fontsize, labelpad=labelpad, **kwargs)
            ax2.set_xscale(scale)
            
            if lim is not None:
                ax2.set_xlim(lim)
            
            ax2.tick_params(axis='x', labelcolor=color, colors=color)
            ax2.spines['top'].set_color(color)
        
        return ax2
    
    #################### P O W E R  L A W ####################
    
    @staticmethod
    def power_law_guide(ax, x_range, exponent, *, add_label: bool = True, label=None, position='lower right', color='gray', ls='--', lw=1.5, offset_log=0, zorder=3, **kwargs):
        r"""
        Add a power-law guide line to a log-log plot.
        
        Useful for showing scaling behavior (e.g., y ~ x^{-2}).
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis with log-log scale.
        x_range : tuple
            (x_start, x_end) for the guide line.
        exponent : float
            Power-law exponent (slope in log-log).
        label : str, optional
            Label (e.g., r'$\\sim N^{-2}$'). If None, auto-generates.
        position : str, default='lower right'
            Where to anchor the line: 'lower right', 'upper left', etc.
        color : str, default='gray'
            Line color.
        ls : str, default='--'
            Line style.
        lw : float, default=1.5
            Line width.
        offset_log : float, default=0
            Vertical offset in log10 units.
        **kwargs
            Additional arguments passed to ax.plot.
        
        Returns
        -------
        line : Line2D
            The guide line object.
        
        Examples
        --------
        >>> # Show y ~ x^{-2} scaling
        >>> Plotter.power_law_guide(ax, (10, 1000), -2, label=r'$\\sim N^{-2}$')
        """
        x0, x1      = x_range
        x           = np.array([x0, x1])
        
        # Determine y values based on position
        ylim        = ax.get_ylim()
        y_log_range = np.log10(ylim[1]) - np.log10(ylim[0])
        
        if 'lower' in position:
            y_anchor = ylim[0] * 10**(0.2 * y_log_range)
        else:  # upper
            y_anchor = ylim[1] * 10**(-0.3 * y_log_range)
        
        # Power law: y = A * x^exponent
        # At x0: y0 = A * x0^exponent => A = y0 / x0^exponent
        if 'right' in position:
            y0  = y_anchor * (x1/x0)**(-exponent)  # Anchor at x1
            A   = y0 / (x0**exponent)
        else:  # left
            A   = y_anchor / (x0**exponent)
        
        y = A * x**exponent * 10**offset_log
        
        if label is None and add_label:
            if exponent == int(exponent):
                label = rf'$\sim x^{{{int(exponent)}}}$'
            else:
                label = rf'$\sim x^{{{exponent:.1f}}}$'
        
        line, = ax.plot(x, y, color=color, ls=ls, lw=lw, label=label, zorder=zorder, **kwargs)
        return line
    
    #################### I N S E T ####################

    @staticmethod
    def get_inset(ax, 
                position    = [0.0, 0.0, 1.0, 1.0], 
                add_box     = False, 
                box_alpha   = 0.5, 
                box_ext     = 0.02,
                facecolor   = 'white',
                zorder      = 1,
                **kwargs):
        """
        Create an inset axis within the given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The parent axis.
        position : list
            [x0, y0, width, height] for the inset axis in relative coordinates.
        add_box : bool, default=False
            Whether to add a semi-transparent box around the inset.
        box_alpha : float, default=0.5
            Transparency of the box.
        box_ext : float, default=0.02
            Extension of the box beyond the inset axis.
        facecolor : str, default='white'
            Face color of the box.
        zorder : int, default=1
            Z-order of the inset axis.
        **kwargs
            Additional arguments passed to fig.add_axes.

        Returns:
        - ax2: The inset axis.
        """
        # Create the inset axis
        bbox    = ax.get_position()
        fig     = ax.figure
        inset_position = [
            bbox.x0 + position[0] * bbox.width,
            bbox.y0 + position[1] * bbox.height,
            position[2] * bbox.width,
            position[3] * bbox.height,
        ]
        ax2 = fig.add_axes(inset_position, **kwargs, zorder=zorder)

        if add_box:
            # Add a semi-transparent white box around the inset
            rect = Rectangle((-box_ext, -box_ext), 1 + 2 * box_ext, 1 + 2 * box_ext, 
                             transform=ax2.transAxes, facecolor=facecolor, 
                             edgecolor='none', alpha=box_alpha, zorder=zorder)
            ax2.add_patch(rect)

        return ax2
    
    ##################### L O O K #####################
    
    @staticmethod
    def set_transparency(ax, alpha = 0.0):
        ax.patch.set_alpha(alpha)
    
    ################### L E G E N D ###################
    
    @staticmethod
    def set_legend(ax,
                fontsize                    = None,
                frameon         : bool      = False,
                loc             : str       = 'best',
                alignment       : str       = 'left',
                markerfirst     : bool      = False,
                framealpha      : float     = 1.0,
                reverse         : bool      = False,
                style                       = None,
                labelspacing    : float     = 0.5,
                handlelength    : float     = 1.5,
                handletextpad   : float     = 0.4,
                borderpad       : float     = 0.4,
                columnspacing   : float     = 1.0,
                ncol            : int       = 1,
                **kwargs):
        '''
        Sets the legend with a preferred style for publication-quality plots.

        Parameters:
        - ax              : Axis to which the legend will be added.
        - fontsize        : Font size of the legend labels.
        - frameon         : Whether to draw a frame around the legend.
        - loc             : Location of the legend ('best', 'upper right', etc.).
        - alignment       : Text alignment ('left', 'center', 'right').
        - markerfirst     : Whether the marker or label appears first in the legend.
        - framealpha      : Transparency of the legend frame (1.0 is opaque).
        - reverse         : Reverse the order of legend items.
        - style           : Predefined style for the legend ('minimal', 'boxed', etc.).
        - labelspacing    : Vertical space between legend entries.
        - handlelength    : Length of the legend markers.
        - handletextpad   : Space between legend markers and text.
        - borderpad       : Padding inside the legend box.
        - columnspacing   : Spacing between legend columns.
        - ncol            : Number of columns in the legend.
        - kwargs          : Additional arguments passed to `ax.legend()`.
        '''
        ax = Plotter.ax(ax)
        if isinstance(ax, IgnoredAxis):
            return None

        # Get legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        if reverse:
            handles = handles[::-1]
            labels = labels[::-1]

        # Apply predefined styles
        if style == 'minimal':
            frameon = False
            fontsize = fontsize or 10
            loc = loc or 'best'
        elif style == 'boxed':
            frameon = True
            framealpha = 0.9
            loc = loc or 'upper right'
        elif style == 'publication':
            frameon = True
            framealpha = 1.0
            loc = loc or 'best'
            fontsize = fontsize or 12
            handlelength = 1.0
            handletextpad = 0.5
            labelspacing = 0.4
            columnspacing = 1.2
            ncol = 1

        # Set legend
        legend = ax.legend(
            handles, 
            labels,
            fontsize=fontsize,
            frameon=frameon,
            loc=loc,
            markerfirst=markerfirst,
            framealpha=framealpha,
            labelspacing=labelspacing,
            handlelength=handlelength,
            handletextpad=handletextpad,
            borderpad=borderpad,
            columnspacing=columnspacing,
            ncol=ncol,
            **kwargs
        )

        # Adjust alignment
        if alignment != 'left':
            for text in legend.get_texts():
                text.set_horizontalalignment(alignment)

    @staticmethod
    def set_legend_custom(  ax,
                            conditions  : list,
                            fontsize    = None,
                            frameon     = False,
                            loc         = 'best',
                            alignment   = 'left',
                            markerfirst = False,
                            framealpha  = 1.0,
                            reverse     = False,
                            **kwargs):
        '''
        Set the legend with custom conditions for the labels
        - ax        :   axis to use
        - conditions:   list of conditions
        - fontsize  :   fontsize
        - frameon   :   frame on or off
        - loc       :   location of the legend
        - alignment :   alignment of the legend
        - markerfirst:  marker first or not
        - framealpha:   alpha of the frame
        '''
        ax = Plotter.ax(ax)
        if isinstance(ax, IgnoredAxis):
            return None

        lines, labels   = ax.get_legend_handles_labels()

        # go through the conditions'
        lbl_idx         = []
        for idx in range(len(labels)):
            for cond in conditions:
                if cond(labels[idx]):
                    lbl_idx.append(idx)
                    break
        
        # set the legend
        ax.legend([lines[idx] for idx in lbl_idx], 
                  [labels[idx] for idx in lbl_idx],
                  fontsize      = fontsize, 
                  frameon       = frameon, 
                  loc           = loc,
                  markerfirst   = markerfirst,
                  framealpha    = framealpha,
                  **kwargs)

    ######### S U B A X S #########

    @staticmethod
    def get_subplots(   nrows           =   1,
                        ncols           =   1,
                        sizex           =   10.,                    # total width [in] OR list of per-col ratios
                        sizey           =   10.,                    # total height [in] OR list of per-row ratios
                        sizex_def       =   3,                      # inches per unit of sizex ratio (if sizex is a sequence)
                        sizey_def       =   3,                      # inches per unit of sizey ratio (if sizey is a sequence)
                        annot_x_pos     =   None,                   # position for annotation - x
                        annot_y_pos     =   None,                   # position for annotation - y
                        panel_labels    =   False,
                        single_if_1     =   False,                  # if True, return ax instead of [ax] when nrows=ncols=1
                        share_x         =   False,
                        share_y         =   False,
                        width_ratios    =   None,                   # list of relative widths for columns (overrides sizex if provided)
                        height_ratios   =   None,                   # list of relative heights for rows (overrides sizey if provided)
                        constrained_layout = None,                  # explicit layout-engine override
                        tight_layout    =   False,                  # call fig.tight_layout() at the end
                        layout          =   None,                   # mpl>=3.6 layout engine, e.g. 'constrained', 'tight'
                        mosaic          =   None,                   # subplot_mosaic specification
                        spans           =   None,                   # dict[name] -> span spec on nrows x ncols grid
                        named_panels    =   None,                   # names for regular grid panels
                        **kwargs) -> Tuple[plt.Figure, AxesList]:
        """
        Create subplot layouts and return a list-like ``AxesList`` wrapper.

        Parameters
        ----------
        nrows, ncols : int, default=(1, 1)
            Grid shape used for regular subplot creation and for ``spans``.
        sizex, sizey : float or sequence, default=10.0
            Figure width/height in inches, or ratio sequences per column/row.
        sizex_def, sizey_def : float, default=3
            Inch scaling used when ``sizex``/``sizey`` are ratio sequences.
        annot_x_pos, annot_y_pos : float or sequence, optional
            Position(s) for panel label annotations in axes-fraction units.
        panel_labels : bool or sequence, default=False
            If truthy, annotate each axis with labels (auto or user-provided).
        single_if_1 : bool, default=False
            If True and only one axis is created, return that axis instead of
            an ``AxesList``.
        share_x, share_y : bool, default=False
            Share x/y axes across created panels.
        width_ratios, height_ratios : sequence, optional
            GridSpec ratios overriding ratios inferred from ``sizex``/``sizey``.
        constrained_layout : bool, optional
            Explicitly control constrained layout engine.
        tight_layout : bool, default=False
            Call ``fig.tight_layout()`` after creation (when compatible).
        layout : str, optional
            Matplotlib layout engine name (e.g. ``'constrained'``, ``'tight'``).
        mosaic : subplot-mosaic spec, optional
            Use ``plt.subplot_mosaic`` with named panels.
        spans : dict, optional
            Named span panels on a regular grid.
            Example: ``{'main': (0, 2, 0, 3), 'side': (0, 2, 3, 4)}``.
        named_panels : sequence or dict, optional
            Panel aliases for regular grids or mosaic alias remapping.
        **kwargs : dict
            Forwarded Matplotlib options. Common keys include:
            ``dpi``, ``subplot_kw``, ``gridspec_kw``, ``hspace``, ``wspace``,
            ``left/right/top/bottom``, ``grid``, ``grid_kws``, ``despine``,
            ``axis_off``, ``suptitle``, ``suptitle_kws``, ``post_hook``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        axes : AxesList or matplotlib.axes.Axes
            ``AxesList`` wrapper (list-compatible, grid-aware, named-panel
            access). Returns single axis only when ``single_if_1=True``.

        Notes
        -----
        ``AxesList`` supports:
        - list operations (iterate, append-like access, slicing)
        - grid indexing: ``axes[row, col]``
        - named access: ``axes['main']``
        - helpers: ``row()``, ``col()``, ``span()``, ``select()``, ``apply()``

        Examples
        --------
        Standard grid:
        ``fig, axes = Plotter.get_subplots(2, 3, sizex=9, sizey=5)``
        ``axes[1, 2].plot(x, y)``

        Named aliases:
        ``fig, axes = Plotter.get_subplots(1, 3, named_panels=['left', 'mid', 'right'])``
        ``axes['mid'].set_title('Center')``

        Mosaic:
        ``fig, axes = Plotter.get_subplots(mosaic=[['A', 'A', 'B'], ['C', 'D', 'D']])``
        ``axes['A'].plot(x, y)``

        Spans:
        ``fig, axes = Plotter.get_subplots(nrows=3, ncols=4, spans={'main': (0, 2, 0, 3), 'side': (0, 2, 3, 4), 'bottom': (2, 3, 0, 4)})``
        ``axes['main'].plot(x, y)``
        """

        #! Extract utility kwargs (do not pass to plt.subplots)
        gridspec_kw     = dict(kwargs.pop('gridspec_kw', {}) or {})
        subplot_kw      = dict(kwargs.pop('subplot_kw', {}) or {})
        panel_labels    = kwargs.pop('panel_labels', None)         # True or list/tuple of labels
        grid_on         = kwargs.pop('grid', False)
        grid_kws        = dict(kwargs.pop('grid_kws', {}) or {})
        despine         = kwargs.pop('despine', False)
        axis_off        = kwargs.pop('axis_off', False)
        suptitle        = kwargs.pop('suptitle', None)
        suptitle_kws    = dict(kwargs.pop('suptitle_kws', {}) or {})
        post_hook       = kwargs.pop('post_hook', None)
        width_ratios    = kwargs.pop('width_ratios', None)
        height_ratios   = kwargs.pop('height_ratios', None)
        share_x         = kwargs.pop('sharex', share_x)
        share_y         = kwargs.pop('sharey', share_y)
        wspace          = kwargs.pop('wspace', None)
        hspace          = kwargs.pop('hspace', None)
        layout          = kwargs.pop('layout', layout)
        constrained_layout = kwargs.pop('constrained_layout', constrained_layout)
        tight_layout    = bool(kwargs.pop('tight_layout', tight_layout))
        mosaic          = kwargs.pop('mosaic', mosaic)
        spans           = kwargs.pop('spans', spans)
        named_panels    = kwargs.pop('named_panels', named_panels)

        if layout is not None:
            kwargs['layout'] = layout
        elif constrained_layout is not None:
            kwargs['constrained_layout'] = bool(constrained_layout)

        # Use constrained_layout by default unless user opted for tight_layout or already set it
        if 'layout' not in kwargs and 'constrained_layout' not in kwargs and not tight_layout:
            kwargs['constrained_layout'] = True

        # Map spacing/bounds kwargs into gridspec_kw if provided
        for k in ('hspace', 'wspace', 'left', 'right', 'top', 'bottom'):
            if k in kwargs:
                gridspec_kw[k] = kwargs.pop(k)

        #! Figure size & ratios
        width_ratios_local = None
        height_ratios_local = None
        # sizex can be total inches (number) or per-column *ratios* (sequence)
        if isinstance(sizex, (list, tuple)):
            if len(sizex) != ncols:
                raise ValueError(f"sizex length {len(sizex)} != ncols {ncols}")
            width_ratios_local = list(sizex)
            total_w         = sizex_def * fsum(width_ratios_local)
        else:
            # If set to None, infer from defaults
            total_w         = float(sizex if sizex is not None else sizex_def * ncols)

        # sizey can be total inches (number) or per-row *ratios* (sequence)
        if isinstance(sizey, (list, tuple)):
            if len(sizey) != nrows:
                raise ValueError(f"sizey length {len(sizey)} != nrows {nrows}")
            height_ratios_local = list(sizey)
            total_h         = sizey_def * fsum(height_ratios_local)
        else:
            total_h         = float(sizey if sizey is not None else sizey_def * nrows)

        if width_ratios is not None:
            width_ratios_local = list(width_ratios)
        if height_ratios is not None:
            height_ratios_local = list(height_ratios)

        if width_ratios_local is not None:
            gridspec_kw['width_ratios'] = width_ratios_local
        if height_ratios_local is not None:
            gridspec_kw['height_ratios'] = height_ratios_local

        figsize                             = kwargs.pop('figsize', (total_w, total_h))

        panel_map: Dict[str, Any] = {}

        def _parse_span(spec):
            # Supported:
            # - (r0, r1, c0, c1)
            # - ((r0, r1), (c0, c1))
            # - {'rows': (r0, r1), 'cols': (c0, c1)}
            if isinstance(spec, dict):
                rows = spec.get("rows", spec.get("row"))
                cols = spec.get("cols", spec.get("col"))
                if rows is None or cols is None:
                    raise ValueError(f"Invalid span dict {spec}.")
                if isinstance(rows, int):
                    rows = (rows, rows + 1)
                if isinstance(cols, int):
                    cols = (cols, cols + 1)
                return slice(int(rows[0]), int(rows[1])), slice(int(cols[0]), int(cols[1]))
            if isinstance(spec, tuple) and len(spec) == 4:
                r0, r1, c0, c1 = spec
                return slice(int(r0), int(r1)), slice(int(c0), int(c1))
            if isinstance(spec, tuple) and len(spec) == 2:
                rs, cs = spec
                if isinstance(rs, int):
                    rs = slice(rs, rs + 1)
                if isinstance(cs, int):
                    cs = slice(cs, cs + 1)
                if isinstance(rs, slice) and isinstance(cs, slice):
                    return rs, cs
            raise ValueError(f"Unsupported span spec: {spec}")

        #! Create
        if mosaic is not None:
            fig, axd = plt.subplot_mosaic(
                mosaic,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
                subplot_kw=subplot_kw,
                sharex=share_x,
                sharey=share_y,
                **kwargs,
            )
            panel_map = {str(k): v for k, v in axd.items()}
            if isinstance(named_panels, dict):
                for alias, target in named_panels.items():
                    if isinstance(target, str) and target in panel_map:
                        panel_map[str(alias)] = panel_map[target]
            # Keep insertion order and unique axes
            axes_flat = list(dict.fromkeys(panel_map.values()))
            axes_list = AxesList(axes_flat, nrows=None, ncols=None, panel_map=panel_map)
        elif spans is not None:
            fig = plt.figure(figsize=figsize, **kwargs)
            gs = fig.add_gridspec(nrows=nrows, ncols=ncols, **gridspec_kw)
            sharex_ref = None
            sharey_ref = None
            for name, spec in dict(spans).items():
                rs, cs = _parse_span(spec)
                add_kw = dict(subplot_kw)
                if bool(share_x) and sharex_ref is not None:
                    add_kw["sharex"] = sharex_ref
                if bool(share_y) and sharey_ref is not None:
                    add_kw["sharey"] = sharey_ref
                ax_one = fig.add_subplot(gs[rs, cs], **add_kw)
                if sharex_ref is None:
                    sharex_ref = ax_one
                if sharey_ref is None:
                    sharey_ref = ax_one
                panel_map[str(name)] = ax_one
            axes_flat = list(dict.fromkeys(panel_map.values()))
            axes_list = AxesList(axes_flat, nrows=nrows, ncols=ncols, panel_map=panel_map)
        else:
            fig, ax = plt.subplots(
                        nrows=nrows,
                        ncols=ncols,
                        figsize=figsize,
                        gridspec_kw=gridspec_kw,
                        subplot_kw=subplot_kw,
                        sharex=share_x,
                        sharey=share_y,
                        **kwargs,
                    )

            #! Normalize axes to a flat list, regardless of (1,1), (1,N), (M,1), (M,N)
            if isinstance(ax, (list, tuple)):
                axes_flat = list(ax)
            else:
                try:
                    axes_flat = ax.ravel().tolist()     # ndarray of Axes
                except Exception:
                    axes_flat = [ax]                    # single Axes
            axes_list = AxesList(axes_flat, nrows=nrows, ncols=ncols)

            # Optional name aliases for standard grids
            if named_panels is not None:
                if isinstance(named_panels, (list, tuple)):
                    if len(named_panels) != len(axes_list):
                        raise ValueError("named_panels list length must match number of axes.")
                    panel_map.update({str(name): axes_list[idx] for idx, name in enumerate(named_panels)})
                elif isinstance(named_panels, dict):
                    for name, spec in named_panels.items():
                        if isinstance(spec, int):
                            panel_map[str(name)] = axes_list[spec]
                        elif isinstance(spec, tuple) and len(spec) == 2:
                            panel_map[str(name)] = axes_list[spec[0], spec[1]]
                        else:
                            raise ValueError(f"Unsupported named panel selector: {spec}")
                else:
                    raise ValueError("named_panels must be list/tuple or dict.")
                axes_list._panel_map = panel_map

        #! Optional cosmetics
        if grid_on:
            for a in axes_list:
                a.grid(True, **grid_kws)

        if despine:
            for a in axes_list:
                for side in ('top', 'right'):
                    a.spines[side].set_visible(False)

        if axis_off:
            for a in axes_list:
                a.axis('off')

        if tight_layout and not kwargs.get('constrained_layout', False) and kwargs.get('layout', None) != 'constrained':
            fig.tight_layout()

        #! Panel labels
        if panel_labels is not None and (panel_labels != False):
            if panel_labels is True:
                labels = [f'({str(chr(97+i))})' for i in range(len(axes_list))]   # a, b, c, ...
            else:
                labels = list(panel_labels)
                if len(labels) != len(axes_list):
                    raise ValueError("panel_labels length must match number of axes")
            
            if annot_x_pos is None:
                annot_x_pos = [0.97] * len(axes_list)
            elif isinstance(annot_x_pos, (int, float)):
                annot_x_pos = [annot_x_pos] * len(axes_list)
            elif isinstance(annot_x_pos, (list, tuple)) and len(annot_x_pos) == len(axes_list):
                annot_x_pos = list(annot_x_pos)
            else:
                raise ValueError("annot_x_pos must be None, a number, or a list/tuple of the same length as axes_list")

            if annot_y_pos is None:
                annot_y_pos = [0.03] * len(axes_list)
            elif isinstance(annot_y_pos, (int, float)):
                annot_y_pos = [annot_y_pos] * len(axes_list)
            elif isinstance(annot_y_pos, (list, tuple)) and len(annot_y_pos) == len(axes_list):
                annot_y_pos = list(annot_y_pos)
            else:
                raise ValueError("annot_y_pos must be None, a number, or a list/tuple of the same length as axes_list")

            for i, (lbl, a) in enumerate(zip(labels, axes_list)):
                if annot_x_pos[i] is not None and annot_y_pos[i] is not None:
                    a.annotate(lbl, xy=(annot_x_pos[i], annot_y_pos[i]), ha='left', va='top', fontweight='bold', fontsize='large', xycoords='axes fraction', zorder = 1000)

        if suptitle is not None:
            fig.suptitle(suptitle, **suptitle_kws)

        #! Final user hook (receives fig and the flat list of axes)
        if callable(post_hook):
            post_hook(fig, axes_list)
        
        return fig, axes_list if not (single_if_1 and len(axes_list) == 1) else axes_list[0]

    @staticmethod
    def subplots(*args, **kwargs):
        """
        Alias of :meth:`Plotter.get_subplots`.
        """
        return Plotter.get_subplots(*args, **kwargs)

    @staticmethod
    def subplot_mosaic(mosaic, *args, **kwargs):
        """
        Convenience alias for mosaic layouts.

        Equivalent to:
        ``Plotter.get_subplots(mosaic=mosaic, *args, **kwargs)``
        """
        kwargs["mosaic"] = mosaic
        return Plotter.get_subplots(*args, **kwargs)

    ######### S A V I N G #########

    @staticmethod
    def save_fig(directory  :   str,
                filename    :   str,
                format      =   'pdf',
                dpi         =   200,
                adjust      =   True,
                fig         =   None,
                **kwargs):
        '''
        Save figure to a specific directory. 
        - directory : directory to save the file
        - filename  : name of the file
        - format    : format of the file
        - dpi       : dpi of the file
        - adjust    : adjust the figure
        '''
        if fig is None:
            fig = plt.gcf()
            
        if adjust:
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(directory + "/" + filename, format = format, dpi = dpi, bbox_inches = 'tight', **kwargs)
    
    # alias
    @staticmethod
    def savefig(directory, filename, format, dpi, adjust, fig = None, **kwargs):
        return Plotter.save_fig(directory, filename, format=format, dpi=dpi, adjust=adjust, fig=fig, **kwargs)
    
    ###############################
    
    @staticmethod
    def plot_heatmaps(dfs       : list,
                      colormap  = 'viridis',
                      cb_width  = 0.1,
                      movefirst = True,
                      index     = None,
                      columns   = None,
                      values    = None,
                      sortidx   = True,  
                      zlabel    = '',
                      sizemult  = 3,
                      xvals     = True,
                      yvals     = True,
                      vmin      = None,
                      vmax      = None,
                      **kwargs):
        import seaborn as sns
        
        size        = len(dfs)
        plotwidth   = 1.0 / len(dfs)
        cbarwidth   = 0.1 * plotwidth
        normalwidth = plotwidth - cbarwidth / size
        extendwidth = plotwidth + cbarwidth
        fig, ax     = Plotter.get_subplots(nrows = 1,
                                           ncols = size, 
                                           sizex = size * sizemult, 
                                           sizey = sizemult,
                                           dpi   = 150,
                                           width_ratios = (size - 1) * [normalwidth] + [extendwidth],
                                           **kwargs)
        dfsorted    = [df.sort_values(by = [index, columns]) for df in dfs] if sortidx else dfs
        dfpivoted   = [df.pivot_table(index = index, columns = columns, values = values) for df in dfsorted]
        heatmap     = [df.sort_index(ascending = True) for i, df in enumerate(dfpivoted)]
        if movefirst:
            for i in range(size):
                heatmap[i].index += heatmap[i].index[1] - heatmap[i].index[0]
        plots       = []
        
        for i, df in enumerate(heatmap):
            plot = sns.heatmap(df, ax = ax[i], cbar = i == size - 1, 
                               cbar_kws = {
                                            # 'shrink': cb_width,
                                            'label' : zlabel
                                        }, 
                               vmin = vmin, vmax = vmax,
                               cmap = colormap)
            plots.append(plot)
        
        # set axes
        for i, axis in enumerate(plots):
            # ax.set_xlabel('')
            # ax.set_ylabel('')
            Plotter.set_tickparams(axis)
            axis.set_title(f'{i + 1}')
            if xvals:
                axis.set_xticks(range(len(heatmap[i].columns)))                                                         # Set positions for x ticks
                axis.set_xticklabels([f"{col:.2f}" if k % 2 == 0 else '' for k, col in enumerate(heatmap[i].columns)])  # Set x labels
            # y ticks
            if yvals:
                axis.set_yticks(range(len(heatmap[i].index)))                                                           # Set positions for y ticks
                axis.set_yticklabels([f"{ind:.2f}" if k % 2 == 0 else '' for k, ind in enumerate(heatmap[i].index)])    # Set y labels
        
        return fig, ax, plots

##########################################################################

class PlotterSave:
    
#################################################

    @staticmethod
    def dict2json(  directory    :   str,
                    fileName     :   str,
                    data):
        '''
        Save dictionary to json file
        - directory : directory to save the file
        - fileName  : name of the file
        - data      : dictionary to save
        '''
        with open(directory + fileName + '.json', 'w') as fp:
            json.dump(data, fp)

#################################################

    @staticmethod
    def json2dict(  directory    :   str,
                    fileName     :   str) -> dict:
        '''
        Load dictionary from json file
        '''
        dict2load = {}
        with open(directory + f"{fileName}.json", "r") as readfile:
            dict2load = json.loads(readfile.read())
        return dict2load
    
#################################################

    @staticmethod
    def json2dict_multiple(directory    : str,
                        keys            : list):
        '''
        Based on the specified keys, load the dictionaries from the json files
        The keys are the names of the files as well!
        '''
        # create the dictionary
        data2plot = {}
        
        # create empty dictionaries
        for key in keys:
            data2plot[key] = {}
            
        for f_op in data2plot.keys():
            data2plot[f_op] = PlotterSave.json2dict(directory, f_op)
        
        return data2plot
    
#################################################

    @staticmethod
    def singleColumnData(   directory    :   str,
                            fileName     :   str,
                            y,
                            typ          =   '.npy'):
        '''
        Stores the values as a single vector
        '''
        toSave          = np.array(y)
        
        if typ == '.npy':
            np.save(directory + fileName + ".npy", toSave)
        elif typ == '.txt':
            np.savetxt(directory + fileName + ".npy", toSave)
#################################################

    @staticmethod
    def twoColumnsData(     directory    :   str,
                            fileName     :   str,
                            x,
                            y,
                            typ          =   '.npy'):
        '''
        Stores the x, y vectors in 2D form (multiple rows and two columns)
        '''
        if len(x) != len(y):
            raise Exception("Sizes incompatible.")
        
        toSave          = np.zeros((len(x), 2))
        toSave[:, 0]    = x
        toSave[:, 1]    = y
        
        if typ == '.npy':
            np.save(directory + fileName + typ, toSave)
        elif typ == '.txt':
            np.savetxt(directory + fileName + typ, toSave)
        elif typ == '.dat':
            np.savetxt(directory + fileName + typ, toSave)
            
        
    #################################################   
    @staticmethod
    def matrixData(         directory    :   str,
                            fileName     :   str,   
                            x,
                            y,
                            typ          =   '.npy'):
        '''
        Stores the x, y vectors in matrix form (appending single column
        at start for x values)
        '''
        if len(x) != len(y):
            raise Exception("Sizes incompatible.")
        
        # first element for x
        toSave          = np.zeros((y.shape[0], y.shape[1] + 1))
        toSave[:, 0]    = x
        toSave[:, 1:]   = y
        
        if typ == '.npy':
            np.save(directory + fileName + typ, toSave)
        else:
            np.savetxt(directory + fileName + typ, toSave)

    ##########################################################################
    @staticmethod
    def app_df(df, colname: str, y, fill_value=np.nan):
        """
        Appends the data to the dataframe.
        
        Parameters:
        - df (pd.DataFrame): The dataframe to append data to.
        - colname (str): The column name to append data under.
        - y (array-like): The data to append.
        - fill_value: The value to use for filling if resizing is needed.
        """
        # Ensure the length of y matches the length of the dataframe
        original_len = len(y)
        if original_len != len(df):
            y = np.resize(y, len(df))
            # Fill the rest of the array with fill_value
            if original_len > len(df):
                y[len(df):] = fill_value
            y[len(df):] = fill_value
        
        df[colname] = y
    
    ##########################################################################
    @staticmethod
    def app_array(arr, y):
        """
        Appends the data to a numpy array.
        
        Parameters:
        - arr (np.ndarray): The numpy array to append data to.
        - y (np.ndarray): The data to append.
        
        Returns:
        - np.ndarray: The updated numpy array with appended data.
        """
        return np.append(arr, y, axis=0)

##########################################################################

try:
    from IPython.display import display
    from sympy import Matrix, init_printing
        
    class MatrixPrinter:
        '''
        Class for printing matrices and vectors using IPython and sympy.
        
        This class provides methods for displaying matrices and vectors
        in a nicely formatted way in Jupyter notebooks.
        '''
        
        def __init__(self):
            init_printing()
        
        @staticmethod
        def print_matrix(matrix: np.ndarray):
            '''Prints the matrix in a nice form.'''
            display(Matrix(matrix))
        
        @staticmethod
        def print_vector(vector: np.ndarray):
            '''Prints the vector in a nice form.'''
            display(Matrix(vector))
        
        @staticmethod
        def print_matrices(matrices: list):
            '''Prints a list of matrices in a nice form.'''
            for matrix in matrices:
                display(Matrix(matrix))
        
        @staticmethod
        def print_vectors(vectors: list):
            '''Prints a list of vectors in a nice form.'''
            for vector in vectors:
                display(Matrix(vector))

except ImportError:
    print("IPython is not installed. Matrix display will not work.")

    class MatrixPrinter:
        '''
        Class for printing matrices and vectors
        '''
        
        @staticmethod
        def print_matrix(matrix: np.ndarray):
            '''
            Prints the matrix in a nice form
            '''
            print("Matrix:")
            print(matrix)
        
        @staticmethod
        def print_vector(vector: np.ndarray):
            '''
            Prints the vector in a nice form
            '''
            print("Vector:")
            print(vector)
        
        @staticmethod
        def print_matrices(matrices: list):
            '''
            Prints a list of matrices in a nice form
            '''
            for matrix in matrices:
                MatrixPrinter.print_matrix(matrix)
        
        @staticmethod
        def print_vectors(vectors: list):
            '''
            Prints a list of vectors in a nice form
            '''
            for vector in vectors:
                MatrixPrinter.print_vector(vector)

##########################################################################
#! EOF
##########################################################################
