'''
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
>>> from QES.general_python.common.plot import Plotter
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
import  itertools
import  numpy as np
from    math import fsum
from    typing import Tuple, Union, Optional, List, TYPE_CHECKING

try:
    import scienceplots
except ImportError:
    print("scienceplots not found, some styles may not be available.")

# Matplotlib
import matplotlib               as mpl
import matplotlib.pyplot        as plt
import matplotlib.colors        as mcolors
import matplotlib.ticker        as mticker

# Grids
from matplotlib.colors          import Normalize, ListedColormap
from matplotlib.gridspec        import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches         import Rectangle
from matplotlib.ticker          import FixedLocator, NullFormatter, LogLocator, LogFormatterMathtext
from matplotlib.legend          import Legend
from matplotlib.legend_handler  import HandlerBase, HandlerPatch

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
    style : str, default='publication'
        Preset style to use:
        - 'publication': Nature/Science journal style (compact, clean)
        - 'presentation': Larger fonts for slides
        - 'poster': Very large fonts for posters
        - 'minimal': Bare minimum styling
        - 'default': Reset to matplotlib defaults
    font_size : int, default=10
        Base font size. Other sizes are scaled relative to this.
    use_latex : bool, default=False
        If True, use LaTeX for text rendering (slower but prettier).
    dpi : int, default=150
        Figure resolution for display.
    **overrides
        Additional rcParams to override. E.g., `axes.linewidth=2`.
    
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
                plt.style.use(['science', 'no-latex', 'colors5-light'])
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
        
        # Axes
        'axes.facecolor'        : 'white',
        'axes.edgecolor'        : 'black',
        'axes.labelcolor'       : 'black',
        'axes.axisbelow'        : True,
        'axes.grid'             : False,
        'axes.spines.top'       : True,
        'axes.spines.right'     : True,
        
        # Ticks
        'xtick.direction'       : 'in',
        'ytick.direction'       : 'in',
        'xtick.top'             : True,
        'ytick.right'           : True,
        'xtick.color'           : 'black',
        'ytick.color'           : 'black',
        
        # Grid
        'grid.color'            : '#E0E0E0',
        'grid.linestyle'        : '-',
        'grid.linewidth'        : 0.5,
        'grid.alpha'            : 0.8,
        
        # Lines
        'lines.linewidth'       : 1.5,
        'lines.markersize'      : 5,
        
        # Fonts
        'font.family'           : 'sans-serif' if not use_latex else 'serif',
        'mathtext.fontset'      : 'stix',
        
        # Legend
        'legend.frameon'        : False,
        'legend.framealpha'     : 1.0,
        'legend.edgecolor'      : 'none',
        'legend.fancybox'       : False,
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
            'axes.linewidth'        : 0.8,
            'xtick.major.width'     : 0.8,
            'ytick.major.width'     : 0.8,
            'xtick.minor.width'     : 0.5,
            'ytick.minor.width'     : 0.5,
            'xtick.major.size'      : 4,
            'ytick.major.size'      : 4,
            'xtick.minor.size'      : 2,
            'ytick.minor.size'      : 2,
            'lines.linewidth'       : 1.2,
            'lines.markersize'      : 4,
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
        'axes.linewidth', 'xtick.major.size', 'ytick.major.size',
        'figure.dpi', 'savefig.dpi',
        'axes.spines.top', 'axes.spines.right',
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
except Exception as e:
    print("Error applying style:", e)
    
# Safely set additional rcParams (for compatibility with documentation build systems)
try:
    mpl.rcParams['mathtext.fontset']    = 'stix'
    mpl.rcParams['font.family']         = 'STIXGeneral'
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
linestylesList                      =   ['-', '--', '-.', ':']
linestylesCycle                     =   itertools.cycle(['-', '--', '-.', ':'])
linestylesCycleExtended             =   itertools.cycle(['-', '--', '-.', ':'] + list(ADDITIONAL_LINESTYLES.keys()))
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

#####################################

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
    def __init__(self, decimals=2, symbol='%'):
        """
        Initialize the object with a format string.
        
        Args:
        decimals (int): The number of decimal places to use.
        symbol (str): The symbol to use for percentage.
        """
        super().__init__(decimals=decimals, symbol=symbol)

class MathTextSciFormatter(mticker.Formatter):
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

########################### plotter ###########################

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
    **Plotting Methods**    : plot, scatter, semilogy, semilogx, loglog, errorbar, fill_between, histogram
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
        'linewidth'     : 1.5,
        'markersize'    : 5,
        'capsize'       : 2,
        'fontsize'      : 10,
        'tick_length'   : 4,
        'tick_width'    : 0.8,
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
        help_text = {
            'overview': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PLOTTER - Scientific Plotting Utilities                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Topics (use Plotter.help('topic_name') for details):

'plot'    - Basic plotting: plot, scatter, semilogy, semilogx, loglog,
            errorbar, fill_between, histogram, contourf

'axis'    - Axis setup: set_ax_params, set_tickparams, setup_log_x,
            setup_log_y, set_smart_lim, unset_spines, unset_ticks, unset_all

'color'   - Colors: add_colorbar, get_colormap, discrete_colormap,
            Color cycles: colorsCycle, colorsCyclePlastic, etc.

'layout'  - Subplots: get_subplots, make_grid, GridBuilder, get_inset

'grid'    - Advanced grids: make_grid (full control), GridBuilder (nested)
            width_ratios, height_ratios, wspace, hspace, margins

'legend'  - Legends: set_legend (with style='publication'), set_legend_custom

'annotate'- Annotations: set_annotate, set_annotate_letter, set_arrow

'style'   - configure_style('nature'), configure_style('science'), 'prl', 'aps'

'save'    - Saving: save_fig(directory, filename, format='pdf', dpi=300)

──────────────────────────────────────────────────────────────────────────────
Quick Example:

    fig, axes = Plotter.get_subplots(1, 2, sizex=8, sizey=3)
    Plotter.plot(axes[0], x, y, label='Data', color='C0')
    Plotter.semilogy(axes[1], x, y_log, color='C1')
    Plotter.set_ax_params(axes[0], xlabel='x', ylabel='y', title='Linear')
    Plotter.set_legend(axes[0], style='publication')
    Plotter.save_fig('.', 'my_figure', format='pdf')
──────────────────────────────────────────────────────────────────────────────
''',
            
            'plot': '''
================
PLOTTING METHODS
================

Basic Plots:
    Plotter.plot(ax, x, y, ls='-', lw=2, color='black', label=None, marker=None)
    Plotter.scatter(ax, x, y, s=10, c='blue', marker='o', alpha=1.0, label=None)

Logarithmic Scales:
    Plotter.semilogy(ax, x, y, **kwargs)    # log y-axis
    Plotter.semilogx(ax, x, y, **kwargs)    # log x-axis  
    Plotter.loglog(ax, x, y, **kwargs)      # log both axes

Error Bars:
    Plotter.errorbar(ax, x, y, yerr=None, xerr=None, fmt='o', capsize=2, **kwargs)

Histogram:
    Plotter.histogram(ax, data, bins=50, density=True, alpha=0.7, **kwargs)

Filled Regions:
    Plotter.fill_between(ax, x, y1, y2, color='blue', alpha=0.5)

Reference Lines:
    Plotter.hline(ax, val, ls='--', lw=2, color='black', label=None)
    Plotter.vline(ax, val, ls='--', lw=2, color='black', label=None)

Contour:
    cs = Plotter.contourf(ax, x, y, z, levels=20, cmap='viridis')
''',
            
            'axis': '''
==================
AXIS CONFIGURATION
==================

Comprehensive Setup:
    Plotter.set_ax_params(
        ax,
        xlabel='x', ylabel='y',          # Axis labels
        xlim=(0, 10), ylim=(0, 1),        # Axis limits
        xscale='linear', yscale='log',   # Scale ('linear', 'log')
        fontsize=12,                      # Label font size
        title='My Plot',                  # Plot title
    )

Log-Scale Setup (with proper tick formatting):
    Plotter.setup_log_y(ax, ylims=(1e-6, 1e0), decade_step=2)
    Plotter.setup_log_x(ax, xlims=(1e-3, 1e3), decade_step=2)

Tick Parameters:
    Plotter.set_tickparams(ax, labelsize=10, maj_tick_l=6, min_tick_l=3)

Smart Limits (auto-compute from data):
    Plotter.set_smart_lim(ax, which='y', data=my_data, margin_p=0.1)

──────────────────────────────────────────────────────────────────────────────
REMOVING SPINES & TICKS (Publication Style)
──────────────────────────────────────────────────────────────────────────────

Remove Spines:
    # Remove specific spines (True = REMOVE, False = KEEP)
    Plotter.unset_spines(ax, top=True, right=True, bottom=False, left=False)
    
    # Nature-style (only left and bottom spines)
    Plotter.unset_spines(ax, top=True, right=True)
    
    # Remove all spines (frameless plot)
    Plotter.unset_spines(ax, top=True, right=True, bottom=True, left=True)

Remove Tick Labels (keep spines):
    # Hide x-tick labels (useful for stacked plots)
    Plotter.unset_ticks(ax, xticks=True, yticks=False)
    
    # Hide all tick labels
    Plotter.unset_ticks(ax, xticks=True, yticks=True)
    
    # Also remove axis labels
    Plotter.unset_ticks(ax, xticks=True, xlabel=True)

Scientific Notation:
    set_formatter(ax, formatter_type='sci', fmt='%1.1e', axis='y')

──────────────────────────────────────────────────────────────────────────────
QUICK REFERENCE - Common Axis Operations
──────────────────────────────────────────────────────────────────────────────

# Shared x-axis in stacked plots (hide middle ticks)
for ax in axes[:-1]:
    Plotter.unset_ticks(ax, xticks=True, xlabel=True)

# Clean Nature-style axes
for ax in axes:
    Plotter.unset_spines(ax, top=True, right=True)

# Log scale with nice decade ticks
Plotter.setup_log_y(ax, ylims=(1e-6, 1e0), decade_step=1)
''',
            
            'color': '''
==================
COLORS & COLORBARS
==================

Add Colorbar:
    cbar, cax = Plotter.add_colorbar(
        fig, pos=[0.92, 0.15, 0.02, 0.7],  # [left, bottom, width, height]
        mappable=data,                      # or ScalarMappable
        cmap='viridis',
        scale='log',                        # 'linear', 'log', 'symlog'
        label=r'$|\psi|^2$',
        orientation='vertical',
    )

Get Colormap Function:
    getcolor, cmap, norm = Plotter.get_colormap(values, cmap='coolwarm')
    color = getcolor(value)  # Returns RGBA for value

Discrete Colormap:
    cmap = Plotter.discrete_colormap(N=5, base_cmap='viridis')

Available Color Cycles (use next(cycle) to get colors):
    colorsCycle         - Tableau colors (default)
    colorsCyclePlastic  - Colorblind-safe palette
    colorsCycleBright   - CSS4 colors
    colorsCyclePastel   - XKCD colors

Reset Cycles:
    reset_color_cycles()  # Reset all
    reset_color_cycles('Plastic')  # Reset specific
''',
            
            'layout': '''
=================
LAYOUT & SUBPLOTS
=================

Simple Subplots (recommended for quick plots):
    fig, axes = Plotter.get_subplots(
        nrows=2, ncols=3,
        sizex=10, sizey=6,                # Total figure size in inches
        panel_labels=True,                # Add (a), (b), (c) labels
        despine=True,                     # Remove top/right spines
        constrained_layout=True,          # Auto-adjust spacing
    )
    # axes is always a flat list: [ax0, ax1, ax2, ...]

──────────────────────────────────────────────────────────────────────────────
ADVANCED GRIDS - Full Control Over Layout
──────────────────────────────────────────────────────────────────────────────

Plotter.make_grid() - Static Method for Complex Layouts:
    fig, axes = Plotter.make_grid(
        nrows=2, ncols=3,
        figsize=(10, 6),
        
        # Column/Row Sizing
        width_ratios=[1, 2, 1],           # Column width proportions
        height_ratios=[1, 3],              # Row height proportions
        
        # Spacing Control
        wspace=0.3,                        # Horizontal gap (0-1)
        hspace=0.2,                        # Vertical gap (0-1)
        left=0.1, right=0.95,              # Figure margins
        top=0.95, bottom=0.1,
        
        # Per-Axis Options
        sharex='col',                      # 'row', 'col', 'all', False
        sharey='row',
        
        # Publication Options
        panel_labels=True,                 # Add (a), (b), (c)
        panel_label_style='parenthesis',   # 'parenthesis', 'plain', 'bold'
        despine=True,
    )

──────────────────────────────────────────────────────────────────────────────
GridBuilder - Class for Maximum Flexibility
──────────────────────────────────────────────────────────────────────────────

For nested/complex layouts, use the GridBuilder class:

    builder = Plotter.GridBuilder(figsize=(12, 8))
    
    # Add a full-width row at top
    builder.add_row(ncols=1, height_ratio=1)
    
    # Add 3-column row with custom width ratios
    builder.add_row(ncols=3, height_ratio=2, width_ratios=[2, 1, 1])
    
    # Add 2-column row
    builder.add_row(ncols=2, height_ratio=1.5)
    
    # Build the figure
    fig, axes = builder.build()
    # axes is a list of lists: [[ax00], [ax10, ax11, ax12], [ax20, ax21]]

GridBuilder supports:
    .add_row(ncols, height_ratio=1, width_ratios=None)  # Add horizontal row
    .add_column(nrows, width_ratio=1, height_ratios=None)  # Add vertical column
    .add_subplot(rows=1, cols=1)  # Add to current position (call multiple times)
    .build(wspace=0.2, hspace=0.2, **margins)  # Finalize

──────────────────────────────────────────────────────────────────────────────
COMMON LAYOUT RECIPES
──────────────────────────────────────────────────────────────────────────────

# Recipe 1: Two-column figure with shared x-axis
fig, axes = Plotter.make_grid(2, 1, figsize=(4, 6), height_ratios=[2, 1], 
                               sharex='col', hspace=0.05)

# Recipe 2: Main plot with colorbar column
fig, axes = Plotter.make_grid(1, 2, figsize=(6, 4), width_ratios=[20, 1])
ax_main, ax_cbar = axes

# Recipe 3: 2x2 grid with panel labels for publication
fig, axes = Plotter.make_grid(2, 2, figsize=(8, 8), panel_labels=True, despine=True)

# Recipe 4: Asymmetric layout using GridBuilder
builder = Plotter.GridBuilder(figsize=(12, 6))
builder.add_row(ncols=1, height_ratio=1)     # Wide plot at top
builder.add_row(ncols=3, height_ratio=2)     # 3 panels below
fig, axes = builder.build(hspace=0.3)

──────────────────────────────────────────────────────────────────────────────
INSETS & OVERLAYS
──────────────────────────────────────────────────────────────────────────────

Inset Axes:
    inset = Plotter.get_inset(
        ax,
        position=[0.6, 0.6, 0.35, 0.35],  # [x, y, width, height] relative to ax
        add_box=True,                      # Semi-transparent background
    )

Twin Axes (secondary y-axis):
    ax2 = Plotter.twin_axis(ax, which='y', label='Secondary Y', color='C1')
    ax2.plot(x, secondary_data, color='C1')

Unify Axis Limits Across Panels:
    Plotter.unify_limits(axes, which='y')  # Same y-limits for all
''',
            
            'legend': '''
=======
LEGENDS
=======

Publication-Style Legend:
    Plotter.set_legend(
        ax,
        style='publication',    # 'minimal', 'boxed', 'publication'
        loc='best',
        ncol=1,
        frameon=True,
        fontsize=10,
    )

Custom Legend (filter by condition):
    Plotter.set_legend_custom(
        ax,
        conditions=[lambda lbl: 'exp' in lbl],  # Only labels containing 'exp'
    )

Available Styles:
    'minimal'     - No frame, small font
    'boxed'       - Semi-transparent frame
    'publication' - Solid frame, Nature-style formatting
''',
            
            'save': '''
==============
SAVING FIGURES
==============

Save Figure:
    Plotter.save_fig(
        directory='./figures',
        filename='my_plot',
        format='pdf',           # 'pdf', 'png', 'svg', 'eps', 'tiff'
        dpi=300,                # For rasterized formats
        adjust=True,            # Tight bounding box
    )

Recommended Formats:
    - PDF/EPS: Vector format for journals (scalable, small file)
    - PNG: Rasterized for presentations (transparent background)
    - TIFF: High-resolution for print (300+ dpi)
    - SVG: Web/interactive (editable in Inkscape/Illustrator)

Tip: Use constrained_layout=True in get_subplots() to avoid clipping.
''',
            
            'grid': '''
=====================
ADVANCED GRID CONTROL
=====================

For complex layouts with precise control over spacing, sizing, and alignment.

──────────────────────────────────────────────────────────────────────────────
Plotter.make_grid() - Recommended for Most Cases
──────────────────────────────────────────────────────────────────────────────

    fig, axes = Plotter.make_grid(
        nrows=2, ncols=3,
        figsize=(10, 6),
        
        # Size Ratios (list matching number of rows/cols)
        width_ratios=[1, 2, 1],     # Columns: narrow, wide, narrow
        height_ratios=[1, 3],        # Rows: short header, tall main
        
        # Spacing (0-1, fraction of average subplot size)
        wspace=0.3,                  # Horizontal gap between columns
        hspace=0.2,                  # Vertical gap between rows
        
        # Figure Margins (0-1, fraction of figure size)
        left=0.1, right=0.95,
        top=0.95, bottom=0.1,
        
        # Axis Sharing
        sharex='col',                # 'row', 'col', 'all', or False
        sharey='row',
    )

──────────────────────────────────────────────────────────────────────────────
Plotter.GridBuilder - For Nested/Complex Layouts
──────────────────────────────────────────────────────────────────────────────

    builder = Plotter.GridBuilder(figsize=(12, 8))
    
    # Add rows with different column configurations
    builder.add_row(ncols=1, height_ratio=1)           # Single wide panel
    builder.add_row(ncols=3, height_ratio=2,           # 3 columns
                    width_ratios=[2, 1, 1])            # with custom widths
    builder.add_row(ncols=2, height_ratio=1.5)         # 2 columns
    
    fig, axes = builder.build(wspace=0.2, hspace=0.3)
    # axes = [[ax00], [ax10, ax11, ax12], [ax20, ax21]]

──────────────────────────────────────────────────────────────────────────────
Example Recipes
──────────────────────────────────────────────────────────────────────────────

# Stacked panels with shared x-axis (no gap)
fig, axes = Plotter.make_grid(3, 1, figsize=(6, 8), hspace=0.05, sharex='col')
for ax in axes[:-1]:
    Plotter.unset_ticks(ax, xticks=True, xlabel=True)

# Main plot + colorbar
fig, axes = Plotter.make_grid(1, 2, figsize=(7, 5), width_ratios=[20, 1])

# 2x2 for publication with panels (a)-(d)
fig, axes = Plotter.make_grid(2, 2, figsize=(8, 8), panel_labels=True)
''',
            
            'style': '''
==================
PUBLICATION STYLES
==================

Configure global Matplotlib style for different journals.

    configure_style('nature')    # Nature, Science, Cell
    configure_style('science')   # Science-family journals
    configure_style('prl')       # Physical Review Letters
    configure_style('aps')       # APS journals (PRB, PRX, etc.)
    configure_style('default')   # Matplotlib defaults

What it Sets:
    - Font family (STIX/serif for publication)
    - Font sizes (labels, ticks, legend)
    - Line widths
    - Figure DPI
    - Axes styling
    - Color cycle

Example:
    from QES.general_python.common.plot import configure_style, Plotter
    
    configure_style('nature')  # Apply once at start
    
    fig, ax = Plotter.get_subplots(1, 1)
    Plotter.plot(ax, x, y)
    Plotter.save_fig('.', 'figure1', format='pdf')

Custom Adjustments:
    After configure_style(), you can fine-tune with:
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
''',
        }
        
        if topic is None:
            print(help_text['overview'])
        elif topic.lower() in help_text:
            print(help_text[topic.lower()])
        else:
            print(f"Unknown topic: '{topic}'. Available: plot, axis, color, layout, grid, legend, annotate, style, save")
    
    ###########################################################
    #! Utilities for plotting
    ###########################################################
    
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
    
    ###########################################################
    #! Filter results
    ###########################################################

    @staticmethod
    def filter_results(results, filters=None, get_params_fun: callable = lambda r: r['params'], *, tol=1e-9):
        """
        Filter results based on flexible parameter conditions.
        For example, you can filter by:
        [
            { 'data' : [...], 'params' : {'Ns': 8, 'Gz': 0.5, 'hx': 0.3} },
            { 'data' : [...], 'params' : {'Ns': 16, 'Gz': 1.0, 'hx': 0.7} },
        ]
        
        Parameters:
        -----------
        results : list
            List of result dictionaries. We assume it has data and parameters
            For example, 
        filters : dict
            Dictionary of parameter filters. Each key is a parameter name, and value can be:
            - Single value: param == value (e.g., {'Ns': 8})
            - Tuple ('eq', value): param ==     value
            - Tuple ('neq', value): param !=     value
            - Tuple ('lt', value): param <      value
            - Tuple ('le', value): param <=     value
            - Tuple ('gt', value): param >      value
            - Tuple ('ge', value): param >=     value
            - Tuple ('between', (min, max)): min <= param <= max
            - List of values: param in [values]
        
        Returns:
        --------
        filtered : list
            Filtered list of results
        
        Examples:
        ---------
        # Equal to a value
        filter_results(results, {'Ns': 8, 'Gz': 0.5})
        
        # Greater than
        filter_results(results, {'hx': ('gt', 0.5)})
        
        # Between range
        filter_results(results, {'hx': ('between', (0.2, 0.8))})
        
        # In list
        filter_results(results, {'Gz': [0.0, 0.5, 1.0]})
        
        # Combined
        filter_results(results, {
            'Ns': 8,
            'hx': ('between', (0.0, 1.0)),
            'Gz': ('ge', 0.5)
        })

        >> filter_results(results, {'hx': ('lt', 0.5)})
        """
        if filters is None:
            return results
        
        filtered = []
        
        for r in results:
            params  = get_params_fun(r)
            matches = True
            
            for param_name, condition in filters.items():
                param_value = params.get(param_name, None)
                
                if param_value is None:
                    matches = False
                    break
                
                # Handle different condition types
                if isinstance(condition, (list, tuple)) and len(condition) > 0:
                    # Check if it's a comparison operator tuple
                    if isinstance(condition, tuple) and len(condition) == 2 and isinstance(condition[0], str):
                        op, value = condition
                        
                        if isinstance(value, (list, tuple)):
                            raise ValueError(f"Value for operator '{op}' cannot be a list or tuple.")
                        
                        if isinstance(value, str):
                            # E.g., ('gt', 'other_param_name')
                            value = float(params.get(value, None))
                            if value is None:
                                matches = False
                                break
                        
                        if op == 'eq':
                            if not abs(param_value - value) < tol:
                                matches = False
                                break
                        elif op == 'neq':
                            if abs(param_value - value) < tol:
                                matches = False
                                break
                        elif op == 'lt':
                            if not param_value < value:
                                matches = False
                                break
                        elif op == 'le':
                            if not param_value <= value:
                                matches = False
                                break
                        elif op == 'gt':
                            if not param_value > value:
                                matches = False
                                break
                        elif op == 'ge':
                            if not param_value >= value:
                                matches = False
                                break
                        elif op == 'between':
                            min_val, max_val = value
                            if not (min_val <= param_value <= max_val):
                                matches = False
                                break
                        else:
                            raise ValueError(f"Unknown operator: {op}")
                    else:
                        # It's a list of acceptable values
                        if param_value not in condition:
                            matches = False
                            break
                else:
                    # Single value - exact match
                    if isinstance(condition, str):
                        condition = float(params.get(condition, None)) # Convert string to float using params
                    
                    if condition is None:
                        matches = False
                        break
                    
                    if abs(param_value - condition) >= 1e-9:
                        matches = False
                        break
            
            if matches:
                filtered.append(r)
        
        return filtered
    
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

        fig_width_pt = columnwidth * wf
        inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        fig_width = fig_width_pt * inches_per_pt  # width in inches
        fig_height = fig_width * hf  # height in inches
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
            cmap='PuBu', elsecolor='blue', get_mappable=False, norm=None, scale='linear', **kwargs):
        """
        Get a colormap for the given values.
        
        Parameters:
        - values (array-like): The values to map to colors.
        - cmap (str, optional): The colormap to use. Defaults to 'PuBu'.
        - elsecolor (str, optional): The color to use if there is only one value. Defaults to 'blue'.
        
        Returns:
        - getcolor (function): A function that maps a value to a color.
        - colors (Colormap): The colormap object.
        - norm (Normalize): The normalization object.
        
        Example:
        >>> getcolor, colors, norm = Plotter.get_colormap([1, 2, 3], cmap='viridis')
        >>> color = getcolor(2.5)
        """
        from matplotlib.colors import Normalize, LogNorm, SymLogNorm
        
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

        # Create Mappable
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=colors)
        
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
        ax, 
        iter        : int,
        x           : float = 0,
        y           : float = 0,
        fontsize    = 12,
        xycoords    = 'axes fraction',
        addit       = '',
        condition   = True,
        zorder      = 50,
        boxaround   = True,
        fontweight  = 'bold',
        color       = 'black',
        **kwargs        
        ):
        """
        Annotate plot with the letter.
        
        Params:
        -----------
        ax: 
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
        kwargs: 
            additional arguments for annotation
            - color : color of the text
            - weight: weight of the text
            
        Example:
        --------
        >>> Plotter.set_annotate_letter(ax, 0, x=0.1, y=0.9, fontsize=14, addit=' Test', color='red')
        """
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
                labelcond = True,
                **kwargs):
        '''
        horizontal line plotting
        '''
        
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]

        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
        
        if label is None or label == '':
            labelcond = False

        ax.axhline(val, ls = ls,  lw = lw,
                label = label if (label is not None and len(label) != 0 and labelcond) else None,
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
                labelcond   = True,
                **kwargs):
        '''
        vertical line plotting
        '''
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]

        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]

        if label is None or label == '':
            labelcond = False

        ax.axvline(val, 
                ls      = ls,  
                lw      = lw, 
                label   = label if (label is not None and len(label) != 0 and labelcond) else None, 
                color   = color,
                zorder  = zorder,
                **kwargs)
    
    ################## S C A T T E R ##################
    
    @staticmethod
    def scatter(ax, x, y,
        s           =   10,
        c           =   'blue',
        marker      =   'o',
        alpha       =   1.0,
        label       =   None,
        edgecolor   =   None,
        zorder      =   5,
        labelcond   =   True,
        linewidths  =   1.0,
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
            
        if isinstance(c, int):
            c = colorsList[c % len(colorsList)]

        if isinstance(marker, int):
            marker = markersList[marker % len(markersList)]

        if label is None or label == '':
            labelcond = False    
        
        ax.scatter(x, y, linewidths = linewidths,
                s=s, c=c, marker=marker, 
                alpha=alpha, label=label if labelcond else '', edgecolor=edgecolor, zorder=zorder, **kwargs)
    
    #################### P L O T S ####################
    
    @staticmethod
    def plot(  ax, 
                x, 
                y,
                ls          = '-',
                lw          = 2.0,
                color       = 'black',
                # label 
                label       = None,
                labelcond   = True,
                # marker
                marker      = None,
                ms          = None,
                # other
                zorder      = 5,
                **kwargs):
        '''
        plot the data
        '''
        if 'linestyle' in kwargs:
            ls = kwargs['linestyle']
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
            labelcond = False
        
        # use the defaults
        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
            
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]
        
        if isinstance(marker, int):
            marker = markersList[marker % len(markersList)]
        
        ax.plot(x, y, 
                ls      = ls, 
                lw      = lw, 
                color   = color, 
                label   = label if labelcond else '', zorder = zorder, marker = marker, ms = ms, **kwargs)

    @staticmethod
    def fill_between(
        ax,
        x,
        y1,
        y2,
        color       =   'blue',
        alpha       =   0.5,
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
        ax.fill_between(x, y1, y2, color=color, alpha=alpha, **kwargs)
    
    # ################ LOG SCALE PLOTS ################
    
    @staticmethod
    def semilogy(ax, x, y, ls='-', lw=1.5, color='black', label=None, marker=None, ms=None, labelcond=True, zorder=5, **kwargs):
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
            labelcond = False
            
        ax.semilogy(x, y, ls=ls, lw=lw, color=color, label=label if labelcond else '', marker=marker, ms=ms, zorder=zorder, **kwargs)
    
    @staticmethod
    def semilogx(ax, x, y, ls='-', lw=1.5, color='black', label=None, marker=None, ms=None, labelcond=True, zorder=5, **kwargs):
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
        if isinstance(color, int):
            color = colorsList[color % len(colorsList)]
        if isinstance(ls, int):
            ls = linestylesList[ls % len(linestylesList)]
        if isinstance(marker, int):
            marker = markersList[marker % len(markersList)]
        if label is None or label == '':
            labelcond = False
            
        ax.semilogx(x, y, ls=ls, lw=lw, color=color, label=label if labelcond else '', marker=marker, ms=ms, zorder=zorder, **kwargs)
    
    @staticmethod
    def loglog(ax, x, y, ls='-', lw=1.5, color='black', label=None, marker=None, ms=None, labelcond=True, zorder=5, **kwargs):
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
            labelcond = False
            
        ax.loglog(x, y, ls=ls, lw=lw, color=color, label=label if labelcond else '', marker=marker, ms=ms, zorder=zorder, **kwargs)
    
    # -------------------- ERROR BARS --------------------
    
    @staticmethod
    def errorbar(ax, x, y, yerr=None, xerr=None, fmt='o', color='black', capsize=2, capthick=1.0, elinewidth=1.0, markersize=5, label=None, labelcond=True, alpha=1.0, zorder=5, **kwargs):
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
            labelcond = False
        
        ax.errorbar(x, y, yerr=yerr, xerr=xerr,
                    fmt=fmt, color=color,
                    capsize=capsize, capthick=capthick,
                    elinewidth=elinewidth, markersize=markersize,
                    label=label if labelcond else '',
                    alpha=alpha, zorder=zorder, **kwargs)
    
    # -------------------- HISTOGRAM --------------------
    
    @staticmethod
    def histogram(ax, data, bins=50, density=True, histtype='stepfilled', alpha=0.7, color='C0', edgecolor='black',
                linewidth=1.0, label=None, orientation='vertical', cumulative=False, log=False, labelcond=True, zorder=5, **kwargs):
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
            labelcond = False
        
        return ax.hist(data, bins=bins, density=density,
                    histtype=histtype, alpha=alpha,
                    color=color, edgecolor=edgecolor,
                    linewidth=linewidth, 
                    label=label if labelcond else '',
                    orientation=orientation,
                    cumulative=cumulative, log=log,
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
                        labelsize   =   None,
                        left        =   True,
                        right       =   True,
                        top         =   True,
                        bottom      =   True,
                        xticks      =   None,
                        yticks      =   None,
                        maj_tick_l  =   6,
                        min_tick_l  =   3,
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

    @staticmethod
    def set_ax_params(
            ax,
            which           : str           = 'both',       # which axis to update
            label           : Union[dict, str]    = "",           # general label
            xlabel          : Union[dict, str]    = "",           # x label - works when which is 'x' or 'both'
            ylabel          : str           = "",           # y label - works when which is 'y' or 'both'
            title           : str           = "",           # title of the axis
            scale           : Union[str, dict]   = "linear",     # scale of the axis
            xscale                          = None,         # scale of the x-axis
            yscale                          = None,         # scale of the y-axis
            lim             : Union[dict, tuple, None] = None,   # fallback for axis limits
            xlim                            = None,         # specific limits for x-axis
            ylim                            = None,         # specific limits for y-axis
            fontsize        : int           = plt.rcParams['font.size'],
            labelPad        : float         = 0.0,          # padding for axis labels
            labelCond       : Union[bool, dict]   = True,
            labelPos        : Union[dict, str, None] = None,     # label positions
            xlabelPos       : Union[str, None]   = None,         # position of the xlabel
            ylabelPos       : Union[str, None]   = None,         # position of the ylabel
            tickPos         : Union[dict, str, None] = None,     # tick positions
            labelCords      : Union[dict, str, None] = None,     # manual label coordinates
            ticks           : Union[dict, str, None] = None,     # custom ticks
            labels          : Union[dict, str, None] = None,     # custom tick labels
            maj_tick_l      : float         = 2,
            min_tick_l      : float         = 1,
            tick_params     : Union[dict, str, None] = None,
            title_fontsize  : int           = 14,
        ):
            """
            Sets various axis parameters for publication-ready plots.

            Parameters:
                ax (matplotlib.axes.Axes): The axis to modify.
                which (str): Specifies which axes to update ('x', 'y', 'both').
                label (str): Label for the axis (fallback for `xlabel` and `ylabel`).
                xlabel, ylabel (str): Labels for x and y axes (overrides `label`).
                title (str): Title for the axis.
                scale (str): Scale of the axis ('linear', 'log').
                lim (tuple): Fallback for axis limits.
                xlim, ylim (tuple): Specific limits for x and y axes (overrides `lim`).
                fontsize (int): Font size for axis labels.
                labelPad (float): Padding for axis labels.
                labelCond (bool): Whether to show the label.
                labelPos (dict): Label positions (e.g., {"x": "top", "y": "right"}).
                tickPos (dict): Tick positions (e.g., {"x": "bottom", "y": "left"}).
                labelCords (dict): Manual label coordinates (e.g., {"x": (0.5, -0.1)}).
                ticks (dict): Custom ticks (e.g., {"x": [1, 2, 3], "y": [0.1, 0.2]}).
                labels (dict): Custom tick labels (e.g., {"x": ["A", "B", "C"]}).
                maj_tick_l (float): Length of major ticks.
                min_tick_l (float): Length of minor ticks.
                tick_params (dict): Additional tick parameters.
                title_fontsize (int): Font size for the title.
            """
            # Resolve limits
            if True:
                # limits
                if isinstance(lim, (tuple, list)):
                    lim = {"x": lim, "y": lim}
                elif not isinstance(lim, dict):
                    lim = {}
                if xlim is not None and isinstance(xlim, (tuple, list)):
                    lim['x'] = xlim
                if ylim is not None and isinstance(ylim, (tuple, list)):
                    lim['y'] = ylim

                # label positions
                if labelPos is None:
                    labelPos = {'x':'left', 'y':'bottom'}
                elif isinstance(labelPos, str):
                    labelPos = {"x": labelPos, "y": labelPos}
                elif not isinstance(labelPos, dict):
                    labelPos = {}
                if xlabelPos is not None:
                    labelPos['x'] = xlabelPos
                if ylabelPos is not None:
                    labelPos['y'] = ylabelPos

                # tick positions
                if tickPos is None:
                    tickPos = {'x':'bottom', 'y':'left'}
                elif isinstance(tickPos, str):
                    tickPos = {"x": tickPos, "y": tickPos}
                elif not isinstance(tickPos, dict):
                    tickPos = {}
                if xlabelPos is not None:
                    tickPos['x'] = xlabelPos
                if ylabelPos is not None:
                    tickPos['y'] = ylabelPos

                # label coordinates
                if labelCords is None:
                    labelCords = {}
                elif isinstance(labelCords, str):
                    labelCords = {"x": labelCords, "y": labelCords}
                elif not isinstance(labelCords, dict):
                    labelCords = {}
                if 'x' in labelCords and isinstance(labelCords['x'], (tuple, list)):
                    labelCords['x'] = labelCords['x']
                if 'y' in labelCords and isinstance(labelCords['y'], (tuple, list)):
                    labelCords['y'] = labelCords['y']

                # scale
                if scale is None:
                    scale = {"x": "linear", "y": "linear"}
                elif isinstance(scale, str):
                    scale = {"x": scale, "y": scale}
                elif not isinstance(scale, dict):
                    scale = {}
                if xscale is not None and isinstance(xscale, str) and xscale != "":
                    scale['x'] = xscale
                if yscale is not None and isinstance(yscale, str) and yscale != "":
                    scale['y'] = yscale
                    
                
                # ticks
                if ticks is None:
                    ticks = {}
                elif isinstance(ticks, str):
                    ticks = {"x": ticks, "y": ticks}
                elif not isinstance(ticks, dict):
                    ticks = {}
                if 'x' in ticks and isinstance(ticks['x'], (tuple, list)):
                    ticks['x'] = ticks['x']
                if 'y' in ticks and isinstance(ticks['y'], (tuple, list)):
                    ticks['y'] = ticks['y']

                # labelcond 
                if labelCond is None:
                    labelCond = {"x": True, "y": True}
                elif isinstance(labelCond, bool):
                    labelCond = {"x": labelCond, "y": labelCond}
                elif not isinstance(labelCond, dict):
                    labelCond = {}
                
                # labels
                if labels is None:
                    labels = {}
                elif isinstance(labels, str):
                    labels = {"x": labels, "y": labels}
                elif not isinstance(labels, dict):
                    labels = {}
                if 'x' in labels and isinstance(labels['x'], (tuple, list)):
                    labels['x'] = labels['x']
                if 'y' in labels and isinstance(labels['y'], (tuple, list)):
                    labels['y'] = labels['y']

                # label 
                if label == "":
                    label       = {"x": "", "y": ""}
                elif label is None:
                    label       = {"x": "", "y": ""}
                    
                if isinstance(label, str):
                    label = {"x": label, "y": label}
                elif not isinstance(label, dict):
                    label = {}
                
                if xlabel is not None and isinstance(xlabel, str):
                    if xlabel == "":
                        labelCond["x"] = False
                    label['x'] = xlabel
                if ylabel is not None and isinstance(ylabel, str):
                    if ylabel == "":
                        labelCond["y"] = False
                    label['y'] = ylabel

            # Default tick parameters
            default_tick_params = {
                "length"    : maj_tick_l,
                "width"     : 0.8,
                "direction" : "in",
                "labelsize" : fontsize - 2 if fontsize else 10,
            }
            if tick_params:
                if isinstance(tick_params, dict):
                    default_tick_params.update(tick_params)

            # Update x-axis parameters
            if "x" in which or "both" in which:
                ax.set_xlabel(label['x'] if labelCond['x'] else "", fontsize=fontsize, labelpad=labelPad)
                if lim and "x" in lim:
                    ax.set_xlim(lim["x"])
                    
                # scale 
                if True:
                    if scale and "x" in scale and scale["x"] is not None:
                        ax.set_xscale(scale["x"])
                    if scale and "x" in scale and scale["x"] == "log":
                        ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='all', numticks=100))
                # ax.xaxis.set_tick_params(**default_tick_params)
                    
                if labelPos and "x" in labelPos and labelPos["x"] in ['top', 'bottom']:
                    ax.xaxis.set_label_position(labelPos["x"])
                    # add small padding if top
                    if labelPos["x"] == 'top':
                        ax.xaxis.set_label_coords(0.5, 1.1)
                if tickPos and "x" in tickPos:
                    getattr(ax.xaxis, f"tick_{tickPos['x']}")()
                if labelCords and "x" in labelCords:
                    ax.xaxis.set_label_coords(*labelCords["x"])
                if ticks and "x" in ticks:
                    ax.set_xticks(ticks["x"])
                    if labels and "x" in labels:
                        ax.set_xticklabels(labels["x"])

            # Update y-axis parameters
            if "y" in which or "both" in which:
                ax.set_ylabel(label['y'] if labelCond['y'] else "", fontsize=fontsize, labelpad=labelPad)
                if lim and "y" in lim:
                    ax.set_ylim(lim["y"])
                    
                # scale 
                if scale and "y" in scale and scale["y"] is not None:
                    ax.set_yscale(scale["y"])
                if scale and "y" in scale and scale["y"] == "log":
                    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='all', numticks=100))
                # ax.yaxis.set_tick_params(**default_tick_params)
                    
                if labelPos and "y" in labelPos and labelPos["y"] in ['left', 'right']:
                    ax.yaxis.set_label_position(labelPos["y"])
                    # add small padding if right
                    if labelPos["y"] == 'right':
                        ax.yaxis.set_label_coords(1.1, 0.5)
                if tickPos and "y" in tickPos:
                    getattr(ax.yaxis, f"tick_{tickPos['y']}")()
                if labelCords and "y" in labelCords:
                    ax.yaxis.set_label_coords(*labelCords["y"])
                if ticks and "y" in ticks:
                    ax.set_yticks(ticks["y"])
                    if labels and "y" in labels:
                        ax.set_yticklabels(labels["y"])

            # Set title
            if title:
                ax.set_title(title, fontsize=title_fontsize)    
    
    @staticmethod
    def set_ax_labels(  ax,
                        fontsize    =   None,
                        xlabel      =   "",
                        ylabel      =   "",
                        title       =   "",
                        xPad        =   0,
                        yPad        =   0):
        '''
        Sets the labels of the x and y axes
        '''
        if xlabel != "":
            ax.set_xlabel(xlabel, 
                            fontsize = fontsize,
                            labelpad = xPad if xPad != 0 else None)
        if ylabel != "":
            ax.set_ylabel(ylabel, 
                            fontsize = fontsize,
                            labelpad = yPad if yPad != 0 else None)
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
    
    # @staticmethod
    # def set_ax_ticks(ax, 
    #                   )
    
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
        """
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
        """
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
    def get_subplots(   nrows       =   1,
                        ncols       =   1,
                        sizex       =   10.,                    # total width [in] OR list of per-col ratios
                        sizey       =   10.,                    # total height [in] OR list of per-row ratios
                        sizex_def   =   3,                      # inches per unit of sizex ratio (if sizex is a sequence)
                        sizey_def   =   3,                      # inches per unit of sizey ratio (if sizey is a sequence)
                        annot_x_pos =   None,                   # position for annotation - x
                        annot_y_pos =   None,                   # position for annotation - y
                        panel_labels=   False,
                        **kwargs) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Flexible subplot factory that *always* returns (fig, flat_axes_list).

        niceties (all optional, via kwargs):
        - width_ratios/height_ratios via sizex/sizey if they are sequences
        - hspace/wspace/left/right/top/bottom (mapped to gridspec_kw)
        - constrained_layout=True by default (unless tight_layout is explicitly set)
        - panel_labels=True or a list/tuple (strings) to annotate axes (A,B,C,...)
        - grid=True to enable grid on all axes (use grid_kws for kwargs)
        - despine=True to hide top/right spines, axis_off=True to turn entire axes off
        - suptitle="...", suptitle_kws={...}
        - dpi=..., sharex=..., sharey=..., subplot_kw={...}, gridspec_kw={...}
        - post_hook=<callable(fig, axes_list)> for custom last-mile tweaks
        """

        # ---- Extract utility kwargs (do not pass to plt.subplots)
        gridspec_kw   = dict(kwargs.pop('gridspec_kw', {}) or {})
        subplot_kw    = dict(kwargs.pop('subplot_kw', {}) or {})
        panel_labels  = kwargs.pop('panel_labels', None)         # True or list/tuple of labels
        grid_on       = kwargs.pop('grid', False)
        grid_kws      = dict(kwargs.pop('grid_kws', {}) or {})
        despine       = kwargs.pop('despine', False)
        axis_off      = kwargs.pop('axis_off', False)
        suptitle      = kwargs.pop('suptitle', None)
        suptitle_kws  = dict(kwargs.pop('suptitle_kws', {}) or {})
        post_hook     = kwargs.pop('post_hook', None)

        # Use constrained_layout by default unless user opted for tight_layout or already set it
        if 'constrained_layout' not in kwargs and not kwargs.get('tight_layout', False):
            kwargs['constrained_layout'] = True

        # Map spacing/bounds kwargs into gridspec_kw if provided
        for k in ('hspace', 'wspace', 'left', 'right', 'top', 'bottom'):
            if k in kwargs:
                gridspec_kw[k] = kwargs.pop(k)

        #! Figure size & ratios
        width_ratios = height_ratios = None
        # sizex can be total inches (number) or per-column *ratios* (sequence)
        if isinstance(sizex, (list, tuple)):
            if len(sizex) != ncols:
                raise ValueError(f"sizex length {len(sizex)} != ncols {ncols}")
            width_ratios    = list(sizex)
            total_w         = sizex_def * fsum(width_ratios)
        else:
            # If set to None, infer from defaults
            total_w         = float(sizex if sizex is not None else sizex_def * ncols)

        # sizey can be total inches (number) or per-row *ratios* (sequence)
        if isinstance(sizey, (list, tuple)):
            if len(sizey) != nrows:
                raise ValueError(f"sizey length {len(sizey)} != nrows {nrows}")
            height_ratios   = list(sizey)
            total_h         = sizey_def * fsum(height_ratios)
        else:
            total_h         = float(sizey if sizey is not None else sizey_def * nrows)

        if width_ratios is not None:
            gridspec_kw['width_ratios']     = width_ratios
        if height_ratios is not None:
            gridspec_kw['height_ratios']    = height_ratios

        figsize                             = kwargs.pop('figsize', (total_w, total_h))

        #! Create
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, gridspec_kw=gridspec_kw, subplot_kw=subplot_kw, **kwargs)

        #! Normalize axes to a flat list, regardless of (1,1), (1,N), (M,1), (M,N)
        if isinstance(ax, (list, tuple)):
            axes_list = list(ax)
        else:
            try:
                axes_list = ax.ravel().tolist()     # ndarray of Axes
            except Exception:
                axes_list = [ax]                    # single Axes

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

        return fig, axes_list

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

##############################################################################

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
    
##############################################################################