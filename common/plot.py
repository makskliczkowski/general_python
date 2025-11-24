import itertools
from math import fsum
from typing import Tuple, Union, Optional, List

try:
    import scienceplots
except ImportError:
    print("scienceplots not found, some styles may not be available.")

########################## matplotlib ##########################
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

########################## grids
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.ticker as mticker
from matplotlib import ticker
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter, LogLocator, LogFormatterMathtext
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase, HandlerPatch
########################## other
import numpy as np
mpl.rcParams.update(mpl.rcParamsDefault)

# Safely set rcParams (for compatibility with documentation build systems)
try:
    plt.rcParams['axes.facecolor']      =   'white'
    plt.rcParams['savefig.facecolor']   =   'w'
except (TypeError, AttributeError, KeyError):
    # Handle cases where rcParams doesn't support item assignment (e.g., during doc builds)
    pass

########################## labellines
try:
    from labellines import labelLine, labelLines
except ImportError:
    print("labellines not found, labelLine and labelLines functions will not be available.")
    
########################## style
SMALL_SIZE                          =   12
MEDIUM_SIZE                         =   14
BIGGER_SIZE                         =   16
ADDITIONAL_LINESTYLES               =   {
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

# plt.rcParams['text.usetex']         = True
# latex_engine                        = 'pdflatex'
# latex_elements                      = {
#                                         'extrapackages': r'\usepackage{physics}',
#                                         'extrapackages': r'\usepackage{amsmath}'
#                                     }

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
    A Plotter class that handles the methods of plotting.
    """
    
    def __init__(self, default_cmap='viridis', font_size=12, dpi=200):
        '''
        Initialize the Plotter class with default parameters.
        Parameters:
            default_cmap [str]:
                Default colormap to use for plots.
            font_size [int]:
                Default font size for text in plots.
            dpi [int]:
                Dots per inch for the figure resolution.
        '''
        self.default_cmap   = default_cmap
        self.font_size      = font_size
        self.dpi            = dpi
    
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
                    extend              : str = 'neutral',
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
        fig : Figure
            Parent figure.
        pos : list[float]
            [left, bottom, width, height] in figure coordinates (0-1).
        mappable : array-like or ScalarMappable
            Data array or existing plot object.
        cmap : str or Colormap
            Colormap scheme.
        norm : Normalize, optional
            Manual normalization.
        vmin, vmax : float, optional
            Data limits.
        scale : {'linear', 'log', 'symlog'}
            Scale type.
        orientation : {'vertical', 'horizontal'}
            Orientation of the bar.
        label : str
            Text label along the long axis (e.g., "Intensity").
        title : str
            Text title at the end/top of the bar (e.g., "(a)").
        ticks : list
            Manual tick positions.
        ticklabels : list
            Manual tick strings.
        tick_location : {'auto', 'left', 'right', 'top', 'bottom'}
            Where to place ticks/labels.
        discrete : bool or int
            If True, discretizes cmap into distinct steps. If int, uses that many steps.
        boundaries : list
            Specific values for discrete color transitions (creates BoundaryNorm).
        invert : bool
            If True, flips the axis direction.
        remove_pdf_lines : bool
            Fixes white lines between color segments in PDF exports.
        format : str or Formatter
            Tick format (e.g., '%.2e').
        **kwargs : 
            Passed to fig.colorbar (e.g., alpha, spacing).

        Returns
        -------
        cbar, cax
        
        Examples
        --------
        >>> # 1. Standard Linear Colorbar (Vertical)
        >>> cbar, cax = Plotter.add_colorbar(fig, [0.92, 0.15, 0.02, 0.7], data_array, 
        ...                                      label='Magnetization $M_z$')

        >>> # 2. Log-Scale with Scientific Notation (Horizontal)
        >>> # Places ticks on top, adds arrows for clipping, formats as 1e-5
        >>> cbar, cax = Plotter.add_colorbar(fig, [0.2, 0.9, 0.6, 0.03], data_array,
        ...                                      scale='log', orientation='horizontal',
        ...                                      format='%.0e', extend='both', 
        ...                                      tick_location='top', label='Conductance')

        >>> # 3. Discrete Phase Diagram (Categorical)
        >>> # Divides colormap into 3 distinct blocks with custom labels
        >>> cbar, cax = Plotter.add_colorbar(fig, [0.85, 0.1, 0.03, 0.8], 
        ...                                      mappable=[0, 1, 2], cmap='Set1', 
        ...                                      discrete=3,
        ...                                      ticklabels=['Insulator', 'Metal', 'Supercond.'])

        >>> # 4. Custom Boundaries (Non-uniform steps)
        >>> # Useful for contour plots with specific threshold levels
        >>> cbar, cax = Plotter.add_colorbar(fig, pos, data, 
        ...                                      boundaries=[0, 0.5, 2.0, 10.0],
        ...                                      spacing='proportional')        
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
    def get_colormap(values, cmap='PuBu', elsecolor='blue', get_mappable=False, **kwargs):
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
        norm        = Normalize(np.min(values), np.max(values))
        colors      = plt.get_cmap(cmap)
        values      = np.sort(values)
        getcolor    = lambda x: colors((x - values[0]) / (values[-1] - values[0])) if len(values) != 1 else elsecolor
        # get the mappable as well 
        mappable    = plt.cm.ScalarMappable(norm=norm, cmap=colors)
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
        Plotter.set_annotate(ax, elem = f'({chr(97 + iter)})' + addit, x = x, y = y, 
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
    def unset_spines(   ax,
                        xticks      =   False,
                        yticks      =   False,
                        left        =   False,
                        right       =   False,
                        top         =   False,
                        bottom      =   False
                     ):
        """
        Disables specified spines and optionally hides ticks on the axis.

        Parameters:
        - ax: Matplotlib axis object.
        - top, right, bottom, left: Booleans to show/hide respective spines.
        - xticks, yticks: Booleans to show/hide tick labels on the x and y axis.
        """
        ax.spines['top'].set_visible(top)
        ax.spines['right'].set_visible(right)
        ax.spines['bottom'].set_visible(bottom)
        ax.spines['left'].set_visible(left)
        
        if not xticks:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        if not yticks:
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    @staticmethod
    def unset_ticks(ax,
                    xticks=False,
                    yticks=False,
                    remove_labels_only=True,
                    erase=False,
                    spines=True):
        '''
        Disables or customizes the ticks on the axis.

        Parameters:
        - ax                : Axis to modify.
        - xticks            : Disable x-ticks if False.
        - yticks            : Disable y-ticks if False.
        - remove_labels_only: Remove only the labels, keeping ticks visible.
        - erase             : Completely hide the axis if True.
        - spines            : Keep or remove spines based on tick visibility.
        '''
        if not xticks:
            if remove_labels_only:
                ax.set_xticklabels([], minor=False)
                ax.set_xticklabels([], minor=True)
            else:
                ax.set_xticks([], minor=False)
                ax.set_xticks([], minor=True)
                ax.set_xticklabels([], minor=False)
                ax.set_xticklabels([], minor=True)
                ax.xaxis.set_tick_params(which='both', labelbottom=False)

            if erase:
                ax.axes.get_xaxis().set_visible(False)

        if not yticks:
            if remove_labels_only:
                ax.set_yticklabels([], minor=False)
                ax.set_yticklabels([], minor=True)
            else:
                ax.set_yticks([], minor=False)
                ax.set_yticks([], minor=True)
                ax.set_yticklabels([], minor=False)
                ax.set_yticklabels([], minor=True)
                ax.yaxis.set_tick_params(which='both', labelleft=False)

            if erase:
                ax.axes.get_yaxis().set_visible(False)

        if not spines:
            # Handle spines
            Plotter.unset_spines(
                ax,
                xticks=xticks,
                yticks=yticks,
                left=not ((not spines) and (not yticks)),
                right=not ((not spines) and (not yticks)),
                top=not ((not spines) and (not xticks)),
                bottom=not ((not spines) and (not xticks))
            )

    @staticmethod
    def unset_ticks_and_spines(ax, 
                                xticks  = False,
                                yticks  = False,
                                erease  = False,
                                spines  = True,
                                left    = False,
                                right   = False,
                                top     = False,
                                bottom  = False
                               ):
        '''
        Unset ticks and spines
        '''
        Plotter.unset_spines(ax, xticks = False, yticks = False, left = left, right = right, top = top, bottom = bottom)
        Plotter.unset_ticks(ax, xticks = xticks, yticks = yticks, erease = erease, spines = spines)
    
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
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: "%g"%x))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "%g"%x))
        if 'y' in axis:
            ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: "%g"%x))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "%g"%x))
    
    #################### G R I D S ####################
    
    @staticmethod
    def get_grid(nrows          :   int,
                 ncols          :   int,
                 wspace         =   None,
                 hspace         =   None,
                 width_ratios   =   None,
                 height_ratios  =   None,
                 ax_sub         =   None,
                 **kwargs):
        '''
        Obtain grid from GridSubplotSpec
        '''
        if ax_sub is not None:
            return GridSpecFromSubplotSpec(nrows        =   nrows, 
                                           ncols        =   ncols,
                                           subplot_spec =   ax_sub, 
                                           wspace       =   wspace,
                                           hspace       =   hspace,
                                           width_ratios =   width_ratios,
                                           height_ratios=   height_ratios,
                                           **kwargs)
        else:
            return GridSpec               (nrows        =   nrows, 
                                           ncols        =   ncols,                                           
                                           wspace       =   wspace,
                                           hspace       =   hspace,
                                           width_ratios =   width_ratios,
                                           height_ratios=   height_ratios,
                                           **kwargs)
    
    @staticmethod
    def get_grid_ax(nrows           : int,
                    ncols           : int,
                    wspace          = None,
                    hspace          = None,
                    width_ratios    = None,
                    height_ratios   = None,
                    ax_sub          = None,
                    **kwargs):
        '''
        Obtain grid from GridSubplotSpec
        '''
        return Plotter.get_grid(nrows, ncols, wspace, hspace, width_ratios, height_ratios, ax_sub, **kwargs), []
    
    @staticmethod
    def get_grid_subplot(gs,
                        fig,
                        i: int,
                        sharex      = None,
                        sharey      = None,
                        padding     = None,
                        spacing     = None,
                        empty_space = None,
                        **kwargs):
        '''
        Creates the subaxis for the GridSpec with optional custom paddings.
        
        Parameters:
            gs (GridSpec): The GridSpec object.
            fig (Figure): The matplotlib figure object.
            i (int): Index of the subplot in the GridSpec.
            sharex (Axes, optional)     : Share the x-axis with another subplot.
            sharey (Axes, optional)     : Share the y-axis with another subplot.
            padding (dict, optional)    : Custom paddings for the subplot layout.
                                        Keys can include 'left', 'right', 'top', 'bottom'.
            **kwargs: Additional keyword arguments passed to `add_subplot`.
        
        Returns:
            Axes: The created subplot.
        '''
        ax = fig.add_subplot(gs[i], sharex=sharex, sharey=sharey, **kwargs)

        # Apply custom paddings if provided
        if padding is not None:
            if not isinstance(padding, dict):
                raise ValueError("Padding must be a dictionary with keys 'left', 'right', 'top', and 'bottom'.")
            fig.subplots_adjust(**{k: v for k, v in padding.items() if k in ['left', 'right', 'top', 'bottom']})
        # Apply custom spacings if provided
        if spacing is not None:
            if not isinstance(spacing, dict):
                raise ValueError("Spacing must be a dictionary with keys 'wspace' and 'hspace'.")
            fig.subplots_adjust(**{k: v for k, v in spacing.items() if k in ['wspace', 'hspace']})
        
        if empty_space is not None:
            
            valid_sides = {'left', 'right', 'top', 'bottom'}        # Validate and parse empty_space
            empty_space = empty_space or {}                         # Ensure empty_space is a dictionary
            
            if not all(side in valid_sides for side in empty_space):
                raise ValueError(f"Keys in empty_space must be one of {valid_sides}, got {list(empty_space.keys())}.")

            # Get current subplot adjustment parameters
            bbox = ax.get_position()

            # Apply adjustments based on the dictionary
            if 'left' in empty_space:
                bbox.x0 = empty_space['left']
            if 'right' in empty_space:
                bbox.x1 = 1 - empty_space['right']
            if 'bottom' in empty_space:
                bbox.y0 = empty_space['bottom']
            if 'top' in empty_space:
                bbox.y1 = 1 - empty_space['top']

            # Adjust the subplot layout for the given axis
            ax.set_position(bbox)
            
        return ax

    @staticmethod
    def app_grid_subplot(   axes   : list,
                            gs, 
                            fig, 
                            i      : int,
                            sharex = None,
                            sharey = None,
                            **kwargs):
        '''
        Appends the subplot to the axes
        '''
        axes.append(Plotter.get_grid_subplot(gs, fig, i, sharex = sharex, sharey = sharey, **kwargs))
    
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

        Parameters:
        - ax: The parent axis.
        - position: List of [x0, y0, width, height] for the inset axis in relative coordinates.
        - add_box: Whether to add a semi-transparent box around the inset.
        - box_alpha: Transparency of the box.
        - box_ext: Extension of the box beyond the inset axis.
        - facecolor: Face color of the box.
        - zorder: Z-order of the inset axis.
        - kwargs: Additional arguments passed to plt.axes.

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
        post_hook     = kwargs.pop('post_hook', None)            # type: Optional[PostHook]

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


import numpy as np
import json

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

from sympy import Matrix, init_printing
try:
    from IPython.display import display
        
    class MatrixPrinter:
        '''
        Class for printing matrices and vectors
        '''
        
        def __init__(self):
            init_printing()
        
        @staticmethod
        def print_matrix(matrix: np.ndarray):
            '''
            Prints the matrix in a nice form
            '''
            display(Matrix(matrix))
        
        @staticmethod
        def print_vector(vector: np.ndarray):
            '''
            Prints the vector in a nice form
            '''
            display(Matrix(vector))
        
        @staticmethod
        def print_matrices(matrices: list):
            '''
            Prints a list of matrices in a nice form
            '''
            for matrix in matrices:
                display(Matrix(matrix))
        
        @staticmethod
        def print_vectors(vectors: list):
            '''
            Prints a list of vectors in a nice form
            '''
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