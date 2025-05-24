import itertools
try:
    import style
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
from matplotlib.ticker import ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase, HandlerPatch
########################## other
import numpy as np
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.facecolor']      =   'white'
plt.rcParams['savefig.facecolor']   =   'w'

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
    print(e)
    plt.style.use(['science', 'no-latex'])
    
mpl.rcParams['mathtext.fontset']    = 'stix'
mpl.rcParams['font.family']         = 'STIXGeneral'

# plt.rcParams['text.usetex']         = True
# latex_engine                        = 'pdflatex'
# latex_elements                      = {
#                                         'extrapackages': r'\usepackage{physics}',
#                                         'extrapackages': r'\usepackage{amsmath}'
#                                     }

#####################################
colorsList                          =   (list(mcolors.TABLEAU_COLORS))
colorsCycle                         =   itertools.cycle(list(mcolors.TABLEAU_COLORS))
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

############################ grids ############################

########################### plotter ###########################

import seaborn as sns

class Plotter:
    """ 
    A Plotter class that handles the methods of plotting.
    """
    
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
                  alpha = None,
                  edgecolor = (0,0,0,1), 
                  facecolor = (1,1,1,0)
                  ):
        dictionary = dict(facecolor = facecolor, edgecolor = edgecolor, color=color)
        if alpha is not None:
            dictionary['alpha'] = alpha
        return dictionary
    
    ####################### C O L O R S #######################

    @staticmethod
    def add_colorbar(axes, fig, cmap, title = '', norm = None, *args, **kwargs):
        '''
        Add colorbar to the plot. 
        - axes      :   axis to add the colorbar to
        - fig       :   figure to add the colorbar to
        - cmap      :   colormap to use
        - title     :   title of the colorbar
        - norm      :   normalization of the colorbar
        '''
        sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
        sm.set_array([])
        
        # add colorbar
        cbar    = fig.colorbar(sm, ax = axes, *args, **kwargs)
        cbar.ax.set_title(title)
        return cbar
    
    @staticmethod
    def add_colorbar_to_subplot(axes, 
                                fig, 
                                cmap, 
                                title           =   '', 
                                norm            =   None, 
                                position        =   'right', 
                                size            =   '5%', 
                                pad             =   0.1, 
                                within_plot     =   False, 
                                locator_kwargs  =   None, 
                                xlabel          =   None,
                                xlabelcords     =   None,
                                xlabelpos       =   'bottom',
                                ylabel          =   None,
                                ylabelcords     =   None,
                                ylabelpos       =   'left',
                                xtickscords     =   None,
                                xtickpos        =   None,
                                ytickscords     =   None,
                                ytickpos        =   None,
                                rotation        =   None,
                                xticks          =   None,
                                yticks          =   None,
                                single          =   False,
                                *args, **kwargs):
        '''
        Add a colorbar within or next to a specific subplot axis (or multiple axes).
        - axes           : axis or list of axes to add the colorbar to
        - fig            : figure to add the colorbar to
        - cmap           : colormap to use
        - title          : title of the colorbar
        - norm           : normalization of the colorbar
        - position       : position of the colorbar ('right', 'left', 'top', 'bottom') or within the plot
        - size           : size of the colorbar (e.g., '5%') for external positioning
        - pad            : padding between the colorbar and the axes (for external positioning)
        - within_plot    : if True, places the colorbar within the plot using an automatic locator
        - locator_kwargs : dictionary of kwargs for locating the colorbar within the plot (applies if within_plot=True)
        '''

        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        if not isinstance(axes, (list, np.ndarray)):        # Handle single or multiple axes input
            axes = [axes]

        locator_kwargs = locator_kwargs or {}               # Ensure locator_kwargs is always a valid dictionary

        colorbars = []                                      # Add a colorbar to each subplot axis
        if within_plot:                                     # Add colorbar as a legend-like object within the plot
            for ax in axes:
                # Set up a proxy patch for the colorbar
                proxy_patch = Rectangle((0, 0), 1, 1, fc=cmap(0.5), edgecolor="none")
                legend = Legend(ax, [proxy_patch], [''], loc='best', handler_map={proxy_patch: HandlerPatch()}, **locator_kwargs)
                ax.add_artist(legend)
        else:                                               # Add colorbar to the side of the plot
            if len(axes) > 1 or not single:                # When more than one axis is present, combine them for a unified colorbar
                                                            # Ensure `size` is numerical, converting percentage strings if necessary
                bbox = axes[0].get_position()               # Determine the bounding box for all the axes
                for ax in axes[1:]:
                    other_bbox = ax.get_position()
                    bbox = Bbox.union([bbox, other_bbox])  # Combine the bounding boxes of all axes 

                # Create a new axis for the colorbar based on the combined bounding box
                fig_width, fig_height = fig.get_size_inches()
                
                if isinstance(size, str) and size.endswith('%'):
                    size = float(size.strip('%')) / 100 * fig_height
                
                # Determine the position and orientation of the colorbar
                if position in ['right', 'left']:
                    if position == 'right':
                        cax_position = [bbox.x1 + pad / fig_width, bbox.y0, size / fig_width, bbox.height]
                    else:  # 'left'
                        cax_position = [bbox.x0 - size / fig_width - pad / fig_width, bbox.y0, size / fig_width, bbox.height]
                    orientation = 'vertical'
                elif position in ['top', 'bottom']:
                    if position == 'top':
                        cax_position = [bbox.x0, bbox.y1 + pad / fig_height, bbox.width, size / fig_height]
                    else:  # 'bottom'
                        cax_position = [bbox.x0, bbox.y0 - size / fig_height - pad / fig_height, bbox.width, size / fig_height]
                    orientation = 'horizontal'
                else:
                    raise ValueError(f"Invalid position '{position}'. Use 'right', 'left', 'top', 'bottom'.")

                # Add colorbar that spans across the axes
                cax     = fig.add_axes(cax_position)
                cbar    = fig.colorbar(sm, cax=cax, orientation=orientation, *args, **kwargs)
                cbar.ax.set_title(title, pad=10 if position in ['top', 'bottom'] else 5)
                colorbars.append(cbar)
            else:                                           # When there is only one axis, add a colorbar to it
                for ax in axes:
                    divider = make_axes_locatable(ax)
                    if position in ['right', 'left']:
                        cax = divider.append_axes(position, size=size, pad=pad)
                    elif position in ['top', 'bottom']:
                        cax = divider.append_axes(position, size=size, pad=pad)
                    else:
                        raise ValueError(f"Invalid position '{position}'. Use 'right', 'left', 'top', 'bottom'.")

                    # Add colorbar to the created axis
                    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal' if position in ['top', 'bottom'] else 'vertical', *args, **kwargs)
                    cbar.ax.set_title(title, pad=10 if position in ['top', 'bottom'] else 5)
                    colorbars.append(cbar)
        
        for cbar in colorbars:
            # set the labels
            if xlabel is not None:
                cbar.ax.set_xlabel(xlabel)
                # adjust tick positions
                if xlabelpos == 'top':
                    cbar.ax.xaxis.set_label_coords(0.5, 1.1)
                if xlabelcords is not None and isinstance(xlabelcords, tuple):
                    cbar.ax.xaxis.set_label_coords(xlabelcords[0], xlabelcords[1])
                if rotation is not None:
                    cbar.ax.xaxis.label.set_rotation(rotation)
            if xtickpos is not None:
                cbar.ax.xaxis.set_ticks_position(xtickpos)
            if xtickscords is not None and isinstance(xtickscords, tuple):
                cbar.ax.xaxis.set_ticks_position(xtickscords)
            
            if ylabel is not None:
                cbar.ax.set_ylabel(ylabel)
                if ylabelpos == 'right':
                    cbar.ax.yaxis.set_label_coords(1.1, 0.5)
                else:
                    cbar.ax.yaxis.set_label_coords(-0.1, 0.5)
                if ylabelcords is not None and isinstance(ylabelcords, tuple):
                    cbar.ax.yaxis.set_label_coords(ylabelcords[0], ylabelcords[1])
                if rotation is not None:
                    cbar.ax.yaxis.label.set_rotation(rotation)
                    
            if ytickpos is not None:
                cbar.ax.yaxis.set_ticks_position(ytickpos)
            # adjust tick positions
            if ytickscords is not None and isinstance(ytickscords, tuple):
                cbar.ax.yaxis.set_ticks_position(ytickscords)
            # unset ticks if specified
            if xticks is not None and xticks == []:
                # cbar.ax.set_xticks([])
                cbar.ax.set_xticklabels([])
            if yticks is not None and yticks == []:
                # cbar.ax.set_yticks([])
                cbar.ax.set_yticklabels([])
        return colorbars    
    
    @staticmethod
    def add_colorbar_pos(fig, 
                        cmap,
                        mapable,
                        pos,
                        norm = None,
                        vmin = None, 
                        vmax = None,
                        orientation = 'vertical', 
                        xlabel      = '',
                        xlabelcords = (0.5, -0.05),
                        ylabel      = '',
                        ylabelcords = (-0.05, 0.5),
                        xticks      = None,
                        yticks      = None,
                        ):
        '''
        Add colorbar to the plot.
        - fig :   figure to add the colorbar to
        - cmap      :   colormap to use
        - mapable   :   mapable object
        - pos       :   position of the colorbar
        - norm      :   normalization of the colorbar
        - vmin      :   minimum value
        - vmax      :   maximum value
        - orientation:  orientation of the colorbar
        - xlabel    :   xlabel for the colorbar
        - xlabelcords:  coordinates of the xlabel
        - ylabel    :   ylabel for the colorbar
        - ylabelcords:  coordinates of the ylabel
        - xticks    :   ticks for the x-axis
        - yticks    :   ticks for the y-axis
        '''
        if norm is None:
            norm = mpl.colors.Normalize(vmin = vmin if vmin is not None else np.min(mapable), 
                                        vmax = vmax if vmax is not None else np.max(mapable))
        
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # create axes 
        cax     = fig.add_axes(pos)
        cbar    = fig.colorbar(sm, cax = cax, aspect = 1, orientation = orientation)
        
        # set the labels
        if xlabel != '':
            cbar.ax.set_xlabel(xlabel)
            cbar.ax.xaxis.set_label_coords(xlabelcords[0], xlabelcords[1])
        
        if ylabel != '':
            cbar.ax.set_ylabel(ylabel)
            cbar.ax.yaxis.set_label_coords(ylabelcords[0], ylabelcords[1])
        
        # set the ticks
        if xticks is not None:
            cbar.ax.set_xticks(xticks)
        if yticks is not None:
            cbar.ax.set_yticks(yticks)
    
    ###########################################################
    
    @staticmethod
    def get_colormap(values, cmap='PuBu', elsecolor='blue'):
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
        """
        norm = Normalize(np.min(values), np.max(values))
        colors = plt.get_cmap(cmap)
        values = np.sort(values)
        getcolor = lambda x: colors((x - values[0]) / (values[-1] - values[0])) if len(values) != 1 else elsecolor
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
        norm = Normalize(np.min(data), np.max(data))
        img = ax.imshow(data, cmap=cmap, norm=norm, **kwargs)
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
    
    ######################## A N N O T ########################
    
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
        **kwargs        
        ):
        '''
        Annotate plot with the letter.
        - ax        : axis to annotate on
        - iter      : iteration number
        - x         : x coordinate
        - y         : y coordinate
        - fontsize  : fontsize 
        '''
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
    def hline(  ax, 
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
        ax.axvline(val, 
                ls = ls,  
                lw = lw, 
                label = label if (label is not None and len(label) != 0 and labelcond) else None, 
                color = color,
                zorder = zorder,
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
        **kwargs,
    ):
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
                label       = None,
                zorder      = 5,
                labelcond   = True,
                **kwargs):
        '''
        plot the data
        '''
        if 'linestyle' in kwargs:
            ls = None
        if label is None or label == '':
            labelcond = False
        ax.plot(x, y, ls = ls, lw = lw, color = color, label = label if labelcond else '', zorder = zorder, **kwargs)

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
            label           : dict | str    = "",           # general label
            xlabel          : dict | str    = "",           # x label - works when which is 'x' or 'both'
            ylabel          : str           = "",           # y label - works when which is 'y' or 'both'
            title           : str           = "",           # title of the axis
            scale           : str  | dict   = "linear",     # scale of the axis
            xscale                          = None,         # scale of the x-axis
            yscale                          = None,         # scale of the y-axis
            lim             : dict | tuple | None = None,   # fallback for axis limits
            xlim                            = None,         # specific limits for x-axis
            ylim                            = None,         # specific limits for y-axis
            fontsize        : int           = plt.rcParams['font.size'],
            labelPad        : float         = 0.0,          # padding for axis labels
            labelCond       : bool | dict   = True,
            labelPos        : dict | str | None = None,     # label positions
            xlabelPos       : str  | None   = None,         # position of the xlabel
            ylabelPos       : str  | None   = None,         # position of the ylabel
            tickPos         : dict | str | None = None,     # tick positions
            labelCords      : dict | str | None = None,     # manual label coordinates
            ticks           : dict | str | None = None,     # custom ticks
            labels          : dict | str | None = None,     # custom tick labels
            maj_tick_l      : float         = 2,
            min_tick_l      : float         = 1,
            tick_params     : dict | str | None = None,
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
    def set_smart_lim(
        ax,
        *,
        which           : str = "both",             # "x", "y", or "both"
        data            : np.ndarray | None = None, # 1-D or 2-D; if None infer from artists
        margin_p        : float = 0,                # log_scale moves
        margin_m        : float = 1,                # log_scale moves
        xlim            : tuple | None = None,      # x limits
        ylim            : tuple | None = None,      # y limits
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
        
        ax2 = plt.axes((0, 0, 1, 1), **kwargs, zorder=zorder)
        ip  = InsetPosition(ax, position)
        ax2.set_axes_locator(ip)
        
        if add_box:
            # Add a semi-transparent white box around the inset
            rect = Rectangle((0, 0), 1 + box_ext, 1 + box_ext, transform=ax2.transAxes, 
                            facecolor=facecolor, edgecolor='none', alpha=box_alpha, zorder=zorder)
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
    def get_subplots(   nrows  =   1,
                        ncols  =   1,
                        sizex  =   10.,
                        sizey  =   10.,
                        **kwargs):
        figsize             = kwargs.get('figsize', (sizex, sizey))
        kwargs['figsize']   = figsize
        
        if ncols == 1 and nrows == 1:
            fig, ax = plt.subplots(nrows, ncols, **kwargs)
            return fig, [ax]
        elif (ncols == 1 and nrows > 1) or (nrows == 1 and ncols > 1):
            return plt.subplots(nrows, ncols, **kwargs)
        else:
            fig, ax = plt.subplots(nrows, ncols, **kwargs)
            return fig, [axis for row in ax for axis in row]
        
    ######### S A V I N G #########

    @staticmethod
    def save_fig(directory  :   str,
                 filename   :   str,
                 format     =   'pdf',
                 dpi        =   200,
                 adjust     =   True,
                 **kwargs):
        '''
        Save figure to a specific directory. 
        - directory : directory to save the file
        - filename  : name of the file
        - format    : format of the file
        - dpi       : dpi of the file
        - adjust    : adjust the figure
        '''
        if adjust:
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(directory + filename, format = format, dpi = dpi, bbox_inches = 'tight', **kwargs)
        
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