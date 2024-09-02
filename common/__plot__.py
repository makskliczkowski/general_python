import itertools
import scienceplots
########################## matplotlib ##########################
import latex
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

########################## grids
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.ticker as mticker
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

import numpy as np
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.facecolor']      =   'white'
plt.rcParams['savefig.facecolor']   =   'w'

########################## labellines
from labellines import labelLine, labelLines

########################## style
SMALL_SIZE                          =   12
MEDIUM_SIZE                         =   14
BIGGER_SIZE                         =   16
ADDITIONAL_LINESTYLES = {
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
     'densely dashdotdotted' : (0, (3, 1, 1, 1, 1, 1))
}

try:
    plt.style.use(['science', 'no-latex', 'colors5-light'])
except Exception as e:
    print(e)
mpl.rcParams['mathtext.fontset']    = 'stix'
mpl.rcParams['font.family']         = 'STIXGeneral'

# plt.rcParams['text.usetex']         = True
# latex_engine                        = 'pdflatex'
# latex_elements                      = {
#                                         'extrapackages': r'\usepackage{physics}',
#                                         'extrapackages': r'\usepackage{amsmath}'
#                                     }

colorsList                          =   (list(mcolors.TABLEAU_COLORS))
colorsCycle                         =   itertools.cycle(list(mcolors.TABLEAU_COLORS))
colorsCyclePlastic                  =   itertools.cycle(list(["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
markersList                         =   ['o','s','v', '+', 'o', '*']
markersCycle                        =   itertools.cycle(["4", "2", "3", "1", "+", "x", "."] + markersList)

########################## functions ##########################

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

########################### plotter ###########################

class Plotter:
    """ 
    A Plotter class that handles the methods of plotting.
    """
    
    @staticmethod
    def get_figsize(columnwidth, wf = 0.5, hf = (5.**0.5-1.0) / 2.0):
      """
      Parameters:
        - wf [float]:  width fraction in columnwidth units
        - hf [float]:  height fraction in columnwidth units.
                       Set by default to golden ratio.
        - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                               using \showthe\columnwidth
      Returns:  [fig_width,fig_height]: that should be given to matplotlib
      """
      fig_width_pt  = columnwidth * wf 
      inches_per_pt = 1.0 / 72.27                       # Convert pt to inch
      fig_width     = fig_width_pt * inches_per_pt      # width in inches
      fig_height    = fig_width * hf                # height in inches
      return [fig_width, fig_height]
    
    @staticmethod 
    def get_color(color,
                  alpha = None,
                  edgecolor = (0,0,0,1), 
                  facecolor = (1,1,1,0)
                  ):
        dictionary = dict(facecolor = facecolor, edgecolor = edgecolor)
        if alpha is not None:
            dictionary['alpha'] = alpha
        return dictionary
    
    ################### C O L O R S ###################

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
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # add colorbar
        cbar    = fig.colorbar(sm, ax = axes, *args, **kwargs)
        cbar.ax.set_title(title)
        return cbar
    
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
                         ):
        '''
        Add colorbar to the plot.
        - axes      :   axis to add the colorbar to
        - fig       :   figure to add the colorbar to
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
            cbar.set_label(xlabel)
            cbar.ax.xaxis.set_label_coords(xlabelcords[0], xlabelcords[1])
        
        if ylabel != '':
            cbar.set_label(ylabel)
            cbar.ax.yaxis.set_label_coords(ylabelcords[0], ylabelcords[1])
    
    @staticmethod
    def get_colormap(values, cmap = 'PuBu', elsecolor = 'blue'):
        norm        = plt.Normalize(np.min(values), np.max(values))
        colors      = plt.get_cmap(cmap)
        values      = np.sort(values)
        getcolor    = lambda x: colors((x - values[0]) / (values[-1] - values[0])) if len(values) != 1 else elsecolor
        return getcolor, colors, norm
    
    #################### A N N O T ####################
    
    @staticmethod
    def set_annotate(   ax,
                        elem    : str,
                        x       : float,
                        y       : float,
                        fontsize= None,
                        xycoords= 'axes fraction',
                        cond    = True,
                        **kwargs):
        '''
        @staticmethod
        
        Make an annotation on the plot.
        - ax        :   axis to annotate on
        - elem      :   annotation string
        - x         :   x coordinate
        - y         :   y coordinate
        - fontsize  :   fontsize
        - xycoords  :   how to interpret the coordinates (from MPL)
        '''
        if cond:
            ax.annotate(elem, xy=(x, y), fontsize=fontsize, xycoords=xycoords, **kwargs)

    @staticmethod
    def set_annotate_letter(
        ax, 
        iter        : int,
        x           : float,
        y           : float,
        fontsize    = 12,
        addit       = '',
        condition   = True,
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
        Plotter.set_annotate(ax, elem = f'({chr(97 + iter)})' + addit, x = x, y = y, fontsize = fontsize, condition = condition, **kwargs)
    
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
        ax[0].annotate(start_T, 
                   xy           =   xystart, 
                   xytext       =   xystart_T, 
                   arrowprops   =   arrowprops, 
                   color        =   startcolor)
        
        ax[0].annotate(end_T,
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
                ls      = '-',
                lw      = 2.0,
                color   = 'black',
                label   = None,
                **kwargs):
        '''
        horizontal line plotting
        '''
        ax.axhline(val, ls = ls,  lw = lw, 
                label = label if (label is not None and len(label) != 0) else None, 
                color = color, 
                **kwargs)
    
    @staticmethod
    def vline(  ax, 
                val     : float,
                ls      = '-',
                lw      = 2.0,
                color   = 'black',
                label   = None,
                **kwargs):
        '''
        vertical line plotting
        '''
        ax.axvline(val, 
                ls = ls,  
                lw = lw, 
                label = label if (label is not None and len(label) != 0) else None, 
                color = color,
                **kwargs)
    
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
    def set_ax_params(  ax, 
                        which       :   str,
                        scale       =   'linear',
                        label       =   "",
                        labelPad    =   0.0,
                        lim         =   None,
                        fontsize    =   None,
                        title       =   '',
                        labelCond   =   True,
                        # manual
                        labelPos    =   None,
                        tickPos     =   None,
                        labelCords  =   None,
                        ticks       =   None,
                        labels      =   None,
                        maj_tick_l  =   2,
                        min_tick_l  =   1):
        '''
        Sets the parameters of the axes
        - ax        : axis to use
        - which     : string, x, y, xy
        - scale     : linear, log
        - label     : label for the axis
        - labelPad  : label padding
        - lim       : limits for the axis
        - fontsize  : font size
        - title     : title for the axis
        '''
        
        # check x axis
        if 'x' in which:
            # set the ticks
            if True:
                Plotter.set_ax_params(ax, 'x', scale = scale, maj_tick_l = maj_tick_l, min_tick_l = min_tick_l)
            
            if label != "":
                ax.set_xlabel(label if labelCond else "", 
                            fontsize = fontsize,
                            labelpad = labelPad if labelPad != 0 else None)
            # check limits
            if lim is not None:
                ax.set_xlim(lim[0], lim[1])
                
            # set the scale
            ax.set_xscale(scale)
            if scale == 'log':
                ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))
            
            # manual settings          
            if True:
                if labelPos is not None and labelPos in ['bottom', 'top']:
                    ax.xaxis.set_label_position(labelPos)
                    
                # ticks :)
                if tickPos is not None and tickPos in ['bottom', 'top']:
                    if tickPos == 'top':
                        ax.xaxis.tick_top()
                        
                if labelCords is not None:
                    ax.xaxis.set_label_coords(labelCords[0], labelCords[1])

            if ticks is not None:
                ax.set_xticks(ticks)
                # try to set the labels
                if labels is not None and len(labels) == len(ticks):
                    ax.set_xticklabels(labels)
            
        # check y axis
        if 'y' in which:
            
            # set the ticks
            if True:
                Plotter.set_ax_params(ax, 'y', scale = scale, maj_tick_l = maj_tick_l, min_tick_l = min_tick_l)
                
            if label != "":
                ax.set_ylabel(label if labelCond else "", 
                            fontsize = fontsize,
                            labelpad = labelPad if labelPad != 0 else None)
            if lim is not None:
                ax.set_ylim(lim[0], lim[1])             
            ax.set_yscale(scale)
            if scale == 'log':
                ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))
            
            if True:
                # label position
                if labelPos is not None and labelPos in ['left', 'right']:
                    ax.yaxis.set_label_position(labelPos)
                    
                # ticks :)
                if tickPos is not None and tickPos in ['left', 'right']:
                    if tickPos == 'right':
                        ax.yaxis.tick_right()
                
                # coordinates  
                if labelCords is not None:
                    ax.yaxis.set_label_coords(labelCords[0], labelCords[1])
            
            if ticks is not None:
                ax.set_yticks(ticks)
                # try to set the labels
                if labels is not None and len(labels) == len(ticks):
                    ax.set_yticklabels(labels)
            
            
        # check the title
        if len(title) != 0:
            ax.set_title(title)       
    
    @staticmethod
    def set_ax_labels(  ax,
                        fontsize    =   None,
                        xlabel      =   "",
                        ylabel      =   "",
                        title       =   "",
                        xPad        =   0,
                        yPad        =   0
                      ):
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
    def unset_ticks(    ax,
                        xticks      =   False,
                        yticks      =   False,
                        erease      =   False,
                        spines      =   True
                    ):
        '''
        Disables the ticks on the axis
        '''
        if not xticks:
            ax.set_xticks([], minor=False)
            ax.set_xticklabels([], minor=False)
            ax.xaxis.set_tick_params(which='both', labelbottom = False)
            if erease:
                ax.axes.get_xaxis().set_visible(False)
        if not yticks:
            ax.set_yticks([], minor=False)
            ax.set_yticklabels([], minor=False)
            ax.yaxis.set_tick_params(which='both', labelleft = False)
            if erease:
                ax.axes.get_yaxis().set_visible(False)
            
        Plotter.unset_spines(ax,    xticks = xticks, 
                                    yticks = yticks, 
                                    left = not ((not spines) and (not yticks)), 
                                    right = not((not spines) and (not yticks)), 
                                    top = not ((not spines) and (not xticks)), 
                                    bottom = not ((not spines) and (not xticks)))
    
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
            ax.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, pos: "%g"%x))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: "%g"%x))
        if 'y' in axis:
            ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, pos: "%g"%x))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: "%g"%x))
    
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
                         i      :   int, 
                         sharex =   None, 
                         sharey =   None,
                         **kwargs):
        '''
        Creates the subaxis for the GridSpec
        '''
        return fig.add_subplot(gs[i], sharex = sharex, sharey = sharey, **kwargs)

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
                  position = [0.0, 0.0, 1.0, 1.0], **kwargs):
        ax2 = plt.axes((0, 0, 1, 1), **kwargs)
        ip  = InsetPosition(ax, position)
        ax2.set_axes_locator(ip)
        return ax2
    
    ##################### L O O K #####################
    
    @staticmethod
    def set_transparency(ax, 
                         alpha = 0.0):
        ax.patch.set_alpha(alpha)
    
    ################### L E G E N D ###################
    
    @staticmethod
    def set_legend(ax,
                   fontsize     =   None,
                   frameon      =   False,
                   loc          =   'best',
                   alignment    =   'left',
                   markerfirst  =   False,
                   framealpha   =   1.0,
                   reverse      =   False,
                   **kwargs
                   ):
        '''
        Sets the legend with preferred style
        '''
        handles, labels = ax.get_legend_handles_labels()
        if reverse:
            handles     = handles[::-1]
            labels      = labels[::-1]
            
        ax.legend(handles,
                  labels,
                  fontsize      = fontsize, 
                  frameon       = frameon, 
                  loc           = loc,
                  markerfirst   = markerfirst,
                  framealpha    = framealpha,
                  **kwargs)  
    
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
        if ncols == 1 and nrows == 1:
            fig, ax = plt.subplots(nrows, ncols, figsize = (sizex, sizey), **kwargs)
            return fig, [ax]
        elif (ncols == 1 and nrows > 1) or (nrows == 1 and ncols > 1):
            return plt.subplots(nrows, ncols, figsize = (sizex, sizey), **kwargs)
        else:
            fig, ax = plt.subplots(nrows, ncols, figsize = (sizex, sizey), **kwargs)
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
    def json2dict_multiple(directory : str,
                           keys      : list):
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

from sympy import Matrix, init_printing

class MatrixPrinter:
    
    
    def __init__(self):
        init_printing()
    
    '''
    Class for printing matrices
    '''
    @staticmethod
    def print_matrix(matrix : np.ndarray):
        '''
        Prints the matrix in a nice form
        '''
        display(Matrix(matrix))
        
