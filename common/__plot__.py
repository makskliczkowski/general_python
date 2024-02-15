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

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.facecolor']      =   'white'
plt.rcParams['savefig.facecolor']   =   'w'

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
    
    #################### A N N O T ####################
    
    @staticmethod
    def set_annotate(   ax,
                        elem    : str,
                        x       : float,
                        y       : float,
                        fontsize= None,
                        xycoords= 'axes fraction',
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
        ax.annotate(elem, xy=(x, y), fontsize=fontsize, xycoords=xycoords, **kwargs)

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
        y   =   funct(x)
        ax.plot(x, y, **kwargs)
    
    #################### L I N E S ####################
    
    @staticmethod
    def hline(  ax, 
                val     : float,
                ls      = '-',
                lw      = 2,
                color   = 'black',
                label   = None,
                **kwargs):
        '''
        HLINE
        '''
        ax.axhline(val, ls = ls,  lw = lw, 
                label = label if (label is not None and len(label) != 0) else None, 
                color = color, 
                **kwargs)
    
    @staticmethod
    def vline(  ax, 
                val     : float,
                ls      = '-',
                lw      = 2,
                color   = 'black',
                label   = None,
                **kwargs):
        '''
        VLINE
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
                        **kwargs
                        ):
        '''
        Sets tickparams to the desired ones
        '''
        ax.tick_params(axis='both', which='major', left=left, right=right, top=top, bottom=bottom, labelsize=labelsize)
        ax.tick_params(axis="both", which='major', left=left, right=right, top=top, bottom=bottom, direction="in",length=6, **kwargs)
        ax.tick_params(axis="both", which='minor', left=left, right=right, top=top, bottom=bottom, direction="in",length=3)

        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

    @staticmethod
    def set_ax_params(  ax, 
                        which       :   str,
                        label       =   "",
                        labelPad    =   0.0,
                        lim         =   None,
                        title       =   '',
                        fontsize    =   None,
                        scale       =   'linear'):
        '''
        Sets the parameters of the axes
        '''
        # check x axis
        if 'x' in which:
            if label != "":
                ax.set_xlabel(label, 
                            fontsize = fontsize,
                            labelpad = labelPad if labelPad != 0 else None)
            if lim is not None:
                ax.set_xlim(lim[0], lim[1])
            ax.set_xscale(scale)
            if scale == 'log':
                ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))

        # check y axis
        if 'y' in which:
            if label != "":
                ax.set_ylabel(label, 
                            fontsize = fontsize,
                            labelpad = labelPad if labelPad != 0 else None)
            if lim is not None:
                ax.set_ylim(lim[0], lim[1])             
            ax.set_yscale(scale)
            if scale == 'log':
                ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))
            
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
        if 'both' in which:
            ax.xaxis.set_label_coords(inX, inY, **kwargs)
            ax.yaxis.set_label_coords(inX, inY, **kwargs)
    
    @staticmethod
    def unset_spines(   ax,
                        xticks      =   False,
                        yticks      =   False,
                        left        =   False,
                        right       =   False,
                        top         =   False,
                        bottom      =   False
                     ):
        '''
        Disables the spines on the axis
        '''
        ax.spines['top'].set_visible(top)
        ax.spines['right'].set_visible(right)
        ax.spines['bottom'].set_visible(bottom)
        ax.spines['left'].set_visible(left)
        if not xticks:
            plt.setp(ax.get_xticklabels(), visible=False)
        if not yticks:
            plt.setp(ax.get_yticklabels(), visible=False)

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
        if not xticks and erease:
            ax.set_xticks([])
        if not yticks and erease:
            ax.set_yticks([])            
        Plotter.unset_spines(ax,    xticks = xticks, 
                                    yticks = yticks, 
                                    left = not ((not spines) and (not yticks)), 
                                    right = not((not spines) and (not yticks)), 
                                    top = not ((not spines) and (not xticks)), 
                                    bottom = not ((not spines) and (not xticks)))
        
    ################### F O R M A T ###################
        
    @staticmethod
    def set_formater(ax, 
                     formater = "%.1e",
                     axis     = 'both'):
        """
        Sets the formatter for the given axis on the plot.
        Args:
            ax (object): The axis object on which to set the formatter.
            formater (str, optional): The format string for the axis labels. Defaults to "%.1e".
            axis (str, optional): The axis on which to set the formatter. Defaults to 'both'.
        Returns:
            None
        """
        if axis == 'y' or axis == 'both':
            ax.yaxis.set_major_formatter(MathTextSciFormatter(formater))
        if axis == 'x' or axis == 'both':
            ax.xaxis.set_major_formatter(MathTextSciFormatter(formater))
              
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
    def get_grid_subplot(gs,
                         fig,    
                         i      :   int, 
                         sharex =   None, 
                         sharey =   None):
        '''
        Creates the subaxis for the GridSpec
        '''
        return fig.add_subplot(gs[i], sharex = sharex, sharey = sharey)

    #################### I N S E T ####################

    @staticmethod
    def get_inset(ax, 
                  position = [0.0, 0.0, 1.0, 1.0]):
        ax2 = plt.axes((0, 0, 1, 1))
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
                #   alignment     = alignment, 
                  markerfirst   = markerfirst,
                  framealpha    = framealpha,
                  **kwargs)  
        
    ######### S U B A X S #########

    @staticmethod
    def get_subplots(   nrows  :   int,
                        sizex  =   10,
                        sizey  =   10,
                        ncols  =   1,
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
                 **kwargs):
        '''
        Save figure to a specific directory
        '''
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(directory + filename, format = format, dpi = dpi, bbox_inches = 'tight', **kwargs)
        

##########################################################################


import numpy as np
    
    
class PlotterSave:
    
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
            np.save(directory + fileName + ".npy", toSave)
        elif typ == '.txt':
            np.savetxt(directory + fileName + ".npy", toSave)
