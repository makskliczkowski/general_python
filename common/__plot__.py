import itertools
import scienceplots
########################## matplotlib ##########################
import latex
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

########################## grids
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.facecolor']      =   'white'
plt.rcParams['savefig.facecolor']   =   'w'

########################## style
SMALL_SIZE                          =   12
MEDIUM_SIZE                         =   14
BIGGER_SIZE                         =   16
# plt.rc('font', size=MEDIUM_SIZE)                                       # controls default text sizes
# plt.rc('axes'   , titlesize=MEDIUM_SIZE , labelsize=MEDIUM_SIZE )       # fontsize of the axes title
# plt.rc('xtick'  , labelsize=SMALL_SIZE  , direction='in'        )       # fontsize of the tick labels
# plt.rc('ytick'  , labelsize=SMALL_SIZE  , direction='in'        )       # fontsize of the tick labels
# plt.rc('legend' , fontsize=SMALL_SIZE   , loc = 'best'          )       # legend fontsize
# plt.rc('figure' , titlesize=BIGGER_SIZE                         )       # fontsize of the figure title
mpl.rcParams['mathtext.fontset']    = 'stix'
mpl.rcParams['font.family']         = 'STIXGeneral'
# plt.rcParams['text.usetex']         = True
# latex_engine                        = 'pdflatex'
# latex_elements                      = {
#                                         'extrapackages': r'\usepackage{physics}',
#                                         'extrapackages': r'\usepackage{amsmath}'
#                                     }
# plt.style.use(['science', 'nolatex'])

colorsList                          =   (list(mcolors.TABLEAU_COLORS))
colorsCycle                         =   itertools.cycle(list(mcolors.TABLEAU_COLORS))
markersList                         =   ['o','s','v', '+', 'o', '*']
markersCycle                        =   itertools.cycle(markersList)

########################## functions ##########################

class Plotter:
    
    ########## A N N O T ##########
    
    @staticmethod
    def set_annotate(   ax,
                        elem    : str,
                        x       : float,
                        y       : float,
                        fontsize= None,
                        xycoords= 'axes fraction',
                        **kwargs):
        '''
        Make an annotation
        '''
        ax.annotate(elem, xy=(x, y), fontsize=fontsize, xycoords=xycoords, **kwargs)
    
    ########## F I T T S ##########
    
    @staticmethod
    def plot_fit(   ax,   
                    funct,
                    x,
                    **kwargs):
        y   =   funct(x)
        ax.plot(x, y, **kwargs)
    
    ########## L I N E S ##########
    
    @staticmethod
    def hline(ax, 
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
                color = color)
    
    @staticmethod
    def vline(ax, 
                val     : float,
                ls      = '-',
                lw      = 2,
                color   = 'black',
                label   = None,
                **kwargs):
        '''
        VLINE
        '''
        ax.axvline(val, ls = ls,  lw = lw, 
                label = label if (label is not None and len(label) != 0) else None, 
                color = color)
    
    ########## T I C K S ##########
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
        ax.tick_params(axis="both", which='major', left=left, right=right, top=top, bottom=bottom, direction="in",length=6)
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
                        yticks      =   False
                    ):
        '''
        Disables the ticks on the axis
        '''
        if not xticks:
            ax.set_xticks([])
        if not yticks:
            ax.set_yticks([])            
        Plotter.unset_spines(ax, xticks = xticks, yticks = yticks, left = True, right = True, top = True, bottom = True)
            
    ########## G R I D S ##########
    
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

    ########## I N S E T ##########

    @staticmethod
    def get_inset(ax, 
                  position = [0.0, 0.0, 1.0, 1.0]):
        ax2 = plt.axes((0, 0, 1, 1))
        ip  = InsetPosition(ax, position)
        ax2.set_axes_locator(ip)
        return ax2
    
    @staticmethod
    def set_transparency(ax, 
                         alpha = 0.0):
        ax.patch.set_alpha(alpha)
    
    ######### L E G E N D #########
    
    @staticmethod
    def set_legend(ax,
                   fontsize     =   None,
                   frameon      =   False,
                   loc          =   'best',
                   alignment    =   'left',
                   markerfirst  =   False,
                   framealpha   =   1,
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
    def get_subplots(   n  :   int,
                        sizex  =   10,
                        sizey  =   10,
                        **kwargs):
        if n == 1:
            fig, ax = plt.subplots(n, figsize = (sizex, sizey), **kwargs)
            return fig, [ax]
        else:
            return plt.subplots(n, figsize = (sizex, sizey), **kwargs)
        
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
        

#####################################

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
            np.save(directory + fileName + ".npy", toSave)
        elif typ == '.txt':
            np.savetxt(directory + fileName + ".npy", toSave)
        
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
