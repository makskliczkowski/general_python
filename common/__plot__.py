import itertools
import scienceplots
########################## matplotlib ##########################
import latex
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

########################## grids
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.facecolor']      =   'white'
plt.rcParams['savefig.facecolor']   =   'w'

########################## style
SMALL_SIZE                          =   12
MEDIUM_SIZE                         =   14
BIGGER_SIZE                         =   16
plt.rc('font', size=MEDIUM_SIZE)                                       # controls default text sizes
plt.rc('axes'   , titlesize=MEDIUM_SIZE , labelsize=MEDIUM_SIZE )       # fontsize of the axes title
plt.rc('xtick'  , labelsize=SMALL_SIZE  , direction='in'        )       # fontsize of the tick labels
plt.rc('ytick'  , labelsize=SMALL_SIZE  , direction='in'        )       # fontsize of the tick labels
plt.rc('legend' , fontsize=SMALL_SIZE   , loc = 'best'          )       # legend fontsize
plt.rc('figure' , titlesize=BIGGER_SIZE                         )       # fontsize of the figure title
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
    
    ########## T I C K S ##########
    
    def set_tickparams( ax,
                        labelsize   =   MEDIUM_SIZE,
                        left        =   True,
                        right       =   True,
                        top         =   True,
                        bottom      =   True,
                        **kwargs
                        ):
        '''
        Sets tickparams to the desired ones
        '''
        ax.tick_params(axis='both',which='major', left=left, right=right, top=top, bottom=bottom, labelsize=labelsize)
        ax.tick_params(axis="both",which='major', left=left, right=right, top=top, bottom=bottom, direction="in",length=6)
        ax.tick_params(axis="both",which='minor', left=left, right=right, top=top, bottom=bottom, direction="in",length=3)
    
    def set_ax_params(  ax, 
                        which       :   str,
                        label       :   str,
                        labelPad    =   0,
                        lim         =   None,
                        title       =   '',
                        fontsize    =   MEDIUM_SIZE):
        '''
        Sets the parameters of the axes
        '''
        # check x axis
        if 'x' in which:
            ax.set_xlabel(label, 
                          fontsize = fontsize,
                          labelpad = labelPad if labelPad != 0 else None)
            if lim is not None:
                ax.set_xlim(lim[0], lim[1])
        # check y axis
        if 'y' in which:
            ax.set_ylabel(label, 
                          fontsize = fontsize,
                          labelpad = labelPad if labelPad != 0 else None)
            if lim is not None:
                ax.set_ylim(lim[0], lim[1])             
                
        # check the title
        if len(title) != 0:
            ax.set_title(title)       
    
    def set_ax_labels(  ax,
                        fontsize    =   MEDIUM_SIZE,
                        xlabel      =   "",
                        ylabel      =   "",
                        title       =   "",
                        xPad        =   0,
                        yPad        =   0
                      ):
        '''
        Sets the labels of the x and y axes
        '''
        ax.set_xlabel(xlabel, 
                        fontsize = fontsize,
                        labelpad = xPad if xPad != 0 else None)
        ax.set_ylabel(ylabel, 
                        fontsize = fontsize,
                        labelpad = yPad if yPad != 0 else None)
        # check the title
        if len(title) != 0:
            ax.set_title(title)    
    
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
            ax.set_xticks([])
        if not yticks:
            ax.set_yticks([])

    ########## G R I D S ##########

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
    
    def get_grid_subplot(gs,
                         fig,    
                         i      :   int, 
                         sharex =   None, 
                         sharey =   None):
        '''
        Creates the subaxis for the GridSpec
        '''
        return fig.add_subplot(gs[i], sharex = sharex, sharey = sharey)

    ######### L E G E N D #########
    
    def set_legend(ax,
                   fontsize     =   16,
                   frameon      =   False,
                   loc          =   'best',
                   alignment    =   'left',
                   markerfirst  =   False,
                   **kwargs
                   ):
        '''
        Sets the legend with preferred style
        '''
        ax.legend(fontsize      = fontsize, 
                  frameon       = frameon, 
                  loc           = loc,
                  alignment     = alignment, 
                  markerfirst   = markerfirst,
                  **kwargs)  
        
    ######### S U B A X S #########

    def get_subplots(n  :   int,
                     sizex  =   10,
                     sizey  =   10,
                     **kwargs):
        if n == 1:
            fig, ax = plt.subplots(n, figsize = (sizex, sizey), **kwargs)
            return fig, [ax]
        else:
            return plt.subplots(n, figsize = (sizex, sizey), **kwargs)