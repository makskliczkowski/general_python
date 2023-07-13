import itertools
import scienceplots
########################## matplotlib ##########################
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.facecolor']      =   'white'
plt.rcParams['savefig.facecolor']   =   'w'

########################## style
SMALL_SIZE                          =   12
MEDIUM_SIZE                         =   14
BIGGER_SIZE                         =   16
#plt.rc('font', size=MEDIUM_SIZE)                                       # controls default text sizes
plt.rc('axes'   , titlesize=MEDIUM_SIZE , labelsize=MEDIUM_SIZE )       # fontsize of the axes title
plt.rc('xtick'  , labelsize=SMALL_SIZE  , direction='in'        )       # fontsize of the tick labels
plt.rc('ytick'  , labelsize=SMALL_SIZE  , direction='in'        )       # fontsize of the tick labels
plt.rc('legend' , fontsize=SMALL_SIZE   , loc = 'best'          )       # legend fontsize
plt.rc('figure' , titlesize=BIGGER_SIZE                         )       # fontsize of the figure title
mpl.rcParams['mathtext.fontset']    = 'stix'
mpl.rcParams['font.family']         = 'STIXGeneral'
plt.style.use(['science', 'no-latex'])


colorsList                          =   (list(mcolors.TABLEAU_COLORS))
colorsCycle                         =   itertools.cycle(list(mcolors.TABLEAU_COLORS))
markersList                         =   ['o','s','v', '+', 'o', '*']
markersCycle                        =   itertools.cycle(markersList)

