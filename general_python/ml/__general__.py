import os  
import matplotlib.pyplot as plt   

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("sklearn not found, please install it to use train_test_split.")
    train_test_split = None

try:
    from plot_keras_history import show_history, plot_history
except:
    pass
#########################################################################    
   
class MLParams:
    '''
    Parameters to use in the learning procedure. Includes the most general information about
    the MachineLearning model
    !TODO
    '''
    
    saveDir = lambda directory: directory + os.pathsep + 'weights'
    
    def __init__(self,  epo     : int   ,
                        batch   : int   , 
                        lr      : float ,
                        reg     : dict  , 
                        loss, 
                        fNum    : int   ,
                        shape   : tuple , 
                        optimizer       ,
                        saveDir         = lambda directory: directory + os.pathsep + 'weights',
                        trainSize       = 0.7, 
                        early_stopping  = 20):
        """
        Initializes The Machine Learning Parameters class
        Args:
            epo (int)           : number of epochs to be used
            batch (int)         : the batch size
            lr (float)          : learning rate for the optimizer
            reg (dict)          : regression parameters
            loss                : loss function for the model
            fNum (int)          : number of files to read
            shape (tuple)       : shape of the input data
            optimizer (_type_)  : optimizer
            saveDir             : directory to be the weights to
            trainSize           : training percentage from the database
            early_stopping      : Defaults to 20.
        """

        self.epo            = epo
        self.batch          = batch
        self.lr             = lr
        self.loss           = loss                          
        self.reg            = reg                              
        self.fNum           = fNum
        self.trainSize      = trainSize
        self.early_stopping = early_stopping
        
        self.saveDir        = saveDir
        self.optimizer      = optimizer
        # get the shape of the basic input
        self.shape          = shape

        self.history        = {}
        
    def plotHistory(self, logscale = True, fig = 2):
        """
        Plots the history of the training
        Args:
            - logscale (bool, optional): Defaults to True.
            - fig (int, optional): new figure index 
        """
                    # plot history
        plt.figure(fig)
        plot_history(self.history, log_scale_metrics = logscale)    

#########################################################################

    
