import os

'''
Parameters to use in the learning procedure
'''
class ML_params:
    def __init__(self,  epo     : int   , batch : int   , lr    : float,
                        reg     : dict  , loss  : str, 
                        fNum    : int   ,
                        shape   : tuple , 
                        optimizer,
                        saveDir         = lambda saveDir: saveDir + os.pathsep + 'weights',
                        trainSize       = 0.7           , early_stopping = 20):
        self.epo            = epo
        self.batch          = batch
        self.lr             = lr
        self.loss           = loss
        self.reg            = reg                               # regularizations
        
        self.fNum           = fNum
        self.trainSize      = trainSize
        self.early_stopping = early_stopping
        
        self.saveDir = saveDir
        self.optimizer = optimizer
        # get the shape of the basic input
        self.shape = shape
    
    '''
    Save the weights for the model to hdf5 file
    '''    
    def set_weights(directory):
        pass
    
    '''
    Load the weights for the model from hdf5 file
    '''
    def load_hdf5(directory):
        pass
    

#########################################################################

