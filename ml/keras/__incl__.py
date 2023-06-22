######## T E N S O R F L O W ########
import tensorflow as tf

TF_TYPE = tf.float64
############# TENSORFLOW ############

if TF_TYPE == tf.float64:
    tf.keras.backend.set_floatx('float64')
    policy = tf.keras.mixed_precision.Policy("float64")
    tf.keras.mixed_precision.set_global_policy(policy)

import tensorflow_probability as tfp

########## S K   L E A R N ##########
from sklearn.model_selection import train_test_split
############# K E R A S #############
from tensorflow import keras
import gc

# import models
from tensorflow.keras.models import Sequential

# import optimizers
from tensorflow.keras.optimizers.legacy import Adam, RMSprop
from tensorflow.keras.optimizers.legacy import SGD

# import layers
from tensorflow.keras import layers, losses, Input, Model, callbacks
from tensorflow.keras.layers import Dense, LeakyReLU, ELU, BatchNormalization, Input, LayerNormalization, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling1D, Conv2DTranspose, Dropout, Dense, Input, BatchNormalization, AveragePooling2D, MaxPooling2D, MaxPooling1D, Flatten, GlobalAveragePooling2D, Dot, Lambda, Reshape, Conv1DTranspose
from tensorflow.keras.layers import DenseFeatures, DepthwiseConv1D, ConvLSTM1D,LocallyConnected1D
from tensorflow.keras.layers import TimeDistributed, Conv1D, SpatialDropout1D, Conv1DTranspose, LSTM, UpSampling2D, Average, Embedding, Bidirectional,RepeatVector
from keras.layers import CuDNNLSTM, ConvLSTM1D, UpSampling1D

# import activations
from tensorflow.keras.activations import relu, softmax, elu

# import backend
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from plot_keras_history import show_history, plot_history

# constraints
from keras.constraints import max_norm, MinMaxNorm

# from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

# import losses
from tensorflow.keras.losses import mean_squared_error, mean_squared_logarithmic_error , KLDivergence, log_cosh, huber, categorical_crossentropy, mean_absolute_error, binary_crossentropy, sparse_categorical_crossentropy, BinaryCrossentropy, Poisson, Hinge, CategoricalCrossentropy

""" TQDM """
from tqdm.keras import TqdmCallback

#####################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#####################################
from .__general__ import *

######################################################## CALLBACK ########################################################

''' 
Evaluation callback
'''
class EvalCallback(callbacks.Callback):
    def __init__(self, val_data_generator = None):
        self.val_generator = val_data_generator
        self.eval_inter =  2

    def on_epoch_end(self, epoch: int, logs=None):
        # Analyze every 'n' epochs
        if (epoch + 1) % self.eval_inter == 0:
            keys = list(logs.keys())
            if self.val_generator is not None:
                metrics = self.model.evaluate(self.val_generator, verbose=0)
            #logs["val_loss"] = # val loss value
            #logs["val_metric"] = # val metric value
        # Housekeeping
        gc.collect()
        K.clear_session()

'''
Giving callbacks inline
'''
def giveCallbacks(savename, save = False, verbose = False, gc = False, patience = 40, monitor='val_loss'):
    # making callbacks
    early_stopping_cb   = callbacks.EarlyStopping(patience=patience, monitor=monitor)
    callback            = [early_stopping_cb] if patience is not None else []
    if gc:
        callback.append(EvalCallback())
    if save:
        # save best models as h5 files to load that later
        callback.append(keras.callbacks.ModelCheckpoint(savename + ".h5", save_best_only=True))
    
    callback.append(TqdmCallback(verbose=verbose))
    return callback

######################################################## RESET ########################################################

'''
Hard memory reset
'''
def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    # clear session
    tf.compat.v1.keras.backend.clear_session()
    # close session
    sess.close()
    # open new session
    sess = tf.compat.v1.keras.backend.get_session()
    # garbage collector
    gc.collect()


    #print(K.gc.collect()) # if it's done something you should see a number being outputted
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    
