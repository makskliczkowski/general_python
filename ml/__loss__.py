from .keras.__reg__ import *

############################# MAHALONOBIS ############################

def mahalonobisCH(CholeskyW, rCond = 1):
    '''
    Returns the Mahalonobis distance between the variables.
    Uses the Cholesky decomposition calculated beforehand.
    '''
    def lossfun(y_true, y_pred):
        y_t     = tf.reduce_mean(y_true, axis = -1)
        y_p     = tf.reduce_mean(y_pred, axis = -1)
        
        out     = tf.linalg.matvec(CholeskyW, y_t - y_p)
        # tf.print(tf.shape(out))
        out     = tf.math.reduce_euclidean_norm(out, axis=1)# / tf.math.reduce_variance(y_t, axis = 1) / tf.cast(tf.shape(out)[1], dtype = tf.float64)
        # tf.print(out)
        return tf.math.sqrt(tf.math.sqrt(rCond) * tf.reduce_mean(out)) + mean_absolute_error(y_t, y_p)
    return lossfun 

def mahalonobis(y_true, y_pred):
    '''
    Returns the Mahalonobis distance between the variables
    '''
    y_t     = tf.reduce_mean(y_true, axis = -1)
    y_p     = tf.reduce_mean(y_pred, axis = -1)
    # create the covariance matrix
    cov     = tfp.stats.covariance(y_t)
    # tf.print(tf.shape(cov))    
    covi    = tf.linalg.inv(cov)
    W       = tf.linalg.cholesky(covi)
    # tf.print(tf.shape(W))
    out     = tf.linalg.matvec(W, y_t - y_p)
    # out     = tf.linalg.matvec(W, tf.reduce_mean(y_t, axis = 0)) - tf.linalg.matvec(W, y_p)
    # tf.print(tf.shape(out))
    out     = tf.math.reduce_euclidean_norm(out, axis=1) / tf.math.reduce_std(y_t, axis = 1)
    # tf.print(out)
    return tf.math.sqrt(tf.reduce_mean(out))

def mahalonobis_pseudo(Kinv):
    '''
    Use the Mahalonobis distance with the pseudoinverse of Kernel 
    to ensure the importnace.
    - Kinv : inverse of the integral kernel
    '''
    def lossfun(y_true, y_pred):
        diff    = y_true - y_pred
        cov     = tfp.stats.covariance(tf.transpose(y_true))
        mull    = K.dot(tf.linalg.inv(cov), diff)
        mull2   = K.dot(mull, tf.transpose(diff))
        dist    = tf.sqrt(mull2)
        return tf.reduce_mean(Kinv * dist)
    return lossfun

############################### IMAGES ###############################

def ssim_loss(y_true, y_pred):
    y_t = tf.expand_dims(y_true, axis = -1)
    y_p = tf.expand_dims(y_pred, axis = -1)
    return 1.0 - tf.reduce_mean((1.0 + tf.image.ssim(y_t, y_p, 2.0))/2.0)

############################### CUSTOM ###############################

def Custom_Hamming_Loss(y_true, y_pred):
    return K.mean(y_true*(1-y_pred)+(1-y_true)*y_pred)

############################# CHI SQUARED ############################

def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx

def chi_2_loss(y_true, y_pred):
    shape   = tf.shape(y_pred)
    y_t     = tf.reshape(y_true, (shape[0], shape[1]))
    y_p     = tf.reshape(y_pred, (shape[0], shape[1]))
    # take the difference
    diff    = (y_t - y_p)
    cov     = tfp.stats.covariance(y_t, sample_axis = 0, event_axis = -1)
    
    # inverse covariance
    covinv  = tf.linalg.pinv(cov)
    covinv  = tf.expand_dims(covinv, 0)
    covinv  = tf.repeat(covinv, tf.shape(diff)[0], axis = 0)

    # multiply
    mull    = tf.einsum('kij,ki->kj', covinv, diff)
    mull2   = tf.einsum('ki,ki->k', diff, mull)
    return tf.reduce_mean(mull2)

######################### CROSSENTRO AVERAGED ########################

def crossentro_av(y_true, y_pred):
    y_t = tf.reduce_mean(y_true, axis =0)
    y_p = tf.transpose(y_pred, [0, 2, 1])
    return tf.reduce_mean(tf.reduce_mean(binary_crossentropy(y_t, y_p, from_logits=False, axis = -1), axis = -1), axis = -1)

def cat_crossentro_av(y_true, y_pred):
    y_t = tf.transpose(y_true, [0, 2, 1])
    y_p = tf.transpose(y_pred, [0, 2, 1])
    return tf.reduce_mean(tf.reduce_mean(categorical_crossentropy(y_t, y_p, from_logits=False, axis = -1), axis = -1), axis = -1)
    
############################ MSA AVERAGED ############################

def msa_pinv(Kinv):
    def losfun(y_true, y_pred):
        out = tf.reduce_mean(y_true - y_pred, axis = -1)
        out = tf.reduce_mean(out, axis = 0)
        return tf.reduce_mean(tf.abs(Kinv * out))

############################ MSE AVERAGED ############################

def mse_my(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.cast(y_true, dtype = tf.float64) - tf.cast(y_pred, dtype = tf.float64)))

'''
Take the average before taking the mean squared error
'''
def mse_av(y_true, y_pred):
    y_t         =   tf.reduce_mean(y_true, axis = 0)
    #y_p = tf.transpose(y_pred, [0, 2, 1])
    y_t         =   tf.reduce_mean(tf.square(y_t - y_pred), axis = -1)
    return tf.reduce_mean(y_t)


def L_i(i):
    return lambda y_true, y_pred: tf.reduce_mean(tf.math.pow(y_true - y_pred, i))

############################ MAE AVERAGED ############################

def mae_av(y_true, y_pred):
    '''
    Take the average before taking the mean squared error
    '''
    y_ta = tf.reduce_mean(y_true, axis = 0)
    y_pa = tf.reduce_mean(y_pred, axis = 0)
    return (tf.reduce_mean(tf.reduce_mean(tf.abs(y_ta - y_pa), axis = 1), axis = -1))

############################ KL ############3

def kl(y_true, y_pred):
    y_t = tf.transpose(y_true, [0, 2, 1])
    y_p = tf.transpose(y_pred, [0, 2, 1])
    return abs(tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(y_t * tf.math.log(tf.math.divide_no_nan(y_t, tf.abs(y_p))),axis = 2), axis = 1)))

def kl_inv(y_true, y_pred):
    '''
    Inverse ...
    '''
    y_t = tf.transpose(y_true, [0, 2, 1])
    y_p = tf.transpose(y_pred, [0, 2, 1])
    return abs(tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(y_p * tf.math.log(tf.math.divide_no_nan(y_t, tf.abs(y_p))),axis = 2), axis = 1)))

######################## BERNOULLI LIKELIHOOD ########################

def nll(epos, epo = 1):
    '''
    Negative log likelihood (Bernoulli). 
    '''
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    def nlll(y_true, y_pred):
        # loss should not be affected by constant multiplication
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)# * weight
    return nlll

############################ CHOOSE THE LOSS FUNCTION ############################

def getLoss(loss_str : str, Kinv = None, rCond = 1.0):
    '''
    Returns a loss function for the model
    '''
    print(f"Using {loss_str}", 2)
    # mean squared error by default
    if loss_str == 'crossentropy_average':
        return crossentro_av
    elif loss_str == 'categorical_crossentropy_average':
        return cat_crossentro_av
    elif loss_str == 'categorical_crossentropy':
        return CategoricalCrossentropy()
    elif loss_str == 'sparse_crossentropy':
        return sparse_categorical_crossentropy
    elif loss_str == 'kl':
        return kl
    elif loss_str == 'kl_inverse':
        return kl_inv
    elif loss_str == 'binary_crossentropy':
        return BinaryCrossentropy()
    elif loss_str == 'poisson':
        return Poisson()
    elif loss_str == 'hinge':
        return Hinge()
    elif loss_str == 'huber':
        return huber
    elif loss_str == 'log_cosh':
        return log_cosh
    elif loss_str == 'msel':
        return mean_squared_logarithmic_error
    ############## MSA ##############
    elif loss_str == 'msa_pinv':
        return msa_pinv(Kinv)
    ############## MSE ##############
    elif loss_str == 'msea':
        return mse_av       
    elif loss_str == 'maea':
        return mae_av       
    elif loss_str == 'mae':
        return mean_absolute_error
    elif loss_str == 'hamming':
        return Custom_Hamming_Loss
    elif loss_str == 'ssim':
        return ssim_loss
    ############# OTHER #############
    elif loss_str == 'mahalonobis':
        return mahalonobis
    elif loss_str == 'mahalonobis_ch':
        return mahalonobisCH(Kinv)
    elif loss_str == 'mahalonobis_k':
        return mahalonobisCH(Kinv, rCond)
    elif loss_str == 'mahalonobis_pinv':
        return mahalonobis_pseudo(Kinv)
    ############ L NORMS ############
    elif loss_str.startswith("L") and len(loss_str) == 2:
        return L_i(int(loss_str[1]))
    elif loss_str == ("chi2"):
        return chi_2_loss
    elif loss_str == 'mse_my':
        return mse_my
    elif loss_str == 'mse':
        return tf.keras.losses.mean_squared_error
    elif loss_str == 'rmse':
        return lambda x, y: tf.sqrt(tf.keras.losses.mean_squared_error(x, y))
    elif loss_str == 'msle':
        return tf.keras.losses.mean_squared_logarithmic_error
    elif loss_str == 'none':
        return None
    else:
        print("\t\t->Loss: mse")
    return mean_squared_error

############################ DISTANCES ############################

'''
Bhattacharyya distance
'''
def Bhattacharyya(x):
    return (lambda p1, p2: np.trapz(np.sqrt(p1 * p2), x = x))

'''
Chi probabilistic distance
'''
def chiDistance(p1, p2):
    difference = p1 - p2
    suma       = (p1 + p2) 
    return np.sqrt(np.sum(np.square(difference) / suma)) 


from scipy.stats import entropy

'''
Bhattacharyya distance
'''
def KLDivergence(p1, p2):
    return entropy(p1, p2)