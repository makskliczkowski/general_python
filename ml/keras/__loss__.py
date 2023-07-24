from .__reg__ import *

############################ IMAGES ############################

def ssim_loss(y_true, y_pred):
    y_t = tf.expand_dims(y_true, axis = -1)
    y_p = tf.expand_dims(y_pred, axis = -1)
    return 1.0 - tf.reduce_mean((1.0 + tf.image.ssim(y_t, y_p, 2.0))/2.0)

############################ CUSTOM ############################

def Custom_Hamming_Loss(y_true, y_pred):
    return K.mean(y_true*(1-y_pred)+(1-y_true)*y_pred)

############################ CHI SQUARED ############################

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

############################ CROSSENTRO AVERAGED ############################

def crossentro_av(y_true, y_pred):
    y_t = tf.reduce_mean(y_true, axis =0)
    y_p = tf.transpose(y_pred, [0, 2, 1])
    return tf.reduce_mean(tf.reduce_mean(binary_crossentropy(y_t, y_p, from_logits=False, axis = -1), axis = -1), axis = -1)

def cat_crossentro_av(y_true, y_pred):
    y_t = tf.transpose(y_true, [0, 2, 1])
    y_p = tf.transpose(y_pred, [0, 2, 1])
    return tf.reduce_mean(tf.reduce_mean(categorical_crossentropy(y_t, y_p, from_logits=False, axis = -1), axis = -1), axis = -1)
    
############################ MSE AVERAGED ############################

def mse_my(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred, dtype = tf.float64), dtype = tf.float64)

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

'''
Take the average before taking the mean squared error
'''
def mae_av(y_true, y_pred):
    y_ta = tf.reduce_mean(y_true, axis = 0)
    y_pa = tf.reduce_mean(y_pred, axis = 0)
    return (tf.reduce_mean(tf.reduce_mean(tf.abs(y_ta - y_pa), axis = 1), axis = -1))

############################ KL ############3

def kl(y_true, y_pred):
    y_t = tf.transpose(y_true, [0, 2, 1])
    y_p = tf.transpose(y_pred, [0, 2, 1])
    return abs(tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(y_t * tf.math.log(tf.math.divide_no_nan(y_t, tf.abs(y_p))),axis = 2), axis = 1)))


def kl_inv(y_true, y_pred):
    y_t = tf.transpose(y_true, [0, 2, 1])
    y_p = tf.transpose(y_pred, [0, 2, 1])
    return abs(tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(y_p * tf.math.log(tf.math.divide_no_nan(y_t, tf.abs(y_p))),axis = 2), axis = 1)))

############################ BERNOULLI LIKELIHOOD ############################

'''
Negative log likelihood (Bernoulli). 
'''
def nll(epos,epo = 1):
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    def nlll(y_true, y_pred):
        # loss should not be affected by constant multiplication
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)# * weight
    return nlll



############################ CHOOSE THE LOSS FUNCTION ############################

'''
Returns a loss function for the model
'''
def getLoss(loss_str : str):
    # mean squared error by default
    if loss_str == 'crossentro':
        return crossentro_av
    elif loss_str == 'cat_crossentro':
        return cat_crossentro_av
    elif loss_str == 'sparse_crossentropy':
        return sparse_categorical_crossentropy
    elif loss_str == 'kl':
        return kl
    elif loss_str == 'kli':
        return kl_inv
    elif loss_str == 'cat_entro':
        return CategoricalCrossentropy()
    elif loss_str == 'bin_entro':
        return BinaryCrossentropy()
    elif loss_str == 'poisson':
        return Poisson()
    elif loss_str == 'kld':
        return KLDivergence()
    elif loss_str == 'hinge':
        return Hinge()
    elif loss_str == 'huber':
        return huber
    elif loss_str == 'log_cosh':
        return log_cosh
    elif loss_str == 'msel':
        return mean_squared_logarithmic_error
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
    #### L NORMS
    elif loss_str.startswith("L") and len(loss_str) == 2:
        return L_i(int(loss_str[1]))
    elif loss_str == ("chi2"):
        return chi_2_loss
    elif loss_str == 'mse_my':
        return mse_my
    elif loss_str == 'mse':
        return tf.keras.losses.mean_squared_error
    else:
        print("\t\t->Loss: mse")
    print(f'\t\t->Loss: {loss_str}')      
    return mean_squared_error