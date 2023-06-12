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

def chi_2_loss(y_true, y_pred):
    diff = y_true - y_pred
    cov = tfp.stats.covariance(tf.transpose(y_pred), event_axis = 1)
    mull = K.dot(tf.linalg.inv(cov), diff)
    mull2 = K.dot(mull, tf.transpose(diff))
    dist = tf.sqrt(mull2)
    return dist

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

'''
Take the average before taking the mean squared error
'''
def mse_av(y_true, y_pred):
    y_t = tf.reduce_mean(y_true, axis =0)
    #y_p = tf.transpose(y_pred, [0, 2, 1])
    return tf.reduce_mean(tf.square(y_t - y_pred))

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
    else:
        print("\t\t->Loss: mse")
    print(f'\t\t->Loss: {loss_str}')      
    return mean_squared_error