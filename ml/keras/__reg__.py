if False:
    from ml.keras.__init__ import *
    from scipy.signal import find_peaks

    ########################################### FIRST DERIVATIVE ##########################################

    '''
    First derivative regularization using two point method
    '''
    def reg_1_two_point(x):
        return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(x[:,:,1:] - x[:,:,:-1]), axis = 2), axis = -1))

    '''
    First derivative regularization using five point stencil
    '''
    def reg_1_five_point(x):
        return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs((x[:,:,:-4] - 8*x[:,:,1:-3] + 8 * x[:,:,3:-1] - x[:,:,4:])/12), axis = 2), axis = -1))

    '''
    First derivative using center formula
    '''
    def reg_1_center_point(x):
        return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs((x[:,:,2:]-x[:,:,:-2])/2), axis = 2), axis = -1))

    ########################################## SECOND DERIVATIVE #########################################

    '''
    Second derivative using three point formula
    '''
    def reg_2_three_point(x):
        return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean((tf.abs(x[:,:,2:] + x[:,:,:-2] - 2*x[:,:,1:-1])), axis = 2), axis = -1))

    '''
    Second derivative using five point formula
    '''
    def reg_2_five_point(x):
        return tf.reduce_mean(tf.reduce_mean(tf.abs((-x[:,:,:-4] + 16*x[:,:,1:-3] -30*x[:,:,2:-2] + 16 * x[:,:,3:-1] - x[:,:,4:])/12), axis = 2))


    ############################################# POSITIVITY #############################################

    '''
    Regulariser that does restrict the values to be positive.
    '''
    def reg_pos(x):
        return tf.reduce_mean(tf.reduce_mean(tf.abs(x-tf.abs(x)), axis=2))
        #return abs(tf.reduce_mean(tf.reduce_sum(tf.clip_by_value(x, -1e13, 1e-5), axis=-1)))

    def reg_zeros(x):
        x_t = tf.clip_by_value(x,1e-1,1e-4)    
        
        res = tf.reduce_sum(tf.math.count_nonzero(x_t))
        return tf.cast(res, TF_TYPE)

    ########################################### PEAK DETECTION ###########################################

    def reg_peak_many(x):
        peaks = tf.reshape(x, [-1])
        peak_width = 4
        peak_prominence = 0.01
        peaks_f = tf.numpy_function(lambda x : tf.convert_to_tensor(find_peaks(x, prominence=peak_prominence, width=peak_width)[0], dtype = TF_TYPE),[peaks], TF_TYPE)
        return tf.cast(tf.size(peaks), dtype = TF_TYPE) / tf.cast(tf.size(peaks_f), dtype=TF_TYPE)

    def reg_peak_few(x):
        peaks = tf.reshape(x, [-1])
        peak_width = 4
        peak_prominence = 0.01
        peaks_f = tf.numpy_function(lambda x : tf.convert_to_tensor(find_peaks(x, prominence=peak_prominence, width=peak_width)[0], dtype = TF_TYPE),[peaks], TF_TYPE)
        return tf.cast(tf.size(peaks_f), dtype = TF_TYPE) / tf.cast(tf.size(peaks), dtype=TF_TYPE)

    def reg_local_min(x):
        xx = x#tf.transpose(x, [0, 2, 1])
        max_pooled = tf.nn.pool(xx, window_shape=(5,), strides = (5,), pooling_type='MAX', padding='SAME')
        #maxima = tf.where(tf.equal(xx, max_pooled), xx, tf.zeros_like(xx))
        return tf.convert_to_tensor(1.0, dtype = TF_TYPE)/tf.cast(tf.reduce_sum(max_pooled), dtype = TF_TYPE)


    ####################################### CHOOSE THE REG FUNCTION #######################################

    '''
    Returns a choosen regression function
    - ml_p      : params for ML model
    - verbose   : wanna talk?
    '''
    def getReg(reg, verbose = False, logger = None) -> dict:
        # create a dictionary of regularizers
        regs = {}
        for r, val in reg.items():
            reg = None
            if r    ==  'l1':
                reg = tf.keras.regularizers.L1(val)
                val = 1
            elif r  ==  'l2':
                reg = tf.keras.regularizers.L2(val)
                val = 1
            elif r  ==  'l1l2':
                reg = tf.keras.regularizers.L1L2(val[0], val[1])
            # --------------------------- first derivative ---------------------------
            elif r  ==  'der1':
                reg = reg_1_two_point
            elif r  ==  'der1_5':
                reg = reg_1_five_point
            elif r  ==  'der1_c':
                reg = reg_1_center_point
            # --------------------------- second derivative ---------------------------
            elif r  ==  'der2':
                reg = reg_2_three_point
            elif r  ==  'der2_5':
                reg = reg_2_five_point
            # --------------------------- positivity ---------------------------
            elif r == 'pos':
                reg = reg_pos
            elif r == 'peak_m':
                reg = reg_peak_many
            elif r == 'peak_f':
                reg = reg_peak_few
            elif r == 'minima':
                reg = reg_local_min
            elif r == 'zeros':
                reg = reg_zeros
            elif r == 'pos':
                reg = reg_pos
            else:
                if logger is not None:
                    logger.info(f'couldnt find {r} regularizer - skipping', 3)
                continue
            if logger is not None:
                logger.info(f'using {r}:{val} regularizer', 2)
            # put in the dictionary
            regs[r] = val, reg
        return regs

    ############################################## REG LAYER ##############################################

    class RegLayer(tf.keras.layers.Layer):
        ''' 
        Layer for calculation of the regularization.
        '''
        
        def __init__(self, reg, *args, **kwargs):
            '''
            Constructor of the class
            '''
            self.is_placeholder = False
            self.reg            = reg
            self.regs           = getReg(reg)
            super(RegLayer, self).__init__(*args, **kwargs)
            
        def call(self, inputs):
            '''
            Fundamental call function
            '''
            # iterate moments with their corresponding value
            for r, (v, f) in self.regs.items():
                val         =       v * f(inputs)
                self.add_metric(val / v,
                                aggregation =   'mean',
                                name        =   f'{self.reg[r]:.1e}*{r}')
                self.add_loss(val)
                    
            return inputs
        
        def get_config(self):
            
            ''' 
            For saving 
            '''
            return {}
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)

    ############################################## ACTIVATION #############################################

    def sigmoidCustom(mult):
        '''
        Custom sigmoid multiplied by multiplier - m * sigmoid(x)
        '''
        def sigm(x):
            return mult * tf.keras.activations.sigmoid(x)
        return sigm    

    def sigmoidCustom2(mult):
        '''
        Custom sigmoid multiplied by multiplier in the argument - sigmoid(m * x)
        '''
        def sigm(x):
            return tf.keras.activations.sigmoid(mult * x)
        return sigm    

    def sigmoidCustom3(mult):
        '''
        Custom sigmoid multiplied by multiplier and
        divided by multiplier in the argument - mult * sigmoid(x / mult)
        '''
        def sigm(x):
            return mult * tf.keras.activations.sigmoid(1.0/mult * x)
        return sigm    

    def sigmoidCustom4(mult):
        '''
        Custom sigmoid multiplied by multiplier and
        multiplied by multiplier in the argument - mult * sigmoid(x * mult)
        '''
        def sigm(x):
            return mult * tf.keras.activations.sigmoid(mult * x)
        return sigm  

    def softpluscut(mult):
        def soft(x):
            return tf.clip_by_value(tf.keras.activations.softplus(x), 0.0, mult)
        return soft
    