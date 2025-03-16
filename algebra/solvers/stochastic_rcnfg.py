import numpy as np
import numba 
from typing import Union, Tuple, Union, Callable, Optional
from functools import partial

from abc import ABC, abstractmethod

from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
import general_python.algebra.solver as solver_utils

#####################################

    
def loss_centered(loss, loss_m):
    '''
    '''
    return loss - loss_m

def derivatives_centered(derivatives, derivatives_m):
    '''
    '''
    return derivatives - derivatives_m


# jax specific
if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    
    def covariance_jax_minres(derivatives_c, derivatives_c_h, num_samples):
        '''
        Calculates the covariance matrix for stochastic reconfiguration from
        the variational derivatives.
        
        Parameters:
            -   
        '''
        return jnp.matmul(derivatives_c, derivatives_c_h) / num_samples
    
    def covariance_jax(derivatives_c, derivatives_c_h, num_samples):
        '''
        '''
        return jnp.matmul(derivatives_c_h, derivatives_c) / num_samples
    
    def gradient_jax(derivatives_c_h,  loss_c, num_samples):
        '''
        '''
        return jnp.matmul(derivatives_c_h, loss_c) / num_samples

# numpy specific
if True:
    
    def covariance_np_minres(derivatives_c, derivatives_c_h, num_samples):
        '''
        Calculates the covariance matrix for stochastic reconfiguration from
        the variational derivatives.
        
        Parameters:
            -   
        '''
        return np.matmul(derivatives_c, derivatives_c_h) / num_samples
    
    def covariance_np(derivatives_c, derivatives_c_h, num_samples):
        '''
        '''
        return np.matmul(derivatives_c_h, derivatives_c) / num_samples
    
    def gradient_np(derivatives_c_h, loss_c, num_samples):
        '''
        '''
        return np.matmul(derivatives_c_h, loss_c) / num_samples
    
#####################################

class StochasticReconfiguration(ABC):
    '''
    This is a class that handles the stochastic reconfiguration process
    '''
    
    def __init__(self,
                solver      : solver_utils.Solver,
                backend     : str = 'default'):
        '''
        
        '''
        super().__init__()
        
        self._backend           = get_backend(backend)
        self._isjax             = self._backend != np
        self._backendstr        = "jax" if self._isjax else "numpy"
        
        # info size and methods
        self._full_size         = 1
        self._nsamples          = 1
        self._minsr             = False
        
        # arrays
        self._loss_m            = None      # mean loss
        self._loss              = None      # loss function (can be energies) 
        self._loss_c            = None      # centered loss function L_c = L - <L>_{samples} [(E_k - <E_k>)]
        self._derivatives_m     = None      # mean derivatives <O_k>                    - (full_size)
        self._derivatives       = None      # variational derivatives O_k               - (n_samples x full_size)
        self._derivatives_c     = None      # centered derivatives (O_k - <O_k>)        - (n_samples x full_size)
        self._derivatives_c_h   = None      # centered derivatives hermitian conjugate 
        self._s                 = None      # the covariance matrix S_{kk'} = (<O_k^*O_k'> - <O_k^*><O_k>) / n_samples
        self._f                 = None      # the variational gradient of loss F_k = Cov[O_k, L_k]
        self._solution          = None
        self._solver            = solver
        
        # functions
        if self._isjax:
            self._covariance_minres_fun = jax.jit(covariance_jax_minres)
            self._covariance_fun        = jax.jit(covariance_jax)
            self._gradient_fun          = jax.jit(gradient_jax)
            self._der_c_fun             = jax.jit(derivatives_centered)
            self._loss_c_fun            = jax.jit(loss_centered)
        else:
            self._covariance_minres_fun = numba.njit(covariance_np_minres)
            self._covariance_fun        = numba.njit(covariance_np)
            self._gradient_fun          = numba.njit(gradient_np)
            self._der_c_fun             = numba.njit(derivatives_centered)
            self._loss_c_fun            = numba.njit(loss_centered)
        
    ##################################
    #! SETTERS
    ##################################
    
    def _calculate_loss(self, mean_loss = None):
        '''
        '''
        self._loss_m    = self._backend.mean(self._loss, axis = 0) if mean_loss is None else mean_loss
        self._loss_c    = self._loss_c_fun(self._loss, self._loss_m)
        
    def _calculate_derivatives(self, mean_deriv = None):
        '''
        '''
        self._derivatives_m     = self._backend.mean(self._derivatives, axis = 0) if mean_deriv is None else mean_deriv
        self._derivatives_c     = self._der_c_fun(self._derivatives, self._derivatives_m)
        self._derivatives_c_h   = self._backend.conj(self._derivatives_c).T
    
    def _calculate_s(self):
        '''
        '''
        if self._minsr:
            return self._covariance_minres_fun(self._derivatives_c, self._derivatives_c_h, self._nsamples)
        return self._covariance_fun(self._derivatives_c, self._derivatives_c_h, self._nsamples)
    
    def set_values(self,
                loss,
                derivatives,
                mean_loss    = None,
                mean_deriv   = None,
                calculate_s  = False,
                use_minsr    : Optional[bool] = None):
        '''
        Sets the values for the Stochastic Reconfiguration (Natural Gradient)
        '''
        # get the loss
        self._loss          = loss
        self._nsamples      = self._loss.shape[0]
        self._calculate_loss(mean_loss)
        self._derivatives   = derivatives
        self._full_size     = derivatives.shape[1]
        self._calculate_derivatives(mean_deriv)
        
        # use min_sr?
        if use_minsr is not None:
            self._minsr     = use_minsr
            
        # calculate covariance
        if calculate_s:
            self._s = self._calculate_s()
        else:
            self._s = None
            
        # calculate F
        self._f     = self._gradient_fun(self._derivatives_c_h, self._loss_c, self._nsamples)
    
    def solve(self, use_s = False, use_minsr = False):
        '''
        '''
        
        self._minsr = use_minsr
        
        if use_s:
            self._s = self._calculate_s()
            # solve with s
            #! TODO, add my solver
            # self._solution = self._backend.linalg.solve(self._s, self._f)
            self._solution = self._backend.linalg.pinv(self._s) @ self._f
        else:
            # solve without creating a matrix explicitely (using the Fisher form)
            #! TODO, add my solver
            if self._minsr:
                def mat_vec_mult(x):
                    applied = self._derivatives_c_h @ x
                    return self._derivatives_c @ applied / self._nsamples
                self._solution = jax.lax.custom_linear_solve(mat_vec_mult, self._loss_c, solve=jax.jit(jsp.linalg.solve))
            else:
                def mat_vec_mult(x):
                    applied = self._derivatives_c @ x
                    return self._derivatives_c_h @ applied / self._nsamples
                self._solution = jax.lax.custom_linear_solve(mat_vec_mult, self._f, solve=jax.jit(jsp.linalg.solve))
        
        # return the forces
        if self._minsr:
            return self._backend.matmul(self._derivatives_c_h, self._solution)
        return self._solution
    
    ##################################
    #! PROPERTIES
    ##################################

    @property
    def forces(self):
        ''''''
        return self._f
    
    @property
    def derivatives(self):
        ''''''
        return self._derivatives
    
    @property
    def derivatives_c(self):
        ''''''
        return self._derivatives_c
    
    @property
    def derivatives_c_h(self):
        ''''''
        return self.derivatives_c_h
    
    @property
    def solution(self):
        ''''''
        return self._solution
    
    @property
    def covariance_matrix(self):
        ''''''
        if self._s is None:
            self._s = self._calculate_s()
        return self._s
    
    ##################################


######################################