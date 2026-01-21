"""
general_python.ml.net_impl.networks.net_amplitude_phase
=======================================================

Amplitude-Phase Separated Ansatz.

Represents the wavefunction as:
    log_psi(s) = log_amp(s) + 1j * phase(s)

Where `log_amp` and `phase` are two independent real-valued neural networks.
This architecture avoids the difficulties of training fully complex-valued networks
and allows specialized architectures for amplitude (e.g. symmetric) and phase (e.g. autoregressive or antisymmetric).

Usage
-----
    from general_python.ml.networks import choose_network
    
    # Create an Amp-Phase network using MLPs for both
    ap_net = choose_network(
        'amplitude_phase',
        input_shape         =   (64,),
        amplitude_net       =   'mlp',
        phase_net           =   'mlp',
        amplitude_kwargs    =   {'hidden_dims': (32, 32)},
        phase_kwargs        =   {'hidden_dims': (16, 16)}
    )

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-21
License         : MIT
----------------------------------------------------------
"""

import jax
import jax.numpy    as jnp
import flax.linen   as nn
from typing         import Any, Optional, Dict, Union, Callable

try:
    from ....ml.net_impl.interface_net_flax import FlaxInterface
    from ....ml.networks import choose_network
    JAX_AVAILABLE = True
except ImportError:
    raise ImportError("AmplitudePhase requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxAmpPhase(nn.Module):
    amplitude_module    : nn.Module
    phase_module        : nn.Module
    dtype               : Any = jnp.float64 # Internal dtype is real

    @nn.compact
    def __call__(self, x):
        # x is (batch, n_sites)
        # Cast to real for internal networks
        x_real          = jnp.real(x) if jnp.iscomplexobj(x) else x
        
        # Amplitude (log-modulus)
        # Should return scalar (batch,) or (batch, 1)
        log_amp         = self.amplitude_module(x_real)
        
        # Phase (angle)
        phase           = self.phase_module(x_real)
        
        # Squeeze if needed
        if log_amp.ndim > 1: 
            log_amp     = log_amp.squeeze(-1)
        if phase.ndim > 1:   
            phase       = phase.squeeze(-1)
        
        # Combine: log_psi = log_amp + i * phase
        return log_amp.astype(jnp.complex128) + 1j * phase.astype(jnp.complex128)

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class AmplitudePhase(FlaxInterface):
    """
    Amplitude-Phase Separated Ansatz Interface.

    Parameters:
        input_shape (tuple): Shape of input (n_sites,).
        amplitude_net (str): Network type for amplitude (e.g. 'mlp', 'cnn').
        phase_net (str): Network type for phase.
        amplitude_kwargs (dict): Config for amplitude network.
        phase_kwargs (dict): Config for phase network.
    """
    def __init__(self,
                input_shape     : tuple,
                amplitude_net   : str   = 'mlp',
                phase_net       : str   = 'mlp',
                amplitude_kwargs: Optional[Dict] = None,
                phase_kwargs    : Optional[Dict] = None,
                dtype           : Any   = jnp.complex128,
                seed            : int   = 0,
                backend         : str   = 'jax',
                **kwargs):

        if not JAX_AVAILABLE: raise ImportError("AmplitudePhase requires JAX.")
        
        amplitude_kwargs    = amplitude_kwargs or {'hidden_dims': (32,), 'output_shape': (1,), 'dtype': jnp.float64}
        phase_kwargs        = phase_kwargs or {'hidden_dims': (32,), 'output_shape': (1,), 'dtype': jnp.float64}
        
        # Ensure output shapes are correct for sub-networks
        amplitude_kwargs.setdefault('output_shape', (1,))
        phase_kwargs.setdefault('output_shape', (1,))
        
        # Force real dtype for sub-networks
        amplitude_kwargs['dtype']   = jnp.float64
        phase_kwargs['dtype']       = jnp.float64

        # Helper to get module class and kwargs without instantiation
        from ....ml.networks import _NETWORK_REGISTRY, _lazy_load_class
        
        def resolve_net(net_type):
            if isinstance(net_type, str):
                return _lazy_load_class(net_type)
            return net_type
    
        amp_cls, amp_kws    = self._resolve_inner(amplitude_net, amplitude_kwargs, input_shape)
        phs_cls, phs_kws    = self._resolve_inner(phase_net, phase_kwargs, input_shape)
        
        amp_module_instance = amp_cls(**amp_kws)
        phs_module_instance = phs_cls(**phs_kws)

        net_kwargs          = {
                                'amplitude_module'  : amp_module_instance,
                                'phase_module'      : phs_module_instance,
                                'dtype'             : dtype
                            }

        super().__init__(
            net_module  =   _FlaxAmpPhase,
            net_kwargs  =   net_kwargs,
            input_shape =   input_shape,
            backend     =   backend,
            dtype       =   dtype,
            seed        =   seed,
            **kwargs
        )
        self._name = 'amplitude_phase'

    def _resolve_inner(self, net_type, net_kwargs, input_shape):
        """Resolves string keys to inner Flax Module classes and prepares kwargs."""
        
        # Import inner classes (lazy import to avoid cycles)
        if net_type == 'mlp':
            from .net_mlp import _FlaxMLP
            
            cls = _FlaxMLP
            kws = net_kwargs.copy()
            
            # Normalize activations
            if 'activations' in kws:
                
                from ....ml.net_impl.activation_functions import get_activation_jnp
                acts = kws['activations']
                
                if isinstance(acts, str):
                    act_fn, _ = get_activation_jnp(acts)
                    kws['activations'] = (act_fn,) * len(kws['hidden_dims'])
            else:
                
                from flax import linen as nn
                kws['activations'] = (nn.relu,) * len(kws['hidden_dims'])
                
            kws.setdefault('output_dim', 1)
            kws.pop('output_shape', None) # Remove wrapper arg
            
            return cls, kws
            
        elif net_type == 'cnn':
            
            from .net_cnn import _FlaxCNN
            
            cls = _FlaxCNN
            kws = net_kwargs.copy()
            kws.setdefault('output_dim', 1)
            kws.pop('output_shape', None)
            
            # Handle reshape_dims if not present but needed
            if 'reshape_dims' not in kws:
                # Assume 1D or square
                L = int(input_shape[0]**0.5)
                
                if L*L == input_shape[0]: 
                    kws['reshape_dims'] = (L, L)
                else: 
                    kws['reshape_dims'] = (input_shape[0],)
            # Default activations
            if 'activations' not in kws:
                from flax import linen as nn
                kws['activations'] = [nn.elu] * len(kws['features'])
                
            return cls, kws
            
        else:
            raise NotImplementedError(f"Inner network type '{net_type}' not yet supported for AmplitudePhase composition.")

    def __repr__(self) -> str:
        return f"AmplitudePhase(amp={self._flax_module.amplitude_module}, phase={self._flax_module.phase_module})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------