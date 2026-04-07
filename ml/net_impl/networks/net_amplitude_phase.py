"""
general_python.ml.net_impl.networks.net_amplitude_phase
=======================================================

Composite wrapper that combines separate amplitude and phase sub-networks.

The wrapper evaluates two real-valued modules and combines them as
``log_psi = log_amp + 1j * phase``. It is useful when the magnitude and phase
benefit from different backbones or optimization scales.

----------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2026-01-21
License         : MIT
----------------------------------------------------------
"""

import jax
import jax.numpy    as jnp
import flax.linen   as nn
import numpy        as np
from typing         import Any, Optional, Dict

try:
    from ....ml.net_impl.interface_net_flax         import FlaxInterface
    from ....ml.net_impl.utils.net_wrapper_utils    import (
                                                        as_spatial_tuple,
                                                        configure_nqs_metadata,
                                                        normalize_activation_sequence,
                                                        normalize_layerwise_spec,
                                                        prepare_split_complex_input,
                                                    )
    JAX_AVAILABLE = True
except ImportError:
    raise ImportError("AmplitudePhase requires JAX/Flax and general_python modules.")

# ----------------------------------------------------------------------
# Inner Flax Module
# ----------------------------------------------------------------------

class _FlaxAmpPhase(nn.Module):
    amplitude_module    : nn.Module
    phase_module        : nn.Module
    dtype               : Any = jnp.complex128

    @nn.compact
    def __call__(self, x):
        # Prepare input for separate amplitude and phase modules
        x_real          = prepare_split_complex_input(x)
        
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
        complex_dtype = jnp.dtype(self.dtype)
        return log_amp.astype(complex_dtype) + 1j * phase.astype(complex_dtype)

# ----------------------------------------------------------------------
# Wrapper Interface
# ----------------------------------------------------------------------

class AmplitudePhase(FlaxInterface):
    """
    Composite amplitude/phase wrapper.

    Parameters
    ----------
    input_shape:
        Shape of the shared input.
    amplitude_net:
        Inner network type for the log-amplitude branch.
    phase_net:
        Inner network type for the phase branch.
    amplitude_kwargs:
        Keyword arguments forwarded to the amplitude branch.
    phase_kwargs:
        Keyword arguments forwarded to the phase branch.
    """
    def __init__(self,
                input_shape         : tuple,
                amplitude_net       : str   = 'mlp',
                phase_net           : str   = 'mlp',
                amplitude_kwargs    : Optional[Dict] = None,
                phase_kwargs        : Optional[Dict] = None,
                dtype               : Any   = jnp.complex128,
                seed                : int   = 0,
                backend             : str   = 'jax',
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

        amp_cls, amp_kws            = self._resolve_inner(amplitude_net, amplitude_kwargs, input_shape)
        phs_cls, phs_kws            = self._resolve_inner(phase_net, phase_kwargs, input_shape)
        
        amp_module_instance         = amp_cls(**amp_kws)
        phs_module_instance         = phs_cls(**phs_kws)

        net_kwargs                  = {
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
        self._name                          = 'amplitude_phase'
        configure_nqs_metadata(self, family="amplitude_phase")

    def _resolve_inner(self, net_type, net_kwargs, input_shape):
        """Resolves string keys to inner Flax Module classes and prepares kwargs."""
        
        # Import inner classes (lazy import to avoid cycles)
        if net_type == 'mlp':
            from .net_mlp import _FlaxMLPDirect

            cls     = _FlaxMLPDirect
            kws     = net_kwargs.copy()
            for key in ('backend', 'seed', 'split_complex', 'input_shape'):
                kws.pop(key, None)
                
            hidden_dims         = tuple(kws.get('hidden_dims', (32,)))
            kws['hidden_dims']  = hidden_dims
            kws['activations']  = normalize_activation_sequence(
                                    kws.get('activations'),
                                    len(hidden_dims),
                                    default=nn.relu,
                                    container=tuple,
                                )
            kws['output_dim']   = int(np.prod(kws.pop('output_shape', (1,))))
            kws.setdefault('use_bias', True)
            kws.setdefault('param_dtype', kws['dtype'])
            
            return cls, kws
            
        elif net_type == 'cnn':
            
            from .net_cnn import _FlaxCNNDirect

            cls     = _FlaxCNNDirect
            kws     = net_kwargs.copy()
            for key in ('backend', 'seed', 'split_complex', 'input_shape'):
                kws.pop(key, None)
            kws['output_feats'] = int(np.prod(kws.pop('output_shape', (1,))))
            
            # Handle reshape_dims if not present but needed
            if 'reshape_dims' not in kws:
                # Assume 1D or square
                L = int(input_shape[0]**0.5)
                
                if L*L == input_shape[0]: 
                    kws['reshape_dims'] = (L, L)
                else: 
                    kws['reshape_dims'] = (input_shape[0],)
                    
            reshape_dims        = tuple(kws['reshape_dims'])
            n_dim               = len(reshape_dims)
            features            = tuple(kws.get('features', (8,)))
            n_layers            = len(features)
            kws['features']     = features
            kws['kernel_sizes'] = tuple(
                                    as_spatial_tuple(item, n_dim, name='kernel_sizes')
                                    for item in normalize_layerwise_spec(kws.get('kernel_sizes', 3), n_layers, name='kernel_sizes')
                                )
            kws['strides']      = tuple(
                                    as_spatial_tuple(item, n_dim, name='strides')
                                    for item in normalize_layerwise_spec(kws.get('strides', 1), n_layers, name='strides')
                                )
            kws['use_bias']     = tuple(bool(b) for b in normalize_layerwise_spec(kws.get('use_bias', True), n_layers, name='use_bias'))
            kws.setdefault('input_channels',    1)
            kws.setdefault('periodic',          True)
            kws.setdefault('use_sum_pool',      True)
            kws.setdefault('input_adapter',     None)
            kws.setdefault('in_act',            None)
            kws.setdefault('out_act',           None)
            kws.setdefault('islog',             True)
            kws.setdefault('param_dtype',       kws['dtype'])
            kws['activations']  = normalize_activation_sequence(
                                    kws.get('activations'),
                                    n_layers,
                                    default=nn.elu,
                                    container=tuple,
                                )
                
            return cls, kws
            
        else:
            raise NotImplementedError(f"Inner network type '{net_type}' not yet supported for AmplitudePhase composition.")

    def __repr__(self) -> str:
        return f"AmplitudePhase(amp={self._flax_module.amplitude_module}, phase={self._flax_module.phase_module})"

# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
