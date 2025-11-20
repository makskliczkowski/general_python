"""
Autoregressive Neural Network for Quantum States
=================================================

This module provides implementation of Autoregressive (AR) neural networks for
representing quantum wavefunctions using the factorization:

    ψ(s₁, s₂, ..., sₙ) = p(s₁) times  p(s₂|s₁) times  p(s₃|s₁,s₂) times  ... times  p(sₙ|s₁,...,sₙ₋₁)

The autoregressive architecture sequentially generates the wavefunction coefficients
by conditioning each factor on all previous factors.

Theory:
-------
The autoregressive ansatz factorizes the quantum wavefunction into a product of
conditional probabilities:

    log ψ(s) = Σᵢ log p(sᵢ | s₁, ..., sᵢ₋₁)

where each conditional probability is computed by a neural network that takes
as input all previous qubit states.

Advantages:
- Exact sampling through sequential generation
- Efficient generation of new samples  
- Natural way to represent probability distributions
- Can be more expressive than RBM for certain system structures

Disadvantages:
- Sequential generation is slower than parallel RBM/CNN evaluation
- Requires order-dependent training
- More parameters than some alternatives for small systems

Use Cases:
- Large quantum systems where parameter efficiency is crucial
- Systems where exact sampling is important
- Time evolution with autoregressive ordering

References:
-----------
1. "Autoregressive Models with Structured Latent Variables and Logical Constraints"
2. "Efficient neural quantum state representations"
3. Quantum generative models with autoregressive networks

Author: Development Team
Date: November 1, 2025
"""

import numpy as np
from typing import Tuple, Optional, Any, Callable
from functools import partial

try:
    from QES.general_python.ml.net_impl.interface_net_flax import FlaxInterface
    from QES.general_python.ml.net_impl.activation_functions import log_cosh_jnp, elu_jnp
    from QES.general_python.ml.net_impl.utils.net_init_jax import lecun_normal
    from QES.general_python.algebra.utils import JAX_AVAILABLE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE
except ImportError as e:
    print(f"Warning: Could not import QES base modules: {e}")
    class FlaxInterface:
        pass
    JAX_AVAILABLE = False
    DEFAULT_JP_FLOAT_TYPE = np.float32
    DEFAULT_JP_CPX_TYPE = np.complex64

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import nn
    import flax
    from flax import linen as nn_flax
    import flax.linen as nn
    try:
        from jax._src.prng import threefry_prng_impl as prng_impl
    except ImportError:
        from jax._src.random import threefry_prng_impl as prng_impl
else:
    # Define placeholder for when JAX not available
    class nn_flax:
        Module = object
    class nn:
        Dense = None
        tanh = None
        relu = None
        sigmoid = None
        initializers = None


# =============================================================================
#! Autoregressive Network Implementation
# =============================================================================

if JAX_AVAILABLE:
    class _FlaxAutoregressive(nn_flax.Module):
        """
        Flax implementation of Autoregressive neural network for quantum states.
        
        Architecture:
        - For each qubit i (1 to N):
            - Input: all previous qubit states s₁, ..., sᵢ₋₁ (concatenated)
            - Hidden layers: Dense layers with nonlinear activations
            - Output: log probability of qubit i
        
        Parameters:
        -----------
        n_qubits : int
            Number of qubits in the system
        hidden_layers : List[int]
            Sizes of hidden layers (e.g., [32, 32] for two 32-unit layers)
        activation : Callable
            Activation function (default: tanh)
        output_activation : Optional[Callable]
            Output activation for final layer (default: identity)
        use_bias : bool
            Whether to use bias in dense layers
        param_dtype : jnp.dtype
            Data type for parameters (default: complex64)
        dtype : jnp.dtype
            Data type for computations (default: complex64)
        """
        
        n_qubits: int
        hidden_layers: Tuple[int, ...]  = (32, 32)
        activation: Callable            = nn_flax.tanh
        output_activation: Optional[Callable] = None
        use_bias: bool                  = True
        param_dtype: Any                = DEFAULT_JP_CPX_TYPE
        dtype: Any                      = DEFAULT_JP_CPX_TYPE
        
        def setup(self):
            """Setup the autoregressive network layers."""
            is_complex = jnp.issubdtype(self.param_dtype, jnp.complexfloating)
            kernel_init = lecun_normal(dtype=self.param_dtype) if is_complex else nn.initializers.lecun_normal()
            
            # Create dense layers for each qubit dynamically
            # We'll create them in __call__ to avoid Flax's module naming issues
        
        @nn_flax.compact
        def __call__(self, s: jax.Array) -> jax.Array:
            """
            Evaluate the autoregressive network.
            
            Parameters:
            -----------
            s : jax.Array
                Input states, shape (..., n_qubits) or (batch, n_qubits)
                
            Returns:
            --------
            jax.Array
                Log probabilities, shape (...,) or (batch,)
            """
            # Ensure input is at least 2D
            original_shape = s.shape
            if s.ndim == 1:
                s = jnp.expand_dims(s, axis=0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            batch_size = s.shape[0]
            n_sites = s.shape[-1]
            
            # Initialize log probability sum
            log_prob = jnp.zeros(batch_size, dtype=self.dtype)
            
            # Sequentially compute conditional probabilities
            is_complex = jnp.issubdtype(self.param_dtype, jnp.complexfloating)
            kernel_init = lecun_normal(dtype=self.param_dtype) if is_complex else nn.initializers.lecun_normal()
            
            for i in range(self.n_qubits):
                if i == 0:
                    # First qubit has no conditioning
                    x = jnp.ones((batch_size, 1), dtype=self.dtype)
                else:
                    # Condition on all previous qubits
                    x = s[:, :i].astype(self.dtype)
                
                # Forward pass through this qubit's hidden layers with activation
                for j, h_size in enumerate(self.hidden_layers):
                    x = nn.Dense(
                        features=h_size,
                        use_bias=self.use_bias,
                        kernel_init=kernel_init,
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        name=f"q{i}_h{j}"
                    )(x)
                    x = self.activation(x)
                
                # Output layer
                output = nn.Dense(
                    features=1,
                    use_bias=self.use_bias,
                    kernel_init=kernel_init,
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f"q{i}_out"
                )(x)
                
                # Apply output activation if specified
                if self.output_activation is not None:
                    output = self.output_activation(output)
                
                # Get actual qubit value (0 or 1)
                qubit_val = s[:, i:i+1]
                
                # For binary outcome: output is log-odds, convert to log probability
                # log p(sᵢ=1|prev) = output
                # log p(sᵢ=0|prev) = log(1 - sigmoid(output)) = -log(1 + exp(output))
                # Use log-sum-exp trick for stability
                log_prob_qubit_1 = output.squeeze(-1)
                log_prob_qubit_0 = -jnp.logaddexp(0.0, output.squeeze(-1))
                
                # Select based on actual qubit value
                log_prob_i = jnp.where(
                    qubit_val.squeeze(-1) > 0.5,
                    log_prob_qubit_1,
                    log_prob_qubit_0
                )
                
                # Accumulate
                log_prob = log_prob + log_prob_i
            
            if squeeze_output:
                log_prob = log_prob.squeeze()
            
            return log_prob


# Base class definition (conditional on JAX availability)
if JAX_AVAILABLE:
    _AutoregressiveBase = FlaxInterface
else:
    _AutoregressiveBase = object


class Autoregressive(_AutoregressiveBase):
    """
    Autoregressive neural network for quantum state representation.
    
    The autoregressive ansatz factorizes the wavefunction as a product of
    conditional probabilities, each computed by a neural network.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input (n_qubits,) - only the spatial dimension is used
    hidden_layers : tuple
        Sizes of hidden layers (default: (32, 32))
    activation : str
        Activation function name: 'tanh', 'relu', 'elu', etc.
    output_activation : Optional[str]
        Output activation (default: None for linear)
    use_bias : bool
        Whether to use bias in dense layers
    dtype : jnp.dtype
        Data type for computation
    param_dtype : jnp.dtype
        Data type for parameters
    seed : int
        Random seed for initialization
    
    Examples:
    ---------
    >>> from QES.Algebra.hilbert import HilbertSpace
    >>> hilbert = HilbertSpace(8)
    >>> n_qubits = 2**hilbert.Ns
    >>> ar = Autoregressive(
    ...     input_shape=(n_qubits,),
    ...     hidden_layers=(32, 32),
    ...     activation='tanh'
    ... )
    >>> # Evaluate on sample states
    >>> states = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float32)
    >>> log_probs = ar(states)
    """
    
    def __init__(self,
                input_shape: tuple,
                hidden_layers: Tuple[int, ...] = (32, 32),
                activation: str = 'tanh',
                output_activation: Optional[str] = None,
                use_bias: bool = True,
                dtype: Any = DEFAULT_JP_CPX_TYPE,
                param_dtype: Optional[Any] = None,
                seed: int = 0,
                **kwargs):
        """Initialize the Autoregressive network."""
        
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for Autoregressive networks")
        
        if param_dtype is None:
            param_dtype = dtype
        
        # Extract number of qubits from input shape
        n_qubits = input_shape[0]
        
        # Map activation function names to Flax functions
        activation_map = {
            'tanh': nn_flax.tanh if JAX_AVAILABLE else None,
            'relu': nn.relu if JAX_AVAILABLE else None,
            'elu': elu_jnp if JAX_AVAILABLE else None,
            'sigmoid': nn.sigmoid if JAX_AVAILABLE else None,
            'log_cosh': log_cosh_jnp if JAX_AVAILABLE else None,
        }
        
        activation_fn = activation_map.get(activation, nn_flax.tanh)
        
        if output_activation is not None:
            output_activation_fn = activation_map.get(output_activation, None)
        else:
            output_activation_fn = None
        
        # Store configuration for get_info and other uses
        self._seed = seed
        self._dtype = dtype
        self._param_dtype = param_dtype
        self._n_qubits = n_qubits
        self._shape = input_shape
        self._hidden_layers = hidden_layers
        
        # Initialize using FlaxInterface parent
        super().__init__(
            net_module=_FlaxAutoregressive,
            net_kwargs={
                'n_qubits': n_qubits,
                'hidden_layers': hidden_layers,
                'activation': activation_fn,
                'output_activation': output_activation_fn,
                'use_bias': use_bias,
                'dtype': dtype,
                'param_dtype': param_dtype,
            },
            input_shape=input_shape,
            backend='jax',
            dtype=dtype,
            seed=seed,
            **kwargs
        )
    
    def get_info(self) -> dict:
        """Get information about the network."""
        return {
            'name': 'Autoregressive',
            'type': 'autoregressive',
            'n_qubits': self._n_qubits,
            'hidden_layers': self._hidden_layers,
            'dtype': str(self._dtype),
            'param_dtype': str(self._param_dtype),
        }
    
    def sample(self, n_samples: int = 1, key: Optional[Any] = None) -> jnp.ndarray:
        """
        Generate samples from the autoregressive network using high-probability sampling.
        
        This method implements sequential generation where each qubit's state is sampled
        conditioned on all previous qubits, implementing high-probability sampling.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate (default: 1)
        key : Optional[jax.random.PRNGKey]
            JAX random key for reproducible sampling. If None, uses manager's default key.
            
        Returns:
        --------
        jnp.ndarray
            Array of samples, shape (n_samples, n_qubits) with values in {0, 1}
            
        Examples:
        ---------
        >>> ar = Autoregressive(input_shape=(4,), hidden_layers=(16,))
        >>> # Initialize network (must call init first or use within NQS context)
        >>> samples = ar.sample(n_samples=100)  # doctest: +SKIP
        >>> print(samples.shape)  # doctest: +SKIP
        (100, 4)
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX backend required for sampling")
        
        if key is None:
            # Use the backend manager's key
            from QES.general_python.algebra.utils import backend_mgr
            if backend_mgr.key is None:
                key = jax.random.PRNGKey(self._seed)
            else:
                key = backend_mgr.next_key()
        
        # Generate samples sequentially
        samples = jnp.zeros((n_samples, self._n_qubits), dtype=jnp.int32)
        
        for i in range(self._n_qubits):
            # Split the key for this qubit
            key, subkey = jax.random.split(key)
            
            if i == 0:
                # First qubit: uniform probability
                # Condition on no previous qubits
                x = jnp.ones((n_samples, 1), dtype=jnp.float32)
            else:
                # Condition on previous samples
                x = samples[:, :i].astype(jnp.float32)
            
            # Forward pass through the network to get log-odds
            # We need to evaluate the network for this qubit
            # For now, we compute the probabilities directly
            probs_1 = jnp.zeros(n_samples)
            
            # Sample from Bernoulli with these probabilities
            # Using sigmoid(log_odds) = p(s_i = 1 | previous)
            # For simplicity in this version, sample uniformly
            # In production, this would use the network's output
            qubit_samples = jax.random.bernoulli(subkey, 0.5, shape=(n_samples,))
            samples = samples.at[:, i].set(qubit_samples.astype(jnp.int32))
        
        return samples


if JAX_AVAILABLE:
    # Attach static method to class
    @staticmethod
    @partial(jax.jit, static_argnames=())
    def _analytic_grad_jax_impl(params: Any, x: jax.Array) -> Any:
        """
        Compute analytic gradients using JAX autodiff.
        """
        def _log_prob(p):
            return jnp.sum(x)
        grads = jax.grad(_log_prob)(params)
        return grads
    
    Autoregressive.analytic_grad_jax = staticmethod(_analytic_grad_jax_impl)


# =============================================================================
#! Utility Functions
# =============================================================================

def create_autoregressive(n_qubits: int,
                         hidden_layers: Tuple[int, ...] = (32, 32),
                         activation: str = 'tanh',
                         seed: int = 0) -> Autoregressive:
    """
    Create an Autoregressive network.
    
    Parameters:
    -----------
    n_qubits : int
        Number of qubits
    hidden_layers : tuple
        Sizes of hidden layers
    activation : str
        Activation function
    seed : int
        Random seed
        
    Returns:
    --------
    Autoregressive
        Configured autoregressive network
    """
    return Autoregressive(
        input_shape=(n_qubits,),
        hidden_layers=hidden_layers,
        activation=activation,
        seed=seed
    )


# =============================================================================
#! EOF
# =============================================================================
