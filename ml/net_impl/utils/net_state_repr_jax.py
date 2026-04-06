"""
general_python.ml.net_impl.utils.net_state_repr_jax
===================================================

Small JAX helpers for explicit state-representation handling in networks.
"""

import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------------
#! Maps an explicit input convention (e.g. {0, 1} or {-1, +1}) to the internal representation used by the network (e.g. {-1, +1}).
# --------------------------------------------------------------------------------

def map_state_to_pm1(x: jax.Array, input_is_spin: bool, input_value: float) -> jax.Array:
	"""Map an explicit input convention to {-1, +1}."""
 
	one = jnp.asarray(1.0, dtype=x.dtype)
	two = jnp.asarray(2.0, dtype=x.dtype)
	if input_is_spin:
		scale = jnp.asarray(abs(float(input_value)), dtype=x.dtype)
		scale = jnp.where(scale == 0, one, scale)
		return x / scale
	repr_value = jnp.asarray(float(input_value), dtype=x.dtype)
	repr_value = jnp.where(repr_value == 0, one, repr_value)
	return x * (two / repr_value) - one

# --------------------------------------------------------------------------------
#! Converts state convention to binary {0, 1} indices for internal processing, if needed.
# --------------------------------------------------------------------------------

def state_to_binary_index(s: jax.Array, input_is_spin: bool, input_value: float) -> jax.Array:
	"""Convert an explicit binary or spin convention to {0, 1} indices."""
	s_real = jnp.real(s)
 
	if input_is_spin:
		threshold 	= jnp.asarray(0.0, dtype=s_real.dtype)
	else:
		repr_value 	= jnp.asarray(float(input_value), dtype=s_real.dtype)
		threshold 	= jnp.where(
			repr_value == 0,
			jnp.asarray(0.0, dtype=s_real.dtype),
			0.5 * repr_value,
		)
	return (s_real > threshold).astype(jnp.int32)

# --------------------------------------------------------------------------------
#! Returns the preferred external input representation based on configuration.
# --------------------------------------------------------------------------------

def preferred_state_representation(transform_input: bool, input_is_spin: bool) -> str:
	"""Return the configured external input representation."""
	_ = transform_input
	return "spin_pm" if bool(input_is_spin) else "binary_01"

# --------------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------------