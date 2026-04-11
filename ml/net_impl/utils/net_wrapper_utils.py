"""
Shared helpers for Flax network wrappers.

These helpers keep the network implementations in ``net_impl`` focused on
generic model construction. 

NQS-specific integration can still inspect wrapper
metadata, but the wrappers now attach that metadata and explicit input-state
conventions through one consistent path.

--------------------------------
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
License     : MIT
Version     : 1.0
--------------------------------
"""

from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence

try:
    import jax.numpy as jnp
    from ....algebra.utils      import BACKEND_DEF_SPIN, BACKEND_REPR
    from ..activation_functions import get_activation_jnp
    from .net_state_repr_jax    import map_state_to_pm1, preferred_state_representation
except ImportError:
    raise ImportError("net_wrapper_utils requires general_python modules.")

# ----------------------------------------------------------------------
#! Activation Specification Normalization
# ----------------------------------------------------------------------

def resolve_activation_spec(spec: Optional[Any]) -> Optional[Callable]:
    """
    Normalize an activation specification to a callable.

    Supported inputs:
    - ``None``
    - a callable
    - a string accepted by ``get_activation_jnp``
    - ``(callable, params)``
    - ``(name, params)``
    """
    if spec is None:
        return None
    if callable(spec):
        return spec

    params = None
    if isinstance(spec, str):
        fn, params = get_activation_jnp(spec) # Resolve string to function and default params
        
    elif isinstance(spec, (tuple, list)) and len(spec) > 0:
        head = spec[0]
        if callable(head):
            fn                  = head
            params              = spec[1] if len(spec) > 1 else None
        elif isinstance(head, str):
            fn, default_params  = get_activation_jnp(head)
            params              = spec[1] if len(spec) > 1 else default_params
        else:
            raise ValueError(f"Invalid activation specification: {spec!r}")
    else:
        raise ValueError(f"Invalid activation specification: {spec!r}")

    if params:
        return partial(fn, **dict(params))
    return fn

def normalize_activation_sequence(spec: Optional[Any], length: int, *, default: Optional[Any] = None, container: type = tuple) -> Sequence[Optional[Callable]]:
    """
    Normalize an activation specification to a fixed-length callable sequence.

    ``spec`` may be a single activation spec, a sequence of specs, or ``None``.
    When ``None`` is provided, ``default`` is used instead.
    """
    use = default if spec is None else spec
    if use is None:
        values = [None] * length
    elif isinstance(use, str) or callable(use):
        values = [resolve_activation_spec(use)] * length
    elif isinstance(use, (tuple, list)):
        items = list(use)
        if len(items) == 1 and length > 1:
            values = [resolve_activation_spec(items[0])] * length
        elif len(items) == length:
            values = [resolve_activation_spec(item) for item in items]
        else:
            raise ValueError(f"Activation specification must have length 1 or {length}, got {len(items)}.")
    else:
        raise ValueError(f"Invalid activation specification: {use!r}")

    return container(values)

def normalize_layerwise_spec(spec: Any, length: int, *, name: str) -> Sequence[Any]:
    """
    Normalize a per-layer specification to a fixed-length sequence.

    Scalar-like values are broadcast across layers. Sequences must either have
    length ``1`` or match ``length`` exactly.
    """
    if isinstance(spec, (list, tuple)):
        items = list(spec)
        if len(items) == 0:
            raise ValueError(f"{name} cannot be empty")
        if len(items) == 1 and length > 1:
            items = items * length
        elif len(items) != length:
            raise ValueError(f"{name} must have length 1 or {length}, got {len(items)}.")
        return tuple(items)
    return tuple(spec for _ in range(length))

def as_spatial_tuple(item: Any, ndim: int, *, name: str) -> tuple[int, ...]:
    """Convert an integer or spatial sequence to a normalized tuple."""
    if isinstance(item, int):
        return (item,) * ndim
    if isinstance(item, (list, tuple)) and len(item) == ndim and all(isinstance(k, int) for k in item):
        return tuple(int(k) for k in item)
    raise ValueError(f"{name} entry {item!r} must be an int or a sequence of length {ndim}.")

# ----------------------------------------------------------------------
#! Split-Complex Helpers
# ----------------------------------------------------------------------

def complex_dtype_from_real(real_dtype: Any) -> Any:
    """Return the matching complex dtype for a real computation dtype."""
    return jnp.complex64 if jnp.dtype(real_dtype) == jnp.float32 else jnp.complex128

def real_dtype_from_complex(dtype: Any) -> Any:
    """Return the matching real dtype for a possibly complex dtype."""
    dt = jnp.dtype(dtype)
    if dt == jnp.complex64:
        return jnp.float32
    if dt == jnp.complex128:
        return jnp.float64
    return dt

# ----------------------------------------------------------------------

def resolve_split_complex_dtypes(dtype: Any, param_dtype: Any) -> tuple[Any, Any, Any]:
    """
    Resolve computation, parameter, and reconstructed output dtypes for split-complex backbones.
    """
    dtype       = jnp.dtype(dtype)
    param_dtype = jnp.dtype(param_dtype)

    if jnp.issubdtype(dtype, jnp.complexfloating):
        comp_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
    else:
        comp_dtype = dtype

    if jnp.issubdtype(param_dtype, jnp.complexfloating):
        param_real_dtype = jnp.float32 if param_dtype == jnp.complex64 else jnp.float64
    else:
        param_real_dtype = param_dtype

    return comp_dtype, param_real_dtype, complex_dtype_from_real(comp_dtype)

# ----------------------------------------------------------------------
#! Input Convention Extraction
# ----------------------------------------------------------------------

def prepare_split_complex_input(x: Any) -> Any:
    """Drop the imaginary part of complex inputs for split-complex backbones."""
    return x.real if jnp.iscomplexobj(x) else x

# ----------------------------------------------------------------------
# Split complex output recombination
# ----------------------------------------------------------------------

def combine_split_complex_output(re: Any, im: Any, real_dtype: Any) -> Any:
    """Recombine paired real/imaginary outputs into one complex tensor."""
    return (re + 1j * im).astype(complex_dtype_from_real(real_dtype))

def map_over_complex_parts(x: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply a real-valued function to real and imaginary parts independently."""
    if jnp.issubdtype(jnp.asarray(x).dtype, jnp.complexfloating):
        return fn(jnp.real(x)) + 1j * fn(jnp.imag(x))
    return fn(x)

# ----------------------------------------------------------------------
#! NQS Integration Helpers
# ----------------------------------------------------------------------

def extract_input_convention(
    wrapper_kwargs      : Mapping[str, Any],
    *,
    transform_input     : Optional[bool] = None,
    map_input_to_spin   : Optional[bool] = None,
) -> dict:
    """
    Extract explicit input-state convention kwargs for a Flax module.
    This tells how the wrapper expects the input states to be represented and whether 
    it will transform them to the internal convention (e.g. {-1, +1} for spins) or not.
    
    The returned dictionary can be passed to the network implementation for consistent handling.
    
    Parameters
    ----------
    wrapper_kwargs : Mapping[str, Any]
        The kwargs passed to the wrapper constructor, which may include input convention settings.
    transform_input : Optional[bool], default=None
        If not None, explicitly set the 'transform_input' convention. If None, it will be inferred from wrapper_kwargs or default to False.
    map_input_to_spin : Optional[bool], default=None
        If not None, explicitly set the 'map_input_to_spin' convention. If None, it will be inferred from wrapper_kwargs or default to False.
        
    Returns
    -------
    dict
        A dictionary containing the resolved input convention settings, which can be passed to the network implementation.
    
    """
    resolved_input_is_spin  = wrapper_kwargs.get(
                                "input_is_spin",
                                wrapper_kwargs.get("input_spin", BACKEND_DEF_SPIN),
                            )
    convention              = {
                                "input_is_spin" : bool(resolved_input_is_spin),
                                "input_value"   : float(wrapper_kwargs.get("input_value", BACKEND_REPR)),
                            }
    if transform_input is not None:
        convention["transform_input"]   = bool(transform_input)
    elif "transform_input" in wrapper_kwargs:
        convention["transform_input"]   = bool(wrapper_kwargs.get("transform_input"))
    if map_input_to_spin is not None:
        convention["map_input_to_spin"] = bool(map_input_to_spin)
    elif "map_input_to_spin" in wrapper_kwargs:
        convention["map_input_to_spin"] = bool(wrapper_kwargs.get("map_input_to_spin"))
    return convention

# ----------------------------------------------------------------------

def make_state_input_adapter(input_convention: Mapping[str, Any]) -> Optional[Callable]:
    """Build the shared signed-spin adapter implied by an explicit state convention."""
    if not bool(input_convention.get("transform_input", False) or input_convention.get("map_input_to_spin", False)):
        return None
    return partial(map_state_to_pm1, input_is_spin=bool(input_convention.get("input_is_spin", BACKEND_DEF_SPIN)), input_value=float(input_convention.get("input_value", BACKEND_REPR)))

def make_state_flip_update(input_convention: Mapping[str, Any]) -> Callable:
    """Build the canonical local flip update for the given explicit state convention."""
    input_is_spin   = bool(input_convention.get("input_is_spin", BACKEND_DEF_SPIN))
    input_value     = float(input_convention.get("input_value", BACKEND_REPR))

    def _flip(values):
        arr = jnp.asarray(values)
        if input_is_spin:
            return -arr
        return jnp.asarray(input_value, dtype=arr.dtype) - arr

    return _flip

# ----------------------------------------------------------------------

def infer_native_representation(
    input_convention: Mapping[str, Any],
    *,
    transform_key: str = "transform_input",
    map_key: str = "map_input_to_spin",
) -> str:
    """
    Infer the preferred external representation for the wrapper.
    
    This is used for NQS metadata to indicate the expected input state convention.
    The logic is based on the input convention settings, and it prioritizes explicit transformation or mapping flags.
    """
    return preferred_state_representation(
        bool(input_convention.get(transform_key, False) or input_convention.get(map_key, False)),
        bool(input_convention.get("input_is_spin", BACKEND_DEF_SPIN)),
    )

def resolve_input_adapter(
    wrapper_kwargs      : Mapping[str, Any],
    input_adapter       : Optional[Callable] = None,
    *,
    transform_input     : Optional[bool] = None,
    map_input_to_spin   : Optional[bool] = None,
) -> tuple[dict, Optional[Callable]]:
    """
    Resolve one explicit input convention together with the effective adapter.
    """
    input_convention = extract_input_convention(
        wrapper_kwargs,
        transform_input=transform_input,
        map_input_to_spin=map_input_to_spin,
    )
    if input_adapter is None:
        input_adapter = make_state_input_adapter(input_convention)
    return input_convention, input_adapter

def configure_nqs_metadata(
    net                     : Any,
    *,
    family                  : str,
    variant                 : str = "general",
    native_representation   : Optional[str] = None,
    supports_fast_updates   : bool = False,
    supports_exact_sampling : bool = False,
    preferred_sampler       : str = "MCSampler",
) -> Any:
    """
    Attach NQS-facing metadata to a generic wrapper.
    
    Parameters
    ----------
    net : Any
        The wrapper instance to which the metadata will be attached.
    family : str
        The NQS family (e.g. 'rbm', 'cnn', 'mlp', 'resnet', 'gcnn', etc.) that this wrapper belongs to.
    variant : str, default="general"
        A more specific variant label within the family, if needed.
    native_representation : Optional[str], default=None
        The preferred external input representation (e.g. "spin_pm", "binary_01"). If None, it will be inferred from the wrapper's input convention.
    supports_fast_updates : bool, default=False
        Whether the wrapper supports fast updates for local sampling.
    supports_exact_sampling : bool, default=False
        Whether the wrapper supports exact sampling of the wavefunction.
    preferred_sampler : str, default="MCSampler"
        The preferred sampler type for this wrapper, if any.
    """
    net._nqs_family                     = family
    net._nqs_variant                    = variant
    net._nqs_native_representation      = native_representation
    net._nqs_supports_fast_updates      = bool(supports_fast_updates)
    net._nqs_supports_exact_sampling    = bool(supports_exact_sampling)
    net._nqs_preferred_sampler          = preferred_sampler
    return net

# ----------------------------------------------------------------------
#! EOF
# ----------------------------------------------------------------------
