# file        :   general_python/algebra/utils.py
# author      :   Maksymilian Kliczkowski
# copyright   :   (c) 2025, Maksymilian Kliczkowski

'''
This module provides utilities for importing the linear algebra backend,
including NumPy and JAX, and managing their configurations.

- It includes functions for checking backend availability, setting random seeds,
and handling JIT compilation.

- It also provides a BackendManager class for managing the backend state,
including the random module and SciPy functionalities.

- It is designed to be used in a flexible and extensible manner,
allowing for easy switching between backends.

- The module also includes functions for padding arrays and checking if an array is JAX-traced.

- The module is designed to be used in a flexible and extensible manner,
allowing for easy switching between backends.

Provides:
- BackendManager: 
    Class to detect and manage the active backend (NumPy/JAX),
    including linear algebra, random number generation, and SciPy modules.
- Functions to retrieve backend components (`get_backend`, `get_global_backend`).
- Global access to the active backend via the `backend_mgr` instance.
- Utilities for JIT compilation (`maybe_jit`), hardware info (`get_hardware_info`),
    and array padding (`pad_array`).
- Clear version reporting and backend status printing.

'''

# Import the required modules
import os
import sys
import inspect
import logging
import random as py_random
import multiprocessing
from functools import wraps
from contextlib import contextmanager
from typing import Union, Optional, TypeAlias, Type, Tuple, Any, Callable, List, Dict, Literal
from dataclasses import dataclass

# ---------------------------------------------------------------------
#! Try to import the global logger
# ---------------------------------------------------------------------

log : 'Logger' = None

# ---------------------------------------------------------------------

import numpy as np
import numpy.random as np_random
import scipy as sp

# ---------------------------------------------------------------------
#! Enviroment variable names
num_cores                                   = os.cpu_count()
PY_NUM_CORES_STR        : str               = "PY_NUM_CORES"

#! os environment variables
PY_JAX_AVAILABLE_STR    : str               = "PY_JAX_AVAILABLE"
PY_FLOATING_POINT_STR   : str               = "PY_FLOATING_POINT"
PY_BACKEND_STR          : str               = "PY_BACKEND"

PY_GLOBAL_SEED_STR      : str               = "PY_GLOBAL_SEED"
PY_INFO_VERBOSE         : str               = "PY_BACKEND_INFO"
PY_UTILS_INIT_DONE_STR  : str               = "PY_UTILS_INIT_DONE"
# ---------------------------------------------------------------------

JIT                     : Callable          = lambda x: x # Default JIT function (identity)
DEFAULT_SEED            : int               = 42
DEFAULT_BACKEND         : str               = "numpy"
DEFAULT_BACKEND_KEY     : Optional[str]     = None
DEFAULT_NP_INT_TYPE     : Type              = np.int64
DEFAULT_NP_FLOAT_TYPE   : Type              = np.float64
DEFAULT_NP_CPX_TYPE     : Type              = np.complex128
BACKEND_REPR            : float             = 0.5
BACKEND_DEF_SPIN        : bool              = True

DEFAULT_JP_INT_TYPE     : Optional[Type]    = None
DEFAULT_JP_FLOAT_TYPE   : Optional[Type]    = None
DEFAULT_JP_CPX_TYPE     : Optional[Type]    = None

# ---------------------------------------------------------------------

def _log_message(msg, lvl = 0, **kwargs):
    """
    Logs a message using the global logger.
    This function ensures the logger is only imported when needed.
    
    Parameters:
        msg (str):
            The message to log.
        lvl (int):
            The indentation level for the message.
    """
    if not PY_INFO_VERBOSE:
        return
    
    text = "\t" * lvl + msg
    if log is None:
        # Lazy import of the logger
        print(msg)
    else:
        log.info(text, **kwargs)

# ---------------------------------------------------------------------
#! SET VARIABLES
# ---------------------------------------------------------------------

PY_GLOBAL_SEED         : int                = int(os.environ.get(PY_GLOBAL_SEED_STR, DEFAULT_SEED))
os.environ[PY_GLOBAL_SEED_STR]              = str(PY_GLOBAL_SEED)

PY_NUM_CORES            : int               = int(os.environ.get(PY_NUM_CORES_STR, str(num_cores)))
os.environ[PY_NUM_CORES_STR]                = str(PY_NUM_CORES)

PREFER_32BIT            : bool              = os.environ.get(PY_FLOATING_POINT_STR, "64bit").lower() in ["32bit", "32", "float32", "float"]
PY_FLOATING_POINT       : str               = os.environ.get(PY_FLOATING_POINT_STR, "float32" if PREFER_32BIT else "float64")
PY_USE_32BIT            : bool              = PREFER_32BIT
os.environ[PY_FLOATING_POINT_STR]           = PY_FLOATING_POINT

PY_NP_INT_TYPE          : Type              = np.int32 if PY_USE_32BIT else np.int64
PY_NP_FLOAT_TYPE        : Type              = np.float32 if PY_USE_32BIT else np.float64
PY_NP_CPX_TYPE          : Type              = np.complex64 if PY_USE_32BIT else np.complex128
PY_BACKEND              : str               = os.environ.get(PY_BACKEND_STR, DEFAULT_BACKEND).lower()
os.environ[PY_BACKEND_STR]                  = PY_BACKEND

# by default, use numpy
PREFER_JAX              : bool              = PY_BACKEND != "numpy" and PY_BACKEND != "np"
PREFER_64BIT            : bool              = True if not PREFER_32BIT else False

#! Backend Detection
JAX_AVAILABLE: bool     = False
jax                     = None
jnp                     = None
jsp                     = None
jrn                     = None
jax_jit                 = lambda x: x
jcfg                    = None

# --- JAX-related placeholders ---
JAX_AVAILABLE: bool     = False
jax: Optional[Any]      = None
jnp: Optional[Any]      = None
jsp: Optional[Any]      = None
jrn: Optional[Any]      = None
jcfg: Optional[Any]     = None

# --- Type Aliases (with defaults) ---
Array: TypeAlias        = np.ndarray
PRNGKey: TypeAlias      = Any # Keep as 'Any' to avoid import errors if JAX is not present
JaxDevice: TypeAlias    = Any

# ---------------------------------------------------------------------

PY_JAX_AVAILABLE        : bool = JAX_AVAILABLE
os.environ[PY_JAX_AVAILABLE_STR] = "1" if PY_JAX_AVAILABLE else "0"

# ---------------------------------------------------------------------

#! Type defaults

DEFAULT_NP_INT_TYPE     = PY_NP_INT_TYPE
DEFAULT_NP_FLOAT_TYPE   = PY_NP_FLOAT_TYPE
DEFAULT_NP_CPX_TYPE     = PY_NP_CPX_TYPE
DEFAULT_JP_INT_TYPE     = None
DEFAULT_JP_FLOAT_TYPE   = None
DEFAULT_JP_CPX_TYPE     = None

#! Type Aliases
if JAX_AVAILABLE and jnp:
    Array       : TypeAlias = Union[np.ndarray, jnp.ndarray]
    PRNGKey     : TypeAlias = Any # jax.random.PRNGKeyArray
    JaxDevice   : TypeAlias = Any # Placeholder for jax device type
else:
    Array       : TypeAlias = np.ndarray
    PRNGKey     : TypeAlias = None
    JaxDevice   : TypeAlias = None

#! These will be updated by the backend_mgr after initialization.
ACTIVE_BACKEND_NAME     : str       = "numpy"
ACTIVE_NP_MODULE        : Any       = np
ACTIVE_RANDOM           : Any       = np_random.default_rng(DEFAULT_SEED)  # Start with a default numpy RNG
ACTIVE_SCIPY_MODULE     : Any       = sp
ACTIVE_JIT              : Callable  = JIT
ACTIVE_JAX_KEY          : Optional[PRNGKey] = None
ACTIVE_INT_TYPE         : Type      = np.int64
ACTIVE_FLOAT_TYPE       : Type      = np.float64
ACTIVE_COMPLEX_TYPE     : Type      = np.complex128
backend_mgr             : 'BackendManager' = None  # Will be set after BackendManager is defined

# ---------------------------------------------------------------------
#! Global methods
# ---------------------------------------------------------------------

def is_jax_array(x: Any) -> bool:
    '''
    Checks if an object is likely a JAX array (including traced).

    Parameters
    ----------
    x : Any
        The object to check.

    Returns
    -------
    bool
        True if x is a JAX array (including traced), False otherwise.
    '''
    if not JAX_AVAILABLE:
        return False
    try:
        # modern JAX
        from jax import Array as JaxArray  # type: ignore
        return isinstance(x, JaxArray)
    except Exception:
        # fallback (older versions / tracers)
        try:
            from jax.core import Tracer as JaxTracer  # type: ignore
            return hasattr(x, "aval") or isinstance(x, JaxTracer)
        except Exception:
            return hasattr(x, "aval")
# ----
is_traced_jax = is_jax_array

# ---------------------------------------------------------------------

def get_backend(backend_spec    : Union[str, Any, None] = None,
                random          : bool = False,
                seed            : Optional[int] = None,
                scipy           : bool = False) -> Union[Any, Tuple[Any, ...]]:
    """
    Return backend modules based on the provided specifier.

    Delegates to the global `backend_mgr.get_backend_modules`.

    Parameters
    ----------
    backend_spec : str or module or None, optional
        Backend specifier ("numpy", "jax", `np`, `jnp`, "default", None).
        Defaults to the globally active backend.
    random : bool, optional
        If True, include the random module/state. For JAX, also return a PRNG key.
        For NumPy, returns a seeded RNG instance. Default is False.
        **For JAX backend, ensure you split the returned PRNG key before use.**
    seed : int, optional
        Seed for the random component. If None, uses the global default seed
        to generate the component for this call. Providing a seed here creates
        a *new* RNG/Key for this call based on that specific seed.
    scipy : bool, optional
        If True, also return the associated SciPy module. Default is False.

    Returns
    -------
    module or tuple
        Requested backend components. See `BackendManager.get_backend_modules` docs.

    **Example for using JAX backend with random number generation:**

    >>> import general_python.algebra.utils as abu
    >>> import numpy as np
    >>> if abu.JAX_AVAILABLE:
    ...     jax_np, (jax_rnd, key), jax_sp = abu.get_backend("jax", random=True, scipy=True, seed=42)
    ...     key, subkey = jax_rnd.split(key) # Split the key!
    ...     random_vector = jax_rnd.uniform(subkey, shape=(5,)) # Use subkey
    ...     print(random_vector)
    """
    return backend_mgr.get_backend_modules(backend_spec, use_random=random, seed=seed, use_scipy=scipy)

# ---------------------------------------------------------------------

def get_global_backend(random: bool = False, seed: Optional[int] = None, scipy: bool = False) -> Union[Any, Tuple[Any, ...]]:
    """
    Return the globally configured default backend modules.

    Delegates to `backend_mgr.get_global_backend_modules`.

    Parameters
    ----------
    random : bool, optional
        If True, include the random module/state and potentially a key/RNG.
    seed : int, optional
        Optional seed for this specific request's random component.
    scipy : bool, optional
        If True, also return the associated SciPy module.

    Returns
    -------
    module or tuple
        The global default backend module(s). See `BackendManager.get_backend_modules`.
    """
    return backend_mgr.get_global_backend_modules(use_random=random, seed=seed, use_scipy=scipy)

# ---------------------------------------------------------------------

def maybe_jit(func):
    """
    Maybe apply JAX JIT compilation to the function.
    """
    
    if not JAX_AVAILABLE or os.getenv("QES_JIT", "1") in ("0", "false", "False"):
        return func
    from jax import jit as _jit

    sig = inspect.signature(func)
    if 'backend' not in sig.parameters:
        raise ValueError(f"@maybe_jit: '{func.__name__}' must accept 'backend' kwarg.")

    jitted = _jit(func, static_argnames=("backend",))

    @wraps(func)
    def wrapper(*args, **kwargs):
        b = kwargs.get("backend", None)
        if b is None:
            return jitted(*args, **kwargs)
        if (isinstance(b, str) and b.lower() in ("np","numpy")) or (b is np):
            return func(*args, **kwargs)  # no JIT for NumPy
        return jitted(*args, **kwargs)
    return wrapper

# ---------------------------------------------------------------------
#! Types
# ---------------------------------------------------------------------


#! Define a registry for NumPy and JAX dtypes
DType = Union[Type[np.generic], Any]
_DTYPE_REGISTRY: Dict[str, Dict[Literal['numpy', 'jax'], DType]] = {}

#! Create a reverse mapping from dtype to name
_TYPE_TO_NAME: Dict[Any, str] = {}
for name, backends in _DTYPE_REGISTRY.items():
    for backend, dtype in backends.items():
        if dtype is not None:
            _TYPE_TO_NAME[dtype] = name
            
_TYPE_TO_NAME[np.complex64]     = 'complex64'
_TYPE_TO_NAME[np.complex128]    = 'complex128'
_TYPE_TO_NAME[np.float32]       = 'float32'
_TYPE_TO_NAME[np.float64]       = 'float64'
_TYPE_TO_NAME[np.int32]         = 'int32'
_TYPE_TO_NAME[np.int64]         = 'int64'
_TYPE_TO_NAME[int]              = 'int64'
_TYPE_TO_NAME[float]            = 'float64'
_TYPE_TO_NAME[complex]          = 'complex128'

def distinguish_type(typek: Any, backend: Literal['numpy', 'jax'] = 'numpy') -> DType:
    """
    Given a type (e.g. np.float32, jnp.int64, or int), return the corresponding
    dtype object in either NumPy or JAX.

    Parameters
    ----------
    typek
        A dtype class or Python int; may be from numpy, jax.numpy, or the builtin int.
    backend
        'numpy' or 'jax' — which library the returned dtype should belong to.

    Returns
    -------
    dtype
        The requested dtype object (e.g. np.float64 or jnp.int32).

    Raises
    ------
    ValueError
        If `typek` isn't one of the supported types, or if you ask for JAX
        but JAX isn't available.
    """

    try:
        name = _TYPE_TO_NAME[typek]
    except KeyError:
        raise ValueError(f"Unsupported state type: {typek!r}")

    entry   = _DTYPE_REGISTRY[name]
    dtype   = entry.get(backend)

    if dtype is None:
        if backend == 'jax' and not JAX_AVAILABLE:
            raise ValueError("JAX not installed or not available")
        raise ValueError(f"Type {name!r} not defined for backend {backend!r}")
    return dtype

# ---------------------------------------------------------------------
#! Functions
# ---------------------------------------------------------------------

def get_hardware_info() -> Tuple[int, int]:
    """
    Get the number of available JAX devices and CPU cores.

    Returns:
        n_devices :
            Number of JAX devices (e.g., GPUs/TPUs) if JAX is available, else 0.
        n_threads :
            Number of CPU cores available to the system.
    """
    n_devices   = 0
    if backend_mgr.is_jax_available and jax:
        try:
            n_devices = jax.device_count()
        except Exception as e:
            log.warning(f"Could not get JAX device count: {e}")
            n_devices = 0 # Fallback if detection fails

    n_threads   = multiprocessing.cpu_count()
    _log_message(f"Detected CPU cores: {n_threads}", 1)
    if n_devices > 0:
        _log_message(f"Detected devices: {n_devices}", 1)
    else:
        _log_message("No device detected.", 1)
    return n_devices, n_threads

# ---------------------------------------------------------------------

@dataclass
class RNGManager:
    np_rng  : np.random.Generator   | None
    jax_rng : Any                   | None
    py_rng  : py_random.Random      | None

# ---------------------------------------------------------------------

class BackendManager:
    """
    Manages the numerical backend (NumPy or JAX) state.

    Provides access to the appropriate linear algebra module (np/jnp),
    random number generator, SciPy module, JIT compiler, and backend info.

    Attributes:
        is_jax_available (bool): 
            True if JAX was successfully imported.
        name (str): 
            Name of the active backend ("numpy" or "jax").
        np (module): 
            The active array module (numpy or jax.numpy).
        random (module): 
            The active random module (numpy.random or jax.random).
        scipy (module): 
            The active SciPy module (scipy or jax.scipy).
        key (Optional[PRNGKey]): 
            The default JAX PRNG key (if JAX is active).
        jit (Callable): 
            The JIT compiler function (jax.jit or identity).
        default_seed (int): 
            The seed used for default RNG initialization.
        default_rng (np.random.Generator | np_random): 
            Default NumPy RNG instance.
        default_jax_key (Optional[PRNGKey]): 
            Default JAX key instance.
        int_dtype (Type): 
            Default integer type for the *active* backend.
        float_dtype (Type): 
            Default float type for the *active* backend.
        complex_dtype (Type): 
            Default complex type for the *active* backend.
    """
    
    def __init__(self, default_seed: int = DEFAULT_SEED, prefer_jax: bool = PREFER_JAX):
        """
        Initializes the manager, detects JAX, and sets the active backend.

        Args:
            default_seed:
                The seed for initializing default random generators.
            prefer_jax:
                If True and JAX is available, use JAX as the default.
                Otherwise, use NumPy.
        """
        self.default_seed           : int   = default_seed
        self.is_jax_available       : bool  = JAX_AVAILABLE

        #! Initialize NumPy components first as fallback
        self._np_module             = np
        self._sp_module             = sp
        self._np_random_module      = np_random # Store the base module
        self.default_rng            = self._create_numpy_rng(self.default_seed)
        
        #! Active backend defaults (start with NumPy)
        self.name                   : str               = "numpy"
        self.np                     : Any               = self._np_module
        self.random                 : Any               = self.default_rng  # Use the Generator instance by default
        self.scipy                  : Any               = self._sp_module   # SciPy module
        self.key                    : Optional[PRNGKey] = None              # Key for the random module
        self.jit                    : Callable          = lambda x: x       # Identity function

        #! JAX specific components (if available)
        self._jax_module            = None
        self._jnp_module            = None
        self._jsp_module            = None
        self._jrn_module            = None
        self._jax_jit               = None
        self.default_jax_key        : Optional[PRNGKey] = None

        if self.is_jax_available and jax and jnp and jsp and jrn and jax_jit and jcfg:
            self._jax_module        = jax
            self._jnp_module        = jnp
            self._jsp_module        = jsp
            self._jrn_module        = jrn
            self._jax_jit           = jax_jit # The imported jax.jit
            self.default_jax_key    = self._create_jax_key(self.default_seed)
            self._update_device()

            if prefer_jax:
                log.info("Setting JAX as the active backend.")
                self.set_active_backend("jax")
    
        self.detected_jax_backend: Optional[str]                = getattr(self, "detected_jax_backend", None)
        self.detected_jax_devices: Optional[List[JaxDevice]]    = getattr(self, "detected_jax_devices", None)
    
        #! Set active dtypes based on the chosen backend
        self._update_dtypes()

        env_seed = os.getenv(PY_GLOBAL_SEED_STR, "").strip()
        if len(env_seed) > 0:
            try:
                self.reseed(int(env_seed))
            except Exception as e:
                log.warning(f"Ignoring PY_GLOBAL_SEED={env_seed!r}: {e}")
        
    # ---------------------------------------------------------------------
    
    def _update_device(self):
        '''
        Detects the JAX backend and devices after import.
        Checks if JAX is available and lists the available devices plus number of threads.
        Otherwise, sets the backend to NumPy.
        '''
        
        # Reset state before detection
        self.detected_jax_backend = None
        self.detected_jax_devices = None
        self._jax_functional      = False

        if not self.is_jax_available or not jax or not jax.lib: # Extra safety check
            log.warning("Attempted _update_device without JAX being available/imported.")
            return

        try:
            # Use xla_bridge from the imported jax module
            self.detected_jax_backend   = jax.default_backend() # 'cpu'/'gpu'/'tpu'
            # Use preferred jax.local_devices()
            try:
                self.detected_jax_devices   = jax.devices() 
            except Exception:
                self.detected_jax_devices   = jax.local_devices()

            if not self.detected_jax_devices:
                log.warning("JAX backend detected, but no devices found!")
                self._jax_functional        = False # No devices = not functional for most purposes
            else:
                # Found backend AND devices
                self._jax_functional        = True
                # Logging moved to __init__ after this call returns

        except AttributeError as ae:
            # Handle cases where xla_bridge might be missing parts (unlikely with full install)
            log.error(f"AttributeError during JAX backend/device detection: {ae}. JAX likely not fully functional.", exc_info=True)
            self.detected_jax_backend   = "Detection Error (Attribute)"
            self.detected_jax_devices   = []
            self._jax_functional        = False
        except Exception as e:
            log.error(f"An unexpected error occurred during JAX backend/device detection: {e}. "
                        f"JAX backend might not be functional.", exc_info=True)
            self.detected_jax_backend   = "Detection Error (Exception)"
            self.detected_jax_devices   = []
            self._jax_functional        = False

    # ---------------------------------------------------------------------
    
    def _update_dtypes(self):
        """
        Updates active dtype attributes based on the active backend.
        Sets int_dtype, float_dtype, and complex_dtype to the default types
        for the active backend (NumPy or JAX).
        """
        if self.name == "jax" and self.is_jax_available:
            # Use the JAX defaults stored globally if JAX is active
            self.int_dtype      = DEFAULT_JP_INT_TYPE if DEFAULT_JP_INT_TYPE else DEFAULT_NP_INT_TYPE # Fallback
            self.float_dtype    = DEFAULT_JP_FLOAT_TYPE if DEFAULT_JP_FLOAT_TYPE else DEFAULT_NP_FLOAT_TYPE
            self.complex_dtype  = DEFAULT_JP_CPX_TYPE if DEFAULT_JP_CPX_TYPE else DEFAULT_NP_CPX_TYPE
        else:
            # Use NumPy defaults otherwise
            self.int_dtype      = DEFAULT_NP_INT_TYPE
            self.float_dtype    = DEFAULT_NP_FLOAT_TYPE
            self.complex_dtype  = DEFAULT_NP_CPX_TYPE
            
        log.debug(f"Active dtypes set for backend '{self.name}': "
                    f"int={getattr(self.int_dtype, '__name__', 'N/A')}, "
                    f"float={getattr(self.float_dtype, '__name__', 'N/A')}, "
                    f"complex={getattr(self.complex_dtype, '__name__', 'N/A')}")
    
    # ----------------------------------------------------------------------
    #! Active Backend Management
    # ----------------------------------------------------------------------    
    
    def set_active_backend(self, name: str):
        """
        Explicitly sets the active backend globally managed by this instance.

        Args:
            name :
                "numpy, npy, np" or "jax".

        Raises:
            ValueError:
                If 'jax' is requested but not available, or invalid name.
        """
        name = name.lower()
        if name == "numpy" or name == "npy" or name == "np":
            self.name       = "numpy"
            self.np         = self._np_module
            self.random     = self.default_rng # Use the numpy generator instance
            self.scipy      = self._sp_module
            self.key        = None
            self.jit        = lambda x: x
            log.info("Switched active backend to NumPy.")
        elif name == "jax":
            if not self.is_jax_available or not self._jnp_module or not self._jrn_module or not self._jsp_module or not self._jax_jit:
                raise ValueError("Cannot set 'jax' backend: JAX components not fully available.")
            self.name       = "jax"
            self.np         = self._jnp_module
            self.random     = self._jrn_module # Use the jax random module
            self.scipy      = self._jsp_module
            self.key        = self.default_jax_key # Use the default stored key
            self.jit        = self._jax_jit
            log.info("Switched active backend to JAX.")
        else:
            raise ValueError(f"Invalid backend name: {name}. Choose 'numpy' or 'jax'.")
        self._update_dtypes() # Update dtypes after switching

    # ---------------------------------------------------------------------
    #! Random Number Generation Initialization
    # ---------------------------------------------------------------------

    @staticmethod
    def _create_numpy_rng(seed: Optional[int]) -> Union[np_random.Generator, np_random.RandomState]:
        """
        Creates a NumPy random number generator instance.
        If NumPy >= 1.17, uses the Generator API.
        Otherwise, falls back to the legacy RandomState API.
        
        Parameters:
            seed (int or None):
                Seed for the random number generator. If None, uses the default seed.
                
        Returns:
            Union[np_random.Generator, np_random.RandomState]:
                A NumPy random number generator instance.
        """
        
        if hasattr(np_random, 'default_rng'):
            if seed is not None:
                # Seed legacy global state only if a specific seed is given
                # Avoids potentially unwanted side effects if seed is None
                try:
                    np.random.seed(seed)
                except ValueError: # Handle potential large seed issues for legacy
                    log.warning(f"Could not seed legacy np.random with seed {seed}. Using default.")
                    np.random.seed(DEFAULT_SEED)
            return np_random.default_rng(seed)
        else:
            #! Legacy RandomState API for NumPy < 1.17
            log.warning(f"NumPy version {np.__version__} < 1.17. Using legacy np.random state.")
            rng_instance = np_random.RandomState(seed)
            # Monkey patch default_rng onto the instance if it doesn't exist, for potential API consistency attempts
            # This is somewhat fragile and mainly for internal consistency here.
            if not hasattr(rng_instance, 'default_rng'):
                rng_instance.default_rng = lambda s=seed: np_random.RandomState(s)
            return rng_instance

    @staticmethod
    def _create_jax_key(seed: int) -> Optional[PRNGKey]:
        """
        Creates a JAX PRNG key.
        If JAX is available, uses the PRNGKey function.
        Otherwise, returns None.
        
        Parameters:
            seed (int):
                Seed for the PRNG key.
        
        Returns:
            Optional[PRNGKey]:
                A JAX PRNG key if JAX is available, otherwise None.
        """
        if JAX_AVAILABLE and jrn:
            return jrn.PRNGKey(seed)
        return None

    # ---------------------------------------------------------------------
    
    def print_info(self):
        """
        Prints backend configuration and library versions in a table.
        Displays the active backend, available libraries, and their versions.
        Also shows the active integer, float, and complex types.
        
        The output is formatted for better readability.
        """

        # Collect version information for each library.
        backend_versions = {
            "NumPy": getattr(np, '__version__', 'Unknown'),
            "SciPy": getattr(sp, '__version__', 'Unknown'), 
            "JAX": "Not Available"
        }

        # If JAX is available, get its actual version
        if self.is_jax_available and self._jax_module:
            backend_versions["JAX"] = getattr(self._jax_module, '__version__', 'Unknown')

        # Print header.
        _log_message("*"*50, 0)
        _log_message("Backend Configuration:", 0)

        # Log version info.
        for lib, version in backend_versions.items():
            _log_message(f"{lib} Version: {version}", 2)

        # Log active backend details.
        _log_message(f"Active Backend: {self.name}", 2)
        _log_message(f"JAX Available: {self.is_jax_available}", 3)
        _log_message(f"Default Seed: {self.default_seed}", 3)

        # Log current backend modules.
        if self.name == "jax":
            _log_message("JAX Backend Details:", 2)
            _log_message(f"\tMain Module: {self.np.__name__}", 3)
            _log_message(f"\tRandom Module: {self.random.__name__} (+ PRNGKey)", 3)
            _log_message(f"\tSciPy Module: {self.scipy.__name__}", 3)
            _log_message(f"\tDefault JAX Key: PRNGKey({self.default_seed})", 3)
        elif self.name == "numpy":
            _log_message("NumPy Backend Details:", 2)
            _log_message(f"\tMain Module: {self.np.__name__}", 3)
            _log_message(f"\tRandom Module: {self.random.__class__.__name__}", 3)
            _log_message(f"\tSciPy Module: {self.scipy.__name__}", 3)

        # Log active data types.
        _log_message("Active Data Types:", 2)
        _log_message(f"\tInteger Type: {self.int_dtype.__name__}", 3)
        _log_message(f"\tFloat Type: {self.float_dtype.__name__}", 3)
        _log_message(f"\tComplex Type: {self.complex_dtype.__name__}", 3)

        #! Format device detection results
        _log_message("Hardware & Device Detection:", 2)
        # Use manager's stored info
        n_threads = multiprocessing.cpu_count()
        _log_message(f"CPU Cores: {n_threads}", 3)

        if self.is_jax_available:
            detected_backend_str = (self.detected_jax_backend or "Detection Failed").upper()
            _log_message(f"Detected JAX Platform: {detected_backend_str}", 3)

            device_summary = "N/A"
            if self.detected_jax_devices is not None: # Check if detection was attempted
                if self.detected_jax_devices: # Check if list is not empty
                    try:
                        platforms       = [d.platform.upper() for d in self.detected_jax_devices if hasattr(d, 'platform')]
                        # Get client kind for more detail if available
                        kinds           = [d.client.platform if hasattr(d, 'client') and hasattr(d.client,'platform') else platforms[i] for i, d in enumerate(self.detected_jax_devices)]
                        device_summary  = f"{len(self.detected_jax_devices)} devices ({', '.join(kinds)})"
                    except Exception:
                        device_summary  = f"{len(self.detected_jax_devices)} devices (Details Error)"
                else:
                    device_summary = "No JAX devices found!"
            else:
                device_summary = "Detection Failed or Not Run"
            _log_message(f"JAX Devices Found: {device_summary}", 2)
        else:
            _log_message(f"JAX Platform: Not Applicable", 2)
            _log_message(f"JAX Devices Found: Not Applicable", 2)


        # Footer.
        _log_message("*" * 50 + "\n\n\n", 0)

    # ---------------------------------------------------------------------
    #! Backend Module Retrieval
    # ---------------------------------------------------------------------
    
    def _get_numpy_modules(self, use_random: bool = False, seed: Optional[int] = None, use_scipy: bool = False) -> Union[Any, Tuple[Any, ...]]:
        """
        Returns NumPy backend modules.
        
        - If use_random is True, returns the random module and a key (rng, key=None).
        
        - If use_scipy is True, returns the SciPy module.
        
        Args:
            use_random (bool):
                If True, include the random module/state.
            seed (int or None):
                Seed for the random number generator.
                If None, uses the manager's default seed.
            use_scipy (bool):
                If True, include the SciPy module.
        """
        
        # get the main NumPy-like module
        main_module         = self._np_module
        results: list[Any]  = [main_module]

        if use_random:
            # If a specific seed is requested, create a new RNG for that seed.
            # Otherwise, use the manager's default RNG instance.
            current_seed    = seed if seed is not None else self.default_seed
            # Always create a new RNG instance when requested via get_backend,
            # even if seed matches default, to ensure independence.
            rng_instance    = self._create_numpy_rng(current_seed)
            #! Tuple format (rng, key=None)
            results.append((rng_instance, None)) 

        if use_scipy:
            results.append(self._sp_module)
            
        return tuple(results) if len(results) > 1 else main_module

    def _get_jax_modules(self, use_random: bool = False, seed: Optional[int] = None, use_scipy: bool = False) -> Union[Any, Tuple[Any, ...]]:
        """
        Returns JAX backend modules.
        
        - If use_random is True, returns the random module and a key (jax.random, key).
        
        - If use_scipy is True, returns the SciPy module.
        
        - If JAX is not available, raises a ValueError.
        
        Args:
            use_random (bool):
                If True, include the random module/state.
            seed (int or None):
                Seed for the random number generator.
                If None, uses the manager's default seed.
            use_scipy (bool):
                If True, include the SciPy module.
        Raises:
            ValueError:
                If JAX is not available or if required JAX components are missing.
        Returns:
            Union[Any, Tuple[Any, ...]]:
                The requested module(s) as a single module or a tuple.
                Format: (main_module) or (main_module, random_part, scipy_module)
                where random_part is (jax.random, key) for JAX.
        """
        if not self.is_jax_available or not self._jnp_module or not self._jrn_module or not self._jsp_module:
            raise ValueError("JAX backend requested but required JAX components are not available.")

        main_module         = self._jnp_module
        results: list[Any]  = [main_module]

        if use_random:
            # If a specific seed is requested, create a new key.
            # Otherwise, use the manager's default key.
            current_seed    = seed if seed is not None else self.default_seed
            # Always create a new key when requested via get_backend.
            current_key     = self._create_jax_key(current_seed)
            # Tuple format (module, key)
            results.append((self._jrn_module, current_key))

        if use_scipy:
            results.append(self._jsp_module)

        return tuple(results) if len(results) > 1 else main_module

    # ---------------------------------------------------------------------
    
    def get_backend_modules(self, 
                            backend_spec    : Union[str, Any, None],
                            use_random      : bool          = False,
                            seed            : Optional[int] = None,
                            use_scipy       : bool          = False) -> Union[Any, Tuple[Any, ...]]:
        """
        Returns backend modules based on the specifier.

        Args:
            backend_spec :
                Backend identifier. Can be:
                - String: "numpy", "np", "jax", "jnp", "default".
                - Module: `numpy` or `jax.numpy`.
                - None: Uses the manager's active backend.
            use_random :
                If True, include the random module/state. For JAX,
                returns (jax.random, key). For NumPy, returns
                (rng_instance, None).
            seed :
                Seed for the random number generator. If None, uses
                the manager's default seed. If provided, creates a
                *new* RNG/Key for this request, independent of the
                manager's default state.
            use_scipy       : If True, include the SciPy module.

        Returns:
            The requested module(s) as a single module or a tuple.
            Format:
                (main_module) or (main_module, random_part, scipy_module)
                where random_part is (rng_instance, None) for numpy or (jrn_module, key) for jax.
        """
        if backend_spec is None or backend_spec == "default":
            backend_name = self.name
        elif isinstance(backend_spec, str):
            backend_name = backend_spec.lower()
        elif backend_spec is np or backend_spec is self._np_module:
            backend_name = "numpy"
        elif self.is_jax_available and self._jnp_module and (backend_spec is jnp or backend_spec is self._jnp_module):
            backend_name = "jax"
        else:
            raise ValueError(f"Unsupported backend specification: {backend_spec}")

        #! Dispatch based on name
        if backend_name in ("numpy", "np", "npy"):
            return self._get_numpy_modules(use_random=use_random, seed=seed, use_scipy=use_scipy)
        elif backend_name in ("jax", "jnp", "jaxpy"):
            # _get_jax_modules performs its own availability check
            return self._get_jax_modules(use_random=use_random, seed=seed, use_scipy=use_scipy)
        else:
            raise ValueError(f"Unknown backend name derived: {backend_name}")

    # ---------------------------------------------------------------------
    
    def get_global_backend_modules(self,
                                use_random  : bool          = False,
                                seed        : Optional[int] = None,
                                use_scipy   : bool          = False) -> Union[Any, Tuple[Any, ...]]:
        """
        Returns the globally configured default backend modules from the manager.

        Uses the manager's current active backend (`self.name`). If a specific `seed`
        is provided, it generates a random component based on that seed for this
        request, otherwise uses the manager's default seed/key framework.

        Args:
            use_random :
                If True, include the random module/state.
            seed :
                Optional seed for this specific request's random component.
            use_scipy :
                If True, include the SciPy module.

        Returns:
            The global default backend module(s). See `get_backend_modules` for format.
        """
        # Pass the current name and arguments to the main getter
        return self.get_backend_modules(self.name, use_random=use_random, seed=seed, use_scipy=use_scipy)

    # ---------------------------------------------------------------------
    #! RANDOMNESS
    # ---------------------------------------------------------------------

    def reseed(self, seed: int) -> RNGManager:
        """
        Reseed the manager's RNGs without doing work at import time.
        Returns an RNGManager instance you can stash if needed.
        """
        self.default_seed = int(seed)

        # NumPy: use Generator; avoid global np.random state unless explicitly requested
        self.default_rng = self._create_numpy_rng(self.default_seed)

        # Python stdlib random (optional, handy for code that uses it)
        py_random.seed(self.default_seed)
        py_state = py_random.getstate()

        # JAX: reset the main key if available
        if self.is_jax_available:
            self.default_jax_key    = self._create_jax_key(self.default_seed)
            self.key                = self.default_jax_key

        # Keep ACTIVE_* mirrors in sync if you expose them
        if self.name == "numpy":
            self.random             = self.default_rng
            self.key                = None
        else:  # jax active
            self.random             = self._jrn_module
            self.key                = self.default_jax_key
            
        return RNGManager(self.default_rng, self.default_jax_key if self.is_jax_available else None, py_state)

    def next_key(self) -> PRNGKey:
        """
        Return a fresh JAX subkey and advance the manager's internal key.
        """
        if not (self.is_jax_available and self.key is not None):
            raise RuntimeError("JAX key not available; ensure JAX backend and call reseed() first.")
        self.key, sub = self._jrn_module.split(self.key)
        return sub

    def split_keys(self, n: int) -> Any:
        """
        Return `n` fresh subkeys and advance the manager's internal key once.
        """
        if not (self.is_jax_available and self.key is not None):
            raise RuntimeError("JAX key not available; ensure JAX backend and call reseed() first.")
        self.key, k0 = self._jrn_module.split(self.key)
        return self._jrn_module.split(k0, n)

    @contextmanager
    def seed_scope(self, seed: int, *, touch_numpy_global: bool = False, touch_python_random: bool = True):
        """
        Temporarily set deterministic seeds and restore previous states on exit.
        Use it as:
        ```
        with seed_scope(seed):
            # Your code here
        ```

        Parameters
        ----------
        seed : int
            The seed to set.
        touch_numpy_global : bool
            Whether to touch the global NumPy random state.
        touch_python_random : bool
            Whether to touch the Python random state.
        """
        # Save old states
        old_np_rng = self.default_rng
        old_seed   = self.default_seed
        old_key    = self.key
        old_py     = py_random.getstate()

        old_np_global_state = None
        if touch_numpy_global:
            old_np_global_state = np.random.get_state()

        # Set new
        suite = self.reseed(seed)
        if touch_numpy_global:
            np.random.seed(seed)  # legacy global
        if touch_python_random:
            py_random.seed(seed)

        try:
            yield suite
        finally:
            # Restore
            self.default_seed = old_seed
            self.default_rng  = old_np_rng
            self.key          = old_key
            if touch_python_random:
                py_random.setstate(old_py)
            if touch_numpy_global and old_np_global_state is not None:
                np.random.set_state(old_np_global_state)

    # Multithreaded jobs
    
    def spawn_np_generators(root_seed: int, n: int) -> list[np.random.Generator]:
        """
        Create `n` independent NumPy generators using SeedSequence.
        Use one per worker/process to avoid correlated streams.
        """
        ss = np.random.SeedSequence(root_seed)
        return [np.random.Generator(np.random.PCG64(s)) for s in ss.spawn(n)]

    def spawn_jax_keys(root_key: PRNGKey, n: int):
        """
        Deterministically produce `n` independent JAX keys.
        """
        return jrn.split(root_key, n) if JAX_AVAILABLE else [None] * n

# ---------------------------------------------------------------------
# ONE-TIME INITIALIZATION LOGIC
#
# This function modifies the global variables declared previously.
# It runs only once per Python session.
# ---------------------------------------------------------------------

def _qes_initialize_utils():
    """
    Performs one-time setup of the backend environment. This function
    modifies the module's global variables.
    """
    # Tell this function we are modifying the module-level (global) variables
    global log, JAX_AVAILABLE, jax, jnp, jsp, jrn, jcfg, JIT
    global Array, PRNGKey, JaxDevice
    global DEFAULT_JP_INT_TYPE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE
    global PREFER_JAX, PREFER_64BIT, PREFER_32BIT
    global _DTYPE_REGISTRY, _TYPE_TO_NAME
    global backend_mgr
    global ACTIVE_BACKEND_NAME, ACTIVE_NP_MODULE, ACTIVE_RANDOM
    global ACTIVE_SCIPY_MODULE, ACTIVE_JIT, ACTIVE_JAX_KEY
    global ACTIVE_INT_TYPE, ACTIVE_FLOAT_TYPE, ACTIVE_COMPLEX_TYPE

    # 1. Setup Logger
    try:
        from ..common.flog import get_global_logger
        log = get_global_logger()
    except ImportError:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.info("QES global logger not found. Using standard logging.")

    _log_message("Initializing QES.general_python.algebra.utils...")

    # 2. Environment and Core Settings
    num_cores                       = os.cpu_count() or 1
    os.environ[PY_NUM_CORES_STR]    = os.getenv(PY_NUM_CORES_STR, str(num_cores))

    # 3. JAX Detection and Import
    if PREFER_JAX:
        try:
            import jax
            from jax import config as jax_config
            jcfg = jax_config

            if PREFER_64BIT:
                jcfg.update("jax_enable_x64", True)
                _log_message("JAX 64-bit precision enabled.", 1)

            logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)
            logging.getLogger('jax').setLevel(logging.WARNING)

            import jax.numpy as jnp
            import jax.scipy as jsp
            import jax.random as jrn

            JAX_AVAILABLE = True
            JIT = jax.jit  # Overwrite the global JIT function
            os.environ[PY_JAX_AVAILABLE_STR] = '1'
            _log_message("JAX backend available and successfully imported.", 0)

        except ImportError:
            _log_message("JAX backend not available. Falling back to NumPy.", 0)
            JAX_AVAILABLE = False
    else:
        _log_message("JAX is not preferred. Using NumPy backend.", 0)
        JAX_AVAILABLE = False

    if JAX_AVAILABLE:
        # Type aliases for JAX
        PRNGKey     = jrn.PRNGKey
        JaxDevice   = jax.lib.xla_client.Device if hasattr(jax.lib, 'xla_client') and hasattr(jax.lib.xla_client, 'Device') else Any
        Array       = Union[np.ndarray, jnp.ndarray]
        if PREFER_32BIT:
            jcfg.update("jax_enable_x64", False)
            _log_message("JAX 32-bit precision enforced.", 1)
        DEFAULT_JP_INT_TYPE     = getattr(jnp, 'int64', getattr(jnp, 'int32'))          # Prefer 64bit if available
        DEFAULT_JP_FLOAT_TYPE   = getattr(jnp, 'float64', getattr(jnp, 'float32'))      # Prefer 64bit if available
        DEFAULT_JP_CPX_TYPE     = getattr(jnp, 'complex128', getattr(jnp, 'complex64')) # Prefer 128bit if available
        _log_message(f"JAX default types: int={DEFAULT_JP_INT_TYPE.__name__}, float={DEFAULT_JP_FLOAT_TYPE.__name__}, complex={DEFAULT_JP_CPX_TYPE.__name__}", 2)
    else:
        # Type aliases for NumPy only
        PRNGKey                 = Any
        JaxDevice               = Any
        Array                   = np.ndarray
        DEFAULT_JP_INT_TYPE     = None
        DEFAULT_JP_FLOAT_TYPE   = None
        DEFAULT_JP_CPX_TYPE     = None
    
    # 4. Update Type Aliases and Registries based on JAX status
    if JAX_AVAILABLE and jnp:
        Array                               = Union[np.ndarray, jnp.ndarray]
        _TYPE_TO_NAME[jnp.complex64]        = 'complex64'
        _TYPE_TO_NAME[jnp.complex128]       = 'complex128'
        _TYPE_TO_NAME[jnp.float32]          = 'float32'
        _TYPE_TO_NAME[jnp.float64]          = 'float64'
        _TYPE_TO_NAME[jnp.int32]            = 'int32'
        _TYPE_TO_NAME[jnp.int64]            = 'int64'
    _log_message(f"Type registries updated. Supported types: {list(_TYPE_TO_NAME.values())}")

    # Register NumPy types as well
    _DTYPE_REGISTRY['float32']      = {'numpy': np.float32,     'jax': jnp.float32 if JAX_AVAILABLE else None}
    _DTYPE_REGISTRY['float64']      = {'numpy': np.float64,     'jax': jnp.float64 if JAX_AVAILABLE else None}
    _DTYPE_REGISTRY['int32']        = {'numpy': np.int32,       'jax': jnp.int32 if JAX_AVAILABLE else None}
    _DTYPE_REGISTRY['int64']        = {'numpy': np.int64,       'jax': jnp.int64 if JAX_AVAILABLE else None}
    _DTYPE_REGISTRY['complex64']    = {'numpy': np.complex64,   'jax': jnp.complex64 if JAX_AVAILABLE else None}
    _DTYPE_REGISTRY['complex128']   = {'numpy': np.complex128,  'jax': jnp.complex128 if JAX_AVAILABLE else None}
    _log_message(f"Data type registry populated with NumPy and JAX types.", 2)


    # 5. Instantiate and Configure the Backend Manager
    backend_mgr = BackendManager(default_seed=DEFAULT_SEED, prefer_jax=PREFER_JAX)
    _log_message(f"BackendManager instantiated with default seed {DEFAULT_SEED}.", 1)

    # 6. Update the Global ACTIVE_* Mirrors from the Manager
    # This makes the active backend components directly accessible.
    ACTIVE_BACKEND_NAME     = backend_mgr.name
    ACTIVE_NP_MODULE        = backend_mgr.np
    ACTIVE_RANDOM           = backend_mgr.random
    ACTIVE_SCIPY_MODULE     = backend_mgr.scipy
    ACTIVE_JIT              = backend_mgr.jit
    ACTIVE_JAX_KEY          = backend_mgr.key
    ACTIVE_INT_TYPE         = backend_mgr.int_dtype
    ACTIVE_FLOAT_TYPE       = backend_mgr.float_dtype
    ACTIVE_COMPLEX_TYPE     = backend_mgr.complex_dtype

    # 7. Final Info Printout
    if os.getenv(PY_INFO_VERBOSE, "0").lower() in ("1", "true", "yes", "on"):
        backend_mgr.print_info()

    os.environ[PY_UTILS_INIT_DONE_STR] = '1'
    _log_message("'[General Python].algebra.utils initialization complete.", 0)

# ---------------------------------------------------------------------
# EXECUTION GUARD
#
# This code runs when the module is imported. It ensures that the
# initialization function is called only once.
# ---------------------------------------------------------------------

if "PY_UTILS_INIT_DONE" not in globals() or PY_UTILS_INIT_DONE_STR not in os.environ:
    PY_UTILS_INIT_DONE = True  # Mark as done immediately
    try:
        _qes_initialize_utils()
    except Exception as e:
        log.error(f"CRITICAL ERROR during backend initialization: {e}")
        # We exit because the library is in an unusable state.
        os._exit(1)
else:
    # This message is helpful for debugging re-import issues.
    _log_message("QES.general_python.algebra.utils already initialized; skipping re-initialization.", 0)
    _log_message("---------------------------------------------------------------------------------", 0)
    
# ---------------------------------------------------------------------
#! EOF
# ---------------------------------------------------------------------

def pad_array(x, target_size: int, pad_value, *, backend=None):
    xp  = backend or (jnp if (JAX_AVAILABLE and is_jax_array(x)) else np)
    out = xp.full((target_size,), pad_value, dtype=x.dtype)
    if xp is np:
        out[:x.shape[0]] = x
        return out
    # JAX path
    return out.at[:x.shape[0]].set(x)

# ---------------------------------------------------------------------
