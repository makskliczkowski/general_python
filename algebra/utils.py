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
import inspect
import multiprocessing
import logging
from typing import Union, Optional, TypeAlias, Type, Tuple, Any, Callable, List, Dict, Literal
from general_python.common.flog import get_global_logger

# ---------------------------------------------------------------------
log = get_global_logger()
# ---------------------------------------------------------------------\

def _log_message(msg, lvl = 0):
    """
    Logs a message using the global logger.
    This function ensures the logger is only imported when needed.
    
    Parameters:
        msg (str):
            The message to log.
        lvl (int):
            The indentation level for the message.
    """
    text = "\t" * lvl + msg
    log.info(text)

# ---------------------------------------------------------------------

import numpy as np
import numpy.random as np_random
import scipy as sp

num_cores                           = os.cpu_count()
os.environ['NUMEXPR_MAX_THREADS']   = str(num_cores)

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

PREFER_JAX              : bool              = os.environ.get("BACKEND", "jax").lower() != "numpy"


#! Backend Detection
JAX_AVAILABLE           = False
jax                     = None
jnp                     = None
jsp                     = None
jrn                     = None
jax_jit                 = lambda x: x
jcfg                    = None

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import jax.random as jrn
    from jax import config as jax_config
    
    try:
        logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)
        logging.getLogger('jax').setLevel(logging.WARNING)
        # jax.config.update('jax_log_compiles', True)
    except Exception as e:
        log.debug(f"Could not configure JAX logger levels: {e}")
    
    JAX_AVAILABLE           = True
    jit                     = jax.jit # Use real JIT if JAX is available
    jcfg                    = jax_config
    _log_message("JAX backend available and successfully imported", 0)

    # Set JAX specific default types *after* successful import
    DEFAULT_JP_INT_TYPE     = getattr(jnp, 'int64', getattr(jnp, 'int32'))          # Prefer 64bit if available
    DEFAULT_JP_FLOAT_TYPE   = getattr(jnp, 'float64', getattr(jnp, 'float32'))      # Prefer 64bit if available
    DEFAULT_JP_CPX_TYPE     = getattr(jnp, 'complex128', getattr(jnp, 'complex64')) # Prefer 128bit if available

    # Set JAX global configuration by enabling 64-bit precision if available
    try:
        jcfg.update("jax_enable_x64", True)
        _log_message("JAX 64-bit precision enabled.", 1)
    except Exception as e:
        _log_message(f"Could not enable JAX 64-bit precision: {e}", 1)
except ImportError:
    _log_message("JAX backend not available. Falling back to NumPy.", 0)
    pass

#! Type Aliases
if JAX_AVAILABLE and jnp:
    Array       : TypeAlias = Union[np.ndarray, jnp.ndarray]
    PRNGKey     : TypeAlias = Any # jax.random.PRNGKeyArray
    JaxDevice   : TypeAlias = Any # Placeholder for jax device type
else:
    Array       : TypeAlias = np.ndarray
    PRNGKey     : TypeAlias = None
    JaxDevice   : TypeAlias = None

# ---------------------------------------------------------------------
#! Types
# ---------------------------------------------------------------------

DType           = Union[Type[np.generic], Any]

#! Define a registry for NumPy and JAX dtypes
_DTYPE_REGISTRY: Dict[str, Dict[Literal['numpy', 'jax'], DType]] = {
    'float32': {
        'numpy': np.float32,
        'jax':  jnp.float32 if JAX_AVAILABLE else None,
    },
    'float64': {
        'numpy': np.float64,
        'jax':  jnp.float64 if JAX_AVAILABLE else None,
    },
    'int32': {
        'numpy': np.int32,
        'jax':  jnp.int32 if JAX_AVAILABLE else None,
    },
    'int64': {
        'numpy': np.int64,
        'jax':  jnp.int64 if JAX_AVAILABLE else None,
    },
    'complex64': {
        'numpy': np.complex64,
        'jax':  jnp.complex64 if JAX_AVAILABLE else None,
    },
    'complex128': {
        'numpy': np.complex128,
        'jax':  jnp.complex128 if JAX_AVAILABLE else None,
    },
}

#! Create a reverse mapping from dtype to name
_TYPE_TO_NAME: Dict[Any, str] = {}
for name, backends in _DTYPE_REGISTRY.items():
    for backend, dtype in backends.items():
        if dtype is not None:
            _TYPE_TO_NAME[dtype] = name
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
        If `typek` isn’t one of the supported types, or if you ask for JAX
        but JAX isn’t available.
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

# try:
#     N_DEVICES, N_THREADS    = get_hardware_info()
# except Exception as e:
#     log.warning(f"Could not get hardware info: {e}")
#     N_DEVICES               = 0
#     N_THREADS               = multiprocessing.cpu_count()

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
        self.default_rng            = self._create_numpy_rng(self.default_seed) # Default instance
        np.random.seed(self.default_seed)                                   # Also seed legacy global state if needed

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

        if self.is_jax_available and jax and jnp and jsp and jrn and jit and jcfg:
            self._jax_module        = jax
            self._jnp_module        = jnp
            self._jsp_module        = jsp
            self._jrn_module        = jrn
            self._jax_jit           = jit # The imported jax.jit
            self.default_jax_key    = self._create_jax_key(self.default_seed)
            self._update_device()

            if prefer_jax:
                log.info("Setting JAX as the active backend.")
                self.name           = "jax"
                self.np             = self._jnp_module
                self.random         = self._jrn_module # JAX random module
                self.scipy          = self._jsp_module
                self.key            = self.default_jax_key # Default JAX key
                self.jit            = self._jax_jit
            else:
                _log_message("JAX is available but not set as the active backend.", 1)
        else:
            log.info("JAX is not available. Using NumPy as the active backend.")

        self.detected_jax_backend   : Optional[str] = None
        self.detected_jax_devices   : Optional[List[JaxDevice]] = None
    
        #! Set active dtypes based on the chosen backend
        self._update_dtypes()
    
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
            self.detected_jax_backend   = jax.lib.xla_bridge.get_backend().platform
            # Use preferred jax.local_devices()
            self.detected_jax_devices   = jax.local_devices()

            if not self.detected_jax_devices:
                log.warning("JAX backend detected, but no devices found!")
                self._jax_functional    = False # No devices = not functional for most purposes
            else:
                # Found backend AND devices
                self._jax_functional    = True
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
            "NumPy": np.__version__,
            "SciPy": sp.__version__,
            "JAX": "Not Available"
        }

        # If JAX is available, get its actual version
        if self.is_jax_available and self._jax_module:
            backend_versions["JAX"] = self._jax_module.__version__

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
    
# -------------------------------------------------------------------------

try:
    # Instantiate the BackendManager globally.
    backend_mgr             = BackendManager(default_seed=DEFAULT_SEED, prefer_jax=PREFER_JAX)
    
    # Expose core ACTIVE components globally for convenience, derived from the manager.
    ACTIVE_BACKEND_NAME     = backend_mgr.name          # Active backend name ("numpy" or "jax")
    ACTIVE_NP_MODULE        = backend_mgr.np            # Active NumPy-like module (numpy or jax.numpy)
    ACTIVE_RANDOM           = backend_mgr.random        # Active random module (rng instance for numpy, module for jax)
    ACTIVE_SCIPY_MODULE     = backend_mgr.scipy         # Active SciPy module (scipy or jax.scipy)
    ACTIVE_JIT              = backend_mgr.jit           # Active JIT function (identity for numpy, jax.jit for jax)
    ACTIVE_JAX_KEY          = backend_mgr.key           # Default JAX PRNG key (if JAX is active)
    ACTIVE_INT_TYPE         = backend_mgr.int_dtype     # Active integer type for the backend
    ACTIVE_FLOAT_TYPE       = backend_mgr.float_dtype   # Active float type for the backend
    ACTIVE_COMPLEX_TYPE     = backend_mgr.complex_dtype # Active complex type for the backend
    DEFAULT_BACKEND         = backend_mgr._np_module    # Default NumPy module (numpy or jax.numpy)
    JIT                     = backend_mgr.jit           # JIT function (identity for numpy, jax.jit for jax)
    DEFAULT_BACKEND_KEY     = backend_mgr.key           # Default JAX key (if JAX is active)
    backend_mgr.print_info() # Print backend info
except ImportError as e:
    log.error(f"Error importing backend modules: {e}")
except AttributeError as e:
    log.error(f"Error accessing backend attributes: {e}")
except Exception as e:
    log.error(f"Error printing backend info: {e}")

# ---------------------------------------------------------------------

def is_jax_array(n: Any):
    """
    Checks if an object is likely a JAX array (including traced).
    Parameters:
    ----------
    n : Any
        The object to check.
    Returns:
    -------
    bool
        True if n is a JAX array (including traced), False otherwise.
    """
    if not backend_mgr.is_jax_available or not backend_mgr._jnp_module:
        return False
    # Check if it originates from jax.numpy or is a Tracer
    # isinstance check handles concrete arrays, hasattr handles tracers/abstract values
    return isinstance(n, backend_mgr._jnp_module.ndarray) or hasattr(n, 'aval')

def is_traced_jax(n):
    """
    Internal helper function to check if an array-like n is JAX-traced integer.
    """
    return is_jax_array(n)

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
    Decorator that applies JAX JIT compilation only when the backend is JAX.
    Raises ValueError if decorated function does not accept 'backend' keyword argument.
    If the function is called with backend set to 'np' (or the NumPy module), then
    the original (non-jitted) function is used.

    The decorator *enforces* that the decorated function has a keyword argument
    called 'backend' (which is also marked as static during JIT compilation).
    """
    
    
    if not JAX_AVAILABLE:
        return func

    from jax import jit
    
    # Check if the function has 'backend' as a keyword argument
    func_signature = inspect.signature(func)  
    if 'backend' not in func_signature.parameters:
        raise ValueError(f"Function '{func.__name__}' decorated with @maybe_jit must accept 'backend' as a keyword argument.")


    jitted_func = jit(func, static_argnames=("backend",))

    def wrapper(*args, **kwargs):
        backend = kwargs.get("backend", None)
        if backend is None:
            return jitted_func(*args, **kwargs)
        if isinstance(backend, str):
            if backend.lower() in ("np", "numpy"):
                return func(*args, **kwargs)
            else:
                return jitted_func(*args, **kwargs)
        if backend is np:
            return func(*args, **kwargs)
        else:
            return jitted_func(*args, **kwargs)

    return wrapper

# ---------------------------------------------------------------------
#! PADDING AND OTHER UTILITIES
# ---------------------------------------------------------------------

if JAX_AVAILABLE:
    import jax.numpy as jnp
    
    def pad_array_jax(x, target_size: int, pad_value):
        """
        Creates a padded version of x with a fixed target_size.
        The first x.shape[0] entries are x and the rest are filled with pad_value.
        """
        padded = jnp.full((target_size,), pad_value, dtype=x.dtype)
        padded = padded.at[:x.shape[0]].set(x)
        return padded
    
def pad_array_np(x, target_size: int, pad_value):
    """
    Creates a padded version of x with a fixed target_size.
    The first x.shape[0] entries are x and the rest are filled with pad_value.
    """
    padded = np.full((target_size,), pad_value, dtype=x.dtype)
    padded[:x.shape[0]] = x
    return padded

# ---------------------------------------------------------------------