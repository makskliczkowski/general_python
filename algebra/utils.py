'''
file:
This module provides linear algebra functions and utilities.
'''

# Import the required modules
import numpy as np
import numpy.random as nrn
import scipy as sp
import inspect

# ---------------------------------------------------------------------

def _log_message(msg, lvl = 0):
    """
    Logs a message using the global logger.
    This function ensures the logger is only imported when needed.
    """
    while lvl > 0:
        print("\t", end="")
        lvl -= 1
    print(msg)

# ---------------------------------------------------------------------

_JAX_AVAILABLE              = False
_KEY                        = None
DEFAULT_NP_INT_TYPE         = np.int64
DEFAULT_NP_FLOAT_TYPE       = np.float64
DEFAULT_NP_CPX_TYPE         = np.complex128
DEFAULT_JP_INT_TYPE         = None
DEFAULT_JP_FLOAT_TYPE       = None
DEFAULT_JP_CPX_TYPE         = None
DEFAULT_INT_TYPE            = np.int64
DEFAULT_FLOAT_TYPE          = np.float64
DEFAULT_CPX_TYPE            = np.complex128
DEFAULT_BACKEND             = np
DEFAULT_BACKEND_NAME        = "numpy"
DEFAULT_BACKEND_RANDOM      = nrn
DEFAULT_BACKEND_SCIPY       = sp
DEFAULT_BACKEND_KEY         = None
_DEFAULT_SEED               = 12345
JIT                         = lambda x: x

# ---------------------------------------------------------------------

class BackendManager:
    """
    BackendManager is a utility class for managing numerical backends used in the application.
    It provides methods for initializing and selecting between NumPy and JAX (if available) backends,
    including support for randomness and SciPy functionalities.
    Attributes:
        _jax_available (bool): Indicates whether JAX is available.
        np: Reference to the numerical backend module (NumPy or jax.numpy).
        random: Random module associated with the backend (NumPy's default_rng or jax.random).
        scipy: Reference to the backend's SciPy module (SciPy or jax.scipy).
        name (str): The name of the currently active backend ("numpy" or "jax").
        key: Random seed or key used by the backend if applicable (e.g., JAX PRNGKey).
    Methods:
        __init__():
            Attempts to initialize the JAX backend by importing jax.numpy, jax.scipy, and jax.random.
            If successful, updates the backend configuration to use JAX and sets a default PRNG key.
            If JAX is not available, falls back to NumPy as the backend.
        print_info():
            Logs information about the initialized backend, including versions of NumPy, SciPy,
            and JAX (if available), as well as the current backend name and other state details.
        get_backend_modules(backend_spec, random=False, seed=None, scipy=False):
            Returns the backend modules based on the provided specification, which may be a module
            (NumPy/jax.numpy) or a string ("default", "numpy", or "jax"). The method considers additional
            flags for including random functionality and SciPy modules, and it handles seeding appropriately.
        _numpy_backend_modules(random=False, seed=None, scipy=False):
            Internal helper method that returns a tuple containing the NumPy module. Optionally includes:
            - a random module (either np.random or np.random.default_rng) depending on the NumPy version and seed.
            - the SciPy module if requested.
        _jax_backend_modules(random=False, seed=None, scipy=False):
            Internal helper method that returns a tuple containing the JAX numerical module.
            If random functionality is requested, also returns the JAX random module along with an appropriate
            PRNGKey generated from the provided seed (or the default key if no seed is provided).
            Optionally includes the JAX SciPy module if requested.
        get_global_backend_modules(random=False, seed=None, scipy=False):
            Retrieves the global backend modules based on the current default backend in use by the manager,
            delegating to get_backend_modules with the manager's current configuration.
    Usage:
        # Instantiate the BackendManager.
        backend_manager = BackendManager()
        # Print backend information.
        backend_manager.print_info()
        # Retrieve backend modules with randomness and SciPy capabilities.
        modules = backend_manager.get_backend_modules("jax", random=True, seed=42, scipy=True)
    """
    
    def __init__(self, seed = _DEFAULT_SEED):
        '''
        Initializes the backend manager by attempting to import JAX modules.
        If JAX is available, sets the backend to JAX and configures the default PRNG key.
        If JAX is not available, falls back to NumPy as the backend.
        '''
        self._jax_available     = False
        self.np                 = np
        self.seed               = seed
        # numpy
        self.random_np          = nrn.default_rng(self.seed) if np.__version__ >= "1.17" else nrn
        if not np.__version__ >= "1.17":
            _log_message(f"NumPy version is {np.__version__}. Using legacy random module.", 1)
        np.random.seed(self.seed)
        self.scipy_np           = sp
        self.random             = self.random_np
        self.scipy              = self.scipy_np
        self.name               = "numpy"
        self.key                = None
        self.jit                = lambda x: x # Dummy JIT function

        try:
            import jax
            import jax.numpy as jnp
            import jax.scipy as jsp
            import jax.random as jrn
            from jax import jit, config as jcfg
            self._jax_available = True
            self.jax            = jnp
            self.np             = self.jax
            self.random         = jrn
            self.scipy          = jsp
            self.name           = "jax"
            self.key            = jrn.PRNGKey(self.seed)
            self.jit            = jit
            
            DEFAULT_JP_INT_TYPE     = jnp.int32
            DEFAULT_JP_FLOAT_TYPE   = jnp.float64
            DEFAULT_JP_CPX_TYPE     = jnp.complex128
            
            # Set JAX global configuration by enabling 64-bit precision
            jcfg.update("jax_enable_x64", True)
            
        except ImportError as e:
            # _log_message(f"JAX not available: {e}. Using NumPy as backend.", lvl=1)
            pass
    # ---------------------------------------------------------------------
    
    @property
    def jax_available(self):
        """Returns the availability of JAX as a backend."""
        return self._jax_available

    # ---------------------------------------------------------------------
    
    def print_info(self):
        """
        Prints detailed backend initialization and configuration information.
        """
        # Collect version information for each library.
        backend_versions = {
            "NumPy": np.__version__,
            "SciPy": sp.__version__,
            "JAX": "Not Available"
        }
        
        # If JAX is available, get its actual version
        if self._jax_available:
            try:
                if hasattr(self.np, "__version__"):
                    backend_versions["JAX"] = self.np.__version__
                elif hasattr(self.np, "version"):
                    backend_versions["JAX"] = self.np.version
                elif hasattr(self.np, 'array_api'):
                    backend_versions["JAX"] = self.np.array_api.__version__
            except (ImportError, AttributeError):
                pass
        
        # Print header.
        _log_message("=== Backend Initialization ===", 0)
        
        # Log version info.
        for lib, version in backend_versions.items():
            _log_message(f"{lib} Version: {version}", 1)
        
        # Log active backend details.
        _log_message(f"\tActive Backend: {self.name}", 1)
        _log_message(f"\tJAX Available: {self._jax_available}", 1)
        _log_message(f"\tDefault Random Key: {self.key}", 1)
        
        # Log current backend modules.
        _log_message("Active Backend Modules:", 1)
        _log_message(f"\tMain Module: {self.np}", 2)
        _log_message(f"\tRandom Module: {self.random}", 2)
        _log_message(f"\tSciPy Module: {self.scipy}", 2)
        
        # Footer.
        _log_message("=== End of Backend Info ===", 0)

    # ---------------------------------------------------------------------
    
    def get_backend_modules(self, backend_spec, random=False, seed=None, scipy=False):
        """
        Returns backend modules based on specifier, now using BackendManager's state.
        Parameters:
        
            backend_spec    : The specification for the backend, can be a module or string.
            random          : A boolean indicating if random functionality should be included.
            seed            : The random seed to use with the backend.
            scipy           : A boolean indicating if the SciPy module should be included.
        Returns:
            module or tuple : The backend module(s) based on the specifier. If random or scipy is requested,
                            it returns the corresponding modules along with their settings. 
        """
        if isinstance(backend_spec, str):
            b_str = backend_spec.lower()
        else:
            b_str = "default"

        seed = _DEFAULT_SEED if seed is None else seed
        
        if backend_spec == np: # Direct module comparison
            if DEFAULT_BACKEND == np:
                return self.get_global_backend_modules(random=random, seed=seed, scipy=scipy)
            return self._numpy_backend_modules(random, seed, scipy) # Use NumPy specifically
        elif backend_spec == self.np:
            if DEFAULT_BACKEND == self.np: # Use global default if it's the manager's current numpy (jax.numpy or numpy)
                return self.get_global_backend_modules(random=random, seed=seed, scipy=scipy)
            if not self._jax_available and backend_spec == self.np and self.name == "numpy": # Handle numpy specifically if jax is not available
                return self._numpy_backend_modules(random, seed, scipy)
            elif self._jax_available and backend_spec == self.np and self.name == "jax": # Handle jax.numpy specifically if jax is available and manager is jax
                return self._jax_backend_modules(random, seed, scipy)
            else:
                raise ValueError(f"Backend module mismatch or unsupported scenario.")


        if b_str == "default":
            b_str = self.name # Use the manager's current default backend name

        if b_str in ("np", "numpy"):
            return self._numpy_backend_modules(random, seed, scipy)
        elif b_str in ("jnp", "jax"):
            if not self._jax_available:
                raise ValueError("JAX backend requested but JAX is not available.")
            return self._jax_backend_modules(random, seed, scipy)
        elif b_str == 'int':
            return None
        else:
            raise ValueError(f"Unsupported backend string: {backend_spec}")

    # ---------------------------------------------------------------------

    def _numpy_backend_modules(self, random=False, seed=_DEFAULT_SEED, scipy=False):
        """Returns NumPy backend modules."""
        
        # reset seed if needed
        if seed != self.seed:
            self.seed       = seed
            self.random_np  = nrn.default_rng(self.seed) if np.__version__ >= "1.17" else nrn
        
        main_module     = np
        rnd_module      = self.random_np
        scipy_module    = self.scipy_np
        ret             = [main_module]
        if random:
            ret.append((rnd_module, None))
        if scipy:
            ret.append(scipy_module)
        return tuple(ret) if len(ret) > 1 else main_module

    # ---------------------------------------------------------------------

    def _jax_backend_modules(self, random=False, seed=None, scipy=False):
        """
        Returns JAX backend modules, handling PRNG key.
        
        For JAX backend, ensure you split and manage the returned PRNG key for each random operation to maintain reproducibility.
        """
        from jax import random as jrn
        
        if seed != self.seed:
            self.seed       = seed
            if self._jax_available:
                self.key    = jrn.PRNGKey(self.seed)
        
        if not self._jax_available:
            raise ValueError("JAX backend requested but JAX is not available.")
        main_module     = self.np
        rnd_module      = self.random if random else None
        scipy_module    = self.scipy if scipy else None
        key             = self.key if random else None
        
        ret = [main_module]
        if random:
            ret.append((rnd_module, key)) # Return random module AND key
        if scipy:
            ret.append(scipy_module)
        return tuple(ret) if len(ret) > 1 else main_module

    # ---------------------------------------------------------------------

    def get_global_backend_modules(self, random=False, seed=None, scipy=False):
        """
        Returns global default backend modules from the manager.
        Parameters:
            random          : A boolean indicating if random functionality should be included.
            seed            : The random seed to use with the backend.
            scipy           : A boolean indicating if the SciPy module should be included.
        Returns:
            module or tuple : The global default backend module(s) based on the manager's current configuration.
                    
        """
        return self.get_backend_modules(self.name, random=random, seed=seed, scipy=scipy) 

    # ---------------------------------------------------------------------

# ---------------------------------------------------------------------

# Instantiate the BackendManager as a global object
backend_mgr = BackendManager()

# Initialize global defaults using BackendManager - for backward compatibility mainly
try:
    # Set the global defaults based on the backend manager's current state
    DEFAULT_BACKEND         = backend_mgr.np
    DEFAULT_BACKEND_NAME    = backend_mgr.name
    DEFAULT_BACKEND_RANDOM  = backend_mgr.random
    DEFAULT_BACKEND_SCIPY   = backend_mgr.scipy
    DEFAULT_BACKEND_KEY     = backend_mgr.key
    DEFAULT_INT_TYPE        = backend_mgr.np.int64
    DEFAULT_FLOAT_TYPE      = backend_mgr.np.float64
    DEFAULT_CPX_TYPE        = backend_mgr.np.complex128
    JIT                     = backend_mgr.jit
    _JAX_AVAILABLE          = backend_mgr.jax_available
    
except Exception as e:
    _log_message(f"Error initializing global defaults: {e}")
    
# Short aliases for convenience - these now point to the BackendManager's current backend
d_bcknd     = DEFAULT_BACKEND
d_bcknd_rnd = DEFAULT_BACKEND_RANDOM
d_bcknd_sp  = DEFAULT_BACKEND_SCIPY
d_bcknd_key = DEFAULT_BACKEND_KEY

# Print backend info *after* BackendManager is initialized, to reflect the *actual* backend in use.
try:
    backend_mgr.print_info()
except Exception as e:
    _log_message(f"Error printing backend info: {e}")

# ---------------------------------------------------------------------

def is_traced_jax(n):
    """
    Internal helper function to check if an array-like n is JAX-traced integer.
    """
    if _JAX_AVAILABLE:
        return hasattr(n, 'shape') and not hasattr(n, 'len')
    return False

# ---------------------------------------------------------------------

def get_backend(backend, random=False, seed=None, scipy=False) -> tuple:
    """
    Return backend modules based on the provided specifier.
    Delegates to BackendManager's get_backend_modules for actual logic.

    Parameters
    ----------
    backend : str or module or None
        Backend specifier. If a string:
            - "default"         : Uses DEFAULT_BACKEND (JAX if available, else NumPy).
            - "np" or "numpy"   : Returns the NumPy backend.
            - "jnp" or "jax"    : Returns the JAX backend (if available).
        If None or not a string, "default" is assumed.
    random : bool, optional
        If True, include the random module. For JAX, also return a PRNG key (if seed is provided).
        **For JAX backend, ensure you split and manage the returned PRNG key for each random operation to maintain reproducibility.**
    seed : int, optional
        If provided, sets the seed for the random module.
    scipy : bool, optional
        If True, also return the associated SciPy module.

    Returns
    -------
    module or tuple
        If neither random nor scipy is requested, returns the main backend module.
        Otherwise, returns a tuple containing:
            (main_module, random_module, scipy_module)
        For the JAX backend with random=True, the "random_module" entry is itself a tuple: (jax.random, key),
        where key is the initial PRNGKey (or a seeded one if seed is provided). **Remember to split this key before use in JAX.**

    **Example for using JAX backend with random number generation:**

    >>> import algebra_utils as alg_utils
    >>> jax_np, (jax_rnd, key), jax_sp = alg_utils.get_backend("jax", random=True, scipy=True)
    >>> key, subkey = jax_rnd.split(key) # Split the key!
    >>> random_vector = jax_rnd.uniform(subkey, shape=(5,)) # Use subkey for random op
    >>> print(random_vector)
    """
    return backend_mgr.get_backend_modules(backend, random=random, seed=seed, scipy=scipy)

# ---------------------------------------------------------------------

def get_global_backend(random=False, seed=None, scipy=False):
    """
    Return the global default backend modules.
    Delegates to BackendManager's get_global_backend_modules.

    Parameters
    ----------
    random : bool, optional
        If True, include the random module. For JAX, also return a PRNG key (if seed is provided).
    seed : int, optional
        If provided, sets the seed for the random module.
    scipy : bool, optional
        If True, also return the associated SciPy module.

    Returns
    -------
    module or tuple
        If neither random nor scipy is requested, returns the main backend module.
        Otherwise, returns a tuple containing:
            (main_module, random_module, scipy_module)
        For the JAX backend, the "random_module" entry is itself a tuple: (jax.random, key) else (numpy.random, None).
        where key is a PRNGKey (or None if seed is not provided).
    """
    return backend_mgr.get_global_backend_modules(random=random, seed=seed, scipy=scipy)

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
    
    
    if not _JAX_AVAILABLE:
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

import multiprocessing

if _JAX_AVAILABLE:
    def _get_gpu_info():
        from jax.lib import xla_bridge
        return xla_bridge.device_count()

def get_hardware_info():
    """
    Get the number of available devices and CPU threads.
    Returns:
        n_devices : int
            The number of available devices (GPUs) if JAX is available, otherwise 0.
        n_threads : int
            The number of CPU threads available.
    """
    n_devices       = 0
    if _JAX_AVAILABLE:
        n_devices   = _get_gpu_info()
    n_threads       = multiprocessing.cpu_count()
    return n_devices, n_threads

# ---------------------------------------------------------------------

#! PADDING AND OTHER UTILITIES

# ---------------------------------------------------------------------

if _JAX_AVAILABLE:
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