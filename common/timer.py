from functools import wraps
from timeit import default_timer
import time
from abc import ABC

################################################################################

class Timer(ABC):
    """
    Enhanced timer class for measuring elapsed time.

    This class can be used as a context manager, a decorator, or directly to time code.
    It supports:
        - Starting, stopping, and resetting the timer.
        - Recording multiple laps.
        - Verbose output to automatically print timing information.
    
    Attributes:
        name (str):
            Optional name to identify the timer.
        verbose (bool):
            If True, prints timing information on stop.
        format:
            Optional format for the output timing information.
    """
    def __init__(self, name: str = None, verbose: bool = False, precision: str = 's'):
        """
        Args:
            name (str):
                Optional name to identify the timer.
            verbose (bool):
                If True, print elapsed time on stop.
            precision (str):
                Time unit for display: 's', 'ms', 'us', or 'ns'.
        """
        self.name           = name
        self.verbose        = verbose
        self.precision      = precision
        self._start_time    = None
        self._elapsed_ns    = 0
        self._laps_ns       = []

        self._unit_factors = {
            's': 1e-9,
            'ms': 1e-6,
            'us': 1e-3,
            'ns': 1.0
        }
        if precision not in self._unit_factors:
            raise ValueError("Precision must be one of: 's', 'ms', 'us', 'ns'")

    def restart(self):
        """Start or restart the timer."""
        self._start_time    = time.perf_counter_ns()
        return self

    def stop(self):
        """Stop the timer and return elapsed time in the selected unit."""
        now = time.perf_counter_ns()
        self._elapsed_ns    = now - self._start_time
        if self.verbose:
            print(f"{self.name or 'Timer'}: {self._format_time(self._elapsed_ns)}")
        return self._to_unit(self._elapsed_ns)

    def lap(self):
        """Record a lap time and return it in the selected unit."""
        now             = time.perf_counter_ns()
        lap_time        = now - self._start_time
        self._laps_ns.append(lap_time)
        return self._to_unit(lap_time)

    @property
    def laps(self):
        """Return a list of recorded lap times in the selected unit."""
        return [self._to_unit(lap) for lap in self._laps_ns]

    @property
    def elapsed(self):
        """Return the total elapsed time (in selected unit), including current run if active."""
        if self._start_time is not None:
            now         = time.perf_counter_ns()
            return self._to_unit(self._elapsed_ns + (now - self._start_time))
        return self._to_unit(self._elapsed_ns)

    def reset(self):
        """Reset the timer and all laps."""
        self._start_time    = None
        self._elapsed_ns    = 0
        self._laps_ns       = []

    def __enter__(self):
        self.restart()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.reset()
            self.restart()
            result      = func(*args, **kwargs)
            elapsed     = self.stop()
            if self.verbose:
                print(f"{self.name or func.__name__} executed in {self._format_time(self._elapsed_ns)}")
            return result
        return wrapper

    def _to_unit(self, ns):
        """Convert nanoseconds to the selected unit."""
        return ns * self._unit_factors[self.precision]

    def _format_time(self, ns):
        """Return a formatted string based on the precision."""
        unit    = self.precision
        value   = self._to_unit(ns)
        return f"{value:.6f} {unit}"

################################################################################

def timeit(fn, *args, **kwargs):
    """
    Measures the execution time of a function, optionally handling JAX DeviceArrays for accurate timing.
    Args:
        fn (callable):
            The function to be timed.
        *args:
            Positional arguments to pass to the function.
        **kwargs:
            Keyword arguments to pass to the function.
    Returns:
        tuple: A tuple containing:
            - res: The result returned by the function `fn`.
            - float: The elapsed time in seconds.
    Notes:
        If the result is a JAX DeviceArray or a tuple containing DeviceArrays, 
        the function waits for computation to finish using `block_until_ready()` 
        to ensure accurate timing.
    """
    
    t0      = time.time()
    res     = fn(*args, **kwargs)
    # If JAX DeviceArray or tuple thereof, block until ready for accurate timing
    if hasattr(res, "block_until_ready"):
        res.block_until_ready()
    elif isinstance(res, tuple):
        for x in res:
            if hasattr(x, "block_until_ready"):
                x.block_until_ready()
    return res, time.time() - t0

################################################################################