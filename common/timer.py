from functools import wraps
from timeit import default_timer
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
        name (str): Optional name to identify the timer.
        verbose (bool): If True, prints timing information on stop.
    """
    def __init__(self, name: str = None, verbose: bool = False):
        self.name           = name
        self.verbose        = verbose
        self._start_time    = None
        self._elapsed       = 0.0
        self._laps          = []
    
    def restart(self):
        """Start the timer."""
        self._start_time = default_timer()
        return self
    
    def stop(self):
        """Stop the timer and optionally print elapsed time."""
        self._elapsed = default_timer() - self._start_time
        if self.verbose:
            print(f"{self.name}: {self._elapsed:.4f} seconds")
        return self._elapsed
    
    def lap(self):
        """Record a lap time."""
        self._laps.append(default_timer() - self._start_time)
        return self._laps[-1]
    
    @property
    def laps(self):
        """Return the recorded lap times."""
        return self._laps.copy()
    
    @property
    def elapsed(self):
        """
        Return the total elapsed time. If the timer is running,
        include the time from the current lap.
        """
        if self._start_time is not None:
            return self._elapsed + (default_timer() - self._start_time)
        return self._elapsed
    
    def reset(self):
        """Reset the timer and laps."""
        self._start_time    = None
        self._elapsed       = 0.0
        self._laps          = []
    
    def __enter__(self):
        """Start the timer when entering a context manager."""
        self.restart()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer when exiting a context manager."""
        self.stop()

    def __call__(self, func):
        """
        Allow the Timer to be used as a decorator.

        When decorating a function, it resets the timer, times the execution,
        and optionally prints the elapsed time.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.reset()
            self.restart()
            result  = func(*args, **kwargs)
            lap     = self.stop()
            if self.verbose:
                timer_name = self.name or func.__name__
                print(f"{timer_name} executed in {lap:.6f} seconds")
            return result
        return wrapper
    
################################################################################