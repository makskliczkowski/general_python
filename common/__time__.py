from contextlib import contextmanager
from timeit import default_timer

################################################################################

@contextmanager
def elapsed_timer():
    start = default_timer()
    try:
        yield lambda: default_timer() - start
    finally:
        end = default_timer()
        elapser = lambda: end - start

################################################################################

class Timer:
    """
    A simple timer class to measure elapsed time.
    Methods:
    --------
    __init__():
        Initializes the timer and starts it.
    elapsed() -> float:
        Returns the elapsed time since the timer was started.
    reset():
        Resets the timer to the current time.
    __enter__() -> Timer:
        Starts the timer and returns the Timer instance (for use with 'with' statement).
    __exit__(exc_type, exc_val, exc_tb):
        Handles the exit of the 'with' statement (does nothing in this implementation).
    """
    def __init__(self):
        self.start = default_timer()
    
    def elapsed(self):
        return default_timer() - self.start
    
    def reset(self):
        self.start = default_timer()
    
    def __enter__(self):
        self.start = default_timer()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # The __exit__ method is intentionally left empty because 
        # there is no specific cleanup action required when exiting the 'with' statement.
        pass