from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    

class Timer:
    def __init__(self):
        self.start  =   default_timer()
    def elapsed(self):
        return default_timer() - self.start
    def reset(self):
        self.start = default_timer()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass