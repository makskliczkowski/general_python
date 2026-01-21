class SolversTests:
    """
    This class implements tests for our algebra module.
    It exercises change-of-basis operations and can be extended to test solvers.
    """
    def __init__(self, backend = "default", logger_backend = None):
        valid_backends = ["default", "np", "jnp", "jax", "numpy"]
        if not isinstance(backend, str) or backend.lower() not in valid_backends:
            raise ValueError("Backend must be one of: " + ", ".join(valid_backends))
        self.loggerbackend  = logger_backend
        self.backend        = backend
        self.test_count     = 0
        self.logger         = self.loggerbackend(logfile="algebra_tests.log") if self.loggerbackend is not None else None
        if self.logger is not None:
            self.logger.configure(directory="./logs")
            self.logger.say("Starting algebra tests...", log=0, lvl=1)

    def _log(self, message, test_number, color = "white"):
        # Log the test message; you can add formatting here.
        if self.logger is not None:
            self.logger.say(f"[TEST {test_number}] {message}", log=0, lvl=1)
        else:
            print(f"[TEST {test_number}] {message}")
