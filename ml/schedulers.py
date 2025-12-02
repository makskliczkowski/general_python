'''

Scheduler implementations for machine learning training.
It includes various learning rate schedulers and an early stopping mechanism.

Namely, it provides:
- ConstantScheduler
- ExponentialDecayScheduler
- StepDecayScheduler
- CosineAnnealingScheduler
- AdaptiveScheduler (ReduceLROnPlateau)

For the usage, either create scheduler instances directly or use the
`choose_scheduler` factory function.

>>> # Example: Create an exponential decay scheduler
>>> from QES.general_python.ml.schedulers import choose_scheduler
>>> scheduler = choose_scheduler('exponential', initial_lr=0.01, max_epochs=100, lr_decay=0.1)
>>> for epoch in range(10):
>>>     lr = scheduler(epoch)
>>>     print(f"Epoch {epoch}: Learning Rate = {lr:.6f}")

---------------------------------------------------------------
file    : general_python/ml/schedulers.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
---------------------------------------------------------------
'''

import math
import enum
import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Type, Dict, Any
from ..common.flog import Logger
import numpy as np

# ##############################################################################
#! Constants
# ##############################################################################

_INF = float('inf')

# ##############################################################################
#! Early Stopping
# ##############################################################################

class BaseSchedulerLogger(ABC):
    """
    Abstract Base Class providing logging capabilities to schedulers.
    """
    
    def __init__(self, logger: Optional[Logger]):
        super().__init__()
        self._logger = logger
    
    def _log(self, message: str, log: Union[int, str] = 'info', lvl: int = 0, color: str = "white", **kwargs):
        """Internal logging helper."""
        
        # Format the message with class name
        full_msg = f"[{self.__class__.__name__}] {message}"
        
        if self._logger:
            if color:
                full_msg = self._logger.colorize(full_msg, color)
            
            # Map string log levels to integer if needed, or rely on Logger defaults
            if isinstance(log, str) and hasattr(Logger, 'LEVELS_R'):
                 log_val = Logger.LEVELS_R.get(log, 20) # Default to INFO
            else:
                 log_val = log

            self._logger.say(full_msg, lvl=lvl, log=log_val, **kwargs)
        else:
            # Fallback if no logger provided
            if log not in ['debug', 'info', 10, 20]:
                print(f"{str(log).upper()}: {full_msg}")

    @property
    def logger(self) -> Optional[Logger]:
        return self._logger
    
    @logger.setter
    def logger(self, logger: Logger):
        self._logger = logger

# ##############################################################################

class EarlyStopping(BaseSchedulerLogger):
    """
    Monitors a metric and determines if training should stop.
    """
    
    def __init__(self, patience: int = 0, min_delta: float = 1e-3, logger: Optional[Logger] = None):
        super().__init__(logger=logger)
        
        if patience is not None and patience < 0: raise ValueError("Patience must be non-negative.")
        if min_delta < 0.0: raise ValueError("min_delta must be non-negative.")

        self._patience          = patience
        self._min_delta         = min_delta
        self._best_metric       = _INF
        self._epoch_since_best  = 0
        self._stop_training     = False

    def __call__(self, _metric: Union[float, complex, np.number]) -> bool:
        """
        Args:
            _metric: The metric value (e.g. loss). Real part used if complex.
        """
        
        # Type check and conversion
        if not isinstance(_metric, (float, complex, np.number)):
            try:
                _metric = float(_metric)
            except (ValueError, TypeError):
                raise TypeError("Metric must be numeric.")

        # Extract real part
        val_r = _metric.real if isinstance(_metric, complex) else float(_metric)
        val_i = _metric.imag if isinstance(_metric, complex) else 0.0
        
        # Check NaN/Inf
        if np.isnan(val_r) or np.isnan(val_i) or np.isinf(val_r) or np.isinf(val_i):
            self._log("Received NaN or Inf metric. Stopping.", log='error', color='red')
            return True
        
        # Check Disabled
        if not self._patience:
            return False

        # Logic
        if val_r < (self._best_metric - self._min_delta):
            self._log(f"Metric improved to {val_r:.4e}.", log='debug', lvl=1)
            self._best_metric       = val_r
            self._epoch_since_best  = 0
            self._stop_training     = False
        else:
            self._epoch_since_best += 1
            self._log(f"No improvement for {self._epoch_since_best} epoch(s). Best: {self._best_metric:.4e}", log='debug', lvl=1)

        if self._epoch_since_best >= self._patience:
            self._log(f"Patience ({self._patience}) exceeded. Stopping.", log='info', color='yellow')
            self._stop_training     = True

        return self._stop_training

    def reset(self):
        self._best_metric           = _INF
        self._epoch_since_best      = 0
        self._stop_training         = False

    @property
    def best_metric(self) -> float: return self._best_metric

# ##############################################################################
#! Base Parameters & Schedulers
# ##############################################################################

class SchedulerType(enum.Enum):
    CONSTANT    = 0
    EXPONENTIAL = 1
    STEP        = 2
    COSINE      = 3
    ADAPTIVE    = 4
    LINEAR      = 5
    
    def __str__(self):
        return self.name.lower()
    
    def __repr__(self):
        return self.__str__()

class Parameters(BaseSchedulerLogger, ABC):
    
    def __init__(self, 
                 initial_lr     : float,
                 max_epochs     : int,
                 lr_decay       : float,
                 lr_clamp       : Optional[float]           = None,
                 logger         : Optional[Logger]          = None,
                 es             : Optional[EarlyStopping]   = None):
        
        super().__init__(logger=logger)

        if initial_lr <= 0: raise ValueError("Initial LR must be positive.")
        if max_epochs <= 0: raise ValueError("Max epochs must be positive.")

        self._initial_lr        = initial_lr
        self._max_epochs        = max_epochs
        self._lr_decay          = lr_decay
        self._lr_clamp          = lr_clamp
        self._early_stopping    = es
        
        self._lr                = initial_lr
        self._lr_history        = []
        self._typek             = None # To be set by subclass

    @abstractmethod
    def __call__(self, _epoch: int, _metric: Optional[Any] = None) -> float:
        """Calculate LR for the epoch."""
        pass

    def _update_and_log_lr(self, new_lr: float) -> float:
        """Helper to clamp, update state, and log."""
        if self._lr_clamp is not None:
            new_lr = np.maximum(new_lr, self._lr_clamp)
        
        self._lr = new_lr
        self._lr_history.append(self._lr)
        return self._lr

    #! Common Properties
    @property
    def lr(self)                    -> float: return self._lr
    
    @property       
    def history(self)               -> List[float]: return self._lr_history
    
    @property
    def early_stopping(self)        -> Optional[EarlyStopping]: return self._early_stopping

    #! ES Proxies
    def set_early_stopping(self, patience: int, min_delta: float = 1e-3):
        self._early_stopping = EarlyStopping(patience, min_delta, self._logger)

    def check_stop(self, _metric) -> bool:
        if self._early_stopping: return self._early_stopping(_metric)
        return False

# ##############################################################################
#! CONCRETE SCHEDULERS
# ##############################################################################

class ConstantScheduler(Parameters):
    def __init__(self, initial_lr: float, max_epochs: int, lr_clamp=None, logger=None, es=None, **kwargs):
        super().__init__(initial_lr, max_epochs, lr_decay=1.0, lr_clamp=lr_clamp, logger=logger, es=es)
        self._typek = SchedulerType.CONSTANT

    def __call__(self, _epoch: int, _metric=None) -> float:
        return self._update_and_log_lr(self._initial_lr)

# ------------------------------------------------------------------------------

class ExponentialDecayScheduler(Parameters):
    """ 
    Multiplicative exponential decay: lr = initial_lr * (gamma ^ epoch)
    
    This follows PyTorch's ExponentialLR convention where:
    - gamma = 0.99 means 1% decay per epoch
    - gamma = 0.999 means 0.1% decay per epoch
    
    For the old exp(-rate * epoch) behavior, use gamma = exp(-rate).
    """
    def __init__(self, initial_lr: float, max_epochs: int, lr_decay: float = 0.99, lr_clamp=None, logger=None, es=None, **kwargs):
        super().__init__(initial_lr, max_epochs, lr_decay, lr_clamp, logger, es)
        self._typek = SchedulerType.EXPONENTIAL

    def __call__(self, _epoch: int, _metric=None) -> float:
        current_epoch   = max(0, _epoch)
        # Multiplicative decay: lr = lr_0 * gamma^epoch (like PyTorch ExponentialLR)
        new_lr          = self._initial_lr * np.power(self._lr_decay, current_epoch)
        return self._update_and_log_lr(new_lr)

# ------------------------------------------------------------------------------

class LinearScheduler(Parameters):
    """ 
    Linearly decays LR from initial_lr to min_lr (default 0) over max_epochs.
    """
    def __init__(self, initial_lr: float, max_epochs: int, min_lr: float = 0.0, lr_clamp=None, logger=None, es=None, **kwargs):
        # We store min_lr but pass 1.0 as dummy decay
        super().__init__(initial_lr, max_epochs, lr_decay=1.0, lr_clamp=lr_clamp, logger=logger, es=es)
        self._min_lr    = min_lr
        self._typek     = SchedulerType.LINEAR

    def __call__(self, _epoch: int, _metric=None) -> float:
        current_epoch   = max(0, min(_epoch, self._max_epochs))
        alpha           = current_epoch / self._max_epochs
        # Interpolate: (1-alpha)*start + alpha*end
        new_lr          = (1 - alpha) * self._initial_lr + alpha * self._min_lr
        return self._update_and_log_lr(new_lr)

# ------------------------------------------------------------------------------

class StepDecayScheduler(Parameters):
    """ lr = initial_lr * decay_factor ^ floor(epoch / step_size) """
    def __init__(self, initial_lr: float, max_epochs: int, lr_decay: float, step_size: int, lr_clamp=None, logger=None, es=None, **kwargs):
        super().__init__(initial_lr, max_epochs, lr_decay, lr_clamp, logger, es)
        if step_size <= 0: raise ValueError("Step size must be positive.")
        self.step_size  = step_size
        self._typek     = SchedulerType.STEP

    def __call__(self, _epoch: int, _metric=None) -> float:
        exponent        = np.floor(max(0, _epoch) / self.step_size)
        new_lr          = self._initial_lr * np.power(self._lr_decay, exponent)
        return self._update_and_log_lr(new_lr)

# ------------------------------------------------------------------------------

class CosineAnnealingScheduler(Parameters):
    def __init__(self, initial_lr: float, max_epochs: int, min_lr: float = 0.0, lr_clamp=None, logger=None, es=None, **kwargs):
        super().__init__(initial_lr, max_epochs, lr_decay=0.0, lr_clamp=lr_clamp, logger=logger, es=es)
        self.min_lr     = min_lr
        self._typek     = SchedulerType.COSINE

    def __call__(self, _epoch: int, _metric=None) -> float:
        cur_ep          = np.clip(_epoch, 0, self._max_epochs)
        if self._max_epochs <= 0: cosine_term = -1.0
        else: cosine_term = np.cos(np.pi * cur_ep / self._max_epochs)
        
        lr_range        = self._initial_lr - self.min_lr
        new_lr          = self.min_lr + 0.5 * lr_range * (1.0 + cosine_term)
        return self._update_and_log_lr(new_lr)

# ------------------------------------------------------------------------------

class AdaptiveScheduler(Parameters):
    """ ReduceLROnPlateau logic """
    def __init__(self, initial_lr: float, max_epochs: int, lr_decay: float, patience: int, 
                 min_lr: float = 1e-5, cooldown: int = 0, min_delta: float = 1e-4, 
                 lr_clamp=None, logger=None, es=None, **kwargs):
        super().__init__(initial_lr, max_epochs, lr_decay, lr_clamp, logger, es)
        
        self.patience           = patience
        self.min_lr             = min_lr
        self.cooldown           = cooldown
        self.min_delta          = min_delta
        
        self._cooldown_counter  = 0
        self._best_metric       = _INF
        self._num_bad_epochs    = 0
        self._typek             = SchedulerType.ADAPTIVE

    def __call__(self, _epoch: int, _metric: Optional[Any]) -> float:
        if _metric is None: raise ValueError("AdaptiveScheduler requires a metric.")
        
        # Safe metric extraction
        if isinstance(_metric, complex): 
            metric_val = _metric.real
        else: 
            metric_val = float(_metric)

        if np.isnan(metric_val) or np.isinf(metric_val):
            return self._update_and_log_lr(self._lr)

        # Cooldown check
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self._num_bad_epochs    = 0
            return self._update_and_log_lr(self._lr)

        # Improvement Check
        if metric_val < self._best_metric - self.min_delta:
            self._best_metric       = metric_val
            self._num_bad_epochs    = 0
        else:
            self._num_bad_epochs   += 1

        # Reduction Logic
        if self._num_bad_epochs > self.patience:
            new_lr = max(self._lr * self._lr_decay, self.min_lr)
            
            if new_lr < self._lr:
                self._log(f"Reducing LR to {new_lr:.2e} (Plateau).", log='info', color='yellow')
                self._lr                = new_lr
                self._cooldown_counter  = self.cooldown
            self._num_bad_epochs    = 0

        return self._update_and_log_lr(self._lr)

    def reset(self):
        self._cooldown_counter  = 0
        self._best_metric       = _INF
        self._num_bad_epochs    = 0

# ##############################################################################
#! FACTORY
# ##############################################################################

SCHEDULER_CLASS_MAP = {
    SchedulerType.CONSTANT:     {'class': ConstantScheduler,            'args': []},
    SchedulerType.EXPONENTIAL:  {'class': ExponentialDecayScheduler,    'args': ['lr_decay']},
    SchedulerType.STEP:         {'class': StepDecayScheduler,           'args': ['lr_decay', 'step_size']},
    SchedulerType.COSINE:       {'class': CosineAnnealingScheduler,     'args': ['min_lr']},
    SchedulerType.ADAPTIVE:     {'class': AdaptiveScheduler,            'args': ['lr_decay', 'patience', 'min_lr', 'cooldown', 'min_delta']},
    SchedulerType.LINEAR:       {'class': LinearScheduler,              'args': ['min_lr']},
}

# Add string aliases
SCHEDULER_CLASS_MAP["constant"]     = SCHEDULER_CLASS_MAP[SchedulerType.CONSTANT]
SCHEDULER_CLASS_MAP["exponential"]  = SCHEDULER_CLASS_MAP[SchedulerType.EXPONENTIAL]
SCHEDULER_CLASS_MAP["step"]         = SCHEDULER_CLASS_MAP[SchedulerType.STEP]
SCHEDULER_CLASS_MAP["cosine"]       = SCHEDULER_CLASS_MAP[SchedulerType.COSINE]
SCHEDULER_CLASS_MAP["adaptive"]     = SCHEDULER_CLASS_MAP[SchedulerType.ADAPTIVE]
SCHEDULER_CLASS_MAP["linear"]       = SCHEDULER_CLASS_MAP[SchedulerType.LINEAR]

def choose_scheduler(scheduler_type : Union[str, SchedulerType, Parameters],
                     initial_lr     : float,
                     max_epochs     : int,
                     logger         : Optional[Logger] = None,
                     **kwargs) -> Parameters:
    """
    Factory to create a scheduler instance.
    """
    
    temp_log = logger if logger else Logger()
    
    # 1. Handle Existing Instance
    if isinstance(scheduler_type, Parameters):
        if logger:                  scheduler_type.logger = logger
        if kwargs.get('lr_clamp'):  scheduler_type._lr_clamp = kwargs['lr_clamp']
        
        # Reconfigure ES if args present
        if 'early_stopping_patience' in kwargs:
             scheduler_type.set_early_stopping(kwargs['early_stopping_patience'], kwargs.get('early_stopping_min_delta', 1e-4))
        return scheduler_type

    # 2. Resolve Type
    key     = scheduler_type.lower() if isinstance(scheduler_type, str) else scheduler_type
    config  = SCHEDULER_CLASS_MAP.get(key)
    
    if not config:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    cls     = config['class']
    
    # 3. Setup Early Stopping
    es = None
    if 'early_stopping_patience' in kwargs:
        es = EarlyStopping(kwargs['early_stopping_patience'], kwargs.get('early_stopping_min_delta', 1e-4), logger)

    # 4. Build Args
    # Filter kwargs to only what the specific scheduler needs + base args
    valid_args = {'initial_lr', 'max_epochs', 'lr_clamp', 'logger', 'es'}
    valid_args.update(config['args'])
    
    build_kwargs = {
        'initial_lr'    : initial_lr, 
        'max_epochs'    : max_epochs,
        'logger'        : logger,
        'es'            : es,
        'lr_clamp'      : kwargs.get('lr_clamp')
    }
    
    # Add specific args from kwargs if they exist
    for arg in config['args']:
        if arg in kwargs:
            build_kwargs[arg] = kwargs[arg]
        elif arg == 'lr_decay':
            build_kwargs['lr_decay'] = 0.99 # Default
            
    try:
        return cls(**build_kwargs)
    except TypeError as e:
        temp_log.say(f"Failed to instantiate {cls.__name__}: {e}", log='error', color='red')
        raise
    
###############################################################################
#! End of file
###############################################################################