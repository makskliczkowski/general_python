'''
file    : general_python/ml/schedulers.py
author  : Maksymilian Kliczkowski
'''

import math
import enum
import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Type, Dict, Any, Callable
from general_python.common.flog import Logger
import numpy as np

# ##############################################################################
#! Constants
# ##############################################################################

_INF = float('inf')

# ##############################################################################
#! Early Stopping
# ##############################################################################

class InitialSchedulerClass(ABC):
    """
    Abstract Base Class for Schedulers.
    This is a placeholder for any common functionality or attributes
    that might be needed in the future.
    """
    
    def __init__(self, logger: Optional[Logger]):
        """
        Args:
            logger (Optional[Logger]):
                Optional logger instance for logging messages.
        """
                
        super().__init__()
        self._logger    = logger
        self._log_level = 'info'
    
    # -----------------------------------------------------------------
    
    def _log(self, message: str, log : Union[int, str] = Logger.LEVELS_R['info'], lvl : int = 0, color : str = "white", append_msg = True, **kwargs):
        """
        Internal logging helper.
        
        Args:
            message (str):
                The message to log.
            log (str):
                The logging level (e.g., 'info', 'debug', 'warning').
            lvl (int):
                The logging tabulating level.
            color (str):
                The color for the message (if applicable).
            append_msg (bool):
                If True, appends the message to the logger.
            **kwargs:
                Additional keyword arguments for the logger.            
        """
        if append_msg:
            message = f"[{self.__class__.__name__}] {message}"
        
        if color:
            message = self._logger.colorize(message, color)

        if self._logger:
            
            #! fallback
            if isinstance(lvl, str):
                lvl = 0
            
            if isinstance(log, str):
                log = Logger.LEVELS_R[log]

            self._logger.say(message, lvl=lvl, log=log, **kwargs)
        else:
            # Basic print fallback
            if log not in ['debug', 'info']:
                print(f"{log.upper()}: {message}")
            else:
                print(f"{message}")

    def set_logger(self, logger: Logger):
        """
        Set a new logger instance.
        
        Args:
            logger (Logger):
                The new logger instance to use.
        """
        self._logger = logger
    
    def get_logger(self) -> Optional[Logger]:
        """
        Get the current logger instance.
        
        Returns:
            Optional[Logger]:
                The current logger instance, or None if not set.
        """
        return self._logger
    
    @property
    def logger(self) -> Optional[Logger]:
        """Returns the logger instance."""
        return self._logger
    
    @logger.setter
    def logger(self, logger: Logger):
        """Sets the logger instance."""
        self._logger = logger
        
# ##############################################################################

class EarlyStopping(InitialSchedulerClass):
    """
    EarlyStopping monitors a specified metric during training and determines whether training should be terminated early based on its performance.

    Attributes:
        patience (int):
            The maximum number of epochs to tolerate no significant improvement.
        min_delta (float):
            The threshold for considering a change in metric as an improvement.
        best_metric (float):
            The best metric value observed so far.
        epoch_since_best (int):
            The count of consecutive epochs without significant improvement.
        stop_training (bool): 
            Flag indicating whether training should be terminated.
        
    Methods:
        __call__(metric: float) -> bool:
            Updates the internal state based on the current metric value. If the current metric shows a significant improvement 
            (i.e., lower than the best metric by at least min_delta), it resets the counter tracking epochs with no improvement.
            Otherwise, it increments the counter. Training is signaled to stop if the counter exceeds the defined patience or if the 
            metric is NaN or infinite.
        reset() -> None:
            Resets the early stopping mechanism to its initial state by restoring best_metric to infinity, resetting the epoch counter,
            and clearing the stop_training flag.
        __repr__() -> str:
            Returns an informative string representation of the EarlyStopping instance, including its configuration (patience and min_delta)
            and current tracking values (best_metric, epoch_since_best, and stop_training).
    """
    
    _ERR_PATIENCE           = "Patience should be non-negative."
    _ERR_MIN_DELTA          = "min_delta should be non-negative."
    _ERR_METRIC             = "Metric should be a float."
    _ERR_METRIC_NAN         = "Metric is NaN or Inf."
    _ERR_METRIC_INF         = "Metric is Inf."
    _ERR_METRIC_NEG         = "Metric should be non-negative."
    _ERR_METRIC_POS         = "Metric should be positive."

    _LOG_IMPROVEMENT        = "EarlyStopping: Metric improved to {:.4e}."
    _LOG_NO_IMPROVEMENT     = "EarlyStopping: No improvement for {} epoch(s)."
    _LOG_PATIENCE_EXCEEDED  = "EarlyStopping: Metric did not improve for {} epochs. Stopping."
    _LOG_NAN_INF            = "EarlyStopping: Received NaN or Inf metric. Stopping."
    
    def __init__(self, patience: int = 0, min_delta: float = 1e-3, logger: Optional[Logger] = None):
        """
        Args:
            patience (int):
                Number of epochs to wait after last time validation metric improved.
                If 0 or None, early stopping is disabled. Default: 0.
            min_delta (float):
                Minimum change in the monitored quantity (real part) to qualify
                as an improvement. Default: 1e-3.
            logger (Optional[Logger]):
                A logger instance conforming to the expected interface.
                If None, uses standard print. Default: None.
        """
        
        super().__init__(logger=logger)
        
        if patience is not None and patience < 0:
            raise ValueError(self._ERR_PATIENCE)
        if min_delta < 0.0:
            raise ValueError(self._ERR_MIN_DELTA)

        self._patience          : Optional[int] = patience
        self._min_delta         : float         = min_delta
        self._best_metric       : float         = _INF      # Stores the real part
        self._epoch_since_best  : int           = 0
        self._stop_training     : bool          = False

    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the EarlyStopping instance
        for better readability.
        """
        state = (
            f"Patience           : {self._patience}\n"
            f"  Min Delta        : {self._min_delta:.2e}\n"
            f"  Best Metric      : {self._best_metric:.4e}\n"
            f"  Epoch Since Best : {self._epoch_since_best}\n"
            f"  Stop Training    : {self._stop_training}"
        )
        return f"{self.__class__.__name__}:\n  {state.replace(chr(10), chr(10) + '  ')}"
    
    ##########################################################################
    
    def __call__(self, _metric: Union[float, complex, np.number]) -> bool:
        """
        Updates the state based on the current metric and returns whether to stop.

        Args:
            _metric (Union[float, complex, np.number]):
                The metric value for the current epoch (e.g., validation loss).
                If complex, the real part is used for comparison.
                Lower real part values are assumed to be better.

        Returns:
            bool:
                True if training should stop, False otherwise.

        Raises:
            TypeError: If the metric is not a supported numeric type.
        """
        
        # Check if the metric is a valid numeric type
        if not isinstance(_metric, (float, complex, np.number)):
            # transform to float if possible
            try:
                _metric = float(_metric)
            except (ValueError, TypeError):
                raise TypeError(self._ERR_METRIC)

        # Extract the real part for comparison if complex
        metric_val      = _metric.real if isinstance(_metric, complex) else float(_metric)
        metric_val_im   = _metric.imag if isinstance(_metric, complex) else 0.0
        
        # Check if the metric value is NaN or infinity
        if self._check_nan(metric_val, metric_val_im):
            return True
        
        if self._check_infty(metric_val, metric_val_im):
            return True
        
        # If patience is 0 or None, early stopping is disabled
        if self._patience is None or self._patience == 0:
            return False

        # Check if current metric has improved significantly
        self._check_metric(metric_val)
        
        # Check if patience has been exceeded
        self._check_patience()

        return self._stop_training

    ##########################################################################
    
    def reset(self):
        """Resets the state of the early stopper."""
        self._best_metric      = _INF
        self._epoch_since_best = 0
        self._stop_training    = False
        self._log("EarlyStopping reset.", log='info', lvl=1)

    ###########################################################################
    
    def _check_infty(self, _metric_r, _metric_im) -> bool:
        """
        Check if the given metric values are infinite.
        """
        istrue              = np.isinf(_metric_r) or np.isinf(_metric_im)
        if istrue:
            self._log(self._ERR_METRIC_INF, log='error', lvl=1, color='red')
            self._stop_training = istrue
        return istrue
    
    def _check_nan(self, _metric_r: float, _metric_i: float) -> bool:
        """Checks if real or imaginary parts of the metric are NaN."""
        is_nan = np.isnan(_metric_r) or np.isnan(_metric_i)
        if is_nan:
            self._log(self._ERR_METRIC_NAN, log='error', lvl=1, color='red')
            self._stop_training = True
        return is_nan

    def _check_metric(self, _metric_r: float):
        """Checks if the metric's real part has improved significantly."""
        if _metric_r < (self._best_metric - self._min_delta):
            self._log(self._LOG_IMPROVEMENT.format(_metric_r), log='debug', lvl=1)
            self._best_metric      = _metric_r
            self._epoch_since_best = 0
            self._stop_training    = False # Reset stop flag on improvement
        else:
            self._epoch_since_best += 1
            self._log(self._LOG_NO_IMPROVEMENT.format(self._epoch_since_best, self._best_metric), log='debug', lvl=1)

    def _check_patience(self):
        """Checks if the patience count has been exceeded."""
        patience_exceeded = self._epoch_since_best >= self._patience
        if patience_exceeded and not self._stop_training:
            self._log(self._LOG_PATIENCE_EXCEEDED.format(self._patience), log='info', lvl=0, color='yellow')
            self._stop_training = True

    ###########################################################################
    
    @property
    def patience(self) -> Optional[int]:
        """Number of epochs to wait for improvement before stopping."""
        return self._patience

    @property
    def min_delta(self) -> float:
        """Minimum change in the monitored quantity to qualify as an improvement."""
        return self._min_delta

    @property
    def best_metric(self) -> float:
        """Best metric value observed so far."""
        return self._best_metric

    @property
    def epoch_since_best(self) -> int:        
        """Number of epochs since the best metric was observed."""
        return self._epoch_since_best

    @property
    def stop_training(self) -> bool:
        """Flag indicating whether training should be stopped."""
        return self._stop_training

    ############################################################################

# ##############################################################################
#! Base Parameters & Schedulers
# ##############################################################################


class SchedulerType(enum.Enum):
    """Enum defining the types of available learning rate schedulers."""
    CONSTANT    = 0
    EXPONENTIAL = 1
    STEP        = 2
    COSINE      = 3
    ADAPTIVE    = 4 # Corresponds to ReduceLROnPlateau from PyTorch

# ##############################################################################

class Parameters(InitialSchedulerClass, ABC):
    """
    Abstract Base Class for learning rate schedulers.
    Holds common parameters, manages early stopping, and provides logging.
    """
    
    _ERR_LR            = "Initial learning rate must be positive."
    _ERR_MAX_EPOCHS    = "Maximum epochs must be positive."
    _ERR_LR_DECAY      = "Learning rate decay must be positive."
    _ERR_LR_CLAMP      = "Learning rate clamp must be positive."
    
    
    def __init__(self, 
                initial_lr  : float,
                max_epochs  : int,
                lr_decay    : float,
                lr_clamp    : Optional[float]           = None,
                logger      : Optional[Logger]          = None,
                es          : Optional[EarlyStopping]   = None):
        """
        Initializes the base Parameters class.

        Args:
            initial_lr (float): The starting learning rate. Must be positive.
            max_epochs (int):
                Maximum number of training epochs. Must be positive.
            lr_decay (float):
                Learning rate decay factor. Interpretation depends on the subclass.
            lr_clamp (Optional[float]):
                Optional minimum learning rate clamp. Defaults to None.
            logger (Optional[Logger]):
                Logger instance. Defaults to None (uses print).
        """
        super().__init__(logger=logger)

        if initial_lr <= 0:
            raise ValueError(self._ERR_LR)
        if max_epochs <= 0:
            raise ValueError(self._ERR_MAX_EPOCHS)
        if lr_decay <= 0:
            raise ValueError(self._ERR_LR_DECAY)

        # --- Core Attributes ---
        self._typek             : Optional[SchedulerType] = None    # Set by subclasses
        self._max_epochs        : int           = max_epochs        # Maximum epochs
        self._initial_lr        : float         = initial_lr        # Initial learning rate
        self._lr                : float         = initial_lr        # Current learning rate
        self._lr_history        : List[float]   = []
        self._lr_decay          : float         = lr_decay
        self._lr_clamp          : Optional[float] = lr_clamp        # Optional minimum LR clamp
        
        if lr_clamp is not None and lr_clamp <= 0:
            raise ValueError(self._ERR_LR_CLAMP)
        
        # --- Early Stopping ---
        self._early_stopping    : Optional[EarlyStopping] = es

    #################################################################
    
    def __repr__(self) -> str:
        """Provides a detailed multi-line representation."""
        
        es_repr = f"\n  Early Stopping   : {self._early_stopping!r}".replace(chr(10), chr(10) + '    ') if self._early_stopping else "\n  Early Stopping   : Not Configured"
        state   = (
            f"Type             : {self._typek.name if self._typek else 'None'}\n"
            f"  Initial LR       : {self._initial_lr:.4e}\n"
            f"  Max Epochs       : {self._max_epochs}\n"
            f"  LR Decay         : {self._lr_decay:.4f}\n"
            f"  Current LR       : {self._lr:.4e}"
            f"{es_repr}"
        )
        return f"{self.__class__.__name__}:\n  {state.replace(chr(10), chr(10) + '  ')}"

    #################################################################
    
    @property
    def lr(self) -> float:
        """Current learning rate."""
        return self._lr

    @property
    def lr_decay(self) -> float:
        """Learning rate decay factor (meaning depends on scheduler type)."""
        return self._lr_decay

    @lr_decay.setter
    def lr_decay(self, value: float):
        self._lr_decay = value

    @property
    def lr_clamp(self) -> Optional[float]:
        """Optional minimum learning rate clamp."""
        return self._lr_clamp
    
    @lr_clamp.setter
    def lr_clamp(self, value: Optional[float]):
        if value is not None and value <= 0:
            raise ValueError(self._ERR_LR_CLAMP)
        self._lr_clamp = value
    
    @property
    def max_epochs(self) -> int:
        """Maximum number of training epochs."""
        return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, value: int):
        if value <= 0:
            raise ValueError(self._ERR_MAX_EPOCHS)
        self._max_epochs = value

    @property
    def type(self) -> Optional[SchedulerType]:
        """The type of the scheduler (SchedulerType enum member)."""
        return self._typek

    @property
    def history(self) -> List[float]:
        """History of learning rates generated by calls to the scheduler."""
        return self._lr_history

    @property
    def best_metric(self) -> float:
        """Best metric value observed by the early stopper (if configured), else infinity."""
        return self._early_stopping.best_metric if self._early_stopping else _INF

    @property
    def early_stopping(self) -> Optional[EarlyStopping]:
        """Returns the EarlyStopping instance if configured, else None."""
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, es: EarlyStopping):
        """Sets the EarlyStopping instance."""
        if not isinstance(es, EarlyStopping):
            raise TypeError("Expected an instance of EarlyStopping.")
        self._early_stopping = es
        
    ###########################################################################
    
    def _update_and_log_lr(self, new_lr: float) -> float:
        """
        Internal helper to update current LR, add to history, and return it.
        Potentially clamps the LR to a minimum value if specified.
        Args:
            new_lr (float): The new learning rate to set.
        Returns:
            float: The updated learning rate.
        """
        
        # Could add min_lr clamping here globally if desired
        new_lr      = np.maximum(new_lr, self.lr_clamp) if self.lr_clamp is not None else new_lr
        self._lr    = new_lr
        self._lr_history.append(self._lr)
        return self._lr

    ###########################################################################

    @abstractmethod
    def __call__(self, _epoch: int, _metric: Optional[Union[float, complex, np.number]] = None) -> float:
        """
        Calculate and return the learning rate for the given epoch.
        Subclasses must implement this method.

        Args:
            epoch (int):
                The current epoch number (usually 0-based).
            metric (Optional[Union[float, complex, np.number]]):
                An optional metric value (e.g., validation loss) used by some
                schedulers (like Adaptive). If complex, the real part may be used.

        Returns:
            float: The calculated learning rate for this epoch.
        """
        pass

    ###########################################################################
    #! Setters and Getters
    ###########################################################################

    # --- Early Stopping Integration ---
    def set_early_stopping(self, patience: int, min_delta: float = 1e-3):
        """
        Configures or reconfigures the internal early stopping mechanism.

        Args:
            patience (int):
                Number of epochs to wait for metric improvement.
            min_delta (float):
                Minimum change considered an improvement.
        """
        self._log(f"Setting EarlyStopping: patience={patience}, min_delta={min_delta:.2e}", log='info', lvl=1)
        self._early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, logger=self._logger)

    def check_stop(self, _metric: Union[float, complex, np.number]) -> bool:
        """
        Checks if the early stopping criteria are met based on the provided metric.

        Args:
            metric (Union[float, complex, np.number]):
                The metric value for the current epoch.

        Returns:
            bool:
                True if training should stop based on early stopping, False otherwise.
                Returns False if early stopping is not configured.
        """
        if self._early_stopping:
            return self._early_stopping(_metric)
        self._log("check_stop called but EarlyStopping is not configured.", log='warning', lvl=1)
        return False

    def reset_early_stopping(self):
        """Resets the internal state of the early stopper, if configured."""
        if self._early_stopping:
            self._early_stopping.reset()
        else:
            self._log("reset_early_stopping called but EarlyStopping is not configured.", log='debug', lvl=1)

    ############################################################################

################################################################################
#! CONCRETE SCHEDULERS
################################################################################

class ConstantScheduler(Parameters):
    """Keeps the learning rate constant throughout training."""

    ##########################################################################

    def __init__(self,
                initial_lr  : float,
                max_epochs  : int,
                lr_clamp    : Optional[float]         = None,
                logger      : Optional[Logger]        = None,
                es          : Optional[EarlyStopping] = None,
                **kwargs):
        """
        Args:
            initial_lr (float):
                The constant learning rate to use.
            max_epochs (int):
                Max epochs (used for context, doesn't affect LR).
            lr_clamp (Optional[float]):
                Optional minimum LR clamp.
            logger (Optional[Logger]):
                Logger instance.
            es (Optional[EarlyStopping]):
                Pre-configured EarlyStopping instance.
            **kwargs: Ignored.
        """
        # lr_decay is unused, set arbitrarily to 1.0 for base class
        super().__init__(initial_lr=initial_lr, max_epochs=max_epochs, lr_decay=1.0, lr_clamp=lr_clamp, logger=logger, es=es)
        self._typek = SchedulerType.CONSTANT
        self._log(f"Initialized with constant LR: {self._initial_lr:.4e}", log='info')

    ##########################################################################

    def __call__(self, _epoch: int, _metric: Optional[Union[float, complex, np.number]] = None) -> float:
        """
        Returns the constant initial learning rate.
        
        Note:
            The learning rate remains constant throughout training.
            The epoch and metric are ignored in this case.
        Args:
            _epoch (int):
                The current epoch number (ignored).
            _metric (Optional[Union[float, complex, np.number]]):
                Optional metric value (ignored).
        """
        
        # The LR doesn't change, but clamping might apply via _update_and_log_lr
        return self._update_and_log_lr(self._initial_lr)

# ##############################################################################

class ExponentialDecayScheduler(Parameters):
    """
    Applies exponential decay: 
        lr = initial_lr * exp(-decay_rate * epoch).
    Note:
        The learning rate changes over time based on the decay rate.
    """

    _WARN_LR_DECAY = "ExponentialDecay 'lr_decay' (rate constant) is usually non-negative. Got {}"

    ##########################################################################

    def __init__(self,
                initial_lr  : float,
                max_epochs  : int,
                lr_decay    : float, # Here, lr_decay is the decay *rate* constant
                lr_clamp    : Optional[float]         = None,
                logger      : Optional[Logger]        = None,
                es          : Optional[EarlyStopping] = None,
                **kwargs):
        """
        Args:
            initial_lr (float):
                Initial learning rate.
            max_epochs (int):
                Maximum number of epochs.
            lr_decay (float):
                The exponential decay rate constant (often k or gamma). Must be >= 0.
            lr_clamp (Optional[float]):
                Optional minimum LR clamp.
            logger (Optional[Logger]):
                Logger instance.
            es (Optional[EarlyStopping]):
                Pre-configured EarlyStopping instance.
            **kwargs: Ignored.
        """
        
        super().__init__(initial_lr=initial_lr, max_epochs=max_epochs, lr_decay=lr_decay, lr_clamp=lr_clamp, logger=logger, es=es)
        
        # Ensure lr_decay is non-negative
        if lr_decay < 0.0:
            self._log(self._WARN_LR_DECAY.format(lr_decay), log='warning')

        self._typek = SchedulerType.EXPONENTIAL
        self._log(f"Initialized with initial LR: {self._initial_lr:.4e}, decay rate: {self._lr_decay:.4f}", log='info')

    ##########################################################################

    def __call__(self, _epoch: int, _metric: Optional[Union[float, complex, np.number]] = None) -> float:
        """
        Calculates LR using exponential decay formula.
        
        Args:
            _epoch (int):
                The current epoch number.
            _metric (Optional[Union[float, complex, np.number]]):
                Optional metric value (ignored).
        Returns:
            float: The calculated learning rate for this epoch.
        Note:
            The learning rate decreases exponentially based on the decay rate.
        """
        current_epoch = max(0, _epoch) # Ensure epoch >= 0
        # self._lr_decay holds the decay *rate* here
        new_lr        = self._initial_lr * np.exp(-self._lr_decay * current_epoch)
        # Clamping applied here
        return self._update_and_log_lr(new_lr)

# ##############################################################################

class StepDecayScheduler(Parameters):
    """
    Decays the learning rate by a factor 'lr_decay' every 'step_size' epochs.
    
    Formula: lr = initial_lr * decay_factor ^ floor(epoch / step_size).
    Note:
        The learning rate changes at regular intervals (step_size) based on the decay factor.
    """

    _ERR_STEP_SIZE    = "Step size must be positive."
    _WARN_LR_DECAY    = "StepDecay 'lr_decay' factor is typically in (0, ...]. Got {}"

    ##########################################################################

    def __init__(self,
                initial_lr  : float,
                max_epochs  : int,
                lr_decay    : float, # Here, lr_decay is the decay *factor*
                step_size   : int,
                lr_clamp    : Optional[float]         = None,
                logger      : Optional[Logger]        = None,
                es          : Optional[EarlyStopping] = None,
                **kwargs):
        """
        Args:
            initial_lr (float):
                Initial learning rate.
            max_epochs (int):
                Maximum number of epochs.
            lr_decay (float):
                Multiplicative factor of LR decay. Typically (0, 1].
            step_size (int):
                Period of learning rate decay (epochs). Must be > 0.
            lr_clamp (Optional[float]):
                Optional minimum LR clamp.
            logger (Optional[Logger]):
                Logger instance.
            es (Optional[EarlyStopping]):
                Pre-configured EarlyStopping instance.
            **kwargs: Ignored.
        """
        super().__init__(initial_lr=initial_lr, max_epochs=max_epochs, lr_decay=lr_decay, lr_clamp=lr_clamp, logger=logger, es=es)
        
        if step_size <= 0:
            raise ValueError(self._ERR_STEP_SIZE)
        if not (0.0 < lr_decay):
            self._log(self._WARN_LR_DECAY.format(lr_decay), log='warning')

        self.step_size : int            = step_size
        self._typek    : SchedulerType  = SchedulerType.STEP
        self._log(f"Initialized with initial LR: {self._initial_lr:.4e}, decay factor: {self._lr_decay:.4f}, step size: {self.step_size}", log='info')

    ##########################################################################

    def __call__(self, _epoch: int, _metric: Optional[Union[float, complex, np.number]] = None) -> float:
        """
        Calculates LR using step decay formula.
        
        Args:
            _epoch (int):
                The current epoch number.
            _metric (Optional[Union[float, complex, np.number]]):
                Optional metric value (ignored).
        """
        
        # Formula: lr = initial_lr * decay_factor ^ floor(epoch / step_size)
        
        current_epoch = max(0, _epoch)
        exponent      = np.floor(current_epoch / self.step_size)
        factor        = np.power(self._lr_decay, exponent)
        new_lr        = self._initial_lr * factor
        return self._update_and_log_lr(new_lr)

    ##########################################################################

    def __repr__(self) -> str:
        """Adds step_size to the base representation."""
        base_repr       = super().__repr__()
        lines           = base_repr.split('\n')
        es_line_idx     = -1
        clamp_line_idx  = -1
        
        # Find the indices of the lines to insert before
        for i, line in enumerate(lines):
            if 'Early Stopping' in line:
                    es_line_idx = i
            if 'LR Clamp' in line:
                clamp_line_idx = i

        insert_idx = es_line_idx if es_line_idx != -1 else len(lines)
        if clamp_line_idx != -1 and clamp_line_idx < insert_idx: insert_idx = clamp_line_idx

        step_line = f"  Step Size        : {self.step_size}"
        
        # Insert before clamp or ES info
        lines.insert(insert_idx, step_line)
        return '\n'.join(lines)

# ##############################################################################

class CosineAnnealingScheduler(Parameters):
    """
    Sets the learning rate using a cosine annealing schedule.
    Formula: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / T_max)).
    """

    _ERR_MIN_LR       = "Minimum learning rate cannot be negative."

    ##########################################################################

    def __init__(self,
                initial_lr  : float,
                max_epochs  : int,
                min_lr      : float = 0.0,
                lr_clamp    : Optional[float]         = None,
                logger      : Optional[Logger]        = None,
                es          : Optional[EarlyStopping] = None,
                **kwargs):
        """
        Args:
            initial_lr (float):
                Initial learning rate (upper bound).
            max_epochs (int):
                Number of epochs for one cycle of annealing (T_max).
            min_lr (float):
                Minimum learning rate (lower bound, eta_min). Default: 0.0.
            lr_clamp (Optional[float]):
                Optional minimum LR clamp (applied after cosine calculation).
            logger (Optional[Logger]):
                Logger instance.
            es (Optional[EarlyStopping]):
                Pre-configured EarlyStopping instance.
            **kwargs: Ignored.
        """
        # lr_decay is unused, set arbitrarily to 0.0 for base class
        super().__init__(initial_lr=initial_lr, max_epochs=max_epochs, lr_decay=0.0, lr_clamp=lr_clamp, logger=logger, es=es)
        
        if min_lr < 0:
            raise ValueError(self._ERR_MIN_LR)

        # Mathematical lower bound for cosine curve
        self.min_lr : float         = min_lr
        self._typek : SchedulerType = SchedulerType.COSINE
        self._log(f"Initialized with LR range: [{self.min_lr:.4e}, {self._initial_lr:.4e}], T_max: {self._max_epochs}", log='info')

    ##########################################################################

    def __call__(self, _epoch: int, _metric: Optional[Union[float, complex, np.number]] = None) -> float:
        """
        Calculates LR using cosine annealing formula.
        
        Args:
            _epoch (int):
                The current epoch number.
            _metric (Optional[Union[float, complex, np.number]]):
                Optional metric value (ignored).
        Returns:
            float: The calculated learning rate for this epoch.
        """
        
        # Formula: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / T_max))
        current_epoch = np.clip(_epoch, 0, self._max_epochs) # Clamp epoch for calculation
        
        # Avoid division by zero, return min_lr
        if self._max_epochs <= 0:
            cosine_term = -1.0
        elif self._max_epochs == 1:
            cosine_term = -1.0 if current_epoch >= 1 else 1.0
        else:
            cosine_term = np.cos(np.pi * current_epoch / self._max_epochs)

        lr_range = self._initial_lr - self.min_lr
        new_lr   = self.min_lr + 0.5 * lr_range * (1.0 + cosine_term)

        # Ensure LR doesn't go below the mathematical min_lr from the formula
        new_lr = np.maximum(self.min_lr, new_lr)

        # Apply final clamp if specified via lr_clamp (which might be > min_lr)
        return self._update_and_log_lr(new_lr)

    ##########################################################################

    def __repr__(self) -> str:
        """Adds min_lr to the base representation."""
        base_repr       = super().__repr__()
        lines           = base_repr.split('\n')
        es_line_idx     = -1
        clamp_line_idx  = -1
        for i, line in enumerate(lines):
            if 'Early Stopping' in line:
                es_line_idx = i
            if 'LR Clamp' in line:
                clamp_line_idx = i

        insert_idx      = es_line_idx if es_line_idx != -1 else len(lines)
        if clamp_line_idx != -1 and clamp_line_idx < insert_idx:
            insert_idx = clamp_line_idx

        min_lr_line     = f"  Min LR (Cosine)  : {self.min_lr:.4e}"
        lines.insert(insert_idx, min_lr_line)
        return '\n'.join(lines)

# ##############################################################################

class AdaptiveScheduler(Parameters):
    """
    Reduces learning rate when a metric (real part if complex) has stopped improving.
    Similar to PyTorch's ReduceLROnPlateau.
    """
    # --- Error Messages ---
    _ERR_PATIENCE       = "Patience must be non-negative."
    _ERR_MIN_LR         = "Minimum learning rate (min_lr) cannot be negative."
    _ERR_COOLDOWN       = "Cooldown must be non-negative."
    _ERR_MIN_DELTA      = "min_delta must be non-negative."
    _ERR_METRIC_REQ     = "AdaptiveScheduler requires a metric to be provided."
    _ERR_METRIC_TYPE    = "AdaptiveScheduler metric must be a number (float, complex, np.number)."

    # --- Logging Messages ---
    _LOG_REDUCE_LR      = "Reducing learning rate to {:.2e} due to lack of improvement for {} epochs."
    _LOG_COOLDOWN_START = "Entering cooldown period for {} epochs."
    _LOG_COOLDOWN_STEP  = "In cooldown... {} epochs remaining."
    _LOG_IMPROVEMENT    = "Metric improved to {:.4e}. Resetting patience."
    _LOG_NO_IMPROVEMENT = "No improvement for {} epoch(s). Best: {:.4e}"
    _LOG_RESET          = "AdaptiveScheduler state reset (best_metric, cooldown, bad_epochs)."
    _LOG_NAN_INF        = "Received NaN or Inf metric. Cannot adapt LR, returning current LR."
    _WARN_LR_DECAY      = "AdaptiveScheduler 'lr_decay' factor is typically in (0, 1). Got {}"

    ##########################################################################

    def __init__(self,
                initial_lr  : float,
                max_epochs  : int,
                lr_decay    : float, # Factor to reduce LR by
                patience    : int,
                min_lr      : float = 1e-5,
                cooldown    : int = 0,
                min_delta   : float = 1e-4,
                lr_clamp    : Optional[float]         = None, # Separate clamp applied at the end
                logger      : Optional[Logger]        = None,
                es          : Optional[EarlyStopping] = None,
                **kwargs):
        """
        Args:
            initial_lr (float):
                Initial learning rate.
            max_epochs (int):
                Maximum number of training epochs.
            lr_decay (float):
                Factor to reduce LR by (new_lr = lr * lr_decay). Typically (0, 1).
            patience (int):
                Epochs with no improvement before reducing LR (>= 0).
            min_lr (float):
                Lower bound for LR reduction (>= 0). Default: 1e-5.
            cooldown (int):
                Epochs to wait after LR reduction (>= 0). Default: 0.
            min_delta (float):
                Min change considered an improvement (>= 0). Default: 1e-4.
            lr_clamp (Optional[float]):
                Optional final minimum LR clamp. Default: None.
            logger (Optional[Logger]):
                Logger instance.
            es (Optional[EarlyStopping]):
                Pre-configured EarlyStopping instance.
            **kwargs: Ignored.
        """
        super().__init__(initial_lr=initial_lr, max_epochs=max_epochs, lr_decay=lr_decay, lr_clamp=lr_clamp, logger=logger, es=es)

        if not (0.0 < lr_decay < 1.0):
            self._log(self._WARN_LR_DECAY.format(lr_decay), log='warning')
        if patience < 0:
            raise ValueError(self._ERR_PATIENCE)
        if min_lr < 0:
            raise ValueError(self._ERR_MIN_LR)
        if cooldown < 0:
            raise ValueError(self._ERR_COOLDOWN)
        if min_delta < 0:
            raise ValueError(self._ERR_MIN_DELTA)

        # --- Adaptive Specific Attributes ---
        # Note: self._lr_decay from base class is used as the reduction factor
        self.patience           : int       = patience
        self.min_lr             : float     = min_lr    # Lower bound for reduction logic
        self.cooldown           : int       = cooldown
        self.min_delta          : float     = min_delta # Threshold for improvement check

        # --- Internal State ---
        self._cooldown_counter  : int       = 0
        self._best_metric       : float     = _INF      # Tracks real part of metric
        self._num_bad_epochs    : int       = 0

        self._typek             : SchedulerType = SchedulerType.ADAPTIVE
        self._log(f"Initialized with initial LR: {self._initial_lr:.4e}, decay: {self._lr_decay:.2f}, "
                f"patience: {self.patience}, min_lr: {self.min_lr:.1e}, cooldown: {self.cooldown}, "
                f"min_delta: {self.min_delta:.1e}", log='info')

    ##########################################################################

    def __call__(self, _epoch: int, _metric: Optional[Union[float, complex, np.number]]) -> float:
        """
        Calculates and potentially reduces the learning rate based on metric improvement.
        Args:
            _epoch (int):
                The current epoch number.
            _metric (Optional[Union[float, complex, np.number]]):
                The metric value for the current epoch (real part used).
        Returns:
            float: The updated learning rate for this epoch.
        """
        if _metric is None:
            self._log(self._ERR_METRIC_REQ, log='error', color='red')
            raise ValueError(self._ERR_METRIC_REQ)

        if not isinstance(_metric, (float, complex, np.number)):
            self._log(self._ERR_METRIC_TYPE, log='error', color='red')
            raise TypeError(self._ERR_METRIC_TYPE)

        metric_val_r = np.real(_metric)
        metric_val_i = np.imag(_metric)

        if np.isnan(metric_val_r) or np.isnan(metric_val_i) or np.isinf(metric_val_r) or np.isinf(metric_val_i):
            self._log(self._LOG_NAN_INF, level='error', color='red')
            return self._update_and_log_lr(self._lr) # Return current LR

        current_lr = self._lr # Start with LR from previous step

        if self._cooldown_counter > 0:
            self._cooldown_counter      -= 1
            self._num_bad_epochs        = 0
            self._log(self._LOG_COOLDOWN_STEP.format(self._cooldown_counter), log='debug', lvl=1)
            # In cooldown, return current LR
            return self._update_and_log_lr(current_lr)

        # Check for improvement vs best metric (using min_delta)
        is_improvement = metric_val_r < self._best_metric - self.min_delta

        if is_improvement:
            self._log(self._LOG_IMPROVEMENT.format(metric_val_r), log='debug', lvl=1)
            self._best_metric    = metric_val_r
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            self._log(self._LOG_NO_IMPROVEMENT.format(self._num_bad_epochs, self._best_metric), log='debug', lvl=1)

        # Check if patience is exceeded
        if self._num_bad_epochs > self.patience:
            reduced = self._reduce_lr(_epoch) # Try to reduce LR
            if reduced:
                current_lr = self._lr # Use the newly reduced LR
            self._num_bad_epochs = 0 # Reset counter after check/reduction

        # _reduce_lr already updated self._lr if reduction happened.
        return self._update_and_log_lr(self._lr)

    ##########################################################################

    def _reduce_lr(self, _epoch: int) -> bool:
        """
        Attempts to reduce the learning rate, applies cooldown, logs changes.

        Returns:
            bool: True if the learning rate was reduced, False otherwise.
        """
        
        # Apply the specific minimum LR for reduction logic
        new_lr_candidate    = self._lr * self._lr_decay
        new_lr              = np.maximum(new_lr_candidate, self.min_lr)

        if new_lr < self._lr:
            self._log(self._LOG_REDUCE_LR.format(new_lr, self.patience), log='info', lvl=0, color='yellow')
            # Update internal LR *before* cooldown logic
            self._lr                = new_lr 
            self._cooldown_counter  = self.cooldown
            if self.cooldown > 0:
                self._log(self._LOG_COOLDOWN_START.format(self.cooldown), log='debug', lvl=1)
            return True
        # else: LR is already at or below min_lr, no reduction applied
        return False

    ##########################################################################

    def reset(self):
        """
        Resets the adaptive state (cooldown, best metric, bad epochs).
        Note:
            This does not reset the learning rate or other base parameters.
        """
        self._cooldown_counter  = 0
        self._best_metric       = _INF
        self._num_bad_epochs    = 0
        self._log(self._LOG_RESET, log='debug', lvl=1)

    ##########################################################################

    def __repr__(self) -> str:
        """
        Adds adaptive-specific parameters to the base representation.
        """
        base_repr       = super().__repr__()
        lines           = base_repr.split('\n')
        es_line_idx     = -1
        clamp_line_idx  = -1
        for i, line in enumerate(lines):
            if 'Early Stopping' in line:
                es_line_idx = i
            if 'LR Clamp' in line:
                clamp_line_idx = i

        insert_idx      = es_line_idx if es_line_idx != -1 else len(lines)
        if clamp_line_idx != -1 and clamp_line_idx < insert_idx:
            insert_idx = clamp_line_idx

        adaptive_lines = [
            f"  Patience         : {self.patience}",
            f"  Min LR (Adaptive): {self.min_lr:.4e}",
            f"  Cooldown         : {self.cooldown}",
            f"  Min Delta        : {self.min_delta:.4e}",
            f"  Best Metric      : {self._best_metric:.4e}",
            f"  Cooldown Counter : {self._cooldown_counter}",
            f"  Num Bad Epochs   : {self._num_bad_epochs}",
        ]
        # Insert before clamp or ES info
        lines.insert(insert_idx, *adaptive_lines)
        return '\n'.join(lines)

    ###########################################################################

# ##############################################################################
#! Factory Function
# ##############################################################################

# Mapping from string/enum to class and required extra args
SCHEDULER_CLASS_MAP: Dict[Union[str, SchedulerType], Dict[str, Any]] = {
    SchedulerType.CONSTANT:    {'class': ConstantScheduler,         'extra_args': []},
    "constant":                {'class': ConstantScheduler,         'extra_args': []},

    SchedulerType.EXPONENTIAL: {'class': ExponentialDecayScheduler, 'extra_args': ['lr_decay']},
    "exponential":             {'class': ExponentialDecayScheduler, 'extra_args': ['lr_decay']},

    SchedulerType.STEP:        {'class': StepDecayScheduler,        'extra_args': ['lr_decay', 'step_size']},
    "step":                    {'class': StepDecayScheduler,        'extra_args': ['lr_decay', 'step_size']},

    SchedulerType.COSINE:      {'class': CosineAnnealingScheduler,  'extra_args': ['min_lr']},
    "cosine":                  {'class': CosineAnnealingScheduler,  'extra_args': ['min_lr']},

    SchedulerType.ADAPTIVE:    {'class': AdaptiveScheduler,         'extra_args': ['lr_decay', 'patience', 'min_lr', 'cooldown', 'min_delta']},
    "adaptive":                {'class': AdaptiveScheduler,         'extra_args': ['lr_decay', 'patience', 'min_lr', 'cooldown', 'min_delta']},
    "reducelronplateau":       {'class': AdaptiveScheduler,         'extra_args': ['lr_decay', 'patience', 'min_lr', 'cooldown', 'min_delta']}, 
}

DEFAULT_SCHEDULER_TYPE = SchedulerType.EXPONENTIAL

# ------------------------------------------------------------------------------

def _handle_existing_instance(scheduler_type: Parameters,
                            logger          : Optional[Logger],
                            kwargs          : Dict[str, Any],
                            temp_logger     : Logger) -> Parameters:
    
    """
    Handles logic when an existing scheduler instance is passed.
    Updates the logger, reconfigures early stopping, and updates lr_clamp if needed.
    Args:
        scheduler_type (Parameters):
            Existing scheduler instance.
        logger (Optional[Logger]):
            Logger instance.
        kwargs (Dict[str, Any]):
            Additional arguments for reconfiguration.
        temp_logger (Logger):
            Temporary logger instance.
    Returns:
        Parameters: The updated scheduler instance.
    Raises:
        ValueError: If the scheduler_type is not a valid Parameters instance.
    """
    factory_prefix = "[choose_scheduler]"
    temp_logger.say(f"{factory_prefix} Received existing instance: {scheduler_type.__class__.__name__}", log='debug')
    instance        = scheduler_type

    # Update logger if new and different
    if logger is not None and instance.logger is not logger:
        temp_logger.say(f"{factory_prefix} Updating logger on existing instance.", log='debug')
        instance.logger = logger
        if instance.early_stopping:
            instance.early_stopping.logger = logger

    # Reconfigure Early Stopping if requested
    es_patience     = kwargs.get('early_stopping_patience')
    es_min_delta    = kwargs.get('early_stopping_min_delta')
    if es_patience is not None:
        if es_patience > 0:
            es_min_delta_val = es_min_delta if es_min_delta is not None else 1e-4
            temp_logger.say(f"{factory_prefix} Reconfiguring ES: pat={es_patience}, delta={es_min_delta_val:.2e}", log='info')
            instance.set_early_stopping(patience=es_patience, min_delta=es_min_delta_val)
        else:
            temp_logger.say(f"{factory_prefix} Disabling ES on existing instance (pat={es_patience}).", log='info')
            instance.early_stopping = None

    # Update lr_clamp if provided
    lr_clamp = kwargs.get('lr_clamp')
    if lr_clamp is not None:
        temp_logger.say(f"{factory_prefix} Updating lr_clamp on existing instance to {lr_clamp:.4e}.", log='debug')
        instance.lr_clamp = lr_clamp
    return instance

def _resolve_scheduler_class(scheduler_type : Union[str, SchedulerType],
                            temp_logger     : Logger) -> Type[Parameters]:
    """
    Resolves the scheduler class from string or enum.
    Args:
        scheduler_type (Union[str, SchedulerType]):
            Type of scheduler (string or enum).
        temp_logger (Logger):
            Temporary logger instance.    
    Returns:
        Type[Parameters]: The resolved scheduler class.
    """
    factory_prefix = "[choose_scheduler]"
    if isinstance(scheduler_type, str):
        lookupJAX_RND_DEFAULT_KEY = scheduler_type.lower()
    elif isinstance(scheduler_type, SchedulerType):
        lookupJAX_RND_DEFAULT_KEY = scheduler_type
    else:
        err_msg = f"scheduler_type must be string, Enum, or Parameters, got {type(scheduler_type)}"
        temp_logger.say(f"{factory_prefix} {err_msg}", log='error', color='red'); raise TypeError(err_msg)

    scheduler_class = SCHEDULER_CLASS_MAP.get(lookupJAX_RND_DEFAULT_KEY)
    if scheduler_class is None:
        temp_logger.say(f"{factory_prefix} Unknown type '{scheduler_type}'. Using default: {DEFAULT_SCHEDULER_TYPE.name}", 'warning', color='yellow')
        scheduler_class = SCHEDULER_CLASS_MAP.get(DEFAULT_SCHEDULER_TYPE)
        if scheduler_class is None:
            err_msg = "Default scheduler config missing."
            temp_logger.say(f"{factory_prefix} {err_msg}", 'critical', color='red')
            raise ValueError(err_msg)
    temp_logger.say(f"{factory_prefix} Resolved class: {scheduler_class.__name__}", log='debug')
    return scheduler_class

def _create_early_stopping(logger: Optional[Logger],
                        kwargs: Dict[str, Any], temp_logger: Logger) -> Optional[EarlyStopping]:
    """
    Create an EarlyStopping instance based on the provided keyword arguments.

    Args:
        logger      : Optional[Logger]
            Logger instance.
        kwargs      : Dict[str, Any]
            Additional configuration parameters.
        temp_logger : Logger
            Logger for internal messages.

    Returns:
        Optional[EarlyStopping]
            Configured EarlyStopping instance if enabled; otherwise, None.
    """
    factory_prefix      = "[choose_scheduler]"
    es_patience         = kwargs.get('early_stopping_patience')
    es_min_delta        = kwargs.get('early_stopping_min_delta')
    es_instance         = None

    if es_patience is not None and es_patience > 0:
        es_min_delta_val    = es_min_delta if es_min_delta is not None else 1e-4
        temp_logger.say(f"{factory_prefix} Creating ES instance: pat={es_patience}, delta={es_min_delta_val:.2e}",
                        log='debug')
        es_instance         = EarlyStopping(patience=es_patience, min_delta=es_min_delta_val, logger=logger)
    elif es_patience is not None and es_patience <= 0:
        temp_logger.say(f"{factory_prefix} Early stopping explicitly disabled (pat={es_patience}).",
                        log='info')
    
    return es_instance

def _prepare_constructor_args(initial_lr: float,
                            max_epochs  : int,
                            logger      : Optional[Logger],
                            es_instance : Optional[EarlyStopping],
                            kwargs      : Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare the keyword arguments for initializing a scheduler.

    Args:
        initial_lr  : float
            Initial learning rate.
        max_epochs  : int
            Maximum number of epochs.
        logger      : Optional[Logger]
            Logger instance.
        es_instance : Optional[EarlyStopping]
            EarlyStopping instance.
        kwargs      : Dict[str, Any]
            Additional parameters.

    Returns:
        Dict[str, Any]
            Dictionary of arguments for scheduler instantiation.
    """
    constructor_kwargs  = {
        'initial_lr' : initial_lr,
        'max_epochs' : max_epochs,
        'lr_decay'   : kwargs.get('lr_decay', 0.9),
        'lr_clamp'   : kwargs.get('lr_clamp'),
        'logger'     : logger,
        'es'         : es_instance,
    }
    constructor_kwargs.update({ k: v for k, v in kwargs.items() if k not in constructor_kwargs })
    return constructor_kwargs

def _instantiate_scheduler(scheduler_class  : Type[Parameters],
                        constructor_kwargs  : Dict[str, Any],
                        temp_logger         : Logger) -> Parameters:
    """
    Instantiate the scheduler with the specified class and constructor arguments.

    Args:
        scheduler_class   : Type[Parameters]
            Scheduler class to instantiate.
        constructor_kwargs: Dict[str, Any]
            Arguments for the scheduler's constructor.
        temp_logger       : Logger
            Logger for internal messages.

    Returns:
        Parameters
            The instantiated scheduler object.

    Raises:
        TypeError:
            If the constructor arguments do not match the scheduler's requirements.
        ValueError:
            If any constructor argument is invalid.
    """
    factory_prefix      = "[choose_scheduler]"
    temp_logger.say(f"{factory_prefix} Constructor Args for {scheduler_class.__name__}: {constructor_kwargs}", 
                    log='debug')
    
    try:
        scheduler       = scheduler_class(**constructor_kwargs)
        temp_logger.say(f"{factory_prefix} Instantiated scheduler: {scheduler_class.__name__}", log='info')
        return scheduler
    except TypeError as e:
        sig             = inspect.signature(scheduler_class.__init__)
        expected        = [p for p in sig.parameters if p not in ['self', 'kwargs']]
        err             = (
            f"Failed to instantiate {scheduler_class.__name__}. Check kwargs. Error: {e}\n"
            f"  Expected params: {expected}\n"
            f"  Provided extra kwargs: { {k: v for k, v in constructor_kwargs.items() if k not in ['initial_lr', 'max_epochs', 'logger', 'es']} }"
        )
        temp_logger.say(f"{factory_prefix} {err}", log='error', color='red')
        raise TypeError(err) from e
    except ValueError as e:
        err             = f"Failed to instantiate {scheduler_class.__name__}. Invalid argument value? Error: {e}"
        temp_logger.say(f"{factory_prefix} {err}", log='error', color='red')
        raise ValueError(err) from e

def choose_scheduler(
    scheduler_type                  : Union[str, SchedulerType, Parameters],
    initial_lr                      : float,
    max_epochs                      : int,
    logger                          : Optional[Logger] = None,
    **kwargs) -> Parameters:
    """
    Factory function to create or return a learning rate scheduler instance.

    Args:
        scheduler_type: Type (instance, string name, or Enum).
        initial_lr: Starting learning rate (> 0).
        max_epochs: Maximum training epochs (> 0).
        logger: Optional Logger instance.
        **kwargs: Additional arguments for scheduler/ES config (see Parameters/Specific Schedulers).

    Returns:
        Parameters: An instance of the configured learning rate scheduler.

    Raises:
        ValueError: If unknown scheduler type or invalid argument value.
        TypeError: If invalid scheduler_type or missing required arguments.
    """
    temp_logger = logger if logger else Logger()

    #! 1. Handle existing instance
    if isinstance(scheduler_type, Parameters):
        return _handle_existing_instance(scheduler_type, logger, kwargs, temp_logger)

    #! 2. Resolve class type
    scheduler_class = _resolve_scheduler_class(scheduler_type, temp_logger)

    #! 3. Prepare Early Stopping (if requested)
    es_instance = _create_early_stopping(logger, kwargs, temp_logger)

    #! 4. Prepare Constructor Arguments
    constructor_args = _prepare_constructor_args(initial_lr, max_epochs, logger, es_instance, kwargs)

    #! 5. Instantiate Scheduler
    scheduler = _instantiate_scheduler(scheduler_class, constructor_args, temp_logger)

    return scheduler

# ##############################################################################
#! Example Usage Class
# ##############################################################################

class SchedulerTester(InitialSchedulerClass):
    """
    SchedulerTester is a demonstration and testing class for various learning rate scheduler
    mechanisms and the EarlyStopping feature. It inherits from InitialSchedulerClass and
    provides a series of tests that simulate different training scenarios, metric inputs, and
    scheduler behaviors to validate their functionality in a controlled environment.

    Class Methods:
        __init__(logger: Optional[Logger] = None):
            - Sets up default testing parameters such as max_epochs_test (default 20) and
              initial_lr_test (default 0.1).
            - Optionally accepts a logger for detailed output during tests.
            - Logs an initialization message.

        run_all_tests():
            - Runs the complete suite of scheduler tests in sequence, including:
                • test_early_stopping: Validates early stopping behavior.
                • test_exponential: Tests the Exponential Decay scheduler.
                • test_adaptive: Tests the Adaptive (ReduceLROnPlateau) scheduler.
                • test_cosine: Tests the Cosine Annealing scheduler.
                • test_step: Tests the Step Decay scheduler.
                • test_constant: Tests the Constant Learning Rate scheduler.
                • test_pass_instance: Verifies that an already instantiated scheduler can be passed
                  to the factory for updates.
                • test_missing_arg: Checks for proper error handling when required arguments are missing.
            - Provides logging before and after running all tests.

        test_early_stopping():
            - Directly tests the EarlyStopping mechanism using a series of metrics, including:
                • Real numbers with gradually improving values.
                • Complex numbers to verify robustness.
                • NaN and Inf values to validate error handling.
            - Logs the outcome at each epoch, indicating when early stopping is triggered.

        _run_scheduler_simulation(scheduler: Parameters, metrics: Optional[List[Any]] = None):
            - Simulates the scheduler behavior over a number of epochs.
            - Logs scheduler details and the learning rate progression along with the metric input.
            - Resets the scheduler state if early stopping is configured.
            - Returns a list of learning rates computed during the simulation.

        test_exponential():
            - Creates an exponential decay scheduler with a specified decay factor and clamp value.
            - Runs the simulation to record and log the learning rate history.

        test_step():
            - Configures a step decay scheduler with a decay factor and step size.
            - Simulates the learning rate progression and logs the details.

        test_cosine():
            - Sets up a cosine annealing scheduler with a defined minimum learning rate.
            - Simulates the annealing behavior and logs the learning rate changes.

        test_constant():
            - Tests a constant learning rate scheduler to ensure the learning rate remains unchanged.

        test_adaptive():
            - Tests the Adaptive Scheduler (similar to ReduceLROnPlateau) by simulating adaptive
              learning rate reduction.
            - Configures parameters such as decay rate, patience, cooldown, and min_delta.
            - Uses a mixture of metric values (including complex numbers) to simulate adaptive training.
            - Logs both the learning rate adjustments and any early stopping triggers.

        test_pass_instance():
            - Validates behavior when an existing scheduler instance is passed to the factory.
            - Updates the instance with new parameters (like logger and early stopping configuration).
            - Checks if the same object is returned after updating.

        test_missing_arg():
            - Tests the scheduler factory's error handling by attempting to create a scheduler with
              a missing required argument.
            - Expects to catch a TypeError and logs the error appropriately.

    Overall, SchedulerTester provides a comprehensive framework to verify that various scheduler
    configurations and the EarlyStopping mechanism operate correctly under different simulated
    training conditions. This framework aids in debugging and fine-tuning scheduler behaviors before
    integrating them into more complex training pipelines.
    """

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initializes the SchedulerTester instance.

        Args:
            logger (Optional[Logger]): Optional logger for logging messages.
        """
        super().__init__(logger=logger)
        self._log("Scheduler Tester Initialized", log='info')
        self.max_epochs_test	= 20
        self.initial_lr_test	= 0.1

    # --------------------------------------------------------------------------

    def run_all_tests(self):
        """
        Runs all example test methods.
        """
        self._log("=" * 60 + "\n--- Running All Scheduler Tests ---\n" + "=" * 60, log='critical')
        self.test_early_stopping()
        self.test_exponential()
        self.test_adaptive()
        self.test_cosine()
        self.test_step()
        self.test_constant()
        self.test_pass_instance()
        self.test_missing_arg()
        self._log("\n" + "=" * 60 + "\n--- All Scheduler Tests Completed ---\n" + "=" * 60, log='critical')

    # --------------------------------------------------------------------------

    def test_early_stopping(self):
        """
        Tests the EarlyStopping class directly with various metric inputs.
        """
        self._log("\n" + "-" * 50 + "\n--- Testing Early Stopping ---\n" + "-" * 50, log='info')
        es = EarlyStopping(patience=3, min_delta=0.01, logger=self._logger)
        metrics_r = [10.0, 9.9, 9.8, 9.81, 9.82, 9.83, 9.75, 9.8]
        metrics_c = [10.0 + 1j, 9.9 + 0j, 9.8 - 1j, 9.81 + 2j, np.complex128(9.82), 9.83, 9.75 - 0.5j, 9.8 + 0j]

        self._log(">>> With Real Metrics:", log='info')
        es.reset()
        for i, met_r in enumerate(metrics_r):
            stop = es(met_r)
            self._log(f"Epoch {i}: Met={met_r:.3f}, Stop={stop}", lvl=1)
            if stop:
                break

        self._log("\n>>> With Complex Metrics:", log='info')
        es.reset()
        for i, met_c in enumerate(metrics_c):
            stop = es(met_c)
            self._log(f"Epoch {i}: Met={met_c}, Stop={stop}", lvl=1)
            if stop:
                break

        self._log("\n>>> With NaN Metric:", log='info')
        es.reset()
        stop_nan = es(np.nan)
        self._log(f"Stop after NaN: {stop_nan}", lvl=1)

        self._log("\n>>> With Inf Metric:", log='info')
        es.reset()
        stop_inf = es(np.inf)
        self._log(f"Stop after Inf: {stop_inf}", lvl=1)

    # --------------------------------------------------------------------------

    def _run_scheduler_simulation(self, scheduler: Parameters, metrics: Optional[List[Any]] = None):
        """
        Helper to run a simulation for a given scheduler.

        Args:
            scheduler (Parameters): The scheduler instance to simulate.
            metrics (Optional[List[Any]]): List of metric values to simulate training; if None, uses N/A values.
        Returns:
            List[float]: The history of calculated learning rates.
        """
        self._log(f"Scheduler Details:\n{scheduler!r}", log='info')
        lrs = []
        if scheduler.early_stopping:
            scheduler.reset_early_stopping()
        
        if hasattr(scheduler, 'reset'):
            scheduler.reset()  # Reset adaptive state, if available

        metrics_to_use = metrics if metrics is not None else [None] * scheduler.max_epochs

        for epoch in range(scheduler.max_epochs):
            metric = metrics_to_use[epoch % len(metrics_to_use)]
            lr     = scheduler(epoch, _metric=metric)
            lrs.append(lr)
            metric_str = f"{metric}" if metric is not None else "N/A"
            self._log(f"Epoch {epoch}: Metric={metric_str}, LR={lr:.4e}", lvl=1, log='debug')

            if scheduler.early_stopping and scheduler.check_stop(_metric=metric):
                self._log(f"Epoch {epoch}: Early stopping triggered.", lvl=0, color='yellow')
                break

        self._log(f"Final LR: {lrs[-1]:.4e}", log='info')
        self._log(f"LR History (first 5): {[f'{lr:.4e}' for lr in lrs[:5]]}", log='info')
        return lrs

    # --------------------------------------------------------------------------

    def test_exponential(self):
        """
        Tests the Exponential Decay scheduler.
        """
        self._log("\n" + "-" * 50 + "\n--- Testing Exponential Decay ---\n" + "-" * 50, log='info')
        scheduler = choose_scheduler("exponential", self.initial_lr_test, self.max_epochs_test, self._logger,
                                    lr_decay=0.85, lr_clamp=1e-5)
        self._run_scheduler_simulation(scheduler)

    # --------------------------------------------------------------------------

    def test_step(self):
        """
        Tests the Step Decay scheduler.
        """
        self._log("\n" + "-" * 50 + "\n--- Testing Step Decay ---\n" + "-" * 50, log='info')
        scheduler = choose_scheduler("step", self.initial_lr_test, self.max_epochs_test, self._logger,
                                    lr_decay=0.5, step_size=5)
        self._run_scheduler_simulation(scheduler)

    # --------------------------------------------------------------------------

    def test_cosine(self):
        """
        Tests the Cosine Annealing scheduler.
        """
        self._log("\n" + "-" * 50 + "\n--- Testing Cosine Annealing ---\n" + "-" * 50, log='info')
        scheduler = choose_scheduler("cosine", self.initial_lr_test, self.max_epochs_test, self._logger,
                                    min_lr=0.001)
        self._run_scheduler_simulation(scheduler)

    # --------------------------------------------------------------------------

    def test_constant(self):
        """
        Tests the Constant Learning Rate scheduler.
        """
        self._log("\n" + "-" * 50 + "\n--- Testing Constant LR ---\n" + "-" * 50, log='info')
        scheduler = choose_scheduler("constant", 0.05, 10, self._logger)
        self._run_scheduler_simulation(scheduler)

    # --------------------------------------------------------------------------

    def test_adaptive(self):
        """
        Tests the Adaptive Scheduler (ReduceLROnPlateau) with simulated metrics.
        """
        self._log("\n" + "-" * 50 + "\n--- Testing Adaptive Scheduler (ReduceLROnPlateau) ---\n" + "-" * 50, log='info')
        scheduler = choose_scheduler(SchedulerType.ADAPTIVE, self.initial_lr_test, self.max_epochs_test, self._logger,
                                    lr_decay=0.5, patience=3, cooldown=1, min_lr=1e-4, min_delta=0.01,
                                    early_stopping_patience=6, early_stopping_min_delta=0.005,
                                    lr_clamp=5e-5)
        sim_metrics = [5.0 + 0j, 4.8, 4.7 - 1j, 4.71, 4.72 + 0.1j, 4.73, 4.74, 4.5, 4.51, 4.52, 4.53, 4.54, 4.55] * 2
        self._log(">>> Simulating Adaptive Training:", log='info')
        self._run_scheduler_simulation(scheduler, metrics=sim_metrics)

    # --------------------------------------------------------------------------

    def test_pass_instance(self):
        """
        Tests behavior when an existing scheduler instance is passed to the factory.
        """
        self._log("\n" + "-" * 50 + "\n--- Testing Passing Instance ---\n" + "-" * 50, log='info')
        original_scheduler = StepDecayScheduler(0.2, 15, 0.5, 4, logger=None, lr_clamp=1e-3)
        self._log(f"Original Instance:\n{original_scheduler!r}", log='info')

        chosen_scheduler = choose_scheduler(original_scheduler, 0.999, 999,
                                            logger=self._logger,
                                            early_stopping_patience=3,
                                            lr_clamp=5e-4)
        self._log(f"\nReturned Instance (updated):\n{chosen_scheduler!r}", log='info')
        self._log(f"Is same object? {chosen_scheduler is original_scheduler}", log='info')

    # --------------------------------------------------------------------------

    def test_missing_arg(self):
        """
        Tests the scheduler factory's behavior when a required argument is missing.
        """
        self._log("\n" + "-" * 50 + "\n--- Testing Missing Argument ---\n" + "-" * 50, log='info')
        try:
            scheduler_bad = choose_scheduler("step", self.initial_lr_test, self.max_epochs_test, self._logger,
                                             lr_decay=0.7)
            self._log("Error: Expected TypeError but scheduler was created.", log='error')
        except TypeError as e:
            self._log(f"Caught expected TypeError:\n  {e}", log='info', color='green')
        except Exception as e:
            self._log(f"Caught unexpected error: {type(e).__name__}: {e}", log='error', color='red')

# ------------------------------------------------------------------------------