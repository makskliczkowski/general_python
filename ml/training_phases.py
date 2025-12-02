"""
Learning phase framework for Neural Quantum State training.

This module implements a multi-phase training system for NQS, allowing:
- Phase transitions with configurable parameters
- Phase-specific callbacks and hooks
- Adaptive learning rates per phase
- Regularization scheduling per phase
- Progress tracking and reporting

Learning phases represent different stages of optimization:

1. **Pre-training**: Initialize network with simple loss, high learning rate
2. **Main Optimization**: Full Hamiltonian, adaptive learning rate
3. **Refinement**: Fine-tune observables, low learning rate, high regularization

Quick Start
-----------
**Using Presets:**

>>> from QES.general_python.ml.training_phases import create_phase_schedulers
>>> lr_sched, reg_sched = create_phase_schedulers('default')
>>> # Pass to NQSTrainer: phases=(lr_sched, reg_sched)

**Creating Custom Phases:**

>>> from QES.general_python.ml.training_phases import LearningPhase, PhaseType, PhaseScheduler
>>> 
>>> my_phases = [
...     LearningPhase(
...         name="warmup", epochs=50,
...         lr=0.1, lr_schedule="exponential", lr_kwargs={'lr_decay': 0.05},
...         reg=0.01
...     ),
...     LearningPhase(
...         name="main", epochs=300,
...         lr=0.02, lr_schedule="adaptive", lr_kwargs={'patience': 20, 'lr_decay': 0.5},
...         reg=0.001
...     ),
... ]
>>> lr_sched = PhaseScheduler(my_phases, param_type='lr')
>>> reg_sched = PhaseScheduler(my_phases, param_type='reg')

Available Scheduler Types
-------------------------
- ``'constant'``: Fixed value
- ``'exponential'``: Exponential decay: lr * exp(-decay * epoch)
- ``'step'``: Step decay: lr * gamma^floor(epoch/step_size)
- ``'cosine'``: Cosine annealing to min_lr
- ``'linear'``: Linear decay to min_lr
- ``'adaptive'``: ReduceLROnPlateau (requires loss)

Available Presets
-----------------
- ``'default'``: 3-phase (pre_training: 50, main: 200, refinement: 100)
- ``'kitaev'``: Specialized for frustrated spin systems (pre: 100, main: 300, fine: 150)

----------------------------------------
File        : NQS/src/learning_phases.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Date        : November 1, 2025
----------------------------------------
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum, auto
import numpy as np

try:
    from .schedulers import choose_scheduler, Parameters, SchedulerType
except ImportError as e:
    raise ImportError("Failed to import schedulers module. Ensure QES package is correctly installed.") from e

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Union, Tuple
from enum import Enum, auto
import numpy as np

# ----------------------------------------------------------------------
# Core Data Structures
# ----------------------------------------------------------------------

class PhaseType(Enum):
    PRE_TRAINING        = auto()
    MAIN                = auto()
    REFINEMENT          = auto()
    CUSTOM              = auto()

@dataclass
class LearningPhase:
    """
    Configuration for a specific training phase.
    
    Each phase defines learning rate and regularization schedules that are
    active for a specific number of epochs. Phases are processed sequentially
    by the PhaseScheduler.
    
    Attributes
    ----------
    name : str
        Human-readable phase identifier (e.g., 'warmup', 'main', 'fine').
        
    epochs : int
        Number of epochs this phase lasts.
        
    phase_type : PhaseType
        Semantic type (PRE_TRAINING, MAIN, REFINEMENT, CUSTOM).
        
    lr : float
        Initial learning rate for this phase.
        
    lr_schedule : str
        Scheduler type for LR. Options:
        - 'constant': Fixed lr throughout phase
        - 'exponential': lr * exp(-lr_decay * local_epoch)
        - 'step': lr * lr_decay^floor(local_epoch/step_size)
        - 'cosine': Cosine annealing from lr to min_lr
        - 'linear': Linear decay from lr to min_lr
        - 'adaptive': ReduceLROnPlateau (requires loss)
        
    lr_kwargs : Dict[str, Any]
        Extra arguments for the LR scheduler. Common keys:
        - 'lr_decay': Decay rate (exponential, step, adaptive)
        - 'step_size': Steps between decays (step scheduler)
        - 'min_lr': Minimum LR (cosine, linear, adaptive)
        - 'patience': Epochs before reduction (adaptive)
        - 'min_delta': Minimum improvement threshold (adaptive)
        
    reg : float
        Initial regularization (diagonal shift) for this phase.
        
    reg_schedule : str
        Scheduler type for regularization. Same options as lr_schedule.
        
    reg_kwargs : Dict[str, Any]
        Extra arguments for the regularization scheduler.
        
    loss_type : str
        Loss function type (default: 'energy').
        
    beta_penalty : float
        Penalty coefficient for excited state targeting.
        
    on_phase_start : Callable, optional
        Callback executed when phase begins.
        
    on_phase_end : Callable, optional
        Callback executed when phase ends.
        
    Examples
    --------
    >>> # Exponential decay warmup
    >>> warmup = LearningPhase(
    ...     name='warmup', epochs=50,
    ...     lr=0.1, lr_schedule='exponential', lr_kwargs={'lr_decay': 0.05},
    ...     reg=0.01
    ... )
    >>> 
    >>> # Adaptive main phase (ReduceLROnPlateau)
    >>> main = LearningPhase(
    ...     name='main', epochs=300,
    ...     lr=0.02, lr_schedule='adaptive',
    ...     lr_kwargs={'patience': 20, 'lr_decay': 0.5, 'min_lr': 1e-4},
    ...     reg=0.001
    ... )
    >>> 
    >>> # Cosine annealing refinement
    >>> refine = LearningPhase(
    ...     name='fine', epochs=100,
    ...     lr=0.01, lr_schedule='cosine', lr_kwargs={'min_lr': 1e-5},
    ...     reg=0.005
    ... )
    """
    name                : str                   = "phase"
    epochs              : int                   = 100
    phase_type          : PhaseType             = PhaseType.MAIN
    
    # LR Configuration
    lr                  : float                 = 1e-2 # Initial LR for this phase
    lr_schedule         : str                   = "constant" 
    # Extra args passed to scheduler factory (e.g., {'lr_decay': 0.9, 'step_size': 10})
    lr_kwargs           : Dict[str, Any]        = field(default_factory=dict) 
    
    # Regularization Configuration  
    reg                 : float                 = 1e-3 # Initial Reg for this phase
    reg_schedule        : str                   = "constant"
    # Extra args passed to scheduler factory
    reg_kwargs          : Dict[str, Any]        = field(default_factory=dict)
    
    # Physics/Loss specifics
    loss_type           : str                   = "energy"
    beta_penalty        : float                 = 0.0
    
    # Callbacks
    on_phase_start      : Optional[Callable]    = None
    on_phase_end        : Optional[Callable]    = None

# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------

def _get_presets() -> Dict[str, List[LearningPhase]]:
    return {
        "default": [
            # 1. Pre-training: Initialize network with simple loss, high learning rate
            LearningPhase(
                name            =   "pre_training", 
                epochs          =   50, 
                phase_type      =   PhaseType.PRE_TRAINING, 
                lr              =   1e-1, 
                lr_schedule     =   "exponential", 
                lr_kwargs       =   {'lr_decay': 1e-2}, # decay rate
                reg             =   5e-2, 
                reg_schedule    =   "constant"
            ),
            # 2. Main Optimization: Full Hamiltonian, adaptive learning rate
            LearningPhase(
                name            =   "main", 
                epochs          =   200, 
                phase_type      =   PhaseType.MAIN, 
                lr              =   3e-2, 
                lr_schedule     =   "adaptive", 
                lr_kwargs       =   {'patience': 20, 'lr_decay': 0.5},
                reg             =   1e-3, 
                reg_schedule    =   "constant"
            ),
            # 3. Refinement: Fine-tune observables with low learning rate
            LearningPhase(
                name            =   "refinement", 
                epochs          =   100, 
                phase_type      =   PhaseType.REFINEMENT, 
                lr              =   1e-2, 
                lr_schedule     =   "cosine", 
                lr_kwargs       =   {'min_lr': 1e-5},
                reg             =   5e-3, 
                reg_schedule    =   "constant"
            )
        ],
        # Example of a specialized preset for frustrated systems
        "kitaev": [
            LearningPhase(
                name            =   "pre", 
                lr              =   5e-2, 
                lr_schedule     =   "step", 
                lr_kwargs       =   {'step_size': 35, 'lr_decay': 0.5},
                reg             =   5e-2
            ),
            LearningPhase(
                name            =   "main", 
                epochs          =   300, 
                phase_type      =   PhaseType.MAIN, 
                lr              =   3e-2, 
                lr_schedule     =   "adaptive", 
                lr_kwargs       =   {'patience': 100, 'min_delta': 1e-4},
                reg             =   1e-3
            ),
            # In refinement, we might want to increase regularization (annealing)
            LearningPhase(
                name            =   "fine", 
                epochs          =   150, 
                phase_type      =   PhaseType.REFINEMENT, 
                lr              =   5e-3, 
                lr_schedule     =   "cosine",
                reg             =   1e-3, 
                reg_schedule    =   "linear", 
                reg_kwargs      =   {}
            )
        ]
    }

# ----------------------------------------------------------------------
# The Smart Scheduler (The Orchestrator)
# ----------------------------------------------------------------------

class PhaseScheduler:
    """
    Manages transitions between training phases.
    
    The PhaseScheduler orchestrates multi-phase training by:
    1. Tracking the current phase based on global epoch count
    2. Instantiating appropriate low-level schedulers for each phase
    3. Firing callbacks on phase transitions
    4. Returning scheduled values via __call__
    
    Parameters
    ----------
    phases : List[LearningPhase]
        Ordered list of training phases to execute.
        
    param_type : str, default='lr'
        Which parameter to schedule ('lr' or 'reg').
        
    logger : Logger, optional
        Logger for phase transition messages.
        
    Attributes
    ----------
    current_phase : LearningPhase
        Currently active phase.
        
    history : List[float]
        All scheduled values returned.
        
    Examples
    --------
    >>> from QES.general_python.ml.training_phases import LearningPhase, PhaseScheduler
    >>> 
    >>> phases = [
    ...     LearningPhase(name='warmup', epochs=50, lr=0.1, lr_schedule='exponential',
    ...                   lr_kwargs={'lr_decay': 0.05}),
    ...     LearningPhase(name='main', epochs=200, lr=0.02, lr_schedule='constant'),
    ... ]
    >>> 
    >>> lr_scheduler = PhaseScheduler(phases, param_type='lr')
    >>> reg_scheduler = PhaseScheduler(phases, param_type='reg')
    >>> 
    >>> # Use in training loop
    >>> for epoch in range(250):
    ...     lr = lr_scheduler(epoch, loss=current_loss)  # Auto phase transition
    ...     reg = reg_scheduler(epoch, loss=current_loss)
    """
    def __init__(self, phases: List[LearningPhase], param_type: str = 'lr', logger=None):
        self.phases                                 = phases
        self.param_type                             = param_type # 'lr' or 'reg'
        self.logger                                 = logger
        self.history                                = []
        
        # State
        self._current_phase_idx                     = 0
        self._epochs_completed_in_prev_phases       = 0
        
        # The active engine (instance of Parameters from schedulers.py)
        self._active_engine: Optional[Parameters]   = None
        self._init_current_phase_engine()
    
    @property
    def current_phase(self) -> LearningPhase:
        if self._current_phase_idx >= len(self.phases):
            return self.phases[-1]
        return self.phases[self._current_phase_idx]

    # ---------

    def _init_current_phase_engine(self):
        """Uses the Factory to create the scheduler for the current phase."""
        phase = self.current_phase
        
        if self.param_type == 'lr' or self.param_type == 'dt':
            init_val    = phase.lr
            sched_type  = phase.lr_schedule
            kwargs      = phase.lr_kwargs
        else:
            init_val    = phase.reg
            sched_type  = phase.reg_schedule
            kwargs      = phase.reg_kwargs
            
        # Logging
        if self.logger:
            self.logger.info(f"Phase '{phase.name}': Init {self.param_type.upper()} scheduler '{sched_type}, val={init_val:.2e}, epochs={phase.epochs}", color='cyan')

        # Instantiate via your factory
        self._active_engine     = choose_scheduler(
                                    scheduler_type  =   sched_type,
                                    initial_lr      =   init_val,    # Factory expects 'initial_lr', works for Reg too
                                    max_epochs      =   phase.epochs,
                                    logger          =   self.logger,
                                    **kwargs
                                )

    def _update_phase_state(self, global_epoch: int):
        """Check if we need to switch phases."""
        local_epoch      = global_epoch - self._epochs_completed_in_prev_phases
        
        # Check transition condition
        while local_epoch >= self.current_phase.epochs and self._current_phase_idx < len(self.phases) - 1:
            
            # Fire end callback
            if self.current_phase.on_phase_end: 
                self.current_phase.on_phase_end()
            
            # Advance
            self._epochs_completed_in_prev_phases   += self.current_phase.epochs
            self._current_phase_idx                 += 1
            local_epoch                             = global_epoch - self._epochs_completed_in_prev_phases
            
            # Fire start callback
            if self.current_phase.on_phase_start: 
                self.current_phase.on_phase_start()
            
            # Re-Initialize the Engine for the new phase
            self._init_current_phase_engine()

        return max(0, local_epoch)

    def __call__(self, global_epoch: int, loss: float = None) -> float:
        """
        Delegates calculation to the specific scheduler instance.
        """
        # Delegate math to the active engine (schedulers.py)
        local_epoch = self._update_phase_state(global_epoch)
        val         = self._active_engine(local_epoch, _metric=loss)
        self.history.append(val)
        return val

# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_phase_schedulers(preset: str = 'default', logger=None):
    """
    Factory function to create LR and Reg schedulers from a preset.
    
    Parameters
    ----------
    preset : str, default='default'
        Preset name. Available:
        - 'default': 3-phase training (350 total epochs)
        - 'kitaev': Specialized for frustrated systems (550 total epochs)
        
    logger : Logger, optional
        Logger for scheduler messages.
        
    Returns
    -------
    Tuple[PhaseScheduler, PhaseScheduler]
        (lr_scheduler, reg_scheduler) tuple.
        
    Raises
    ------
    ValueError
        If preset name is not recognized.
        
    Examples
    --------
    >>> lr_sched, reg_sched = create_phase_schedulers('default')
    >>> 
    >>> # Pass to NQSTrainer
    >>> trainer = NQSTrainer(nqs, phases=(lr_sched, reg_sched))
    >>> 
    >>> # Or use preset string directly
    >>> trainer = NQSTrainer(nqs, phases='default')  # Equivalent
    """
    presets = _get_presets()
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")
    
    phases      = presets[preset]
    lr_sched    = PhaseScheduler(phases, 'lr', logger)
    reg_sched   = PhaseScheduler(phases, 'reg', logger)
    return lr_sched, reg_sched

PRESETS = _get_presets()

# -----------------------------------
#! End of file
# -----------------------------------
