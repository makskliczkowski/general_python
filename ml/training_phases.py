"""
Learning phase framework for Neural Quantum State training.

This module implements a multi-phase training system for NQS, allowing:
- Phase transitions with configurable parameters
- Phase-specific callbacks and hooks
- Adaptive learning rates per phase
- Regularization scheduling per phase
- Progress tracking and reporting

Learning phases represent different stages of optimization:
1. Pre-training: Initialize network with simple loss, high learning rate
2. Main Optimization: Full Hamiltonian, adaptive learning rate
3. Refinement: Fine-tune observables, low learning rate, high regularization

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

"""
Multi-stage Training Phase Scheduler.

This module manages dynamic hyperparameters (learning rate, regularization)
across defined training phases (Pre-training, Main, Refinement).

----------------------------------------------------------
file        : general_python/ml/training_phases.py
author      : Maksymilian Kliczkowski
----------------------------------------------------------
"""

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
    """Configuration for a specific duration of training."""
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
                lr_kwargs       =   {'step_size': 25, 'lr_decay': 0.5},
                reg             =   5e-2
            ),
            LearningPhase(
                name            =   "main", 
                epochs          =   300, 
                phase_type      =   PhaseType.MAIN, 
                lr              =   2e-2, 
                lr_schedule     =   "adaptive", 
                lr_kwargs       =   {'patience': 15, 'min_delta': 1e-4},
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
    Manages the transition between phases and instantiates the 
    appropriate low-level Scheduler from schedulers.py for each phase.
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
    """Returns (lr_scheduler, reg_scheduler) tuple."""
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
