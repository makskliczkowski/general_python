from __future__ import annotations
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Iterable, Optional, Dict, List, Tuple, Any
from enum import Enum
import time
import logging

################################################################################
# High-precision, monotonic clock in nanoseconds
_now_ns: Callable[[], int] = time.perf_counter_ns

class TimerState(Enum):
    RUNNING     = "running"
    PAUSED      = "paused"
    STOPPED     = "stopped"

@dataclass(slots=True)
class Timer:
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
    name                    : Optional[str]                     = None
    logger                  : Optional[logging.Logger]          = None
    logger_args             : Optional[Dict[str, Any]]          = None
    verbose                 : bool                              = False
    unit                    : str                               = "auto"
    deadline_s              : Optional[float]                   = None
    synchronizer            : Optional[Callable[[Any], None]]   = None

    # internal state
    _start_ns               : Optional[int]                     = field(default=None, init=False)
    _paused                 : bool                              = field(default=False, init=False)
    _stopped                : bool                              = field(default=False, init=False)
    _elapsed_ns             : int                               = field(default=0, init=False)
    
    _laps_ns                : List[int]                         = field(default_factory=list, init=False)
    _laps_names             : List[str]                         = field(default_factory=list, init=False)
    _last_lap_anchor_ns     : Optional[int]                     = field(default=None, init=False)
    _marks_ns               : Dict[str, int]                    = field(default_factory=dict, init=False)

    ################################################################################

    def start(self) -> "Timer":
        """Start (or resume) the timer; no-op if already running."""
        if self._start_ns is None:
            now             = _now_ns()
            self._start_ns  = now
            if self._last_lap_anchor_ns is None:
                self._last_lap_anchor_ns = now
        return self

    def pause(self) -> "Timer":
        """Pause the timer, accumulating elapsed time."""
        if self._start_ns is not None:
            now                 = _now_ns()
            self._elapsed_ns   += now - self._start_ns
            self._start_ns      = None
            self._paused        = True
        return self

    def resume(self) -> "Timer":
        """Resume after pause."""
        if self._paused:
            self._paused    = False
            self._start_ns  = _now_ns()
        return self

    def stop(self) -> float:
        """Stop and return elapsed time in seconds."""
        self.pause()
        self._stopped = True
        return self.elapsed_s()

    def reset(self) -> "Timer":
        """Clear state (elapsed, laps, marks) and stop."""
        self._start_ns              = None
        self._elapsed_ns            = 0
        self._last_lap_anchor_ns    = None
        self._laps_ns.clear()
        self._marks_ns.clear()
        return self

    ################################################################################

    def lap(self, name: Optional[str] = None) -> float:
        """
        Record a lap (time since last lap or start) and return lap in seconds.
        """
        now         = _now_ns()
        anchor      = self._last_lap_anchor_ns if self._last_lap_anchor_ns is not None else now
        lap_ns      = now - anchor
        self._laps_ns.append(lap_ns)
        self._last_lap_anchor_ns = now
        if name:
            self._laps_names.append(name)
            self._marks_ns[name] = now
        else:
            self._laps_names.append(f"lap{len(self._laps_ns)}")
        return lap_ns / 1e9

    ################################################################################
    
    def mark(self, name: Optional[str] = None) -> None:
        """
        Create/update a named absolute anchor at current time. Later use since('name').
        """
        self._marks_ns[name] = _now_ns()

    def since(self, name: Optional[str] = None, ts: Optional[int] = None) -> float:
        """
        Seconds elapsed since the named mark. Raises KeyError if mark not set.
        """
        if name is not None:
            if name not in self._marks_ns:
                raise KeyError(f"Mark '{name}' not found")
            return (_now_ns() - self._marks_ns[name]) / 1e9
        elif ts is not None:
            return (_now_ns() - ts) / 1e9
        raise ValueError("Either 'name' or 'ts' must be provided")

    ################################################################################
    #! queries 
    ################################################################################
    
    def elapsed_ns(self) -> int:
        """Total elapsed nanoseconds (includes current running span)."""
        if self._start_ns is None:
            return self._elapsed_ns
        return self._elapsed_ns + (_now_ns() - self._start_ns)

    def elapsed_ms(self) -> float:
        """Elapsed milliseconds (float)."""
        return self.elapsed_ns() / 1e6

    def elapsed_us(self) -> float:
        """Elapsed microseconds (float)."""
        return self.elapsed_ns() / 1e3

    def elapsed_s(self) -> float:
        """Elapsed seconds (float)."""
        return self.elapsed_ns() / 1e9

    ################################################################################

    def laps(self) -> Tuple[List[float], List[str]]:
        """Recorded laps (seconds) and their names."""
        return [ns / 1e9 for ns in self._laps_ns], list(self._laps_names)

    ################################################################################
    
    def remaining_s(self, buffer_s: float = 0.0) -> Optional[float]:
        """
        If deadline_s is set, return remaining seconds (can be negative). Otherwise None.
        """
        if self.deadline_s is None:
            return None
        return self.deadline_s - buffer_s - self.elapsed_s()

    ################################################################################
    
    def overtime(self, buffer_s: float = 0.0) -> bool:
        """
        True if elapsed >= deadline_s - buffer_s; False if no deadline is set.
        """
        rem = self.remaining_s(buffer_s)
        return (rem is not None) and (rem <= 0.0)

    @property
    def state(self) -> TimerState:
        if self._start_ns is not None:
            return TimerState.RUNNING
        if self._paused:
            return TimerState.PAUSED
        return TimerState.STOPPED

    ################################################################################
    #! formatting & reporting
    ################################################################################

    def _format_unit(self, seconds: float) -> Tuple[float, str]:
        if self.unit == "auto":
            if seconds >= 1.0:
                return (seconds, "s")
            ms = seconds * 1e3
            if ms >= 1.0:
                return (ms, "ms")
            us = seconds * 1e6
            if us >= 1.0:
                return (us, "us")
            return (seconds * 1e9, "ns")
        elif self.unit == "s":
            return (seconds, "s")
        elif self.unit == "ms":
            return (seconds * 1e3, "ms")
        elif self.unit == "us":
            return (seconds * 1e6, "us")
        elif self.unit == "ns":
            return (seconds * 1e9, "ns")
        else:
            raise ValueError("unit must be one of {'auto','s','ms','us','ns'}")

    def format_elapsed(self) -> str:
        v, u = self._format_unit(self.elapsed_s())
        return f"{v:.6f} {u}"

    ################################################################################

    def report(self, include_laps: bool = True) -> str:
        parts           = [f"{self.name or 'Timer'}: {self.format_elapsed()}"]
        if include_laps and self._laps_ns:
            laps_sec, laps_names    = self.laps()
            laps_fmt                = ", ".join(f"{n}={t:.6f}s" for n, t in zip(laps_names, laps_sec))
            parts.append(f"[laps: {laps_fmt}]")
        if self.deadline_s is not None:
            rem                     = self.remaining_s()
            parts.append(f"[deadline rem: {rem:.3f}s]" if rem is not None else "[deadline rem: n/a]")
        return " ".join(parts)

    def _emit(self, msg: str, logger_args: Dict[str, Any] = None) -> None:
        if self.logger is not None and self.verbose:
            self.logger.info(msg, **(logger_args or {}))
        elif self.verbose:
            print(msg)

    # -------- context manager --------

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.pause()
        self._emit(self.report(include_laps=True), logger_args=self.logger_args)

    # -------- decorator (re-entrant, thread-safe) --------

    @classmethod
    def decorator(cls,
                name            : Optional[str] = None,
                logger          : Optional[logging.Logger] = None,
                verbose         : bool = False,
                unit            : str = "auto",
                deadline_s      : Optional[float] = None,
                synchronizer    : Optional[Callable[[Any], None]] = None):
        """
        Decorator for timing a function.

        Usage:
            @Timer.decorator("block", verbose=True)
            def fn(...): ...

        Parameters:
            - name: 
                The name of the timer (default: function name)
            - logger: 
                Optional logger for logging (default: None)
            - verbose: 
                If True, print timing info (default: False)
            - unit: 
                Time unit for reporting (default: "auto")
            - deadline_s: 
                Optional deadline in seconds (default: None)
            - synchronizer: 
                Optional synchronizer function (default: None)
        """
        def deco(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                t = cls(name or func.__name__, logger=logger, verbose=verbose,
                        unit=unit, deadline_s=deadline_s, synchronizer=synchronizer)
                t.start()
                try:
                    res = func(*args, **kwargs)
                    # Optional synchronization for lazy backends (e.g., JAX)
                    if t.synchronizer is not None:
                        try:
                            t.synchronizer(res)
                        except ReferenceError as e:
                            if logger is not None:
                                logger.warning(f"Synchronizer skipped: underlying object vanished ({e})")
                        except Exception as e:
                            # Try tuple unpack
                            if isinstance(res, tuple):
                                for x in res:
                                    try:
                                        t.synchronizer(x)
                                    except Exception:
                                        pass
                            
                    return res
                finally:
                    t.pause()
                    t._emit(t.report(include_laps=True), logger_args=t.logger_args)
            return wrapper
        return deco

################################################################################
# Utility: function timing with optional synchronizer (JAX, etc.)
def timeit(fn: Callable, *args, synchronizer: Optional[Callable[[Any], None]] = None, **kwargs) -> Tuple[Any, float]:
    """
    Measures wall time for a callable. If `synchronizer` is provided, it will be called
    on the result (and on tuple elements) to force completion.
    Returns (result, elapsed_seconds).
    """
    t0 = _now_ns()
    res = fn(*args, **kwargs)
    if synchronizer is not None:
        try:
            synchronizer(res)
        except Exception:
            if isinstance(res, tuple):
                for x in res:
                    try:
                        synchronizer(x)
                    except Exception:
                        pass
    dt = (_now_ns() - t0) / 1e9
    return res, dt

################################################################################