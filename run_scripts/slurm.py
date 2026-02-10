'''


'''

import os
import numpy as np
import time
from typing     import Dict, List, Optional, Union, TYPE_CHECKING
from pathlib    import Path

if TYPE_CHECKING:
    from ..common.flog import Logger

#########################################################

try:
    import psutil
    import argparse
    def _cpu_count_psutil(logical: bool = True): return psutil.cpu_count(logical)
except ImportError:
    print("Psutil is not available for CPU handling...")
    def _cpu_count_psutil(logical: bool = True): return 1

import subprocess
from typing         import Callable
from contextlib     import contextmanager
from dataclasses    import dataclass

try:
    from ..algebra.ran_wrapper import set_global_seed
except ImportError:
    raise ImportError("Failed to import set_global_seed from algebra.ran_wrapper")

#########################################################
#! Validators
#########################################################

def calculate_optimal_workers(params, available_memory: int, memory_per_worker: int, max_cores: Optional[int] = None, logger: Optional['Logger'] = None):
    """
    Calculate optimal number of workers based on available resources
    Parameters
    ----------
    params : list
        List of parameter values to process (e.g. site labels, parameter identifiers).
    available_memory : int
        Total available memory in bytes.
    memory_per_worker : int
        Estimated memory required per worker in bytes.
    max_cores : int, optional
        Maximum number of CPU cores to use. If None, use all available cores.
    logger : Logger, optional
        Logger instance for logging information.
    Returns
    -------
    int
        Optimal number of workers to use.
    """
    
    if os.getenv("PYSINGLETHREAD", "0") == "1":
        if logger:
            logger.info("PYSINGLETHREAD is set, using 1 worker", lvl=2)
        return 1
    
    # Get system info
    cpu_count = psutil.cpu_count(logical=False) or 1
    if max_cores:
        cpu_count = min(cpu_count, max_cores)
    
    # Calculate based on memory
    memory_limited_workers  = max(1, int(available_memory / memory_per_worker))

    # Calculate based on work (don't create more workers than parameter values)
    work_limited_workers    = len(params)
    
    # Calculate based on CPU cores (leave some for system)
    cpu_limited_workers     = max(1, cpu_count - 1)
    
    # Take the minimum of all constraints
    optimal_workers         = min(memory_limited_workers, work_limited_workers, cpu_limited_workers)
    
    if logger:
        logger.info(f"System info:")
        logger.info(f"  CPU count: {cpu_count}")
        logger.info(f"  Available memory: {available_memory} bytes")
        logger.info(f"  Memory per worker: {memory_per_worker} bytes")
        logger.info(f"  Workers based on memory: {memory_limited_workers}")
        logger.info(f"  Workers based on work: {work_limited_workers}")
        logger.info(f"  Workers based on CPU cores: {cpu_limited_workers}")
        logger.info(f"Optimal number of workers: {optimal_workers}")
            
    return optimal_workers

def calculate_realisations_per_parameter(parameters     : List[str], 
                                        n_realisations  : str,
                                        logger          : Optional['Logger'] = None) -> Dict[str, int]:
    """
    Calculate the number of realizations per parameter.

    Parameters
    ----------
    parameters : list of str
        List of parameter names (e.g. site labels, parameter identifiers).
    n_realisations : str
        String specifying realizations:
        - Single int: "10"
        - Comma-separated list: "10,20,30"
        - Dictionary-like: "p1:5;p2:10"

    Returns
    -------
    dict
        Mapping from parameter -> number of realizations.
        
    Example
    --------
    >>> calculate_realisations_per_parameter(['p1', 'p2'], '10')
    {'p1': 10, 'p2': 10}
    >>> calculate_realisations_per_parameter(['p1', 'p2'], '10,20')
    {'p1': 10, 'p2': 20}
    >>> calculate_realisations_per_parameter(['p1', 'p2'], 'p1:5;p2:10')
    {'p1': 5, 'p2': 10}
    """
    if not parameters:
        raise ValueError("Parameters list cannot be empty.")

    n_realisations = n_realisations.strip()

    # Case 3: Dictionary-like string
    if ":" in n_realisations and ";" in n_realisations:
        result = {}
        for pair in n_realisations.split(";"):
            if not pair.strip():
                continue
            try:
                key, val = pair.split(":")
                result[key.strip()] = int(val.strip())
            except ValueError:
                raise ValueError(f"Invalid dictionary entry: '{pair}'. Expected 'key:value'.")
        
        # Validate all parameters are covered
        missing = [p for p in parameters if p not in result]
        if missing:
            result.update({p: 0 for p in missing})
        if logger:
            logger.info(f"Realizations per parameter (dict): {result}")
        return result

    # Case 2: Comma-separated list of integers
    if "," in n_realisations:
        parts = [x.strip() for x in n_realisations.split(",")]
        if len(parts) != len(parameters):
            # append with zeros
            if len(parts) < len(parameters):
                parts += ['0'] * (len(parameters) - len(parts))
            else:
                parts = parts[:len(parameters)]
        try:
            if logger:
                logger.info(f"Realizations per parameter (list): {parts}")
            return {param: int(val) for param, val in zip(parameters, parts)}
        except ValueError:
            raise ValueError("All realizations must be integers in the list.")

    # Case 1: Single integer string
    if n_realisations.isdigit():
        val = int(n_realisations)
        return {param: val for param in parameters}

    raise ValueError("Invalid format. Use a single integer ('10'), "
                    "a comma-separated list ('10,20'), "
                    "or a dictionary string ('p1:10;p2:20').")

def validate_realisations_against_time(
    n_realisations      : Dict[str, int],
    t_realisations      : Optional[Union[str, List[str], Dict[str, str]]] = None,
    t_total             : Optional[str] = None,
    parameters_reals    : Optional[List[str]] = None,
    parameters_multiple : Optional[Dict[str, int]] = None,
    *,
    logger              : Optional['Logger'] = None) -> Dict[str, int]:
    """
    Validate and adjust number of realizations per parameter against total available time.
    If total required time > t_total, scale down realizations proportionally.
    Supports t_realisations in multiple formats: single string, comma-separated string, or dictionary string.
    
    Parameters
    ----------
    n_realisations : dict
        Mapping from parameter -> number of realizations.
    t_realisations : str, list of str, or dict, optional
        Time per realization, formats:
        - Single string: "0:10:00" (hh:mm:ss)
        - Comma-separated: "0:10:00,0:20:00"
        - Dictionary-like: "p1:0:10:00;p2:0:20:00"
        - List of strings: ["0:10:00", "0:20:00"]
    t_total : str, optional
        Total allowed time, format: "h:m:s"
    parameters_reals : list of str, optional
        List of parameters to consider from n_realisations. If None, use all keys.
    parameters_multiple : dict, optional
        Per-parameter multiplier for inner calculations (e.g. if each realization involves multiple runs).
    logger : Logger, optional
    """

    if not n_realisations:
        raise ValueError("n_realisations must be provided and non-empty.")

    if t_realisations is None or len(t_realisations) == 0 or t_total is None:
        if logger:
            logger.warning("[validate] No time limits provided — keeping realizations unchanged.", lvl=2)
        return n_realisations

    def to_seconds(tim: str) -> int:
        if isinstance(tim, (int, float)):
            return int(tim) # assume already in seconds
        
        parts = tim.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid time format '{tim}', expected h:m:s.")
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s

    #! Convert total allowed time to seconds
    t_total_sec = to_seconds(t_total)
    parameters  = parameters_reals if parameters_reals else list(n_realisations.keys())

    # Flexible parsing for t_realisations
    t_per_param : Dict[str, int] = {}

    # Dictionary-like format: "p1:0:10:00;p2:0:20:00"
    if isinstance(t_realisations, str) and ":" in t_realisations and ";" in t_realisations:
        for pair in t_realisations.split(";"):
            if not pair.strip():
                continue
            if ":" not in pair:
                raise ValueError(f"Invalid time entry '{pair}', expected p:hh:mm:ss")
            key, val            = pair.split(":", 1)
            key, val            = key.strip(), val.strip()
            t_per_param[key]    = to_seconds(val)
        missing = [p for p in parameters if p not in t_per_param]
        if missing:
            # Does not make sense to fill with zeros here (because we will divide by zero later)
            raise ValueError(f"Missing time values for parameters: {missing}")
    # Comma-separated list: "0:10:00,0:20:00"
    elif isinstance(t_realisations, str) and "," in t_realisations:
        parts = [x.strip() for x in t_realisations.split(",")]
        if len(parts) != len(parameters):
            raise ValueError(f"Expected {len(parameters)} time values, got {len(parts)}.")
        t_per_param = {p: to_seconds(t) for p, t in zip(parameters, parts)}
    # Single string: "0:10:00"
    elif isinstance(t_realisations, str):
        t_sec       = to_seconds(t_realisations)
        t_per_param = {p: t_sec for p in parameters}
    # Already a dict or list
    elif isinstance(t_realisations, dict):
        t_per_param = {p: to_seconds(t_realisations[p]) for p in parameters}
    elif isinstance(t_realisations, list):
        if len(t_realisations) != len(parameters):
            raise ValueError("Number of t_realisations entries must match parameters.")
        t_per_param = {p: to_seconds(t) for p, t in zip(parameters, t_realisations)}
    else:
        raise ValueError("Invalid t_realisations format.")

    # Account for per-parameter multiplier (inner calculations)
    if isinstance(parameters_multiple, int):
        parameters_multiple = {p: parameters_multiple for p in parameters}
    elif isinstance(parameters_multiple, list):
        parameters_multiple = {p: parameters_multiple[i] if i < len(parameters_multiple) else 1 for i, p in enumerate(parameters)}
    multiplier      = parameters_multiple or {}
    multiplier      = {p: multiplier.get(p, 1) for p in parameters}
    total_required  = sum(n_realisations[p] * t_per_param[p] * multiplier[p] for p in parameters)

    if logger:
        logger.info(f"[validate] Total available time: {t_total_sec // 60} min", lvl=3)
        logger.info(f"[validate] Total required time:  {total_required // 60} min", lvl=3)

    if total_required <= t_total_sec:
        if logger:
            logger.info("[validate] Realizations fit within total time — no adjustment needed.", lvl=4)
        return n_realisations

    # Scale down realizations
    scaling = t_total_sec / total_required
    if logger:
        logger.warning(f"[validate] Scaling realizations by factor {scaling:.3f} to fit within time.", lvl=4)

    adjusted = {}
    for p in parameters:
        original    = n_realisations[p]
        new_count   = max(int(original * scaling), 1)
        adjusted[p] = new_count
        if logger:
            logger.warning(f"[validate] {p}: {original} * {multiplier[p]} -> {new_count}", lvl=4)

    if parameters_reals:
        for p in n_realisations:
            if p not in adjusted:
                adjusted[p] = n_realisations[p]

    return adjusted

def initialize_random_seed(args_seed: Optional[int], logger: Optional['Logger'] = None):
    """
    Initialize a random seed for reproducibility.

    Parameters
    ----------
    args_seed : int | None
        The seed provided by the user (e.g., from argparse). If None, a time-based seed is used.
    logger : object
        Logger instance supporting `.info(msg, lvl, color=...)`.

    Returns
    -------
    tuple
        (seed, rng) where:
        - seed is the final integer seed used.
        - rng is a numpy.random.Generator initialized with that seed.
    """
    if args_seed is not None:
        if logger:
            logger.info(f"Using provided seed: {args_seed}", lvl=1, color='green')
        seed = args_seed
    else:
        if logger:
            logger.info("No seed provided, using current time for random seed", lvl=1, color='yellow')
        seed = time.time_ns() % (2**32 - 1)

    #! Initialize RNG
    rng = np.random.default_rng(seed=seed)

    #! Ensure global state is consistent if you rely on global RNG elsewhere
    set_global_seed(seed, backend=np)

    logger.info(f"Random seed set to {seed}", lvl=2, color='green' if args_seed is not None else 'yellow')
    return seed, rng

#########################################################

@dataclass
class SimulationParams:
    data_dir    : str
    seed        : int
    rand_num    : int
    worker_id   : int
    # times
    start_time  : int | float | None = None
    job_time    : int | float | None = None
    # node
    max_memory  : int | float | None = None
    max_cores   : int | None = None

    def copy(self) -> "SimulationParams":
        """Return a shallow copy of the simulation parameters."""
        return SimulationParams(
            data_dir    = self.data_dir,
            seed        = self.seed,
            rand_num    = self.rand_num,
            worker_id   = self.worker_id,
            start_time  = self.start_time,
            job_time    = self.job_time,
            max_memory  = self.max_memory,
            max_cores   = self.max_cores
        )

    # ---------------------------------------------------------------

    @staticmethod
    def add_slurm_cmd_args(
        ap: argparse.ArgumentParser,
        *,
        include_directory: bool     = True,
        # logging and reproducibility
        include_verbose: bool       = True,
        include_verbose_every: bool = True,
        # random seed
        include_seed: bool          = True,
        # hardware constraints
        include_maxcores: bool      = True,
        include_maxmemory: bool     = True,
        include_maxmemperwrkr: bool = True,
        # realizations
        include_realiz: bool        = True,
        include_t_per_realiz: bool  = True):
        """
        Add common simulation and SLURM-related arguments to an argparse parser.

        Parameters
        ----------
        ap : argparse.ArgumentParser
            The argument parser to which arguments will be added.
        include_* : bool
            Flags controlling which argument groups to include.
        """
        # Output directory
        if include_directory:
            ap.add_argument(
                '-d', '--save_dir',
                type     = str,
                default  = Path(os.getcwd()) / 'data',
                required = False,
                help     = 'Directory to save the data to. Defaults to current working directory if not provided.'
            )

        # Verbosity level
        if include_verbose:
            ap.add_argument(
                '-v', '--verbose',
                type    = int,
                default = 1,
                choices = [0, 1, 2, 3],
                help    = 'Verbosity level (0 = silent, 3 = very detailed logging).'
            )
            
        if include_verbose_every:
            ap.add_argument(
                '-ve', '--verbose_every',
                type    = float,
                default = 0.1,
                help    = 'Log progress every N realizations (default: 0.1). If >= 1, logs every N realizations. If < 1, logs every fraction of total realizations.'
            )

        # Reproducibility
        if include_seed:
            ap.add_argument(
                '-S', '--seed',
                type    = int,
                default = None,
                help    = 'Random seed for reproducibility. If None, uses a time-based seed.'
            )

        # Hardware constraints
        if include_maxcores:
            ap.add_argument(
                '-mc', '--max_cores',
                type    = int,
                default = _cpu_count_psutil(),
                help    = 'Maximum number of CPU cores to use (default: all available cores).'
            )

        if include_maxmemory:
            ap.add_argument(
                '-mm', '--max_memory',
                type    = float,
                default = 196.0,
                help    = 'Maximum memory in GB (default: 196.0).'
            )
            
        if include_maxmemperwrkr:
            ap.add_argument(
                '-mw', '--memory_per_worker',
                type    = float,
                default = 4.0,
                help    = 'Maximum memory per worker in GB (default: 4.0).'
            )

        # Realization-related arguments
        if include_realiz:
            ap.add_argument(
                '-nr', '--nrealis',
                type    = str,
                default = '1',
                help    = 'Number of realizations per parameter. '
                        "Formats supported: '10', '10,20,30', or 'p1:5;p2:10'."
            )

        if include_t_per_realiz:
            ap.add_argument(
                '-tr', '--trealis',
                type    = str,
                default = '',
                help    = 'Time per realization (h:m:s). '
                        "Supports single value, comma-separated list, or dictionary-like format 'p1:0:10:00;p2:0:20:00'."
            )
    
    #########################################################
    
    @staticmethod
    def is_verbose(verbose: bool, verbose_every: float, current_realiz: int, total_realiz: int) -> bool:
        """
        Determine if logging should occur based on verbosity settings.
        
        Parameters
        ----------
        verbose : bool
            Whether verbose logging is enabled.
        verbose_every : float
            Frequency of logging. If >= 1, log every N realizations. If < 1, log every fraction of total realizations.
        current_realiz : int
            The current realization index (0-based).
        total_realiz : int
            The total number of realizations.
        Returns
        -------
        bool
            True if logging should occur at the current realization, False otherwise.        
        """
        
        if not verbose:
            return False
        
        if verbose_every is None or verbose_every <= 0:
            return True

        if verbose_every >= 1:
            return (current_realiz % int(verbose_every)) == 0
        
        if verbose_every and total_realiz > 0:
            interval = max(1, int(total_realiz * verbose_every))
            return (current_realiz % interval) == 0
        
        return True
    
    
    #########################################################

    @staticmethod
    def random_identifier(rng = None, seed = None, max_int = 1000000):
        if rng is None:
            rng = np.random.default_rng(seed)    
        return f"{rng.integers(0, max_int)}"
    
    #########################################################

#########################################################

class SlurmMonitor:
    """SLURM job monitoring utilities"""
    
    try:
        from ..common.flog  import get_global_logger
        logger              = get_global_logger()
    except ImportError:
        import logging
        logger              = logging.getLogger("SLURMMonitor")   
    
    @staticmethod
    def is_slurm():
        """Check if running in SLURM environment"""
        return os.getenv("SLURM_JOB_ID") is not None
    
    #####################################################
    
    @staticmethod
    def get_memory_used(logger: Optional['Logger'] = None):
        try:
            mem_info = psutil.Process().memory_info()
            if logger: logger.info(f"Memory: {mem_info.rss / 1024**3:.2f} GB", lvl=3, color='red')
        except Exception as e:
            if logger: logger.warning(f"Failed to get memory info: {e}")
            pass
    
    #####################################################
    
    @staticmethod
    def get_remaining_time():
        """Get remaining time in seconds, returns -1 if not available"""
        if not SlurmMonitor.is_slurm():
            return -1
            
        try:
            job_id  = os.getenv("SLURM_JOB_ID")
            cmd     = f"scontrol show job {job_id}"
            result  = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return -1
                
            output  = result.stdout
            
            # Parse TimeLimit and RunTime
            time_limit_key  = "TimeLimit="
            run_time_key    = "RunTime="
            
            time_limit_pos  = output.find(time_limit_key)
            run_time_pos    = output.find(run_time_key)
            
            if time_limit_pos == -1 or run_time_pos == -1:
                return -1
                
            # Extract time strings
            time_limit_start    = time_limit_pos + len(time_limit_key)
            time_limit_end      = output.find(" ", time_limit_start)
            if time_limit_end == -1:
                time_limit_end  = output.find("\n", time_limit_start)
            time_limit_str      = output[time_limit_start:time_limit_end].strip()
            
            run_time_start      = run_time_pos + len(run_time_key)
            run_time_end        = output.find(" ", run_time_start)
            if run_time_end == -1:
                run_time_end    = output.find("\n", run_time_start)
            run_time_str        = output[run_time_start:run_time_end].strip()
            
            SlurmMonitor.logger.info("SLURM job information found:", lvl=3)
            SlurmMonitor.logger.info(f"Time limit: {time_limit_str}", lvl=4)
            SlurmMonitor.logger.info(f"Run time: {run_time_str}", lvl=4)
            
            # Parse time strings to seconds
            def parse_time(time_str):
                """Parse SLURM time format to seconds"""
                try:
                    # Handle formats: DD-HH:MM:SS, DD-HH:MM, HH:MM:SS, MM:SS
                    if '-' in time_str:
                        if time_str.count(':') == 2:  # DD-HH:MM:SS
                            days, time_part = time_str.split('-')
                            hours, minutes, seconds = map(int, time_part.split(':'))
                            return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
                        elif time_str.count(':') == 1:  # DD-HH:MM
                            days, time_part = time_str.split('-')
                            hours, minutes = map(int, time_part.split(':'))
                            return int(days) * 86400 + hours * 3600 + minutes * 60
                    else:
                        if time_str.count(':') == 2:  # HH:MM:SS
                            hours, minutes, seconds = map(int, time_str.split(':'))
                            return hours * 3600 + minutes * 60 + seconds
                        elif time_str.count(':') == 1:  # MM:SS
                            minutes, seconds = map(int, time_str.split(':'))
                            return minutes * 60 + seconds
                    return 0
                except (ValueError, AttributeError):
                    return 0
            
            total_time_limit = parse_time(time_limit_str)
            total_run_time = parse_time(run_time_str)
            
            if total_time_limit > 0 and total_run_time >= 0:
                SlurmMonitor.logger.info(f"Total time limit: {total_time_limit} seconds", lvl=3)
                SlurmMonitor.logger.info(f"Total run time: {total_run_time} seconds", lvl=3)
                
                remaining_time = total_time_limit - total_run_time
                SlurmMonitor.logger.info(f"Remaining time: {remaining_time} seconds", lvl=3)
                
                return max(0, remaining_time)  # Prevent negative values
                
        except (Exception) as e:
            SlurmMonitor.logger.warning(f"Failed to get SLURM time info: {e}")
        return -1
    
    @staticmethod
    def get_remaining_and_current_time(default: int = 60 * 60 * 24 * 4):
        """Get remaining time and current time, returns (default, current_time) if not in SLURM"""
        try:
            if not SlurmMonitor.is_slurm():
                return default, time.perf_counter()
            remaining = SlurmMonitor.get_remaining_time()
            if remaining != -1:
                return remaining, time.perf_counter()
        except Exception:
            pass
        return default, time.perf_counter()
    
    #####################################################
    
    @staticmethod
    def is_overtime(limit=1000, start_time=None, job_time=None, logger: Optional['Logger'] = None, verbose: bool = False, **kwargs):
        """Check if remaining time is less than limit seconds"""
        if not SlurmMonitor.is_slurm() and (start_time is None or job_time is None):
            return False
        
        # Take logger from kwargs if provided, otherwise use class logger
        logger = logger or SlurmMonitor.logger
        
        # 1) Local stopwatch path
        if start_time is not None and job_time is not None:
            elapsed     = time.perf_counter() - start_time
            remaining   = job_time - elapsed
            if verbose:
                logger.info(f"Elapsed={elapsed:.1f}s, remaining={remaining:.1f}s, limit={limit:.1f}s", lvl=3, **kwargs)
            if remaining < limit:
                logger.warning(f"Remaining time {remaining:.1f}s is below limit {limit:.1f}s", lvl=3, **kwargs)
                return True
            return False

        # 2) SLURM path
        remaining = SlurmMonitor.get_remaining_time()  # expected to return seconds, -1 on failure
        if remaining == -1:
            logger.warning("Failed to get remaining time from SLURM", lvl=3, **kwargs)
            return False

        if verbose:
            logger.info(f"Remaining time in SLURM job: {remaining:.1f}s", lvl=3, **kwargs)
        
        if remaining < limit:
            logger.warning(f"Remaining time {remaining:.1f}s is below limit {limit:.1f}s", lvl=3, **kwargs)
            return True
        return False

    @staticmethod
    @contextmanager
    def is_overtime_scope(limit                 : float             = 1000.0,
                        start_time              : float | None      = None,
                        job_time                : float | None      = None,
                        *,
                        slurm_poll_interval     : float             = 30.0,
                        on_trigger              : Callable | None   = None,
                        raise_on_trigger        : bool              = False,
                        logger                  : Optional['Logger']= None,
                        **kwargs
                        ):
        """
        Yield a callable `should_stop()` to poll inside your loops.
        - Caches SLURM remaining time for `slurm_poll_interval` seconds to avoid hammering scontrol.
        - If overtime is detected: logs, optionally calls `on_trigger()`, and may raise SystemExit.
        For instance, on_trigger could be used to send a notification or log an error. It could also
        save the current state or progress to resume later.

        Parameters:
            limit (float):
                The time limit in seconds.
            start_time (float | None):
                The start time of the job.
            job_time (float | None):
                The total time allocated for the job.
            slurm_poll_interval (float):
                The polling interval for SLURM jobs.
            on_trigger (callable | None):
                A callback function to call when overtime is detected.
            raise_on_trigger (bool):
                Whether to raise SystemExit when overtime is detected.
        """
        last_slurm_check    = -1e30
        cached_remaining    = None
        logger              = logger or SlurmMonitor.logger

        def _check_once() -> bool:
            nonlocal last_slurm_check, cached_remaining
            now = time.perf_counter()

            # Prefer precise local stopwatch if available
            if start_time is not None and job_time is not None:
                remaining = job_time - (now - start_time)
            else:
                if not SlurmMonitor.is_slurm():
                    return False
                if cached_remaining is None or (now - last_slurm_check) >= slurm_poll_interval:
                    cached_remaining = SlurmMonitor.get_remaining_time()
                    last_slurm_check = now
                # check SLURM
                remaining = cached_remaining
                if remaining == -1:
                    logger.warning("Failed to get remaining time from SLURM", lvl=3, **kwargs)
                    return False

            if remaining < limit:
                logger.warning(f"Remaining time {remaining:.1f}s < limit {limit:.1f}s", lvl=3, **kwargs)
                if on_trigger:
                    try:
                        on_trigger()
                    except Exception as e:
                        logger.error(f"on_trigger() failed: {e}", lvl=2, **kwargs)
                if raise_on_trigger:
                    raise SystemExit(0)
                return True
            return False

        try:
            yield _check_once
        finally:
            _check_once()

    #####################################################

#########################################################
#! EOF
#########################################################