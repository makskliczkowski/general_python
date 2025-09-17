import sys
import os
import argparse
from typing import Callable
import numpy as np
import pandas as pd
import time
import subprocess
import psutil
from contextlib import contextmanager
from dataclasses import dataclass

try:
    from ..common.flog import Logger, get_global_logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    
    def get_global_logger():
        """Get the global logger instance"""
        return logging.getLogger("SLURM")

#########################################################

@dataclass
class SimulationParams:
    data_dir            : str
    seed                : int
    rand_num            : int
    worker_id           : int
    # times
    start_time          : int | float | None = None
    job_time            : int | float | None = None

    def copy(self):
        return SimulationParams(
            data_dir    =self.data_dir,
            seed        =self.seed,
            rand_num    =self.rand_num,
            worker_id   =self.worker_id,
            start_time  =self.start_time,
            job_time    =self.job_time
        )
        
    @staticmethod
    def random_identifier(rng = None, seed = None, max_int = 1000000):
        if rng is None:
            np.random.default_rng(seed)    
        return f"{rng.integers(0, max_int)}"

class SlurmMonitor:
    """SLURM job monitoring utilities"""
    
    logger = get_global_logger()
    
    @staticmethod
    def is_slurm():
        """Check if running in SLURM environment"""
        return os.getenv("SLURM_JOB_ID") is not None
    
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
    def get_remaining_and_current_time():
        return SlurmMonitor.get_remaining_time(), time.perf_counter()
    
    #####################################################
    
    @staticmethod
    def is_overtime(limit=1000, start_time=None, job_time=None):
        """Check if remaining time is less than limit seconds"""
        if not SlurmMonitor.is_slurm() and (start_time is None or job_time is None):
            return False
        
        # 1) Local stopwatch path
        if start_time is not None and job_time is not None:
            elapsed     = time.perf_counter() - start_time
            remaining   = job_time - elapsed
            SlurmMonitor.logger.info(f"Elapsed={elapsed:.1f}s, remaining={remaining:.1f}s, limit={limit:.1f}s", lvl=3)
            if remaining < limit:
                SlurmMonitor.logger.warning(f"Remaining time {remaining:.1f}s is below limit {limit:.1f}s", lvl=3)
                return True
            return False

        # 2) SLURM path
        remaining = SlurmMonitor.get_remaining_time()  # expected to return seconds, -1 on failure
        if remaining == -1:
            SlurmMonitor.logger.warning("Failed to get remaining time from SLURM", lvl=3)
            return False

        SlurmMonitor.logger.info(f"Remaining time in SLURM job: {remaining:.1f}s", lvl=3)
        if remaining < limit:
            SlurmMonitor.logger.warning(f"Remaining time {remaining:.1f}s is below limit {limit:.1f}s", lvl=3)
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
                        raise_on_trigger        : bool              = False):
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
        last_slurm_check = -1e30
        cached_remaining = None

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
                    SlurmMonitor.logger.warning("Failed to get remaining time from SLURM", lvl=3)
                    return False

            if remaining < limit:
                SlurmMonitor.logger.warning(f"Remaining time {remaining:.1f}s < limit {limit:.1f}s", lvl=3)
                if on_trigger:
                    try:
                        on_trigger()
                    except Exception as e:
                        SlurmMonitor.logger.error(f"on_trigger() failed: {e}", lvl=2)
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