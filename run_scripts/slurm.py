import sys
import os
import argparse
import numpy as np
import pandas as pd
import time
import subprocess
import psutil

from general_python.common.flog import Logger, get_global_logger

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
    
    #####################################################
    
    @staticmethod
    def is_overtime(limit=1000, start_time=None, job_time=None):
        """Check if remaining time is less than limit seconds"""
        if not SlurmMonitor.is_slurm():
            return False
        
        if start_time is not None:
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time >= job_time - limit:
                SlurmMonitor.logger.info(f"Elapsed time exceeds limit: {elapsed_time} seconds", lvl=3)
                return True
        
        remaining_time = SlurmMonitor.get_remaining_time()
        
        if remaining_time == -1:
            return False
            
        SlurmMonitor.logger.info(f"Remaining time in SLURM job: {remaining_time} seconds", lvl=3)
        return remaining_time < limit
    
    #####################################################
    
#########################################################