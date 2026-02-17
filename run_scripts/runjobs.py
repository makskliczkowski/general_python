#!/usr/bin/env python3
"""
Job Runner Script for SLURM
Reads job configurations from an INI file and submits them sequentially.

This script is designed to run jobs on a SLURM cluster.
It reads job configurations from an INI file, submits them to the SLURM scheduler,
and logs the results. The script handles job submission errors and maintains a log of successful and failed submissions.

Author      :  Maksymilian Kliczkowski
Date        :  2025-05-24
Version     :  1.0
"""

import subprocess
import sys
import os
import time

from datetime import datetime

def log_message(message, log_file=None):
    """Log a message with timestamp"""
    timestamp       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg   = f"[{timestamp}] {message}"
    print(formatted_msg)
    if log_file and os.path.exists(log_file):
        with open(log_file, "a") as f:
            f.write(formatted_msg + "\n")

# ----------------------------------------------------

def parse_ini_line(line):
    """
    Parse a line from the INI file
    Expected format: script_path "parameters" slurm_args
    Example: ./submit_job.sh "param1 param2" --time=1:00:00 --mem=4gb
    
    Parameters and SLURM arguments are optional.
    Params:
        line (str): A line from the INI file.
    Returns:
        dict: A dictionary with keys 'script', 'parameters', and 'slurm_args'.
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    # Split by quotes to separate script, parameters, and SLURM args
    parts = line.split('"')
    if len(parts) < 3:
        raise ValueError(f"Invalid line format: {line}")
    
    # Extract components
    script_parts    = parts[0].strip().split()
    script_path     = script_parts[0]
    
    parameters      = parts[1]
    slurm_args      = parts[2].strip().split()[1:] if len(parts[2].strip().split()) > 1 else []
    
    #! return a dictionary with script, parameters, and slurm_args
    # Example:
    # {
    #     'script': './submit_job.sh',
    #     'parameters': 'param1 param2',
    #     'slurm_args': ['--time=1:00:00', '--mem=4gb']
    # }
    return {
        'script'        : script_path,
        'parameters'    : parameters.split(),
        'slurm_args'    : slurm_args,
        'full_line'     : line
    }

# ----------------------------------------------------

def check_job_success(result):
    """
    Check if job submission was successful
    
    Parameters:
        result (subprocess.CompletedProcess): The result of the job submission command.
    Returns:
        bool: True if the job was submitted successfully, False otherwise.
    """
    if result.returncode != 0:
        return False
    
    stdout = result.stdout.strip() if isinstance(result.stdout, str) else result.stdout.decode('utf-8').strip()
    stderr = result.stderr.strip() if isinstance(result.stderr, str) else result.stderr.decode('utf-8').strip()
    
    # Check for common success indicators
    if 'Submitted batch job' in stdout:
        return True
    
    # Check for failure indicators
    failure_indicators  = ['error', 'failed', 'invalid', 'denied']
    full_output         = (stdout + ' ' + stderr).lower()
    return not any(indicator in full_output for indicator in failure_indicators)

# ----------------------------------------------------

def run_job_runner(ini_file, runmode):
    """Main function to run jobs from INI file"""
    
    if not ini_file.endswith(".ini"):
        raise ValueError(f"{ini_file} is not a '.ini' file")
    
    if not os.path.exists(ini_file):
        raise FileNotFoundError(f"INI file not found: {ini_file}")
    
    # take tha name out of the basename that can be a directory and it s in the 
    # current working directory
    init_file_base  = os.path.basename(ini_file)
    
    # Setup output files
    base_name       = init_file_base.replace('.ini', '')
    output_file     = f"output_{base_name}.dat"
    log_file        = f"runner_log_{base_name}.log"

    log_message(f"Starting job runner for: {ini_file}", log_file)
    log_message(f"Output will be logged to: {output_file}", log_file)
    
    # Get submission delay from environment variable
    try:
        delay = float(os.getenv("JOB_SUBMISSION_DELAY", "0.1"))
    except ValueError:
        log_message("Warning: JOB_SUBMISSION_DELAY is not a valid float. Using default 0.1s.")
        delay = 0.1

    jobs_run        = 0
    jobs_failed     = 0
    
    # Read and process jobs
    with open(ini_file, "r") as f:
        lines = f.readlines()
    
    remaining_lines = []
    with open(output_file, "a") as output_f:
        for i, line in enumerate(lines):
            try:
                job_config = parse_ini_line(line)
                if job_config is None:
                    # Skip empty lines or comments
                    remaining_lines.append(line)
                    continue
                
                # Prepare command
                cmd = [job_config['script']] + job_config['parameters'] + job_config['slurm_args']
                log_message(f"Job {jobs_run + 1}: Running command: {' '.join(cmd)}", log_file)
                
                # Run the job
                if runmode == "--test":
                    log_message(f"Job {jobs_run + 1}: Test mode, not submitting", log_file)
                    output_f.write(f"{datetime.now().isoformat()}: {' '.join(cmd)}\n")
                    output_f.flush()
                    jobs_run += 1
                    continue
                
                result = subprocess.run(
                    cmd,
                    stdout  =   subprocess.PIPE,
                    stderr  =   subprocess.PIPE,
                    text    =   True,
                    timeout =   300  # 5 minute timeout for job submission
                )
                
                # Check if job submission was successful
                if check_job_success(result):
                    log_message(f"Job {jobs_run + 1}: Submitted successfully", log_file)
                    
                    # Log the successful submission
                    output_f.write(f"{datetime.now().isoformat()}: {' '.join(cmd)}\n")
                    output_f.flush()
                    
                    jobs_run += 1
                    
                    # Add a small delay between submissions
                    if delay > 0:
                        time.sleep(delay)
                    
                else:
                    log_message(f"Job {jobs_run + 1}: Submission failed", log_file)
                    log_message(f"STDOUT: {result.stdout}", log_file)
                    log_message(f"STDERR: {result.stderr}", log_file)
                    
                    jobs_failed += 1
                    remaining_lines.append(line)
                    
                    # Stop on first failure (matching original behavior)
                    break
                    
            except subprocess.TimeoutExpired:
                log_message(f"Job {jobs_run + 1}: Submission timed out", log_file)
                jobs_failed += 1
                remaining_lines.append(line)
                break
                
            except Exception as e:
                log_message(f"Job {jobs_run + 1}: Error - {str(e)}", log_file)
                jobs_failed += 1
                remaining_lines.append(line)
                break
    
    #! Update the INI file with remaining jobs
    with open(ini_file, "w") as f:
        f.writelines(remaining_lines[jobs_run:])
    
    #! Summary
    log_message(f"Summary: {jobs_run} jobs submitted, {jobs_failed} failed", log_file)
    
    #! Remove empty INI file
    if os.path.getsize(ini_file) == 0:
        log_message(f"Removing empty INI file: {ini_file}", log_file)
        os.remove(ini_file)
    
    return jobs_run, jobs_failed

# ----------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 runjobs.py <ini_file> <runmode>")
        sys.exit(1)
    
    ini_file    = sys.argv[1]
    if len(sys.argv) > 2:
        # Check if runmode is valid
        if sys.argv[2] not in ["--run", "--test"]:
            print("Invalid runmode. Use '--run' or '--test'.")
            sys.exit(1)
        runmode = sys.argv[2]
    try:
        jobs_run, jobs_failed = run_job_runner(ini_file, runmode)
        
        if jobs_failed > 0:
            print(f"Job runner completed with failures: {jobs_run} successful, {jobs_failed} failed")
            sys.exit(1)
        else:
            print(f"Job runner completed successfully: {jobs_run} jobs submitted")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
# ----------------------------------------------------
# End of script
# ----------------------------------------------------