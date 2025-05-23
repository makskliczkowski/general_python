'''
This module prepares the ini file for the run
It contains the function prepareIni which prepares the ini file for the run
Provides flexible configuration for generating job submission files.

author      : Maksymilian Kliczkowski

'''

import os
import json
import numpy as np
from typing import List, Dict, Any, Union, Optional
from itertools import product
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SlurmOptions:
    """Container for SLURM-specific options"""
    time            : str               = "1:00:00"
    memory          : str               = "4gb"
    cpus            : int               = 1
    partition       : Optional[str]     = None
    qos             : Optional[str]     = None
    account         : Optional[str]     = None
    gres            : Optional[str]     = None
    nodes           : int               = 1
    exclusive       : bool              = False
    mail_type       : Optional[str]     = None
    mail_user       : Optional[str]     = None
    array           : Optional[str]     = None
    dependency      : Optional[str]     = None
    custom_options  : Dict[str, str]    = field(default_factory=dict)
    
    def to_args(self) -> List[str]:
        """
        Convert SLURM options to command line arguments
        
        Returns:
            List[str]: List of SLURM command line arguments
        """
        args = []
        
        # Standard options
        args.extend([f"--time={self.time}", f"--mem={self.memory}", f"-c{self.cpus}"])
        
        if self.nodes != 1:
            args.append(f"-N{self.nodes}")
        
        # Optional standard options
        optional_options = {
            'partition'     : self.partition,
            'qos'           : self.qos,
            'account'       : self.account,
            'gres'          : self.gres,
            'mail-type'     : self.mail_type,
            'mail-user'     : self.mail_user,
            'array'         : self.array,
            'dependency'    : self.dependency
        }
        
        for option, value in optional_options.items():
            if value:
                args.append(f"--{option}={value}")
        
        if self.exclusive:
            args.append("--exclusive")
        
        # Custom options
        for option, value in self.custom_options.items():
            if value:
                args.append(f"--{option}={value}")
            else:
                args.append(f"--{option}")
        
        return args

@dataclass 
class JobConfig:
    """Configuration for a single job"""
    script          : str
    parameters      : List[str]     = field(default_factory=list)
    slurm_options   : SlurmOptions  = field(default_factory=SlurmOptions)
    comment         : str           = ""
    
    def to_ini_line(self) -> str:
        """Convert job configuration to INI file line"""
        
        # Build the command
        cmd_parts = [self.script]
        
        # Add parameters (quoted if they contain spaces or multiple args)
        if self.parameters:
            param_str = " ".join(str(p) for p in self.parameters)
            cmd_parts.append(f'"{param_str}"')
        
        # Add SLURM arguments
        slurm_args = self.slurm_options.to_args()
        cmd_parts.extend(slurm_args)
        
        line = " ".join(cmd_parts)
        
        # Add comment if provided
        if self.comment:
            line = f"{line}  # {self.comment}"
        
        return line

class IniGenerator:
    """Enhanced INI file generator with multiple configuration methods"""
    
    def __init__(self):
        self.jobs: List[JobConfig] = []
    
    def add_job(self, job_config: JobConfig):
        """Add a single job configuration"""
        self.jobs.append(job_config)
    
    # -------- Add simple job with basic parameters --------
    
    def add_simple_job(self, 
                    script          : str, 
                    parameters      : List[Any] = None,
                    time            : str = "1:00:00",
                    memory          : str = "4gb", 
                    cpus            : int = 1,
                    **slurm_kwargs) -> 'IniGenerator':
        """Add a simple job with basic parameters (backward compatibility)"""
        
        params      = [str(p) for p in (parameters or [])]
        slurm_opts  = SlurmOptions(time=time, memory=memory, cpus=cpus)
        
        # Handle additional SLURM options
        for key, value in slurm_kwargs.items():
            if hasattr(slurm_opts, key):
                setattr(slurm_opts, key, value)
            else:
                slurm_opts.custom_options[key.replace('_', '-')] = str(value)
        
        job = JobConfig(script=script, parameters=params, slurm_options=slurm_opts)
        self.add_job(job)
        return self
    
    # -------- Add multiple jobs with different parameter sets --------
    
    def add_parametric_jobs(self,
                        script              : str,
                        parameter_sets      : List[List[Any]],
                        slurm_template      : SlurmOptions = None,
                        comment_template    : str = "Job {index}") -> 'IniGenerator':
        """Add multiple jobs with different parameter sets"""
        
        slurm_template = slurm_template or SlurmOptions()
        
        for i, param_set in enumerate(parameter_sets):
            params  = [str(p) for p in param_set]
            comment = comment_template.format(index=i+1, params=params)
            
            job     = JobConfig(
                script          =   script,
                parameters      =   params,
                slurm_options   =   slurm_template,
                comment         =   comment
            )
            self.add_job(job)
        
        return self

    # -------- Add jobs for a parameter grid search --------
    
    def add_grid_search(self,
                    script          : str,
                    parameter_grid  : Dict[str, List[Any]],
                    slurm_template  : SlurmOptions = None,
                    fixed_params    : List[Any] = None) -> 'IniGenerator':
        """Add jobs for a parameter grid search"""

        slurm_template  = slurm_template or SlurmOptions()
        fixed_params    = fixed_params or []
        
        # Generate all combinations
        param_names     = list(parameter_grid.keys())
        param_values    = list(parameter_grid.values())

        for i, combination in enumerate(product(*param_values)):
            # Build parameter list
            params      = list(fixed_params)  # Start with fixed parameters
            params.extend(combination)
            
            # Create descriptive comment
            param_desc  = ", ".join(f"{name}={val}" for name, val in zip(param_names, combination))
            comment     = f"Grid search {i+1}: {param_desc}"
            
            job         = JobConfig(
                script          =   script,
                parameters      =   [str(p) for p in params],
                slurm_options   =   slurm_template,
                comment         =   comment
            )
            self.add_job(job)
        
        return self
    
    # -------- Load and save functions --------
    
    def load_from_json(self, json_file: str) -> 'IniGenerator':
        """Load job configurations from a JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for job_data in data.get('jobs', []):
            # Parse SLURM options
            slurm_data = job_data.get('slurm_options', {})
            slurm_opts = SlurmOptions(**slurm_data)
            
            # Create job config
            job = JobConfig(
                script          =   job_data['script'],
                parameters      =   job_data.get('parameters', []),
                slurm_options   =   slurm_opts,
                comment         =   job_data.get('comment', '')
            )
            self.add_job(job)
        
        return self
    
    # -------- Save template JSON function --------
    
    def save_template_json(self, filename: str):
        """Save a template JSON configuration file"""
        template = {
            "jobs": [
                {
                    "script"        : "./example_script.sh",
                    "parameters"    : ["param1", "param2", "param3"],
                    "slurm_options" : {
                        "time"      : "2:00:00",
                        "memory"    : "8gb",
                        "cpus"      : 4,
                        "partition" : "compute",
                        "qos"       : "normal"
                    },
                    "comment"       : "Example job"
                }
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(template, f, indent=2)
    
    # -------- Write INI file function --------
    
    def write_ini(self, filename: str, header_comment: str = None):
        """Write all jobs to an INI file"""
        with open(filename, 'w') as f:
            # Write header
            f.write(f"# Generated INI file - {datetime.now().isoformat()}\n")
            if header_comment:
                f.write(f"# {header_comment}\n")
            f.write(f"# Total jobs: {len(self.jobs)}\n\n")
            
            # Write jobs
            for job in self.jobs:
                f.write(job.to_ini_line() + "\n")
    
    # -------- Preview function --------
    
    def preview(self, max_lines: int = 10) -> str:
        """Preview the first few lines of the INI file"""
        lines = [f"# Preview - Total jobs: {len(self.jobs)}"]
        
        for i, job in enumerate(self.jobs[:max_lines]):
            lines.append(job.to_ini_line())
        
        if len(self.jobs) > max_lines:
            lines.append(f"# ... and {len(self.jobs) - max_lines} more jobs")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all jobs"""
        self.jobs.clear()
    
    def count(self) -> int:
        """Get number of jobs"""
        return len(self.jobs)
    
    def __len__(self) -> int:
        """Get number of jobs"""
        return len(self.jobs)
    
    def __getitem__(self, index: int) -> JobConfig:
        """Get job by index"""
        return self.jobs[index]
    
    def __iter__(self):
        """Iterate over jobs"""
        return iter(self.jobs)
    
    def __str__(self) -> str:
        """String representation of the INI generator"""
        return "\n".join(job.to_ini_line() for job in self.jobs)
    
    def __repr__(self) -> str:
        """String representation of the INI generator"""
        return self.__str__()

################################################################
#! LEGACY - BACKWARD COMPATIBILITY
################################################################

def prepare_ini(script  : str, 
            first       : str,
            middle      : str,
            time        : str,
            mem         : str,
            cpus        : int,
            fun         : int = None) -> str:
    """ Allows to prepare the ini file for the run
    
    Parameters
    ----------
    script : str
        The script to be executed
    first  : str
        The first argument
    middle : str
        The middle argument
    time   : str
        The time argument
    mem    : str
        The memory argument
    cpus   : int
        The cpus argument    
    fun    : int
        The function to be executed
    """
    params = [first, middle]
    if fun is not None:
        params.append(str(fun))
    
    generator = IniGenerator()
    generator.add_simple_job(script, params, time, mem, cpus)
    
    return generator.jobs[0].to_ini_line()

def middle_ini(*args) -> str:
    """Prepare middle argument - backward compatibility - older scripts like QSM and others"""
    return " ".join(str(arg) for arg in args)

################################################################
#! PARAMETER SWEEP
################################################################

def create_parameter_sweep(script       : str,
                        param_ranges    : Dict[str, List[Any]],
                        slurm_config    : Dict[str, Any] = None) -> IniGenerator:
    """Quick function to create parameter sweep"""
    
    slurm_opts  = SlurmOptions(**(slurm_config or {}))
    generator   = IniGenerator()
    generator.add_grid_search(script, param_ranges, slurm_opts)
    return generator

################################################################
#! EXAMPLE USAGE
################################################################

def example_basic_usage():
    """Example of basic usage"""
    generator = IniGenerator()
    
    # Add simple jobs
    generator.add_simple_job(
        script      =   "./my_script.sh",
        parameters  =   ["10", "20", "0.1"],
        time        =   "2:00:00",
        memory      =   "8gb",
        cpus        =   4,
        partition   =   "compute"
    )
    
    return generator

def example_grid_search():
    """Example of parameter grid search"""
    generator = IniGenerator()
    
    # Define parameter grid
    param_grid = {
        'learning_rate'     : [0.01, 0.1, 0.5],
        'batch_size'        : [32, 64, 128],
        'epochs'            : [10, 50, 100]
    }
    
    # SLURM configuration template
    slurm_config = SlurmOptions(
        time        = "4:00:00",
        memory      = "16gb",
        cpus        = 8,
        partition   = "gpu",
        gres        = "gpu:1"
    )
    
    generator.add_grid_search(
        script          =   "./train_model.sh",
        parameter_grid  =   param_grid,
        slurm_template  =   slurm_config,
        fixed_params    =   ["dataset.csv", "output_dir"]
    )
    
    return generator

if __name__ == "__main__":
    # Demo usage
    print("=== Basic Usage Demo ===")
    basic_gen = example_basic_usage()
    print(basic_gen.preview())
    
    print("\n=== Grid Search Demo ===")
    grid_gen = example_grid_search()
    print(f"Generated {grid_gen.count()} jobs for parameter sweep")
    print(grid_gen.preview(5))
    
    # Save examples
    basic_gen.write_ini("basic_jobs.ini", "Basic job example")  
    grid_gen.write_ini("grid_search_jobs.ini", "Parameter grid search")
    grid_gen.save_template_json("job_template.json")
    
    print("\nFiles generated: basic_jobs.ini, grid_search_jobs.ini, job_template.json")
    
################################################################

class ManyBodyEstimator:
    """Class for many-body estimator"""
    
    base_memory_per_element     = 8     # bytes (for double precision)
    memory_overhead_factor      = 1.2   # Empirical factor for overhead
    
    def __init__(self, script: str):
        self.script = script
    
    @staticmethod
    def estimate_inner_realizations(Ns: int, custom: Optional[int] = None) -> int:
        """
        Estimate number of inner loop realizations based on system size
        Uses adaptive scaling for larger systems
        Parameters:
            Ns (int):
                System size
            custom (int, optional):
                Custom number of realizations
        """
        if custom is not None:
            return custom
            
        # Base realizations for different system sizes
        base_realizations = {
            6: 500, 7: 200, 8: 200, 9: 200, 10: 200,
            11: 150, 12: 100, 13: 100, 14: 80, 15: 10, 16: 5
        }
        
        if Ns in base_realizations:
            return base_realizations[Ns]
        elif Ns > 16:
            # For very large systems, use logarithmic scaling
            return ManyBodyEstimator.estimate_inner_realizations(int(np.log2(Ns)))
        else:
            return 1
    
    @staticmethod
    def estimate_outer_realizations(Ns: int) -> int:
        """Estimate number of outer loop realizations"""
        if Ns < 14:
            return 1
        elif Ns == 14:
            return 2
        elif Ns == 15:
            return 3
        elif Ns <= 16:
            return 5
        elif Ns > 16:
            return ManyBodyEstimator.estimate_outer_realizations(int(np.log2(Ns)))
        else:
            return 1
    
    @staticmethod
    def estimate_matrix_memory(Ns: int) -> int:
        """
        Estimate memory requirements for matrices in GB
        Accounts for different matrix structures per model type
        """
        # Base matrix size (Hilbert space dimension)
        if Ns <= 16:
            hilbert_size = 2 ** Ns
        else:
            hilbert_size = 2 ** int(np.log2(Ns))
            
        # Memory estimation for different models
                # Memory for main matrices (Hamiltonian, eigenvectors, etc.)
        matrix_memory_bytes = (
            hilbert_size * hilbert_size * ManyBodyEstimator.base_memory_per_element *
            ManyBodyEstimator.memory_overhead_factor
        )
        
        # Convert to GB and add safety margin
        memory_gb = int(max(1, matrix_memory_bytes / 1e9 * 1.2))
        
        # Apply empirical corrections based on your original data
        if Ns < 10:
            memory_gb = max(memory_gb, 10)
        elif Ns < 12:
            memory_gb = max(memory_gb, 16)
        elif Ns < 14:
            memory_gb = max(memory_gb, 16)
        elif Ns == 14:
            memory_gb = max(memory_gb, 64)
        elif Ns == 15:
            memory_gb = max(memory_gb, 144)
        elif Ns == 16:
            memory_gb = max(memory_gb, 296)
        elif Ns > 16:
            # For very large systems, use more conservative estimates
            memory_gb = ManyBodyEstimator.estimate_matrix_memory(int(np.log2(Ns)))
        return memory_gb

    @staticmethod
    def estimate_simulation_time(Ns: int, realizations: int = None) -> str:
        """
        Estimate wall-clock time needed for simulation.
        """
        
        # Base time estimates (in minutes)
        base_times = {
            6   : 30*60-1, 7    : 30*60-1,   8     : 30*60-1,    9    : 30*60-1,
            10  : 50*60-1, 11   : 60*60-1,   12    : 100*60-1,  13    : 160*60-1,
            14  : 50*100-1, 15  : 100*80-1,  16    : 100*80-1
        }
        
        if Ns in base_times:
            minutes         = base_times[Ns]
        elif Ns > 16:
            # For very large systems, use logarithmic scaling
            base_minutes    = base_times.get(int(np.log2(Ns)), 200*60)
            minutes         = int(base_minutes * 1.5)  # Scale up for larger systems
        else:
            minutes         = 30*60-1  # Default 30 hours

        # Adjust for realizations if provided
        if realizations:
            base_realizations   = ManyBodyEstimator.estimate_inner_realizations(Ns)
            if realizations != base_realizations:
                minutes         = int(minutes * (realizations / base_realizations))

        # Convert to HH:MM:SS format
        hours   = minutes // 60
        mins    = minutes % 60
        return f"{hours:02d}:{mins:02d}:59"
    
    @staticmethod
    def estimate_cpu_requirements(Ns: int) -> int:
        """Estimate optimal number of CPU cores"""
        if Ns < 12:
            return 1
        elif Ns <= 14:
            return 4
        elif Ns < 16:
            return 12
        elif Ns == 16:
            return 16
        elif Ns > 16:
            base_cpus = self.estimate_cpu_requirements(int(np.log2(Ns)))
            return min(32, base_cpus * 2)  # Cap at 32 cores
        else:
            return 1
        
################################################################
#! END OF FILE
################################################################