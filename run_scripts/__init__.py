'''
This module provides functionality to calculate the optimal number of workers
'''

import psutil

def calculate_optimal_workers(alphas, available_memory, memory_per_worker, max_cores=None, logger=None):
    """
    Calculate optimal number of workers based on available resources
    """
    
    # Get system info
    cpu_count = psutil.cpu_count(logical=False) or 1
    if max_cores:
        cpu_count = min(cpu_count, max_cores)
    
    # Calculate based on memory
    memory_limited_workers = max(1, int(available_memory / memory_per_worker))
    
    # Calculate based on work (don't create more workers than alpha values)
    work_limited_workers = len(alphas)
    
    # Calculate based on CPU cores (leave some for system)
    cpu_limited_workers = max(1, cpu_count - 1)
    
    # Take the minimum of all constraints
    optimal_workers = min(memory_limited_workers, work_limited_workers, cpu_limited_workers)
    
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

####################################################################

def calculate_realisations_per_parameter(parameters: list, n_realisations: str):
    '''
    Calculate the number of realizations per parameter based on user input
    - parameters      : list of parameters, e.g., list of sites
    - n_realisations  : str, either a single integer or a comma-separated list of integers
    '''

    if not parameters:
        raise ValueError("Parameters list cannot be empty")

    # If comma-separated, split into list
    if ',' in n_realisations:
        n_realisations = n_realisations.split(',')
    else:
        n_realisations = [n_realisations] * len(parameters)
        
    # Validate and convert
    if all(x.strip().isdigit() for x in n_realisations):
        n_realisations = [int(x.strip()) for x in n_realisations]
    elif len(n_realisations) == 1 and n_realisations[0].strip().isdigit():
        n_realisations = int(n_realisations[0].strip())
        n_realisations = [n_realisations] * len(parameters)
    else:
        raise ValueError("--number_of_realizations must be an integer or a comma-separated list of integers")

    # If they don't match, raise an error
    if len(n_realisations) != len(parameters):
        raise ValueError("The number of realizations must match the number of parameters")

    # Convert to dictionary if it's a list
    if isinstance(n_realisations, list):
        n_realisations = {param: n_realisations[i] for i, param in enumerate(parameters)}
    else:
        n_realisations = {param: n_realisations for param in parameters}
    return n_realisations

########################################################################################################################