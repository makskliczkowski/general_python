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
