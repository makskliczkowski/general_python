'''
Memory-related utilities for general Python QES modules.

----------------------------------------------------------------------------
File        : common/memory.py
Author      : Maks Kliczkowski
----------------------------------------------------------------------------
'''

from    typing import Dict, Optional, TYPE_CHECKING
import  gc
import  numpy as np

if TYPE_CHECKING:
    from general_python.common.flog import Logger

# --------------------------------------------------------------------------------

# Memory monitoring (optional but recommended)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not installed. Memory monitoring disabled. Install with: pip install psutil")

# --------------------------------------------------------------------------------

def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    return float('inf')  # Assume unlimited if psutil not available

def get_used_memory_gb() -> float:
    """Get memory used by current process in GB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    return 0.0

def check_memory_for_operation(required_gb: float, operation_name: str, safety_factor: float = 0.8, logger: Optional["Logger"] = None) -> bool:
    """
    Check if there's enough memory for an operation.
    
    Args:
        required_gb: Estimated memory required in GB
        operation_name: Name of operation for logging
        safety_factor: Fraction of available memory to use (default 0.8)
    
    Returns:
        True if safe to proceed, False otherwise
    """
    available       = get_available_memory_gb()
    safe_available  = available * safety_factor
    
    if required_gb > safe_available:
        if logger is not None:
            logger.error(f"Insufficient memory for {operation_name}!", lvl=0)
            logger.error(f"  Required: {required_gb:.2f} GB, Available: {available:.2f} GB (safe: {safe_available:.2f} GB)", lvl=0)
        return False
    
    if logger is not None:
        logger.info(f"Memory check passed for {operation_name}: {required_gb:.2f} GB / {available:.2f} GB available", lvl=2)
    return True

def log_memory_status(context: str = "", logger: Optional["Logger"] = None, lvl: int = 0) -> None:
    """Log current memory usage."""
    if PSUTIL_AVAILABLE:
        used        = get_used_memory_gb()
        available   = get_available_memory_gb()
        
        if logger is not None:
            # Display in mB for smaller values
            if used < 1.0 or available < 1.0:
                used_mB         = used * 1024
                available_mB    = available * 1024
                logger.info(f"Memory [{context}]: Used={used_mB:.2f}mB, Available={available_mB:.2f}mB", lvl=lvl, color='yellow')
            else:
                logger.info(f"Memory [{context}]: Used={used:.2f}GB, Available={available:.2f}GB", lvl=lvl, color='yellow')

# --------------------------------------------------------------------------------
#! End of File
# --------------------------------------------------------------------------------