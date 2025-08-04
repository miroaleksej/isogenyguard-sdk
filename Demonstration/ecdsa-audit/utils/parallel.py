"""
Module for parallel processing.
Implements efficient parallel execution of tasks.
"""

import multiprocessing
from typing import Callable, Iterable, List, Any
import functools

def parallel_map(func: Callable, items: Iterable, num_workers: int = None, chunk_size: int = 1) -> List[Any]:
    """
    Parallel version of map function.
    
    Args:
        func: function to apply to each item
        items: iterable of items to process
        num_workers: number of worker processes (None = use all available cores)
        chunk_size: number of items to process in each chunk
        
    Returns:
        List of results
    """
    num_workers = num_workers or multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(func, items, chunksize=chunk_size)
    
    return results

def parallel_starmap(func: Callable, arg_tuples: Iterable[tuple], num_workers: int = None, chunk_size: int = 1) -> List[Any]:
    """
    Parallel version of starmap function (for functions with multiple arguments).
    
    Args:
        func: function to apply to each argument tuple
        arg_tuples: iterable of argument tuples
        num_workers: number of worker processes (None = use all available cores)
        chunk_size: number of items to process in each chunk
        
    Returns:
        List of results
    """
    num_workers = num_workers or multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(func, arg_tuples, chunksize=chunk_size)
    
    return results

class ParallelExecutor:
    """Helper class for more complex parallel execution patterns"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or multiprocessing.cpu_count()
    
    def map(self, func: Callable, items: Iterable, chunk_size: int = 1) -> List[Any]:
        """Execute map operation in parallel"""
        return parallel_map(func, items, self.num_workers, chunk_size)
    
    def starmap(self, func: Callable, arg_tuples: Iterable[tuple], chunk_size: int = 1) -> List[Any]:
        """Execute starmap operation in parallel"""
        return parallel_starmap(func, arg_tuples, self.num_workers, chunk_size)
    
    def apply_async(self, func: Callable, args: tuple = (), kwargs: dict = {}) -> Any:
        """Execute a single function asynchronously"""
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            result = pool.apply_async(func, args, kwargs)
            return result.get()
    
    def chunked_map(self, func: Callable, items: Iterable, chunk_size: int = 100) -> List[Any]:
        """
        Process items in chunks for better memory management.
        
        Args:
            func: function to apply to each chunk
            items: iterable of items to process
            chunk_size: number of items per chunk
            
        Returns:
            List of results
        """
        def process_chunk(chunk):
            return [func(item) for item in chunk]
        
        # Create chunks
        chunks = []
        chunk = []
        for item in items:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)
        
        # Process chunks in parallel
        chunk_results = self.map(process_chunk, chunks)
        
        # Flatten results
        results = [item for chunk_result in chunk_results for item in chunk_result]
        return results

# Example usage:
# def process_region(args):
#     Q, u_r, u_z_min, u_z_max = args
#     # ... region processing code ...
#     return region_data
#
# with multiprocessing.Pool() as pool:
#     regions = pool.starmap(process_region, region_args)
