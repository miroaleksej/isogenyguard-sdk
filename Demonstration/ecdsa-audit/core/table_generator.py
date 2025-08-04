"""
Module for generating Rₓ table subregions.
Implements sampling strategies for efficient auditing.
"""

from typing import List, Tuple
import random
from .curve_operations import CurveOperations

class TableGenerator:
    def __init__(self, curve_ops: CurveOperations):
        self.curve_ops = curve_ops
    
    def generate_random_regions(self, 
                              Q: 'coincurve.PublicKey', 
                              num_regions: int = 10, 
                              region_size: int = 50) -> List[List[List[int]]]:
        """
        Generates random subregions of the Rₓ table for auditing
        
        Args:
            Q: public key
            num_regions: number of subregions
            region_size: size of each subregion (region_size x region_size)
            
        Returns:
            List of table subregions
        """
        regions = []
        n = self.curve_ops.n
        
        for _ in range(num_regions):
            # Randomly select subregion start
            u_r_start = random.randint(1, n - region_size - 1)
            u_z_start = random.randint(1, n - region_size - 1)
            
            region = self.curve_ops.compute_subregion(
                Q, 
                u_r_start, u_r_start + region_size,
                u_z_start, u_z_start + region_size
            )
            regions.append(region)
        
        return regions
    
    def generate_optimal_regions(self, 
                               Q: 'coincurve.PublicKey', 
                               num_regions: int = 5, 
                               region_size: int = 50) -> List[List[List[int]]]:
        """
        Generates subregions in optimal zones for auditing.
        Based on the theorem about optimal d_opt ≈ n/2.
        
        Args:
            Q: public key
            num_regions: number of subregions
            region_size: size of each subregion
            
        Returns:
            List of subregions in optimal zones
        """
        regions = []
        n = self.curve_ops.n
        d_opt = n // 2  # Optimal zone for anomaly detection
        
        for i in range(num_regions):
            # Offset from optimal zone for diversity
            offset = (i * region_size) % (n // 4)
            u_r_start = (d_opt + offset) % (n - region_size)
            u_z_start = (d_opt + offset * 2) % (n - region_size)
            
            region = self.curve_ops.compute_subregion(
                Q, 
                u_r_start, u_r_start + region_size,
                u_z_start, u_z_start + region_size
            )
            regions.append(region)
        
        return regions
    
    def generate_symmetry_regions(self, 
                                Q: 'coincurve.PublicKey', 
                                num_regions: int = 5, 
                                region_size: int = 50) -> List[Tuple[List[List[int]], int, int]]:
        """
        Generates subregions around symmetry points for detailed analysis.
        
        Args:
            Q: public key
            num_regions: number of subregions
            region_size: size of each subregion
            
        Returns:
            List of subregions with their special point coordinates
        """
        regions = []
        n = self.curve_ops.n
        
        for _ in range(num_regions):
            u_r = random.randint(1, n - 1)
            # Randomly select a symmetry point (we don't know d, so we'll check multiple points)
            u_z_center = random.randint(1, n - 1)
            
            u_r_start = max(1, u_r - region_size // 2)
            u_z_start = max(1, u_z_center - region_size // 2)
            
            region = self.curve_ops.compute_subregion(
                Q, 
                u_r_start, u_r_start + region_size,
                u_z_start, u_z_start + region_size
            )
            regions.append((region, u_r, u_z_center))
        
        return regions
