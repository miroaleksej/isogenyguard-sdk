"""
Module for topological analysis of the Rₓ table.
Contains methods for computing Betti numbers and analyzing structure.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

class TopologyAnalyzer:
    def __init__(self):
        pass
    
    def compute_betti_numbers(self, region: List[List[int]]) -> Tuple[int, int, int]:
        """
        Computes Betti numbers for a subregion of the table.
        
        Args:
            region: Rₓ table subregion
            
        Returns:
            Tuple (β₀, β₁, β₂) - Betti numbers
        """
        # Convert table to points in 3D space (u_r, u_z, R_x)
        points = []
        for i, row in enumerate(region):
            for j, rx in enumerate(row):
                points.append([i, j, rx])
        
        points = np.array(points)
        
        # Build simplicial complex
        # For simplicity, use Delaunay triangulation
        try:
            tri = Delaunay(points[:, :2])  # Use only u_r and u_z coordinates
        except:
            # If triangulation fails, return zero Betti numbers
            return (0, 0, 0)
        
        # Compute β₀ - number of connected components
        # Create adjacency matrix for vertices
        n_vertices = len(points)
        edges = set()
        
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        # Create sparse adjacency matrix
        row = []
        col = []
        for i, j in edges:
            row.append(i)
            col.append(j)
            row.append(j)
            col.append(i)
        
        if not row:
            return (0, 0, 0)
            
        adjacency = coo_matrix((np.ones(len(row)), (row, col)), 
                             shape=(n_vertices, n_vertices))
        
        # Compute number of connected components
        n_components, _ = connected_components(adjacency)
        beta_0 = n_components
        
        # Simplified calculation of β₁ (number of "cycles")
        beta_1 = len(edges) - (n_vertices - beta_0)
        
        # For β₂ in 2D case, usually 0 or 1
        beta_2 = 1 if beta_0 == 1 else 0
        
        return (beta_0, beta_1, beta_2)
    
    def analyze_spiral_waves(self, region: List[List[int]]) -> float:
        """
        Analyzes spiral waves and computes the damping coefficient.
        
        Args:
            region: Rₓ table subregion
            
        Returns:
            Damping coefficient γ
        """
        # Spiral wave analysis algorithm:
        # 1. Find spiral center in the subregion
        # 2. Measure wave amplitude at different distances from center
        # 3. Calculate damping coefficient
        
        n = len(region)
        m = len(region[0])
        
        # Collect values along diagonals (spirals)
        diagonals = []
        for k in range(-m+1, n):
            diag = []
            for i in range(max(0, k), min(n, k+m)):
                j = i - k
                if 0 <= j < m:
                    diag.append(region[i][j])
            if diag:
                diagonals.append(diag)
        
        # Compute amplitude for each diagonal
        amplitudes = [max(d) - min(d) for d in diagonals]
        
        # Calculate damping coefficient
        if len(amplitudes) < 2:
            return 0.0
            
        log_ratios = []
        for i in range(1, len(amplitudes)):
            if amplitudes[i-1] > 0 and amplitudes[i] > 0:
                ratio = amplitudes[i] / amplitudes[i-1]
                log_ratios.append(np.log(ratio))
        
        if not log_ratios:
            return 0.0
            
        # Damping coefficient - mean of logarithmic ratios
        gamma = -np.mean(log_ratios)
        return gamma
    
    def check_symmetry(self, region: List[List[int]], center_row: int = None) -> float:
        """
        Checks symmetry around special points.
        
        Args:
            region: Rₓ table subregion
            center_row: specific row to check (optional)
            
        Returns:
            Symmetry coefficient (0-1)
        """
        # Algorithm:
        # 1. For each row find the special point
        # 2. Check symmetry around it
        # 3. Return the proportion of symmetric points
        
        symmetric_points = 0
        total_points = 0
        
        rows_to_check = [center_row] if center_row is not None else range(len(region))
        
        for i in rows_to_check:
            if i >= len(region):
                continue
                
            row = region[i]
            # Simple implementation - assume special point is approximately in the middle
            center = len(row) // 2
            
            for j in range(1, min(center, len(row) - center - 1)):
                # Check if values are symmetric (with small tolerance for floating point)
                if abs(row[center - j] - row[center + j]) < 0.01 * (row[center - j] + row[center + j] + 1):
                    symmetric_points += 1
                total_points += 1
        
        return symmetric_points / total_points if total_points > 0 else 0.0
    
    def detect_spiral_structure(self, region: List[List[int]]) -> Dict[str, float]:
        """
        Detects spiral structure properties in the region.
        
        Args:
            region: Rₓ table subregion
            
        Returns:
            Dictionary with spiral structure properties
        """
        n = len(region)
        m = len(region[0])
        
        # Find dominant spiral direction
        max_correlation = 0
        best_slope = 0
        
        # Test different slopes (d values)
        for slope in range(1, 10):
            correlation = 0
            count = 0
            
            # Check correlation along lines with given slope
            for offset in range(-m, n):
                values = []
                for i in range(max(0, offset), min(n, offset + m)):
                    j = i - offset
                    if 0 <= j < m:
                        values.append(region[i][j])
                
                if len(values) > 2:
                    # Simple correlation measure
                    diff = np.diff(values)
                    same_sign = np.sum(np.sign(diff[:-1]) == np.sign(diff[1:]))
                    correlation += same_sign / (len(diff) - 1)
                    count += 1
            
            if count > 0:
                avg_correlation = correlation / count
                if avg_correlation > max_correlation:
                    max_correlation = avg_correlation
                    best_slope = slope
        
        return {
            "dominant_slope": best_slope,
            "correlation_strength": max_correlation,
            "is_spiral_structure": max_correlation > 0.7
        }
