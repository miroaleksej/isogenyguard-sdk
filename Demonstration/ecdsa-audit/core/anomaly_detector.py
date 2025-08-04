"""
Module for detecting anomalies in the Rₓ table structure.
Implements methods to identify vulnerabilities based on topological analysis.
"""

from typing import List, Tuple, Dict
import numpy as np

class AnomalyDetector:
    def __init__(self):
        # Thresholds for secure implementation
        self.betti_thresholds = {
            "beta_0": 1.0,  # β₀ should be 1
            "beta_1": 2.0,  # β₁ should be 2
            "beta_2": 1.0   # β₂ should be 1
        }
        self.gamma_threshold = 0.1  # Threshold for damping coefficient
        self.symmetry_threshold = 0.85  # Threshold for symmetry
        self.spiral_threshold = 0.7   # Threshold for spiral structure
    
    def detect_topology_anomaly(self, 
                              betti: Tuple[int, int, int], 
                              gamma: float, 
                              symmetry: float,
                              spiral_info: Dict[str, float]) -> Dict[str, bool]:
        """
        Detects topological anomalies in the Rₓ table.
        
        Args:
            betti: tuple (β₀, β₁, β₂)
            gamma: damping coefficient
            symmetry: symmetry coefficient
            spiral_info: spiral structure information
            
        Returns:
            Dictionary with detected anomalies
        """
        anomalies = {
            "betti_anomaly": False,
            "low_damping": False,
            "broken_symmetry": False,
            "missing_spiral": False,
            "reused_k_attack": False
        }
        
        # Check Betti numbers
        beta_0, beta_1, beta_2 = betti
        if abs(beta_0 - self.betti_thresholds["beta_0"]) > 0.5:
            anomalies["betti_anomaly"] = True
        if abs(beta_1 - self.betti_thresholds["beta_1"]) > 1.0:
            anomalies["betti_anomaly"] = True
        if abs(beta_2 - self.betti_thresholds["beta_2"]) > 0.5:
            anomalies["betti_anomaly"] = True
        
        # Check damping coefficient
        if gamma < self.gamma_threshold:
            anomalies["low_damping"] = True
        
        # Check symmetry
        if symmetry < self.symmetry_threshold:
            anomalies["broken_symmetry"] = True
        
        # Check spiral structure
        if not spiral_info["is_spiral_structure"] or spiral_info["correlation_strength"] < self.spiral_threshold:
            anomalies["missing_spiral"] = True
        
        # Detect reused k attack
        # If multiple anomalies are present, especially low damping and broken symmetry
        if (anomalies["low_damping"] and anomalies["broken_symmetry"]) or anomalies["betti_anomaly"]:
            anomalies["reused_k_attack"] = True
            
        return anomalies
    
    def calculate_anomaly_score(self, 
                             betti: Tuple[int, int, int], 
                             gamma: float, 
                             symmetry: float,
                             spiral_info: Dict[str, float]) -> float:
        """
        Calculates anomaly score (0-1, where 1 is maximally anomalous).
        
        Args:
            betti: tuple (β₀, β₁, β₂)
            gamma: damping coefficient
            symmetry: symmetry coefficient
            spiral_info: spiral structure information
            
        Returns:
            Anomaly score
        """
        # Calculate deviations
        beta_0, beta_1, beta_2 = betti
        
        beta_0_dev = abs(beta_0 - self.betti_thresholds["beta_0"])
        beta_1_dev = abs(beta_1 - self.betti_thresholds["beta_1"])
        beta_2_dev = abs(beta_2 - self.betti_thresholds["beta_2"])
        
        # Normalize deviations (assuming typical ranges)
        beta_0_score = min(1.0, beta_0_dev / 1.0)
        beta_1_score = min(1.0, beta_1_dev / 2.0)
        beta_2_score = min(1.0, beta_2_dev / 1.0)
        
        # Normalize other metrics (0-1 scale)
        gamma_score = max(0.0, 1.0 - gamma / self.gamma_threshold)
        symmetry_score = max(0.0, 1.0 - symmetry / self.symmetry_threshold)
        spiral_score = 0.0 if spiral_info["is_spiral_structure"] else 1.0
        
        # Weighted average
        anomaly_score = (
            0.25 * beta_0_score +
            0.25 * beta_1_score +
            0.15 * beta_2_score +
            0.15 * gamma_score +
            0.10 * symmetry_score +
            0.10 * spiral_score
        )
        
        return min(1.0, anomaly_score)
    
    def detect_vulnerability_level(self, anomaly_score: float) -> str:
        """
        Determines vulnerability level based on anomaly score.
        
        Args:
            anomaly_score: anomaly score (0-1)
            
        Returns:
            Vulnerability level ("safe", "warning", "critical")
        """
        if anomaly_score < 0.3:
            return "safe"
        elif anomaly_score < 0.7:
            return "warning"
        else:
            return "critical"
    
    def analyze_multiple_regions(self, 
                               betti_results: List[Tuple[int, int, int]], 
                               gamma_values: List[float],
                               symmetry_scores: List[float],
                               spiral_infos: List[Dict[str, float]]) -> Dict:
        """
        Analyzes multiple regions to detect overall anomalies.
        
        Args:
            betti_results: list of Betti number results
            gamma_values: list of damping coefficients
            symmetry_scores: list of symmetry scores
            spiral_infos: list of spiral structure information
            
        Returns:
            Analysis results
        """
        # Calculate averages
        avg_beta_0 = np.mean([b[0] for b in betti_results])
        avg_beta_1 = np.mean([b[1] for b in betti_results])
        avg_beta_2 = np.mean([b[2] for b in betti_results])
        avg_gamma = np.mean(gamma_values)
        avg_symmetry = np.mean(symmetry_scores)
        
        # Calculate individual anomaly scores
        anomaly_scores = []
        for i in range(len(betti_results)):
            score = self.calculate_anomaly_score(
                betti_results[i],
                gamma_values[i],
                symmetry_scores[i],
                spiral_infos[i]
            )
            anomaly_scores.append(score)
        
        # Overall anomaly score
        overall_score = np.mean(anomaly_scores)
        
        # Detect overall vulnerabilities
        overall_anomalies = self.detect_topology_anomaly(
            (avg_beta_0, avg_beta_1, avg_beta_2),
            avg_gamma,
            avg_symmetry,
            spiral_infos[0]  # Using first as representative
        )
        
        # Determine vulnerability level
        vulnerability_level = self.detect_vulnerability_level(overall_score)
        
        return {
            "anomaly_scores": anomaly_scores,
            "overall_score": overall_score,
            "vulnerability_level": vulnerability_level,
            "anomalies": overall_anomalies,
            "average_metrics": {
                "beta_0": avg_beta_0,
                "beta_1": avg_beta_1,
                "beta_2": avg_beta_2,
                "gamma": avg_gamma,
                "symmetry": avg_symmetry,
                "spiral_strength": np.mean([s["correlation_strength"] for s in spiral_infos])
            }
        }
