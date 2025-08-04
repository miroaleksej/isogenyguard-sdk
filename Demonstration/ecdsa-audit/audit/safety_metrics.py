"""
Module for calculating safety metrics based on topological analysis.
"""

from typing import List, Tuple, Dict
import numpy as np

class SafetyMetrics:
    def __init__(self):
        # Thresholds for secure implementation
        self.betti_thresholds = {
            "beta_0": 1.0,  # β₀ should be 1
            "beta_1": 2.0,  # β₁ should be 2
            "beta_2": 1.0   # β₂ should be 1
        }
        self.gamma_threshold = 0.1  # Threshold for damping coefficient
        self.symmetry_threshold = 0.85  # Threshold for symmetry
    
    def calculate_betti_anomaly(self, betti: Tuple[int, int, int]) -> float:
        """
        Calculates Betti numbers anomaly.
        
        Args:
            betti: tuple (β₀, β₁, β₂)
            
        Returns:
            Anomaly value (higher means more deviation)
        """
        beta_0, beta_1, beta_2 = betti
        delta_beta_0 = abs(beta_0 - self.betti_thresholds["beta_0"])
        delta_beta_1 = abs(beta_1 - self.betti_thresholds["beta_1"])
        delta_beta_2 = abs(beta_2 - self.betti_thresholds["beta_2"])
        
        # Weighted sum of deviations
        anomaly = (
            0.3 * delta_beta_0 +
            0.5 * delta_beta_1 +
            0.2 * delta_beta_2
        )
        return anomaly
    
    def calculate_safety_score(self, 
                             betti_results: List[Tuple[int, int, int]],
                             gamma_values: List[float],
                             symmetry_scores: List[float]) -> float:
        """
        Calculates overall safety score.
        
        Args:
            betti_results: list of Betti number results for subregions
            gamma_values: list of damping coefficient values
            symmetry_scores: list of symmetry scores
            
        Returns:
            Overall safety score (0-1, where 1 is maximally safe)
        """
        # Calculate average anomalies
        avg_betti_anomaly = np.mean([self.calculate_betti_anomaly(b) for b in betti_results])
        avg_gamma = np.mean(gamma_values)
        avg_symmetry = np.mean(symmetry_scores)
        
        # Normalize Betti anomaly (lower is better)
        normalized_betti = max(0, 1 - avg_betti_anomaly)
        
        # Normalize gamma (should be above threshold)
        normalized_gamma = min(1.0, avg_gamma / self.gamma_threshold) if self.gamma_threshold > 0 else 1.0
        
        # Weight metrics
        safety_score = (
            0.4 * normalized_betti +
            0.4 * min(1.0, normalized_gamma) +
            0.2 * min(1.0, avg_symmetry / self.symmetry_threshold)
        )
        
        return max(0.0, min(1.0, safety_score))
    
    def detect_vulnerability(self, safety_score: float) -> Dict[str, bool]:
        """
        Determines vulnerabilities based on safety score.
        
        Args:
            safety_score: safety score
            
        Returns:
            Dictionary with detected vulnerabilities
        """
        vulnerabilities = {
            "reused_k_attack": False,  # Reused k attack
            "weak_nonce_generation": False,  # Weak nonce generation
            "anomalous_topology": False  # Anomalous topology
        }
        
        # Detect reused k attack
        # If gamma is too low, it may indicate reused k values
        if safety_score < 0.7:
            vulnerabilities["anomalous_topology"] = True
            
        if safety_score < 0.5:
            vulnerabilities["weak_nonce_generation"] = True
            
        # Additional analysis for detecting reused k
        if safety_score < 0.3:
            vulnerabilities["reused_k_attack"] = True
            
        return vulnerabilities
    
    def generate_safety_report(self, 
                             betti_results: List[Tuple[int, int, int]],
                             gamma_values: List[float],
                             symmetry_scores: List[float]) -> Dict:
        """
        Generates a comprehensive safety report.
        
        Args:
            betti_results: list of Betti number results
            gamma_values: list of damping coefficient values
            symmetry_scores: list of symmetry scores
            
        Returns:
            Safety report
        """
        safety_score = self.calculate_safety_score(betti_results, gamma_values, symmetry_scores)
        vulnerabilities = self.detect_vulnerability(safety_score)
        
        # Calculate average metrics
        avg_beta_0 = np.mean([b[0] for b in betti_results])
        avg_beta_1 = np.mean([b[1] for b in betti_results])
        avg_beta_2 = np.mean([b[2] for b in betti_results])
        avg_gamma = np.mean(gamma_values)
        avg_symmetry = np.mean(symmetry_scores)
        
        return {
            "safety_score": safety_score,
            "vulnerabilities": vulnerabilities,
            "betti_analysis": {
                "average": (avg_beta_0, avg_beta_1, avg_beta_2),
                "expected": (1.0, 2.0, 1.0),
                "anomaly": self.calculate_betti_anomaly((avg_beta_0, avg_beta_1, avg_beta_2))
            },
            "spiral_analysis": {
                "average_gamma": avg_gamma,
                "threshold": self.gamma_threshold,
                "is_safe": avg_gamma > self.gamma_threshold
            },
            "symmetry_analysis": {
                "average_symmetry": avg_symmetry,
                "threshold": self.symmetry_threshold,
                "is_safe": avg_symmetry > self.symmetry_threshold
            },
            "recommendations": self.generate_recommendations(vulnerabilities, safety_score)
        }
    
    def generate_recommendations(self, vulnerabilities: Dict[str, bool], safety_score: float) -> List[str]:
        """
        Generates security recommendations based on vulnerabilities.
        
        Args:
            vulnerabilities: detected vulnerabilities
            safety_score: safety score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if safety_score >= 0.8:
            recommendations.append("The ECDSA implementation appears to be secure based on topological analysis.")
            recommendations.append("No critical vulnerabilities were detected.")
        
        if vulnerabilities["anomalous_topology"]:
            recommendations.append("Anomalous topology detected in Rₓ table structure.")
            recommendations.append("This may indicate implementation issues or potential vulnerabilities.")
        
        if vulnerabilities["weak_nonce_generation"]:
            recommendations.append("Weak nonce generation detected.")
            recommendations.append("Ensure that random k values are properly generated with a cryptographically secure random number generator.")
        
        if vulnerabilities["reused_k_attack"]:
            recommendations.append("High probability of reused k values detected.")
            recommendations.append("Reused k values can lead to private key recovery. Immediately rotate keys and fix nonce generation.")
        
        if not any(vulnerabilities.values()):
            recommendations.append("The implementation shows expected topological properties for a secure ECDSA system.")
        
        return recommendations
