"""
Tests for anomaly detection functionality.
Verifies that vulnerabilities are correctly identified based on topological analysis.
"""

import unittest
import numpy as np
from core.topology_analyzer import TopologyAnalyzer
from core.anomaly_detector import AnomalyDetector

class TestAnomalyDetection(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.analyzer = TopologyAnalyzer()
        self.detector = AnomalyDetector()
    
    def test_safe_implementation_detection(self):
        """Test detection of a safe implementation"""
        # Create a table that mimics a safe ECDSA implementation
        safe_table = [
            [4, 5, 6, 0, 1, 2],
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 0, 1],
            [6, 0, 1, 2, 3, 4],
            [2, 3, 4, 5, 6, 0],
            [5, 6, 0, 1, 2, 3]
        ]
        
        # Analyze the table
        betti = self.analyzer.compute_betti_numbers(safe_table)
        gamma = self.analyzer.analyze_spiral_waves(safe_table)
        symmetry = self.analyzer.check_symmetry(safe_table)
        spiral_info = self.analyzer.detect_spiral_structure(safe_table)
        
        # Detect anomalies
        anomalies = self.detector.detect_topology_anomaly(betti, gamma, symmetry, spiral_info)
        anomaly_score = self.detector.calculate_anomaly_score(betti, gamma, symmetry, spiral_info)
        vuln_level = self.detector.detect_vulnerability_level(anomaly_score)
        
        # Verify results
        self.assertFalse(anomalies["betti_anomaly"], "No Betti anomaly should be detected in a safe implementation")
        self.assertFalse(anomalies["low_damping"], "No low damping should be detected in a safe implementation")
        self.assertFalse(anomalies["broken_symmetry"], "No broken symmetry should be detected in a safe implementation")
        self.assertFalse(anomalies["missing_spiral"], "No missing spiral should be detected in a safe implementation")
        self.assertFalse(anomalies["reused_k_attack"], "No reused k attack should be detected in a safe implementation")
        
        self.assertLess(anomaly_score, 0.3, "Anomaly score should be low for a safe implementation")
        self.assertEqual(vuln_level, "safe", "Vulnerability level should be 'safe' for a safe implementation")
    
    def test_reused_k_attack_detection(self):
        """Test detection of reused k attack vulnerability"""
        # Create a table with reused k values (repeating pattern)
        # This simulates what happens when the same k is used for multiple signatures
        reused_k_table = [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6]
        ]
        
        # Analyze the table
        betti = self.analyzer.compute_betti_numbers(reused_k_table)
        gamma = self.analyzer.analyze_spiral_waves(reused_k_table)
        symmetry = self.analyzer.check_symmetry(reused_k_table)
        spiral_info = self.analyzer.detect_spiral_structure(reused_k_table)
        
        # Detect anomalies
        anomalies = self.detector.detect_topology_anomaly(betti, gamma, symmetry, spiral_info)
        anomaly_score = self.detector.calculate_anomaly_score(betti, gamma, symmetry, spiral_info)
        vuln_level = self.detector.detect_vulnerability_level(anomaly_score)
        
        # Verify results
        self.assertTrue(anomalies["betti_anomaly"], "Betti anomaly should be detected in reused k attack")
        self.assertTrue(anomalies["low_damping"], "Low damping should be detected in reused k attack")
        self.assertTrue(anomalies["broken_symmetry"], "Broken symmetry should be detected in reused k attack")
        self.assertTrue(anomalies["missing_spiral"], "Missing spiral should be detected in reused k attack")
        self.assertTrue(anomalies["reused_k_attack"], "Reused k attack should be detected")
        
        self.assertGreater(anomaly_score, 0.7, "Anomaly score should be high for reused k attack")
        self.assertEqual(vuln_level, "critical", "Vulnerability level should be 'critical' for reused k attack")
    
    def test_weak_nonce_generation_detection(self):
        """Test detection of weak nonce generation"""
        # Create a table with weak nonce generation (linear pattern)
        weak_nonce_table = [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 0],
            [3, 4, 5, 6, 0, 1],
            [4, 5, 6, 0, 1, 2],
            [5, 6, 0, 1, 2, 3]
        ]
        
        # Analyze the table
        betti = self.analyzer.compute_betti_numbers(weak_nonce_table)
        gamma = self.analyzer.analyze_spiral_waves(weak_nonce_table)
        symmetry = self.analyzer.check_symmetry(weak_nonce_table)
        spiral_info = self.analyzer.detect_spiral_structure(weak_nonce_table)
        
        # Detect anomalies
        anomalies = self.detector.detect_topology_anomaly(betti, gamma, symmetry, spiral_info)
        anomaly_score = self.detector.calculate_anomaly_score(betti, gamma, symmetry, spiral_info)
        vuln_level = self.detector.detect_vulnerability_level(anomaly_score)
        
        # Verify results
        self.assertTrue(anomalies["betti_anomaly"], "Betti anomaly should be detected in weak nonce generation")
        self.assertTrue(anomalies["low_damping"], "Low damping should be detected in weak nonce generation")
        self.assertTrue(anomalies["broken_symmetry"], "Broken symmetry should be detected in weak nonce generation")
        self.assertFalse(anomalies["missing_spiral"], "Spiral structure might still be present")
        self.assertFalse(anomalies["reused_k_attack"], "Not a full reused k attack")
        
        self.assertGreater(0.7, anomaly_score, "Anomaly score should be medium for weak nonce generation")
        self.assertLess(0.3, anomaly_score, "Anomaly score should be medium for weak nonce generation")
        self.assertEqual(vuln_level, "warning", "Vulnerability level should be 'warning' for weak nonce generation")
    
    def test_multiple_regions_analysis(self):
        """Test analysis of multiple regions"""
        # Create a mix of safe and vulnerable regions
        safe_table = [
            [4, 5, 6, 0, 1, 2],
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 0, 1],
            [6, 0, 1, 2, 3, 4],
            [2, 3, 4, 5, 6, 0],
            [5, 6, 0, 1, 2, 3]
        ]
        
        reused_k_table = [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6]
        ]
        
        # Analyze multiple regions
        betti_results = []
        gamma_values = []
        symmetry_scores = []
        spiral_infos = []
        
        # 3 safe regions
        for _ in range(3):
            betti = self.analyzer.compute_betti_numbers(safe_table)
            gamma = self.analyzer.analyze_spiral_waves(safe_table)
            symmetry = self.analyzer.check_symmetry(safe_table)
            spiral_info = self.analyzer.detect_spiral_structure(safe_table)
            
            betti_results.append(betti)
            gamma_values.append(gamma)
            symmetry_scores.append(symmetry)
            spiral_infos.append(spiral_info)
        
        # 2 vulnerable regions
        for _ in range(2):
            betti = self.analyzer.compute_betti_numbers(reused_k_table)
            gamma = self.analyzer.analyze_spiral_waves(reused_k_table)
            symmetry = self.analyzer.check_symmetry(reused_k_table)
            spiral_info = self.analyzer.detect_spiral_structure(reused_k_table)
            
            betti_results.append(betti)
            gamma_values.append(gamma)
            symmetry_scores.append(symmetry)
            spiral_infos.append(spiral_info)
        
        # Analyze multiple regions
        analysis = self.detector.analyze_multiple_regions(
            betti_results, gamma_values, symmetry_scores, spiral_infos
        )
        
        # Verify results
        self.assertEqual(analysis["vulnerability_level"], "warning",
                        "Mixed regions should result in 'warning' level")
        self.assertTrue(0.3 < analysis["overall_score"] < 0.7,
                       "Anomaly score should be medium for mixed regions")
        
        # Most anomalies should be detected
        self.assertTrue(analysis["anomalies"]["betti_anomaly"])
        self.assertTrue(analysis["anomalies"]["low_damping"])
        self.assertTrue(analysis["anomalies"]["broken_symmetry"])
        self.assertTrue(analysis["anomalies"]["missing_spiral"])
        self.assertFalse(analysis["anomalies"]["reused_k_attack"])  # Not dominant enough
    
    def test_edge_cases(self):
        """Test anomaly detection on edge cases"""
        # Test empty table
        empty_table = []
        with self.assertRaises(Exception):
            self.analyzer.compute_betti_numbers(empty_table)
        
        # Test single row
        single_row = [[1]][[2]][[3]][[4]][[5]]
        betti = self.analyzer.compute_betti_numbers(single_row)
        gamma = self.analyzer.analyze_spiral_waves(single_row)
        symmetry = self.analyzer.check_symmetry(single_row)
        spiral_info = self.analyzer.detect_spiral_structure(single_row)
        
        anomaly_score = self.detector.calculate_anomaly_score(
            betti, gamma, symmetry, spiral_info
        )
        self.assertLess(anomaly_score, 1.0, "Anomaly score should be calculable for single row")
        
        # Test single column
        single_col = [[1], [2], [3], [4], [5]]
        betti = self.analyzer.compute_betti_numbers(single_col)
        gamma = self.analyzer.analyze_spiral_waves(single_col)
        symmetry = self.analyzer.check_symmetry(single_col)
        spiral_info = self.analyzer.detect_spiral_structure(single_col)
        
        anomaly_score = self.detector.calculate_anomaly_score(
            betti, gamma, symmetry, spiral_info
        )
        self.assertLess(anomaly_score, 1.0, "Anomaly score should be calculable for single column")

if __name__ == '__main__':
    unittest.main()
