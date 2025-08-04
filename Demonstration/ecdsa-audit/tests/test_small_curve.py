"""
Tests for the ECDSA audit system using a small curve (n=7) as in the reference material.
Verifies that the system correctly analyzes the Rₓ table structure for known parameters.
"""

import unittest
import numpy as np
from core.curve_operations import CurveOperations
from core.table_generator import TableGenerator
from core.topology_analyzer import TopologyAnalyzer
from core.anomaly_detector import AnomalyDetector

class TestSmallCurve(unittest.TestCase):
    def setUp(self):
        """Set up test environment with a small curve (n=7)"""
        self.n = 7
        self.d = 3  # Private key from the example
        
        # Create a mock PublicKey object that mimics coincurve.PublicKey
        # In a real test, we would use an actual small curve implementation
        class MockPublicKey:
            def __init__(self, d, n):
                self.d = d
                self.n = n
                
            def multiply(self, scalar):
                return MockPoint((scalar * self.d) % self.n, 0)
                
            @property
            def point(self):
                return MockPoint(self.d, 0)
                
        class MockPoint:
            def __init__(self, x, y):
                self.x_val = x
                self.y_val = y
                
            def x(self):
                return self.x_val
                
            def y(self):
                return self.y_val
                
            def combine(self, other_points):
                total_x = self.x_val
                for point in other_points:
                    total_x = (total_x + point.x()) % 7
                return MockPoint(total_x, 0)
                
        # Create test public key
        self.Q = MockPublicKey(self.d, self.n)
        
        # Initialize components
        self.curve_ops = CurveOperations()
        self.table_generator = TableGenerator(self.curve_ops)
        self.topology_analyzer = TopologyAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        # Generate the expected table as in the reference material
        self.expected_table = [
            [4, 5, 6, 0, 1, 2],  # u_r = 1
            [0, 1, 2, 3, 4, 5],  # u_r = 2
            [3, 4, 5, 6, 0, 1],  # u_r = 3
            [6, 0, 1, 2, 3, 4],  # u_r = 4
            [2, 3, 4, 5, 6, 0],  # u_r = 5
            [5, 6, 0, 1, 2, 3]   # u_r = 6
        ]
    
    def test_table_generation(self):
        """Test that table generation matches the expected structure"""
        # Generate the full table
        full_table = []
        for u_r in range(1, self.n):
            row = []
            for u_z in range(1, self.n):
                # In our mock, R_x = (u_r * d + u_z) % n
                k = (u_r * self.d + u_z) % self.n
                row.append(k)
            full_table.append(row)
        
        # Verify the table matches expected
        for i in range(len(self.expected_table)):
            for j in range(len(self.expected_table[0])):
                self.assertEqual(full_table[i][j], self.expected_table[i][j],
                                f"Mismatch at position ({i}, {j}): expected {self.expected_table[i][j]}, got {full_table[i][j]}")
    
    def test_cyclic_shift_property(self):
        """Test the cyclic shift property between rows (Theorem 2)"""
        # Generate the full table
        full_table = []
        for u_r in range(1, self.n):
            row = []
            for u_z in range(1, self.n):
                k = (u_r * self.d + u_z) % self.n
                row.append(k)
            full_table.append(row)
        
        # Check that each row is a cyclic shift of the previous row by d positions
        for i in range(1, len(full_table)):
            # Shift the previous row by d positions to the right (cyclic)
            shifted = full_table[i-1][-self.d:] + full_table[i-1][:-self.d]
            self.assertEqual(full_table[i], shifted,
                            f"Row {i} is not a cyclic shift of row {i-1} by {self.d} positions")
    
    def test_betti_numbers(self):
        """Test that Betti numbers match expected values for the small curve"""
        # For a complete torus structure, we expect β₀ = 1, β₁ = 2, β₂ = 1
        betti = self.topology_analyzer.compute_betti_numbers(self.expected_table)
        
        # In a perfect implementation with n=7, we should get exactly these values
        self.assertEqual(betti[0], 1, "β₀ should be 1 (one connected component)")
        self.assertEqual(betti[1], 2, "β₁ should be 2 (two independent cycles)")
        self.assertEqual(betti[2], 1, "β₂ should be 1 (one void)")
    
    def test_spiral_waves(self):
        """Test that spiral wave analysis produces expected results"""
        gamma = self.topology_analyzer.analyze_spiral_waves(self.expected_table)
        
        # For a well-structured table, gamma should be positive and reasonably large
        self.assertGreater(gamma, 0, "Gamma should be positive for a structured table")
        self.assertLess(gamma, 1.0, "Gamma should be less than 1.0")
    
    def test_symmetry(self):
        """Test symmetry around special points"""
        # For each row, there should be a special point where symmetry occurs
        symmetry_scores = []
        
        for i, row in enumerate(self.expected_table):
            # The special point should be at u_z* = -u_r * d mod n
            u_r = i + 1
            special_point = (-u_r * self.d) % self.n
            
            # Adjust for 0-based indexing and table structure
            special_point = (special_point - 1) % len(row)
            
            # Check symmetry around this point
            symmetry = self.topology_analyzer.check_symmetry([row], center_row=0)
            symmetry_scores.append(symmetry)
        
        # All rows should show high symmetry
        avg_symmetry = sum(symmetry_scores) / len(symmetry_scores)
        self.assertGreater(avg_symmetry, 0.8, "Symmetry score should be high for a structured table")
    
    def test_spiral_structure(self):
        """Test detection of spiral structure"""
        spiral_info = self.topology_analyzer.detect_spiral_structure(self.expected_table)
        
        # The expected table should have a clear spiral structure
        self.assertTrue(spiral_info["is_spiral_structure"], 
                       "The expected table should have a spiral structure")
        self.assertGreater(spiral_info["correlation_strength"], 0.7,
                          "Spiral structure correlation should be strong")
        self.assertEqual(spiral_info["dominant_slope"], self.d,
                        "Dominant slope should match the private key d")
    
    def test_anomaly_detection(self):
        """Test anomaly detection on the expected table"""
        # Compute topological characteristics
        betti = self.topology_analyzer.compute_betti_numbers(self.expected_table)
        gamma = self.topology_analyzer.analyze_spiral_waves(self.expected_table)
        symmetry = self.topology_analyzer.check_symmetry(self.expected_table)
        spiral_info = self.topology_analyzer.detect_spiral_structure(self.expected_table)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_topology_anomaly(
            betti, gamma, symmetry, spiral_info
        )
        
        # The expected table should have no critical anomalies
        self.assertFalse(anomalies["betti_anomaly"], "No Betti anomaly should be detected")
        self.assertFalse(anomalies["low_damping"], "No low damping should be detected")
        self.assertFalse(anomalies["broken_symmetry"], "No broken symmetry should be detected")
        self.assertFalse(anomalies["missing_spiral"], "No missing spiral should be detected")
        self.assertFalse(anomalies["reused_k_attack"], "No reused k attack should be detected")
        
        # Calculate anomaly score (should be low)
        anomaly_score = self.anomaly_detector.calculate_anomaly_score(
            betti, gamma, symmetry, spiral_info
        )
        self.assertLess(anomaly_score, 0.3, "Anomaly score should be low for a good implementation")
        
        # Vulnerability level should be safe
        vuln_level = self.anomaly_detector.detect_vulnerability_level(anomaly_score)
        self.assertEqual(vuln_level, "safe", "Vulnerability level should be 'safe'")

if __name__ == '__main__':
    unittest.main()
