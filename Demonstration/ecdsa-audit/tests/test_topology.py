"""
Tests for topological analysis functionality.
Verifies that Betti numbers, spiral wave analysis, and symmetry detection work correctly.
"""

import unittest
import numpy as np
from core.topology_analyzer import TopologyAnalyzer

class TestTopologyAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.analyzer = TopologyAnalyzer()
    
    def test_betti_numbers_simple(self):
        """Test Betti numbers for simple structures"""
        # Test a single point (should have β₀=1, β₁=0, β₂=0)
        single_point = [[5]]
        betti = self.analyzer.compute_betti_numbers(single_point)
        self.assertEqual(betti[0], 1, "β₀ should be 1 for a single point")
        self.assertEqual(betti[1], 0, "β₁ should be 0 for a single point")
        self.assertEqual(betti[2], 0, "β₂ should be 0 for a single point")
        
        # Test a line (should have β₀=1, β₁=0, β₂=0)
        line = [[1]][[2]][[3]][[4]][[5]]
        betti = self.analyzer.compute_betti_numbers(line)
        self.assertEqual(betti[0], 1, "β₀ should be 1 for a line")
        self.assertLessEqual(betti[1], 1, "β₁ should be small for a line")
        self.assertEqual(betti[2], 0, "β₂ should be 0 for a line")
        
        # Test a circle (should have β₀=1, β₁=1, β₂=0)
        circle = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        betti = self.analyzer.compute_betti_numbers(circle)
        self.assertEqual(betti[0], 1, "β₀ should be 1 for a circle")
        self.assertGreaterEqual(betti[1], 1, "β₁ should be at least 1 for a circle")
        self.assertEqual(betti[2], 0, "β₂ should be 0 for a circle")
    
    def test_betti_numbers_torus(self):
        """Test Betti numbers for a torus-like structure"""
        # Create a small torus-like structure (6x6)
        torus = [
            [4, 5, 6, 0, 1, 2],
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 0, 1],
            [6, 0, 1, 2, 3, 4],
            [2, 3, 4, 5, 6, 0],
            [5, 6, 0, 1, 2, 3]
        ]
        
        betti = self.analyzer.compute_betti_numbers(torus)
        
        # For a torus, we expect β₀=1, β₁=2, β₂=1
        self.assertEqual(betti[0], 1, "β₀ should be 1 for a torus")
        self.assertTrue(1 <= betti[1] <= 3, f"β₁ should be around 2 for a torus, got {betti[1]}")
        self.assertEqual(betti[2], 1, "β₂ should be 1 for a torus")
    
    def test_spiral_wave_analysis(self):
        """Test spiral wave analysis with known patterns"""
        # Test a perfect spiral pattern
        spiral = [
            [0, 1, 2, 3, 4],
            [7, 8, 9, 10, 5],
            [6, 5, 4, 3, 2],
            [1, 0, 1, 2, 3],
            [8, 7, 6, 5, 4]
        ]
        gamma = self.analyzer.analyze_spiral_waves(spiral)
        self.assertGreater(gamma, 0.1, "Gamma should be positive for a spiral pattern")
        
        # Test a random pattern (should have low gamma)
        np.random.seed(42)
        random_pattern = np.random.randint(0, 10, (10, 10)).tolist()
        gamma = self.analyzer.analyze_spiral_waves(random_pattern)
        self.assertLess(gamma, 0.1, "Gamma should be low for a random pattern")
        
        # Test a linear pattern (should have medium gamma)
        linear = [[i + j for j in range(10)] for i in range(10)]
        gamma = self.analyzer.analyze_spiral_waves(linear)
        self.assertGreater(gamma, 0, "Gamma should be positive for a linear pattern")
        self.assertLess(gamma, 0.5, "Gamma should not be too high for a linear pattern")
    
    def test_symmetry_detection(self):
        """Test symmetry detection around center points"""
        # Test a perfectly symmetric pattern
        symmetric = [
            [1, 2, 3, 2, 1],
            [4, 5, 6, 5, 4],
            [7, 8, 9, 8, 7],
            [4, 5, 6, 5, 4],
            [1, 2, 3, 2, 1]
        ]
        symmetry = self.analyzer.check_symmetry(symmetric)
        self.assertGreater(symmetry, 0.9, "Symmetry score should be high for a symmetric pattern")
        
        # Test an asymmetric pattern
        asymmetric = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ]
        symmetry = self.analyzer.check_symmetry(asymmetric)
        self.assertLess(symmetry, 0.2, "Symmetry score should be low for an asymmetric pattern")
        
        # Test symmetry in a specific row
        row = [1, 2, 3, 4, 3, 2, 1]
        symmetry = self.analyzer.check_symmetry([row], center_row=0)
        self.assertGreater(symmetry, 0.9, "Symmetry score should be high for a symmetric row")
    
    def test_spiral_structure_detection(self):
        """Test detection of spiral structure properties"""
        # Test a clear spiral structure
        spiral = [
            [0, 1, 2, 3, 4, 5],
            [11, 12, 13, 14, 15, 6],
            [10, 17, 18, 19, 16, 7],
            [9, 8, 7, 6, 5, 8],
            [8, 7, 6, 5, 4, 9],
            [7, 6, 5, 4, 3, 10]
        ]
        spiral_info = self.analyzer.detect_spiral_structure(spiral)
        
        self.assertTrue(spiral_info["is_spiral_structure"], 
                       "Should detect spiral structure in a clear spiral pattern")
        self.assertGreater(spiral_info["correlation_strength"], 0.8,
                          "Spiral structure correlation should be strong")
        self.assertEqual(spiral_info["dominant_slope"], 1,
                        "Dominant slope should be 1 for this spiral pattern")
        
        # Test a random pattern
        np.random.seed(42)
        random_pattern = np.random.randint(0, 10, (10, 10)).tolist()
        spiral_info = self.analyzer.detect_spiral_structure(random_pattern)
        
        self.assertFalse(spiral_info["is_spiral_structure"],
                        "Should not detect spiral structure in a random pattern")
        self.assertLess(spiral_info["correlation_strength"], 0.5,
                       "Spiral structure correlation should be weak in random pattern")
    
    def test_small_curve_topology(self):
        """Test topology analysis on the small curve example (n=7)"""
        small_curve = [
            [4, 5, 6, 0, 1, 2],
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 0, 1],
            [6, 0, 1, 2, 3, 4],
            [2, 3, 4, 5, 6, 0],
            [5, 6, 0, 1, 2, 3]
        ]
        
        # Test Betti numbers
        betti = self.analyzer.compute_betti_numbers(small_curve)
        self.assertEqual(betti[0], 1, "β₀ should be 1 for the small curve table")
        self.assertTrue(1 <= betti[1] <= 3, f"β₁ should be around 2, got {betti[1]}")
        self.assertEqual(betti[2], 1, "β₂ should be 1 for the small curve table")
        
        # Test spiral waves
        gamma = self.analyzer.analyze_spiral_waves(small_curve)
        self.assertGreater(gamma, 0.1, "Gamma should be > 0.1 for a secure implementation")
        
        # Test symmetry
        symmetry = self.analyzer.check_symmetry(small_curve)
        self.assertGreater(symmetry, 0.8, "Symmetry score should be high for the small curve")
        
        # Test spiral structure
        spiral_info = self.analyzer.detect_spiral_structure(small_curve)
        self.assertTrue(spiral_info["is_spiral_structure"], 
                       "Should detect spiral structure in the small curve table")
        self.assertEqual(spiral_info["dominant_slope"], 3,
                        "Dominant slope should match d=3 for the small curve")

if __name__ == '__main__':
    unittest.main()
