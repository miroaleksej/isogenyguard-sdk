"""
Tests for table generation functionality.
Verifies that subregions are generated correctly and follow expected properties.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock
from core.curve_operations import CurveOperations
from core.table_generator import TableGenerator

class TestTableGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Mock CurveOperations
        self.curve_ops = MagicMock()
        self.curve_ops.n = 100  # Mock curve order
        
        # Set up mock for compute_subregion
        def mock_compute_subregion(Q, u_r_min, u_r_max, u_z_min, u_z_max):
            # Create a deterministic pattern for testing
            region = []
            for u_r in range(u_r_min, u_r_max):
                row = []
                for u_z in range(u_z_min, u_z_max):
                    # Create a pattern where R_x = (u_r + u_z) % 10
                    row.append((u_r + u_z) % 10)
                region.append(row)
            return region
            
        self.curve_ops.compute_subregion = MagicMock(side_effect=mock_compute_subregion)
        
        # Initialize TableGenerator with mock
        self.table_generator = TableGenerator(self.curve_ops)
        
        # Create a mock public key
        self.Q = MagicMock()
    
    def test_random_regions_generation(self):
        """Test generation of random regions"""
        regions = self.table_generator.generate_random_regions(self.Q, num_regions=5, region_size=10)
        
        # Check number of regions
        self.assertEqual(len(regions), 5, "Should generate exactly 5 regions")
        
        # Check region dimensions
        for region in regions:
            self.assertEqual(len(region), 10, "Each region should have 10 rows")
            self.assertEqual(len(region[0]), 10, "Each region should have 10 columns")
        
        # Check that regions are different (not the same pattern)
        unique_regions = set(str(region) for region in regions)
        self.assertGreater(len(unique_regions), 1, "Regions should be different")
    
    def test_optimal_regions_generation(self):
        """Test generation of optimal regions around d_opt ≈ n/2"""
        regions = self.table_generator.generate_optimal_regions(self.Q, num_regions=5, region_size=10)
        
        # Check number of regions
        self.assertEqual(len(regions), 5, "Should generate exactly 5 regions")
        
        # Check that regions are centered around n/2
        n_half = self.curve_ops.n // 2
        for region in regions:
            # The first region should be around n/2
            # Note: We can't directly check the position, but we can verify the pattern
            self.assertEqual(len(region), 10, "Each region should have 10 rows")
            self.assertEqual(len(region[0]), 10, "Each region should have 10 columns")
    
    def test_symmetry_regions_generation(self):
        """Test generation of regions around symmetry points"""
        regions_with_positions = self.table_generator.generate_symmetry_regions(
            self.Q, num_regions=5, region_size=10
        )
        
        # Check number of regions
        self.assertEqual(len(regions_with_positions), 5, "Should generate exactly 5 regions")
        
        # Check structure of returned data
        for region_data in regions_with_positions:
            region, u_r, u_z_center = region_data
            self.assertEqual(len(region), 10, "Each region should have 10 rows")
            self.assertEqual(len(region[0]), 10, "Each region should have 10 columns")
            self.assertTrue(1 <= u_r < self.curve_ops.n, "u_r should be within valid range")
            self.assertTrue(1 <= u_z_center < self.curve_ops.n, "u_z_center should be within valid range")
    
    def test_region_size_consistency(self):
        """Test that regions have consistent size"""
        for size in [5, 10, 20, 50]:
            regions = self.table_generator.generate_random_regions(self.Q, num_regions=3, region_size=size)
            for region in regions:
                self.assertEqual(len(region), size, f"Region should have {size} rows")
                self.assertEqual(len(region[0]), size, f"Region should have {size} columns")
    
    def test_edge_case_regions(self):
        """Test generation of regions at the edges of the table"""
        # Mock a smaller curve order for edge case testing
        self.curve_ops.n = 20
        
        # Test region at the very beginning
        region1 = self.table_generator.generate_random_regions(
            self.Q, num_regions=1, region_size=10
        )[0]
        
        # Test region at the very end
        self.curve_ops.compute_subregion = MagicMock(return_value=[[1]*10]*10)
        region2 = self.table_generator.generate_random_regions(
            self.Q, num_regions=1, region_size=10
        )[0]
        
        # Both should be valid regions
        self.assertEqual(len(region1), 10)
        self.assertEqual(len(region2), 10)
    
    def test_cyclic_property_in_regions(self):
        """Test that generated regions maintain the cyclic property"""
        # Create a custom mock that simulates the cyclic property
        def cyclic_mock_compute_subregion(Q, u_r_min, u_r_max, u_z_min, u_z_max):
            region = []
            for u_r in range(u_r_min, u_r_max):
                row = []
                for u_z in range(u_z_min, u_z_max):
                    # Create a pattern with cyclic shift of d=3 between rows
                    k = (u_r * 3 + u_z) % 7
                    row.append(k)
                region.append(row)
            return region
        
        self.curve_ops.compute_subregion = MagicMock(side_effect=cyclic_mock_compute_subregion)
        
        # Generate a region
        region = self.table_generator.generate_random_regions(self.Q, num_regions=1, region_size=6)[0]
        
        # Check that each row is a cyclic shift of the previous row by 3 positions
        for i in range(1, len(region)):
            # Shift the previous row by 3 positions to the right (cyclic)
            shifted = region[i-1][-3:] + region[i-1][:-3]
            self.assertEqual(region[i], shifted,
                            f"Row {i} is not a cyclic shift of row {i-1} by 3 positions")

if __name__ == '__main__':
    unittest.main()
