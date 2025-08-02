import pytest
import numpy as np
from isogenyguard.core import recover_private_key, check_special_points

def test_recover_private_key_success():
    """Test private key recovery with known secure parameters (d=27, n=79)"""
    # Test data from research paper (Table 3: d=27, n=79)
    ur_values = [5, 13, 21, 34, 42]
    uz_values = [23, 52, 3, 35, 64]
    r_values = [41, 41, 41, 41, 41]  # All have the same R_x
    n = 79
    
    d = recover_private_key(ur_values, uz_values, r_values, n)
    
    assert d is not None, "Key recovery failed - returned None"
    assert d == 27, f"Expected d=27, but got {d}"
    assert 0 <= d < n, f"Recovered key {d} is outside valid range [0, {n-1}]"

def test_recover_private_key_vulnerable():
    """Test key recovery with vulnerable parameters (d=40, n=79)"""
    # Test data with d=40 (higher F1-score in Table 3)
    ur_values = [8, 16, 24, 32, 40]
    uz_values = [15, 30, 45, 60, 75]
    r_values = [22, 22, 22, 22, 22]
    n = 79
    
    d = recover_private_key(ur_values, uz_values, r_values, n)
    
    assert d is not None
    assert d == 40, f"Expected d=40, but got {d}"

def test_recover_private_key_edge_cases():
    """Test edge cases for key recovery"""
    # Empty inputs
    assert recover_private_key([], [], [], 79) is None
    
    # Single point (not enough for recovery)
    assert recover_private_key([5], [23], [41], 79) is None
    
    # Invalid n (zero)
    assert recover_private_key([5, 13], [23, 52], [41, 41], 0) is None

def test_check_special_points():
    """Test detection of special points in signature data"""
    # Test data with known special points (d=27, n=79)
    ur_values = [5, 13, 21, 34, 42]
    uz_values = [23, 52, 3, 35, 64]
    n = 79
    
    special_points = check_special_points(ur_values, uz_values, n)
    
    # According to Theorem 9, all points should be special for consistent data
    assert len(special_points) == 4, f"Expected 4 special points, found {len(special_points)}"
    assert all(1 <= idx < len(ur_values) for idx in special_points), "Invalid special point indices"

def test_check_special_points_vulnerable():
    """Test special point detection with vulnerable parameters"""
    # Test data with d=1 (low F1-score in Table 3)
    ur_values = [1, 2, 3, 4, 5]
    uz_values = [1, 2, 3, 4, 5]
    n = 79
    
    special_points = check_special_points(ur_values, uz_values, n)
    
    # For d=1, most points should not be special
    assert len(special_points) < 3, f"Too many special points detected for vulnerable system: {len(special_points)}"

def test_check_special_points_no_points():
    """Test special point detection with no special points"""
    # Test data with random values (no consistent d)
    ur_values = [5, 10, 15, 20, 25]
    uz_values = [10, 30, 50, 70, 15]
    n = 79
    
    special_points = check_special_points(ur_values, uz_values, n)
    
    assert len(special_points) == 0, f"Detected special points where none should exist: {special_points}"

def test_recover_private_key_with_special_points():
    """Test key recovery using only special points"""
    # Test data with some special points
    ur_values = [5, 13, 21, 34, 42, 50, 58]
    uz_values = [23, 52, 3, 35, 64, 12, 41]
    r_values = [41, 41, 41, 41, 41, 22, 22]
    n = 79
    
    # First identify special points
    special_indices = check_special_points(ur_values, uz_values, n)
    
    # Filter data to only include special points
    special_ur = [ur_values[i] for i in special_indices]
    special_uz = [uz_values[i] for i in special_indices]
    special_r = [r_values[i] for i in special_indices]
    
    # Recover key using only special points
    d = recover_private_key(special_ur, special_uz, special_r, n)
    
    assert d is not None
    assert d == 27, f"Key recovery using special points failed. Expected 27, got {d}"
