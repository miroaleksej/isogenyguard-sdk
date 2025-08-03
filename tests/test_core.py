"""
IsogenyGuard — Unit tests for core module
Based on Theorem 9 and Table 3 from the scientific work.
"""

import pytest
import numpy as np
from isogenyguard.core import recover_private_key


def test_recover_private_key_success():
    """
    Test private key recovery with known secure parameters (d=27, n=79).
    Data from Table 3: special points with consistent d.
    """
    # Test data from research paper (Table 3: d=27, n=79)
    ur_values = [5, 13, 21, 34, 42]
    uz_values = [23, 52, 3, 35, 64]
    r_values = [41, 41, 41, 41, 41]  # All have the same R_x
    n = 79

    d = recover_private_key(ur_values, uz_values, r_values, n)

    assert d is not None, "Key recovery failed - returned None"
    assert d == 27, f"Expected d=27, but got {d}"


def test_recover_private_key_vulnerable():
    """
    Test key recovery with vulnerable parameters (d=40, n=79).
    Higher F1-score in Table 3 indicates better detectability.
    """
    ur_values = [8, 16, 24, 32, 40]
    uz_values = [15, 30, 45, 60, 75]
    r_values = [22, 22, 22, 22, 22]
    n = 79

    d = recover_private_key(ur_values, uz_values, r_values, n)

    assert d is not None
    assert d == 40, f"Expected d=40, but got {d}"


def test_recover_private_key_edge_cases():
    """
    Test edge cases: empty input, single point, invalid n.
    """
    # Empty inputs
    assert recover_private_key([], [], [], 79) is None

    # Single point (not enough for recovery)
    assert recover_private_key([5], [23], [41], 79) is None

    # Invalid n (zero)
    assert recover_private_key([5, 13], [23, 52], [41, 41], 0) is None

    # Negative n
    assert recover_private_key([5, 13], [23, 52], [41, 41], -1) is None


def test_recover_private_key_with_noise():
    """
    Test robustness to small noise in u_z values.
    Should still recover correct d.
    """
    ur_values = [5, 13, 21, 34, 42]
    uz_clean = [23, 52, 3, 35, 64]
    n = 79
    r_values = [41] * 5

    # Add small noise
    uz_noisy = [uz + np.random.normal(0, 0.5) for uz in uz_clean]
    uz_noisy = [int(round(uz)) % n for uz in uz_noisy]

    d = recover_private_key(ur_values, uz_noisy, r_values, n)

    assert d is not None
    assert d == 27, f"Expected d=27, but got {d}"


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
