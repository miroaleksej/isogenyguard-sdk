"""
IsogenyGuard — Unit tests for topology module
Based on Theorem 21, Theorem 24, and Table 3.
"""

import pytest
import numpy as np
from isogenyguard.topology import check_betti_numbers, calculate_topological_entropy


def test_secure_system_betti_numbers():
    """
    Test Betti numbers for secure system (d=27, n=79).
    According to Table 3 and Theorem 21: β₀=1, β₁=2, β₂=1.
    """
    # j-invariants from secure system (d=27)
    j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82, 0.70, 0.69, 0.73]
    result = check_betti_numbers(j_invariants, n=2)

    # Expected Betti numbers for 2D torus
    assert result["betti_0"] == 1, f"Expected β₀=1, got {result['betti_0']}"
    assert result["betti_1"] == 2, f"Expected β₁=2, got {result['betti_1']}"
    assert result["betti_2"] == 1, f"Expected β₂=1, got {result['betti_2']}"
    assert result["is_secure"] is True, "Secure system incorrectly flagged as vulnerable"

    # Topological entropy should be ~3.3 for d=27 (Table 3)
    assert 3.0 <= result["topological_entropy"] <= 3.6, \
        f"Topological entropy {result['topological_entropy']:.2f} outside expected range for d=27"


def test_vulnerable_system_betti_numbers():
    """
    Test Betti numbers for vulnerable system (low entropy).
    Should have anomalous Betti numbers or low h_top.
    """
    # j-invariants from vulnerable system (uniform, low entropy)
    j_invariants = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    result = check_betti_numbers(j_invariants, n=2)

    # Should not satisfy (1,2,1)
    assert not (result["betti_0"] == 1 and result["betti_1"] == 2 and result["betti_2"] == 1), \
        "Vulnerable system incorrectly flagged as secure"

    # Topological entropy should be low
    assert result["topological_entropy"] < 2.0, \
        f"Topological entropy {result['topological_entropy']:.2f} too high for vulnerable system"


def test_betti_numbers_edge_cases():
    """
    Test Betti number calculation with edge cases.
    """
    # Empty input
    result_empty = check_betti_numbers([])
    assert result_empty["betti_0"] == 0
    assert result_empty["betti_1"] == 0
    assert result_empty["betti_2"] == 0
    assert result_empty["is_secure"] is False

    # Single point
    result_single = check_betti_numbers([0.5])
    assert result_single["betti_0"] == 1
    assert result_single["betti_1"] == 0
    assert result_single["betti_2"] == 0
    assert result_single["is_secure"] is False

    # Two points
    result_two = check_betti_numbers([0.5, 0.6])
    assert result_two["betti_0"] == 1
    assert 0 <= result_two["betti_1"] <= 1
    assert result_two["betti_2"] == 0
    assert result_two["is_secure"] is False


def test_topological_entropy_values():
    """
    Test topological entropy calculation with known values.
    """
    # Secure system (d=27)
    entropy_secure = calculate_topological_entropy([0.72, 0.68, 0.75, 0.65, 0.82])
    assert 3.0 <= entropy_secure <= 3.6, \
        f"Entropy {entropy_secure:.2f} outside expected range for secure system"

    # Vulnerable system
    entropy_vulnerable = calculate_topological_entropy([0.1, 0.11, 0.12, 0.13, 0.14])
    assert entropy_vulnerable < 2.0, \
        f"Entropy {entropy_vulnerable:.2f} too high for vulnerable system"

    # Perfectly uniform distribution
    uniform_data = [i / 10 for i in range(10)]
    entropy_uniform = calculate_topological_entropy(uniform_data)
    assert entropy_uniform < 0.5, \
        f"Uniform data should have low entropy, got {entropy_uniform:.2f}"


def test_betti_numbers_with_noise():
    """
    Test Betti number stability with added noise.
    Small noise should not change topological structure.
    """
    # Base secure system
    base_j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82, 0.70, 0.69, 0.73]

    # Add small noise
    noisy_j_invariants = [x + np.random.normal(0, 0.01) for x in base_j_invariants]
    result_noisy = check_betti_numbers(noisy_j_invariants, n=2)

    # Betti numbers should remain the same
    assert result_noisy["betti_0"] == 1
    assert result_noisy["betti_1"] == 2
    assert result_noisy["betti_2"] == 1
    assert result_noisy["is_secure"] is True


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
