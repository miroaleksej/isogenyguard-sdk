"""
IsogenyGuard — Unit tests for topology module
Based on Theorem 21 (Torus structure), Theorem 24 (Topological entropy), and Table 3.
"""

import pytest
import numpy as np
from unittest.mock import patch
from isogenyguard.topology import check_betti_numbers, calculate_topological_entropy


def generate_j_invariants_for_d(d: int, n: int = 79, size: int = 50) -> list:
    """
    Simulates j-invariants from isogeny lattice for a given private key d.
    In real use: extracted from network or side-channel.
    For d=27, expected h_top ≈ 3.3 (Table 3).

    Args:
        d (int): Private key
        n (int): Group order
        size (int): Number of points to generate

    Returns:
        list: Simulated j-invariants
    """
    np.random.seed(d)  # Deterministic for testing
    base = d * 10
    return [float(base + np.random.normal(0, 5)) for _ in range(size)]


def test_secure_system_betti_numbers():
    """
    Test Betti numbers for a secure system with d=27, n=79.
    According to Table 3 and Theorem 21: β₀=1, β₁=2, β₂=1.
    """
    j_invariants = generate_j_invariants_for_d(d=27, size=40)
    result = check_betti_numbers(j_invariants, n=2)

    assert result["betti_0"] == 1, f"Expected β₀=1, got {result['betti_0']}"
    assert result["betti_1"] == 2, f"Expected β₁=2, got {result['betti_1']}"
    assert result["betti_2"] == 1, f"Expected β₂=1, got {result['betti_2']}"
    assert result["is_secure"] is True, "Secure system should be flagged as secure"
    assert 3.0 <= result["topological_entropy"] <= 3.6, \
        f"Entropy {result['topological_entropy']:.2f} not in expected range for d=27"


def test_vulnerable_system_anomalous_betti():
    """
    Test system with anomalous Betti numbers (vulnerable).
    According to Corollary 15: deviation from (1,2,1) → vulnerable.
    """
    # Simulate weak system: uniform j-invariants (low entropy, no structure)
    j_invariants = [0.1] * 50

    result = check_betti_numbers(j_invariants, n=2)

    assert result["betti_0"] == 1
    assert result["betti_1"] == 0  # No cycles
    assert result["betti_2"] == 0
    assert result["is_secure"] is False, "System with β₁=0 should be flagged as vulnerable"


def test_topological_entropy_secure():
    """
    Test topological entropy for secure key (d=27).
    According to Table 3: h_top ≈ 3.3 for d=27.
    """
    j_invariants = generate_j_invariants_for_d(d=27, size=30)
    entropy = calculate_topological_entropy(j_invariants)

    assert 3.0 <= entropy <= 3.6, f"Expected h_top ≈ 3.3, got {entropy:.2f}"


def test_topological_entropy_vulnerable():
    """
    Test low topological entropy → vulnerable system.
    According to Theorem 24: h_top < log n - δ → vulnerable.
    """
    # Simulate vulnerable system: very predictable j-invariants
    j_invariants = [0.5, 0.51, 0.52, 0.53, 0.54, 0.55]
    entropy = calculate_topological_entropy(j_invariants)

    assert entropy < 1.0, f"Low-entropy system has h_top={entropy:.2f}, should be near 0"
    assert entropy >= 0.0


def test_topological_entropy_extreme_cases():
    """
    Test entropy for extreme cases: uniform vs. clustered data.
    - Uniform data: high entropy
    - Clustered data: low entropy
    """
    # Uniform distribution → high entropy
    uniform_data = [i / 10 for i in range(10)]
    entropy_uniform = calculate_topological_entropy(uniform_data)
    assert entropy_uniform > 2.0, "Uniform data should have high entropy"

    # Clustered data → low entropy
    clustered_data = [0.1, 0.11, 0.12, 0.88, 0.89, 0.90]
    entropy_clustered = calculate_topological_entropy(clustered_data)
    assert entropy_clustered < entropy_uniform, \
        "Clustered data should have lower entropy than uniform data"


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


def test_betti_numbers_dimension_parameter():
    """
    Test Betti numbers with different expected dimensions (n).
    For ECDSA: n=2 → expect (1,2,1)
    For CSIDH-like: n=3 → expect different structure
    """
    j_invariants = generate_j_invariants_for_d(d=27, size=30)

    # Test for ECDSA (n=2)
    result_ecdsa = check_betti_numbers(j_invariants, n=2)
    assert result_ecdsa["betti_0"] == 1
    assert result_ecdsa["betti_1"] == 2
    assert result_ecdsa["betti_2"] == 1
    assert result_ecdsa["is_secure"] is True

    # Test for higher dimension (e.g., CSIDH-512, n=78 → expect β₁=77)
    result_high_dim = check_betti_numbers(j_invariants, n=77)
    # In real use, this would require proper (n-1)-torus model
    assert result_high_dim["betti_0"] == 1  # Always one connected component


def test_topological_security_metrics():
    """
    Test overall security metrics based on topological properties.
    Compares secure vs. vulnerable systems.
    """
    # Secure system (d=27)
    secure_j_invariants = generate_j_invariants_for_d(d=27, size=40)
    secure_result = check_betti_numbers(secure_j_invariants, n=2)

    # Vulnerable system (d=1, low entropy)
    vulnerable_j_invariants = generate_j_invariants_for_d(d=1, size=40)
    vulnerable_result = check_betti_numbers(vulnerable_j_invariants, n=2)

    # Verify security flags
    assert secure_result["is_secure"] is True
    assert vulnerable_result["is_secure"] is False

    # Verify entropy comparison
    assert secure_result["topological_entropy"] > vulnerable_result["topological_entropy"], \
        "Secure system should have higher entropy"


def test_noise_robustness():
    """
    Test robustness of Betti numbers to small noise in j-invariants.
    Small noise should not change topological structure.
    """
    base_data = generate_j_invariants_for_d(d=27, size=30)
    noisy_data = [x + np.random.normal(0, 0.1) for x in base_data]

    result_clean = check_betti_numbers(base_data, n=2)
    result_noisy = check_betti_numbers(noisy_data, n=2)

    # Betti numbers should remain stable
    assert result_clean["betti_0"] == result_noisy["betti_0"]
    assert abs(result_clean["betti_1"] - result_noisy["betti_1"]) <= 1
    assert result_clean["is_secure"] == result_noisy["is_secure"]


if __name__ == "__main__":
    import sys
    print("Running IsogenyGuard topology tests...\n")
    pytest.main([__file__, "-v"] + sys.argv[1:])
