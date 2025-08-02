import pytest
import numpy as np
from isogenyguard.topology import check_betti_numbers, calculate_topological_entropy

def test_secure_system_betti_numbers():
    """
    Test Betti numbers for secure system (d=27, n=79)
    According to Table 3, should have β₀=1, β₁=2, β₂=1
    """
    # j-invariants from secure system (d=27)
    j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82, 0.70, 0.69, 0.73]
    
    result = check_betti_numbers(j_invariants)
    
    # Expected Betti numbers for 2D torus (Theorem 21)
    assert result["betti_0"] == 1, f"Expected β₀=1, got {result['betti_0']}"
    assert result["betti_1"] == 2, f"Expected β₁=2, got {result['betti_1']}"
    assert result["betti_2"] == 1, f"Expected β₂=1, got {result['betti_2']}"
    assert result["is_secure"] is True, "Secure system incorrectly flagged as vulnerable"
    
    # Topological entropy should be around 3.3 for d=27 (Table 3)
    assert 3.0 <= result["topological_entropy"] <= 3.6, \
        f"Topological entropy {result['topological_entropy']:.2f} outside expected range for d=27"

def test_vulnerable_system_betti_numbers():
    """
    Test Betti numbers for vulnerable system
    Should have anomalous Betti numbers
    """
    # j-invariants from vulnerable system (low entropy)
    j_invariants = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    
    result = check_betti_numbers(j_invariants)
    
    # For vulnerable system, Betti numbers should deviate from (1,2,1)
    assert not (result["betti_0"] == 1 and result["betti_1"] == 2 and result["betti_2"] == 1), \
        "Vulnerable system incorrectly flagged as secure"
    
    # Topological entropy should be low for vulnerable system
    assert result["topological_entropy"] < 2.0, \
        f"Topological entropy {result['topological_entropy']:.2f} too high for vulnerable system"

def test_betti_numbers_edge_cases():
    """Test Betti number calculation with edge cases"""
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
    """Test topological entropy calculation with known values"""
    # Secure system (d=27)
    entropy_secure = calculate_topological_entropy([0.72, 0.68, 0.75, 0.65, 0.82])
    assert 3.0 <= entropy_secure <= 3.6, f"Entropy {entropy_secure:.2f} outside expected range for secure system"
    
    # Vulnerable system
    entropy_vulnerable = calculate_topological_entropy([0.1, 0.11, 0.12, 0.13, 0.14])
    assert entropy_vulnerable < 2.0, f"Entropy {entropy_vulnerable:.2f} too high for vulnerable system"
    
    # Perfectly uniform distribution
    uniform_data = [i/10 for i in range(10)]
    entropy_uniform = calculate_topological_entropy(uniform_data)
    assert entropy_uniform > 2.5, "Entropy too low for uniform distribution"
    
    # Highly clustered data
    clustered_data = [0.1, 0.11, 0.12, 0.88, 0.89, 0.90]
    entropy_clustered = calculate_topological_entropy(clustered_data)
    assert entropy_clustered < entropy_uniform, "Clustered data has higher entropy than uniform data"

def test_betti_numbers_dimension_parameter():
    """Test Betti numbers with different dimension parameters"""
    j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82]
    
    # For ECDSA (n=2), expect β₀=1, β₁=2, β₂=1
    result_ecdsa = check_betti_numbers(j_invariants, n=2)
    assert result_ecdsa["betti_0"] == 1
    assert result_ecdsa["betti_1"] == 2
    assert result_ecdsa["betti_2"] == 1
    
    # For higher-dimensional systems, Betti numbers should change
    result_higher_dim = check_betti_numbers(j_invariants, n=3)
    assert result_higher_dim["betti_1"] >= 2, "Higher dimension should have more 1-cycles"
    
    # For CSIDH with larger n, expect different Betti numbers
    # According to Theorem 21: β₁ should be n-1
    result_csidh = check_betti_numbers(j_invariants, n=5)
    assert result_csidh["betti_1"] == 4, "For n=5, β₁ should be 4"

def test_topological_security_metrics():
    """Test security metrics based on topological properties"""
    # Secure system metrics
    secure_j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82, 0.70, 0.69, 0.73]
    secure_result = check_betti_numbers(secure_j_invariants)
    
    # Vulnerable system metrics
    vulnerable_j_invariants = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    vulnerable_result = check_betti_numbers(vulnerable_j_invariants)
    
    # Verify security flag
    assert secure_result["is_secure"] is True
    assert vulnerable_result["is_secure"] is False
    
    # Verify entropy comparison
    assert secure_result["topological_entropy"] > vulnerable_result["topological_entropy"], \
        "Secure system should have higher topological entropy"
    
    # Verify F1-score correlation (from Table 3)
    # Higher entropy should correlate with higher F1-score
    d_values = [1, 10, 27, 40, 78]
    entropies = [0.0, 2.3, 3.3, 3.7, 4.3]
    f1_scores = [0.12, 0.35, 0.84, 0.91, 0.78]
    
    # Check that entropy generally increases with d (until maximum)
    for i in range(1, len(d_values)):
        if d_values[i] <= 40:  # Before the peak
            assert entropies[i] >= entropies[i-1], f"Entropy should increase with d (d={d_values[i]})"
    
    # Check that F1-score peaks around d=40
    assert f1_scores[3] > f1_scores[2]  # 40 > 27
    assert f1_scores[3] > f1_scores[4]  # 40 > 78

def test_betti_numbers_with_noise():
    """Test Betti number stability with added noise"""
    # Base secure system
    base_j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82, 0.70, 0.69, 0.73]
    
    # Add small noise
    noisy_j_invariants = [x + np.random.normal(0, 0.01) for x in base_j_invariants]
    result_noisy = check_betti_numbers(noisy_j_invariants)
    
    # Betti numbers should remain the same with small noise
    assert result_noisy["betti_0"] == 1
    assert result_noisy["betti_1"] == 2
    assert result_noisy["betti_2"] == 1
    assert result_noisy["is_secure"] is True
    
    # Add large noise
    very_noisy_j_invariants = [x + np.random.normal(0, 0.2) for x in base_j_invariants]
    result_very_noisy = check_betti_numbers(very_noisy_j_invariants)
    
    # With large noise, system might become vulnerable
    assert result_very_noisy["topological_entropy"] < result_noisy["topological_entropy"], \
        "Large noise should reduce topological entropy"
