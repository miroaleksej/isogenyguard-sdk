"""
IsogenyGuard — Realistic topological audit test
Workflow: Public key → generate j-invariants → check Betti numbers and entropy
"""

import pytest
import numpy as np
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


def test_topology_audit_end_to_end():
    """
    Test full topological audit:
    1. Given public key (d)
    2. Generate j-invariants from isogeny lattice
    3. Perform topological audit
    4. Check Betti numbers and entropy
    """
    # Setup
    n = 79
    d_true = 27

    # Step 1: Generate j-invariants from isogeny lattice (Q = [a]E_0)
    j_invariants = generate_j_invariants_for_d(d_true, size=40)

    # Step 2: Perform topological audit
    result = check_betti_numbers(j_invariants, n=2)

    # Step 3: Check Betti numbers (Theorem 6)
    assert result["betti_0"] == 1, f"Expected β₀=1, got {result['betti_0']}"
    assert result["betti_1"] == 2, f"Expected β₁=2, got {result['betti_1']}"
    assert result["betti_2"] == 1, f"Expected β₂=1, got {result['betti_2']}"
    assert result["is_secure"] is True, "Secure system should be flagged as secure"

    # Step 4: Check topological entropy (Theorem 24)
    entropy = result["topological_entropy"]
    assert 3.0 <= entropy <= 3.6, f"Expected h_top ≈ 3.3, got {entropy:.2f}"


def test_topology_audit_vulnerable_system():
    """
    Test audit of vulnerable system (low entropy)
    """
    # Simulate weak system: uniform j-invariants (low entropy)
    j_invariants = [0.1] * 50

    result = check_betti_numbers(j_invariants, n=2)

    # Should have anomalous Betti numbers
    assert not (result["betti_0"] == 1 and result["betti_1"] == 2 and result["betti_2"] == 1), \
        "Vulnerable system incorrectly flagged as secure"

    # Low entropy
    assert result["topological_entropy"] < 2.0, \
        "Vulnerable system should have low entropy"
