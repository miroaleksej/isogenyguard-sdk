"""
IsogenyGuard — Realistic unit test for private key recovery
Simulates real-world workflow: from signatures to d via (u_r, u_z) and R_x table.
"""

import pytest
from unittest.mock import MagicMock
from isogenyguard.core import recover_private_key_from_signatures
from isogenyguard.utils import rsz_to_uruz


# Mock elliptic curve operations
def mock_x_of_kG(k, n=79):
    """Mock x(kG) — in real code, uses secp256k1"""
    return (k * k + 1) % n  # Simplified deterministic mapping


def generate_signature(d, z, k, n=79):
    """Generate valid ECDSA signature for given d, z, k"""
    r = mock_x_of_kG(k, n)
    s = pow(k, -1, n) * (z + r * d) % n
    return r, s, z


def test_recover_private_key_realistic():
    """
    Test end-to-end key recovery:
    1. Generate multiple signatures with fixed d
    2. Convert (r,s,z) → (u_r, u_z)
    3. Build R_x table and find repeated R_x values
    4. Extract special points (same R_x, consecutive u_r)
    5. Recover d using Theorem 9
    """
    # Setup
    n = 79
    d_true = 27
    Q = f"mock_pubkey_{d_true}"  # Q = dG

    # Step 1: Collect real signatures from "network"
    signatures = []
    z_values = [10, 20, 30, 40, 50]
    k_values = [68, 5, 13, 21, 34, 42]  # Include special points

    for z in z_values:
        for k in k_values:
            r, s, _ = generate_signature(d_true, z, k, n)
            signatures.append((r, s, z))

    # Step 2: Convert to (u_r, u_z) — Theorem 1
    uruz_points = []
    for r, s, z in signatures:
        ur, uz = rsz_to_uruz(r, s, z, n)
        uruz_points.append((ur, uz, r))

    # Step 3: Build R_x table and find repeated R_x
    from collections import defaultdict
    rx_map = defaultdict(list)  # r → list of (ur, uz)
    for ur, uz, r in uruz_points:
        rx_map[r].append((ur, uz))

    # Step 4: Find special points: same R_x, consecutive u_r
    special_points = []
    for r, points in rx_map.items():
        if len(points) < 2:
            continue
        sorted_points = sorted(points, key=lambda x: x[0])  # sort by u_r
        for i in range(len(sorted_points) - 1):
            ur1, uz1 = sorted_points[i]
            ur2, uz2 = sorted_points[i+1]
            if ur2 == ur1 + 1:
                special_points.append((ur1, uz1))
                special_points.append((ur2, uz2))

    # Step 5: Recover d — Theorem 9
    recovered_d = None
    diffs = []
    sorted_sp = sorted(special_points, key=lambda x: x[0])
    for i in range(len(sorted_sp) - 1):
        if sorted_sp[i+1][0] == sorted_sp[i][0] + 1:
            delta_uz = (sorted_sp[i+1][1] - sorted_sp[i][1]) % n
            d = (-delta_uz) % n
            diffs.append(d)
    if diffs:
        recovered_d = max(set(diffs), key=diffs.count)

    # Step 6: Assert
    assert recovered_d is not None, "Failed to recover d — no special points found"
    assert recovered_d == d_true, f"Expected d={d_true}, got {recovered_d}"
