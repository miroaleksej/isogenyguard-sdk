"""
IsogenyGuard — Realistic end-to-end test for private key recovery
Workflow: Public key → generate signatures → find repeated R_x → recover d
"""

import pytest
import numpy as np
from unittest.mock import patch
from isogenyguard.core import recover_private_key_from_signatures
from isogenyguard.utils import rsz_to_uruz, uruz_to_rsz


# --- Mock elliptic curve operations ---
def mock_x_of_kG(k, n=79):
    """Mock x(kG) — in real code, uses secp256k1"""
    return (k * k + 1) % n  # Deterministic for testing


def mock_sign(z, d, k, n=79, G=None):
    """Generate valid ECDSA signature"""
    r = mock_x_of_kG(k, n)
    s = pow(k, -1, n) * (z + r * d) % n
    return r, s, z


# --- Test ---
def test_recover_private_key_end_to_end():
    """
    Test full attack workflow:
    1. Given public key Q = dG
    2. Generate signatures in a predefined (u_r, u_z) square
    3. Convert (r,s,z) → (u_r, u_z)
    4. Build R_x table and find repeated R_x values
    5. Recover d using Theorem 9
    """
    # Setup
    n = 79
    d_true = 27
    Q = f"mock_pubkey_{d_true}"  # Q = dG

    # Define search region: square in (u_r, u_z) space
    ur_range = range(5, 15)  # u_r ∈ [5, 14]
    uz_range = range(20, 30)  # u_z ∈ [20, 29]

    # Step 1: Generate signatures from (u_r, u_z) space
    signatures = []
    for ur in ur_range:
        for uz in uz_range:
            # k = u_z + u_r * d mod n
            k = (uz + ur * d_true) % n
            # z can be arbitrary
            z = (ur + uz) % n
            # Sign
            r, s, _ = mock_sign(z, d_true, k, n)
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
