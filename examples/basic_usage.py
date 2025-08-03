"""
Basic usage example for IsogenyGuard SDK
Demonstrates private key recovery and topological audit.
"""

from IsogenyGuard import (
    recover_private_key,
    check_betti_numbers,
    calculate_topological_entropy,
    info
)

# Parameters from scientific work
n = 79  # Group order
d_true = 27  # True private key (for simulation)

# Simulated special points from network traffic (u_r, u_z)
# From scientific work: u_z^(r+1) - u_z^(r) = -d mod n
special_points = [
    (5, 23),
    (6, 75),   # 75 - 23 = 52 ≡ -27 mod 79
    (7, 48),   # 48 - 75 = -27 ≡ 52 mod 79
    (8, 21),
    (9, 73)
]

# Step 1: Recover private key from special points (Theorem 9)
d_recovered = recover_private_key(special_points, n)
print(f"[+] Recovered private key: d = {d_recovered}")
print(f"[+] True private key: d = {d_true}")
print(f"[+] Match: {d_recovered == d_true}\n")

# Step 2: Simulate j-invariants from isogeny lattice
# In practice: from network or side-channel
j_invariants = [float((i * d_recovered) % 1000) for i in range(50)]

# Step 3: Topological audit (Theorem 21, Table 3)
betti_result = check_betti_numbers(j_invariants, n=2)
entropy = calculate_topological_entropy(j_invariants)

print(f"[+] Betti numbers: β₀={betti_result['betti_0']}, "
      f"β₁={betti_result['betti_1']}, β₂={betti_result['betti_2']}")
print(f"[+] Expected: β₀=1, β₁=2, β₂=1")
print(f"[+] System secure: {betti_result['is_secure']}\n")

print(f"[+] Topological entropy h_top = {entropy:.2f}")
print(f"[+] log|d| = {import math: math.log(d_recovered):.2f}\n")

# Step 4: Print SDK info
print(info())
