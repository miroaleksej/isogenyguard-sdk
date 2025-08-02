"""
Example of using IsogenyGuard for auditing a real cryptographic system
"""

from isogenyguard import check_betti_numbers, calculate_topological_entropy
from cryptosec import analyze_ecdsa_signatures

# Load signatures from wallet (example)
signatures = [
    {"r": 12345, "s": 67890, "z": 54321},
    {"r": 23456, "s": 78901, "z": 65432},
    # ... more signatures
]

# Analyze using IsogenyGuard
j_invariants = [0.72 * (sig["r"] / 79) for sig in signatures]
betti_result = check_betti_numbers(j_invariants)

print("="*50)
print("TOPOLOGICAL AUDIT OF REAL SYSTEM")
print("="*50)
print(f"Betti numbers: β₀={betti_result['betti_0']}, β₁={betti_result['betti_1']}, β₂={betti_result['betti_2']}")
print(f"Topological entropy: {betti_result['topological_entropy']:.4f}")
print(f"Security status: {'SECURE' if betti_result['is_secure'] else 'VULNERABLE!'}")

# Additional analysis through Cryptosec
security_report = analyze_ecdsa_signatures(signatures)
print("\nADDITIONAL ANALYSIS:")
for issue in security_report.issues:
    print(f"- {issue}")
