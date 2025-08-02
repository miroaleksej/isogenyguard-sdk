from isogenyguard import core, topology

def main():
    print("="*50)
    print("IsogenyGuard SDK - Basic Usage Example")
    print("="*50)
    
    # Example 1: Private key recovery (Theorem 9)
    print("\nExample 1: Private Key Recovery")
    print("-"*40)
    
    # Data from research (d=27, n=79)
    ur_values = [5, 13, 21, 34, 42]
    uz_values = [23, 52, 3, 35, 64]
    r_values = [41, 41, 41, 41, 41]  # All have the same R_x
    n = 79
    
    d = core.recover_private_key(ur_values, uz_values, r_values, n)
    print(f"Recovered private key: d = {d}")
    print(f"Expected result: d = 27")
    print(f"Verification: {'SUCCESS' if d == 27 else 'ERROR'}")
    
    # Example 2: Topological security audit
    print("\nExample 2: Topological Security Audit")
    print("-"*40)
    
    # Example j-invariants from secure system (d=27)
    j_invariants_secure = [0.72, 0.68, 0.75, 0.65, 0.82, 0.70, 0.69, 0.73]
    
    result = topology.check_betti_numbers(j_invariants_secure)
    
    print("Topological audit results:")
    print(f"Betti numbers: β₀={result['betti_0']}, β₁={result['betti_1']}, β₂={result['betti_2']}")
    print(f"Topological entropy: {result['topological_entropy']:.4f}")
    print(f"Security status: {'SECURE' if result['is_secure'] else 'VULNERABLE!'}")
    
    # Example 3: Checking vulnerable system
    print("\nExample 3: Vulnerable System Check")
    print("-"*40)
    
    # Example j-invariants from vulnerable system
    j_invariants_vulnerable = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    
    result_vuln = topology.check_betti_numbers(j_invariants_vulnerable)
    
    print("Vulnerable system analysis results:")
    print(f"Betti numbers: β₀={result_vuln['betti_0']}, β₁={result_vuln['betti_1']}, β₂={result_vuln['betti_2']}")
    print(f"Topological entropy: {result_vuln['topological_entropy']:.4f}")
    print(f"Security status: {'SECURE' if result_vuln['is_secure'] else 'VULNERABLE!'}")
    
    print("\n" + "="*50)
    print("IsogenyGuard SDK is ready for use!")
    print("Topology is not a hacking tool, but a microscope for vulnerability diagnostics")
    print("="*50)

if __name__ == "__main__":
    main()
