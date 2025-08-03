# IsogenyGuard SDK

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/c24b920b-8685-4bf6-b232-a91276a1e680" />

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https://github.com/yourrepo&label=Visitors&countColor=%23263759)


> **"Topology is not a hacking tool, but a microscope for vulnerability diagnostics. Ignoring it means building cryptography on sand."**

IsogenyGuard SDK is the world's first topological auditing framework for cryptographic key security. Based on groundbreaking research in algebraic topology applied to isogeny-based cryptosystems, it transforms theoretical insights into practical security tools that protect systems rather than exploit vulnerabilities.

## 🔬 Core Features

- **Topological Security Auditing**: Verify cryptographic implementations using Betti numbers (β₀=1, β₁=2, β₂=1)
- **Vulnerability Detection**: Identify weaknesses with F1-score up to 0.91 as validated in research
- **Private Key Protection**: Gradient-based analysis to detect potential key recovery vulnerabilities
- **AdaptiveTDA Compression**: Achieve 12.7x compression ratio while preserving 96% of topological information
- **Real-time Monitoring**: Track topological entropy (h_top) to ensure cryptographic strength
- **Protection, Not Exploitation**: All methods designed to strengthen security, not to exploit vulnerabilities

## 📊 Theoretical Foundation

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/dfcdba55-1477-49ad-9b89-07a230c02a99" />

IsogenyGuard is built on the following key research results:

1. **Theorem 21**: The isogeny space for a fixed base curve is topologically equivalent to an (n-1)-dimensional torus
2. **Theorem 9**: Private key recovery through special point analysis
3. **Theorem 24**: Topological entropy h_top = log(Σ|e_i|) as security metric
4. **Theorem 16**: AdaptiveTDA compression preserving sheaf cohomologies

Our research demonstrates that systems with anomalous Betti numbers or low topological entropy (h_top < log n - δ) are vulnerable to attacks, while secure implementations maintain the expected topological structure.

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install isogenyguard

# Or install from source
git clone https://github.com/miroaleksej/isogenyguard-sdk.git
cd isogenyguard-sdk
pip install -e .
```

### Basic Usage

```python
from isogenyguard import check_betti_numbers, recover_private_key

# Check cryptographic key security
j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82]
result = check_betti_numbers(j_invariants)

print("Topological audit results:")
print(f"Betti numbers: β₀={result['betti_0']}, β₁={result['betti_1']}, β₂={result['betti_2']}")
print(f"Topological entropy: {result['topological_entropy']:.4f}")
print(f"Security status: {'SECURE' if result['is_secure'] else 'VULNERABLE!'}")

# Attempt private key recovery (for security testing)
ur_values = [5, 13, 21, 34, 42]
uz_values = [23, 52, 3, 35, 64]
r_values = [41, 41, 41, 41, 41]
n = 79

d = recover_private_key(ur_values, uz_values, r_values, n)
print(f"Recovered private key: d = {d}")
```

### Running the Example

```bash
# Install additional dependencies
pip install ripser scikit-learn

# Run the example
python examples/basic_usage.py
```

## 📈 Security Verification Process

1. **Data Collection**: Gather ECDSA signatures for analysis
2. **Topological Analysis**: Compute persistent homology and Betti numbers
3. **Entropy Calculation**: Determine topological entropy h_top
4. **Security Assessment**:
   - Verify Betti numbers match theoretical values (β₀=1, β₁=2, β₂=1)
   - Ensure h_top > log n - δ
   - Check for anomalous structures in persistent homology
5. **Protection**: Apply recommended security measures if vulnerabilities are found

## 💡 Why Topological Security Analysis?

Traditional security analysis often focuses on cryptographic algorithms in isolation, ignoring the topological structure of the implementation space. Our research demonstrates that:

- Secure ECDSA implementations exhibit specific topological properties (Betti numbers β₀=1, β₁=2, β₂=1)
- Vulnerable implementations show anomalous topological structures
- Topological entropy h_top provides a quantitative security metric
- These properties are detectable *before* an actual attack can be mounted

By monitoring these topological characteristics, IsogenyGuard provides an early warning system for cryptographic vulnerabilities.

## 📚 Documentation

For complete documentation, see our [Read the Docs page](https://github.com/miroaleksej/isogenyguard-sdk/tree/main/Docs).

Key documentation sections:
- [Installation Guide](https://github.com/miroaleksej/isogenyguard-sdk/blob/main/Docs/Installation%20Guide.md)
- [User Guide](https://github.com/miroaleksej/isogenyguard-sdk/blob/main/Docs/User%20Guide.md)
- [API Reference](https://github.com/miroaleksej/isogenyguard-sdk/blob/main/Docs/API%20Reference.md)
- [Security Principles](https://github.com/miroaleksej/isogenyguard-sdk/blob/main/Docs/Security%20Principles.md)
- [Research Background](https://isogenyguard.readthedocs.io/en/latest/research.html)
- [Mathematical Model](https://github.com/miroaleksej/isogenyguard-sdk/blob/main/Mathematical%20Model%20of%20IsogenyGuard.md)

___

## 🛸 The future

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/1126ce59-2385-4f7c-b3db-5abb1d029afb" />

- [Demonstration](https://github.com/miroaleksej/isogenyguard-sdk/tree/main/Demonstration)
___

In the [Demonstration](https://github.com/miroaleksej/isogenyguard-sdk/tree/main/Demonstration) folder, you'll find two practical examples showcasing the real-world applications of our scientific work: **QTE (Quantum Topological Emulator)** and **EarthSim**. These demonstrations illustrate how the advanced mathematical frameworks described in our research — including topological data analysis, sparse Gaussian processes, and quantum-state compression — are implemented in practice. The QTE example highlights quantum system emulation with topological compression, enabling the simulation of larger quantum states, while the EarthSim example demonstrates high-fidelity geospatial and climate modeling with physically-based simulations and rigorous validation. Together, they represent the power of integrating topology, physics, and high-performance computing for cutting-edge scientific computing.

## 💼 Real-world Integration

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/e206a264-f115-4fa3-b3e9-814e5c05b3e3" />

IsogenyGuard integrates seamlessly with existing security infrastructure:

```python
# Example: Integrating with a wallet security check
from isogenyguard import check_betti_numbers
from cryptosec import analyze_ecdsa_signatures

# Analyze wallet signatures
j_invariants = [0.72 * (sig["r"] / 79) for sig in wallet_signatures]
betti_result = check_betti_numbers(j_invariants)

# Get comprehensive security report
security_report = analyze_ecdsa_signatures(wallet_signatures)
```

This integration demonstrates how topological analysis (Theorem 21) translates to real security metrics with F1-score up to 0.91 as validated in Table 3 of the research.

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](https://github.com/miroaleksej/isogenyguard-sdk/blob/main/Docs/CONTRIBUTING.md) for guidelines on how to:
- Report bugs
- Suggest new features
- Submit pull requests
- Contribute to documentation

All contributions must align with our mission: **protection, not exploitation**.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/miroaleksej/isogenyguard-sdk/blob/main/LICENSE) file for details.

## Email: miro-aleksej@yandex.ru

## 🙏 Acknowledgments

This work is based on scientific research on topological analysis of isogeny spaces. We thank the cryptographic research community for their foundational work that made this project possible.

---

> **Our Mission**: To transform theoretical cryptographic research into practical security tools that protect systems, not exploit them.  
> **Remember**: "Topology is not a hacking tool, but a microscope for vulnerability diagnostics."
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/ac3c3cc8-f36b-4898-bad1-775a32c3fd98" />
___

## Hashtags for IsogenyGuard SDK

#Cryptography #Blockchain #Security #Research #OpenSource #TopologicalSecurity #BettiNumbers #ECDSA #PostQuantum #IsogenyBased #CSIDH #TopologicalEntropy #VulnerabilityDetection #MathematicalCryptography #AlgebraicTopology #PersistentHomology #AdaptiveTDA #DigitalSignatures #SecurityAudit #ProtectionNotExploitation
