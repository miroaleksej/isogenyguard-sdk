# ECDSA Topological Audit System - README

## Overview
The ECDSA Topological Audit System is a groundbreaking tool for analyzing the security of ECDSA implementations through topological analysis of the Rₓ table structure. Unlike traditional audit methods, this system detects vulnerabilities **without requiring access to the private key**, using only the public key to identify potential weaknesses in the implementation.

This innovative approach applies algebraic topology (Betti numbers, spiral wave analysis) to detect anomalies in ECDSA implementations that could lead to critical vulnerabilities like private key recovery through "reused k" attacks.

## Key Features

- **Black-box analysis**: No need for private key or source code access
- **Topological vulnerability detection**: Uses Betti numbers (β₀=1, β₁=2, β₂=1) as security indicators
- **Spiral wave analysis**: Measures damping coefficient (γ > 0.1 for secure implementations)
- **Symmetry verification**: Checks expected symmetry around special points
- **Optimized for Bitcoin**: Works with secp256k1 curve used in Bitcoin
- **Partial table analysis**: Efficiently analyzes subregions instead of full table (critical for large n)
- **Comprehensive reporting**: Generates detailed HTML/PDF reports with visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecdsa-audit.git
cd ecdsa-audit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: For optimal performance with secp256k1, ensure you have the `libsecp256k1` library installed on your system.

## Quick Start

Audit a single Bitcoin public key:
```bash
python main.py 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
```

Audit multiple keys from a file:
```bash
python main.py -f public_keys.txt
```

Generate verbose output with visualizations:
```bash
python main.py -f public_keys.txt --verbose
```

## Project Structure

```
ecdsa-audit/
├── core/                   # Core system components
│   ├── curve_operations.py # Elliptic curve operations
│   ├── table_generator.py  # Rₓ table subregion generation
│   ├── topology_analyzer.py# Topological property analysis
│   └── anomaly_detector.py # Vulnerability detection
├── audit/                  # Audit modules
│   ├── audit_engine.py     # Main audit coordinator
│   ├── safety_metrics.py   # Security metrics calculation
│   └── vulnerability_scanner.py # Vulnerability pattern detection
├── utils/                  # Utility modules
│   ├── config.py           # Configuration management
│   ├── parallel.py         # Parallel processing
│   ├── visualization.py    # Result visualization
│   └── report_generator.py # Report generation
├── tests/                  # Test suite
├── config/                 # Configuration files
├── requirements.txt        # Dependencies
└── main.py                 # Command-line entry point
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# ECDSA Topological Audit System - User Manual

## 1. Introduction

### 1.1 What is Topological ECDSA Audit?

The ECDSA Topological Audit System analyzes the security of ECDSA implementations by examining the topological properties of the Rₓ table structure. Instead of traditional code analysis, this system:

- Treats the Rₓ table as a topological space (specifically a torus)
- Measures topological invariants (Betti numbers)
- Analyzes spiral wave patterns and symmetry properties
- Detects anomalies that indicate potential vulnerabilities

The key innovation is that this analysis requires **only the public key**, making it a powerful black-box auditing technique.

### 1.2 Why Topological Analysis?

Traditional ECDSA security analysis often requires:
- Access to source code
- Knowledge of implementation details
- Testing with known private keys

Our topological approach overcomes these limitations by:
- Working with only public information (public key)
- Providing quantitative security metrics
- Detecting structural anomalies that indicate implementation flaws
- Identifying specific vulnerabilities like "reused k" attacks

The system is particularly valuable for auditing production systems where private keys and source code are unavailable.

## 2. System Requirements

### 2.1 Hardware Requirements
- Minimum: 2 GHz processor, 2 GB RAM
- Recommended: 4+ core processor, 8+ GB RAM (for faster analysis)
- For Bitcoin analysis: SSD storage recommended for cache

### 2.2 Software Requirements
- Python 3.7+
- libsecp256k1 (for optimal Bitcoin analysis performance)
- Required Python packages (see requirements.txt)

## 3. Installation and Setup

### 3.1 Installing Dependencies

On Ubuntu/Debian:
```bash
sudo apt-get install libsecp256k1-dev
pip install -r requirements.txt
```

On macOS (using Homebrew):
```bash
brew install libsecp256k1
pip install -r requirements.txt
```

### 3.2 Configuration

The system includes two configuration profiles:
- `default_config.yaml`: General-purpose configuration
- `bitcoin_secp256k1.yaml`: Optimized for Bitcoin ECDSA analysis

To use the Bitcoin configuration:
```bash
python main.py -c config/bitcoin_secp256k1.yaml PUBLIC_KEY
```

Key configuration parameters:
- `default_num_regions`: Number of subregions to analyze (default: 10)
- `default_region_size`: Size of each subregion (default: 50)
- `gamma_threshold`: Minimum damping coefficient (default: 0.1)
- `symmetry_threshold`: Minimum symmetry score (default: 0.85)

## 4. Using the Audit System

### 4.1 Basic Usage

Audit a single public key:
```bash
python main.py PUBLIC_KEY_HEX
```

Example:
```bash
python main.py 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
```

### 4.2 Advanced Usage

Audit multiple keys from a file:
```bash
python main.py -f public_keys.txt
```

Customize analysis parameters:
```bash
python main.py PUBLIC_KEY -n 15 -s 60
```
Where `-n` sets number of regions and `-s` sets region size.

Generate PDF report:
```bash
python main.py PUBLIC_KEY --format pdf
```

Enable verbose output with visualizations:
```bash
python main.py PUBLIC_KEY --verbose
```

### 4.3 Input Format

Public keys must be in hexadecimal format, either:
- Compressed (starts with 02 or 03, 66 characters)
- Uncompressed (starts with 04, 130 characters)

Example valid keys:
- Compressed: `0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798`
- Uncompressed: `0479be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8`

## 5. Understanding the Results

### 5.1 Vulnerability Levels

The system classifies implementations into three vulnerability levels:

| Level | Anomaly Score | Description |
|-------|---------------|-------------|
| Safe | < 0.3 | No critical vulnerabilities detected |
| Warning | 0.3-0.7 | Potential anomalies requiring investigation |
| Critical | > 0.7 | Critical vulnerabilities detected |

### 5.2 Key Security Metrics

The audit evaluates three primary metrics:

1. **Topological Analysis (Betti Numbers)**
   - Expected values: β₀ = 1, β₁ = 2, β₂ = 1
   - Deviations indicate structural anomalies

2. **Spiral Wave Analysis**
   - Damping coefficient (γ) should be > 0.1
   - Low values indicate potential nonce generation issues

3. **Symmetry Analysis**
   - Symmetry score should be > 0.85
   - Broken symmetry indicates implementation anomalies

### 5.3 Detected Vulnerabilities

The system identifies specific vulnerabilities:

| Vulnerability | Description | Risk Level |
|---------------|-------------|------------|
| Reused k attack | Same k value used for multiple signatures | Critical |
| Weak nonce generation | Predictable or biased nonce values | High |
| Broken symmetry | Deviation from expected symmetry patterns | Medium |
| Missing spiral | Absence of expected spiral structure | Medium |

## 6. Practical Examples

### 6.1 Example 1: Secure Implementation

```
$ python main.py 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798

[2023-10-15 14:30:22] INFO - Starting audit for public key: 0279be667e...
[2023-10-15 14:31:45] INFO - Audit completed in 83.2s. Vulnerability level: safe

ECDSA AUDIT RESULTS
Public Key: 0279be667e... (truncated)
Vulnerability Level: SAFE
Anomaly Score: 0.12

SECURITY METRICS:
- Betti Numbers: β₀=1.0, β₁=2.0, β₂=1.0 (EXPECTED: 1, 2, 1)
- Damping Coefficient: γ=0.15 (THRESHOLD: 0.1)
- Symmetry Score: 0.92 (THRESHOLD: 0.85)

RECOMMENDATIONS:
The ECDSA implementation appears to be secure based on topological analysis.
No critical vulnerabilities were detected in the Rₓ table structure.
```

### 6.2 Example 2: Vulnerable Implementation (Reused k)

```
$ python main.py 02c6047f9441ed7d6d3045406e95c07cd85c7a8e8e5b9bcb9d578f74e04e1f451b

[2023-10-15 14:35:12] INFO - Starting audit for public key: 02c6047f94...
[2023-10-15 14:36:28] INFO - Audit completed in 76.5s. Vulnerability level: critical

ECDSA AUDIT RESULTS
Public Key: 02c6047f94... (truncated)
Vulnerability Level: CRITICAL
Anomaly Score: 0.87

SECURITY METRICS:
- Betti Numbers: β₀=1.0, β₁=0.5, β₂=0.0 (EXPECTED: 1, 2, 1)
- Damping Coefficient: γ=0.03 (THRESHOLD: 0.1)
- Symmetry Score: 0.62 (THRESHOLD: 0.85)

DETECTED VULNERABILITIES:
- Reused k attack vulnerability (CRITICAL)
- Broken symmetry (MEDIUM)
- Missing spiral structure (MEDIUM)

RECOMMENDATIONS:
CRITICAL: High probability of reused k values detected.
Reused k values can lead to private key recovery. Immediate action required.
Ensure that random k values are generated using a cryptographically secure random number generator.
Consider implementing deterministic nonce generation according to RFC 6979.
Rotate all affected keys immediately.
```

## 7. Advanced Configuration

### 7.1 Custom Configuration

Create a custom configuration file (e.g., `my_config.yaml`):

```yaml
# Custom configuration for high-security audit
default_num_regions: 20
default_region_size: 70
gamma_threshold: 0.15
symmetry_threshold: 0.90
log_level: "DEBUG"
```

Use it with:
```bash
python main.py PUBLIC_KEY -c my_config.yaml
```

### 7.2 Parallel Processing

To speed up batch audits, configure parallel processing in your config file:

```yaml
max_workers: 4  # Use 4 CPU cores
chunk_size: 2   # Process 2 keys per chunk
```

## 8. Troubleshooting

### 8.1 Common Issues

**Problem**: "Invalid public key" error
- **Solution**: Ensure the public key is in correct hexadecimal format (compressed or uncompressed)

**Problem**: Slow analysis performance
- **Solution**: 
  1. Verify libsecp256k1 is installed
  2. Increase region size but decrease number of regions
  3. Enable parallel processing

**Problem**: False positive warnings
- **Solution**: Adjust thresholds in configuration:
  ```yaml
  gamma_threshold: 0.08
  symmetry_threshold: 0.80
  ```

### 8.2 Debugging

Enable debug logging:
```bash
python main.py PUBLIC_KEY --verbose
```

Or set in config:
```yaml
log_level: "DEBUG"
```

## 9. Technical Background

### 9.1 How It Works

The system leverages the mathematical structure of ECDSA:

1. For any public key Q = d·G, we can compute Rₓ(uᵣ,u_z) = x(uᵣ·Q + u_z·G)
2. This creates a table with specific topological properties
3. Secure implementations produce tables with:
   - Betti numbers: β₀ = 1, β₁ = 2, β₂ = 1
   - Damping coefficient: γ > 0.1
   - High symmetry around special points

Vulnerabilities like "reused k" attacks disrupt these properties, making them detectable through topological analysis.

### 9.2 The Rₓ Table Structure

The Rₓ table has these key properties:
- Each row is a cyclic shift of the previous row by d positions
- Values are symmetric around special points u_z* = -u_r·d mod n
- Points with the same Rₓ value form spiral patterns
- The complete structure is topologically equivalent to a torus

These properties allow us to detect anomalies without knowing the private key d.

## 10. Support and Community

For support, bug reports, or feature requests:
- Open an issue on GitHub
- Email: miro-aleksej@yandex.ru

The project welcomes contributions! Please see CONTRIBUTING.md for details.

---

*This system implements novel research in topological cryptography. For academic references and theoretical background, see the accompanying research paper.*
