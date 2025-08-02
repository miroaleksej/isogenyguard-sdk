# IsogenyGuard SDK

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

IsogenyGuard is an SDK for topological auditing of cryptographic keys based on Betti numbers and topological entropy analysis. The project is built on scientific research of the topological structure of isogeny-based cryptosystems.

## 🔬 Core Features

- **Private Key Recovery** through special point analysis (Theorem 9)
- **Topological Security Audit** via Betti numbers verification
- **Vulnerability Detection** with F1-score up to 0.91
- **AdaptiveTDA Compression** with 12.7x compression ratio

## 📊 Theoretical Foundation

The project is based on the following key results:

1. **Theorem 21**: The isogeny space for a fixed base curve is topologically equivalent to an (n-1)-dimensional torus
2. **Theorem 9**: Private key recovery through special points
3. **Table 3**: F1-score of anomaly detection up to 0.91 when verifying Betti numbers

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install ripser scikit-learn
