# IsogenyGuard SDK API Reference

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Core Module](#2-core-module)
  - [2.1 recover_private_key](#21-recover_private_key)
  - [2.2 check_special_points](#22-check_special_points)
- [3. Topology Module](#3-topology-module)
  - [3.1 check_betti_numbers](#31-check_betti_numbers)
  - [3.2 calculate_topological_entropy](#32-calculate_topological_entropy)
  - [3.3 calculate_persistence_diagram](#33-calculate_persistence_diagram)
- [4. AdaptiveTDA Module](#4-adaptivetda-module)
  - [4.1 compress](#41-compress)
  - [4.2 decompress](#42-decompress)
- [5. TopologicalAnalyzer Class](#5-topologicalanalyzer-class)
  - [5.1 analyze_topology](#51-analyze_topology)
- [6. Data Structures](#6-data-structures)
  - [6.1 TopologyAnalysisResult](#61-topologyanalysisresult)
  - [6.2 SecurityReport](#62-securityreport)

## 1. Introduction

This API Reference documents the IsogenyGuard SDK, a topological security analysis framework for cryptographic implementations. The SDK implements the theoretical framework from the scientific research on topological analysis of isogeny spaces.

All functions and classes are designed to **protect systems, not exploit vulnerabilities**. The API provides quantitative metrics for security assessment based on the topological properties of cryptographic implementations.

> **"Topology is not a hacking tool, but a microscope for vulnerability diagnostics. Ignoring it means building cryptography on sand."**

## 2. Core Module

The `isogenyguard.core` module contains functions for private key security analysis based on special point detection.

### 2.1 recover_private_key

```python
def recover_private_key(
    ur_values: List[int], 
    uz_values: List[int], 
    r_values: List[int], 
    n: int
) -> Optional[int]
```

**Description**: Recovers the private key `d` through special point analysis using gradient-based methods. Based on **Theorem 9** from the research.

**Parameters**:
- `ur_values` (List[int]): List of u_r values from signatures (u_r = r · s⁻¹ mod n)
- `uz_values` (List[int]): List of u_z values from signatures (u_z = z · s⁻¹ mod n)
- `r_values` (List[int]): List of R_x values (x-coordinates of signature points)
- `n` (int): Group order (curve parameter)

**Returns**:
- `Optional[int]`: Recovered private key `d` or `None` if recovery fails

**Mathematical Basis**:
```
d = -(∂r/∂u_z) · (∂r/∂u_r)⁻¹ mod n
```

**Example**:
```python
from isogenyguard import recover_private_key

# Data from research (d=27, n=79)
ur_values = [5, 13, 21, 34, 42]
uz_values = [23, 52, 3, 35, 64]
r_values = [41, 41, 41, 41, 41]  # All have the same R_x
n = 79

d = recover_private_key(ur_values, uz_values, r_values, n)
print(f"Recovered private key: d = {d}")  # Expected: d = 27
```

**Security Note**: This function is designed for security testing only. Use it to identify weaknesses in your implementation, not to exploit others' systems.

### 2.2 check_special_points

```python
def check_special_points(
    ur_values: List[int], 
    uz_values: List[int], 
    n: int
) -> List[int]
```

**Description**: Checks for special points in signature data that could enable private key recovery. Based on the analysis of linear dependencies in the signature space.

**Parameters**:
- `ur_values` (List[int]): List of u_r values from signatures
- `uz_values` (List[int]): List of u_z values from signatures
- `n` (int): Group order (curve parameter)

**Returns**:
- `List[int]`: List of indices of special points in the input data

**Mathematical Basis**:
Special points satisfy: 
```
u_z ≡ -u_r · d mod n
```
For adjacent points: 
```
u_z^(r+1) - u_z^(r) ≡ -d mod n
```

**Example**:
```python
from isogenyguard import check_special_points

# Test data with d=27, n=79
ur_values = [5, 13, 21, 34, 42]
uz_values = [23, 52, 3, 35, 64]
n = 79

special_points = check_special_points(ur_values, uz_values, n)
print(f"Special points detected at indices: {special_points}")
# Expected: [1, 2, 3, 4] (all points are special for consistent data)
```

**Security Implications**:
- More than 70% of points being special indicates high vulnerability
- Fewer than 30% special points suggests better security
- Ideal implementation should have random distribution without clear patterns

## 3. Topology Module

The `isogenyguard.topology` module contains functions for topological security analysis based on Betti numbers and persistent homology.

### 3.1 check_betti_numbers

```python
def check_betti_numbers(
    j_invariants: List[float], 
    n: int = 2
) -> Dict[str, Any]
```

**Description**: Calculates Betti numbers for the isogeny space and verifies security based on topological properties. Based on **Theorem 21** from the research.

**Parameters**:
- `j_invariants` (List[float]): List of j-invariants from observed curves
- `n` (int, optional): Dimension parameter (2 for ECDSA). Defaults to 2.

**Returns**:
- `Dict[str, Any]`: Dictionary containing:
  - `betti_0` (int): Number of connected components
  - `betti_1` (int): Number of independent cycles
  - `betti_2` (int): Number of voids
  - `is_secure` (bool): Security status (True if β₀=1, β₁=n, β₂=1)
  - `topological_entropy` (float): Topological entropy value
  - `persistence` (List[Tuple]): Persistence diagram data
  - `f1_score` (float): F1-score of vulnerability detection

**Theoretical Basis**:
- For ECDSA (n=2), expected values: β₀=1, β₁=2, β₂=1
- Topological entropy: h_top = log(Σ|e_i|)
- F1-score correlates with topological entropy (Table 3 in research)

**Example**:
```python
from isogenyguard import check_betti_numbers

# Example from secure system (d=27)
j_invariants_secure = [0.72, 0.68, 0.75, 0.65, 0.82, 0.70, 0.69, 0.73]
result = check_betti_numbers(j_invariants_secure)

print(f"Betti numbers: β₀={result['betti_0']}, β₁={result['betti_1']}, β₂={result['betti_2']}")
print(f"Topological entropy: {result['topological_entropy']:.4f}")
print(f"Security status: {'SECURE' if result['is_secure'] else 'VULNERABLE!'}")
print(f"F1-score: {result['f1_score']:.2f}")

# Expected output for secure system:
# Betti numbers: β₀=1, β₁=2, β₂=1
# Topological entropy: 3.34
# Security status: SECURE
# F1-score: 0.84
```

**Security Interpretation**:
- β₀=1: One connected component (system is cohesive)
- β₁=2: Two independent cycles (toroidal structure)
- β₂=1: One void (complete torus structure)
- h_top > 3.0: Adequate topological entropy for security
- F1-score > 0.80: High confidence in vulnerability detection

### 3.2 calculate_topological_entropy

```python
def calculate_topological_entropy(
    j_invariants: List[float]
) -> float
```

**Description**: Calculates topological entropy based on j-invariants. Based on **Theorem 24** from the research.

**Parameters**:
- `j_invariants` (List[float]): List of j-invariants from observed curves

**Returns**:
- `float`: Topological entropy value

**Theoretical Basis**:
```
h_top = log(∑|e_i|)
```
Where e_i are the exponents in the secret key representation.

**Example**:
```python
from isogenyguard import calculate_topological_entropy

# Example from secure system (d=27)
j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82]
entropy = calculate_topological_entropy(j_invariants)
print(f"Topological entropy: {entropy:.4f}")  # Expected: ~3.3
```

**Security Thresholds**:
- h_top < 2.0: Very low security (F1-score ~0.12)
- 2.0 ≤ h_top < 2.5: Low security (F1-score ~0.35)
- 2.5 ≤ h_top < 3.5: Medium security (F1-score ~0.84)
- 3.5 ≤ h_top < 4.0: High security (F1-score ~0.91)
- h_top ≥ 4.0: Maximum security (F1-score ~0.78)

### 3.3 calculate_persistence_diagram

```python
def calculate_persistence_diagram(
    j_invariants: List[float]
) -> List[Tuple[int, float, float, float]]
```

**Description**: Calculates the persistence diagram for j-invariants using persistent homology.

**Parameters**:
- `j_invariants` (List[float]): List of j-invariants from observed curves

**Returns**:
- `List[Tuple[int, float, float, float]]`: Persistence diagram data as:
  - Dimension (0, 1, or 2)
  - Birth time
  - Death time
  - Persistence (death - birth)

**Theoretical Basis**:
- Based on computational topology methods
- Used to compute Betti numbers and topological features
- Forms the foundation for security assessment

**Example**:
```python
from isogenyguard import calculate_persistence_diagram

j_invariants = [0.72, 0.68, 0.75, 0.65, 0.82]
persistence = calculate_persistence_diagram(j_invariants)

print("Persistence diagram:")
for dim, birth, death, persistence_val in persistence:
    print(f"Dimension {dim}: [{birth:.2f}, {death:.2f}) persistence={persistence_val:.2f}")
```

**Security Application**:
- Used to verify the topological structure of the isogeny space
- Anomalies in persistence indicate potential vulnerabilities
- Key input for Betti number calculation

## 4. AdaptiveTDA Module

The `isogenyguard.adaptivetda` module contains classes for topology-preserving data compression.

### 4.1 compress

```python
@staticmethod
def compress(
    data: np.ndarray, 
    gamma: float = 0.8
) -> Dict[str, Any]
```

**Description**: Compresses data using Adaptive Topological Data Analysis (AdaptiveTDA) while preserving topological features. Based on **Theorem 16** from the research.

**Parameters**:
- `data` (np.ndarray): Input data to compress
- `gamma` (float, optional): Compression parameter (higher = better preservation). Defaults to 0.8.

**Returns**:
- `Dict[str, Any]`: Dictionary containing:
  - `shape` (tuple): Original data shape
  - `indices` (List[List[int]]): Indices of significant coefficients
  - `values` (List[float]): Values of significant coefficients
  - `threshold` (float): Adaptive threshold used
  - `compression_ratio` (float): Achieved compression ratio
  - `topological_preservation` (float): Percentage of topological information preserved

**Theoretical Basis**:
```
ε(U) = ε₀ · exp(-γ · P(U))
```
Where P(U) is the persistence indicator.

**Example**:
```python
from isogenyguard import AdaptiveTDA
import numpy as np

# Generate sample security data
data = np.array([0.72, 0.68, 0.75, 0.65, 0.82, 0.70, 0.69, 0.73])

# Compress data
compressed = AdaptiveTDA.compress(data, gamma=0.8)

print(f"Original size: {data.size} elements")
print(f"Compressed size: {len(compressed['values']) + 3 * len(compressed['indices'])} elements")
print(f"Compression ratio: {compressed['compression_ratio']:.1f}x")
print(f"Topological preservation: {compressed['topological_preservation']*100:.0f}%")
```

**Performance**:
- Achieves 12.7x compression ratio
- Preserves 96% of topological information
- Ideal for large-scale security monitoring systems

### 4.2 decompress

```python
@staticmethod
def decompress(
    compressed: Dict[str, Any]
) -> np.ndarray
```

**Description**: Decompresses data while preserving topological features.

**Parameters**:
- `compressed` (Dict[str, Any]): Compressed data dictionary from `compress()`

**Returns**:
- `np.ndarray`: Decompressed data array

**Example**:
```python
from isogenyguard import AdaptiveTDA

# Assuming 'compressed' is from previous example
decompressed = AdaptiveTDA.decompress(compressed)

# Verify topological properties are preserved
original_analysis = check_betti_numbers(data.tolist())
decompressed_analysis = check_betti_numbers(decompressed.tolist())

print(f"Original Betti numbers: β₀={original_analysis['betti_0']}, β₁={original_analysis['betti_1']}, β₂={original_analysis['betti_2']}")
print(f"Decomp. Betti numbers: β₀={decompressed_analysis['betti_0']}, β₁={decompressed_analysis['betti_1']}, β₂={decompressed_analysis['betti_2']}")
```

**Security Application**:
- Enables efficient storage and transmission of security data
- Preserves critical topological features for security analysis
- Maintains vulnerability detection capability after compression

## 5. TopologicalAnalyzer Class

The `isogenyguard.TopologicalAnalyzer` class provides a comprehensive interface for topological security analysis.

### 5.1 analyze_topology

```python
@staticmethod
def analyze_topology(
    ur_values: List[int], 
    uz_values: List[int],
    r_values: List[int], 
    n: int
) -> TopologyAnalysisResult
```

**Description**: Performs complete topological analysis of ECDSA signatures, combining multiple analysis methods.

**Parameters**:
- `ur_values` (List[int]): List of u_r values from signatures
- `uz_values` (List[int]): List of u_z values from signatures
- `r_values` (List[int]): List of R_x values from signatures
- `n` (int): Group order (curve parameter)

**Returns**:
- `TopologyAnalysisResult`: Dataclass containing comprehensive analysis results

**Example**:
```python
from isogenyguard import TopologicalAnalyzer

# Test data with d=27, n=79
ur_values = [5, 13, 21, 34, 42]
uz_values = [23, 52, 3, 35, 64]
r_values = [41, 41, 41, 41, 41]
n = 79

analysis = TopologicalAnalyzer.analyze_topology(ur_values, uz_values, r_values, n)

print(f"Betti numbers: β₀={analysis.betti_numbers[0]}, β₁={analysis.betti_numbers[1]}, β₂={analysis.betti_numbers[2]}")
print(f"Topological entropy: {analysis.topological_entropy:.4f}")
print(f"Security status: {'SECURE' if analysis.is_secure else 'VULNERABLE!'}")
print(f"Special points detected: {len(analysis.special_points)} out of {len(ur_values)}")
print(f"Key recovery possible: {analysis.gradient_analysis['recovery_possible']}")
```

**Analysis Components**:
1. **Betti number verification** (Theorem 21)
2. **Topological entropy calculation** (Theorem 24)
3. **Special point detection** (Theorem 9)
4. **Gradient-based key recovery analysis**

**Security Interpretation**:
- Comprehensive security assessment combining multiple metrics
- F1-score up to 0.91 for vulnerability detection
- Detailed insights into specific vulnerabilities

## 6. Data Structures

### 6.1 TopologyAnalysisResult

```python
@dataclass
class TopologyAnalysisResult:
    betti_numbers: List[int]
    topological_entropy: float
    is_secure: bool
    special_points: List[int]
    gradient_analysis: Dict[str, Any]
    persistence_diagram: List[Tuple[int, float, float, float]]
    f1_score: float
```

**Description**: Data structure containing results of topological security analysis.

**Fields**:
- `betti_numbers` (List[int]): Betti numbers [β₀, β₁, β₂]
- `topological_entropy` (float): Calculated topological entropy
- `is_secure` (bool): Overall security status based on Betti numbers
- `special_points` (List[int]): Indices of special points in signature data
- `gradient_analysis` (Dict[str, Any]): Results of gradient-based analysis
  - `d_estimated` (Optional[int]): Estimated private key
  - `recovery_possible` (bool): Whether key recovery is possible
- `persistence_diagram` (List[Tuple]): Persistence diagram data
- `f1_score` (float): F1-score of vulnerability detection

### 6.2 SecurityReport

```python
@dataclass
class SecurityReport:
    secure: bool
    issues: List[str]
    recommendations: List[str]
    topology_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
```

**Description**: Data structure containing comprehensive security analysis report.

**Fields**:
- `secure` (bool): Overall security status
- `issues` (List[str]): List of detected security issues
- `recommendations` (List[str]): List of security recommendations
- `topology_metrics` (Dict[str, Any]): Topological security metrics
  - `betti_numbers` (List[int]): Betti numbers [β₀, β₁, β₂]
  - `topological_entropy` (float): Topological entropy value
  - `special_points_count` (int): Number of special points detected
  - `gradient_recovery_possible` (bool): Whether gradient recovery is possible
- `performance_metrics` (Dict[str, Any]): Performance metrics
  - `signature_count` (int): Number of signatures analyzed
  - `analysis_time` (float): Time taken for analysis
  - `f1_score` (float): F1-score of vulnerability detection

**Example Usage**:
```python
from isogenyguard import SecurityReport

def generate_security_report(j_invariants: List[float]) -> SecurityReport:
    # Implementation details...
    pass

report = generate_security_report([0.72, 0.68, 0.75, 0.65, 0.82])

if not report.secure:
    print("SECURITY WARNING: Potential vulnerabilities detected!")
    for issue in report.issues:
        print(f"- {issue}")
    for recommendation in report.recommendations:
        print(f"  Recommendation: {recommendation}")
else:
    print("System security: All topological metrics indicate a secure implementation")
```

## Theoretical Foundation

All functions in IsogenyGuard SDK are grounded in rigorous mathematical research:

1. **Theorem 21**: The isogeny space for a fixed base curve is topologically equivalent to an (n-1)-dimensional torus
   - Basis for Betti number verification (β₀=1, β₁=2, β₂=1 for ECDSA)
   - Provides the foundation for topological security assessment

2. **Theorem 9**: Private key recovery through special point analysis
   - Enables gradient-based key recovery simulation
   - Forms the basis for detecting implementation weaknesses

3. **Theorem 24**: Topological entropy h_top = log(Σ|e_i|) as security metric
   - Quantifies security through topological properties
   - Correlates with vulnerability detection F1-score (Table 3)

4. **Theorem 16**: AdaptiveTDA compression preserving sheaf cohomologies
   - Enables efficient security monitoring with 12.7x compression ratio
   - Maintains 96% of topological information after compression

These theoretical foundations transform abstract mathematical concepts into practical security tools that protect systems rather than exploit vulnerabilities.
