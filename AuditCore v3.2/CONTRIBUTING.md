# CONTRIBUTING to AuditCore v3.2

Thank you for your interest in contributing to AuditCore v3.2! This document outlines our contribution guidelines and processes to ensure smooth collaboration while maintaining the high standards of this industrial-grade topological security analysis framework.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving Documentation](#improving-documentation)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Mathematical Validation](#mathematical-validation)
- [License Considerations](#license-considerations)

## Code of Conduct
By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). We expect all contributors to maintain a professional, respectful attitude toward the sensitive nature of cryptographic security research.

## How Can I Contribute?

### Reporting Bugs
AuditCore v3.2 is a complex system that combines topological data analysis with elliptic curve cryptography. When reporting bugs:

1. **Check if the issue already exists** in our [issue tracker](https://github.com/yourusername/auditcore-v3.2/issues)
2. **Provide a clear, concise title** that summarizes the problem
3. **Include detailed information**:
   - Expected behavior vs. actual behavior
   - Steps to reproduce the issue
   - Relevant code snippets
   - System configuration (OS, Python version, hardware specs)
   - Error logs with stack traces
   - Mathematical context (if related to topological analysis)
4. **Label the issue** appropriately (bug, security, enhancement, etc.)

**Important**: If you discover a potential security vulnerability in ECDSA implementations through AuditCore, please follow our [responsible disclosure process](SECURITY.md) rather than creating a public issue.

### Suggesting Enhancements
We welcome suggestions for improvements to AuditCore v3.2. When proposing enhancements:

1. **Explain the problem** you're trying to solve
2. **Propose a specific solution** with technical details
3. **Include mathematical justification** where applicable (e.g., how it relates to Theorems 3, 5, or 9 from the documentation)
4. **Consider implementation complexity** and how it fits with the existing architecture
5. **Discuss potential impact** on:
   - Topological analysis capabilities
   - Performance characteristics
   - Resource requirements
   - Security implications

### Your First Code Contribution
1. **Fork the repository** and create a new branch for your feature/fix
2. **Ensure you understand the mathematical foundations** before modifying core algorithms
3. **Follow the existing code structure** and naming conventions
4. **Write comprehensive tests** for new functionality
5. **Update documentation** to reflect your changes
6. **Submit a pull request** with a clear description of your changes

### Improving Documentation
Clear documentation is critical for this mathematically complex system. Contributions can include:
- Expanding README with practical usage examples
- Improving docstrings for complex mathematical operations
- Creating Jupyter notebooks demonstrating specific analysis techniques
- Developing visualizations of topological concepts
- Translating documentation to other languages (with original English preserved)

## Development Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for acceleration)
- 16+ GB RAM (for large-scale topological analysis)
- Required libraries: giotto-tda (with GPU support), fastecdsa, NumPy, SciPy

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/auditcore-v3.2.git
cd auditcore-v3.2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install optional GPU acceleration
pip install "giotto-tda[gpu]"

# Install development dependencies
pip install -r dev-requirements.txt
```

### Testing Setup
AuditCore v3.2 requires rigorous testing due to its security-critical nature:
```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run mathematical validation tests
pytest tests/mathematical

# Check code style
flake8 .
```

## Pull Request Process

1. **Create a detailed description** of your changes, including:
   - The problem being solved
   - Mathematical justification (if applicable)
   - Performance impact analysis
   - Any new dependencies introduced

2. **Ensure all tests pass** before submitting

3. **Include appropriate documentation updates**

4. **Reference related issues** in your PR description

5. **Be prepared to discuss** the mathematical soundness of your implementation

6. **Maintain a clean commit history** with descriptive commit messages

7. **Respond promptly** to review comments

**Note**: Pull requests that modify core mathematical algorithms (bijective parameterization, topological analysis, key recovery mechanisms) require additional review from domain experts in both cryptography and topological data analysis.

## Coding Standards

AuditCore v3.2 follows strict coding standards due to its security-critical nature:

- **PEP 8 compliance** with some exceptions for mathematical clarity
- **Type hints** required for all functions and methods
- **Docstrings** following Google style format with mathematical notation
- **Immutable data structures** where possible
- **No global state** in analysis components
- **Resource management** with context managers for GPU/distributed resources
- **Error handling** for all cryptographic operations
- **Mathematical comments** explaining non-obvious operations

Example of acceptable code style:
```python
def compute_betti_numbers(
    point_cloud: np.ndarray,
    homology_dimensions: List[int] = [0, 1, 2],
    filtration_max: float = 0.5
) -> Dict[int, float]:
    """Computes Betti numbers for ECDSA signature space analysis.
    
    Implements persistent homology analysis as described in 
    "Logic, Structure, and Mathematical Model" (Section 4.1.1).
    
    Args:
        point_cloud: The R_x values from ECDSA signatures
        homology_dimensions: Dimensions to compute (typically [0, 1, 2])
        filtration_max: Maximum filtration value for VR complex
        
    Returns:
        Dictionary mapping dimension k to β_k (Betti number)
        
    Raises:
        ValueError: If point_cloud has insufficient points for analysis
        RuntimeError: If topological computation fails
    """
    # Verify we have enough points for meaningful topological analysis
    if len(point_cloud) < 10:
        raise ValueError("Insufficient points for topological analysis (minimum 10 required)")
    
    # Create Vietoris-Rips complex with appropriate parameters
    vr = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=homology_dimensions,
        n_jobs=-1
    )
    
    # Compute persistence diagram
    persistence_diagram = vr.fit_transform([point_cloud])[0]
    
    # Extract Betti numbers from infinite intervals
    betti_numbers = {}
    for dim in homology_dimensions:
        infinite_intervals = persistence_diagram[
            (persistence_diagram[:, 2] == float("inf")) & 
            (persistence_diagram[:, 1] == dim)
        ]
        betti_numbers[dim] = len(infinite_intervals)
    
    return betti_numbers
```

## Testing Requirements

Due to the mathematical and security-critical nature of AuditCore v3.2, we have stringent testing requirements:

### Unit Tests
- 100% coverage for core mathematical functions
- Boundary condition testing for all numeric operations
- Test cases for edge conditions (R = infinity, r = 0, s = 0)

### Integration Tests
- End-to-end analysis of known secure ECDSA implementations
- Analysis of intentionally vulnerable implementations (linear pattern, spiral pattern)
- Resource management tests across different data sizes
- GPU vs. CPU consistency checks

### Mathematical Validation
- Verification against theoretical expectations (e.g., torus structure β₀=1, β₁=2, β₂=1)
- Comparison with manual calculations for small test cases
- Statistical validation of pattern detection algorithms

### Performance Testing
- Benchmarking across different hardware configurations
- Memory usage profiling
- Time complexity verification

## Mathematical Validation

Contributions that modify core mathematical algorithms require special validation:

1. **Theoretical justification** must reference relevant theorems from the documentation
2. **Mathematical proofs** or derivations should accompany significant changes
3. **Verification against known results** must be provided
4. **Peer review** from domain experts is required for core algorithm changes

When modifying topological analysis components, contributors must demonstrate:
- Correctness of persistent homology calculations
- Proper interpretation of Betti numbers in ECDSA context
- Stability of results across parameter variations
- Mathematical soundness of vulnerability detection logic

## License Considerations

AuditCore v3.2 is released under a custom ethical license that permits research and educational use while prohibiting malicious application.

By contributing to this project, you agree that:
- Your contributions will be licensed under the same terms as the project
- You have the right to contribute the code
- You understand the ethical restrictions on usage

**Important**: Contributions must not include code that could be used to directly compromise cryptographic systems without proper authorization. All security findings must follow responsible disclosure practices.

---

Thank you for contributing to AuditCore v3.2! Your efforts help advance the field of topological security analysis while maintaining the highest ethical standards in cryptographic research.
