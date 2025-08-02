# CONTRIBUTING

# Contributing to IsogenyGuard SDK

Thank you for considering contributing to IsogenyGuard SDK! This document outlines the process and guidelines for contributing to the project.

## 🤝 Our Philosophy

**Our mission: Protection, not exploitation.** All contributions must align with this core principle. We welcome contributions that:
- Improve security analysis capabilities
- Enhance vulnerability detection
- Strengthen protection mechanisms
- Improve documentation and usability

We do not accept contributions that:
- Focus on exploitation techniques
- Provide methods for attacking secure systems
- Bypass security measures without corresponding protection mechanisms
- Enable unauthorized key recovery

> "Topology is not a hacking tool, but a microscope for vulnerability diagnostics. Ignoring it means building cryptography on sand."

## 🛠 How to Contribute

### Reporting Bugs

1. Search through [existing issues](https://github.com/miroaleksej/isogenyguard-sdk/issues) to ensure the bug hasn't been reported already
2. Create a new issue with:
   - Clear title describing the problem
   - Steps to reproduce the issue
   - Expected behavior vs. actual behavior
   - System information (OS, Python version, etc.)
   - Any relevant logs or screenshots
   - Reference to relevant sections of the research (Theorems, Tables)

### Suggesting Enhancements

1. Search through [existing issues](https://github.com/miroaleksej/isogenyguard-sdk/issues) to ensure the enhancement hasn't been suggested already
2. Create a new issue with:
   - Clear title describing the enhancement
   - Detailed description of the problem it solves
   - Proposed solution with implementation details
   - Reference to relevant sections of the research (Theorems, Tables)
   - Expected impact on security metrics (Betti numbers, topological entropy, F1-score)

### Submitting Pull Requests

1. Fork the repository
2. Create a new branch for your feature/bugfix: `git checkout -b feature/your-feature-name`
3. Make your changes and commit them with descriptive messages
4. Push to your fork: `git push origin feature/your-feature-name`
5. Create a pull request with:
   - Clear title describing the changes
   - Detailed description of what was changed and why
   - Reference to related issue (if applicable)
   - Documentation updates (if applicable)
   - Test coverage for new functionality
   - Reference to theoretical basis from research (Theorems, Lemmas)

## 📚 Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for all public functions and classes
- Reference relevant theorems from the research in comments where applicable
- Maintain high test coverage (minimum 80%)
- Keep the "protection, not exploitation" principle in mind
- All new features must include:
  - Implementation of topological security metrics
  - Validation against research results (Table 3)
  - Documentation of security implications

## 🧪 Testing

All contributions must include appropriate tests:

```bash
# Run the test suite
pytest tests/

# Run tests with coverage report
pytest tests/ --cov=isogenyguard --cov-report=html
```

Your tests should:
- Verify correctness against known values from research (d=27, n=79)
- Check Betti numbers (β₀=1, β₁=2, β₂=1) for secure systems
- Validate topological entropy calculations
- Test vulnerability detection with F1-score up to 0.91
- Include edge cases and error handling

## 🔬 Research Integration

Contributions that integrate with the underlying research are highly valued. When adding new features:

1. Reference the relevant theorem or lemma from the research
2. Provide validation against Table 3 values
3. Document how the feature improves security metrics
4. Include visualizations of topological properties when applicable

Example of good documentation:

```python
def check_betti_numbers(j_invariants, n=2):
    """
    Calculate Betti numbers for the isogeny space
    
    Based on Theorem 21: The isogeny space for a fixed base curve 
    is topologically equivalent to an (n-1)-dimensional torus
    
    Expected values: β₀=1, β₁=n, β₂=1 (for n=2 as in ECDSA)
    
    Args:
        j_invariants: List of j-invariants from observed curves
        n: Space dimension (2 for ECDSA)
        
    Returns:
        Dictionary with Betti numbers and security flag
    """
    # Implementation...
```

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

## 💬 Questions?

If you have any questions about contributing, please open an issue with the label "question" or contact the maintainers.

---

**Remember**: All contributions must align with our mission of protection, not exploitation. We welcome your help in making cryptographic systems more secure through topological analysis.
