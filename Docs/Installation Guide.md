# IsogenyGuard SDK Installation Guide

## 📋 System Requirements

Before installing IsogenyGuard SDK, ensure your system meets these requirements:

- **Operating System**: Windows, macOS, or Linux
- **Python Version**: 3.8 or higher
- **Memory**: Minimum 2GB RAM (4GB recommended for large-scale analysis)
- **Disk Space**: 100MB for core installation, additional space required for dependencies

## 📦 Installation Methods

### Method 1: Install from PyPI (Recommended)

The simplest way to install IsogenyGuard SDK is via PyPI:

```bash
# Create a virtual environment (recommended)
python -m venv isogenyguard-env
source isogenyguard-env/bin/activate  # Linux/MacOS
# isogenyguard-env\Scripts\activate  # Windows

# Install the package
pip install isogenyguard
```

### Method 2: Install from Source

For developers or those who want the latest features:

```bash
# Clone the repository
git clone https://github.com/miroaleksej/isogenyguard-sdk.git
cd isogenyguard-sdk

# Create a virtual environment
python -m venv isogenyguard-dev
source isogenyguard-dev/bin/activate  # Linux/MacOS
# isogenyguard-dev\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for topological analysis
pip install ripser scikit-learn

# Install the package in development mode
pip install -e .
```

## ⚙️ Required Dependencies

IsogenyGuard SDK requires the following dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Numerical computations |
| ripser | >=0.6.0 | Persistent homology calculations |
| scikit-learn | >=1.0.0 | Machine learning utilities |
| scipy | >=1.7.0 | Scientific computing |

These will be automatically installed when you install IsogenyGuard via pip. If you need to install them manually:

```bash
pip install numpy>=1.21.0 ripser>=0.6.0 scikit-learn>=1.0.0 scipy>=1.7.0
```

## ✅ Verification

After installation, verify that IsogenyGuard is working correctly:

```bash
# Run the basic example
python -c "from isogenyguard import check_betti_numbers; \
           result = check_betti_numbers([0.72, 0.68, 0.75, 0.65, 0.82]); \
           print(f'Security status: {"SECURE" if result["is_secure"] else "VULNERABLE"}')"
```

You should see output similar to:
```
Security status: SECURE
```

For a more comprehensive verification, run the example script:

```bash
# Run the full example
python examples/basic_usage.py
```

Expected output should show:
- Recovered private key: d = 27 (matching the expected value)
- Betti numbers: β₀=1, β₁=2, β₂=1 for secure system
- Topological entropy around 3.3 for secure system

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Installation fails with "ModuleNotFoundError"

**Problem**: `ModuleNotFoundError: No module named 'isogenyguard'` after installation

**Solution**: Ensure you're using the correct Python environment:
```bash
# Check which Python is being used
which python  # Linux/MacOS
# where python  # Windows

# Reinstall in the correct environment
pip uninstall isogenyguard -y
pip install isogenyguard
```

#### 2. Ripser installation fails

**Problem**: Compilation errors when installing ripser

**Solution**: Install system dependencies first:
- **Ubuntu/Debian**: `sudo apt-get install build-essential python3-dev`
- **macOS**: Install Xcode command line tools: `xcode-select --install`
- **Windows**: Install Microsoft Visual C++ Build Tools

#### 3. Low topological entropy in verification

**Problem**: Example shows low topological entropy despite secure system

**Solution**: This may indicate an issue with the ripser installation. Try:
```bash
pip uninstall ripser -y
pip install --no-cache-dir ripser
```

## 🧪 Testing the Installation

To verify your installation is fully functional, run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Expected output: 7 passed in X.XXs
```

A successful test run should show all tests passing, confirming that:
- Topological analysis functions work correctly
- Betti number calculation matches expected values
- Key recovery works as documented in Theorem 9
- AdaptiveTDA compression achieves 12.7x ratio as in research

## 💡 Tips for Production Use

1. **Virtual Environments**: Always use virtual environments for isolation
2. **Dependency Pinning**: For production use, pin specific versions:
   ```bash
   pip install "isogenyguard==0.1.0" "ripser==0.6.0" "scikit-learn==1.0.0"
   ```
3. **GPU Acceleration**: For large-scale analysis, consider installing GPU-accelerated versions of dependencies:
   ```bash
   pip install tensorflow  # For GPU-accelerated computations
   ```
