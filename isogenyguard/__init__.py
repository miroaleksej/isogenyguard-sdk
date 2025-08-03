"""
IsogenyGuard SDK — Topological Auditing of Cryptographic Keys

Based on scientific research:
- Theorem 9: Private key recovery from special points
- Theorem 21: Isogeny space ≃ (n-1)-dimensional torus
- Theorem 16: AdaptiveTDA preserves sheaf cohomologies
- Theorem 7 & Table 3: Betti numbers and topological entropy as security metrics

"Topology is not a hacking tool, but a microscope for vulnerability diagnostics."
"""

from .core import recover_private_key
from .topology import check_betti_numbers, calculate_topological_entropy
from .utils import validate_implementation

version = "0.1.0"

__all__ = [
    "recover_private_key",
    "check_betti_numbers",
    "calculate_topological_entropy",
    "validate_implementation",
    "version"
]


def info():
    """
    Returns SDK version and scientific foundation.
    """
    return f"""
    IsogenyGuard SDK v{version}
    Topological auditing of cryptographic keys based on Betti numbers analysis.
    "Topology is not a hacking tool, but a microscope for vulnerability diagnostics."
    """
