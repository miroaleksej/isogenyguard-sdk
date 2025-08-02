# Import main functions for easy usage
from .core import recover_private_key, check_special_points
from .topology import check_betti_numbers, calculate_topological_entropy

# Package version
__version__ = "0.1.0"

# Define what is imported when using "from isogenyguard import *"
__all__ = [
    "recover_private_key",
    "check_special_points",
    "check_betti_numbers",
    "calculate_topological_entropy"
]

# Information function about the project
def info():
    """Returns information about IsogenyGuard project"""
    return """
    IsogenyGuard SDK v{version}
    Topological auditing of cryptographic keys based on Betti numbers analysis.
    "Topology is not a hacking tool, but a microscope for vulnerability diagnostics."
    """.format(version=__version__)
