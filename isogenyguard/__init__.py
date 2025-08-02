from .core import recover_private_key, check_special_points
from .topology import check_betti_numbers, calculate_topological_entropy

# Версия пакета
__version__ = "0.1.0"

# Определяем, что импортируется при использовании "from isogenyguard import *"
__all__ = [
    "recover_private_key",
    "check_special_points",
    "check_betti_numbers",
    "calculate_topological_entropy"
]

# Информационное сообщение о проекте
def info():
    """Возвращает информацию о проекте IsogenyGuard"""
    return """
    IsogenyGuard SDK v{version}
    Topological auditing of cryptographic keys based on Betti numbers analysis.
    "Topology is not a hacking tool, but a microscope for vulnerability diagnostics."
    """.format(version=__version__)
