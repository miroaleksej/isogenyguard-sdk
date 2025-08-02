import numpy as np
from sklearn.neighbors import KernelDensity
from .utils import calculate_persistence_diagram

def check_betti_numbers(j_invariants, n=2):
    """
    Checks Betti numbers for the isogeny space
    Expected values: β₀=1, β₁=n, β₂=1 (for n=2 as in ECDSA)
    
    :param j_invariants: List of j-invariants from observed curves
    :param n: Space dimension (2 for ECDSA)
    :return: Dictionary with Betti numbers and security flag
    """
    # Calculate persistent homology
    persistence = calculate_persistence_diagram(j_invariants)
    
    # Calculate Betti numbers
    betti_0 = len([p for p in persistence if p[0] == 0 and p[1] == float('inf')])
    betti_1 = len([p for p in persistence if p[0] == 1])
    betti_2 = len([p for p in persistence if p[0] == 2])
    
    # Check against theoretical values
    is_secure = (betti_0 == 1 and 
                 betti_1 == n and 
                 betti_2 == 1)
    
    # Calculate topological entropy
    topological_entropy = np.log(sum(abs(p[2]) for p in persistence) + 1e-10) if persistence else 0.0
    
    return {
        "betti_0": betti_0,
        "betti_1": betti_1,
        "betti_2": betti_2,
        "is_secure": is_secure,
        "topological_entropy": topological_entropy,
        "persistence": persistence
    }

def calculate_topological_entropy(j_invariants):
    """
    Calculates topological entropy based on j-invariants
    
    :param j_invariants: List of j-invariants
    :return: Topological entropy value
    """
    if not j_invariants:
        return 0.0
    
    # Density distribution estimation
    kde = KernelDensity(bandwidth=0.5).fit(np.array(j_invariants).reshape(-1, 1))
    log_dens = kde.score_samples(np.array(j_invariants).reshape(-1, 1))
    
    # Entropy calculation
    entropy = -np.mean(log_dens)
    
    return entropy
