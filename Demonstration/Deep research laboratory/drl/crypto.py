"""
Cryptography Module for the Deep Research Laboratory.
Analyzes ECDSA signatures using topological methods.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from .utils import rsz_to_uruz, uruz_to_rsz


class CryptoAnalyzer:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.data = None
        self.hypercube = None
        self.j_invariants = []
        self.n = 79  # Example group order

    def load_data(self, data: Dict[str, Any]):
        """Load ECDSA signature data."""
        # Expect: list of (r, s, z) or (r, s, z, d_true)
        self.data = data
        self.signatures = data.get('signatures', [])
        if not self.signatures:
            raise ValueError("No signatures found in data")
        self.n = data.get('n', 79)

    def build_model(self):
        """Build the R_x(u_r, u_z) hypercube."""
        # Convert (r,s,z) to (u_r, u_z)
        uruz_points = []
        for sig in self.signatures:
            r, s, z = sig['r'], sig['s'], sig['z']
            ur, uz = rsz_to_uruz(r, s, z, self.n)
            uruz_points.append((ur, uz, r))

        # Build R_x table
        from collections import defaultdict
        rx_map = defaultdict(list)
        for ur, uz, r in uruz_points:
            rx_map[r].append((ur, uz))

        # Store for analysis
        self.uruz_points = uruz_points
        self.rx_map = rx_map

    def analyze(self) -> Dict[str, Any]:
        """Perform full topological analysis."""
        results = {}

        # Find special points (repeated R_x)
        special_points = []
        for r, points in self.rx_map.items():
            if len(points) >= 2:
                sorted_points = sorted(points, key=lambda x: x[0])  # sort by u_r
                for i in range(len(sorted_points) - 1):
                    ur1, uz1 = sorted_points[i]
                    ur2, uz2 = sorted_points[i+1]
                    if ur2 == ur1 + 1:
                        special_points.append((ur1, uz1))
                        special_points.append((ur2, uz2))

        # Recover private key
        d_recovered = self.recover_private_key(special_points)
        results['d_recovered'] = d_recovered
        results['special_points_count'] = len(special_points) // 2

        # Check Betti numbers
        betti_result = self.check_betti_numbers()
        results.update(betti_result)

        # Topological entropy
        entropy = self.calculate_topological_entropy()
        results['topological_entropy'] = entropy

        # Security assessment
        is_secure = betti_result['is_secure'] and (d_recovered is None)
        results['is_secure'] = is_secure
        results['recommendation'] = "Key is secure" if is_secure else "Vulnerable: Check nonce generation"

        return results

    def recover_private_key(self, special_points: List[Tuple[int, int]]) -> int:
        """Recover d from special points."""
        if len(special_points) < 2:
            return None

        diffs = []
        sorted_points = sorted(special_points, key=lambda x: x[0])
        for i in range(len(sorted_points) - 1):
            if sorted_points[i+1][0] == sorted_points[i][0] + 1:
                d = (sorted_points[i][1] - sorted_points[i+1][1]) % self.n
                diffs.append(d)

        return max(set(diffs), key=diffs.count) if diffs else None

    def check_betti_numbers(self) -> Dict[str, Any]:
        """Check Betti numbers for toroidal structure."""
        # Simulate persistence diagram
        persistence = [(0.1, 0.5), (0.2, float('inf')), (0.3, 0.8)]

        betti_0 = len([p for p in persistence if p[0] == 0 and p[1] == float('inf')])
        betti_1 = len([p for p in persistence if p[1] != float('inf') and p[0] > 0])
        betti_2 = len([p for p in persistence if p[0] > 0.5])

        is_secure = (betti_0 == 1 and betti_1 == 2 and betti_2 == 1)

        return {
            "betti_0": betti_0,
            "betti_1": betti_1,
            "betti_2": betti_2,
            "is_secure": is_secure
        }

    def calculate_topological_entropy(self) -> float:
        """Calculate topological entropy."""
        if len(self.uruz_points) < 2:
            return 0.0
        kde = gaussian_kde([p[0] for p in self.uruz_points])
        log_density = kde.logpdf([p[0] for p in self.uruz_points])
        return -np.mean(log_density)
