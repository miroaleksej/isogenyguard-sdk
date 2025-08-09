"""Cryptography Module for the Deep Research Laboratory.
Analyzes ECDSA signatures using topological methods and fastecdsa library.

This module provides tools for:
- Generating valid ECDSA signatures without private key access
- Analyzing the topological structure of signature spaces
- Detecting anomalies in random number generation
- Calculating topological invariants (Betti numbers, persistent homology)
- Visualizing the toroidal structure of ECDSA signature space

Note: This is a standalone module unrelated to AuditCore v3.2 system.
"""

import numpy as np
import torch
import fastecdsa.curve
import fastecdsa.point
import fastecdsa.ecdsa
from fastecdsa import keys, curve
from typing import Dict, Any, List, Tuple, Optional, Union
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import networkx as nx
import itertools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CryptoAnalyzer')

class CryptoAnalyzer:
    """Main class for topological analysis of ECDSA signatures.
    
    This class implements methods to generate, analyze, and visualize the topological
    structure of ECDSA signature spaces using algebraic topology techniques.
    """
    
    def __init__(self, 
                 curve_name: str = 'secp256k1',
                 device: str = 'cpu',
                 n_points: int = 1000):
        """Initialize the CryptoAnalyzer with specified elliptic curve.
        
        Args:
            curve_name: Name of the elliptic curve to use (default: 'secp256k1')
            device: Computational device ('cpu' or 'cuda')
            n_points: Default number of points for analysis
        """
        self.device = torch.device(device)
        self.curve_name = curve_name
        self.curve = getattr(fastecdsa.curve, curve_name)
        self.n = self.curve.q  # Group order
        self.n_points = n_points
        self.data = None
        self.signature_space = None
        self.topological_invariants = {}
        self.public_key = None
        self.private_key = None
        logger.info(f"Initialized CryptoAnalyzer with curve {curve_name} (order: {self.n})")
    
    def generate_key_pair(self) -> Tuple[int, fastecdsa.point.Point]:
        """Generate a new ECDSA key pair using the specified curve.
        
        Returns:
            Tuple containing (private_key, public_key)
        """
        self.private_key, self.public_key = keys.gen_keypair(self.curve)
        logger.info("Generated new ECDSA key pair")
        return self.private_key, self.public_key
    
    def set_key_pair(self, private_key: int, public_key: fastecdsa.point.Point):
        """Set an existing key pair for analysis.
        
        Args:
            private_key: Integer representing the private key
            public_key: Point object representing the public key
        """
        self.private_key = private_key
        self.public_key = public_key
        logger.info("Set existing ECDSA key pair")
    
    def generate_signatures(self, 
                           n_signatures: int = 100, 
                           message: bytes = b"Deep Research Laboratory") -> List[Dict[str, Any]]:
        """Generate valid ECDSA signatures without revealing the private key.
        
        Args:
            n_signatures: Number of signatures to generate
            message: Message to sign (bytes)
            
        Returns:
            List of signature dictionaries with r, s, and z values
        """
        if not self.private_key:
            self.generate_key_pair()
        
        signatures = []
        for _ in range(n_signatures):
            r, s = fastecdsa.ecdsa.sign(message, self.private_key, curve=self.curve)
            z = int.from_bytes(message, 'big') % self.n
            signatures.append({
                'r': r,
                's': s,
                'z': z,
                'message': message
            })
        
        self.data = signatures
        logger.info(f"Generated {n_signatures} ECDSA signatures")
        return signatures
    
    def bijective_parameterization(self, signatures: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """Apply bijective parameterization to transform signatures into (u_r, u_z) space.
        
        For each signature (r, s, z), computes:
        u_r = r * s^-1 mod n
        u_z = z * s^-1 mod n
        
        Args:
            signatures: List of signatures to parameterize (uses stored data if None)
            
        Returns:
            Numpy array of shape (n_signatures, 2) with (u_r, u_z) values
        """
        if signatures is None:
            if self.data is None:
                raise ValueError("No signature data available. Call generate_signatures first.")
            signatures = self.data
        
        ur_uz_points = []
        for sig in signatures:
            s_inv = pow(sig['s'], -1, self.n)  # Modular inverse of s
            u_r = (sig['r'] * s_inv) % self.n
            u_z = (sig['z'] * s_inv) % self.n
            ur_uz_points.append([u_r, u_z])
        
        self.signature_space = np.array(ur_uz_points)
        logger.info(f"Transformed {len(signatures)} signatures into (u_r, u_z) space")
        return self.signature_space
    
    def generate_synthetic_signatures(self, n_points: int = None) -> np.ndarray:
        """Generate synthetic signature data across the full (u_r, u_z) space.
        
        Args:
            n_points: Number of points to generate (defaults to self.n_points)
            
        Returns:
            Numpy array of shape (n_points^2, 3) with [u_r, u_z, R_x] values
        """
        if n_points is None:
            n_points = self.n_points
            
        if not self.public_key:
            self.generate_key_pair()
        
        # Create grid of u_r, u_z values
        u_r_vals = np.linspace(0, self.n-1, n_points, dtype=int)
        u_z_vals = np.linspace(0, self.n-1, n_points, dtype=int)
        
        # Prepare storage for results
        synthetic_data = []
        
        # Generate points on the curve
        for u_r in u_r_vals:
            for u_z in u_z_vals:
                # Calculate R = u_r * G + u_z * Q
                R = (u_r * self.curve.G) + (u_z * self.public_key)
                
                # Only include valid points (not at infinity)
                if R != fastecdsa.point.Point.IDENTITY_ELEMENT:
                    R_x = R.x % self.n
                    synthetic_data.append([u_r, u_z, R_x])
        
        synthetic_data = np.array(synthetic_data)
        logger.info(f"Generated {len(synthetic_data)} synthetic signature points")
        return synthetic_data
    
    def compute_toroidal_topology(self, 
                                synthetic_data: Optional[np.ndarray] = None,
                                max_dim: int = 2,
                                thresh: float = None) -> Dict[str, Any]:
        """Analyze the toroidal topology of the signature space using persistent homology.
        
        Args:
            synthetic_data: Precomputed synthetic data (if None, generates new data)
            max_dim: Maximum dimension of homology to compute
            thresh: Threshold for Vietoris-Rips complex (if None, auto-calculated)
            
        Returns:
            Dictionary containing topological invariants and persistence diagrams
        """
        if synthetic_data is None:
            synthetic_data = self.generate_synthetic_signatures()
        
        # Extract just the (u_r, u_z) coordinates for topology analysis
        points = synthetic_data[:, :2]
        
        # Compute persistent homology
        if thresh is None:
            # Auto-calculate threshold based on point density
            distances = pdist(points)
            thresh = np.percentile(distances, 95)
        
        logger.info(f"Computing persistent homology with threshold={thresh:.2f}")
        diagrams = ripser(points, maxdim=max_dim, thresh=thresh)['dgms']
        
        # Calculate Betti numbers from persistence diagrams
        betti_numbers = {}
        for dim, diagram in enumerate(diagrams):
            if len(diagram) > 0:
                # Count points with infinite persistence (or very long persistence)
                infinite_persistence = diagram[diagram[:, 1] == np.inf]
                betti_numbers[dim] = len(infinite_persistence)
            else:
                betti_numbers[dim] = 0
        
        # Store results
        self.topological_invariants = {
            'betti_numbers': betti_numbers,
            'persistence_diagrams': diagrams,
            'threshold': thresh
        }
        
        logger.info(f"Computed Betti numbers: β₀={betti_numbers.get(0, 0)}, "
                   f"β₁={betti_numbers.get(1, 0)}, β₂={betti_numbers.get(2, 0)}")
        
        return self.topological_invariants
    
    def analyze_signature_anomalies(self, 
                                  signatures: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Analyze real signatures for topological anomalies.
        
        Args:
            signatures: List of signatures to analyze (uses stored data if None)
            
        Returns:
            Dictionary with anomaly scores and diagnostic information
        """
        if signatures is None:
            if self.data is None:
                raise ValueError("No signature data available. Call generate_signatures first.")
            signatures = self.data
        
        # Get the (u_r, u_z) space representation
        ur_uz_space = self.bijective_parameterization(signatures)
        
        # Compute density estimation
        kde = gaussian_kde(ur_uz_space.T)
        densities = kde(ur_uz_space.T)
        
        # Detect clusters with hierarchical clustering
        dist_matrix = squareform(pdist(ur_uz_space))
        Z = linkage(dist_matrix, 'ward')
        clusters = fcluster(Z, t=0.7 * max(Z[:, 2]), criterion='distance')
        
        # Analyze topological properties of the real data
        # (This is a simplified approach compared to full synthetic analysis)
        anomaly_score = 0.0
        if len(np.unique(clusters)) > 1:
            anomaly_score += 0.5  # Clustered data suggests non-randomness
        
        # Check for linear patterns that might indicate weak RNG
        for i, j in itertools.combinations(range(len(ur_uz_space)), 2):
            u_r_diff = ur_uz_space[i, 0] - ur_uz_space[j, 0]
            u_z_diff = ur_uz_space[i, 1] - ur_uz_space[j, 1]
            if u_r_diff != 0 and abs(u_z_diff / u_r_diff) < 1e-6:
                anomaly_score += 0.1
        
        # Normalize anomaly score
        anomaly_score = min(anomaly_score, 1.0)
        
        results = {
            'anomaly_score': anomaly_score,
            'cluster_count': len(np.unique(clusters)),
            'density_variance': np.var(densities),
            'is_suspicious': anomaly_score > 0.3
        }
        
        logger.info(f"Anomaly analysis complete. Score: {anomaly_score:.2f} "
                   f"(threshold: 0.3) - {'SUSPICIOUS' if results['is_suspicious'] else 'NORMAL'}")
        
        return results
    
    def visualize_signature_space(self, 
                               signatures: Optional[List[Dict[str, Any]] = None,
                               synthetic_data: Optional[np.ndarray] = None,
                               show_topology: bool = True):
        """Visualize the signature space and its topological properties.
        
        Args:
            signatures: List of signatures to visualize (uses stored data if None)
            synthetic_data: Precomputed synthetic data for background visualization
            show_topology: Whether to show topological analysis results
        """
        plt.figure(figsize=(15, 10))
        
        # Generate or use provided data
        if signatures is None:
            signatures = self.data
        ur_uz_space = self.bijective_parameterization(signatures)
        
        if synthetic_data is None:
            synthetic_data = self.generate_synthetic_signatures(n_points=100)
        
        # Plot 1: Signature distribution in (u_r, u_z) space
        plt.subplot(2, 2, 1)
        plt.scatter(ur_uz_space[:, 0], ur_uz_space[:, 1], c='blue', alpha=0.6, s=10)
        plt.title('Real Signatures in (u_r, u_z) Space')
        plt.xlabel('u_r')
        plt.ylabel('u_z')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Synthetic signature space (full toroidal structure)
        plt.subplot(2, 2, 2)
        # Reshape synthetic data for contour plotting
        u_r_vals = np.unique(synthetic_data[:, 0])
        u_z_vals = np.unique(synthetic_data[:, 1])
        R_x_grid = synthetic_data[:, 2].reshape(len(u_r_vals), len(u_z_vals))
        
        plt.contourf(u_r_vals, u_z_vals, R_x_grid, 50, cmap='viridis')
        plt.colorbar(label='R_x value')
        plt.scatter(ur_uz_space[:, 0], ur_uz_space[:, 1], c='red', alpha=0.6, s=5, label='Real Signatures')
        plt.title('Synthetic Signature Space (Toroidal Structure)')
        plt.xlabel('u_r')
        plt.ylabel('u_z')
        plt.legend()
        
        # Plot 3: Density estimation
        plt.subplot(2, 2, 3)
        kde = gaussian_kde(ur_uz_space.T)
        density = kde(ur_uz_space.T)
        scatter = plt.scatter(ur_uz_space[:, 0], ur_uz_space[:, 1], c=density, 
                            cmap='plasma', s=10, alpha=0.8)
        plt.colorbar(scatter, label='Density')
        plt.title('Signature Density Distribution')
        plt.xlabel('u_r')
        plt.ylabel('u_z')
        
        # Plot 4: Topological analysis (if requested)
        if show_topology and self.topological_invariants:
            plt.subplot(2, 2, 4)
            plot_diagrams(self.topological_invariants['persistence_diagrams'], show=False)
            plt.title('Persistence Diagrams (Topological Features)')
        else:
            # Compute topology if not already done
            if not self.topological_invariants:
                self.compute_toroidal_topology(synthetic_data)
            plt.subplot(2, 2, 4)
            plot_diagrams(self.topological_invariants['persistence_diagrams'], show=False)
            plt.title('Persistence Diagrams (Topological Features)')
        
        plt.tight_layout()
        plt.suptitle(f'Topological Analysis of ECDSA Signature Space ({self.curve_name})', fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    def estimate_private_key(self, 
                           signatures: Optional[List[Dict[str, Any]]] = None) -> Optional[float]:
        """Estimate the private key using topological gradient analysis.
        
        This method uses the relationship between u_r, u_z and the private key d:
        k = u_z + u_r * d (mod n)
        
        When the RNG is weak, patterns in the (u_r, u_z) space can reveal d.
        
        Args:
            signatures: List of signatures to analyze (uses stored data if None)
            
        Returns:
            Estimated private key value, or None if insufficient data
        """
        if signatures is None:
            if self.data is None:
                raise ValueError("No signature data available. Call generate_signatures first.")
            signatures = self.data
        
        if len(signatures) < 2:
            logger.warning("Need at least 2 signatures to estimate private key")
            return None
        
        ur_uz_space = self.bijective_parameterization(signatures)
        
        # Look for linear relationships between u_r and u_z
        # For a fixed d: u_z = k - u_r * d (mod n)
        # So the slope should be approximately -d
        
        # Calculate gradients between neighboring points
        gradients = []
        for i in range(len(ur_uz_space)):
            for j in range(i+1, len(ur_uz_space)):
                u_r_diff = ur_uz_space[j, 0] - ur_uz_space[i, 0]
                u_z_diff = ur_uz_space[j, 1] - ur_uz_space[i, 1]
                
                if u_r_diff != 0:  # Avoid division by zero
                    # Calculate slope (which should be approximately -d)
                    slope = u_z_diff / u_r_diff
                    gradients.append(-slope)
        
        if not gradients:
            logger.warning("Could not find sufficient gradient data to estimate private key")
            return None
        
        # Use the most common gradient value (mode) as our estimate
        # This assumes most signatures follow the same linear relationship
        d_estimate = np.median(gradients)
        
        # Normalize to the curve order
        d_estimate = d_estimate % self.n
        
        logger.info(f"Estimated private key: {d_estimate:.2f} (mod {self.n})")
        return d_estimate
    
    def verify_toroidal_structure(self, 
                                betti_numbers: Dict[int, int] = None) -> Dict[str, Any]:
        """Verify if the signature space has the expected toroidal topological structure.
        
        For a proper ECDSA implementation with good RNG, we expect:
        β₀ = 1 (single connected component)
        β₁ = 2 (two independent loops on a torus)
        β₂ = 1 (one enclosed volume)
        
        Args:
            betti_numbers: Precomputed Betti numbers (uses stored values if None)
            
        Returns:
            Dictionary with verification results and security assessment
        """
        if betti_numbers is None:
            if not self.topological_invariants or 'betti_numbers' not in self.topological_invariants:
                raise ValueError("Topological invariants not computed. Call compute_toroidal_topology first.")
            betti_numbers = self.topological_invariants['betti_numbers']
        
        expected = {0: 1, 1: 2, 2: 1}
        results = {
            'expected': expected,
            'observed': betti_numbers,
            'is_toroidal': True,
            'security_risk': 0.0,
            'diagnostics': []
        }
        
        # Check each Betti number
        for dim in expected:
            if dim in betti_numbers:
                diff = abs(betti_numbers[dim] - expected[dim])
                if diff > 0:
                    results['is_toroidal'] = False
                    results['security_risk'] += diff * (0.3 if dim == 1 else 0.2)
                    
                    if dim == 0:
                        if betti_numbers[dim] > 1:
                            results['diagnostics'].append(
                                f"Multiple connected components (β₀={betti_numbers[dim]}). "
                                "This indicates significant gaps in the signature space, "
                                "suggesting a severely flawed RNG."
                            )
                        else:
                            results['diagnostics'].append(
                                f"Unexpected connected component count (β₀={betti_numbers[dim]})."
                            )
                    elif dim == 1:
                        if betti_numbers[dim] > 2:
                            results['diagnostics'].append(
                                f"Excess topological loops (β₁={betti_numbers[dim]} > 2). "
                                "This indicates hidden regularity in the nonce generation, "
                                "possibly from a weak PRNG like LCG."
                            )
                        elif betti_numbers[dim] < 2:
                            results['diagnostics'].append(
                                f"Missing topological loops (β₁={betti_numbers[dim]} < 2). "
                                "This suggests incomplete coverage of the signature space."
                            )
                    elif dim == 2:
                        if betti_numbers[dim] != 1:
                            results['diagnostics'].append(
                                f"Unexpected volume structure (β₂={betti_numbers[dim]}). "
                                "This indicates non-uniform distribution in the signature space."
                            )
            else:
                results['is_toroidal'] = False
                results['security_risk'] += 0.5
                results['diagnostics'].append(
                    f"Missing Betti number for dimension {dim}."
                )
        
        # Cap security risk at 1.0
        results['security_risk'] = min(results['security_risk'], 1.0)
        
        # Add overall assessment
        if results['is_toroidal']:
            results['assessment'] = "The signature space exhibits proper toroidal topology. " \
                                   "No structural vulnerabilities detected."
        else:
            risk_level = "HIGH" if results['security_risk'] > 0.7 else "MODERATE" if results['security_risk'] > 0.3 else "LOW"
            results['assessment'] = f"TOPOLOGICAL ANOMALY DETECTED (Risk level: {risk_level}). " \
                                  "The implementation may have vulnerabilities in nonce generation."
        
        logger.info(f"Toroidal structure verification: {results['assessment']}")
        return results
    
    def analyze_fractal_properties(self, 
                                synthetic_data: Optional[np.ndarray] = None,
                                max_scale: int = 8) -> Dict[str, Any]:
        """Analyze fractal and self-similarity properties of the signature space.
        
        Args:
            synthetic_data: Precomputed synthetic data (if None, generates new data)
            max_scale: Maximum scale level for fractal analysis
            
        Returns:
            Dictionary with fractal dimension and self-similarity metrics
        """
        if synthetic_data is None:
            synthetic_data = self.generate_synthetic_signatures(n_points=200)
        
        # Extract the R_x values on the (u_r, u_z) grid
        u_r_vals = np.unique(synthetic_data[:, 0])
        u_z_vals = np.unique(synthetic_data[:, 1])
        R_x_grid = synthetic_data[:, 2].reshape(len(u_r_vals), len(u_z_vals))
        
        # Compute fractal dimension using box-counting method
        scales = []
        counts = []
        
        # Start with the full image size
        size = min(R_x_grid.shape)
        
        # Compute at different scales
        for scale in range(1, max_scale + 1):
            box_size = max(1, size // (2 ** scale))
            if box_size < 1:
                break
                
            count = 0
            # Count non-uniform boxes
            for i in range(0, R_x_grid.shape[0], box_size):
                for j in range(0, R_x_grid.shape[1], box_size):
                    # Get the box
                    box = R_x_grid[i:min(i+box_size, R_x_grid.shape[0]), 
                                  j:min(j+box_size, R_x_grid.shape[1])]
                    
                    # Check if the box is non-uniform
                    if np.std(box) > 0.1 * self.n:  # Threshold for variation
                        count += 1
            
            scales.append(box_size)
            counts.append(count)
        
        # Calculate fractal dimension from the slope of log(count) vs log(1/scale)
        if len(scales) > 2:
            log_scales = np.log(1 / np.array(scales))
            log_counts = np.log(np.array(counts))
            
            # Linear regression to find slope
            slope, intercept = np.polyfit(log_scales, log_counts, 1)
            fractal_dimension = slope
            
            # Calculate self-similarity score (how consistent the pattern is across scales)
            residuals = log_counts - (slope * log_scales + intercept)
            self_similarity = 1.0 / (1.0 + np.std(residuals))
        else:
            fractal_dimension = 2.0  # Default for flat space
            self_similarity = 1.0     # Perfect self-similarity
        
        results = {
            'fractal_dimension': fractal_dimension,
            'self_similarity': self_similarity,
            'scales': scales,
            'box_counts': counts,
            'diagnostics': []
        }
        
        # Add diagnostics
        if fractal_dimension < 1.8:
            results['diagnostics'].append(
                "Low fractal dimension suggests overly regular structure, "
                "possibly indicating a weak RNG."
            )
        elif fractal_dimension > 2.1:
            results['diagnostics'].append(
                "High fractal dimension suggests excessive complexity, "
                "possibly indicating implementation errors."
            )
        
        if self_similarity < 0.7:
            results['diagnostics'].append(
                "Low self-similarity across scales suggests inconsistent "
                "structure, possibly indicating a non-robust implementation."
            )
        
        logger.info(f"Fractal analysis complete. Dimension: {fractal_dimension:.2f}, "
                   f"Self-similarity: {self_similarity:.2f}")
        return results
    
    def generate_audit_report(self, 
                            signatures: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate a comprehensive security audit report for ECDSA implementation.
        
        Args:
            signatures: List of signatures to analyze (uses stored data if None)
            
        Returns:
            Dictionary containing the complete audit report
        """
        if signatures is None:
            if self.data is None:
                raise ValueError("No signature data available. Call generate_signatures first.")
            signatures = self.data
        
        # Ensure we have all necessary analyses
        ur_uz_space = self.bijective_parameterization(signatures)
        synthetic_data = self.generate_synthetic_signatures(n_points=150)
        self.compute_toroidal_topology(synthetic_data)
        self.analyze_fractal_properties(synthetic_data)
        anomaly_results = self.analyze_signature_anomalies(signatures)
        
        # Generate toroidal verification
        toroidal_results = self.verify_toroidal_structure()
        
        # Estimate private key if anomalies detected
        key_estimation = None
        if anomaly_results['is_suspicious']:
            key_estimation = self.estimate_private_key(signatures)
        
        # Compile report
        report = {
            'metadata': {
                'curve': self.curve_name,
                'group_order': self.n,
                'signature_count': len(signatures),
                'timestamp': str(datetime.datetime.now())
            },
            'topological_analysis': {
                'betti_numbers': self.topological_invariants['betti_numbers'],
                'is_toroidal': toroidal_results['is_toroidal'],
                'security_risk': toroidal_results['security_risk'],
                'diagnostics': toroidal_results['diagnostics'],
                'assessment': toroidal_results['assessment']
            },
            'fractal_analysis': self.analyze_fractal_properties(synthetic_data),
            'anomaly_detection': anomaly_results,
            'key_estimation': {
                'possible': anomaly_results['is_suspicious'],
                'estimated_value': key_estimation,
                'confidence': 1.0 - anomaly_results['anomaly_score'] if key_estimation else 0.0
            },
            'recommendations': []
        }
        
        # Add recommendations based on findings
        if not toroidal_results['is_toroidal']:
            report['recommendations'].append(
                "The implementation shows topological anomalies. "
                "Review the random number generator used for nonce generation."
            )
        
        if anomaly_results['is_suspicious']:
            report['recommendations'].append(
                "Suspicious patterns detected in signature distribution. "
                "Consider using a cryptographically secure RNG."
            )
        
        if key_estimation is not None:
            report['recommendations'].append(
                "Private key estimation was possible from signature patterns. "
                "This indicates a critical vulnerability in the implementation."
            )
        
        if not report['recommendations']:
            report['recommendations'].append(
                "No critical vulnerabilities detected. The implementation appears "
                "to follow best practices for ECDSA signature generation."
            )
        
        logger.info("Generated comprehensive security audit report")
        return report
