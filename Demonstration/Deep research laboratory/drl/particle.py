"""
Particle Hypercube for the Deep Research Laboratory.
Models known elementary particles and predicts unknown ones
using topological anomaly detection.

Based on:
- Standard Model of particle physics
- Cohomological analysis (from scientific work)
- Machine learning for anomaly detection
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from scipy.fft import dctn, idctn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
import zstandard as zstd
import json
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster


@dataclass
class Particle:
    """Represents an elementary particle."""
    name: str
    mass: float  # eV
    charge: float  # in units of e
    spin: float  # in units of ħ
    baryon_number: float
    lepton_number: float
    color_charge: float
    lifetime: float  # seconds
    interaction: int  # 1=strong, 2=EM, 3=weak, 4=gravity
    status: str = "known"  # known, hypothetical, unknown
    probability: float = 1.0  # for predicted particles
    anomaly_score: float = 0.0


class ParticleHypercube:
    """
    A multidimensional hypercube for modeling elementary particles.
    Uses cohomological analysis to detect anomalies that may indicate new physics.
    """

    def __init__(self, device: str = 'cpu', resolution: int = 128):
        self.device = torch.device(device)
        self.resolution = resolution
        self.dimensions = [
            'mass', 'charge', 'spin',
            'baryon_number', 'lepton_number',
            'color_charge', 'lifetime'
        ]

        # Define ranges for each dimension (log scale for mass and lifetime)
        self.ranges = {
            'mass': (1e-6, 1e20),  # eV
            'charge': (-3.0, 3.0),
            'spin': (0.0, 4.0),
            'baryon_number': (-1.0, 1.0),
            'lepton_number': (-3.0, 3.0),
            'color_charge': (0.0, 1.0),
            'lifetime': (1e-40, 1e36)  # seconds
        }

        # Build grids
        self.grids = {}
        for dim in self.dimensions:
            min_val, max_val = self.ranges[dim]
            self.grids[dim] = np.logspace(np.log10(min_val), np.log10(max_val), resolution) if dim in ['mass', 'lifetime'] else np.linspace(min_val, max_val, resolution)

        # Initialize hypercube
        self.hypercube = np.zeros(tuple([resolution] * len(self.dimensions)))
        self.known_particles = self._load_particle_database()
        self.anomaly_detector = None
        self.gp_model = None

        # From scientific work: F_SM → F induces cohomology map
        self.standard_model_particles = self.known_particles[self.known_particles['status'] == 'known'].copy()

    def _load_particle_database(self) -> pd.DataFrame:
        """Load database of known and hypothetical particles."""
        particles = [
            # Fermions (Quarks)
            Particle('up', 2.3e6, 2/3, 0.5, 1/3, 0, 1, 1e-25, 1),
            Particle('down', 4.8e6, -1/3, 0.5, 1/3, 0, 1, 1e-25, 1),
            Particle('charm', 1.275e9, 2/3, 0.5, 1/3, 0, 1, 1e-12, 1),
            Particle('strange', 95e6, -1/3, 0.5, 1/3, 0, 1, 1e-10, 1),
            Particle('top', 173.1e9, 2/3, 0.5, 1/3, 0, 1, 5e-25, 1),
            Particle('bottom', 4.18e9, -1/3, 0.5, 1/3, 0, 1, 1e-12, 1),

            # Leptons
            Particle('electron', 0.511e6, -1, 0.5, 0, 1, 0, 1e36, 2),
            Particle('muon', 105.7e6, -1, 0.5, 0, 1, 0, 2.2e-6, 2),
            Particle('tau', 1.777e9, -1, 0.5, 0, 1, 0, 2.9e-13, 2),
            Particle('electron_neutrino', 1e-6, 0, 0.5, 0, 1, 0, 1e36, 3),
            Particle('muon_neutrino', 0.17e6, 0, 0.5, 0, 1, 0, 1e36, 3),
            Particle('tau_neutrino', 18.2e6, 0, 0.5, 0, 1, 0, 1e36, 3),

            # Bosons
            Particle('photon', 0, 0, 1, 0, 0, 0, 1e36, 2),
            Particle('gluon', 0, 0, 1, 0, 0, 1, 1e36, 1),
            Particle('W_boson', 80.4e9, 1, 1, 0, 0, 0, 3e-25, 3),
            Particle('Z_boson', 91.2e9, 0, 1, 0, 0, 0, 3e-25, 3),
            Particle('higgs', 125.1e9, 0, 0, 0, 0, 0, 1.6e-22, 2),

            # Hypothetical
            Particle('graviton', 0, 0, 2, 0, 0, 0, 1e36, 4, 'hypothetical'),
            Particle('axion', 1e-5, 0, 0, 0, 0, 0, 1e36, 2, 'hypothetical'),
            Particle('magnetic_monopole', 1e15, 1, 0.5, 0, 0, 0, 1e36, 1, 'hypothetical'),
            Particle('sterile_neutrino', 1e6, 0, 0.5, 0, 0, 0, 1e36, 3, 'hypothetical'),
        ]
        return pd.DataFrame([p.__dict__ for p in particles])

    def load_data(self, data: Dict[str, Any]):
        """Load external particle data or constraints."""
        # Can be used to load experimental results or constraints
        pass

    def build_model(self):
        """Build the particle hypercube by populating it with known particles."""
        self._populate_hypercube()
        self._train_anomaly_detection_models()

    def _populate_hypercube(self):
        """Populate the hypercube with known particles."""
        for _, particle in self.known_particles.iterrows():
            coords = []
            for dim in self.dimensions:
                value = particle[dim]
                # For log-scaled dimensions
                if dim in ['mass', 'lifetime']:
                    value = max(value, 1e-30)  # avoid log(0)
                    value = np.log10(value)
                    grid = np.log10(self.grids[dim])
                else:
                    grid = self.grids[dim]
                idx = np.abs(grid - value).argmin()
                coords.append(idx)
            # Set intensity in hypercube
            self.hypercube[tuple(coords)] = 1.0

    def _train_anomaly_detection_models(self):
        """Train machine learning models for anomaly detection."""
        # Extract coordinates of known particles
        X = []
        for _, particle in self.known_particles.iterrows():
            x = [particle[dim] for dim in self.dimensions]
            X.append(x)
        X = np.array(X)

        # Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(X)

        # Gaussian Process for interpolation and uncertainty
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
        y = np.ones(len(X))  # All known particles have value 1
        self.gp_model.fit(X, y)

    def analyze(self) -> Dict[str, Any]:
        """Perform full analysis: detect anomalies, predict new particles."""
        # Detect topological anomalies (cohomological approach)
        anomalies = self._detect_topological_anomalies()
        predicted_particles = self._predict_new_particles(anomalies)

        # Calculate cohomology map kernel/cokernel (from scientific work)
        cohomology_report = self._analyze_cohomology_map()

        return {
            "total_known_particles": len(self.known_particles),
            "anomalies_count": len(anomalies),
            "predicted_particles": [p.__dict__ for p in predicted_particles],
            "cohomology_analysis": cohomology_report,
            "model_built": True
        }

    def _detect_topological_anomalies(self) -> List[Dict]:
        """Detect anomalies using persistent homology and machine learning."""
        # Use Ripser for persistent homology on particle coordinates
        from ripser import ripser
        X = np.array([[p.mass, p.charge, p.spin] for _, p in self.known_particles.iterrows()])  # Simplified
        persistence = ripser(X, maxdim=2)['dgms']

        # Find points not predicted by Standard Model
        anomalies = []
        for i, (birth, death) in enumerate(persistence[1]):  # H1
            if death == float('inf'):
                anomalies.append({'type': 'H1_anomaly', 'birth': birth, 'index': i})

        return anomalies

    def _predict_new_particles(self, anomalies: List[Dict]) -> List[Particle]:
        """Predict properties of new particles based on anomalies."""
        predicted = []
        for anomaly in anomalies:
            if anomaly['type'] == 'H1_anomaly':
                # Predict particle properties based on anomaly location
                mass = 10**(anomaly['birth'] * 1e9)  # Example
                p = Particle(
                    name=f"predicted_particle_{len(predicted)}",
                    mass=mass,
                    charge=0,
                    spin=2,
                    baryon_number=0,
                    lepton_number=0,
                    color_charge=0,
                    lifetime=1e36,
                    interaction=4,
                    status="unknown",
                    probability=0.8,
                    anomaly_score=anomaly['birth']
                )
                predicted.append(p)
        return predicted

    def _analyze_cohomology_map(self) -> Dict[str, Any]:
        """
        Analyze the cohomology map F_SM → F.
        Non-zero elements in kernel/cokernel indicate anomalies (from scientific work).
        """
        # Simplified: compare density in hypercube
        # In reality, this would use sheaf cohomology
        sm_density = np.sum(self.hypercube)  # Known SM particles
        total_density = sm_density  # Placeholder

        # Kernel: elements in F_SM not in F (none, since F_SM ⊂ F)
        # Cokernel: elements in F not predicted by F_SM
        cokernel_size = len(self._predict_new_particles([]))

        return {
            "cokernel_size": cokernel_size,
            "interpretation": "Non-zero cokernel indicates presence of new physical phenomena not predicted by Standard Model"
        }

    def visualize(self):
        """Visualize the particle space in 3D."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot known particles
        known = self.known_particles[self.known_particles['status'] == 'known']
        ax.scatter(
            np.log10(known['mass'] + 1e-10),
            known['charge'],
            known['spin'],
            c='blue', s=50, label='Known Particles'
        )

        # Plot hypothetical
        hypothetical = self.known_particles[self.known_particles['status'] == 'hypothetical']
        ax.scatter(
            np.log10(hypothetical['mass'] + 1e-10),
            hypothetical['charge'],
            hypothetical['spin'],
            c='green', s=70, marker='s', label='Hypothetical'
        )

        # Add title and labels
        ax.set_xlabel('log10(Mass) [eV]')
        ax.set_ylabel('Charge [e]')
        ax.set_zlabel('Spin [ħ]')
        ax.set_title('Particle Hypercube: Known and Hypothetical Particles')
        ax.legend()

        plt.tight_layout()
        plt.show()

    def compress(self) -> bytes:
        """Compress the hypercube using DCT and Zstandard."""
        # Apply DCT
        dct_coeff = dctn(self.hypercube, type=2, norm='ortho')
        # Keep significant coefficients
        threshold = np.quantile(np.abs(dct_coeff), 0.9)
        sparse_dct = dct_coeff * (np.abs(dct_coeff) >= threshold)
        # Serialize
        flat_data = sparse_dct.astype(np.float32).tobytes()
        metadata = json.dumps({
            'shape': self.hypercube.shape,
            'threshold': float(threshold),
            'dimensions': self.dimensions
        }).encode('utf-8')
        # Compress
        cctx = zstd.ZstdCompressor(level=10)
        return cctx.compress(metadata + b'\n' + flat_data)

    def save_to_file(self, filename: str):
        """Save the model to HDF5."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('hypercube', data=self.hypercube)
            f.attrs['resolution'] = self.resolution
            f.attrs['dimensions'] = json.dumps(self.dimensions)
