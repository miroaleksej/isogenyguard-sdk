"""
example_usage.py - Demonstration of the Deep Research Laboratory

This script shows how to use the unified framework for:
- Cryptography (ECDSA analysis)
- Quantum Physics (Atom simulation)
- Particle Physics (Anomaly detection)
- Biology (Rejuvenation modeling)
- Earth Observation (Anomaly detection)
- Quantum Topological Emulation
"""

import os
import json
import numpy as np
from drl.crypto import CryptoAnalyzer
from drl.quantum import QuantumAtomDecomposer
from drl.particle import ParticleHypercube
from drl.bio import RejuvenationHypercube
from drl.earth import EarthAnomalyHypercube
from drl.qte import QuantumTopologicalEmulator
import matplotlib.pyplot as plt


def demo_cryptography():
    """Demonstrate ECDSA topological analysis."""
    print("\n" + "="*60)
    print("DEMONSTRATION: CRYPTOGRAPHY ANALYSIS (ECDSA)")
    print("="*60)
    
    # Initialize analyzer
    crypto = CryptoAnalyzer()
    
    # Create synthetic ECDSA signatures for d=27, n=79
    n = 79
    d_true = 27
    signatures = []
    
    # Generate signatures with special points
    for ur in range(5, 15):
        for uz in range(20, 30):
            k = (uz + ur * d_true) % n
            r = (k * k + 1) % n  # Mock x(kG)
            s = np.random.randint(1, n)
            z = (ur + uz) % n
            signatures.append({'r': r, 's': s, 'z': z})
    
    # Load and analyze
    crypto.load_data({'signatures': signatures, 'n': n})
    crypto.build_model()
    results = crypto.analyze()
    
    print(f"Recovered private key: d = {results['d_recovered']}")
    print(f"Expected: d = {d_true}")
    print(f"Match: {results['d_recovered'] == d_true}")
    print(f"Betti numbers: β₀={results['betti_0']}, β₁={results['betti_1']}, β₂={results['betti_2']}")
    print(f"Security status: {'SECURE' if results['is_secure'] else 'VULNERABLE'}")


def demo_quantum():
    """Demonstrate quantum atom simulation."""
    print("\n" + "="*60)
    print("DEMONSTRATION: QUANTUM ATOM DECOMPOSER")
    print("="*60)
    
    # Initialize QAD
    qad = QuantumAtomDecomposer()
    
    # Configure for Hydrogen
    qad.load_data({'element': 'Hydrogen', 'atomic_number': 1, 'resolution': 32})
    qad.build_model()
    results = qad.analyze()
    
    print(f"Element: {results['element']}")
    print(f"Atomic number: {results['atomic_number']}")
    print(f"Total quantum states: {results['total_states']}")
    print(f"Energy levels (eV): {results['energy_levels_eV']}")
    print(f"Anomalies detected: {results['anomalies_count']}")
    
    # Save to file
    qad.save_to_file("hydrogen_state.h5")
    print("Quantum state saved to 'hydrogen_state.h5'")


def demo_particle():
    """Demonstrate particle hypercube analysis."""
    print("\n" + "="*60)
    print("DEMONSTRATION: PARTICLE HYPERCUBE")
    print("="*60)
    
    # Initialize particle analyzer
    particle = ParticleHypercube()
    particle.build_model()
    results = particle.analyze()
    
    print(f"Total known particles: {results['total_known_particles']}")
    print(f"Anomalies detected: {results['anomalies_count']}")
    print(f"Predicted new particles: {len(results['predicted_particles'])}")
    
    for i, p in enumerate(results['predicted_particles'][:3]):
        print(f"  Particle {i+1}: mass={p['mass']:.2e} eV, spin={p['spin']}, anomaly_score={p['anomaly_score']:.3f}")
    
    print(f"Cokernel size: {results['cohomology_analysis']['cokernel_size']}")
    print("Interpretation:", results['cohomology_analysis']['interpretation'])


def demo_bio():
    """Demonstrate rejuvenation hypercube."""
    print("\n" + "="*60)
    print("DEMONSTRATION: REJUVENATION HYPERCUBE")
    print("="*60)
    
    # Initialize bio analyzer
    bio = RejuvenationHypercube()
    bio.build_model()
    results = bio.analyze()
    
    print(f"Current biological age: {results['current_biological_age']:.1f} years")
    print(f"Rejuvenation score: {results['rejuvenation_score']:.1f}/100")
    print(f"Anomalies detected: {results['anomalies_count']}")
    
    print("Optimal intervention:")
    for param, value in results['optimal_intervention'].items():
        if abs(value) > 0.1:
            print(f"  {param}: {value:+.2f}")
    
    print("Predicted rejuvenated state:")
    for param, value in results['predicted_rejuvenated_state'].items():
        if param in ['telomerase_activity', 'epigenetic_age']:
            print(f"  {param}: {value:.2f}")


def demo_earth():
    """Demonstrate Earth anomaly detection."""
    print("\n" + "="*60)
    print("DEMONSTRATION: EARTH ANOMALY HYPERCUBE")
    print("="*60)
    
    # Initialize earth analyzer
    earth = EarthAnomalyHypercube(resolution=0.1, temporal_depth=10)
    
    # Create synthetic satellite data
    data_shape = (100, 100, 10)  # lat, lon, time
    sat_data = np.random.rand(*data_shape)
    
    # Add an anomaly
    sat_data[50:55, 50:55, 5] *= 2.0
    
    # Ingest data
    earth.ingest_satellite_data(
        source='multispectral',
        data=sat_data,
        time='2023-10-01',
        bounds=(0.0, 1.0, 0.0, 1.0)
    )
    
    earth.build_model()
    results = earth.analyze()
    
    print(f"Anomalies detected: {results['anomalies_count']}")
    print(f"Topological analysis: {results['topological_analysis']}")
    
    # Visualize
    if results['significant_anomalies']:
        earth.visualize_anomalies(results['significant_anomalies'])


def demo_qte():
    """Demonstrate Quantum Topological Emulator."""
    print("\n" + "="*60)
    print("DEMONSTRATION: QUANTUM TOPOLOGICAL EMULATOR")
    print("="*60)
    
    # Initialize QTE
    qte = QuantumTopologicalEmulator()
    qte.build_model()
    results = qte.analyze()
    
    print(f"Anomalies detected: {results['anomalies_count']}")
    print(f"Betti numbers: {results['betti_numbers']}")
    print(f"Topological entropy: {results['topological_entropy']:.3f}")
    print(f"System intact: {results['is_intact']}")
    
    # Compress data
    compressed = qte.compress(ratio=0.1)
    print(f"Data compressed. Size: {len(compressed)} bytes")
    
    # Visualize
    qte.visualize()


def main():
    """Run all demonstrations."""
    print("Deep Research Laboratory - Example Usage")
    print("Topology is not a hacking tool, but a microscope for vulnerability diagnostics")
    print("="*80)
    
    # Run demonstrations
    demo_cryptography()
    demo_quantum()
    demo_particle()
    demo_bio()
    demo_earth()
    demo_qte()
    
    print("\n" + "="*80)
    print("All demonstrations completed successfully.")
    print("The Deep Research Laboratory is ready for real-world research.")


if __name__ == "__main__":
    main()
