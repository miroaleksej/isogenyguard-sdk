"""
Main ECDSA audit module.
Coordinates all components for conducting an audit.
"""

from typing import Dict, List, Tuple, Optional
import time
import logging
from core.curve_operations import CurveOperations
from core.table_generator import TableGenerator
from core.topology_analyzer import TopologyAnalyzer
from core.anomaly_detector import AnomalyDetector
from utils.parallel import parallel_map

class AuditEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.curve_ops = CurveOperations()
        self.table_generator = TableGenerator(self.curve_ops)
        self.topology_analyzer = TopologyAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.logger = logger or logging.getLogger(__name__)
    
    def audit_public_key(self, public_key_hex: str, 
                        num_regions: int = 10, 
                        region_size: int = 50) -> Dict:
        """
        Conducts audit of a public key.
        
        Args:
            public_key_hex: public key in hexadecimal format
            num_regions: number of subregions to analyze
            region_size: size of each subregion
            
        Returns:
            Audit results as a dictionary
        """
        start_time = time.time()
        self.logger.info(f"Starting audit for public key: {public_key_hex[:10]}...")
        
        # Convert public key to coincurve object
        try:
            Q = coincurve.PublicKey(bytes.fromhex(public_key_hex))
        except Exception as e:
            self.logger.error(f"Invalid public key: {str(e)}")
            return {"error": f"Invalid public key: {str(e)}"}
        
        # Generate subregions for analysis
        self.logger.debug(f"Generating {num_regions} optimal regions of size {region_size}x{region_size}")
        regions = self.table_generator.generate_optimal_regions(
            Q, num_regions, region_size
        )
        
        # Analyze each subregion
        self.logger.debug("Analyzing subregions...")
        betti_results = []
        gamma_values = []
        symmetry_scores = []
        spiral_infos = []
        
        for i, region in enumerate(regions):
            self.logger.debug(f"Analyzing region {i+1}/{num_regions}")
            
            # Compute topological characteristics
            betti = self.topology_analyzer.compute_betti_numbers(region)
            gamma = self.topology_analyzer.analyze_spiral_waves(region)
            symmetry = self.topology_analyzer.check_symmetry(region)
            spiral_info = self.topology_analyzer.detect_spiral_structure(region)
            
            betti_results.append(betti)
            gamma_values.append(gamma)
            symmetry_scores.append(symmetry)
            spiral_infos.append(spiral_info)
        
        # Detect anomalies
        self.logger.debug("Detecting anomalies...")
        analysis = self.anomaly_detector.analyze_multiple_regions(
            betti_results, gamma_values, symmetry_scores, spiral_infos
        )
        
        # Format results
        result = {
            "public_key": public_key_hex[:16] + "..." if len(public_key_hex) > 16 else public_key_hex,
            "vulnerability_level": analysis["vulnerability_level"],
            "anomaly_score": analysis["overall_score"],
            "anomalies": analysis["anomalies"],
            "metrics": {
                "betti_numbers": {
                    "average": (
                        analysis["average_metrics"]["beta_0"],
                        analysis["average_metrics"]["beta_1"],
                        analysis["average_metrics"]["beta_2"]
                    ),
                    "expected": (1.0, 2.0, 1.0)
                },
                "damping_coefficient": {
                    "average": analysis["average_metrics"]["gamma"],
                    "threshold": self.anomaly_detector.gamma_threshold,
                    "is_safe": analysis["average_metrics"]["gamma"] > self.anomaly_detector.gamma_threshold
                },
                "symmetry": {
                    "average": analysis["average_metrics"]["symmetry"],
                    "threshold": self.anomaly_detector.symmetry_threshold,
                    "is_safe": analysis["average_metrics"]["symmetry"] > self.anomaly_detector.symmetry_threshold
                },
                "spiral_structure": {
                    "average_strength": analysis["average_metrics"]["spiral_strength"],
                    "threshold": self.anomaly_detector.spiral_threshold,
                    "is_present": analysis["average_metrics"]["spiral_strength"] > self.anomaly_detector.spiral_threshold
                }
            },
            "analysis_details": {
                "num_regions": num_regions,
                "region_size": region_size,
                "execution_time": round(time.time() - start_time, 2),
                "per_region_analysis": [
                    {
                        "betti_numbers": betti_results[i],
                        "damping_coefficient": gamma_values[i],
                        "symmetry_score": symmetry_scores[i],
                        "spiral_info": spiral_infos[i],
                        "anomaly_score": analysis["anomaly_scores"][i]
                    }
                    for i in range(num_regions)
                ]
            }
        }
        
        self.logger.info(f"Audit completed in {result['analysis_details']['execution_time']}s. "
                        f"Vulnerability level: {result['vulnerability_level']}")
        
        return result
    
    def audit_batch(self, public_keys: List[str], 
                   num_regions: int = 5, 
                   region_size: int = 50) -> List[Dict]:
        """
        Conducts audit of multiple public keys.
        
        Args:
            public_keys: list of public keys
            num_regions: number of subregions for analysis
            region_size: size of each subregion
            
        Returns:
            List of audit results for each key
        """
        self.logger.info(f"Starting batch audit of {len(public_keys)} public keys")
        start_time = time.time()
        
        # Use parallel processing for speedup
        results = parallel_map(
            lambda pk: self.audit_public_key(pk, num_regions, region_size),
            public_keys
        )
        
        elapsed = time.time() - start_time
        self.logger.info(f"Batch audit completed in {elapsed:.2f} seconds "
                        f"({elapsed/len(public_keys):.2f} seconds per key)")
        
        return results
    
    def audit_from_file(self, file_path: str, 
                       num_regions: int = 5, 
                       region_size: int = 50) -> List[Dict]:
        """
        Conducts audit of public keys from a file.
        
        Args:
            file_path: path to file containing public keys (one per line)
            num_regions: number of subregions for analysis
            region_size: size of each subregion
            
        Returns:
            List of audit results
        """
        self.logger.info(f"Reading public keys from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                public_keys = [line.strip() for line in f if line.strip()]
            
            self.logger.info(f"Found {len(public_keys)} public keys")
            return self.audit_batch(public_keys, num_regions, region_size)
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return [{"error": f"File error: {str(e)}"}]
