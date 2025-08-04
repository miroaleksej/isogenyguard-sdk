# ECDSA Topological Audit System - API Reference

## Core Package

### CurveOperations

```python
class CurveOperations(curve_name: str = "secp256k1")
```

Provides operations for elliptic curve cryptography, specifically for working with the Rₓ table structure.

**Parameters:**
- `curve_name` (str): Name of the elliptic curve to use (default: "secp256k1")

**Methods:**

#### compute_R
```python
compute_R(u_r: int, u_z: int, Q: coincurve.PublicKey) -> coincurve.Point
```
Computes the elliptic curve point R = u_r * Q + u_z * G.

**Parameters:**
- `u_r` (int): Vertical coordinate in the Rₓ table
- `u_z` (int): Horizontal coordinate in the Rₓ table
- `Q` (coincurve.PublicKey): Public key point on the curve

**Returns:**
- `coincurve.Point`: The computed point R on the elliptic curve

#### get_Rx
```python
get_Rx(u_r: int, u_z: int, Q: coincurve.PublicKey) -> int
```
Computes the x-coordinate of point R = u_r * Q + u_z * G.

**Parameters:**
- `u_r` (int): Vertical coordinate in the Rₓ table
- `u_z` (int): Horizontal coordinate in the Rₓ table
- `Q` (coincurve.PublicKey): Public key point on the curve

**Returns:**
- `int`: x-coordinate of point R modulo n (curve order)

#### compute_subregion
```python
compute_subregion(Q: coincurve.PublicKey, u_r_min: int, u_r_max: int, u_z_min: int, u_z_max: int) -> List[List[int]]
```
Computes a subregion of the Rₓ table.

**Parameters:**
- `Q` (coincurve.PublicKey): Public key point on the curve
- `u_r_min` (int): Minimum vertical coordinate
- `u_r_max` (int): Maximum vertical coordinate
- `u_z_min` (int): Minimum horizontal coordinate
- `u_z_max` (int): Maximum horizontal coordinate

**Returns:**
- `List[List[int]]`: 2D list containing x-coordinates of points in the subregion

#### get_special_point
```python
get_special_point(u_r: int, d: Optional[int] = None) -> int
```
Calculates the special point u_z* for a given u_r where symmetry occurs.

**Parameters:**
- `u_r` (int): Vertical coordinate
- `d` (Optional[int]): Private key (optional, not required for actual computations)

**Returns:**
- `int`: Special point u_z* where symmetry occurs (u_z* = -u_r * d mod n)

#### get_mirror_point
```python
get_mirror_point(u_r: int, u_z: int, d: Optional[int] = None) -> int
```
Calculates the mirror point for a given (u_r, u_z) pair.

**Parameters:**
- `u_r` (int): Vertical coordinate
- `u_z` (int): Horizontal coordinate
- `d` (Optional[int]): Private key (optional)

**Returns:**
- `int`: Mirror point u_z' where R_x is the same (u_z' = -u_z - 2 * u_r * d mod n)

---

### TableGenerator

```python
class TableGenerator(curve_ops: CurveOperations)
```

Generates subregions of the Rₓ table for auditing purposes.

**Parameters:**
- `curve_ops` (CurveOperations): Curve operations instance

**Methods:**

#### generate_random_regions
```python
generate_random_regions(Q: coincurve.PublicKey, num_regions: int = 10, region_size: int = 50) -> List[List[List[int]]]
```
Generates random subregions of the Rₓ table for auditing.

**Parameters:**
- `Q` (coincurve.PublicKey): Public key to analyze
- `num_regions` (int): Number of subregions to generate (default: 10)
- `region_size` (int): Size of each subregion (region_size x region_size) (default: 50)

**Returns:**
- `List[List[List[int]]]`: List of generated subregions

#### generate_optimal_regions
```python
generate_optimal_regions(Q: coincurve.PublicKey, num_regions: int = 5, region_size: int = 50) -> List[List[List[int]]]
```
Generates subregions in optimal zones for auditing based on the theorem about optimal d_opt ≈ n/2.

**Parameters:**
- `Q` (coincurve.PublicKey): Public key to analyze
- `num_regions` (int): Number of subregions to generate (default: 5)
- `region_size` (int): Size of each subregion (default: 50)

**Returns:**
- `List[List[List[int]]]`: List of subregions in optimal zones

#### generate_symmetry_regions
```python
generate_symmetry_regions(Q: coincurve.PublicKey, num_regions: int = 5, region_size: int = 50) -> List[Tuple[List[List[int]], int, int]]
```
Generates subregions around symmetry points for detailed analysis.

**Parameters:**
- `Q` (coincurve.PublicKey): Public key to analyze
- `num_regions` (int): Number of subregions to generate (default: 5)
- `region_size` (int): Size of each subregion (default: 50)

**Returns:**
- `List[Tuple[List[List[int]], int, int]]`: List of subregions with their special point coordinates

---

### TopologyAnalyzer

```python
class TopologyAnalyzer()
```

Analyzes topological properties of the Rₓ table subregions.

**Methods:**

#### compute_betti_numbers
```python
compute_betti_numbers(region: List[List[int]]) -> Tuple[int, int, int]
```
Computes Betti numbers for a subregion of the table.

**Parameters:**
- `region` (List[List[int]]): Rₓ table subregion

**Returns:**
- `Tuple[int, int, int]`: Tuple (β₀, β₁, β₂) - Betti numbers

#### analyze_spiral_waves
```python
analyze_spiral_waves(region: List[List[int]]) -> float
```
Analyzes spiral waves and computes the damping coefficient.

**Parameters:**
- `region` (List[List[int]]): Rₓ table subregion

**Returns:**
- `float`: Damping coefficient γ

#### check_symmetry
```python
check_symmetry(region: List[List[int]], center_row: int = None) -> float
```
Checks symmetry around special points.

**Parameters:**
- `region` (List[List[int]]): Rₓ table subregion
- `center_row` (int, optional): Specific row to check

**Returns:**
- `float`: Symmetry coefficient (0-1)

#### detect_spiral_structure
```python
detect_spiral_structure(region: List[List[int]]) -> Dict[str, float]
```
Detects spiral structure properties in the region.

**Parameters:**
- `region` (List[List[int]]): Rₓ table subregion

**Returns:**
- `Dict[str, float]`: Dictionary with spiral structure properties

---

### AnomalyDetector

```python
class AnomalyDetector()
```

Detects anomalies in the Rₓ table structure based on topological analysis.

**Methods:**

#### detect_topology_anomaly
```python
detect_topology_anomaly(betti: Tuple[int, int, int], gamma: float, symmetry: float, spiral_info: Dict[str, float]) -> Dict[str, bool]
```
Detects topological anomalies in the Rₓ table.

**Parameters:**
- `betti` (Tuple[int, int, int]): Betti numbers (β₀, β₁, β₂)
- `gamma` (float): Damping coefficient
- `symmetry` (float): Symmetry coefficient
- `spiral_info` (Dict[str, float]): Spiral structure information

**Returns:**
- `Dict[str, bool]`: Dictionary with detected anomalies

#### calculate_anomaly_score
```python
calculate_anomaly_score(betti: Tuple[int, int, int], gamma: float, symmetry: float, spiral_info: Dict[str, float]) -> float
```
Calculates anomaly score (0-1, where 1 is maximally anomalous).

**Parameters:**
- `betti` (Tuple[int, int, int]): Betti numbers (β₀, β₁, β₂)
- `gamma` (float): Damping coefficient
- `symmetry` (float): Symmetry coefficient
- `spiral_info` (Dict[str, float]): Spiral structure information

**Returns:**
- `float`: Anomaly score (0-1)

#### detect_vulnerability_level
```python
detect_vulnerability_level(anomaly_score: float) -> str
```
Determines vulnerability level based on anomaly score.

**Parameters:**
- `anomaly_score` (float): Anomaly score (0-1)

**Returns:**
- `str`: Vulnerability level ("safe", "warning", "critical")

#### analyze_multiple_regions
```python
analyze_multiple_regions(betti_results: List[Tuple[int, int, int]], gamma_values: List[float], symmetry_scores: List[float], spiral_infos: List[Dict[str, float]]) -> Dict
```
Analyzes multiple regions to detect overall anomalies.

**Parameters:**
- `betti_results` (List[Tuple[int, int, int]]): List of Betti number results
- `gamma_values` (List[float]): List of damping coefficients
- `symmetry_scores` (List[float]): List of symmetry scores
- `spiral_infos` (List[Dict[str, float]]): List of spiral structure information

**Returns:**
- `Dict`: Analysis results

---

## Audit Package

### AuditEngine

```python
class AuditEngine(logger: Optional[logging.Logger] = None)
```

Main engine for conducting ECDSA audits.

**Parameters:**
- `logger` (Optional[logging.Logger]): Logger instance (default: None)

**Methods:**

#### audit_public_key
```python
audit_public_key(public_key_hex: str, num_regions: int = 10, region_size: int = 50) -> Dict
```
Conducts audit of a public key.

**Parameters:**
- `public_key_hex` (str): Public key in hexadecimal format
- `num_regions` (int): Number of subregions to analyze (default: 10)
- `region_size` (int): Size of each subregion (default: 50)

**Returns:**
- `Dict`: Audit results

#### audit_batch
```python
audit_batch(public_keys: List[str], num_regions: int = 5, region_size: int = 50) -> List[Dict]
```
Conducts audit of multiple public keys.

**Parameters:**
- `public_keys` (List[str]): List of public keys
- `num_regions` (int): Number of subregions for analysis (default: 5)
- `region_size` (int): Size of each subregion (default: 50)

**Returns:**
- `List[Dict]`: List of audit results for each key

#### audit_from_file
```python
audit_from_file(file_path: str, num_regions: int = 5, region_size: int = 50) -> List[Dict]
```
Conducts audit of public keys from a file.

**Parameters:**
- `file_path` (str): Path to file containing public keys (one per line)
- `num_regions` (int): Number of subregions for analysis (default: 5)
- `region_size` (int): Size of each subregion (default: 50)

**Returns:**
- `List[Dict]`: List of audit results

---

### SafetyMetrics

```python
class SafetyMetrics()
```

Calculates safety metrics based on topological analysis.

**Methods:**

#### calculate_betti_anomaly
```python
calculate_betti_anomaly(betti: Tuple[int, int, int]) -> float
```
Calculates Betti numbers anomaly.

**Parameters:**
- `betti` (Tuple[int, int, int]): Tuple (β₀, β₁, β₂)

**Returns:**
- `float`: Anomaly value (higher means more deviation)

#### calculate_safety_score
```python
calculate_safety_score(betti_results: List[Tuple[int, int, int]], gamma_values: List[float], symmetry_scores: List[float]) -> float
```
Calculates overall safety score.

**Parameters:**
- `betti_results` (List[Tuple[int, int, int]]): List of Betti number results
- `gamma_values` (List[float]): List of damping coefficient values
- `symmetry_scores` (List[float]): List of symmetry scores

**Returns:**
- `float`: Overall safety score (0-1, where 1 is maximally safe)

#### detect_vulnerability
```python
detect_vulnerability(safety_score: float) -> Dict[str, bool]
```
Determines vulnerabilities based on safety score.

**Parameters:**
- `safety_score` (float): Safety score

**Returns:**
- `Dict[str, bool]`: Dictionary with detected vulnerabilities

#### generate_safety_report
```python
generate_safety_report(betti_results: List[Tuple[int, int, int]], gamma_values: List[float], symmetry_scores: List[float]) -> Dict
```
Generates a comprehensive safety report.

**Parameters:**
- `betti_results` (List[Tuple[int, int, int]]): List of Betti number results
- `gamma_values` (List[float]): List of damping coefficient values
- `symmetry_scores` (List[float]): List of symmetry scores

**Returns:**
- `Dict`: Safety report

---

### VulnerabilityScanner

```python
class VulnerabilityScanner()
```

Scans ECDSA implementations for specific vulnerabilities.

**Methods:**

#### scan_for_reused_k
```python
scan_for_reused_k(region: List[List[int]]) -> Dict[str, float]
```
Scans for reused k vulnerability pattern.

**Parameters:**
- `region` (List[List[int]]): Rₓ table subregion

**Returns:**
- `Dict[str, float]`: Detection metrics for reused k vulnerability

#### scan_for_weak_drbg
```python
scan_for_weak_drbg(regions: List[List[List[int]]]) -> Dict[str, float]
```
Scans for weak DRBG (Deterministic Random Bit Generator) patterns.

**Parameters:**
- `regions` (List[List[List[int]]]): List of Rₓ table subregions

**Returns:**
- `Dict[str, float]`: Detection metrics for weak DRBG vulnerability

#### scan_for_special_point_anomalies
```python
scan_for_special_point_anomalies(region: List[List[int]], u_r: int) -> Dict[str, float]
```
Scans for anomalies around special points.

**Parameters:**
- `region` (List[List[int]]): Rₓ table subregion
- `u_r` (int): Specific row to check

**Returns:**
- `Dict[str, float]`: Detection metrics for special point anomalies

#### comprehensive_vulnerability_scan
```python
comprehensive_vulnerability_scan(regions: List[List[List[int]]], u_r: int = None) -> Dict
```
Performs comprehensive vulnerability scan across multiple dimensions.

**Parameters:**
- `regions` (List[List[List[int]]]): List of Rₓ table subregions
- `u_r` (int, optional): Specific row to check for special point anomalies

**Returns:**
- `Dict`: Comprehensive vulnerability assessment

---

## Utils Package

### Config

```python
class Config(config_path: Optional[str] = None)
```

Handles configuration for the ECDSA audit system.

**Parameters:**
- `config_path` (Optional[str]): Path to YAML configuration file

**Methods:**

#### get
```python
get(key: str, default: Any = None) -> Any
```
Get configuration value.

**Parameters:**
- `key` (str): Configuration key (can use dot notation for nested keys)
- `default` (Any): Default value if key not found

**Returns:**
- `Any`: Configuration value

#### set
```python
set(key: str, value: Any)
```
Set configuration value.

**Parameters:**
- `key` (str): Configuration key (can use dot notation for nested keys)
- `value` (Any): Value to set

#### validate
```python
validate() -> bool
```
Validate configuration values.

**Returns:**
- `bool`: True if configuration is valid, False otherwise

#### from_dict
```python
@classmethod
from_dict(config_dict: Dict[str, Any]) -> 'Config'
```
Create configuration from dictionary.

**Parameters:**
- `config_dict` (Dict[str, Any]): Dictionary with configuration values

**Returns:**
- `Config`: Config object

---

### ParallelExecutor

```python
class ParallelExecutor(num_workers: int = None)
```

Helper class for parallel execution of tasks.

**Parameters:**
- `num_workers` (int): Number of worker processes (None = use all available cores)

**Methods:**

#### map
```python
map(func: Callable, items: Iterable, chunk_size: int = 1) -> List[Any]
```
Execute map operation in parallel.

**Parameters:**
- `func` (Callable): Function to apply to each item
- `items` (Iterable): Iterable of items to process
- `chunk_size` (int): Number of items to process in each chunk (default: 1)

**Returns:**
- `List[Any]`: List of results

#### starmap
```python
starmap(func: Callable, arg_tuples: Iterable[tuple], chunk_size: int = 1) -> List[Any]
```
Execute starmap operation in parallel.

**Parameters:**
- `func` (Callable): Function to apply to each argument tuple
- `arg_tuples` (Iterable[tuple]): Iterable of argument tuples
- `chunk_size` (int): Number of items to process in each chunk (default: 1)

**Returns:**
- `List[Any]`: List of results

#### chunked_map
```python
chunked_map(func: Callable, items: Iterable, chunk_size: int = 100) -> List[Any]
```
Process items in chunks for better memory management.

**Parameters:**
- `func` (Callable): Function to apply to each chunk
- `items` (Iterable): Iterable of items to process
- `chunk_size` (int): Number of items per chunk (default: 100)

**Returns:**
- `List[Any]`: List of results

---

### Visualizer

```python
class Visualizer(output_dir: str = "visualizations", dpi: int = 300)
```

Provides functions to visualize audit results.

**Parameters:**
- `output_dir` (str): Directory to save visualizations (default: "visualizations")
- `dpi` (int): Resolution for saved images (default: 300)

**Methods:**

#### visualize_region
```python
visualize_region(region: List[List[int]], title: str = "Rₓ Table Subregion", save_as: Optional[str] = None) -> None
```
Visualizes a subregion of the Rₓ table.

**Parameters:**
- `region` (List[List[int]]): Rₓ table subregion
- `title` (str): Plot title (default: "Rₓ Table Subregion")
- `save_as` (Optional[str]): Filename to save the visualization

#### visualize_betti_results
```python
visualize_betti_results(betti_results: List[Tuple[int, int, int]], title: str = "Betti Numbers Analysis", save_as: Optional[str] = None) -> None
```
Visualizes Betti numbers analysis results.

**Parameters:**
- `betti_results` (List[Tuple[int, int, int]]): List of Betti number results
- `title` (str): Plot title (default: "Betti Numbers Analysis")
- `save_as` (Optional[str]): Filename to save the visualization

#### visualize_spiral_analysis
```python
visualize_spiral_analysis(gamma_values: List[float], symmetry_scores: List[float], spiral_strengths: List[float], title: str = "Spiral Structure Analysis", save_as: Optional[str] = None) -> None
```
Visualizes spiral wave and symmetry analysis.

**Parameters:**
- `gamma_values` (List[float]): List of damping coefficients
- `symmetry_scores` (List[float]): List of symmetry scores
- `spiral_strengths` (List[float]): List of spiral structure strengths
- `title` (str): Plot title (default: "Spiral Structure Analysis")
- `save_as` (Optional[str]): Filename to save the visualization

#### visualize_anomaly_heatmap
```python
visualize_anomaly_heatmap(anomaly_scores: List[float], region_positions: List[Tuple[int, int]], grid_size: Tuple[int, int] = (10, 10), title: str = "Anomaly Score Heatmap", save_as: Optional[str] = None) -> None
```
Visualizes anomaly scores as a heatmap over the Rₓ table.

**Parameters:**
- `anomaly_scores` (List[float]): List of anomaly scores for regions
- `region_positions` (List[Tuple[int, int]]): List of (u_r_start, u_z_start) for each region
- `grid_size` (Tuple[int, int]): Size of the grid to visualize (default: (10, 10))
- `title` (str): Plot title (default: "Anomaly Score Heatmap")
- `save_as` (Optional[str]): Filename to save the visualization

#### generate_audit_report_visuals
```python
generate_audit_report_visuals(audit_result: Dict) -> None
```
Generates all visualizations for an audit result.

**Parameters:**
- `audit_result` (Dict): Result from audit_public_key

---

### ReportGenerator

```python
class ReportGenerator(template_dir: str = "templates", output_dir: str = "reports")
```

Generates professional reports from audit results.

**Parameters:**
- `template_dir` (str): Directory containing report templates (default: "templates")
- `output_dir` (str): Directory to save generated reports (default: "reports")

**Methods:**

#### generate_html_report
```python
generate_html_report(audit_result: Dict, template_name: str = "audit_report.html", output_filename: Optional[str] = None) -> str
```
Generates an HTML report from audit results.

**Parameters:**
- `audit_result` (Dict): Result from audit_public_key
- `template_name` (str): Name of the HTML template (default: "audit_report.html")
- `output_filename` (Optional[str]): Name for the output file

**Returns:**
- `str`: Path to the generated report

#### generate_markdown_report
```python
generate_markdown_report(audit_result: Dict, output_filename: Optional[str] = None) -> str
```
Generates a Markdown report from audit results.

**Parameters:**
- `audit_result` (Dict): Result from audit_public_key
- `output_filename` (Optional[str]): Name for the output file

**Returns:**
- `str`: Path to the generated report

#### generate_pdf_report
```python
generate_pdf_report(audit_result: Dict, output_filename: Optional[str] = None) -> str
```
Generates a PDF report from audit results.

**Parameters:**
- `audit_result` (Dict): Result from audit_public_key
- `output_filename` (Optional[str]): Name for the output file

**Returns:**
- `str`: Path to the generated report

#### generate_batch_report
```python
generate_batch_report(audit_results: List[Dict], output_filename: Optional[str] = None) -> str
```
Generates a consolidated report for multiple audit results.

**Parameters:**
- `audit_results` (List[Dict]): List of results from audit_public_key
- `output_filename` (Optional[str]): Name for the output file

**Returns:**
- `str`: Path to the generated report
