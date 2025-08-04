"""
Module for visualizing audit results.
Provides functions to create informative plots of Rₓ table properties.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import os
from matplotlib.colors import LinearSegmentedColormap

class Visualizer:
    def __init__(self, output_dir: str = "visualizations", dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: directory to save visualizations
            dpi: resolution for saved images
        """
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def visualize_region(self, region: List[List[int]], 
                       title: str = "Rₓ Table Subregion",
                       save_as: Optional[str] = None) -> None:
        """
        Visualizes a subregion of the Rₓ table.
        
        Args:
            region: Rₓ table subregion
            title: plot title
            save_as: filename to save the visualization (if None, just displays)
        """
        plt.figure(figsize=(12, 10))
        
        # Create a custom colormap for better visualization
        colors = ["#00008B", "#0000FF", "#87CEFA", "#FFFFFF", "#FFA500", "#FF4500", "#8B0000"]
        cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors)
        
        ax = sns.heatmap(region, cmap=cmap, cbar_kws={'label': 'Rₓ value'})
        plt.title(title, fontsize=16)
        plt.xlabel('u_z', fontsize=14)
        plt.ylabel('u_r', fontsize=14)
        plt.tight_layout()
        
        if save_as:
            plt.savefig(os.path.join(self.output_dir, save_as), dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def visualize_betti_results(self, 
                             betti_results: List[Tuple[int, int, int]],
                             title: str = "Betti Numbers Analysis",
                             save_as: Optional[str] = None) -> None:
        """
        Visualizes Betti numbers analysis results.
        
        Args:
            betti_results: list of Betti number results
            title: plot title
            save_as: filename to save the visualization
        """
        beta_0 = [b[0] for b in betti_results]
        beta_1 = [b[1] for b in betti_results]
        beta_2 = [b[2] for b in betti_results]
        regions = list(range(1, len(betti_results) + 1))
        
        plt.figure(figsize=(14, 10))
        
        # Beta_0 plot
        plt.subplot(3, 1, 1)
        plt.plot(regions, beta_0, 'o-', linewidth=2, markersize=8)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        plt.title('β₀ (Connected Components)')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(min(beta_0) - 0.5, max(beta_0) + 0.5)
        
        # Beta_1 plot
        plt.subplot(3, 1, 2)
        plt.plot(regions, beta_1, 's-', linewidth=2, markersize=8, color='green')
        plt.axhline(y=2, color='r', linestyle='--', alpha=0.7)
        plt.title('β₁ (Cycles)')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(min(beta_1) - 0.5, max(beta_1) + 0.5)
        
        # Beta_2 plot
        plt.subplot(3, 1, 3)
        plt.plot(regions, beta_2, 'd-', linewidth=2, markersize=8, color='purple')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        plt.title('β₂ (Voids)')
        plt.xlabel('Region Index')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(min(beta_2) - 0.5, max(beta_2) + 0.5)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_as:
            plt.savefig(os.path.join(self.output_dir, save_as), dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def visualize_spiral_analysis(self, 
                                gamma_values: List[float], 
                                symmetry_scores: List[float],
                                spiral_strengths: List[float],
                                title: str = "Spiral Structure Analysis",
                                save_as: Optional[str] = None) -> None:
        """
        Visualizes spiral wave and symmetry analysis.
        
        Args:
            gamma_values: list of damping coefficients
            symmetry_scores: list of symmetry scores
            spiral_strengths: list of spiral structure strengths
            title: plot title
            save_as: filename to save the visualization
        """
        regions = list(range(1, len(gamma_values) + 1))
        
        plt.figure(figsize=(14, 12))
        
        # Gamma (damping coefficient) plot
        plt.subplot(3, 1, 1)
        plt.bar(regions, gamma_values, color='blue', alpha=0.7)
        plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
        plt.title('Damping Coefficient (γ) of Spiral Waves')
        plt.ylabel('γ Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Symmetry scores plot
        plt.subplot(3, 1, 2)
        plt.bar(regions, symmetry_scores, color='green', alpha=0.7)
        plt.axhline(y=0.85, color='r', linestyle='--', label='Threshold (0.85)')
        plt.title('Symmetry Score Around Special Points')
        plt.ylabel('Symmetry Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Spiral structure strength plot
        plt.subplot(3, 1, 3)
        plt.bar(regions, spiral_strengths, color='purple', alpha=0.7)
        plt.axhline(y=0.7, color='r', linestyle='--', label='Threshold (0.7)')
        plt.title('Spiral Structure Strength')
        plt.xlabel('Region Index')
        plt.ylabel('Strength')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_as:
            plt.savefig(os.path.join(self.output_dir, save_as), dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def visualize_anomaly_heatmap(self,
                                anomaly_scores: List[float],
                                region_positions: List[Tuple[int, int]],
                                grid_size: Tuple[int, int] = (10, 10),
                                title: str = "Anomaly Score Heatmap",
                                save_as: Optional[str] = None) -> None:
        """
        Visualizes anomaly scores as a heatmap over the Rₓ table.
        
        Args:
            anomaly_scores: list of anomaly scores for regions
            region_positions: list of (u_r_start, u_z_start) for each region
            grid_size: size of the grid to visualize
            title: plot title
            save_as: filename to save the visualization
        """
        # Create a grid to store anomaly scores
        grid = np.zeros(grid_size)
        count_grid = np.zeros(grid_size)
        
        # Fill the grid with anomaly scores
        for score, (u_r, u_z) in zip(anomaly_scores, region_positions):
            # Map positions to grid indices
            i = u_r % grid_size[0]
            j = u_z % grid_size[1]
            
            grid[i, j] += score
            count_grid[i, j] += 1
        
        # Calculate average scores
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_grid = np.divide(grid, count_grid)
            avg_grid[count_grid == 0] = np.nan
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        ax = sns.heatmap(avg_grid, cmap="YlOrRd", annot=True, fmt=".2f",
                        cbar_kws={'label': 'Anomaly Score (0-1)'})
        plt.title(title, fontsize=16)
        plt.xlabel('u_z Position (mod grid)', fontsize=14)
        plt.ylabel('u_r Position (mod grid)', fontsize=14)
        plt.tight_layout()
        
        if save_as:
            plt.savefig(os.path.join(self.output_dir, save_as), dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def generate_audit_report_visuals(self, audit_result: Dict) -> None:
        """
        Generates all visualizations for an audit result.
        
        Args:
            audit_result: result from audit_public_key
        """
        # 1. Visualize one representative region
        if 'analysis_details' in audit_result and 'per_region_analysis' in audit_result['analysis_details']:
            region = audit_result['analysis_details']['per_region_analysis'][0]
            self.visualize_region(
                region=region['region_data'],  # Assuming region data is stored
                title=f"Rₓ Table Subregion - {audit_result['public_key'][:8]}",
                save_as=f"region_{audit_result['public_key'][:8]}.png"
            )
            
            # Extract betti numbers, gamma, symmetry
            betti_results = [a['betti_numbers'] for a in audit_result['analysis_details']['per_region_analysis']]
            gamma_values = [a['damping_coefficient'] for a in audit_result['analysis_details']['per_region_analysis']]
            symmetry_scores = [a['symmetry_score'] for a in audit_result['analysis_details']['per_region_analysis']]
            spiral_strengths = [a['spiral_info']['correlation_strength'] for a in audit_result['analysis_details']['per_region_analysis']]
            
            # 2. Betti numbers visualization
            self.visualize_betti_results(
                betti_results,
                title=f"Betti Numbers Analysis - {audit_result['public_key'][:8]}",
                save_as=f"betti_{audit_result['public_key'][:8]}.png"
            )
            
            # 3. Spiral structure visualization
            self.visualize_spiral_analysis(
                gamma_values,
                symmetry_scores,
                spiral_strengths,
                title=f"Spiral Structure Analysis - {audit_result['public_key'][:8]}",
                save_as=f"spiral_{audit_result['public_key'][:8]}.png"
            )
            
            # 4. Anomaly heatmap (if region positions are available)
            if 'region_positions' in audit_result:
                self.visualize_anomaly_heatmap(
                    [a['anomaly_score'] for a in audit_result['analysis_details']['per_region_analysis']],
                    audit_result['region_positions'],
                    title=f"Anomaly Heatmap - {audit_result['public_key'][:8]}",
                    save_as=f"anomaly_heatmap_{audit_result['public_key'][:8]}.png"
                )
        
        # 5. Overall safety score visualization
        self._visualize_safety_score(audit_result)
    
    def _visualize_safety_score(self, audit_result: Dict) -> None:
        """
        Creates a visualization of the safety score components.
        
        Args:
            audit_result: result from audit_public_key
        """
        metrics = audit_result['metrics']
        
        # Extract values
        betti_anomaly = metrics['betti_analysis']['anomaly']
        gamma_ratio = metrics['damping_coefficient']['average'] / metrics['damping_coefficient']['threshold']
        symmetry_ratio = metrics['symmetry']['average'] / metrics['symmetry']['threshold']
        
        # Prepare data for radar chart
        labels = ['Betti Numbers', 'Damping Coefficient', 'Symmetry']
        stats = [
            1 - min(1.0, betti_anomaly),  # Invert anomaly to safety
            min(1.0, gamma_ratio),
            min(1.0, symmetry_ratio)
        ]
        
        # Number of variables
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        stats.append(stats[0])  # Close the polygon
        angles.append(angles[0])
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one axe per variable + add labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=10)
        plt.ylim(0, 1.0)
        
        # Plot data
        ax.plot(angles, stats, linewidth=2, linestyle='solid', label='Safety Metrics')
        ax.fill(angles, stats, alpha=0.25)
        
        # Add labels
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        # Set title and legend
        plt.title(f"ECDSA Safety Profile - {audit_result['public_key'][:8]}", size=16, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        # Add safety score text
        plt.figtext(0.5, 0.02, 
                   f"Overall Safety Score: {audit_result['safety_score']:.2f} | Vulnerability Level: {audit_result['vulnerability_level'].upper()}",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Save the visualization
        plt.savefig(os.path.join(self.output_dir, f"safety_profile_{audit_result['public_key'][:8]}.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def visualize_special_points(self, 
                              region: List[List[int]], 
                              u_r: int,
                              special_point: int,
                              title: str = "Symmetry Around Special Point",
                              save_as: Optional[str] = None) -> None:
        """
        Visualizes symmetry around a special point in a specific row.
        
        Args:
            region: Rₓ table subregion
            u_r: row index within the region
            special_point: column index of the special point
            title: plot title
            save_as: filename to save the visualization
        """
        if u_r >= len(region):
            return
            
        row = region[u_r]
        x = list(range(len(row)))
        
        plt.figure(figsize=(12, 8))
        
        # Plot the row values
        plt.plot(x, row, 'b-o', linewidth=2, markersize=4, label='Rₓ values')
        
        # Highlight the special point
        plt.plot(special_point, row[special_point], 'ro', markersize=10, label='Special Point')
        
        # Draw symmetry lines
        for offset in range(1, min(special_point, len(row) - special_point - 1)):
            plt.plot([special_point - offset, special_point + offset], 
                    [row[special_point - offset], row[special_point + offset]], 
                    'g--', alpha=0.3)
        
        plt.title(title, fontsize=16)
        plt.xlabel('u_z', fontsize=14)
        plt.ylabel('Rₓ value', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_as:
            plt.savefig(os.path.join(self.output_dir, save_as), dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
