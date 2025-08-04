"""
Module for generating audit reports.
Converts audit results into professional, human-readable reports.
"""

from typing import Dict, List, Any, Optional
import os
import datetime
from jinja2 import Environment, FileSystemLoader
import json
import markdown
from pathlib import Path

class ReportGenerator:
    def __init__(self, template_dir: str = "templates", output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            template_dir: directory containing report templates
            output_dir: directory to save generated reports
        """
        self.template_dir = template_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )
    
    def generate_html_report(self, audit_result: Dict, 
                           template_name: str = "audit_report.html",
                           output_filename: Optional[str] = None) -> str:
        """
        Generates an HTML report from audit results.
        
        Args:
            audit_result: result from audit_public_key
            template_name: name of the HTML template
            output_filename: name for the output file (if None, generates based on timestamp)
            
        Returns:
            Path to the generated report
        """
        # Prepare data for the template
        report_data = self._prepare_report_data(audit_result)
        
        # Load template
        template = self.env.get_template(template_name)
        
        # Render HTML
        html_content = template.render(**report_data)
        
        # Determine output filename
        if output_filename is None:
            key_snippet = audit_result['public_key'][:8] if 'public_key' in audit_result else "unknown"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ecdsa_audit_{key_snippet}_{timestamp}.html"
        
        # Save HTML file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_markdown_report(self, audit_result: Dict, 
                              output_filename: Optional[str] = None) -> str:
        """
        Generates a Markdown report from audit results.
        
        Args:
            audit_result: result from audit_public_key
            output_filename: name for the output file
            
        Returns:
            Path to the generated report
        """
        # Determine output filename
        if output_filename is None:
            key_snippet = audit_result['public_key'][:8] if 'public_key' in audit_result else "unknown"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ecdsa_audit_{key_snippet}_{timestamp}.md"
        
        # Generate Markdown content
        md_content = self._generate_markdown_content(audit_result)
        
        # Save Markdown file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return output_path
    
    def generate_pdf_report(self, audit_result: Dict, 
                          output_filename: Optional[str] = None) -> str:
        """
        Generates a PDF report from audit results.
        Requires additional libraries like weasyprint or pdfkit.
        
        Args:
            audit_result: result from audit_public_key
            output_filename: name for the output file
            
        Returns:
            Path to the generated report
        """
        # Generate HTML first
        html_path = self.generate_html_report(audit_result, output_filename=output_filename)
        
        # Convert HTML to PDF (simplified - would need additional setup)
        pdf_path = html_path.replace('.html', '.pdf')
        
        # In a real implementation, you would use weasyprint or similar:
        # from weasyprint import HTML
        # HTML(html_path).write_pdf(pdf_path)
        
        # For now, just return a placeholder
        with open(pdf_path, 'w') as f:
            f.write(f"PDF report would be generated from {html_path}")
        
        return pdf_path
    
    def _prepare_report_data(self, audit_result: Dict) -> Dict:
        """Prepares data for HTML template rendering"""
        # Create a deep copy to avoid modifying the original
        report_data = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'audit_result': audit_result,
            'vulnerability_level_class': self._get_vulnerability_class(audit_result['vulnerability_level'])
        }
        
        # Add safety score description
        report_data['safety_description'] = self._get_safety_description(
            audit_result['anomaly_score']
        )
        
        # Add detailed vulnerability explanations
        report_data['vulnerability_explanations'] = self._get_vulnerability_explanations(
            audit_result.get('anomalies', {})
        )
        
        return report_data
    
    def _generate_markdown_content(self, audit_result: Dict) -> str:
        """Generates Markdown content for the report"""
        md = []
        
        # Header
        md.append(f"# ECDSA Implementation Audit Report")
        md.append(f"**Public Key:** {audit_result['public_key']}")
        md.append(f"**Audit Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**Vulnerability Level:** {audit_result['vulnerability_level'].upper()}")
        md.append(f"**Anomaly Score:** {audit_result['anomaly_score']:.4f}")
        md.append("")
        
        # Executive summary
        md.append("## Executive Summary")
        safety_desc = self._get_safety_description(audit_result['anomaly_score'])
        md.append(f"{safety_desc}")
        md.append("")
        
        # Detailed analysis
        md.append("## Detailed Analysis")
        
        # Betti numbers analysis
        betti = audit_result['metrics']['betti_numbers']['average']
        md.append("### Topological Analysis (Betti Numbers)")
        md.append(f"- **β₀ (Connected Components):** {betti[0]:.2f} (Expected: 1.0)")
        md.append(f"- **β₁ (Cycles):** {betti[1]:.2f} (Expected: 2.0)")
        md.append(f"- **β₂ (Voids):** {betti[2]:.2f} (Expected: 1.0)")
        md.append("")
        
        # Damping coefficient analysis
        gamma = audit_result['metrics']['damping_coefficient']
        md.append("### Spiral Wave Analysis")
        md.append(f"- **Damping Coefficient (γ):** {gamma['average']:.4f} {'✅' if gamma['is_safe'] else '⚠️'}")
        md.append(f"  - Threshold: {gamma['threshold']:.2f}")
        md.append(f"  - {'Above threshold - normal spiral wave decay' if gamma['is_safe'] else 'Below threshold - potential nonce generation issue'}")
        md.append("")
        
        # Symmetry analysis
        symmetry = audit_result['metrics']['symmetry']
        md.append("### Symmetry Analysis")
        md.append(f"- **Symmetry Score:** {symmetry['average']:.4f} {'✅' if symmetry['is_safe'] else '⚠️'}")
        md.append(f"  - Threshold: {symmetry['threshold']:.2f}")
        md.append(f"  - {'Above threshold - expected symmetry present' if symmetry['is_safe'] else 'Below threshold - potential implementation anomaly'}")
        md.append("")
        
        # Vulnerabilities section
        md.append("## Detected Vulnerabilities")
        
        anomalies = audit_result.get('anomalies', {})
        if any(anomalies.values()):
            md.append("| Vulnerability | Status |")
            md.append("|---------------|--------|")
            
            if anomalies.get('betti_anomaly', False):
                md.append("| Topological Anomaly | ⚠️ Critical |")
            if anomalies.get('low_damping', False):
                md.append("| Low Damping Coefficient | ⚠️ High Risk |")
            if anomalies.get('broken_symmetry', False):
                md.append("| Broken Symmetry | ⚠️ Medium Risk |")
            if anomalies.get('missing_spiral', False):
                md.append("| Missing Spiral Structure | ⚠️ Medium Risk |")
            if anomalies.get('reused_k_attack', False):
                md.append("| Reused k Attack Vulnerability | ❌ Critical |")
        else:
            md.append("No critical vulnerabilities detected. The ECDSA implementation shows expected topological properties.")
        
        md.append("")
        
        # Recommendations
        md.append("## Security Recommendations")
        
        # In a real implementation, these would come from the audit engine
        recommendations = [
            "The implementation appears to have a critical vulnerability: Reused k values detected.",
            "Reused k values can lead to private key recovery. Immediate action required.",
            "Ensure that random k values are generated using a cryptographically secure random number generator.",
            "Consider implementing deterministic nonce generation according to RFC 6979.",
            "Rotate all affected keys immediately."
        ]
        
        for i, rec in enumerate(recommendations, 1):
            md.append(f"{i}. {rec}")
        
        # Technical details
        md.append("")
        md.append("## Technical Details")
        md.append("```json")
        md.append(json.dumps(audit_result, indent=2))
        md.append("```")
        
        return "\n".join(md)
    
    def _get_vulnerability_class(self, level: str) -> str:
        """Returns CSS class for vulnerability level"""
        level = level.lower()
        if level == "critical":
            return "danger"
        elif level == "warning":
            return "warning"
        else:
            return "success"
    
    def _get_safety_description(self, anomaly_score: float) -> str:
        """Returns a descriptive text for the safety score"""
        if anomaly_score < 0.3:
            return ("The ECDSA implementation appears to be secure based on topological analysis. "
                   "No critical vulnerabilities were detected in the Rₓ table structure.")
        elif anomaly_score < 0.7:
            return ("The analysis detected potential anomalies in the Rₓ table structure. "
                   "While not immediately critical, these anomalies warrant further investigation "
                   "as they may indicate implementation issues or potential vulnerabilities.")
        else:
            return ("Critical vulnerabilities were detected in the ECDSA implementation. "
                   "The topological analysis reveals significant deviations from expected "
                   "properties, indicating a high risk of private key exposure or other security issues.")
    
    def _get_vulnerability_explanations(self, anomalies: Dict[str, bool]) -> Dict[str, str]:
        """Returns detailed explanations for detected vulnerabilities"""
        explanations = {}
        
        if anomalies.get('betti_anomaly', False):
            explanations['betti_anomaly'] = (
                "The Betti numbers (topological invariants) of the Rₓ table deviate significantly "
                "from expected values (β₀=1, β₁=2, β₂=1). This indicates an anomalous topological "
                "structure that may be caused by improper nonce generation or implementation flaws."
            )
        
        if anomalies.get('low_damping', False):
            explanations['low_damping'] = (
                "The damping coefficient of spiral waves (γ) is below the threshold of 0.1. "
                "In a secure implementation, spiral waves should exhibit a certain rate of decay. "
                "A low damping coefficient suggests potential issues with the randomness of nonce values, "
                "which could lead to vulnerabilities such as key recovery attacks."
            )
        
        if anomalies.get('broken_symmetry', False):
            explanations['broken_symmetry'] = (
                "The expected symmetry around special points in the Rₓ table is broken. "
                "In a secure ECDSA implementation, values should be symmetric around specific points "
                "determined by the private key. Broken symmetry indicates potential implementation "
                "anomalies that could compromise security."
            )
        
        if anomalies.get('missing_spiral', False):
            explanations['missing_spiral'] = (
                "The expected spiral structure in the Rₓ table is missing or significantly weakened. "
                "This structure is a fundamental property of secure ECDSA implementations. Its absence "
                "may indicate issues with the underlying random number generation or implementation flaws."
            )
        
        if anomalies.get('reused_k_attack', False):
            explanations['reused_k_attack'] = (
                "High probability of reused k values detected. When the same k value is used for multiple "
                "signatures with the same private key, an attacker can recover the private key. "
                "This is a critical vulnerability that requires immediate remediation."
            )
        
        return explanations
    
    def generate_batch_report(self, audit_results: List[Dict], 
                            output_filename: Optional[str] = None) -> str:
        """
        Generates a consolidated report for multiple audit results.
        
        Args:
            audit_results: list of results from audit_public_key
            output_filename: name for the output file
            
        Returns:
            Path to the generated report
        """
        # Create summary data
        summary = {
            'total_keys': len(audit_results),
            'safe_count': sum(1 for r in audit_results if r['vulnerability_level'] == 'safe'),
            'warning_count': sum(1 for r in audit_results if r['vulnerability_level'] == 'warning'),
            'critical_count': sum(1 for r in audit_results if r['vulnerability_level'] == 'critical'),
            'results': audit_results
        }
        
        # Prepare data for template
        report_data = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': summary,
            'vulnerability_levels': ['safe', 'warning', 'critical']
        }
        
        # Load template
        template = self.env.get_template("batch_audit_report.html")
        
        # Render HTML
        html_content = template.render(**report_data)
        
        # Determine output filename
        if output_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ecdsa_batch_audit_{timestamp}.html"
        
        # Save HTML file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
