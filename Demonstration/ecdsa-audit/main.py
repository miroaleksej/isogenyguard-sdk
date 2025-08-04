"""
ECDSA Audit System - Main entry point
"""

import argparse
import logging
import sys
import os
from audit.audit_engine import AuditEngine
from utils.config import Config
from utils.visualization import Visualizer
from utils.report_generator import ReportGenerator

def setup_logging(config: Config):
    """Set up logging based on configuration"""
    log_level = config.get("log_level", "INFO")
    log_file = config.get("log_file")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        ]
    )
    
    # Add a handler for warnings
    logging.captureWarnings(True)
    
    return logging.getLogger("ecdsa_audit")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ECDSA Audit System")
    parser.add_argument("public_key", nargs="?", help="Public key to audit (hex format)")
    parser.add_argument("-f", "--file", help="File containing public keys (one per line)")
    parser.add_argument("-c", "--config", default="config/default_config.yaml", 
                        help="Configuration file path")
    parser.add_argument("-n", "--num_regions", type=int, 
                        help="Number of regions to analyze (overrides config)")
    parser.add_argument("-s", "--region_size", type=int, 
                        help="Size of each region (overrides config)")
    parser.add_argument("--format", choices=["html", "pdf", "markdown"], 
                        default="html", help="Output report format")
    parser.add_argument("--output", help="Output directory for reports")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Validate configuration
    if not config.validate():
        print("Configuration validation failed. Exiting.")
        sys.exit(1)
    
    # Set up logging
    logger = setup_logging(config)
    
    # Create audit engine
    audit_engine = AuditEngine(logger=logger)
    
    try:
        # Determine number of regions and region size
        num_regions = args.num_regions or config.get("default_num_regions")
        region_size = args.region_size or config.get("default_region_size")
        
        # Perform audit based on input
        if args.public_key:
            # Audit a single public key
            logger.info(f"Auditing public key: {args.public_key[:10]}...")
            result = audit_engine.audit_public_key(
                args.public_key, num_regions, region_size
            )
            
            # Generate report
            report_gen = ReportGenerator()
            if args.format == "html":
                report_path = report_gen.generate_html_report(result)
            elif args.format == "pdf":
                report_path = report_gen.generate_pdf_report(result)
            else:  # markdown
                report_path = report_gen.generate_markdown_report(result)
            
            logger.info(f"Audit completed. Report generated at: {report_path}")
            
            # Visualize results if requested
            if args.verbose:
                visualizer = Visualizer()
                visualizer.generate_audit_report_visuals(result)
                logger.info(f"Visualizations generated in {visualizer.output_dir}")
        
        elif args.file:
            # Audit multiple public keys from file
            logger.info(f"Auditing public keys from file: {args.file}")
            results = audit_engine.audit_from_file(
                args.file, num_regions, region_size
            )
            
            # Generate batch report
            report_gen = ReportGenerator()
            report_path = report_gen.generate_batch_report(results)
            logger.info(f"Batch audit completed. Report generated at: {report_path}")
            
            # Visualize results if requested
            if args.verbose:
                visualizer = Visualizer()
                for i, result in enumerate(results[:5]):  # Only visualize first 5 for brevity
                    visualizer.generate_audit_report_visuals(result)
                logger.info(f"Visualizations generated in {visualizer.output_dir}")
        
        else:
            # No input provided
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.exception("An error occurred during audit")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
