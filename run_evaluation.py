#!/usr/bin/env python3
"""
Run complete emotional intelligence evaluation

Usage:
    python run_evaluation.py [--api-key YOUR_KEY] [--models MODEL1,MODEL2] [--quick]
"""

import argparse
import os
import sys
from pathlib import Path
from ei_framework import EmotionalIntelligenceFramework
from ei_analyzer import EIAnalyzer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run emotional intelligence evaluation for LLMs'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('OPENROUTER_API_KEY'),
        help='OpenRouter API key (or set OPENROUTER_API_KEY env variable)'
    )
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of models to test (default: all)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with fewer scenarios'
    )
    parser.add_argument(
        '--analyze-only',
        type=str,
        help='Only analyze existing results file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not args.api_key and not args.analyze_only:
        logger.error("No API key provided! Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)
    
    # Analyze only mode
    if args.analyze_only:
        logger.info(f"Analyzing existing results: {args.analyze_only}")
        analyzer = EIAnalyzer(args.analyze_only)
        
        # Generate visualizations
        viz_path = analyzer.create_visualizations(f"{args.output_dir}/visualizations")
        logger.info(f"Visualizations saved to: {viz_path}")
        
        # Generate report
        report = analyzer.generate_detailed_report()
        report_path = Path(args.output_dir) / "analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")
        
        # Print insights
        print("\n=== Key Insights ===")
        for insight in analyzer.generate_insights():
            print(f"• {insight}")
        
        return
    
    # Initialize framework
    logger.info("Initializing Emotional Intelligence Framework")
    framework = EmotionalIntelligenceFramework(args.api_key)
    
    # Filter models if specified
    if args.models:
        model_list = [m.strip() for m in args.models.split(',')]
        framework.models = {k: v for k, v in framework.models.items() if k in model_list}
        logger.info(f"Testing models: {list(framework.models.keys())}")
    
    # Quick mode - use fewer scenarios
    if args.quick:
        framework.scenarios = framework.scenarios[:3]
        logger.info("Quick mode: Using first 3 scenarios only")
    
    # Show test configuration
    print("\n=== Test Configuration ===")
    print(f"Models: {len(framework.models)}")
    print(f"Scenarios: {len(framework.scenarios)}")
    print(f"Total tests: {len(framework.models) * len(framework.scenarios)}")
    print(f"Output directory: {args.output_dir}")
    
    # Run evaluation
    print("\n=== Running Evaluation ===")
    results = framework.run_comprehensive_evaluation()
    
    # Save results
    results_file = framework.save_results(args.output_dir)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Generate initial report
    report = framework.generate_report()
    report_path = Path(args.output_dir) / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to: {report_path}")
    
    # Run analysis
    print("\n=== Running Analysis ===")
    analyzer = EIAnalyzer(str(results_file))
    
    # Generate visualizations
    viz_path = analyzer.create_visualizations(f"{args.output_dir}/visualizations")
    print(f"✓ Visualizations saved to: {viz_path}")
    
    # Generate detailed report
    detailed_report = analyzer.generate_detailed_report()
    detailed_report_path = Path(args.output_dir) / "detailed_analysis.md"
    with open(detailed_report_path, 'w') as f:
        f.write(detailed_report)
    print(f"✓ Detailed analysis saved to: {detailed_report_path}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Duration: {results['metadata']['duration']:.1f} seconds")
    print(f"Tests completed: {results['metadata']['total_tests']}")
    
    # Print insights
    print("\n=== Key Insights ===")
    for insight in analyzer.generate_insights():
        print(f"• {insight}")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()