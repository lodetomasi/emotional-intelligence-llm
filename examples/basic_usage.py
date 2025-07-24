"""
Basic usage example for Emotional Intelligence Framework
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ei_framework import EmotionalIntelligenceFramework

def main():
    """Basic usage example"""
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Initialize framework
    print("Initializing Emotional Intelligence Framework...")
    framework = EmotionalIntelligenceFramework(api_key)
    
    # Run a quick test with one model
    print("\nRunning quick test with Mixtral-8x22B...")
    
    # Select only one model for quick test
    framework.models = {
        k: v for k, v in framework.models.items() 
        if k == "Mixtral-8x22B"
    }
    
    # Use only first 2 scenarios
    framework.scenarios = framework.scenarios[:2]
    
    # Run evaluation
    results = framework.run_comprehensive_evaluation()
    
    # Print results
    print(f"\nTest completed in {results['metadata']['duration']:.1f} seconds")
    print(f"Total tests: {results['metadata']['total_tests']}")
    
    # Generate and print report
    report = framework.generate_report()
    print("\n" + "="*50)
    print(report)
    
    # Save results
    output_file = framework.save_results()
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()