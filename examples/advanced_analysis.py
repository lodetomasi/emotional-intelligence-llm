"""
Advanced analysis example for Emotional Intelligence Framework
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ei_framework import EIAnalyzer
import matplotlib.pyplot as plt

def main():
    """Advanced analysis example"""
    
    # Find latest results file
    import glob
    result_files = glob.glob("results/ei_results_*.json")
    
    if not result_files:
        print("No results files found. Run basic_usage.py first.")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"Analyzing: {latest_file}")
    
    # Initialize analyzer
    analyzer = EIAnalyzer(latest_file)
    
    # Generate insights
    print("\n=== Key Insights ===")
    for insight in analyzer.generate_insights():
        print(f"• {insight}")
    
    # Emotion analysis
    print("\n=== Emotion Analysis ===")
    emotion_data = analyzer.emotion_analysis()
    
    for model, data in emotion_data['emotion_diversity'].items():
        print(f"\n{model}:")
        print(f"  Unique emotions: {data['unique_emotions']}")
        print(f"  Diversity score: {data['diversity_score']:.3f}")
    
    # Response quality
    print("\n=== Response Quality ===")
    quality = analyzer.response_quality_analysis()
    
    for model, metrics in quality.items():
        print(f"\n{model}:")
        print(f"  Avg response length: {metrics['avg_response_length']:.0f} chars")
        print(f"  Token efficiency: {metrics['tokens_per_char']:.3f}")
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    viz_path = analyzer.create_visualizations("results/visualizations")
    print(f"Visualizations saved to: {viz_path}")
    
    # Generate detailed report
    report = analyzer.generate_detailed_report()
    report_path = "results/detailed_analysis.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()