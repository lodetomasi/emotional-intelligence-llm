"""
Script to save and organize emotional intelligence test results
"""
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path

class ResultsSaver:
    def __init__(self, base_dir="results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.base_dir / "raw_responses"
        self.processed_dir = self.base_dir / "processed_data"
        self.visualizations_dir = self.base_dir / "visualizations"
        
        for dir in [self.raw_dir, self.processed_dir, self.visualizations_dir]:
            dir.mkdir(exist_ok=True)
    
    def save_raw_results(self, results_dict, model_name=None):
        """Save raw API responses"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_name:
            filename = f"{model_name}_{timestamp}_raw.json"
        else:
            filename = f"all_models_{timestamp}_raw.json"
        
        filepath = self.raw_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Raw results saved to: {filepath}")
        return filepath
    
    def save_processed_scores(self, scores_df):
        """Save processed scores as CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_path = self.processed_dir / f"ei_scores_{timestamp}.csv"
        scores_df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.processed_dir / f"ei_scores_{timestamp}.json"
        scores_df.to_json(json_path, orient='records', indent=2)
        
        print(f"Processed scores saved to:\n  CSV: {csv_path}\n  JSON: {json_path}")
        return csv_path, json_path
    
    def save_analysis_summary(self, summary_data):
        """Save analysis summary with key findings"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            "timestamp": timestamp,
            "analysis_date": datetime.now().isoformat(),
            "summary_statistics": summary_data,
            "metadata": {
                "models_tested": summary_data.get("models_tested", []),
                "total_scenarios": summary_data.get("total_scenarios", 0),
                "dimensions_evaluated": summary_data.get("dimensions", [])
            }
        }
        
        filepath = self.processed_dir / f"analysis_summary_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis summary saved to: {filepath}")
        return filepath
    
    def create_results_report(self, all_results, scores):
        """Create a comprehensive results report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_path = self.base_dir / f"EI_Research_Report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Emotional Intelligence Test Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Models Tested**: {len(all_results)}\n")
            f.write(f"- **Test Scenarios**: {scores.get('total_scenarios', 'N/A')}\n")
            f.write(f"- **Dimensions Evaluated**: {', '.join(scores.get('dimensions', []))}\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | Overall Score | Emotion Recognition | Empathy | Regulation | Social Awareness |\n")
            f.write("|-------|---------------|-------------------|---------|------------|------------------|\n")
            
            # Add model scores (placeholder structure)
            for model in all_results.keys():
                f.write(f"| {model} | TBD | TBD | TBD | TBD | TBD |\n")
            
            f.write("\n## Key Findings\n\n")
            f.write("1. **Emotion Recognition**: [Add insights]\n")
            f.write("2. **Empathy Responses**: [Add insights]\n")
            f.write("3. **Emotional Regulation**: [Add insights]\n")
            f.write("4. **Social Awareness**: [Add insights]\n\n")
            
            f.write("## Sample Responses\n\n")
            # Add sample responses for each model
            
            f.write("## Methodology Notes\n\n")
            f.write("- Evaluation criteria and scoring methodology\n")
            f.write("- Limitations and considerations\n")
            f.write("- Future research directions\n")
        
        print(f"Comprehensive report saved to: {report_path}")
        return report_path

# Helper function to be used in the notebook
def save_all_results(results_dict, scores_df=None, summary_data=None):
    """Main function to save all results"""
    saver = ResultsSaver()
    
    # Save raw results
    raw_path = saver.save_raw_results(results_dict)
    
    # Save processed scores if provided
    if scores_df is not None:
        csv_path, json_path = saver.save_processed_scores(scores_df)
    
    # Save summary if provided
    if summary_data is not None:
        summary_path = saver.save_analysis_summary(summary_data)
    
    # Create comprehensive report
    report_path = saver.create_results_report(results_dict, summary_data or {})
    
    print("\n✅ All results saved successfully!")
    return {
        "raw": raw_path,
        "report": report_path
    }

if __name__ == "__main__":
    print("Results saver module loaded. Use save_all_results() function in your notebook.")