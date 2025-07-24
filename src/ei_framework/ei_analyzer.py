"""
Emotional Intelligence Analysis Module
Advanced analysis and visualization for EI test results

Author: Lorenzo De Tomasi
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import re
from datetime import datetime

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EIAnalyzer:
    """Advanced analyzer for emotional intelligence test results"""
    
    def __init__(self, results_file: Optional[str] = None):
        self.results = None
        self.analysis = None
        
        if results_file:
            self.load_results(results_file)
    
    def load_results(self, results_file: str):
        """Load results from JSON file"""
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        self.results = data.get('results', [])
        self.metadata = data.get('metadata', {})
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.results)
    
    def emotion_analysis(self) -> Dict[str, Any]:
        """Analyze emotion detection patterns"""
        emotion_stats = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            if result['success']:
                model = result['model_name']
                for emotion in result['emotions_detected']:
                    emotion_stats[model][emotion] += 1
        
        # Calculate emotion diversity
        emotion_diversity = {}
        for model, emotions in emotion_stats.items():
            unique_emotions = len(emotions)
            total_detections = sum(emotions.values())
            emotion_diversity[model] = {
                'unique_emotions': unique_emotions,
                'total_detections': total_detections,
                'diversity_score': unique_emotions / total_detections if total_detections > 0 else 0
            }
        
        return {
            'emotion_counts': dict(emotion_stats),
            'emotion_diversity': emotion_diversity
        }
    
    def response_quality_analysis(self) -> Dict[str, Any]:
        """Analyze response quality metrics"""
        quality_metrics = defaultdict(dict)
        
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            successful = model_data[model_data['success'] == True]
            
            if len(successful) > 0:
                # Response length analysis
                response_lengths = successful['response'].apply(len)
                
                # Token efficiency
                total_tokens = successful['tokens_used'].apply(lambda x: x.get('total_tokens', 0))
                
                quality_metrics[model] = {
                    'avg_response_length': response_lengths.mean(),
                    'response_length_std': response_lengths.std(),
                    'avg_tokens_used': total_tokens.mean(),
                    'tokens_per_char': total_tokens.mean() / response_lengths.mean() if response_lengths.mean() > 0 else 0
                }
        
        return dict(quality_metrics)
    
    def dimension_performance(self) -> pd.DataFrame:
        """Create dimension performance matrix"""
        # Create pivot table
        performance = self.df.pivot_table(
            values='success',
            index='model_name',
            columns='dimension',
            aggfunc='mean'
        ) * 100  # Convert to percentage
        
        return performance
    
    def empathy_scoring(self) -> Dict[str, float]:
        """Score empathy responses based on keywords and patterns"""
        empathy_keywords = [
            'sorry', 'understand', 'feel', 'support', 'here for you',
            'difficult', 'hard', 'compassion', 'care', 'listen',
            'acknowledge', 'valid', 'heart goes out'
        ]
        
        empathy_scores = {}
        
        for model in self.df['model_name'].unique():
            empathy_results = self.df[
                (self.df['model_name'] == model) & 
                (self.df['dimension'] == 'empathy') &
                (self.df['success'] == True)
            ]
            
            if len(empathy_results) > 0:
                scores = []
                for _, row in empathy_results.iterrows():
                    response_lower = row['response'].lower()
                    score = sum(1 for keyword in empathy_keywords if keyword in response_lower)
                    scores.append(score / len(empathy_keywords))
                
                empathy_scores[model] = np.mean(scores)
        
        return empathy_scores
    
    def generate_insights(self) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        # Success rates
        success_rates = self.df.groupby('model_name')['success'].mean() * 100
        best_success = success_rates.idxmax()
        insights.append(f"Highest success rate: {best_success} ({success_rates[best_success]:.1f}%)")
        
        # Response times
        avg_times = self.df[self.df['success'] == True].groupby('model_name')['response_time'].mean()
        fastest = avg_times.idxmin()
        insights.append(f"Fastest average response: {fastest} ({avg_times[fastest]:.2f}s)")
        
        # Emotion detection
        emotion_analysis = self.emotion_analysis()
        best_detector = max(
            emotion_analysis['emotion_diversity'].items(),
            key=lambda x: x[1]['unique_emotions']
        )
        insights.append(
            f"Best emotion detector: {best_detector[0]} "
            f"({best_detector[1]['unique_emotions']} unique emotions)"
        )
        
        # Empathy scores
        empathy_scores = self.empathy_scoring()
        if empathy_scores:
            best_empathy = max(empathy_scores.items(), key=lambda x: x[1])
            insights.append(f"Highest empathy score: {best_empathy[0]} ({best_empathy[1]:.2f})")
        
        return insights
    
    def create_visualizations(self, output_dir: str = "visualizations"):
        """Create all visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Success Rate Comparison
        plt.figure(figsize=(10, 6))
        success_rates = self.df.groupby('model_name')['success'].mean() * 100
        success_rates.plot(kind='bar')
        plt.title('Model Success Rates')
        plt.ylabel('Success Rate (%)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'success_rates.png', dpi=300)
        plt.close()
        
        # 2. Response Time Distribution
        plt.figure(figsize=(10, 6))
        successful_df = self.df[self.df['success'] == True]
        successful_df.boxplot(column='response_time', by='model_name')
        plt.title('Response Time Distribution by Model')
        plt.ylabel('Response Time (seconds)')
        plt.xlabel('Model')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(output_path / 'response_times.png', dpi=300)
        plt.close()
        
        # 3. Emotion Detection Heatmap
        emotion_data = self.emotion_analysis()
        emotion_matrix = []
        models = []
        all_emotions = set()
        
        for model, emotions in emotion_data['emotion_counts'].items():
            models.append(model)
            all_emotions.update(emotions.keys())
        
        all_emotions = sorted(all_emotions)
        
        for model in models:
            row = [emotion_data['emotion_counts'][model].get(emotion, 0) 
                   for emotion in all_emotions]
            emotion_matrix.append(row)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            emotion_matrix,
            xticklabels=all_emotions,
            yticklabels=models,
            cmap='YlOrRd',
            annot=True,
            fmt='d'
        )
        plt.title('Emotion Detection Frequency Heatmap')
        plt.xlabel('Emotions')
        plt.ylabel('Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'emotion_heatmap.png', dpi=300)
        plt.close()
        
        # 4. Dimension Performance Radar
        dimension_perf = self.dimension_performance()
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        dimensions = dimension_perf.columns.tolist()
        num_vars = len(dimensions)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        dimensions += dimensions[:1]
        angles += angles[:1]
        
        for idx, (model, row) in enumerate(dimension_perf.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d.replace('_', ' ').title() for d in dimensions[:-1]])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Success Rate (%)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Performance by EI Dimension', y=1.08)
        plt.tight_layout()
        plt.savefig(output_path / 'dimension_radar.png', dpi=300)
        plt.close()
        
        return output_path
    
    def generate_detailed_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("# Emotional Intelligence Analysis Report")
        report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall statistics
        report.append("\n## Overall Statistics")
        report.append(f"- Total tests: {len(self.df)}")
        report.append(f"- Successful tests: {self.df['success'].sum()}")
        report.append(f"- Overall success rate: {self.df['success'].mean() * 100:.1f}%")
        
        # Model performance
        report.append("\n## Model Performance")
        success_rates = self.df.groupby('model_name')['success'].mean() * 100
        avg_times = self.df[self.df['success'] == True].groupby('model_name')['response_time'].mean()
        
        for model in self.df['model_name'].unique():
            report.append(f"\n### {model}")
            report.append(f"- Success rate: {success_rates.get(model, 0):.1f}%")
            report.append(f"- Average response time: {avg_times.get(model, 0):.2f}s")
            
            # Emotions detected
            emotion_data = self.emotion_analysis()
            if model in emotion_data['emotion_diversity']:
                stats = emotion_data['emotion_diversity'][model]
                report.append(f"- Unique emotions detected: {stats['unique_emotions']}")
                report.append(f"- Emotion diversity score: {stats['diversity_score']:.3f}")
        
        # Dimension analysis
        report.append("\n## Dimension Analysis")
        dimension_perf = self.dimension_performance()
        report.append("\n```")
        report.append(dimension_perf.to_string())
        report.append("```")
        
        # Key insights
        report.append("\n## Key Insights")
        for insight in self.generate_insights():
            report.append(f"- {insight}")
        
        # Response quality
        report.append("\n## Response Quality Analysis")
        quality = self.response_quality_analysis()
        
        for model, metrics in quality.items():
            report.append(f"\n### {model}")
            report.append(f"- Average response length: {metrics['avg_response_length']:.0f} characters")
            report.append(f"- Average tokens used: {metrics['avg_tokens_used']:.0f}")
            report.append(f"- Token efficiency: {metrics['tokens_per_char']:.3f} tokens/char")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    import glob
    
    # Find most recent results file
    result_files = glob.glob("results/ei_results_*.json")
    if not result_files:
        print("No results files found!")
        return
    
    latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Analyzing: {latest_file}")
    
    # Initialize analyzer
    analyzer = EIAnalyzer(latest_file)
    
    # Generate visualizations
    viz_path = analyzer.create_visualizations()
    print(f"Visualizations saved to: {viz_path}")
    
    # Generate report
    report = analyzer.generate_detailed_report()
    
    # Save report
    report_path = Path("results") / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    print("\nKey Insights:")
    for insight in analyzer.generate_insights():
        print(f"  - {insight}")


if __name__ == "__main__":
    main()