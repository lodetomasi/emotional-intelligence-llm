# Python Implementation Guide

## Overview

The Emotional Intelligence Framework has been completely reimplemented as a modular Python package with the following components:

### Core Modules

1. **`ei_framework.py`** - Main framework class
   - `EmotionalIntelligenceFramework`: Core testing engine
   - Model management and API integration
   - Test execution and result collection
   - Automatic emotion extraction and analysis

2. **`ei_analyzer.py`** - Advanced analysis module
   - `EIAnalyzer`: Statistical analysis and visualization
   - Performance metrics calculation
   - Insight generation
   - Visualization creation (matplotlib/seaborn)

3. **`ei_config.py`** - Configuration module
   - Model specifications
   - Test scenario definitions
   - API settings
   - Analysis parameters

4. **`run_evaluation.py`** - Command-line interface
   - Argument parsing
   - Orchestration of testing and analysis
   - Report generation

## Usage

### Basic Usage

```python
from ei_framework import EmotionalIntelligenceFramework

# Initialize framework
framework = EmotionalIntelligenceFramework(api_key="your-key")

# Run evaluation
results = framework.run_comprehensive_evaluation()

# Save results
framework.save_results("results")

# Generate report
report = framework.generate_report()
print(report)
```

### Command Line Interface

```bash
# Run full evaluation
python run_evaluation.py --api-key YOUR_KEY

# Test specific models
python run_evaluation.py --models "Claude Opus 4,Llama 3.3 70B"

# Quick test (fewer scenarios)
python run_evaluation.py --quick

# Analyze existing results
python run_evaluation.py --analyze-only results/ei_results_20250124.json
```

### Advanced Analysis

```python
from ei_analyzer import EIAnalyzer

# Load results
analyzer = EIAnalyzer("results/ei_results_20250124.json")

# Generate insights
insights = analyzer.generate_insights()

# Create visualizations
analyzer.create_visualizations("output/viz")

# Generate detailed report
report = analyzer.generate_detailed_report()
```

## Architecture

### Class Structure

```
EmotionalIntelligenceFramework
├── models: Dict[str, ModelConfig]
├── scenarios: List[TestScenario]
├── results: List[TestResult]
├── query_model()
├── run_test()
├── run_comprehensive_evaluation()
├── analyze_results()
└── save_results()

EIAnalyzer
├── load_results()
├── emotion_analysis()
├── response_quality_analysis()
├── dimension_performance()
├── empathy_scoring()
├── generate_insights()
└── create_visualizations()
```

### Data Flow

1. **Configuration** → Load models and scenarios
2. **Execution** → Query each model with each scenario
3. **Collection** → Gather responses and metrics
4. **Analysis** → Calculate statistics and insights
5. **Visualization** → Generate charts and reports
6. **Output** → Save results in multiple formats

## Extending the Framework

### Adding New Models

```python
# In ei_config.py
MODELS["New Model"] = {
    "api_id": "provider/model-name",
    "provider": "Provider Name",
    "architecture": "Architecture Type",
    "context_length": 4096,
    "strengths": ["strength1", "strength2"]
}
```

### Adding Test Scenarios

```python
# In ei_config.py
TEST_SCENARIOS["emotion_recognition"].append({
    "id": "ER4",
    "name": "Complex Emotion",
    "prompt": "Your test prompt here",
    "expected_emotions": ["emotion1", "emotion2"],
    "difficulty": "hard"
})
```

### Custom Analysis

```python
class CustomAnalyzer(EIAnalyzer):
    def custom_metric(self):
        # Your custom analysis logic
        return results
```

## Output Formats

- **JSON**: Complete raw results with metadata
- **CSV**: Summary statistics for data analysis
- **Markdown**: Human-readable reports
- **PNG/SVG**: Publication-quality visualizations

## Performance Considerations

- Implements exponential backoff for API retries
- Rate limiting between requests (configurable)
- Batch processing support
- Async execution planned for v2.0

## Dependencies

See `requirements.txt` for full list. Core dependencies:
- pandas: Data manipulation
- matplotlib/seaborn: Visualization
- requests: API communication
- numpy: Numerical computation