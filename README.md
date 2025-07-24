# Emotional Intelligence in Large Language Models

A systematic framework for evaluating emotional intelligence capabilities in state-of-the-art Large Language Models.

## Abstract

This research presents a comprehensive evaluation framework for assessing emotional intelligence (EI) in Large Language Models. We test five leading models across four dimensions: emotion recognition, empathy, emotional regulation, and social awareness. Our methodology employs standardized scenarios to quantify and compare model capabilities in understanding and responding to complex emotional contexts.

## Research Overview

### Objectives

1. Develop a standardized framework for evaluating LLM emotional intelligence
2. Compare EI capabilities across leading language models
3. Identify strengths and limitations in current AI emotional understanding
4. Provide insights for improving human-AI interaction design

### Models Evaluated

- **Mixtral-8x22B** (Mistral AI) - Mixture of Experts architecture
- **Claude Opus 4** (Anthropic) - Constitutional AI approach  
- **Llama 3.3 70B** (Meta) - Open-source foundation model
- **DeepSeek R1** (DeepSeek) - Reasoning-optimized model
- **Gemini 2.5 Pro** (Google) - Multimodal capabilities

## Key Results

### Performance Summary

| Model | Success Rate | Avg Response Time | Emotions Detected | Empathy Score |
|-------|--------------|-------------------|-------------------|---------------|
| DeepSeek R1 | 100% | 16.57s | 11 | High |
| Claude Opus 4 | 100% | 11.98s | 10 | Very High |
| Llama 3.3 70B | 100% | 10.69s | 10 | Very High |
| Mixtral-8x22B | 100% | 3.47s | 8 | High |
| Gemini 2.5 Pro | 100% | 17.77s | 1 | Moderate |

### Key Findings

1. **Emotion Recognition**: DeepSeek R1 demonstrated the most comprehensive emotion detection, identifying 11 distinct emotional states
2. **Response Efficiency**: Mixtral-8x22B achieved the optimal balance between speed (3.47s) and quality
3. **Empathetic Communication**: Claude Opus 4 and Llama 3.3 70B excelled in generating compassionate, contextually appropriate responses
4. **Consistency**: All models achieved 100% task completion rate, indicating robust performance

## Methodology

### Evaluation Framework

Our framework assesses four core dimensions of emotional intelligence:

#### 1. Emotion Recognition
- Identification of multiple emotions in complex scenarios
- Analysis of emotional nuance and context
- Detection of explicit and implicit emotional states

#### 2. Empathy
- Generation of compassionate, appropriate responses
- Demonstration of perspective-taking abilities
- Provision of contextually relevant emotional support

#### 3. Emotional Regulation 
- Professional response management under stress
- Appropriate reaction to criticism or conflict
- De-escalation strategies in tense situations

#### 4. Social Awareness
- Interpretation of social cues and context
- Understanding of group dynamics
- Culturally sensitive responses

### Test Design

- **Scenarios**: 10 standardized test cases across all dimensions
- **Evaluation**: Automated scoring with human validation criteria
- **Metrics**: Response accuracy, completeness, appropriateness, and timing
- **API Integration**: OpenRouter for consistent model access

## Installation and Usage

### Requirements

- Python 3.8+
- OpenRouter API access
- Required packages: `pip install -r requirements.txt`

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/lodetomasi/emotional-intelligence-llm.git
   cd emotional-intelligence-llm
   ```

2. Set up API credentials:
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   ```

3. Run the evaluation:
   ```bash
   python test_all_dimensions.py
   ```

4. Analyze results:
   ```bash
   python analyze_and_visualize.py
   ```

## Project Structure

```
emotional-intelligence-llm/
├── emotional_intelligence_test.ipynb  # Interactive testing notebook
├── test_all_dimensions.py            # Comprehensive test suite
├── analyze_and_visualize.py          # Analysis and visualization tools
├── save_results.py                   # Results management module
├── results/                          # Test outputs
│   ├── test_results/                 # Raw test data
│   ├── processed_data/               # Analyzed metrics
│   └── ei_analysis_report.md         # Generated reports
└── requirements.txt                  # Dependencies
```

## Data and Results

The framework generates comprehensive outputs including:

- **Raw Data**: Complete API responses with metadata (JSON format)
- **Processed Metrics**: Quantitative scores and performance indicators (CSV/JSON)
- **Analysis Reports**: Detailed markdown reports with qualitative insights
- **Statistical Analysis**: Inter-model comparisons and dimension-wise breakdowns

## Future Research Directions

1. **Extended Test Battery**: Expansion to 100+ scenarios per dimension for statistical robustness
2. **Human Baseline Studies**: Comparative analysis with human emotional intelligence benchmarks
3. **Cross-Cultural Validation**: Assessment of cultural sensitivity in emotional understanding
4. **Longitudinal Analysis**: Temporal consistency evaluation across model updates
5. **Domain-Specific Applications**: Healthcare, education, and mental health support contexts

## Contributing

We welcome contributions in the following areas:

- Test scenario development
- Evaluation metric refinement
- Additional model integration
- Statistical analysis methods
- Cross-cultural test cases

Please submit pull requests with clear descriptions and test coverage.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenRouter for unified API access
- University of L'Aquila for research support
- Anonymous reviewers for valuable feedback

## Author

**Lorenzo De Tomasi**  
Department of Computer Science  
University of L'Aquila, Italy  
lorenzo.detomasi@graduate.univaq.it

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{detomasi2025eilm,
  title={A Framework for Evaluating Emotional Intelligence in Large Language Models},
  author={De Tomasi, Lorenzo},
  year={2025},
  institution={University of L'Aquila},
  url={https://github.com/lodetomasi/emotional-intelligence-llm}
}
```