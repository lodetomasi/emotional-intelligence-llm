# Emotional Intelligence in Large Language Models

> A comprehensive framework for evaluating emotional intelligence capabilities in state-of-the-art Large Language Models through systematic testing and analysis.

## Abstract

This research investigates the emotional intelligence (EI) capabilities of five leading Large Language Models, examining their ability to recognize emotions, demonstrate empathy, regulate emotional responses, and exhibit social awareness. Our framework provides both quantitative metrics and qualitative analysis of model performance across emotionally complex scenarios.

## 📊 Key Findings

Our initial tests reveal significant variations in emotional intelligence across models:

| Model | Emotions Detected | Analysis Depth | Key Strengths |
|-------|------------------|----------------|---------------|
| **Llama 3.3 70B** | 7 emotions | Excellent | Most comprehensive emotional taxonomy, identifies nuanced states like resignation |
| **Claude Opus 4** | 6 emotions | Excellent | Superior contextual understanding, recognizes underlying sadness and stress |
| **Mixtral-8x22B** | 3 emotions | Good | Accurate identification of core emotions with clear explanations |

### Sample Analysis: Work-Life Balance Scenario

*Test prompt: "I can't believe I have to work late again tonight. I promised my daughter I'd be at her recital..."*

- **Core emotions identified by all models**: Frustration, Guilt, Disappointment
- **Advanced emotions (Claude/Llama only)**: Helplessness, Anxiety, Sadness, Resignation
- **Response quality**: Models showed varying degrees of nuance in understanding the emotional complexity

## 🔬 Methodology

### Four Dimensions of Emotional Intelligence

1. **Emotion Recognition**
   - Identifying multiple emotions in complex texts
   - Understanding emotional nuance and context
   - Detecting both explicit and implicit emotional states

2. **Empathy**
   - Generating compassionate responses to emotional scenarios
   - Demonstrating understanding of others' perspectives
   - Providing appropriate emotional support

3. **Emotional Regulation**
   - Managing responses in high-stress situations
   - Maintaining professionalism under criticism
   - De-escalating emotional conflicts

4. **Social Awareness**
   - Reading social cues and body language descriptions
   - Understanding group dynamics
   - Responding appropriately to social contexts

### Models Tested

- **Mixtral-8x22B** (Mistral AI) - Mixture of Experts architecture
- **Claude Opus 4** (Anthropic) - Constitutional AI approach
- **Llama 3.3 70B** (Meta) - Open-source foundation model
- **DeepSeek R1** (DeepSeek) - Reasoning-optimized model
- **Gemini 2.5 Pro** (Google) - Multimodal capabilities

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configuration

1. Obtain an OpenRouter API key from [openrouter.ai](https://openrouter.ai)
2. Set your API key:
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   ```

### Running the Tests

```bash
jupyter notebook emotional_intelligence_test.ipynb
```

## 📁 Project Structure

```
emotional-intelligence-llm/
├── emotional_intelligence_test.ipynb  # Main testing notebook
├── save_results.py                    # Results management module
├── results/                           # Test outputs
│   ├── raw_responses/                 # Complete API responses
│   ├── processed_data/                # Analyzed scores and metrics
│   └── visualizations/                # Charts and graphs
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 📈 Results Format

The framework generates multiple output formats:

- **Raw API Responses**: Complete model outputs with metadata
- **Processed Scores**: CSV/JSON files with quantitative metrics
- **Analysis Reports**: Markdown reports with qualitative insights
- **Visualizations**: Publication-ready charts and comparison matrices

## 🔮 Future Work

1. **Expand Test Scenarios**: Scale to 100+ scenarios per dimension
2. **Human Baseline**: Compare with human emotional intelligence scores
3. **Cross-Cultural Analysis**: Test emotion recognition across cultures
4. **Temporal Consistency**: Evaluate response stability over time
5. **Fine-tuning Impact**: Assess how EI-focused training affects performance

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new test scenarios
- Implement additional evaluation metrics
- Test additional models
- Improve visualization methods

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Acknowledgments

- OpenRouter for API access to multiple models
- The LLM research community for ongoing discussions on AI capabilities
- University of L'Aquila for research support

## 👨‍🔬 Author

**Lorenzo De Tomasi**  
PhD Candidate in Computer Science  
University of L'Aquila, Italy  
📧 lorenzo.detomasi@graduate.univaq.it  
🔗 [GitHub](https://github.com/lodetomasi) | [LinkedIn](https://linkedin.com/in/lorenzodetomasi)

## 📝 Citation

```bibtex
@article{detomasi2025emotional,
  title={Emotional Intelligence in Large Language Models: A Comparative Analysis},
  author={De Tomasi, Lorenzo},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025},
  institution={University of L'Aquila}
}
```

---

<p align="center">
  <i>"The question is not whether intelligent machines can have emotions, but whether machines can be intelligent without them."</i><br>
  - Marvin Minsky
</p>