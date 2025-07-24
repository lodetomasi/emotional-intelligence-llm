# Project Structure

```
emotional-intelligence-llm/
├── src/                      # Source code
│   ├── ei_framework/         # Main framework package
│   │   ├── __init__.py
│   │   ├── ei_framework.py   # Core framework
│   │   ├── ei_improved_framework.py  # Enhanced version with security
│   │   ├── ei_analyzer.py    # Analysis and visualization
│   │   └── ei_config.py      # Configuration definitions
│   └── tests/                # Test modules
│       ├── __init__.py
│       ├── test_all_dimensions.py
│       └── test_all_verbose.py
│
├── scripts/                  # Executable scripts
│   ├── run_evaluation.py     # Main CLI interface
│   ├── run_comprehensive_tests.py
│   └── analyze_and_visualize.py
│
├── config/                   # Configuration files
│   └── default_config.json   # Default settings
│
├── examples/                 # Usage examples
│   ├── basic_usage.py        # Simple example
│   ├── advanced_analysis.py  # Analysis example
│   └── emotional_intelligence_test.ipynb  # Jupyter notebook
│
├── docs/                     # Documentation
│   ├── README_PYTHON.md      # Python implementation guide
│   ├── SECURITY.md           # Security guidelines
│   ├── RESULTS.md            # Test results documentation
│   └── methodology/          # Methodology details
│       └── emotion_detection.md
│
├── results/                  # Output directory (gitignored)
│   ├── test_results/         # Raw test outputs
│   ├── processed_data/       # Analyzed data
│   └── visualizations/       # Generated charts
│
├── setup.py                  # Package installation
├── pyproject.toml           # Modern Python packaging
├── requirements.txt         # Dependencies
├── README.md               # Main documentation
├── LICENSE                 # MIT License
├── .env.example           # Environment template
└── .gitignore            # Git ignore rules
```

## Installation

```bash
# Clone repository
git clone https://github.com/lodetomasi/emotional-intelligence-llm.git
cd emotional-intelligence-llm

# Install package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

```bash
# Set API key
export OPENROUTER_API_KEY="your-key"

# Run evaluation
python scripts/run_evaluation.py

# Or use as module
python -m ei_framework.run_evaluation
```