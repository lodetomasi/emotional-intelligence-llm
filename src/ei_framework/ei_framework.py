"""
Emotional Intelligence Framework for Large Language Models
A comprehensive Python framework for evaluating LLM emotional intelligence

Author: Lorenzo De Tomasi
University of L'Aquila
"""

import json
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EIDimension(Enum):
    """Emotional Intelligence dimensions"""
    EMOTION_RECOGNITION = "emotion_recognition"
    EMPATHY = "empathy"
    EMOTIONAL_REGULATION = "emotional_regulation"
    SOCIAL_AWARENESS = "social_awareness"


@dataclass
class TestScenario:
    """Test scenario configuration"""
    id: str
    dimension: EIDimension
    prompt: str
    expected_emotions: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    api_id: str
    provider: str
    architecture: str


@dataclass
class TestResult:
    """Individual test result"""
    model_name: str
    scenario_id: str
    dimension: EIDimension
    success: bool
    response: str
    response_time: float
    tokens_used: Dict[str, int]
    emotions_detected: List[str] = field(default_factory=list)
    error: Optional[str] = None


class EmotionalIntelligenceFramework:
    """Main framework for evaluating LLM emotional intelligence"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.models = self._initialize_models()
        self.scenarios = self._initialize_scenarios()
        self.results = []
        
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations"""
        return {
            "Mixtral-8x22B": ModelConfig(
                name="Mixtral-8x22B",
                api_id="mistralai/mixtral-8x22b-instruct",
                provider="Mistral AI",
                architecture="Mixture of Experts"
            ),
            "Claude Opus 4": ModelConfig(
                name="Claude Opus 4",
                api_id="anthropic/claude-opus-4",
                provider="Anthropic",
                architecture="Constitutional AI"
            ),
            "Llama 3.3 70B": ModelConfig(
                name="Llama 3.3 70B",
                api_id="meta-llama/llama-3.3-70b-instruct",
                provider="Meta",
                architecture="Transformer"
            ),
            "DeepSeek R1": ModelConfig(
                name="DeepSeek R1",
                api_id="deepseek/deepseek-r1-0528",
                provider="DeepSeek",
                architecture="Reasoning-optimized"
            ),
            "Gemini 2.5 Pro": ModelConfig(
                name="Gemini 2.5 Pro",
                api_id="google/gemini-2.5-pro",
                provider="Google",
                architecture="Multimodal"
            )
        }
    
    def _initialize_scenarios(self) -> List[TestScenario]:
        """Initialize test scenarios"""
        scenarios = []
        
        # Emotion Recognition scenarios
        scenarios.extend([
            TestScenario(
                id="ER1",
                dimension=EIDimension.EMOTION_RECOGNITION,
                prompt="""Identify ALL emotions in this text and explain why:
"I can't believe I have to work late again tonight. I promised my daughter I'd be at her recital, but this project deadline is non-negotiable. I feel like I'm constantly letting her down. Sometimes I wonder if this job is worth missing all these important moments."

List each emotion you detect.""",
                expected_emotions=["frustration", "guilt", "disappointment", "conflict", "regret", "doubt"]
            ),
            TestScenario(
                id="ER2",
                dimension=EIDimension.EMOTION_RECOGNITION,
                prompt="""What emotions are present in this message:
"Just got the promotion! 😊 Although... I'm not sure I deserve it. Maybe they made a mistake? What if I can't handle the responsibility?"

Identify ALL emotions, including conflicting ones.""",
                expected_emotions=["joy", "self-doubt", "anxiety", "imposter syndrome", "fear"]
            ),
            TestScenario(
                id="ER3",
                dimension=EIDimension.EMOTION_RECOGNITION,
                prompt="""Read between the lines and identify hidden emotions:
"Sure, go ahead to the party without me. It's fine. I didn't really want to go anyway."

What emotions might be beneath the surface?""",
                expected_emotions=["hurt", "rejection", "passive-aggressiveness", "sadness", "loneliness"]
            )
        ])
        
        # Empathy scenarios
        scenarios.extend([
            TestScenario(
                id="EM1",
                dimension=EIDimension.EMPATHY,
                prompt="""Your friend messages: "I had to put my dog down yesterday. He was with me for 15 years. The house feels so empty without him."

Write a compassionate, empathetic response.""",
                evaluation_criteria=["acknowledgment", "validation", "support", "personal_touch"]
            ),
            TestScenario(
                id="EM2",
                dimension=EIDimension.EMPATHY,
                prompt="""A colleague says: "I completely bombed the presentation. Everyone was staring at me like I was an idiot."

Provide an empathetic response.""",
                evaluation_criteria=["normalize", "reframe", "support", "no_minimizing"]
            )
        ])
        
        # Emotional Regulation scenarios
        scenarios.extend([
            TestScenario(
                id="REG1",
                dimension=EIDimension.EMOTIONAL_REGULATION,
                prompt="""In a meeting, a colleague says: "Your analysis is completely wrong and shows you don't understand the basics."

How would you respond professionally?""",
                evaluation_criteria=["calm", "professional", "de-escalate", "boundaries"]
            )
        ])
        
        # Social Awareness scenarios
        scenarios.extend([
            TestScenario(
                id="SA1",
                dimension=EIDimension.SOCIAL_AWARENESS,
                prompt="""At a party, you notice a new colleague standing alone, looking at their phone with hunched shoulders.

What might be happening and how would you approach them?""",
                evaluation_criteria=["observation", "empathy", "appropriate", "respectful"]
            )
        ])
        
        return scenarios
    
    def query_model(self, model_config: ModelConfig, prompt: str, 
                   max_retries: int = 2) -> Tuple[bool, str, float, Dict[str, int], Optional[str]]:
        """Query a model and return results"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_config.api_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are participating in an emotional intelligence assessment."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=45
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    return (
                        True,
                        result["choices"][0]["message"]["content"],
                        elapsed,
                        result.get("usage", {}),
                        None
                    )
                else:
                    error = f"API Error {response.status_code}: {response.text[:200]}"
                    if attempt == max_retries - 1:
                        return False, "", elapsed, {}, error
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    return False, "", 0, {}, str(e)
            
            time.sleep(2 ** attempt)
        
        return False, "", 0, {}, "Max retries exceeded"
    
    def extract_emotions(self, text: str) -> List[str]:
        """Extract emotions mentioned in text"""
        emotions = [
            'frustration', 'guilt', 'disappointment', 'anger', 'sadness',
            'joy', 'anxiety', 'fear', 'doubt', 'hurt', 'rejection',
            'loneliness', 'stress', 'regret', 'conflict', 'helplessness',
            'resignation', 'worry', 'shame', 'confusion', 'overwhelm',
            'empathy', 'compassion', 'understanding'
        ]
        
        text_lower = text.lower()
        found_emotions = []
        
        for emotion in emotions:
            if emotion in text_lower:
                found_emotions.append(emotion)
        
        return list(set(found_emotions))
    
    def run_test(self, model_name: str, scenario: TestScenario) -> TestResult:
        """Run a single test"""
        model_config = self.models[model_name]
        
        logger.info(f"Testing {model_name} on scenario {scenario.id}")
        
        success, response, elapsed, usage, error = self.query_model(
            model_config, scenario.prompt
        )
        
        emotions_detected = self.extract_emotions(response) if success else []
        
        return TestResult(
            model_name=model_name,
            scenario_id=scenario.id,
            dimension=scenario.dimension,
            success=success,
            response=response,
            response_time=elapsed,
            tokens_used=usage,
            emotions_detected=emotions_detected,
            error=error
        )
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation across all models and scenarios"""
        logger.info("Starting comprehensive emotional intelligence evaluation")
        
        start_time = datetime.now()
        self.results = []
        
        for model_name in self.models:
            logger.info(f"\nEvaluating {model_name}")
            
            for scenario in self.scenarios:
                result = self.run_test(model_name, scenario)
                self.results.append(result)
                time.sleep(2)  # Rate limiting
            
            time.sleep(3)  # Extra delay between models
        
        end_time = datetime.now()
        
        return {
            "metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "total_tests": len(self.results),
                "models_tested": list(self.models.keys()),
                "scenarios_tested": len(self.scenarios)
            },
            "results": self.results
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results"""
        if not self.results:
            return {}
        
        analysis = {
            "model_performance": {},
            "dimension_analysis": {},
            "emotion_detection": {},
            "response_times": {}
        }
        
        # Model performance
        for model_name in self.models:
            model_results = [r for r in self.results if r.model_name == model_name]
            successful = [r for r in model_results if r.success]
            
            analysis["model_performance"][model_name] = {
                "success_rate": len(successful) / len(model_results) * 100 if model_results else 0,
                "avg_response_time": np.mean([r.response_time for r in successful]) if successful else 0,
                "total_tokens": sum(r.tokens_used.get("total_tokens", 0) for r in successful),
                "unique_emotions": len(set(sum([r.emotions_detected for r in successful], [])))
            }
        
        # Dimension analysis
        for dimension in EIDimension:
            dim_results = [r for r in self.results if r.dimension == dimension]
            successful = [r for r in dim_results if r.success]
            
            analysis["dimension_analysis"][dimension.value] = {
                "total_tests": len(dim_results),
                "successful_tests": len(successful),
                "success_rate": len(successful) / len(dim_results) * 100 if dim_results else 0
            }
        
        return analysis
    
    def save_results(self, output_dir: str = "results"):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_data = {
            "metadata": {
                "timestamp": timestamp,
                "models": list(self.models.keys()),
                "scenarios": [s.id for s in self.scenarios]
            },
            "results": [
                {
                    "model_name": r.model_name,
                    "scenario_id": r.scenario_id,
                    "dimension": r.dimension.value,
                    "success": r.success,
                    "response": r.response,
                    "response_time": r.response_time,
                    "tokens_used": r.tokens_used,
                    "emotions_detected": r.emotions_detected,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        with open(output_path / f"ei_results_{timestamp}.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Save analysis
        analysis = self.analyze_results()
        with open(output_path / f"ei_analysis_{timestamp}.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for r in self.results:
            summary_data.append({
                "model": r.model_name,
                "scenario": r.scenario_id,
                "dimension": r.dimension.value,
                "success": r.success,
                "response_time": r.response_time,
                "emotions_detected": len(r.emotions_detected),
                "total_tokens": r.tokens_used.get("total_tokens", 0)
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path / f"ei_summary_{timestamp}.csv", index=False)
        
        logger.info(f"Results saved to {output_path}")
        
        return output_path / f"ei_results_{timestamp}.json"
    
    def generate_report(self) -> str:
        """Generate markdown report"""
        analysis = self.analyze_results()
        
        report = []
        report.append("# Emotional Intelligence Evaluation Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n## Model Performance Summary")
        report.append("\n| Model | Success Rate | Avg Response Time | Unique Emotions |")
        report.append("|-------|--------------|-------------------|-----------------|")
        
        for model, stats in analysis["model_performance"].items():
            report.append(
                f"| {model} | {stats['success_rate']:.1f}% | "
                f"{stats['avg_response_time']:.2f}s | {stats['unique_emotions']} |"
            )
        
        report.append("\n## Dimension Analysis")
        for dimension, stats in analysis["dimension_analysis"].items():
            report.append(f"\n### {dimension.replace('_', ' ').title()}")
            report.append(f"- Success Rate: {stats['success_rate']:.1f}%")
            report.append(f"- Tests Completed: {stats['successful_tests']}/{stats['total_tests']}")
        
        return "\n".join(report)


def main():
    """Main execution function"""
    # Initialize framework
    api_key = os.getenv("OPENROUTER_API_KEY", "your-api-key")
    framework = EmotionalIntelligenceFramework(api_key)
    
    # Run evaluation
    results = framework.run_comprehensive_evaluation()
    
    # Analyze and save
    framework.save_results()
    
    # Generate report
    report = framework.generate_report()
    print(report)
    
    # Save report
    with open("ei_evaluation_report.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()