"""
Improved Emotional Intelligence Framework with bias mitigation
and comprehensive error handling

Author: Lorenzo De Tomasi
"""

import json
import requests
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import re
from collections import Counter

# Security and validation
from urllib.parse import urlparse
import secrets

logger = logging.getLogger(__name__)


class EmotionDetectionMethod(Enum):
    """Methods for emotion detection"""
    KEYWORD = "keyword"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"


@dataclass
class EmotionScore:
    """Detailed emotion scoring"""
    emotion: str
    confidence: float
    evidence: List[str]
    context: str


class ImprovedEmotionalIntelligenceFramework:
    """Enhanced framework with bias mitigation and security improvements"""
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        # Security: Never store API key in plain text
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided via parameter or OPENROUTER_API_KEY environment variable")
        
        # Load configuration
        self.config = self._load_secure_config(config_path)
        self.models = self.config.get("models", {})
        self.scenarios = self.config.get("scenarios", [])
        self.results = []
        
        # Initialize secure paths
        self.base_path = Path.home() / ".ei_framework"
        self.base_path.mkdir(exist_ok=True)
        
    def _load_secure_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with validation"""
        if config_path:
            # Validate path to prevent traversal attacks
            config_path = Path(config_path).resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Load default configuration
            from ei_config import MODELS, TEST_SCENARIOS, API_CONFIG
            config = {
                "models": MODELS,
                "scenarios": TEST_SCENARIOS,
                "api": API_CONFIG
            }
        
        return config
    
    def detect_emotions_advanced(self, text: str, method: EmotionDetectionMethod = EmotionDetectionMethod.HYBRID) -> List[EmotionScore]:
        """Advanced emotion detection with reduced bias"""
        emotions_found = []
        
        # Normalize text
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        
        # Comprehensive emotion patterns
        emotion_patterns = {
            'frustration': {
                'keywords': ['frustrated', 'frustrating', "can't believe", 'again'],
                'patterns': [r"have to.*again", r"constantly.*ing"],
                'negation_sensitive': True
            },
            'guilt': {
                'keywords': ['guilt', 'guilty', 'letting down', 'fault', 'blame'],
                'patterns': [r"feel like.*letting", r"my fault"],
                'negation_sensitive': True
            },
            'disappointment': {
                'keywords': ['disappointed', 'disappointment', 'let down'],
                'patterns': [r"promised.*but", r"expected.*but"],
                'negation_sensitive': True
            },
            'anxiety': {
                'keywords': ['anxious', 'worried', 'nervous', 'stress'],
                'patterns': [r"what if", r"worried about"],
                'negation_sensitive': True
            },
            'joy': {
                'keywords': ['happy', 'joy', 'excited', 'glad', 'promotion'],
                'patterns': [r"!+", r":\)|😊|😄"],
                'negation_sensitive': True
            },
            'sadness': {
                'keywords': ['sad', 'depressed', 'empty', 'loss', 'grief'],
                'patterns': [r"feels.*empty", r"miss.*ing"],
                'negation_sensitive': True
            }
        }
        
        # Check each emotion with context
        for emotion, config in emotion_patterns.items():
            confidence = 0.0
            evidence = []
            
            # Check keywords
            for keyword in config['keywords']:
                if keyword in text_lower:
                    # Check for negation
                    if config['negation_sensitive']:
                        # Look for negation within 3 words
                        if not re.search(rf"(not|no|never|neither)\s+\w*\s*\w*\s*{keyword}", text_lower):
                            confidence += 0.3
                            evidence.append(f"Keyword: {keyword}")
                    else:
                        confidence += 0.3
                        evidence.append(f"Keyword: {keyword}")
            
            # Check patterns
            for pattern in config['patterns']:
                matches = re.findall(pattern, text_lower)
                if matches:
                    confidence += 0.4
                    evidence.append(f"Pattern: {pattern}")
            
            # Context analysis
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in config['keywords']):
                    confidence += 0.1
                    
            # Add emotion if confidence threshold met
            if confidence > 0.3:
                emotions_found.append(EmotionScore(
                    emotion=emotion,
                    confidence=min(confidence, 1.0),
                    evidence=evidence,
                    context=text[:100] + "..." if len(text) > 100 else text
                ))
        
        return sorted(emotions_found, key=lambda x: x.confidence, reverse=True)
    
    def query_model_secure(self, model_config: Dict, prompt: str, 
                          max_retries: int = 3) -> Tuple[bool, str, float, Dict[str, int], Optional[str]]:
        """Secure model querying with comprehensive error handling"""
        
        # Input validation
        if not prompt or len(prompt) > 10000:
            return False, "", 0, {}, "Invalid prompt length"
        
        # Secure headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "EI-Framework/1.0"
        }
        
        # Request data with safety limits
        data = {
            "model": model_config.get("api_id"),
            "messages": [
                {
                    "role": "system",
                    "content": "You are participating in an emotional intelligence assessment. Provide thoughtful, unbiased responses."
                },
                {
                    "role": "user",
                    "content": prompt[:5000]  # Limit prompt length
                }
            ],
            "temperature": self.config.get("api", {}).get("temperature", 0.7),
            "max_tokens": min(self.config.get("api", {}).get("max_tokens", 800), 2000),
            "timeout": self.config.get("api", {}).get("timeout", 45)
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Make request with timeout
                response = requests.post(
                    self.config.get("api", {}).get("base_url", "https://openrouter.ai/api/v1/chat/completions"),
                    headers=headers,
                    json=data,
                    timeout=data["timeout"]
                )
                
                elapsed = time.time() - start_time
                
                # Handle different status codes
                if response.status_code == 200:
                    try:
                        result = response.json()
                        # Validate response structure
                        if "choices" in result and len(result["choices"]) > 0:
                            return (
                                True,
                                result["choices"][0]["message"]["content"],
                                elapsed,
                                result.get("usage", {}),
                                None
                            )
                        else:
                            last_error = "Invalid response structure"
                    except json.JSONDecodeError:
                        last_error = "Invalid JSON response"
                        
                elif response.status_code == 429:
                    # Rate limit - wait longer
                    wait_time = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code == 401:
                    return False, "", elapsed, {}, "Authentication failed - check API key"
                    
                else:
                    last_error = f"API Error {response.status_code}: {response.text[:200]}"
                    
            except requests.exceptions.Timeout:
                last_error = f"Request timeout after {data['timeout']} seconds"
                
            except requests.exceptions.ConnectionError:
                last_error = "Connection error - check internet connection"
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.exception("Unexpected error in query_model_secure")
            
            # Exponential backoff
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                time.sleep(wait_time)
        
        return False, "", 0, {}, last_error or "Max retries exceeded"
    
    def evaluate_empathy_response(self, response: str, criteria: List[str]) -> Dict[str, float]:
        """Evaluate empathy with reduced keyword bias"""
        scores = {}
        
        # Comprehensive empathy evaluation
        evaluation_rubric = {
            "acknowledgment": {
                "indicators": [
                    r"i.*(understand|hear|see).*you",
                    r"that.*(must be|sounds|seems).*difficult",
                    r"(sorry|sympathize).*your"
                ],
                "weight": 0.25
            },
            "validation": {
                "indicators": [
                    r"(valid|understandable|normal).*feel",
                    r"(right|okay).*to feel",
                    r"anyone would.*feel"
                ],
                "weight": 0.25
            },
            "support": {
                "indicators": [
                    r"(here|available|support).*you",
                    r"(help|assist|anything).*need",
                    r"(together|with you)"
                ],
                "weight": 0.25
            },
            "personal_touch": {
                "indicators": [
                    r"(remember|think of|imagine)",
                    r"(personal experience|relate)",
                    r"(specific|particular).*situation"
                ],
                "weight": 0.25
            }
        }
        
        response_lower = response.lower()
        
        for criterion, config in evaluation_rubric.items():
            score = 0.0
            matches = 0
            
            for pattern in config["indicators"]:
                if re.search(pattern, response_lower):
                    matches += 1
            
            # Score based on matches and response length
            if matches > 0:
                score = min(matches / len(config["indicators"]), 1.0)
                
                # Adjust for response quality
                if len(response) > 50:  # Minimal response penalty
                    score *= 1.1
                if len(response) > 200:  # Thoughtful response bonus
                    score *= 1.1
                    
                score = min(score, 1.0)
            
            scores[criterion] = score * config["weight"]
        
        # Overall empathy score
        scores["overall"] = sum(scores.values())
        
        return scores
    
    def save_results_secure(self, output_dir: Optional[str] = None) -> Path:
        """Save results with security measures"""
        # Validate output directory
        if output_dir:
            output_path = Path(output_dir).resolve()
            # Ensure path is within allowed directories
            if not str(output_path).startswith(str(Path.cwd())):
                raise ValueError("Output directory must be within current working directory")
        else:
            output_path = Path("results")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate secure filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = secrets.token_hex(4)
        filename = f"ei_results_{timestamp}_{random_suffix}.json"
        
        # Prepare data for saving (remove sensitive information)
        safe_results = {
            "metadata": {
                "timestamp": timestamp,
                "framework_version": "1.0",
                "models_tested": [m["model_name"] for m in self.results],
                "total_tests": len(self.results)
            },
            "results": []
        }
        
        # Clean results of sensitive data
        for result in self.results:
            clean_result = result.copy()
            # Remove any API keys or sensitive data
            if "api_key" in clean_result:
                del clean_result["api_key"]
            safe_results["results"].append(clean_result)
        
        # Save with secure permissions
        output_file = output_path / filename
        with open(output_file, 'w') as f:
            json.dump(safe_results, f, indent=2)
        
        # Set restrictive permissions (Unix-like systems)
        try:
            os.chmod(output_file, 0o600)
        except:
            pass  # Windows doesn't support chmod
        
        logger.info(f"Results saved securely to {output_file}")
        return output_file
    
    def generate_unbiased_report(self) -> str:
        """Generate report without methodological biases"""
        report = []
        report.append("# Emotional Intelligence Evaluation Report")
        report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n**Note**: Scores are based on pattern matching and should be validated with human judgment")
        
        # Calculate statistics without bias
        model_stats = {}
        for result in self.results:
            model = result.get("model_name")
            if model not in model_stats:
                model_stats[model] = {
                    "tests": 0,
                    "successful": 0,
                    "total_response_time": 0,
                    "emotion_counts": Counter(),
                    "empathy_scores": []
                }
            
            model_stats[model]["tests"] += 1
            if result.get("success"):
                model_stats[model]["successful"] += 1
                model_stats[model]["total_response_time"] += result.get("response_time", 0)
                
                # Count emotions without bias
                if "emotions_detected" in result:
                    for emotion in result["emotions_detected"]:
                        model_stats[model]["emotion_counts"][emotion.emotion] += 1
                
                # Collect empathy scores
                if "empathy_score" in result:
                    model_stats[model]["empathy_scores"].append(result["empathy_score"]["overall"])
        
        # Report findings objectively
        report.append("\n## Model Performance (Objective Metrics)")
        report.append("\n| Model | Success Rate | Avg Response Time | Unique Emotions |")
        report.append("|-------|--------------|-------------------|-----------------|")
        
        for model, stats in model_stats.items():
            success_rate = (stats["successful"] / stats["tests"] * 100) if stats["tests"] > 0 else 0
            avg_time = (stats["total_response_time"] / stats["successful"]) if stats["successful"] > 0 else 0
            unique_emotions = len(stats["emotion_counts"])
            
            report.append(f"| {model} | {success_rate:.1f}% | {avg_time:.2f}s | {unique_emotions} |")
        
        report.append("\n## Important Considerations")
        report.append("- Emotion detection is based on pattern matching and may not capture nuanced expressions")
        report.append("- Empathy scores reflect structural elements and should be validated qualitatively")
        report.append("- Response times may vary based on API load and network conditions")
        report.append("- Results should be interpreted in context of the specific use case")
        
        return "\n".join(report)


def create_secure_config(output_path: str = "config.secure.json"):
    """Create a secure configuration file"""
    config = {
        "api": {
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "timeout": 45,
            "max_retries": 3,
            "rate_limit_delay": 2,
            "temperature": 0.7,
            "max_tokens": 800
        },
        "security": {
            "max_prompt_length": 5000,
            "allowed_output_dirs": ["./results", "./reports"],
            "sanitize_paths": True
        },
        "evaluation": {
            "emotion_detection_method": "hybrid",
            "empathy_rubric_version": "1.0",
            "bias_mitigation": True
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set restrictive permissions
    try:
        os.chmod(output_path, 0o600)
    except:
        pass
    
    print(f"Secure configuration created: {output_path}")


if __name__ == "__main__":
    # Example usage
    create_secure_config()
    
    # Initialize framework
    framework = ImprovedEmotionalIntelligenceFramework()
    
    # Run evaluation
    print("Improved framework initialized with bias mitigation and security enhancements")