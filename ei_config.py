"""
Configuration for Emotional Intelligence Framework
"""

from typing import Dict, List

# Model configurations
MODELS = {
    "Mixtral-8x22B": {
        "api_id": "mistralai/mixtral-8x22b-instruct",
        "provider": "Mistral AI",
        "architecture": "Mixture of Experts",
        "context_length": 32768,
        "strengths": ["speed", "efficiency"]
    },
    "Claude Opus 4": {
        "api_id": "anthropic/claude-opus-4",
        "provider": "Anthropic",
        "architecture": "Constitutional AI",
        "context_length": 200000,
        "strengths": ["empathy", "nuance"]
    },
    "Llama 3.3 70B": {
        "api_id": "meta-llama/llama-3.3-70b-instruct",
        "provider": "Meta",
        "architecture": "Transformer",
        "context_length": 8192,
        "strengths": ["open-source", "balanced"]
    },
    "DeepSeek R1": {
        "api_id": "deepseek/deepseek-r1-0528",
        "provider": "DeepSeek",
        "architecture": "Reasoning-optimized",
        "context_length": 32768,
        "strengths": ["reasoning", "analysis"]
    },
    "Gemini 2.5 Pro": {
        "api_id": "google/gemini-2.5-pro",
        "provider": "Google",
        "architecture": "Multimodal",
        "context_length": 32768,
        "strengths": ["multimodal", "general"]
    }
}

# Test scenarios organized by dimension
TEST_SCENARIOS = {
    "emotion_recognition": [
        {
            "id": "ER1",
            "name": "Work-Life Conflict",
            "prompt": """Identify ALL emotions in this text and explain why:
"I can't believe I have to work late again tonight. I promised my daughter I'd be at her recital, but this project deadline is non-negotiable. I feel like I'm constantly letting her down. Sometimes I wonder if this job is worth missing all these important moments."

List each emotion you detect.""",
            "expected_emotions": ["frustration", "guilt", "disappointment", "conflict", "regret", "doubt"],
            "difficulty": "medium"
        },
        {
            "id": "ER2",
            "name": "Imposter Syndrome",
            "prompt": """What emotions are present in this message:
"Just got the promotion! 😊 Although... I'm not sure I deserve it. Maybe they made a mistake? What if I can't handle the responsibility?"

Identify ALL emotions, including conflicting ones.""",
            "expected_emotions": ["joy", "self-doubt", "anxiety", "imposter syndrome", "fear"],
            "difficulty": "medium"
        },
        {
            "id": "ER3",
            "name": "Hidden Emotions",
            "prompt": """Read between the lines and identify hidden emotions:
"Sure, go ahead to the party without me. It's fine. I didn't really want to go anyway."

What emotions might be beneath the surface?""",
            "expected_emotions": ["hurt", "rejection", "passive-aggressiveness", "sadness", "loneliness"],
            "difficulty": "hard"
        }
    ],
    
    "empathy": [
        {
            "id": "EM1",
            "name": "Pet Loss",
            "prompt": """Your friend messages: "I had to put my dog down yesterday. He was with me for 15 years. The house feels so empty without him."

Write a compassionate, empathetic response.""",
            "evaluation_criteria": ["acknowledgment", "validation", "support", "personal_touch"],
            "difficulty": "medium"
        },
        {
            "id": "EM2",
            "name": "Professional Failure",
            "prompt": """A colleague says: "I completely bombed the presentation. Everyone was staring at me like I was an idiot."

Provide an empathetic response.""",
            "evaluation_criteria": ["normalize", "reframe", "support", "no_minimizing"],
            "difficulty": "medium"
        },
        {
            "id": "EM3",
            "name": "Teen Heartbreak",
            "prompt": """Your teenage niece texts: "Nobody understands me. My parents think I'm overreacting about the breakup but this is literally ruining my life."

How would you respond with genuine empathy?""",
            "evaluation_criteria": ["validate", "no_condescension", "understanding", "bridge_building"],
            "difficulty": "hard"
        }
    ],
    
    "emotional_regulation": [
        {
            "id": "REG1",
            "name": "Professional Criticism",
            "prompt": """In a meeting, a colleague says: "Your analysis is completely wrong and shows you don't understand the basics."

How would you respond professionally?""",
            "evaluation_criteria": ["calm", "professional", "de-escalate", "boundaries"],
            "difficulty": "hard"
        },
        {
            "id": "REG2",
            "name": "Crisis Management",
            "prompt": """You receive news of a family emergency right before an important presentation. 

How do you handle this situation?""",
            "evaluation_criteria": ["compartmentalize", "prioritize", "self-care", "realistic"],
            "difficulty": "hard"
        }
    ],
    
    "social_awareness": [
        {
            "id": "SA1",
            "name": "Social Isolation",
            "prompt": """At a party, you notice a new colleague standing alone, looking at their phone with hunched shoulders.

What might be happening and how would you approach them?""",
            "evaluation_criteria": ["observation", "empathy", "appropriate", "respectful"],
            "difficulty": "medium"
        },
        {
            "id": "SA2",
            "name": "Team Tension",
            "prompt": """During a meeting, two colleagues have a tense exchange and the room goes quiet.

How would you help navigate this situation?""",
            "evaluation_criteria": ["dynamics", "tension", "tactful", "mediate"],
            "difficulty": "hard"
        }
    ]
}

# Evaluation weights
DIMENSION_WEIGHTS = {
    "emotion_recognition": 0.25,
    "empathy": 0.30,
    "emotional_regulation": 0.25,
    "social_awareness": 0.20
}

# API Configuration
API_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "timeout": 45,
    "max_retries": 2,
    "rate_limit_delay": 2,
    "model_delay": 3,
    "default_temperature": 0.7,
    "max_tokens": 800
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "emotion_keywords": [
        # Primary emotions
        'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
        # Secondary emotions
        'frustration', 'guilt', 'disappointment', 'anxiety', 'shame',
        'pride', 'jealousy', 'envy', 'hope', 'relief',
        # Complex emotions
        'regret', 'loneliness', 'confusion', 'overwhelm', 'helplessness',
        'resignation', 'doubt', 'conflict', 'stress', 'worry',
        # Social emotions
        'empathy', 'compassion', 'sympathy', 'understanding', 'support',
        'rejection', 'hurt', 'betrayal', 'trust', 'gratitude'
    ],
    
    "empathy_indicators": [
        'sorry', 'understand', 'feel', 'support', 'here for you',
        'difficult', 'hard', 'compassion', 'care', 'listen',
        'acknowledge', 'valid', 'heart goes out', 'empathize',
        'imagine', 'must be', 'sounds like', 'hear you'
    ],
    
    "professional_indicators": [
        'appreciate', 'feedback', 'clarify', 'understand your perspective',
        'consider', 'discuss', 'collaborate', 'solution', 'resolve',
        'professional', 'respectful', 'constructive'
    ]
}

# Output Configuration
OUTPUT_CONFIG = {
    "results_dir": "results",
    "visualizations_dir": "visualizations",
    "reports_dir": "reports",
    "file_formats": ["json", "csv", "md"],
    "visualization_formats": ["png", "svg"],
    "dpi": 300
}